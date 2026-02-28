"""
Reconciler — continuous broker state integrity guardrail.

Two entry points:
  reconcile_full()  — called at startup
                      fetches openTrades + pendingOrders + accountSummary,
                      rebuilds internal state from broker truth.
  reconcile_light() — called before any state-changing action and every 60s
                      (LIVE_REAL only). Fetches openTrades + pendingOrders
                      (+ summary for LIVE_REAL). Fails open on API errors.

Four invariants checked on every call:

  1. EXTERNAL_CLOSE     : local open position absent from broker
                          → mark close, clear local, pause entries
  2. RECOVERED_POSITION : broker position absent from local state
                          → reconstruct minimal local state, log
                          → pause if detected during operation (not startup)
  3. EXTERNAL_MODIFY    : broker stop/TP diverges from local expectation
                          → adopt broker value; pause if stop moved away from entry
  4. RECONCILE_PAUSE    : any safety-relevant mismatch sets pause_new_entries

Design:
  - Fails open on network error (never blocks trading due to API hiccup)
  - dry_run=True blocks all mutations (test-safe)
  - throttle: minimum 30s between light reconciles (pre-action calls)
  - Does NOT change signal logic; state integrity only
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..exchange.oanda_client import OandaClient
    from ..strategy.forex.set_and_forget import SetAndForgetStrategy
    from .trade_journal import TradeJournal
    from .notifier import Notifier
    from .control_state import ControlState
    from .account_state import AccountState

logger = logging.getLogger(__name__)

# ── Pip tolerance constants ────────────────────────────────────────────────
def _stop_tolerance(pair: str) -> float:
    """2-pip tolerance for stop-price comparison."""
    return 0.02 if "JPY" in pair else 0.0002


def _price_to_pips(pair: str, diff: float) -> float:
    return diff / (0.01 if "JPY" in pair else 0.0001)


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class ReconcileResult:
    clean:             bool       = True
    actions:           List[str]  = field(default_factory=list)
    pause_triggered:   bool       = False
    external_closes:   List[str]  = field(default_factory=list)
    recovered:         List[str]  = field(default_factory=list)
    external_modifies: List[str]  = field(default_factory=list)

    def summary(self) -> str:
        if self.clean:
            return "clean"
        parts = []
        if self.external_closes:   parts.append(f"closed={self.external_closes}")
        if self.recovered:         parts.append(f"recovered={self.recovered}")
        if self.external_modifies: parts.append(f"modified={self.external_modifies}")
        if self.pause_triggered:   parts.append("PAUSED")
        return " | ".join(parts)


# ── Reconciler ────────────────────────────────────────────────────────────

class Reconciler:
    """
    Continuous broker reconciliation guard.

    Instantiate once in the orchestrator:

        self.reconciler = Reconciler(
            oanda        = self.oanda,
            strategy     = self.strategy,
            journal      = self.journal,
            notifier     = self.notifier,
            control      = self.control,
            account      = self.account,
            account_mode = self.account.mode.value,
            dry_run      = self.dry_run,
        )

    Then call:
        self.reconciler.reconcile_full()           # startup
        self.reconciler.reconcile_light()          # pre-action / periodic
    """

    #: Minimum seconds between throttled light reconcile calls
    MIN_LIGHT_INTERVAL_S: float = 30.0

    def __init__(
        self,
        oanda,
        strategy,
        journal,
        notifier,
        control,
        account=None,
        account_mode: str = "LIVE_PAPER",
        dry_run: bool = True,
    ) -> None:
        self.oanda        = oanda
        self.strategy     = strategy
        self.journal      = journal
        self.notifier     = notifier
        self.control      = control
        self.account      = account
        self.account_mode = account_mode   # "LIVE_REAL" | "LIVE_PAPER" | "BACKTEST"
        self.dry_run      = dry_run
        self._last_light_ts: Optional[datetime] = None

    # ── Public API ────────────────────────────────────────────────────

    def reconcile_full(self) -> ReconcileResult:
        """
        Full startup reconcile.

        Fetches openTrades, pendingOrders, accountSummary.
        Rebuilds strategy.open_positions from broker truth.
        Does NOT pause on RECOVERED_POSITION (normal restart scenario).
        Fails open on broker API error so startup isn't blocked.
        """
        logger.info("🔍 reconcile_full() — startup broker state verification")
        try:
            broker_trades  = self.oanda.get_open_trades()
            pending_orders = self.oanda.get_pending_orders()
            summary        = self.oanda.get_account_summary()
        except Exception as e:
            logger.error(f"reconcile_full(): broker fetch failed — starting in unverified state: {e}")
            return ReconcileResult(clean=True)  # fail-open at startup

        result = self._run_invariants(
            broker_trades  = broker_trades,
            pending_orders = pending_orders,
            summary        = summary,
            is_startup     = True,
        )
        _log_result("full", result)
        return result

    def reconcile_light(self, throttle: bool = True) -> ReconcileResult:
        """
        Quick pre-action invariant check.

        Called:
          - Before placing an order
          - Before modifying stop/trail
          - Before computing risk that depends on equity/open exposure
          - Every 60s on LIVE_REAL (orchestrator timer)

        Fails open on broker API error — never blocks an action due to
        a transient network hiccup.  Throttled to at most once per
        MIN_LIGHT_INTERVAL_S seconds when throttle=True.
        """
        if throttle and self._last_light_ts is not None:
            elapsed = (datetime.now(timezone.utc) - self._last_light_ts).total_seconds()
            if elapsed < self.MIN_LIGHT_INTERVAL_S:
                return ReconcileResult(clean=True)  # fresh enough — skip

        try:
            broker_trades  = self.oanda.get_open_trades()
            pending_orders = self.oanda.get_pending_orders()
            summary = (
                self.oanda.get_account_summary()
                if self.account_mode == "LIVE_REAL"
                else None
            )
        except Exception as e:
            logger.warning(f"reconcile_light(): broker fetch failed (fail-open): {e}")
            return ReconcileResult(clean=True)

        self._last_light_ts = datetime.now(timezone.utc)
        result = self._run_invariants(
            broker_trades  = broker_trades,
            pending_orders = pending_orders,
            summary        = summary,
            is_startup     = False,
        )
        if not result.clean:
            _log_result("light", result)
        return result

    # ── Invariant Engine ──────────────────────────────────────────────

    def _run_invariants(
        self,
        broker_trades:  List[Dict],
        pending_orders: List[Dict],
        summary:        Optional[Dict],
        is_startup:     bool,
    ) -> ReconcileResult:
        result = ReconcileResult()

        # broker_open keyed by pair ("GBP/JPY"), value = raw broker trade dict
        broker_open = {
            t["instrument"].replace("_", "/"): t
            for t in (broker_trades or [])
        }
        # Pairs with an unfilled pending limit order (don't flag as external_close)
        pending_pairs = {
            o.get("instrument", "").replace("_", "/")
            for o in (pending_orders or [])
        }

        local_open = dict(self.strategy.open_positions)

        # ── Invariant 1: EXTERNAL_CLOSE ──────────────────────────────
        for pair, pos in local_open.items():
            if pair not in broker_open:
                if pair in pending_pairs:
                    # Limit order still pending — not yet filled → not a close
                    logger.debug(f"reconcile: {pair} not in open trades but has pending order — skipping")
                    continue
                self._handle_external_close(pair, pos, result)

        # ── Invariant 2: RECOVERED_POSITION ──────────────────────────
        for pair, broker_trade in broker_open.items():
            if pair not in local_open:
                self._handle_recovered_position(
                    pair, broker_trade, result, is_startup=is_startup
                )

        # ── Invariant 3: EXTERNAL_MODIFY ─────────────────────────────
        for pair in local_open:
            if pair in broker_open:
                self._check_external_modify(
                    pair, local_open[pair], broker_open[pair], result
                )

        # ── Invariant 4: Pause on safety-relevant mismatches ─────────
        # Safety-relevant = external close (unexpected), or recovered
        # position found during live operation (not startup).
        needs_pause = (
            bool(result.external_closes)
            or (bool(result.recovered) and not is_startup)
        )
        if needs_pause and not self.dry_run:
            parts = []
            if result.external_closes:
                parts.append(f"external_close={result.external_closes}")
            if result.recovered and not is_startup:
                parts.append(f"recovered_position={result.recovered}")
            reason = (
                "RECONCILE: "
                + " + ".join(parts)
                + " — manual review required before resuming"
            )
            self.control.pause(reason=reason, updated_by="reconciler")
            result.pause_triggered = True
            result.actions.append("RECONCILE_PAUSE")
            logger.warning(f"⛔ Reconciler paused entries: {reason}")

        result.clean = len(result.actions) == 0
        return result

    # ── Invariant 1: External Close ───────────────────────────────────

    def _handle_external_close(
        self, pair: str, pos: dict, result: ReconcileResult
    ) -> None:
        """Local position gone from broker → mark external close."""
        direction = pos.get("direction", "?")
        entry     = pos.get("entry", 0.0)
        stop      = pos.get("stop", entry)
        units     = pos.get("units", 0)
        trade_id  = pos.get("oanda_trade_id")

        logger.warning(
            f"⚠️  EXTERNAL_CLOSE: {pair} {direction.upper()} "
            f"entry={entry:.5f} stop={stop:.5f} id={trade_id} — "
            f"position no longer on broker"
        )

        # Approximate PnL: use stop as exit price (best guess without broker fill data)
        pip      = 0.01 if "JPY" in pair else 0.0001
        mult     = 1 if direction == "long" else -1
        pnl_est  = mult * (stop - entry) * units * pip

        self._journal_reconcile(pair, "EXTERNAL_CLOSE", {
            "direction":       direction,
            "entry_price":     entry,
            "exit_price_est":  stop,
            "units":           units,
            "pnl_estimate":    round(pnl_est, 2),
            "oanda_trade_id":  trade_id,
            "notes": (
                "Position no longer present on broker — externally closed "
                "or stop hit outside sync window. PnL is estimated from last stop."
            ),
        })

        # Clear local state so bot doesn't attempt trailing on a ghost position
        if not self.dry_run:
            try:
                self.strategy.close_position(pair, stop)
            except Exception as e:
                logger.warning(f"close_position({pair}) failed during external_close: {e}")
                # Force-remove as fallback
                self.strategy.open_positions.pop(pair, None)

        result.external_closes.append(pair)
        result.actions.append(f"EXTERNAL_CLOSE:{pair}")

        self.notifier.send(
            f"⚠️ RECONCILER: *{pair} external close detected*\n"
            f"Position no longer on broker.\n"
            f"Direction: {direction.upper()} | Entry: {entry:.5f} | Last stop: {stop:.5f}\n"
            f"Est. PnL: ${pnl_est:+,.2f}\n"
            f"Entries PAUSED — resume manually after review."
        )

    # ── Invariant 2: Recovered Position ──────────────────────────────

    def _handle_recovered_position(
        self,
        pair: str,
        broker_trade: dict,
        result: ReconcileResult,
        is_startup: bool,
    ) -> None:
        """Broker has a position we don't know about → reconstruct minimal state."""
        direction = broker_trade.get("direction", "?")
        entry     = float(broker_trade.get("entry", 0.0))
        stop      = broker_trade.get("stop_loss")
        units     = broker_trade.get("units", 0)
        trade_id  = broker_trade.get("id", "?")
        open_time = broker_trade.get("open_time", "")

        log_fn = logger.info if is_startup else logger.warning
        icon   = "ℹ️" if is_startup else "⚠️"
        log_fn(
            f"{icon}  RECOVERED_POSITION: {pair} {direction.upper()} "
            f"entry={entry:.5f} stop={stop} id={trade_id} "
            f"({'startup recovery' if is_startup else 'LIVE DRIFT — state inconsistency'})"
        )

        if stop is None:
            logger.error(
                f"RECOVERED_POSITION {pair}: NO STOP LOSS on broker trade {trade_id}! "
                f"Position is unprotected. Manual intervention required."
            )

        # Reconstruct minimal local state so trailing and monitoring work.
        # trail_locked=False, trail_max=entry → conservative: assume no MFE progress.
        # If at startup, the real trail state is unknown; monitor will re-evaluate.
        reconstructed: Dict = {
            "pair":           pair,
            "direction":      direction,
            "entry":          entry,
            "stop":           stop if stop is not None else entry,
            "initial_risk":   abs(entry - stop) if stop is not None else 0.0,
            "units":          units,
            "oanda_trade_id": trade_id,
            "trail_locked":   False,
            "trail_max":      entry,
            "pattern_type":   "RECOVERED",
            "confidence":     0.0,
            "entry_ts":       open_time,
            "source":         "RECONCILER_RECOVERED",
        }

        if not self.dry_run:
            self.strategy.open_positions[pair] = reconstructed

        self._journal_reconcile(pair, "RECOVERED_POSITION", {
            "direction":      direction,
            "entry_price":    entry,
            "stop":           stop,
            "units":          units,
            "oanda_trade_id": trade_id,
            "is_startup":     is_startup,
            "notes": (
                "Normal startup recovery — state file was stale or missing"
                if is_startup
                else "⚠️ Position appeared on broker during live operation — state drift detected"
            ),
        })

        result.recovered.append(pair)
        result.actions.append(f"RECOVERED_POSITION:{pair}")

        if is_startup:
            self.notifier.send(
                f"♻️ RECONCILER: *{pair} recovered at startup*\n"
                f"{direction.upper()} @ {entry:.5f}  SL={stop}  units={int(units)}\n"
                f"Trail state unknown — conservative (no MFE progress assumed)."
            )
        else:
            self.notifier.send(
                f"⚠️ RECONCILER: *{pair} RECOVERED POSITION (live)*\n"
                f"Broker has position not tracked locally.\n"
                f"{direction.upper()} @ {entry:.5f}  SL={stop}\n"
                f"State reconstructed. Entries PAUSED — manual review required."
            )

    # ── Invariant 3: External Modify ──────────────────────────────────

    def _check_external_modify(
        self,
        pair: str,
        local_pos: dict,
        broker_trade: dict,
        result: ReconcileResult,
    ) -> None:
        """Broker stop/TP differs from local → adopt broker value; flag safety violations."""
        local_stop  = local_pos.get("stop")
        broker_stop = broker_trade.get("stop_loss")
        direction   = local_pos.get("direction", "long")
        entry       = float(local_pos.get("entry", 0.0))
        tol         = _stop_tolerance(pair)

        # ── Stop mismatch ─────────────────────────────────────────────
        if local_stop is not None and broker_stop is not None:
            diff = abs(local_stop - broker_stop)
            if diff > tol:
                pips = _price_to_pips(pair, diff)

                # Safety check: did stop move FURTHER from entry? (never allowed)
                stop_worse = (
                    (direction == "long"  and broker_stop < local_stop) or
                    (direction == "short" and broker_stop > local_stop)
                )

                logger.warning(
                    f"⚠️  EXTERNAL_MODIFY {pair}: stop "
                    f"local={local_stop:.5f} → broker={broker_stop:.5f} "
                    f"({pips:.1f} pips {'WORSE ⛔' if stop_worse else 'tighter ✓'})"
                )

                # Always adopt broker as ground truth
                if not self.dry_run:
                    local_pos["stop"] = broker_stop
                    self.strategy.open_positions[pair] = local_pos

                self._journal_reconcile(pair, "EXTERNAL_MODIFY", {
                    "field":          "stop_loss",
                    "local_value":    local_stop,
                    "broker_value":   broker_stop,
                    "diff_pips":      round(pips, 1),
                    "stop_worse":     stop_worse,
                    "direction":      direction,
                    "entry":          entry,
                    "oanda_trade_id": local_pos.get("oanda_trade_id"),
                    "notes": (
                        "⚠️ Stop moved AWAY from entry — safety violation; entries paused"
                        if stop_worse
                        else "Stop tightened externally (manual trail?) — adopted"
                    ),
                })

                result.external_modifies.append(pair)
                result.actions.append(f"EXTERNAL_MODIFY:{pair}:stop")

                self.notifier.send(
                    f"{'⚠️' if stop_worse else 'ℹ️'} RECONCILER: *{pair} stop changed externally*\n"
                    f"Local: {local_stop:.5f} → Broker: {broker_stop:.5f} ({pips:.1f} pips)\n"
                    + (
                        "⛔ Stop moved AWAY from entry — SAFETY VIOLATION\n"
                        "Entries PAUSED — resume manually after review."
                        if stop_worse
                        else "Adopted broker stop (tighter)."
                    )
                )

                # Pause only if stop moved AWAY from entry
                if stop_worse and not self.dry_run:
                    self.control.pause(
                        reason=(
                            f"RECONCILE: {pair} stop moved away from entry "
                            f"({local_stop:.5f}→{broker_stop:.5f}) — safety violation"
                        ),
                        updated_by="reconciler",
                    )
                    result.pause_triggered = True
                    result.actions.append(f"RECONCILE_PAUSE:stop_worse:{pair}")

        # ── TP check (should never exist per Alex's no-TP rule) ───────
        broker_tp = broker_trade.get("take_profit")
        if broker_tp is not None:
            logger.warning(
                f"⚠️  EXTERNAL_MODIFY {pair}: unexpected take-profit on broker: {broker_tp} "
                f"(violates no-TP rule)"
            )
            self._journal_reconcile(pair, "EXTERNAL_MODIFY", {
                "field":          "take_profit",
                "broker_value":   broker_tp,
                "oanda_trade_id": local_pos.get("oanda_trade_id"),
                "notes": "Take-profit found on broker — violates no-TP strategy rule",
            })
            result.external_modifies.append(pair)
            result.actions.append(f"EXTERNAL_MODIFY:{pair}:take_profit")

    # ── Helper ────────────────────────────────────────────────────────

    def _journal_reconcile(self, pair: str, event: str, data: dict) -> None:
        try:
            self.journal.log_reconcile_event(pair=pair, event=event, data=data)
        except Exception as e:
            logger.warning(f"Failed to journal reconcile event {event}/{pair}: {e}")


# ── Module-level log helper ───────────────────────────────────────────────

def _log_result(mode: str, result: ReconcileResult) -> None:
    if result.clean:
        logger.info(f"✅ reconcile_{mode}(): clean")
    else:
        logger.warning(f"⚠️  reconcile_{mode}(): {result.summary()}")
