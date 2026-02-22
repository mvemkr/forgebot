"""
Trade Executor — Wires strategy decisions to live OANDA orders.

Takes a TradeDecision with decision=ENTER and:
  1. Validates through risk manager (kill switch, counter-trend, one-trade rule)
  2. Calculates units based on balance + risk %
  3. Places a limit order on OANDA (no take profit — EVER)
  4. Registers the position on the strategy for one-trade tracking
  5. Logs everything to the trade journal

Safety defaults:
  - dry_run=True until explicitly disabled
  - No take profit is EVER set (hard-enforced)
  - All orders are LIMIT orders (no market orders on entry)
"""
import logging
from pathlib import Path
from typing import Optional, Dict

from ..strategy.forex.set_and_forget import SetAndForgetStrategy, TradeDecision, Decision
from ..exchange.oanda_client import OandaClient
from .trade_journal import TradeJournal
from .risk_manager_forex import ForexRiskManager

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Connects SetAndForgetStrategy signals to live OANDA execution.

    Parameters
    ----------
    strategy : SetAndForgetStrategy
        The running strategy instance (used for position registration)
    oanda : OandaClient
        Live OANDA client
    journal : TradeJournal
        Trade journal for logging
    risk_manager : ForexRiskManager
        Risk controls and position sizing
    dry_run : bool
        If True (default), logs orders without submitting. Set to False to go live.
    """

    def __init__(
        self,
        strategy: SetAndForgetStrategy,
        oanda: OandaClient,
        journal: TradeJournal,
        risk_manager: ForexRiskManager,
        dry_run: bool = True,
    ):
        self.strategy = strategy
        self.oanda = oanda
        self.journal = journal
        self.risk = risk_manager
        self.dry_run = dry_run

        if dry_run:
            logger.info("TradeExecutor initialized in DRY RUN mode — no live orders will be placed")
        else:
            logger.warning("TradeExecutor initialized in LIVE mode — orders WILL be placed on OANDA")

    # ── Main Entry Point ───────────────────────────────────────────────

    def execute(self, decision: TradeDecision, account_balance: float) -> Dict:
        """
        Process a TradeDecision. Only acts on Decision.ENTER.
        Returns a result dict with status and details.

        Parameters
        ----------
        decision : TradeDecision
            Output from SetAndForgetStrategy.evaluate()
        account_balance : float
            Current account balance (used for risk sizing)
        """
        pair = decision.pair

        # Only execute ENTER decisions
        if decision.decision != Decision.ENTER:
            logger.debug(f"{pair}: Decision is {decision.decision.value} — nothing to execute")
            return {"status": "skipped", "reason": decision.decision.value}

        # ── Kill switch check ─────────────────────────────────────────
        halted, halt_reason = self.risk.check_and_halt_if_needed(account_balance)
        if halted:
            logger.warning(f"🛑 Kill switch active — blocking {pair} entry: {halt_reason}")
            self.journal.log_blocked(pair, halt_reason, ["kill_switch"])
            return {"status": "blocked", "reason": halt_reason}

        # ── Counter-trend check ───────────────────────────────────────
        trend_weekly = decision.trend_weekly.value if decision.trend_weekly else "neutral"
        trend_daily  = decision.trend_daily.value  if decision.trend_daily  else "neutral"
        trend_4h     = decision.trend_4h.value     if decision.trend_4h     else "neutral"

        ct_blocked, ct_reason = self.risk.is_counter_trend(
            decision.direction, trend_weekly, trend_daily
        )
        if ct_blocked:
            logger.warning(f"{pair}: {ct_reason}")
            self.journal.log_blocked(pair, ct_reason, ["counter_trend"])
            return {"status": "blocked", "reason": ct_reason}

        # ── Dual-trade eligibility (belt and suspenders) ─────────────
        # Checks: max concurrent positions, currency overlap, book budget.
        pos_blocked, pos_reason = self.risk.check_entry_eligibility(
            pair, self.strategy.open_positions, account_balance
        )
        if pos_blocked:
            logger.warning(f"{pair}: {pos_reason}")
            self.journal.log_blocked(pair, pos_reason, ["entry_eligibility"])
            return {"status": "blocked", "reason": pos_reason}

        # ── Validate entry fields ─────────────────────────────────────
        if decision.entry_price is None or decision.stop_loss is None:
            logger.error(f"{pair}: ENTER decision missing entry_price or stop_loss")
            return {"status": "error", "reason": "Missing entry_price or stop_loss"}

        # ── NO TAKE PROFIT — EVER ─────────────────────────────────────
        # This is a hard rule from the strategy. Enforce it at the executor level.
        take_profit = None  # explicitly None, always

        # ── Risk sizing (book-aware) ──────────────────────────────────
        # First trade: full tier rate. Second trade: capped at remaining
        # book budget so total exposure never exceeds MAX_BOOK_EXPOSURE.
        risk_pct = self.risk.get_book_risk_pct(account_balance, self.strategy.open_positions)
        if risk_pct <= 0:
            reason = "Book exposure budget exhausted — no room for another trade."
            logger.warning(f"{pair}: {reason}")
            return {"status": "blocked", "reason": reason}
        self.strategy.risk_pct = risk_pct  # keep strategy in sync

        stop_pips = abs(decision.entry_price - decision.stop_loss)
        stop_pips_display = stop_pips * (100 if "JPY" in pair else 10000)

        units = self.oanda.calculate_units(
            pair=pair,
            account_balance=account_balance,
            risk_pct=risk_pct,
            stop_pips=stop_pips_display,
        )

        risk_dollars = account_balance * (risk_pct / 100)
        lot_size = units / 100_000  # convert units to lots for journal

        logger.info(
            f"EXECUTING: {pair} {decision.direction.upper()} "
            f"@ {decision.entry_price:.5f}  "
            f"SL={decision.stop_loss:.5f}  "
            f"Units={units:,} ({lot_size:.2f} lots)  "
            f"Risk={risk_pct}% (${risk_dollars:.2f})  "
            f"Stop={stop_pips_display:.1f} pips  "
            f"{'[DRY RUN]' if self.dry_run else '[LIVE]'}"
        )

        # ── Check for existing OANDA position on this pair ────────────
        oanda_trades = self.oanda.get_open_trades()
        existing = next(
            (t for t in oanda_trades if t["instrument"].replace("_", "/") == pair),
            None
        )
        if existing:
            reason = f"OANDA already has open position on {pair} (trade {existing['id']}) — skipping"
            logger.warning(reason)
            return {"status": "blocked", "reason": reason}

        # ── Place order ───────────────────────────────────────────────
        result = self.oanda.place_limit_order(
            pair=pair,
            direction=decision.direction,
            units=units,
            limit_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            take_profit=take_profit,   # always None
            dry_run=self.dry_run,
        )

        # Extract OANDA trade ID from response
        oanda_trade_id: Optional[str] = None
        if not self.dry_run:
            fill = result.get("orderFillTransaction", {})
            oanda_trade_id = fill.get("tradeOpened", {}).get("tradeID")
            if not oanda_trade_id:
                oanda_trade_id = result.get("orderCreateTransaction", {}).get("id")

        # ── Register position on strategy ─────────────────────────────
        self.strategy.register_open_position(
            pair=pair,
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            direction=decision.direction,
            pattern_type=decision.pattern.pattern_type if decision.pattern else None,
            neckline_ref=decision.neckline_ref,
            risk_pct=risk_pct,   # stored for book exposure tracking on 2nd trade
        )

        # Enrich position dict with metadata for post-trade analysis
        if pair in self.strategy.open_positions:
            pos = self.strategy.open_positions[pair]
            if oanda_trade_id:
                pos["oanda_trade_id"]  = oanda_trade_id
            pos["signal_strength"]     = decision.entry_signal.strength if decision.entry_signal else 0.0
            pos["key_level_score"]     = decision.nearest_level.score   if decision.nearest_level else 0
            pos["psych_level"]         = decision.nearest_level.price   if decision.nearest_level else 0.0
            pos["confidence"]          = decision.confidence
            pos["risk_pct"]            = risk_pct
            pos["risk_dollars"]        = risk_dollars
            pos["units"]               = units
            pos["entry_ts"]            = decision.entry_price  # approximation; journal has real ts
            # Session extraction
            for part in (decision.reason or "").split("|"):
                if "Session:" in part:
                    pos["session"] = part.strip().replace("Session:", "").strip()
                    break

        # ── Journal ───────────────────────────────────────────────────
        session = ""
        if decision.entry_signal:
            session = ""
        # Get session from the decision reason if available
        for part in (decision.reason or "").split("|"):
            if "Session:" in part:
                session = part.strip().replace("Session:", "").strip()

        self.journal.log_trade_entered(
            pair=pair,
            direction=decision.direction,
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            lot_size=lot_size,
            units=units,
            risk_pct=risk_pct,
            risk_dollars=risk_dollars,
            account_balance=account_balance,
            pattern=decision.pattern.pattern_type if decision.pattern else "unknown",
            pattern_clarity=decision.pattern.clarity if decision.pattern else 0.0,
            signal_type=decision.entry_signal.signal_type if decision.entry_signal else "unknown",
            signal_strength=decision.entry_signal.strength if decision.entry_signal else 0.0,
            level_score=decision.nearest_level.score if decision.nearest_level else 0,
            trend_weekly=trend_weekly,
            trend_daily=trend_daily,
            trend_4h=trend_4h,
            session=session,
            confidence=decision.confidence,
            oanda_trade_id=oanda_trade_id,
            dry_run=self.dry_run,
            notes=f"Entry via orchestrator. R={stop_pips_display:.0f}pips",
        )

        mode = "DRY RUN" if self.dry_run else "LIVE"
        logger.info(f"✅ [{mode}] {pair} {decision.direction.upper()} order placed. Trade ID: {oanda_trade_id or 'pending'}")

        return {
            "status": "executed",
            "pair": pair,
            "direction": decision.direction,
            "entry": decision.entry_price,
            "stop_loss": decision.stop_loss,
            "units": units,
            "risk_pct": risk_pct,
            "risk_dollars": risk_dollars,
            "dry_run": self.dry_run,
            "oanda_result": result,
        }
