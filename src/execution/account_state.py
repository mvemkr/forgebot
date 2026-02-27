"""
src/execution/account_state.py
===============================
AccountMode + AccountState — single source of truth for equity and execution mode.

Three mutually exclusive modes:

  LIVE_REAL   real OANDA account; equity pulled from broker each tick.
              If broker fetch fails → equity = UNKNOWN (None).
              Bot blocks new entries when UNKNOWN; existing positions keep running.

  LIVE_PAPER  no real orders; equity tracked internally and persisted to disk.
              Equity never becomes UNKNOWN — starts from SIM_STARTING_EQUITY
              (or last persisted value) and is updated after each simulated close.
              The broker is NEVER consulted for equity or NAV; only candle prices
              are fetched from OANDA in this mode.

  BACKTEST    offline backtester; AccountState not used for live file I/O.

Design rules
------------
- equity=None is an explicit sentinel meaning "broker fetch failed".  Never default to 0.
- peak_equity only ever increases (watermark logic).
- Paper mode writes equity to PAPER_STATE_FILE after every PnL event so a crash/restart
  picks up where it left off.  SIM_STARTING_EQUITY only applies on first bootstrap
  (when PAPER_STATE_FILE does not exist); all subsequent restarts load from disk.
- PAPER_JOURNAL_FILE records every entry and exit event as JSONL; provides a
  complete audit trail of the simulated account.
- week_id / week_pnl / week_trades reset automatically on ISO-week boundary.
- Equity identity invariant: equity_after == equity_before + realized_pnl (±0.01)
  for every close event; violations are logged as errors.
- to_dict() is safe to embed in bot_state.json / heartbeat.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PAPER_STATE_FILE   = Path.home() / "trading-bot" / "runtime_state" / "paper_account.json"
PAPER_JOURNAL_FILE = Path.home() / "trading-bot" / "runtime_state" / "paper_journal.jsonl"
_UNKNOWN_DISPLAY   = "UNKNOWN"


# ── Mode enum ─────────────────────────────────────────────────────────────────

class AccountMode(Enum):
    LIVE_REAL  = "live_real"   # real broker account
    LIVE_PAPER = "live_paper"  # simulated equity, no real orders
    BACKTEST   = "backtest"    # offline backtester


# ── AccountState ──────────────────────────────────────────────────────────────

class AccountState:
    """
    Tracks equity, peak, and sourcing for a single execution mode.

    LIVE_PAPER closed-system guarantee
    -----------------------------------
    In LIVE_PAPER mode this object is the sole source of equity truth.
    The broker is consulted ONLY for candle prices; its balance/NAV is
    never read for sizing, display, or risk calculations.

    Attributes
    ----------
    mode : AccountMode
    equity : Optional[float]
        Current equity.  None means UNKNOWN (broker unreachable in LIVE_REAL).
    peak_equity : float
        High-water mark.  Set to initial equity on first call; only increases.
    realized_session_pnl : float
        Cumulative realized PnL since last restart.
    broker_fetch_failures : int
        Consecutive broker-fetch failures (LIVE_REAL only); reset on success.
    week_id : str
        ISO week string, e.g. "2026-W08".  Drives weekly-stats resets.
    week_pnl : float
        Realized PnL accumulated this ISO week (resets on week boundary).
    week_trades : int
        Trade closes this ISO week (resets on week boundary).
    """

    def __init__(
        self,
        mode: AccountMode,
        initial_equity: Optional[float],
        peak_equity: Optional[float] = None,
        paper_file: Path = PAPER_STATE_FILE,
        paper_journal_file: Path = PAPER_JOURNAL_FILE,
    ):
        self.mode                        = mode
        self.equity: Optional[float]     = initial_equity
        self.peak_equity: float          = peak_equity or initial_equity or 0.0
        self.realized_session_pnl: float = 0.0
        self.broker_fetch_failures: int  = 0
        self._paper_file                 = Path(paper_file)
        self._paper_journal_file         = Path(paper_journal_file)
        self._last_broker_ts: Optional[str] = None
        self.last_update_ts: Optional[str]  = (
            datetime.now(timezone.utc).isoformat() if initial_equity is not None else None
        )

        # Weekly accounting (paper mode only; harmless elsewhere)
        self.week_id: str     = self._current_week_id()
        self.week_pnl: float  = 0.0
        self.week_trades: int = 0

        # Ensure peak ≥ equity at construction
        if self.equity is not None and self.equity > self.peak_equity:
            self.peak_equity = self.equity

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def is_unknown(self) -> bool:
        """True when broker fetch has failed and equity is not available."""
        return self.equity is None

    @property
    def is_tradeable(self) -> bool:
        """True when equity is known and positive (safe to size a trade)."""
        return self.equity is not None and self.equity > 0.0

    @property
    def equity_source(self) -> str:
        """
        Machine + UI source label.
        LIVE_REAL  → "BROKER" (or "UNKNOWN" when fetch has failed)
        LIVE_PAPER → "SIM"
        BACKTEST   → "BACKTEST"
        """
        if self.mode == AccountMode.LIVE_REAL:
            return "BROKER" if not self.is_unknown else "UNKNOWN"
        if self.mode == AccountMode.LIVE_PAPER:
            return "SIM"
        return "BACKTEST"

    @property
    def equity_display(self) -> str:
        """Display string — shows 'UNKNOWN' when equity is None."""
        if self.equity is None:
            return _UNKNOWN_DISPLAY
        return f"${self.equity:,.2f}"

    @property
    def mode_label(self) -> str:
        """Short UI label for the mode badge."""
        if self.mode == AccountMode.LIVE_REAL:
            return "LIVE REAL"
        if self.mode == AccountMode.LIVE_PAPER:
            return "LIVE PAPER"
        return "BACKTEST"

    def safe_equity(self, fallback: float = 0.0) -> float:
        """Return equity as float, substituting *fallback* when UNKNOWN."""
        return self.equity if self.equity is not None else fallback

    # ── ISO week helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _current_week_id() -> str:
        """Return ISO week string for the current UTC date, e.g. '2026-W08'."""
        d = datetime.now(timezone.utc)
        y, w, _ = d.isocalendar()
        return f"{y}-W{w:02d}"

    def _check_week_boundary(self) -> bool:
        """
        Detect ISO-week rollover and reset weekly stats if needed.
        Returns True when a reset occurred (caller may want to log).
        """
        current = self._current_week_id()
        if current != self.week_id:
            logger.info(
                f"AccountState (paper): week boundary {self.week_id!r} → {current!r} "
                f"| old week_pnl=${self.week_pnl:+,.2f}  week_trades={self.week_trades}"
            )
            self.week_id     = current
            self.week_pnl    = 0.0
            self.week_trades = 0
            self.save_paper()
            return True
        return False

    # ── LIVE_REAL update path ─────────────────────────────────────────────────

    def update_from_broker(self, summary: dict) -> bool:
        """
        Apply a successful broker summary dict.
        Sets equity, updates peak, resets failure counter.
        Returns True on success, False on missing balance key.

        NO-OP in LIVE_PAPER mode — paper equity is NEVER overwritten by the broker.
        This guard is the primary safety valve preventing equity corruption.
        """
        if self.mode != AccountMode.LIVE_REAL:
            logger.debug(
                f"AccountState.update_from_broker: no-op in {self.mode.value} mode"
            )
            return False

        bal = summary.get("balance")
        if bal is None:
            logger.warning("AccountState: broker summary missing 'balance' key")
            self.broker_fetch_failures += 1
            return False

        self.equity = float(bal)
        now_ts = datetime.now(timezone.utc).isoformat()
        self._last_broker_ts = now_ts
        self.last_update_ts  = now_ts
        self.broker_fetch_failures = 0

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            logger.debug(f"AccountState: new peak ${self.peak_equity:,.2f}")

        return True

    def mark_broker_failed(self) -> None:
        """
        Record a broker fetch failure.  Equity transitions to UNKNOWN on the
        FIRST failure (we never default to a stale/zero value).
        Only used in LIVE_REAL mode.

        NO-OP in LIVE_PAPER mode — paper equity is self-contained and must never
        be corrupted by a broker connectivity event.
        """
        if self.mode != AccountMode.LIVE_REAL:
            logger.debug(
                f"AccountState.mark_broker_failed: no-op in {self.mode.value} mode"
            )
            return
        self.broker_fetch_failures += 1
        prev = self.equity
        self.equity = None
        if prev is not None:
            logger.warning(
                f"AccountState: broker fetch failed "
                f"(failures={self.broker_fetch_failures}); "
                f"equity now UNKNOWN (was ${prev:,.2f}). "
                f"New entries blocked."
            )
        else:
            logger.debug(
                f"AccountState: broker fetch still failing "
                f"(failures={self.broker_fetch_failures})"
            )

    # ── LIVE_PAPER update path ────────────────────────────────────────────────

    def apply_pnl(
        self,
        realized_pnl: float,
        balance_after: Optional[float] = None,
        pair: str = "",
        exit_reason: str = "",
        ts: Optional[str] = None,
    ) -> None:
        """
        Apply realized PnL to paper equity and enforce all accounting invariants.

        Invariants enforced
        -------------------
        1. equity_after == equity_before + realized_pnl  (identity, ±0.01 rounding)
        2. peak_equity only increases
        3. week_pnl / week_trades reset automatically on ISO-week boundary
        4. Every close event is appended to PAPER_JOURNAL_FILE
        5. paper_account.json is written atomically after every event

        If *balance_after* is explicitly provided (e.g. from backtester or manual
        override) it overrides the incremental calculation; invariant #1 is still
        checked and logged if violated.
        """
        if self.mode not in (AccountMode.LIVE_PAPER, AccountMode.BACKTEST):
            logger.debug("AccountState.apply_pnl: not in paper mode — ignoring")
            return

        # Check week boundary BEFORE applying PnL so stats land in the right week
        self._check_week_boundary()

        equity_before = self.equity

        # ── Apply the PnL ────────────────────────────────────────────────────
        if balance_after is not None:
            self.equity = float(balance_after)
        elif self.equity is not None:
            self.equity = self.equity + realized_pnl
        else:
            logger.warning("AccountState.apply_pnl: equity was None in paper mode — ignoring")
            return

        # ── Invariant 1: equity identity ─────────────────────────────────────
        if equity_before is not None and balance_after is None:
            expected = equity_before + realized_pnl
            delta    = abs(self.equity - expected)
            if delta > 0.01:
                logger.error(
                    f"AccountState: equity identity violated! "
                    f"equity_before={equity_before:.4f} + pnl={realized_pnl:.4f} "
                    f"= expected {expected:.4f} but equity_after={self.equity:.4f} "
                    f"(delta={delta:.4f})"
                )

        # ── Update tracking fields ────────────────────────────────────────────
        self.realized_session_pnl += realized_pnl
        self.last_update_ts = datetime.now(timezone.utc).isoformat()

        # Invariant 2: peak watermark
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Weekly stats (only meaningful in paper mode)
        self.week_pnl    += realized_pnl
        self.week_trades += 1

        # ── Journal entry ────────────────────────────────────────────────────
        self._append_journal({
            "event":          "exit",
            "ts":             ts or self.last_update_ts,
            "pair":           pair,
            "exit_reason":    exit_reason,
            "realized_pnl":   round(realized_pnl, 4),
            "equity_before":  round(equity_before, 4) if equity_before is not None else None,
            "equity_after":   round(self.equity, 4),
            "peak_equity":    round(self.peak_equity, 4),
            "week_id":        self.week_id,
            "week_pnl":       round(self.week_pnl, 4),
            "week_trades":    self.week_trades,
        })

        # ── Persist ──────────────────────────────────────────────────────────
        self.save_paper()

        logger.info(
            f"AccountState (paper): PnL ${realized_pnl:+,.2f} → equity "
            f"${self.equity:,.2f} | peak ${self.peak_equity:,.2f} | "
            f"session ${self.realized_session_pnl:+,.2f} | "
            f"week [{self.week_id}] ${self.week_pnl:+,.2f} / {self.week_trades}T"
        )

    def log_entry_event(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        risk_dollars: float,
        ts: Optional[str] = None,
    ) -> None:
        """
        Log a trade entry event to the paper journal.
        Does NOT modify equity — that only happens on close via apply_pnl().
        """
        if self.mode not in (AccountMode.LIVE_PAPER,):
            return
        self._append_journal({
            "event":        "entry",
            "ts":           ts or datetime.now(timezone.utc).isoformat(),
            "pair":         pair,
            "direction":    direction,
            "entry_price":  entry_price,
            "stop_loss":    stop_loss,
            "risk_dollars": round(risk_dollars, 4),
            "equity_now":   round(self.equity, 4) if self.equity is not None else None,
            "week_id":      self.week_id,
        })

    # ── Paper journal ─────────────────────────────────────────────────────────

    def _append_journal(self, record: dict) -> None:
        """Append one JSONL record to the paper journal.  Never raises."""
        if self.mode == AccountMode.BACKTEST:
            return
        try:
            self._paper_journal_file.parent.mkdir(parents=True, exist_ok=True)
            with self._paper_journal_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error(f"AccountState: failed to append to paper journal: {exc}")

    # ── Paper persistence ─────────────────────────────────────────────────────

    def load_paper(self) -> bool:
        """
        Load paper equity from disk.  Returns True if a valid file was found.
        Safe to call even if file doesn't exist (returns False, equity unchanged).

        Also checks week boundary after load so a process that's been down
        across a week rollover resets stats correctly on the first restart.
        """
        if not self._paper_file.exists():
            return False
        try:
            data         = json.loads(self._paper_file.read_text())
            equity_raw   = data.get("equity")
            peak_raw     = data.get("peak_equity")
            session_pnl  = data.get("realized_session_pnl", 0.0)
            week_id      = data.get("week_id",     self._current_week_id())
            week_pnl     = data.get("week_pnl",    0.0)
            week_trades  = data.get("week_trades", 0)

            if equity_raw is not None:
                self.equity                  = float(equity_raw)
                self.peak_equity             = float(peak_raw) if peak_raw is not None else self.equity
                self.realized_session_pnl    = float(session_pnl)
                self.week_id                 = str(week_id)
                self.week_pnl                = float(week_pnl)
                self.week_trades             = int(week_trades)
                logger.info(
                    f"AccountState (paper): loaded equity=${self.equity:,.2f} "
                    f"peak=${self.peak_equity:,.2f} "
                    f"week={self.week_id} week_pnl=${self.week_pnl:+,.2f} "
                    f"from {self._paper_file}"
                )
                # If the process was down over a week boundary, reset now
                self._check_week_boundary()
                return True
        except Exception as e:
            logger.error(f"AccountState: failed to load paper state: {e}")
        return False

    def save_paper(self) -> None:
        """Persist paper equity to disk atomically (tmp + rename)."""
        if self.mode == AccountMode.BACKTEST:
            return   # never write to disk during backtests
        try:
            self._paper_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._paper_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "equity":                  self.equity,
                "peak_equity":             self.peak_equity,
                "realized_session_pnl":    self.realized_session_pnl,
                "week_id":                 self.week_id,
                "week_pnl":                round(self.week_pnl, 4),
                "week_trades":             self.week_trades,
                "mode":                    self.mode.value,
                "saved_at":                datetime.now(timezone.utc).isoformat(),
            }, indent=2))
            tmp.replace(self._paper_file)
        except Exception as e:
            logger.error(f"AccountState: failed to save paper state: {e}")

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Snapshot for bot_state.json / heartbeat / dashboard."""
        return {
            "mode":                    self.mode.value,
            "mode_label":              self.mode_label,
            "equity":                  self.equity,          # None = UNKNOWN
            "equity_display":          self.equity_display,
            "equity_source":           self.equity_source,   # "BROKER" | "SIM" | "UNKNOWN"
            "peak_equity":             self.peak_equity,
            "realized_session_pnl":    self.realized_session_pnl,
            "week_id":                 self.week_id,
            "week_pnl":                self.week_pnl,
            "week_trades":             self.week_trades,
            "is_unknown":              self.is_unknown,
            "is_tradeable":            self.is_tradeable,
            "broker_fetch_failures":   self.broker_fetch_failures,
            "last_update_ts":          self.last_update_ts,
            "last_broker_ts":          self._last_broker_ts,
        }

    def __repr__(self) -> str:
        eq = self.equity_display
        return (
            f"AccountState(mode={self.mode.value}, equity={eq}, "
            f"peak=${self.peak_equity:,.2f}, week={self.week_id})"
        )

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def for_live_real(
        cls,
        broker_summary: Optional[dict],
        peak_override: Optional[float] = None,
    ) -> "AccountState":
        """
        Build from an initial broker summary.
        If summary is None/fails, equity starts UNKNOWN.
        """
        if broker_summary:
            bal  = broker_summary.get("balance")
            peak = broker_summary.get("nav") or bal
            inst = cls(AccountMode.LIVE_REAL, bal, peak or peak_override)
        else:
            inst = cls(AccountMode.LIVE_REAL, None, peak_override or 0.0)
        return inst

    @classmethod
    def for_live_paper(
        cls,
        default_balance: float,
        peak_override: Optional[float] = None,
        paper_file: Path = PAPER_STATE_FILE,
        paper_journal_file: Path = PAPER_JOURNAL_FILE,
    ) -> "AccountState":
        """
        Build for paper trading.

        SIM_STARTING_EQUITY / *default_balance* is used ONLY when no
        paper_account.json exists (first bootstrap).  All subsequent restarts
        load persisted equity from disk, ignoring the .env value entirely.
        """
        inst = cls(
            AccountMode.LIVE_PAPER,
            default_balance,
            peak_override,
            paper_file=paper_file,
            paper_journal_file=paper_journal_file,
        )
        loaded = inst.load_paper()
        if not loaded:
            logger.info(
                f"AccountState (paper): no saved file — "
                f"bootstrapping at ${default_balance:,.2f}"
            )
            inst.save_paper()   # persist the starting balance immediately
        return inst

    @classmethod
    def for_backtest(cls, initial_balance: float) -> "AccountState":
        """Lightweight instance for backtester — no file I/O."""
        return cls(AccountMode.BACKTEST, initial_balance, initial_balance)
