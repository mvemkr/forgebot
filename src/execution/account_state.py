"""
src/execution/account_state.py
===============================
AccountMode + AccountState — single source of truth for equity and execution mode.

Three mutually exclusive modes:

  LIVE_REAL   real OANDA account; equity pulled from broker each tick.
              If broker fetch fails → equity = UNKNOWN (None).
              Bot blocks new entries when UNKNOWN; existing positions keep running.

  LIVE_PAPER  no real orders; equity tracked internally and persisted to disk.
              Equity never becomes UNKNOWN — starts from DRY_RUN_PAPER_BALANCE
              (or last persisted value) and is updated after each simulated close.

  BACKTEST    offline backtester; AccountState not used for live file I/O.

Design rules
------------
- equity=None is an explicit sentinel meaning "broker fetch failed".  Never default to 0.
- peak_equity only ever increases (watermark logic).
- Paper mode writes equity to PAPER_STATE_FILE after every PnL event so a crash/restart
  picks up where it left off.
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

PAPER_STATE_FILE = Path.home() / "trading-bot" / "runtime_state" / "paper_account.json"
_UNKNOWN_DISPLAY = "UNKNOWN"


# ── Mode enum ─────────────────────────────────────────────────────────────────

class AccountMode(Enum):
    LIVE_REAL  = "live_real"   # real broker account
    LIVE_PAPER = "live_paper"  # simulated equity, no real orders
    BACKTEST   = "backtest"    # offline backtester


# ── AccountState ──────────────────────────────────────────────────────────────

class AccountState:
    """
    Tracks equity, peak, and sourcing for a single execution mode.

    Attributes
    ----------
    mode : AccountMode
    equity : Optional[float]
        Current equity.  None means UNKNOWN (broker unreachable in LIVE_REAL).
    peak_equity : float
        High-water mark.  Set to initial equity on first call; only increases.
    realized_session_pnl : float
        Cumulative realized PnL since last restart (paper mode tracking).
    broker_fetch_failures : int
        Consecutive broker-fetch failures (LIVE_REAL only); reset on success.
    """

    def __init__(
        self,
        mode: AccountMode,
        initial_equity: Optional[float],
        peak_equity: Optional[float] = None,
        paper_file: Path = PAPER_STATE_FILE,
    ):
        self.mode                   = mode
        self.equity: Optional[float] = initial_equity
        self.peak_equity: float      = peak_equity or initial_equity or 0.0
        self.realized_session_pnl: float = 0.0
        self.broker_fetch_failures: int  = 0
        self._paper_file             = Path(paper_file)
        self._last_broker_ts: Optional[str] = None
        self.last_update_ts: Optional[str]  = (
            datetime.now(timezone.utc).isoformat() if initial_equity is not None else None
        )

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

    # ── LIVE_REAL update path ─────────────────────────────────────────────────

    def update_from_broker(self, summary: dict) -> bool:
        """
        Apply a successful broker summary dict.
        Sets equity, updates peak, resets failure counter.
        Returns True on success, False on missing balance key.

        NO-OP in LIVE_PAPER mode — paper equity is never overwritten by the broker.
        This guard prevents accidental equity corruption if the method is called
        while in the wrong mode.
        """
        if self.mode != AccountMode.LIVE_REAL:
            logger.debug(
                f"AccountState.update_from_broker: no-op in {self.mode.value} mode"
            )
            return False

        bal = summary.get("balance")
        if bal is None:
            # Summary present but balance missing — treat as soft failure
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
        """
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

    def apply_pnl(self, realized_pnl: float, balance_after: Optional[float] = None) -> None:
        """
        Apply realized PnL to paper equity.

        If *balance_after* is provided (e.g. from backtester or a manual override),
        it takes precedence over the incremental calculation.

        Always updates peak.  Always persists to disk.
        """
        if self.mode not in (AccountMode.LIVE_PAPER, AccountMode.BACKTEST):
            logger.debug("AccountState.apply_pnl: not in paper mode — ignoring")
            return

        if balance_after is not None:
            self.equity = float(balance_after)
        elif self.equity is not None:
            self.equity = self.equity + realized_pnl
        else:
            # Should never happen for paper mode, but be safe
            logger.warning("AccountState.apply_pnl: equity was None in paper mode — ignoring")
            return

        self.realized_session_pnl += realized_pnl
        self.last_update_ts = datetime.now(timezone.utc).isoformat()

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.save_paper()
        logger.info(
            f"AccountState (paper): PnL ${realized_pnl:+,.2f} → equity "
            f"${self.equity:,.2f} | peak ${self.peak_equity:,.2f} | "
            f"session PnL ${self.realized_session_pnl:+,.2f}"
        )

    # ── Paper persistence ─────────────────────────────────────────────────────

    def load_paper(self) -> bool:
        """
        Load paper equity from disk.  Returns True if a valid file was found.
        Safe to call even if file doesn't exist (returns False, equity unchanged).
        """
        if not self._paper_file.exists():
            return False
        try:
            data = json.loads(self._paper_file.read_text())
            equity_raw = data.get("equity")
            peak_raw   = data.get("peak_equity")
            session_pnl = data.get("realized_session_pnl", 0.0)
            if equity_raw is not None:
                self.equity                  = float(equity_raw)
                self.peak_equity             = float(peak_raw) if peak_raw else self.equity
                self.realized_session_pnl    = float(session_pnl)
                logger.info(
                    f"AccountState (paper): loaded equity=${self.equity:,.2f} "
                    f"peak=${self.peak_equity:,.2f} from {self._paper_file}"
                )
                return True
        except Exception as e:
            logger.error(f"AccountState: failed to load paper state: {e}")
        return False

    def save_paper(self) -> None:
        """Persist paper equity to disk atomically."""
        if self.mode == AccountMode.BACKTEST:
            return   # never write to disk during backtests
        try:
            self._paper_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._paper_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "equity":                 self.equity,
                "peak_equity":            self.peak_equity,
                "realized_session_pnl":   self.realized_session_pnl,
                "mode":                   self.mode.value,
                "saved_at":               datetime.now(timezone.utc).isoformat(),
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
            "is_unknown":              self.is_unknown,
            "is_tradeable":            self.is_tradeable,
            "broker_fetch_failures":   self.broker_fetch_failures,
            "last_update_ts":          self.last_update_ts,
            "last_broker_ts":          self._last_broker_ts,
        }

    def __repr__(self) -> str:
        eq = self.equity_display
        return f"AccountState(mode={self.mode.value}, equity={eq}, peak=${self.peak_equity:,.2f})"

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
    ) -> "AccountState":
        """
        Build for paper trading.  Loads persisted equity if the file exists;
        otherwise starts from *default_balance*.
        """
        inst = cls(AccountMode.LIVE_PAPER, default_balance, peak_override, paper_file=paper_file)
        loaded = inst.load_paper()
        if not loaded:
            logger.info(
                f"AccountState (paper): no saved file — starting at ${default_balance:,.2f}"
            )
            inst.save_paper()   # persist the starting balance
        return inst

    @classmethod
    def for_backtest(cls, initial_balance: float) -> "AccountState":
        """Lightweight instance for backtester — no file I/O."""
        return cls(AccountMode.BACKTEST, initial_balance, initial_balance)
