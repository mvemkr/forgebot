"""
Forex Risk Manager

Handles:
  1. Auto-scaling risk % based on account balance (Scenario B tiers)
  2. Kill switch — REGROUP mode (pause + learn, NOT stop forever)
  3. Hard-block counter-trend trades
  4. Dual-trade book exposure management (max 2 simultaneous positions)

Risk tiers (Scenario B — agreed with Mike, 2026-02-21):
  < $8,000  → 10% risk per trade
  < $15,000 → 15% risk per trade
  < $30,000 → 20% risk per trade
  $30,000+  → 25% risk per trade

Kill switch (40% DD threshold — agreed 2026-02-21):
  When account drops 40%+ from rolling peak over 7 days:
    → Enter REGROUP mode (NOT a permanent halt)
    → Bot goes observation-only for COOLDOWN_DAYS (default 14)
    → Sends Mike a full analysis of what went wrong
    → Auto-resumes after cooldown unless Mike explicitly keeps it paused
    → This matches Alex's behaviour: step back, watch, rebuild coherence, re-engage

Dual-trade rules (agreed 2026-02-21 after Monte Carlo analysis):
  MAX_BOOK_EXPOSURE = 35% — total capital at risk across all open positions
  MAX_CONCURRENT_TRADES = 2
  MIN_SECOND_TRADE_PCT = 5% — don't open second trade if budget < 5%
  Currency overlap rule: no two open positions may share a currency
    (e.g. GBP/USD + EUR/USD both expose USD — blocked)
  Second trade risk = min(tier_rate, MAX_BOOK_EXPOSURE - trade1_risk_pct)
"""
import json
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple

from .trade_journal import TradeJournal

logger = logging.getLogger(__name__)

KILL_SWITCH_LOG = Path.home() / "trading-bot" / "logs" / "kill_switch.log"
REGROUP_STATE_FILE = Path.home() / "trading-bot" / "logs" / "regroup_state.json"


class BotMode(Enum):
    ACTIVE    = "active"       # normal trading
    REGROUP   = "regroup"      # cooldown — observe only, no new trades
    PAUSED    = "paused"       # Mike manually paused


class ForexRiskManager:
    """
    Risk controls for the Set & Forget strategy.

    Kill switch enters REGROUP mode (not a hard stop):
      - No new positions opened
      - Existing positions continue to be monitored
      - Sends daily "observations" to Mike during cooldown
      - Auto-resumes after COOLDOWN_DAYS
      - Mike can extend cooldown or resume early
    """

    # Scenario B risk tiers (agreed 2026-02-21)
    RISK_TIERS = [
        (8_000,          10.0),
        (15_000,         15.0),
        (30_000,         20.0),
        (float("inf"),   25.0),
    ]

    # Kill switch: 40% drawdown from 7-day rolling peak triggers regroup
    DRAWDOWN_THRESHOLD_PCT = 40.0
    DRAWDOWN_WINDOW_DAYS   = 7
    COOLDOWN_DAYS          = 14   # days in regroup before auto-resume

    # Dual-trade book limits (Monte Carlo validated 2026-02-21)
    MAX_BOOK_EXPOSURE    = 35.0   # % — max total risk across all open positions
    MAX_CONCURRENT_TRADES = 4     # hard cap on simultaneous at-risk positions (Alex runs 4+)
    MIN_SECOND_TRADE_PCT  = 5.0   # % — skip 2nd trade if budget below this

    def __init__(
        self,
        journal: TradeJournal,
        # Legacy params kept for compatibility — now baked into class constants
        weekly_drawdown_limit_pct: float = 40.0,
    ):
        self.journal = journal
        self._mode: BotMode = BotMode.ACTIVE
        self._regroup_reason: Optional[str] = None
        self._regroup_started: Optional[datetime] = None
        self._regroup_ends: Optional[datetime] = None
        self._peak_balance: float = 0.0

        KILL_SWITCH_LOG.parent.mkdir(parents=True, exist_ok=True)

        # Restore regroup state if bot was restarted mid-cooldown
        self._load_regroup_state()

    # ── Mode ───────────────────────────────────────────────────────────

    @property
    def mode(self) -> BotMode:
        return self._mode

    @property
    def is_halted(self) -> bool:
        """True when bot should not open new positions."""
        return self._mode in (BotMode.REGROUP, BotMode.PAUSED)

    @property
    def in_regroup(self) -> bool:
        return self._mode == BotMode.REGROUP

    @property
    def regroup_ends(self) -> Optional[datetime]:
        return self._regroup_ends

    @property
    def regroup_reason(self) -> Optional[str]:
        return self._regroup_reason

    # ── Risk Scaling ───────────────────────────────────────────────────

    def get_risk_pct(self, account_balance: float) -> float:
        """Base tier risk — no drawdown adjustment. Use get_risk_pct_with_dd for live trading."""
        for max_bal, risk_pct in self.RISK_TIERS:
            if account_balance < max_bal:
                logger.debug(f"Risk tier: {risk_pct}% (balance ${account_balance:,.0f})")
                return risk_pct
        return 25.0  # fallback

    def get_risk_pct_with_dd(self, account_balance: float,
                              peak_equity: Optional[float] = None) -> tuple[float, str]:
        """
        Returns (final_risk_pct, reason_flag) with graduated drawdown caps applied.

        DD ≥ DD_L2_PCT (25%) → cap at DD_L2_CAP (6%)   flag: DD_CAP_6
        DD ≥ DD_L1_PCT (15%) → cap at DD_L1_CAP (10%)  flag: DD_CAP_10
        Otherwise             → full tier risk           flag: ""

        Caps lift only when equity recovers above DD_RESUME_PCT (95%) of peak.
        peak_equity defaults to self._peak_balance when not supplied.
        """
        from src.strategy.forex.strategy_config import (DD_CIRCUIT_BREAKER_ENABLED,
                                                         DD_L1_PCT, DD_L1_CAP,
                                                         DD_L2_PCT, DD_L2_CAP,
                                                         DD_RESUME_PCT)
        base = self.get_risk_pct(account_balance)

        if not DD_CIRCUIT_BREAKER_ENABLED:
            return base, ""

        peak = peak_equity if peak_equity is not None else self._peak_balance
        if peak <= 0:
            return base, ""

        dd = (peak - account_balance) / peak   # 0.0 → 1.0

        if dd >= DD_L2_PCT / 100:
            capped = min(base, DD_L2_CAP)
            logger.warning(f"DD circuit breaker L2: DD={dd:.1%} ≥ {DD_L2_PCT}% "
                           f"→ risk capped {base:.1f}%→{capped:.1f}%")
            return capped, "DD_CAP_6"

        if dd >= DD_L1_PCT / 100:
            capped = min(base, DD_L1_CAP)
            logger.warning(f"DD circuit breaker L1: DD={dd:.1%} ≥ {DD_L1_PCT}% "
                           f"→ risk capped {base:.1f}%→{capped:.1f}%")
            return capped, "DD_CAP_10"

        return base, ""

    def get_tier_label(self, account_balance: float) -> str:
        # Note: use &lt; not < — these labels are sent via Telegram HTML parse mode
        # and bare < is treated as an opening HTML tag, causing a 400 parse error.
        labels = {
            10.0: "Tier 1 (&lt;$8K) — 10%",
            15.0: "Tier 2 (&lt;$15K) — 15%",
            20.0: "Tier 3 (&lt;$30K) — 20%",
            25.0: "Tier 4 ($30K+) — 25%",
        }
        return labels.get(self.get_risk_pct(account_balance), "?")

    # ── Kill Switch / Regroup ──────────────────────────────────────────

    def check_and_update_mode(self, current_balance: float) -> Tuple[BotMode, str]:
        """
        Run all mode checks. Returns (mode, reason).
        Call at the start of every orchestrator cycle.

        Logic:
          1. If PAUSED — stay paused (Mike must manually resume)
          2. If REGROUP — check if cooldown expired → auto-resume
          3. If ACTIVE — check drawdown → enter regroup if triggered
        """
        if self._mode == BotMode.PAUSED:
            return BotMode.PAUSED, "Manually paused by Mike"

        if self._mode == BotMode.REGROUP:
            return self._check_regroup_expiry(current_balance)

        # Active — check if we need to regroup
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance

        triggered, reason = self._check_drawdown(current_balance)
        if triggered:
            self._enter_regroup(reason, current_balance)
            return BotMode.REGROUP, reason

        return BotMode.ACTIVE, ""

    # Legacy alias so orchestrator.py still works unchanged
    def check_and_halt_if_needed(self, current_balance: float) -> Tuple[bool, str]:
        mode, reason = self.check_and_update_mode(current_balance)
        return mode != BotMode.ACTIVE, reason

    def _check_drawdown(self, current_balance: float) -> Tuple[bool, str]:
        """Compare current balance to 7-day peak."""
        weekly_pnl = self.journal.get_weekly_pnl()
        if not weekly_pnl:
            return False, ""

        now    = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.DRAWDOWN_WINDOW_DAYS)

        recent_pnl = 0.0
        for week_key, pnl in weekly_pnl.items():
            try:
                year, week = week_key.split("-W")
                week_start = datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")
                week_start = week_start.replace(tzinfo=timezone.utc)
                if week_start >= cutoff:
                    recent_pnl += pnl
            except Exception:
                pass

        if recent_pnl < 0:
            approx_start   = current_balance - recent_pnl
            drawdown_pct   = abs(recent_pnl) / approx_start * 100
            if drawdown_pct >= self.DRAWDOWN_THRESHOLD_PCT:
                reason = (
                    f"Account down ${abs(recent_pnl):,.0f} "
                    f"({drawdown_pct:.1f}%) in {self.DRAWDOWN_WINDOW_DAYS} days. "
                    f"Threshold: {self.DRAWDOWN_THRESHOLD_PCT}%."
                )
                return True, reason

        return False, ""

    def _check_regroup_expiry(self, current_balance: float) -> Tuple[BotMode, str]:
        """Check if the cooldown period has elapsed → auto-resume."""
        now = datetime.now(timezone.utc)
        if self._regroup_ends and now >= self._regroup_ends:
            days_in_regroup = (now - self._regroup_started).days if self._regroup_started else self.COOLDOWN_DAYS
            self._exit_regroup(
                f"Cooldown complete ({days_in_regroup} days). Resuming live trading. "
                f"Current balance: ${current_balance:,.2f}"
            )
            return BotMode.ACTIVE, "Regroup complete — auto-resumed"

        remaining = (self._regroup_ends - now) if self._regroup_ends else timedelta(days=0)
        days_left = remaining.days + (1 if remaining.seconds > 0 else 0)
        return BotMode.REGROUP, f"In regroup cooldown — {days_left} day(s) remaining"

    def _enter_regroup(self, reason: str, balance: float):
        now = datetime.now(timezone.utc)
        self._mode            = BotMode.REGROUP
        self._regroup_reason  = reason
        self._regroup_started = now
        self._regroup_ends    = now + timedelta(days=self.COOLDOWN_DAYS)

        msg = (
            f"🟡 REGROUP MODE ACTIVATED\n"
            f"Reason: {reason}\n"
            f"Balance: ${balance:,.2f}\n"
            f"Cooldown: {self.COOLDOWN_DAYS} days (auto-resume: "
            f"{self._regroup_ends.strftime('%b %d')})\n\n"
            f"What this means:\n"
            f"  • No new positions will be opened\n"
            f"  • Existing positions are still monitored\n"
            f"  • Bot will scan and observe daily — rebuilding coherence\n"
            f"  • Auto-resumes in {self.COOLDOWN_DAYS} days\n\n"
            f"To resume early: message the bot 'resume trading'\n"
            f"To extend cooldown: message 'extend cooldown X days'"
        )
        logger.critical(f"🟡 REGROUP: {reason}")
        self._log_kill_switch(reason, f"Entering {self.COOLDOWN_DAYS}-day regroup. Balance: ${balance:,.2f}")
        self.journal.log_kill_switch(reason, "Regroup mode entered", balance)
        self._save_regroup_state()
        return msg  # returned so orchestrator can send to Telegram

    def _exit_regroup(self, reason: str):
        logger.info(f"✅ Regroup complete: {reason}")
        self._log_kill_switch("REGROUP ENDED", reason)
        self._mode            = BotMode.ACTIVE
        self._regroup_reason  = None
        self._regroup_started = None
        self._regroup_ends    = None
        self._save_regroup_state()

    def resume_early(self, reason: str = "Manual resume by Mike"):
        """Mike explicitly resumes trading before cooldown expires."""
        logger.warning(f"Early resume: {reason}")
        self._exit_regroup(reason)

    def pause(self, reason: str = "Manually paused"):
        """Mike manually pauses the bot (indefinite — must manually resume)."""
        self._mode = BotMode.PAUSED
        self._regroup_reason = reason
        logger.warning(f"Bot PAUSED: {reason}")
        self._save_regroup_state()

    def unpause(self):
        """Mike manually unpauses."""
        self._mode = BotMode.ACTIVE
        self._regroup_reason = None
        logger.info("Bot unpaused → ACTIVE")
        self._save_regroup_state()

    # ── State Persistence ──────────────────────────────────────────────

    def _save_regroup_state(self):
        state = {
            "mode": self._mode.value,
            "regroup_reason": self._regroup_reason,
            "regroup_started": self._regroup_started.isoformat() if self._regroup_started else None,
            "regroup_ends":    self._regroup_ends.isoformat()    if self._regroup_ends    else None,
            "peak_balance":    self._peak_balance,
        }
        try:
            REGROUP_STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.warning(f"Could not save regroup state: {e}")

    def _load_regroup_state(self):
        if not REGROUP_STATE_FILE.exists():
            return
        try:
            state = json.loads(REGROUP_STATE_FILE.read_text())
            self._mode            = BotMode(state.get("mode", "active"))
            self._regroup_reason  = state.get("regroup_reason")
            self._peak_balance    = state.get("peak_balance", 0.0)
            rs = state.get("regroup_started")
            re = state.get("regroup_ends")
            self._regroup_started = datetime.fromisoformat(rs) if rs else None
            self._regroup_ends    = datetime.fromisoformat(re) if re else None
            if self._mode == BotMode.REGROUP:
                logger.warning(
                    f"Restored REGROUP state from disk. "
                    f"Reason: {self._regroup_reason} | "
                    f"Ends: {self._regroup_ends}"
                )
        except Exception as e:
            logger.warning(f"Could not load regroup state: {e}")

    def _log_kill_switch(self, reason: str, action: str):
        ts = datetime.now(timezone.utc).isoformat()
        with open(KILL_SWITCH_LOG, "a") as f:
            f.write(f"[{ts}] {reason} | ACTION: {action}\n")

    # ── Counter-Trend Guard ────────────────────────────────────────────

    def is_counter_trend(
        self,
        trade_direction: str,
        trend_weekly: str,
        trend_daily: str,
    ) -> Tuple[bool, str]:
        """
        Weekly trend MUST align with trade direction — non-negotiable.
        Daily can pull back against weekly (that's the entry setup).
        """
        bullish_weekly = trend_weekly in ("bullish", "strong_bullish")
        bearish_weekly = trend_weekly in ("bearish", "strong_bearish")

        if trade_direction == "long" and bearish_weekly:
            return True, (
                f"⛔ COUNTER-TREND BLOCKED: LONG but weekly is {trend_weekly}. "
                f"#1 cause of plan-break losses."
            )
        if trade_direction == "short" and bullish_weekly:
            return True, (
                f"⛔ COUNTER-TREND BLOCKED: SHORT but weekly is {trend_weekly}. "
                f"#1 cause of plan-break losses."
            )
        return False, ""

    # ── One-Trade-At-A-Time ────────────────────────────────────────────

    # ── Dual-trade book management ─────────────────────────────────────

    @staticmethod
    def pair_currencies(pair: str) -> set:
        """Return the two currency codes for a pair. 'GBP/USD' → {'GBP','USD'}."""
        parts = pair.replace("_", "/").upper().split("/")
        return set(parts) if len(parts) == 2 else set()

    def currencies_in_use(self, open_positions: dict) -> set:
        """Return the set of all currencies currently tied up in open positions."""
        in_use = set()
        for p in open_positions:
            in_use |= self.pair_currencies(p)
        return in_use

    def committed_book_pct(self, open_positions: dict) -> float:
        """
        Return the total % of account currently committed across open positions.
        Uses the stored risk_pct per position; falls back to current tier if missing.
        """
        return sum(
            pos.get("risk_pct", 0.0) for pos in open_positions.values()
        )

    def get_book_risk_pct(self, account_balance: float, open_positions: dict) -> float:
        """
        Return the appropriate risk % for the NEXT trade given current book exposure.

        - First trade:  full tier rate (10/15/20/25%)
        - Second trade: min(tier_rate, MAX_BOOK_EXPOSURE - committed_pct)
        - Returns 0.0 if budget is below MIN_SECOND_TRADE_PCT (caller should block)
        """
        tier_rate  = self.get_risk_pct(account_balance)
        committed  = self.committed_book_pct(open_positions)
        budget     = self.MAX_BOOK_EXPOSURE - committed

        if budget < self.MIN_SECOND_TRADE_PCT:
            return 0.0   # not enough room for a meaningful second trade

        return min(tier_rate, budget)

    @staticmethod
    def _open_position_theme_gate(
        pair: str,
        direction: str,
        open_positions: dict,
        threshold: int = 2,
    ) -> Tuple[bool, str]:
        """
        Open-position theme contradiction gate.

        If ≥ threshold open positions all express the same directional bias on a
        currency, that IS the theme — regardless of price momentum scores.
        Block any new position that expresses the OPPOSITE bias on that currency.

        Key example: GBP/JPY SHORT + USD/JPY SHORT both BUY JPY → JPY theme = LONG.
        NZD/JPY LONG would SELL JPY → contradiction → blocked.
        NZD/JPY SHORT would BUY JPY → aligns → allowed (this is the "push to short").

        threshold=2 means 2 confirming positions establish the theme.
        Returns (blocked: bool, reason: str).
        """
        if "/" not in pair:
            return False, ""

        # Build currency bias from open positions: +N = that currency is being bought
        from collections import Counter
        currency_bias: Counter = Counter()
        for op, pos in open_positions.items():
            if "/" not in op:
                continue
            ob, oq = op.split("/")
            d = pos.get("direction", "")
            # SHORT = sell base, buy quote  →  base -1, quote +1
            # LONG  = buy base, sell quote  →  base +1, quote -1
            bias = 1 if d == "long" else -1
            currency_bias[ob] += bias
            currency_bias[oq] -= bias

        # Only gate on macro carry/safe-haven currencies.
        # USD, EUR, GBP selling in two pairs is coincidental (USD/JPY + USD/CHF
        # both sell USD but that's JPY strong + CHF strong — not a "USD is crashing"
        # theme). Applying the gate to USD would block valid Alex trades like
        # EUR/USD SHORT when USD/JPY + USD/CHF are running.
        MACRO_THEME_CCYS = {"JPY", "CHF"}

        base, quote = pair.split("/")
        new_bias = 1 if direction == "long" else -1   # +1 = buying base, -1 = selling base

        for ccy, trade_impact in [(base, new_bias), (quote, -new_bias)]:
            if ccy not in MACRO_THEME_CCYS:
                continue
            existing = currency_bias.get(ccy, 0)
            if existing >= threshold and trade_impact < 0:
                return True, (
                    f"⛔ THEME CONTRADICTION: {ccy} is being bought in "
                    f"{existing} open positions — new {direction} {pair} would sell it."
                )
            if existing <= -threshold and trade_impact > 0:
                return True, (
                    f"⛔ THEME CONTRADICTION: {ccy} is being sold in "
                    f"{abs(existing)} open positions — new {direction} {pair} would buy it."
                )

        return False, ""

    def check_entry_eligibility(
        self,
        pair: str,
        open_positions: dict,
        account_balance: float,
        is_macro_theme_pair: bool = False,
        direction: str = "",
    ) -> Tuple[bool, str]:
        """
        Gate check for opening a new position.  Four layered rules:

          1. Max concurrent positions (hard cap = MAX_CONCURRENT_TRADES)
             EXCEPTION: macro theme pairs → cap raised to STACK_MAX (4)
          2. Same-pair block — can't enter the same instrument twice
          3. Book exposure — remaining budget must be ≥ MIN_SECOND_TRADE_PCT
          4. Open-position theme gate — if ≥2 positions express the same
             directional bias on a currency, block contradicting new trades.
             E.g. GBP/JPY SHORT + USD/JPY SHORT → JPY theme = LONG →
             blocks NZD/JPY LONG, allows NZD/JPY SHORT.

        is_macro_theme_pair: set by set_and_forget.evaluate() when a macro
        currency theme is active and this pair is one of the stacked trades.
        Matches the backtester's _entry_eligible() exception exactly.

        Returns (blocked: bool, reason: str).
        """
        from src.strategy.forex.currency_strength import STACK_MAX
        n = len(open_positions)

        # 1 — Hard cap (STACK_MAX for theme, MAX_CONCURRENT_TRADES for non-theme).
        # BE positions are risk-free — don't count them toward the limit.
        # Alex takes new trades when existing ones hit BE (locked in profit = free slot).
        def _pos_at_be(pos: dict) -> bool:
            entry = pos.get("entry", 0)
            stop  = pos.get("stop", pos.get("stop_loss", 0))
            direction = pos.get("direction", "")
            if direction == "short": return stop <= entry
            if direction == "long":  return stop >= entry
            return False

        if is_macro_theme_pair:
            at_risk_count = sum(1 for p in open_positions if not _pos_at_be(open_positions[p]))
            max_allowed = STACK_MAX
            if at_risk_count >= max_allowed:
                return True, (
                    f"⛔ MAX POSITIONS: {at_risk_count} at-risk trades open "
                    f"(max {max_allowed} [theme stack]). Waiting for one to close or hit BE."
                )
        else:
            # Non-theme: only count non-BE, non-theme positions
            at_risk_count = sum(
                1 for p, pos in open_positions.items()
                if not pos.get("macro_theme") and not _pos_at_be(pos)
            )
            max_allowed = self.MAX_CONCURRENT_TRADES
            if at_risk_count >= max_allowed:
                return True, (
                    f"⛔ MAX POSITIONS: {at_risk_count} non-BE positions open "
                    f"(max {max_allowed}). Waiting for one to close or hit BE."
                )

        if n == 0:
            return False, ""   # first trade — no further checks needed

        # 2 — Same-pair block only (currency overlap removed — Alex stacks GBP/JPY + GBP/CHF,
        # USD/JPY + NZD/JPY simultaneously. Broad currency overlap was blocking valid setups.
        # Only hard rule: don't enter the same instrument twice.)
        if pair in open_positions:
            return True, (
                f"⛔ SAME PAIR OPEN: {pair} already has an open position. "
                f"Can't enter the same instrument twice."
            )

        # 3 — Book budget
        book_risk = self.get_book_risk_pct(account_balance, open_positions)
        if book_risk <= 0:
            committed = self.committed_book_pct(open_positions)
            return True, (
                f"⛔ BOOK FULL: {committed:.1f}% already committed — "
                f"only {self.MAX_BOOK_EXPOSURE - committed:.1f}% remaining, "
                f"below {self.MIN_SECOND_TRADE_PCT:.0f}% minimum. "
                f"Waiting for a position to close."
            )

        # 4 — Open-position theme gate (only when direction is known)
        if direction:
            blocked, reason = self._open_position_theme_gate(pair, direction, open_positions)
            if blocked:
                return True, reason

        return False, ""

    def check_position_count(self, open_positions: dict) -> Tuple[bool, str]:
        """
        Legacy belt-and-suspenders check kept for backward compatibility.
        Prefer check_entry_eligibility() for full dual-trade logic.
        """
        if len(open_positions) >= self.MAX_CONCURRENT_TRADES:
            pairs = list(open_positions.keys())
            return True, f"Max positions ({self.MAX_CONCURRENT_TRADES}) reached: {pairs}."
        return False, ""

    # ── Status ─────────────────────────────────────────────────────────

    def status(self, account_balance: float) -> Dict:
        return {
            "mode":           self._mode.value,
            "is_halted":      self.is_halted,
            "regroup_reason": self._regroup_reason,
            "regroup_ends":   self._regroup_ends.isoformat() if self._regroup_ends else None,
            "risk_pct":       self.get_risk_pct(account_balance),
            "tier_label":     self.get_tier_label(account_balance),
            "account_balance": account_balance,
            "peak_balance":   self._peak_balance,
            "drawdown_threshold_pct": self.DRAWDOWN_THRESHOLD_PCT,
        }
