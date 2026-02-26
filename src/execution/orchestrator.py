"""
Forex Bot Orchestrator — The main loop.

Schedule:
  Every 1H (London/NY sessions):
    → Fetch 1H data for watchlist pairs
    → Run strategy.evaluate() on each pair
    → Execute ENTER decisions (auto-approved if London overnight)

  Every 4H:
    → PositionMonitor.check_all() — stop sync, breakeven, exit signals

  Every 6H:
    → Send standings update to Mike (balance, P&L, positions, mode)

  Sunday 20:00 ET:
    → WeeklyScanner across all pairs → send weekly brief

  Daily 08:00 ET:
    → Morning brief

  Every cycle:
    → Write heartbeat file
    → Check mode (active / regroup / paused)

Overnight auto-execute:
    London session (3AM–8AM ET, 08:00–13:00 UTC) trades auto-execute
    per Mike's standing pre-approval (2026-02-21).
    Mike is notified immediately via Telegram with full trade details.

Regroup mode:
    When kill switch fires (40% DD in 7 days):
    → Bot enters REGROUP (not a hard halt)
    → No new positions opened
    → Existing positions monitored as normal
    → Daily observations sent to Mike
    → Auto-resumes after 14 days (or Mike resumes early)
"""
import os
import time
import logging
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv(Path.home() / "trading-bot" / ".env")

LOG_DIR = Path.home() / "trading-bot" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "forex_orchestrator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("orchestrator")

HEARTBEAT_FILE  = LOG_DIR / "forex_orchestrator.heartbeat"
CONTROL_FILE    = LOG_DIR / "bot_control.json"   # Mike → Forge → bot command relay
WHITELIST_FILE  = LOG_DIR / "whitelist.json"      # pair whitelist (persistent, UI-managed)

WATCHLIST = [
    "USD/JPY", "GBP/CHF", "USD/CHF", "USD/CAD", "GBP/JPY",
    "EUR/USD", "GBP/USD", "NZD/USD", "GBP/NZD", "EUR/GBP",
    "AUD/USD", "NZD/JPY",
]

# ── Shared execution config (single source of truth) ─────────────────────────
# Imported from strategy_config — same values used by the backtester.
# To change any threshold, edit strategy_config.py. Never hardcode here.
from ..strategy.forex import strategy_config as _sc   # module-ref for lever access
from ..strategy.forex.strategy_config import (
    MIN_CONFIDENCE,
    BLOCK_ENTRY_WHILE_WINNER_RUNNING,
    ATR_STOP_MULTIPLIER,
    ATR_MIN_MULTIPLIER,
    ATR_LOOKBACK,
    MAX_CONCURRENT_TRADES,
    LONDON_SESSION_START_UTC,
    LONDON_SESSION_END_UTC,
    STOP_COOLDOWN_DAYS,
    DRY_RUN_PAPER_BALANCE,
    winner_rule_check,
)


class ForexOrchestrator:
    """
    Main bot loop for the Set & Forget Forex strategy.

    dry_run : bool
        If True (default), no live orders placed. Log + simulate only.
    """

    def __init__(self, dry_run: bool = True):
        from ..exchange.oanda_client import OandaClient
        from ..strategy.forex.set_and_forget import SetAndForgetStrategy
        from ..strategy.forex.weekly_scanner import WeeklyScanner
        from .trade_journal import TradeJournal
        from .risk_manager_forex import ForexRiskManager, BotMode
        from .trade_executor import TradeExecutor
        from .position_monitor import PositionMonitor
        from .notifier import Notifier
        from .bot_state import BotState

        self.dry_run  = dry_run
        self.BotMode  = BotMode
        mode_label    = "DRY RUN" if dry_run else "⚠️  LIVE TRADING"
        logger.info(f"Initializing ForexOrchestrator [{mode_label}]")

        self.oanda    = OandaClient()
        self.journal  = TradeJournal()
        self.notifier = Notifier()
        self.state    = BotState()

        try:
            summary              = self.oanda.get_account_summary()
            real_balance         = summary.get("balance", 0.0)
            self.account_nav     = summary.get("nav", real_balance)
            self.unrealized_pnl  = summary.get("unrealized_pnl", 0.0)
            # In dry_run with an unfunded account ($0), use a paper balance so
            # risk sizing and kill-switch math work correctly.
            if dry_run and real_balance < 100.0:
                self.account_balance = DRY_RUN_PAPER_BALANCE
                logger.info(
                    f"DRY RUN — OANDA balance ${real_balance:.2f} (unfunded). "
                    f"Using paper balance ${DRY_RUN_PAPER_BALANCE:,.0f} for sizing."
                )
            else:
                self.account_balance = real_balance
                logger.info(f"Account balance: ${self.account_balance:,.2f}")
        except Exception as e:
            logger.error(f"Could not fetch account balance: {e}")
            self.account_balance = DRY_RUN_PAPER_BALANCE if dry_run else 4000.0
            self.account_nav     = self.account_balance
            self.unrealized_pnl  = 0.0

        self.risk             = ForexRiskManager(self.journal)

        # ── Dry-run peak isolation ─────────────────────────────────────
        # If running in dry_run with a paper balance, the persisted peak may
        # be from a previous live/practice run (e.g., OANDA practice $10K).
        # Inheriting that peak permanently caps risk during development and
        # makes testing misleading.  Reset peak to the paper balance so DD
        # logic only measures this bot's simulated performance.
        if dry_run and self.risk._peak_balance > self.account_balance * 1.02:
            old_peak = self.risk._peak_balance
            self.risk._peak_balance = self.account_balance
            self.risk._save_regroup_state()
            logger.info(
                f"DRY RUN — reset inherited peak ${old_peak:,.0f} → "
                f"${self.account_balance:,.0f} (paper balance). "
                f"DD tracking now starts from scratch."
            )

        initial_risk_pct      = self.risk.get_risk_pct(self.account_balance)

        self.strategy = SetAndForgetStrategy(
            account_balance=self.account_balance,
            risk_pct=initial_risk_pct,
        )

        self.executor = TradeExecutor(
            strategy=self.strategy,
            oanda=self.oanda,
            journal=self.journal,
            risk_manager=self.risk,
            dry_run=dry_run,
        )

        self.monitor = PositionMonitor(
            strategy=self.strategy,
            oanda=self.oanda,
            journal=self.journal,
            notifier=self.notifier,
            dry_run=dry_run,
        )

        self.scanner = WeeklyScanner(self.strategy)

        from .trade_analyzer import TradeAnalyzer
        self.analyzer = TradeAnalyzer(notifier=self.notifier)

        self._last_hourly:         Optional[datetime] = None
        self._last_4h:             Optional[datetime] = None
        self._last_weekly:         Optional[datetime] = None
        self._last_daily_brief:    Optional[datetime] = None
        self._last_standings:      Optional[datetime] = None
        self._last_regroup_obs:    Optional[datetime] = None
        self._last_monthly_report: Optional[datetime] = None
        self._prev_mode:           Optional[str]      = None

        # Per-pair confluence state — written every scan, read by dashboard.
        # Shows what each pair is waiting for before the bot will enter.
        self._confluence_state:    Dict[str, dict]   = {}

        # Consecutive loss streak — used by risk manager for streak brake cap.
        # Loaded from saved state on startup; updated when positions close.
        # A ratchet exit near 0R counts as a win (doesn't increment streak).
        self._consecutive_losses:  int               = 0

        # ── Crash recovery — restore full state from last save ───────────
        # Reconciles saved state with OANDA live data:
        #   - Open positions: restores full context (entry, stop, direction,
        #     pattern, trends, confidence, entry_reason, tier — everything)
        #   - Pattern memory: marks exhausted patterns so we don't re-enter
        # If the bot restarts mid-trade, it picks up exactly where it left off.
        try:
            recovery = self.state.reconcile_with_oanda(self.oanda, self.strategy)
            if recovery.get("recovered"):
                n_pos = len(recovery.get("recovered_positions", {}))
                age_m = int(recovery.get("state_age_seconds", 0) / 60)
                if n_pos:
                    logger.info(
                        f"✅ Recovered {n_pos} open position(s) from state "
                        f"({age_m}m ago): {list(recovery['recovered_positions'].keys())}"
                    )
                    self.notifier.send(
                        f"♻️ Bot restarted — recovered {n_pos} open position(s) from saved state "
                        f"({age_m}m ago):\n" +
                        "\n".join(
                            f"  • {p}: {v.get('direction','?')} @ {v.get('entry','?'):.5f}  "
                            f"SL={v.get('stop','?'):.5f}  pattern={v.get('pattern_type','?')}"
                            for p, v in recovery["recovered_positions"].items()
                        )
                    )
                else:
                    logger.info("State reconciled — no open positions to recover")
        except Exception as e:
            logger.warning(f"State recovery failed: {e}")

        # Restore pattern memory from last saved state
        try:
            saved = self.state.load()
            if saved:
                patterns = saved.get("stats", {}).get("traded_pattern_keys", [])
                if patterns:
                    restored = {k: "exhausted" for k in patterns}
                    self.strategy.restore_traded_patterns(restored)
                    logger.info(f"Restored {len(restored)} exhausted pattern(s) from state")
        except Exception as e:
            logger.warning(f"Pattern memory restore failed: {e}")

        # Restore consecutive loss streak from saved state
        try:
            saved = self.state.load()
            self._consecutive_losses = int(saved.get("stats", {}).get("consecutive_losses", 0))
            if self._consecutive_losses:
                logger.info(f"Restored consecutive_losses={self._consecutive_losses} from state")
        except Exception:
            pass

        # Load Tier 1 news calendar on startup
        try:
            self.strategy.news_filter.refresh_if_needed()
            upcoming = self.strategy.news_filter.format_upcoming(hours_ahead=48)
            logger.info(f"News calendar loaded. {upcoming}")
        except Exception as e:
            logger.warning(f"News calendar load failed: {e}")

        logger.info(
            f"Orchestrator ready. Risk: {initial_risk_pct}%/trade. "
            f"Watchlist: {len(WATCHLIST)} pairs. "
            f"Overnight auto-execute: London session {LONDON_SESSION_START_UTC}:00–"
            f"{LONDON_SESSION_END_UTC}:00 UTC"
        )

    # ── Main Loop ──────────────────────────────────────────────────────

    def run_forever(self):
        logger.info("🚀 Starting ForexOrchestrator main loop")
        self.notifier.send(
            f"🚀 Forex bot started ({'DRY RUN' if self.dry_run else '⚠️ LIVE'})\n"
            f"Balance: ${self.account_balance:,.2f} | "
            f"Risk/trade: {self.risk.get_risk_pct(self.account_balance):.0f}%\n"
            f"Kill switch: {self.risk.DRAWDOWN_THRESHOLD_PCT:.0f}% DD → Regroup mode\n"
            f"Overnight: London session trades auto-execute 🌙"
        )

        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt — shutting down")
                break
            except Exception as e:
                logger.error(f"Tick error: {e}", exc_info=True)

            self._write_heartbeat("sleeping")
            time.sleep(60)

    def _tick(self):
        now = datetime.now(timezone.utc)
        self._write_heartbeat("running")

        # Check control file for commands from Mike (relayed via Forge)
        self._check_control_file()

        # Save full state to disk (dashboard reads this)
        self._save_state()

        # Refresh balance
        try:
            summary    = self.oanda.get_account_summary()
            real_bal   = summary.get("balance", self.account_balance)
            # Dry run with unfunded account: always use paper balance so risk
            # sizing and kill-switch math work. Never let $0 bleed through.
            if self.dry_run and real_bal < 100.0:
                self.account_balance = DRY_RUN_PAPER_BALANCE
            else:
                self.account_balance = real_bal
            self.account_nav     = summary.get("nav",     self.account_nav)
            self.unrealized_pnl  = summary.get("unrealized_pnl", 0.0)
            self.strategy.update_balance(self.account_balance)
            self.strategy.risk_pct = self.risk.get_risk_pct(self.account_balance)
        except Exception as e:
            logger.warning(f"Balance refresh failed: {e}")

        # Mode check
        mode, mode_reason = self.risk.check_and_update_mode(self.account_balance)
        self._handle_mode_transitions(mode, mode_reason)

        # ── Regroup-only path ─────────────────────────────────────────
        if mode == self.BotMode.REGROUP:
            self._run_regroup_cycle(now)
            return

        if mode == self.BotMode.PAUSED:
            logger.info("Bot is PAUSED — skipping all execution")
            return

        # ── Normal active path ────────────────────────────────────────
        if self._should_run_weekly(now):
            self._run_weekly_scan()
            self._last_weekly = now

        if self._should_run_daily_brief(now):
            self._send_daily_brief()
            self._last_daily_brief = now

        if self._should_run_standings(now):
            self._send_standings()
            self._last_standings = now

        if self._should_run_monthly_report(now):
            self._send_monthly_report()
            self._last_monthly_report = now

        if self._should_run_4h(now):
            self._run_position_monitor()
            self._last_4h = now

        if self._should_run_hourly(now):
            self._run_strategy_evaluation(now)
            self._last_hourly = now

    # ── Mode Transition Handling ───────────────────────────────────────

    def _handle_mode_transitions(self, mode, reason: str):
        """Detect mode changes and fire appropriate Telegram alerts."""
        mode_str = mode.value
        if mode_str == self._prev_mode:
            return

        prev = self._prev_mode
        self._prev_mode = mode_str

        if mode_str == "regroup" and prev == "active":
            # Just entered regroup — alert Mike
            self.notifier.send_regroup_alert(
                reason=self.risk.regroup_reason or reason,
                balance=self.account_balance,
                cooldown_days=self.risk.COOLDOWN_DAYS,
                resume_date=self.risk.regroup_ends,
            )
            logger.critical(f"🟡 Mode transition: ACTIVE → REGROUP")

        elif mode_str == "active" and prev == "regroup":
            # Cooldown over — send resume message + standings
            days_in_regroup = self.risk.COOLDOWN_DAYS  # approximate
            self.notifier.send_regroup_resume(self.account_balance, days_in_regroup)
            self._send_standings()
            logger.info(f"✅ Mode transition: REGROUP → ACTIVE")

    # ── Regroup Cycle ─────────────────────────────────────────────────

    def _run_regroup_cycle(self, now: datetime):
        """
        During regroup: monitor open positions + send daily observations.
        No new positions. This is the 'step back and watch' phase.
        """
        # 4H position monitor still runs (protect open positions)
        if self._should_run_4h(now):
            self._run_position_monitor()
            self._last_4h = now

        # Standings every 6h even during regroup
        if self._should_run_standings(now):
            self._send_standings()
            self._last_standings = now

        # Daily regroup observation (scan but don't trade)
        if self._should_run_regroup_obs(now):
            self._send_regroup_observation()
            self._last_regroup_obs = now

    def _send_regroup_observation(self):
        """Scan all pairs during regroup — observe and log, don't trade."""
        observations = []
        for pair in WATCHLIST[:6]:  # reduced scan to save API calls
            try:
                df_4h = self._fetch_oanda_candles(pair, "H4", count=50)
                df_1h = self._fetch_oanda_candles(pair, "H1", count=30)
                if df_4h is None or df_1h is None:
                    continue
                # Just note if price is near a psych level
                last_close = float(df_1h["close"].iloc[-1])
                observations.append(f"{pair}: {last_close:.5f} — watching for structure")
            except Exception:
                pass

        days_remaining = 0
        if self.risk.regroup_ends:
            remaining = self.risk.regroup_ends - datetime.now(timezone.utc)
            days_remaining = max(0, remaining.days)

        self.notifier.send_regroup_observation(observations or ["No clear setups today."], days_remaining)

    # ── Strategy Evaluation ───────────────────────────────────────────

    def _run_strategy_evaluation(self, now: datetime):
        """Fetch 1H data and evaluate all pairs for entry signals."""
        logger.info("=== Hourly strategy evaluation ===")
        is_overnight = self._is_london_session(now)

        # ── Macro theme detection ─────────────────────────────────────────
        # Before evaluating individual pairs, check if a dominant currency
        # theme is active. If JPY is weakening across 4+ pairs, we allow
        # stacking up to 4 correlated positions at reduced size each.
        # Alex's Week 7-8: JPY SHORT theme = $70K in one week.
        macro_theme = self._detect_macro_theme()
        if macro_theme:
            logger.info(
                f"🌊 MACRO THEME ACTIVE: {macro_theme} "
                f"— stacking up to {macro_theme.trade_count} positions"
            )
            self.notifier.send(
                f"🌊 <b>Macro theme detected:</b> {macro_theme.currency} "
                f"{'weak ↓' if macro_theme.direction == 'weak' else 'strong ↑'} "
                f"(score={macro_theme.score:.1f}) — "
                f"watching {', '.join(macro_theme.confirming_pairs[:4])}"
            )

        # ── Pair whitelist ────────────────────────────────────────────────────
        wl_enabled, wl_pairs = self._load_whitelist()
        if wl_enabled:
            logger.info(f"Whitelist ACTIVE — trading only: {', '.join(sorted(wl_pairs))}")

        for pair in WATCHLIST:
            try:
                self._evaluate_pair(
                    pair, overnight=is_overnight, macro_theme=macro_theme,
                    whitelist_enabled=wl_enabled, whitelist_pairs=wl_pairs,
                )
            except Exception as e:
                logger.error(f"{pair}: Evaluation error: {e}")

    def _detect_macro_theme(self):
        """
        Scan all watchlist pairs for a dominant macro currency theme.
        Uses the same CurrencyStrengthAnalyzer as the backtester so
        live bot and backtest behave identically.
        Returns a CurrencyTheme if one is found, else None.
        """
        from ..strategy.forex.currency_strength import CurrencyStrengthAnalyzer, CurrencyTheme
        try:
            analyzer     = CurrencyStrengthAnalyzer()
            candle_data  = {}
            for pair in WATCHLIST:
                df_d = self._fetch_oanda_candles(pair, "D", count=220)
                if df_d is not None and len(df_d) >= 22:
                    candle_data[pair] = {"d": df_d}
            if not candle_data:
                return None
            return analyzer.get_dominant_theme(candle_data)
        except Exception as e:
            logger.warning(f"Macro theme detection failed: {e}")
            return None

    def _capture_confluence(self, pair: str, decision, macro_theme) -> None:
        """
        Build a structured confluence snapshot for one pair and store it.
        The dashboard reads this to show what the bot is waiting for.
        Called after every strategy.evaluate() — ENTER, WAIT, or NO_TRADE.
        """
        from datetime import datetime, timezone
        ff = set(decision.failed_filters or [])

        # ── Derive each check ─────────────────────────────────────────
        has_pattern  = decision.pattern is not None
        has_level    = decision.nearest_level is not None
        trend_ok     = "trend_alignment" not in ff
        has_signal   = decision.entry_signal is not None
        session_ok   = "session" not in ff
        news_ok      = "news_blackout" not in ff
        conf_val     = decision.confidence
        n_open       = len(self.strategy.open_positions)
        # Winner rule: confidence bar doesn't rise for 2nd trade — it's a
        # binary block (winner running = door closed entirely, not higher bar).
        conf_req     = MIN_CONFIDENCE
        conf_ok      = conf_val >= conf_req

        # ── Pip equity: measured move in pips (neckline → target_1) ─────────
        # Must be computed BEFORE the waiting-for list (used in pip equity gate).
        from src.strategy.forex import strategy_config as _sc_live
        _pip_mult   = 100.0 if "JPY" in pair.upper() else 10000.0
        _pip_equity = 0.0
        if has_pattern and decision.pattern.target_1 and decision.pattern.neckline:
            _is_cb = 'consolidation_breakout' in (decision.pattern.pattern_type or '')
            _pe_target = (decision.pattern.target_2
                          if _is_cb and decision.pattern.target_2
                          else decision.pattern.target_1)
            _pip_equity = abs(decision.pattern.neckline - _pe_target) * _pip_mult

        # ── Whitelist-blocked shortcut ────────────────────────────────────────
        is_whitelist_blocked = "whitelist_blocked" in ff

        # ── Derive "waiting for" — ordered by what needs to happen first ─────
        waiting = []
        if is_whitelist_blocked:
            waiting.append("Not in active whitelist — edit in Backtests → Whitelist")
        else:
            if not has_pattern:
                waiting.append("Pattern to form")
            if not has_level:
                waiting.append("Price near key level")
            if not trend_ok:
                waiting.append("Multi-TF trend alignment")
            if has_pattern and has_level and trend_ok and not conf_ok:
                waiting.append(f"Confidence {conf_val:.0%} → {conf_req:.0%}")
            if has_pattern and has_level and trend_ok and conf_ok and not has_signal:
                waiting.append("Engulfing candle / entry signal")
            # Pip equity gate
            if has_pattern and _pip_equity > 0 and _pip_equity < _sc_live.MIN_PIP_EQUITY:
                waiting.append(f"Pip equity {_pip_equity:.0f}p → {_sc_live.MIN_PIP_EQUITY:.0f}p min")
            if not session_ok:
                waiting.append("London session (3–8 AM ET)")
            if not news_ok:
                waiting.append("News blackout to clear")
            if "max_concurrent" in ff:
                waiting.append("Open position to close first")
            if "currency_overlap" in ff:
                waiting.append("Currency exposure to free up")
            if "winner_rule" in ff:
                waiting.append("Winner running — door closed until it stops")
            if not waiting and decision.decision.value == "ENTER":
                waiting.append("Nothing — ready to enter")
            elif not waiting:
                waiting.append("Building setup")

        self._confluence_state[pair] = {
            "pair":        pair,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "decision":    decision.decision.value,
            "direction":   decision.direction,
            "confidence":  round(conf_val, 3),
            "conf_required": conf_req,
            "pattern":     decision.pattern.pattern_type if has_pattern else None,
            "pattern_clarity": round(decision.pattern.clarity, 2) if has_pattern else None,
            "pattern_tf":  getattr(decision.pattern, '_source_tf', 'daily').upper() if has_pattern else None,
            "pip_equity":  round(_pip_equity, 1),
            "level_price": round(decision.nearest_level.price, 5) if has_level else None,
            "level_score": decision.nearest_level.score if has_level else None,
            "trend_weekly": decision.trend_weekly.value if decision.trend_weekly else "?",
            "trend_daily":  decision.trend_daily.value  if decision.trend_daily  else "?",
            "trend_4h":     decision.trend_4h.value     if decision.trend_4h     else "?",
            "entry_signal": decision.entry_signal.signal_type if has_signal else None,
            "macro_theme":  f"{macro_theme.currency}_{macro_theme.direction}" if macro_theme else None,
            "checks": {
                "pattern":       has_pattern,
                "key_level":     has_level,
                "trend":         trend_ok,
                "confidence":    conf_ok,
                "entry_signal":  has_signal,
                "session":       session_ok,
                "news":          news_ok,
            },
            "failed_filters":     list(ff),
            "waiting_for":        waiting,
            "whitelist_blocked":  is_whitelist_blocked,
        }

    def _evaluate_pair(
        self, pair: str, overnight: bool = False, macro_theme=None,
        whitelist_enabled: bool = False, whitelist_pairs: set = None,
    ):
        # ── Whitelist gate (before any API fetch — saves quota) ──────────────
        if whitelist_enabled and whitelist_pairs is not None:
            if pair not in whitelist_pairs:
                from ..strategy.forex.set_and_forget import Decision, TradeDecision
                blocked = TradeDecision(
                    decision=Decision.BLOCKED,
                    pair=pair,
                    direction=None,
                    reason=f"⛔ WHITELIST: {pair} not in active whitelist. "
                           f"Enabled pairs: {', '.join(sorted(whitelist_pairs))}",
                    confidence=0.0,
                    failed_filters=["whitelist_blocked"],
                )
                self._capture_confluence(pair, blocked, macro_theme)
                logger.debug(f"{pair}: WHITELIST_BLOCKED — skipping data fetch")
                return

        df_w  = self._fetch_oanda_candles(pair, "W",  count=100)
        df_d  = self._fetch_oanda_candles(pair, "D",  count=200)
        df_4h = self._fetch_oanda_candles(pair, "H4", count=200)
        df_1h = self._fetch_oanda_candles(pair, "H1", count=100)

        if any(df is None or len(df) < 20 for df in [df_w, df_d, df_4h, df_1h]):
            logger.debug(f"{pair}: Insufficient data, skipping")
            return

        decision = self.strategy.evaluate(
            pair=pair,
            df_weekly=df_w,
            df_daily=df_d,
            df_4h=df_4h,
            df_1h=df_1h,
            macro_theme=macro_theme,
        )

        # ── Capture confluence state for dashboard ────────────────────────
        # Runs for EVERY pair, EVERY scan — dashboard shows what each pair
        # is waiting for before the bot will enter.
        self._capture_confluence(pair, decision, macro_theme)

        if decision.decision.value == "ENTER":
            # ── Confidence gate (matches backtester MIN_CONFIDENCE=65%) ──
            if decision.confidence < MIN_CONFIDENCE:
                logger.info(
                    f"⚠ {pair}: ENTER signal BLOCKED — confidence {decision.confidence:.0%} "
                    f"< {MIN_CONFIDENCE:.0%} threshold. Waiting for stronger setup."
                )
                return

            # ── Macro theme direction gate ────────────────────────────────
            # PARITY: identical logic to backtester (oanda_backtest_v2.py).
            # If a macro theme is active and this pair is one of its suggested
            # trades, block entries whose direction CONTRADICTS the theme.
            # E.g. USD_strong theme → EUR/USD SHORT is fine, EUR/USD LONG is blocked.
            # Controlled by REQUIRE_THEME_GATE lever in strategy_config.py.
            if _sc.REQUIRE_THEME_GATE and macro_theme:
                _theme_dir_map = dict(macro_theme.suggested_trades)
                _theme_dir     = _theme_dir_map.get(pair)
                if _theme_dir and _theme_dir != decision.direction:
                    logger.info(
                        f"⚠ {pair}: ENTER BLOCKED — theme direction conflict. "
                        f"Theme={macro_theme.currency}_{macro_theme.direction} "
                        f"wants {_theme_dir}, pattern wants {decision.direction}."
                    )
                    return

            # ── Winner rule: don't compete with your winner ───────────────
            # Block new entries only when any open position is ACTIVELY up
            # ≥WINNER_THRESHOLD_R in unrealized profit at current price.
            # Uses live price fetch — NOT the be_moved flag (which stays True
            # even after price drifts back to entry and sits there for months).
            _max_open_r = 0.0
            for _open_pair, _open_pos in self.strategy.open_positions.items():
                _open_entry = _open_pos.get("entry", 0)
                _open_stop  = _open_pos.get("stop", 0)
                _open_dir   = _open_pos.get("direction", "long")
                _open_risk  = abs(_open_entry - _open_stop)
                if _open_risk == 0:
                    continue
                # Reuse already-fetched df_1h for same pair; otherwise fetch
                if _open_pair == pair:
                    _open_price = float(df_1h["close"].iloc[-1])
                else:
                    _open_df = self._fetch_oanda_candles(_open_pair, "H1", count=2)
                    if _open_df is None or len(_open_df) == 0:
                        continue
                    _open_price = float(_open_df["close"].iloc[-1])
                _wr = ((_open_price - _open_entry) / _open_risk
                       if _open_dir == "long"
                       else (_open_entry - _open_price) / _open_risk)
                if _wr > _max_open_r:
                    _max_open_r = _wr

            win_blocked, win_reason = winner_rule_check(
                n_open=len(self.strategy.open_positions),
                max_unrealized_r=_max_open_r,
            )
            if win_blocked:
                logger.info(f"⚠ {pair}: ENTER BLOCKED — {win_reason}")
                return

            # ── Session gate — only auto-execute during London session ────
            # Outside London: log the signal so Mike can act manually, but
            # don't auto-execute. Alex always enters London session (1-3 AM ET).
            if not overnight:
                logger.info(
                    f"⏰ {pair}: ENTER signal in non-London session — "
                    f"logging for review, not auto-executing. (conf={decision.confidence:.0%})"
                )
                self.notifier.send(
                    f"⏰ <b>{pair} signal (non-London)</b> — {decision.direction} @ "
                    f"{decision.entry_price:.5f if decision.entry_price else '?'} | "
                    f"conf={decision.confidence:.0%}\n"
                    f"<i>Not auto-executing — outside London session. Review manually.</i>"
                )
                return

            # ── ATR stop check (matches backtester _stop_ok logic) ────────
            # Stop must be ≤ ATR_STOP_MULTIPLIER × ATR (rejects ancient levels)
            # Stop must be ≥ ATR_MIN_MULTIPLIER × ATR (rejects micro-stop noise)
            if decision.entry_price and decision.stop_loss and len(df_d) >= ATR_LOOKBACK + 1:
                import numpy as np
                recent   = df_d.tail(ATR_LOOKBACK + 1)
                hl       = (recent["high"] - recent["low"]).values
                hc       = np.abs(recent["high"].values[1:] - recent["close"].values[:-1])
                lc       = np.abs(recent["low"].values[1:]  - recent["close"].values[:-1])
                tr       = np.maximum(hl[1:], np.maximum(hc, lc))
                atr      = float(np.mean(tr))
                dist     = abs(decision.entry_price - decision.stop_loss)
                pip      = 0.01 if "JPY" in pair else 0.0001
                if dist > atr * ATR_STOP_MULTIPLIER:
                    logger.info(
                        f"⚠ {pair}: ENTER BLOCKED — stop too wide: "
                        f"{dist/pip:.0f}p > max {atr*ATR_STOP_MULTIPLIER/pip:.0f}p "
                        f"({ATR_STOP_MULTIPLIER:.0f}×ATR)"
                    )
                    return
                if dist < atr * ATR_MIN_MULTIPLIER:
                    logger.info(
                        f"⚠ {pair}: ENTER BLOCKED — stop too tight: "
                        f"{dist/pip:.0f}p < min {atr*ATR_MIN_MULTIPLIER/pip:.0f}p "
                        f"({ATR_MIN_MULTIPLIER:.2f}×ATR) — micro-stop, daily noise will hit it"
                    )
                    return

            logger.info(f"🎯 {pair}: ENTER signal! overnight={overnight} conf={decision.confidence:.0%}")
            result = self.executor.execute(decision, self.account_balance)

            if result.get("status") in ("filled", "pending", "dry_run"):
                # Send trade notification — extra detail if overnight
                self.notifier.send_trade_entry(
                    pair=pair,
                    direction=decision.direction or "?",
                    entry_price=decision.entry_price or 0,
                    stop_price=decision.stop_price or 0,
                    risk_amount=result.get("risk_amount", 0),
                    risk_pct=self.risk.get_risk_pct(self.account_balance),
                    pattern=str(decision.pattern.pattern_type if decision.pattern else "?"),
                    key_level=decision.nearest_level.price if decision.nearest_level else 0,
                    overnight=overnight,
                    dry_run=self.dry_run,
                )

        elif decision.confidence >= 0.60:
            logger.info(
                f"👁 {pair}: Setup forming (conf={decision.confidence:.0%}) — "
                f"{decision.reason[:80]}"
            )
            if decision.nearest_level and decision.pattern:
                self.journal.log_setup_detected(
                    pair=pair,
                    direction=decision.direction or "?",
                    key_level=decision.nearest_level.price,
                    level_score=decision.nearest_level.score,
                    pattern=decision.pattern.pattern_type,
                    pattern_clarity=decision.pattern.clarity,
                    trend_weekly=decision.trend_weekly.value if decision.trend_weekly else "?",
                    trend_daily=decision.trend_daily.value if decision.trend_daily else "?",
                    trend_4h=decision.trend_4h.value if decision.trend_4h else "?",
                    notes=f"Awaiting entry signal. Conf={decision.confidence:.0%}",
                )

    def _is_london_session(self, now: datetime) -> bool:
        """True if current UTC hour is within London session window."""
        return LONDON_SESSION_START_UTC <= now.hour < LONDON_SESSION_END_UTC

    # ── Position Monitor ──────────────────────────────────────────────

    def _run_position_monitor(self):
        if not self.strategy.open_positions:
            logger.debug("4H check: no open positions")
            return

        logger.info(f"=== 4H position monitor ({len(self.strategy.open_positions)} open) ===")
        candle_data = {}
        for pair in self.strategy.open_positions:
            candle_data[pair] = {
                "daily": self._fetch_oanda_candles(pair, "D",  count=50),
                "4h":    self._fetch_oanda_candles(pair, "H4", count=50),
                "1h":    self._fetch_oanda_candles(pair, "H1", count=20),
            }
        self.monitor.check_all(candle_data)

    # ── Weekly Scan ───────────────────────────────────────────────────

    def _run_weekly_scan(self):
        logger.info("=== Weekly Scanner ===")

        # Refresh Tier 1 news calendar for the new week
        try:
            refreshed = self.strategy.news_filter.refresh_if_needed(force=True)
            upcoming = self.strategy.news_filter.format_upcoming(hours_ahead=168)  # full week
            if refreshed:
                self.notifier.send(f"📅 <b>Weekly News Calendar</b>\n\n{upcoming}")
            logger.info(f"News calendar refreshed. {upcoming}")
        except Exception as e:
            logger.warning(f"News calendar refresh failed: {e}")

        pair_data = {}
        for pair in WATCHLIST:
            df_w  = self._fetch_oanda_candles(pair, "W",  count=100)
            df_d  = self._fetch_oanda_candles(pair, "D",  count=200)
            df_4h = self._fetch_oanda_candles(pair, "H4", count=200)
            df_1h = self._fetch_oanda_candles(pair, "H1", count=100)
            if any(df is None or len(df) < 20 for df in [df_w, df_d, df_4h, df_1h]):
                continue
            pair_data[pair] = {"weekly": df_w, "daily": df_d, "4h": df_4h, "1h": df_1h}

        if not pair_data:
            logger.error("Weekly scan: no valid pair data")
            return

        setups   = self.scanner.scan(pair_data)
        prime    = [s.pair for s in setups if s.status.value == "PRIME"]
        watching = [s.pair for s in setups if s.status.value == "WATCHING"]

        self.journal.log_scan_complete(
            pairs_scanned=len(pair_data),
            prime_setups=prime,
            watching=watching,
            notes=f"Top picks: {prime[:3]}",
        )
        self.notifier.send_weekly_brief(
            setups=setups,
            account_balance=self.account_balance,
            risk_pct=self.risk.get_risk_pct(self.account_balance),
        )

    # ── Daily Brief ───────────────────────────────────────────────────

    def _send_daily_brief(self):
        try:
            summary = self.oanda.get_account_summary()
        except Exception:
            summary = {}

        # Include today's Tier 1 news events in the brief
        try:
            news_str = self.strategy.news_filter.format_upcoming(hours_ahead=24)
        except Exception:
            news_str = ""

        self.notifier.send_daily_brief(
            open_positions=self.strategy.open_positions,
            account_summary=summary,
            recent_signals=[news_str] if news_str else [],
        )

    # ── Standings ─────────────────────────────────────────────────────

    def _send_standings(self):
        """
        Send a regular standings update — 6H cadence.
        Mike always knows where things stand, even after overnight trades.
        """
        try:
            week_stats   = self.journal.get_stats()
            weekly_pnl   = sum(self.journal.get_weekly_pnl().values()) if self.journal.get_weekly_pnl() else 0.0
            trades_week  = week_stats.get("trades_this_week", 0)
            wins_week    = week_stats.get("wins_this_week", 0)
            losses_week  = week_stats.get("losses_this_week", 0)
        except Exception:
            weekly_pnl = trades_week = wins_week = losses_week = 0

        risk_status = self.risk.status(self.account_balance, consecutive_losses=self._consecutive_losses, dry_run=self.dry_run)

        self.notifier.send_standings(
            account_balance  = self.account_balance,
            nav              = self.account_nav,
            unrealized_pnl   = self.unrealized_pnl,
            weekly_pnl       = weekly_pnl,
            peak_balance     = self.risk._peak_balance or self.account_balance,
            risk_pct         = risk_status["risk_pct"],
            tier_label       = risk_status["tier_label"],
            open_positions   = self.strategy.open_positions,
            trades_this_week = trades_week,
            wins_this_week   = wins_week,
            losses_this_week = losses_week,
            mode             = risk_status["mode"],
            regroup_ends     = self.risk.regroup_ends,
        )

    # ── Data Fetching ─────────────────────────────────────────────────

    def _fetch_oanda_candles(self, pair: str, granularity: str, count: int):
        try:
            candles = self.oanda.get_candles(pair, granularity=granularity, count=count)
            if not candles:
                return None
            df = pd.DataFrame(candles)
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")[["open", "high", "low", "close", "volume"]]
            return df
        except Exception as e:
            logger.warning(f"{pair}/{granularity}: Data fetch failed: {e}")
            return None

    # ── Schedule Logic ─────────────────────────────────────────────────

    def _should_run_hourly(self, now: datetime) -> bool:
        if self._last_hourly is None:
            return True
        return (now - self._last_hourly).total_seconds() >= 3600

    def _should_run_4h(self, now: datetime) -> bool:
        if self._last_4h is None:
            return True
        return (now - self._last_4h).total_seconds() >= 14400

    def _should_run_standings(self, now: datetime) -> bool:
        if self._last_standings is None:
            return True
        return (now - self._last_standings).total_seconds() >= 21600  # 6 hours

    def _should_run_weekly(self, now: datetime) -> bool:
        if self._last_weekly is not None:
            if (now - self._last_weekly).total_seconds() < 150 * 3600:
                return False
        return now.weekday() == 0 and now.hour == 1  # Monday 01:00 UTC

    def _should_run_daily_brief(self, now: datetime) -> bool:
        if self._last_daily_brief is not None:
            if (now - self._last_daily_brief).total_seconds() < 20 * 3600:
                return False
        return now.hour == 13  # 08:00 ET

    def _should_run_regroup_obs(self, now: datetime) -> bool:
        if self._last_regroup_obs is None:
            return True
        return (now - self._last_regroup_obs).total_seconds() >= 86400  # 24 hours

    def _send_monthly_report(self):
        """Send a full monthly performance report + lessons summary."""
        try:
            all_trades = self.journal.get_all_trades()
            self.analyzer.send_monthly_report(all_trades, self.account_balance)
        except Exception as e:
            logger.error(f"Monthly report failed: {e}")

    def _should_run_monthly_report(self, now: datetime) -> bool:
        """First day of month at 09:00 UTC."""
        if self._last_monthly_report is not None:
            if (now - self._last_monthly_report).days < 25:
                return False
        return now.day == 1 and now.hour == 9

    # ── Control File ──────────────────────────────────────────────────

    def _load_whitelist(self) -> tuple[bool, set]:
        """
        Read logs/whitelist.json and return (enabled, pairs_set).
        Returns (False, empty_set) if file missing → no whitelist active.
        Called once per scan cycle (not per-pair) to avoid repeated disk reads.
        """
        if not WHITELIST_FILE.exists():
            return False, set()
        try:
            data = json.loads(WHITELIST_FILE.read_text())
            enabled = bool(data.get("enabled", False))
            pairs   = set(str(p) for p in data.get("pairs", []))
            return enabled, pairs
        except Exception as e:
            logger.warning(f"Could not read whitelist.json: {e}")
            return False, set()

    def _check_control_file(self):
        """
        Read bot_control.json for commands relayed by Forge from Mike's Telegram.
        Commands are consumed (deleted) after processing so they only fire once.

        Supported commands:
          {"command": "pause",  "reason": "Mike requested"}
          {"command": "resume", "reason": "Mike requested"}
          {"command": "extend_cooldown", "days": 7}
        """
        if not CONTROL_FILE.exists():
            return
        try:
            ctrl = json.loads(CONTROL_FILE.read_text())
            cmd = ctrl.get("command", "").lower()
            reason = ctrl.get("reason", "Command from control file")

            if cmd == "pause":
                logger.info(f"Control: PAUSE received — {reason}")
                self.risk.pause(reason)
                self.notifier.send(f"⏸ <b>Bot paused</b> — {reason}\nSay 'resume trading' to restart.")
                CONTROL_FILE.unlink()

            elif cmd == "resume":
                logger.info(f"Control: RESUME received — {reason}")
                self.risk.resume_early(reason)
                self.notifier.send(f"▶️ <b>Bot resumed</b> — {reason}\nBack to scanning. 🎯")
                self._send_standings()
                CONTROL_FILE.unlink()

            elif cmd == "extend_cooldown":
                extra_days = int(ctrl.get("days", 7))
                if self.risk.regroup_ends:
                    self.risk._regroup_ends = self.risk.regroup_ends + timedelta(days=extra_days)
                    self.risk._save_regroup_state()
                    self.notifier.send(
                        f"🟡 Cooldown extended by {extra_days} days.\n"
                        f"New resume date: {self.risk.regroup_ends.strftime('%b %d, %Y')}"
                    )
                CONTROL_FILE.unlink()

            elif cmd:
                logger.warning(f"Unknown control command: {cmd}")
                CONTROL_FILE.unlink()

        except Exception as e:
            logger.error(f"Control file error: {e}")
            try:
                CONTROL_FILE.unlink()
            except Exception:
                pass

    # ── State Save ────────────────────────────────────────────────────

    def _refresh_consecutive_losses(self):
        """
        Recompute _consecutive_losses from recent journal entries.
        Called at the start of each state save so the dashboard is always current.
        Scratch exits (|pnl| < 0.10 × risk) count as wins (don't extend streak).
        """
        try:
            trades = self.journal.get_recent_trades(30)  # most-recent first
            count = 0
            for t in trades:
                pnl = t.get("pnl", 0) or 0
                r   = t.get("r",   None)
                # Use R if available (more robust than pnl for micro-positions)
                is_loss = (r is not None and r < -0.10) or (r is None and pnl < 0)
                if is_loss:
                    count += 1
                else:
                    break
            self._consecutive_losses = count
        except Exception:
            pass  # keep existing value on error

    def _save_state(self):
        """Save full bot state to disk — dashboard reads this."""
        try:
            self._refresh_consecutive_losses()
            risk_status  = self.risk.status(
                self.account_balance,
                consecutive_losses=self._consecutive_losses,
                dry_run=self.dry_run,
            )
            session_info = self._session_status()
            self.state.save(
                account_balance  = self.account_balance,
                dry_run          = self.dry_run,
                halted           = self.risk.is_halted,
                halt_reason      = self.risk.regroup_reason,
                risk_pct         = risk_status["risk_pct"],
                open_positions   = self.strategy.open_positions,
                pair_analysis    = {},
                recent_decisions = [],
                mode             = risk_status["mode"],
                confluence_state = self._confluence_state,
                stats            = {
                    "traded_patterns":     len(self.strategy.traded_patterns),
                    "mode":                risk_status["mode"],
                    "tier":                risk_status["tier_label"],
                    "peak_balance":        risk_status["peak_balance"],
                    "drawdown_pct":        risk_status["drawdown_pct"],
                    "regroup_ends":        risk_status["regroup_ends"],
                    "base_risk_pct":       risk_status["base_risk_pct"],
                    "final_risk_pct":      risk_status["final_risk_pct"],
                    "dd_flag":             risk_status["dd_flag"],
                    "active_cap_label":    risk_status["active_cap_label"],
                    "final_risk_dollars":  risk_status["final_risk_dollars"],
                    "consecutive_losses":  self._consecutive_losses,
                    "paused":              risk_status["paused"],
                    "paused_since":        risk_status["paused_since"],
                    "peak_source":         risk_status["peak_source"],
                    "session_allowed":     session_info["session_allowed"],
                    "session_reason":      session_info["session_reason"],
                    "next_session":        session_info["next_session"],
                    "next_session_mins":   session_info["next_session_mins"],
                    "traded_pattern_keys": list(self.strategy.traded_patterns.keys()),
                },
            )
        except Exception as e:
            logger.debug(f"State save failed: {e}")

    # ── Heartbeat ─────────────────────────────────────────────────────

    def _session_status(self) -> dict:
        """Current session block status + minutes to next valid entry window."""
        try:
            sf = self.strategy.session_filter
            now = datetime.now(timezone.utc)
            allowed, reason = sf.is_entry_allowed(now)
            next_session, mins_until = sf.next_entry_window(now)
            return {
                "session_allowed":     allowed,
                "session_reason":      reason if not allowed else "",
                "next_session":        next_session,
                "next_session_mins":   mins_until,
            }
        except Exception:
            return {"session_allowed": True, "session_reason": "", "next_session": "", "next_session_mins": 0}

    def _write_heartbeat(self, status: str = "ok"):
        try:
            risk_status   = self.risk.status(
                self.account_balance,
                consecutive_losses=self._consecutive_losses,
                dry_run=self.dry_run,
            )
            session_info  = self._session_status()
            HEARTBEAT_FILE.write_text(json.dumps({
                "timestamp":          datetime.now(timezone.utc).isoformat(),
                "status":             status,
                "mode":               risk_status["mode"],
                "open_positions":     list(self.strategy.open_positions.keys()),
                "account_balance":    self.account_balance,
                "risk_pct":           risk_status["risk_pct"],
                "base_risk_pct":      risk_status["base_risk_pct"],
                "final_risk_pct":     risk_status["final_risk_pct"],
                "dd_flag":            risk_status["dd_flag"],
                "active_cap_label":   risk_status["active_cap_label"],
                "final_risk_dollars": risk_status["final_risk_dollars"],
                "consecutive_losses": self._consecutive_losses,
                "tier":               risk_status["tier_label"],
                "drawdown_pct":       risk_status["drawdown_pct"],
                "peak_balance":       risk_status["peak_balance"],
                "regroup_ends":        risk_status["regroup_ends"],
                "dry_run":             self.dry_run,
                "session_allowed":     session_info["session_allowed"],
                "session_reason":      session_info["session_reason"],
                "next_session":        session_info["next_session"],
                "next_session_mins":   session_info["next_session_mins"],
            }, indent=2))
        except Exception:
            pass


# ── Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Forex Set & Forget Bot")
    parser.add_argument("--live", action="store_true",
                        help="LIVE mode — places real OANDA orders. Default: dry run.")
    args = parser.parse_args()

    if args.live:
        print("\n⚠️  WARNING: LIVE MODE — real orders will be placed")
        print("Press Ctrl+C within 5 seconds to abort...")
        time.sleep(5)

    bot = ForexOrchestrator(dry_run=not args.live)
    bot.run_forever()
