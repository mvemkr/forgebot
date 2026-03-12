"""
Futures Orchestrator — ES / MES intraday trading loop.

Runs a 60-second tick loop:
  Before 8:15 AM ET  : wait
  At 8:15 AM ET      : capture opening range
  8:15 – 11:00 AM ET : check ORB breakout
  10:00 – 3:00 PM ET : check session fade
  After 4:00 PM ET   : stop for the day, send daily summary

Usage:
    python -m src.execution.futures_orchestrator           # dry run (default)
    python -m src.execution.futures_orchestrator --live    # live orders
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import pytz

from ..exchange.schwab_client import SchwabClient
from ..strategy.futures.es_strategy import ESFuturesStrategy, TradeSetup
from ..strategy.futures import es_config as cfg
from .futures_executor import FuturesExecutor
from .notifier import Notifier

logger = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

# ── File paths ─────────────────────────────────────────────────────────────
LOG_DIR         = Path.home() / "trading-bot" / "logs"
RUNTIME_DIR     = Path.home() / "trading-bot" / "runtime_state"
HEARTBEAT_FILE  = LOG_DIR / "futures_orchestrator.heartbeat"
STATE_FILE      = LOG_DIR / "futures_bot_state.json"
DECISION_LOG    = LOG_DIR / "futures_decision_log.jsonl"
JOURNAL_FILE    = LOG_DIR / "futures_journal.jsonl"
PAPER_ACCOUNT   = RUNTIME_DIR / "futures_paper_account.json"


class FuturesOrchestrator:
    """
    Main loop for the ES futures intraday strategy.

    State persisted across restarts via futures_bot_state.json.
    Paper equity persisted in runtime_state/futures_paper_account.json.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run  = dry_run
        LOG_DIR.mkdir(parents=True,    exist_ok=True)
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"FuturesOrchestrator starting — "
            f"{'DRY RUN' if dry_run else '⚠️  LIVE MODE'}"
        )

        # Wire dependencies
        self.schwab    = SchwabClient()
        self.notifier  = Notifier()
        self.strategy  = ESFuturesStrategy(schwab_client=self.schwab)
        self.executor  = FuturesExecutor(
            schwab_client = self.schwab,
            notifier      = self.notifier,
            dry_run       = dry_run,
            journal_path  = JOURNAL_FILE,
        )

        # Runtime state
        self.open_positions:     list  = []
        self.daily_pnl:          float = 0.0
        self.session_started:    bool  = False
        self.orb_range_captured: bool  = False
        self._load_state()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """Tick every 60 seconds until killed."""
        logger.info("Futures orchestrator: entering main loop")
        self.notifier.send(
            f"🚀 <b>Futures Orchestrator started</b>\n"
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}\n"
            f"Strategy: ORB Breakout + Session Fade (ES/MES)"
        )
        while True:
            try:
                self._tick()
                self._write_heartbeat()
                self._save_state()
            except Exception as e:
                logger.error(f"Tick error: {e}", exc_info=True)
            time.sleep(cfg.SCAN_INTERVAL_SECONDS)

    # ── Tick ──────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now_utc = datetime.now(timezone.utc)
        now_et  = now_utc.astimezone(ET)
        hour    = now_et.hour

        # Outside all active windows
        if hour < cfg.PREMARKET_START_ET:
            return
        if hour >= cfg.RTH_CLOSE_ET:
            if self.session_started:
                self._end_of_day_summary()
                self.strategy.reset_session()
                self.session_started    = False
                self.orb_range_captured = False
            return

        self.session_started = True

        # ── Capture opening range at 8:15 AM ─────────────────────────
        if not self.orb_range_captured and hour == cfg.PREMARKET_START_ET and now_et.minute >= 15:
            candles_15m = self.schwab.get_es_candles(timeframe="15m", lookback=10)
            captured = self.strategy.update_opening_range(candles_15m)
            self.orb_range_captured = True
            if captured and self.strategy.orb_high is not None:
                self.notifier.send(
                    f"📊 <b>ES Opening Range Set</b>\n"
                    f"High: {self.strategy.orb_high:.2f}  "
                    f"Low: {self.strategy.orb_low:.2f}  "
                    f"Range: {self.strategy.orb_high - self.strategy.orb_low:.1f}pts"
                )

        # ── Monitor open position ─────────────────────────────────────
        if self.open_positions:
            self._monitor_positions(now_utc)

        # ── Strategy evaluation ───────────────────────────────────────
        if len(self.open_positions) >= cfg.MAX_CONCURRENT_CONTRACTS:
            return   # at max capacity

        setup = self.strategy.evaluate(now_utc=now_utc)
        if setup:
            self._execute_setup(setup, now_utc)

    # ── Execute ───────────────────────────────────────────────────────────────

    def _execute_setup(self, setup: TradeSetup, now_utc: datetime) -> None:
        equity = self._paper_equity()

        result = self.executor.execute(
            setup          = setup,
            account_equity = equity,
            daily_pnl      = self.daily_pnl,
            open_contracts = sum(p.get("contracts", 1) for p in self.open_positions),
        )

        self._log_decision(setup, result, now_utc)

        if result["status"] == "executed":
            self.open_positions.append({
                "strategy":    setup.strategy_type,
                "direction":   setup.direction,
                "entry_price": setup.entry_price,
                "stop_price":  setup.stop_price,
                "target_price":setup.target_price,
                "contracts":   result.get("contracts", 1),
                "dollar_risk": result.get("dollar_risk", 0),
                "opened_at":   now_utc.isoformat(),
            })

    # ── Position monitoring ───────────────────────────────────────────────────

    def _monitor_positions(self, now_utc: datetime) -> None:
        """
        Check open positions against current price for stop/target hits.
        Simple mark-to-market — in paper mode uses last Schwab quote.
        """
        if not self.open_positions:
            return

        quote = self.schwab.get_es_quote()
        price = quote.get("mid") or quote.get("last") or 0.0
        if price == 0.0:
            return

        still_open = []
        for pos in self.open_positions:
            direction = pos["direction"]
            stop      = pos["stop_price"]
            target    = pos["target_price"]
            entry     = pos["entry_price"]
            contracts = pos.get("contracts", 1)
            pv        = cfg.MES_POINT_VALUE if cfg.USE_MES else cfg.ES_POINT_VALUE

            hit_stop   = (direction == "long"  and price <= stop)  or \
                         (direction == "short" and price >= stop)
            hit_target = (direction == "long"  and price >= target) or \
                         (direction == "short" and price <= target)

            if hit_stop or hit_target:
                pts   = (price - entry) if direction == "long" else (entry - price)
                pnl   = pts * contracts * pv
                self.daily_pnl += pnl
                outcome = "TARGET ✅" if hit_target else "STOP ❌"
                msg = (
                    f"{'📈' if direction=='long' else '📉'} <b>ES Position Closed — {outcome}</b>\n"
                    f"Strategy: {pos['strategy']}\n"
                    f"Entry: {entry:.2f}  Exit: {price:.2f}  Pts: {pts:+.1f}\n"
                    f"P&L: ${pnl:+.2f}  Daily P&L: ${self.daily_pnl:+.2f}"
                    + ("\n⚠️ <i>DRY RUN</i>" if self.dry_run else "")
                )
                self.notifier.send(msg)
                logger.info(f"Position closed: {outcome}  P&L=${pnl:+.2f}  daily=${self.daily_pnl:+.2f}")
                self._update_paper_equity(pnl)
            else:
                still_open.append(pos)

        self.open_positions = still_open

    # ── End of day ────────────────────────────────────────────────────────────

    def _end_of_day_summary(self) -> None:
        equity = self._paper_equity()
        msg = (
            f"🏁 <b>ES Session Summary</b>\n"
            f"Daily P&L: ${self.daily_pnl:+.2f}\n"
            f"Paper Equity: ${equity:,.2f}\n"
            f"Open positions at close: {len(self.open_positions)}"
            + ("\n⚠️ <i>DRY RUN</i>" if self.dry_run else "")
        )
        self.notifier.send(msg)
        # Reset daily P&L for next session
        self.daily_pnl      = 0.0
        self.open_positions = []
        logger.info("End-of-day summary sent, daily state reset")

    # ── Paper equity helpers ──────────────────────────────────────────────────

    def _paper_equity(self) -> float:
        try:
            if PAPER_ACCOUNT.exists():
                d = json.loads(PAPER_ACCOUNT.read_text())
                return float(d.get("equity", 10_000))
        except Exception:
            pass
        return 10_000.0

    def _update_paper_equity(self, pnl: float) -> None:
        try:
            d = json.loads(PAPER_ACCOUNT.read_text()) if PAPER_ACCOUNT.exists() else {}
            equity = float(d.get("equity", 10_000)) + pnl
            peak   = max(float(d.get("peak_equity", equity)), equity)
            d.update({
                "equity":               round(equity, 2),
                "peak_equity":          round(peak, 2),
                "realized_session_pnl": round(float(d.get("realized_session_pnl", 0)) + pnl, 2),
                "saved_at":             datetime.now(timezone.utc).isoformat(),
            })
            PAPER_ACCOUNT.write_text(json.dumps(d, indent=2))
        except Exception as e:
            logger.error(f"Paper equity update failed: {e}")

    # ── State persistence ─────────────────────────────────────────────────────

    def _save_state(self) -> None:
        state: Dict[str, Any] = {
            "open_positions":     self.open_positions,
            "daily_pnl":          round(self.daily_pnl, 2),
            "orb_high":           self.strategy.orb_high,
            "orb_low":            self.strategy.orb_low,
            "orb_set":            self.strategy.orb_set,
            "orb_fired":          self.strategy.orb_fired,
            "session_high":       self.strategy.session_high,
            "session_low":        self.strategy.session_low,
            "fade_fired":         self.strategy.fade_fired,
            "heartbeat":          datetime.now(timezone.utc).isoformat(),
            "dry_run":            self.dry_run,
            "mode":               "DRY_RUN" if self.dry_run else "LIVE",
        }
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.error(f"State save failed: {e}")

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            d = json.loads(STATE_FILE.read_text())
            self.open_positions     = d.get("open_positions", [])
            self.daily_pnl          = float(d.get("daily_pnl", 0))
            self.strategy.orb_high  = d.get("orb_high")
            self.strategy.orb_low   = d.get("orb_low")
            self.strategy.orb_set   = d.get("orb_set", False)
            self.strategy.orb_fired = d.get("orb_fired", False)
            self.strategy.session_high = d.get("session_high")
            self.strategy.session_low  = d.get("session_low")
            self.strategy.fade_fired   = d.get("fade_fired", False)
            logger.info(f"State loaded from {STATE_FILE}")
        except Exception as e:
            logger.warning(f"State load failed (starting fresh): {e}")

    def _write_heartbeat(self) -> None:
        try:
            HEARTBEAT_FILE.write_text(json.dumps({
                "timestamp":  datetime.now(timezone.utc).isoformat(),
                "daily_pnl":  round(self.daily_pnl, 2),
                "open_pos":   len(self.open_positions),
                "dry_run":    self.dry_run,
            }))
        except Exception as e:
            logger.error(f"Heartbeat write failed: {e}")

    def _log_decision(self, setup: TradeSetup, result: Dict, now_utc: datetime) -> None:
        entry = {
            "ts":       now_utc.isoformat(),
            "strategy": setup.strategy_type,
            "direction":setup.direction,
            "entry":    setup.entry_price,
            "stop":     setup.stop_price,
            "target":   setup.target_price,
            "rr":       round(setup.rr_ratio, 2),
            "status":   result.get("status"),
            "reason":   result.get("reason") or setup.reason,
            "dry_run":  self.dry_run,
        }
        try:
            with open(DECISION_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Decision log write failed: {e}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="ES Futures Orchestrator")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live order submission (default: dry run)",
    )
    args = parser.parse_args()
    FuturesOrchestrator(dry_run=not args.live).run_forever()
