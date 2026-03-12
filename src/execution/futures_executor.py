"""
Futures Executor — Validates risk, sizes contracts, places ES/MES orders.

Mirrors trade_executor.py but for futures via SchwabClient.

Safety:
  - dry_run=True by default — never submits without explicit opt-in
  - MAX_DAILY_LOSS_DOLLARS hard stop enforced before every order
  - MAX_CONCURRENT_CONTRACTS hard cap enforced before every order
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict

from ..strategy.futures.es_strategy import TradeSetup
from ..strategy.futures import es_config as cfg
from .notifier import Notifier

logger = logging.getLogger(__name__)

FUTURES_JOURNAL_PATH = Path.home() / "trading-bot" / "logs" / "futures_journal.jsonl"
FUTURES_PAPER_PATH   = Path.home() / "trading-bot" / "runtime_state" / "futures_paper_account.json"


class FuturesExecutor:
    """
    Validates risk and executes futures TradeSetups via SchwabClient.

    Parameters
    ----------
    schwab_client : SchwabClient
        Live Schwab client (or a mock/test double).
    notifier      : Notifier
        Shared Telegram notifier.
    dry_run       : bool
        True = log only, no live orders (default).
    """

    def __init__(
        self,
        schwab_client,
        notifier:   Optional[Notifier] = None,
        dry_run:    bool               = True,
        journal_path: Path             = FUTURES_JOURNAL_PATH,
    ):
        self.client     = schwab_client
        self.notifier   = notifier or Notifier()
        self.dry_run    = dry_run
        self.journal    = Path(journal_path)
        self.journal.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            logger.info("FuturesExecutor: DRY RUN — no live orders will be submitted")
        else:
            logger.warning("FuturesExecutor: LIVE MODE — orders WILL be submitted to Schwab")

    # ── Main entry point ──────────────────────────────────────────────────────

    def execute(
        self,
        setup:          TradeSetup,
        account_equity: float,
        daily_pnl:      float = 0.0,
        open_contracts: int   = 0,
    ) -> Dict:
        """
        Validate risk checks, size contracts, place order, log, notify.

        Parameters
        ----------
        setup          : TradeSetup from ESFuturesStrategy
        account_equity : current paper / live equity
        daily_pnl      : realized P&L so far today (negative = loss)
        open_contracts : currently open contract count

        Returns
        -------
        dict  {"status": "executed"|"blocked"|"error", ...}
        """

        # ── 1. Daily loss limit ───────────────────────────────────────
        if daily_pnl <= -cfg.MAX_DAILY_LOSS_DOLLARS:
            reason = (f"Daily loss limit hit: ${daily_pnl:.2f} ≤ "
                      f"-${cfg.MAX_DAILY_LOSS_DOLLARS}")
            logger.warning(f"🛑 {reason}")
            self._log_blocked(setup, reason)
            return {"status": "blocked", "reason": reason}

        # ── 2. Max concurrent contracts ───────────────────────────────
        if open_contracts >= cfg.MAX_CONCURRENT_CONTRACTS:
            reason = (f"Max concurrent contracts reached: "
                      f"{open_contracts}/{cfg.MAX_CONCURRENT_CONTRACTS}")
            logger.warning(f"🛑 {reason}")
            self._log_blocked(setup, reason)
            return {"status": "blocked", "reason": reason}

        # ── 3. Contract sizing ────────────────────────────────────────
        contracts = self._size_contracts(setup, account_equity)
        if contracts < 1:
            reason = (f"Risk sizing returned 0 contracts "
                      f"(equity={account_equity:.2f}, risk={cfg.MAX_RISK_PCT}%, "
                      f"stop={setup.risk_points}pts)")
            logger.warning(reason)
            self._log_blocked(setup, reason)
            return {"status": "blocked", "reason": reason}

        # ── 4. Place order ────────────────────────────────────────────
        point_value = cfg.MES_POINT_VALUE if cfg.USE_MES else cfg.ES_POINT_VALUE
        dollar_risk = contracts * setup.risk_points * point_value

        logger.info(
            f"FUTURES EXECUTE: {setup.direction.upper()} {contracts}× "
            f"{'MES' if cfg.USE_MES else 'ES'}  "
            f"entry={setup.entry_price:.2f}  stop={setup.stop_price:.2f}  "
            f"target={setup.target_price:.2f}  "
            f"risk=${dollar_risk:.2f}  RR={setup.rr_ratio:.1f}  "
            f"{'[DRY RUN]' if self.dry_run else '[LIVE]'}"
        )

        result = self.client.place_futures_order(
            direction   = setup.direction,
            quantity    = contracts,
            order_type  = "LIMIT",
            limit_price = setup.entry_price,
            stop_price  = setup.stop_price,
            dry_run     = self.dry_run,
        )

        # ── 5. Journal ────────────────────────────────────────────────
        self._log_executed(setup, contracts, dollar_risk, account_equity, result)

        # ── 6. Notify ─────────────────────────────────────────────────
        emoji = "📈" if setup.direction == "long" else "📉"
        msg = (
            f"{emoji} <b>ES FUTURES — {setup.strategy_type}</b>\n"
            f"Direction: <b>{setup.direction.upper()}</b>\n"
            f"Entry: {setup.entry_price:.2f}  "
            f"Stop: {setup.stop_price:.2f}  "
            f"Target: {setup.target_price:.2f}\n"
            f"Contracts: {contracts}× {'MES' if cfg.USE_MES else 'ES'}  "
            f"Risk: ${dollar_risk:.2f}  RR: {setup.rr_ratio:.1f}\n"
            f"Reason: {setup.reason}\n"
            + ("⚠️ <i>DRY RUN — no order submitted</i>" if self.dry_run else "✅ <i>Live order submitted</i>")
        )
        self.notifier.send(msg)

        return {
            "status":       "executed",
            "direction":    setup.direction,
            "strategy":     setup.strategy_type,
            "entry":        setup.entry_price,
            "stop":         setup.stop_price,
            "target":       setup.target_price,
            "contracts":    contracts,
            "dollar_risk":  dollar_risk,
            "rr":           setup.rr_ratio,
            "dry_run":      self.dry_run,
            "broker_result": result,
        }

    # ── Risk sizing ───────────────────────────────────────────────────────────

    def _size_contracts(self, setup: TradeSetup, equity: float) -> int:
        """
        Calculate contract count: floor(equity × risk_pct / (stop_pts × point_value))
        """
        point_value  = cfg.MES_POINT_VALUE if cfg.USE_MES else cfg.ES_POINT_VALUE
        max_risk_usd = equity * (cfg.MAX_RISK_PCT / 100.0)
        risk_per_contract = setup.risk_points * point_value
        if risk_per_contract <= 0:
            return 0
        contracts = int(max_risk_usd / risk_per_contract)
        return min(contracts, cfg.MAX_CONCURRENT_CONTRACTS)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_blocked(self, setup: TradeSetup, reason: str) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event":     "FUTURES_BLOCKED",
            "strategy":  setup.strategy_type,
            "direction": setup.direction,
            "entry":     setup.entry_price,
            "reason":    reason,
        }
        self._append(entry)

    def _log_executed(
        self,
        setup:         TradeSetup,
        contracts:     int,
        dollar_risk:   float,
        equity:        float,
        broker_result: Dict,
    ) -> None:
        entry = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "event":         "FUTURES_ENTERED",
            "strategy":      setup.strategy_type,
            "direction":     setup.direction,
            "entry_price":   setup.entry_price,
            "stop_price":    setup.stop_price,
            "target_price":  setup.target_price,
            "risk_points":   setup.risk_points,
            "reward_points": setup.reward_points,
            "rr":            round(setup.rr_ratio, 2),
            "contracts":     contracts,
            "dollar_risk":   round(dollar_risk, 2),
            "equity":        round(equity, 2),
            "reason":        setup.reason,
            "dry_run":       self.dry_run,
            "broker_result": broker_result,
        }
        self._append(entry)

    def _append(self, entry: dict) -> None:
        try:
            with open(self.journal, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"FuturesExecutor journal write failed: {e}")
