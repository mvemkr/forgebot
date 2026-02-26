"""
alex_policy.py — Shared Alex small-account hard rules
======================================================
Single source of truth for the two equity-scaled entry gates.
Called by BOTH:
  - backtesting/oanda_backtest_v2.py  (backtest loop)
  - src/execution/orchestrator.py     (live entry gate)

This ensures exact parity: changing a threshold here affects both paths
simultaneously, with no copy-paste drift.

Rules
-----
1. Dynamic MIN_RR
   When account equity < SMALL_ACCOUNT_THRESHOLD ($25 K):
     require exec_rr ≥ MIN_RR_SMALL_ACCOUNT (3.0 R)
   Otherwise:
     require exec_rr ≥ MIN_RR_STANDARD (2.5 R)

2. Weekly trade punch-card
   When equity < SMALL_ACCOUNT_THRESHOLD:
     max MAX_TRADES_PER_WEEK_SMALL (1) closed trade per ISO week
   Otherwise:
     max MAX_TRADES_PER_WEEK_STANDARD (2) closed trades per ISO week
"""
from __future__ import annotations

from typing import Tuple

import src.strategy.forex.strategy_config as _cfg


def check_dynamic_min_rr(exec_rr: float, balance: float) -> Tuple[bool, str]:
    """
    Returns (blocked: bool, reason: str).

    Blocks the entry when exec_rr is below the equity-tier minimum.

    Args:
        exec_rr: the R:R ratio computed by the strategy for this setup.
        balance: current account equity in account currency.
    """
    small = balance < _cfg.SMALL_ACCOUNT_THRESHOLD
    threshold = _cfg.MIN_RR_SMALL_ACCOUNT if small else _cfg.MIN_RR_STANDARD
    if exec_rr < threshold:
        tier = f"small-acct (<${_cfg.SMALL_ACCOUNT_THRESHOLD:,.0f})" if small else "standard"
        return True, (
            f"MIN_RR_SMALL_ACCOUNT: exec_rr={exec_rr:.2f}R < {threshold:.1f}R required "
            f"[{tier}, equity=${balance:,.0f}]"
        )
    return False, ""


def check_weekly_trade_limit(trades_this_week: int, balance: float) -> Tuple[bool, str]:
    """
    Returns (blocked: bool, reason: str).

    Blocks the entry when the ISO-week punch-card is exhausted.

    Args:
        trades_this_week: number of trades already *entered* in the current
                          ISO calendar week (Mon–Sun).  Caller is responsible
                          for computing this from the trade journal or the
                          backtest weekly counter.
        balance: current account equity in account currency.
    """
    small = balance < _cfg.SMALL_ACCOUNT_THRESHOLD
    cap   = _cfg.MAX_TRADES_PER_WEEK_SMALL if small else _cfg.MAX_TRADES_PER_WEEK_STANDARD
    if trades_this_week >= cap:
        tier = f"small-acct (<${_cfg.SMALL_ACCOUNT_THRESHOLD:,.0f})" if small else "standard"
        return True, (
            f"WEEKLY_TRADE_LIMIT: {trades_this_week}/{cap} trades this week "
            f"[{tier}]"
        )
    return False, ""
