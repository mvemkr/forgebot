"""
alex_policy.py — Shared Alex small-account hard rules
======================================================
Single source of truth for entry gates applied on BOTH paths:
  - backtesting/oanda_backtest_v2.py  (backtest loop)
  - src/execution/orchestrator.py     (live entry gate)

Rules
-----
1. Alignment-based MIN_RR  (replaces equity-based version)
   Pro-trend trade (weekly + daily + 4H all agree with direction):
     require exec_rr ≥ MIN_RR_STANDARD   (2.5 R)  — high-conviction setup
   Non-protrend / mixed / unknown HTF alignment:
     require exec_rr ≥ MIN_RR_COUNTERTREND (3.0 R) — extra cushion needed
   Rationale: Alex's best trades have full HTF backing.  Counter-trend or
   mixed-signal setups need a wider buffer to compensate for lower conviction.

2. Weekly trade punch-card
   When equity < SMALL_ACCOUNT_THRESHOLD ($25 K):
     max MAX_TRADES_PER_WEEK_SMALL  (1) entered trade per ISO week
   Otherwise:
     max MAX_TRADES_PER_WEEK_STANDARD (2) entered trades per ISO week

Helper
------
  htf_aligned(direction, trend_weekly, trend_daily, trend_4h) → bool | None
    Computes HTF alignment given a Decision's trend fields.
    Returns True (all 3 agree), False (any opposes), or None (data missing).
"""
from __future__ import annotations

from typing import Optional, Tuple, Any

import src.strategy.forex.strategy_config as _cfg


# ── HTF alignment helper ─────────────────────────────────────────────────────

def htf_aligned(
    direction: str,
    trend_weekly: Any = None,
    trend_daily:  Any = None,
    trend_4h:     Any = None,
) -> Optional[bool]:
    """
    Returns:
      True  — all 3 HTFs agree with direction (pro-trend)
      False — at least one HTF opposes direction (counter/mixed)
      None  — insufficient trend data to determine

    trend_* can be a Trend enum or None.
    """
    if trend_weekly is None or trend_daily is None or trend_4h is None:
        return None

    def _bull(t) -> bool:
        return t is not None and hasattr(t, "value") and t.value in ("bullish", "strong_bullish")

    def _bear(t) -> bool:
        return t is not None and hasattr(t, "value") and t.value in ("bearish", "strong_bearish")

    if direction == "long":
        return _bull(trend_weekly) and _bull(trend_daily) and _bull(trend_4h)
    elif direction == "short":
        return _bear(trend_weekly) and _bear(trend_daily) and _bear(trend_4h)
    return None


# ── Gate functions (called by both backtester and orchestrator) ───────────────

def check_dynamic_min_rr(
    exec_rr:     float,
    htf_aligned_flag: Optional[bool] = None,
    balance:     float = 0.0,  # kept for legacy callers; logic is alignment-based now
) -> Tuple[bool, str]:
    """
    Returns (blocked: bool, reason: str).

    Alignment-based MIN_RR gate (replaces equity-based version):
      • pro-trend (htf_aligned_flag=True)  → MIN_RR_STANDARD   (2.5 R)
      • non-protrend / mixed / unknown      → MIN_RR_COUNTERTREND (3.0 R)

    Args:
        exec_rr:          R:R ratio computed by the strategy for this setup.
        htf_aligned_flag: result of alex_policy.htf_aligned() or a pre-computed bool.
                          None = alignment unknown → apply stricter threshold.
        balance:          account equity (kept for API compat; not used for threshold).
    """
    if htf_aligned_flag is True:
        threshold = _cfg.MIN_RR_STANDARD       # 2.5 R — pro-trend, full HTF backing
        tier = "protrend (W+D+4H agree)"
    else:
        threshold = _cfg.MIN_RR_COUNTERTREND   # 3.0 R — mixed/counter/unknown
        tier = ("non-protrend (HTF mixed/counter)"
                if htf_aligned_flag is False
                else "htf-unknown")

    if exec_rr < threshold:
        return True, (
            f"MIN_RR_ALIGN: exec_rr={exec_rr:.2f}R < {threshold:.1f}R required "
            f"[{tier}]"
        )
    return False, ""


def check_weekly_trade_limit(
    trades_this_week: int,
    balance:          float,
) -> Tuple[bool, str]:
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
