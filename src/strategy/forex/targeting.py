"""
targeting.py — Shared target selection for live strategy + backtester.

Single source of truth for TP selection and exec RR gating.
Both set_and_forget.py and oanda_backtest_v2.py import from here.
This eliminates the "strategy clears RR on target_1; backtester exits at 0.3R" bug.

Usage:
    from .targeting import select_target, find_next_structure_level

    candidates = [
        (struct_level, "4h_structure"),   # preferred — Alex's actual method
        (pattern.target_1, "measured_move"),
        (pattern.target_2, "measured_move_t2"),
    ]
    target, target_type, exec_rr = select_target(
        direction="short", entry=1.3500, stop=1.3600,
        candidates=candidates, min_rr=cfg.MIN_RR,
    )
    if target is None:
        # block — no qualifying target
"""

from typing import Optional, List, Tuple
import pandas as pd


def select_target(
    direction: str,
    entry: float,
    stop: float,
    candidates: List[Tuple[float, str]],
    min_rr: float,
) -> Tuple[Optional[float], str, float]:
    """
    Pick the first candidate target that satisfies ALL of:
      1. Correct side   (SHORT target < entry, LONG target > entry)
      2. exec_rr        (|target - entry| / |entry - stop|) >= min_rr

    Candidates are evaluated IN ORDER — first qualifying one wins.
    Recommended order: [4h_structure, measured_move, measured_move_t2]

    Returns:
        (chosen_target, target_type, exec_rr)
        (None, "no_qualifying_target", 0.0)  — block this entry
    """
    risk = abs(entry - stop)
    if risk < 1e-8:
        return None, "zero_risk", 0.0

    for price, target_type in candidates:
        if price is None:
            continue

        # ── Wrong-side sanity (Wk3 bug class) ────────────────────────────────
        # break_retest_bearish had target set ABOVE entry on a SHORT.
        # One-liner fix: a SHORT target must be strictly below entry;
        # a LONG target must be strictly above entry. If not, discard silently.
        if direction == "short" and price >= entry:
            continue
        if direction == "long"  and price <= entry:
            continue

        exec_rr = abs(price - entry) / risk
        if exec_rr >= min_rr:
            return price, target_type, exec_rr

    return None, "no_qualifying_target", 0.0


def find_next_structure_level(
    df_4h: pd.DataFrame,
    direction: str,
    reference_price: float,
    swing_bars: int = 5,
) -> Optional[float]:
    """
    Find the nearest prior 4H swing level beyond reference_price in trade direction.

    Alex explicitly places his target at "the next 4-hour low/high" visible on
    the chart at entry time — not the geometric measured move.

    For SHORT: nearest swing LOW strictly below reference_price
    For LONG:  nearest swing HIGH strictly above reference_price

    Returns the level price, or None if no swing found.
    """
    if df_4h is None or len(df_4h) < swing_bars * 2 + 1:
        return None

    lows  = df_4h["low"].values  if "low"  in df_4h.columns else df_4h["Low"].values
    highs = df_4h["high"].values if "high" in df_4h.columns else df_4h["High"].values
    n     = len(df_4h)

    candidates: list = []
    for i in range(swing_bars, n - swing_bars):
        if direction == "short":
            price = lows[i]
            if price >= reference_price:
                continue
            if (all(price <= lows[i - j]  for j in range(1, swing_bars + 1)) and
                    all(price <= lows[i + j]  for j in range(1, swing_bars + 1))):
                candidates.append(price)
        else:
            price = highs[i]
            if price <= reference_price:
                continue
            if (all(price >= highs[i - j] for j in range(1, swing_bars + 1)) and
                    all(price >= highs[i + j] for j in range(1, swing_bars + 1))):
                candidates.append(price)

    if not candidates:
        return None
    # nearest = highest swing low (SHORT) or lowest swing high (LONG)
    return max(candidates) if direction == "short" else min(candidates)
