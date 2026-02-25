"""
targeting.py — Shared target selection AND stop selection for live strategy + backtester.

Single source of truth: both set_and_forget.py and oanda_backtest_v2.py import from here.

Functions:
  select_target()         — Pick the best TP candidate ≥ MIN_RR.  Fixes "exec at 0.3R" bug.
  get_structure_stop()    — Compute a tight structure-based stop from pattern.stop_anchor.
  find_next_structure_level() — Nearest prior 4H swing level (Alex's TP method).

Stop selection order (tightest valid first):
  H&S / IH&S        1. right shoulder extreme + buffer
                    2. ATR fallback (3× 1H ATR)
  Double top/bottom 1. second peak/trough extreme + buffer
                    2. ATR fallback
  Break/retest      1. retest rejection swing extreme + buffer
                    2. broken level + buffer
                    3. ATR fallback
  Consol breakout   1. range boundary + buffer
                    2. ATR fallback
  Liquidity sweep   1. sweep wick extreme + buffer
                    2. ATR fallback
  Any               Legacy pattern.stop_loss as absolute last resort

Buffer spec (per Alex's tight invalidation style):
  Majors: 3 pips   (pip_size=0.0001 → buffer=0.0003)
  JPY / crosses: 5 pips
"""

from typing import Optional, List, Tuple, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .pattern_detector import PatternResult


def select_target(
    direction: str,
    entry: float,
    stop: float,
    candidates: List[Tuple[float, str]],
    min_rr: float,
    rejected_log: Optional[List[dict]] = None,
) -> Tuple[Optional[float], str, float]:
    """
    Pick the first candidate target that satisfies ALL of:
      1. Correct side   (SHORT target < entry, LONG target > entry)
      2. exec_rr        (|target - entry| / |entry - stop|) >= min_rr

    Candidates are evaluated IN ORDER — first qualifying one wins.
    Recommended order: [4h_structure, measured_move, measured_move_t2]

    Optional: pass a list as `rejected_log` to capture all rejected candidates
    with their reason codes (for funnel/report diagnostics).

    Returns:
        (chosen_target, target_type, exec_rr)
        (None, "no_qualifying_target", 0.0)  — block this entry
    """
    risk = abs(entry - stop)
    if risk < 1e-8:
        if rejected_log is not None:
            rejected_log.append({"type": "all", "reason": "zero_risk", "rr": 0.0})
        return None, "zero_risk", 0.0

    for price, target_type in candidates:
        if price is None:
            continue

        # ── Wrong-side sanity (Wk3 bug class) ────────────────────────────────
        # break_retest_bearish had target set ABOVE entry on a SHORT.
        # One-liner fix: a SHORT target must be strictly below entry;
        # a LONG target must be strictly above entry. If not, discard silently.
        if direction == "short" and price >= entry:
            if rejected_log is not None:
                rejected_log.append({"type": target_type, "price": price,
                                     "reason": "wrong_side", "rr": 0.0})
            continue
        if direction == "long"  and price <= entry:
            if rejected_log is not None:
                rejected_log.append({"type": target_type, "price": price,
                                     "reason": "wrong_side", "rr": 0.0})
            continue

        exec_rr = abs(price - entry) / risk
        if exec_rr >= min_rr:
            return price, target_type, exec_rr

        # Target valid side but RR too low
        if rejected_log is not None:
            rejected_log.append({"type": target_type, "price": price,
                                 "reason": "rr_too_low", "rr": round(exec_rr, 3)})

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


# ─────────────────────────────────────────────────────────────────────────────
# Structure Stop Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stop_buffer(pip_size: float, is_jpy_or_cross: bool) -> float:
    """Tight structure buffer: 3 pips for majors, 5 pips for JPY/crosses."""
    n_pips = 5 if is_jpy_or_cross else 3
    return n_pips * pip_size


def _compute_atr_1h(df: pd.DataFrame, n: int = 14) -> float:
    """ATR(14) from 1H OHLC data. Returns 0.0 if insufficient data."""
    if df is None or len(df) < n + 1:
        return 0.0
    highs  = df["high"].values  if "high"  in df.columns else df["High"].values
    lows   = df["low"].values   if "low"   in df.columns else df["Low"].values
    closes = df["close"].values if "close" in df.columns else df["Close"].values
    hl  = highs[1:] - lows[1:]
    hpc = np.abs(highs[1:] - closes[:-1])
    lpc = np.abs(lows[1:]  - closes[:-1])
    tr  = np.maximum(hl, np.maximum(hpc, lpc))
    return float(np.mean(tr[-n:]))


def _find_retest_rejection_extreme(
    df_1h: pd.DataFrame,
    direction: str,
    anchor: float,
    n_bars: int = 25,
    tol: float = 0.003,
) -> Optional[float]:
    """
    For break/retest patterns: find the retest rejection swing extreme.

    SHORT break/retest:
      Price broke DOWN through anchor, then RETESTED from below.
      Find the highest bar HIGH in the retest zone (within tol of anchor, last n_bars).
      Stop goes above that high + buffer.

    LONG break/retest:
      Price broke UP through anchor, then RETESTED from above.
      Find the lowest bar LOW in the retest zone.
      Stop goes below that low + buffer.
    """
    if df_1h is None or len(df_1h) < 3:
        return None

    highs = df_1h["high"].values  if "high"  in df_1h.columns else df_1h["High"].values
    lows  = df_1h["low"].values   if "low"   in df_1h.columns else df_1h["Low"].values
    n     = len(highs)
    start = max(0, n - n_bars)

    if direction == "short":
        # Zone: slightly above / below the broken level (retest comes from below)
        zone_hi = anchor * (1 + tol * 3)
        zone_lo = anchor * (1 - tol)
        retest_highs = [h for h in highs[start:] if zone_lo <= h <= zone_hi * 1.01]
        return max(retest_highs) if retest_highs else None
    else:
        zone_lo = anchor * (1 - tol * 3)
        zone_hi = anchor * (1 + tol)
        retest_lows = [lo for lo in lows[start:] if zone_lo * 0.99 <= lo <= zone_hi]
        return min(retest_lows) if retest_lows else None


# ─────────────────────────────────────────────────────────────────────────────
# get_structure_stop()
# ─────────────────────────────────────────────────────────────────────────────

def get_structure_stop(
    pattern_type: str,
    direction: str,
    entry: float,
    df_1h: pd.DataFrame,
    pattern: "PatternResult",
    pip_size: float = 0.0001,
    is_jpy_or_cross: bool = False,
    atr_fallback_mult: float = 3.0,
    stop_log: Optional[list] = None,
) -> Tuple[float, str, float]:
    """
    Compute the tightest valid structure-based stop for a trade.

    Uses pattern.stop_anchor (the raw structural extreme) + a tight pip buffer.
    Falls back to ATR-based stop, then to pattern.stop_loss (legacy) as last resort.

    Parameters:
        pattern_type        — e.g. "head_and_shoulders", "double_top"
        direction           — "long" or "short"
        entry               — entry price
        df_1h               — 1H OHLC DataFrame (history up to entry bar)
        pattern             — PatternResult with stop_anchor set
        pip_size            — 0.01 for JPY pairs, 0.0001 for others
        is_jpy_or_cross     — True for JPY pairs or currency crosses
        atr_fallback_mult   — ATR multiplier for fallback (default 3.0; Alex uses ~1.5–2.0)
        stop_log            — optional list; gets appended with STOP_CANDIDATE_REJECTED
                              and STOP_SELECTED entries for diagnostics

    Returns:
        (stop_price, stop_type, stop_pips)
    """
    buf = _stop_buffer(pip_size, is_jpy_or_cross)
    candidates: List[Tuple[float, str]] = []

    def _log(label: str, price: float, reason: str) -> None:
        if stop_log is not None:
            pips = abs(price - entry) / pip_size if pip_size > 0 else 0
            stop_log.append({
                "action": reason,   # "STOP_SELECTED" or "STOP_CANDIDATE_REJECTED:<why>"
                "type":   label,
                "price":  round(price, 6),
                "pips":   round(pips, 1),
            })

    # ── Build candidate list ──────────────────────────────────────────────────
    anchor = pattern.stop_anchor

    if anchor is not None:
        if "break_retest" in pattern_type:
            # Preferred: actual retest rejection swing extreme from 1H data
            retest_ext = _find_retest_rejection_extreme(df_1h, direction, anchor)
            if retest_ext is not None:
                candidates.append((
                    retest_ext + buf if direction == "short" else retest_ext - buf,
                    "retest_swing",
                ))
            # Fallback: broken level + buffer (always available when anchor is set)
            candidates.append((
                anchor + buf if direction == "short" else anchor - buf,
                "broken_level",
            ))
        else:
            # H&S: right shoulder extreme
            # IH&S: right shoulder low
            # DT: highest peak
            # DB: lowest trough
            # CB: range boundary
            # Sweep: wick extreme
            candidates.append((
                anchor + buf if direction == "short" else anchor - buf,
                "structural_anchor",
            ))

    # ATR fallback (3× tighter than the old 8× used in pattern_detector)
    atr = _compute_atr_1h(df_1h)
    if atr > 0:
        candidates.append((
            entry + atr * atr_fallback_mult if direction == "short"
            else entry - atr * atr_fallback_mult,
            "atr_fallback",
        ))

    # Absolute last resort: legacy pattern.stop_loss
    candidates.append((pattern.stop_loss, "legacy_pattern_stop"))

    # ── Choose tightest valid stop ─────────────────────────────────────────────
    # Valid = on the correct side of entry.
    # Tightest = smallest distance from entry (closest invalidation level).
    valid: List[Tuple[float, str, float]] = []
    for price, label in candidates:
        if direction == "short" and price <= entry:
            _log(label, price, "STOP_CANDIDATE_REJECTED:wrong_side")
            continue
        if direction == "long" and price >= entry:
            _log(label, price, "STOP_CANDIDATE_REJECTED:wrong_side")
            continue
        dist_pips = abs(price - entry) / pip_size
        valid.append((price, label, dist_pips))

    if not valid:
        # Shouldn't happen — legacy_pattern_stop always in candidates
        _log("no_valid_fallback", pattern.stop_loss, "STOP_CANDIDATE_REJECTED:none_valid")
        dist_pips = abs(entry - pattern.stop_loss) / pip_size
        return pattern.stop_loss, "no_valid_fallback", dist_pips

    # Sort ascending by distance → tightest first
    valid.sort(key=lambda x: x[2])

    # Reject any with identical or near-zero distance (degenerate)
    for price, label, dist_pips in valid:
        if dist_pips < 1.0:
            _log(label, price, "STOP_CANDIDATE_REJECTED:too_tight(<1pip)")
            continue
        _log(label, price, f"STOP_SELECTED:{label}")
        return price, label, dist_pips

    # All candidates degenerate — use the widest valid one
    price, label, dist_pips = valid[-1]
    _log(label, price, f"STOP_SELECTED:{label}_fallback")
    return price, label, dist_pips
