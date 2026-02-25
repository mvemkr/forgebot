"""
targeting.py — Shared target selection AND stop selection for live strategy + backtester.

Single source of truth: both set_and_forget.py and oanda_backtest_v2.py import from here.

Functions:
  select_target()         — Pick the best TP candidate ≥ MIN_RR.  Fixes "exec at 0.3R" bug.
  get_structure_stop()    — Compute a tight structure-based stop from pattern.stop_anchor.
  find_next_structure_level() — Nearest prior 4H swing level (Alex's TP method).

Stop selection order (PRIORITY-first, not tightest-first):
  H&S / IH&S        1. H1 retest rejection extreme near neckline  (6-bar window; 30-bar fallback)
                    2. Right shoulder anchor + buffer
                    3. ATR fallback (3× 1H ATR) — only if structural candidates fail bounds
  Double top/bottom 1. Second peak/trough extreme + buffer
                    2. ATR fallback
  Break/retest      1. H1 retest rejection extreme near broken level  (same)
                    2. Broken level + buffer
                    3. ATR fallback
  Consol breakout   1. Range boundary + buffer
                    2. ATR fallback
  Liquidity sweep   1. Sweep wick extreme + buffer
                    2. ATR fallback
  Any               Legacy pattern.stop_loss as absolute last resort

Key insight (ab7 → ab8): candidates used to be sorted tightest-first, so 3×1H ATR
(~48p on GBP/JPY) almost always beat structural candidates that were further away.
Fix: evaluate candidates IN PRIORITY ORDER; use the first one that passes bounds check.
ATR fallback is the safety net, not the default.

Bounds check (applied to every candidate):
  Too tight: dist_pips < max(8p, MIN_FRAC_ATR × 1H ATR)  → noise will hit it; skip
  Too wide:  dist_pips > MAX_FRAC_ATR × 1H ATR            → oversized; continue to next

Buffer spec (per Alex's tight invalidation style):
  Majors: max(2×spread, 3 pips)    (pip_size=0.0001 → min buffer=0.0003)
  JPY / crosses: max(2×spread, 5 pips)
"""

from typing import Optional, List, Tuple, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .pattern_detector import PatternResult

# ─────────────────────────────────────────────────────────────────────────────
# Stop bounds constants
# ─────────────────────────────────────────────────────────────────────────────
# Minimum stop as fraction of 1H ATR (below this = micro-stop, noise will hit it)
_MIN_FRAC_ATR: float = 0.15   # 0.15 × ATR(1H, 14)
_MIN_ABS_PIPS: float = 8.0    # hard floor regardless of ATR (avoids zero-ATR edge cases)
# Maximum stop as fraction of 1H ATR (above this = oversized; try next candidate)
_MAX_FRAC_ATR: float = 10.0   # generous cap — only blocks truly degenerate stops


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

def _stop_buffer(pip_size: float, is_jpy_or_cross: bool,
                 spread: Optional[float] = None) -> float:
    """
    Tight structure buffer: max(2×spread, n_pips) for correct invalidation padding.

    Majors:       max(2×spread, 3 pips)
    JPY/crosses:  max(2×spread, 5 pips)

    If spread is None (live feed unavailable), uses flat pip buffer only.
    """
    n_pips = 5 if is_jpy_or_cross else 3
    flat_buf = n_pips * pip_size
    if spread is not None and spread > 0:
        return max(flat_buf, 2.0 * spread)
    return flat_buf


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


def _find_h1_retest_rejection_extreme(
    df_1h: pd.DataFrame,
    direction: str,
    level: float,
    n_bars_primary: int = 6,
    n_bars_fallback: int = 30,
    tol: float = 0.005,
) -> Optional[float]:
    """
    Find the H1 retest rejection extreme near a key level (neckline, broken level).

    For SHORT entry (H&S break / bearish break-retest):
      Price broke DOWN through level, retested from BELOW.
      Find H1 candle(s) touching the level zone in the last n_bars.
      Anchor = max(high of touching candles)  ← stop goes just above this

    For LONG entry (IH&S break / bullish break-retest):
      Price broke UP through level, retested from ABOVE.
      Find H1 candle(s) touching the level zone in the last n_bars.
      Anchor = min(low of touching candles)  ← stop goes just below this

    Two-pass search:
      Pass 1: last n_bars_primary (6) bars — most recent retest
      Pass 2: last n_bars_fallback (30) bars — wider search if retest older

    Zone definition: level ± tol×level (default ±0.5%)
    Returns: raw rejection extreme (no buffer applied here — caller adds buffer)
    """
    if df_1h is None or len(df_1h) < 3:
        return None

    highs = df_1h["high"].values  if "high"  in df_1h.columns else df_1h["High"].values
    lows  = df_1h["low"].values   if "low"   in df_1h.columns else df_1h["Low"].values
    n     = len(highs)

    zone_lo = level * (1.0 - tol)
    zone_hi = level * (1.0 + tol)

    def _search(n_bars: int) -> Optional[float]:
        start = max(0, n - n_bars)
        if direction == "short":
            # Retest from below: HIGH of bar touches the level zone
            touching = [highs[i] for i in range(start, n)
                        if zone_lo <= highs[i] <= zone_hi * 1.005]
            return max(touching) if touching else None
        else:
            # Retest from above: LOW of bar touches the level zone
            touching = [lows[i] for i in range(start, n)
                        if zone_lo * 0.995 <= lows[i] <= zone_hi]
            return min(touching) if touching else None

    # Pass 1: tight recent window
    result = _search(n_bars_primary)
    if result is not None:
        return result

    # Pass 2: wider fallback (retest may be older)
    return _search(n_bars_fallback)


def _find_retest_rejection_extreme(
    df_1h: pd.DataFrame,
    direction: str,
    anchor: float,
    n_bars: int = 30,
    tol: float = 0.005,
) -> Optional[float]:
    """
    Alias: same logic as _find_h1_retest_rejection_extreme with single window.
    Kept for backward-compat with break/retest patterns that call this directly.
    """
    return _find_h1_retest_rejection_extreme(
        df_1h, direction, anchor,
        n_bars_primary=6, n_bars_fallback=n_bars, tol=tol,
    )


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
    spread: Optional[float] = None,
) -> Tuple[float, str, float]:
    """
    Compute the best structure-based stop for a trade using PRIORITY-FIRST selection.

    Candidates are evaluated in semantic priority order (most preferred first).
    The first candidate that passes bounds check is used — ATR is NOT preferred
    just because it happens to be numerically tighter.

    Bounds check per candidate:
      Too tight: dist_pips < max(_MIN_ABS_PIPS, _MIN_FRAC_ATR × 1H ATR) → skip
      Too wide:  dist_pips > _MAX_FRAC_ATR × 1H ATR                     → skip (continue)
    If all structural candidates fail bounds, ATR fallback is used.
    Legacy pattern.stop_loss is absolute last resort.

    Parameters:
        pattern_type        — e.g. "head_and_shoulders", "double_top"
        direction           — "long" or "short"
        entry               — entry price
        df_1h               — 1H OHLC DataFrame (history up to entry bar)
        pattern             — PatternResult with stop_anchor set
        pip_size            — 0.01 for JPY pairs, 0.0001 for others
        is_jpy_or_cross     — True for JPY pairs or currency crosses
        atr_fallback_mult   — ATR multiplier for fallback stop (default 3.0)
        stop_log            — optional list for STOP_CANDIDATE_REJECTED / STOP_SELECTED entries
        spread              — current bid/ask spread (used for max(2×spread, n_pips) buffer)

    Returns:
        (stop_price, stop_type, stop_pips)
    """
    buf = _stop_buffer(pip_size, is_jpy_or_cross, spread)
    atr = _compute_atr_1h(df_1h)

    # Compute bounds in price terms for fast comparison
    min_dist = max(_MIN_ABS_PIPS * pip_size,
                   _MIN_FRAC_ATR * atr if atr > 0 else _MIN_ABS_PIPS * pip_size)
    max_dist = _MAX_FRAC_ATR * atr if atr > 0 else 1e9

    def _log(label: str, price: float, reason: str) -> None:
        if stop_log is not None:
            pips = abs(price - entry) / pip_size if pip_size > 0 else 0
            stop_log.append({
                "action": reason,
                "type":   label,
                "price":  round(price, 6),
                "pips":   round(pips, 1),
            })

    def _stop_side(price: float, label: str) -> Optional[Tuple[float, str, float]]:
        """
        Validate a candidate stop price:
          1. Must be on the correct side of entry (wrong-side guard)
          2. Must pass bounds check (not too tight, not too wide)
        Returns (price, label, dist_pips) if valid, None if rejected.
        """
        if direction == "short" and price <= entry:
            _log(label, price, "STOP_CANDIDATE_REJECTED:wrong_side")
            return None
        if direction == "long" and price >= entry:
            _log(label, price, "STOP_CANDIDATE_REJECTED:wrong_side")
            return None

        dist = abs(price - entry)
        if dist < min_dist:
            _log(label, price, f"STOP_CANDIDATE_REJECTED:too_tight({dist/pip_size:.1f}p)")
            return None
        if dist > max_dist:
            _log(label, price, f"STOP_CANDIDATE_REJECTED:too_wide({dist/pip_size:.1f}p)")
            return None

        dist_pips = dist / pip_size
        _log(label, price, f"STOP_SELECTED:{label}")
        return (price, label, dist_pips)

    # ── Build candidate list IN PRIORITY ORDER ───────────────────────────────
    anchor   = pattern.stop_anchor
    neckline = pattern.neckline
    ordered_candidates: List[Tuple[float, str]] = []

    if anchor is not None:
        if "head_and_shoulders" in pattern_type:
            # Priority order for H&S / IH&S entries (entering at neckline retest):
            #   1. H1 retest rejection extreme near neckline  ← Alex's invalidation
            #   2. Right shoulder anchor + buffer             ← structural fallback
            retest_ext = _find_h1_retest_rejection_extreme(df_1h, direction, neckline)
            if retest_ext is not None:
                ordered_candidates.append((
                    retest_ext + buf if direction == "short" else retest_ext - buf,
                    "neckline_retest_swing",
                ))
            ordered_candidates.append((
                anchor + buf if direction == "short" else anchor - buf,
                "shoulder_anchor",
            ))

        elif "break_retest" in pattern_type:
            # Priority order for break/retest entries:
            #   1. H1 retest rejection extreme near the broken level
            #   2. Broken level itself + buffer
            retest_ext = _find_h1_retest_rejection_extreme(df_1h, direction, anchor)
            if retest_ext is not None:
                ordered_candidates.append((
                    retest_ext + buf if direction == "short" else retest_ext - buf,
                    "neckline_retest_swing",
                ))
            ordered_candidates.append((
                anchor + buf if direction == "short" else anchor - buf,
                "broken_level",
            ))

        else:
            # DT / DB / CB / Sweep: structural anchor is the natural stop
            #   1. Peak/trough/range boundary + buffer
            ordered_candidates.append((
                anchor + buf if direction == "short" else anchor - buf,
                "structural_anchor",
            ))

    # ATR fallback — only reaches here if all structural candidates fail bounds
    if atr > 0:
        ordered_candidates.append((
            entry + atr * atr_fallback_mult if direction == "short"
            else entry - atr * atr_fallback_mult,
            "atr_fallback",
        ))

    # Absolute last resort: legacy pattern.stop_loss
    ordered_candidates.append((pattern.stop_loss, "legacy_pattern_stop"))

    # ── Select first candidate that passes bounds check ───────────────────────
    # Priority-first: structural anchors preferred over ATR, ATR over legacy.
    for price, label in ordered_candidates:
        result = _stop_side(price, label)
        if result is not None:
            return result

    # All candidates rejected — emergency fallback: use legacy stop regardless of bounds
    _log("emergency_fallback", pattern.stop_loss, "STOP_SELECTED:emergency_fallback")
    dist_pips = abs(pattern.stop_loss - entry) / pip_size
    return pattern.stop_loss, "emergency_fallback", dist_pips
