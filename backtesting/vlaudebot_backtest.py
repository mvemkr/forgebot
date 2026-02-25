"""
VlaudeBot Backtester — Structure-First Price Action Engine v1
=============================================================
Implements the fully-defined algorithmic ruleset (schema v1.0.0):
  - D1/H4 bias detection (HH/HL vs LL/LH swing structure)
  - B1 weekend filter (Thu 12pm / Fri hard block, Fri 15:00 force close)
  - B2 ATR stop sanity — per-class floors/caps (Diddy prevention)
  - B3 chop killer — range + midline crosses on H4
  - Setups: H&S, IH&S, Double Top/Bottom, Range Break (ported from pattern_detector)
  - TP: measured-move first → HTF liquidity fallback, 2.8–3.5R window
  - Management: BE at +2R, F4 v2 conditional time stop, Friday force close

Phase 1: Alex's 5 traded pairs + EUR/USD as reject control.
Goal: reproduce Alex's 10 winners, reject his 4 losers.

Usage:
    python -m backtesting.vlaudebot_backtest
    python -m backtesting.vlaudebot_backtest --start 2024-07-15 --end 2024-10-31 --balance 8000
"""

import sys
import os
import json
import hashlib
import subprocess
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.exchange.oanda_client import OandaClient, INSTRUMENT_MAP
from src.strategy.forex.pattern_detector import PatternDetector

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (single source of truth — mirrors JSON schema v1.0.0 + refinements)
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "schema_version": "1.0.0",
    "execution_mode": "H1",          # "H1" (MODE_A) | "M15" (MODE_B, Phase 2)

    # Session windows (NY time)
    "session": {
        "london_start": "02:00",
        "london_end":   "05:00",
        "ny_start":     "08:00",
        "ny_end":       "11:00",
    },

    # Pair whitelist + class mapping
    "pair_whitelist_enabled": True,
    "pair_whitelist": [
        "GBP_JPY", "USD_JPY", "USD_CHF", "GBP_CHF", "USD_CAD",
        "EUR_USD",   # control: must be rejected for counter-trend
    ],
    "pair_class_map": {
        "USD_JPY":  "JPY",
        "GBP_JPY":  "CROSS",
        "GBP_CHF":  "CROSS",
        "USD_CAD":  "MAJOR",
        "USD_CHF":  "MAJOR",
        "EUR_USD":  "MAJOR",
    },

    # ATR constants
    "atr": {
        "period":               14,
        "breakout_close_frac":  0.10,
        "retest_tol_frac":      0.15,
        "buffer_frac":          0.10,
    },

    # B2 — Volatility bounds per pair class
    # Calibrated against Alex's actual stop sizes:
    #   GBP/JPY ~40p, USD/JPY ~40p, USD/CHF ~36p, GBP/CHF ~34p, USD/CAD ~24p
    "volatility_bounds": {
        "min_atr_mult": 0.4,   # allow tighter stops on low-ATR pairs
        "max_atr_mult": 2.2,   # allow wider stops on JPY/CROSS at high ATR
        "floor_pips": {"MAJOR": 10, "JPY": 12, "CROSS": 12},
        "cap_pips":   {"MAJOR": 70, "JPY": 100, "CROSS": 110},
    },

    # B3 — Regime filter (chop killer v1)
    "regime_filter": {
        "enabled":             True,
        "h4_lookback_bars":    40,
        "min_range_atr_mult":  2.2,
        "max_midline_crosses": 12,
    },

    # Pivot detection
    "pivot": {
        "k_d1": 3,
        "k_h4": 2,
        "k_h1": 2,
        "min_confirmed_swings": 3,
    },

    # H&S geometry
    "hs": {
        "head_over_shoulder_atr_mult": 0.5,
        "shoulder_symmetry_atr_mult":  0.35,
    },

    # Double top/bottom
    "double_top": {
        "equal_high_tol_atr_mult":       0.25,
        "min_h1_bars_between_peaks":     10,
        "max_h1_bars_between_peaks":     120,
    },

    # Range break
    "range": {
        "min_h1_bars":              20,
        "max_h1_bars":              80,
        "max_range_atr_mult":       1.8,
        "touch_tol_atr_frac":       0.2,
        "min_touches":              4,
    },

    # Retest window (H1 bars)
    "retest": {
        "min_bars": 1,
        "max_bars": 3,
    },

    # R:R policy
    "rr": {
        "min_rr":         2.8,
        "cap_rr":         3.5,
        "preferred_low":  3.2,
        "tp_priority": ["MEASURED_MOVE", "HTF_LIQUIDITY"],
    },

    # Trade management
    "management": {
        "move_be_at_r":  2.0,
        "be_buffer_r":   0.1,
        "time_stop": {
            "enabled":                     True,
            "hours_by_class": {"MAJOR": 24, "JPY": 30, "CROSS": 36},
            "min_progress_r":              0.5,
            "requires_reentry_of_level":   True,
        },
    },

    # Risk — mirrors Alex's tier (25% of equity at full size)
    "risk_pct": 0.25,

    # Position limits
    "max_open_positions": 1,   # Alex's rule: ONE trade at a time, globally

    # Backtest defaults
    "default_start":   "2024-07-15",
    "default_end":     "2024-10-31",
    "default_balance": 8000.0,
}

NY  = ZoneInfo("America/New_York")
PIP = {"JPY": 0.01, "MAJOR": 0.0001, "CROSS": 0.0001}

# ─────────────────────────────────────────────────────────────────────────────
# PIP / ATR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def pip_size(pair: str) -> float:
    # JPY-quoted pairs always use 0.01 regardless of class
    if pair.endswith("_JPY"):
        return 0.01
    cls = CFG["pair_class_map"].get(pair, "MAJOR")
    return PIP.get(cls, 0.0001)


def to_pips(pair: str, price_delta: float) -> float:
    return abs(price_delta) / pip_size(pair)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"]
    prev_cl = cl.shift(1)
    tr = pd.concat([hi - lo,
                    (hi - prev_cl).abs(),
                    (lo - prev_cl).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def atr_pips(df: pd.DataFrame, pair: str, period: int = 14) -> float:
    atr = compute_atr(df, period).iloc[-1]
    if pd.isna(atr) or atr == 0:
        return 0.0
    return to_pips(pair, atr)


# ─────────────────────────────────────────────────────────────────────────────
# B1 — WEEKEND FILTER
# ─────────────────────────────────────────────────────────────────────────────

def b1_can_enter(ts: datetime) -> bool:
    ny  = ts.astimezone(NY)
    wd  = ny.strftime("%A").upper()[:3]
    hm  = ny.strftime("%H:%M")
    if wd == "FRI":
        return False
    if wd == "THU" and hm >= "12:00":
        return False
    return True


def b1_force_close(ts: datetime) -> bool:
    ny = ts.astimezone(NY)
    wd = ny.strftime("%A").upper()[:3]
    hm = ny.strftime("%H:%M")
    return wd == "FRI" and hm >= "15:00"


def b1_preferred_session(ts: datetime) -> bool:
    ny  = ts.astimezone(NY)
    hm  = ny.strftime("%H:%M")
    lon = CFG["session"]["london_start"] <= hm <= CFG["session"]["london_end"]
    nyc = CFG["session"]["ny_start"]     <= hm <= CFG["session"]["ny_end"]
    return lon or nyc


# ─────────────────────────────────────────────────────────────────────────────
# B2 — VOLATILITY SANITY FILTER
# ─────────────────────────────────────────────────────────────────────────────

def b2_stop_valid(stop_pips: float, atr_h1_pips: float,
                  pair_class: str) -> tuple[bool, str]:
    vb  = CFG["volatility_bounds"]
    mn  = max(vb["min_atr_mult"] * atr_h1_pips, vb["floor_pips"][pair_class])
    mx  = min(vb["max_atr_mult"] * atr_h1_pips, vb["cap_pips"][pair_class])
    if atr_h1_pips <= 0:
        return False, "atr_unavailable"
    if stop_pips < mn:
        return False, f"stop_too_tight_{stop_pips:.1f}p_<_{mn:.1f}p"
    if stop_pips > mx:
        return False, f"stop_too_wide_{stop_pips:.1f}p_>_{mx:.1f}p"
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# B3 — REGIME FILTER  (chop killer v1)
# ─────────────────────────────────────────────────────────────────────────────

def b3_regime_ok(h4: pd.DataFrame, pair: str,
                 direction: str) -> tuple[bool, str]:
    rf  = CFG["regime_filter"]
    if not rf["enabled"]:
        return True, "filter_disabled"

    n      = rf["h4_lookback_bars"]
    window = h4.iloc[-n:]
    if len(window) < n:
        return False, "insufficient_h4_bars"

    atr_val = compute_atr(window).iloc[-1]
    if pd.isna(atr_val) or atr_val <= 0:
        return False, "atr_invalid"

    hi  = window["high"].max()
    lo  = window["low"].min()
    rng = hi - lo
    rng_pips = to_pips(pair, rng)
    atr_pips_val = to_pips(pair, atr_val)

    if rng_pips < rf["min_range_atr_mult"] * atr_pips_val:
        return False, f"chop_range_{rng_pips/atr_pips_val:.2f}x_atr"

    # Midline cross count — too many = chop, not trending
    mid    = (hi + lo) / 2
    closes = window["close"].values
    crosses = int(np.sum(np.diff(np.sign(closes - mid)) != 0))
    if crosses > rf["max_midline_crosses"]:
        return False, f"chop_midline_{crosses}_crosses"

    # NOTE: directional price-vs-midline check REMOVED.
    # Alex enters at reversals *at* extremes — price can be on "wrong" side
    # of midline precisely because it's at an extreme before the move starts.
    # The range-width check already ensures there's a real trend in the window.

    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# C — BIAS DETECTION  (D1 + H4 swing structure)
# ─────────────────────────────────────────────────────────────────────────────

def _pivot_highs(series: np.ndarray, k: int) -> list[int]:
    idx = []
    for i in range(k, len(series) - k):
        if series[i] == max(series[i-k:i+k+1]):
            idx.append(i)
    return idx


def _pivot_lows(series: np.ndarray, k: int) -> list[int]:
    idx = []
    for i in range(k, len(series) - k):
        if series[i] == min(series[i-k:i+k+1]):
            idx.append(i)
    return idx


def _swing_bias(df: pd.DataFrame, k: int) -> str:
    ph = _pivot_highs(df["high"].values,  k)
    pl = _pivot_lows( df["low"].values,   k)
    if len(ph) < 2 or len(pl) < 2:
        return "MIXED"
    h1, h2 = df["high"].values[ph[-2]], df["high"].values[ph[-1]]
    l1, l2 = df["low"].values[pl[-2]],  df["low"].values[pl[-1]]
    hh, hl = h2 > h1, l2 > l1
    ll, lh = l2 < l1, h2 < h1
    if hh and hl:
        return "BULL"
    if ll and lh:
        return "BEAR"
    return "MIXED"


def _close_based_bias(df: pd.DataFrame, lookback: int = 20) -> tuple[str, str]:
    """
    Transitional close-based bias — no pivot confirmation lag.

    Compares recent vs prior range structure on the last `lookback` bars:
      - recent = last 8 bars  (half a day on H4 = ~2 trading days)
      - prior  = bars [-16:-8]

    Returns (bias, detail):
      BEAR if LH + LL  (or one + confirming mean shift)
      BULL if HH + HL  (or one + confirming mean shift)
      MIXED otherwise
    """
    if len(df) < lookback + 4:
        return "MIXED", "insufficient_data"

    window = df.iloc[-lookback:]
    atr_val = compute_atr(window).iloc[-1]
    if pd.isna(atr_val) or atr_val <= 0:
        return "MIXED", "atr_invalid"

    half = lookback // 2
    qtr  = half // 2

    recent_high = df["high"].iloc[-qtr:].max()
    prior_high  = df["high"].iloc[-half:-qtr].max()
    recent_low  = df["low"].iloc[-qtr:].min()
    prior_low   = df["low"].iloc[-half:-qtr].min()

    lh = recent_high < prior_high - atr_val * 0.1
    ll = recent_low  < prior_low  - atr_val * 0.1
    hh = recent_high > prior_high + atr_val * 0.1
    hl = recent_low  > prior_low  + atr_val * 0.1

    # Mean shift as tiebreaker
    mean_recent = df["close"].iloc[-qtr:].mean()
    mean_prior  = df["close"].iloc[-half:-qtr].mean()
    mean_down   = mean_recent < mean_prior - atr_val * 0.25
    mean_up     = mean_recent > mean_prior + atr_val * 0.25

    if lh and ll:
        return "BEAR", f"LH({prior_high:.4f}>{recent_high:.4f})+LL({prior_low:.4f}>{recent_low:.4f})"
    if hh and hl:
        return "BULL", f"HH({prior_high:.4f}<{recent_high:.4f})+HL({prior_low:.4f}<{recent_low:.4f})"
    if (lh or ll) and mean_down:
        return "BEAR", f"partial_LH/LL+mean_down"
    if (hh or hl) and mean_up:
        return "BULL", f"partial_HH/HL+mean_up"

    return "MIXED", f"no_clear_structure"


def detect_bias(d1: pd.DataFrame, h4: pd.DataFrame) -> tuple[str, str]:
    """
    H4-primary close-based transitional bias.
    D1 used only as a confirming/blocking layer — never vetoes H4 signal alone.

    Logic:
      1. Compute H4 bias (close-based, no pivot lag)
      2. If H4 is clear → use it (allows early counter-trend entries)
      3. If H4 is MIXED → fall back to D1 (daily context)
      4. If both MIXED → MIXED

    This matches Alex's entry timing: H4 structure starts turning before D1
    confirms. The pivot-confirmed version nuked entries before confirmation.
    """
    h4_b, h4_r = _close_based_bias(h4, lookback=20)
    if h4_b != "MIXED":
        return h4_b, f"H4={h4_b}({h4_r})"

    # H4 inconclusive — try D1 as fallback
    d1_b, d1_r = _close_based_bias(d1, lookback=20)
    if d1_b != "MIXED":
        return d1_b, f"D1_fallback={d1_b}({d1_r})"

    return "MIXED", f"H4=MIXED D1=MIXED"


# ─────────────────────────────────────────────────────────────────────────────
# TP CONSTRUCTION  (measured-move → HTF liquidity, 2.8–3.5R)
# ─────────────────────────────────────────────────────────────────────────────

def build_tp(entry: float, stop: float, direction: str,
             measured_move: float,
             htf_liquidity: Optional[float] = None) -> tuple[Optional[float], float, str]:
    """Returns (tp, rr, method) or (None, 0, 'rejected')."""
    risk   = abs(entry - stop)
    mn, cap = CFG["rr"]["min_rr"], CFG["rr"]["cap_rr"]
    if risk <= 0:
        return None, 0.0, "rejected_zero_risk"

    def rr(tp):
        return abs(tp - entry) / risk

    def clamp(tp):
        r = rr(tp)
        if r <= cap:
            return tp
        return (entry - cap * risk) if direction == "SHORT" else (entry + cap * risk)

    # 1. Measured move
    tp1 = (entry - measured_move) if direction == "SHORT" else (entry + measured_move)
    r1  = rr(tp1)
    if r1 >= mn:
        tp1c = clamp(tp1)
        method = "MEASURED_MOVE" if r1 <= cap else "MEASURED_MOVE_CAPPED"
        return tp1c, rr(tp1c), method

    # 2. HTF liquidity
    if htf_liquidity is not None:
        r2 = rr(htf_liquidity)
        if r2 >= mn:
            tp2c = clamp(htf_liquidity)
            method = "HTF_LIQUIDITY" if r2 <= cap else "HTF_LIQUIDITY_CAPPED"
            return tp2c, rr(tp2c), method

    return None, 0.0, "rejected_no_valid_tp"


# ─────────────────────────────────────────────────────────────────────────────
# HTF LIQUIDITY FINDER  (nearest prior swing in direction of trade)
# ─────────────────────────────────────────────────────────────────────────────

def find_htf_liquidity(h4: pd.DataFrame, direction: str,
                       entry: float) -> Optional[float]:
    """Find the nearest prior H4 swing low (SHORT) or high (LONG) beyond entry."""
    k  = CFG["pivot"]["k_h4"]
    if direction == "SHORT":
        pl  = _pivot_lows(h4["low"].values, k)
        liq = [h4["low"].values[i] for i in pl if h4["low"].values[i] < entry]
        return max(liq) if liq else None
    else:
        ph  = _pivot_highs(h4["high"].values, k)
        liq = [h4["high"].values[i] for i in ph if h4["high"].values[i] > entry]
        return min(liq) if liq else None


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN ADAPTER  (PatternDetector → vlaudebot signal format)
# ─────────────────────────────────────────────────────────────────────────────
# The existing PatternDetector has battle-tested H&S / double-top /
# consolidation_breakout geometry.  We run it on the H1 window and filter
# its results through the vlaudebot spec (B2, B3, TP policy, bias).

_BEARISH_SETUP_TYPES = {
    "head_and_shoulders",
    "double_top",
    "consolidation_breakout_bearish",
    "break_retest_bearish",
}
_BULLISH_SETUP_TYPES = {
    "inverted_head_and_shoulders",
    "double_bottom",
    "consolidation_breakout_bullish",
    "break_retest_bullish",
}

def direction_of(pattern_type: str) -> Optional[str]:
    if pattern_type in _BEARISH_SETUP_TYPES:
        return "SHORT"
    if pattern_type in _BULLISH_SETUP_TYPES:
        return "LONG"
    return None


def adapt_pattern(pr, pair: str, h1: pd.DataFrame, h4: pd.DataFrame,
                  bias: str, ts: datetime) -> Optional[dict]:
    """
    Convert a PatternResult to a vlaudebot signal dict after running all filters.
    Returns the signal dict or None if rejected (with reason logged).
    """
    direction = direction_of(pr.pattern_type)
    if direction is None:
        return None

    pair_class = CFG["pair_class_map"].get(pair, "MAJOR")
    atr_h1     = atr_pips(h1, pair)

    # ── Bias gate (pattern-dependent) ─────────────────────────────────────
    # Structural reversal patterns (H&S, DT, DB) carry their own directional
    # evidence — the pattern geometry IS the bias signal. They do NOT need
    # a separate H4 bias confirmation (Alex enters these before H4 confirms).
    #
    # Range breaks and pullback continuations DO need H4 bias alignment
    # because they lack the structural reversal evidence.
    _SELF_CONFIRMING = {
        "head_and_shoulders", "inverted_head_and_shoulders",
        "double_top", "double_bottom",
    }
    _NEEDS_BIAS = {
        "consolidation_breakout_bearish", "consolidation_breakout_bullish",
        "break_retest_bearish", "break_retest_bullish",
    }

    if pr.pattern_type in _NEEDS_BIAS:
        # Range/break patterns: require H4 bias alignment
        if bias == "MIXED":
            return _reject(pr, "bias_mixed_range")
        if direction == "SHORT" and bias != "BEAR":
            return _reject(pr, "counter_trend_range")
        if direction == "LONG"  and bias != "BULL":
            return _reject(pr, "counter_trend_range")
    else:
        # Self-confirming reversal patterns (H&S, DT, DB):
        # H4 bias not required because Alex enters BEFORE H4 confirms.
        # But: STRONGLY opposite H4 bias = block (the EUR/USD Wk4 loser case).
        #   BULL + SHORT → block (don't fight clear uptrend with reversal short)
        #   BEAR + LONG  → block (don't fight clear downtrend with reversal long)
        #   MIXED + any  → allow (transitional = exactly Alex's setup window)
        # This preserves early-reversal entries while blocking pure counter-trend.
        if direction == "SHORT" and bias == "BULL":
            return _reject(pr, "counter_trend_confirmed_bull")
        if direction == "LONG"  and bias == "BEAR":
            return _reject(pr, "counter_trend_confirmed_bear")

    # ── B3 regime ─────────────────────────────────────────────────────────
    ok3, r3 = b3_regime_ok(h4, pair, direction)
    if not ok3:
        return _reject(pr, f"B3_{r3}")

    # ── Freshness: current bar must INTERACT with the retest zone ────────────
    # Alex's retest entry: candle wick touches neckline, body rejects back.
    # Zone check based on:  wick interaction OR close within 2× tolerance.
    # "Exact bar close in zone" is too narrow — entry triggers on rejection bar
    # which can close slightly outside the zone after touching it.
    # ── Zone check: touch + rejection (two-stage) ─────────────────────────
    # Stage 1 (touch): wick must reach within tol_touch of neckline
    # Stage 2 (reject): close must be within tol_close, on the BREAK side
    # For SHORT (neckline broke bearishly, price retests from below):
    #   wick reaches UP to neckline, close stays BELOW neckline
    # For LONG (neckline broke bullishly, price retests from above):
    #   wick reaches DOWN to neckline, close stays ABOVE neckline
    atr_raw     = compute_atr(h1.iloc[-50:]).iloc[-1] if len(h1) >= 50 else 0
    if pd.isna(atr_raw) or atr_raw <= 0:
        atr_raw = pr.neckline * 0.003
    tol_touch   = atr_raw * 0.50   # wick must reach within 0.5×ATR of level
    tol_close   = atr_raw * 0.25   # close within 0.25×ATR (on correct side)
    level       = pr.neckline
    bar         = h1.iloc[-1]
    if direction == "SHORT":
        # Wick reached UP to neckline from below; close rejected back below
        wick_ok  = bar["high"] >= level - tol_touch
        close_ok = bar["close"] <= level + tol_close
    else:
        # Wick reached DOWN to neckline from above; close rejected back above
        wick_ok  = bar["low"]  <= level + tol_touch
        close_ok = bar["close"] >= level - tol_close
    if not (wick_ok and close_ok):
        return _reject(pr, "not_in_entry_zone")

    # ── Rejection trigger required (engulfing / pin / strong rejection) ───
    # Must see an actual rejection candle AT the zone — not just price being near it.
    # Uses last 2 H1 bars (current + prior) to detect trigger.
    if len(h1) >= 2:
        triggered, trigger_name = detect_rejection_trigger(
            h1.iloc[-2:], direction, level, atr_raw)
        if not triggered:
            return _reject(pr, f"no_trigger_{trigger_name}")
    else:
        return _reject(pr, "insufficient_bars_for_trigger")

    # ── Confidence gate (existing system score) ────────────────────────────
    if pr.clarity < 0.40:
        return _reject(pr, f"low_clarity_{pr.clarity:.2f}")

    # ── Stop model v2 ─────────────────────────────────────────────────────
    # Priority: use tightest stop that (a) truly invalidates the setup and
    # (b) passes B2 ATR sanity. PatternDetector stop anchors the pattern
    # extreme; micro-stop anchors the retest rejection swing.
    entry = pr.entry_zone_low if direction == "SHORT" else pr.entry_zone_high

    atr_price = atr_h1 * pip_size(pair)
    buf_price = max(2 * pip_size(pair), 0.10 * atr_price)   # ≥2 pips or 10% ATR

    # Candidate 1: PatternDetector stop (right shoulder / peak / trough)
    stop_pattern = pr.stop_loss

    # Candidate 2: Micro-structure stop from recent H1 retest swing
    N_recent = 12
    recent   = h1.iloc[-N_recent:] if len(h1) >= N_recent else h1
    if direction == "SHORT":
        rejection_hi = recent["high"].max()
        stop_micro   = rejection_hi + buf_price
    else:
        rejection_lo = recent["low"].min()
        stop_micro   = rejection_lo - buf_price

    # Sanity: stops must be on the correct side of entry
    if direction == "SHORT":
        stop_pattern = max(stop_pattern, entry + pip_size(pair))  # at least 1 pip above entry
        stop_micro   = max(stop_micro,   entry + pip_size(pair))
    else:
        stop_pattern = min(stop_pattern, entry - pip_size(pair))
        stop_micro   = min(stop_micro,   entry - pip_size(pair))

    # Choose: try tightest valid stop first (micro), then pattern
    def try_stop(st):
        sp   = to_pips(pair, abs(entry - st))
        ok, r = b2_stop_valid(sp, atr_h1, pair_class)
        return ok, sp, r

    ok_micro, sp_micro, r_micro = try_stop(stop_micro)
    ok_pat,   sp_pat,   r_pat   = try_stop(stop_pattern)

    if ok_micro:
        stop, stop_pips, stop_model = stop_micro,   sp_micro, "MICRO_REJECTION"
    elif ok_pat:
        stop, stop_pips, stop_model = stop_pattern, sp_pat,   "PATTERN_ANCHOR"
    else:
        return _reject(pr, f"B2_no_valid_stop(micro={r_micro},pat={r_pat})")

    # ── TP (measured move first → HTF liquidity fallback) ─────────────────
    measured_move = abs(pr.target_1 - pr.neckline) if pr.target_1 else 0
    htf_liq       = find_htf_liquidity(h4, direction, entry)
    tp, rr, tp_method = build_tp(entry, stop, direction, measured_move, htf_liq)
    if tp is None:
        return _reject(pr, f"no_valid_tp_({tp_method})")

    return {
        "pair":        pair,
        "pair_class":  pair_class,
        "direction":   direction,
        "setup_type":  pr.pattern_type,
        "neckline":    round(pr.neckline, 6),
        "entry":       round(entry, 6),
        "stop":        round(stop,  6),
        "take_profit": round(tp,    6),
        "rr":          round(rr,    3),
        "tp_method":   tp_method,
        "stop_model":  stop_model,
        "stop_pips":   round(stop_pips, 1),
        "atr_h1_pips": round(atr_h1, 1),
        "clarity":     round(pr.clarity, 3),
        "open_time":   ts,
        "notes":       pr.notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VERSION STAMP  (audit trail for every run)
# ─────────────────────────────────────────────────────────────────────────────

def version_stamp(data: dict) -> dict:
    """Return {git_sha, cfg_hash, data_hash} for audit logging."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        sha = "unknown"

    cfg_hash  = hashlib.md5(json.dumps(CFG, sort_keys=True, default=str).encode()).hexdigest()[:8]
    # data hash: checksum of first+last close of each pair/TF
    data_repr = {
        p: {tf: (float(df.iloc[0]["close"]), float(df.iloc[-1]["close"]))
            for tf, df in tfs.items() if len(df) > 0}
        for p, tfs in data.items()
    } if data else {}
    data_hash = hashlib.md5(json.dumps(data_repr, sort_keys=True).encode()).hexdigest()[:8]
    return {"git_sha": sha, "cfg_hash": cfg_hash, "data_hash": data_hash}


# ─────────────────────────────────────────────────────────────────────────────
# ALEX DIAGNOSTIC — force-run engine on Alex's known entry timestamps
# ─────────────────────────────────────────────────────────────────────────────

# Known Alex entry timestamps (UTC, approximate to nearest hour)
ALEX_TRADES = [
    {"wk": 1,  "pair": "GBP_JPY", "dir": "SHORT", "ts": "2024-07-16T07:00:00Z", "note": "H&S @205"},
    {"wk": 2,  "pair": "USD_JPY", "dir": "SHORT", "ts": "2024-07-23T01:00:00Z", "note": "H&S @157.5"},
    {"wk": 3,  "pair": "USD_CHF", "dir": "SHORT", "ts": "2024-07-30T07:00:00Z", "note": "Break/retest @0.889"},
    {"wk": 6,  "pair": "GBP_CHF", "dir": "SHORT", "ts": "2024-08-20T07:00:00Z", "note": "DT @1.125"},
    {"wk": 8,  "pair": "USD_JPY", "dir": "SHORT", "ts": "2024-09-03T07:00:00Z", "note": "H&S @144"},
    {"wk": 10, "pair": "USD_CAD", "dir": "SHORT", "ts": "2024-09-17T07:00:00Z", "note": "Break/retest @1.35"},
    {"wk": 11, "pair": "USD_CAD", "dir": "SHORT", "ts": "2024-09-24T07:00:00Z", "note": "Re-entry @1.35"},
    {"wk": 12, "pair": "GBP_CHF", "dir": "SHORT", "ts": "2024-10-03T01:00:00Z", "note": "Consol break @1.125"},
    {"wk": 13, "pair": "USD_CHF", "dir": "LONG",  "ts": "2024-10-08T07:00:00Z", "note": "Retest @0.854"},
]

def run_alex_diagnostics(data: dict, verbose: bool = True) -> None:
    """
    For each known Alex trade, find the nearest H1 bar and explain
    exactly what the engine sees at that moment.
    """
    pattern_dt = PatternDetector()
    print(f"\n{'='*62}")
    print("Alex Trade Diagnostics — per-trade engine output")
    print(f"{'='*62}")

    for trade in ALEX_TRADES:
        pair = trade["pair"]
        ts_target = datetime.fromisoformat(trade["ts"].replace("Z", "+00:00"))
        if pair not in data:
            print(f"\n  Wk{trade['wk']:2d} {pair} — DATA MISSING")
            continue

        h1  = data[pair].get("H1")
        h4  = data[pair].get("H4")
        d1  = data[pair].get("D")
        if h1 is None or h4 is None or d1 is None:
            print(f"\n  Wk{trade['wk']:2d} {pair} — TF DATA MISSING")
            continue

        # Find closest H1 bar at or before the target timestamp
        mask = h1.index <= ts_target
        if not mask.any():
            print(f"\n  Wk{trade['wk']:2d} {pair} — NO H1 BAR BEFORE {ts_target}")
            continue

        h1_now = h1[mask]
        h4_now = h4[h4.index <= ts_target]
        d1_now = d1[d1.index <= ts_target]

        bar_ts   = h1_now.index[-1]
        bar      = h1_now.iloc[-1]
        h1_win   = h1_now.iloc[-200:]
        atr_val  = atr_pips(h1_win, pair)

        # Bias
        bias, bias_reason = detect_bias(d1_now, h4_now)

        # B3
        b3_ok, b3_reason = b3_regime_ok(h4_now, pair, trade["dir"])

        # Patterns
        patterns = pattern_dt.detect_all(h1_win)

        print(f"\n  Wk{trade['wk']:2d} {pair} {trade['dir']} — {trade['note']}")
        print(f"       Bar: {bar_ts}  close={bar['close']:.5f}  ATR={atr_val:.1f}p")
        print(f"       Bias: {bias} ({bias_reason})")
        print(f"       B3:   {'PASS' if b3_ok else 'FAIL'} ({b3_reason})")
        print(f"       Patterns found: {len(patterns)}")

        if patterns:
            for pr in patterns[:5]:
                _reject_counts.clear()
                sig = adapt_pattern(pr, pair, h1_win, h4_now, bias, ts_target)
                rejects = list(_reject_counts.keys())
                status = "✅ SIGNAL" if sig else f"❌ {', '.join(rejects)}"
                print(f"         {pr.pattern_type:35s} neckline={pr.neckline:.4f}  "
                      f"clarity={pr.clarity:.2f}  {status}")
        else:
            print(f"         (no patterns in 200-bar H1 window)")

    _reject_counts.clear()
    print()


_reject_counts: dict = {}

def _reject(pr, reason: str) -> None:
    _reject_counts[reason] = _reject_counts.get(reason, 0) + 1
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TRADE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def manage_position(pos: dict, bar: pd.Series, ts: datetime) -> tuple[str, str]:
    """
    Returns (action, reason):
      action: "hold" | "close" | "move_be"
    """
    mgmt  = CFG["management"]
    entry = pos["entry"]
    stop  = pos["stop"]
    tp    = pos["take_profit"]
    dir_  = pos["direction"]
    risk  = abs(entry - stop)
    pc    = pos["pair_class"]

    hi, lo = bar["high"], bar["low"]
    cl     = bar["close"]

    if dir_ == "SHORT":
        pnl_r  = (entry - cl) / risk
        hit_tp = lo <= tp
        hit_sl = hi >= stop
    else:
        pnl_r  = (cl - entry) / risk
        hit_tp = hi >= tp
        hit_sl = lo <= stop

    # Friday force close
    if b1_force_close(ts):
        return "close", "friday_force_close"

    # Stop/TP
    if hit_sl:
        return "close", "stop_loss"
    if hit_tp:
        return "close", "take_profit"

    # Move to BE at +2R
    if pnl_r >= mgmt["move_be_at_r"] and not pos.get("be_moved"):
        return "move_be", "be_at_2R"

    # F4 v2 conditional time stop
    ts_cfg = mgmt["time_stop"]
    if ts_cfg["enabled"]:
        hours_limit = ts_cfg["hours_by_class"][pc]
        elapsed     = (ts - pos["open_time"]).total_seconds() / 3600
        if elapsed >= hours_limit:
            if pnl_r < ts_cfg["min_progress_r"]:
                if ts_cfg["requires_reentry_of_level"]:
                    level_zone = pos.get("neckline", entry)
                    zone_tol   = CFG["atr"]["retest_tol_frac"] * risk
                    in_zone    = abs(cl - level_zone) <= zone_tol
                    if in_zone:
                        return "close", f"time_stop_{elapsed:.0f}h_zone_reentry"
                else:
                    return "close", f"time_stop_{elapsed:.0f}h"

    return "hold", ""


def exit_price(pos: dict, bar: pd.Series, reason: str) -> float:
    """Return the bar's fill price for the given exit reason."""
    if reason == "stop_loss":
        return pos["stop"]
    if reason == "take_profit":
        return pos["take_profit"]
    if reason == "friday_force_close":
        return bar["close"]
    return bar["close"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

_CACHE: dict = {}

def _candle_rows(raw: list) -> list:
    """Handle both OandaClient-unwrapped and raw OANDA mid-price formats."""
    rows = []
    for c in raw:
        if "open" in c:
            # Already unwrapped by OandaClient.get_candles
            rows.append({
                "time":  pd.Timestamp(c["time"]),
                "open":  float(c["open"]),
                "high":  float(c["high"]),
                "low":   float(c["low"]),
                "close": float(c["close"]),
            })
        else:
            # Raw OANDA format: mid.o / mid.h / mid.l / mid.c
            mid = c.get("mid", {})
            rows.append({
                "time":  pd.Timestamp(c["time"]),
                "open":  float(mid.get("o", 0)),
                "high":  float(mid.get("h", 0)),
                "low":   float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
            })
    return rows


def fetch_candles_range(client: OandaClient, pair: str, granularity: str,
                        from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """Fetch candles for a specific date range using OANDA from/to params."""
    from src.exchange.oanda_client import INSTRUMENT_MAP
    instrument = INSTRUMENT_MAP.get(pair, pair.replace("/", "_"))
    # OANDA expects RFC3339; chunk into ≤5000-bar requests
    all_rows = []
    chunk_start = from_dt
    while chunk_start < to_dt:
        params = {
            "granularity": granularity,
            "from": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to":   to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": "M",
        }
        try:
            data = client._get(f"/v3/instruments/{instrument}/candles", params=params)
            candles = data.get("candles", [])
        except Exception:
            break
        if not candles:
            break
        all_rows.extend(_candle_rows(candles))
        last_ts = pd.Timestamp(candles[-1]["time"])
        if len(candles) < 5000:
            break
        chunk_start = last_ts.to_pydatetime() + timedelta(seconds=1)

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.DataFrame(all_rows).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_all_tfs(client: OandaClient, pair: str,
                  from_dt: Optional[datetime] = None,
                  to_dt:   Optional[datetime] = None) -> dict:
    """
    Fetch D, H4, H1 (and M15 if MODE_B) for a pair.
    If from_dt/to_dt provided, uses date-range fetch (required for backtest windows
    more than ~2 months in the past).  Falls back to count-based fetch for recent data.
    """
    if from_dt is not None and to_dt is not None:
        # Add lookback buffer for bias/ATR warmup
        buf = timedelta(days=90)
        fetch_from = from_dt - buf
        tfs = ["D", "H4", "H1"]
        if CFG["execution_mode"] == "M15":
            tfs.append("M15")
        return {tf: fetch_candles_range(client, pair, tf, fetch_from, to_dt)
                for tf in tfs}
    else:
        # Recent data (no historical window)
        tfs = {"D": 500, "H4": 800, "H1": 1500}
        if CFG["execution_mode"] == "M15":
            tfs["M15"] = 2000
        key_base = (pair, "count")
        if key_base in _CACHE:
            return _CACHE[key_base]
        result = {}
        for tf, cnt in tfs.items():
            raw = client.get_candles(pair, granularity=tf, count=cnt)
            rows = _candle_rows(raw)
            df = pd.DataFrame(rows).set_index("time").sort_index()
            result[tf] = df
        _CACHE[key_base] = result
        return result


# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZING  (fractional risk)
# ─────────────────────────────────────────────────────────────────────────────

def position_size(balance: float, stop_pips: float, pair: str) -> float:
    """Dollar risk = risk_pct × balance. Units = risk / (stop_pips × pip_value)."""
    risk_amt   = balance * CFG["risk_pct"]
    ps         = pip_size(pair)
    # For majors: 1 unit moves ~$0.0001 per unit; use 100K unit lot as base
    # OANDA: pip_value per unit ≈ pip_size (for non-JPY vs USD crosses, approximately)
    pip_value_per_unit = ps
    if stop_pips <= 0:
        return 0
    return risk_amt / (stop_pips * pip_value_per_unit)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(start: str, end: str, balance: float,
                 verbose: bool = True, diag: bool = False) -> dict:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    client = OandaClient(
        api_key=os.environ["OANDA_API_KEY"],
        account_id=os.environ["OANDA_ACCOUNT_ID"],
    )

    pairs    = CFG["pair_whitelist"] if CFG["pair_whitelist_enabled"] else list(CFG["pair_class_map"])
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    print(f"\n{'='*62}")
    print(f"VlaudeBot Backtest  |  {start} → {end}  |  ${balance:,.0f}")
    print(f"Pairs: {pairs}")
    print(f"Mode: {CFG['execution_mode']}")
    print(f"{'='*62}\n")

    # ── Fetch all data ────────────────────────────────────────────────────
    data = {}
    for pair in pairs:
        print(f"  Fetching {pair}...", end=" ", flush=True)
        try:
            data[pair] = fetch_all_tfs(client, pair, from_dt=start_dt, to_dt=end_dt)
            rows = len(data[pair].get("H1", pd.DataFrame()))
            print(f"ok  ({rows} H1 bars)")
        except Exception as e:
            print(f"ERROR: {e}")

    # ── Alex diagnostics (optional) ──────────────────────────────────────
    if diag:
        run_alex_diagnostics(data)

    # ── Build unified H1 timeline ─────────────────────────────────────────
    # Use USD_JPY H1 as the master clock (most liquid, fewest gaps)
    master_pair = "USD_JPY" if "USD_JPY" in data else pairs[0]
    h1_master   = data[master_pair]["H1"]
    timeline    = h1_master.index[
        (h1_master.index >= start_dt) &
        (h1_master.index <= end_dt)
    ]

    # Version stamp
    vstamp = version_stamp(data)
    print(f"\n  Timeline: {len(timeline)} H1 bars  "
          f"({timeline[0].strftime('%Y-%m-%d')} → {timeline[-1].strftime('%Y-%m-%d')})")
    print(f"  Version:  git={vstamp['git_sha']}  cfg={vstamp['cfg_hash']}  data={vstamp['data_hash']}\n")

    # ── State ─────────────────────────────────────────────────────────────
    equity       = balance
    open_pos     = {}    # pair → position dict
    trades       = []
    pattern_dt   = PatternDetector()
    last_closed    = {}    # pair → datetime of last close (for cooldown)
    last_entry_key = {}    # pair → (neckline_rounded, setup_type) — cleared after cooldown
    COOLDOWN_HOURS = 120   # 5 days — same-pair cooldown after any close

    # ── Bar loop ──────────────────────────────────────────────────────────
    for ts in timeline:
        ts_aware = ts.to_pydatetime().replace(tzinfo=timezone.utc)

        # ── 1. Manage open positions ───────────────────────────────────────
        for pair in list(open_pos.keys()):
            pos  = open_pos[pair]
            h1   = data[pair].get("H1")
            if h1 is None or ts not in h1.index:
                continue
            bar = h1.loc[ts]

            action, reason = manage_position(pos, bar, ts_aware)

            if action == "move_be":
                risk = abs(pos["entry"] - pos["stop"])
                be_buf = CFG["management"]["be_buffer_r"] * risk
                pos["stop"]     = (pos["entry"] + be_buf) if pos["direction"] == "SHORT" else (pos["entry"] - be_buf)
                pos["be_moved"] = True

            elif action == "close":
                fill = exit_price(pos, bar, reason)
                if pos["direction"] == "SHORT":
                    pnl_pips = (pos["entry"] - fill) / pip_size(pair)
                else:
                    pnl_pips = (fill - pos["entry"]) / pip_size(pair)
                # Always use INITIAL stop pips for R — stop may have moved to BE
                risk_pips  = pos.get("initial_stop_pips") or \
                             to_pips(pair, abs(pos["entry"] - pos["stop"]))
                pnl_r      = pnl_pips / risk_pips if risk_pips > 0 else 0
                risk_amount = equity * CFG["risk_pct"]
                pnl_dollar  = pnl_r * risk_amount

                equity += pnl_dollar

                trade_rec = {
                    "pair":       pair,
                    "direction":  pos["direction"],
                    "setup_type": pos["setup_type"],
                    "entry":      pos["entry"],
                    "stop":       pos["stop"],
                    "take_profit": pos["take_profit"],
                    "fill":       fill,
                    "open_time":  pos["open_time"].isoformat(),
                    "close_time": ts_aware.isoformat(),
                    "reason":     reason,
                    "pnl_pips":   round(pnl_pips, 1),
                    "pnl_r":      round(pnl_r, 3),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "equity":     round(equity, 2),
                    "rr":         pos["rr"],
                }
                trades.append(trade_rec)
                last_closed[pair]   = ts_aware
                del open_pos[pair]

                emoji = "✅" if pnl_dollar > 0 else "❌"
                print(f"  {emoji} CLOSE {pair:8s} {pos['direction']:5s}  "
                      f"{reason:25s}  {pnl_r:+.2f}R  "
                      f"${pnl_dollar:+,.0f}  →  ${equity:,.0f}")

        # ── 2. Scan for new entries ────────────────────────────────────────
        if not b1_can_enter(ts_aware):
            continue

        # Global position limit — Alex's rule: ONE trade at a time
        if len(open_pos) >= CFG["max_open_positions"]:
            continue

        for pair in pairs:
            if pair in open_pos:
                continue
            if pair not in data:
                continue

            # Cooldown: don't re-enter within COOLDOWN_HOURS of last close
            if pair in last_closed:
                elapsed_h = (ts_aware - last_closed[pair]).total_seconds() / 3600
                if elapsed_h < COOLDOWN_HOURS:
                    continue

            h1  = data[pair].get("H1")
            h4  = data[pair].get("H4")
            d1  = data[pair].get("D")
            if h1 is None or h4 is None or d1 is None:
                continue

            # Slice to current bar (no look-ahead)
            mask_h1 = h1.index <= ts
            mask_h4 = h4.index <= ts
            mask_d1 = d1.index <= ts

            h1_now = h1[mask_h1]
            h4_now = h4[mask_h4]
            d1_now = d1[mask_d1]

            if len(h1_now) < 50 or len(h4_now) < 20 or len(d1_now) < 10:
                continue

            # Bias
            bias, bias_reason = detect_bias(d1_now, h4_now)

            # Run pattern detection on H1 window (last 200 bars)
            h1_window = h1_now.iloc[-200:]
            patterns  = pattern_dt.detect_all(h1_window)

            for pr in patterns:
                sig = adapt_pattern(pr, pair, h1_window, h4_now,
                                    bias, ts_aware)
                if sig is None:
                    continue

                # Dedup: within cooldown window, block the EXACT same setup
                key_sig = (round(sig["neckline"], 4), sig["setup_type"])
                if last_entry_key.get(pair) == key_sig and pair in last_closed:
                    elapsed_h = (ts_aware - last_closed[pair]).total_seconds() / 3600
                    if elapsed_h < COOLDOWN_HOURS:
                        continue   # still in cooldown on same pattern

                # Weekend gate
                if not b1_can_enter(ts_aware):
                    break

                # Enter position — store initial stop pips for consistent R calc
                initial_stop_pips = to_pips(pair, abs(sig["entry"] - sig["stop"]))
                open_pos[pair]       = {**sig, "open_time": ts_aware,
                                        "initial_stop_pips": initial_stop_pips}
                last_entry_key[pair] = key_sig

                print(f"  📈 ENTER {pair:8s} {sig['direction']:5s}  "
                      f"{sig['setup_type']:30s}  "
                      f"entry={sig['entry']:.5f}  "
                      f"stop={sig['stop']:.5f}  "
                      f"tp={sig['take_profit']:.5f}  "
                      f"RR={sig['rr']:.2f}  "
                      f"({sig['tp_method']})")
                break  # one signal per pair per bar

    # ── Force close anything still open at end ────────────────────────────
    final_ts = timeline[-1].to_pydatetime().replace(tzinfo=timezone.utc)
    for pair, pos in list(open_pos.items()):
        h1  = data[pair].get("H1")
        if h1 is None:
            continue
        final_bar = h1.iloc[-1]
        fill      = final_bar["close"]
        if pos["direction"] == "SHORT":
            pnl_pips = (pos["entry"] - fill) / pip_size(pair)
        else:
            pnl_pips = (fill - pos["entry"]) / pip_size(pair)
        risk_pips   = pos.get("initial_stop_pips") or \
                      to_pips(pair, abs(pos["entry"] - pos["stop"]))
        pnl_r       = pnl_pips / risk_pips if risk_pips > 0 else 0
        risk_amount = equity * CFG["risk_pct"]
        pnl_dollar  = pnl_r * risk_amount
        equity     += pnl_dollar

        trade_rec = {
            "pair":       pair,
            "direction":  pos["direction"],
            "setup_type": pos["setup_type"],
            "entry":      pos["entry"],
            "stop":       pos["stop"],
            "take_profit": pos["take_profit"],
            "fill":       fill,
            "open_time":  pos["open_time"].isoformat(),
            "close_time": final_ts.isoformat(),
            "reason":     "window_end",
            "pnl_pips":   round(pnl_pips, 1),
            "pnl_r":      round(pnl_r, 3),
            "pnl_dollar": round(pnl_dollar, 2),
            "equity":     round(equity, 2),
            "rr":         pos["rr"],
        }
        trades.append(trade_rec)
        del open_pos[pair]

    # ── Summary ───────────────────────────────────────────────────────────
    wins   = [t for t in trades if t["pnl_dollar"] > 0]
    losses = [t for t in trades if t["pnl_dollar"] <= 0]
    ret    = (equity / balance - 1) * 100

    print(f"\n{'='*62}")
    print(f"  Trades:   {len(trades)}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Win rate: {len(wins)/len(trades)*100:.0f}%"  if trades else "  No trades")
    print(f"  Return:   {ret:+.1f}%")
    print(f"  Balance:  ${balance:,.0f} → ${equity:,.2f}")
    print(f"{'='*62}\n")

    # Rejection breakdown
    if _reject_counts:
        print("\n── Rejection breakdown ──")
        for reason, cnt in sorted(_reject_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {cnt:5d}  {reason}")

    # Save results
    vstamp = version_stamp(data)
    result = {
        "start":       start,
        "end":         end,
        "balance_in":  balance,
        "balance_out": round(equity, 2),
        "return_pct":  round(ret, 2),
        "n_trades":    len(trades),
        "n_wins":      len(wins),
        "n_losses":    len(losses),
        "version":     vstamp,
        "trades":      trades,
    }
    out = Path(__file__).parent.parent / "logs" / "vlaudebot_results.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            **{k: v for k, v in result.items() if k != "trades"},
        }) + "\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="VlaudeBot backtester v1")
    ap.add_argument("--start",   default=CFG["default_start"])
    ap.add_argument("--end",     default=CFG["default_end"])
    ap.add_argument("--balance", type=float, default=CFG["default_balance"])
    ap.add_argument("--mode",    default="H1", choices=["H1", "M15"])
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--diag",    action="store_true", default=False,
                    help="Run Alex-trade diagnostics before main loop")
    args = ap.parse_args()
    CFG["execution_mode"] = args.mode
    run_backtest(args.start, args.end, args.balance, args.verbose, args.diag)
