"""
At-Level Candle Confirmations
==============================
Alex's "daily driver" confirmation patterns, valid ONLY when price has recently
tested the key level (psychological round number / pattern neckline).

Two families:
  1. Morning Star / Evening Star  — 3-candle reversal sequence
  2. Strict Hammer / Shooting Star — tight single-bar rejection

RULE (from transcript): confirmation is only meaningful when the retest zone
was touched recently. We define "recently" as within the last LEVEL_TOUCH_H1_BARS
H1 bars (≈ "5 M15 bars" in Alex's wording).

The confirmation candle must also close back on the correct side of the level:
  - bearish setup: confirmation closes at or below the level
  - bullish setup: confirmation closes at or above the level

All thresholds are configurable via strategy_config.py.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

from .entry_signal import EntrySignal, is_indecision
from . import strategy_config as _cfg


# ─────────────────────────────────────────────────────────────────────────────
# Level-touch gate
# ─────────────────────────────────────────────────────────────────────────────

def level_was_touched_recently(
    df: pd.DataFrame,
    level: float,
    lookback: int,
    tolerance_pct: float,
) -> bool:
    """
    Returns True if any of the last `lookback` *closed* H1 bars traded within
    `tolerance_pct` of `level`.

    We check the full [low, high] range — not just close — because a wick into
    the level is the "test" in Alex's terminology.
    """
    if len(df) < lookback + 1 or level <= 0:
        return False
    tol = abs(level) * tolerance_pct
    recent = df.iloc[-(lookback + 1):-1]   # last N closed bars (exclude current)
    for _, bar in recent.iterrows():
        if bar["low"] <= level + tol and bar["high"] >= level - tol:
            return True
    return False


def _body_pct(bar: pd.Series) -> float:
    """Absolute body size as fraction of candle range. Returns 0 if doji/flat."""
    r = bar["high"] - bar["low"]
    if r < 1e-10:
        return 0.0
    return abs(bar["close"] - bar["open"]) / r


def _is_bearish(bar: pd.Series) -> bool:
    return bar["close"] < bar["open"]


def _is_bullish(bar: pd.Series) -> bool:
    return bar["close"] > bar["open"]


def _candle_midpoint(bar: pd.Series) -> float:
    """Midpoint of the candle body."""
    return (bar["open"] + bar["close"]) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Morning Star / Evening Star
# ─────────────────────────────────────────────────────────────────────────────

def detect_morning_evening_star(
    df: pd.DataFrame,
    direction: str,
    level: float,
    lookback: Optional[int] = None,
) -> Optional[EntrySignal]:
    """
    Detect Morning Star (bullish) or Evening Star (bearish) at a key level.

    Pattern (uses last 3 closed bars: bar1, bar2, bar3=signal):
      Evening Star (bearish) — direction="short":
        bar1: large bullish candle  (body ≥ STAR_BAR1_MIN_BODY_PCT)
        bar2: small star             (body ≤ STAR_BAR2_MAX_BODY_PCT)
        bar3: large bearish candle,  closes below midpoint of bar1

      Morning Star (bullish) — direction="long":
        bar1: large bearish candle
        bar2: small star
        bar3: large bullish candle,  closes above midpoint of bar1

    Additional gate:
      - Level must have been touched in the last LEVEL_TOUCH_H1_BARS bars
      - Signal candle (bar3) must close on the correct side of the level

    Returns an EntrySignal or None.
    """
    if len(df) < 4:
        return None

    _lookback = lookback if lookback is not None else _cfg.LEVEL_TOUCH_H1_BARS
    if not level_was_touched_recently(df, level, _lookback, _cfg.LEVEL_TOUCH_TOLERANCE_PCT):
        return None

    bar1 = df.iloc[-3]
    bar2 = df.iloc[-2]
    bar3 = df.iloc[-1]   # current / just-closed confirmation bar

    b1_body = _body_pct(bar1)
    b2_body = _body_pct(bar2)
    b3_body = _body_pct(bar3)

    if direction == "short":
        # ── Evening Star ─────────────────────────────────────────────
        # bar1: large bullish
        if not (_is_bullish(bar1) and b1_body >= _cfg.STAR_BAR1_MIN_BODY_PCT):
            return None
        # bar2: small body (star — ALLOWED to be indecision/doji; that's the pattern)
        if b2_body > _cfg.STAR_BAR2_MAX_BODY_PCT:
            return None
        # bar3: bearish, closes below midpoint of bar1, must NOT be indecision
        if not _is_bearish(bar3):
            return None
        if b3_body < _cfg.STAR_BAR3_MIN_BODY_PCT:
            return None
        if _cfg.INDECISION_FILTER_ENABLED and is_indecision(bar3):
            return None   # confirmation candle must be decisive (rule: "no doji at trigger")
        bar1_mid = _candle_midpoint(bar1)
        if bar3["close"] >= bar1_mid:
            return None
        # bar3 close must be at or below the level (back on sell side)
        if bar3["close"] > level * (1 + _cfg.LEVEL_TOUCH_TOLERANCE_PCT):
            return None

        # Strength: how far below bar1 midpoint does bar3 close?
        penetration = (bar1_mid - bar3["close"]) / (bar1_mid - bar1["open"] + 1e-10)
        strength = min(1.0, 0.55 + 0.30 * penetration + 0.15 * b3_body)
        return EntrySignal(
            signal_type="evening_star",
            direction="short",
            strength=round(strength, 3),
            candle_index=-1,
            close=float(bar3["close"]),
            body_size=float(abs(bar3["open"] - bar3["close"])),
            notes=(f"Evening Star at level {level:.5f} | "
                   f"bar1 body={b1_body:.0%} bar2 body={b2_body:.0%} bar3 body={b3_body:.0%}"),
        )

    elif direction == "long":
        # ── Morning Star ─────────────────────────────────────────────
        if not (_is_bearish(bar1) and b1_body >= _cfg.STAR_BAR1_MIN_BODY_PCT):
            return None
        # bar2: allowed to be indecision — that's the "star" in the middle
        if b2_body > _cfg.STAR_BAR2_MAX_BODY_PCT:
            return None
        # bar3: bullish confirmation — must NOT be indecision
        if not _is_bullish(bar3):
            return None
        if b3_body < _cfg.STAR_BAR3_MIN_BODY_PCT:
            return None
        if _cfg.INDECISION_FILTER_ENABLED and is_indecision(bar3):
            return None   # confirmation candle must be decisive
        bar1_mid = _candle_midpoint(bar1)
        if bar3["close"] <= bar1_mid:
            return None
        # bar3 close must be at or above the level (back on buy side)
        if bar3["close"] < level * (1 - _cfg.LEVEL_TOUCH_TOLERANCE_PCT):
            return None

        penetration = (bar3["close"] - bar1_mid) / (bar1["open"] - bar1_mid + 1e-10)
        strength = min(1.0, 0.55 + 0.30 * penetration + 0.15 * b3_body)
        return EntrySignal(
            signal_type="morning_star",
            direction="long",
            strength=round(strength, 3),
            candle_index=-1,
            close=float(bar3["close"]),
            body_size=float(abs(bar3["open"] - bar3["close"])),
            notes=(f"Morning Star at level {level:.5f} | "
                   f"bar1 body={b1_body:.0%} bar2 body={b2_body:.0%} bar3 body={b3_body:.0%}"),
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Strict Hammer / Shooting Star
# ─────────────────────────────────────────────────────────────────────────────

def detect_strict_hammer_shooting_star(
    df: pd.DataFrame,
    direction: str,
    level: float,
    lookback: Optional[int] = None,
) -> Optional[EntrySignal]:
    """
    Detect a Strict Hammer (bullish) or Shooting Star (bearish) at a key level.

    Stricter than the existing pin_bar() in entry_signal.py:
      - Rejection wick ≥ STRICT_HAMMER_WICK_BODY_RATIO × body (default 3× vs 2×)
      - Body ≤ STRICT_HAMMER_MAX_BODY_RANGE_PCT of range (default 25%)
      - Opposite wick ≤ STRICT_HAMMER_MAX_OPPOSITE_PCT of range (default 10%)
      - Close in outer STRICT_HAMMER_CLOSE_OUTER_PCT of range (default 35%)
      - Level touched within LEVEL_TOUCH_H1_BARS bars
      - Signal candle closes on correct side of the level

    Checks the last 2 closed bars (like the existing detector lookback).
    """
    if len(df) < 3:
        return None

    _lookback = lookback if lookback is not None else _cfg.LEVEL_TOUCH_H1_BARS
    if not level_was_touched_recently(df, level, _lookback, _cfg.LEVEL_TOUCH_TOLERANCE_PCT):
        return None

    signals: list[EntrySignal] = []

    for i in (-2, -1):
        try:
            bar = df.iloc[i]
        except IndexError:
            continue

        o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
        candle_range = h - l
        if candle_range < 1e-10:
            continue

        body      = abs(c - o)
        body_high = max(o, c)
        body_low  = min(o, c)
        upper_wick = h - body_high
        lower_wick = body_low - l

        body_pct     = body / candle_range
        upper_wick_pct = upper_wick / candle_range
        lower_wick_pct = lower_wick / candle_range

        # Minimum body floor to avoid doji-trivial ratio
        body_floor = max(body, candle_range * 0.005)

        if direction == "short":
            # Shooting Star: dominant upper wick
            if not (
                upper_wick >= _cfg.STRICT_HAMMER_WICK_BODY_RATIO * body_floor
                and upper_wick > lower_wick
                and body_pct   <= _cfg.STRICT_HAMMER_MAX_BODY_RANGE_PCT
                and lower_wick_pct <= _cfg.STRICT_HAMMER_MAX_OPPOSITE_PCT
                and c <= l + _cfg.STRICT_HAMMER_CLOSE_OUTER_PCT * candle_range
                and c <= level * (1 + _cfg.LEVEL_TOUCH_TOLERANCE_PCT)  # closes on sell side
            ):
                continue
            strength = min(1.0,
                0.50
                + 0.20 * min(1.0, upper_wick_pct / 0.60)   # reward big upper wick
                + 0.15 * min(1.0, (1 - body_pct) / 0.80)   # reward small body
                + 0.15 * (1 - lower_wick_pct / max(_cfg.STRICT_HAMMER_MAX_OPPOSITE_PCT, 1e-9))  # reward tiny lower wick
            )
            signals.append(EntrySignal(
                signal_type="shooting_star_strict",
                direction="short",
                strength=round(strength, 3),
                candle_index=i,
                close=c,
                body_size=body,
                notes=(f"Strict Shooting Star at level {level:.5f} | "
                       f"uwk={upper_wick_pct:.0%} body={body_pct:.0%} lwk={lower_wick_pct:.0%}"),
            ))

        elif direction == "long":
            # Hammer: dominant lower wick
            if not (
                lower_wick >= _cfg.STRICT_HAMMER_WICK_BODY_RATIO * body_floor
                and lower_wick > upper_wick
                and body_pct   <= _cfg.STRICT_HAMMER_MAX_BODY_RANGE_PCT
                and upper_wick_pct <= _cfg.STRICT_HAMMER_MAX_OPPOSITE_PCT
                and c >= h - _cfg.STRICT_HAMMER_CLOSE_OUTER_PCT * candle_range
                and c >= level * (1 - _cfg.LEVEL_TOUCH_TOLERANCE_PCT)  # closes on buy side
            ):
                continue
            strength = min(1.0,
                0.50
                + 0.20 * min(1.0, lower_wick_pct / 0.60)
                + 0.15 * min(1.0, (1 - body_pct) / 0.80)
                + 0.15 * (1 - upper_wick_pct / max(_cfg.STRICT_HAMMER_MAX_OPPOSITE_PCT, 1e-9))
            )
            signals.append(EntrySignal(
                signal_type="hammer_strict",
                direction="long",
                strength=round(strength, 3),
                candle_index=i,
                close=c,
                body_size=body,
                notes=(f"Strict Hammer at level {level:.5f} | "
                       f"lwk={lower_wick_pct:.0%} body={body_pct:.0%} uwk={upper_wick_pct:.0%}"),
            ))

    if not signals:
        return None
    signals.sort(key=lambda s: s.strength, reverse=True)
    return signals[0]


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher — called from entry_signal.has_signal()
# ─────────────────────────────────────────────────────────────────────────────

def detect_at_level_confirmation(
    df: pd.DataFrame,
    direction: str,
    level: float,
    mode: str,
) -> Optional[EntrySignal]:
    """
    Dispatch to the correct at-level detector based on ENTRY_TRIGGER_MODE.

    Called when mode is one of:
      engulf_or_star_at_level
      engulf_or_strict_pin_at_level
      engulf_or_star_or_strict_pin_at_level

    Returns the strongest matching signal or None.
    """
    candidates: list[EntrySignal] = []

    want_star = mode in (
        "engulf_or_star_at_level",
        "engulf_or_star_or_strict_pin_at_level",
    )
    want_strict_pin = mode in (
        "engulf_or_strict_pin_at_level",
        "engulf_or_star_or_strict_pin_at_level",
    )

    if want_star:
        sig = detect_morning_evening_star(df, direction, level)
        if sig:
            candidates.append(sig)

    if want_strict_pin:
        sig = detect_strict_hammer_shooting_star(df, direction, level)
        if sig:
            candidates.append(sig)

    if not candidates:
        return None
    candidates.sort(key=lambda s: s.strength, reverse=True)
    return candidates[0]
