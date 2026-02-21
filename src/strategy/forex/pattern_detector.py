"""
Pattern Detector

Detects the three core patterns from the Set & Forget strategy:
  1. Head & Shoulders / Inverted H&S
  2. Double Top / Double Bottom
  3. Break & Retest (the backbone of every trade)

Also handles:
  - Structure shift detection (higher high/lower low series)
  - Trend identification across timeframes

All patterns are scored by clarity (0.0–1.0).
A pattern by itself is NOT an entry signal — it identifies the setup zone.
The engulfing candle (entry_signal.py) is required to enter.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class Trend(Enum):
    STRONG_BULLISH  = "strong_bullish"
    BULLISH         = "bullish"
    NEUTRAL         = "neutral"
    BEARISH         = "bearish"
    STRONG_BEARISH  = "strong_bearish"


@dataclass
class PatternResult:
    pattern_type: str        # 'head_and_shoulders', 'double_top', 'break_retest', etc.
    direction: str           # 'bearish' (sell) or 'bullish' (buy)
    neckline: float          # key level to watch for break/retest
    entry_zone_low: float    # entry zone bottom
    entry_zone_high: float   # entry zone top
    stop_loss: float         # logical stop placement
    target_1: float          # conservative target (1:2 R:R)
    target_2: float          # extended target (1:4+ R:R)
    clarity: float           # 0.0–1.0 (1.0 = textbook pattern)
    notes: str = ""

    @property
    def risk(self) -> float:
        if self.direction == 'bearish':
            return abs(self.entry_zone_high - self.stop_loss)
        return abs(self.entry_zone_low - self.stop_loss)

    @property
    def reward_1(self) -> float:
        if self.direction == 'bearish':
            return abs(self.entry_zone_high - self.target_1)
        return abs(self.target_1 - self.entry_zone_low)

    @property
    def rr_ratio_1(self) -> float:
        return self.reward_1 / self.risk if self.risk > 0 else 0


class PatternDetector:
    """
    Detects trade patterns on OHLC data.

    Recommended: run on daily candles for primary detection,
    then confirm on 4H for entry timing.
    """

    def __init__(
        self,
        swing_window: int = 5,
        min_pattern_bars: int = 10,
        max_pattern_bars: int = 60,
        tolerance_pct: float = 0.3,
    ):
        self.swing_window = swing_window
        self.min_bars = min_pattern_bars
        self.max_bars = max_pattern_bars
        self.tol = tolerance_pct / 100

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect_trend(self, df: pd.DataFrame) -> Trend:
        """
        Detect market structure trend from OHLC.
        Uses higher-high/higher-low and lower-high/lower-low series.
        """
        if len(df) < 20:
            return Trend.NEUTRAL

        highs = df['high'].values
        lows  = df['low'].values

        # Find last 4 significant swing points
        swing_highs = self._find_swings(highs, 'high')[-4:]
        swing_lows  = self._find_swings(lows,  'low')[-4:]

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return Trend.NEUTRAL

        # Check HH/HL series (bullish)
        hh = all(swing_highs[i] < swing_highs[i+1] for i in range(len(swing_highs)-1))
        hl = all(swing_lows[i]  < swing_lows[i+1]  for i in range(len(swing_lows)-1))

        # Check LH/LL series (bearish)
        lh = all(swing_highs[i] > swing_highs[i+1] for i in range(len(swing_highs)-1))
        ll = all(swing_lows[i]  > swing_lows[i+1]  for i in range(len(swing_lows)-1))

        if hh and hl:
            return Trend.STRONG_BULLISH
        if hh or hl:
            return Trend.BULLISH
        if lh and ll:
            return Trend.STRONG_BEARISH
        if lh or ll:
            return Trend.BEARISH
        return Trend.NEUTRAL

    def detect_all(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detect all patterns, return sorted by clarity descending."""
        results = []
        results.extend(self._detect_head_and_shoulders(df))
        results.extend(self._detect_double_top_bottom(df))
        results.extend(self._detect_break_retest(df))
        results.sort(key=lambda r: r.clarity, reverse=True)
        return results

    def best_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Return the highest-clarity pattern, or None."""
        results = self.detect_all(df)
        return results[0] if results else None

    # ------------------------------------------------------------------ #
    # Head & Shoulders
    # ------------------------------------------------------------------ #

    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        if len(df) < self.min_bars:
            return results

        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        n = len(df)

        # Bearish H&S: left shoulder (high) → head (higher high) → right shoulder (lower high)
        swing_highs_idx = self._find_swing_indices(highs, 'high')
        for i in range(len(swing_highs_idx) - 2):
            ls_idx  = swing_highs_idx[i]
            h_idx   = swing_highs_idx[i+1]
            rs_idx  = swing_highs_idx[i+2]

            # Check spacing
            if not (self.min_bars <= rs_idx - ls_idx <= self.max_bars):
                continue

            ls  = highs[ls_idx]
            h   = highs[h_idx]
            rs  = highs[rs_idx]

            # Head must be higher than both shoulders
            if not (h > ls and h > rs):
                continue

            # Shoulders should be roughly equal (within tolerance)
            shoulder_diff = abs(ls - rs) / ls
            if shoulder_diff > self.tol * 3:
                continue

            # Find neckline: lowest lows between ls→head and head→rs
            trough1 = min(lows[ls_idx:h_idx+1])
            trough2 = min(lows[h_idx:rs_idx+1])
            neckline = (trough1 + trough2) / 2

            # Has price broken the neckline?
            current_close = closes[-1]
            broke_neckline = current_close < neckline

            # Clarity: how well-formed is the pattern
            clarity = 1.0
            clarity -= shoulder_diff * 5           # penalize unequal shoulders
            clarity -= abs(trough1 - trough2) / h  # penalize uneven troughs
            clarity = max(0.1, min(1.0, clarity))

            stop = rs + (h - neckline) * 0.3   # just above right shoulder

            notes = "Neckline broken — waiting for retest" if broke_neckline else "Watching for neckline break"
            results.append(PatternResult(
                pattern_type='head_and_shoulders',
                direction='bearish',
                neckline=neckline,
                entry_zone_low=neckline,
                entry_zone_high=neckline * (1 + self.tol),
                stop_loss=stop,
                target_1=neckline - (h - neckline),       # measured move
                target_2=neckline - (h - neckline) * 1.5,
                clarity=clarity,
                notes=notes,
            ))

        # Bullish Inverted H&S (mirror logic)
        swing_lows_idx = self._find_swing_indices(lows, 'low')
        for i in range(len(swing_lows_idx) - 2):
            ls_idx = swing_lows_idx[i]
            h_idx  = swing_lows_idx[i+1]
            rs_idx = swing_lows_idx[i+2]

            if not (self.min_bars <= rs_idx - ls_idx <= self.max_bars):
                continue

            ls = lows[ls_idx]
            h  = lows[h_idx]
            rs = lows[rs_idx]

            if not (h < ls and h < rs):
                continue

            shoulder_diff = abs(ls - rs) / ls
            if shoulder_diff > self.tol * 3:
                continue

            peak1 = max(highs[ls_idx:h_idx+1])
            peak2 = max(highs[h_idx:rs_idx+1])
            neckline = (peak1 + peak2) / 2

            clarity = 1.0 - shoulder_diff * 5
            clarity = max(0.1, min(1.0, clarity))

            stop = rs - abs(neckline - h) * 0.3

            results.append(PatternResult(
                pattern_type='inverted_head_and_shoulders',
                direction='bullish',
                neckline=neckline,
                entry_zone_low=neckline * (1 - self.tol),
                entry_zone_high=neckline,
                stop_loss=stop,
                target_1=neckline + abs(neckline - h),
                target_2=neckline + abs(neckline - h) * 1.5,
                clarity=clarity,
                notes="Inverted H&S — bullish reversal setup",
            ))

        return results

    # ------------------------------------------------------------------ #
    # Double Top / Bottom
    # ------------------------------------------------------------------ #

    def _detect_double_top_bottom(self, df: pd.DataFrame) -> List[PatternResult]:
        results = []
        if len(df) < self.min_bars:
            return results

        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values

        # Double top
        swing_highs_idx = self._find_swing_indices(highs, 'high')
        for i in range(len(swing_highs_idx) - 1):
            t1_idx = swing_highs_idx[i]
            t2_idx = swing_highs_idx[i+1]

            if not (self.min_bars // 2 <= t2_idx - t1_idx <= self.max_bars):
                continue

            t1 = highs[t1_idx]
            t2 = highs[t2_idx]

            # Tops must be approximately equal
            if abs(t1 - t2) / t1 > self.tol:
                continue

            top_level = (t1 + t2) / 2
            valley = min(lows[t1_idx:t2_idx+1])

            clarity = 1.0 - abs(t1 - t2) / t1 * 10
            clarity = max(0.1, min(1.0, clarity))

            stop = top_level * (1 + self.tol * 2)
            measured_move = top_level - valley

            results.append(PatternResult(
                pattern_type='double_top',
                direction='bearish',
                neckline=valley,
                entry_zone_low=valley,
                entry_zone_high=valley * (1 + self.tol),
                stop_loss=stop,
                target_1=valley - measured_move,
                target_2=valley - measured_move * 1.5,
                clarity=clarity,
                notes=f"Double top at {top_level:.5f}",
            ))

        # Double bottom (mirror)
        swing_lows_idx = self._find_swing_indices(lows, 'low')
        for i in range(len(swing_lows_idx) - 1):
            b1_idx = swing_lows_idx[i]
            b2_idx = swing_lows_idx[i+1]

            if not (self.min_bars // 2 <= b2_idx - b1_idx <= self.max_bars):
                continue

            b1 = lows[b1_idx]
            b2 = lows[b2_idx]

            if abs(b1 - b2) / b1 > self.tol:
                continue

            bottom_level = (b1 + b2) / 2
            peak = max(highs[b1_idx:b2_idx+1])

            clarity = 1.0 - abs(b1 - b2) / b1 * 10
            clarity = max(0.1, min(1.0, clarity))

            stop = bottom_level * (1 - self.tol * 2)
            measured_move = peak - bottom_level

            results.append(PatternResult(
                pattern_type='double_bottom',
                direction='bullish',
                neckline=peak,
                entry_zone_low=peak * (1 - self.tol),
                entry_zone_high=peak,
                stop_loss=stop,
                target_1=peak + measured_move,
                target_2=peak + measured_move * 1.5,
                clarity=clarity,
                notes=f"Double bottom at {bottom_level:.5f}",
            ))

        return results

    # ------------------------------------------------------------------ #
    # Break & Retest
    # ------------------------------------------------------------------ #

    def _detect_break_retest(self, df: pd.DataFrame) -> List[PatternResult]:
        """
        The backbone of the strategy. Detects:
        - Consolidation zone
        - A break of that zone
        - Price returning to test the broken level (retest)
        """
        results = []
        if len(df) < 15:
            return results

        closes = df['close'].values
        highs  = df['high'].values
        lows   = df['low'].values
        n = len(df)

        # Look at last 40 candles for consolidation + break
        lookback = min(40, n - 5)
        consol = df.iloc[-(lookback+5):-5]
        recent = df.iloc[-5:]

        if len(consol) < 10:
            return results

        consol_high = consol['high'].max()
        consol_low  = consol['low'].min()
        consol_range = consol_high - consol_low

        current_close = closes[-1]

        # Bearish break: price broke below consolidation and is retesting from below
        if current_close < consol_low:
            retest_zone_low  = consol_low * (1 - self.tol * 2)
            retest_zone_high = consol_low * (1 + self.tol * 2)
            if retest_zone_low <= current_close <= retest_zone_high * 1.005:
                clarity = min(1.0, consol_range / current_close * 50)  # larger range = clearer
                results.append(PatternResult(
                    pattern_type='break_retest_bearish',
                    direction='bearish',
                    neckline=consol_low,
                    entry_zone_low=retest_zone_low,
                    entry_zone_high=retest_zone_high,
                    stop_loss=consol_low * (1 + self.tol * 4),
                    target_1=current_close - consol_range,
                    target_2=current_close - consol_range * 2,
                    clarity=clarity,
                    notes=f"Break of {consol_low:.5f} — price retesting from below",
                ))

        # Bullish break: price broke above consolidation and is retesting from above
        if current_close > consol_high:
            retest_zone_low  = consol_high * (1 - self.tol * 2)
            retest_zone_high = consol_high * (1 + self.tol * 2)
            if retest_zone_low * 0.995 <= current_close <= retest_zone_high:
                clarity = min(1.0, consol_range / current_close * 50)
                results.append(PatternResult(
                    pattern_type='break_retest_bullish',
                    direction='bullish',
                    neckline=consol_high,
                    entry_zone_low=retest_zone_low,
                    entry_zone_high=retest_zone_high,
                    stop_loss=consol_high * (1 - self.tol * 4),
                    target_1=current_close + consol_range,
                    target_2=current_close + consol_range * 2,
                    clarity=clarity,
                    notes=f"Break of {consol_high:.5f} — price retesting from above",
                ))

        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _find_swings(self, values: np.ndarray, kind: str) -> List[float]:
        n = self.swing_window
        swings = []
        for i in range(n, len(values) - n):
            window = values[i-n:i+n+1]
            if kind == 'high' and values[i] == max(window):
                swings.append(values[i])
            elif kind == 'low' and values[i] == min(window):
                swings.append(values[i])
        return swings

    def _find_swing_indices(self, values: np.ndarray, kind: str) -> List[int]:
        n = self.swing_window
        idxs = []
        for i in range(n, len(values) - n):
            window = values[i-n:i+n+1]
            if kind == 'high' and values[i] == max(window):
                idxs.append(i)
            elif kind == 'low' and values[i] == min(window):
                idxs.append(i)
        return idxs
