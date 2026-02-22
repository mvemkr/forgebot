"""
Pattern Detector

Detects institutional price action patterns used by smart money:

  1. Head & Shoulders / Inverted H&S
  2. Double Top / Double Bottom
  3. Break & Retest / SR Flip
  4. Liquidity Sweep (QM / Stop Hunt / Fakeout)
     — The core institutional play: price wicks above a swing high
       (or below a swing low) to trigger retail stops, then reverses.
       This is the QM pattern, Fakeout V1/V2, and stop hunt from the
       CMS institutional price action framework.
  5. Equal Highs / Equal Lows detection (stop pool identification)

The institutional insight: round numbers attract retail stop orders.
Institutions know this. They sweep those levels (trigger the stops,
fill their own orders in the opposite direction), then move away.
The liquidity sweep IS the entry signal — not a candle pattern to avoid.

All patterns output a pattern_level: the structural price to validate
against major round numbers. Patterns at round numbers = institutional.
Patterns at random swing levels = noise.

A pattern by itself is NOT an entry signal — it identifies the setup zone.
The engulfing candle (entry_signal.py) is required to confirm entry.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

try:
    from scipy.signal import find_peaks as _scipy_find_peaks
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


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
    pattern_level: float = 0.0  # the structural price to validate against round numbers:
                                 #   double_bottom → average of the two lows (the support floor)
                                 #   double_top    → average of the two highs (the resistance ceiling)
                                 #   H&S / IH&S   → neckline (the level everyone is watching)
                                 #   break_retest  → the broken level (same as neckline)
                                 # Round-number check fires on THIS price, not on current price.

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
        results.extend(self._detect_liquidity_sweep(df))      # institutional sweeps
        results.extend(self._detect_head_and_shoulders(df))
        results.extend(self._detect_double_top_bottom(df))
        results.extend(self._detect_break_retest(df))
        results.extend(self._detect_engulfing_rejection(df))  # daily/4H candle rejection
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
                pattern_level=neckline,   # neckline is THE level everyone watches on H&S
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
                pattern_level=neckline,   # neckline is THE level everyone watches on IH&S
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
                pattern_level=top_level,  # the tops are what formed at the round number
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
                pattern_level=bottom_level,  # the bottoms are what formed at the round number
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
                    pattern_level=consol_low,  # the broken level is the round number
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
                    pattern_level=consol_high,  # the broken level is the round number
                ))

        return results

    # ------------------------------------------------------------------ #
    # Liquidity Sweep / QM / Stop Hunt
    # ------------------------------------------------------------------ #

    def _detect_liquidity_sweep(self, df: pd.DataFrame) -> List[PatternResult]:
        """
        Detect liquidity sweeps — the core institutional reversal pattern.

        What this detects (from the CMS cheat sheet):
          • QM Quick Retest / QM Late Retest
          • Fakeout V1 (Default) / Fakeout V2 (SR Flip)
          • Any candle that wicks beyond a swing high/low but closes back inside

        The institutional mechanism:
          1. Retail traders place stops just beyond swing highs/lows
          2. Institutions push price past that level (triggers stops → fills their orders)
          3. Price snaps back — the stop hunt candle CLOSES back inside
          4. If that swept level is at a round number → institutions were defending it

        Entry logic:
          Bearish: wick above swing high (close back below) → sell the reversal
          Bullish: wick below swing low (close back above) → buy the reversal

        pattern_level = the swept swing high/low
          → validated against round numbers in set_and_forget.py
        """
        results = []
        if len(df) < self.min_bars:
            return results

        highs  = df['high'].values
        lows   = df['low'].values
        opens  = df['open'].values
        closes = df['close'].values
        n = len(df)

        # Find swing points in the historical data (not including the sweep candle itself)
        swing_h_idx = self._find_swing_indices(highs, 'high')
        swing_l_idx = self._find_swing_indices(lows,  'low')

        # Scan the last 20 candles for sweeps
        scan_start = max(self.swing_window + 1, n - 20)

        for i in range(scan_start, n):
            bar_high  = highs[i]
            bar_low   = lows[i]
            bar_open  = opens[i]
            bar_close = closes[i]
            bar_range = bar_high - bar_low
            if bar_range == 0:
                continue

            # ── Bearish sweep: wick above swing high, close back below ──────
            for sh_idx in reversed(swing_h_idx):
                if sh_idx >= i:
                    continue
                if i - sh_idx > 35:
                    break   # swing too old

                swing_high = highs[sh_idx]

                # Must wick above and CLOSE back below
                if not (bar_high > swing_high and bar_close < swing_high):
                    continue

                wick_above = bar_high - swing_high
                sweep_ratio = wick_above / bar_range

                # High-quality sweep: wick must be ≥50% of candle range.
                # This filters out noise — we want a DECISIVE rejection candle,
                # not a small doji that happened to poke above a level.
                # Alex is watching for the big stop hunt candle, not micro wicks.
                if sweep_ratio < 0.50:
                    continue

                # Bearish close is ideal (candle turned back down)
                bearish_body = bar_close < bar_open

                # Clarity: how decisive the rejection was
                body_size = abs(bar_close - bar_open)
                body_ratio = body_size / bar_range
                clarity = min(1.0,
                    0.40 * sweep_ratio           # how far above it swept
                  + 0.40 * body_ratio            # how strong the close-back was
                  + (0.20 if bearish_body else 0)
                )
                if clarity < self.min_bars / 100:
                    clarity = max(0.30, clarity)

                # Stop: above the sweep wick
                stop = bar_high * (1 + self.tol * 2)

                # Target: measured from sweep high, minimum 3R
                sweep_height = bar_high - swing_high
                target_dist  = max(sweep_height * 3, swing_high - bar_close)

                results.append(PatternResult(
                    pattern_type  = 'liquidity_sweep_bearish',
                    direction     = 'bearish',
                    neckline      = swing_high,
                    entry_zone_low  = bar_close * (1 - self.tol),
                    entry_zone_high = swing_high,
                    stop_loss     = stop,
                    target_1      = swing_high - target_dist,
                    target_2      = swing_high - target_dist * 1.5,
                    clarity       = clarity,
                    notes         = (
                        f"Bearish sweep above {swing_high:.5f} — "
                        f"wick {wick_above / 0.0001:.0f}p, closed back at {bar_close:.5f}. "
                        f"Retail stops triggered. Institutional supply."
                    ),
                    pattern_level = swing_high,   # this is the round number to validate
                ))
                break   # one sweep per scan candle is enough

            # ── Bullish sweep: wick below swing low, close back above ────────
            for sl_idx in reversed(swing_l_idx):
                if sl_idx >= i:
                    continue
                if i - sl_idx > 35:
                    break

                swing_low = lows[sl_idx]

                if not (bar_low < swing_low and bar_close > swing_low):
                    continue

                wick_below = swing_low - bar_low
                sweep_ratio = wick_below / bar_range

                # Same quality filter — wick must be dominant (≥50% of bar range)
                if sweep_ratio < 0.50:
                    continue

                bullish_body = bar_close > bar_open

                body_size  = abs(bar_close - bar_open)
                body_ratio = body_size / bar_range
                clarity = min(1.0,
                    0.40 * sweep_ratio
                  + 0.40 * body_ratio
                  + (0.20 if bullish_body else 0)
                )
                clarity = max(0.30, clarity)

                stop = bar_low * (1 - self.tol * 2)
                sweep_height = swing_low - bar_low
                target_dist  = max(sweep_height * 3, bar_close - swing_low)

                results.append(PatternResult(
                    pattern_type  = 'liquidity_sweep_bullish',
                    direction     = 'bullish',
                    neckline      = swing_low,
                    entry_zone_low  = swing_low,
                    entry_zone_high = bar_close * (1 + self.tol),
                    stop_loss     = stop,
                    target_1      = swing_low + target_dist,
                    target_2      = swing_low + target_dist * 1.5,
                    clarity       = clarity,
                    notes         = (
                        f"Bullish sweep below {swing_low:.5f} — "
                        f"wick {wick_below / 0.0001:.0f}p, closed back at {bar_close:.5f}. "
                        f"Retail stops triggered. Institutional demand."
                    ),
                    pattern_level = swing_low,
                ))
                break

        return results

    def _detect_engulfing_rejection(self, df: pd.DataFrame) -> List[PatternResult]:
        """
        Detect large engulfing / rejection candles at key levels.

        This is Alex's most-used and most-profitable pattern type. It appears
        in two forms:

        1. CLASSIC ENGULFING: current candle body completely contains the
           previous candle's body AND closes in opposite direction.
           Most reliable form. Alex says "no engulfing, no trade" about this.

        2. STRONG REJECTION: current candle body is ≥ 2× average body size
           and body/range ratio ≥ 0.55 — even without strict engulfing of
           previous candle. Used for: USD/CAD Week 10 daily bullish engulfing
           at support that launched him from $44K → $107K.

        Key properties:
          - Only looks at the 3 most recent closed candles (immediate signal)
          - Stop: behind the candle's structural extreme (lowest low / highest
            high of the past 5 bars) rather than just the wick — gives the
            trade room to breathe like Alex's set-and-forget approach
          - pattern_level: the candle's open price (the level being tested)
          - Clarity: combination of body ratio, engulf completeness, size vs ATR

        NOT the same as entry_signal.py which handles 1H confirmation.
        This is the DAILY or 4H pattern that IS the setup itself.
        """
        if len(df) < 10:
            return []

        opens  = df["open"].values
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        n      = len(df)

        # Average body size over last 20 candles (baseline for significance check)
        avg_body = float(np.mean(np.abs(closes[-20:] - opens[-20:])))
        if avg_body <= 0:
            return []

        results = []

        # Check only the last 3 completed candles — this is an IMMEDIATE signal
        for i in range(max(1, n - 3), n):
            curr_open  = opens[i]
            curr_close = closes[i]
            curr_high  = highs[i]
            curr_low   = lows[i]
            prev_open  = opens[i - 1]
            prev_close = closes[i - 1]

            curr_body  = abs(curr_close - curr_open)
            prev_body  = abs(prev_close - prev_open)
            curr_range = curr_high - curr_low
            if curr_range <= 0:
                continue

            body_ratio = curr_body / curr_range          # how much of range is body
            size_ratio = curr_body / avg_body if avg_body > 0 else 1.0

            # ── Bullish engulfing ─────────────────────────────────────────
            is_bullish = curr_close > curr_open          # green candle
            if is_bullish:
                # Classic: current body engulfs previous body (higher close, lower open)
                classic_engulf = (
                    curr_open  <= min(prev_open, prev_close) and
                    curr_close >= max(prev_open, prev_close)
                )
                # Strong rejection: body large enough even without full engulf
                strong_rejection = (
                    size_ratio >= 2.0 and
                    body_ratio >= 0.55
                )
                if not (classic_engulf or strong_rejection):
                    continue
                if body_ratio < 0.40:
                    continue   # too wick-dominated to be reliable

                # Structural stop: below the lowest low of past 5 bars
                struct_low   = float(np.min(lows[max(0, i-4):i+1]))
                stop_loss    = struct_low * (1 - self.tol)
                target_range = curr_close - struct_low

                # Clarity: weighted combination of quality factors
                engulf_score  = 1.0 if classic_engulf else 0.6
                clarity       = min(1.0, (
                    0.4 * body_ratio
                  + 0.3 * min(size_ratio / 3.0, 1.0)
                  + 0.3 * engulf_score
                ))

                notes = ("Classic bullish engulfing" if classic_engulf
                         else "Strong bullish rejection candle")
                results.append(PatternResult(
                    pattern_type     = 'engulfing_bullish',
                    direction        = 'bullish',
                    neckline         = curr_close,          # most recent price
                    entry_zone_low   = curr_close * (1 - self.tol),
                    entry_zone_high  = curr_close * (1 + self.tol),
                    stop_loss        = stop_loss,
                    target_1         = curr_close + target_range,
                    target_2         = curr_close + target_range * 1.5,
                    clarity          = clarity,
                    pattern_level    = curr_open,           # the level being tested/broken
                    notes            = f"{notes} (body {body_ratio:.0%}, {size_ratio:.1f}× avg)",
                ))

            # ── Bearish engulfing ─────────────────────────────────────────
            is_bearish = curr_close < curr_open           # red candle
            if is_bearish:
                classic_engulf = (
                    curr_open  >= max(prev_open, prev_close) and
                    curr_close <= min(prev_open, prev_close)
                )
                strong_rejection = (
                    size_ratio >= 2.0 and
                    body_ratio >= 0.55
                )
                if not (classic_engulf or strong_rejection):
                    continue
                if body_ratio < 0.40:
                    continue

                struct_high  = float(np.max(highs[max(0, i-4):i+1]))
                stop_loss    = struct_high * (1 + self.tol)
                target_range = struct_high - curr_close

                engulf_score = 1.0 if classic_engulf else 0.6
                clarity      = min(1.0, (
                    0.4 * body_ratio
                  + 0.3 * min(size_ratio / 3.0, 1.0)
                  + 0.3 * engulf_score
                ))

                notes = ("Classic bearish engulfing" if classic_engulf
                         else "Strong bearish rejection candle")
                results.append(PatternResult(
                    pattern_type     = 'engulfing_bearish',
                    direction        = 'bearish',
                    neckline         = curr_close,
                    entry_zone_low   = curr_close * (1 - self.tol),
                    entry_zone_high  = curr_close * (1 + self.tol),
                    stop_loss        = stop_loss,
                    target_1         = curr_close - target_range,
                    target_2         = curr_close - target_range * 1.5,
                    clarity          = clarity,
                    pattern_level    = curr_open,
                    notes            = f"{notes} (body {body_ratio:.0%}, {size_ratio:.1f}× avg)",
                ))

        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _find_swings(self, values: np.ndarray, kind: str) -> List[float]:
        return [values[i] for i in self._find_swing_indices(values, kind)]

    def _find_swing_indices(self, values: np.ndarray, kind: str) -> List[int]:
        """
        Find swing high/low indices using fixed sliding window.

        Simple and reliable: a swing high is the highest value in a ±swing_window
        bar window. Produces clean, predictable pattern detection without the
        noise amplification that scipy prominence can introduce during volatile
        market periods (elections, major news events).

        scipy prominence is architecturally superior but requires volatility
        regime detection to avoid finding patterns in every choppy period.
        Keeping the controlled version until we add volatility regime filtering.
        """
        n = self.swing_window
        idxs = []
        for i in range(n, len(values) - n):
            window = values[i-n:i+n+1]
            if kind == 'high' and values[i] == max(window):
                idxs.append(i)
            elif kind == 'low' and values[i] == min(window):
                idxs.append(i)
        return idxs
