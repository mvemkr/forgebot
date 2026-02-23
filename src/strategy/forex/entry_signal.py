"""
Entry Signal Detector — The Engulfing Candle

THE most important module in the system. This is the patience filter.

From the strategy research:
  "Price reaching a key level is NOT an entry.
   The entry is: price at key level + bearish/bullish engulfing candle."

Rules:
  - Bearish engulfing: current body CLOSES below previous candle's full body
  - Bullish engulfing: current body CLOSES above previous candle's full body
  - Body must be meaningful (not a doji — body > wick ratio threshold)
  - No entry without the candle. No exceptions. No FOMO.

Additional confirmation:
  - Marubozu (large body, small wicks) = stronger signal
  - Tweezer top/bottom (two candles with near-equal highs/lows) = valid
  - Pin bar with strong body in direction = valid
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .strategy_config import ENGULFING_ONLY


@dataclass
class EntrySignal:
    signal_type: str       # 'bearish_engulfing', 'bullish_engulfing', 'pin_bar_bear', etc.
    direction: str         # 'long' or 'short'
    strength: float        # 0.0–1.0 (1.0 = perfect textbook candle)
    candle_index: int      # which candle fired (usually -1 = latest closed)
    close: float           # close price of signal candle
    body_size: float       # absolute body size
    notes: str = ""

    @property
    def is_strong(self) -> bool:
        return self.strength >= 0.65

    @property
    def is_valid(self) -> bool:
        return self.strength >= 0.40


class EntrySignalDetector:
    """
    Detects engulfing candles and other entry confirmation signals.

    Parameters
    ----------
    min_body_ratio : float
        Minimum ratio of body to total candle range (filter dojis).
        Default 0.5 = body must be ≥50% of high–low range.
    lookback_candles : int
        How many recent candles to scan for signals (default 2 = last 2 closed candles).
    """

    def __init__(
        self,
        min_body_ratio: float = 0.50,
        lookback_candles: int = 2,
    ):
        self.min_body_ratio = min_body_ratio
        self.lookback = lookback_candles

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(self, df: pd.DataFrame) -> Optional[EntrySignal]:
        """
        Scan recent candles for entry signals.
        Returns the strongest signal found, or None.

        df: OHLC dataframe, latest candle last.
        NOTE: The latest candle (index -1) should be CLOSED, not currently forming.
        """
        if len(df) < 3:
            return None

        signals = []

        for i in range(-self.lookback, 0):
            try:
                current = df.iloc[i]
                prev    = df.iloc[i - 1]
            except IndexError:
                continue

            # Bearish engulfing
            be = self._bearish_engulfing(current, prev)
            if be:
                signals.append(EntrySignal(
                    signal_type='bearish_engulfing',
                    direction='short',
                    strength=be,
                    candle_index=i,
                    close=current['close'],
                    body_size=abs(current['open'] - current['close']),
                    notes=f"Bearish engulfing at {current['close']:.5f}",
                ))

            # Bullish engulfing
            bue = self._bullish_engulfing(current, prev)
            if bue:
                signals.append(EntrySignal(
                    signal_type='bullish_engulfing',
                    direction='long',
                    strength=bue,
                    candle_index=i,
                    close=current['close'],
                    body_size=abs(current['open'] - current['close']),
                    notes=f"Bullish engulfing at {current['close']:.5f}",
                ))

            # Pin bar — controlled by ENGULFING_ONLY config flag.
            # Alex's rule: "No engulfing candle = no trade." Every video.
            # Default: ENGULFING_ONLY=True → pin bars are NOT valid entry signals.
            if not ENGULFING_ONLY:
                pb = self._pin_bar(current)
                if pb:
                    signals.append(pb)

        if not signals:
            return None

        # Return strongest signal
        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals[0]

    def has_signal(self, df: pd.DataFrame, direction: str) -> tuple[bool, Optional[EntrySignal]]:
        """
        Check if there's a valid entry signal in the given direction.
        Returns (True/False, signal_or_None).
        direction: 'long' or 'short'
        """
        signal = self.detect(df)
        if signal and signal.direction == direction and signal.is_valid:
            return True, signal
        return False, None

    # ------------------------------------------------------------------ #
    # Individual signal detectors
    # ------------------------------------------------------------------ #

    def _bearish_engulfing(self, current: pd.Series, prev: pd.Series) -> float:
        """
        Returns strength (0.0 = not engulfing, >0 = engulfing).

        Classic bearish engulfing:
        - Current candle is bearish (close < open)
        - Current body fully engulfs previous body
        - Previous candle was bullish (close > open) — ideal
        """
        curr_open  = current['open']
        curr_close = current['close']
        curr_high  = current['high']
        curr_low   = current['low']

        prev_open  = prev['open']
        prev_close = prev['close']

        # Must be bearish
        if curr_close >= curr_open:
            return 0.0

        curr_body = curr_open - curr_close
        curr_range = curr_high - curr_low

        # Filter doji
        if curr_range > 0 and curr_body / curr_range < self.min_body_ratio:
            return 0.0

        # Body must engulf previous candle's body
        prev_body_high = max(prev_open, prev_close)
        prev_body_low  = min(prev_open, prev_close)

        if curr_open < prev_body_high or curr_close > prev_body_low:
            return 0.0

        # Strength: how much does it over-engulf?
        engulf_ratio = curr_body / max(abs(prev_open - prev_close), 0.00001)
        strength = min(1.0, engulf_ratio / 2)

        # Bonus: previous candle was bullish (cleaner reversal signal)
        if prev_close > prev_open:
            strength = min(1.0, strength + 0.15)

        # Bonus: small wicks (marubozu-like)
        upper_wick = curr_high - curr_open
        wick_ratio = upper_wick / curr_range if curr_range > 0 else 1
        if wick_ratio < 0.15:
            strength = min(1.0, strength + 0.10)

        return strength

    def _bullish_engulfing(self, current: pd.Series, prev: pd.Series) -> float:
        """Bullish mirror of _bearish_engulfing."""
        curr_open  = current['open']
        curr_close = current['close']
        curr_high  = current['high']
        curr_low   = current['low']

        prev_open  = prev['open']
        prev_close = prev['close']

        # Must be bullish
        if curr_close <= curr_open:
            return 0.0

        curr_body = curr_close - curr_open
        curr_range = curr_high - curr_low

        if curr_range > 0 and curr_body / curr_range < self.min_body_ratio:
            return 0.0

        prev_body_high = max(prev_open, prev_close)
        prev_body_low  = min(prev_open, prev_close)

        if curr_close < prev_body_high or curr_open > prev_body_low:
            return 0.0

        engulf_ratio = curr_body / max(abs(prev_open - prev_close), 0.00001)
        strength = min(1.0, engulf_ratio / 2)

        if prev_close < prev_open:
            strength = min(1.0, strength + 0.15)

        lower_wick = curr_open - curr_low
        wick_ratio = lower_wick / curr_range if curr_range > 0 else 1
        if wick_ratio < 0.15:
            strength = min(1.0, strength + 0.10)

        return strength

    def _pin_bar(self, current: pd.Series) -> Optional[EntrySignal]:
        """
        Pin bar / rejection wick.
        Long wick in one direction with small body at the other end.
        """
        o = current['open']
        h = current['high']
        l = current['low']
        c = current['close']

        candle_range = h - l
        if candle_range == 0:
            return None

        body = abs(c - o)
        body_ratio = body / candle_range

        # Body must be small (pin bar = mostly wick)
        if body_ratio > 0.35:
            return None

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # Bearish pin bar: large upper wick, body at bottom
        if upper_wick > candle_range * 0.6 and upper_wick > lower_wick * 2:
            strength = min(1.0, upper_wick / candle_range - 0.1)
            return EntrySignal(
                signal_type='pin_bar_bearish',
                direction='short',
                strength=strength * 0.8,  # pin bars slightly weaker than engulfing
                candle_index=-1,
                close=c,
                body_size=body,
                notes=f"Bearish pin bar — rejection wick at {h:.5f}",
            )

        # Bullish pin bar: large lower wick, body at top
        if lower_wick > candle_range * 0.6 and lower_wick > upper_wick * 2:
            strength = min(1.0, lower_wick / candle_range - 0.1)
            return EntrySignal(
                signal_type='pin_bar_bullish',
                direction='long',
                strength=strength * 0.8,
                candle_index=-1,
                close=c,
                body_size=body,
                notes=f"Bullish pin bar — rejection wick at {l:.5f}",
            )

        return None


if __name__ == "__main__":
    # Quick test
    import pandas as pd

    # Bearish engulfing example
    data = {
        'open':  [1.3000, 1.3010, 1.3020, 1.3040, 1.3060],
        'high':  [1.3020, 1.3030, 1.3050, 1.3070, 1.3070],
        'low':   [1.2990, 1.3000, 1.3010, 1.3030, 1.2990],
        'close': [1.3010, 1.3020, 1.3040, 1.3060, 1.3000],  # last candle: big down candle
    }
    df = pd.DataFrame(data)
    detector = EntrySignalDetector()
    signal = detector.detect(df)
    print(f"Signal: {signal}")
