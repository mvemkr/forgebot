"""
Key Level Detector

Identifies significant price levels through confluence stacking:
  1. Previous structure highs/lows (swing points)
  2. Round psychological numbers (XX.XX000, XX.X0000)
  3. EMA proximity (20, 50, 200 on daily/weekly)
  4. Previous day/week high & low

Levels score by confluence count. Minimum 2 confluences to flag.
This is not entry logic — it flags WHERE to watch. Entry comes from
the engulfing candle at the level.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KeyLevel:
    price: float
    level_type: str           # 'resistance' or 'support'
    confluences: List[str]    # what makes this level significant
    score: int                # number of confluences (higher = stronger)
    distance_pct: float       # how far current price is from this level (%)
    notes: str = ""

    def __repr__(self):
        return (f"KeyLevel({self.price:.5f} [{self.level_type}] "
                f"score={self.score} confluences={self.confluences} "
                f"dist={self.distance_pct:.2f}%)")


class LevelDetector:
    """
    Detects key price levels with confluence scoring.

    Parameters
    ----------
    swing_lookback : int
        Number of candles to look back for swing highs/lows (default 20)
    round_number_tolerance_pct : float
        How close to a round number to count as a confluence (default 0.05%)
    ema_tolerance_pct : float
        How close to EMA to count as a confluence (default 0.1%)
    min_confluence : int
        Minimum confluences to flag a level (default 2)
    cluster_tolerance_pct : float
        Merge levels within this % of each other (default 0.15%)
    """

    def __init__(
        self,
        swing_lookback: int = 20,
        round_number_tolerance_pct: float = 0.05,
        ema_tolerance_pct: float = 0.10,
        min_confluence: int = 2,
        cluster_tolerance_pct: float = 0.15,
    ):
        self.swing_lookback = swing_lookback
        self.round_pct = round_number_tolerance_pct / 100
        self.ema_pct = ema_tolerance_pct / 100
        self.min_confluence = min_confluence
        self.cluster_pct = cluster_tolerance_pct / 100

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(self, df: pd.DataFrame, current_price: float) -> List[KeyLevel]:
        """
        Main entry point. Returns list of KeyLevel objects sorted by distance.

        df must have columns: open, high, low, close (OHLC), indexed by datetime.
        Expects daily candles for best results (at least 100 candles recommended).
        """
        if len(df) < 30:
            return []

        # Calculate EMAs
        df = df.copy()
        df['ema20']  = df['close'].ewm(span=20,  adjust=False).mean()
        df['ema50']  = df['close'].ewm(span=50,  adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

        raw_levels = {}  # price → list of confluence strings

        # 1. Swing highs and lows
        self._add_swing_levels(df, raw_levels)

        # 2. Round psychological numbers
        self._add_round_numbers(df, raw_levels, current_price)

        # 3. EMA levels
        self._add_ema_levels(df, raw_levels)

        # 4. Previous week high/low
        self._add_prev_week_levels(df, raw_levels)

        # 5. Cluster nearby levels
        clustered = self._cluster_levels(raw_levels)

        # 6. Build KeyLevel objects
        levels = []
        for price, confluences in clustered.items():
            if len(confluences) < self.min_confluence:
                continue
            distance_pct = abs(price - current_price) / current_price * 100
            level_type = 'resistance' if price > current_price else 'support'
            levels.append(KeyLevel(
                price=price,
                level_type=level_type,
                confluences=list(set(confluences)),
                score=len(set(confluences)),
                distance_pct=distance_pct,
            ))

        # Sort by distance (closest first), then by score descending
        levels.sort(key=lambda l: (l.distance_pct, -l.score))
        return levels

    def nearest_levels(
        self, df: pd.DataFrame, current_price: float, max_distance_pct: float = 2.0
    ) -> tuple[Optional[KeyLevel], Optional[KeyLevel]]:
        """
        Returns (nearest_support, nearest_resistance) within max_distance_pct.
        """
        levels = self.detect(df, current_price)
        supports    = [l for l in levels if l.level_type == 'support'    and l.distance_pct <= max_distance_pct]
        resistances = [l for l in levels if l.level_type == 'resistance' and l.distance_pct <= max_distance_pct]
        return (
            supports[0]    if supports    else None,
            resistances[0] if resistances else None,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _add_swing_levels(self, df: pd.DataFrame, levels: dict):
        """Detect swing highs and lows using local maxima/minima."""
        n = self.swing_lookback
        highs = df['high'].values
        lows  = df['low'].values

        for i in range(n, len(df) - n):
            # Swing high: highest point in the window
            if highs[i] == max(highs[i-n:i+n+1]):
                price = round(highs[i], 5)
                levels.setdefault(price, []).append('swing_high')

            # Swing low: lowest point in the window
            if lows[i] == min(lows[i-n:i+n+1]):
                price = round(lows[i], 5)
                levels.setdefault(price, []).append('swing_low')

    def _add_round_numbers(self, df: pd.DataFrame, levels: dict, current_price: float):
        """
        Add round psychological number levels near current price.
        For most forex pairs: every 0.00500, 0.01000 increment.
        """
        # Determine granularity based on price magnitude
        if current_price > 100:    # JPY pairs (e.g. 155.000)
            increments = [1.0, 0.5]
        elif current_price > 10:   # e.g. GBP/JPY at 190
            increments = [1.0, 0.5]
        else:                       # majors (1.3000, etc.)
            increments = [0.01000, 0.00500]

        search_range = current_price * 0.05  # Search ±5% from current price

        for inc in increments:
            # Generate round numbers in range
            start = round((current_price - search_range) / inc) * inc
            end   = current_price + search_range
            candidate = start
            while candidate <= end:
                candidate = round(candidate, 5)
                distance = abs(candidate - current_price) / current_price
                if distance <= 0.05:
                    levels.setdefault(candidate, []).append(f'round_number_{inc}')
                candidate += inc

    def _add_ema_levels(self, df: pd.DataFrame, levels: dict):
        """Add EMA values as dynamic levels."""
        for col, label in [('ema20', 'ema20'), ('ema50', 'ema50'), ('ema200', 'ema200')]:
            if col in df.columns and not df[col].empty:
                price = round(df[col].iloc[-1], 5)
                levels.setdefault(price, []).append(label)

    def _add_prev_week_levels(self, df: pd.DataFrame, levels: dict):
        """Add previous week's high and low."""
        if len(df) < 10:
            return
        # Approximate: last 5 daily candles = previous week
        week_slice = df.iloc[-10:-5]
        if len(week_slice) == 0:
            return
        prev_high = round(week_slice['high'].max(), 5)
        prev_low  = round(week_slice['low'].min(),  5)
        levels.setdefault(prev_high, []).append('prev_week_high')
        levels.setdefault(prev_low,  []).append('prev_week_low')

    def _cluster_levels(self, levels: dict) -> dict:
        """Merge levels that are within cluster_tolerance_pct of each other."""
        if not levels:
            return {}

        sorted_prices = sorted(levels.keys())
        clustered = {}
        used = set()

        for i, price in enumerate(sorted_prices):
            if price in used:
                continue
            cluster_prices = [price]
            cluster_confluences = list(levels[price])
            used.add(price)

            for j in range(i + 1, len(sorted_prices)):
                other = sorted_prices[j]
                if other in used:
                    continue
                if abs(other - price) / price <= self.cluster_pct:
                    cluster_prices.append(other)
                    cluster_confluences.extend(levels[other])
                    used.add(other)
                else:
                    break  # sorted, so no more close ones

            # Use median price of cluster
            center = round(float(np.median(cluster_prices)), 5)
            clustered[center] = cluster_confluences

        return clustered


if __name__ == "__main__":
    # Quick test with fake data
    import pandas as pd
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    close = 1.3000 + np.cumsum(np.random.randn(200) * 0.003)
    df = pd.DataFrame({
        'open':  close - 0.001,
        'high':  close + 0.003,
        'low':   close - 0.003,
        'close': close,
    }, index=dates)

    detector = LevelDetector()
    current = close[-1]
    levels = detector.detect(df, current)
    print(f"Current price: {current:.5f}")
    print(f"Found {len(levels)} key levels:")
    for l in levels[:10]:
        print(f"  {l}")
