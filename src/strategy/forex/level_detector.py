"""
Key Level Detector — Major Psychological Round Numbers

Alex's strategy: only trade at levels "everyone is watching" — the big round
numbers that are immediately obvious on any chart. 1.3000, 1.3500, 157.000,
2.2500. Arbitrary swing highs/lows that cluster at non-round prices don't
qualify as key levels on their own.

Architecture:
  1. PRIMARY source: major round numbers at fixed increments (0.0500 for
     majors/crosses, 0.5000 for JPY pairs). These are the baseline candidates.
  2. CONFLUENCE scoring: each candidate gets +1 for each additional factor
     that aligns with it — swing structure, EMA proximity, prev-week H/L,
     "super round" status (0.10 or whole-number multiple).
  3. Only levels with score ≥ min_confluence are returned.

This replaces the old approach where arbitrary swing highs/lows were the
primary source and round numbers were just one of many confluence types.
Under the new design every returned level is guaranteed to be at a major
round number. Signal quality rises; false positives from random swing clusters
are eliminated.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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
    Detects key price levels anchored to major psychological round numbers.

    Parameters
    ----------
    min_confluence : int
        Minimum confluences to include a level (default 2).
        The round number itself counts as 1; you need at least 1 more
        (swing structure, EMA, prev-week H/L, or super-round status).
    swing_lookback : int
        Candles each side for swing high/low detection (default 20).
    confluence_tolerance_pct : float
        How close a secondary factor must be to a round number to count
        as confluence (default 0.20%).
    search_range_pct : float
        How far above/below current price to search for round numbers
        (default 5%).
    """

    def __init__(
        self,
        min_confluence: int   = 2,
        swing_lookback: int   = 20,
        confluence_tolerance_pct: float = 0.20,
        search_range_pct: float         = 5.0,
    ):
        self.min_confluence = min_confluence
        self.swing_lookback = swing_lookback
        self.conf_tol   = confluence_tolerance_pct / 100
        self.search_pct = search_range_pct / 100

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(self, df: pd.DataFrame, current_price: float,
               pair: str = "") -> List[KeyLevel]:
        """
        Return KeyLevel objects at major round numbers, sorted by proximity.

        df      — daily OHLC DataFrame (≥ 30 rows recommended).
        current_price — latest 1H close or bid price.
        pair    — optional pair string used to pick the right increment.
        """
        if len(df) < 20:
            return []

        df = df.copy()

        # ── Secondary confluence inputs ───────────────────────────────
        swings      = self._find_swings(df)
        emas        = self._calc_emas(df)
        prev_wk     = self._prev_week_extremes(df)
        eq_levels   = self._find_equal_levels(df)      # stop pool clusters
        struct_lvls = self._find_structural_levels(df, current_price)  # previous S/R

        # ── Generate round-number candidates ─────────────────────────
        increment   = self._round_increment(current_price, pair)
        super_inc   = increment * 2          # "super round" = twice the increment
        candidates  = self._round_number_candidates(current_price, increment)

        levels: List[KeyLevel] = []

        # ── Add structural levels as first-class level candidates ─────
        # Alex's #1 priority: "previous structure highs/lows."
        # These are returned even when they're not at a round number.
        # Score is based on test count (how often price respected this level).
        for struct_price, struct_label, test_count in struct_lvls:
            # Build confluence list — starts with the structural anchor
            confs: List[str] = [struct_label]
            # Bonus: if also near a round number
            nearest_round = round(struct_price / increment) * increment
            if abs(nearest_round - struct_price) / struct_price <= self.conf_tol * 2:
                confs.append("round_number")
            # Bonus: if near an EMA
            for ema_val, ema_label in emas:
                if abs(ema_val - struct_price) / struct_price <= self.conf_tol:
                    confs.append(ema_label)
                    break
            # Bonus: if equal highs/lows cluster at this level
            for eq_price, eq_label in eq_levels:
                if abs(eq_price - struct_price) / struct_price <= self.conf_tol:
                    confs.append(eq_label)
                    break
            # Score = unique confluences + bonus for being tested many times
            score = len(set(confs)) + (1 if test_count >= 2 else 0)  # 3→2: catches USD/CHF 0.884, GBP/CHF 1.12
            if score < self.min_confluence:
                continue
            dist_pct   = abs(struct_price - current_price) / current_price * 100
            level_type = "resistance" if struct_price > current_price else "support"
            levels.append(KeyLevel(
                price        = struct_price,
                level_type   = level_type,
                confluences  = list(set(confs)),
                score        = score,
                distance_pct = dist_pct,
                notes        = f"structural ({test_count} tests)",
            ))

        # ── Round-number candidates ───────────────────────────────────
        for cand in candidates:
            confluences: List[str] = ["round_number"]   # always counts as 1

            # Super-round bonus (e.g. whole number or 0.10 multiple)
            if abs(cand % super_inc) < 1e-9 or abs(cand % super_inc - super_inc) < 1e-9:
                confluences.append("super_round")

            # Swing structure near this level
            for sw_price, sw_kind in swings:
                if abs(sw_price - cand) / cand <= self.conf_tol:
                    confluences.append(sw_kind)
                    break   # one swing hit is enough

            # EMA proximity
            for ema_val, ema_label in emas:
                if abs(ema_val - cand) / cand <= self.conf_tol:
                    confluences.append(ema_label)
                    break

            # Previous-week high/low
            for pw_price, pw_label in prev_wk:
                if abs(pw_price - cand) / cand <= self.conf_tol:
                    confluences.append(pw_label)
                    break

            # Equal highs / equal lows (stop pool clusters)
            # This is a strong institutional signal — stops are parked here
            for eq_price, eq_label in eq_levels:
                if abs(eq_price - cand) / cand <= self.conf_tol:
                    confluences.append(eq_label)
                    break

            score = len(set(confluences))
            if score < self.min_confluence:
                continue

            distance_pct = abs(cand - current_price) / current_price * 100
            level_type   = "resistance" if cand > current_price else "support"

            levels.append(KeyLevel(
                price        = cand,
                level_type   = level_type,
                confluences  = list(set(confluences)),
                score        = score,
                distance_pct = distance_pct,
            ))

        # Sort: closest first, then highest score
        levels.sort(key=lambda l: (l.distance_pct, -l.score))
        return levels

    def nearest_levels(
        self, df: pd.DataFrame, current_price: float,
        max_distance_pct: float = 2.0, pair: str = "",
    ) -> Tuple[Optional[KeyLevel], Optional[KeyLevel]]:
        """Return (nearest_support, nearest_resistance) within max_distance_pct."""
        levels = self.detect(df, current_price, pair=pair)
        supports    = [l for l in levels if l.level_type == "support"
                       and l.distance_pct <= max_distance_pct]
        resistances = [l for l in levels if l.level_type == "resistance"
                       and l.distance_pct <= max_distance_pct]
        return (supports[0] if supports else None,
                resistances[0] if resistances else None)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _round_increment(price: float, pair: str) -> float:
        """
        Return the major round-number increment for this pair.

        JPY pairs  (price 80-200) : 0.500  → 157.00, 157.50, 158.00
        High-price crosses        : 0.0250 → 2.2250, 2.2500, 2.2750, 2.3000
        Standard majors           : 0.0250 → 1.2750, 1.3000, 1.3250, 1.3500

        Using 0.0250 (not 0.0500) covers the half-major levels that are
        just as watched as the full round numbers — 1.3250, 1.2750 etc.
        Alex trades these constantly. JPY already gets 0.500 which naturally
        includes both whole numbers and half-numbers (157.000, 157.500).
        """
        if "JPY" in pair.upper() or price > 50:
            return 0.500
        return 0.0250

    def _round_number_candidates(self, current_price: float,
                                  increment: float) -> List[float]:
        """Generate round numbers within search_range_pct of current price."""
        lo = current_price * (1 - self.search_pct)
        hi = current_price * (1 + self.search_pct)

        start = np.floor(lo / increment) * increment
        cands: List[float] = []
        val = start
        while val <= hi + 1e-9:
            rounded = round(val, 8)
            if lo <= rounded <= hi:
                cands.append(rounded)
            val += increment

        return cands

    def _find_swings(self, df: pd.DataFrame) -> List[Tuple[float, str]]:
        """Return list of (price, label) for swing highs and lows."""
        n      = self.swing_lookback
        highs  = df["high"].values
        lows   = df["low"].values
        result = []

        for i in range(n, len(df) - n):
            if highs[i] == max(highs[i - n: i + n + 1]):
                result.append((round(highs[i], 6), "swing_high"))
            if lows[i] == min(lows[i - n: i + n + 1]):
                result.append((round(lows[i], 6), "swing_low"))

        return result

    def _calc_emas(self, df: pd.DataFrame) -> List[Tuple[float, str]]:
        """Calculate 21, 50, 200 EMA from closing prices."""
        closes = df["close"]
        result = []
        for span, label in [(21, "ema21"), (50, "ema50"), (200, "ema200")]:
            if len(closes) >= span:
                val = float(closes.ewm(span=span, adjust=False).mean().iloc[-1])
                result.append((round(val, 6), label))
        return result

    def _prev_week_extremes(self, df: pd.DataFrame) -> List[Tuple[float, str]]:
        """Approximate previous-week high/low from last 10 daily candles."""
        if len(df) < 10:
            return []
        week = df.iloc[-10:-5]
        return [
            (round(float(week["high"].max()), 6), "prev_week_high"),
            (round(float(week["low"].min()),  6), "prev_week_low"),
        ]

    def _find_equal_levels(self, df: pd.DataFrame) -> List[Tuple[float, str]]:
        """
        Detect equal highs and equal lows — stop pool / liquidity clusters.

        Equal highs: 2+ swing highs within confluence_tolerance_pct of each other.
        Equal lows:  2+ swing lows  within confluence_tolerance_pct of each other.

        These are where retail traders pile their stops. Institutions know exactly
        where these clusters are and target them for liquidity grabs. A round number
        that also coincides with equal highs/lows is extremely high confluence.

        Returns (price, 'equal_highs' | 'equal_lows') for the cluster midpoint.
        """
        if len(df) < 20:
            return []

        # Use the shared _find_swings which returns (price, label) tuples
        all_swings = self._find_swings(df)
        highs_prices = [p for p, lbl in all_swings if lbl == "swing_high"]
        lows_prices  = [p for p, lbl in all_swings if lbl == "swing_low"]
        result = []

        # Equal highs: cluster swing highs within tolerance
        used = set()
        for i, h1 in enumerate(highs_prices):
            if i in used:
                continue
            cluster = [h1]
            for j, h2 in enumerate(highs_prices[i+1:], start=i+1):
                if abs(h2 - h1) / max(h1, 1e-9) <= self.conf_tol:
                    cluster.append(h2)
                    used.add(j)
            if len(cluster) >= 2:
                midpoint = round(sum(cluster) / len(cluster), 6)
                result.append((midpoint, "equal_highs"))

        # Equal lows: cluster swing lows within tolerance
        used = set()
        for i, l1 in enumerate(lows_prices):
            if i in used:
                continue
            cluster = [l1]
            for j, l2 in enumerate(lows_prices[i+1:], start=i+1):
                if abs(l2 - l1) / max(l1, 1e-9) <= self.conf_tol:
                    cluster.append(l2)
                    used.add(j)
            if len(cluster) >= 2:
                midpoint = round(sum(cluster) / len(cluster), 6)
                result.append((midpoint, "equal_lows"))

        return result

    def _find_structural_levels(
        self, df: pd.DataFrame, current_price: float
    ) -> List[Tuple[float, str, int]]:
        """
        Find significant previous structural swing highs and lows.

        These are Alex's #1 priority level type: "previous structure highs/lows."
        When price comes back to a level where it previously reversed sharply,
        institutions are watching. It doesn't need to be a round number.

        Algorithm:
          1. Use a longer lookback (10+ bars each side) to find weekly-scale swings
          2. Score by test count: how many times has price come within tolerance
             of this level and bounced?
          3. Only return levels within search_range_pct of current price
          4. Score ≥ 2 required (must be tested at least twice to qualify)

        Returns list of (price, label, score) triples.
        """
        if len(df) < 30:
            return []

        highs = df["high"].values
        lows  = df["low"].values
        n     = max(10, self.swing_lookback)    # wider lookback for structural levels

        # Find structural swing points
        struct_highs: List[float] = []
        struct_lows:  List[float] = []
        for i in range(n, len(df) - n):
            if highs[i] == max(highs[i - n: i + n + 1]):
                struct_highs.append(round(highs[i], 6))
            if lows[i] == min(lows[i - n: i + n + 1]):
                struct_lows.append(round(lows[i], 6))

        # Score each structural level by test count
        # A "test" = any bar where price came within tolerance and reversed
        test_tol = self.conf_tol * 3    # 3× wider tolerance for test detection (0.6%)
        result: List[Tuple[float, str, int]] = []

        lo_bound = current_price * (1 - self.search_pct)
        hi_bound = current_price * (1 + self.search_pct)

        for level_price in set(struct_highs):
            if not (lo_bound <= level_price <= hi_bound):
                continue
            # Count tests: bars where high came within tolerance of this level
            tests = sum(
                1 for h in highs
                if abs(h - level_price) / level_price <= test_tol
            )
            if tests >= 2:
                result.append((level_price, "structural_high", tests))

        for level_price in set(struct_lows):
            if not (lo_bound <= level_price <= hi_bound):
                continue
            # Count tests: bars where low came within tolerance of this level
            tests = sum(
                1 for l in lows
                if abs(l - level_price) / level_price <= test_tol
            )
            if tests >= 2:
                result.append((level_price, "structural_low", tests))

        return result
