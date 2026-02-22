"""
Currency Strength / Macro Theme Detector

Detects when a single currency is dominant (strongly weak or strong) across
ALL of its pairs — the condition that lets Alex stack 4 correlated positions
on one macro theme for massive gains.

Alex's Week 7-8 (Sep 2024): JPY was weakening across every pair.
  - USD/JPY SHORT ✅
  - GBP/JPY SHORT ✅
  - NZD/JPY SHORT ✅
  - AUD/JPY SHORT ✅
  → $20K → $90K in one week ($70K profit)

He didn't take 4 separate trades. He recognised ONE macro theme (JPY weakness)
and expressed it across 4 vehicles simultaneously.

Algorithm
─────────
1. For each pair in the watchlist, compute:
   - 5-day momentum  (short-term: is the move happening now?)
   - 20-day momentum (medium-term: is this a sustained theme?)
   - Trend direction from detect_trend()

2. For each of the 8 currencies, aggregate performance:
   - As BASE currency: count bullish/bearish momentum
   - As QUOTE currency: count inverse
   - Result: strength_score in [-8, +8] (one point per pair)

3. Dominant theme threshold: abs(score) ≥ THEME_THRESHOLD (default 3)
   Meaning: a currency is outperforming/underperforming on 3+ of its pairs

4. When a macro theme is detected:
   - Return the dominant currency, direction (strong/weak), and score
   - Orchestrator uses this to allow correlated position stacking
   - Position size per stacked trade = normal_size / stack_count
     (same total exposure, just spread across correlated vehicles)

Currency coverage (from 18-pair watchlist):
  GBP: GBP/USD, GBP/JPY, GBP/CHF, GBP/NZD, EUR/GBP (as quote)
  EUR: EUR/USD, EUR/GBP, EUR/JPY (via crosses), EUR/NZD, EUR/CAD, EUR/AUD
  USD: GBP/USD (q), EUR/USD (q), USD/JPY, USD/CHF, USD/CAD
  AUD: AUD/USD, AUD/CAD, AUD/NZD
  NZD: NZD/USD, EUR/NZD (q), GBP/NZD (q), NZD/CAD, AUD/NZD (q)
  CAD: USD/CAD (q), EUR/CAD (q), AUD/CAD (q), NZD/CAD (q)
  CHF: USD/CHF (q), GBP/CHF (q), EUR/CHF not in list
  JPY: USD/JPY (q), GBP/JPY (q), AUD/JPY (q, if added)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ── Configuration ───────────────────────────────────────────────────────── #

THEME_THRESHOLD  = 3      # currency must dominate ≥ 3 pairs for theme to fire
STACK_MAX        = 4      # max concurrent positions in a macro theme
MOMENTUM_DAYS_S  = 5      # short-term momentum window
MOMENTUM_DAYS_L  = 20     # long-term momentum window

# Which currency is BASE (index 0) and QUOTE (index 1) in each pair
# +1 = pair direction measures BASE currency strength
# -1 = pair direction measures QUOTE currency strength (inverted)
PAIR_CURRENCY_MAP: Dict[str, Tuple[str, str]] = {
    "GBP/USD": ("GBP", "USD"),
    "EUR/USD": ("EUR", "USD"),
    "USD/JPY": ("USD", "JPY"),
    "USD/CHF": ("USD", "CHF"),
    "USD/CAD": ("USD", "CAD"),
    "AUD/USD": ("AUD", "USD"),
    "NZD/USD": ("NZD", "USD"),
    "GBP/JPY": ("GBP", "JPY"),
    "EUR/JPY": ("EUR", "JPY"),
    "GBP/CHF": ("GBP", "CHF"),
    "EUR/GBP": ("EUR", "GBP"),
    "AUD/JPY": ("AUD", "JPY"),
    "NZD/JPY": ("NZD", "JPY"),
    "EUR/NZD": ("EUR", "NZD"),
    "EUR/CAD": ("EUR", "CAD"),
    "EUR/AUD": ("EUR", "AUD"),
    "AUD/CAD": ("AUD", "CAD"),
    "AUD/NZD": ("AUD", "NZD"),
    "NZD/CAD": ("NZD", "CAD"),
    "GBP/NZD": ("GBP", "NZD"),
    "GBP/CAD": ("GBP", "CAD"),
}


# ── Data classes ─────────────────────────────────────────────────────────── #

@dataclass
class CurrencyTheme:
    """A detected macro currency theme."""
    currency:      str           # e.g. "JPY"
    direction:     str           # "weak" (sell everything vs this) or "strong" (buy)
    score:         float         # how many pairs confirm (-8 to +8)
    confirming_pairs: List[str]  # pairs that confirm this theme
    suggested_trades: List[Tuple[str, str]]  # [(pair, direction), ...] to stack

    @property
    def trade_count(self) -> int:
        return min(len(self.suggested_trades), STACK_MAX)

    @property
    def position_fraction(self) -> float:
        """Each trade's position size as fraction of normal.
        E.g. 4 stacked trades → each gets 0.25 of normal risk.
        Total exposure = same as one normal trade.
        """
        return 1.0 / max(self.trade_count, 1)

    def __str__(self) -> str:
        direction_label = "WEAK ↓" if self.direction == "weak" else "STRONG ↑"
        pairs_str = ", ".join(self.confirming_pairs[:4])
        return (f"{self.currency} {direction_label} "
                f"(score={self.score:.1f}, {self.trade_count} trades: {pairs_str})")


# ── Main analyzer ────────────────────────────────────────────────────────── #

class CurrencyStrengthAnalyzer:
    """
    Analyses all available pair data to detect dominant macro currency themes.

    Usage:
        analyzer = CurrencyStrengthAnalyzer()
        theme = analyzer.get_dominant_theme(candle_data)
        if theme:
            # Stack positions on theme.suggested_trades
    """

    def __init__(
        self,
        threshold: float = THEME_THRESHOLD,
        momentum_short: int = MOMENTUM_DAYS_S,
        momentum_long:  int = MOMENTUM_DAYS_L,
    ):
        self.threshold      = threshold
        self.momentum_short = momentum_short
        self.momentum_long  = momentum_long

    def compute_strength(self, candle_data: Dict[str, dict]) -> Dict[str, float]:
        """
        Compute a strength score for each currency based on available pair data.

        Score interpretation:
          +3.0 = currency is strong on 3 pairs (bullish bias)
          -3.0 = currency is weak on 3 pairs (bearish bias)
          0.0  = neutral or mixed signals

        Each pair contributes ±1.0 (short momentum) plus ±0.5 (long momentum bonus)
        when both timeframes agree. Disagreement cancels.
        """
        scores: Dict[str, float] = {c: 0.0 for c in
                                     ["GBP","EUR","USD","AUD","NZD","CAD","CHF","JPY"]}
        pair_signals: Dict[str, float] = {}   # pair → net signal (-1 to +1)

        for pair, pdata in candle_data.items():
            if pair not in PAIR_CURRENCY_MAP:
                continue
            df_d = pdata.get("d")
            if df_d is None or len(df_d) < self.momentum_long + 2:
                continue

            closes = df_d["close"].values

            # Short-term momentum: N-day return (sign only)
            ret_s = (closes[-1] - closes[-self.momentum_short]) / closes[-self.momentum_short]
            # Long-term momentum: direction over wider window
            ret_l = (closes[-1] - closes[-self.momentum_long])  / closes[-self.momentum_long]

            # Signal: both agree → full weight; disagree → zero
            if ret_s > 0 and ret_l > 0:
                signal =  1.0    # pair trending up = base strong, quote weak
            elif ret_s < 0 and ret_l < 0:
                signal = -1.0    # pair trending down = base weak, quote strong
            else:
                signal =  0.0    # mixed signals — skip this pair

            if signal == 0.0:
                continue

            pair_signals[pair] = signal

            base, quote = PAIR_CURRENCY_MAP[pair]
            scores[base]  += signal          # base benefits when pair goes up
            scores[quote] -= signal          # quote suffers when pair goes up

        return scores

    def get_dominant_theme(
        self, candle_data: Dict[str, dict]
    ) -> Optional[CurrencyTheme]:
        """
        Returns a CurrencyTheme if any currency is dominant enough to stack trades.

        Priority: weakest currency first (more opportunities when something is
        clearly in free-fall than when something is merely strong).
        """
        scores = self.compute_strength(candle_data)

        # Find the currency with the most extreme score
        most_extreme_currency = max(scores.items(), key=lambda x: abs(x[1]))
        currency, score = most_extreme_currency

        if abs(score) < self.threshold:
            return None    # no dominant theme

        direction = "weak" if score < 0 else "strong"

        # Build list of confirming pairs and trade suggestions
        confirming_pairs: List[str] = []
        suggested_trades: List[Tuple[str, str]] = []

        for pair, (base, quote) in PAIR_CURRENCY_MAP.items():
            if pair not in candle_data:
                continue

            if currency == base:
                pair_direction = "long"  if direction == "strong" else "short"
                confirming_pairs.append(pair)
                suggested_trades.append((pair, pair_direction))
            elif currency == quote:
                pair_direction = "short" if direction == "strong" else "long"
                confirming_pairs.append(pair)
                suggested_trades.append((pair, pair_direction))

        # Rank suggested trades by momentum strength (strongest first)
        ranked: List[Tuple[str, str, float]] = []
        for pair, trade_dir in suggested_trades:
            pdata = candle_data.get(pair, {})
            df_d  = pdata.get("d")
            if df_d is None or len(df_d) < 5:
                continue
            closes = df_d["close"].values
            momentum = abs(closes[-1] - closes[-5]) / closes[-5]
            ranked.append((pair, trade_dir, momentum))
        ranked.sort(key=lambda x: x[2], reverse=True)
        suggested_trades = [(p, d) for p, d, _ in ranked[:STACK_MAX]]

        return CurrencyTheme(
            currency          = currency,
            direction         = direction,
            score             = score,
            confirming_pairs  = [p for p, _ in suggested_trades],
            suggested_trades  = suggested_trades,
        )

    def get_all_themes(
        self, candle_data: Dict[str, dict], min_score: float = 2.0
    ) -> List[CurrencyTheme]:
        """
        Returns ALL currencies with score above min_score, ranked by strength.
        Useful for scanning multiple concurrent themes.
        """
        scores = self.compute_strength(candle_data)
        themes = []
        for currency, score in sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(score) >= min_score:
                # Build a minimal theme object for each
                direction = "weak" if score < 0 else "strong"
                themes.append(CurrencyTheme(
                    currency         = currency,
                    direction        = direction,
                    score            = score,
                    confirming_pairs = [],
                    suggested_trades = [],
                ))
        return themes
