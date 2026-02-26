"""
RegimeScore — Market Regime Measurement

Computes a deterministic 0–4 score across four independent dimensions:

  1. Volatility Expansion  (0 / 0.5 / 1)
  2. Trend Persistence     (0 / 0.5 / 1)
  3. Recent Performance    (0 / 0.5 / 1)
  4. Correlation Cluster   (0 / 1)

RegimeScore total = sum of four components.

Escalation eligibility thresholds (for future Risk Mode AUTO gating):
  score ≥ 3.0  →  eligible_high    (HIGH risk mode allowed)
  score ≥ 3.5  →  eligible_extreme (EXTREME risk mode allowed)

NOTE: This module scores only. No risk-sizing changes.
No trailing changes. No signal changes. Pure read-only observation.

Usage (backtester):
    from src.strategy.forex.regime_score import compute_regime_score
    rs = compute_regime_score(df_h4, recent_trades, candle_data_h4)

Usage (live orchestrator):
    rs = compute_regime_score(df_h4, journal.get_recent_trades(10), live_h4_slices)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from .pattern_detector import PatternDetector, Trend

logger = logging.getLogger(__name__)

# ── Thresholds (tunable without code changes) ──────────────────────────────
VOL_HIGH_RATIO     = 1.15   # ATR ratio → full vol score
VOL_MED_RATIO      = 1.05   # ATR ratio → half vol score
VOL_LOOKBACK       = 20     # bars for ATR average

PERF_HIGH_SUMR     = 1.5    # last-10 sum_R → full perf score
PERF_MED_SUMR      = 0.0    # last-10 sum_R → half perf score

CLUSTER_MIN_PAIRS  = 3      # pairs in same currency direction → cluster bonus

SCORE_HIGH_THRESH    = 3.0  # eligible for HIGH risk mode
SCORE_EXTREME_THRESH = 3.5  # eligible for EXTREME risk mode

_DETECTOR = PatternDetector()   # module-level singleton (stateless, thread-safe)


# ── Result ─────────────────────────────────────────────────────────────────

@dataclass
class RegimeScore:
    """Regime measurement at a single point in time."""
    vol_expansion:      float   # 0, 0.5, or 1.0
    trend_persistence:  float   # 0, 0.5, or 1.0
    recent_performance: float   # 0, 0.5, or 1.0
    correlation_cluster: float  # 0 or 1.0
    total:              float   # sum 0–4
    eligible_high:      bool    # total >= 3.0
    eligible_extreme:   bool    # total >= 3.5
    notes:              List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "regime_score":         round(self.total, 2),
            "regime_vol":           self.vol_expansion,
            "regime_trend":         self.trend_persistence,
            "regime_perf":          self.recent_performance,
            "regime_cluster":       self.correlation_cluster,
            "eligible_high":        self.eligible_high,
            "eligible_extreme":     self.eligible_extreme,
            "regime_notes":         self.notes,
        }

    def band(self) -> str:
        """Textual band for display."""
        if self.total >= SCORE_EXTREME_THRESH:
            return "EXTREME_ELIGIBLE"
        if self.total >= SCORE_HIGH_THRESH:
            return "HIGH_ELIGIBLE"
        if self.total >= 2.0:
            return "MEDIUM"
        return "LOW"


# ── Component scorers ──────────────────────────────────────────────────────

def _vol_expansion_score(df_h4: pd.DataFrame) -> tuple[float, str]:
    """
    ATR_H4_current / ATR_H4_20bar_avg.
      > 1.15  → 1.0  (expansion)
      > 1.05  → 0.5  (slight expansion)
      else    → 0.0  (quiet / contracting)

    Uses high-low range as ATR proxy (avoids gap sensitivity on H4).
    """
    if len(df_h4) < VOL_LOOKBACK + 1:
        return 0.0, "insufficient_h4_bars"

    tr = (df_h4["high"] - df_h4["low"]).values
    current_atr = float(tr[-1])
    avg_atr     = float(np.mean(tr[-(VOL_LOOKBACK + 1):-1]))

    if avg_atr <= 0:
        return 0.0, "zero_avg_atr"

    ratio = current_atr / avg_atr
    if ratio > VOL_HIGH_RATIO:
        return 1.0, f"vol_high ratio={ratio:.2f}"
    if ratio > VOL_MED_RATIO:
        return 0.5, f"vol_med ratio={ratio:.2f}"
    return 0.0, f"vol_quiet ratio={ratio:.2f}"


def _trend_persistence_score(df_h4: pd.DataFrame) -> tuple[float, str]:
    """
    H4 market structure quality.
      STRONG_BULLISH / STRONG_BEARISH → 1.0  (clean directional)
      BULLISH / BEARISH               → 0.5  (mild structure)
      NEUTRAL                         → 0.0  (chop)

    Delegates to PatternDetector.detect_trend() which uses HH/HL / LL/LH swing logic.
    """
    if len(df_h4) < 20:
        return 0.0, "insufficient_h4_bars"

    trend = _DETECTOR.detect_trend(df_h4)
    if trend in (Trend.STRONG_BULLISH, Trend.STRONG_BEARISH):
        return 1.0, f"trend_strong:{trend.value}"
    if trend in (Trend.BULLISH, Trend.BEARISH):
        return 0.5, f"trend_mild:{trend.value}"
    return 0.0, f"trend_neutral:{trend.value}"


def _recent_performance_score(recent_trades: List[dict]) -> tuple[float, str]:
    """
    Last N closed trades sum_R.
      sum_R >= 1.5  → 1.0
      sum_R >= 0.0  → 0.5
      sum_R <  0.0  → 0.0

    Trades must have an 'r' field (realized R multiple).
    Uses last min(10, len(trades)) trades.
    """
    if not recent_trades:
        return 0.0, "no_recent_trades"

    last = recent_trades[-10:]
    sum_r = sum(t.get("r", 0.0) for t in last)
    n     = len(last)

    if sum_r >= PERF_HIGH_SUMR:
        return 1.0, f"perf_strong sum_R={sum_r:.2f} n={n}"
    if sum_r >= PERF_MED_SUMR:
        return 0.5, f"perf_neutral sum_R={sum_r:.2f} n={n}"
    return 0.0, f"perf_weak sum_R={sum_r:.2f} n={n}"


def _correlation_cluster_score(
    h4_slices: Dict[str, pd.DataFrame],
) -> tuple[float, str]:
    """
    Check if ≥3 pairs express the same directional bias on a single currency.

    Method:
      For each pair in h4_slices, detect H4 trend.
      Map to currency bias: GBP/USD BULLISH → GBP +1, USD -1.
      If any currency's net directional count ≥ CLUSTER_MIN_PAIRS → bonus.

    Example: USD/JPY↓ + GBP/JPY↓ + EUR/JPY↓ → JPY +3 → cluster detected.
    """
    if len(h4_slices) < CLUSTER_MIN_PAIRS:
        return 0.0, "too_few_pairs"

    ccy_bias: dict[str, int] = {}

    for pair, df_h4 in h4_slices.items():
        if "/" not in pair or len(df_h4) < 20:
            continue
        base, quote = pair.split("/")
        trend = _DETECTOR.detect_trend(df_h4)

        if trend in (Trend.BULLISH, Trend.STRONG_BULLISH):
            ccy_bias[base]  = ccy_bias.get(base,  0) + 1
            ccy_bias[quote] = ccy_bias.get(quote, 0) - 1
        elif trend in (Trend.BEARISH, Trend.STRONG_BEARISH):
            ccy_bias[base]  = ccy_bias.get(base,  0) - 1
            ccy_bias[quote] = ccy_bias.get(quote, 0) + 1
        # NEUTRAL: no contribution

    # Check if any currency has |bias| >= threshold
    dominated = [(ccy, abs(score)) for ccy, score in ccy_bias.items()
                 if abs(score) >= CLUSTER_MIN_PAIRS]

    if dominated:
        top = max(dominated, key=lambda x: x[1])
        return 1.0, f"cluster:{top[0]}×{top[1]}"
    return 0.0, f"no_cluster max={max(abs(v) for v in ccy_bias.values()) if ccy_bias else 0}"


# ── Main entry point ───────────────────────────────────────────────────────

def compute_regime_score(
    df_h4: pd.DataFrame,
    recent_trades: List[dict],
    h4_slices: Optional[Dict[str, pd.DataFrame]] = None,
) -> RegimeScore:
    """
    Compute all four regime components and return a RegimeScore.

    Args:
        df_h4         : H4 candles for the primary pair (or any representative pair)
                        Used for vol_expansion and trend_persistence.
        recent_trades : Last N closed trades with 'r' field.  Use [] if none.
        h4_slices     : Optional dict {pair → H4 DataFrame} for all tradeable pairs.
                        Required for correlation_cluster score.  Pass None to skip.

    Returns:
        RegimeScore dataclass with all components + total + eligibility flags.
    """
    notes = []

    vol_score,   vol_note   = _vol_expansion_score(df_h4)
    trend_score, trend_note = _trend_persistence_score(df_h4)
    perf_score,  perf_note  = _recent_performance_score(recent_trades)

    if h4_slices:
        cluster_score, cluster_note = _correlation_cluster_score(h4_slices)
    else:
        cluster_score, cluster_note = 0.0, "cluster_skipped"

    notes = [vol_note, trend_note, perf_note, cluster_note]
    total = vol_score + trend_score + perf_score + cluster_score

    return RegimeScore(
        vol_expansion       = vol_score,
        trend_persistence   = trend_score,
        recent_performance  = perf_score,
        correlation_cluster = cluster_score,
        total               = total,
        eligible_high       = total >= SCORE_HIGH_THRESH,
        eligible_extreme    = total >= SCORE_EXTREME_THRESH,
        notes               = notes,
    )
