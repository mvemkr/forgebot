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


# ══════════════════════════════════════════════════════════════════════════════
# Risk Mode System (Regime-based risk scaling)
# ══════════════════════════════════════════════════════════════════════════════
#
# Simplified integer score (0–4) → RiskMode mapping.
# Used by ForexRiskManager to select risk %, weekly cap, and DD thresholds.
#
# Score rules (one point each, no partial credit):
#   +1  Weekly trend == Daily trend  (macro alignment)
#   +1  H4 ATR ratio > 1.1           (volatility expanding)
#   +1  last-5-trades sum_R > 0      (recent positive edge)
#   +1  loss_streak == 0             (no active loss run)
#
# Mode thresholds:
#   score 0–1  → LOW     (reduce risk, tighten cap)
#   score 2    → MEDIUM  (normal risk)
#   score 3    → HIGH    (increase risk)
#   score 4    → EXTREME (max risk, zero streak required — automatic when score=4)

from enum import Enum


class RiskMode(Enum):
    LOW     = "LOW"
    MEDIUM  = "MEDIUM"
    HIGH    = "HIGH"
    EXTREME = "EXTREME"


# ── Per-mode risk parameters ───────────────────────────────────────────────
# Multipliers applied to the base tier risk %.
# Example: base=6%, HIGH multiplier=1.5 → 9% risk.
# Weekly cap: max trades per ISO week regardless of account size.
RISK_MODE_PARAMS: dict[str, dict] = {
    # LOW — capital-preservation mode: half risk, tighter DD/streak caps
    "LOW": {
        "risk_mult":        0.5,
        "weekly_cap_small": 1, "weekly_cap_std": 1,
        "dd_l1_cap":        6.0,    # tighten DD_L1 10% → 6%
        "dd_l2_cap":        3.0,    # tighten DD_L2 6% → 3%
        "streak_l2_cap":    3.0,    # tighten streak-L2 6% → 3%
        "streak_l3_cap":    1.5,    # tighten streak-L3 3% → 1.5%
    },
    # MEDIUM — baseline/normal: default constants unchanged
    "MEDIUM": {
        "risk_mult":        1.0,
        "weekly_cap_small": 1, "weekly_cap_std": 2,
        "dd_l1_cap":        10.0,   # baseline DD_L1
        "dd_l2_cap":        6.0,    # baseline DD_L2
        "streak_l2_cap":    6.0,    # baseline streak-L2
        "streak_l3_cap":    3.0,    # baseline streak-L3
    },
    # HIGH — growth mode: 1.5× risk, looser DD/streak caps
    "HIGH": {
        "risk_mult":        1.5,
        "weekly_cap_small": 2, "weekly_cap_std": 3,
        "dd_l1_cap":        15.0,   # loosen DD_L1 10% → 15%
        "dd_l2_cap":        10.0,   # loosen DD_L2 6% → 10%
        "streak_l2_cap":    9.0,    # loosen streak-L2 6% → 9%
        "streak_l3_cap":    6.0,    # loosen streak-L3 3% → 6%
    },
    # EXTREME — max-compounding mode: 2× risk, maximum caps
    "EXTREME": {
        "risk_mult":        2.0,
        "weekly_cap_small": 3, "weekly_cap_std": 4,
        "dd_l1_cap":        20.0,   # loosen DD_L1 10% → 20%
        "dd_l2_cap":        15.0,   # loosen DD_L2 6% → 15%
        "streak_l2_cap":    12.0,   # loosen streak-L2 6% → 12%
        "streak_l3_cap":    9.0,    # loosen streak-L3 3% → 9%
    },
}

ATR_RATIO_THRESH   = 1.10  # H4 ATR expansion threshold for HIGH promotion
ATR_DEMOTE_THRESH  = 1.00  # ATR below this → immediate demotion (hysteresis band: 1.00–1.10)
LAST5_SUMR_THRESH  = 0.0   # last-5 sum_R must be > this for HIGH
LAST10_SUMR_EXTREME = 1.5  # last-10 sum_R must be ≥ this for EXTREME
DD_EXTREME_MAX_PCT  = 10.0  # drawdown % must be < this for EXTREME


@dataclass
class RegimeModeScore:
    """Simplified integer-score regime assessment → RiskMode."""
    wd_aligned:    bool    # Weekly trend == Daily trend
    atr_expanding: bool    # H4 ATR ratio ≥ ATR_RATIO_THRESH (1.10)
    edge_positive: bool    # last-5 sum_R > LAST5_SUMR_THRESH
    streak_clear:  bool    # loss_streak == 0
    score:         int     # 0–4
    mode:          RiskMode
    atr_ratio:     float = 0.0
    last5_sum_r:   float = 0.0
    loss_streak:   int   = 0
    # ── Hysteresis state (caller must persist and pass back each call) ─────
    consecutive_high_bars: int   = 0    # output: consecutive bars where high_conds_met=True
    # ── Extended diagnostics ──────────────────────────────────────────────
    last10_sum_r:   float = 0.0
    dd_pct:         float = 0.0
    promotion_note: str   = ""   # non-empty when mode == HIGH/EXTREME; use for debug logging

    def to_dict(self) -> dict:
        d = {
            "risk_mode":       self.mode.value,
            "regime_score_v2": self.score,
            "wd_aligned":      self.wd_aligned,
            "atr_expanding":   self.atr_expanding,
            "edge_positive":   self.edge_positive,
            "streak_clear":    self.streak_clear,
            "atr_ratio":       round(self.atr_ratio, 3),
            "last5_sum_r":     round(self.last5_sum_r, 2),
            "loss_streak":     self.loss_streak,
            "consec_high":     self.consecutive_high_bars,
        }
        if self.promotion_note:
            d["promotion_note"] = self.promotion_note
        return d

    @property
    def params(self) -> dict:
        return RISK_MODE_PARAMS[self.mode.value]


def compute_risk_mode(
    trend_weekly:          str,
    trend_daily:           str,
    df_h4:                 "pd.DataFrame",
    recent_trades:         list,
    loss_streak:           int,
    dd_pct:                float = 0.0,
    consecutive_high_bars: int   = 0,
) -> RegimeModeScore:
    """Compute risk mode from four independent conditions + hysteresis state.

    HIGH promotion rules (ALL required — not any-3-of-4):
      1. W==D aligned        – weekly and daily trend point same direction
      2. ATR expanding       – ATR_H4 / ATR_H4_20avg >= ATR_RATIO_THRESH (1.10)
      3. Recent edge         – last-5 sum_R > LAST5_SUMR_THRESH (0.0)
      4. No loss streak      – loss_streak == 0
      5. Hysteresis          – conditions held for ≥2 consecutive evaluations
                               (caller must persist consecutive_high_bars across calls)

    EXTREME promotion: all HIGH conditions + score==4 + last10_sum_R>=1.5 + dd<10%

    Immediate demotion (resets hysteresis counter):
      W!=D  OR  ATR ratio < ATR_DEMOTE_THRESH (1.00)  OR  loss_streak >= 1

    Args:
        trend_weekly, trend_daily : bias strings ("bullish" / "bearish" / ...)
        df_h4                     : H4 OHLC DataFrame, ≥21 bars for ATR
        recent_trades             : trade dicts with 'r' field (last 5/10 used)
        loss_streak               : consecutive loss count from risk manager
        dd_pct                    : current drawdown % from peak (EXTREME gate)
        consecutive_high_bars     : number of prior consecutive qualifying bars
                                    (caller persists from previous call's output)

    Returns:
        RegimeModeScore — note: `consecutive_high_bars` in the result is the
        UPDATED counter the caller should store for the next call.
    """
    # ── Component 1: Weekly == Daily alignment ────────────────────────────
    def _bull(t: str) -> bool:
        return "bullish" in (t or "").lower()

    def _bear(t: str) -> bool:
        return "bearish" in (t or "").lower()

    wd_aligned = (
        (_bull(trend_weekly) and _bull(trend_daily)) or
        (_bear(trend_weekly) and _bear(trend_daily))
    )

    # ── Component 2: H4 ATR expansion ─────────────────────────────────────
    atr_ratio     = 0.0
    atr_expanding = False
    if df_h4 is not None and len(df_h4) >= 21:
        try:
            hi = df_h4["high"].values if "high" in df_h4.columns else df_h4["High"].values
            lo = df_h4["low"].values  if "low"  in df_h4.columns else df_h4["Low"].values
            tr = hi - lo
            current_atr = float(tr[-1])
            avg_atr     = float(np.mean(tr[-21:-1]))
            if avg_atr > 0:
                atr_ratio     = current_atr / avg_atr
                atr_expanding = atr_ratio >= ATR_RATIO_THRESH   # >= 1.10
        except Exception:
            pass

    # ── Component 3: Last-5 sum_R ─────────────────────────────────────────
    last5       = (recent_trades or [])[-5:]
    last5_sum_r = sum(t.get("r", 0.0) for t in last5)
    edge_positive = last5_sum_r > LAST5_SUMR_THRESH

    # ── Component 4: Loss streak clear ────────────────────────────────────
    streak_clear = loss_streak == 0

    # ── Integer score (0–4) for EXTREME eligibility check ─────────────────
    score = sum([wd_aligned, atr_expanding, edge_positive, streak_clear])

    # ── HIGH: ALL 4 conditions required (strict gate, replaces any-3-of-4) ─
    # Every condition must be True — W==D is not just a prerequisite but a
    # full veto; ATR must exceed the promotion threshold (1.10); edge and
    # streak together ensure we only scale up during confirmed good conditions.
    high_conds_met = (
        wd_aligned
        and atr_expanding                # atr_ratio >= 1.10
        and edge_positive                # last5_sum_r > 0
        and streak_clear                 # loss_streak == 0
    )

    # ── Immediate demotion: resets hysteresis counter ─────────────────────
    # Triggers when: W!=D  OR  ATR ratio < ATR_DEMOTE_THRESH (1.00)  OR  streak>=1
    # ATR hysteresis band: promote at >= 1.10, demote at < 1.00.
    # Gray zone [1.00, 1.10): not qualifying but no hard demotion → counter
    # still resets to 0 because high_conds_met is False (atr < 1.10).
    _immediate_demote = (
        not wd_aligned
        or atr_ratio < ATR_DEMOTE_THRESH   # < 1.00
        or loss_streak >= 1
    )

    # ── Hysteresis counter update ─────────────────────────────────────────
    # Increment when conditions met; reset on any demotion signal.
    # Using _immediate_demote as override ensures ATR < 1.00 and streak
    # always reset even if somehow high_conds_met were True (defensive).
    consec_out = (
        (consecutive_high_bars + 1) if (high_conds_met and not _immediate_demote) else 0
    )

    # ── HIGH / EXTREME eligibility ────────────────────────────────────────
    # HIGH: all conditions held for this bar AND at least 1 prior qualifying bar
    # (i.e., current is the 2nd consecutive qualifying bar).
    high_eligible = high_conds_met and (consecutive_high_bars >= 1) and not _immediate_demote

    # EXTREME: HIGH eligible + score=4 + sustained performance + low DD
    last10       = (recent_trades or [])[-10:]
    last10_sum_r = sum(t.get("r", 0.0) for t in last10)
    extreme_eligible = (
        high_eligible
        and score == 4
        and last10_sum_r >= LAST10_SUMR_EXTREME   # >= +1.5R across last 10 trades
        and dd_pct < DD_EXTREME_MAX_PCT            # drawdown < 10%
    )

    # ── Mode mapping ──────────────────────────────────────────────────────
    if extreme_eligible:
        mode = RiskMode.EXTREME
    elif high_eligible:
        mode = RiskMode.HIGH
    elif score >= 2:
        mode = RiskMode.MEDIUM
    else:
        mode = RiskMode.LOW

    # ── Promotion debug note ──────────────────────────────────────────────
    promotion_note = ""
    if mode in (RiskMode.HIGH, RiskMode.EXTREME):
        promotion_note = (
            f"W={trend_weekly[:4] if trend_weekly else '?'} "
            f"D={trend_daily[:4] if trend_daily else '?'} "
            f"ATR={atr_ratio:.3f} "
            f"last5R={last5_sum_r:+.2f} "
            f"streak={loss_streak} "
            f"dd={dd_pct:.1f}% "
            f"consec={consecutive_high_bars}→{consec_out} "
            f"score={score}"
        )

    return RegimeModeScore(
        wd_aligned             = wd_aligned,
        atr_expanding          = atr_expanding,
        edge_positive          = edge_positive,
        streak_clear           = streak_clear,
        score                  = score,
        mode                   = mode,
        atr_ratio              = atr_ratio,
        last5_sum_r            = last5_sum_r,
        loss_streak            = loss_streak,
        consecutive_high_bars  = consec_out,
        last10_sum_r           = last10_sum_r,
        dd_pct                 = dd_pct,
        promotion_note         = promotion_note,
    )
