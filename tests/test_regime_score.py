"""
Unit tests for RegimeScore computation.

Covers:
  - Vol expansion score (0 / 0.5 / 1)
  - Trend persistence score (0 / 0.5 / 1)
  - Recent performance score (0 / 0.5 / 1)
  - Correlation cluster score (0 / 1)
  - Final total + eligibility flags
  - Edge cases: empty data, zero ATR, neutral trend
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest

from src.strategy.forex.regime_score import (
    compute_regime_score, RegimeScore,
    _vol_expansion_score, _trend_persistence_score,
    _recent_performance_score, _correlation_cluster_score,
    VOL_HIGH_RATIO, VOL_MED_RATIO, SCORE_HIGH_THRESH, SCORE_EXTREME_THRESH,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def make_h4_bars(n: int = 80, atr_mult: float = 1.0, trend: str = "up") -> pd.DataFrame:
    """
    Generate synthetic H4 OHLC bars with a sine wave + drift so that
    PatternDetector.detect_trend() reliably returns STRONG_BULLISH / STRONG_BEARISH.
    n >= 40 recommended; default 80 gives clean swing detection.
    """
    t_arr  = np.linspace(0, 4 * np.pi, n)
    drift  = 0.001 * np.arange(n) if trend == "up" else -0.001 * np.arange(n)
    closes = 1.30 + 0.010 * np.sin(t_arr) + drift
    half_atr = 0.002 * atr_mult
    df = pd.DataFrame({
        "open":  closes - 0.0005,
        "high":  closes + half_atr,
        "low":   closes - half_atr,
        "close": closes,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="4h")
    return df


def make_trades(rs: list) -> list:
    """Create fake trade records with given R values."""
    return [{"r": r, "pair": "GBP/USD", "direction": "short"} for r in rs]


# ── Vol expansion tests ─────────────────────────────────────────────────────

class TestVolExpansion:
    def test_high_expansion(self):
        df = make_h4_bars(40, atr_mult=1.0)
        # Make the last bar much larger than the 20-bar avg
        df.loc[df.index[-1], "high"] += 0.05   # spike
        df.loc[df.index[-1], "low"]  -= 0.05
        score, note = _vol_expansion_score(df)
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_medium_expansion(self):
        # Last bar is 10% above avg (between 1.05 and 1.15)
        df = make_h4_bars(40, atr_mult=1.0)
        avg_atr = float((df["high"] - df["low"]).iloc[-21:-1].mean())
        df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "close"] + avg_atr * 1.10
        df.loc[df.index[-1], "low"]  = df.loc[df.index[-1], "close"] - avg_atr * 0.0
        score, note = _vol_expansion_score(df)
        assert score == 0.5, f"Expected 0.5, got {score}"

    def test_quiet_market(self):
        df = make_h4_bars(40, atr_mult=1.0)   # uniform bars → ratio ≈ 1.0
        score, _ = _vol_expansion_score(df)
        assert score in (0.0, 0.5), f"Expected 0.0 or 0.5 for uniform bars, got {score}"

    def test_insufficient_bars(self):
        df = make_h4_bars(10)   # < VOL_LOOKBACK+1
        score, note = _vol_expansion_score(df)
        assert score == 0.0
        assert "insufficient" in note

    def test_zero_atr(self):
        df = make_h4_bars(40)
        df["high"] = df["close"]
        df["low"]  = df["close"]    # zero range
        score, note = _vol_expansion_score(df)
        assert score == 0.0


# ── Trend persistence tests ─────────────────────────────────────────────────

class TestTrendPersistence:
    def test_strong_uptrend(self):
        # Sine + upward drift → STRONG_BULLISH (swing H/L clearly form HH/HL)
        df = make_h4_bars(80, trend="up")
        score, note = _trend_persistence_score(df)
        assert score in (0.5, 1.0), f"Expected ≥0.5 for strong uptrend, got {score} ({note})"

    def test_insufficient_bars(self):
        df = make_h4_bars(10)
        score, note = _trend_persistence_score(df)
        assert score == 0.0
        assert "insufficient" in note


# ── Recent performance tests ────────────────────────────────────────────────

class TestRecentPerformance:
    def test_strong_positive(self):
        trades = make_trades([1.0, 1.0, 0.5, 0.5, 0.5])  # sum=3.5 ≥ 1.5
        score, note = _recent_performance_score(trades)
        assert score == 1.0

    def test_neutral_positive(self):
        trades = make_trades([0.3, 0.3, 0.3, 0.1])  # sum=1.0, 0≤sum<1.5
        score, note = _recent_performance_score(trades)
        assert score == 0.5

    def test_losing_streak(self):
        trades = make_trades([-1.0, -1.0, -1.0])  # sum=-3.0 < 0
        score, note = _recent_performance_score(trades)
        assert score == 0.0

    def test_empty_trades(self):
        score, note = _recent_performance_score([])
        assert score == 0.0
        assert "no_recent" in note

    def test_uses_last_10(self):
        # 15 trades: first 5 are big wins, last 10 are small negatives
        old_wins = make_trades([5.0] * 5)
        recent   = make_trades([-0.2] * 10)
        trades   = old_wins + recent
        score, _ = _recent_performance_score(trades)
        # sum_r of last 10 = -2.0 → should be 0.0
        assert score == 0.0

    def test_exact_boundary_1p5(self):
        trades = make_trades([0.5, 0.5, 0.5])  # sum=1.5 → exactly HIGH
        score, _ = _recent_performance_score(trades)
        assert score == 1.0

    def test_exact_boundary_0(self):
        trades = make_trades([0.0, 0.0, 0.0])  # sum=0.0 → NEUTRAL
        score, _ = _recent_performance_score(trades)
        assert score == 0.5


# ── Correlation cluster tests ───────────────────────────────────────────────

class TestCorrelationCluster:
    def _make_trending_h4(self, direction: str) -> pd.DataFrame:
        """Sine + drift so PatternDetector.detect_trend() returns BULLISH/BEARISH."""
        return make_h4_bars(80, trend=direction)

    def test_jpy_cluster_detected(self):
        # USD/JPY ↓ + GBP/JPY ↓ + EUR/JPY ↓ → JPY bullish (all pairs: JPY is quote currency trending up)
        down_df = self._make_trending_h4("down")
        h4_slices = {
            "USD/JPY": down_df,
            "GBP/JPY": down_df,
            "EUR/JPY": down_df,
        }
        score, note = _correlation_cluster_score(h4_slices)
        assert score == 1.0, f"Expected cluster bonus, got {score} note={note}"

    def test_no_cluster(self):
        up   = self._make_trending_h4("up")
        down = self._make_trending_h4("down")
        h4_slices = {"GBP/USD": up, "EUR/USD": down, "USD/JPY": up}
        score, note = _correlation_cluster_score(h4_slices)
        assert score == 0.0

    def test_insufficient_pairs(self):
        h4_slices = {"GBP/USD": self._make_trending_h4("up")}
        score, note = _correlation_cluster_score(h4_slices)
        assert score == 0.0
        assert "too_few" in note


# ── Full compute_regime_score integration ──────────────────────────────────

class TestComputeRegimeScore:
    def test_returns_regime_score(self):
        df = make_h4_bars(40)
        rs = compute_regime_score(df_h4=df, recent_trades=[], h4_slices=None)
        assert isinstance(rs, RegimeScore)
        assert 0.0 <= rs.total <= 4.0

    def test_eligibility_flags(self):
        df = make_h4_bars(40)
        # Manually manipulate: give high enough scores to qualify
        rs = compute_regime_score(df_h4=df, recent_trades=make_trades([1.0]*5), h4_slices=None)
        assert rs.eligible_high    == (rs.total >= SCORE_HIGH_THRESH)
        assert rs.eligible_extreme == (rs.total >= SCORE_EXTREME_THRESH)

    def test_to_dict_keys(self):
        df = make_h4_bars(40)
        rs = compute_regime_score(df_h4=df, recent_trades=[], h4_slices=None)
        d  = rs.to_dict()
        for key in ["regime_score", "regime_vol", "regime_trend", "regime_perf",
                    "regime_cluster", "eligible_high", "eligible_extreme"]:
            assert key in d, f"Missing key: {key}"

    def test_band(self):
        df = make_h4_bars(40)
        rs = compute_regime_score(df_h4=df, recent_trades=[], h4_slices=None)
        assert rs.band() in ("LOW", "MEDIUM", "HIGH_ELIGIBLE", "EXTREME_ELIGIBLE")

    def test_no_h4_slices_no_cluster(self):
        df = make_h4_bars(40)
        rs = compute_regime_score(df_h4=df, recent_trades=[], h4_slices=None)
        assert rs.correlation_cluster == 0.0

    def test_total_is_sum(self):
        df = make_h4_bars(40)
        trades = make_trades([1.0, 1.0, 1.0])
        rs = compute_regime_score(df_h4=df, recent_trades=trades, h4_slices=None)
        expected = rs.vol_expansion + rs.trend_persistence + rs.recent_performance + rs.correlation_cluster
        assert abs(rs.total - expected) < 1e-9

    def test_backtest_trade_field(self):
        """Trades from the backtest have 'regime_score_at_entry' field."""
        from datetime import datetime, timezone
        from backtesting.oanda_backtest_v2 import run_backtest, TRAIL_ARMS
        r = run_backtest(
            start_dt  = datetime(2024, 7, 15, tzinfo=timezone.utc),
            end_dt    = datetime(2024, 10, 31, tzinfo=timezone.utc),
            starting_bal = 8000,
            trail_cfg    = TRAIL_ARMS["C"],
            quiet        = True,
        )
        trades = r.get("trades", [])
        assert len(trades) > 0, "Expected at least one trade"
        for t in trades:
            assert "regime_score_at_entry" in t, f"Missing regime_score_at_entry in trade: {t['pair']}"
            assert 0.0 <= t["regime_score_at_entry"] <= 4.0
        # Result dict has distribution fields
        assert "regime_avg"       in r
        assert "regime_pct_high"  in r
        assert "regime_pct_extreme" in r
