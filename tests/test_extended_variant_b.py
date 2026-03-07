"""
tests/test_extended_variant_b.py
Unit tests for the extended Variant B multi-window confirmation study.

Coverage (6 classes):
  1. Config defaults         — production values unchanged
  2. Window definitions      — dates, ordering, roles, variant shape
  3. Helper functions        — _r, _is_win, _pct, _rs, _wr_str, _wr_pct,
                               _avg_r_val, _worst3l, _mae, _mfe,
                               _signal_type, _pattern, _is_strict_pin, _is_engulf
  4. WindowComparison        — unlock algebra, trigger breakdown,
                               strict-pin detection, pair/pattern concentration
  5. Promotion gate          — all 4 criteria, small-n bypass, pass/fail edges
  6. Script structure        — constants, callables, safety guards
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.strategy.forex.strategy_config as _sc


def _restore():
    _sc.ENTRY_TRIGGER_MODE           = "engulf_only"
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 2


# ──────────────────────────────────────────────────────────────────────────────
# 1. Config defaults
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigDefaults:

    def test_entry_trigger_mode_exists(self):
        assert hasattr(_sc, "ENTRY_TRIGGER_MODE")

    def test_entry_trigger_mode_is_engulf_only(self):
        assert _sc.ENTRY_TRIGGER_MODE == "engulf_only"

    def test_lookback_exists(self):
        assert hasattr(_sc, "ENGULF_CONFIRM_LOOKBACK_BARS")

    def test_lookback_is_2(self):
        assert _sc.ENGULF_CONFIRM_LOOKBACK_BARS == 2

    def test_engulfing_only_flag_true(self):
        assert _sc.ENGULFING_ONLY is True

    def test_strict_pin_mode_valid(self):
        assert hasattr(_sc, "ENTRY_TRIGGER_MODE")  # mode validated by strategy


# ──────────────────────────────────────────────────────────────────────────────
# 2. Window definitions
# ──────────────────────────────────────────────────────────────────────────────

class TestWindowDefinitions:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.extended_variant_b as m
        self.m = m

    def test_eight_windows(self):
        assert len(self.m.WINDOWS) == 8

    def test_full_year_is_first(self):
        label, ws, we, is_sub = self.m.WINDOWS[0]
        assert label == "Full-year"
        assert is_sub is False

    def test_full_year_dates(self):
        _, ws, we, _ = self.m.WINDOWS[0]
        assert ws == datetime(2025, 1,  1, tzinfo=timezone.utc)
        assert we == datetime(2026, 3,  1, tzinfo=timezone.utc)

    def test_q1_2025_dates(self):
        _, ws, we, is_sub = self.m.WINDOWS[1]
        assert ws == datetime(2025, 1,  1, tzinfo=timezone.utc)
        assert we == datetime(2025, 3, 31, tzinfo=timezone.utc)
        assert is_sub is True

    def test_q2_2025_dates(self):
        _, ws, we, _ = self.m.WINDOWS[2]
        assert ws == datetime(2025, 4, 1, tzinfo=timezone.utc)
        assert we == datetime(2025, 6, 30, tzinfo=timezone.utc)

    def test_q3_2025_dates(self):
        _, ws, we, _ = self.m.WINDOWS[3]
        assert ws == datetime(2025, 7, 1, tzinfo=timezone.utc)
        assert we == datetime(2025, 9, 30, tzinfo=timezone.utc)

    def test_q4_2025_dates(self):
        _, ws, we, _ = self.m.WINDOWS[4]
        assert ws == datetime(2025, 10, 1,  tzinfo=timezone.utc)
        assert we == datetime(2025, 12, 31, tzinfo=timezone.utc)

    def test_jan_feb_2026_dates(self):
        _, ws, we, _ = self.m.WINDOWS[5]
        assert ws == datetime(2026, 1,  1, tzinfo=timezone.utc)
        assert we == datetime(2026, 2, 28, tzinfo=timezone.utc)

    def test_w1_dates(self):
        label, ws, we, is_sub = self.m.WINDOWS[6]
        assert label == "W1"
        assert ws == datetime(2026, 2,  1, tzinfo=timezone.utc)
        assert we == datetime(2026, 2, 14, tzinfo=timezone.utc)
        assert is_sub is False

    def test_w2_dates(self):
        label, ws, we, is_sub = self.m.WINDOWS[7]
        assert label == "W2"
        assert ws == datetime(2026, 2, 15, tzinfo=timezone.utc)
        assert we == datetime(2026, 2, 28, tzinfo=timezone.utc)
        assert is_sub is False

    def test_five_subperiods(self):
        subs = [w for w in self.m.WINDOWS if w[3] is True]
        assert len(subs) == 5

    def test_windows_start_chronological(self):
        starts = [w[1] for w in self.m.WINDOWS]
        # Full-year and Q1 both start Jan 2025 (same); rest ascend
        assert starts[1] <= starts[2] <= starts[3] <= starts[4] <= starts[5]

    def test_two_variants_only(self):
        labels = [v[0] for v in self.m.VARIANTS]
        assert labels == ["A", "B"]

    def test_variant_a_engulf_only(self):
        _, mode, lb, _ = self.m.VARIANTS[0]
        assert mode == "engulf_only"
        assert lb   == 2

    def test_variant_b_strict_pin(self):
        _, mode, lb, _ = self.m.VARIANTS[1]
        assert mode == "engulf_or_strict_pin_at_level"
        assert lb   == 2

    def test_full_year_label_constant(self):
        assert self.m.FULL_YEAR_LABEL == "Full-year"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Helper functions
# ──────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.extended_variant_b as m
        self.m = m

    # _r
    def test_r_from_r_key(self):
        assert self.m._r({"r": 1.5}) == pytest.approx(1.5)

    def test_r_from_realised_r(self):
        assert self.m._r({"realised_r": -0.7}) == pytest.approx(-0.7)

    def test_r_from_result_r(self):
        assert self.m._r({"result_r": 2.1}) == pytest.approx(2.1)

    def test_r_missing_zero(self):
        assert self.m._r({}) == 0.0

    def test_r_prefers_r_key(self):
        assert self.m._r({"r": 1.0, "realised_r": 99.0}) == pytest.approx(1.0)

    # _is_win
    def test_is_win_positive(self):
        assert self.m._is_win({"r": 0.01}) is True

    def test_is_win_zero_is_not_win(self):
        assert self.m._is_win({"r": 0.0}) is False

    def test_is_win_negative(self):
        assert self.m._is_win({"r": -1.0}) is False

    # _pct / _rs
    def test_pct_positive(self):
        assert self.m._pct(5.5) == "+5.5%"

    def test_pct_negative(self):
        assert self.m._pct(-2.3) == "-2.3%"

    def test_rs_positive(self):
        assert self.m._rs(1.03) == "+1.03R"

    def test_rs_negative(self):
        assert self.m._rs(-0.84) == "-0.84R"

    # _wr_str / _wr_pct
    def test_wr_str_empty(self):
        assert self.m._wr_str([]) == "—"

    def test_wr_str_all_wins(self):
        assert self.m._wr_str([{"r": 1.0}, {"r": 0.5}]) == "100%"

    def test_wr_str_half(self):
        assert self.m._wr_str([{"r": 1.0}, {"r": -1.0}]) == "50%"

    def test_wr_pct_none_empty(self):
        assert self.m._wr_pct([]) is None

    def test_wr_pct_all_wins(self):
        assert self.m._wr_pct([{"r": 1.0}, {"r": 0.5}]) == pytest.approx(100.0)

    # _avg_r_val / _worst3l
    def test_avg_r_empty(self):
        assert self.m._avg_r_val([]) == 0.0

    def test_avg_r_value(self):
        ts = [{"r": 1.0}, {"r": -0.5}]
        assert self.m._avg_r_val(ts) == pytest.approx(0.25)

    def test_worst3l_fewer_than_3(self):
        ts = [{"r": -1.0}, {"r": -0.5}]
        assert self.m._worst3l(ts) == pytest.approx(-1.5)

    def test_worst3l_many_trades(self):
        ts = [{"r": -3.0}, {"r": -2.0}, {"r": -1.0}, {"r": 1.0}, {"r": 2.0}]
        assert self.m._worst3l(ts) == pytest.approx(-6.0)

    # _mae / _mfe
    def test_mae_from_mae_r(self):
        assert self.m._mae({"mae_r": -1.5}) == pytest.approx(-1.5)

    def test_mfe_from_mfe_r(self):
        assert self.m._mfe({"mfe_r": 2.0}) == pytest.approx(2.0)

    def test_mae_none_missing(self):
        assert self.m._mae({}) is None

    def test_mfe_none_missing(self):
        assert self.m._mfe({}) is None

    # _signal_type / _pattern
    def test_signal_type_from_signal_type_key(self):
        assert self.m._signal_type({"signal_type": "bearish_engulfing_4h"}) == "bearish_engulfing_4h"

    def test_signal_type_fallback_trigger_type(self):
        assert self.m._signal_type({"trigger_type": "shooting_star_strict"}) == "shooting_star_strict"

    def test_signal_type_unknown_fallback(self):
        assert self.m._signal_type({}) == "unknown"

    def test_pattern_from_pattern_key(self):
        assert self.m._pattern({"pattern": "head_and_shoulders"}) == "head_and_shoulders"

    def test_pattern_fallback_pattern_type(self):
        assert self.m._pattern({"pattern_type": "double_top"}) == "double_top"

    def test_pattern_unknown_fallback(self):
        assert self.m._pattern({}) == "unknown"

    # _is_strict_pin / _is_engulf
    def test_is_strict_pin_shooting_star(self):
        assert self.m._is_strict_pin({"signal_type": "shooting_star_strict_4h"}) is True

    def test_is_strict_pin_hammer(self):
        assert self.m._is_strict_pin({"signal_type": "hammer_strict"}) is True

    def test_is_strict_pin_false_for_engulf(self):
        assert self.m._is_strict_pin({"signal_type": "bearish_engulfing_4h"}) is False

    def test_is_engulf_bearish(self):
        assert self.m._is_engulf({"signal_type": "bearish_engulfing_4h"}) is True

    def test_is_engulf_bullish(self):
        assert self.m._is_engulf({"signal_type": "bullish_engulfing"}) is True

    def test_is_engulf_false_for_pin(self):
        assert self.m._is_engulf({"signal_type": "shooting_star_strict_4h"}) is False

    def test_strict_pin_and_engulf_mutually_exclusive(self):
        """A real trade should not be both strict-pin and engulf."""
        t1 = {"signal_type": "shooting_star_strict_4h"}
        t2 = {"signal_type": "bearish_engulfing_4h"}
        assert not (self.m._is_strict_pin(t1) and self.m._is_engulf(t1))
        assert not (self.m._is_strict_pin(t2) and self.m._is_engulf(t2))


# ──────────────────────────────────────────────────────────────────────────────
# 4. WindowComparison
# ──────────────────────────────────────────────────────────────────────────────

class _FakeResult:
    def __init__(self, trades, ret=0.0, dd=0.0, wr=0.0, avg_r=0.0):
        self.trades     = trades
        self.n_trades   = len(trades)
        self.return_pct = ret
        self.max_dd_pct = dd
        self.win_rate   = wr
        self.avg_r      = avg_r
        self.candle_data = None


def _t(pair="GBP/JPY", ts="2026-02-10T08:00:00+00:00", r=1.0,
        pattern="head_and_shoulders", signal="bearish_engulfing_4h",
        mae=-0.5, mfe=1.5):
    return {
        "pair":        pair,
        "entry_ts":    ts,
        "r":           r,
        "pattern":     pattern,
        "signal_type": signal,
        "direction":   "short",
        "mae_r":       mae,
        "mfe_r":       mfe,
    }


class TestWindowComparison:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.extended_variant_b as m
        self.WC = m.WindowComparison
        self.m  = m

    def _wc(self, trades_a, trades_b, ret_a=0.0, ret_b=0.0, dd_a=0.0, dd_b=0.0,
             label="W1", is_sub=False):
        ra = _FakeResult(trades_a, ret=ret_a, dd=dd_a)
        rb = _FakeResult(trades_b, ret=ret_b, dd=dd_b)
        return self.WC(
            label,
            datetime(2026, 2, 1,  tzinfo=timezone.utc),
            datetime(2026, 2, 14, tzinfo=timezone.utc),
            ra, rb, is_sub,
        )

    # Unlock algebra
    def test_no_unlock_same_trades(self):
        t = _t()
        wc = self._wc([t], [t])
        assert wc.unlocked_b == []
        assert wc.locked_by_b == []

    def test_one_unlocked_different_ts(self):
        t_a  = _t(ts="2026-02-05T08:00:00+00:00")
        t_b1 = _t(ts="2026-02-05T08:00:00+00:00")
        t_b2 = _t(ts="2026-02-10T08:00:00+00:00")
        wc = self._wc([t_a], [t_b1, t_b2])
        assert len(wc.unlocked_b)  == 1
        assert len(wc.locked_by_b) == 0

    def test_locked_by_b(self):
        t1 = _t(ts="2026-02-05T08:00:00+00:00")
        t2 = _t(ts="2026-02-07T08:00:00+00:00")
        wc = self._wc([t1, t2], [t1])
        assert len(wc.locked_by_b) == 1

    # Properties
    def test_n_a_n_b(self):
        wc = self._wc([_t()], [_t(), _t(ts="2026-02-11T08:00:00+00:00")])
        assert wc.n_a == 1
        assert wc.n_b == 2

    def test_wr_a_wr_b_from_win_rate(self):
        ra = _FakeResult([], wr=0.67)
        rb = _FakeResult([], wr=0.50)
        wc = self.WC("T", datetime(2026,2,1,tzinfo=timezone.utc),
                     datetime(2026,2,14,tzinfo=timezone.utc), ra, rb)
        assert wc.wr_a == 67
        assert wc.wr_b == 50

    # Trigger breakdown
    def test_strict_pin_trades_identified(self):
        t1 = _t(signal="shooting_star_strict_4h")
        t2 = _t(ts="2026-02-11T08:00:00+00:00", signal="bearish_engulfing_4h")
        wc = self._wc([], [t1, t2])
        sp = wc.strict_pin_trades()
        eg = wc.engulf_trades()
        assert len(sp) == 1
        assert len(eg) == 1

    def test_strict_pin_trades_empty_when_none(self):
        t = _t(signal="bearish_engulfing_4h")
        wc = self._wc([], [t])
        assert wc.strict_pin_trades() == []

    def test_trigger_table_groups_by_signal(self):
        t1 = _t(signal="shooting_star_strict_4h")
        t2 = _t(ts="2026-02-11T08:00:00+00:00", signal="bearish_engulfing_4h")
        t3 = _t(ts="2026-02-12T08:00:00+00:00", signal="bearish_engulfing_4h")
        wc = self._wc([], [t1, t2, t3])
        tbl = wc.trigger_table()
        assert tbl["bearish_engulfing_4h"] and len(tbl["bearish_engulfing_4h"]) == 2
        assert tbl["shooting_star_strict_4h"] and len(tbl["shooting_star_strict_4h"]) == 1

    # MAE/MFE
    def test_unlock_mae_mfe_computed(self):
        t = {**_t(ts="2026-02-10T08:00:00+00:00"), "mae_r": -1.0, "mfe_r": 2.0}
        wc = self._wc([], [t])
        mae, mfe = wc.unlock_mae_mfe()
        assert mae == "-1.00R"
        assert mfe == "+2.00R"

    def test_unlock_mae_mfe_none_when_missing(self):
        t = {k: v for k, v in _t().items() if k not in ("mae_r", "mfe_r")}
        wc = self._wc([], [t])
        mae, mfe = wc.unlock_mae_mfe()
        assert mae is None
        assert mfe is None

    # Pair/pattern concentration
    def test_pair_concentration(self):
        ts = [_t(pair="USD/JPY", ts=f"2026-02-0{i}T08:00:00+00:00") for i in range(1, 4)]
        wc = self._wc([], ts)
        conc = wc.pair_concentration()
        assert conc.get("USD/JPY") == 3

    def test_pattern_concentration(self):
        t1 = _t(ts="2026-02-08T08:00:00+00:00", pattern="head_and_shoulders")
        t2 = _t(ts="2026-02-09T08:00:00+00:00", pattern="double_top")
        wc = self._wc([], [t1, t2])
        conc = wc.pattern_concentration()
        assert "head_and_shoulders" in conc
        assert "double_top" in conc

    def test_is_subperiod_flag(self):
        wc = self._wc([], [], is_sub=True)
        assert wc.is_subperiod is True

    def test_all_b_trades_contains_shared_and_unlocked(self):
        shared  = _t(ts="2026-02-05T08:00:00+00:00")
        unlocked = _t(ts="2026-02-10T08:00:00+00:00")
        wc = self._wc([shared], [shared, unlocked])
        assert len(wc.all_b_trades) == 2


# ──────────────────────────────────────────────────────────────────────────────
# 5. Promotion gate
# ──────────────────────────────────────────────────────────────────────────────

class TestPromotionGate:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.extended_variant_b as m
        self.m   = m
        self.WC  = m.WindowComparison
        self.chk = m._promotion_check

    def _wc(self, label, trades_a, trades_b, ret_a=0.0, ret_b=0.0, dd_a=0.0, dd_b=0.0):
        ra = _FakeResult(trades_a, ret=ret_a, dd=dd_a)
        rb = _FakeResult(trades_b, ret=ret_b, dd=dd_b)
        return self.WC(label,
                       datetime(2025, 1, 1, tzinfo=timezone.utc),
                       datetime(2026, 3, 1, tzinfo=timezone.utc),
                       ra, rb)

    # Criterion 1 — full-year return
    def test_crit1_pass_b_equal_a(self):
        wcs = [self._wc("Full-year", [], [], ret_a=5.0, ret_b=5.0)]
        _, notes = self.chk(wcs)
        assert any("Criterion 1" in n and "PASS" in n for n in notes)

    def test_crit1_pass_b_within_tolerance(self):
        wcs = [self._wc("Full-year", [], [], ret_a=5.0, ret_b=4.6)]
        _, notes = self.chk(wcs)
        assert any("Criterion 1" in n and "PASS" in n for n in notes)

    def test_crit1_fail_b_below_tolerance(self):
        wcs = [self._wc("Full-year", [], [], ret_a=5.0, ret_b=4.0)]
        promote, notes = self.chk(wcs)
        assert any("Criterion 1" in n and "FAIL" in n for n in notes)
        assert promote is False

    # Criterion 2 — full-year maxDD
    def test_crit2_pass_dd_unchanged(self):
        wcs = [self._wc("Full-year", [], [], dd_a=3.0, dd_b=3.0)]
        _, notes = self.chk(wcs)
        assert any("Criterion 2" in n and "PASS" in n for n in notes)

    def test_crit2_pass_within_1p5(self):
        wcs = [self._wc("Full-year", [], [], dd_a=2.0, dd_b=3.4)]
        _, notes = self.chk(wcs)
        assert any("Criterion 2" in n and "PASS" in n for n in notes)

    def test_crit2_fail_over_limit(self):
        wcs = [self._wc("Full-year", [], [], dd_a=2.0, dd_b=4.0)]
        promote, notes = self.chk(wcs)
        assert any("Criterion 2" in n and "FAIL" in n for n in notes)
        assert promote is False

    # Criterion 3 — concentration (small-n bypass)
    def test_crit3_small_n_not_block(self):
        """< 5 unlocked trades → criterion 3 should be N/A, not FAIL."""
        trades_b = [_t(ts=f"2026-02-0{i}T08:00:00+00:00") for i in range(1, 4)]
        wcs = [self._wc("Full-year", [], trades_b, ret_a=0.0, ret_b=3.0)]
        promote, notes = self.chk(wcs)
        # With <5 unlocks criterion 3 is N/A — should not block promotion
        crit3 = [n for n in notes if "Criterion 3" in n]
        assert crit3
        assert not any("FAIL" in n for n in crit3)

    def test_crit3_large_n_concentrated_fails(self):
        """≥5 unlocks all from same pair → FAIL."""
        trades_b = [_t(pair="USD/JPY", ts=f"2025-0{m+1}-0{d+1}T08:00:00+00:00")
                    for m in range(1) for d in range(5)]
        wcs = [self._wc("Full-year", [], trades_b, ret_a=0.0, ret_b=5.0)]
        promote, notes = self.chk(wcs)
        assert any("Criterion 3" in n and "FAIL" in n for n in notes)

    def test_crit3_large_n_spread_passes(self):
        """≥5 unlocks spread across pairs and patterns → PASS."""
        pairs    = ["USD/JPY", "GBP/JPY", "USD/CHF", "GBP/CHF", "EUR/USD", "GBP/USD"]
        patterns = ["head_and_shoulders", "double_top", "double_bottom",
                    "inv_head_and_shoulders", "break_retest", "head_and_shoulders"]
        trades_b = [
            _t(pair=p, ts=f"2025-0{i+1}-01T08:00:00+00:00", pattern=pt)
            for i, (p, pt) in enumerate(zip(pairs, patterns))
        ]
        wcs = [self._wc("Full-year", [], trades_b, ret_a=0.0, ret_b=6.0)]
        _, notes = self.chk(wcs)
        crit3 = [n for n in notes if "Criterion 3" in n]
        assert any("PASS" in n for n in crit3)

    # Criterion 4 — strict-pin WR
    def test_crit4_no_strict_pin_trades_na(self):
        """No strict-pin trades → criterion N/A (not blocked)."""
        wcs = [self._wc("Full-year", [], [])]
        _, notes = self.chk(wcs)
        assert any("Criterion 4" in n for n in notes)

    def test_crit4_pass_wr_above_45(self):
        sp_trades = [
            _t(ts=f"2025-0{m+1}-01T08:00:00+00:00",
               signal="shooting_star_strict_4h",
               r=(1.0 if m < 3 else -1.0))
            for m in range(5)
        ]
        wcs = [self._wc("Full-year", [], sp_trades, ret_a=0.0, ret_b=3.0)]
        _, notes = self.chk(wcs)
        crit4 = [n for n in notes if "Criterion 4" in n]
        # 3W/2L = 60% WR ≥ 45% → PASS
        assert any("PASS" in n for n in crit4)

    def test_crit4_fail_wr_below_45(self):
        sp_trades = [
            _t(ts=f"2025-0{m+1}-01T08:00:00+00:00",
               signal="shooting_star_strict_4h",
               r=(-1.0 if m < 4 else 1.0))
            for m in range(5)
        ]
        wcs = [self._wc("Full-year", [], sp_trades, ret_a=0.0, ret_b=0.0, dd_a=0.0, dd_b=0.0)]
        promote, notes = self.chk(wcs)
        crit4 = [n for n in notes if "Criterion 4" in n]
        # 1W/4L = 20% WR < 45% → FAIL
        assert any("FAIL" in n for n in crit4)

    # All pass → promoted
    def test_all_pass_returns_promote(self):
        wcs = [self._wc("Full-year", [], [], ret_a=5.0, ret_b=6.0,
                        dd_a=2.0, dd_b=2.0)]
        promote, _ = self.chk(wcs)
        assert promote is True

    # Any fail → not promoted
    def test_one_fail_blocks_promotion(self):
        wcs = [self._wc("Full-year", [], [], ret_a=10.0, ret_b=2.0)]
        promote, _ = self.chk(wcs)
        assert promote is False


# ──────────────────────────────────────────────────────────────────────────────
# 6. Script structure
# ──────────────────────────────────────────────────────────────────────────────

class TestScriptStructure:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.extended_variant_b as m
        self.m = m

    def test_report_path_attribute(self):
        assert hasattr(self.m, "REPORT_PATH")

    def test_report_path_under_backtesting_results(self):
        p = self.m.REPORT_PATH
        assert "backtesting/results" in str(p)
        assert p.name == "extended_variant_b.md"

    def test_capital_8000(self):
        assert self.m.CAPITAL == 8_000.0

    def test_main_callable(self):
        assert callable(self.m.main)

    def test_build_report_callable(self):
        assert callable(self.m.build_report)

    def test_run_variant_callable(self):
        assert callable(self.m.run_variant)

    def test_promotion_check_callable(self):
        assert callable(self.m._promotion_check)

    def test_window_comparison_class_exists(self):
        assert hasattr(self.m, "WindowComparison")

    def test_atexit_guard_in_source(self):
        src = (REPO / "scripts/extended_variant_b.py").read_text()
        assert "atexit.register" in src

    def test_reset_config_restores_production_values(self):
        _sc.ENTRY_TRIGGER_MODE           = "engulf_or_strict_pin_at_level"
        _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 99
        self.m._reset_config()
        assert _sc.ENTRY_TRIGGER_MODE           == "engulf_only"
        assert _sc.ENGULF_CONFIRM_LOOKBACK_BARS == 2

    def test_no_live_trading_calls_in_source(self):
        src = (REPO / "scripts/extended_variant_b.py").read_text()
        for forbidden in ("submit_order", "place_order", "create_order", "dry_run=False"):
            assert forbidden not in src, f"Found forbidden call: {forbidden}"

    def test_full_year_label_matches_window(self):
        """FULL_YEAR_LABEL must match the label field of the first WINDOWS entry."""
        assert self.m.FULL_YEAR_LABEL == self.m.WINDOWS[0][0]
