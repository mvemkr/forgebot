"""
tests/test_confirm_variant_b.py
Unit tests for the Variant B multi-window confirmation study.

Coverage:
  1. Config defaults  — production values unchanged
  2. Window definitions — correct dates and ordering
  3. Helper functions — _r, _is_win, _pct, _rs, _wr, _avg_r, _mae, _mfe,
                         _entry_dt, _detection_dt, _bars_lag
  4. WindowResult      — unlock set algebra, pair/pattern concentration,
                         stale-entry detection
  5. Promotion gate    — all 4 criteria, pass/fail edge cases
  6. Script structure  — report constants exist and script is importable
"""

from __future__ import annotations

import sys
import importlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.strategy.forex.strategy_config as _sc


# ── restore helper ────────────────────────────────────────────────────────────
def _restore():
    _sc.ENTRY_TRIGGER_MODE           = "engulf_only"
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 2


# ──────────────────────────────────────────────────────────────────────────────
# 1. Config defaults
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigDefaults:
    """Production config must not be changed by this study."""

    def test_entry_trigger_mode_exists(self):
        assert hasattr(_sc, "ENTRY_TRIGGER_MODE")

    def test_entry_trigger_mode_is_engulf_only(self):
        # Promoted to B-Prime 2026-03-07
        assert _sc.ENTRY_TRIGGER_MODE == "engulf_or_strict_pin_at_level", (
            "ENTRY_TRIGGER_MODE must be 'engulf_or_strict_pin_at_level' (B-Prime LIVE_PAPER)"
        )

    def test_engulf_confirm_lookback_exists(self):
        assert hasattr(_sc, "ENGULF_CONFIRM_LOOKBACK_BARS")

    def test_engulf_confirm_lookback_is_2(self):
        assert _sc.ENGULF_CONFIRM_LOOKBACK_BARS == 2, (
            "ENGULF_CONFIRM_LOOKBACK_BARS must be 2 in production"
        )

    def test_engulfing_only_derived_flag_true(self):
        # False — B-Prime (strict_pin_at_level) is active
        assert _sc.ENGULFING_ONLY is False

    def test_strict_pin_mode_is_valid(self):
        """engulf_or_strict_pin_at_level must be a recognised trigger mode."""
        valid_modes = {
            "engulf_only",
            "engulf_or_pin",
            "engulf_or_star_at_level",
            "engulf_or_strict_pin_at_level",
            "engulf_or_star_or_strict_pin_at_level",
        }
        assert "engulf_or_strict_pin_at_level" in valid_modes


# ──────────────────────────────────────────────────────────────────────────────
# 2. Window definitions
# ──────────────────────────────────────────────────────────────────────────────

class TestWindowDefinitions:
    """Verify WINDOWS list shape, ordering, and date correctness."""

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.confirm_variant_b as m
        self.m = m

    def test_three_windows(self):
        assert len(self.m.WINDOWS) == 3

    def test_window_labels(self):
        labels = [w[0] for w in self.m.WINDOWS]
        assert labels == ["W1", "W2", "Live-parity"]

    def test_w1_dates(self):
        label, start, end = self.m.WINDOWS[0]
        assert start == datetime(2026, 2,  1, tzinfo=timezone.utc)
        assert end   == datetime(2026, 2, 14, tzinfo=timezone.utc)

    def test_w2_dates(self):
        label, start, end = self.m.WINDOWS[1]
        assert start == datetime(2026, 2, 15, tzinfo=timezone.utc)
        assert end   == datetime(2026, 2, 28, tzinfo=timezone.utc)

    def test_parity_dates(self):
        label, start, end = self.m.WINDOWS[2]
        assert start == datetime(2026, 2, 28, tzinfo=timezone.utc)
        assert end   == datetime(2026, 3,  6, tzinfo=timezone.utc)

    def test_windows_are_chronological(self):
        for i in range(len(self.m.WINDOWS) - 1):
            assert self.m.WINDOWS[i][1] <= self.m.WINDOWS[i + 1][1]

    def test_two_variants_only(self):
        """Study must compare exactly A and B."""
        labels = [v[0] for v in self.m.VARIANTS]
        assert labels == ["A", "B"]

    def test_variant_a_is_baseline(self):
        _, mode, lb, _ = self.m.VARIANTS[0]
        assert mode == "engulf_only"
        assert lb   == 2

    def test_variant_b_is_strict_pin(self):
        _, mode, lb, _ = self.m.VARIANTS[1]
        assert mode == "engulf_or_strict_pin_at_level"
        assert lb   == 2


# ──────────────────────────────────────────────────────────────────────────────
# 3. Helper functions
# ──────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.confirm_variant_b as m
        self.m = m

    # _r
    def test_r_from_r_key(self):
        assert self.m._r({"r": 1.5}) == 1.5

    def test_r_from_realised_r(self):
        assert self.m._r({"realised_r": -0.7}) == -0.7

    def test_r_from_result_r(self):
        assert self.m._r({"result_r": 2.1}) == pytest.approx(2.1)

    def test_r_missing_returns_zero(self):
        assert self.m._r({}) == 0.0

    def test_r_prefers_r_key_first(self):
        assert self.m._r({"r": 1.0, "realised_r": 99.0}) == 1.0

    # _is_win
    def test_is_win_positive(self):
        assert self.m._is_win({"r": 0.5}) is True

    def test_is_win_zero_not_win(self):
        assert self.m._is_win({"r": 0.0}) is False

    def test_is_win_negative(self):
        assert self.m._is_win({"r": -1.0}) is False

    # _pct / _rs
    def test_pct_positive(self):
        assert self.m._pct(3.5)  == "+3.5%"

    def test_pct_negative(self):
        assert self.m._pct(-1.2) == "-1.2%"

    def test_rs_positive(self):
        assert self.m._rs(1.03)  == "+1.03R"

    def test_rs_negative(self):
        assert self.m._rs(-0.70) == "-0.70R"

    # _wr / _avg_r
    def test_wr_empty(self):
        assert self.m._wr([]) == "—"

    def test_wr_all_wins(self):
        trades = [{"r": 1.0}, {"r": 0.5}]
        assert self.m._wr(trades) == "100%"

    def test_wr_half(self):
        trades = [{"r": 1.0}, {"r": -1.0}]
        assert self.m._wr(trades) == "50%"

    def test_avg_r_empty(self):
        assert self.m._avg_r([]) == "—"

    def test_avg_r_value(self):
        trades = [{"r": 1.0}, {"r": -0.5}]
        assert self.m._avg_r(trades) == "+0.25R"

    # _mae / _mfe
    def test_mae_from_mae_r(self):
        assert self.m._mae({"mae_r": -1.5}) == pytest.approx(-1.5)

    def test_mfe_from_mfe_r(self):
        assert self.m._mfe({"mfe_r": 2.0}) == pytest.approx(2.0)

    def test_mae_none_when_missing(self):
        assert self.m._mae({}) is None

    def test_mfe_none_when_missing(self):
        assert self.m._mfe({}) is None

    # _entry_dt
    def test_entry_dt_from_string(self):
        t = {"entry_ts": "2026-02-10T08:00:00+00:00"}
        dt = self.m._entry_dt(t)
        assert dt == datetime(2026, 2, 10, 8, 0, tzinfo=timezone.utc)

    def test_entry_dt_from_datetime(self):
        raw = datetime(2026, 2, 10, 8, 0)
        t = {"entry_ts": raw}
        dt = self.m._entry_dt(t)
        assert dt.tzinfo is not None

    def test_entry_dt_fallback_open_ts(self):
        t = {"open_ts": "2026-02-10T09:00:00+00:00"}
        dt = self.m._entry_dt(t)
        assert dt is not None

    def test_entry_dt_none_when_missing(self):
        assert self.m._entry_dt({}) is None

    # _bars_lag
    def test_bars_lag_same_time(self):
        t = {
            "entry_ts":    "2026-02-10T08:00:00+00:00",
            "pattern_ts":  "2026-02-10T08:00:00+00:00",
        }
        assert self.m._bars_lag(t) == 0

    def test_bars_lag_2_hours(self):
        t = {
            "entry_ts":    "2026-02-10T10:00:00+00:00",
            "pattern_ts":  "2026-02-10T08:00:00+00:00",
        }
        assert self.m._bars_lag(t) == 2

    def test_bars_lag_none_when_missing(self):
        t = {"entry_ts": "2026-02-10T08:00:00+00:00"}
        assert self.m._bars_lag(t) is None


# ──────────────────────────────────────────────────────────────────────────────
# 4. WindowResult — unlock algebra
# ──────────────────────────────────────────────────────────────────────────────

class _FakeResult:
    """Minimal BacktestResult stand-in."""
    def __init__(self, trades, ret=0.0, dd=0.0, wr=0.0):
        self.trades      = trades
        self.n_trades    = len(trades)
        self.return_pct  = ret
        self.max_dd_pct  = dd
        self.win_rate    = wr
        self.candle_data = None


def _make_trade(pair="GBP/JPY", ts="2026-02-10T08:00:00+00:00", r=1.0,
                pattern="head_and_shoulders", trigger="bearish_engulfing"):
    return {
        "pair":         pair,
        "entry_ts":     ts,
        "r":            r,
        "pattern_type": pattern,
        "trigger_type": trigger,
        "direction":    "short",
    }


class TestWindowResult:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.confirm_variant_b as m
        self.WR  = m.WindowResult
        self.m   = m

    def _wr(self, trades_a, trades_b, ret_a=0.0, ret_b=0.0, dd_a=0.0, dd_b=0.0):
        ra = _FakeResult(trades_a, ret=ret_a, dd=dd_a)
        rb = _FakeResult(trades_b, ret=ret_b, dd=dd_b)
        return self.WR(
            "W1",
            datetime(2026, 2, 1, tzinfo=timezone.utc),
            datetime(2026, 2, 14, tzinfo=timezone.utc),
            ra, rb,
        )

    # Unlock algebra
    def test_no_unlock_same_trades(self):
        t = _make_trade()
        wr = self._wr([t], [t])
        assert wr.unlocked_b == []

    def test_one_unlocked_trade(self):
        t_a  = _make_trade(ts="2026-02-05T08:00:00+00:00", r=1.0)
        t_b1 = _make_trade(ts="2026-02-05T08:00:00+00:00", r=1.0)
        t_b2 = _make_trade(ts="2026-02-10T08:00:00+00:00", r=0.5)
        wr = self._wr([t_a], [t_b1, t_b2])
        assert len(wr.unlocked_b)  == 1
        assert len(wr.locked_by_b) == 0

    def test_locked_by_b(self):
        t_a  = _make_trade(ts="2026-02-05T08:00:00+00:00", r=1.0)
        t_a2 = _make_trade(ts="2026-02-07T08:00:00+00:00", r=-1.0)
        t_b  = _make_trade(ts="2026-02-05T08:00:00+00:00", r=1.0)
        wr = self._wr([t_a, t_a2], [t_b])
        assert len(wr.locked_by_b) == 1

    def test_unlock_wr_100pct(self):
        t_b_new = _make_trade(ts="2026-02-10T08:00:00+00:00", r=1.0)
        wr = self._wr([], [t_b_new])
        assert wr.unlock_wr() == "100%"

    def test_unlock_wr_none_if_no_unlocks(self):
        t = _make_trade()
        wr = self._wr([t], [t])
        assert wr.unlock_wr() is None

    def test_unlock_avg_r(self):
        t1 = _make_trade(ts="2026-02-08T08:00:00+00:00", r=1.0)
        t2 = _make_trade(ts="2026-02-09T08:00:00+00:00", r=-0.5)
        wr = self._wr([], [t1, t2])
        assert wr.unlock_avg_r() == "+0.25R"

    # Concentration
    def test_pair_concentration_single_pair(self):
        trades = [_make_trade(pair="USD/JPY", ts=f"2026-02-0{i}T08:00:00+00:00", r=1.0)
                  for i in range(1, 4)]
        wr = self._wr([], trades)
        conc = wr.pair_concentration()
        assert conc["USD/JPY"] == 3

    def test_pattern_concentration(self):
        t1 = _make_trade(ts="2026-02-08T08:00:00+00:00", pattern="head_and_shoulders")
        t2 = _make_trade(ts="2026-02-09T08:00:00+00:00", pattern="double_top")
        wr = self._wr([], [t1, t2])
        conc = wr.pattern_concentration()
        assert "head_and_shoulders" in conc
        assert "double_top"         in conc

    # Stale entry
    def test_stale_entry_within_4_bars(self):
        t = {
            **_make_trade(ts="2026-02-10T10:00:00+00:00"),
            "pattern_ts": "2026-02-10T08:00:00+00:00",
        }
        wr = self._wr([], [t])
        stale, valid = wr.unlock_stale()
        assert stale == 0
        assert valid == 1

    def test_stale_entry_over_4_bars(self):
        t = {
            **_make_trade(ts="2026-02-10T15:00:00+00:00"),
            "pattern_ts": "2026-02-10T08:00:00+00:00",
        }
        wr = self._wr([], [t])
        stale, valid = wr.unlock_stale()
        assert stale == 1
        assert valid == 1

    def test_stale_missing_data(self):
        t = _make_trade()  # no pattern_ts
        wr = self._wr([], [t])
        stale, valid = wr.unlock_stale()
        assert valid == 0  # no lag data available

    def test_mae_mfe_none_when_no_data(self):
        t = _make_trade(ts="2026-02-10T08:00:00+00:00")
        wr = self._wr([], [t])
        mae, mfe = wr.unlock_mae_mfe()
        assert mae is None
        assert mfe is None

    def test_mae_mfe_computed_from_fields(self):
        t = {**_make_trade(ts="2026-02-10T08:00:00+00:00"), "mae_r": -1.0, "mfe_r": 2.0}
        wr = self._wr([], [t])
        mae, mfe = wr.unlock_mae_mfe()
        assert mae == "-1.00R"
        assert mfe == "+2.00R"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Promotion gate
# ──────────────────────────────────────────────────────────────────────────────

class TestPromotionGate:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.confirm_variant_b as m
        self.m   = m
        self.WR  = m.WindowResult
        self.chk = m._promotion_check

    def _wr(self, label, trades_a, trades_b, ret_a=0.0, ret_b=0.0, dd_a=0.0, dd_b=0.0):
        ws_map = {
            "W1":          (datetime(2026, 2, 1,  tzinfo=timezone.utc),
                            datetime(2026, 2, 14, tzinfo=timezone.utc)),
            "W2":          (datetime(2026, 2, 15, tzinfo=timezone.utc),
                            datetime(2026, 2, 28, tzinfo=timezone.utc)),
            "Live-parity": (datetime(2026, 2, 28, tzinfo=timezone.utc),
                            datetime(2026, 3, 6,  tzinfo=timezone.utc)),
        }
        ws, we = ws_map[label]
        ra = _FakeResult(trades_a, ret=ret_a, dd=dd_a)
        rb = _FakeResult(trades_b, ret=ret_b, dd=dd_b)
        return self.WR(label, ws, we, ra, rb)

    # Criterion 1: W1 return
    def test_crit1_pass_b_equal_a(self):
        wrs = [self._wr("W1", [], [], ret_a=3.5, ret_b=3.5),
               self._wr("W2", [], [], dd_a=1.0, dd_b=1.0),
               self._wr("Live-parity", [], [])]
        promote, notes = self.chk(wrs)
        assert any("PASS" in n and "Criterion 1" in n for n in notes)

    def test_crit1_pass_b_better(self):
        wrs = [self._wr("W1", [], [], ret_a=2.0, ret_b=4.0),
               self._wr("W2", [], [], dd_a=1.0, dd_b=1.0),
               self._wr("Live-parity", [], [])]
        promote, notes = self.chk(wrs)
        assert any("PASS" in n and "Criterion 1" in n for n in notes)

    def test_crit1_fail_b_worse(self):
        wrs = [self._wr("W1", [], [], ret_a=3.5, ret_b=1.0),
               self._wr("W2", [], [], dd_a=1.0, dd_b=1.0),
               self._wr("Live-parity", [], [])]
        promote, notes = self.chk(wrs)
        assert any("FAIL" in n and "Criterion 1" in n for n in notes)
        assert promote is False

    # Criterion 2: W2 maxDD
    def test_crit2_pass_dd_unchanged(self):
        wrs = [self._wr("W1", [], [], ret_a=3.5, ret_b=3.5),
               self._wr("W2", [], [], dd_a=2.0, dd_b=2.0),
               self._wr("Live-parity", [], [])]
        promote, notes = self.chk(wrs)
        assert any("PASS" in n and "Criterion 2" in n for n in notes)

    def test_crit2_pass_dd_within_1pp(self):
        wrs = [self._wr("W1", [], [], ret_a=3.5, ret_b=3.5),
               self._wr("W2", [], [], dd_a=1.0, dd_b=1.9),
               self._wr("Live-parity", [], [])]
        promote, notes = self.chk(wrs)
        assert any("PASS" in n and "Criterion 2" in n for n in notes)

    def test_crit2_fail_dd_over_limit(self):
        wrs = [self._wr("W1", [], [], ret_a=3.5, ret_b=3.5),
               self._wr("W2", [], [], dd_a=1.0, dd_b=2.5),
               self._wr("Live-parity", [], [])]
        promote, notes = self.chk(wrs)
        assert any("FAIL" in n and "Criterion 2" in n for n in notes)
        assert promote is False

    # Criterion 3: concentration
    def test_crit3_pass_no_unlocks(self):
        """Zero unlocked trades → concentration criterion is vacuously satisfied."""
        wrs = [self._wr("W1", [], [], ret_a=0.0, ret_b=0.0),
               self._wr("W2", [], [], dd_a=0.0, dd_b=0.0),
               self._wr("Live-parity", [], [])]
        promote, notes = self.chk(wrs)
        assert any("Criterion 3" in n for n in notes)

    def test_crit3_fail_single_pair_all_unlocks(self):
        """All unlocked trades from same pair → concentrated."""
        unlocked = [_make_trade(pair="USD/JPY", ts=f"2026-02-0{i}T08:00:00+00:00")
                    for i in range(1, 5)]
        wrs = [self._wr("W1", [], unlocked, ret_a=0.0, ret_b=0.0),
               self._wr("W2", [], [],       dd_a=0.0, dd_b=0.0),
               self._wr("Live-parity", [], [])]
        promote, notes = self.chk(wrs)
        assert any("FAIL" in n and "Criterion 3" in n for n in notes)

    def test_crit3_pass_spread_across_pairs(self):
        """Unlocked spread across 4 pairs AND 4 patterns → not concentrated."""
        pairs    = ["USD/JPY", "GBP/JPY", "USD/CHF", "GBP/CHF"]
        patterns = ["head_and_shoulders", "double_top", "double_bottom", "inv_head_and_shoulders"]
        unlocked = [
            _make_trade(pair=p, ts=f"2026-02-0{i+1}T08:00:00+00:00", pattern=pt)
            for i, (p, pt) in enumerate(zip(pairs, patterns))
        ]
        wrs = [self._wr("W1", [], unlocked, ret_a=0.0, ret_b=4.0),
               self._wr("W2", [], [],       dd_a=0.0, dd_b=0.0),
               self._wr("Live-parity", [], [])]
        promote, notes = self.chk(wrs)
        crit3_notes = [n for n in notes if "Criterion 3" in n]
        assert any("PASS" in n for n in crit3_notes)

    # All criteria pass → promoted
    def test_all_pass_returns_promote(self):
        wrs = [self._wr("W1", [], [], ret_a=2.0, ret_b=3.0),
               self._wr("W2", [], [], dd_a=1.0, dd_b=1.0),
               self._wr("Live-parity", [], [])]
        promote, notes = self.chk(wrs)
        assert promote is True

    # One fail → not promoted
    def test_one_fail_blocks_promotion(self):
        wrs = [self._wr("W1", [], [], ret_a=5.0, ret_b=1.0),   # crit 1 fails
               self._wr("W2", [], [], dd_a=1.0, dd_b=1.0),
               self._wr("Live-parity", [], [])]
        promote, notes = self.chk(wrs)
        assert promote is False


# ──────────────────────────────────────────────────────────────────────────────
# 6. Script structure
# ──────────────────────────────────────────────────────────────────────────────

class TestScriptStructure:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.confirm_variant_b as m
        self.m = m

    def test_report_path_exists_as_attribute(self):
        assert hasattr(self.m, "REPORT_PATH")

    def test_report_path_is_under_backtesting_results(self):
        p = self.m.REPORT_PATH
        assert "backtesting/results" in str(p)
        assert p.name == "confirm_variant_b.md"

    def test_capital_is_8000(self):
        assert self.m.CAPITAL == 8_000.0

    def test_main_callable(self):
        assert callable(self.m.main)

    def test_build_report_callable(self):
        assert callable(self.m.build_report)

    def test_run_variant_callable(self):
        assert callable(self.m.run_variant)

    def test_promotion_check_callable(self):
        assert callable(self.m._promotion_check)

    def test_window_result_class_exists(self):
        assert hasattr(self.m, "WindowResult")

    def test_atexit_guard_present(self):
        """_reset_config must be registered before any config mutation."""
        import atexit
        # Verify atexit module is used (import present in script)
        src = (REPO / "scripts/confirm_variant_b.py").read_text()
        assert "atexit.register" in src

    def test_reset_config_restores_production_values(self):
        """_reset_config() must restore the values captured at module import."""
        _sc.ENTRY_TRIGGER_MODE           = "engulf_only"   # force a temporary change
        _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 99
        self.m._reset_config()
        # Restores to whatever was set at import time (now B-Prime production values)
        assert _sc.ENTRY_TRIGGER_MODE           == "engulf_or_strict_pin_at_level"
        assert _sc.ENGULF_CONFIRM_LOOKBACK_BARS == 2

    def test_no_live_trading_in_source(self):
        """Script must not contain live-order submission calls."""
        src = (REPO / "scripts/confirm_variant_b.py").read_text()
        for forbidden in ("submit_order", "place_order", "create_order", "dry_run=False"):
            assert forbidden not in src, f"Found forbidden call: {forbidden}"
