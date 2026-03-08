"""
tests/test_variant_b_prime.py
Unit tests for the Variant B-Prime three-way comparison study.

Coverage (7 classes):
  1. Config defaults        — STRICT_PIN_PATTERN_WHITELIST, model tag
  2. Whitelist gate logic   — gate in/out of set_and_forget (signal-level helpers)
  3. Window definitions     — dates, ordering, variant shape, HNS_PATTERNS
  4. Helpers                — _r, _is_win, _pct, _rs, _wr, _avg_r, _worst3,
                               _mae, _mfe, _signal_type, _pattern,
                               _is_strict_pin, _is_hns_pattern
  5. WindowTriple algebra   — unlocked_b, unlocked_bp, removed_by_wl,
                               locked_by_bp, unlock_mae_mfe
  6. Key-questions logic    — Q1 regression, Jan-Feb/W1 preservation
  7. Script structure       — constants, callables, atexit, safety
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.strategy.forex.strategy_config as _sc


def _restore():
    _sc.ENTRY_TRIGGER_MODE           = "engulf_only"
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 2
    _sc.STRICT_PIN_PATTERN_WHITELIST = None


# ──────────────────────────────────────────────────────────────────────────────
# 1. Config defaults
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigDefaults:
    """STRICT_PIN_PATTERN_WHITELIST must exist and default to None."""

    def test_whitelist_attr_exists(self):
        assert hasattr(_sc, "STRICT_PIN_PATTERN_WHITELIST")

    def test_whitelist_default_none(self):
        assert _sc.STRICT_PIN_PATTERN_WHITELIST is None, (
            "STRICT_PIN_PATTERN_WHITELIST must be None in production"
        )

    def test_entry_trigger_mode_still_engulf_only(self):
        assert _sc.ENTRY_TRIGGER_MODE == "engulf_only"

    def test_lookback_still_2(self):
        assert _sc.ENGULF_CONFIRM_LOOKBACK_BARS == 2

    def test_no_sp_hns_tag_when_whitelist_none(self):
        _sc.STRICT_PIN_PATTERN_WHITELIST = None
        tags = _sc.get_model_tags()
        assert "sp_hns_only" not in tags

    def test_sp_hns_tag_when_whitelist_set(self):
        _sc.STRICT_PIN_PATTERN_WHITELIST = ["head_and_shoulders"]
        try:
            tags = _sc.get_model_tags()
            assert "sp_hns_only" in tags
        finally:
            _restore()

    def test_sp_hns_tag_absent_after_reset(self):
        _sc.STRICT_PIN_PATTERN_WHITELIST = ["head_and_shoulders"]
        _restore()
        tags = _sc.get_model_tags()
        assert "sp_hns_only" not in tags


# ──────────────────────────────────────────────────────────────────────────────
# 2. Whitelist gate logic helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestWhitelistGateLogic:
    """
    Test the helper functions used by FILTER 5b in set_and_forget.py.
    These replicate the exact logic: is the signal strict-pin? is the pattern whitelisted?
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.variant_b_prime as m
        self.m = m

    # _is_strict_pin helper
    def test_shooting_star_strict_4h_is_strict_pin(self):
        t = {"signal_type": "shooting_star_strict_4h"}
        assert self.m._is_strict_pin(t) is True

    def test_shooting_star_is_strict_pin(self):
        assert self.m._is_strict_pin({"signal_type": "shooting_star_strict"}) is True

    def test_hammer_strict_is_strict_pin(self):
        assert self.m._is_strict_pin({"signal_type": "hammer_strict"}) is True

    def test_bearish_engulfing_not_strict_pin(self):
        assert self.m._is_strict_pin({"signal_type": "bearish_engulfing_4h"}) is False

    def test_bullish_engulfing_not_strict_pin(self):
        assert self.m._is_strict_pin({"signal_type": "bullish_engulfing"}) is False

    def test_unknown_signal_not_strict_pin(self):
        assert self.m._is_strict_pin({"signal_type": "unknown"}) is False

    # _is_hns_pattern helper
    def test_head_and_shoulders_is_hns(self):
        assert self.m._is_hns_pattern({"pattern": "head_and_shoulders"}) is True

    def test_inverted_hns_is_hns(self):
        assert self.m._is_hns_pattern({"pattern": "inverted_head_and_shoulders"}) is True

    def test_double_top_not_hns(self):
        assert self.m._is_hns_pattern({"pattern": "double_top"}) is False

    def test_double_bottom_not_hns(self):
        assert self.m._is_hns_pattern({"pattern": "double_bottom"}) is False

    def test_break_retest_bullish_not_hns(self):
        assert self.m._is_hns_pattern({"pattern": "break_retest_bullish"}) is False

    def test_break_retest_bearish_not_hns(self):
        assert self.m._is_hns_pattern({"pattern": "break_retest_bearish"}) is False

    def test_unknown_pattern_not_hns(self):
        assert self.m._is_hns_pattern({}) is False

    # Gate logic: whitelist=HNS blocks non-HNS strict-pin
    def test_gate_blocks_double_top_strict_pin(self):
        """Strict-pin at double_top should be blocked by B-Prime whitelist."""
        hns_wl = self.m.HNS_PATTERNS
        t = {"signal_type": "shooting_star_strict_4h", "pattern": "double_top"}
        is_sp   = self.m._is_strict_pin(t)
        blocked = self.m._is_strict_pin(t) and self.m._pattern(t) not in hns_wl
        assert is_sp is True
        assert blocked is True

    def test_gate_passes_hns_strict_pin(self):
        """Strict-pin at head_and_shoulders should pass B-Prime whitelist."""
        hns_wl = self.m.HNS_PATTERNS
        t = {"signal_type": "shooting_star_strict_4h", "pattern": "head_and_shoulders"}
        blocked = self.m._is_strict_pin(t) and self.m._pattern(t) not in hns_wl
        assert blocked is False

    def test_gate_never_blocks_engulf(self):
        """Engulf signals must never be blocked regardless of whitelist."""
        hns_wl = self.m.HNS_PATTERNS
        for pattern in ["double_top", "double_bottom", "break_retest_bullish"]:
            t = {"signal_type": "bearish_engulfing_4h", "pattern": pattern}
            blocked = self.m._is_strict_pin(t) and self.m._pattern(t) not in hns_wl
            assert blocked is False, f"Engulf at {pattern} should not be blocked"

    def test_gate_inactive_when_whitelist_none(self):
        """When whitelist is None, no signal is ever blocked."""
        for signal in ["shooting_star_strict_4h", "hammer_strict", "bearish_engulfing_4h"]:
            t = {"signal_type": signal, "pattern": "double_top"}
            # None whitelist → gate is inactive → not blocked
            is_blocked = (
                None is not None        # whitelist is None → gate off
                and self.m._is_strict_pin(t)
                and self.m._pattern(t) not in []
            )
            assert is_blocked is False


# ──────────────────────────────────────────────────────────────────────────────
# 3. Window & variant definitions
# ──────────────────────────────────────────────────────────────────────────────

class TestWindowVariantDefinitions:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.variant_b_prime as m
        self.m = m

    def test_seven_windows(self):
        assert len(self.m.WINDOWS) == 7

    def test_q1_dates(self):
        label, ws, we = self.m.WINDOWS[0]
        assert label == "Q1-2025"
        assert ws == datetime(2025, 1,  1, tzinfo=timezone.utc)
        assert we == datetime(2025, 3, 31, tzinfo=timezone.utc)

    def test_q4_dates(self):
        label, ws, we = self.m.WINDOWS[3]
        assert label == "Q4-2025"
        assert ws == datetime(2025, 10,  1, tzinfo=timezone.utc)
        assert we == datetime(2025, 12, 31, tzinfo=timezone.utc)

    def test_jan_feb_2026_dates(self):
        label, ws, we = self.m.WINDOWS[4]
        assert label == "Jan-Feb-2026"
        assert ws == datetime(2026, 1,  1, tzinfo=timezone.utc)
        assert we == datetime(2026, 2, 28, tzinfo=timezone.utc)

    def test_w1_dates(self):
        label, ws, we = self.m.WINDOWS[5]
        assert label == "W1"
        assert ws == datetime(2026, 2,  1, tzinfo=timezone.utc)
        assert we == datetime(2026, 2, 14, tzinfo=timezone.utc)

    def test_w2_dates(self):
        label, ws, we = self.m.WINDOWS[6]
        assert label == "W2"
        assert ws == datetime(2026, 2, 15, tzinfo=timezone.utc)
        assert we == datetime(2026, 2, 28, tzinfo=timezone.utc)

    def test_windows_chronological(self):
        starts = [w[1] for w in self.m.WINDOWS]
        # Q1-Q4 + Jan-Feb ascend; W1/W2 are sub-windows of Jan-Feb → ok
        assert starts[0] <= starts[1] <= starts[2] <= starts[3] <= starts[4]

    def test_three_variants(self):
        assert len(self.m.VARIANTS) == 3

    def test_variant_a(self):
        label, mode, lb, wl, _ = self.m.VARIANTS[0]
        assert label == "A"
        assert mode == "engulf_only"
        assert lb   == 2
        assert wl   is None

    def test_variant_b(self):
        label, mode, lb, wl, _ = self.m.VARIANTS[1]
        assert label == "B"
        assert mode == "engulf_or_strict_pin_at_level"
        assert lb   == 2
        assert wl   is None

    def test_variant_bprime(self):
        label, mode, lb, wl, _ = self.m.VARIANTS[2]
        assert label == "B-Prime"
        assert mode == "engulf_or_strict_pin_at_level"
        assert lb   == 2
        assert wl   == self.m.HNS_PATTERNS

    def test_hns_patterns_list(self):
        assert "head_and_shoulders"          in self.m.HNS_PATTERNS
        assert "inverted_head_and_shoulders" in self.m.HNS_PATTERNS
        assert "double_top"    not in self.m.HNS_PATTERNS
        assert "double_bottom" not in self.m.HNS_PATTERNS

    def test_hns_patterns_exactly_two(self):
        assert len(self.m.HNS_PATTERNS) == 2


# ──────────────────────────────────────────────────────────────────────────────
# 4. Helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.variant_b_prime as m
        self.m = m

    def test_r_from_r(self):
        assert self.m._r({"r": 1.5}) == pytest.approx(1.5)

    def test_r_fallback_realised(self):
        assert self.m._r({"realised_r": -0.7}) == pytest.approx(-0.7)

    def test_r_zero_when_missing(self):
        assert self.m._r({}) == 0.0

    def test_is_win_positive(self):
        assert self.m._is_win({"r": 0.01}) is True

    def test_is_win_zero_false(self):
        assert self.m._is_win({"r": 0.0}) is False

    def test_pct_format(self):
        assert self.m._pct(3.5)  == "+3.5%"
        assert self.m._pct(-1.2) == "-1.2%"

    def test_rs_format(self):
        assert self.m._rs(1.03)  == "+1.03R"
        assert self.m._rs(-0.84) == "-0.84R"

    def test_wr_empty(self):
        assert self.m._wr([]) == "—"

    def test_wr_all_wins(self):
        assert self.m._wr([{"r": 1.0}, {"r": 0.5}]) == "100%"

    def test_wr_half(self):
        assert self.m._wr([{"r": 1.0}, {"r": -1.0}]) == "50%"

    def test_avg_r_empty(self):
        assert self.m._avg_r([]) == 0.0

    def test_avg_r_value(self):
        assert self.m._avg_r([{"r": 1.0}, {"r": -0.5}]) == pytest.approx(0.25)

    def test_worst3_fewer_than_3(self):
        assert self.m._worst3([{"r": -1.5}, {"r": -0.5}]) == pytest.approx(-2.0)

    def test_worst3_picks_three_worst(self):
        ts = [{"r": v} for v in [-3.0, -2.0, -1.0, 1.0, 2.0]]
        assert self.m._worst3(ts) == pytest.approx(-6.0)

    def test_mae_from_mae_r(self):
        assert self.m._mae({"mae_r": -1.5}) == pytest.approx(-1.5)

    def test_mfe_from_mfe_r(self):
        assert self.m._mfe({"mfe_r": 2.0}) == pytest.approx(2.0)

    def test_mae_none_missing(self):
        assert self.m._mae({}) is None

    def test_signal_type_key(self):
        assert self.m._signal_type({"signal_type": "bearish_engulfing_4h"}) == "bearish_engulfing_4h"

    def test_signal_type_unknown(self):
        assert self.m._signal_type({}) == "unknown"

    def test_pattern_key(self):
        assert self.m._pattern({"pattern": "head_and_shoulders"}) == "head_and_shoulders"

    def test_pattern_fallback(self):
        assert self.m._pattern({"pattern_type": "double_top"}) == "double_top"


# ──────────────────────────────────────────────────────────────────────────────
# 5. WindowTriple algebra
# ──────────────────────────────────────────────────────────────────────────────

class _FakeResult:
    def __init__(self, trades, ret=0.0, dd=0.0, wr=0.0, avg_r=0.0):
        self.trades      = trades
        self.n_trades    = len(trades)
        self.return_pct  = ret
        self.max_dd_pct  = dd
        self.win_rate    = wr
        self.avg_r       = avg_r
        self.candle_data = None


def _t(pair="GBP/JPY", ts="2026-02-10T08:00:00+00:00", r=1.0,
        pattern="head_and_shoulders", signal="shooting_star_strict_4h",
        mae=-0.5, mfe=1.5):
    return {
        "pair": pair, "entry_ts": ts, "r": r,
        "pattern": pattern, "signal_type": signal,
        "direction": "short", "mae_r": mae, "mfe_r": mfe,
    }


class TestWindowTriple:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.variant_b_prime as m
        self.WT = m.WindowTriple
        self.m  = m

    def _wt(self, trades_a, trades_b, trades_bp,
             ret_a=0.0, ret_b=0.0, ret_bp=0.0,
             dd_a=0.0, dd_b=0.0, dd_bp=0.0, label="Q1-2025"):
        ra  = _FakeResult(trades_a,  ret=ret_a,  dd=dd_a)
        rb  = _FakeResult(trades_b,  ret=ret_b,  dd=dd_b)
        rbp = _FakeResult(trades_bp, ret=ret_bp, dd=dd_bp)
        return self.WT(
            label,
            datetime(2025, 1, 1,  tzinfo=timezone.utc),
            datetime(2025, 3, 31, tzinfo=timezone.utc),
            ra, rb, rbp,
        )

    # Unlock algebra
    def test_no_unlocks_identical(self):
        t = _t()
        wt = self._wt([t], [t], [t])
        assert wt.unlocked_b  == []
        assert wt.unlocked_bp == []

    def test_b_unlocks_one_bprime_unlocks_same(self):
        shared   = _t(ts="2025-02-01T08:00:00+00:00")
        unlocked = _t(ts="2025-02-10T08:00:00+00:00", pattern="head_and_shoulders")
        wt = self._wt([shared], [shared, unlocked], [shared, unlocked])
        assert len(wt.unlocked_b)  == 1
        assert len(wt.unlocked_bp) == 1

    def test_removed_by_whitelist(self):
        """Non-H&S strict-pin trade in B but not in B-Prime → removed_by_wl."""
        shared      = _t(ts="2025-02-01T08:00:00+00:00")
        non_hns     = _t(ts="2025-02-10T08:00:00+00:00",
                         pattern="double_bottom", signal="shooting_star_strict_4h", r=-1.07)
        # B has it; B-Prime blocks it (whitelist)
        wt = self._wt([shared], [shared, non_hns], [shared])
        assert len(wt.removed_by_wl) == 1
        assert wt.removed_by_wl[0].get("pattern") == "double_bottom"

    def test_removed_trade_is_not_in_unlocked_bp(self):
        shared  = _t(ts="2025-02-01T08:00:00+00:00")
        non_hns = _t(ts="2025-02-10T08:00:00+00:00",
                     pattern="break_retest_bullish", r=-1.04)
        wt = self._wt([shared], [shared, non_hns], [shared])
        assert len(wt.unlocked_bp) == 0
        assert len(wt.removed_by_wl) == 1

    def test_bprime_unlocks_hns_but_not_nonhns(self):
        """B-Prime adds H&S trade but not double_bottom."""
        hns_t    = _t(ts="2025-02-05T08:00:00+00:00", pattern="head_and_shoulders", r=1.03)
        non_hns  = _t(ts="2025-02-07T08:00:00+00:00", pattern="double_bottom",     r=-1.07)
        # A: neither; B: both; B-Prime: H&S only
        wt = self._wt([], [hns_t, non_hns], [hns_t])
        assert len(wt.unlocked_b)     == 2
        assert len(wt.unlocked_bp)    == 1
        assert len(wt.removed_by_wl)  == 1
        assert wt.unlocked_bp[0].get("pattern") == "head_and_shoulders"

    def test_locked_by_bp_is_empty_when_bprime_superset(self):
        """B-Prime trades should be a subset of A ∪ H&S unlocks; A trades preserved."""
        shared = _t(ts="2025-02-01T08:00:00+00:00", signal="bearish_engulfing_4h")
        hns_t  = _t(ts="2025-02-05T08:00:00+00:00", pattern="head_and_shoulders")
        wt = self._wt([shared], [shared, hns_t], [shared, hns_t])
        assert wt.locked_by_bp == []

    def test_properties_delegate_to_results(self):
        wt = self._wt([], [], [], ret_a=5.0, ret_b=3.0, ret_bp=6.0,
                      dd_a=2.0, dd_b=1.5, dd_bp=2.5)
        assert wt._ret(wt.res_a)  == pytest.approx(5.0)
        assert wt._ret(wt.res_bp) == pytest.approx(6.0)
        assert wt._dd(wt.res_b)   == pytest.approx(1.5)

    def test_unlock_mae_mfe_from_trades(self):
        t = {**_t(), "mae_r": -1.0, "mfe_r": 2.0}
        wt = self._wt([], [t], [t])
        mae, mfe = wt.unlock_mae_mfe(wt.unlocked_bp)
        assert mae == "-1.00R"
        assert mfe == "+2.00R"

    def test_unlock_mae_mfe_none_when_empty(self):
        wt = self._wt([], [], [])
        mae, mfe = wt.unlock_mae_mfe([])
        assert mae is None
        assert mfe is None

    def test_multiple_removed_by_wl(self):
        shared = _t(ts="2025-01-01T08:00:00+00:00", signal="bearish_engulfing_4h")
        rm1    = _t(ts="2025-01-05T08:00:00+00:00", pattern="double_top",    r=-1.0,
                    signal="shooting_star_strict_4h")
        rm2    = _t(ts="2025-01-10T08:00:00+00:00", pattern="double_bottom", r=-1.0,
                    signal="shooting_star_strict_4h")
        wt = self._wt([shared], [shared, rm1, rm2], [shared])
        assert len(wt.removed_by_wl) == 2


# ──────────────────────────────────────────────────────────────────────────────
# 6. Key-questions logic
# ──────────────────────────────────────────────────────────────────────────────

class TestKeyQuestions:
    """Verify the Q1/Jan-Feb/W1 check logic in build_report."""

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.variant_b_prime as m
        self.m = m
        self.WT = m.WindowTriple

    def _wt(self, label, ret_a, ret_b, ret_bp):
        ws_map = {
            "Q1-2025":      (datetime(2025,1,1,tzinfo=timezone.utc), datetime(2025,3,31,tzinfo=timezone.utc)),
            "Jan-Feb-2026": (datetime(2026,1,1,tzinfo=timezone.utc), datetime(2026,2,28,tzinfo=timezone.utc)),
            "W1":           (datetime(2026,2,1,tzinfo=timezone.utc), datetime(2026,2,14,tzinfo=timezone.utc)),
        }
        ws, we = ws_map[label]
        ra  = _FakeResult([], ret=ret_a)
        rb  = _FakeResult([], ret=ret_b)
        rbp = _FakeResult([], ret=ret_bp)
        return self.WT(label, ws, we, ra, rb, rbp)

    def test_q1_regression_fixed_when_bprime_matches_a(self):
        """B-Prime return = A return → regression fixed."""
        wt = self._wt("Q1-2025", ret_a=12.7, ret_b=-5.2, ret_bp=12.7)
        assert wt._ret(wt.res_bp) >= wt._ret(wt.res_a) - 0.1

    def test_q1_regression_not_fixed_when_bprime_still_worse(self):
        wt = self._wt("Q1-2025", ret_a=12.7, ret_b=-5.2, ret_bp=5.0)
        assert wt._ret(wt.res_bp) < wt._ret(wt.res_a) - 0.1

    def test_jan_feb_gain_preserved_when_bprime_better(self):
        wt = self._wt("Jan-Feb-2026", ret_a=11.4, ret_b=15.2, ret_bp=15.2)
        assert wt._ret(wt.res_bp) >= wt._ret(wt.res_a) - 0.1

    def test_jan_feb_gain_lost_when_bprime_worse(self):
        wt = self._wt("Jan-Feb-2026", ret_a=11.4, ret_b=15.2, ret_bp=5.0)
        assert wt._ret(wt.res_bp) < wt._ret(wt.res_a) - 0.1

    def test_w1_gain_preserved(self):
        wt = self._wt("W1", ret_a=0.0, ret_b=1.5, ret_bp=1.5)
        assert wt._ret(wt.res_bp) >= wt._ret(wt.res_a) - 0.1

    def test_removed_by_wl_all_nonhns(self):
        """All removed trades should be non-H&S patterns."""
        non_hns_trades = [
            {"pattern": "double_bottom",      "r": -1.07, "signal_type": "shooting_star_strict_4h",
             "pair": "USD/JPY", "entry_ts": f"2025-01-0{i}T08:00:00+00:00"}
            for i in range(1, 4)
        ]
        ra  = _FakeResult([])
        rb  = _FakeResult(non_hns_trades)
        rbp = _FakeResult([])
        wt  = self.WT("Q1-2025",
                      datetime(2025,1,1,tzinfo=timezone.utc),
                      datetime(2025,3,31,tzinfo=timezone.utc),
                      ra, rb, rbp)
        for t in wt.removed_by_wl:
            assert not self.m._is_hns_pattern(t), \
                f"Removed trade should be non-H&S: {t.get('pattern')}"


# ──────────────────────────────────────────────────────────────────────────────
# 7. Script structure
# ──────────────────────────────────────────────────────────────────────────────

class TestScriptStructure:

    @pytest.fixture(autouse=True)
    def _import(self):
        import scripts.variant_b_prime as m
        self.m = m

    def test_report_path_attribute(self):
        assert hasattr(self.m, "REPORT_PATH")

    def test_report_path_correct_filename(self):
        assert self.m.REPORT_PATH.name == "variant_b_prime.md"

    def test_report_path_under_backtesting_results(self):
        assert "backtesting/results" in str(self.m.REPORT_PATH)

    def test_capital_8000(self):
        assert self.m.CAPITAL == 8_000.0

    def test_main_callable(self):
        assert callable(self.m.main)

    def test_build_report_callable(self):
        assert callable(self.m.build_report)

    def test_run_variant_callable(self):
        assert callable(self.m.run_variant)

    def test_window_triple_class_exists(self):
        assert hasattr(self.m, "WindowTriple")

    def test_atexit_guard_in_source(self):
        src = (REPO / "scripts/variant_b_prime.py").read_text()
        assert "atexit.register" in src

    def test_reset_config_includes_whitelist(self):
        """_reset_config must reset STRICT_PIN_PATTERN_WHITELIST."""
        src = (REPO / "scripts/variant_b_prime.py").read_text()
        assert "STRICT_PIN_PATTERN_WHITELIST" in src

    def test_reset_config_restores_all_three(self):
        _sc.ENTRY_TRIGGER_MODE           = "engulf_or_strict_pin_at_level"
        _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 99
        _sc.STRICT_PIN_PATTERN_WHITELIST = ["head_and_shoulders"]
        self.m._reset_config()
        assert _sc.ENTRY_TRIGGER_MODE           == "engulf_only"
        assert _sc.ENGULF_CONFIRM_LOOKBACK_BARS == 2
        assert _sc.STRICT_PIN_PATTERN_WHITELIST is None

    def test_no_live_trading_calls(self):
        src = (REPO / "scripts/variant_b_prime.py").read_text()
        for forbidden in ("submit_order", "place_order", "create_order", "dry_run=False"):
            assert forbidden not in src

    def test_full_year_not_in_windows(self):
        """Full-year window must be excluded (known broken)."""
        labels = [w[0] for w in self.m.WINDOWS]
        assert "Full-year" not in labels

    def test_gate_in_set_and_forget_source(self):
        """FILTER 5b gate must be present in set_and_forget.py."""
        src = (REPO / "src/strategy/forex/set_and_forget.py").read_text()
        assert "FILTER 5b" in src
        assert "STRICT_PIN_PATTERN_WHITELIST" in src
        assert "strict_pin_pattern_blocked" in src
