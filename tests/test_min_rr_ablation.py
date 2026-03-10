"""
Tests for scripts/min_rr_ablation.py

Coverage:
  - Config defaults at module load
  - Variant definitions
  - Window definitions
  - Trade identity / unlock helpers
  - Distribution helpers
  - Formatter functions
  - WindowResult computed properties
  - WindowTriple helpers
  - _set_rr / _reset_config guard
  - Report smoke test
  - No live-order calls
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from datetime import datetime, timezone
from typing import List
import pytest

# ── repo root on path ─────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.strategy.forex.strategy_config as _sc

# ── load module under test ────────────────────────────────────────────────
import scripts.min_rr_ablation as _m


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _trade(pair="GBP/JPY", direction="short", ts="2025-01-15T09:00",
           r=1.2, pattern="head_and_shoulders", mae=-0.5, mfe=1.8):
    return {
        "pair": pair, "direction": direction, "entry_ts": ts,
        "r": r, "pattern": pattern, "mae_r": mae, "mfe_r": mfe,
    }


def _restore():
    """Reset config to production values between tests that mutate it."""
    _sc.MIN_RR               = _m._ORIG_MIN_RR
    _sc.MIN_RR_STANDARD      = _m._ORIG_MIN_RR_STANDARD
    _sc.MIN_RR_COUNTERTREND  = _m._ORIG_MIN_RR_COUNTERTREND
    _sc.MIN_RR_SMALL_ACCOUNT = _m._ORIG_MIN_RR_SMALL


# ─────────────────────────────────────────────────────────────────────────
# 1. Config defaults
# ─────────────────────────────────────────────────────────────────────────
class TestConfigDefaults:
    def test_orig_min_rr_is_2_5(self):
        assert _m._ORIG_MIN_RR == 2.5

    def test_orig_min_rr_standard_is_2_5(self):
        assert _m._ORIG_MIN_RR_STANDARD == 2.5

    def test_orig_min_rr_countertrend_is_2_5(self):
        assert _m._ORIG_MIN_RR_COUNTERTREND == 2.5

    def test_orig_min_rr_small_is_2_5(self):
        assert _m._ORIG_MIN_RR_SMALL == 2.5

    def test_trigger_mode_captured(self):
        assert _m._ORIG_TRIGGER_MODE == "engulf_or_strict_pin_at_level"

    def test_whitelist_captured(self):
        assert _m._ORIG_WL == ["head_and_shoulders", "inverted_head_and_shoulders"]

    def test_lookback_captured(self):
        assert _m._ORIG_LB == 2

    def test_production_rr_still_2_5(self):
        assert _sc.MIN_RR == 2.5


# ─────────────────────────────────────────────────────────────────────────
# 2. Variant definitions
# ─────────────────────────────────────────────────────────────────────────
class TestVariantDefinitions:
    def test_three_variants(self):
        assert len(_m.VARIANTS) == 3

    def test_variant_ids(self):
        ids = [v[0] for v in _m.VARIANTS]
        assert ids == ["A", "B", "C"]

    def test_variant_rr_values(self):
        rrs = [v[1] for v in _m.VARIANTS]
        assert rrs == [2.5, 2.0, 1.5]

    def test_variant_a_is_baseline(self):
        assert _m.VARIANTS[0][1] == 2.5

    def test_variant_b_is_2_0(self):
        assert _m.VARIANTS[1][1] == 2.0

    def test_variant_c_is_1_5(self):
        assert _m.VARIANTS[2][1] == 1.5

    def test_variant_labels_non_empty(self):
        for _, _, label in _m.VARIANTS:
            assert label.strip()


# ─────────────────────────────────────────────────────────────────────────
# 3. Window definitions
# ─────────────────────────────────────────────────────────────────────────
class TestWindowDefinitions:
    def test_eight_windows(self):
        assert len(_m.WINDOWS) == 8

    def test_window_names(self):
        names = [w[0] for w in _m.WINDOWS]
        assert "Q1-2025" in names
        assert "Q2-2025" in names
        assert "Q3-2025" in names
        assert "Q4-2025" in names
        assert "Jan-Feb-2026" in names
        assert "W1" in names
        assert "W2" in names
        assert "live-parity" in names

    def test_all_windows_have_start_before_end(self):
        for name, start, end in _m.WINDOWS:
            assert start < end, f"{name}: start not before end"

    def test_windows_are_utc(self):
        for name, start, end in _m.WINDOWS:
            assert start.tzinfo is not None, f"{name} start missing tz"
            assert end.tzinfo is not None, f"{name} end missing tz"

    def test_q1_2025_dates(self):
        w = next(w for w in _m.WINDOWS if w[0] == "Q1-2025")
        assert w[1].year == 2025 and w[1].month == 1
        assert w[2].month == 3

    def test_live_parity_in_2026(self):
        w = next(w for w in _m.WINDOWS if w[0] == "live-parity")
        assert w[1].year == 2026

    def test_pairs_list_has_7_alex_pairs(self):
        assert len(_m.PAIRS) == 7
        assert "GBP/JPY" in _m.PAIRS
        assert "USD/JPY" in _m.PAIRS


# ─────────────────────────────────────────────────────────────────────────
# 4. _set_rr / _reset_config
# ─────────────────────────────────────────────────────────────────────────
class TestSetRR:
    def setup_method(self): _restore()
    def teardown_method(self): _restore()

    def test_set_rr_changes_all_four(self):
        _m._set_rr(2.0)
        assert _sc.MIN_RR == 2.0
        assert _sc.MIN_RR_STANDARD == 2.0
        assert _sc.MIN_RR_COUNTERTREND == 2.0
        assert _sc.MIN_RR_SMALL_ACCOUNT == 2.0

    def test_set_rr_1_5(self):
        _m._set_rr(1.5)
        assert _sc.MIN_RR == 1.5
        assert _sc.MIN_RR_STANDARD == 1.5

    def test_reset_restores_production_values(self):
        _m._set_rr(1.5)
        assert _sc.MIN_RR == 1.5
        _m._reset_config()
        assert _sc.MIN_RR == 2.5
        assert _sc.MIN_RR_STANDARD == 2.5
        assert _sc.MIN_RR_COUNTERTREND == 2.5
        assert _sc.MIN_RR_SMALL_ACCOUNT == 2.5

    def test_reset_restores_trigger_mode(self):
        _sc.ENTRY_TRIGGER_MODE = "engulf_only"
        _m._reset_config()
        assert _sc.ENTRY_TRIGGER_MODE == "engulf_or_strict_pin_at_level"

    def test_reset_restores_whitelist(self):
        _sc.STRICT_PIN_PATTERN_WHITELIST = None
        _m._reset_config()
        assert _sc.STRICT_PIN_PATTERN_WHITELIST == [
            "head_and_shoulders", "inverted_head_and_shoulders"
        ]

    def test_reset_restores_lookback(self):
        _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 99
        _m._reset_config()
        assert _sc.ENGULF_CONFIRM_LOOKBACK_BARS == 2

    def test_recovery_min_rr_unchanged(self):
        _m._set_rr(1.5)
        assert _sc.RECOVERY_MIN_RR == 3.0
        _m._reset_config()
        assert _sc.RECOVERY_MIN_RR == 3.0


# ─────────────────────────────────────────────────────────────────────────
# 5. Trade identity helpers
# ─────────────────────────────────────────────────────────────────────────
class TestTradeKey:
    def test_same_trade_same_key(self):
        t = _trade(pair="GBP/JPY", direction="short", ts="2025-01-15T09:00")
        assert _m._trade_key(t) == _m._trade_key(t)

    def test_different_pair_different_key(self):
        t1 = _trade(pair="GBP/JPY", ts="2025-01-15T09:00")
        t2 = _trade(pair="USD/JPY", ts="2025-01-15T09:00")
        assert _m._trade_key(t1) != _m._trade_key(t2)

    def test_different_direction_different_key(self):
        t1 = _trade(direction="long",  ts="2025-01-15T09:00")
        t2 = _trade(direction="short", ts="2025-01-15T09:00")
        assert _m._trade_key(t1) != _m._trade_key(t2)

    def test_different_hour_different_key(self):
        t1 = _trade(ts="2025-01-15T09:00")
        t2 = _trade(ts="2025-01-15T10:00")
        assert _m._trade_key(t1) != _m._trade_key(t2)

    def test_same_hour_same_key(self):
        t1 = _trade(ts="2025-01-15T09:00")
        t2 = _trade(ts="2025-01-15T09:45")
        # both truncate to "2025-01-15T09"
        assert _m._trade_key(t1) == _m._trade_key(t2)

    def test_key_is_tuple_of_three(self):
        k = _m._trade_key(_trade())
        assert isinstance(k, tuple) and len(k) == 3


# ─────────────────────────────────────────────────────────────────────────
# 6. _find_unlocked / _find_removed
# ─────────────────────────────────────────────────────────────────────────
class TestFindUnlocked:
    def test_empty_base_all_new_are_unlocked(self):
        new = [_trade(ts="2025-01-15T09"), _trade(ts="2025-01-16T09")]
        assert len(_m._find_unlocked([], new)) == 2

    def test_all_same_none_unlocked(self):
        t = _trade(ts="2025-01-15T09")
        assert _m._find_unlocked([t], [t]) == []

    def test_one_new_one_unlocked(self):
        base = [_trade(ts="2025-01-15T09")]
        new  = [_trade(ts="2025-01-15T09"), _trade(pair="USD/JPY", ts="2025-01-16T09")]
        unlocked = _m._find_unlocked(base, new)
        assert len(unlocked) == 1
        assert unlocked[0]["pair"] == "USD/JPY"

    def test_empty_new_empty_unlocked(self):
        base = [_trade(ts="2025-01-15T09")]
        assert _m._find_unlocked(base, []) == []

    def test_find_removed_baseline_not_in_new(self):
        base = [_trade(pair="GBP/JPY", ts="2025-01-15T09"),
                _trade(pair="USD/JPY", ts="2025-01-16T09")]
        new  = [_trade(pair="GBP/JPY", ts="2025-01-15T09")]
        removed = _m._find_removed(base, new)
        assert len(removed) == 1
        assert removed[0]["pair"] == "USD/JPY"

    def test_find_removed_all_present_empty(self):
        base = [_trade(ts="2025-01-15T09")]
        new  = [_trade(ts="2025-01-15T09"), _trade(pair="USD/JPY", ts="2025-01-16T09")]
        assert _m._find_removed(base, new) == []


# ─────────────────────────────────────────────────────────────────────────
# 7. Distribution helpers
# ─────────────────────────────────────────────────────────────────────────
class TestDistributions:
    def _trades(self):
        return [
            _trade(pair="GBP/JPY", pattern="head_and_shoulders",  direction="short"),
            _trade(pair="GBP/JPY", pattern="double_top",           direction="short"),
            _trade(pair="USD/JPY", pattern="head_and_shoulders",   direction="long"),
        ]

    def test_pair_dist_counts(self):
        d = _m._pair_dist(self._trades())
        assert d["GBP/JPY"] == 2
        assert d["USD/JPY"] == 1

    def test_pair_dist_sorted_desc(self):
        d = _m._pair_dist(self._trades())
        assert list(d.keys())[0] == "GBP/JPY"

    def test_pattern_dist_counts(self):
        d = _m._pattern_dist(self._trades())
        assert d["head_and_shoulders"] == 2
        assert d["double_top"] == 1

    def test_dir_dist(self):
        d = _m._dir_dist(self._trades())
        assert d["short"] == 2
        assert d["long"] == 1

    def test_empty_dist_is_empty(self):
        assert _m._pair_dist([]) == {}
        assert _m._pattern_dist([]) == {}


# ─────────────────────────────────────────────────────────────────────────
# 8. Stat helpers
# ─────────────────────────────────────────────────────────────────────────
class TestStatHelpers:
    def test_wr_all_wins(self):
        trades = [_trade(r=1.0), _trade(r=2.0)]
        assert _m._wr(trades) == 1.0

    def test_wr_all_losses(self):
        trades = [_trade(r=-1.0), _trade(r=-1.0)]
        assert _m._wr(trades) == 0.0

    def test_wr_half(self):
        trades = [_trade(r=1.0), _trade(r=-1.0)]
        assert _m._wr(trades) == 0.5

    def test_wr_empty_zero(self):
        assert _m._wr([]) == 0.0

    def test_avg_r(self):
        trades = [_trade(r=2.0), _trade(r=0.0), _trade(r=-1.0)]
        assert abs(_m._avg_r(trades) - (1.0 / 3.0)) < 1e-9

    def test_avg_r_empty(self):
        assert _m._avg_r([]) == 0.0

    def test_sum_r(self):
        trades = [_trade(r=1.5), _trade(r=-1.0)]
        assert abs(_m._sum_r(trades) - 0.5) < 1e-9

    def test_sum_r_empty(self):
        assert _m._sum_r([]) == 0.0

    def test_avg_mae(self):
        trades = [_trade(mae=-0.5), _trade(mae=-1.0)]
        assert abs(_m._avg_mae(trades) - (-0.75)) < 1e-9

    def test_avg_mae_empty(self):
        assert _m._avg_mae([]) is None

    def test_avg_mae_missing_fields(self):
        trades = [{"pair": "GBP/JPY", "r": 1.0}]
        assert _m._avg_mae(trades) is None

    def test_avg_mfe(self):
        trades = [_trade(mfe=1.5), _trade(mfe=2.5)]
        assert abs(_m._avg_mfe(trades) - 2.0) < 1e-9

    def test_avg_mfe_empty(self):
        assert _m._avg_mfe([]) is None


# ─────────────────────────────────────────────────────────────────────────
# 9. Formatter functions
# ─────────────────────────────────────────────────────────────────────────
class TestFormatters:
    def test_fmt_r_positive(self):
        assert _m._fmt_r(1.5) == "+1.50R"

    def test_fmt_r_negative(self):
        assert _m._fmt_r(-1.0) == "-1.00R"

    def test_fmt_r_zero(self):
        assert _m._fmt_r(0.0) == "+0.00R"

    def test_fmt_r_none(self):
        assert _m._fmt_r(None) == "—"

    def test_fmt_pct_positive(self):
        assert _m._fmt_pct(10.5) == "+10.5%"

    def test_fmt_pct_negative(self):
        assert _m._fmt_pct(-3.2) == "-3.2%"

    def test_fmt_wr_100(self):
        assert _m._fmt_wr(1.0) == "100%"

    def test_fmt_wr_zero(self):
        assert _m._fmt_wr(0.0) == "0%"

    def test_fmt_wr_half(self):
        assert _m._fmt_wr(0.5) == "50%"


# ─────────────────────────────────────────────────────────────────────────
# 10. WindowResult computed properties
# ─────────────────────────────────────────────────────────────────────────
class TestWindowResult:
    def _make_result(self, trades):
        from unittest.mock import MagicMock
        r = MagicMock()
        r.n_trades    = len(trades)
        r.return_pct  = 10.0
        r.win_rate    = _m._wr(trades)
        r.avg_r       = _m._avg_r(trades)
        r.max_dd_pct  = 5.0
        r.trades      = trades
        return r

    def _wr_obj(self, trades=None, variant="A", error=""):
        if trades is None:
            trades = [_trade(r=1.5), _trade(r=-1.0)]
        return _m.WindowResult(
            variant=variant, min_rr=2.5, label="test",
            window="Q1-2025",
            result=self._make_result(trades) if not error else None,
            error=error,
        )

    def test_n_from_result(self):
        wr = self._wr_obj(trades=[_trade(r=1.0), _trade(r=2.0)])
        assert wr.n == 2

    def test_n_zero_on_error(self):
        wr = self._wr_obj(error="timeout")
        assert wr.n == 0

    def test_total_r(self):
        wr = self._wr_obj(trades=[_trade(r=1.5), _trade(r=-1.0)])
        assert abs(wr.total_r - 0.5) < 1e-9

    def test_expectancy_positive(self):
        wr = self._wr_obj(trades=[_trade(r=2.0), _trade(r=1.0)])
        assert abs(wr.expectancy - 1.5) < 1e-9

    def test_expectancy_zero_no_trades(self):
        wr = self._wr_obj(error="no data")
        assert wr.expectancy == 0.0

    def test_worst3_sorted_ascending(self):
        trades = [_trade(r=-1.0), _trade(r=-0.5), _trade(r=2.0), _trade(r=-2.0)]
        wr = self._wr_obj(trades=trades)
        w3 = wr.worst3
        assert w3[0] <= w3[-1]
        assert len(w3) == 3

    def test_worst3_fewer_than_3(self):
        wr = self._wr_obj(trades=[_trade(r=-1.0)])
        assert len(wr.worst3) == 1

    def test_mae_r_list(self):
        trades = [_trade(mae=-0.5), _trade(mae=-1.0)]
        wr = self._wr_obj(trades=trades)
        assert len(wr.mae_r_list) == 2

    def test_mfe_r_list(self):
        trades = [_trade(mfe=1.5), _trade(mfe=2.0)]
        wr = self._wr_obj(trades=trades)
        assert len(wr.mfe_r_list) == 2


# ─────────────────────────────────────────────────────────────────────────
# 11. WindowTriple
# ─────────────────────────────────────────────────────────────────────────
class TestWindowTriple:
    def _wt(self):
        _UTC = timezone.utc

        def _wr(v, rr):
            return _m.WindowResult(variant=v, min_rr=rr, label=f"test {v}",
                                   window="Q1-2025", result=None)
        return _m.WindowTriple(
            window="Q1-2025",
            start=datetime(2025, 1, 1, tzinfo=_UTC),
            end=datetime(2025, 3, 31, tzinfo=_UTC),
            result_a=_wr("A", 2.5),
            result_b=_wr("B", 2.0),
            result_c=_wr("C", 1.5),
        )

    def test_results_returns_three(self):
        assert len(self._wt().results) == 3

    def test_by_variant_a(self):
        assert self._wt().by_variant("A").variant == "A"

    def test_by_variant_b(self):
        assert self._wt().by_variant("B").min_rr == 2.0

    def test_by_variant_c(self):
        assert self._wt().by_variant("C").min_rr == 1.5

    def test_invalid_variant_raises(self):
        with pytest.raises(KeyError):
            self._wt().by_variant("Z")

    def test_window_name_preserved(self):
        assert self._wt().window == "Q1-2025"


# ─────────────────────────────────────────────────────────────────────────
# 12. Report smoke test
# ─────────────────────────────────────────────────────────────────────────
class TestReportSmoke:
    """Build a report from synthetic WindowTriples — verifies no crash."""

    def _make_triple(self, window: str) -> _m.WindowTriple:
        from unittest.mock import MagicMock
        _UTC = timezone.utc

        def _br(trades):
            r = MagicMock()
            r.n_trades   = len(trades)
            r.return_pct = 5.0
            r.win_rate   = _m._wr(trades)
            r.avg_r      = _m._avg_r(trades)
            r.max_dd_pct = 3.0
            r.trades     = trades
            r.candle_data = None
            return r

        def _wr_obj(var, rr, trades):
            return _m.WindowResult(
                variant=var, min_rr=rr, label=f"label {var}",
                window=window, result=_br(trades),
            )

        base  = [_trade(ts="2025-01-15T09", r=1.5),
                 _trade(ts="2025-01-20T09", pair="USD/JPY", r=-1.0)]
        mod   = base + [_trade(ts="2025-02-01T09", pair="USD/CHF", r=2.0)]
        lower = mod  + [_trade(ts="2025-02-10T09", pair="GBP/CHF", r=-1.0)]

        return _m.WindowTriple(
            window=window,
            start=datetime(2025, 1, 1, tzinfo=_UTC),
            end=datetime(2025, 3, 31, tzinfo=_UTC),
            result_a=_wr_obj("A", 2.5, base),
            result_b=_wr_obj("B", 2.0, mod),
            result_c=_wr_obj("C", 1.5, lower),
        )

    def test_build_report_returns_string(self):
        triples = [self._make_triple("Q1-2025"), self._make_triple("Q2-2025")]
        report = _m._build_report(triples)
        assert isinstance(report, str)

    def test_report_contains_variant_headers(self):
        triples = [self._make_triple("Q1-2025")]
        report = _m._build_report(triples)
        assert "Variant" in report
        assert "MIN_RR" in report

    def test_report_contains_all_windows(self):
        triples = [self._make_triple("Q1-2025"), self._make_triple("Q4-2025")]
        report = _m._build_report(triples)
        assert "Q1-2025" in report
        assert "Q4-2025" in report

    def test_report_contains_unlock_section(self):
        report = _m._build_report([self._make_triple("Q1-2025")])
        assert "Unlock Analysis" in report
        assert "2.0" in report
        assert "1.5" in report

    def test_report_contains_verdict(self):
        report = _m._build_report([self._make_triple("Q1-2025")])
        assert "Verdict" in report

    def test_report_contains_total_r(self):
        report = _m._build_report([self._make_triple("Q1-2025")])
        assert "Total R Captured" in report

    def test_report_contains_mae_mfe_section(self):
        report = _m._build_report([self._make_triple("Q1-2025")])
        assert "MAE" in report and "MFE" in report

    def test_report_no_crash_all_empty(self):
        _UTC = timezone.utc
        wr = _m.WindowResult(variant="A", min_rr=2.5, label="test",
                              window="Q1-2025", result=None, error="no data")
        triple = _m.WindowTriple(
            window="Q1-2025",
            start=datetime(2025, 1, 1, tzinfo=_UTC),
            end=datetime(2025, 3, 31, tzinfo=_UTC),
            result_a=wr,
            result_b=_m.WindowResult(variant="B", min_rr=2.0, label="test",
                                     window="Q1-2025", result=None),
            result_c=_m.WindowResult(variant="C", min_rr=1.5, label="test",
                                     window="Q1-2025", result=None),
        )
        report = _m._build_report([triple])
        assert isinstance(report, str)


# ─────────────────────────────────────────────────────────────────────────
# 13. Script structure checks
# ─────────────────────────────────────────────────────────────────────────
class TestScriptStructure:
    def _src(self) -> str:
        return (REPO / "scripts" / "min_rr_ablation.py").read_text()

    def test_atexit_registered(self):
        assert "atexit.register" in self._src()

    def test_reset_config_defined(self):
        assert "def _reset_config" in self._src()

    def test_set_rr_defined(self):
        assert "def _set_rr" in self._src()

    def test_no_live_order_calls(self):
        src = self._src()
        for forbidden in ("submit_order", "place_order", "create_order",
                          "dry_run=False", "ACCOUNT_MODE=LIVE_REAL"):
            assert forbidden not in src, f"Found forbidden: {forbidden}"

    def test_preloaded_candle_data_reuse(self):
        assert "preloaded_candle_data" in self._src()

    def test_recovery_min_rr_not_mutated(self):
        src = self._src()
        assert "RECOVERY_MIN_RR" not in src.split("_reset_config")[1].split("def ")[0] or \
               "RECOVERY_MIN_RR = " not in self._src()

    def test_build_report_defined(self):
        assert "def _build_report" in self._src()

    def test_run_ablation_defined(self):
        assert "def run_ablation" in self._src()

    def test_offline_note_in_docstring(self):
        assert "OFFLINE ONLY" in self._src()

    def test_no_master_changes_note(self):
        assert "master" in self._src().lower()


# ─────────────────────────────────────────────────────────────────────────
# 14. Edge cases
# ─────────────────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_find_unlocked_identical_pair_different_direction(self):
        t1 = _trade(pair="GBP/JPY", direction="short", ts="2025-01-15T09")
        t2 = _trade(pair="GBP/JPY", direction="long",  ts="2025-01-15T09")
        # t2 is NOT in base (which only has t1), so it should be unlocked
        unlocked = _m._find_unlocked([t1], [t1, t2])
        assert len(unlocked) == 1
        assert unlocked[0]["direction"] == "long"

    def test_sum_r_negative_only(self):
        trades = [_trade(r=-1.0), _trade(r=-0.5)]
        assert _m._sum_r(trades) < 0

    def test_wr_single_win(self):
        assert _m._wr([_trade(r=1.0)]) == 1.0

    def test_wr_single_loss(self):
        assert _m._wr([_trade(r=-1.0)]) == 0.0

    def test_pattern_dist_with_none_pattern(self):
        t = {"pair": "GBP/JPY", "direction": "short", "r": 1.0}
        d = _m._pattern_dist([t])
        assert "?" in d

    def test_fmt_r_large_positive(self):
        assert _m._fmt_r(10.5) == "+10.50R"

    def test_fmt_pct_zero(self):
        assert _m._fmt_pct(0.0) == "+0.0%"

    def test_window_result_no_result_trades_empty(self):
        wr = _m.WindowResult(variant="A", min_rr=2.5, label="x",
                             window="Q1", result=None)
        assert wr.trades == []
        assert wr.total_r == 0.0
        assert wr.worst3 == []
