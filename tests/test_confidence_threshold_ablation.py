"""tests/test_confidence_threshold_ablation.py
==============================================
Tests for scripts/confidence_threshold_ablation.py.

Covers:
  1. atexit / config restore
  2. Variant constants (correct threshold values)
  3. WINDOWS definition (8 windows, correct order)
  4. _trade_key() stability
  5. _r() field fallback chain
  6. _unlocked() trade diff logic
  7. _displaced() with and without replacement
  8. _flag_unlocked() — low-conf loss, concentration, MAE>MFE flags
  9. build_report() smoke test (1 window, 3 variants)
 10. build_report() verdict logic
 11. ATR floor violation detector
 12. Confidence map lookup (_conf_for_trade)
 13. atexit does not fire prematurely
 14. run_all() result structure (mocked backtester)
"""
from __future__ import annotations

import sys
import json
import types
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ── lightweight BacktestResult stub ──────────────────────────────────────────
@dataclass
class _FakeResult:
    trades:      List[dict] = field(default_factory=list)
    return_pct:  float      = 0.0
    max_dd_pct:  float      = 0.0
    win_rate:    float      = 0.0
    avg_r:       float      = 0.0
    n_trades:    int        = 0
    candle_data: dict       = field(default_factory=dict)


def _fake_trade(pair="GBP/USD", pattern="double_top", direction="short",
                r=1.5, stop_pips=20, mfe_r=1.8, mae_r=-0.3,
                entry_ts="2025-01-15T09:00:00+00:00", confidence=0.74):
    return {
        "pair":               pair,
        "pattern":            pattern,
        "direction":          direction,
        "r":                  r,
        "initial_stop_pips":  stop_pips,
        "mfe_r":              mfe_r,
        "mae_r":              mae_r,
        "entry_ts":           entry_ts,
        "confidence_entry":   confidence,
    }


def _fake_wr(var_id, window, threshold, trades=None, enter_decisions=None):
    from scripts.confidence_threshold_ablation import WindowResult
    trades = trades or []
    result = _FakeResult(
        trades=trades,
        n_trades=len(trades),
        win_rate=sum(1 for t in trades if t.get("r", 0) > 0) / len(trades) if trades else 0.0,
        avg_r=sum(t.get("r", 0) for t in trades) / len(trades) if trades else 0.0,
    )
    return WindowResult(
        variant=var_id,
        window=window,
        threshold=threshold,
        result=result,
        candle_data={},
        enter_decisions=enter_decisions or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. atexit / config restore
# ─────────────────────────────────────────────────────────────────────────────
class TestAtexitRestore(unittest.TestCase):

    def test_original_conf_captured(self):
        from scripts.confidence_threshold_ablation import _ORIG_CONF
        self.assertIsInstance(_ORIG_CONF, float)
        self.assertGreater(_ORIG_CONF, 0.0)
        self.assertLess(_ORIG_CONF, 1.0)

    def test_reset_fn_restores(self):
        import src.strategy.forex.strategy_config as _sc
        from scripts.confidence_threshold_ablation import _reset_conf, _ORIG_CONF
        original = _sc.MIN_CONFIDENCE
        _sc.MIN_CONFIDENCE = 0.50
        _reset_conf()
        self.assertEqual(_sc.MIN_CONFIDENCE, _ORIG_CONF)
        _sc.MIN_CONFIDENCE = original   # restore for other tests

    def test_baseline_conf_unchanged_in_module(self):
        """_ORIG_CONF must match current production default."""
        from scripts.confidence_threshold_ablation import _ORIG_CONF
        self.assertAlmostEqual(_ORIG_CONF, 0.77, places=3)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Variant constants
# ─────────────────────────────────────────────────────────────────────────────
class TestVariants(unittest.TestCase):

    def setUp(self):
        from scripts.confidence_threshold_ablation import VARIANTS
        self.variants = VARIANTS

    def test_three_variants(self):
        self.assertEqual(len(self.variants), 3)

    def test_variant_ids(self):
        ids = [v[0] for v in self.variants]
        self.assertEqual(ids, ["A", "B", "C"])

    def test_thresholds(self):
        thresholds = {v[0]: v[1] for v in self.variants}
        self.assertAlmostEqual(thresholds["A"], 0.77, places=3)
        self.assertAlmostEqual(thresholds["B"], 0.73, places=3)
        self.assertAlmostEqual(thresholds["C"], 0.70, places=3)

    def test_a_is_highest(self):
        thresholds = sorted([v[1] for v in self.variants], reverse=True)
        self.assertEqual(self.variants[0][1], thresholds[0])

    def test_c_is_lowest(self):
        thresholds = sorted([v[1] for v in self.variants])
        self.assertEqual(self.variants[-1][1], thresholds[0])


# ─────────────────────────────────────────────────────────────────────────────
# 3. WINDOWS definition
# ─────────────────────────────────────────────────────────────────────────────
class TestWindows(unittest.TestCase):

    def setUp(self):
        from scripts.confidence_threshold_ablation import WINDOWS
        self.windows = WINDOWS

    def test_eight_windows(self):
        self.assertEqual(len(self.windows), 8)

    def test_window_names(self):
        names = [w[0] for w in self.windows]
        self.assertIn("Q1-2025", names)
        self.assertIn("Q4-2025", names)
        self.assertIn("Jan-Feb-2026", names)
        self.assertIn("W1", names)
        self.assertIn("W2", names)
        self.assertIn("live-parity", names)

    def test_start_before_end(self):
        for name, start, end in self.windows:
            self.assertLess(start, end, f"{name}: start >= end")

    def test_chronological(self):
        starts = [w[1] for w in self.windows]
        self.assertEqual(starts, sorted(starts))


# ─────────────────────────────────────────────────────────────────────────────
# 4. _trade_key() stability
# ─────────────────────────────────────────────────────────────────────────────
class TestTradeKey(unittest.TestCase):

    def setUp(self):
        from scripts.confidence_threshold_ablation import _trade_key
        self.fn = _trade_key

    def test_basic(self):
        t = {"pair": "USD/JPY", "entry_ts": "2025-03-10T09:30:00+00:00"}
        key = self.fn(t)
        self.assertIn("USD/JPY", key)
        self.assertIn("2025031009", key)

    def test_same_trade_same_key(self):
        t = {"pair": "GBP/USD", "entry_ts": "2025-06-01T08:00:00+00:00"}
        self.assertEqual(self.fn(t), self.fn(t))

    def test_different_pairs_different_keys(self):
        t1 = {"pair": "EUR/USD", "entry_ts": "2025-06-01T08:00:00+00:00"}
        t2 = {"pair": "GBP/USD", "entry_ts": "2025-06-01T08:00:00+00:00"}
        self.assertNotEqual(self.fn(t1), self.fn(t2))

    def test_different_hour_different_key(self):
        t1 = {"pair": "USD/CAD", "entry_ts": "2025-06-01T08:00:00+00:00"}
        t2 = {"pair": "USD/CAD", "entry_ts": "2025-06-01T09:00:00+00:00"}
        self.assertNotEqual(self.fn(t1), self.fn(t2))


# ─────────────────────────────────────────────────────────────────────────────
# 5. _r() fallback chain
# ─────────────────────────────────────────────────────────────────────────────
class TestRHelper(unittest.TestCase):

    def setUp(self):
        from scripts.confidence_threshold_ablation import _r
        self.fn = _r

    def test_r_field(self):
        self.assertEqual(self.fn({"r": 2.5}), 2.5)

    def test_realised_r_fallback(self):
        self.assertEqual(self.fn({"realised_r": 1.7}), 1.7)

    def test_result_r_fallback(self):
        self.assertEqual(self.fn({"result_r": -1.0}), -1.0)

    def test_r_takes_priority(self):
        self.assertEqual(self.fn({"r": 3.0, "realised_r": 1.0}), 3.0)

    def test_missing_returns_zero(self):
        self.assertEqual(self.fn({}), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. _unlocked() trade diff
# ─────────────────────────────────────────────────────────────────────────────
class TestUnlocked(unittest.TestCase):

    def setUp(self):
        from scripts.confidence_threshold_ablation import _unlocked
        self.fn = _unlocked

    def _wr(self, trades, var="A", thr=0.77):
        return _fake_wr(var, "Q1-2025", thr, trades=trades)

    def test_no_new_trades(self):
        t = _fake_trade(entry_ts="2025-01-10T09:00:00+00:00")
        base = self._wr([t], "A")
        cmp  = self._wr([t], "B", 0.73)
        self.assertEqual(self.fn(base, cmp), [])

    def test_one_new_trade(self):
        t_base = _fake_trade(entry_ts="2025-01-10T09:00:00+00:00")
        t_new  = _fake_trade(pair="USD/JPY", entry_ts="2025-01-15T09:00:00+00:00")
        base = self._wr([t_base], "A")
        cmp  = self._wr([t_base, t_new], "B", 0.73)
        result = self.fn(base, cmp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pair"], "USD/JPY")

    def test_empty_base(self):
        t = _fake_trade()
        base = self._wr([], "A")
        cmp  = self._wr([t], "B", 0.73)
        self.assertEqual(len(self.fn(base, cmp)), 1)

    def test_empty_cmp_no_unlocked(self):
        t = _fake_trade()
        base = self._wr([t], "A")
        cmp  = self._wr([], "B", 0.73)
        self.assertEqual(self.fn(base, cmp), [])

    def test_multiple_unlocked(self):
        t1 = _fake_trade(entry_ts="2025-01-10T09:00:00+00:00")
        t2 = _fake_trade(pair="USD/CHF", entry_ts="2025-01-20T09:00:00+00:00")
        t3 = _fake_trade(pair="GBP/JPY", entry_ts="2025-01-25T09:00:00+00:00")
        base = self._wr([t1], "A")
        cmp  = self._wr([t1, t2, t3], "B", 0.73)
        result = self.fn(base, cmp)
        self.assertEqual(len(result), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 7. _displaced() logic
# ─────────────────────────────────────────────────────────────────────────────
class TestDisplaced(unittest.TestCase):

    def setUp(self):
        from scripts.confidence_threshold_ablation import _displaced
        self.fn = _displaced

    def test_no_displacement(self):
        t = _fake_trade(entry_ts="2025-01-10T09:00:00+00:00")
        base = _fake_wr("A", "Q1", 0.77, [t])
        cmp  = _fake_wr("B", "Q1", 0.73, [t])
        self.assertEqual(self.fn(base, cmp), [])

    def test_displacement_detected(self):
        # t_base is in A but not in B
        t_base = _fake_trade(entry_ts="2025-01-10T09:00:00+00:00", r=1.5)
        t_repl = _fake_trade(pair="USD/JPY", entry_ts="2025-01-10T10:00:00+00:00", r=2.0)
        base = _fake_wr("A", "Q1", 0.77, [t_base])
        cmp  = _fake_wr("B", "Q1", 0.73, [t_repl])
        result = self.fn(base, cmp)
        self.assertEqual(len(result), 1)
        displaced_trade, replacement = result[0]
        self.assertEqual(displaced_trade["entry_ts"], t_base["entry_ts"])


# ─────────────────────────────────────────────────────────────────────────────
# 8. _flag_unlocked()
# ─────────────────────────────────────────────────────────────────────────────
class TestFlagUnlocked(unittest.TestCase):

    def setUp(self):
        from scripts.confidence_threshold_ablation import _flag_unlocked
        self.fn = _flag_unlocked

    def test_no_flags_clean_trade(self):
        t = _fake_trade(r=1.5, mfe_r=2.0, mae_r=-0.2)
        conf_map = {_trade_key_for(t): 0.74}
        flags = self.fn([t], 10.0, conf_map, 0.73)
        self.assertEqual(flags, [])

    def test_low_conf_loss_flagged(self):
        t = _fake_trade(r=-1.0, mfe_r=0.0, mae_r=-1.0)
        conf_map = {_trade_key_for(t): 0.71}   # < 0.72 AND loss
        flags = self.fn([t], 10.0, conf_map, 0.70)
        self.assertTrue(any("LOW_CONF_LOSS" in f for f in flags))

    def test_low_conf_win_not_flagged(self):
        t = _fake_trade(r=1.5)
        conf_map = {_trade_key_for(t): 0.71}   # < 0.72 but WIN
        flags = self.fn([t], 10.0, conf_map, 0.70)
        self.assertFalse(any("LOW_CONF_LOSS" in f for f in flags))

    def test_high_concentration_flagged(self):
        t = _fake_trade(r=3.0)   # 3/10 = 30% > 20%
        flags = self.fn([t], 10.0, {}, 0.73)
        self.assertTrue(any("HIGH_CONCENTRATION" in f for f in flags))

    def test_concentration_below_threshold_not_flagged(self):
        t = _fake_trade(r=1.5)   # 1.5/10 = 15% < 20%
        flags = self.fn([t], 10.0, {}, 0.73)
        self.assertFalse(any("HIGH_CONCENTRATION" in f for f in flags))

    def test_mae_mfe_pattern_flagged(self):
        # 3 trades: |MAE| > MFE all winners → >50% → flag
        trades = [
            _fake_trade(r=0.5, mfe_r=0.5, mae_r=-1.0),   # |MAE|=1.0 > MFE=0.5 WIN
            _fake_trade(r=0.3, mfe_r=0.4, mae_r=-0.8, entry_ts="2025-01-16T09:00:00+00:00"),
            _fake_trade(r=0.2, mfe_r=0.3, mae_r=-0.9, entry_ts="2025-01-17T09:00:00+00:00"),
        ]
        flags = self.fn(trades, 5.0, {}, 0.73)
        self.assertTrue(any("MAE>MFE" in f for f in flags))

    def test_zero_window_sum_r_no_crash(self):
        t = _fake_trade(r=1.0)
        # window_sum_r=0 → concentration check should not divide by zero
        flags = self.fn([t], 0.0, {}, 0.73)
        # Just ensure no exception
        self.assertIsInstance(flags, list)


def _trade_key_for(t):
    from scripts.confidence_threshold_ablation import _trade_key
    return _trade_key(t)


# ─────────────────────────────────────────────────────────────────────────────
# 9. build_report() smoke test
# ─────────────────────────────────────────────────────────────────────────────
class TestBuildReportSmoke(unittest.TestCase):

    def _make_results(self):
        from scripts.confidence_threshold_ablation import WINDOWS, VARIANTS
        one_win = [WINDOWS[0]]   # Q1-2025 only
        trades_a = [_fake_trade(r=2.0, entry_ts="2025-01-15T09:00:00+00:00")]
        trades_b = [_fake_trade(r=2.0, entry_ts="2025-01-15T09:00:00+00:00"),
                    _fake_trade(pair="USD/CHF", r=1.5, entry_ts="2025-01-20T09:00:00+00:00")]
        trades_c = trades_b + [
            _fake_trade(pair="GBP/JPY", r=-1.0, entry_ts="2025-01-22T09:00:00+00:00")
        ]
        results = {
            "A": [_fake_wr("A", "Q1-2025", 0.77, trades_a)],
            "B": [_fake_wr("B", "Q1-2025", 0.73, trades_b)],
            "C": [_fake_wr("C", "Q1-2025", 0.70, trades_c)],
        }
        return results, one_win

    def test_report_builds(self):
        from scripts.confidence_threshold_ablation import build_report
        results, one_win = self._make_results()
        report = build_report(results, windows=one_win)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)

    def test_report_has_sections(self):
        from scripts.confidence_threshold_ablation import build_report
        results, one_win = self._make_results()
        report = build_report(results, windows=one_win)
        for section in [
            "Aggregate Summary",
            "Per-Window Breakdown",
            "ATR Floor Check",
            "Unlocked Trade Analysis",
            "Cascade Displacement",
            "Confidence Distribution",
            "Pattern Distribution",
            "Verdict",
        ]:
            self.assertIn(section, report, f"Section '{section}' missing from report")

    def test_variant_labels_in_report(self):
        from scripts.confidence_threshold_ablation import build_report
        results, one_win = self._make_results()
        report = build_report(results, windows=one_win)
        for label in ["Variant A", "Variant B", "Variant C"]:
            self.assertIn(label, report)

    def test_threshold_values_in_report(self):
        from scripts.confidence_threshold_ablation import build_report
        results, one_win = self._make_results()
        report = build_report(results, windows=one_win)
        self.assertIn("0.77", report)
        self.assertIn("0.73", report)
        self.assertIn("0.70", report)


# ─────────────────────────────────────────────────────────────────────────────
# 10. build_report() verdict logic
# ─────────────────────────────────────────────────────────────────────────────
class TestVerdictLogic(unittest.TestCase):

    def _report_for(self, a_trades, b_trades, c_trades):
        from scripts.confidence_threshold_ablation import build_report, WINDOWS
        one_win = [WINDOWS[0]]
        results = {
            "A": [_fake_wr("A", "Q1-2025", 0.77, a_trades)],
            "B": [_fake_wr("B", "Q1-2025", 0.73, b_trades)],
            "C": [_fake_wr("C", "Q1-2025", 0.70, c_trades)],
        }
        return build_report(results, windows=one_win)

    def test_reject_when_b_regresses(self):
        """B loses vs A → REJECT."""
        t_a = [_fake_trade(r=2.0, entry_ts="2025-01-15T09:00:00+00:00")]
        t_b = [_fake_trade(r=2.0, entry_ts="2025-01-15T09:00:00+00:00"),
               _fake_trade(pair="USD/CHF", r=-2.5, entry_ts="2025-01-20T09:00:00+00:00")]
        report = self._report_for(t_a, t_b, t_b)
        self.assertIn("REJECT", report)

    def test_no_change_when_same_trades(self):
        """Identical trades for A and B → NO_CHANGE."""
        t = [_fake_trade(r=2.0, entry_ts="2025-01-15T09:00:00+00:00")]
        report = self._report_for(t, t, t)
        self.assertIn("NO_CHANGE", report)

    def test_promote_when_clean_improvement(self):
        """B improves with no flags → PROMOTE.
        Unlocked trade must be < 20% of window SumR to avoid HIGH_CONCENTRATION.
        Baseline has 5× +2R trades → SumR=10R; unlocked +1.5R = 15% < 20%.
        """
        t_shared = [
            _fake_trade(r=2.0, entry_ts=f"2025-01-{10+i:02d}T09:00:00+00:00")
            for i in range(5)
        ]
        t_unlocked = _fake_trade(pair="USD/CHF", r=1.5, mfe_r=2.0, mae_r=-0.2,
                                 entry_ts="2025-01-20T09:00:00+00:00")
        t_a = t_shared
        t_b = t_shared + [t_unlocked]
        report = self._report_for(t_a, t_b, t_b)
        self.assertIn("PROMOTE", report)


# ─────────────────────────────────────────────────────────────────────────────
# 11. ATR floor violation detector
# ─────────────────────────────────────────────────────────────────────────────
class TestATRFloorViolation(unittest.TestCase):

    def test_no_violations(self):
        t = _fake_trade(stop_pips=20)
        wr = _fake_wr("A", "Q1", 0.77, [t])
        self.assertEqual(len(wr.stop_floor_violations()), 0)

    def test_violation_detected(self):
        t = _fake_trade(stop_pips=5)   # < 8 pip floor
        wr = _fake_wr("A", "Q1", 0.77, [t])
        self.assertEqual(len(wr.stop_floor_violations()), 1)

    def test_exactly_8_pips_no_violation(self):
        t = _fake_trade(stop_pips=8)   # exactly 8 — not a violation
        wr = _fake_wr("A", "Q1", 0.77, [t])
        self.assertEqual(len(wr.stop_floor_violations()), 0)

    def test_zero_stop_pips_ignored(self):
        t = _fake_trade(stop_pips=0)   # missing data — not flagged
        wr = _fake_wr("A", "Q1", 0.77, [t])
        self.assertEqual(len(wr.stop_floor_violations()), 0)

    def test_multiple_violations(self):
        trades = [
            _fake_trade(stop_pips=3, entry_ts="2025-01-10T09:00:00+00:00"),
            _fake_trade(stop_pips=20, entry_ts="2025-01-11T09:00:00+00:00"),
            _fake_trade(stop_pips=6, entry_ts="2025-01-12T09:00:00+00:00"),
        ]
        wr = _fake_wr("A", "Q1", 0.77, trades)
        self.assertEqual(len(wr.stop_floor_violations()), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Confidence map lookup
# ─────────────────────────────────────────────────────────────────────────────
class TestConfidenceMap(unittest.TestCase):

    def test_lookup_hit(self):
        ts = "2025-01-15T09:00:00+00:00"
        trade = _fake_trade(pair="USD/JPY", entry_ts=ts)
        decision = {"ts": ts, "pair": "USD/JPY", "decision": "ENTER", "confidence": 0.741}
        wr = _fake_wr("B", "Q1", 0.73, [trade], enter_decisions=[decision])
        from scripts.confidence_threshold_ablation import _conf_for_trade
        result = _conf_for_trade(wr, trade)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.741, places=3)

    def test_lookup_miss_returns_none(self):
        ts = "2025-01-15T09:00:00+00:00"
        trade = _fake_trade(pair="USD/JPY", entry_ts=ts)
        wr = _fake_wr("B", "Q1", 0.73, [trade], enter_decisions=[])
        from scripts.confidence_threshold_ablation import _conf_for_trade
        result = _conf_for_trade(wr, trade)
        self.assertIsNone(result)

    def test_wrong_pair_returns_none(self):
        ts = "2025-01-15T09:00:00+00:00"
        trade = _fake_trade(pair="USD/JPY", entry_ts=ts)
        decision = {"ts": ts, "pair": "GBP/USD", "decision": "ENTER", "confidence": 0.741}
        wr = _fake_wr("B", "Q1", 0.73, [trade], enter_decisions=[decision])
        from scripts.confidence_threshold_ablation import _conf_for_trade
        result = _conf_for_trade(wr, trade)
        self.assertIsNone(result)


# ─────────────────────────────────────────────────────────────────────────────
# 13. atexit not called prematurely
# ─────────────────────────────────────────────────────────────────────────────
class TestAtexitNotPremature(unittest.TestCase):

    def test_module_import_does_not_change_conf(self):
        """Importing the module must not alter MIN_CONFIDENCE."""
        import src.strategy.forex.strategy_config as _sc
        before = _sc.MIN_CONFIDENCE
        import importlib
        import scripts.confidence_threshold_ablation  # noqa — re-import
        self.assertEqual(_sc.MIN_CONFIDENCE, before)


# ─────────────────────────────────────────────────────────────────────────────
# 14. run_all() structure (mocked backtester)
# ─────────────────────────────────────────────────────────────────────────────
class TestRunAllStructure(unittest.TestCase):

    def test_result_keys_and_length(self):
        """run_all() returns {A: [...], B: [...], C: [...]} each with len==WINDOWS."""
        from scripts.confidence_threshold_ablation import WINDOWS, VARIANTS

        fake_result = _FakeResult(trades=[], n_trades=0)

        with patch("scripts.confidence_threshold_ablation.run_backtest",
                   return_value=fake_result) as mock_bt, \
             patch("scripts.confidence_threshold_ablation._load_enter_decisions",
                   return_value=[]):

            from scripts.confidence_threshold_ablation import run_all
            # Only run with one window to keep test fast
            with patch("scripts.confidence_threshold_ablation.WINDOWS", [WINDOWS[0]]):
                results = run_all()

        self.assertIn("A", results)
        self.assertIn("B", results)
        self.assertIn("C", results)
        for var_id, _, _ in VARIANTS:
            self.assertEqual(len(results[var_id]), 1)

    def test_min_confidence_patched_per_variant(self):
        """run_all() must patch MIN_CONFIDENCE to the correct value for each variant."""
        import src.strategy.forex.strategy_config as _sc
        from scripts.confidence_threshold_ablation import WINDOWS, VARIANTS

        captured = []
        fake_result = _FakeResult()

        def fake_run_backtest(**kwargs):
            captured.append(_sc.MIN_CONFIDENCE)
            return fake_result

        with patch("scripts.confidence_threshold_ablation.run_backtest",
                   side_effect=fake_run_backtest), \
             patch("scripts.confidence_threshold_ablation._load_enter_decisions",
                   return_value=[]):

            from scripts.confidence_threshold_ablation import run_all
            with patch("scripts.confidence_threshold_ablation.WINDOWS", [WINDOWS[0]]):
                run_all()

        # 3 variants × 1 window = 3 captured values
        self.assertEqual(len(captured), 3)
        expected_thresholds = {v[1] for v in VARIANTS}
        self.assertEqual(set(captured), expected_thresholds)

    def test_conf_restored_after_each_variant(self):
        """MIN_CONFIDENCE must be restored to _ORIG_CONF after each variant run."""
        import src.strategy.forex.strategy_config as _sc
        from scripts.confidence_threshold_ablation import WINDOWS, _ORIG_CONF

        fake_result = _FakeResult()
        post_run_values = []

        original_run = None

        def fake_run_backtest(**kwargs):
            return fake_result

        # Hook into run_variant to check conf after each call
        import scripts.confidence_threshold_ablation as _mod
        original_run_variant = _mod.run_variant

        def patched_run_variant(*args, **kwargs):
            wr = original_run_variant(*args, **kwargs)
            post_run_values.append(_sc.MIN_CONFIDENCE)
            return wr

        with patch("scripts.confidence_threshold_ablation.run_backtest",
                   side_effect=fake_run_backtest), \
             patch("scripts.confidence_threshold_ablation._load_enter_decisions",
                   return_value=[]), \
             patch("scripts.confidence_threshold_ablation.run_variant",
                   side_effect=patched_run_variant):

            from scripts.confidence_threshold_ablation import run_all
            with patch("scripts.confidence_threshold_ablation.WINDOWS", [WINDOWS[0]]):
                run_all()

        for val in post_run_values:
            self.assertAlmostEqual(val, _ORIG_CONF, places=4,
                                   msg=f"MIN_CONFIDENCE={val} not restored after variant run")


if __name__ == "__main__":
    unittest.main()
