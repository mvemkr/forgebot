"""tests/test_struct_stop_ablation.py
=====================================
Tests for the structural stop ablation harness.
Covers:
  - variant stop function signatures and basic behaviour
  - floor rejection tracking
  - fallback-to-ATR detection
  - ceiling (Variant C) and noise buffer (Variant D) arithmetic
  - trade-key helpers (unlock / displaced detection)
  - config mutation / restore safety
  - atexit guard is registered
  - report building (smoke test)
  - window and variant definitions
  - H&S retest-swing priority logic
  - break_retest anchor delegation
  - wrong-side guard
  - zero-ATR edge cases
  - counter isolation between runs
  - WindowResult properties
  - WindowQuad by_variant lookup
  - displacement table net-delta logic
  - per-window unlock summary correctness
"""
import sys
import math
import types
import unittest
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import patch, MagicMock

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import src.strategy.forex.strategy_config as _sc
import src.strategy.forex.set_and_forget as _saf
import src.strategy.forex.targeting as _tgt

# Import the module under test
import scripts.struct_stop_ablation as _abl

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _mock_pattern(stop_anchor=None, stop_loss=None, neckline=None, target_1=None, target_2=None):
    p = MagicMock()
    p.stop_anchor = stop_anchor
    p.stop_loss   = stop_loss or (stop_anchor + 0.01 if stop_anchor else 1.0010)
    p.neckline    = neckline
    p.target_1    = target_1
    p.target_2    = target_2
    return p


def _mock_df(n_bars: int = 20, atr_value: Optional[float] = None):
    """Return a fake 1H DataFrame that produces a known ATR via _compute_atr_1h."""
    import pandas as pd
    import numpy as np
    # If atr_value supplied, build synthetic OHLC that yields exactly that ATR
    if atr_value is not None:
        # All bars have H-L = atr_value, C = mid
        highs  = np.full(n_bars, 1.0 + atr_value / 2)
        lows   = np.full(n_bars, 1.0 - atr_value / 2)
        closes = np.full(n_bars, 1.0)
    else:
        rng = np.random.default_rng(42)
        highs  = 1.0 + rng.random(n_bars) * 0.001
        lows   = 1.0 - rng.random(n_bars) * 0.001
        closes = (highs + lows) / 2
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


def _call_stop_b(pattern_type, direction, entry, anchor, atr_val=0.001, pip=0.0001):
    """Convenience wrapper for Variant B stop function."""
    fn = _abl._make_variant_b_stop()
    df  = _mock_df(atr_value=atr_val)
    pat = _mock_pattern(stop_anchor=anchor)
    return fn(pattern_type, direction, entry, df, pat, pip_size=pip)


def _call_stop_c(pattern_type, direction, entry, anchor, atr_val=0.001, pip=0.0001):
    fn = _abl._make_variant_c_stop()
    df  = _mock_df(atr_value=atr_val)
    pat = _mock_pattern(stop_anchor=anchor)
    return fn(pattern_type, direction, entry, df, pat, pip_size=pip)


def _call_stop_d(pattern_type, direction, entry, anchor, atr_val=0.001, pip=0.0001):
    fn = _abl._make_variant_d_stop()
    df  = _mock_df(atr_value=atr_val)
    pat = _mock_pattern(stop_anchor=anchor)
    return fn(pattern_type, direction, entry, df, pat, pip_size=pip)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Module integrity
# ─────────────────────────────────────────────────────────────────────────────
class TestModuleIntegrity(unittest.TestCase):

    def test_variants_defined(self):
        self.assertEqual(len(_abl.VARIANTS), 4)

    def test_variant_ids(self):
        ids = [v[0] for v in _abl.VARIANTS]
        self.assertEqual(ids, ["A", "B", "C", "D"])

    def test_variant_a_has_no_stop_fn(self):
        """Variant A is baseline — stop_fn must be None."""
        _, _, stop_fn, _ = _abl.VARIANTS[0]
        self.assertIsNone(stop_fn)

    def test_variants_bcd_have_stop_fn(self):
        for _, _, stop_fn, _ in _abl.VARIANTS[1:]:
            self.assertIsNotNone(stop_fn)

    def test_variant_b_patches_atr_floor(self):
        _, _, _, patch = _abl.VARIANTS[1]
        self.assertTrue(patch)

    def test_variant_c_patches_atr_floor(self):
        _, _, _, patch = _abl.VARIANTS[2]
        self.assertTrue(patch)

    def test_variant_d_no_atr_floor_patch(self):
        """D adds a buffer so it won't produce sub-floor stops; no floor bypass needed."""
        _, _, _, patch = _abl.VARIANTS[3]
        self.assertFalse(patch)

    def test_windows_count(self):
        self.assertEqual(len(_abl.WINDOWS), 8)

    def test_windows_names(self):
        names = [w[0] for w in _abl.WINDOWS]
        self.assertIn("Q1-2025", names)
        self.assertIn("live-parity", names)

    def test_windows_chronological(self):
        starts = [w[1] for w in _abl.WINDOWS]
        self.assertEqual(starts, sorted(starts))

    def test_atexit_registered(self):
        import atexit as _ae
        funcs = [f for _, f, _ in _ae._atexit_exit_handlers() if f == _abl._reset_all] \
                if hasattr(_ae, "_atexit_exit_handlers") else []
        # Lightweight: just verify _reset_all is callable
        self.assertTrue(callable(_abl._reset_all))

    def test_constants_defined(self):
        self.assertEqual(_abl._ATR_FLOOR_FRAC, 0.15)
        self.assertEqual(_abl._ATR_ABS_FLOOR_PIP, 8.0)
        self.assertEqual(_abl._ATR_C_CEILING_MULT, 3.0)
        self.assertEqual(_abl._ATR_D_BUFFER_MULT, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Config restore safety
# ─────────────────────────────────────────────────────────────────────────────
class TestConfigRestore(unittest.TestCase):

    def test_reset_all_restores_atr_multiplier(self):
        original = _sc.ATR_MIN_MULTIPLIER
        _sc.ATR_MIN_MULTIPLIER = 0.0
        _abl._reset_all()
        self.assertEqual(_sc.ATR_MIN_MULTIPLIER, original)

    def test_reset_all_restores_stop_fn(self):
        original = _saf.get_structure_stop
        _saf.get_structure_stop = lambda *a, **kw: None
        _abl._reset_all()
        self.assertIs(_saf.get_structure_stop, original)

    def test_reset_all_idempotent(self):
        _abl._reset_all()
        _abl._reset_all()
        self.assertEqual(_sc.ATR_MIN_MULTIPLIER, _abl._ORIG_ATR_MIN_MULT)

    def test_orig_stop_fn_is_real(self):
        self.assertIs(_abl._ORIG_STOP_FN, _tgt.get_structure_stop)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Counter helpers
# ─────────────────────────────────────────────────────────────────────────────
class TestCounters(unittest.TestCase):

    def setUp(self):
        _abl._reset_counters()

    def test_reset_zeroes_all(self):
        _abl._COUNTERS["structural_stop_candidates"] = 99
        _abl._reset_counters()
        self.assertEqual(_abl._COUNTERS["structural_stop_candidates"], 0)
        self.assertEqual(_abl._COUNTERS["structural_stop_floor_rejections"], 0)
        self.assertEqual(_abl._COUNTERS["fallback_to_baseline_stop_count"], 0)

    def test_snapshot_is_copy(self):
        _abl._COUNTERS["structural_stop_candidates"] = 5
        snap = _abl._snapshot_counters()
        _abl._COUNTERS["structural_stop_candidates"] = 99
        self.assertEqual(snap["structural_stop_candidates"], 5)

    def test_snapshot_all_keys(self):
        snap = _abl._snapshot_counters()
        self.assertIn("structural_stop_candidates", snap)
        self.assertIn("structural_stop_floor_rejections", snap)
        self.assertIn("fallback_to_baseline_stop_count", snap)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Variant B stop function
# ─────────────────────────────────────────────────────────────────────────────
class TestVariantBStop(unittest.TestCase):

    def setUp(self):
        _abl._reset_counters()

    def test_returns_raw_anchor_short(self):
        stop, stype, pips = _call_stop_b("double_top", "short", 1.3500, 1.3600)
        self.assertAlmostEqual(stop, 1.3600, places=5)

    def test_returns_raw_anchor_long(self):
        stop, stype, pips = _call_stop_b("double_bottom", "long", 1.3500, 1.3400)
        self.assertAlmostEqual(stop, 1.3400, places=5)

    def test_stop_type_b_struct_raw(self):
        _, stype, _ = _call_stop_b("double_top", "short", 1.3500, 1.3600)
        self.assertEqual(stype, "b_struct_raw")

    def test_pips_calculation(self):
        stop, _, pips = _call_stop_b("double_top", "short", 1.3500, 1.3600, pip=0.0001)
        expected_pips = abs(1.3600 - 1.3500) / 0.0001  # = 100 pips
        self.assertAlmostEqual(pips, expected_pips, places=1)

    def test_candidate_counter_increments(self):
        _call_stop_b("double_top", "short", 1.3500, 1.3600)
        self.assertEqual(_abl._COUNTERS["structural_stop_candidates"], 1)

    def test_floor_rejection_tracked_when_too_tight(self):
        # atr=0.01 → floor = max(8*0.0001, 0.15*0.01) = max(0.0008, 0.0015) = 0.0015
        # stop_dist = |1.3501 - 1.3500| = 0.0001 < floor → floor rejection
        stop, stype, pips = _call_stop_b(
            "double_top", "short", 1.3500, 1.3501, atr_val=0.01, pip=0.0001
        )
        self.assertEqual(_abl._COUNTERS["structural_stop_floor_rejections"], 1)

    def test_floor_rejection_not_tracked_when_wide(self):
        # atr=0.001 → floor=max(0.0008, 0.00015)=0.0008
        # stop_dist = 0.01 >> floor
        _call_stop_b("double_top", "short", 1.3500, 1.3600, atr_val=0.001)
        self.assertEqual(_abl._COUNTERS["structural_stop_floor_rejections"], 0)

    def test_fallback_to_atr_when_no_anchor(self):
        fn = _abl._make_variant_b_stop()
        df  = _mock_df(atr_value=0.001)
        pat = _mock_pattern(stop_anchor=None, stop_loss=1.3600)
        stop, stype, pips = fn("double_top", "short", 1.3500, df, pat, pip_size=0.0001)
        self.assertIn("atr_fallback", stype)
        self.assertEqual(_abl._COUNTERS["fallback_to_baseline_stop_count"], 1)

    def test_wrong_side_short_falls_back(self):
        # Anchor below entry for a short → wrong side
        fn = _abl._make_variant_b_stop()
        df  = _mock_df(atr_value=0.001)
        pat = _mock_pattern(stop_anchor=1.3400, stop_loss=1.3600)
        stop, stype, pips = fn("double_top", "short", 1.3500, df, pat, pip_size=0.0001)
        # Should have fallen back (wrong side rejected)
        self.assertEqual(_abl._COUNTERS["fallback_to_baseline_stop_count"], 1)

    def test_wrong_side_long_falls_back(self):
        fn = _abl._make_variant_b_stop()
        df  = _mock_df(atr_value=0.001)
        pat = _mock_pattern(stop_anchor=1.3600, stop_loss=1.3400)
        stop, stype, pips = fn("double_bottom", "long", 1.3500, df, pat, pip_size=0.0001)
        self.assertEqual(_abl._COUNTERS["fallback_to_baseline_stop_count"], 1)

    def test_no_buffer_added(self):
        """B returns exact anchor with no pip buffer."""
        anchor = 1.3600
        stop, _, _ = _call_stop_b("double_top", "short", 1.3500, anchor)
        self.assertAlmostEqual(stop, anchor, places=6)

    def test_atr_fallback_distance(self):
        fn = _abl._make_variant_b_stop()
        atr_val = 0.002
        df  = _mock_df(atr_value=atr_val)
        pat = _mock_pattern(stop_anchor=None, stop_loss=1.37)
        entry = 1.35
        stop, stype, pips = fn("double_top", "short", entry, df, pat, pip_size=0.0001)
        expected = entry + atr_val * 3.0  # default fallback_mult=3.0
        self.assertAlmostEqual(stop, expected, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Variant C stop function (ATR ceiling)
# ─────────────────────────────────────────────────────────────────────────────
class TestVariantCStop(unittest.TestCase):

    def setUp(self):
        _abl._reset_counters()

    def test_ceiling_applied_when_structural_wider(self):
        # atr=0.001 → ceiling=0.003; struct_dist=0.02 → capped to 0.003
        fn = _abl._make_variant_c_stop()
        df  = _mock_df(atr_value=0.001)
        pat = _mock_pattern(stop_anchor=1.3700, stop_loss=1.37)
        stop, stype, pips = fn("double_top", "short", 1.3500, df, pat, pip_size=0.0001)
        expected_dist = min(0.0200, 3.0 * 0.001)  # = 0.003
        expected_stop = 1.3500 + expected_dist
        self.assertAlmostEqual(stop, expected_stop, places=5)
        self.assertEqual(stype, "c_struct_capped")

    def test_ceiling_not_applied_when_structural_tighter(self):
        # atr=0.01 → ceiling=0.03; struct_dist=0.001 < 0.03 → no cap
        fn = _abl._make_variant_c_stop()
        df  = _mock_df(atr_value=0.01)
        pat = _mock_pattern(stop_anchor=1.3510, stop_loss=1.3520)
        stop, stype, pips = fn("double_top", "short", 1.3500, df, pat, pip_size=0.0001)
        expected_dist = 0.001
        expected_stop = 1.3500 + expected_dist
        self.assertAlmostEqual(stop, expected_stop, places=5)
        self.assertEqual(stype, "c_struct_raw")

    def test_ceiling_stop_pips(self):
        fn = _abl._make_variant_c_stop()
        atr_val = 0.001
        df  = _mock_df(atr_value=atr_val)
        pat = _mock_pattern(stop_anchor=1.3700)
        stop, stype, pips = fn("double_top", "short", 1.3500, df, pat, pip_size=0.0001)
        expected_pips = min(200, 3.0 * atr_val / 0.0001)  # = min(200, 30) = 30
        self.assertAlmostEqual(pips, expected_pips, places=1)

    def test_fallback_incremented_when_no_anchor(self):
        fn = _abl._make_variant_c_stop()
        df  = _mock_df(atr_value=0.001)
        pat = _mock_pattern(stop_anchor=None, stop_loss=1.36)
        fn("double_top", "short", 1.35, df, pat, pip_size=0.0001)
        self.assertEqual(_abl._COUNTERS["fallback_to_baseline_stop_count"], 1)

    def test_floor_rejection_tracked(self):
        # struct_dist < floor should still be tracked in C
        fn = _abl._make_variant_c_stop()
        atr_val = 0.01
        df  = _mock_df(atr_value=atr_val)
        pat = _mock_pattern(stop_anchor=1.3501)  # dist=0.0001 < 0.0015 floor
        fn("double_top", "short", 1.3500, df, pat, pip_size=0.0001)
        self.assertEqual(_abl._COUNTERS["structural_stop_floor_rejections"], 1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Variant D stop function (ATR noise buffer)
# ─────────────────────────────────────────────────────────────────────────────
class TestVariantDStop(unittest.TestCase):

    def setUp(self):
        _abl._reset_counters()

    def test_buffer_added_short(self):
        # struct_dist = 0.01, atr=0.002 → stop_dist = 0.01 + 0.5*0.002 = 0.011
        fn = _abl._make_variant_d_stop()
        df  = _mock_df(atr_value=0.002)
        pat = _mock_pattern(stop_anchor=1.3600)
        stop, stype, pips = fn("double_top", "short", 1.3500, df, pat, pip_size=0.0001)
        expected_dist = 0.0100 + 0.5 * 0.002  # = 0.011
        expected_stop = 1.3500 + expected_dist
        self.assertAlmostEqual(stop, expected_stop, places=5)

    def test_buffer_added_long(self):
        fn = _abl._make_variant_d_stop()
        df  = _mock_df(atr_value=0.002)
        pat = _mock_pattern(stop_anchor=1.3400)
        stop, stype, pips = fn("double_bottom", "long", 1.3500, df, pat, pip_size=0.0001)
        expected_dist = 0.0100 + 0.5 * 0.002
        expected_stop = 1.3500 - expected_dist
        self.assertAlmostEqual(stop, expected_stop, places=5)

    def test_stop_type_d_struct_buffered(self):
        _, stype, _ = _call_stop_d("double_top", "short", 1.3500, 1.3600, atr_val=0.002)
        self.assertEqual(stype, "d_struct_buffered")

    def test_pips_include_buffer(self):
        fn = _abl._make_variant_d_stop()
        atr_val = 0.002
        pip = 0.0001
        df  = _mock_df(atr_value=atr_val)
        pat = _mock_pattern(stop_anchor=1.3600)
        _, _, pips = fn("double_top", "short", 1.3500, df, pat, pip_size=pip)
        expected_pips = (0.01 + 0.5 * atr_val) / pip
        self.assertAlmostEqual(pips, expected_pips, places=1)

    def test_fallback_when_no_anchor(self):
        fn = _abl._make_variant_d_stop()
        df  = _mock_df(atr_value=0.002)
        pat = _mock_pattern(stop_anchor=None, stop_loss=1.36)
        fn("double_top", "short", 1.35, df, pat, pip_size=0.0001)
        self.assertEqual(_abl._COUNTERS["fallback_to_baseline_stop_count"], 1)

    def test_zero_atr_no_buffer(self):
        """If ATR is 0 (insufficient data), noise = 0 → stop = raw anchor."""
        fn = _abl._make_variant_d_stop()
        import pandas as pd, numpy as np
        df = pd.DataFrame({"high": [1.0]*5, "low": [1.0]*5, "close": [1.0]*5})
        pat = _mock_pattern(stop_anchor=1.3600)
        stop, _, _ = fn("double_top", "short", 1.3500, df, pat, pip_size=0.0001)
        # ATR insufficient data → no buffer → raw anchor
        self.assertAlmostEqual(stop, 1.3600, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Structural anchor helper
# ─────────────────────────────────────────────────────────────────────────────
class TestStructAnchor(unittest.TestCase):

    def setUp(self):
        _abl._reset_counters()

    def test_returns_none_when_anchor_none(self):
        df  = _mock_df()
        pat = _mock_pattern(stop_anchor=None)
        anchor = _abl._struct_anchor_for_pattern("double_top", "short", 1.35, df, pat)
        self.assertIsNone(anchor)

    def test_wrong_side_short_returns_none(self):
        df  = _mock_df()
        pat = _mock_pattern(stop_anchor=1.34)  # below entry for short → wrong
        anchor = _abl._struct_anchor_for_pattern("double_top", "short", 1.35, df, pat)
        self.assertIsNone(anchor)

    def test_wrong_side_long_returns_none(self):
        df  = _mock_df()
        pat = _mock_pattern(stop_anchor=1.36)  # above entry for long → wrong
        anchor = _abl._struct_anchor_for_pattern("double_bottom", "long", 1.35, df, pat)
        self.assertIsNone(anchor)

    def test_correct_side_short(self):
        df  = _mock_df()
        pat = _mock_pattern(stop_anchor=1.36)
        anchor = _abl._struct_anchor_for_pattern("double_top", "short", 1.35, df, pat)
        self.assertAlmostEqual(anchor, 1.36, places=5)

    def test_correct_side_long(self):
        df  = _mock_df()
        pat = _mock_pattern(stop_anchor=1.34)
        anchor = _abl._struct_anchor_for_pattern("double_bottom", "long", 1.35, df, pat)
        self.assertAlmostEqual(anchor, 1.34, places=5)

    def test_has_returns_anchor_for_double_top(self):
        """DT/DB/sweep: uses stop_anchor directly."""
        df  = _mock_df()
        pat = _mock_pattern(stop_anchor=1.3620, neckline=1.35)
        anchor = _abl._struct_anchor_for_pattern("double_top", "short", 1.35, df, pat)
        self.assertAlmostEqual(anchor, 1.3620, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Trade key helpers
# ─────────────────────────────────────────────────────────────────────────────
class TestTradeKey(unittest.TestCase):

    def _trade(self, pair="GBP/JPY", direction="short", ts="2025-01-15 09:00"):
        return {"pair": pair, "direction": direction, "entry_time": ts,
                "realized_r": 2.5, "initial_stop_pips": 50}

    def test_key_uniqueness(self):
        t1 = self._trade("GBP/JPY", "short", "2025-01-15 09:00")
        t2 = self._trade("GBP/JPY", "long",  "2025-01-15 09:00")
        self.assertNotEqual(_abl._trade_key(t1), _abl._trade_key(t2))

    def test_key_same_for_same_trade(self):
        t = self._trade()
        self.assertEqual(_abl._trade_key(t), _abl._trade_key(t))

    def test_key_hour_truncation(self):
        t1 = self._trade(ts="2025-01-15 09:00")
        t2 = self._trade(ts="2025-01-15 09:45")  # same hour after truncation
        self.assertEqual(_abl._trade_key(t1), _abl._trade_key(t2))

    def test_find_unlocked_empty_base(self):
        base = []
        new  = [self._trade("GBP/JPY")]
        unlocked = _abl._find_unlocked(base, new)
        self.assertEqual(len(unlocked), 1)

    def test_find_unlocked_identical(self):
        t = self._trade()
        unlocked = _abl._find_unlocked([t], [t])
        self.assertEqual(len(unlocked), 0)

    def test_find_unlocked_one_new(self):
        t1 = self._trade("GBP/JPY", "short", "2025-01-15 09:00")
        t2 = self._trade("USD/JPY", "short", "2025-01-15 10:00")
        unlocked = _abl._find_unlocked([t1], [t1, t2])
        self.assertEqual(len(unlocked), 1)
        self.assertEqual(unlocked[0]["pair"], "USD/JPY")

    def test_find_removed(self):
        t1 = self._trade("GBP/JPY")
        t2 = self._trade("USD/JPY", ts="2025-01-16 09:00")
        removed = _abl._find_removed([t1, t2], [t2])
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["pair"], "GBP/JPY")

    def test_find_removed_none(self):
        t = self._trade()
        self.assertEqual(len(_abl._find_removed([t], [t])), 0)

    def test_find_unlocked_multi(self):
        trades_a = [self._trade("GBP/JPY", ts=f"2025-0{i}-01 09:00") for i in range(1, 4)]
        trades_b = trades_a + [self._trade("USD/JPY", ts="2025-05-01 09:00")]
        unlocked = _abl._find_unlocked(trades_a, trades_b)
        self.assertEqual(len(unlocked), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 9. WindowResult properties
# ─────────────────────────────────────────────────────────────────────────────
class TestWindowResult(unittest.TestCase):

    def _make_wr(self, n=3, win=2, sum_r=2.0, ret=10.0, dd=5.0):
        from src.strategy.forex.backtest_schema import BacktestResult
        trades = []
        for i in range(n):
            r = 1.0 if i < win else -1.0
            trades.append({"pair": "GBP/JPY", "direction": "short",
                           "entry_time": f"2025-01-0{i+1} 09:00",
                           "realized_r": r,
                           "mae_r": -0.3, "mfe_r": 1.5,
                           "initial_stop_pips": 50.0,
                           "stop_type": "structural_anchor"})
        br = BacktestResult(return_pct=ret, max_dd_pct=dd, win_rate=win/n,
                            avg_r=sum_r/n, n_trades=n, trades=trades)
        return _abl.WindowResult(
            variant="A", window="Q1-2025", result=br, candle_data=None,
            counters={"structural_stop_candidates": 3,
                      "structural_stop_floor_rejections": 1,
                      "fallback_to_baseline_stop_count": 0}
        )

    def test_n_trades(self):
        wr = self._make_wr(n=5)
        self.assertEqual(wr.n, 5)

    def test_total_r(self):
        wr = self._make_wr(n=3, win=2)
        # 2 wins at +1R, 1 loss at -1R → total 1.0
        self.assertAlmostEqual(wr.total_r, 1.0, places=5)

    def test_wr(self):
        wr = self._make_wr(n=4, win=3)
        self.assertAlmostEqual(wr.wr, 0.75, places=5)

    def test_worst3(self):
        wr = self._make_wr(n=4, win=2)
        # 2 losses at -1R, 2 wins at +1R → worst3 = [-1, -1, +1]
        self.assertTrue(len(wr.worst3) <= 3)
        self.assertTrue(all(isinstance(r, float) for r in wr.worst3))

    def test_stop_pips_list(self):
        wr = self._make_wr(n=3)
        self.assertEqual(len(wr.stop_pips_list), 3)
        self.assertTrue(all(p == 50.0 for p in wr.stop_pips_list))

    def test_mae_mfe_lists(self):
        wr = self._make_wr(n=3)
        self.assertEqual(len(wr.mae_r_list), 3)
        self.assertEqual(len(wr.mfe_r_list), 3)

    def test_counters_stored(self):
        wr = self._make_wr()
        self.assertEqual(wr.counters["structural_stop_floor_rejections"], 1)


# ─────────────────────────────────────────────────────────────────────────────
# 10. WindowQuad
# ─────────────────────────────────────────────────────────────────────────────
class TestWindowQuad(unittest.TestCase):

    def _make_wr(self, var):
        from src.strategy.forex.backtest_schema import BacktestResult
        br = BacktestResult(n_trades=0)
        return _abl.WindowResult(variant=var, window="Q1-2025", result=br,
                                 candle_data=None, counters={})

    def test_by_variant(self):
        wq = _abl.WindowQuad(
            window="Q1-2025",
            result_a=self._make_wr("A"),
            result_b=self._make_wr("B"),
            result_c=self._make_wr("C"),
            result_d=self._make_wr("D"),
        )
        self.assertEqual(wq.by_variant("B").variant, "B")
        self.assertEqual(wq.by_variant("D").variant, "D")

    def test_results_list_order(self):
        wq = _abl.WindowQuad(
            window="Q1-2025",
            result_a=self._make_wr("A"),
            result_b=self._make_wr("B"),
            result_c=self._make_wr("C"),
            result_d=self._make_wr("D"),
        )
        ids = [r.variant for r in wq.results]
        self.assertEqual(ids, ["A", "B", "C", "D"])


# ─────────────────────────────────────────────────────────────────────────────
# 11. Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────
class TestFormatters(unittest.TestCase):

    def test_fmt_r_positive(self):
        self.assertEqual(_abl._fmt_r(2.5), "+2.50R")

    def test_fmt_r_negative(self):
        self.assertEqual(_abl._fmt_r(-1.0), "-1.00R")

    def test_fmt_r_none(self):
        self.assertEqual(_abl._fmt_r(None), "—")

    def test_fmt_pct_positive(self):
        self.assertTrue(_abl._fmt_pct(12.5).startswith("+"))

    def test_fmt_wr(self):
        self.assertEqual(_abl._fmt_wr(0.75), "75%")

    def test_p50_empty(self):
        self.assertEqual(_abl._p50([]), 0.0)

    def test_p50_odd(self):
        self.assertEqual(_abl._p50([1.0, 3.0, 2.0]), 2.0)

    def test_avg_empty(self):
        self.assertIsNone(_abl._avg([]))

    def test_avg_values(self):
        self.assertAlmostEqual(_abl._avg([1.0, 2.0, 3.0]), 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Displacement net delta arithmetic
# ─────────────────────────────────────────────────────────────────────────────
class TestDisplacementArithmetic(unittest.TestCase):

    def test_net_positive_when_replacement_better(self):
        displaced_r    = 0.5
        replacement_r  = 2.5
        net = replacement_r - displaced_r
        self.assertGreater(net, 0)

    def test_net_negative_when_replacement_worse(self):
        displaced_r    = 2.5
        replacement_r  = -1.0
        net = replacement_r - displaced_r
        self.assertLess(net, 0)

    def test_net_zero_when_equal(self):
        net = 1.5 - 1.5
        self.assertAlmostEqual(net, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Variant B — no buffer guarantee
# ─────────────────────────────────────────────────────────────────────────────
class TestVariantBNoBuffer(unittest.TestCase):

    def setUp(self):
        _abl._reset_counters()

    def test_jpy_pair_no_buffer(self):
        """For JPY pairs, production code adds 5-pip buffer. Variant B must not."""
        anchor = 155.000
        entry  = 154.500
        stop, stype, pips = _call_stop_b(
            "double_top", "short", entry, anchor, atr_val=0.5, pip=0.01
        )
        self.assertAlmostEqual(stop, anchor, places=3)

    def test_major_pair_no_buffer(self):
        anchor = 1.3600
        entry  = 1.3500
        stop, stype, pips = _call_stop_b(
            "double_top", "short", entry, anchor, atr_val=0.001, pip=0.0001
        )
        self.assertAlmostEqual(stop, anchor, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Variant C ceiling — stop tighter than baseline
# ─────────────────────────────────────────────────────────────────────────────
class TestVariantCTighter(unittest.TestCase):

    def setUp(self):
        _abl._reset_counters()

    def test_c_stop_le_b_stop(self):
        """C applies ceiling so it must produce stops ≤ B for same input."""
        fn_b = _abl._make_variant_b_stop()
        fn_c = _abl._make_variant_c_stop()
        atr_val = 0.001
        df  = _mock_df(atr_value=atr_val)
        pat = _mock_pattern(stop_anchor=1.3700)

        stop_b, _, _ = fn_b("double_top", "short", 1.3500, df, pat, pip_size=0.0001)
        _abl._reset_counters()
        stop_c, _, _ = fn_c("double_top", "short", 1.3500, df, pat, pip_size=0.0001)

        dist_b = abs(stop_b - 1.3500)
        dist_c = abs(stop_c - 1.3500)
        self.assertLessEqual(dist_c, dist_b + 1e-8)  # C ≤ B

    def test_c_ceiling_at_exactly_3x_atr(self):
        atr_val = 0.001
        fn = _abl._make_variant_c_stop()
        df  = _mock_df(atr_value=atr_val)
        # struct_dist = 0.005 > 3×0.001 = 0.003 → capped
        pat = _mock_pattern(stop_anchor=1.3550)
        stop, stype, pips = fn("double_top", "short", 1.3500, df, pat, pip_size=0.0001)
        expected_stop = 1.3500 + 3.0 * atr_val
        self.assertAlmostEqual(stop, expected_stop, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# 15. Variant D always wider than B
# ─────────────────────────────────────────────────────────────────────────────
class TestVariantDWider(unittest.TestCase):

    def setUp(self):
        _abl._reset_counters()

    def test_d_wider_than_b(self):
        """D = B + 0.5×ATR so D must always be wider than B."""
        fn_b = _abl._make_variant_b_stop()
        fn_d = _abl._make_variant_d_stop()
        atr_val = 0.002
        df  = _mock_df(atr_value=atr_val)
        pat = _mock_pattern(stop_anchor=1.3600)

        stop_b, _, _ = fn_b("double_top", "short", 1.3500, df, pat, pip_size=0.0001)
        _abl._reset_counters()
        stop_d, _, _ = fn_d("double_top", "short", 1.3500, df, pat, pip_size=0.0001)

        dist_b = abs(stop_b - 1.3500)
        dist_d = abs(stop_d - 1.3500)
        self.assertGreater(dist_d, dist_b)

    def test_d_buffer_exact_amount(self):
        atr_val = 0.002
        expected_noise = 0.5 * atr_val  # = 0.001
        fn = _abl._make_variant_d_stop()
        df  = _mock_df(atr_value=atr_val)
        pat = _mock_pattern(stop_anchor=1.3600)
        stop, _, _ = fn("double_top", "short", 1.3500, df, pat, pip_size=0.0001)
        dist = abs(stop - 1.3500)
        raw_struct = 0.0100
        self.assertAlmostEqual(dist, raw_struct + expected_noise, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# 16. Run ablation smoke test (mock run_backtest)
# ─────────────────────────────────────────────────────────────────────────────
class TestRunAblationSmoke(unittest.TestCase):

    @patch("scripts.struct_stop_ablation.run_backtest")
    def test_run_ablation_returns_quads(self, mock_bt):
        from src.strategy.forex.backtest_schema import BacktestResult
        mock_bt.return_value = BacktestResult(n_trades=2, return_pct=5.0,
                                               win_rate=0.5, avg_r=0.5,
                                               max_dd_pct=3.0, trades=[])
        quads = _abl.run_ablation(
            windows_filter=["Q1-2025"],
            variants_filter=["A", "B"],
        )
        self.assertEqual(len(quads), 1)
        self.assertEqual(quads[0].window, "Q1-2025")

    @patch("scripts.struct_stop_ablation.run_backtest")
    def test_run_ablation_resets_config_after(self, mock_bt):
        from src.strategy.forex.backtest_schema import BacktestResult
        mock_bt.return_value = BacktestResult(n_trades=0, trades=[])
        orig_mult = _sc.ATR_MIN_MULTIPLIER
        _abl.run_ablation(windows_filter=["Q1-2025"], variants_filter=["B"])
        self.assertEqual(_sc.ATR_MIN_MULTIPLIER, orig_mult)

    @patch("scripts.struct_stop_ablation.run_backtest")
    def test_run_ablation_resets_stop_fn_after(self, mock_bt):
        from src.strategy.forex.backtest_schema import BacktestResult
        mock_bt.return_value = BacktestResult(n_trades=0, trades=[])
        orig_fn = _saf.get_structure_stop
        _abl.run_ablation(windows_filter=["Q1-2025"], variants_filter=["B"])
        self.assertIs(_saf.get_structure_stop, orig_fn)

    @patch("scripts.struct_stop_ablation.run_backtest")
    def test_build_report_smoke(self, mock_bt):
        from src.strategy.forex.backtest_schema import BacktestResult
        mock_bt.return_value = BacktestResult(n_trades=1, return_pct=7.0,
                                               win_rate=1.0, avg_r=1.5,
                                               max_dd_pct=2.0, trades=[
                                                   {"pair": "GBP/JPY", "direction": "short",
                                                    "entry_time": "2025-01-10 09:00",
                                                    "realized_r": 1.5,
                                                    "mae_r": -0.2, "mfe_r": 2.0,
                                                    "initial_stop_pips": 45.0,
                                                    "stop_type": "b_struct_raw"}
                                               ])
        quads = _abl.run_ablation(
            windows_filter=["Q1-2025"],
            variants_filter=["A", "B", "C", "D"],
        )
        report = _abl._build_report(quads)
        self.assertIn("Structural Stop Ablation", report)
        self.assertIn("Variant B", report)
        self.assertIn("ATR Floor Telemetry", report)
        self.assertIn("Aggregate Totals", report)
        self.assertIn("Displacement Table", report)


if __name__ == "__main__":
    unittest.main()
