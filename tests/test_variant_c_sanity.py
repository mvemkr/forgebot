"""tests/test_variant_c_sanity.py
===================================
Tests for Variant C sanity confirmation harness.
Covers: stop functions, floor enforcement, outlier audit,
tiny-stop counts, realism flags, robustness, report smoke.
"""
import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import src.strategy.forex.strategy_config as _sc
import src.strategy.forex.set_and_forget as _saf
import scripts.variant_c_sanity as _s


# ── Helpers ───────────────────────────────────────────────────────────────────
def _mock_df(atr_val=0.001, n=20):
    import pandas as pd, numpy as np
    highs  = np.full(n, 1.0 + atr_val / 2)
    lows   = np.full(n, 1.0 - atr_val / 2)
    closes = np.full(n, 1.0)
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})

def _mock_pat(anchor=None, neckline=None, stop_loss=1.37):
    p = MagicMock()
    p.stop_anchor = anchor
    p.neckline    = neckline
    p.stop_loss   = stop_loss
    return p

def _trade(pair="GBP/JPY", r=1.5, stop_pips=30.0, pattern="head_and_shoulders",
           direction="short", ts="2025-01-15 09:00", rr=3.0, mae=-0.3, mfe=2.0,
           stop_type="c_struct_raw"):
    return {"pair": pair, "direction": direction, "r": r,
            "initial_stop_pips": stop_pips, "pattern": pattern,
            "entry_ts": ts, "planned_rr": rr,
            "mae_r": mae, "mfe_r": mfe, "stop_type": stop_type}


# ── 1. Variant C stop function ────────────────────────────────────────────────
class TestVariantCStop(unittest.TestCase):

    def setUp(self):
        self._fn = _s._make_c_stop()

    def _call(self, entry, anchor, atr_val=0.001, pip=0.0001, direction="short"):
        df  = _mock_df(atr_val)
        pat = _mock_pat(anchor=anchor)
        return self._fn("double_top", direction, entry, df, pat, pip_size=pip)

    def test_ceiling_applied(self):
        # atr=0.001, ceil=0.003; struct_dist=0.02 → capped at 0.003
        stop, stype, pips = self._call(1.3500, 1.3700, atr_val=0.001)
        self.assertAlmostEqual(stop, 1.3500 + 3 * 0.001, places=5)
        self.assertEqual(stype, "c_struct_capped")

    def test_no_ceiling_when_tight(self):
        # atr=0.01, ceil=0.03; struct_dist=0.001 < 0.03 → raw
        stop, stype, pips = self._call(1.3500, 1.3510, atr_val=0.01)
        self.assertAlmostEqual(stop, 1.3510, places=5)
        self.assertEqual(stype, "c_struct_raw")

    def test_no_floor_in_c(self):
        """C does NOT enforce an 8-pip floor — that's C8."""
        # struct_dist = 2 pips (< 8), no floor should be applied
        stop, stype, pips = self._call(1.3500, 1.3502, atr_val=0.001)
        # dist = 0.0002 → pips = 2.0
        self.assertAlmostEqual(pips, 2.0, places=1)

    def test_fallback_when_no_anchor(self):
        df  = _mock_df(0.001)
        pat = _mock_pat(anchor=None)
        stop, stype, _ = self._fn("double_top", "short", 1.35, df, pat, pip_size=0.0001)
        self.assertIn("fallback", stype)

    def test_long_direction(self):
        df  = _mock_df(0.001)
        pat = _mock_pat(anchor=1.34)
        stop, _, pips = self._fn("double_bottom", "long", 1.35, df, pat, pip_size=0.0001)
        self.assertLess(stop, 1.35)

    def test_wrong_side_falls_back(self):
        df  = _mock_df(0.001)
        pat = _mock_pat(anchor=1.34, stop_loss=1.36)  # below entry for short
        stop, stype, _ = self._fn("double_top", "short", 1.35, df, pat, pip_size=0.0001)
        self.assertIn("fallback", stype)


# ── 2. Variant C8 stop function ───────────────────────────────────────────────
class TestVariantC8Stop(unittest.TestCase):

    def setUp(self):
        self._fn = _s._make_c8_stop()

    def _call(self, entry, anchor, atr_val=0.001, pip=0.0001, direction="short"):
        df  = _mock_df(atr_val)
        pat = _mock_pat(anchor=anchor)
        return self._fn("double_top", direction, entry, df, pat, pip_size=pip)

    def test_floor_applied_when_stop_below_8_pips(self):
        # struct_dist = 2 pips < 8-pip floor
        stop, stype, pips = self._call(1.3500, 1.3502, atr_val=0.001, pip=0.0001)
        self.assertAlmostEqual(pips, 8.0, places=1)
        self.assertEqual(stype, "c8_floored")

    def test_floor_not_applied_when_stop_above_8_pips(self):
        # struct_dist = 50 pips > floor
        stop, stype, pips = self._call(1.3500, 1.3550, atr_val=0.1, pip=0.0001)
        self.assertGreater(pips, 8.0)
        self.assertNotEqual(stype, "c8_floored")

    def test_ceiling_still_applied_in_c8(self):
        # struct_dist = 200 pips, ceiling = 3×ATR = 30 pips → capped at 30
        stop, stype, pips = self._call(1.3500, 1.3700, atr_val=0.001, pip=0.0001)
        self.assertAlmostEqual(pips, 30.0, places=0)
        self.assertEqual(stype, "c8_capped")

    def test_floor_is_8_pips_exact(self):
        # struct_dist = 5 pips < 8 → floored to 8
        stop, stype, pips = self._call(1.3500, 1.3505, atr_val=0.001, pip=0.0001)
        self.assertAlmostEqual(pips, 8.0, places=1)

    def test_c8_always_wider_than_or_equal_to_c(self):
        fn_c  = _s._make_c_stop()
        fn_c8 = _s._make_c8_stop()
        df  = _mock_df(0.001)
        pat = _mock_pat(anchor=1.3502)
        _, _, pips_c  = fn_c("double_top",  "short", 1.35, df, pat, pip_size=0.0001)
        _, _, pips_c8 = fn_c8("double_top", "short", 1.35, df, pat, pip_size=0.0001)
        self.assertGreaterEqual(pips_c8, pips_c)

    def test_fallback_respects_floor(self):
        df  = _mock_df(0.0001)   # tiny ATR → atr*3 < 8 pips
        pat = _mock_pat(anchor=None, stop_loss=1.36)
        stop, stype, pips = self._fn("double_top", "short", 1.35, df, pat, pip_size=0.0001)
        self.assertGreaterEqual(pips, 8.0)


# ── 3. Config restore ─────────────────────────────────────────────────────────
class TestConfigRestore(unittest.TestCase):

    def test_reset_all_restores_atr_mult(self):
        orig = _sc.ATR_MIN_MULTIPLIER
        _sc.ATR_MIN_MULTIPLIER = 0.0
        _s._reset_all()
        self.assertEqual(_sc.ATR_MIN_MULTIPLIER, orig)

    def test_reset_all_restores_stop_fn(self):
        orig = _saf.get_structure_stop
        _saf.get_structure_stop = lambda *a, **kw: None
        _s._reset_all()
        self.assertIs(_saf.get_structure_stop, orig)


# ── 4. Unlocked helper ────────────────────────────────────────────────────────
class TestUnlocked(unittest.TestCase):

    def test_empty_base(self):
        t = _trade()
        self.assertEqual(len(_s._unlocked([], [t])), 1)

    def test_same_trades_no_unlock(self):
        t = _trade()
        self.assertEqual(len(_s._unlocked([t], [t])), 0)

    def test_new_trade_detected(self):
        t1 = _trade("GBP/JPY", ts="2025-01-15 09:00")
        t2 = _trade("USD/JPY", ts="2025-01-16 09:00")
        unlocked = _s._unlocked([t1], [t1, t2])
        self.assertEqual(len(unlocked), 1)
        self.assertEqual(unlocked[0]["pair"], "USD/JPY")

    def test_hour_truncation(self):
        t1 = _trade(ts="2025-01-15 09:00")
        t2 = _trade(ts="2025-01-15 09:45")   # same hour
        self.assertEqual(len(_s._unlocked([t1], [t2])), 0)


# ── 5. Target pips helper ─────────────────────────────────────────────────────
class TestTargetPips(unittest.TestCase):

    def test_basic(self):
        t = _trade(stop_pips=30.0, rr=2.5)
        self.assertAlmostEqual(_s._target_pips(t), 75.0)

    def test_zero_rr(self):
        t = _trade(rr=0.0)
        self.assertEqual(_s._target_pips(t), 0.0)

    def test_zero_stop(self):
        t = _trade(stop_pips=0.0, rr=3.0)
        self.assertEqual(_s._target_pips(t), 0.0)


# ── 6. Outlier audit ──────────────────────────────────────────────────────────
class TestOutlierAudit(unittest.TestCase):

    def test_returns_max_10(self):
        trades = [_trade(r=float(i)) for i in range(20)]
        top = _s.outlier_audit(trades)
        self.assertLessEqual(len(top), 10)

    def test_sorted_descending(self):
        trades = [_trade(r=float(i)) for i in range(15)]
        top = _s.outlier_audit(trades)
        rs = [t["r"] for t in top]
        self.assertEqual(rs, sorted(rs, reverse=True))

    def test_fewer_than_10_ok(self):
        trades = [_trade(r=1.0), _trade(r=2.0)]
        top = _s.outlier_audit(trades)
        self.assertEqual(len(top), 2)

    def test_empty_list(self):
        self.assertEqual(_s.outlier_audit([]), [])


# ── 7. Tiny-stop counts ───────────────────────────────────────────────────────
class TestTinyStopCounts(unittest.TestCase):

    def test_counts_correct(self):
        trades = [
            _trade(stop_pips=2.0),
            _trade(stop_pips=4.0),
            _trade(stop_pips=7.0),
            _trade(stop_pips=9.0),
            _trade(stop_pips=15.0),
        ]
        c = _s.tiny_stop_counts(trades)
        self.assertEqual(c["stop_le_3p"], 1)
        self.assertEqual(c["stop_le_5p"], 2)
        self.assertEqual(c["stop_le_8p"], 3)
        self.assertEqual(c["stop_le_10p"], 4)

    def test_zero_stop_not_counted(self):
        trades = [_trade(stop_pips=0.0)]
        c = _s.tiny_stop_counts(trades)
        self.assertEqual(c["stop_le_3p"], 0)

    def test_none_stop_not_counted(self):
        t = {"pair": "GBP/JPY", "r": 1.0, "initial_stop_pips": None}
        c = _s.tiny_stop_counts([t])
        self.assertEqual(c["stop_le_3p"], 0)

    def test_boundary_inclusive(self):
        trades = [_trade(stop_pips=8.0)]
        c = _s.tiny_stop_counts(trades)
        self.assertEqual(c["stop_le_8p"], 1)
        self.assertEqual(c["stop_le_5p"], 0)

    def test_empty_list(self):
        c = _s.tiny_stop_counts([])
        self.assertTrue(all(v == 0 for v in c.values()))


# ── 8. Realism flags ──────────────────────────────────────────────────────────
class TestRealismFlags(unittest.TestCase):

    def test_tiny_stop_flagged(self):
        t = _trade(stop_pips=4.0, r=2.0)
        flags = _s.realism_flags([t], "Q1-2025", 10.0)
        self.assertTrue(any("TINY_STOP" in f for f in flags))

    def test_extreme_r_flagged(self):
        t = _trade(stop_pips=30.0, r=9.5)
        flags = _s.realism_flags([t], "Q1-2025", 20.0)
        self.assertTrue(any("EXTREME_R" in f for f in flags))

    def test_concentration_flagged(self):
        t = _trade(stop_pips=30.0, r=8.0)
        flags = _s.realism_flags([t], "Q1-2025", 10.0)  # 8/10 = 80% > 20%
        self.assertTrue(any("CONCENTRATION" in f for f in flags))

    def test_no_flags_when_normal(self):
        t = _trade(stop_pips=30.0, r=2.5)
        flags = _s.realism_flags([t], "Q1-2025", 30.0)  # 2.5/30 = 8% < 20%
        self.assertEqual(flags, [])

    def test_zero_window_sum_no_concentration(self):
        t = _trade(stop_pips=30.0, r=2.5)
        flags = _s.realism_flags([t], "Q1-2025", 0.0)
        self.assertFalse(any("CONCENTRATION" in f for f in flags))

    def test_stop_exactly_8_not_flagged(self):
        t = _trade(stop_pips=8.0, r=2.0)
        flags = _s.realism_flags([t], "Q1-2025", 20.0)
        self.assertFalse(any("TINY_STOP" in f for f in flags))

    def test_r_exactly_8_not_flagged(self):
        t = _trade(stop_pips=30.0, r=8.0)
        flags = _s.realism_flags([t], "Q1-2025", 100.0)  # 8% → no concentration
        self.assertFalse(any("EXTREME_R" in f for f in flags))


# ── 9. Robustness ─────────────────────────────────────────────────────────────
class TestRobustness(unittest.TestCase):

    def _make_trades(self, rs):
        return [_trade(r=r, stop_pips=30) for r in rs]

    def test_ex_top1_excludes_best(self):
        c = self._make_trades([10.0, 3.0, 2.0, -1.0])
        a = self._make_trades([1.0, 1.0])
        rb = _s.robustness(c, a)
        # ex top-1: drop 10.0 → sum = 4.0
        self.assertAlmostEqual(rb["c_ex_top1"], 4.0)

    def test_ex_top2_excludes_top_two(self):
        c = self._make_trades([10.0, 3.0, 2.0, -1.0])
        a = self._make_trades([1.0])
        rb = _s.robustness(c, a)
        # ex top-2: drop 10.0 and 3.0 → sum = 1.0
        self.assertAlmostEqual(rb["c_ex_top2"], 1.0)

    def test_beats_a_flags(self):
        c = self._make_trades([10.0, 3.0, 2.0])
        a = self._make_trades([4.0])  # a_sum = 4.0
        rb = _s.robustness(c, a)
        self.assertTrue(rb["c_beats_a_ex1"])    # ex-top1: 5 > 4 → True
        self.assertFalse(rb["c_beats_a_ex2"])   # ex-top2: 2 < 4 → False

    def test_top1_metadata(self):
        c = self._make_trades([10.0, 3.0])
        a = self._make_trades([1.0])
        rb = _s.robustness(c, a)
        self.assertAlmostEqual(rb["top1_r"], 10.0)
        self.assertAlmostEqual(rb["top2_r"], 3.0)

    def test_empty_c(self):
        rb = _s.robustness([], [_trade(r=1.0)])
        self.assertEqual(rb["c_sum_r"], 0.0)
        self.assertEqual(rb["c_ex_top1"], 0.0)

    def test_single_c_trade(self):
        c = self._make_trades([5.0])
        a = self._make_trades([3.0])
        rb = _s.robustness(c, a)
        self.assertAlmostEqual(rb["c_ex_top1"], 0.0)
        self.assertAlmostEqual(rb["c_ex_top2"], 0.0)


# ── 10. WinResult construction ────────────────────────────────────────────────
class TestWinResult(unittest.TestCase):

    def test_sum_r_correct(self):
        trades = [_trade(r=2.0), _trade(r=-1.0), _trade(r=1.5)]
        wr = _s.WinResult(window="Q1-2025", n=3, wr=0.67,
                          sum_r=sum(t["r"] for t in trades),
                          ret_pct=10.0, max_dd=5.0, avg_r=0.83, trades=trades)
        self.assertAlmostEqual(wr.sum_r, 2.5)

    def test_fields(self):
        wr = _s.WinResult(window="W1", n=1, wr=1.0, sum_r=2.0,
                          ret_pct=5.0, max_dd=0.0, avg_r=2.0, trades=[])
        self.assertEqual(wr.window, "W1")
        self.assertEqual(wr.n, 1)


# ── 11. Report smoke test ─────────────────────────────────────────────────────
class TestReportSmoke(unittest.TestCase):

    @patch("scripts.variant_c_sanity.run_backtest")
    def test_report_builds(self, mock_bt):
        from src.strategy.forex.backtest_schema import BacktestResult
        trades_a = [_trade(r=1.5, stop_pips=30), _trade(r=-1.0, stop_pips=25)]
        trades_c = trades_a + [_trade(pair="USD/JPY", r=3.0, stop_pips=12,
                                      ts="2025-02-01 09:00")]
        mock_bt.side_effect = [
            BacktestResult(n_trades=len(trades_a), return_pct=5.0, win_rate=0.5,
                           avg_r=0.25, max_dd_pct=3.0, trades=trades_a),
            BacktestResult(n_trades=len(trades_c), return_pct=8.0, win_rate=0.67,
                           avg_r=1.17, max_dd_pct=3.0, trades=trades_c),
            BacktestResult(n_trades=len(trades_c), return_pct=7.5, win_rate=0.67,
                           avg_r=1.0, max_dd_pct=3.0, trades=trades_c),
        ] * 8  # 3 variants × 8 windows

        from scripts.variant_c_sanity import run_all, build_report, WINDOWS
        one_win = [WINDOWS[0]]
        with patch("scripts.variant_c_sanity.WINDOWS", one_win):
            results = run_all()
        report = build_report(results, windows=one_win)

        self.assertIn("Variant C Sanity", report)
        self.assertIn("Tiny-Stop Audit", report)
        self.assertIn("Robustness", report)
        self.assertIn("Recommendation", report)
        self.assertIn("Verdict", report)

    @patch("scripts.variant_c_sanity.run_backtest")
    def test_report_verdict_present(self, mock_bt):
        from src.strategy.forex.backtest_schema import BacktestResult
        mock_bt.return_value = BacktestResult(n_trades=0, trades=[])
        from scripts.variant_c_sanity import run_all, build_report, WINDOWS
        one_win = [WINDOWS[0]]
        with patch("scripts.variant_c_sanity.WINDOWS", one_win):
            results = run_all()
        report = build_report(results, windows=one_win)
        self.assertIn("Verdict", report)

    def test_reset_all_idempotent(self):
        _s._reset_all()
        _s._reset_all()
        self.assertEqual(_sc.ATR_MIN_MULTIPLIER, _s._ORIG_ATR_MIN_MULT)


# ── 12. Structural anchor ─────────────────────────────────────────────────────
class TestStructAnchor(unittest.TestCase):

    def test_none_when_anchor_none(self):
        df  = _mock_df()
        pat = _mock_pat(anchor=None)
        self.assertIsNone(_s._struct_anchor_for_pattern("double_top", "short", 1.35, df, pat))

    def test_correct_side_short(self):
        df  = _mock_df()
        pat = _mock_pat(anchor=1.36)
        a = _s._struct_anchor_for_pattern("double_top", "short", 1.35, df, pat)
        self.assertAlmostEqual(a, 1.36)

    def test_wrong_side_short_returns_none(self):
        df  = _mock_df()
        pat = _mock_pat(anchor=1.34)   # below entry for short
        self.assertIsNone(_s._struct_anchor_for_pattern("double_top", "short", 1.35, df, pat))


# ── 13. Variant count and windows ─────────────────────────────────────────────
class TestDefinitions(unittest.TestCase):

    def test_three_variants(self):
        self.assertEqual(len(_s.VARIANTS), 3)

    def test_variant_ids(self):
        ids = [v[0] for v in _s.VARIANTS]
        self.assertEqual(ids, ["A", "C", "C8"])

    def test_8_windows(self):
        self.assertEqual(len(_s.WINDOWS), 8)

    def test_hard_floor_constant(self):
        self.assertEqual(_s._HARD_FLOOR_PIPS, 8.0)

    def test_ceiling_constant(self):
        self.assertEqual(_s._ATR_C_CEILING_MULT, 3.0)


if __name__ == "__main__":
    unittest.main()
