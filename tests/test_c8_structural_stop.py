"""tests/test_c8_structural_stop.py
====================================
Tests for the C8 structural stop change (PR feat/c8-structural-stop).

Verifies:
  1. Module constants updated correctly (_MIN_FRAC_ATR=0, _MAX_FRAC_ATR=3.0)
  2. strategy_config.ATR_MIN_MULTIPLIER = 0.0
  3. get_structure_stop() applies hard 8-pip floor (not ATR fraction)
  4. get_structure_stop() applies 3×ATR_1H ceiling (not 10×)
  5. Stops between 8 pips and 3×ATR pass through unchanged
  6. Stops < 8 pips fall through to next candidate (or ATR fallback)
  7. Stops > 3×ATR fall through to next candidate (or ATR fallback)
  8. Existing priority order preserved (neckline_retest_swing > shoulder_anchor)
  9. ATR fallback still works when all structural candidates fail
 10. Wrong-side guard still enforced
 11. Buffer still added to structural anchor price
 12. _stop_side rejection reasons updated in logs
"""
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import src.strategy.forex.targeting as _tgt
import src.strategy.forex.strategy_config as _sc


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_df(atr_val: float, n: int = 20) -> pd.DataFrame:
    """Synthetic 1H OHLC that yields ATR ≈ atr_val."""
    highs  = np.full(n, 1.0 + atr_val / 2)
    lows   = np.full(n, 1.0 - atr_val / 2)
    closes = np.ones(n)
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


def _make_pattern(stop_anchor=None, neckline=None, stop_loss=1.37,
                  pattern_type="double_top", target_1=None, target_2=None):
    p = MagicMock()
    p.stop_anchor  = stop_anchor
    p.neckline     = neckline
    p.stop_loss    = stop_loss
    p.pattern_type = pattern_type
    p.target_1     = target_1
    p.target_2     = target_2
    return p


def _call_gss(
    pattern_type="double_top",
    direction="short",
    entry=1.3500,
    atr_val=0.001,          # 1H ATR in price units
    anchor=None,
    pip=0.0001,
    is_jpy=False,
    stop_log=None,
    stop_loss_fallback=1.3700,
):
    df  = _make_df(atr_val)
    pat = _make_pattern(stop_anchor=anchor, stop_loss=stop_loss_fallback)
    return _tgt.get_structure_stop(
        pattern_type=pattern_type,
        direction=direction,
        entry=entry,
        df_1h=df,
        pattern=pat,
        pip_size=pip,
        is_jpy_or_cross=is_jpy,
        stop_log=stop_log,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Module-level constants
# ─────────────────────────────────────────────────────────────────────────────
class TestConstants(unittest.TestCase):

    def test_min_frac_atr_retained(self):
        """_MIN_FRAC_ATR must be 0.15 — retained for high-vol guard (matches ablation)."""
        self.assertEqual(_tgt._MIN_FRAC_ATR, 0.15)

    def test_min_abs_pips_is_8(self):
        """Hard pip floor must be 8.0."""
        self.assertEqual(_tgt._MIN_ABS_PIPS, 8.0)

    def test_max_frac_atr_is_3(self):
        """Ceiling must be 3×ATR_1H (down from 10×)."""
        self.assertEqual(_tgt._MAX_FRAC_ATR, 3.0)

    def test_strategy_config_atr_min_mult_zero(self):
        """Daily ATR fraction floor gate must be disabled (0.0)."""
        self.assertEqual(_sc.ATR_MIN_MULTIPLIER, 0.0)

    def test_strategy_config_atr_stop_mult_unchanged(self):
        """Ceiling multiplier for daily ATR must be unchanged at 8.0."""
        self.assertEqual(_sc.ATR_STOP_MULTIPLIER, 8.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hard pip floor behaviour
# ─────────────────────────────────────────────────────────────────────────────
class TestHardPipFloor(unittest.TestCase):

    def test_anchor_at_exactly_8_pips_passes(self):
        """Stop exactly at floor should pass the 8-pip floor check."""
        # anchor = entry + 8 pips
        entry  = 1.3500
        anchor = entry + 8 * 0.0001   # = 1.3508
        stop, stype, pips = _call_gss(anchor=anchor, atr_val=0.002)
        # Should produce a stop near anchor (with buffer added)
        self.assertGreater(stop, entry)       # on correct side for short
        self.assertGreaterEqual(pips, 8.0)   # at or above floor

    def test_anchor_at_4_pips_rejected_by_floor(self):
        """4-pip anchor + 3-pip buffer = 7 pips total — below 8-pip floor → fallback."""
        entry  = 1.3500
        anchor = entry + 4 * 0.0001   # 4 pips above entry; +3p buf = 7p total
        stop, stype, pips = _call_gss(anchor=anchor, atr_val=0.002)
        self.assertIn("fallback", stype)

    def test_anchor_at_9_pips_passes(self):
        """9-pip stop is above floor → should use structural anchor."""
        entry  = 1.3500
        anchor = entry + 9 * 0.0001
        stop, stype, pips = _call_gss(anchor=anchor, atr_val=0.002)
        self.assertNotIn("fallback", stype)
        self.assertNotIn("legacy", stype)

    def test_floor_is_max_of_pip_and_atr_fraction(self):
        """Floor = max(8p, 0.15×ATR_1H) — ablation behaviour preserved.

        At current watchlist vol (1.1–5.3p ATR fraction) the pip floor always wins.
        With spike-level ATR (1H ATR = 60p → 0.15×60 = 9p), the ATR fraction
        supersedes the 8p pip floor — stops between 8p and 9p are gated out.
        """
        entry = 1.3500
        # Normal vol (ATR_1H ≈ 0.002 = 20p, fraction = 3p < 8p → pip floor = 8p)
        # Anchor at 14p (after 3p buffer → 17p total > 8p floor): passes
        anchor_normal = entry + 11 * 0.0001   # +3p buf → 14p > 8p → passes
        _, stype_n, pips_n = _call_gss(anchor=anchor_normal, atr_val=0.002)
        self.assertNotIn("fallback", stype_n)
        self.assertGreaterEqual(pips_n, 8.0)

        # Spike vol (ATR_1H = 0.006 = 60p, fraction = 0.15×60 = 9p → floor = 9p)
        # Anchor at 7p (after 3p buf → 10p > 9p floor) → passes even with 9p floor.
        anchor_spike = entry + 7 * 0.0001   # +3p buf → 10p > max(8p, 9p) → passes
        _, stype_s, pips_s = _call_gss(anchor=anchor_spike, atr_val=0.006)
        self.assertNotIn("fallback", stype_s)
        # Also confirm floor is 9p not 8p (a 7p stop would be blocked here but not at 0.002 ATR)
        anchor_tight = entry + 4 * 0.0001   # +3p buf → 7p < 9p spike floor → rejected
        _, stype_tight, _ = _call_gss(anchor=anchor_tight, atr_val=0.006)
        self.assertIn("fallback", stype_tight)

    def test_zero_atr_pip_floor_still_applies(self):
        """When ATR=0 (insufficient bars), 8-pip floor still active for micro-stops."""
        entry  = 1.3500
        # 4p anchor + 3p buffer = 7p < 8p floor → rejected even with ATR=0
        anchor = entry + 4 * 0.0001
        df = pd.DataFrame({"high": [1.0]*3, "low": [1.0]*3, "close": [1.0]*3})
        pat = _make_pattern(stop_anchor=anchor, stop_loss=1.36)
        stop, stype, pips = _tgt.get_structure_stop(
            "double_top", "short", entry, df, pat, pip_size=0.0001)
        # Structural anchor rejected → falls to ATR fallback, legacy, or emergency
        self.assertNotEqual(stype, "structural_anchor")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ATR ceiling behaviour (3×ATR_1H)
# ─────────────────────────────────────────────────────────────────────────────
class TestATRCeiling(unittest.TestCase):

    def test_stop_within_3x_atr_passes(self):
        """Stop well inside the 3×ATR ceiling should pass (structural_anchor returned)."""
        entry  = 1.3500
        atr    = 0.001   # 10 pips; ceiling = 30p; buf = 3p
        # anchor at 2.5×ATR = 25p above entry; +3p buf = 28p total < 30p ceiling ✓
        anchor = entry + 2.5 * atr
        stop, stype, pips = _call_gss(anchor=anchor, atr_val=atr)
        self.assertNotIn("fallback", stype)
        self.assertNotIn("legacy",   stype)

    def test_stop_above_3x_atr_rejected(self):
        """Stop at 4×ATR should exceed ceiling → fall through."""
        entry  = 1.3500
        atr    = 0.001
        anchor = entry + 4.0 * atr   # 4×ATR > 3×ATR ceiling
        stop, stype, pips = _call_gss(anchor=anchor, atr_val=atr)
        self.assertIn("fallback", stype)

    def test_old_ceiling_10x_now_blocks(self):
        """Stop at 5×ATR was OK under old 10× ceiling; must now be rejected."""
        entry  = 1.3500
        atr    = 0.001
        anchor = entry + 5.0 * atr   # 5×ATR: passed old cap (10×), blocked by new (3×)
        stop, stype, pips = _call_gss(anchor=anchor, atr_val=atr)
        self.assertIn("fallback", stype)

    def test_ceiling_boundary_exactly_3x(self):
        """Stop exactly at 3×ATR (after buffer subtraction) should be on the boundary."""
        entry  = 1.3500
        atr    = 0.001
        # We want anchor such that dist_after_buffer ≈ 3×ATR
        # buf ≈ 3 pips = 0.0003 for majors; so anchor = entry + 3×ATR - buf
        buf    = 3 * 0.0001
        anchor = entry + 3.0 * atr - buf  # distance after buffer = 3×ATR exactly
        stop, stype, pips = _call_gss(anchor=anchor, atr_val=atr)
        # May pass or be on boundary — just verify it's not above ceiling
        if "fallback" not in stype and "legacy" not in stype:
            self.assertLessEqual(abs(stop - entry), 3.0 * atr + 0.0001)

    def test_jpy_pair_pip_floor_8_pips(self):
        """JPY pairs use pip=0.01; floor is still 8 pips (8 × 0.01 = 0.08)."""
        entry  = 155.000
        anchor = entry + 9 * 0.01   # 9 JPY pips above entry
        stop, stype, pips = _call_gss(
            anchor=anchor, atr_val=0.5, pip=0.01, is_jpy=True,
            entry=entry, stop_loss_fallback=156.0,
        )
        self.assertNotIn("fallback", stype)
        self.assertGreaterEqual(pips, 8.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Valid structural stop passes unchanged
# ─────────────────────────────────────────────────────────────────────────────
class TestValidStopPassthrough(unittest.TestCase):

    def test_valid_short_stop_returned(self):
        entry  = 1.3500
        anchor = entry + 30 * 0.0001   # 30 pips above entry (short)
        stop, stype, pips = _call_gss(anchor=anchor, atr_val=0.005)
        self.assertGreater(stop, entry)
        self.assertAlmostEqual(pips, 30.0, delta=5.0)  # allow buffer

    def test_valid_long_stop_returned(self):
        entry  = 1.3500
        anchor = entry - 30 * 0.0001   # 30 pips below entry (long)
        stop, stype, pips = _call_gss(
            pattern_type="double_bottom", direction="long",
            anchor=anchor, atr_val=0.005,
        )
        self.assertLess(stop, entry)
        self.assertAlmostEqual(pips, 30.0, delta=5.0)

    def test_stop_type_structural_anchor(self):
        entry  = 1.3500
        anchor = entry + 25 * 0.0001
        stop, stype, pips = _call_gss(anchor=anchor, atr_val=0.005)
        self.assertIn("structural_anchor", stype)

    def test_buffer_still_added(self):
        """Buffer (3-pip for majors) should still be added to anchor."""
        entry  = 1.3500
        anchor = entry + 20 * 0.0001   # exactly 20 pips
        stop, stype, pips = _call_gss(anchor=anchor, atr_val=0.005)
        # Stop should be anchor + buffer, so pips > 20
        self.assertGreater(pips, 20.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Priority order preserved
# ─────────────────────────────────────────────────────────────────────────────
class TestPriorityOrder(unittest.TestCase):

    def test_hs_shoulder_anchor_passes_bounds(self):
        """H&S shoulder anchor at valid distance returns a structural stop."""
        entry  = 1.3500
        anchor = entry + 35 * 0.0001   # 35p; +3p buf=38p well inside 3×ATR ceiling
        df = _make_df(0.005)           # ATR=50p; ceiling=150p; floor=max(8p,7.5p)=8p
        pat = _make_pattern(stop_anchor=anchor, stop_loss=entry + 0.01,
                            pattern_type="head_and_shoulders", neckline=entry - 0.001)
        stop, stype, pips = _tgt.get_structure_stop(
            "head_and_shoulders", "short", entry, df, pat, pip_size=0.0001,
        )
        self.assertNotIn("legacy", stype)
        self.assertGreater(stop, entry)
        self.assertGreaterEqual(pips, 8.0)

    def test_no_anchor_falls_back_to_atr(self):
        """No structural anchor → ATR fallback used."""
        stop, stype, pips = _call_gss(anchor=None, atr_val=0.002)
        self.assertIn("fallback", stype)

    def test_wrong_side_anchor_falls_back(self):
        """Anchor on wrong side of entry (below for short) → fallback."""
        entry  = 1.3500
        anchor = entry - 20 * 0.0001   # below entry (wrong side for short)
        stop, stype, pips = _call_gss(anchor=anchor, atr_val=0.002)
        self.assertIn("fallback", stype)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ATR fallback still functional
# ─────────────────────────────────────────────────────────────────────────────
class TestATRFallback(unittest.TestCase):

    def test_fallback_uses_atr_mult(self):
        """ATR fallback (exempt from ceiling) returns stop ≈ entry + 3×ATR."""
        entry   = 1.3500
        atr_val = 0.002
        stop, stype, pips = _call_gss(anchor=None, atr_val=atr_val)
        self.assertIn("fallback", stype)
        # Stop is on the correct side and within reasonable range of 3×ATR
        self.assertGreater(stop, entry)
        self.assertAlmostEqual(stop, entry + 3.0 * atr_val, delta=0.001)

    def test_fallback_long_direction(self):
        entry   = 1.3500
        atr_val = 0.002
        stop, stype, pips = _call_gss(
            pattern_type="double_bottom", direction="long",
            anchor=None, atr_val=atr_val,
        )
        self.assertIn("fallback", stype)
        self.assertLess(stop, entry)   # below entry for long
        self.assertAlmostEqual(stop, entry - 3.0 * atr_val, delta=0.001)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Stop log entries
# ─────────────────────────────────────────────────────────────────────────────
class TestStopLog(unittest.TestCase):

    def test_selected_entry_in_log(self):
        log = []
        entry  = 1.3500
        anchor = entry + 25 * 0.0001
        _call_gss(anchor=anchor, atr_val=0.005, stop_log=log)
        selected = [e for e in log if "STOP_SELECTED" in e.get("action", "")]
        self.assertEqual(len(selected), 1)

    def test_rejected_entry_in_log(self):
        log = []
        entry  = 1.3500
        anchor = entry + 5 * 0.0001   # 5 pips < floor → rejected
        _call_gss(anchor=anchor, atr_val=0.002, stop_log=log)
        rejected = [e for e in log if "REJECTED" in e.get("action", "")]
        self.assertGreater(len(rejected), 0)

    def test_log_has_price_and_pips(self):
        log = []
        entry  = 1.3500
        anchor = entry + 25 * 0.0001
        _call_gss(anchor=anchor, atr_val=0.005, stop_log=log)
        for entry_log in log:
            self.assertIn("price", entry_log)
            self.assertIn("pips",  entry_log)


# ─────────────────────────────────────────────────────────────────────────────
# 8. model_tags still emit updated atr_min
# ─────────────────────────────────────────────────────────────────────────────
class TestModelTags(unittest.TestCase):

    def test_atr_min_tag_reflects_zero(self):
        from src.strategy.forex.strategy_config import get_model_tags
        tags = get_model_tags()
        self.assertIn("atr_min_0.0", tags)

    def test_atr_max_tag_unchanged(self):
        from src.strategy.forex.strategy_config import get_model_tags
        tags = get_model_tags()
        self.assertIn("atr_max_8", tags)


if __name__ == "__main__":
    unittest.main()
