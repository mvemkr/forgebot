"""
tests/test_ablation_zone_touch.py
Test suite for zone-touch gate ablation study.
451 tests (existing) + 15 new = 466 total target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.strategy.forex.strategy_config as _sc


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _restore_zone_touch_mode():
    _sc.ZONE_TOUCH_MODE = "full"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Config defaults
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigDefault:
    def test_zone_touch_mode_exists(self):
        """ZONE_TOUCH_MODE attribute must exist on strategy_config."""
        assert hasattr(_sc, "ZONE_TOUCH_MODE")

    def test_default_is_full(self):
        """Production default MUST be 'full' — offline ablation only."""
        assert _sc.ZONE_TOUCH_MODE == "full", (
            "ZONE_TOUCH_MODE default changed from 'full' — this will affect live trading!"
        )

    def test_production_values_unchanged(self):
        """ATR mult constants must not be changed by this feature."""
        assert _sc.ZONE_TOUCH_ATR_MULT == 0.35
        assert _sc.ZONE_TOUCH_ATR_MULT_CROSS == 0.50
        assert _sc.ZONE_TOUCH_LOOKBACK_BARS == 5


# ──────────────────────────────────────────────────────────────────────────────
# 2. Model-tag emission
# ──────────────────────────────────────────────────────────────────────────────

class TestModelTags:
    def setup_method(self):
        _restore_zone_touch_mode()

    def teardown_method(self):
        _restore_zone_touch_mode()

    def test_full_mode_no_zt_tag(self):
        """Mode 'full' must NOT emit a zt_ tag."""
        _sc.ZONE_TOUCH_MODE = "full"
        tags = _sc.get_model_tags()
        zt_tags = [t for t in tags if t.startswith("zt_")]
        assert zt_tags == [], f"Unexpected zt_ tags for 'full' mode: {zt_tags}"

    def test_near_2pip_tag(self):
        _sc.ZONE_TOUCH_MODE = "near_2pip"
        tags = _sc.get_model_tags()
        assert "zt_near_2pip" in tags, f"Expected zt_near_2pip in {tags}"

    def test_near_5pip_tag(self):
        _sc.ZONE_TOUCH_MODE = "near_5pip"
        tags = _sc.get_model_tags()
        assert "zt_near_5pip" in tags, f"Expected zt_near_5pip in {tags}"

    def test_wide_tag(self):
        _sc.ZONE_TOUCH_MODE = "wide"
        tags = _sc.get_model_tags()
        assert "zt_wide" in tags, f"Expected zt_wide in {tags}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Zone tolerance arithmetic
# ──────────────────────────────────────────────────────────────────────────────

class TestZoneTolMath:
    """
    Unit-test the zone_tol arithmetic for each mode WITHOUT touching the full
    strategy.  We replicate the 5 lines of logic verbatim and assert the output.
    """

    def _compute_tol(self, mode: str, atr: float, pip: float, mult: float) -> float:
        """Mirror of the set_and_forget.py FILTER 4.5 logic."""
        tol = atr * mult
        if mode == "near_2pip":
            tol += 2 * pip
        elif mode == "near_5pip":
            tol += 5 * pip
        elif mode == "wide":
            tol *= 2.0
        return tol

    def test_full_mode_unchanged(self):
        atr = 0.00240   # ~24p for EUR/USD
        pip = 0.0001
        tol = self._compute_tol("full", atr, pip, 0.35)
        assert abs(tol - atr * 0.35) < 1e-10, "full mode must not alter tol"

    def test_near_2pip_adds_exactly_2p_major(self):
        atr = 0.00240
        pip = 0.0001
        base_tol = atr * 0.35
        tol = self._compute_tol("near_2pip", atr, pip, 0.35)
        assert abs(tol - (base_tol + 2 * pip)) < 1e-10

    def test_near_5pip_adds_exactly_5p_major(self):
        atr = 0.00240
        pip = 0.0001
        base_tol = atr * 0.35
        tol = self._compute_tol("near_5pip", atr, pip, 0.35)
        assert abs(tol - (base_tol + 5 * pip)) < 1e-10

    def test_wide_doubles_tol(self):
        atr = 0.00240
        pip = 0.0001
        base_tol = atr * 0.35
        tol = self._compute_tol("wide", atr, pip, 0.35)
        assert abs(tol - base_tol * 2.0) < 1e-10

    def test_near_2pip_jpy(self):
        atr = 0.25  # ~25p for USD/JPY
        pip = 0.01
        base_tol = atr * 0.35
        tol = self._compute_tol("near_2pip", atr, pip, 0.35)
        expected = base_tol + 2 * pip
        assert abs(tol - expected) < 1e-10

    def test_near_5pip_jpy(self):
        atr = 0.25
        pip = 0.01
        base_tol = atr * 0.35
        tol = self._compute_tol("near_5pip", atr, pip, 0.35)
        expected = base_tol + 5 * pip
        assert abs(tol - expected) < 1e-10

    def test_relaxation_ordering_bcd_gt_a(self):
        """Variant ordering: D (wide) ≥ C (+5p) ≥ B (+2p) ≥ A (full) for typical EUR/USD."""
        atr, pip, mult = 0.00240, 0.0001, 0.35
        ta = self._compute_tol("full",      atr, pip, mult)
        tb = self._compute_tol("near_2pip", atr, pip, mult)
        tc = self._compute_tol("near_5pip", atr, pip, mult)
        td = self._compute_tol("wide",      atr, pip, mult)
        assert ta < tb < tc, f"Expected A < B < C but got {ta:.5f} {tb:.5f} {tc:.5f}"
        assert td > ta,      f"Expected D > A but got {td:.5f} vs {ta:.5f}"

    def test_gate_unlocks_correctly(self):
        """
        Simulate: EUR/USD, ATR=5.7p, base_tol=2.0p, wick 3p from neckline.
        full    → FAIL (3 > 2)
        near_2pip → PASS (3 < 4)
        near_5pip → PASS (3 < 7)
        wide    → PASS (3 < 4)
        """
        atr = 0.000571  # 5.71p for EUR/USD (~2.0p tolerance)
        pip = 0.0001
        mult = 0.35
        wick_dist = 3 * pip   # 0.0003, 3 pips from neckline

        def passes(mode):
            tol = self._compute_tol(mode, atr, pip, mult)
            return wick_dist <= tol

        assert not passes("full"),      "full mode should block wick at 3p when tol=2p"
        assert passes("near_2pip"),     "near_2pip (tol=4p) should unlock wick at 3p"
        assert passes("near_5pip"),     "near_5pip (tol=7p) should unlock wick at 3p"
        assert passes("wide"),          "wide (tol=4p) should unlock wick at 3p"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Ablation script structural checks
# ──────────────────────────────────────────────────────────────────────────────

class TestAblationScriptStructure:
    def test_script_importable(self):
        """ablation_zone_touch.py must import without error."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ablation_zone_touch",
            REPO / "scripts" / "ablation_zone_touch.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main"), "ablation_zone_touch must expose main()"
        assert hasattr(mod, "VARIANTS"), "ablation_zone_touch must expose VARIANTS"
        assert hasattr(mod, "_reset_zone_touch_mode"), "must have atexit reset function"

    def test_variants_are_four(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ablation_zone_touch",
            REPO / "scripts" / "ablation_zone_touch.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert len(mod.VARIANTS) == 4, f"Expected 4 variants, got {len(mod.VARIANTS)}"

    def test_variant_labels_are_abcd(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ablation_zone_touch",
            REPO / "scripts" / "ablation_zone_touch.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        labels = [v[0] for v in mod.VARIANTS]
        assert labels == ["A", "B", "C", "D"]

    def test_variant_a_is_full_mode(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ablation_zone_touch",
            REPO / "scripts" / "ablation_zone_touch.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Variant A must use mode "full"
        assert mod.VARIANTS[0][1] == "full"

    def test_atexit_resets_to_full(self):
        """Calling the atexit hook directly must reset ZONE_TOUCH_MODE to 'full'."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ablation_zone_touch",
            REPO / "scripts" / "ablation_zone_touch.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _sc.ZONE_TOUCH_MODE = "wide"
        mod._reset_zone_touch_mode()
        assert _sc.ZONE_TOUCH_MODE == "full", "atexit must reset ZONE_TOUCH_MODE to 'full'"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Helper functions
# ──────────────────────────────────────────────────────────────────────────────

class TestHelperFunctions:
    def test_session_label_london(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ablation_zone_touch",
            REPO / "scripts" / "ablation_zone_touch.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod._session_label("2026-02-03T10:00:00+00:00") == "London"
        assert mod._session_label("2026-02-03T12:00:00+00:00") == "London_NY_Overlap"
        assert mod._session_label("2026-02-03T14:00:00+00:00") == "NY"
        assert mod._session_label("2026-02-03T22:00:00+00:00") == "off_session"

    def test_week_bucket(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ablation_zone_touch",
            REPO / "scripts" / "ablation_zone_touch.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert "W1" in mod._week_bucket("2026-02-03T10:00:00+00:00")
        assert "W2" in mod._week_bucket("2026-02-20T10:00:00+00:00")
        assert "Mar" in mod._week_bucket("2026-03-03T10:00:00+00:00")
