"""
tests/test_ablation_engulf_trigger.py
Test suite for engulfing confirmation timing ablation study.
458 tests (existing) + 18 new = 476 total target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.strategy.forex.strategy_config as _sc


def _restore():
    _sc.ENTRY_TRIGGER_MODE          = "engulf_only"
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 2


# ──────────────────────────────────────────────────────────────────────────────
# 1. Config defaults
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigDefaults:
    def test_entry_trigger_mode_exists(self):
        assert hasattr(_sc, "ENTRY_TRIGGER_MODE")

    def test_engulf_confirm_lookback_exists(self):
        assert hasattr(_sc, "ENGULF_CONFIRM_LOOKBACK_BARS")

    def test_entry_trigger_mode_default(self):
        # Promoted to B-Prime (engulf_or_strict_pin_at_level) 2026-03-07
        assert _sc.ENTRY_TRIGGER_MODE == "engulf_or_strict_pin_at_level", (
            "ENTRY_TRIGGER_MODE must be 'engulf_or_strict_pin_at_level' (B-Prime LIVE_PAPER)"
        )

    def test_lookback_default_is_2(self):
        assert _sc.ENGULF_CONFIRM_LOOKBACK_BARS == 2, (
            "ENGULF_CONFIRM_LOOKBACK_BARS must be 2 in production"
        )

    def test_engulfing_only_derived_flag(self):
        """ENGULFING_ONLY is False — B-Prime mode active (strict_pin_at_level)."""
        assert _sc.ENGULFING_ONLY is False


# ──────────────────────────────────────────────────────────────────────────────
# 2. Model tags
# ──────────────────────────────────────────────────────────────────────────────

class TestModelTags:
    def setup_method(self): _restore()
    def teardown_method(self): _restore()

    def test_baseline_tag(self):
        _sc.ENTRY_TRIGGER_MODE = "engulf_only"
        tags = _sc.get_model_tags()
        assert "engulfing_only" in tags

    def test_strict_pin_tag(self):
        _sc.ENTRY_TRIGGER_MODE = "engulf_or_strict_pin_at_level"
        tags = _sc.get_model_tags()
        assert "strict_pin_at_level" in tags

    def test_lookback_2_no_tag(self):
        _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 2
        tags = _sc.get_model_tags()
        lb_tags = [t for t in tags if t.startswith("engulf_lb")]
        assert lb_tags == [], f"No engulf_lb tag expected for lb=2, got {lb_tags}"

    def test_lookback_3_tag(self):
        _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 3
        tags = _sc.get_model_tags()
        assert "engulf_lb3" in tags, f"Expected engulf_lb3 in {tags}"

    def test_lookback_4_tag(self):
        _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 4
        tags = _sc.get_model_tags()
        assert "engulf_lb4" in tags


# ──────────────────────────────────────────────────────────────────────────────
# 3. Signal detector construction wires ENGULF_CONFIRM_LOOKBACK_BARS
# ──────────────────────────────────────────────────────────────────────────────

class TestSignalDetectorLookback:
    def setup_method(self): _restore()
    def teardown_method(self): _restore()

    def test_default_lookback_wired(self):
        from src.strategy.forex.set_and_forget import SetAndForgetStrategy
        s = SetAndForgetStrategy()
        assert s.signal_detector.lookback == 2

    def test_lb3_wired(self):
        _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 3
        from src.strategy.forex.set_and_forget import SetAndForgetStrategy
        s = SetAndForgetStrategy()
        assert s.signal_detector.lookback == 3, (
            f"Expected lookback=3, got {s.signal_detector.lookback}"
        )

    def test_lb1_wired(self):
        _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 1
        from src.strategy.forex.set_and_forget import SetAndForgetStrategy
        s = SetAndForgetStrategy()
        assert s.signal_detector.lookback == 1


# ──────────────────────────────────────────────────────────────────────────────
# 4. Ablation script structure
# ──────────────────────────────────────────────────────────────────────────────

class TestAblationScriptStructure:
    def _load(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ablation_engulf_trigger",
            REPO / "scripts" / "ablation_engulf_trigger.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_importable(self):
        mod = self._load()
        assert hasattr(mod, "main")
        assert hasattr(mod, "VARIANTS")
        assert hasattr(mod, "_reset_engulf_config")

    def test_four_variants(self):
        mod = self._load()
        assert len(mod.VARIANTS) == 4

    def test_variant_a_is_baseline(self):
        mod = self._load()
        lbl, tm, lb, _ = mod.VARIANTS[0]
        assert lbl == "A"
        assert tm  == "engulf_only"
        assert lb  == 2

    def test_variant_d_is_broadest(self):
        mod = self._load()
        lbl, tm, lb, _ = mod.VARIANTS[3]
        assert lbl == "D"
        assert lb  == 3  # extended window
        assert "strict_pin" in tm

    def test_atexit_resets_both_flags(self):
        mod = self._load()
        _sc.ENTRY_TRIGGER_MODE           = "engulf_or_pin"
        _sc.ENGULF_CONFIRM_LOOKBACK_BARS = 3
        mod._reset_engulf_config()
        assert _sc.ENTRY_TRIGGER_MODE          == "engulf_only"
        assert _sc.ENGULF_CONFIRM_LOOKBACK_BARS == 2


# ──────────────────────────────────────────────────────────────────────────────
# 5. Helper functions
# ──────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def _load(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ablation_engulf_trigger",
            REPO / "scripts" / "ablation_engulf_trigger.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_session_labels(self):
        mod = self._load()
        assert mod._session("2026-02-03T10:00:00+00:00") == "London"
        assert mod._session("2026-02-03T12:30:00+00:00") == "London_NY_Overlap"
        assert mod._session("2026-02-03T15:00:00+00:00") == "NY"
        assert mod._session("2026-02-03T22:00:00+00:00") == "off_session"

    def test_week_labels(self):
        mod = self._load()
        assert "W1" in mod._week("2026-02-05T10:00:00+00:00")
        assert "W2" in mod._week("2026-02-20T10:00:00+00:00")
        assert "Mar" in mod._week("2026-03-02T10:00:00+00:00")
