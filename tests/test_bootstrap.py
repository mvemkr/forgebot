"""
tests/test_bootstrap.py
=======================
Bootstrap guard tests for ControlState.bootstrap().

Invariants:
  1. LIVE_PAPER + missing control.json
     → file created, pause_new_entries=True, bootstrap_event="BOOTSTRAP_CREATED"
  2. LIVE_REAL + missing control.json
     → file NOT created, pause_new_entries=True (in-memory only),
       control_missing=True, bootstrap_event="CONTROL_STATE_MISSING"
  3. File already exists → bootstrap() is a no-op (no overwrite)
  4. LIVE_PAPER + example present → content seeded from example
  5. LIVE_PAPER + example missing → uses defaults (no crash)
  6. LIVE_REAL bootstrap → subsequent reload() does not create the file
  7. is_backtest=True → bootstrap() is always a no-op
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.execution.control_state import ControlState


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ctrl(tmp_path: Path, *, with_example: bool = False, example_content: dict | None = None) -> ControlState:
    """Return a ControlState pointing at a tmp directory (no existing control.json)."""
    ctrl_path = tmp_path / "control.json"
    if with_example:
        content = example_content or {"pause_new_entries": False, "reason": "from_example"}
        (tmp_path / "control.example.json").write_text(json.dumps(content))
    return ControlState(path=ctrl_path, is_backtest=False)


def _ctrl_existing(tmp_path: Path) -> ControlState:
    """Return a ControlState where control.json already exists (paused)."""
    ctrl_path = tmp_path / "control.json"
    ctrl_path.write_text(json.dumps({
        "pause_new_entries": True,
        "reason": "pre-existing",
        "updated_by": "test",
        "last_updated": None,
        "risk_mode": None,
    }))
    return ControlState(path=ctrl_path, is_backtest=False)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LIVE_PAPER missing → file created, paused, BOOTSTRAP_CREATED
# ─────────────────────────────────────────────────────────────────────────────

class TestLivePaperBootstrap:

    def test_file_created(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        assert not (tmp_path / "control.json").exists()
        ctrl.bootstrap(is_live_real=False)
        assert (tmp_path / "control.json").exists(), "LIVE_PAPER: control.json must be created"

    def test_paused_in_memory(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=False)
        assert ctrl.pause_new_entries is True

    def test_paused_on_disk(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=False)
        data = json.loads((tmp_path / "control.json").read_text())
        assert data["pause_new_entries"] is True

    def test_bootstrap_event(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=False)
        assert ctrl.bootstrap_event == "BOOTSTRAP_CREATED"

    def test_reason_on_disk(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=False)
        data = json.loads((tmp_path / "control.json").read_text())
        assert data["reason"] == "BOOTSTRAP_CREATED"

    def test_control_missing_false(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=False)
        assert ctrl.control_missing is False

    def test_updated_by_bootstrap(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=False)
        data = json.loads((tmp_path / "control.json").read_text())
        assert data["updated_by"] == "system:bootstrap"


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIVE_REAL missing → NOT created, in-memory pause, CONTROL_STATE_MISSING
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveRealBootstrap:

    def test_file_not_created(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=True)
        assert not (tmp_path / "control.json").exists(), \
            "LIVE_REAL: control.json must NOT be created"

    def test_paused_in_memory(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=True)
        assert ctrl.pause_new_entries is True

    def test_control_missing_true(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=True)
        assert ctrl.control_missing is True

    def test_bootstrap_event(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=True)
        assert ctrl.bootstrap_event == "CONTROL_STATE_MISSING"

    def test_reason_in_memory(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=True)
        assert ctrl.reason == "CONTROL_STATE_MISSING"

    def test_reload_does_not_create_file(self, tmp_path):
        """reload() after LIVE_REAL bootstrap must not create the file."""
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=True)
        ctrl.reload()
        assert not (tmp_path / "control.json").exists()

    def test_reload_resets_to_defaults(self, tmp_path):
        """reload() after LIVE_REAL bootstrap resets to default (file absent).
        The in-memory CONTROL_STATE_MISSING flag is ephemeral — it is set only
        once at bootstrap time; reload() is not expected to re-set it."""
        ctrl = _ctrl(tmp_path)
        ctrl.bootstrap(is_live_real=True)
        ctrl.reload()
        # File still missing → defaults restored (pause=False per _DEFAULT_STATE)
        # This is intentional: reload() signals the *current* file state;
        # the orchestrator must not call reload() in a loop that erases the flag.
        assert ctrl.pause_new_entries is False  # default after reload with no file


# ─────────────────────────────────────────────────────────────────────────────
# 3. File already exists → no-op
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapNoOp:

    def test_live_paper_noop_when_file_exists(self, tmp_path):
        ctrl = _ctrl_existing(tmp_path)
        ctrl.bootstrap(is_live_real=False)
        assert ctrl.bootstrap_event is None
        # Original content preserved
        data = json.loads((tmp_path / "control.json").read_text())
        assert data["reason"] == "pre-existing"

    def test_live_real_noop_when_file_exists(self, tmp_path):
        ctrl = _ctrl_existing(tmp_path)
        ctrl.bootstrap(is_live_real=True)
        assert ctrl.bootstrap_event is None
        assert ctrl.control_missing is False

    def test_control_missing_defaults_false(self, tmp_path):
        """control_missing should be False on a normal startup (file present)."""
        ctrl = _ctrl_existing(tmp_path)
        assert ctrl.control_missing is False

    def test_bootstrap_event_defaults_none(self, tmp_path):
        ctrl = _ctrl_existing(tmp_path)
        assert ctrl.bootstrap_event is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. LIVE_PAPER + example present → content seeded from example
# ─────────────────────────────────────────────────────────────────────────────

class TestLivePaperExampleSeeding:

    def test_seeds_from_example(self, tmp_path):
        """risk_mode from example should survive into created file (merged with defaults)."""
        ctrl = _ctrl(tmp_path, with_example=True,
                     example_content={"risk_mode": "MEDIUM", "pause_new_entries": False})
        ctrl.bootstrap(is_live_real=False)
        data = json.loads((tmp_path / "control.json").read_text())
        # pause always forced to True regardless of example
        assert data["pause_new_entries"] is True
        # other keys from example are preserved
        assert data.get("risk_mode") == "MEDIUM"

    def test_pause_always_forced_even_if_example_unpaused(self, tmp_path):
        ctrl = _ctrl(tmp_path, with_example=True,
                     example_content={"pause_new_entries": False})
        ctrl.bootstrap(is_live_real=False)
        assert ctrl.pause_new_entries is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. LIVE_PAPER + example missing → uses defaults, no crash
# ─────────────────────────────────────────────────────────────────────────────

class TestLivePaperNoExample:

    def test_no_crash_without_example(self, tmp_path):
        ctrl = _ctrl(tmp_path, with_example=False)
        ctrl.bootstrap(is_live_real=False)  # must not raise
        assert (tmp_path / "control.json").exists()
        assert ctrl.pause_new_entries is True
        assert ctrl.bootstrap_event == "BOOTSTRAP_CREATED"


# ─────────────────────────────────────────────────────────────────────────────
# 6. is_backtest=True → always no-op
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestNoOp:

    def test_backtest_live_paper_noop(self, tmp_path):
        ctrl_path = tmp_path / "control.json"
        ctrl = ControlState(path=ctrl_path, is_backtest=True)
        ctrl.bootstrap(is_live_real=False)
        assert not ctrl_path.exists()
        assert ctrl.bootstrap_event is None

    def test_backtest_live_real_noop(self, tmp_path):
        ctrl_path = tmp_path / "control.json"
        ctrl = ControlState(path=ctrl_path, is_backtest=True)
        ctrl.bootstrap(is_live_real=True)
        assert not ctrl_path.exists()
        assert ctrl.control_missing is False
        assert ctrl.bootstrap_event is None
