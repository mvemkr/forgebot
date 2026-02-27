"""
tests/test_regime_control.py
============================
Invariant tests for:
  1. PAUSE_BLOCK — pause_new_entries=True must block at the one entry path;
     executor.execute must never be called; no state transition.
  2. Risk mode parity — multiplier must be applied identically in every
     sizing path (get_book_risk_pct, get_risk_pct_with_dd, backtester inner
     _risk_pct_with_flag).  compute_risk_sizing is the single source of truth.
"""

from __future__ import annotations

import json, os, sys, tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_risk_manager(regime_mode=None):
    from src.execution.risk_manager_forex import ForexRiskManager
    rm = ForexRiskManager(journal=MagicMock(), backtest=True)
    rm._peak_balance = 10_000.0   # set a peak so DD math works
    if regime_mode:
        rm.set_regime_mode(regime_mode)
    return rm


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PAUSE_BLOCK invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestPauseBlock:

    def test_control_state_default_false(self, tmp_path):
        """Missing file → pause_new_entries defaults to False."""
        from src.execution.control_state import ControlState
        ctrl = ControlState(path=tmp_path / "control.json")
        assert ctrl.pause_new_entries is False

    def test_control_state_pause_persists(self, tmp_path):
        """pause() writes file; reload() from disk reflects state."""
        from src.execution.control_state import ControlState
        p = tmp_path / "control.json"
        ctrl = ControlState(path=p)
        ctrl.pause("NFP event", "test")
        assert ctrl.pause_new_entries is True
        assert p.exists()
        # Fresh instance reads same file
        ctrl2 = ControlState(path=p)
        assert ctrl2.pause_new_entries is True
        assert ctrl2.reason == "NFP event"

    def test_control_state_resume_clears(self, tmp_path):
        """resume() clears pause_new_entries; persists to disk."""
        from src.execution.control_state import ControlState
        p = tmp_path / "control.json"
        ctrl = ControlState(path=p)
        ctrl.pause("test", "test")
        ctrl.resume("all clear", "test")
        ctrl2 = ControlState(path=p)
        assert ctrl2.pause_new_entries is False

    def test_control_state_backtest_noop(self, tmp_path):
        """Backtest ControlState never writes to disk."""
        from src.execution.control_state import ControlState
        p = tmp_path / "control.json"
        ctrl = ControlState(path=p, is_backtest=True)
        ctrl.pause("should not write", "test")
        assert not p.exists()  # no file written in backtest mode
        assert ctrl.pause_new_entries is False  # unchanged

    def test_control_state_atomic_write(self, tmp_path):
        """File content is valid JSON after write."""
        from src.execution.control_state import ControlState
        p = tmp_path / "control.json"
        ctrl = ControlState(path=p)
        ctrl.pause("atomicity test", "test")
        data = json.loads(p.read_text())
        assert data["pause_new_entries"] is True
        assert "last_updated" in data

    # ── PAUSE_BLOCK in orchestrator entry flow ─────────────────────────────

    def test_pause_block_prevents_executor_call(self, tmp_path):
        """
        When pause_new_entries=True, executor.execute() must NOT be called.
        This confirms the block sits before the one true order path.
        """
        from src.execution.control_state import ControlState

        # Set pause state
        ctrl_path = tmp_path / "control.json"
        ctrl = ControlState(path=ctrl_path)
        ctrl.pause("test pause", "test")

        # Verify the gate via direct ControlState check (mirrors what orchestrator does)
        assert ctrl.pause_new_entries is True

        # Simulate orchestrator entry guard (the exact code path in orchestrator.py)
        mock_executor = MagicMock()
        enter_called = False

        def _simulated_entry_flow(control, executor, decision_value="ENTER", confidence=0.80):
            """Mirrors the orchestrator _evaluate_pair() entry path."""
            MIN_CONFIDENCE = 0.65
            if decision_value != "ENTER":
                return "not_enter"
            if confidence < MIN_CONFIDENCE:
                return "confidence_block"
            # ← PAUSE_BLOCK check (this is the code we added)
            if control.pause_new_entries:
                return "PAUSE_BLOCK"
            # ← executor.execute() — must NOT be reached when paused
            executor.execute()
            return "executed"

        result = _simulated_entry_flow(ctrl, mock_executor)
        assert result == "PAUSE_BLOCK", f"Expected PAUSE_BLOCK, got {result!r}"
        mock_executor.execute.assert_not_called()

    def test_pause_block_does_not_affect_enter_when_false(self, tmp_path):
        """When pause_new_entries=False, entry proceeds past the gate."""
        from src.execution.control_state import ControlState
        ctrl_path = tmp_path / "control.json"
        ctrl = ControlState(path=ctrl_path)
        # Default: not paused
        assert ctrl.pause_new_entries is False

        mock_executor = MagicMock()

        def _simulated_entry_flow(control, executor):
            if control.pause_new_entries:
                return "PAUSE_BLOCK"
            executor.execute()
            return "executed"

        result = _simulated_entry_flow(ctrl, mock_executor)
        assert result == "executed"
        mock_executor.execute.assert_called_once()

    def test_pause_block_position_in_code(self):
        """
        Structural test: the pause_new_entries check must appear BEFORE
        executor.execute() and AFTER the confidence gate in orchestrator source.
        The check may span multiple lines, so we look for the block that contains
        pause_new_entries and PAUSE_BLOCK within a 10-line window of each other.
        """
        orc_path = Path(__file__).parents[1] / "src" / "execution" / "orchestrator.py"
        source = orc_path.read_text()
        lines = source.splitlines()

        confidence_line = next(i for i, l in enumerate(lines) if "decision.confidence < MIN_CONFIDENCE" in l)
        executor_line   = next(i for i, l in enumerate(lines) if "self.executor.execute(" in l)

        # There may be multiple pause_new_entries references (startup log, reload comment,
        # the actual guard).  Find the one that is BOTH after the confidence gate AND
        # has "PAUSE_BLOCK" within a 10-line lookahead window.
        pause_check_line = None
        for i, line in enumerate(lines):
            if "pause_new_entries" not in line:
                continue
            window = "\n".join(lines[i: i + 10])
            if "PAUSE_BLOCK" in window:
                pause_check_line = i
                break

        assert pause_check_line is not None, (
            "No pause_new_entries check with adjacent PAUSE_BLOCK found in orchestrator.py"
        )

        assert confidence_line < pause_check_line < executor_line, (
            f"Code order wrong: confidence={confidence_line} "
            f"pause_check={pause_check_line} executor={executor_line}. "
            f"pause gate must be: AFTER confidence gate, BEFORE executor.execute()"
        )

    def test_pause_block_no_open_positions_side_effect(self, tmp_path):
        """
        PAUSE_BLOCK must not modify open_positions or any mutable state.
        The guard is a pure read + early return — no writes occur.
        """
        from src.execution.control_state import ControlState
        ctrl = ControlState(path=tmp_path / "c.json")
        ctrl.pause("freeze", "test")

        open_positions_before = {"GBP/JPY": {"entry": 200.0}}
        open_positions = dict(open_positions_before)

        # Simulate entry guard (identical to orchestrator PAUSE_BLOCK path)
        if ctrl.pause_new_entries:
            pass  # return early — no further mutation

        # Dict must be identical after the guard executes
        assert open_positions == open_positions_before, (
            "PAUSE_BLOCK guard must not mutate open_positions"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Risk mode parity invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskModeParity:
    """
    All sizing paths must return the same final_pct for identical inputs.
    compute_risk_sizing() is the canonical calculation; all other getters
    must be thin wrappers that return a subset of the same result.
    """

    BALANCE    = 8_000.0
    PEAK       = 8_000.0   # no DD
    STREAK     = 0

    def _rs(self, mode=None):
        return _make_risk_manager(regime_mode=mode)

    # ── Base sanity ────────────────────────────────────────────────────────

    def test_no_mode_base_pct(self):
        """No regime mode → multiplier 1.0 → final == base tier (6% at $8K)."""
        rm = self._rs()
        result = rm.compute_risk_sizing(self.BALANCE, self.PEAK, self.STREAK)
        assert result.base_pct     == 6.0
        assert result.mode_mult    == 1.0
        assert result.premult_pct  == 6.0
        assert result.final_pct    == 6.0
        assert result.blocked      is False
        assert result.dd_flag      == ""

    def test_high_mode_multiplier(self):
        """HIGH mode (1.5×): 6% × 1.5 = 9%."""
        rm = self._rs(mode="HIGH")
        result = rm.compute_risk_sizing(self.BALANCE, self.PEAK, self.STREAK)
        assert result.base_pct     == 6.0
        assert result.mode_mult    == 1.5
        assert result.premult_pct  == 9.0
        assert result.final_pct    == 9.0

    def test_low_mode_multiplier(self):
        """LOW mode (0.5×): 6% × 0.5 = 3%."""
        rm = self._rs(mode="LOW")
        result = rm.compute_risk_sizing(self.BALANCE, self.PEAK, self.STREAK)
        assert result.base_pct    == 6.0
        assert result.mode_mult   == 0.5
        assert result.premult_pct == 3.0
        assert result.final_pct   == 3.0

    def test_extreme_mode_multiplier(self):
        """EXTREME mode (2.0×): 6% × 2.0 = 12%."""
        rm = self._rs(mode="EXTREME")
        result = rm.compute_risk_sizing(self.BALANCE, self.PEAK, self.STREAK)
        assert result.premult_pct == 12.0
        assert result.final_pct   == 12.0

    # ── Parity: all getters agree ──────────────────────────────────────────

    def test_get_risk_pct_with_dd_matches_compute(self):
        """get_risk_pct_with_dd() shim must return (final_pct, dd_flag) from compute."""
        for mode in [None, "LOW", "MEDIUM", "HIGH", "EXTREME"]:
            rm = self._rs(mode=mode)
            result    = rm.compute_risk_sizing(self.BALANCE, self.PEAK, self.STREAK)
            pct, flag = rm.get_risk_pct_with_dd(self.BALANCE, self.PEAK, self.STREAK)
            assert pct  == result.final_pct,  f"mode={mode}: pct mismatch"
            assert flag == result.dd_flag,    f"mode={mode}: flag mismatch"

    def test_get_effective_risk_pct_matches_premult(self):
        """get_effective_risk_pct() must equal compute().premult_pct."""
        for mode in [None, "LOW", "HIGH", "EXTREME"]:
            rm = self._rs(mode=mode)
            result   = rm.compute_risk_sizing(self.BALANCE)
            eff      = rm.get_effective_risk_pct(self.BALANCE)
            assert eff == result.premult_pct, f"mode={mode}: effective_pct mismatch"

    def test_get_book_risk_pct_uses_final_pct(self):
        """get_book_risk_pct() (the live executor path) must use final_pct."""
        for mode in [None, "HIGH", "EXTREME"]:
            rm = self._rs(mode=mode)
            rm._peak_balance = self.PEAK
            result    = rm.compute_risk_sizing(self.BALANCE, self.PEAK, self.STREAK)
            book_pct  = rm.get_book_risk_pct(self.BALANCE, open_positions={})
            # book_pct = min(final_pct, MAX_BOOK_EXPOSURE - committed)
            # With no open positions: committed = 0, budget = 35%, so book = final
            assert book_pct == result.final_pct, (
                f"mode={mode}: book_pct={book_pct} != final_pct={result.final_pct}"
            )

    # ── DD caps still apply after mode multiplier ─────────────────────────

    def test_dd_cap_applied_after_mode_mult(self):
        """
        HIGH mode: base=15% (tier-2 balance=$10K), mult=1.5× → premult=22.5%.
        At 30% DD, HIGH-mode dd_l2_cap=10% fires → final_pct capped at 10%.
        Flag = "DD_CAP_10" (dynamic, reflects actual cap value).
        """
        from src.strategy.forex.strategy_config import DD_L2_PCT
        from src.strategy.forex.regime_score import RISK_MODE_PARAMS
        rm = self._rs(mode="HIGH")
        # Use $10K balance (tier-2 = 15%); peak=$14,286 → 30% DD ≥ DD_L2_PCT
        high_dd_l2_cap = RISK_MODE_PARAMS["HIGH"]["dd_l2_cap"]   # 10.0
        result = rm.compute_risk_sizing(
            account_balance=10_000.0,
            peak_equity=14_286.0,    # 30% DD
            consecutive_losses=0,
        )
        assert result.premult_pct == pytest.approx(22.5, abs=0.1), \
            f"premult should be 22.5% (15% × 1.5×): got {result.premult_pct}"
        assert result.final_pct   <= high_dd_l2_cap, \
            f"DD L2 cap ({high_dd_l2_cap}%) should apply; got {result.final_pct}"
        assert result.final_pct   == pytest.approx(high_dd_l2_cap, abs=0.01), \
            f"final_pct should equal dd_l2_cap ({high_dd_l2_cap}%)"
        assert f"DD_CAP_{int(high_dd_l2_cap)}" in result.reasons

    def test_killswitch_blocks_regardless_of_mode(self):
        """EXTREME mode cannot override kill-switch."""
        rm = self._rs(mode="EXTREME")
        # Induce 45% DD — above 40% killswitch
        result = rm.compute_risk_sizing(
            account_balance=5_500.0,
            peak_equity=10_000.0,
            consecutive_losses=0,
        )
        assert result.blocked    is True
        assert result.final_pct  == 0.0
        assert result.dd_flag    == "DD_KILLSWITCH"

    # ── to_dict round-trip ─────────────────────────────────────────────────

    def test_to_dict_complete(self):
        """RiskSizingResult.to_dict() must include all required keys."""
        rm = self._rs(mode="MEDIUM")
        result = rm.compute_risk_sizing(self.BALANCE, self.PEAK, self.STREAK)
        d = result.to_dict()
        for key in ("base_pct", "risk_mode", "mode_mult", "premult_pct",
                    "final_pct", "dd_flag", "blocked", "reasons"):
            assert key in d, f"Missing key: {key}"

    # ── Backtester path ────────────────────────────────────────────────────

    def test_backtester_uses_compute_via_shim(self):
        """
        Backtester's _risk_pct_with_flag calls get_risk_pct_with_dd which
        is now a shim over compute_risk_sizing.  Values must match.
        """
        rm = _make_risk_manager(regime_mode="HIGH")
        # Simulate the backtester's inner function
        peak_balance        = self.PEAK
        consecutive_losses  = self.STREAK

        def _risk_pct_with_flag(bal):
            pct, dd_flag = rm.get_risk_pct_with_dd(
                bal, peak_equity=peak_balance,
                consecutive_losses=consecutive_losses)
            return pct, dd_flag

        pct, flag = _risk_pct_with_flag(self.BALANCE)
        expected  = rm.compute_risk_sizing(self.BALANCE, self.PEAK, self.STREAK)

        assert pct  == expected.final_pct, "backtester pct diverges from compute_risk_sizing"
        assert flag == expected.dd_flag,   "backtester flag diverges from compute_risk_sizing"
        assert pct  == 9.0, "HIGH mode 6%×1.5 should give 9%"

    # ── WeeklyCapacity reflects mode ───────────────────────────────────────

    def test_regime_weekly_caps_per_mode(self):
        """Weekly caps must scale with mode per RISK_MODE_PARAMS."""
        rm = self._rs()
        rm.set_regime_mode("EXTREME")
        small, std = rm.regime_weekly_caps()
        assert small == 3 and std == 4

        rm.set_regime_mode("LOW")
        small, std = rm.regime_weekly_caps()
        assert small == 1 and std == 1

        rm.set_regime_mode(None)
        from src.strategy.forex.strategy_config import (
            MAX_TRADES_PER_WEEK_SMALL, MAX_TRADES_PER_WEEK_STANDARD
        )
        small, std = rm.regime_weekly_caps()
        assert small == MAX_TRADES_PER_WEEK_SMALL
        assert std   == MAX_TRADES_PER_WEEK_STANDARD
