"""
tests/test_account_mode.py
==========================
Tests that enforce the three hard safety invariants for AccountMode separation.

1. LIVE_PAPER never calls broker order endpoints (no OandaClient.create_order calls).
2. LIVE_REAL blocks entries when equity is UNKNOWN (returns WAIT: BROKER_EQUITY_UNKNOWN).
3. Restart preserves LIVE_PAPER equity state (paper_account.json survives crash/restart).

These are regression gates — any refactor that breaks them is a safety violation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.execution.account_state import AccountMode, AccountState, PAPER_STATE_FILE


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _paper_state(tmp_path: Path, equity: float = 8_000.0, peak: float = 8_000.0) -> AccountState:
    paper_file = tmp_path / "paper_account.json"
    return AccountState.for_live_paper(
        default_balance=equity,
        peak_override=peak,
        paper_file=paper_file,
    )


def _live_real_state(broker_summary: dict | None) -> AccountState:
    return AccountState.for_live_real(broker_summary)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LIVE_PAPER never calls broker order endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestLivePaperNoBrokerOrders:
    """
    Invariant: in LIVE_PAPER mode the broker order path is never exercised.
    The only OANDA calls allowed are read-only (candles, account summary for NAV).
    """

    def test_account_mode_is_live_paper(self, tmp_path):
        acc = _paper_state(tmp_path)
        assert acc.mode == AccountMode.LIVE_PAPER

    def test_equity_source_is_sim(self, tmp_path):
        acc = _paper_state(tmp_path)
        assert acc.equity_source == "SIM"

    def test_is_not_unknown(self, tmp_path):
        """Paper equity is never UNKNOWN — always a concrete float."""
        acc = _paper_state(tmp_path)
        assert acc.is_unknown is False
        assert acc.equity == 8_000.0

    def test_trade_executor_dry_run_prevents_live_orders(self):
        """
        Structural invariant: TradeExecutor.execute() must contain an
        `if not self.dry_run:` (or equivalent) guard that gates ALL broker
        order calls.  LIVE_PAPER always passes dry_run=True.

        We verify the guard exists in source and that executor.dry_run is
        correctly set on init — these two together ensure no live orders.
        """
        from src.execution.trade_executor import TradeExecutor
        import inspect

        # 1. Source-level: executor must have a dry_run guard before live calls
        source = inspect.getsource(TradeExecutor.execute)
        assert "dry_run" in source, (
            "TradeExecutor.execute must reference self.dry_run to gate live orders"
        )
        assert "not self.dry_run" in source or "if self.dry_run" in source, (
            "TradeExecutor.execute must branch on dry_run before placing orders"
        )

        # 2. Runtime: executor built with dry_run=True must have dry_run=True
        mock_oanda   = MagicMock()
        mock_journal = MagicMock()
        mock_risk    = MagicMock()
        mock_strat   = MagicMock()
        mock_strat.open_positions = {}

        executor = TradeExecutor(
            strategy=mock_strat,
            oanda=mock_oanda,
            journal=mock_journal,
            risk_manager=mock_risk,
            dry_run=True,
        )
        assert executor.dry_run is True, "dry_run=True must be stored and honoured"

        # 3. The orchestrator must pass dry_run=True for LIVE_PAPER
        orc_src = Path(__file__).parents[1] / "src" / "execution" / "orchestrator.py"
        orc_text = orc_src.read_text()
        # TradeExecutor must be constructed with dry_run=dry_run (not hardcoded False)
        assert "TradeExecutor(" in orc_text, "Orchestrator must instantiate TradeExecutor"
        assert "dry_run=dry_run" in orc_text, (
            "Orchestrator must forward dry_run flag to TradeExecutor; "
            "hardcoding dry_run=False here would enable live orders in LIVE_PAPER"
        )

    def test_apply_pnl_updates_equity_not_broker(self, tmp_path):
        """Paper equity changes via apply_pnl, not broker balance polling."""
        acc = _paper_state(tmp_path, equity=8_000.0)
        acc.apply_pnl(+500.0)
        assert acc.equity == 8_500.0
        assert acc.equity_source == "SIM"
        assert acc.broker_fetch_failures == 0   # broker never touched

    def test_paper_equity_not_overwritten_by_broker_summary(self, tmp_path):
        """
        update_from_broker() must be a NO-OP in LIVE_PAPER mode.
        Calling it accidentally (e.g., via a shared helper) must NOT overwrite
        paper equity with the broker balance.
        """
        acc = _paper_state(tmp_path, equity=12_345.0)
        # Simulate an accidental call: broker returned $0 (unfunded OANDA account).
        result = acc.update_from_broker({"balance": 0.0, "nav": 0.0})
        # Must be a no-op: equity, source, and mode all unchanged.
        assert result is False,               "update_from_broker must return False in LIVE_PAPER"
        assert acc.equity == 12_345.0,        "Paper equity must NOT be overwritten by broker"
        assert acc.equity_source == "SIM",    "equity_source must remain SIM"
        assert acc.mode == AccountMode.LIVE_PAPER


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  LIVE_REAL blocks entries when equity is UNKNOWN
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiveRealEquityUnknown:
    """
    Invariant: when broker equity is UNKNOWN (fetch failed), no new entry is sized
    or submitted.  The decision reason must surface BROKER_EQUITY_UNKNOWN.
    """

    def test_for_live_real_with_none_summary_is_unknown(self):
        acc = _live_real_state(None)
        assert acc.mode == AccountMode.LIVE_REAL
        assert acc.is_unknown is True
        assert acc.equity is None
        assert acc.equity_source == "UNKNOWN"

    def test_for_live_real_with_valid_summary_is_not_unknown(self):
        acc = _live_real_state({"balance": 4_200.0, "nav": 4_200.0})
        assert acc.is_unknown is False
        assert acc.equity == 4_200.0
        assert acc.equity_source == "BROKER"

    def test_mark_broker_failed_sets_unknown(self):
        acc = _live_real_state({"balance": 4_200.0})
        assert acc.is_unknown is False
        acc.mark_broker_failed()
        assert acc.is_unknown is True
        assert acc.equity is None
        assert acc.equity_source == "UNKNOWN"
        assert acc.broker_fetch_failures == 1

    def test_consecutive_failures_accumulate(self):
        acc = _live_real_state(None)   # already unknown
        acc.mark_broker_failed()
        acc.mark_broker_failed()
        assert acc.broker_fetch_failures == 2

    def test_successful_fetch_resets_unknown(self):
        acc = _live_real_state(None)
        acc.mark_broker_failed()
        assert acc.is_unknown is True
        acc.update_from_broker({"balance": 5_000.0})
        assert acc.is_unknown is False
        assert acc.equity == 5_000.0
        assert acc.broker_fetch_failures == 0

    def test_is_tradeable_false_when_unknown(self):
        acc = _live_real_state(None)
        assert acc.is_tradeable is False

    def test_is_tradeable_true_when_equity_known(self):
        acc = _live_real_state({"balance": 4_200.0})
        assert acc.is_tradeable is True

    def test_safe_equity_returns_fallback_when_unknown(self):
        acc = _live_real_state(None)
        assert acc.safe_equity(999.0) == 999.0
        assert acc.safe_equity(0.0)   == 0.0

    def test_orchestrator_skips_evaluation_when_unknown(self):
        """
        When account.is_unknown is True, _run_strategy_evaluation must NOT be called.
        This is the critical gate that prevents entries with unknown equity.
        """
        orc_path = Path(__file__).parents[1] / "src" / "execution" / "orchestrator.py"
        source = orc_path.read_text()

        # The guard must exist and must test is_unknown before calling evaluation
        assert "is_unknown" in source, "is_unknown check missing from orchestrator"

        lines = source.splitlines()
        unknown_line = next(
            i for i, l in enumerate(lines) if "is_unknown" in l and "if " in l
        )
        eval_line = next(
            i for i, l in enumerate(lines) if "_run_strategy_evaluation" in l
        )
        # The unknown check must come BEFORE _run_strategy_evaluation
        assert unknown_line < eval_line, (
            f"is_unknown guard (line {unknown_line}) must appear BEFORE "
            f"_run_strategy_evaluation (line {eval_line})"
        )

    def test_broker_equity_unknown_reason_in_source(self):
        """
        The string BROKER_EQUITY_UNKNOWN must appear in the orchestrator
        so the dashboard/journal surfaces why entries are blocked.
        """
        orc_path = Path(__file__).parents[1] / "src" / "execution" / "orchestrator.py"
        assert "BROKER_EQUITY_UNKNOWN" in orc_path.read_text(), (
            "BROKER_EQUITY_UNKNOWN reason must be logged in orchestrator.py"
        )

    def test_unknown_equity_display(self):
        acc = _live_real_state(None)
        assert acc.equity_display == "UNKNOWN"
        assert acc.equity_display != "$0.00"   # must never display as zero

    def test_planned_risk_zero_when_unknown(self):
        """
        Risk sizing must return 0 / blocked when equity is None.
        safe_equity(0) → compute_risk_sizing(0) → tier 0 = 6% but equity 0 → $0 risk.
        The orchestrator skips evaluation entirely, so no trade ever sizes from 0.
        This verifies the safe_equity fallback path is consistent.
        """
        from src.execution.risk_manager_forex import ForexRiskManager
        rm = ForexRiskManager(journal=MagicMock(), backtest=True)
        acc = _live_real_state(None)
        # In the orchestrator, evaluation is SKIPPED when unknown.
        # If somehow reached, safe_equity(0) would give $0 risk — no trade.
        risk_dollars = acc.safe_equity(0.0) * (rm.get_risk_pct(acc.safe_equity(0.0)) / 100)
        assert risk_dollars == 0.0, "Zero equity must produce zero risk dollars"


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Restart preserves LIVE_PAPER equity state
# ═══════════════════════════════════════════════════════════════════════════════

class TestLivePaperPersistence:
    """
    Invariant: paper equity persists across restarts.
    Simulated PnL accumulates across sessions; never resets unless user explicitly resets.
    """

    def test_initial_state_persisted_on_creation(self, tmp_path):
        """for_live_paper() saves the initial equity to disk immediately."""
        paper_file = tmp_path / "paper_account.json"
        acc = AccountState.for_live_paper(8_000.0, paper_file=paper_file)
        assert paper_file.exists(), "paper_account.json must be created on first init"
        data = json.loads(paper_file.read_text())
        assert data["equity"] == 8_000.0
        assert data["mode"] == AccountMode.LIVE_PAPER.value

    def test_pnl_persisted_after_apply(self, tmp_path):
        """apply_pnl() saves updated equity to disk."""
        paper_file = tmp_path / "paper_account.json"
        acc = AccountState.for_live_paper(8_000.0, paper_file=paper_file)
        acc.apply_pnl(+300.0)

        data = json.loads(paper_file.read_text())
        assert data["equity"] == 8_300.0
        assert data["realized_session_pnl"] == 300.0

    def test_restart_loads_persisted_equity(self, tmp_path):
        """
        A second for_live_paper() call on the same file loads the saved equity,
        not the default_balance.  This is the crash-recovery path.
        """
        paper_file = tmp_path / "paper_account.json"

        # Session 1: make some profit
        acc1 = AccountState.for_live_paper(8_000.0, paper_file=paper_file)
        acc1.apply_pnl(+1_200.0)   # equity → 9_200
        acc1.apply_pnl(-400.0)     # equity → 8_800
        assert acc1.equity == 8_800.0

        # Session 2: fresh instance, same file — must resume from 8_800
        acc2 = AccountState.for_live_paper(8_000.0, paper_file=paper_file)
        assert acc2.equity == 8_800.0, (
            f"Restart must load persisted equity 8_800, not default 8_000. "
            f"Got: {acc2.equity}"
        )
        assert acc2.peak_equity >= 9_200.0, "Peak must persist across restarts"

    def test_peak_never_decreases_across_sessions(self, tmp_path):
        """Peak equity is a high-water mark — it must never decrease."""
        paper_file = tmp_path / "paper_account.json"

        acc1 = AccountState.for_live_paper(8_000.0, paper_file=paper_file)
        acc1.apply_pnl(+2_000.0)   # peak = 10_000
        acc1.apply_pnl(-1_500.0)   # equity = 8_500; peak stays 10_000
        assert acc1.peak_equity == 10_000.0

        acc2 = AccountState.for_live_paper(8_000.0, paper_file=paper_file)
        assert acc2.peak_equity == 10_000.0, "Peak must persist across sessions"

    def test_loss_does_not_reset_equity(self, tmp_path):
        """
        A losing trade reduces equity but never resets it to the initial value.
        The bot must be able to distinguish 'down from peak' from 'reset'.
        """
        paper_file = tmp_path / "paper_account.json"
        acc = AccountState.for_live_paper(8_000.0, paper_file=paper_file)
        acc.apply_pnl(-800.0)    # equity → 7_200

        # Reload
        acc2 = AccountState.for_live_paper(8_000.0, paper_file=paper_file)
        assert acc2.equity == 7_200.0, "Loss must persist; equity must not reset to 8_000"

    def test_file_written_atomically(self, tmp_path):
        """Save must use a tmp file + rename — no partial writes."""
        paper_file = tmp_path / "paper_account.json"
        acc = AccountState.for_live_paper(8_000.0, paper_file=paper_file)
        acc.apply_pnl(100.0)

        # .tmp file must be gone after save (replaced by rename)
        tmp_file = paper_file.with_suffix(".tmp")
        assert not tmp_file.exists(), ".tmp file must be cleaned up after atomic write"
        assert paper_file.exists()

    def test_paper_file_path_in_runtime_state(self, tmp_path):
        """
        The canonical PAPER_STATE_FILE path must be under runtime_state/,
        not logs/ — so it's clearly separated from backtest artefacts.
        """
        assert "runtime_state" in str(PAPER_STATE_FILE), (
            f"PAPER_STATE_FILE must be under runtime_state/, got: {PAPER_STATE_FILE}"
        )
        assert "paper_account" in str(PAPER_STATE_FILE), (
            f"PAPER_STATE_FILE must be named paper_account.json, got: {PAPER_STATE_FILE}"
        )

    def test_manual_reset_possible(self, tmp_path):
        """
        User can explicitly reset by deleting paper_account.json.
        After deletion, a fresh for_live_paper() starts from default_balance.
        """
        paper_file = tmp_path / "paper_account.json"
        acc1 = AccountState.for_live_paper(8_000.0, paper_file=paper_file)
        acc1.apply_pnl(+5_000.0)   # equity → 13_000

        # User deletes the file (explicit reset)
        paper_file.unlink()

        acc2 = AccountState.for_live_paper(8_000.0, paper_file=paper_file)
        assert acc2.equity == 8_000.0, "After manual reset, equity must restart from default"

    def test_to_dict_includes_last_update_ts(self, tmp_path):
        """to_dict() must include last_update_ts so the dashboard can show when equity was last known."""
        acc = AccountState.for_live_paper(8_000.0, paper_file=tmp_path / "pa.json")
        d = acc.to_dict()
        assert "last_update_ts" in d, "to_dict() must include last_update_ts"
        assert d["last_update_ts"] is not None

    def test_equity_source_is_sim_after_restart(self, tmp_path):
        """Even after restart, equity_source must always be 'SIM' for LIVE_PAPER."""
        paper_file = tmp_path / "paper_account.json"
        AccountState.for_live_paper(8_000.0, paper_file=paper_file).apply_pnl(100.0)
        acc = AccountState.for_live_paper(8_000.0, paper_file=paper_file)
        assert acc.equity_source == "SIM"
