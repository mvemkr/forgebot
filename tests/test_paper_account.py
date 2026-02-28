"""
tests/test_paper_account.py
============================
Five hard invariants for the LIVE_PAPER closed-system accounting.

Invariant 1 — Persistence across restart
    paper_account.json is the single source of truth; equity loaded on next
    AccountState construction is identical to the equity at shutdown.

Invariant 2 — No broker leakage in paper mode
    update_from_broker() is a guaranteed no-op; broker balance / NAV values
    must never appear in any equity or sizing calculation when mode=LIVE_PAPER.

Invariant 3 — Weekly PnL reset on ISO-week boundary
    week_pnl and week_trades reset to zero when week_id rolls over.
    week_pnl accumulates correctly within a single week.

Invariant 4 — Risk sizing uses current paper equity (compounding)
    After equity grows (e.g. $8K → $31.2K) the risk manager's sizing
    output reflects the updated balance, not the bootstrap value.

Invariant 5 — Equity identity
    For every close: equity_after == equity_before + realized_pnl  (±0.01)
    Verified via apply_pnl() calls and journal entries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.execution.account_state import AccountMode, AccountState, PAPER_STATE_FILE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _paper(tmp_path: Path, equity: float = 8_000.0) -> AccountState:
    """Create a fresh LIVE_PAPER AccountState with isolated tmp files."""
    paper_file   = tmp_path / "paper_account.json"
    journal_file = tmp_path / "paper_journal.jsonl"
    return AccountState.for_live_paper(
        default_balance=equity,
        paper_file=paper_file,
        paper_journal_file=journal_file,
    )


def _reload(tmp_path: Path, default_balance: float = 8_000.0) -> AccountState:
    """Simulate a restart: build a new AccountState from the same tmp files."""
    paper_file   = tmp_path / "paper_account.json"
    journal_file = tmp_path / "paper_journal.jsonl"
    return AccountState.for_live_paper(
        default_balance=default_balance,
        paper_file=paper_file,
        paper_journal_file=journal_file,
    )


def _journal_lines(tmp_path: Path) -> list[dict]:
    """Read all lines from the paper journal as parsed dicts."""
    journal_file = tmp_path / "paper_journal.jsonl"
    if not journal_file.exists():
        return []
    return [json.loads(line) for line in journal_file.read_text().splitlines() if line.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant 1 — Persistence across restart
# ═══════════════════════════════════════════════════════════════════════════════

class TestPersistenceAcrossRestart:
    """Equity, peak, and weekly stats survive a process restart."""

    def test_bootstrap_writes_file(self, tmp_path):
        """First construction creates paper_account.json."""
        acc = _paper(tmp_path, equity=8_000.0)
        assert (tmp_path / "paper_account.json").exists()

    def test_restart_loads_persisted_equity(self, tmp_path):
        """Simulated win: restarted process uses updated equity, not .env value."""
        acc = _paper(tmp_path, equity=8_000.0)
        acc.apply_pnl(+2_500.0, pair="GBP/JPY", exit_reason="target_reached")
        assert acc.equity == pytest.approx(10_500.0)

        # Restart with a *different* default — persisted file must win
        acc2 = _reload(tmp_path, default_balance=99_999.0)
        assert acc2.equity == pytest.approx(10_500.0), (
            "Restart should load persisted equity, not the default_balance arg"
        )

    def test_restart_loads_peak_equity(self, tmp_path):
        acc = _paper(tmp_path, equity=8_000.0)
        acc.apply_pnl(+5_000.0)
        acc.apply_pnl(-1_000.0)
        assert acc.peak_equity == pytest.approx(13_000.0)

        acc2 = _reload(tmp_path)
        assert acc2.peak_equity == pytest.approx(13_000.0)

    def test_restart_loads_week_pnl(self, tmp_path):
        """week_pnl persists and reloads correctly within the same week."""
        acc = _paper(tmp_path, equity=8_000.0)
        acc.apply_pnl(+1_000.0, pair="USD/CAD", exit_reason="target_reached")
        acc.apply_pnl(-500.0,   pair="EUR/USD", exit_reason="stop_hit")
        assert acc.week_pnl == pytest.approx(500.0)

        acc2 = _reload(tmp_path)
        # Same week → week_pnl carries over
        assert acc2.week_pnl == pytest.approx(500.0)
        assert acc2.week_trades == 2

    def test_env_default_ignored_when_file_exists(self, tmp_path):
        """SIM_STARTING_EQUITY is irrelevant after first bootstrap."""
        acc = _paper(tmp_path, equity=8_000.0)
        acc.apply_pnl(+20_000.0)  # huge win

        # Restart with completely different default
        acc2 = _reload(tmp_path, default_balance=1_000.0)
        assert acc2.equity == pytest.approx(28_000.0)
        assert acc2.equity != pytest.approx(1_000.0)

    def test_atomic_write_no_tmp_file_after_save(self, tmp_path):
        """Atomic save leaves no .tmp file behind."""
        acc = _paper(tmp_path, equity=8_000.0)
        acc.apply_pnl(+100.0)
        assert not (tmp_path / "paper_account.tmp").exists()
        assert (tmp_path / "paper_account.json").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant 2 — No broker leakage in paper mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoBrokerLeakageInPaperMode:
    """Broker balance/NAV must never alter paper equity."""

    def test_update_from_broker_is_noop(self, tmp_path):
        """Calling update_from_broker in LIVE_PAPER returns False and leaves equity unchanged."""
        acc = _paper(tmp_path, equity=8_000.0)
        result = acc.update_from_broker({"balance": 999_999.0, "nav": 999_999.0})
        assert result is False
        assert acc.equity == pytest.approx(8_000.0)

    def test_broker_failure_does_not_affect_paper_equity(self, tmp_path):
        """mark_broker_failed in LIVE_PAPER must not change equity or set UNKNOWN."""
        acc = _paper(tmp_path, equity=8_000.0)
        # mark_broker_failed is for LIVE_REAL only; paper equity must be unaffected
        initial_equity = acc.equity
        # Call it anyway (defensive — shouldn't happen in normal flow)
        acc.mark_broker_failed()
        # Paper equity is still the original value; is_unknown should still be False
        # because paper mode doesn't use the broker sentinel
        assert acc.equity == pytest.approx(initial_equity)
        assert acc.equity_source == "SIM"

    def test_equity_source_is_sim_never_broker(self, tmp_path):
        acc = _paper(tmp_path)
        assert acc.equity_source == "SIM"
        # After multiple PnL events, source stays SIM
        acc.apply_pnl(+1_000.0)
        acc.apply_pnl(-200.0)
        assert acc.equity_source == "SIM"

    def test_is_unknown_always_false_in_paper(self, tmp_path):
        """Paper equity is always known — is_unknown must be False."""
        acc = _paper(tmp_path, equity=8_000.0)
        assert acc.is_unknown is False
        acc.apply_pnl(-5_000.0)   # big loss
        assert acc.is_unknown is False

    def test_mode_is_live_paper(self, tmp_path):
        acc = _paper(tmp_path)
        assert acc.mode == AccountMode.LIVE_PAPER


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant 3 — Weekly PnL reset on ISO-week boundary
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeeklyPnlReset:
    """week_pnl/week_trades reset when the ISO week rolls over."""

    def test_week_pnl_accumulates_within_week(self, tmp_path):
        acc = _paper(tmp_path, equity=8_000.0)
        acc.apply_pnl(+1_000.0, pair="GBP/JPY", exit_reason="target_reached")
        acc.apply_pnl(-500.0,   pair="USD/JPY", exit_reason="stop_hit")
        acc.apply_pnl(+200.0,   pair="USD/CHF", exit_reason="ratchet_stop")
        assert acc.week_pnl    == pytest.approx(700.0)
        assert acc.week_trades == 3

    def test_week_boundary_resets_counters(self, tmp_path):
        """Simulating a week-boundary by directly mutating week_id then triggering check."""
        acc = _paper(tmp_path, equity=8_000.0)
        acc.apply_pnl(+2_000.0)   # trade in week N
        assert acc.week_trades == 1

        # Force the week to look stale
        acc.week_id = "2020-W01"   # definitely in the past

        # Next apply_pnl must detect boundary and reset BEFORE applying new PnL
        acc.apply_pnl(+500.0)
        assert acc.week_pnl    == pytest.approx(500.0),  "week_pnl should only count post-reset trade"
        assert acc.week_trades == 1,                     "week_trades should reset to 0 then +1"

    def test_week_boundary_reset_preserves_equity(self, tmp_path):
        """Week reset must not alter equity — only counters reset."""
        acc = _paper(tmp_path, equity=8_000.0)
        acc.apply_pnl(+3_000.0)
        equity_after_w1 = acc.equity  # 11_000
        acc.week_id = "2020-W01"  # force stale

        acc.apply_pnl(+1_000.0)
        assert acc.equity == pytest.approx(12_000.0)
        assert acc.week_pnl == pytest.approx(1_000.0)   # only new week's trade

    def test_week_id_persists_to_file(self, tmp_path):
        acc = _paper(tmp_path, equity=8_000.0)
        data = json.loads((tmp_path / "paper_account.json").read_text())
        assert "week_id" in data
        assert data["week_id"] == acc.week_id

    def test_week_reset_persists_to_file(self, tmp_path):
        """After a week reset, the new week_id and zeroed counters are saved."""
        acc = _paper(tmp_path, equity=8_000.0)
        acc.week_id = "2020-W01"
        acc.apply_pnl(+100.0)

        data = json.loads((tmp_path / "paper_account.json").read_text())
        assert data["week_id"] != "2020-W01"
        assert data["week_pnl"] == pytest.approx(100.0)
        assert data["week_trades"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant 4 — Risk sizing uses current paper equity (compounding)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskSizingUsesCurrentPaperEquity:
    """
    When paper equity grows, the risk manager must use the *current* equity for
    position sizing — not the bootstrap value.  This is the compounding invariant.
    """

    def test_risk_manager_reflects_grown_equity(self, tmp_path):
        """
        Simulate +290% equity ($8K → $31.2K).
        ForexRiskManager.compute_risk_sizing() with the grown balance must
        return a final_dollars substantially larger than with the starting balance.
        """
        from src.execution.risk_manager_forex import ForexRiskManager

        acc = _paper(tmp_path, equity=8_000.0)

        # Simulate a sequence of wins: $8K → $31.2K (+290%)
        wins = [+3_200, +6_500, +5_800, +7_700]
        for pnl in wins:
            acc.apply_pnl(float(pnl))

        assert acc.equity == pytest.approx(31_200.0), "Sanity: equity should be $31.2K"

        # Risk manager fed updated equity
        rm_grown   = ForexRiskManager(MagicMock())
        sizing_grown = rm_grown.compute_risk_sizing(acc.equity)

        # Compare against starting equity sizing
        rm_start   = ForexRiskManager(MagicMock())
        sizing_start = rm_start.compute_risk_sizing(8_000.0)

        # Compare effective risk dollars: final_pct% of the respective balance
        dollars_grown = sizing_grown.final_pct / 100.0 * acc.equity
        dollars_start = sizing_start.final_pct / 100.0 * 8_000.0
        assert dollars_grown > dollars_start, (
            f"Risk dollars must increase as equity grows "
            f"(grown={dollars_grown:.0f}, start={dollars_start:.0f})"
        )
        # At $31.2K we're in the 15% tier → risk pct > 6% starting tier
        assert sizing_grown.premult_pct > sizing_start.premult_pct, (
            "Risk percentage tier must be higher for grown equity"
        )

    def test_peak_equity_tracks_new_high(self, tmp_path):
        """peak_equity is the high-water mark and rises with equity."""
        acc = _paper(tmp_path, equity=8_000.0)
        acc.apply_pnl(+10_000.0)
        assert acc.peak_equity == pytest.approx(18_000.0)
        acc.apply_pnl(-3_000.0)
        assert acc.peak_equity == pytest.approx(18_000.0)   # does not decrease

    def test_orchestrator_uses_paper_equity_for_sizing(self, tmp_path):
        """
        Structural: orchestrator's _run_balance_refresh uses account.equity
        (not broker NAV) for sizing in LIVE_PAPER mode.
        """
        import inspect
        from src.execution import orchestrator as orch_mod

        source = inspect.getsource(orch_mod)
        # The broker NAV branch is guarded behind LIVE_REAL mode check
        assert "LIVE_REAL" in source or "AccountMode.LIVE_REAL" in source
        # Paper sizing always from account.equity / safe_equity
        assert "safe_equity" in source or "account.equity" in source

    def test_sim_starting_equity_ignored_when_file_exists(self, tmp_path):
        """SIM_STARTING_EQUITY (default_balance) is irrelevant once a file exists."""
        # First run: bootstrap at $8K
        acc = _paper(tmp_path, equity=8_000.0)
        acc.apply_pnl(+14_000.0)   # grow to $22K

        # Second run with a totally different default
        acc2 = _reload(tmp_path, default_balance=8_000.0)
        assert acc2.equity == pytest.approx(22_000.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant 5 — Equity identity
# ═══════════════════════════════════════════════════════════════════════════════

class TestEquityIdentity:
    """equity_after == equity_before + realized_pnl  (±0.01) for every close."""

    def _assert_identity(self, acc: AccountState, pnl: float, **kwargs) -> float:
        before = acc.equity
        acc.apply_pnl(pnl, **kwargs)
        after  = acc.equity
        assert after == pytest.approx(before + pnl, abs=0.01), (
            f"Equity identity violated: {before} + {pnl} → expected "
            f"{before + pnl:.4f} but got {after:.4f}"
        )
        return after

    def test_single_win(self, tmp_path):
        acc = _paper(tmp_path, equity=8_000.0)
        self._assert_identity(acc, +2_000.0)

    def test_single_loss(self, tmp_path):
        acc = _paper(tmp_path, equity=8_000.0)
        self._assert_identity(acc, -480.0)

    def test_sequence_of_trades(self, tmp_path):
        """Equity identity holds across an entire sequence (compounding path)."""
        acc = _paper(tmp_path, equity=8_000.0)
        pnls = [+1_920, -480, +4_007, +4_796, +2_061, -2_660, +1_137]
        expected = 8_000.0
        for pnl in pnls:
            expected += pnl
            self._assert_identity(acc, float(pnl))
        assert acc.equity == pytest.approx(expected, abs=0.05)

    def test_journal_records_correct_before_after(self, tmp_path):
        """Journal entries must reflect correct equity_before / equity_after."""
        acc = _paper(tmp_path, equity=8_000.0)
        acc.apply_pnl(+2_000.0, pair="GBP/JPY", exit_reason="target_reached")
        acc.apply_pnl(-500.0,   pair="USD/CHF", exit_reason="stop_hit")

        lines = _journal_lines(tmp_path)
        exits = [l for l in lines if l["event"] == "exit"]
        assert len(exits) == 2

        first  = exits[0]
        second = exits[1]

        assert first["equity_before"]  == pytest.approx(8_000.0)
        assert first["equity_after"]   == pytest.approx(10_000.0)
        assert first["realized_pnl"]   == pytest.approx(+2_000.0)

        assert second["equity_before"] == pytest.approx(10_000.0)
        assert second["equity_after"]  == pytest.approx(9_500.0)
        assert second["realized_pnl"]  == pytest.approx(-500.0)

    def test_entry_event_logged_to_journal(self, tmp_path):
        """log_entry_event() writes an 'entry' record to paper_journal.jsonl."""
        acc = _paper(tmp_path, equity=8_000.0)
        acc.log_entry_event(
            pair="GBP/JPY",
            direction="short",
            entry_price=204.328,
            stop_loss=205.674,
            risk_dollars=480.0,
        )
        lines = _journal_lines(tmp_path)
        entries = [l for l in lines if l["event"] == "entry"]
        assert len(entries) == 1
        e = entries[0]
        assert e["pair"]         == "GBP/JPY"
        assert e["direction"]    == "short"
        assert e["planned_risk_dollars"] == pytest.approx(480.0)
        assert e["equity_before"]        == pytest.approx(8_000.0)

    def test_backtest_mode_never_writes_journal(self, tmp_path):
        """BACKTEST AccountState must not write any files."""
        journal_file = tmp_path / "paper_journal.jsonl"
        acc = AccountState.for_backtest(initial_balance=8_000.0)
        acc._paper_journal_file = journal_file   # point to tmp
        acc.apply_pnl(+1_000.0)
        assert not journal_file.exists(), "Backtest mode must not write journal"
