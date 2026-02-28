"""
tests/test_account_mode_config.py
==================================
Tests for ACCOUNT_MODE + SIM_STARTING_EQUITY configuration.

Invariants:
  1. LIVE_PAPER seeds equity from SIM_STARTING_EQUITY on first run.
  2. LIVE_PAPER restarts reload persisted equity (SIM_STARTING_EQUITY is NOT re-applied).
  3. LIVE_REAL ignores SIM_STARTING_EQUITY; equity comes from broker only.
  4. OandaClient.for_paper_mode() uses DEMO credentials and normalizes OANDA_DEMO_ENV.
  5. strategy_config exports ACCOUNT_MODE and SIM_STARTING_EQUITY from env.
  6. Orchestrator mode-derivation logic (config vs dry_run flag priority).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.execution.account_state import AccountMode, AccountState


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SIM_STARTING_EQUITY seeds paper equity on first run
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimStartingEquity:

    def test_first_run_uses_sim_starting_equity(self, tmp_path):
        """
        When no paper_account.json exists, AccountState must be seeded from
        the provided default_balance (callers pass SIM_STARTING_EQUITY here).
        """
        paper_file = tmp_path / "paper_account.json"
        assert not paper_file.exists()

        acc = AccountState.for_live_paper(5_000.0, paper_file=paper_file)
        assert acc.equity     == 5_000.0
        assert acc.peak_equity == 5_000.0
        assert paper_file.exists(), "paper_account.json must be created on first run"

        data = json.loads(paper_file.read_text())
        assert data["equity"] == 5_000.0

    def test_restart_uses_persisted_equity_not_sim_starting(self, tmp_path):
        """
        On restart, persisted equity must be loaded — SIM_STARTING_EQUITY is
        only used when there is no existing file.  This is the core persistence
        invariant: paper PnL is never silently reset.
        """
        paper_file = tmp_path / "paper_account.json"

        # Session 1: start at 5_000, earn 2_000
        acc1 = AccountState.for_live_paper(5_000.0, paper_file=paper_file)
        acc1.apply_pnl(+2_000.0)
        assert acc1.equity == 7_000.0

        # Session 2: restart with SAME SIM_STARTING_EQUITY=5_000 — must reload 7_000
        acc2 = AccountState.for_live_paper(5_000.0, paper_file=paper_file)
        assert acc2.equity == 7_000.0, (
            "Restart must load persisted equity (7_000), not SIM_STARTING_EQUITY (5_000)"
        )

    def test_different_sim_starting_equity_values(self, tmp_path):
        """SIM_STARTING_EQUITY can be any positive float."""
        for balance in [1_000.0, 25_000.0, 100_000.0]:
            pf = tmp_path / f"paper_{balance:.0f}.json"
            acc = AccountState.for_live_paper(balance, paper_file=pf)
            assert acc.equity == balance, f"Expected {balance}, got {acc.equity}"

    def test_strategy_config_exports_account_mode(self):
        """strategy_config must export ACCOUNT_MODE as a string."""
        from src.strategy.forex.strategy_config import ACCOUNT_MODE
        assert isinstance(ACCOUNT_MODE, str)
        assert ACCOUNT_MODE in ("LIVE_PAPER", "LIVE_REAL", "BACKTEST"), (
            f"ACCOUNT_MODE must be LIVE_PAPER | LIVE_REAL | BACKTEST, got {ACCOUNT_MODE!r}"
        )

    def test_strategy_config_exports_sim_starting_equity(self):
        """strategy_config must export SIM_STARTING_EQUITY as a positive float."""
        from src.strategy.forex.strategy_config import SIM_STARTING_EQUITY
        assert isinstance(SIM_STARTING_EQUITY, float)
        assert SIM_STARTING_EQUITY > 0, "SIM_STARTING_EQUITY must be positive"

    def test_env_override_account_mode(self, monkeypatch):
        """ACCOUNT_MODE env var must override the default."""
        monkeypatch.setenv("ACCOUNT_MODE", "LIVE_REAL")
        # Reload the module to pick up the env change
        import importlib
        import src.strategy.forex.strategy_config as sc
        importlib.reload(sc)
        assert sc.ACCOUNT_MODE == "LIVE_REAL"
        # Restore
        monkeypatch.setenv("ACCOUNT_MODE", "LIVE_PAPER")
        importlib.reload(sc)

    def test_env_override_sim_starting_equity(self, monkeypatch):
        """SIM_STARTING_EQUITY env var must be read as float."""
        monkeypatch.setenv("SIM_STARTING_EQUITY", "12500.0")
        import importlib
        import src.strategy.forex.strategy_config as sc
        importlib.reload(sc)
        assert sc.SIM_STARTING_EQUITY == 12_500.0
        # Restore
        monkeypatch.setenv("SIM_STARTING_EQUITY", "8000.0")
        importlib.reload(sc)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  LIVE_REAL ignores SIM_STARTING_EQUITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiveRealIgnoresSimEquity:

    def test_live_real_equity_from_broker(self):
        """LIVE_REAL uses broker balance, not SIM_STARTING_EQUITY."""
        broker_bal = 4_321.0
        acc = AccountState.for_live_real({"balance": broker_bal})
        assert acc.equity      == broker_bal
        assert acc.equity_source == "BROKER"
        assert acc.mode        == AccountMode.LIVE_REAL

    def test_live_real_equity_source_is_broker(self):
        acc = AccountState.for_live_real({"balance": 5_000.0})
        assert acc.equity_source == "BROKER"
        assert acc.equity_source != "SIM"

    def test_live_real_broker_fail_sets_unknown_not_sim(self):
        """
        When broker fetch fails in LIVE_REAL mode, equity must be UNKNOWN.
        It must NEVER fall back to SIM_STARTING_EQUITY or any paper constant.
        """
        acc = AccountState.for_live_real(None)  # broker unreachable on startup
        assert acc.equity         is None,        "equity must be None (UNKNOWN)"
        assert acc.equity_source  == "UNKNOWN"
        assert acc.equity_display == "UNKNOWN"
        assert acc.equity_display != "$0.00"
        assert acc.equity_display != "$8,000.00"  # must not use SIM value

    def test_live_real_mark_failed_then_recover(self):
        """mark_broker_failed → unknown; update_from_broker → known again."""
        acc = AccountState.for_live_real({"balance": 4_000.0})
        acc.mark_broker_failed()
        assert acc.is_unknown is True

        acc.update_from_broker({"balance": 4_050.0})  # broker comes back
        assert acc.is_unknown is False
        assert acc.equity == 4_050.0
        assert acc.equity_source == "BROKER"

    def test_live_real_update_from_broker_ignores_sim(self):
        """
        update_from_broker must use the broker value, regardless of what
        SIM_STARTING_EQUITY is set to.
        """
        from src.strategy.forex.strategy_config import SIM_STARTING_EQUITY
        broker_bal = SIM_STARTING_EQUITY + 1_234.0   # deliberately different
        acc = AccountState.for_live_real({"balance": broker_bal})
        assert acc.equity == broker_bal
        assert acc.equity != SIM_STARTING_EQUITY


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  OandaClient.for_paper_mode() uses demo credentials
# ═══════════════════════════════════════════════════════════════════════════════

class TestOandaClientPaperMode:

    def _patch_env(self, monkeypatch, api_key="demo_key", account_id="demo_acct", env="LIVE_PAPER"):
        monkeypatch.setenv("OANDA_DEMO_API_KEY",    api_key)
        monkeypatch.setenv("OANDA_DEMO_ACCOUNT_ID", account_id)
        monkeypatch.setenv("OANDA_DEMO_ENV",        env)

    def test_for_paper_mode_uses_demo_api_key(self, monkeypatch):
        self._patch_env(monkeypatch)
        from src.exchange.oanda_client import OandaClient
        client = OandaClient.for_paper_mode()
        assert client.api_key    == "demo_key"
        assert client.account_id == "demo_acct"

    def test_for_paper_mode_normalizes_live_paper_env(self, monkeypatch):
        """OANDA_DEMO_ENV='LIVE_PAPER' must map to 'practice' (OANDA's practice URL)."""
        self._patch_env(monkeypatch, env="LIVE_PAPER")
        from src.exchange.oanda_client import OandaClient, PRACTICE_BASE
        client = OandaClient.for_paper_mode()
        assert client.env  == "practice"
        assert client.base == PRACTICE_BASE

    def test_for_paper_mode_normalizes_practice_env(self, monkeypatch):
        self._patch_env(monkeypatch, env="practice")
        from src.exchange.oanda_client import OandaClient, PRACTICE_BASE
        client = OandaClient.for_paper_mode()
        assert client.env  == "practice"
        assert client.base == PRACTICE_BASE

    def test_for_paper_mode_normalizes_paper_alias(self, monkeypatch):
        """'paper' is an alias for 'practice'."""
        self._patch_env(monkeypatch, env="paper")
        from src.exchange.oanda_client import OandaClient, PRACTICE_BASE
        client = OandaClient.for_paper_mode()
        assert client.base == PRACTICE_BASE

    def test_for_paper_mode_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OANDA_DEMO_API_KEY",    raising=False)
        monkeypatch.setenv("OANDA_DEMO_ACCOUNT_ID", "demo_acct")
        from src.exchange.oanda_client import OandaClient
        with pytest.raises(ValueError, match="OANDA_DEMO_API_KEY"):
            OandaClient.for_paper_mode()

    def test_for_paper_mode_missing_account_id_raises(self, monkeypatch):
        monkeypatch.setenv("OANDA_DEMO_API_KEY",       "demo_key")
        monkeypatch.delenv("OANDA_DEMO_ACCOUNT_ID",    raising=False)
        from src.exchange.oanda_client import OandaClient
        with pytest.raises(ValueError, match="OANDA_DEMO_ACCOUNT_ID"):
            OandaClient.for_paper_mode()

    def test_for_paper_mode_does_not_use_real_credentials(self, monkeypatch):
        """Demo client must NOT inherit OANDA_API_KEY or OANDA_ACCOUNT_ID."""
        monkeypatch.setenv("OANDA_API_KEY",         "real_key")
        monkeypatch.setenv("OANDA_ACCOUNT_ID",      "real_acct")
        monkeypatch.setenv("OANDA_DEMO_API_KEY",    "demo_key")
        monkeypatch.setenv("OANDA_DEMO_ACCOUNT_ID", "demo_acct")
        monkeypatch.setenv("OANDA_DEMO_ENV",        "LIVE_PAPER")
        from src.exchange.oanda_client import OandaClient
        client = OandaClient.for_paper_mode()
        assert client.api_key    != "real_key",    "Demo client must not use real API key"
        assert client.account_id != "real_acct",   "Demo client must not use real account ID"
        assert client.api_key    == "demo_key"
        assert client.account_id == "demo_acct"


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Orchestrator mode-derivation priority
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorModeDerivation:
    """
    Verify the ACCOUNT_MODE config vs dry_run flag priority rules in orchestrator source.
    We test at the source level to avoid importing the full orchestrator
    (which would require a live OANDA connection).
    """

    def _src(self):
        return (Path(__file__).parents[1] / "src" / "execution" / "orchestrator.py").read_text()

    def test_account_mode_is_imported_from_config(self):
        """Orchestrator must import ACCOUNT_MODE from strategy_config."""
        assert "ACCOUNT_MODE" in self._src(), (
            "Orchestrator must import ACCOUNT_MODE from strategy_config"
        )

    def test_sim_starting_equity_is_imported(self):
        """Orchestrator must import SIM_STARTING_EQUITY from strategy_config."""
        assert "SIM_STARTING_EQUITY" in self._src(), (
            "Orchestrator must import SIM_STARTING_EQUITY from strategy_config"
        )

    def test_for_paper_mode_used_for_live_paper(self):
        """Orchestrator must call OandaClient.for_paper_mode() for LIVE_PAPER."""
        assert "OandaClient.for_paper_mode()" in self._src(), (
            "Orchestrator must use OandaClient.for_paper_mode() for LIVE_PAPER credentials"
        )

    def test_sim_starting_equity_used_for_paper_init(self):
        """Orchestrator must pass SIM_STARTING_EQUITY (not DRY_RUN_PAPER_BALANCE) to for_live_paper."""
        src = self._src()
        # for_live_paper must be called with SIM_STARTING_EQUITY
        assert "for_live_paper(SIM_STARTING_EQUITY)" in src, (
            "AccountState.for_live_paper must receive SIM_STARTING_EQUITY, not DRY_RUN_PAPER_BALANCE"
        )

    def test_dry_run_derived_from_mode(self):
        """self.dry_run must be derived from exec_mode, not from the parameter directly."""
        src = self._src()
        assert "self.dry_run = (exec_mode == AccountMode.LIVE_PAPER)" in src, (
            "dry_run must be derived from exec_mode so it's consistent with ACCOUNT_MODE config"
        )

    def test_equity_fallback_property_exists(self):
        """_equity_fallback property must exist and reference SIM_STARTING_EQUITY."""
        src = self._src()
        assert "_equity_fallback" in src, "_equity_fallback property missing from orchestrator"
        # Find the property body — search the whole file for the combination
        assert "SIM_STARTING_EQUITY" in src, "SIM_STARTING_EQUITY must appear in orchestrator"
        # Both must appear; together they confirm the property uses the config value
        prop_idx = src.index("def _equity_fallback")
        body = src[prop_idx: prop_idx + 400]
        assert "SIM_STARTING_EQUITY" in body, (
            "_equity_fallback property body must reference SIM_STARTING_EQUITY"
        )

    def test_safe_equity_uses_equity_fallback(self):
        """
        DRY_RUN_PAPER_BALANCE must only appear on import lines in orchestrator.
        All safe_equity() fallback calls must use _equity_fallback.
        """
        src = self._src()
        lines = src.splitlines()
        # Find contiguous import block (from/import lines or continuation)
        import_block_end = 0
        in_import = False
        for i, ln in enumerate(lines):
            stripped = ln.strip()
            if stripped.startswith("from ") or stripped.startswith("import "):
                in_import = True
            if in_import:
                import_block_end = i
                # Multi-line import block ends at the closing paren
                if ")" in ln and stripped != "(":
                    in_import = False

        bad_uses = [
            (i + 1, ln) for i, ln in enumerate(lines)
            if "DRY_RUN_PAPER_BALANCE" in ln and i > import_block_end
        ]
        assert bad_uses == [], (
            f"DRY_RUN_PAPER_BALANCE must not appear outside the import block "
            f"(use _equity_fallback). Found at lines: {bad_uses}"
        )
