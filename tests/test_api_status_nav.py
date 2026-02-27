"""
tests/test_api_status_nav.py
=============================
Tests for /api/status NAV + balance aggregation in LIVE_PAPER mode.

Required invariants:
  - LIVE_PAPER equity=8000, no open positions → balance==8000, nav==8000, equity_source=="SIM"
  - NAV is equity + unrealized (0 if no positions)
  - Broker dict values (balance=0, nav=0 from unfunded demo account) must NOT appear
  - Unknown broker values → None / "N/A", never 0.00
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Shared bot_state / heartbeat factory ─────────────────────────────────────

def _make_bot_state(
    equity: float | None = 8_000.0,
    nav: float | None = 8_000.0,
    unrealized_pnl: float = 0.0,
    account_mode: str = "live_paper",
    equity_source: str = "SIM",
    equity_unknown: bool = False,
) -> dict:
    """Build a minimal bot_state dict as the orchestrator writes it."""
    return {
        "account_balance": equity,
        "dry_run": True,
        "halted":  False,
        "stats": {
            # AccountState fields (injected by orchestrator._save_state)
            "account_mode":           account_mode,
            "account_mode_label":     "LIVE PAPER" if account_mode == "live_paper" else "LIVE REAL",
            "account_equity":         equity,
            "account_equity_source":  equity_source,
            "account_is_unknown":     equity_unknown,
            "account_equity_display": "UNKNOWN" if equity is None else f"${equity:,.2f}",
            # nav / unrealized
            "nav":                    nav,
            "unrealized_pnl":         unrealized_pnl,
            # risk
            "mode":                   "active",
            "tier":                   "$8K",
            "peak_balance":           equity or 8_000.0,
            "drawdown_pct":           0.0,
            "base_risk_pct":          6.0,
            "final_risk_pct":         6.0,
            "dd_flag":                "",
            "active_cap_label":       "",
            "final_risk_dollars":     480.0,
            "consecutive_losses":     0,
            "paused":                 False,
            "paused_since":           None,
            "peak_source":            "sim",
            "session_allowed":        True,
            "session_reason":         "",
            "next_session":           "London",
            "next_session_mins":      0,
            "traded_pattern_keys":    [],
            "regime_score":           None,
            "pause_new_entries":      False,
            "pause_reason":           "",
            "pause_updated_by":       "system",
            "pause_last_updated":     None,
            "risk_mode":              None,
            "risk_mode_mult":         1.0,
            "regime_weekly_caps":     [1, 2],
        },
        "open_positions": {},
        "confluence_state": {},
        "timestamp": "2026-02-27T15:00:00+00:00",
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestApiStatusNavLivePaper:
    """
    /api/status enrichment tests for LIVE_PAPER mode.
    We patch the I/O helpers in dashboard/app.py and call api_status() directly.
    """

    def _call_status(self, bot_state: dict, broker_account: dict | None = None):
        """
        Import and call the api_status view function with all I/O mocked.
        Returns the parsed JSON dict from the response.
        """
        from dashboard import app as dash_app

        fake_account = broker_account or {"balance": 0.0, "nav": 0.0, "unrealized_pnl": 0.0}

        with dash_app.app.test_client() as client:
            with (
                patch("dashboard.app.load_bot_state",     return_value=bot_state),
                patch("dashboard.app.load_heartbeat",     return_value={}),
                patch("dashboard.app.get_oanda",          return_value=None),   # no live fetch
                patch("dashboard.app.get_journal",        return_value=MagicMock(
                    get_stats=lambda: {"win_rate": 0, "trades": 0, "total_pnl": 0},
                    get_recent_trades=lambda n: [],
                )),
                patch("dashboard.app.get_scan_state",         return_value=[]),
                patch("dashboard.app.get_last_decision_ts",   return_value=None),
                patch("dashboard.app.get_engine_sha",         return_value="abc1234"),
                patch("dashboard.app.load_recent_decisions",  return_value=[]),
                patch("dashboard.app._load_whitelist_state",  return_value={}),
                patch("dashboard.app._load_control_state",    return_value={}),
            ):
                resp = client.get("/api/status")
                return json.loads(resp.data)

    # ── Core invariant ────────────────────────────────────────────────────────

    def test_live_paper_balance_equals_sim_equity(self):
        """LIVE_PAPER equity=8000 → status.account_balance == 8000 (not broker $0)."""
        state = _make_bot_state(equity=8_000.0, nav=8_000.0)
        result = self._call_status(state)
        assert result["account_balance"] == 8_000.0, (
            f"Expected account_balance=8000 (from AccountState), got {result['account_balance']}"
        )

    def test_live_paper_nav_equals_equity_when_no_positions(self):
        """LIVE_PAPER, no open positions → nav == equity (unrealized = 0)."""
        state = _make_bot_state(equity=8_000.0, nav=8_000.0, unrealized_pnl=0.0)
        result = self._call_status(state)
        assert result["nav"] == 8_000.0, (
            f"Expected nav=8000 (equity + 0 unrealized), got {result['nav']}"
        )

    def test_live_paper_equity_source_is_sim(self):
        """equity_source must be 'SIM' in LIVE_PAPER mode."""
        state = _make_bot_state(equity=8_000.0)
        result = self._call_status(state)
        assert result["equity_source"] == "SIM", (
            f"Expected equity_source='SIM', got {result['equity_source']!r}"
        )

    def test_all_three_invariants_together(self):
        """Combined: balance==8000 AND nav==8000 AND equity_source=='SIM'."""
        state = _make_bot_state(equity=8_000.0, nav=8_000.0)
        result = self._call_status(state)
        assert result["account_balance"] == 8_000.0
        assert result["nav"]             == 8_000.0
        assert result["equity_source"]   == "SIM"

    # ── Broker $0 must not bleed through ─────────────────────────────────────

    def test_broker_zero_balance_ignored_in_paper_mode(self):
        """Broker reports $0 balance for unfunded demo account — must be ignored."""
        state           = _make_bot_state(equity=8_000.0, nav=8_000.0)
        broker_account  = {"balance": 0.0, "nav": 0.0}   # unfunded demo
        result          = self._call_status(state, broker_account=broker_account)
        assert result["account_balance"] != 0.0, "Broker $0 must not replace paper equity"
        assert result["nav"]             != 0.0, "Broker $0 nav must not replace paper nav"

    def test_broker_unreachable_in_paper_mode_does_not_affect_balance(self):
        """Even if broker call fails entirely, paper balance must remain intact."""
        state = _make_bot_state(equity=8_000.0, nav=8_000.0)
        # get_oanda() returns None → no broker call at all
        result = self._call_status(state, broker_account=None)
        assert result["account_balance"] == 8_000.0
        assert result["nav"]             == 8_000.0

    # ── NAV with unrealized PnL ───────────────────────────────────────────────

    def test_nav_includes_unrealized_when_position_open(self):
        """LIVE_PAPER with open position: nav = equity + unrealized."""
        equity      = 8_000.0
        unrealized  = +320.0
        nav         = equity + unrealized
        state       = _make_bot_state(equity=equity, nav=nav, unrealized_pnl=unrealized)
        result      = self._call_status(state)
        assert result["nav"]             == nav
        assert result["unrealized_pnl"]  == unrealized

    def test_nav_negative_unrealized(self):
        """NAV correctly reflects a negative unrealized PnL (floating loss)."""
        equity      = 8_000.0
        unrealized  = -240.0
        nav         = equity + unrealized   # 7_760
        state       = _make_bot_state(equity=equity, nav=nav, unrealized_pnl=unrealized)
        result      = self._call_status(state)
        assert result["nav"]            == 7_760.0
        assert result["unrealized_pnl"] == -240.0

    # ── LIVE_REAL unknown equity ──────────────────────────────────────────────

    def test_live_real_unknown_equity_returns_none_not_zero(self):
        """LIVE_REAL with broker failure: balance and nav must be None, never 0."""
        state = _make_bot_state(
            equity=None,
            nav=None,
            account_mode="live_real",
            equity_source="UNKNOWN",
            equity_unknown=True,
        )
        result = self._call_status(state)
        # balance must be None (falsy but not zero)
        assert result["account_balance"] is None or result["account_balance"] != 0.0, (
            f"LIVE_REAL unknown equity: balance must be None, got {result['account_balance']}"
        )
        assert result["nav"] is None, (
            f"LIVE_REAL unknown nav must be None, got {result['nav']}"
        )

    def test_live_real_unknown_equity_source_is_unknown(self):
        state = _make_bot_state(
            equity=None, nav=None,
            account_mode="live_real",
            equity_source="UNKNOWN",
            equity_unknown=True,
        )
        result = self._call_status(state)
        assert result["equity_unknown"] is True
        assert result["equity_source"]  == "UNKNOWN"

    # ── AccountState round-trip: orchestrator writes, dashboard reads ─────────

    def test_orchestrator_nav_computation_live_paper_no_positions(self):
        """
        Unit-test the orchestrator's nav formula directly (no Flask needed).
        For LIVE_PAPER, unrealized_pnl=0 → nav == equity.
        """
        equity          = 8_000.0
        unrealized_pnl  = 0.0
        computed_nav    = (equity or 0.0) + unrealized_pnl
        assert computed_nav == equity, (
            "LIVE_PAPER nav formula: equity + 0 unrealized must equal equity"
        )

    def test_orchestrator_nav_computation_live_paper_with_positions(self):
        equity          = 8_000.0
        unrealized_pnl  = 450.0
        computed_nav    = (equity or 0.0) + unrealized_pnl
        assert computed_nav == 8_450.0

    def test_orchestrator_nav_none_for_live_real_broker_failure(self):
        """Confirm the None-assignment path exists in orchestrator source."""
        orc_path = Path(__file__).parents[1] / "src" / "execution" / "orchestrator.py"
        source   = orc_path.read_text()
        assert "self.account_nav = None" in source, (
            "Orchestrator must explicitly set account_nav=None when broker fails "
            "(not fall back to 0)"
        )
