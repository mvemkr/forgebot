"""
Acceptance tests for Reconciler — continuous broker state integrity guardrail.

Four scenarios from the spec:
  1. Manual close while bot running → EXTERNAL_CLOSE detected, journal updated,
     no further trailing attempts (position cleared locally)
  2. Manual stop edit while bot running → EXTERNAL_MODIFY detected,
     bot adopts broker stop
  3. Local thinks open position but broker doesn't → EXTERNAL_CLOSE →
     entries paused + conflict recorded
  4. Broker has open position after restart → RECOVERED_POSITION logged

Design:
  - All tests use a fake OandaClient, fake Strategy, and fake Journal.
  - dry_run=True for most tests (no real state mutations).
  - dry_run=False for mutation tests to verify state is actually updated.
  - Reconciler is instantiated directly — no orchestrator dependency.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

from src.execution.reconciler import Reconciler, ReconcileResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_reconciler(
    broker_trades=None,
    broker_pending=None,
    local_positions=None,
    dry_run=False,
    account_mode="LIVE_REAL",
):
    """
    Build a Reconciler with mocked dependencies.

    broker_trades   : list of dicts returned by oanda.get_open_trades()
    broker_pending  : list of dicts returned by oanda.get_pending_orders()
    local_positions : dict {pair: pos_dict} in strategy.open_positions
    """
    broker_trades  = broker_trades  or []
    broker_pending = broker_pending or []
    local_positions = local_positions or {}

    oanda = MagicMock()
    oanda.get_open_trades.return_value   = broker_trades
    oanda.get_pending_orders.return_value = broker_pending
    oanda.get_account_summary.return_value = {
        "balance": 10_000.0,
        "nav":     10_000.0,
    }

    strategy = MagicMock()
    strategy.open_positions = dict(local_positions)  # mutable copy

    journal  = MagicMock()
    notifier = MagicMock()

    control = MagicMock()
    control.pause_new_entries = False
    control.reason            = ""

    reconciler = Reconciler(
        oanda        = oanda,
        strategy     = strategy,
        journal      = journal,
        notifier     = notifier,
        control      = control,
        account_mode = account_mode,
        dry_run      = dry_run,
    )
    return reconciler, strategy, journal, notifier, control


def _broker_trade(pair, direction="long", entry=1.25000, stop=1.24000,
                  units=10_000, trade_id="42"):
    """Build a minimal broker trade dict matching OandaClient.get_open_trades() format."""
    return {
        "id":         trade_id,
        "instrument": pair.replace("/", "_"),   # e.g. "GBP_USD"
        "direction":  direction,
        "units":      units,
        "entry":      entry,
        "stop_loss":  stop,
        "take_profit": None,
        "open_time":  "2026-02-01T02:00:00Z",
    }


def _local_pos(pair, direction="long", entry=1.25000, stop=1.24000,
               units=10_000, trade_id="42"):
    """Build a minimal local position dict."""
    return {
        "pair":           pair,
        "direction":      direction,
        "entry":          entry,
        "stop":           stop,
        "initial_risk":   abs(entry - stop),
        "units":          units,
        "oanda_trade_id": trade_id,
        "trail_locked":   False,
        "trail_max":      entry,
        "pattern_type":   "HEAD_AND_SHOULDERS",
        "confidence":     0.82,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Manual close detected (EXTERNAL_CLOSE)
# ─────────────────────────────────────────────────────────────────────────────

class TestExternalClose:
    """
    Mike manually closes GBP/USD on the OANDA platform while the bot is running.
    The bot's local state still thinks the position is open.
    """

    def test_external_close_detected(self):
        """EXTERNAL_CLOSE appears in result.actions and result.external_closes."""
        pair = "GBP/USD"
        local = {pair: _local_pos(pair)}
        # Broker has NO open trade for GBP/USD (Mike closed it manually)
        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = [],   # empty — trade gone
            local_positions = local,
            dry_run         = False,
        )
        result = rec.reconcile_light(throttle=False)

        assert not result.clean
        assert pair in result.external_closes
        assert f"EXTERNAL_CLOSE:{pair}" in result.actions

    def test_external_close_journals_event(self):
        """Journal receives a RECONCILE_EXTERNAL_CLOSE entry."""
        pair = "GBP/USD"
        local = {pair: _local_pos(pair)}
        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = [],
            local_positions = local,
            dry_run         = False,
        )
        rec.reconcile_light(throttle=False)

        # journal.log_reconcile_event must be called with event="EXTERNAL_CLOSE"
        journal.log_reconcile_event.assert_called()
        call_kwargs = journal.log_reconcile_event.call_args_list[-1]
        assert call_kwargs.kwargs.get("event") == "EXTERNAL_CLOSE" or \
               (call_kwargs.args and "EXTERNAL_CLOSE" in str(call_kwargs))

    def test_external_close_clears_local_position(self):
        """Local position is removed so trailing never fires on a ghost trade."""
        pair = "GBP/USD"
        local = {pair: _local_pos(pair)}
        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = [],
            local_positions = local,
            dry_run         = False,
        )
        rec.reconcile_light(throttle=False)

        strategy.close_position.assert_called_once()

    def test_external_close_notifies_user(self):
        """Mike receives a Telegram notification about the external close."""
        pair = "GBP/USD"
        local = {pair: _local_pos(pair)}
        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = [],
            local_positions = local,
            dry_run         = False,
        )
        rec.reconcile_light(throttle=False)

        notifier.send.assert_called()
        msg = notifier.send.call_args_list[-1][0][0]
        assert "EXTERNAL" in msg.upper() or pair in msg

    def test_external_close_pauses_entries(self):
        """Entries are paused after an unexpected external close."""
        pair = "GBP/USD"
        local = {pair: _local_pos(pair)}
        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = [],
            local_positions = local,
            dry_run         = False,
        )
        result = rec.reconcile_light(throttle=False)

        assert result.pause_triggered
        control.pause.assert_called()

    def test_pending_order_not_flagged_as_external_close(self):
        """
        If local has a position and broker has a PENDING order for same pair
        (limit order not yet filled) — do NOT flag as external close.
        """
        pair = "EUR/USD"
        local = {pair: _local_pos(pair)}
        pending = [{"instrument": "EUR_USD", "type": "LIMIT", "units": 10000}]
        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = [],          # not filled yet
            broker_pending  = pending,
            local_positions = local,
            dry_run         = False,
        )
        result = rec.reconcile_light(throttle=False)

        assert result.clean
        assert not result.external_closes


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Manual stop edit (EXTERNAL_MODIFY)
# ─────────────────────────────────────────────────────────────────────────────

class TestExternalModify:
    """
    Mike manually tightens the stop on USD/JPY from 154.000 to 155.000
    via the OANDA web interface while the bot is managing the position.
    """

    def test_stop_tightened_externally_adopted(self):
        """Broker stop tighter than local → adopt broker value, no pause."""
        pair  = "USD/JPY"
        entry = 156.000
        local_stop  = 154.000  # bot's last known stop
        broker_stop = 155.000  # Mike tightened it manually (closer to entry for long)

        local = {pair: _local_pos(pair, entry=entry, stop=local_stop)}
        broker = [_broker_trade(pair, entry=entry, stop=broker_stop)]

        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = broker,
            local_positions = local,
            dry_run         = False,
        )
        result = rec.reconcile_light(throttle=False)

        assert not result.clean
        assert pair in result.external_modifies
        # Entries should NOT be paused (stop tightened = safer)
        assert not result.pause_triggered

    def test_tightened_stop_adopted_in_local_state(self):
        """Local position dict is updated to reflect broker's tighter stop."""
        pair  = "USD/JPY"
        entry = 156.000
        local_stop  = 154.000
        broker_stop = 155.500   # tighter

        local = {pair: _local_pos(pair, entry=entry, stop=local_stop)}
        broker = [_broker_trade(pair, entry=entry, stop=broker_stop)]

        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = broker,
            local_positions = local,
            dry_run         = False,
        )
        rec.reconcile_light(throttle=False)

        # strategy.open_positions[pair]["stop"] should be broker_stop
        updated_pos = strategy.open_positions.get(pair, {})
        assert updated_pos.get("stop") == broker_stop

    def test_stop_widened_externally_triggers_pause(self):
        """
        If someone moves the stop AWAY from entry (safety violation):
        adopt broker stop AND pause entries.
        """
        pair  = "USD/JPY"
        entry = 156.000
        local_stop  = 155.000   # tighter (better)
        broker_stop = 153.000   # wider — someone messed with it

        local = {pair: _local_pos(pair, entry=entry, stop=local_stop)}
        broker = [_broker_trade(pair, entry=entry, stop=broker_stop)]

        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = broker,
            local_positions = local,
            dry_run         = False,
        )
        result = rec.reconcile_light(throttle=False)

        assert not result.clean
        assert pair in result.external_modifies
        assert result.pause_triggered
        control.pause.assert_called()

    def test_stop_within_tolerance_not_flagged(self):
        """Stop difference within 2-pip tolerance → no EXTERNAL_MODIFY event."""
        pair       = "EUR/USD"
        entry      = 1.08000
        local_stop = 1.07000
        # 1 pip difference — within 2-pip tolerance
        broker_stop = 1.07001

        local = {pair: _local_pos(pair, entry=entry, stop=local_stop)}
        broker = [_broker_trade(pair, entry=entry, stop=broker_stop)]

        rec, _, _, _, _ = _make_reconciler(
            broker_trades   = broker,
            local_positions = local,
        )
        result = rec.reconcile_light(throttle=False)

        assert result.clean
        assert not result.external_modifies

    def test_unexpected_take_profit_logged(self):
        """If broker trade has a TP set (should never happen) — log it."""
        pair = "GBP/JPY"
        local = {pair: _local_pos(pair, entry=195.000, stop=193.000)}
        bt = _broker_trade(pair, entry=195.000, stop=193.000)
        bt["take_profit"] = 198.000    # someone added a TP

        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = [bt],
            local_positions = local,
        )
        result = rec.reconcile_light(throttle=False)

        assert pair in result.external_modifies
        journal.log_reconcile_event.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — Local open position, broker empty → pause + conflict
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalOnlyPosition:
    """
    Local state thinks a position is open but OANDA has nothing.
    This is scenario 3: state inconsistency requiring manual review.
    """

    def test_conflict_banner_pause(self):
        """pause + conflict in result when local has position broker doesn't."""
        pair  = "GBP/CHF"
        local = {pair: _local_pos(pair)}
        rec, _, _, _, control = _make_reconciler(
            broker_trades   = [],
            local_positions = local,
            dry_run         = False,
        )
        result = rec.reconcile_light(throttle=False)

        assert not result.clean
        assert pair in result.external_closes
        assert result.pause_triggered
        control.pause.assert_called()

    def test_multiple_local_only_positions_all_flagged(self):
        """All local-only positions are detected in a single reconcile pass."""
        pairs = ["GBP/CHF", "USD/CAD"]
        local = {p: _local_pos(p) for p in pairs}
        rec, _, _, _, _ = _make_reconciler(
            broker_trades   = [],
            local_positions = local,
            dry_run         = False,
        )
        result = rec.reconcile_light(throttle=False)

        for p in pairs:
            assert p in result.external_closes

    def test_dry_run_does_not_clear_local(self):
        """In dry_run mode, local positions are NOT mutated."""
        pair  = "USD/CHF"
        local = {pair: _local_pos(pair)}
        rec, strategy, _, _, _ = _make_reconciler(
            broker_trades   = [],
            local_positions = local,
            dry_run         = True,    # no mutations
        )
        rec.reconcile_light(throttle=False)

        strategy.close_position.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4 — Broker has position, local doesn't → RECOVERED_POSITION
# ─────────────────────────────────────────────────────────────────────────────

class TestRecoveredPosition:
    """
    Bot restarted (or crashed) and the state file is empty, but OANDA still
    has an open trade. reconcile_full() must detect and reconstruct it.
    """

    def test_recovered_at_startup_no_pause(self):
        """
        At startup (is_startup=True), RECOVERED_POSITION is logged but
        entries are NOT paused — this is the normal restart-recovery path.
        """
        pair  = "GBP/JPY"
        broker = [_broker_trade(pair, direction="long", entry=195.000, stop=193.000)]
        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = broker,
            local_positions = {},       # empty — bot just restarted
            dry_run         = False,
        )
        result = rec.reconcile_full()

        assert pair in result.recovered
        assert f"RECOVERED_POSITION:{pair}" in result.actions
        # No pause at startup
        assert not result.pause_triggered
        control.pause.assert_not_called()

    def test_recovered_position_reconstructed_in_local(self):
        """Local strategy.open_positions gets a reconstructed entry."""
        pair   = "GBP/JPY"
        entry  = 195.000
        stop   = 193.000
        broker = [_broker_trade(pair, direction="short", entry=entry,
                                stop=stop, units=50_000, trade_id="99")]
        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = broker,
            local_positions = {},
            dry_run         = False,
        )
        rec.reconcile_full()

        assert pair in strategy.open_positions
        pos = strategy.open_positions[pair]
        assert pos["direction"]     == "short"
        assert pos["entry"]         == entry
        assert pos["stop"]          == stop
        assert pos["oanda_trade_id"] == "99"
        assert pos["source"]        == "RECONCILER_RECOVERED"

    def test_recovered_during_operation_triggers_pause(self):
        """
        If reconcile_light() (not startup) finds a broker position the bot
        doesn't know about → state drift → pause entries.
        """
        pair   = "USD/CAD"
        broker = [_broker_trade(pair, direction="long", entry=1.38, stop=1.37)]
        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = broker,
            local_positions = {},
            dry_run         = False,
        )
        result = rec.reconcile_light(throttle=False)

        assert pair in result.recovered
        assert result.pause_triggered
        control.pause.assert_called()

    def test_recovered_journals_event(self):
        """journal.log_reconcile_event called with RECOVERED_POSITION."""
        pair   = "EUR/USD"
        broker = [_broker_trade(pair, entry=1.09, stop=1.08)]
        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = broker,
            local_positions = {},
            dry_run         = False,
        )
        rec.reconcile_full()

        journal.log_reconcile_event.assert_called()
        events = [
            (c.kwargs.get("event") or (c.args[1] if len(c.args) > 1 else ""))
            for c in journal.log_reconcile_event.call_args_list
        ]
        assert any("RECOVERED" in e for e in events)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5 — Clean state (no mismatches)
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanState:
    """All positions match → result.clean == True."""

    def test_clean_when_both_sides_agree(self):
        pair   = "EUR/USD"
        entry  = 1.08000
        stop   = 1.07000
        local  = {pair: _local_pos(pair, entry=entry, stop=stop)}
        broker = [_broker_trade(pair, entry=entry, stop=stop)]

        rec, _, _, _, _ = _make_reconciler(
            broker_trades   = broker,
            local_positions = local,
        )
        result = rec.reconcile_light(throttle=False)

        assert result.clean
        assert result.actions == []

    def test_clean_when_both_empty(self):
        rec, _, _, _, _ = _make_reconciler(
            broker_trades   = [],
            local_positions = {},
        )
        result = rec.reconcile_light(throttle=False)

        assert result.clean

    def test_multiple_pairs_all_match(self):
        pairs  = ["GBP/USD", "USD/JPY"]
        local  = {p: _local_pos(p, entry=1.3 if "GBP" in p else 155.0,
                                stop=1.29 if "GBP" in p else 153.0)
                  for p in pairs}
        broker = [
            _broker_trade("GBP/USD", entry=1.3, stop=1.29, trade_id="1"),
            _broker_trade("USD/JPY", entry=155.0, stop=153.0, trade_id="2"),
        ]
        rec, _, _, _, _ = _make_reconciler(
            broker_trades   = broker,
            local_positions = local,
        )
        result = rec.reconcile_light(throttle=False)

        assert result.clean


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6 — Fail-open on API error
# ─────────────────────────────────────────────────────────────────────────────

class TestFailOpen:
    """reconcile_light: fail-closed on LIVE_REAL, fail-open on LIVE_PAPER."""

    def test_api_error_live_real_pauses_entries(self):
        """
        LIVE_REAL + API exception → fail-closed:
          - result.clean == False
          - result.pause_triggered == True
          - control.pause called with reason BROKER_SYNC_FAILED
        """
        oanda = MagicMock()
        oanda.get_open_trades.side_effect = Exception("Connection timeout")

        control = MagicMock()
        control.pause_new_entries = False

        rec = Reconciler(
            oanda        = oanda,
            strategy     = MagicMock(),
            journal      = MagicMock(),
            notifier     = MagicMock(),
            control      = control,
            account_mode = "LIVE_REAL",
            dry_run      = False,
        )
        result = rec.reconcile_light(throttle=False)

        assert not result.clean
        assert result.pause_triggered
        control.pause.assert_called_once()
        pause_reason = control.pause.call_args.kwargs.get("reason", "")
        assert "BROKER_SYNC_FAILED" in pause_reason

    def test_api_error_live_real_aborts_order_via_result(self):
        """
        Pre-order gate in orchestrator checks result.pause_triggered.
        Confirm the returned result signals abort (pause_triggered=True).
        """
        oanda = MagicMock()
        oanda.get_open_trades.side_effect = TimeoutError("read timeout")

        rec = Reconciler(
            oanda        = oanda,
            strategy     = MagicMock(),
            journal      = MagicMock(),
            notifier     = MagicMock(),
            control      = MagicMock(),
            account_mode = "LIVE_REAL",
            dry_run      = False,
        )
        result = rec.reconcile_light(throttle=False)

        # Orchestrator checks: if result.pause_triggered → return (abort order)
        assert result.pause_triggered

    def test_api_error_live_paper_stays_open(self):
        """
        LIVE_PAPER + API exception → fail-open:
          - result.clean == True
          - control.pause NOT called
          - trading continues unblocked
        """
        oanda = MagicMock()
        oanda.get_open_trades.side_effect = Exception("Connection timeout")

        control = MagicMock()

        rec = Reconciler(
            oanda        = oanda,
            strategy     = MagicMock(),
            journal      = MagicMock(),
            notifier     = MagicMock(),
            control      = control,
            account_mode = "LIVE_PAPER",
            dry_run      = False,
        )
        result = rec.reconcile_light(throttle=False)

        assert result.clean
        assert not result.pause_triggered
        control.pause.assert_not_called()

    def test_api_error_on_full_returns_clean(self):
        """reconcile_full fails open — startup must not be blocked."""
        oanda = MagicMock()
        oanda.get_open_trades.side_effect = RuntimeError("timeout")

        rec = Reconciler(
            oanda        = oanda,
            strategy     = MagicMock(),
            journal      = MagicMock(),
            notifier     = MagicMock(),
            control      = MagicMock(),
            dry_run      = True,
        )
        result = rec.reconcile_full()

        assert result.clean


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 7 — Throttle behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestThrottle:
    """reconcile_light with throttle=True skips if called too recently."""

    def test_throttle_skips_second_call(self):
        """Second call within MIN_LIGHT_INTERVAL_S → returns clean without API call."""
        pair  = "GBP/USD"
        local = {pair: _local_pos(pair)}
        rec, strategy, journal, notifier, control = _make_reconciler(
            broker_trades   = [],   # would return EXTERNAL_CLOSE if called
            local_positions = local,
            dry_run         = False,
        )
        # First call (no throttle)
        rec.reconcile_light(throttle=False)

        # Second call immediately — throttle=True → should skip broker fetch
        rec.oanda.reset_mock()
        result_throttled = rec.reconcile_light(throttle=True)

        # Must not have called OANDA again
        rec.oanda.get_open_trades.assert_not_called()
        assert result_throttled.clean   # returned early

    def test_throttle_false_always_calls_oanda(self):
        """throttle=False always fetches from broker regardless of recency."""
        pair  = "USD/JPY"
        local = {pair: _local_pos(pair)}
        rec, strategy, _, _, _ = _make_reconciler(
            broker_trades   = [],
            local_positions = local,
            dry_run         = False,
        )
        for _ in range(3):
            rec.reconcile_light(throttle=False)

        # OANDA called 3 times
        assert rec.oanda.get_open_trades.call_count == 3
