"""
Tests for ES Futures Strategy (ORB + Session Fade) and FuturesExecutor.

All tests use mocked SchwabClient — no live connections required.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytz
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ET = pytz.timezone("America/New_York")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utc(et_dt):
    """Localise a naive ET datetime to UTC."""
    return ET.localize(et_dt).astimezone(pytz.utc)


def _candles_15m(open_et_hour=8, open_et_min=0, high=5700.0, low=5690.0):
    """Single 15-min candle at the given ET time."""
    ts = _utc(datetime(2026, 3, 12, open_et_hour, open_et_min))
    return pd.DataFrame(
        [{"open": (high + low) / 2, "high": high, "low": low, "close": high - 1, "volume": 1000}],
        index=pd.DatetimeIndex([ts], tz="UTC"),
    )


def _daily_candles(session_high=5750.0, session_low=5600.0, n=5):
    """n daily candles with given high/low."""
    rows = []
    for i in range(n):
        ts = datetime(2026, 3, i + 6, 15, 0, tzinfo=pytz.utc)
        rows.append({"open": 5680, "high": session_high, "low": session_low,
                     "close": 5695, "volume": 50000})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(
        [datetime(2026, 3, i + 6, 15, 0, tzinfo=pytz.utc) for i in range(n)], tz="UTC"
    ))


def _make_strategy():
    from src.strategy.futures.es_strategy import ESFuturesStrategy
    return ESFuturesStrategy(schwab_client=None)


# ── 1. test_orb_range_capture ─────────────────────────────────────────────────
def test_orb_range_capture():
    """update_opening_range() stores high/low from the 8:00 AM ET bar."""
    strat = _make_strategy()
    candles = _candles_15m(8, 0, high=5710.0, low=5695.0)
    result = strat.update_opening_range(candles)
    assert result is True
    assert strat.orb_set is True
    assert strat.orb_high == pytest.approx(5710.0)
    assert strat.orb_low  == pytest.approx(5695.0)


# ── 2. test_orb_breakout_long ─────────────────────────────────────────────────
def test_orb_breakout_long():
    """Price above orb_high + buffer during 8:15–11 AM triggers LONG."""
    from src.strategy.futures import es_config as cfg
    strat = _make_strategy()
    candles = _candles_15m(8, 0, high=5700.0, low=5690.0)
    strat.update_opening_range(candles)

    now = _utc(datetime(2026, 3, 12, 9, 0))     # 9 AM ET — valid window
    price = strat.orb_high + cfg.ORB_BREAKOUT_BUFFER_PTS + 0.1

    setup = strat.check_orb_breakout(price, now)
    assert setup is not None
    assert setup.direction    == "long"
    assert setup.strategy_type == "ORB_BREAKOUT"
    assert setup.stop_price   == pytest.approx(price - cfg.ORB_STOP_POINTS)
    assert setup.target_price == pytest.approx(price + cfg.ORB_STOP_POINTS * cfg.ORB_TARGET_MULTIPLIER)


# ── 3. test_orb_breakout_short ────────────────────────────────────────────────
def test_orb_breakout_short():
    """Price below orb_low - buffer during valid window triggers SHORT."""
    from src.strategy.futures import es_config as cfg
    strat = _make_strategy()
    candles = _candles_15m(8, 0, high=5700.0, low=5690.0)
    strat.update_opening_range(candles)

    now   = _utc(datetime(2026, 3, 12, 9, 30))
    price = strat.orb_low - cfg.ORB_BREAKOUT_BUFFER_PTS - 0.1

    setup = strat.check_orb_breakout(price, now)
    assert setup is not None
    assert setup.direction    == "short"
    assert setup.strategy_type == "ORB_BREAKOUT"
    assert setup.stop_price   == pytest.approx(price + cfg.ORB_STOP_POINTS)
    assert setup.target_price == pytest.approx(price - cfg.ORB_STOP_POINTS * cfg.ORB_TARGET_MULTIPLIER)


# ── 4. test_orb_range_too_wide ────────────────────────────────────────────────
def test_orb_range_too_wide():
    """update_opening_range() returns False when range exceeds ORB_MAX_RANGE_PTS."""
    from src.strategy.futures import es_config as cfg
    strat = _make_strategy()
    wide  = cfg.ORB_MAX_RANGE_PTS + 1
    candles = _candles_15m(8, 0, high=5700.0 + wide, low=5700.0)
    result = strat.update_opening_range(candles)
    assert result is False
    assert strat.orb_set  is True    # marked as "set" so we stop retrying
    assert strat.orb_high is None    # not stored


# ── 5. test_orb_range_too_narrow ─────────────────────────────────────────────
def test_orb_range_too_narrow():
    """update_opening_range() returns False when range is below ORB_MIN_RANGE_PTS."""
    from src.strategy.futures import es_config as cfg
    strat = _make_strategy()
    narrow = cfg.ORB_MIN_RANGE_PTS - 0.5
    candles = _candles_15m(8, 0, high=5700.0 + narrow, low=5700.0)
    result = strat.update_opening_range(candles)
    assert result is False
    assert strat.orb_high is None


# ── 6. test_session_fade_at_high ──────────────────────────────────────────────
def test_session_fade_at_high():
    """Price at 5-day session high during 10AM–3PM triggers SESSION_FADE SHORT."""
    from src.strategy.futures import es_config as cfg
    strat   = _make_strategy()
    s_high  = 5750.0
    daily   = _daily_candles(session_high=s_high, session_low=5600.0)
    now     = _utc(datetime(2026, 3, 12, 11, 0))   # 11 AM ET
    price   = s_high - cfg.FADE_ENTRY_BUFFER_PTS + 0.1   # within buffer

    setup = strat.check_session_fade(price, daily, now)
    assert setup is not None
    assert setup.direction     == "short"
    assert setup.strategy_type == "SESSION_FADE"
    assert setup.stop_price    == pytest.approx(price + cfg.FADE_STOP_POINTS)
    assert setup.target_price  == pytest.approx(price - cfg.FADE_STOP_POINTS * cfg.FADE_TARGET_MULTIPLIER)


# ── 7. test_session_fade_at_low ───────────────────────────────────────────────
def test_session_fade_at_low():
    """Price at 5-day session low triggers SESSION_FADE LONG."""
    from src.strategy.futures import es_config as cfg
    strat  = _make_strategy()
    s_low  = 5600.0
    daily  = _daily_candles(session_high=5750.0, session_low=s_low)
    now    = _utc(datetime(2026, 3, 12, 14, 0))   # 2 PM ET
    price  = s_low + cfg.FADE_ENTRY_BUFFER_PTS - 0.1

    setup = strat.check_session_fade(price, daily, now)
    assert setup is not None
    assert setup.direction     == "long"
    assert setup.strategy_type == "SESSION_FADE"
    assert setup.stop_price    == pytest.approx(price - cfg.FADE_STOP_POINTS)
    assert setup.target_price  == pytest.approx(price + cfg.FADE_STOP_POINTS * cfg.FADE_TARGET_MULTIPLIER)


# ── 8. test_outside_rth_no_signal ─────────────────────────────────────────────
def test_outside_rth_no_signal():
    """Fade check returns None outside the 10 AM–3 PM ET window."""
    strat = _make_strategy()
    daily = _daily_candles(session_high=5750.0, session_low=5600.0)

    # 8:30 AM — before window
    now_early = _utc(datetime(2026, 3, 12, 8, 30))
    assert strat.check_session_fade(5750.0, daily, now_early) is None

    # 3:30 PM — after window
    strat.fade_fired = False
    now_late  = _utc(datetime(2026, 3, 12, 15, 30))
    assert strat.check_session_fade(5750.0, daily, now_late) is None

    # ORB also blocked before 8:15 AM
    strat.orb_set  = True
    strat.orb_high = 5705.0
    strat.orb_low  = 5695.0
    now_pre_orb = _utc(datetime(2026, 3, 12, 8, 0))   # 8:00 AM — before 8:15 cutoff
    assert strat.check_orb_breakout(9999.0, now_pre_orb) is None


# ── 9. test_dry_run_no_order ──────────────────────────────────────────────────
def test_dry_run_no_order(tmp_path):
    """FuturesExecutor with dry_run=True logs but does not call place_futures_order."""
    from src.strategy.futures.es_strategy import TradeSetup
    from src.execution.futures_executor import FuturesExecutor

    mock_client   = MagicMock()
    mock_notifier = MagicMock()

    executor = FuturesExecutor(
        schwab_client = mock_client,
        notifier      = mock_notifier,
        dry_run       = True,
        journal_path  = tmp_path / "futures_journal.jsonl",
    )

    setup = TradeSetup(
        strategy_type = "ORB_BREAKOUT",
        direction     = "long",
        entry_price   = 5700.0,
        stop_price    = 5696.0,
        target_price  = 5708.0,
        risk_points   = 4.0,
        reward_points = 8.0,
        reason        = "test",
    )

    result = executor.execute(setup, account_equity=10_000.0)

    # place_futures_order must be called with dry_run=True
    mock_client.place_futures_order.assert_called_once()
    call_kwargs = mock_client.place_futures_order.call_args[1]
    assert call_kwargs["dry_run"] is True
    assert result["status"] == "executed"
    assert result["dry_run"] is True


# ── 10. test_daily_loss_limit ─────────────────────────────────────────────────
def test_daily_loss_limit(tmp_path):
    """FuturesExecutor blocks execution when daily_pnl hits MAX_DAILY_LOSS_DOLLARS."""
    from src.strategy.futures.es_strategy import TradeSetup
    from src.execution.futures_executor import FuturesExecutor
    from src.strategy.futures import es_config as cfg

    mock_client   = MagicMock()
    mock_notifier = MagicMock()

    executor = FuturesExecutor(
        schwab_client = mock_client,
        notifier      = mock_notifier,
        dry_run       = True,
        journal_path  = tmp_path / "futures_journal.jsonl",
    )

    setup = TradeSetup(
        strategy_type = "SESSION_FADE",
        direction     = "short",
        entry_price   = 5750.0,
        stop_price    = 5755.0,
        target_price  = 5737.5,
        risk_points   = 5.0,
        reward_points = 12.5,
        reason        = "fade short at session high",
    )

    result = executor.execute(
        setup          = setup,
        account_equity = 10_000.0,
        daily_pnl      = -cfg.MAX_DAILY_LOSS_DOLLARS,   # hit the limit exactly
    )

    assert result["status"] == "blocked"
    mock_client.place_futures_order.assert_not_called()


# ── 11. test_rr_ratio ─────────────────────────────────────────────────────────
def test_rr_ratio():
    """TradeSetup.rr_ratio == reward / risk."""
    from src.strategy.futures.es_strategy import TradeSetup
    setup = TradeSetup(
        strategy_type="ORB_BREAKOUT", direction="long",
        entry_price=5700.0, stop_price=5696.0, target_price=5708.0,
        risk_points=4.0, reward_points=8.0, reason="test",
    )
    assert setup.rr_ratio == pytest.approx(2.0)


# ── 12. test_orb_fires_once_per_session ──────────────────────────────────────
def test_orb_fires_once_per_session():
    """orb_fired prevents a second ORB signal in the same session."""
    from src.strategy.futures import es_config as cfg
    strat = _make_strategy()
    strat.orb_set   = True
    strat.orb_high  = 5700.0
    strat.orb_low   = 5690.0

    now   = _utc(datetime(2026, 3, 12, 9, 0))
    price = 5701.5   # above high + buffer

    setup1 = strat.check_orb_breakout(price, now)
    assert setup1 is not None
    assert strat.orb_fired is True

    setup2 = strat.check_orb_breakout(price + 5, now)
    assert setup2 is None   # already fired


# ── 13. test_max_contracts_blocked ────────────────────────────────────────────
def test_max_contracts_blocked(tmp_path):
    """FuturesExecutor blocks when open_contracts >= MAX_CONCURRENT_CONTRACTS."""
    from src.strategy.futures.es_strategy import TradeSetup
    from src.execution.futures_executor import FuturesExecutor
    from src.strategy.futures import es_config as cfg

    executor = FuturesExecutor(
        schwab_client = MagicMock(),
        notifier      = MagicMock(),
        dry_run       = True,
        journal_path  = tmp_path / "fj.jsonl",
    )
    setup = TradeSetup("ORB_BREAKOUT","long",5700,5696,5708,4,8,"test")
    result = executor.execute(
        setup,
        account_equity = 10_000.0,
        open_contracts = cfg.MAX_CONCURRENT_CONTRACTS,   # at cap
    )
    assert result["status"] == "blocked"
