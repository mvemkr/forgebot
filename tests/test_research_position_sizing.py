"""
Tests for scripts/research_position_sizing.py

1.  Import without side effects (no _sc mutation)
2.  Variant structure (A=1%, B=3%, C=5%, D=10%)
3.  Variants are strictly increasing risk
4.  Study start < study end and spans >= 12 months
5.  _parse_ts: string, datetime, None
6.  _build_equity_curve: starts at CAPITAL, applies pnl in order, length
7.  _build_equity_curve: empty trades returns single starting point
8.  _monthly_equity: correct month-end values, carry-forward when no trades
9.  _max_drawdown: peak-to-trough tracking
10. _max_drawdown: flat curve (no drawdown)
11. _losing_streak: consecutive losses counted correctly
12. _losing_streak: no losses → (0, 0.0)
13. _losing_streak: streak resets after winner
14. _months_elapsed: correct month arithmetic
15. _months_to_milestone: returns first month equity >= threshold
16. _months_to_milestone: returns None if never reached
17. No atexit required — _sc not mutated by import
18. MILESTONES contains $27K and $100K
19. Report path correct
20. flat_risk_pct passed correctly (not patching _sc)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
UTC = timezone.utc


# ── 1. Import no side effects ─────────────────────────────────────────────────
def test_import_no_side_effects():
    import src.strategy.forex.strategy_config as _sc
    orig = _sc.MAX_TRADES_PER_WEEK_SMALL
    import scripts.research_position_sizing as m
    assert _sc.MAX_TRADES_PER_WEEK_SMALL == orig, "Import must not mutate strategy_config"
    assert hasattr(m, "VARIANTS")
    assert hasattr(m, "STUDY_START")
    assert hasattr(m, "STUDY_END")


# ── 2. Variant structure ──────────────────────────────────────────────────────
def test_variant_structure():
    import scripts.research_position_sizing as m
    assert len(m.VARIANTS) == 4
    labels = [v[0] for v in m.VARIANTS]
    assert labels == ["A", "B", "C", "D"]
    risks  = [v[1] for v in m.VARIANTS]
    assert risks[0] == pytest.approx(0.01)
    assert risks[1] == pytest.approx(0.03)
    assert risks[2] == pytest.approx(0.05)
    assert risks[3] == pytest.approx(0.10)


# ── 3. Strictly increasing risk ───────────────────────────────────────────────
def test_variants_strictly_increasing():
    import scripts.research_position_sizing as m
    risks = [v[1] for v in m.VARIANTS]
    assert risks == sorted(risks)
    assert len(set(risks)) == len(risks)


# ── 4. Study window ───────────────────────────────────────────────────────────
def test_study_window_span():
    import scripts.research_position_sizing as m
    assert m.STUDY_START < m.STUDY_END
    months = (m.STUDY_END.year - m.STUDY_START.year)*12 + (m.STUDY_END.month - m.STUDY_START.month)
    assert months >= 12, "Study must span at least 12 months"


# ── 5. _parse_ts ──────────────────────────────────────────────────────────────
def test_parse_ts_string():
    import scripts.research_position_sizing as m
    dt = m._parse_ts("2026-02-24T11:00:00+00:00")
    assert dt.year == 2026 and dt.month == 2 and dt.day == 24


def test_parse_ts_datetime():
    import scripts.research_position_sizing as m
    dt_in = datetime(2026, 2, 24, 11, tzinfo=UTC)
    dt    = m._parse_ts(dt_in)
    assert dt == dt_in


def test_parse_ts_none():
    import scripts.research_position_sizing as m
    assert m._parse_ts(None) is None


def test_parse_ts_naive_datetime():
    import scripts.research_position_sizing as m
    dt_naive = datetime(2026, 2, 24, 11)
    dt = m._parse_ts(dt_naive)
    assert dt.tzinfo is not None


# ── 6. _build_equity_curve ────────────────────────────────────────────────────
def test_build_equity_curve_basic():
    import scripts.research_position_sizing as m
    trades = [
        {"exit_ts": "2025-02-01T12:00:00+00:00", "pnl":  500.0},
        {"exit_ts": "2025-03-01T12:00:00+00:00", "pnl": -200.0},
    ]
    curve = m._build_equity_curve(trades, 8000.0)
    assert curve[0][1] == pytest.approx(8000.0)   # starting point
    assert curve[1][1] == pytest.approx(8500.0)   # after +500
    assert curve[2][1] == pytest.approx(8300.0)   # after -200


def test_build_equity_curve_sorted_by_exit():
    import scripts.research_position_sizing as m
    # Out-of-order input — must be sorted by exit_ts
    trades = [
        {"exit_ts": "2025-03-01T12:00:00+00:00", "pnl": -200.0},
        {"exit_ts": "2025-02-01T12:00:00+00:00", "pnl":  500.0},
    ]
    curve = m._build_equity_curve(trades, 8000.0)
    assert curve[1][1] == pytest.approx(8500.0)   # Feb first (+500)
    assert curve[2][1] == pytest.approx(8300.0)   # Mar second (-200)


# ── 7. _build_equity_curve empty ─────────────────────────────────────────────
def test_build_equity_curve_empty():
    import scripts.research_position_sizing as m
    curve = m._build_equity_curve([], 8000.0)
    assert len(curve) == 1
    assert curve[0][1] == pytest.approx(8000.0)


# ── 8. _monthly_equity ────────────────────────────────────────────────────────
def test_monthly_equity_carry_forward():
    import scripts.research_position_sizing as m
    # Single trade in Feb 2025, no further trades
    curve = [
        (datetime(2025, 1, 1, tzinfo=UTC), 8000.0),
        (datetime(2025, 2, 15, tzinfo=UTC), 9000.0),
    ]
    monthly = m._monthly_equity(curve)
    assert "2025-01" in monthly
    assert "2025-02" in monthly
    # Jan: no trade yet → 8000 (starting point only)
    assert monthly["2025-01"] == pytest.approx(8000.0)
    # Feb: trade closed on Feb 15 → 9000
    assert monthly["2025-02"] == pytest.approx(9000.0)
    # March and beyond should carry forward 9000
    if "2025-03" in monthly:
        assert monthly["2025-03"] == pytest.approx(9000.0)


# ── 9. _max_drawdown ──────────────────────────────────────────────────────────
def test_max_drawdown_basic():
    import scripts.research_position_sizing as m
    curve = [
        (datetime(2025, 1, 1, tzinfo=UTC), 8000.0),
        (datetime(2025, 2, 1, tzinfo=UTC), 10000.0),  # new peak
        (datetime(2025, 3, 1, tzinfo=UTC), 7000.0),   # -3000 from peak
        (datetime(2025, 4, 1, tzinfo=UTC), 9000.0),
    ]
    dd_pct, dd_usd = m._max_drawdown(curve)
    assert dd_usd == pytest.approx(3000.0)
    assert dd_pct == pytest.approx(30.0)


def test_max_drawdown_flat():
    import scripts.research_position_sizing as m
    curve = [
        (datetime(2025, 1, 1, tzinfo=UTC), 8000.0),
        (datetime(2025, 2, 1, tzinfo=UTC), 9000.0),
        (datetime(2025, 3, 1, tzinfo=UTC), 10000.0),
    ]
    dd_pct, dd_usd = m._max_drawdown(curve)
    assert dd_pct == pytest.approx(0.0)
    assert dd_usd == pytest.approx(0.0)


# ── 10. _losing_streak ────────────────────────────────────────────────────────
def test_losing_streak_basic():
    import scripts.research_position_sizing as m
    trades = [
        {"exit_ts": "2025-01-10T00:00:00+00:00", "pnl": -100.0},
        {"exit_ts": "2025-01-20T00:00:00+00:00", "pnl": -150.0},
        {"exit_ts": "2025-01-30T00:00:00+00:00", "pnl":  200.0},
        {"exit_ts": "2025-02-10T00:00:00+00:00", "pnl": -80.0},
    ]
    streak, usd = m._losing_streak(trades)
    assert streak == 2
    assert usd == pytest.approx(250.0)


def test_losing_streak_no_losses():
    import scripts.research_position_sizing as m
    trades = [
        {"exit_ts": "2025-01-10T00:00:00+00:00", "pnl": 100.0},
        {"exit_ts": "2025-01-20T00:00:00+00:00", "pnl": 200.0},
    ]
    streak, usd = m._losing_streak(trades)
    assert streak == 0
    assert usd == pytest.approx(0.0)


def test_losing_streak_resets_after_win():
    import scripts.research_position_sizing as m
    trades = [
        {"exit_ts": "2025-01-05T00:00:00+00:00", "pnl": -100.0},
        {"exit_ts": "2025-01-10T00:00:00+00:00", "pnl": -100.0},
        {"exit_ts": "2025-01-15T00:00:00+00:00", "pnl":  300.0},  # winner resets
        {"exit_ts": "2025-01-20T00:00:00+00:00", "pnl": -100.0},
        {"exit_ts": "2025-01-25T00:00:00+00:00", "pnl": -100.0},
        {"exit_ts": "2025-01-30T00:00:00+00:00", "pnl": -100.0},
    ]
    streak, usd = m._losing_streak(trades)
    assert streak == 3
    assert usd == pytest.approx(300.0)


# ── 11. _months_elapsed ───────────────────────────────────────────────────────
def test_months_elapsed():
    import scripts.research_position_sizing as m
    assert m._months_elapsed("2025-01", "2025-01") == 0
    assert m._months_elapsed("2025-01", "2025-06") == 5
    assert m._months_elapsed("2025-01", "2026-01") == 12
    assert m._months_elapsed("2025-11", "2026-02") == 3


# ── 12. _months_to_milestone ──────────────────────────────────────────────────
def test_months_to_milestone_reached():
    import scripts.research_position_sizing as m
    curve  = [(datetime(2025, 1, 1, tzinfo=UTC), 8000.0)]
    monthly = {"2025-01": 8000.0, "2025-06": 27500.0, "2025-12": 50000.0}
    result  = m._months_to_milestone(curve, monthly, 27_000.0)
    assert result == "2025-06"


def test_months_to_milestone_not_reached():
    import scripts.research_position_sizing as m
    curve   = [(datetime(2025, 1, 1, tzinfo=UTC), 8000.0)]
    monthly = {"2025-01": 8000.0, "2025-12": 15000.0}
    result  = m._months_to_milestone(curve, monthly, 27_000.0)
    assert result is None


# ── 13. No atexit / _sc mutation ─────────────────────────────────────────────
def test_no_sc_mutation():
    import src.strategy.forex.strategy_config as _sc
    import scripts.research_position_sizing as m
    # flat_risk_pct is a backtester argument, not a _sc field
    assert not hasattr(m, "_ORIG_CAP_SMALL"), \
        "research_position_sizing should not patch _sc — it uses flat_risk_pct kwarg"


# ── 14. MILESTONES ────────────────────────────────────────────────────────────
def test_milestones_contains_27k_and_100k():
    import scripts.research_position_sizing as m
    assert 27_000.0 in m.MILESTONES
    assert 100_000.0 in m.MILESTONES


# ── 15. Report path ───────────────────────────────────────────────────────────
def test_report_path():
    import scripts.research_position_sizing as m
    assert str(m.REPORT_PATH).endswith("research_position_sizing.md")
    assert "backtesting/results" in str(m.REPORT_PATH)
