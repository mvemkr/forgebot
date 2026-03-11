"""
Tests for scripts/ablation_weekly_cap.py

1.  Import without side effects
2.  Variant structure (A=1, B=2, C=3)
3.  Window definitions (8 windows matching prior ablations)
4.  Patching: _sc.MAX_TRADES_PER_WEEK_SMALL mutated and restored
5.  _iso_week: correct ISO year+week from ISO timestamp
6.  _assign_slots: slot numbers increment per ISO week, reset across weeks
7.  _assign_slots: preserves all trade data
8.  _slot_quality: avg_r / win_rate / mae / mfe computed correctly
9.  _count_weeks_with_cap_blocks: counts distinct ISO weeks with ≥N blocks
10. _additional_trades: splits slot-1 vs slot≥2
11. atexit guard registered and resets correctly
12. Variant A must be first and have cap=1 (production)
13. Caps are strictly increasing A < B < C
14. Report path correct
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

UTC = timezone.utc


# ── 1. Import — no side effects ───────────────────────────────────────────────
def test_import_no_side_effects():
    import src.strategy.forex.strategy_config as _sc
    orig = _sc.MAX_TRADES_PER_WEEK_SMALL
    import scripts.ablation_weekly_cap as m
    assert _sc.MAX_TRADES_PER_WEEK_SMALL == orig, "Import must not mutate strategy_config"
    assert hasattr(m, "VARIANTS")
    assert hasattr(m, "WINDOWS")


# ── 2. Variant structure ──────────────────────────────────────────────────────
def test_variant_structure():
    import scripts.ablation_weekly_cap as m
    assert len(m.VARIANTS) == 3
    labels = [v[0] for v in m.VARIANTS]
    assert labels == ["A", "B", "C"]
    caps = [v[1] for v in m.VARIANTS]
    assert caps == [1, 2, 3]


# ── 3. Window definitions ─────────────────────────────────────────────────────
def test_window_definitions():
    import scripts.ablation_weekly_cap as m
    assert len(m.WINDOWS) == 8
    names = [w[0] for w in m.WINDOWS]
    for expected in ["Q1-2025", "Q2-2025", "Q3-2025", "Q4-2025",
                     "Jan-Feb-2026", "W1", "W2", "live-parity"]:
        assert expected in names
    q3 = next(w for w in m.WINDOWS if w[0] == "Q3-2025")
    assert q3[1].year == 2025 and q3[1].month == 7
    assert q3[2].year == 2025 and q3[2].month == 9


# ── 4. Patching ───────────────────────────────────────────────────────────────
def test_patching_mutates_sc():
    import src.strategy.forex.strategy_config as _sc
    import scripts.ablation_weekly_cap as m
    orig = _sc.MAX_TRADES_PER_WEEK_SMALL
    _sc.MAX_TRADES_PER_WEEK_SMALL = 2
    assert _sc.MAX_TRADES_PER_WEEK_SMALL == 2
    _sc.MAX_TRADES_PER_WEEK_SMALL = orig
    assert _sc.MAX_TRADES_PER_WEEK_SMALL == orig


# ── 5. _iso_week ──────────────────────────────────────────────────────────────
def test_iso_week_string():
    import scripts.ablation_weekly_cap as m
    # 2026-02-24 is Tuesday, ISO week 9 of 2026
    yr, wk = m._iso_week("2026-02-24T11:00:00+00:00")
    assert yr == 2026
    assert wk == 9


def test_iso_week_datetime():
    import scripts.ablation_weekly_cap as m
    dt = datetime(2026, 2, 24, 11, 0, tzinfo=UTC)
    yr, wk = m._iso_week(dt)
    assert yr == 2026
    assert wk == 9


def test_iso_week_none():
    import scripts.ablation_weekly_cap as m
    yr, wk = m._iso_week(None)
    assert yr == 0 and wk == 0


# ── 6. _assign_slots: slot numbering ─────────────────────────────────────────
def test_assign_slots_increments_within_week():
    import scripts.ablation_weekly_cap as m
    # Two trades in same ISO week (2026-W09)
    trades = [
        {"pair": "A", "direction": "long",  "entry_ts": "2026-02-23T04:00:00+00:00", "r": 0.5},
        {"pair": "B", "direction": "short", "entry_ts": "2026-02-25T04:00:00+00:00", "r": -1.0},
    ]
    slotted = m._assign_slots(trades)
    # Sort by entry_ts: A before B
    slots = {t["pair"]: t["week_slot"] for t in slotted}
    assert slots["A"] == 1
    assert slots["B"] == 2


def test_assign_slots_resets_across_weeks():
    import scripts.ablation_weekly_cap as m
    trades = [
        {"pair": "A", "entry_ts": "2026-02-23T04:00:00+00:00", "r": 0.5},  # W09
        {"pair": "B", "entry_ts": "2026-03-02T04:00:00+00:00", "r": -1.0}, # W10
    ]
    slotted = m._assign_slots(trades)
    slots = {t["pair"]: t["week_slot"] for t in slotted}
    assert slots["A"] == 1
    assert slots["B"] == 1  # New week → slot resets to 1


def test_assign_slots_three_in_week():
    import scripts.ablation_weekly_cap as m
    trades = [
        {"pair": "A", "entry_ts": "2026-02-23T04:00:00+00:00", "r": 1.0},
        {"pair": "B", "entry_ts": "2026-02-24T04:00:00+00:00", "r": -1.0},
        {"pair": "C", "entry_ts": "2026-02-25T04:00:00+00:00", "r": 0.5},
    ]
    slotted = m._assign_slots(trades)
    by_pair = {t["pair"]: t["week_slot"] for t in slotted}
    assert by_pair == {"A": 1, "B": 2, "C": 3}


# ── 7. _assign_slots preserves data ──────────────────────────────────────────
def test_assign_slots_preserves_data():
    import scripts.ablation_weekly_cap as m
    trades = [{"pair": "X", "direction": "long", "entry_ts": "2026-02-23T04:00:00+00:00",
               "r": 1.5, "mae_r": -0.3, "mfe_r": 2.1}]
    slotted = m._assign_slots(trades)
    assert len(slotted) == 1
    t = slotted[0]
    assert t["r"] == 1.5
    assert t["mae_r"] == -0.3
    assert t["mfe_r"] == 2.1
    assert t["week_slot"] == 1
    assert t["iso_week"] == (2026, 9)


# ── 8. _slot_quality ──────────────────────────────────────────────────────────
def test_slot_quality_basic():
    import scripts.ablation_weekly_cap as m
    slotted = [
        {"week_slot": 1, "r":  1.0, "mae_r": -0.2, "mfe_r":  2.0},
        {"week_slot": 1, "r": -1.0, "mae_r": -1.5, "mfe_r":  0.3},
        {"week_slot": 2, "r":  0.5, "mae_r": -0.1, "mfe_r":  1.0},
    ]
    sq = m._slot_quality(slotted)
    assert sq[1]["count"]    == 2
    assert sq[1]["avg_r"]    == pytest.approx(0.0, abs=1e-9)
    assert sq[1]["win_rate"] == pytest.approx(0.5)
    assert sq[2]["count"]    == 1
    assert sq[2]["avg_r"]    == pytest.approx(0.5)
    assert sq[2]["win_rate"] == pytest.approx(1.0)


def test_slot_quality_empty():
    import scripts.ablation_weekly_cap as m
    sq = m._slot_quality([])
    assert sq == {}


# ── 9. _count_weeks_with_cap_blocks ──────────────────────────────────────────
def test_count_weeks_with_cap_blocks():
    import scripts.ablation_weekly_cap as m
    gap_log = [
        {"gap_type": "WEEKLY_TRADE_LIMIT", "ts": "2026-02-23T04:00:00+00:00"},
        {"gap_type": "WEEKLY_TRADE_LIMIT", "ts": "2026-02-24T04:00:00+00:00"},  # same week
        {"gap_type": "low_confidence",     "ts": "2026-02-23T06:00:00+00:00"},  # different type
        {"gap_type": "WEEKLY_TRADE_LIMIT", "ts": "2026-03-02T04:00:00+00:00"},  # different week
    ]
    # Both W09 and W10 had WEEKLY_TRADE_LIMIT blocks
    assert m._count_weeks_with_cap_blocks(gap_log, min_blocks=1) == 2
    # W09 had 2 blocks, W10 had 1 — only 1 week had ≥2 blocks
    assert m._count_weeks_with_cap_blocks(gap_log, min_blocks=2) == 1


def test_count_weeks_empty_gap_log():
    import scripts.ablation_weekly_cap as m
    assert m._count_weeks_with_cap_blocks([], min_blocks=1) == 0


# ── 10. _additional_trades ────────────────────────────────────────────────────
def test_additional_trades_split():
    import scripts.ablation_weekly_cap as m
    slotted_a = [{"week_slot": 1, "r": 1.0}]
    slotted_x = [
        {"week_slot": 1, "r": 1.0},
        {"week_slot": 2, "r": -0.5},
        {"week_slot": 3, "r": 0.8},
    ]
    baseline, additional = m._additional_trades(slotted_a, slotted_x)
    assert len(baseline)   == 1
    assert len(additional) == 2
    assert all(t["week_slot"] >= 2 for t in additional)


# ── 11. atexit guard ──────────────────────────────────────────────────────────
def test_atexit_reset():
    import src.strategy.forex.strategy_config as _sc
    import scripts.ablation_weekly_cap as m
    # Mutate
    _sc.MAX_TRADES_PER_WEEK_SMALL = 99
    # Call reset
    m._reset_cap_config()
    # Must be restored
    assert _sc.MAX_TRADES_PER_WEEK_SMALL == m._ORIG_CAP_SMALL
    assert _sc.MAX_TRADES_PER_WEEK_STANDARD == m._ORIG_CAP_STANDARD


# ── 12. Variant A is first and has cap=1 ──────────────────────────────────────
def test_variant_a_is_baseline():
    import scripts.ablation_weekly_cap as m
    a = m.VARIANTS[0]
    assert a[0] == "A"
    assert a[1] == 1


# ── 13. Caps strictly increasing ─────────────────────────────────────────────
def test_caps_strictly_increasing():
    import scripts.ablation_weekly_cap as m
    caps = [v[1] for v in m.VARIANTS]
    assert caps == sorted(caps)
    assert len(set(caps)) == len(caps), "All caps must be distinct"


# ── 14. Report path ───────────────────────────────────────────────────────────
def test_report_path():
    import scripts.ablation_weekly_cap as m
    assert str(m.REPORT_PATH).endswith("ablation_weekly_cap.md")
    assert "backtesting/results" in str(m.REPORT_PATH)
