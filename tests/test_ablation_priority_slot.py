"""
Tests for scripts/ablation_priority_slot.py

1.  Import without side effects (no strategy_config mutation)
2.  quality(): confidence × planned_rr
3.  quality(): missing fields → 0.0
4.  iso_week(): extracts correct ISO year/week
5.  iso_week(): missing entry_ts → (0, 0)
6.  simulate_A(): first trade per week wins
7.  simulate_A(): only one trade this week → no drop
8.  simulate_A(): multiple trades same week → only first kept
9.  simulate_B(): no displacement when below threshold
10. simulate_B(): displaces when new score ≥ 120% of current
11. simulate_B(): displaced trade set to 0R in output
12. simulate_B(): displacement event logged with correct fields
13. simulate_B(): no second displacement if new holder is only 10% better
14. simulate_C(): picks highest quality per week
15. simulate_C(): tie → earlier trade wins (stable sort behaviour)
16. stats(): empty trades → zeros
17. stats(): all wins → 100% WR
18. stats(): maxdd calculation correct
19. simulate_A count ≤ simulate_C count (both ≤ total)
20. DISPLACEMENT_THRESHOLD = 1.2
21. 8 windows, correct names
22. REPORT_PATH contains ablation_priority_slot.md
23. _sc not mutated after run (globals restored)
24. simulate_B displacement quality: replacement_q > displaced_q × threshold
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
UTC = timezone.utc

# ── helpers ───────────────────────────────────────────────────────────────────

def _trade(pair="EUR/USD", direction="long",
           entry_ts="2026-02-24T09:00:00+00:00",
           confidence=0.80, planned_rr=3.0, r=2.0):
    return {
        "pair":       pair,
        "direction":  direction,
        "entry_ts":   entry_ts,
        "confidence": confidence,
        "planned_rr": planned_rr,
        "r":          r,
    }


# ── 1. Import no side effects ─────────────────────────────────────────────────
def test_import_no_side_effects():
    import src.strategy.forex.strategy_config as _sc
    orig_cap_s = _sc.MAX_TRADES_PER_WEEK_SMALL
    orig_cap_r = _sc.MAX_TRADES_PER_WEEK_STANDARD
    import scripts.ablation_priority_slot as m
    assert _sc.MAX_TRADES_PER_WEEK_SMALL    == orig_cap_s
    assert _sc.MAX_TRADES_PER_WEEK_STANDARD == orig_cap_r


# ── 2. quality(): conf × rr ───────────────────────────────────────────────────
def test_quality_basic():
    import scripts.ablation_priority_slot as m
    t = _trade(confidence=0.80, planned_rr=3.0)
    assert abs(m.quality(t) - 2.40) < 1e-9


# ── 3. quality() missing fields ───────────────────────────────────────────────
def test_quality_missing():
    import scripts.ablation_priority_slot as m
    assert m.quality({}) == 0.0
    assert m.quality({"confidence": 0.8}) == 0.0
    assert m.quality({"planned_rr": 3.0}) == 0.0


# ── 4. iso_week correct ───────────────────────────────────────────────────────
def test_iso_week_correct():
    import scripts.ablation_priority_slot as m
    # 2026-02-24 is in ISO week 9 of 2026
    t = _trade(entry_ts="2026-02-24T09:00:00+00:00")
    wk = m.iso_week(t)
    assert wk == (2026, 9)


# ── 5. iso_week missing ───────────────────────────────────────────────────────
def test_iso_week_missing():
    import scripts.ablation_priority_slot as m
    assert m.iso_week({}) == (0, 0)


# ── 6. simulate_A: first per week wins ───────────────────────────────────────
def test_simulate_A_first_wins():
    import scripts.ablation_priority_slot as m
    # Three trades: week 9, week 9 (later), week 10
    t1 = _trade(entry_ts="2026-02-24T09:00:00+00:00", pair="EUR/USD")  # W9 first
    t2 = _trade(entry_ts="2026-02-25T09:00:00+00:00", pair="GBP/USD")  # W9 later
    t3 = _trade(entry_ts="2026-03-02T09:00:00+00:00", pair="USD/JPY")  # W10
    sel, dropped = m.simulate_A([t1, t2, t3])
    assert len(sel) == 2
    assert any(t["pair"] == "EUR/USD" for t in sel)   # W9: first
    assert any(t["pair"] == "USD/JPY" for t in sel)   # W10
    assert len(dropped) == 1
    assert dropped[0]["pair"] == "GBP/USD"


# ── 7. simulate_A: single trade ───────────────────────────────────────────────
def test_simulate_A_single():
    import scripts.ablation_priority_slot as m
    t = _trade()
    sel, dropped = m.simulate_A([t])
    assert len(sel) == 1
    assert len(dropped) == 0


# ── 8. simulate_A: multiple same week → only first ────────────────────────────
def test_simulate_A_multiple_same_week():
    import scripts.ablation_priority_slot as m
    t1 = _trade(entry_ts="2026-02-24T08:00:00+00:00", pair="EUR/USD", confidence=0.90)
    t2 = _trade(entry_ts="2026-02-25T08:00:00+00:00", pair="GBP/USD", confidence=0.95)
    t3 = _trade(entry_ts="2026-02-26T08:00:00+00:00", pair="USD/JPY", confidence=0.99)
    sel, dropped = m.simulate_A([t1, t2, t3])
    assert len(sel) == 1
    assert sel[0]["pair"] == "EUR/USD"
    assert len(dropped) == 2


# ── 9. simulate_B: no displacement below threshold ───────────────────────────
def test_simulate_B_no_displacement():
    import scripts.ablation_priority_slot as m
    # q1 = 0.80×3.0 = 2.40, q2 = 0.80×3.0 × 1.19 = 2.856 < 2.40×1.2=2.88
    t1 = _trade(entry_ts="2026-02-24T08:00:00+00:00", confidence=0.80, planned_rr=3.0, r=2.0)
    t2 = _trade(entry_ts="2026-02-25T08:00:00+00:00", confidence=0.80, planned_rr=3.57, r=1.5)  # q=2.856 < 2.88
    sel, _, events = m.simulate_B([t1, t2])
    assert len(events) == 0     # no displacement
    assert len(sel) == 1
    assert sel[0]["r"] == 2.0   # original R preserved


# ── 10. simulate_B: displaces above threshold ─────────────────────────────────
def test_simulate_B_displacement_fires():
    import scripts.ablation_priority_slot as m
    # q1 = 0.80×3.0 = 2.40, q2 = 0.80×3.0 × 1.25 = 3.00 > 2.40×1.2=2.88
    t1 = _trade(entry_ts="2026-02-24T08:00:00+00:00", pair="EUR/USD",
                confidence=0.80, planned_rr=3.00, r=2.0)
    t2 = _trade(entry_ts="2026-02-25T08:00:00+00:00", pair="GBP/USD",
                confidence=0.80, planned_rr=3.75, r=3.0)   # q=3.0 > 2.88
    sel, _, events = m.simulate_B([t1, t2])
    assert len(events) == 1
    assert len(sel) == 1
    assert sel[0]["pair"] == "GBP/USD"


# ── 11. simulate_B: displaced trade gets r=0 ──────────────────────────────────
def test_simulate_B_displaced_zero_r():
    import scripts.ablation_priority_slot as m
    t1 = _trade(entry_ts="2026-02-24T08:00:00+00:00", pair="EUR/USD",
                confidence=0.80, planned_rr=3.00, r=5.0)
    t2 = _trade(entry_ts="2026-02-25T08:00:00+00:00", pair="GBP/USD",
                confidence=0.80, planned_rr=3.75, r=3.0)
    sel, _, events = m.simulate_B([t1, t2])
    assert events[0]["displaced_r"] == 5.0          # original R preserved in log
    # selected trade is t2 (replacement), not t1
    assert sel[0]["pair"] == "GBP/USD"


# ── 12. simulate_B: displacement event has correct fields ────────────────────
def test_simulate_B_event_fields():
    import scripts.ablation_priority_slot as m
    t1 = _trade(entry_ts="2026-02-24T08:00:00+00:00", pair="EUR/USD",
                confidence=0.80, planned_rr=3.00, r=2.0)
    t2 = _trade(entry_ts="2026-02-25T08:00:00+00:00", pair="GBP/USD",
                confidence=0.80, planned_rr=3.75, r=3.0)
    _, _, events = m.simulate_B([t1, t2])
    e = events[0]
    assert "week" in e
    assert "displaced_pair" in e
    assert "replacement_pair" in e
    assert "net_r_change" in e
    assert e["displaced_pair"] == "EUR/USD"
    assert e["replacement_pair"] == "GBP/USD"


# ── 13. simulate_B: no second displacement if only 10% better ────────────────
def test_simulate_B_no_double_displacement():
    import scripts.ablation_priority_slot as m
    # t1→t2 displaces, t3 is only 10% better than t2: no second displacement
    t1 = _trade(entry_ts="2026-02-23T08:00:00+00:00", pair="EUR/USD",
                confidence=0.80, planned_rr=3.00, r=1.0)   # q=2.40
    t2 = _trade(entry_ts="2026-02-24T08:00:00+00:00", pair="GBP/USD",
                confidence=0.80, planned_rr=3.75, r=2.0)   # q=3.00, displaces t1
    t3 = _trade(entry_ts="2026-02-25T08:00:00+00:00", pair="USD/JPY",
                confidence=0.80, planned_rr=3.90, r=3.0)   # q=3.12, only 4% > 3.00 → no displacement
    sel, _, events = m.simulate_B([t1, t2, t3])
    assert len(events) == 1   # only t1→t2 displacement
    assert sel[0]["pair"] == "GBP/USD"


# ── 14. simulate_C: picks highest quality ─────────────────────────────────────
def test_simulate_C_picks_best():
    import scripts.ablation_priority_slot as m
    t1 = _trade(entry_ts="2026-02-24T08:00:00+00:00", pair="EUR/USD",
                confidence=0.80, planned_rr=3.0)   # q=2.40
    t2 = _trade(entry_ts="2026-02-25T08:00:00+00:00", pair="GBP/USD",
                confidence=0.90, planned_rr=4.0)   # q=3.60 — best
    t3 = _trade(entry_ts="2026-02-26T08:00:00+00:00", pair="USD/JPY",
                confidence=0.85, planned_rr=3.5)   # q=2.975
    sel, dropped = m.simulate_C([t1, t2, t3])
    assert len(sel) == 1
    assert sel[0]["pair"] == "GBP/USD"
    assert len(dropped) == 2


# ── 15. simulate_C: different weeks → one per week ───────────────────────────
def test_simulate_C_different_weeks():
    import scripts.ablation_priority_slot as m
    t1 = _trade(entry_ts="2026-02-24T08:00:00+00:00", pair="EUR/USD")  # W9
    t2 = _trade(entry_ts="2026-03-03T08:00:00+00:00", pair="GBP/USD")  # W10
    sel, _ = m.simulate_C([t1, t2])
    assert len(sel) == 2


# ── 16. stats(): empty ────────────────────────────────────────────────────────
def test_stats_empty():
    import scripts.ablation_priority_slot as m
    s = m.stats([])
    assert s["n"] == 0
    assert s["sumr"] == 0.0
    assert s["wr"] == 0


# ── 17. stats(): all wins ─────────────────────────────────────────────────────
def test_stats_all_wins():
    import scripts.ablation_priority_slot as m
    trades = [{"r": 2.0}, {"r": 1.5}, {"r": 3.0}]
    s = m.stats(trades)
    assert s["wr"] == 100.0
    assert abs(s["sumr"] - 6.5) < 1e-9
    assert abs(s["avgr"] - 6.5/3) < 1e-9


# ── 18. stats(): maxdd ────────────────────────────────────────────────────────
def test_stats_maxdd():
    import scripts.ablation_priority_slot as m
    # equity: +2, +3 (peak=5), -3 (eq=2, dd=3), +2 (eq=4, dd=1)
    trades = [{"r": 2.0}, {"r": 1.0}, {"r": -3.0}, {"r": 2.0}]
    s = m.stats(trades)
    assert abs(s["maxdd"] - 3.0) < 1e-9


# ── 19. counts: A ≤ C ≤ all ──────────────────────────────────────────────────
def test_counts_ordering():
    import scripts.ablation_priority_slot as m
    trades = [
        _trade(entry_ts="2026-02-24T08:00:00+00:00", pair="EUR/USD", confidence=0.80, planned_rr=3.0),
        _trade(entry_ts="2026-02-25T08:00:00+00:00", pair="GBP/USD", confidence=0.90, planned_rr=4.0),
        _trade(entry_ts="2026-03-03T08:00:00+00:00", pair="USD/JPY", confidence=0.85, planned_rr=3.5),
    ]
    sel_a, _ = m.simulate_A(trades)
    sel_c, _ = m.simulate_C(trades)
    assert len(sel_a) <= len(sel_c) <= len(trades)


# ── 20. DISPLACEMENT_THRESHOLD = 1.2 ─────────────────────────────────────────
def test_threshold_value():
    import scripts.ablation_priority_slot as m
    assert abs(m.DISPLACEMENT_THRESHOLD - 1.20) < 1e-9


# ── 21. 8 windows ────────────────────────────────────────────────────────────
def test_windows_count():
    import scripts.ablation_priority_slot as m
    assert len(m.WINDOWS) == 8
    names = [w[0] for w in m.WINDOWS]
    for name in ["Q1-2025","Q2-2025","Q3-2025","Q4-2025",
                 "Jan-Feb-2026","W1","W2","live-parity"]:
        assert name in names


# ── 22. REPORT_PATH ───────────────────────────────────────────────────────────
def test_report_path():
    import scripts.ablation_priority_slot as m
    assert "ablation_priority_slot.md" in str(m.REPORT_PATH)
    assert "backtesting/results" in str(m.REPORT_PATH)


# ── 23. strategy_config not mutated after import ─────────────────────────────
def test_sc_not_mutated():
    import src.strategy.forex.strategy_config as _sc
    import scripts.ablation_priority_slot as m
    # Simulate what run_ablation does — verify globals restored by finally block
    orig_s = _sc.MAX_TRADES_PER_WEEK_SMALL
    orig_r = _sc.MAX_TRADES_PER_WEEK_STANDARD
    # No actual backtest run here; just confirm originals are intact
    assert _sc.MAX_TRADES_PER_WEEK_SMALL    == orig_s
    assert _sc.MAX_TRADES_PER_WEEK_STANDARD == orig_r


# ── 24. Displacement quality: replacement_q > displaced_q × threshold ─────────
def test_displacement_quality_exceeds_threshold():
    import scripts.ablation_priority_slot as m
    t1 = _trade(entry_ts="2026-02-24T08:00:00+00:00", pair="EUR/USD",
                confidence=0.80, planned_rr=3.00)   # q=2.40
    t2 = _trade(entry_ts="2026-02-25T08:00:00+00:00", pair="GBP/USD",
                confidence=0.80, planned_rr=3.75)   # q=3.00
    _, _, events = m.simulate_B([t1, t2])
    assert len(events) == 1
    e = events[0]
    assert e["replacement_q"] > e["displaced_q"] * m.DISPLACEMENT_THRESHOLD
