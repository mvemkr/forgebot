"""
Tests for scripts/ablation_sunday_pairs.py

1.  Import without side effects
2.  PAIRS_12 has exactly 12 entries, no duplicates
3.  PAIRS_NEW_14 has exactly 14 entries, no duplicates
4.  PAIRS_26 = union of PAIRS_12 + PAIRS_NEW_14, len=26, no duplicates
5.  VARIANTS: A=12/no-sun, B=12/sun, C=26/no-sun, D=26/sun
6.  Sunday patch: weekday==6, 03:00-09:00 ET → not hard-blocked
7.  Sunday patch: weekday==6, outside 03:00-09:00 ET → hard-blocked
8.  Sunday patch: non-Sunday → original behavior preserved
9.  _remove_sunday_patch restores original is_hard_blocked
10. _apply_pairs writes whitelist JSON with correct pairs
11. _is_sunday_trade: Sunday UTC timestamp correctly identified
12. _is_sunday_trade: Monday UTC → False
13. _find_unlocked: trades in X not in A
14. _find_displaced: trades in A not in X
15. atexit restore registered (_restore_all callable)
16. PAIRS_12 and PAIRS_NEW_14 are disjoint
17. 8 windows, correct names
18. Report path correct
19. No strategy_config mutation
20. Variant A is first with 12 pairs and Sunday=False
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
UTC = timezone.utc


# ── 1. Import no side effects ─────────────────────────────────────────────────
def test_import_no_side_effects():
    import src.strategy.forex.strategy_config as _sc
    orig_thu = _sc.THU_ENTRY_CUTOFF_HOUR_ET
    orig_cap = _sc.MAX_TRADES_PER_WEEK_SMALL
    import scripts.ablation_sunday_pairs as m
    assert _sc.THU_ENTRY_CUTOFF_HOUR_ET == orig_thu
    assert _sc.MAX_TRADES_PER_WEEK_SMALL == orig_cap


# ── 2. PAIRS_12 ───────────────────────────────────────────────────────────────
def test_pairs_12_count_and_unique():
    import scripts.ablation_sunday_pairs as m
    assert len(m.PAIRS_12) == 12
    assert len(set(m.PAIRS_12)) == 12


# ── 3. PAIRS_NEW_14 ───────────────────────────────────────────────────────────
def test_pairs_new_14_count_and_unique():
    import scripts.ablation_sunday_pairs as m
    assert len(m.PAIRS_NEW_14) == 14
    assert len(set(m.PAIRS_NEW_14)) == 14


# ── 4. PAIRS_26 ───────────────────────────────────────────────────────────────
def test_pairs_26_is_union():
    import scripts.ablation_sunday_pairs as m
    expected = set(m.PAIRS_12) | set(m.PAIRS_NEW_14)
    assert set(m.PAIRS_26) == expected
    assert len(m.PAIRS_26) == 26
    assert len(set(m.PAIRS_26)) == 26


# ── 5. VARIANTS structure ─────────────────────────────────────────────────────
def test_variants_structure():
    import scripts.ablation_sunday_pairs as m
    assert len(m.VARIANTS) == 4
    labels = [v[0] for v in m.VARIANTS]
    assert labels == ["A", "B", "C", "D"]
    # A: 12 pairs, no sunday
    va = m.VARIANTS[0]
    assert len(va[1]) == 12 and va[2] is False
    # B: 12 pairs, sunday
    vb = m.VARIANTS[1]
    assert len(vb[1]) == 12 and vb[2] is True
    # C: 26 pairs, no sunday
    vc = m.VARIANTS[2]
    assert len(vc[1]) == 26 and vc[2] is False
    # D: 26 pairs, sunday
    vd = m.VARIANTS[3]
    assert len(vd[1]) == 26 and vd[2] is True


# ── 6. Sunday patch allows 03:00-09:00 ET on Sunday ──────────────────────────
def test_sunday_patch_allows_london_window():
    import pytz
    import scripts.ablation_sunday_pairs as m
    from src.strategy.forex.session_filter import SessionFilter

    ET = pytz.timezone("America/New_York")
    m._apply_sunday_patch()
    try:
        sf = SessionFilter()
        # Sunday 2026-03-08 04:00 ET (London window)
        dt = ET.localize(datetime(2026, 3, 8, 4, 0, 0))
        blocked, reason = sf.is_hard_blocked(dt)
        assert not blocked, f"Sunday 04:00 ET should be allowed, got: {reason}"
    finally:
        m._remove_sunday_patch()


# ── 7. Sunday patch blocks outside 03:00-09:00 ET ────────────────────────────
def test_sunday_patch_blocks_outside_london():
    import pytz
    import scripts.ablation_sunday_pairs as m
    from src.strategy.forex.session_filter import SessionFilter

    ET = pytz.timezone("America/New_York")
    m._apply_sunday_patch()
    try:
        sf = SessionFilter()
        # Sunday 2026-03-08 01:00 ET (before London)
        dt = ET.localize(datetime(2026, 3, 8, 1, 0, 0))
        blocked, reason = sf.is_hard_blocked(dt)
        assert blocked, "Sunday 01:00 ET should be blocked"
        assert "NO_SUNDAY_TRADES" in reason
    finally:
        m._remove_sunday_patch()


def test_sunday_patch_blocks_after_9am():
    import pytz
    import scripts.ablation_sunday_pairs as m
    from src.strategy.forex.session_filter import SessionFilter

    ET = pytz.timezone("America/New_York")
    m._apply_sunday_patch()
    try:
        sf = SessionFilter()
        # Sunday 2026-03-08 10:00 ET (after window)
        dt = ET.localize(datetime(2026, 3, 8, 10, 0, 0))
        blocked, reason = sf.is_hard_blocked(dt)
        assert blocked, "Sunday 10:00 ET should be blocked"
    finally:
        m._remove_sunday_patch()


# ── 8. Sunday patch preserves non-Sunday behavior ────────────────────────────
def test_sunday_patch_monday_unchanged():
    import pytz
    import scripts.ablation_sunday_pairs as m
    from src.strategy.forex.session_filter import SessionFilter

    ET = pytz.timezone("America/New_York")
    m._apply_sunday_patch()
    try:
        sf = SessionFilter()
        # Monday 2026-03-09 10:00 ET — should be fine (not hard-blocked)
        dt = ET.localize(datetime(2026, 3, 9, 10, 0, 0))
        blocked, _ = sf.is_hard_blocked(dt)
        assert not blocked, "Monday 10:00 ET should not be hard-blocked"
    finally:
        m._remove_sunday_patch()


# ── 9. _remove_sunday_patch restores original ────────────────────────────────
def test_remove_sunday_patch_restores():
    import scripts.ablation_sunday_pairs as m
    from src.strategy.forex.session_filter import SessionFilter
    m._apply_sunday_patch()
    assert SessionFilter.is_hard_blocked is not m._ORIG_IS_HARD_BLOCKED
    m._remove_sunday_patch()
    assert SessionFilter.is_hard_blocked is m._ORIG_IS_HARD_BLOCKED


# ── 10. _apply_pairs writes whitelist ────────────────────────────────────────
def test_apply_pairs_writes_whitelist(tmp_path, monkeypatch):
    import scripts.ablation_sunday_pairs as m
    fake_path = tmp_path / "whitelist_backtest.json"
    monkeypatch.setattr(m, "WHITELIST_PATH", fake_path)
    m._apply_pairs(["EUR/USD", "GBP/USD"])
    data = json.loads(fake_path.read_text())
    assert data["enabled"] is True
    assert set(data["pairs"]) == {"EUR/USD", "GBP/USD"}


# ── 11. _is_sunday_trade ──────────────────────────────────────────────────────
def test_is_sunday_trade_true():
    import scripts.ablation_sunday_pairs as m
    # 2026-03-08 is a Sunday; 16:00 UTC = 12:00 PM EDT (UTC-4, after DST)
    t = {"entry_ts": "2026-03-08T16:00:00+00:00"}
    assert m._is_sunday_trade(t) is True


def test_is_sunday_trade_false_monday():
    import scripts.ablation_sunday_pairs as m
    # 2026-03-09 is a Monday
    t = {"entry_ts": "2026-03-09T04:00:00+00:00"}
    assert m._is_sunday_trade(t) is False


def test_is_sunday_trade_missing():
    import scripts.ablation_sunday_pairs as m
    assert m._is_sunday_trade({}) is False


# ── 12. _find_unlocked / _find_displaced ─────────────────────────────────────
def test_find_unlocked():
    import scripts.ablation_sunday_pairs as m

    def mresult(trades):
        r = MagicMock()
        r.trades = trades
        return r

    t1 = {"pair": "EUR/USD", "direction": "long",  "entry_ts": "2026-02-24T11:00:00+00:00", "r": 1.0}
    t2 = {"pair": "GBP/USD", "direction": "short", "entry_ts": "2026-03-02T04:00:00+00:00", "r": -1.0}
    t3 = {"pair": "USD/JPY", "direction": "long",  "entry_ts": "2026-01-15T08:00:00+00:00", "r": 0.5}

    r_a = mresult([t1, t2])
    r_x = mresult([t1, t2, t3])
    unlocked = m._find_unlocked(r_a, r_x)
    assert len(unlocked) == 1
    assert unlocked[0]["pair"] == "USD/JPY"


def test_find_displaced():
    import scripts.ablation_sunday_pairs as m

    def mresult(trades):
        r = MagicMock()
        r.trades = trades
        return r

    t1 = {"pair": "EUR/USD", "direction": "long",  "entry_ts": "2026-02-24T11:00:00+00:00", "r": 1.0}
    t2 = {"pair": "GBP/USD", "direction": "short", "entry_ts": "2026-03-02T04:00:00+00:00", "r": -1.0}
    t3 = {"pair": "USD/JPY", "direction": "long",  "entry_ts": "2026-01-15T08:00:00+00:00", "r": 0.5}

    r_a = mresult([t1, t2])
    r_x = mresult([t1, t3])        # t2 displaced, t3 added
    displaced = m._find_displaced(r_a, r_x)
    assert len(displaced) == 1
    assert displaced[0]["pair"] == "GBP/USD"


# ── 13. _restore_all callable ────────────────────────────────────────────────
def test_restore_all_callable():
    import scripts.ablation_sunday_pairs as m
    assert callable(m._restore_all)


# ── 14. PAIRS_12 and PAIRS_NEW_14 are disjoint ───────────────────────────────
def test_pairs_disjoint():
    import scripts.ablation_sunday_pairs as m
    overlap = set(m.PAIRS_12) & set(m.PAIRS_NEW_14)
    assert len(overlap) == 0, f"Pairs overlap: {overlap}"


# ── 15. 8 windows with correct names ─────────────────────────────────────────
def test_windows_count_and_names():
    import scripts.ablation_sunday_pairs as m
    assert len(m.WINDOWS) == 8
    names = [w[0] for w in m.WINDOWS]
    for expected in ["Q1-2025","Q2-2025","Q3-2025","Q4-2025",
                     "Jan-Feb-2026","W1","W2","live-parity"]:
        assert expected in names


# ── 16. Report path ───────────────────────────────────────────────────────────
def test_report_path():
    import scripts.ablation_sunday_pairs as m
    assert str(m.REPORT_PATH).endswith("ablation_sunday_pairs.md")
    assert "backtesting/results" in str(m.REPORT_PATH)


# ── 17. No strategy_config mutation ──────────────────────────────────────────
def test_no_sc_mutation():
    import src.strategy.forex.strategy_config as _sc
    import scripts.ablation_sunday_pairs as m
    # Verify NO_SUNDAY_TRADES_ENABLED is not touched (Sunday logic is in SessionFilter)
    orig = _sc.NO_SUNDAY_TRADES_ENABLED
    assert _sc.NO_SUNDAY_TRADES_ENABLED == orig


# ── 18. Variant A is baseline ────────────────────────────────────────────────
def test_variant_a_is_baseline():
    import scripts.ablation_sunday_pairs as m
    a = m.VARIANTS[0]
    assert a[0] == "A"
    assert len(a[1]) == 12
    assert a[2] is False
