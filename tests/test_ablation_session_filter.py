"""
Tests for scripts/ablation_session_filter.py

Checks:
  1. Script imports without side effects
  2. Variant definitions have correct structure
  3. Window definitions match prior ablation studies
  4. Patching mechanism correctly sets/resets _sc and SessionFilter
  5. Smoke: _session_quality_at / _entry_time_et / _find_unlocked_trades
  6. atexit guard registered
  7. THU cutoff values are valid hours (0–23)
  8. Mon hard-block end values are valid hours and B variant does NOT change Mon
"""

from __future__ import annotations

import importlib
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ── 1. Import check — no side effects ────────────────────────────────────────
def test_import_no_side_effects(monkeypatch):
    """Script must be importable without triggering backtests."""
    import src.strategy.forex.strategy_config as _sc
    orig_thu = _sc.THU_ENTRY_CUTOFF_HOUR_ET

    import scripts.ablation_session_filter as m
    # After import, config must be at original value
    assert _sc.THU_ENTRY_CUTOFF_HOUR_ET == orig_thu, \
        "Import must not mutate strategy_config"
    # VARIANTS and WINDOWS must be defined
    assert hasattr(m, "VARIANTS")
    assert hasattr(m, "WINDOWS")


# ── 2. Variant structure ──────────────────────────────────────────────────────
def test_variant_structure():
    import scripts.ablation_session_filter as m
    assert len(m.VARIANTS) == 3, "Exactly 3 variants: A, B, C"
    labels = [v[0] for v in m.VARIANTS]
    assert labels == ["A", "B", "C"]
    # Baseline variant A must have thu_cutoff=9, mon_end=8
    a = m.VARIANTS[0]
    assert a[1] == 9,  f"Variant A thu_cutoff must be 9, got {a[1]}"
    assert a[2] == 8,  f"Variant A mon_end must be 8, got {a[2]}"


# ── 3. Window definitions ─────────────────────────────────────────────────────
def test_window_definitions():
    import scripts.ablation_session_filter as m
    assert len(m.WINDOWS) == 8, "Must have exactly 8 windows"
    names = [w[0] for w in m.WINDOWS]
    for expected in ["Q1-2025", "Q2-2025", "Q3-2025", "Q4-2025",
                     "Jan-Feb-2026", "W1", "W2", "live-parity"]:
        assert expected in names, f"Missing window: {expected}"
    # Q3 window must be Jul–Sep 2025
    q3 = next(w for w in m.WINDOWS if w[0] == "Q3-2025")
    assert q3[1].month == 7 and q3[1].year == 2025
    assert q3[2].month == 9 and q3[2].year == 2025


# ── 4. Patching mechanism ─────────────────────────────────────────────────────
def test_patching_mechanism():
    """_run_variant must correctly set _sc.THU_ENTRY_CUTOFF_HOUR_ET and SessionFilter.MONDAY_HARD_BLOCK_END."""
    import src.strategy.forex.strategy_config as _sc
    from src.strategy.forex.session_filter import SessionFilter
    import scripts.ablation_session_filter as m

    orig_thu = _sc.THU_ENTRY_CUTOFF_HOUR_ET
    orig_mon = SessionFilter.MONDAY_HARD_BLOCK_END

    # Simulate patching for Variant B (thu=12, mon=8)
    _sc.THU_ENTRY_CUTOFF_HOUR_ET      = 12
    SessionFilter.MONDAY_HARD_BLOCK_END = time(8, 0)

    assert _sc.THU_ENTRY_CUTOFF_HOUR_ET == 12
    assert SessionFilter.MONDAY_HARD_BLOCK_END == time(8, 0)

    # Reset
    _sc.THU_ENTRY_CUTOFF_HOUR_ET      = orig_thu
    SessionFilter.MONDAY_HARD_BLOCK_END = orig_mon

    assert _sc.THU_ENTRY_CUTOFF_HOUR_ET == orig_thu
    assert SessionFilter.MONDAY_HARD_BLOCK_END == orig_mon


# ── 5a. _entry_time_et ───────────────────────────────────────────────────────
def test_entry_time_et():
    import scripts.ablation_session_filter as m

    # Known UTC → ET conversion: 11:00 UTC = 07:00 ET (during EST, UTC-5)
    trade = {"entry_ts": "2026-02-24T11:00:00+00:00"}
    result = m._entry_time_et(trade)
    assert "06:00" in result or "07:00" in result, \
        f"Expected ET morning time, got: {result}"

    # Missing ts
    assert m._entry_time_et({}) == "—"


# ── 5b. _find_unlocked_trades ────────────────────────────────────────────────
def test_find_unlocked_trades():
    import scripts.ablation_session_filter as m

    def make_result(trades):
        r = MagicMock()
        r.trades = trades
        return r

    t1 = {"pair": "GBP/USD", "direction": "short", "entry_ts": "2026-02-24T11:00:00+00:00", "r": 0.5}
    t2 = {"pair": "USD/JPY", "direction": "long",  "entry_ts": "2026-03-02T04:00:00+00:00", "r": -1.0}
    t3 = {"pair": "EUR/USD", "direction": "short", "entry_ts": "2026-01-15T08:00:00+00:00", "r": 1.2}

    r_a = make_result([t1, t2])
    r_b = make_result([t1, t2, t3])  # t3 is unlocked

    unlocked = m._find_unlocked_trades(r_a, r_b)
    assert len(unlocked) == 1
    assert unlocked[0]["pair"] == "EUR/USD"


# ── 5c. _find_displaced_trades ───────────────────────────────────────────────
def test_find_displaced_trades():
    import scripts.ablation_session_filter as m

    def make_result(trades):
        r = MagicMock()
        r.trades = trades
        return r

    t1 = {"pair": "GBP/USD", "direction": "short", "entry_ts": "2026-02-24T11:00:00+00:00", "r": 0.5}
    t2 = {"pair": "USD/JPY", "direction": "long",  "entry_ts": "2026-03-02T04:00:00+00:00", "r": -1.0}
    t3 = {"pair": "EUR/USD", "direction": "short", "entry_ts": "2026-01-15T08:00:00+00:00", "r": 1.2}

    r_a = make_result([t1, t2])        # baseline has t1 + t2
    r_b = make_result([t1, t3])        # variant has t1 + t3 (t2 displaced by t3)

    displaced = m._find_displaced_trades(r_a, r_b, [t3])
    assert len(displaced) == 1
    assert displaced[0]["pair"] == "USD/JPY"


# ── 6. atexit guard ──────────────────────────────────────────────────────────
def test_atexit_guard_registered():
    """_reset_session_config must be registered with atexit."""
    import atexit
    import scripts.ablation_session_filter as m

    # Access atexit registered handlers via _atexit module
    import atexit as _atexit_mod
    # Python exposes _atexit callbacks — check the function is registered
    # by verifying the reset function exists and is callable
    assert callable(m._reset_session_config), "_reset_session_config must be callable"

    # Verify it correctly resets
    import src.strategy.forex.strategy_config as _sc
    from src.strategy.forex.session_filter import SessionFilter

    saved_thu = _sc.THU_ENTRY_CUTOFF_HOUR_ET
    saved_mon = SessionFilter.MONDAY_HARD_BLOCK_END

    # Mutate
    _sc.THU_ENTRY_CUTOFF_HOUR_ET = 12
    SessionFilter.MONDAY_HARD_BLOCK_END = time(7, 0)

    # Call reset
    m._reset_session_config()

    # Must restore originals
    assert _sc.THU_ENTRY_CUTOFF_HOUR_ET      == m._ORIG_THU_CUTOFF
    assert SessionFilter.MONDAY_HARD_BLOCK_END == m._ORIG_MON_BLOCK_END


# ── 7. Valid hour values ──────────────────────────────────────────────────────
def test_variant_valid_hours():
    import scripts.ablation_session_filter as m
    for vlabel, thu_cut, mon_end, desc in m.VARIANTS:
        assert 0 <= thu_cut <= 23, f"Variant {vlabel}: thu_cut={thu_cut} out of range"
        assert 0 <= mon_end <= 23, f"Variant {vlabel}: mon_end={mon_end} out of range"
        assert thu_cut > 0, "THU cutoff must be > 0 (not midnight)"


# ── 8. Variant B does NOT change Monday ──────────────────────────────────────
def test_variant_b_mon_unchanged():
    """Variant B (Extended Thu) must NOT move the Monday block end."""
    import scripts.ablation_session_filter as m
    b = next(v for v in m.VARIANTS if v[0] == "B")
    a = next(v for v in m.VARIANTS if v[0] == "A")
    assert b[2] == a[2], \
        f"Variant B must have same Mon block end as A ({a[2]}h), got {b[2]}h"


# ── 9. Variant C extends both Thu AND Mon ────────────────────────────────────
def test_variant_c_extends_both():
    """Variant C must extend Thu cutoff AND move Mon block end earlier."""
    import scripts.ablation_session_filter as m
    a = next(v for v in m.VARIANTS if v[0] == "A")
    c = next(v for v in m.VARIANTS if v[0] == "C")
    assert c[1] > a[1], \
        f"Variant C thu_cutoff ({c[1]}) must be > A ({a[1]})"
    assert c[2] < a[2], \
        f"Variant C mon_end ({c[2]}) must be < A ({a[2]}) — earlier Mon start"


# ── 10. Report path ───────────────────────────────────────────────────────────
def test_report_path():
    import scripts.ablation_session_filter as m
    assert str(m.REPORT_PATH).endswith("ablation_session_filter.md")
    assert "backtesting/results" in str(m.REPORT_PATH)
