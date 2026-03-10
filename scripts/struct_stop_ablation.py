"""scripts/struct_stop_ablation.py
====================================
Structural Stop Variants A / B / C / D  ×  8 time windows.

Variants
--------
  A — Baseline: current production stop logic (no change)
  B — Structural Pivot:  raw structural anchor, NO buffer, NO ATR floor/ceiling
  C — Structural + ATR Ceiling: min(structural_dist, 3 × ATR_1H) cap
  D — Structural + ATR Noise Buffer: structural_dist + (0.5 × ATR_1H)

OFFLINE ONLY.  No live changes.  No master merge.
atexit() restores all monkeypatches and config mutations on exit.

Displacement analysis
---------------------
Because the weekly trade cap can displace trades when new entries appear,
for each non-baseline variant we report:
  - displaced_trade  (in A, not in variant) with its R
  - replacement_trade (in variant, not in A) with its R
  - net_delta_R

Floor rejection telemetry
--------------------------
For each structural stop candidate computed in B/C/D, we check whether
the raw structural distance would have been < 0.15 × ATR_1H (the production
floor in targeting.py).  These cases are logged as structural_stop_floor_rejections.
If this count is high, the structural variants are falling back to ATR and
results must be interpreted carefully.
"""
from __future__ import annotations

import atexit
import sys
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

# ── Path setup ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult, STARTING_BAL
import src.strategy.forex.strategy_config as _sc
import src.strategy.forex.set_and_forget as _saf
import src.strategy.forex.targeting as _tgt

# ── Preserve originals ───────────────────────────────────────────────────────
_ORIG_ATR_MIN_MULT: float = _sc.ATR_MIN_MULTIPLIER
_ORIG_STOP_FN = _saf.get_structure_stop  # bound name in set_and_forget module

def _reset_all() -> None:
    """Restore every patched value.  Called by atexit AND by _run_variant finally."""
    _sc.ATR_MIN_MULTIPLIER = _ORIG_ATR_MIN_MULT
    _saf.get_structure_stop = _ORIG_STOP_FN

atexit.register(_reset_all)

# ─────────────────────────────────────────────────────────────────────────────
# Per-run telemetry counters  (mutable dict so closures can write to them)
# ─────────────────────────────────────────────────────────────────────────────
_COUNTERS: Dict[str, int] = {
    "structural_stop_candidates":       0,
    "structural_stop_floor_rejections": 0,
    "fallback_to_baseline_stop_count":  0,
}

def _reset_counters() -> None:
    for k in _COUNTERS:
        _COUNTERS[k] = 0

def _snapshot_counters() -> Dict[str, int]:
    return dict(_COUNTERS)

# ─────────────────────────────────────────────────────────────────────────────
# Structural stop variant implementations
# ─────────────────────────────────────────────────────────────────────────────
_ATR_FLOOR_FRAC:    float = 0.15   # mirrors targeting._MIN_FRAC_ATR
_ATR_ABS_FLOOR_PIP: float = 8.0   # mirrors targeting._MIN_ABS_PIPS
_ATR_C_CEILING_MULT: float = 3.0  # Variant C: min(structural, 3×ATR_1H)
_ATR_D_BUFFER_MULT:  float = 0.5  # Variant D: structural_dist + 0.5×ATR_1H


def _struct_anchor_for_pattern(
    pattern_type: str,
    direction: str,
    entry: float,
    df_1h: Any,
    pattern: Any,
) -> Optional[float]:
    """
    Return the raw structural anchor for the given pattern (no buffer applied).

    H&S / IH&S  → right-shoulder extreme (pattern.stop_anchor)
                  then try retest rejection extreme if break/retest style entry
    break_retest → last rejection candle extreme near the broken level
                   (falls back to pattern.stop_anchor if not found)
    DT / DB / CB / sweep → pattern.stop_anchor (peak_high / trough_low / anchor)

    Returns None if no anchor is available or if anchor is on wrong side of entry.
    """
    anchor = pattern.stop_anchor

    if anchor is None:
        return None

    # For H&S entries: also try retest rejection extreme (neckline retest swing)
    if "head_and_shoulders" in pattern_type:
        neckline = getattr(pattern, "neckline", None)
        if neckline is not None:
            retest_ext = _tgt._find_h1_retest_rejection_extreme(df_1h, direction, neckline)
            if retest_ext is not None:
                # Use retest swing if it is TIGHTER (closer to entry) than shoulder
                retest_dist = abs(retest_ext - entry)
                anchor_dist = abs(anchor - entry)
                if retest_dist < anchor_dist:
                    # Verify correct side
                    if direction == "short" and retest_ext > entry:
                        anchor = retest_ext
                    elif direction == "long" and retest_ext < entry:
                        anchor = retest_ext

    # For break_retest: use retest rejection extreme near the broken level
    elif "break_retest" in pattern_type:
        retest_ext = _tgt._find_h1_retest_rejection_extreme(df_1h, direction, anchor)
        if retest_ext is not None:
            anchor = retest_ext

    # Validate correct side of entry
    if direction == "short" and anchor <= entry:
        return None  # wrong side — stop must be above entry for short
    if direction == "long"  and anchor >= entry:
        return None  # wrong side — stop must be below entry for long

    return anchor


def _make_variant_b_stop():
    """
    Variant B: raw structural anchor, no buffer, no ATR floor or ceiling.
    Falls back to ATR × atr_fallback_mult only when no structural anchor exists.
    """
    def _stop_b(
        pattern_type: str,
        direction: str,
        entry: float,
        df_1h: Any,
        pattern: Any,
        pip_size: float = 0.0001,
        is_jpy_or_cross: bool = False,
        atr_fallback_mult: float = 3.0,
        stop_log: Optional[list] = None,
        spread: Optional[float] = None,
    ) -> Tuple[float, str, float]:
        atr = _tgt._compute_atr_1h(df_1h)
        anchor = _struct_anchor_for_pattern(pattern_type, direction, entry, df_1h, pattern)

        if anchor is not None:
            _COUNTERS["structural_stop_candidates"] += 1
            raw_dist = abs(anchor - entry)
            # Track what the production floor would say
            floor = max(_ATR_ABS_FLOOR_PIP * pip_size,
                        _ATR_FLOOR_FRAC * atr if atr > 0 else _ATR_ABS_FLOOR_PIP * pip_size)
            if raw_dist < floor:
                _COUNTERS["structural_stop_floor_rejections"] += 1
            stop_pips = raw_dist / pip_size if pip_size > 0 else 0.0
            if stop_log is not None:
                stop_log.append({"action": f"STOP_SELECTED:b_struct_raw",
                                 "type": "b_struct_raw",
                                 "price": round(anchor, 6),
                                 "pips": round(stop_pips, 1)})
            return (anchor, "b_struct_raw", stop_pips)

        # No structural anchor → ATR fallback
        _COUNTERS["fallback_to_baseline_stop_count"] += 1
        if atr > 0:
            stop_price = (entry + atr * atr_fallback_mult
                          if direction == "short"
                          else entry - atr * atr_fallback_mult)
            stop_pips = atr * atr_fallback_mult / pip_size if pip_size > 0 else 0.0
            return (stop_price, "b_atr_fallback", stop_pips)
        # Ultimate fallback
        stop_pips = abs(pattern.stop_loss - entry) / pip_size if pip_size > 0 else 0.0
        return (pattern.stop_loss, "b_legacy", stop_pips)

    return _stop_b


def _make_variant_c_stop():
    """
    Variant C: structural anchor + ATR ceiling: stop_dist = min(struct, 3×ATR_1H).
    No floor.  No buffer.  Purpose: prevent excessively wide structural stops.
    """
    def _stop_c(
        pattern_type: str,
        direction: str,
        entry: float,
        df_1h: Any,
        pattern: Any,
        pip_size: float = 0.0001,
        is_jpy_or_cross: bool = False,
        atr_fallback_mult: float = 3.0,
        stop_log: Optional[list] = None,
        spread: Optional[float] = None,
    ) -> Tuple[float, str, float]:
        atr = _tgt._compute_atr_1h(df_1h)
        anchor = _struct_anchor_for_pattern(pattern_type, direction, entry, df_1h, pattern)

        if anchor is not None:
            _COUNTERS["structural_stop_candidates"] += 1
            raw_dist = abs(anchor - entry)
            # Track floor rejection
            floor = max(_ATR_ABS_FLOOR_PIP * pip_size,
                        _ATR_FLOOR_FRAC * atr if atr > 0 else _ATR_ABS_FLOOR_PIP * pip_size)
            if raw_dist < floor:
                _COUNTERS["structural_stop_floor_rejections"] += 1
            # Apply ceiling: cap stop at 3×ATR_1H
            ceiling = _ATR_C_CEILING_MULT * atr if atr > 0 else raw_dist
            stop_dist = min(raw_dist, ceiling)
            stop_price = (entry + stop_dist) if direction == "short" else (entry - stop_dist)
            stop_pips = stop_dist / pip_size if pip_size > 0 else 0.0
            stop_type = "c_struct_capped" if stop_dist < raw_dist else "c_struct_raw"
            if stop_log is not None:
                stop_log.append({"action": f"STOP_SELECTED:{stop_type}",
                                 "type": stop_type,
                                 "price": round(stop_price, 6),
                                 "pips": round(stop_pips, 1)})
            return (stop_price, stop_type, stop_pips)

        _COUNTERS["fallback_to_baseline_stop_count"] += 1
        if atr > 0:
            stop_price = (entry + atr * atr_fallback_mult
                          if direction == "short"
                          else entry - atr * atr_fallback_mult)
            stop_pips = atr * atr_fallback_mult / pip_size if pip_size > 0 else 0.0
            return (stop_price, "c_atr_fallback", stop_pips)
        stop_pips = abs(pattern.stop_loss - entry) / pip_size if pip_size > 0 else 0.0
        return (pattern.stop_loss, "c_legacy", stop_pips)

    return _stop_c


def _make_variant_d_stop():
    """
    Variant D: structural anchor + ATR noise buffer: stop_dist = struct + 0.5×ATR_1H.
    No floor cap.  Purpose: allow small noise beyond structure without resorting to wide ATR.
    """
    def _stop_d(
        pattern_type: str,
        direction: str,
        entry: float,
        df_1h: Any,
        pattern: Any,
        pip_size: float = 0.0001,
        is_jpy_or_cross: bool = False,
        atr_fallback_mult: float = 3.0,
        stop_log: Optional[list] = None,
        spread: Optional[float] = None,
    ) -> Tuple[float, str, float]:
        atr = _tgt._compute_atr_1h(df_1h)
        anchor = _struct_anchor_for_pattern(pattern_type, direction, entry, df_1h, pattern)

        if anchor is not None:
            _COUNTERS["structural_stop_candidates"] += 1
            raw_dist = abs(anchor - entry)
            floor = max(_ATR_ABS_FLOOR_PIP * pip_size,
                        _ATR_FLOOR_FRAC * atr if atr > 0 else _ATR_ABS_FLOOR_PIP * pip_size)
            if raw_dist < floor:
                _COUNTERS["structural_stop_floor_rejections"] += 1
            # Add noise buffer
            noise = _ATR_D_BUFFER_MULT * atr if atr > 0 else 0.0
            stop_dist = raw_dist + noise
            stop_price = (entry + stop_dist) if direction == "short" else (entry - stop_dist)
            stop_pips = stop_dist / pip_size if pip_size > 0 else 0.0
            if stop_log is not None:
                stop_log.append({"action": "STOP_SELECTED:d_struct_buffered",
                                 "type": "d_struct_buffered",
                                 "price": round(stop_price, 6),
                                 "pips": round(stop_pips, 1)})
            return (stop_price, "d_struct_buffered", stop_pips)

        _COUNTERS["fallback_to_baseline_stop_count"] += 1
        if atr > 0:
            stop_price = (entry + atr * atr_fallback_mult
                          if direction == "short"
                          else entry - atr * atr_fallback_mult)
            stop_pips = atr * atr_fallback_mult / pip_size if pip_size > 0 else 0.0
            return (stop_price, "d_atr_fallback", stop_pips)
        stop_pips = abs(pattern.stop_loss - entry) / pip_size if pip_size > 0 else 0.0
        return (pattern.stop_loss, "d_legacy", stop_pips)

    return _stop_d


# ─────────────────────────────────────────────────────────────────────────────
# Variant registry
# ─────────────────────────────────────────────────────────────────────────────
#  (id, label, stop_fn_or_None, patch_atr_floor)
#   stop_fn_or_None = None → use production stop (Variant A baseline)
#   patch_atr_floor = True → set _sc.ATR_MIN_MULTIPLIER = 0 during run
VARIANTS: List[Tuple[str, str, Any, bool]] = [
    ("A", "Baseline (production stop)",              None,                      False),
    ("B", "Structural Pivot (no buffer, no bounds)", _make_variant_b_stop(),    True),
    ("C", "Structural + ATR Ceiling (3×ATR_1H cap)", _make_variant_c_stop(),    True),
    ("D", "Structural + ATR Noise Buffer (+0.5×ATR)", _make_variant_d_stop(),   False),
]

# ─────────────────────────────────────────────────────────────────────────────
# Time windows  (same 8 as MIN_RR ablation for direct comparability)
# ─────────────────────────────────────────────────────────────────────────────
_UTC = timezone.utc
WINDOWS: List[Tuple[str, datetime, datetime]] = [
    ("Q1-2025",      datetime(2025,  1,  1, tzinfo=_UTC), datetime(2025,  3, 31, tzinfo=_UTC)),
    ("Q2-2025",      datetime(2025,  4,  1, tzinfo=_UTC), datetime(2025,  6, 30, tzinfo=_UTC)),
    ("Q3-2025",      datetime(2025,  7,  1, tzinfo=_UTC), datetime(2025,  9, 30, tzinfo=_UTC)),
    ("Q4-2025",      datetime(2025, 10,  1, tzinfo=_UTC), datetime(2025, 12, 31, tzinfo=_UTC)),
    ("Jan-Feb-2026", datetime(2026,  1,  1, tzinfo=_UTC), datetime(2026,  2, 28, tzinfo=_UTC)),
    ("W1",           datetime(2026,  2, 17, tzinfo=_UTC), datetime(2026,  2, 21, tzinfo=_UTC)),
    ("W2",           datetime(2026,  2, 24, tzinfo=_UTC), datetime(2026,  2, 28, tzinfo=_UTC)),
    ("live-parity",  datetime(2026,  3,  2, tzinfo=_UTC), datetime(2026,  3,  8, tzinfo=_UTC)),
]

# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WindowResult:
    variant:              str
    window:               str
    result:               BacktestResult
    candle_data:          Optional[dict]
    counters:             Dict[str, int]   # telemetry snapshot for this run

    @property
    def n(self) -> int:
        return self.result.n_trades

    @property
    def ret(self) -> float:
        return self.result.return_pct

    @property
    def wr(self) -> float:
        return self.result.win_rate

    @property
    def avg_r(self) -> float:
        return self.result.avg_r

    @property
    def max_dd(self) -> float:
        return self.result.max_dd_pct

    @property
    def trades(self) -> List[dict]:
        return self.result.trades or []

    @property
    def total_r(self) -> float:
        return sum(t.get("realized_r", 0.0) for t in self.trades)

    @property
    def expectancy(self) -> float:
        rs = [t.get("realized_r", 0.0) for t in self.trades]
        return sum(rs) / len(rs) if rs else 0.0

    @property
    def worst3(self) -> List[float]:
        rs = sorted(t.get("realized_r", 0.0) for t in self.trades)
        return rs[:3]

    @property
    def mae_r_list(self) -> List[float]:
        return [t.get("mae_r", 0.0) for t in self.trades if t.get("mae_r") is not None]

    @property
    def mfe_r_list(self) -> List[float]:
        return [t.get("mfe_r", 0.0) for t in self.trades if t.get("mfe_r") is not None]

    @property
    def stop_pips_list(self) -> List[float]:
        return [t.get("initial_stop_pips", 0.0) for t in self.trades
                if t.get("initial_stop_pips", 0.0) > 0]

    @property
    def stop_type_counts(self) -> Dict[str, int]:
        return self.result.stop_type_counts or {}

    @property
    def atr_fallback_pct(self) -> float:
        return self.result.atr_fallback_pct


@dataclass
class WindowQuad:
    """All 4 variant results for a single time window."""
    window:    str
    result_a:  WindowResult
    result_b:  WindowResult
    result_c:  WindowResult
    result_d:  WindowResult

    @property
    def results(self) -> List[WindowResult]:
        return [self.result_a, self.result_b, self.result_c, self.result_d]

    def by_variant(self, v: str) -> WindowResult:
        return {"A": self.result_a, "B": self.result_b,
                "C": self.result_c, "D": self.result_d}[v]


# ─────────────────────────────────────────────────────────────────────────────
# Trade-key helpers for unlock / displacement analysis
# ─────────────────────────────────────────────────────────────────────────────
def _trade_key(t: dict) -> Tuple[str, str, str]:
    """Canonical key: (pair, direction, entry_ts rounded to hour)."""
    ts = t.get("entry_time") or t.get("entry_ts") or ""
    pair = t.get("pair", "")
    direction = t.get("direction", "")
    return (pair, direction, str(ts)[:13])


def _find_unlocked(base_trades: List[dict], new_trades: List[dict]) -> List[dict]:
    """Trades in new_trades that are NOT in base_trades (newly unlocked)."""
    base_keys = {_trade_key(t) for t in base_trades}
    return [t for t in new_trades if _trade_key(t) not in base_keys]


def _find_removed(base_trades: List[dict], new_trades: List[dict]) -> List[dict]:
    """Trades in base_trades that are NOT in new_trades (displaced)."""
    new_keys  = {_trade_key(t) for t in new_trades}
    return [t for t in base_trades if _trade_key(t) not in new_keys]


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_r(v: Optional[float]) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}R"


def _fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def _fmt_wr(v: float) -> str:
    return f"{v*100:.0f}%"


def _p50(lst: List[float]) -> float:
    if not lst:
        return 0.0
    s = sorted(lst)
    return s[len(s) // 2]


def _avg(lst: List[float]) -> Optional[float]:
    return sum(lst) / len(lst) if lst else None


# ─────────────────────────────────────────────────────────────────────────────
# Run a single (variant, window) pair
# ─────────────────────────────────────────────────────────────────────────────
def _run_variant(
    var_id: str,
    var_label: str,
    stop_fn: Any,
    patch_atr_floor: bool,
    start_dt: datetime,
    end_dt: datetime,
    window_name: str,
    preloaded: Optional[dict] = None,
) -> Tuple[WindowResult, Optional[dict]]:
    """
    Monkeypatch strategy stop function, run backtest, restore, return result.

    For variants with patch_atr_floor=True we also zero out _sc.ATR_MIN_MULTIPLIER
    so the backtester's _stop_ok() doesn't reject structurally tight stops.
    """
    _reset_counters()

    # Install stop function patch
    if stop_fn is not None:
        _saf.get_structure_stop = stop_fn
    if patch_atr_floor:
        _sc.ATR_MIN_MULTIPLIER = 0.0

    try:
        bt: BacktestResult = run_backtest(
            start_dt=start_dt,
            end_dt=end_dt,
            starting_bal=STARTING_BAL,
            notes=f"struct_stop_{var_id}",
            preloaded_candle_data=preloaded,
            use_cache=True,
            quiet=True,
        )
        # Re-extract candle data for window reuse (first variant fetches, others reuse)
        candle_data = getattr(bt, "_candle_data", None)
    finally:
        _saf.get_structure_stop = _ORIG_STOP_FN
        _sc.ATR_MIN_MULTIPLIER = _ORIG_ATR_MIN_MULT

    wr = WindowResult(
        variant=var_id,
        window=window_name,
        result=bt,
        candle_data=candle_data,
        counters=_snapshot_counters(),
    )
    return wr, candle_data


# ─────────────────────────────────────────────────────────────────────────────
# Main ablation runner
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation(
    windows_filter: Optional[List[str]] = None,
    variants_filter: Optional[List[str]] = None,
) -> List[WindowQuad]:
    """
    Run all (window × variant) combinations.  Returns list of WindowQuad results.
    Candle data is cached after Variant A run and reused for B/C/D (same window).
    """
    windows_to_run  = WINDOWS  if not windows_filter  else [w for w in WINDOWS  if w[0] in windows_filter]
    variants_to_run = VARIANTS if not variants_filter else [v for v in VARIANTS if v[0] in variants_filter]

    quads: List[WindowQuad] = []

    for win_name, start_dt, end_dt in windows_to_run:
        print(f"\n{'='*60}")
        print(f"Window: {win_name}  ({start_dt.date()} → {end_dt.date()})")
        print(f"{'='*60}")

        results: Dict[str, WindowResult] = {}
        cached_candles: Optional[dict] = None

        for var_id, var_label, stop_fn, patch_floor in variants_to_run:
            print(f"  Variant {var_id}: {var_label} ...", end=" ", flush=True)
            wr, candle_data = _run_variant(
                var_id=var_id,
                var_label=var_label,
                stop_fn=stop_fn,
                patch_atr_floor=patch_floor,
                start_dt=start_dt,
                end_dt=end_dt,
                window_name=win_name,
                preloaded=cached_candles,
            )
            if candle_data and cached_candles is None:
                cached_candles = candle_data
            results[var_id] = wr

            # Live progress summary
            c = wr.counters
            print(
                f"trades={wr.n:2d}  WR={_fmt_wr(wr.wr):4s}  "
                f"SumR={_fmt_r(wr.total_r)}  Ret={_fmt_pct(wr.ret):6s}  "
                f"DD={_fmt_pct(wr.max_dd):6s}  "
                f"candidates={c['structural_stop_candidates']}  "
                f"floor_rej={c['structural_stop_floor_rejections']}  "
                f"fallbacks={c['fallback_to_baseline_stop_count']}"
            )

        # If filter excluded some variants, fill with empty sentinel
        def _get(v: str) -> WindowResult:
            if v in results:
                return results[v]
            dummy = BacktestResult()
            return WindowResult(variant=v, window=win_name, result=dummy,
                                candle_data=None, counters=dict(_COUNTERS))

        quads.append(WindowQuad(
            window=win_name,
            result_a=_get("A"),
            result_b=_get("B"),
            result_c=_get("C"),
            result_d=_get("D"),
        ))

    return quads


# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_report(quads: List[WindowQuad]) -> str:
    from datetime import datetime
    import pytz
    now_et = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d %H:%M ET")
    lines: List[str] = []
    W = lines.append

    W("# Structural Stop Ablation Study")
    W(f"Generated: {now_et}\n")

    W("## Config")
    W("| Setting | Value |")
    W("|---------|-------|")
    W(f"| ENTRY_TRIGGER_MODE | `{_sc.ENTRY_TRIGGER_MODE}` |")
    W(f"| STRICT_PIN_PATTERN_WHITELIST | `{_sc.STRICT_PIN_PATTERN_WHITELIST}` |")
    W(f"| ENGULF_CONFIRM_LOOKBACK_BARS | `{_sc.ENGULF_CONFIRM_LOOKBACK_BARS}` |")
    W(f"| MIN_CONFIDENCE | `{_sc.MIN_CONFIDENCE}` |")
    W(f"| MIN_RR (unchanged) | `{_sc.MIN_RR}` |")
    W(f"| RECOVERY_MIN_RR (unchanged) | `{_sc.RECOVERY_MIN_RR}` |")
    W(f"| ORIG ATR_MIN_MULTIPLIER | `{_ORIG_ATR_MIN_MULT}` |\n")

    W("## Variants")
    W("| ID | Label | Stop Logic | ATR Floor Bypassed |")
    W("|----|-------|------------|--------------------|")
    for vid, vlabel, _, pfl in VARIANTS:
        W(f"| {vid} | {vlabel} | see spec | {'Yes ⚠' if pfl else 'No'} |")
    W("")

    W("## Per-Window Results")
    W("")
    W("| Window | Var | Trades | WR | AvgR | SumR | Ret% | MaxDD | Expectancy | Worst3 |")
    W("|--------|-----|:------:|:--:|:----:|:----:|:----:|:-----:|:----------:|--------|")

    for quad in quads:
        first = True
        for wr in quad.results:
            prefix = f"| **{quad.window}**" if first else f"| {quad.window}"
            first = False
            w3 = "  ".join(_fmt_r(r) for r in wr.worst3) or "—"
            W(
                f"{prefix} | **{wr.variant}** | {wr.n} | {_fmt_wr(wr.wr)} | "
                f"{_fmt_r(wr.avg_r)} | {_fmt_r(wr.total_r)} | {_fmt_pct(wr.ret)} | "
                f"{_fmt_pct(wr.max_dd)} | {_fmt_r(wr.expectancy)} | {w3} |"
            )
        W("| | | | | | | | | | |")

    # ── Telemetry (floor rejections) ─────────────────────────────────────────
    W("## ATR Floor Telemetry (Variants B / C / D)")
    W("")
    W("| Window | Var | Stop Candidates | Floor Rejections | Fallback to ATR | Floor Rej % |")
    W("|--------|-----|:---------------:|:----------------:|:---------------:|:-----------:|")
    for quad in quads:
        for wr in [quad.result_b, quad.result_c, quad.result_d]:
            c = wr.counters
            cand = c["structural_stop_candidates"]
            rej  = c["structural_stop_floor_rejections"]
            fb   = c["fallback_to_baseline_stop_count"]
            pct  = f"{rej/cand*100:.0f}%" if cand > 0 else "n/a"
            W(f"| {quad.window} | {wr.variant} | {cand} | {rej} | {fb} | {pct} |")
    W("")

    # ── Stop distribution ────────────────────────────────────────────────────
    W("## Stop Width Distribution (pips)")
    W("")
    W("| Window | Var | Trades | Stop p50 | Min | Max | Type Distribution |")
    W("|--------|-----|:------:|:--------:|:---:|:---:|-------------------|")
    for quad in quads:
        for wr in quad.results:
            sp = wr.stop_pips_list
            p50 = f"{_p50(sp):.0f}" if sp else "—"
            mn  = f"{min(sp):.0f}" if sp else "—"
            mx  = f"{max(sp):.0f}" if sp else "—"
            st  = "  ".join(f"{k}:{v}" for k, v in sorted(wr.stop_type_counts.items())) or "—"
            W(f"| {quad.window} | {wr.variant} | {wr.n} | {p50}p | {mn}p | {mx}p | {st} |")
    W("")

    # ── Aggregate totals ─────────────────────────────────────────────────────
    W("## Aggregate Totals (all windows)")
    W("")
    W("| Var | Total Trades | Total SumR | Wins | Losses | WR | AvgR | AvgDD | Total Ret% |")
    W("|-----|:------------:|:----------:|:----:|:------:|:--:|:----:|:-----:|:----------:|")
    for var_id, var_label, _, _ in VARIANTS:
        all_trades: List[dict] = []
        ret_sum = 0.0
        dd_sum  = 0.0
        n_wins = n_losses = 0
        for quad in quads:
            wr = quad.by_variant(var_id)
            all_trades.extend(wr.trades)
            ret_sum += wr.ret
            dd_sum  += wr.max_dd
            for t in wr.trades:
                r = t.get("realized_r", 0.0)
                if r >= 0:
                    n_wins += 1
                else:
                    n_losses += 1
        n = len(all_trades)
        sum_r = sum(t.get("realized_r", 0.0) for t in all_trades)
        avg_r = sum_r / n if n > 0 else 0.0
        wr_frac = n_wins / n if n > 0 else 0.0
        avg_dd = dd_sum / len(quads) if quads else 0.0
        total_ret = ret_sum
        W(
            f"| **{var_id}** | {n} | {_fmt_r(sum_r)} | {n_wins} | {n_losses} | "
            f"{_fmt_wr(wr_frac)} | {_fmt_r(avg_r)} | {_fmt_pct(avg_dd)} | {_fmt_pct(total_ret)} |"
        )
    W("")

    # ── Unlock analysis ───────────────────────────────────────────────────────
    for vname, var_id, from_var in [
        ("B", "B", "A"),
        ("C", "C", "A"),
        ("D", "D", "A"),
    ]:
        W(f"## Unlock Analysis: A → {vname}")
        W("")
        all_unlocked: List[dict] = []
        all_displaced: List[dict] = []
        for quad in quads:
            base = quad.by_variant(from_var).trades
            cvar = quad.by_variant(var_id).trades
            all_unlocked.extend(_find_unlocked(base, cvar))
            all_displaced.extend(_find_removed(base, cvar))

        if not all_unlocked:
            W("No trades unlocked by this variant.\n")
        else:
            wr_u = sum(1 for t in all_unlocked if t.get("realized_r", 0) >= 0) / len(all_unlocked)
            sum_r_u = sum(t.get("realized_r", 0.0) for t in all_unlocked)
            avg_mae_u = _avg([t.get("mae_r", 0.0) for t in all_unlocked if t.get("mae_r") is not None])
            avg_mfe_u = _avg([t.get("mfe_r", 0.0) for t in all_unlocked if t.get("mfe_r") is not None])

            W(f"Total unlocked (A → {vname}): **{len(all_unlocked)}** trades")
            W(f"- WR: {_fmt_wr(wr_u)}")
            W(f"- AvgR: {_fmt_r(sum_r_u/len(all_unlocked) if all_unlocked else None)}")
            W(f"- SumR: {_fmt_r(sum_r_u)}")
            W(f"- Avg MAE: {_fmt_r(avg_mae_u)}")
            W(f"- Avg MFE: {_fmt_r(avg_mfe_u)}")
            W("")

            W(f"| Window | Pair | Dir | Pattern | StopPips | TargetPips | R | MAE | MFE | StopType |")
            W(f"|--------|------|-----|---------|:--------:|:----------:|:-:|:---:|:---:|----------|")
            for t in sorted(all_unlocked, key=lambda x: x.get("entry_time") or ""):
                pattern = t.get("pattern_type", t.get("pattern", "—"))
                stop_p  = t.get("initial_stop_pips", 0.0)
                tgt_p   = t.get("target_pips", "—")
                W(
                    f"| {str(t.get('entry_time',''))[:10]} "
                    f"| {t.get('pair','—')} | {t.get('direction','—')} "
                    f"| {pattern} | {stop_p:.0f}p | {tgt_p} "
                    f"| {_fmt_r(t.get('realized_r'))} "
                    f"| {_fmt_r(t.get('mae_r'))} | {_fmt_r(t.get('mfe_r'))} "
                    f"| {t.get('stop_type','—')} |"
                )
            W("")

        # ── Displacement table ───────────────────────────────────────────────
        W(f"### Displacement Table (A → {vname})")
        W("")
        if not all_displaced and not all_unlocked:
            W("No displacement — identical trade lists.\n")
        else:
            W("| Displaced Trade | Displaced R | Replacement Trade | Replacement R | Net ΔR |")
            W("|-----------------|:-----------:|-------------------|:-------------:|:------:|")
            # Match displaced to replacements (ordered by window/time)
            disp_map: List[Tuple[Optional[dict], Optional[dict]]] = []
            max_len = max(len(all_displaced), len(all_unlocked))
            for i in range(max_len):
                d = all_displaced[i] if i < len(all_displaced) else None
                u = all_unlocked[i]  if i < len(all_unlocked)  else None
                disp_map.append((d, u))

            net_total = 0.0
            for d, u in disp_map:
                dr = d.get("realized_r", 0.0) if d else 0.0
                ur = u.get("realized_r", 0.0) if u else 0.0
                net = ur - dr
                net_total += net
                d_label = f"{d.get('pair','?')} {d.get('direction','?')} {str(d.get('entry_time',''))[:10]}" if d else "—"
                u_label = f"{u.get('pair','?')} {u.get('direction','?')} {str(u.get('entry_time',''))[:10]}" if u else "—"
                W(f"| {d_label} | {_fmt_r(dr)} | {u_label} | {_fmt_r(ur)} | {_fmt_r(net)} |")

            sign = "+" if net_total >= 0 else ""
            W(f"| **TOTAL** | | | | **{sign}{net_total:.2f}R** |")
        W("")

    # ── Per-window unlock table ───────────────────────────────────────────────
    W("## Per-Window Unlock Summary")
    W("")
    W("| Window | B_unlocked | B_SumR | B_FloorRej | C_unlocked | C_SumR | C_FloorRej | D_unlocked | D_SumR | D_FloorRej |")
    W("|--------|:----------:|:------:|:----------:|:----------:|:------:|:----------:|:----------:|:------:|:----------:|")
    for quad in quads:
        bu = _find_unlocked(quad.result_a.trades, quad.result_b.trades)
        cu = _find_unlocked(quad.result_a.trades, quad.result_c.trades)
        du = _find_unlocked(quad.result_a.trades, quad.result_d.trades)
        W(
            f"| {quad.window} "
            f"| {len(bu)} | {_fmt_r(sum(t.get('realized_r',0) for t in bu))} | {quad.result_b.counters['structural_stop_floor_rejections']} "
            f"| {len(cu)} | {_fmt_r(sum(t.get('realized_r',0) for t in cu))} | {quad.result_c.counters['structural_stop_floor_rejections']} "
            f"| {len(du)} | {_fmt_r(sum(t.get('realized_r',0) for t in du))} | {quad.result_d.counters['structural_stop_floor_rejections']} |"
        )
    W("")

    # ── Pair / pattern distribution of unlocked trades ───────────────────────
    for vname in ["B", "C", "D"]:
        all_unlocked = []
        for quad in quads:
            base = quad.by_variant("A").trades
            cvar = quad.by_variant(vname).trades
            all_unlocked.extend(_find_unlocked(base, cvar))
        if not all_unlocked:
            continue
        W(f"### {vname}-Unlocked: Pair Distribution")
        pair_ctr = Counter(t.get("pair", "?") for t in all_unlocked)
        for pair, cnt in pair_ctr.most_common():
            pts = [t for t in all_unlocked if t.get("pair") == pair]
            sr  = sum(t.get("realized_r", 0.0) for t in pts)
            W(f"  - {pair}: {cnt} trades, SumR {_fmt_r(sr)}")
        W("")
        W(f"### {vname}-Unlocked: Pattern Distribution")
        pat_ctr = Counter(t.get("pattern_type", t.get("pattern", "?")) for t in all_unlocked)
        for pat, cnt in pat_ctr.most_common():
            W(f"  - {pat}: {cnt}")
        W("")
        W(f"### {vname}-Unlocked: Stop Width vs Baseline")
        sp_unlocked = [t.get("initial_stop_pips", 0.0) for t in all_unlocked if t.get("initial_stop_pips", 0.0) > 0]
        all_a = []
        for quad in quads:
            all_a.extend(quad.result_a.trades)
        sp_base = [t.get("initial_stop_pips", 0.0) for t in all_a if t.get("initial_stop_pips", 0.0) > 0]
        W(f"  - Baseline trades stop p50: {_p50(sp_base):.0f}p")
        W(f"  - {vname}-unlocked stop p50: {_p50(sp_unlocked):.0f}p")
        W("")

    # ── Conclusion / recommendation ───────────────────────────────────────────
    W("## Summary & Recommendation")
    W("")
    # Compute total SumR improvement vs baseline
    for vname in ["B", "C", "D"]:
        a_sum = sum(
            sum(t.get("realized_r", 0.0) for t in quad.by_variant("A").trades)
            for quad in quads
        )
        v_sum = sum(
            sum(t.get("realized_r", 0.0) for t in quad.by_variant(vname).trades)
            for quad in quads
        )
        delta = v_sum - a_sum
        sign  = "+" if delta >= 0 else ""
        W(f"- Variant {vname} vs A: total SumR {_fmt_r(v_sum)} (delta {sign}{delta:.2f}R)")

    W("")
    W("*(Promotion decision deferred to Mike — this is a report-only study.)*")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Structural stop ablation study")
    parser.add_argument("--windows",  nargs="+", help="Filter windows by name")
    parser.add_argument("--variants", nargs="+", help="Filter variants by ID (A/B/C/D)")
    parser.add_argument("--out", default="backtesting/results/ablation_structure_stops.md",
                        help="Output path for markdown report")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Structural Stop Ablation Study  (OFFLINE ONLY)        ║")
    print("║   Variants: A(baseline) B(raw struct) C(+ceil) D(+buf)  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"ATR_MIN_MULTIPLIER (original): {_ORIG_ATR_MIN_MULT}")
    print(f"ATR ceiling mult (C): {_ATR_C_CEILING_MULT}×ATR_1H")
    print(f"ATR noise buffer (D): {_ATR_D_BUFFER_MULT}×ATR_1H")
    print()

    quads = run_ablation(
        windows_filter=args.windows,
        variants_filter=args.variants,
    )

    report = _build_report(quads)

    out_path = Path(_ROOT) / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\n✅ Report written to: {out_path}")

    # Also print summary to stdout
    print("\n" + "="*60)
    print("AGGREGATE SUMMARY")
    print("="*60)
    for var_id, var_label, _, _ in VARIANTS:
        all_trades: List[dict] = []
        for quad in quads:
            all_trades.extend(quad.by_variant(var_id).trades)
        n = len(all_trades)
        sum_r = sum(t.get("realized_r", 0.0) for t in all_trades)
        nw = sum(1 for t in all_trades if t.get("realized_r", 0.0) >= 0)
        wr = nw / n if n > 0 else 0.0
        print(f"  Var {var_id}: {n:3d} trades  WR={_fmt_wr(wr)}  SumR={_fmt_r(sum_r)}"
              f"  ({var_label})")


if __name__ == "__main__":
    main()
