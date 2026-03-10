"""scripts/variant_c_sanity.py
================================
Variant C sanity confirmation: is the outperformance real?

Compares only:
  A  — Baseline (production stop, unchanged)
  C  — Structural + 3×ATR_1H ceiling
  C8 — Variant C with hard 8-pip minimum stop floor enforced

Checks:
  1. Outlier audit        — top 10 unlocked C trades by R, full detail
  2. Tiny-stop audit      — count of C trades with stop ≤ 3/5/8/10 pips
  3. Realism audit        — flags: too-tight vs ATR, R>8 from tiny stop,
                            single trade >20% of window SumR
  4. Robustness           — recompute C totals ex-top-1 and ex-top-2 outliers
  5. C with 8-pip floor   — aggregate + per-window, does C still beat A?

OFFLINE ONLY. No live changes. No master merge.
atexit() restores all patches on exit.
"""
from __future__ import annotations

import atexit
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult, STARTING_BAL
import src.strategy.forex.strategy_config as _sc
import src.strategy.forex.set_and_forget as _saf
import src.strategy.forex.targeting as _tgt

# ── Preserve originals ────────────────────────────────────────────────────────
_ORIG_ATR_MIN_MULT: float = _sc.ATR_MIN_MULTIPLIER
_ORIG_STOP_FN = _saf.get_structure_stop

def _reset_all() -> None:
    _sc.ATR_MIN_MULTIPLIER = _ORIG_ATR_MIN_MULT
    _saf.get_structure_stop = _ORIG_STOP_FN

atexit.register(_reset_all)

# ── Constants ─────────────────────────────────────────────────────────────────
_ATR_C_CEILING_MULT: float = 3.0   # min(structural, 3×ATR_1H)
_HARD_FLOOR_PIPS:    float = 8.0   # C8 hard minimum stop in pips
_ATR_FLOOR_FRAC:     float = 0.15

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

# ── Shared structural anchor helper (mirrors struct_stop_ablation) ────────────
def _struct_anchor_for_pattern(
    pattern_type: str, direction: str, entry: float,
    df_1h: Any, pattern: Any,
) -> Optional[float]:
    anchor = pattern.stop_anchor
    if anchor is None:
        return None
    if "head_and_shoulders" in pattern_type:
        neckline = getattr(pattern, "neckline", None)
        if neckline is not None:
            retest_ext = _tgt._find_h1_retest_rejection_extreme(df_1h, direction, neckline)
            if retest_ext is not None:
                retest_dist = abs(retest_ext - entry)
                anchor_dist = abs(anchor - entry)
                if retest_dist < anchor_dist:
                    if direction == "short" and retest_ext > entry:
                        anchor = retest_ext
                    elif direction == "long" and retest_ext < entry:
                        anchor = retest_ext
    elif "break_retest" in pattern_type:
        retest_ext = _tgt._find_h1_retest_rejection_extreme(df_1h, direction, anchor)
        if retest_ext is not None:
            anchor = retest_ext
    if direction == "short" and anchor <= entry:
        return None
    if direction == "long"  and anchor >= entry:
        return None
    return anchor


# ── Variant C stop (no floor, ceiling = 3×ATR_1H) ────────────────────────────
def _make_c_stop():
    def _stop_c(
        pattern_type, direction, entry, df_1h, pattern,
        pip_size=0.0001, is_jpy_or_cross=False,
        atr_fallback_mult=3.0, stop_log=None, spread=None,
    ):
        atr    = _tgt._compute_atr_1h(df_1h)
        anchor = _struct_anchor_for_pattern(pattern_type, direction, entry, df_1h, pattern)
        if anchor is not None:
            raw_dist = abs(anchor - entry)
            ceiling  = _ATR_C_CEILING_MULT * atr if atr > 0 else raw_dist
            stop_dist = min(raw_dist, ceiling)
            stop_price = (entry + stop_dist) if direction == "short" else (entry - stop_dist)
            stop_pips  = stop_dist / pip_size if pip_size > 0 else 0.0
            stype = "c_struct_capped" if stop_dist < raw_dist else "c_struct_raw"
            if stop_log is not None:
                stop_log.append({"action": f"STOP_SELECTED:{stype}", "type": stype,
                                 "price": round(stop_price, 6), "pips": round(stop_pips, 1)})
            return (stop_price, stype, stop_pips)
        # ATR fallback
        if atr > 0:
            stop_price = (entry + atr * atr_fallback_mult if direction == "short"
                          else entry - atr * atr_fallback_mult)
            return (stop_price, "c_atr_fallback", atr * atr_fallback_mult / pip_size)
        pips = abs(pattern.stop_loss - entry) / pip_size if pip_size > 0 else 0.0
        return (pattern.stop_loss, "c_legacy", pips)
    return _stop_c


# ── Variant C8 stop (same as C but hard floor = 8 pips) ──────────────────────
def _make_c8_stop():
    def _stop_c8(
        pattern_type, direction, entry, df_1h, pattern,
        pip_size=0.0001, is_jpy_or_cross=False,
        atr_fallback_mult=3.0, stop_log=None, spread=None,
    ):
        atr    = _tgt._compute_atr_1h(df_1h)
        anchor = _struct_anchor_for_pattern(pattern_type, direction, entry, df_1h, pattern)
        floor_dist = _HARD_FLOOR_PIPS * pip_size   # 8-pip hard minimum
        if anchor is not None:
            raw_dist  = abs(anchor - entry)
            ceiling   = _ATR_C_CEILING_MULT * atr if atr > 0 else raw_dist
            stop_dist = max(min(raw_dist, ceiling), floor_dist)   # ← floor applied
            stop_price = (entry + stop_dist) if direction == "short" else (entry - stop_dist)
            stop_pips  = stop_dist / pip_size if pip_size > 0 else 0.0
            stype = "c8_floored" if stop_dist == floor_dist else (
                    "c8_capped"  if stop_dist < raw_dist else "c8_raw")
            if stop_log is not None:
                stop_log.append({"action": f"STOP_SELECTED:{stype}", "type": stype,
                                 "price": round(stop_price, 6), "pips": round(stop_pips, 1)})
            return (stop_price, stype, stop_pips)
        if atr > 0:
            stop_dist  = max(atr * atr_fallback_mult, floor_dist)
            stop_price = (entry + stop_dist if direction == "short" else entry - stop_dist)
            return (stop_price, "c8_atr_fallback", stop_dist / pip_size)
        pips = abs(pattern.stop_loss - entry) / pip_size if pip_size > 0 else 0.0
        return (pattern.stop_loss, "c8_legacy", pips)
    return _stop_c8


# ── Run a single variant across all windows ───────────────────────────────────
@dataclass
class WinResult:
    window:      str
    n:           int
    wr:          float
    sum_r:       float
    ret_pct:     float
    max_dd:      float
    avg_r:       float
    trades:      List[dict]
    candle_data: Optional[dict] = None


def _run_window(
    var_id: str, stop_fn: Any, patch_floor: bool,
    start_dt: datetime, end_dt: datetime,
    window_name: str, preloaded: Optional[dict],
) -> WinResult:
    if stop_fn is not None:
        _saf.get_structure_stop = stop_fn
    if patch_floor:
        _sc.ATR_MIN_MULTIPLIER = 0.0
    try:
        bt: BacktestResult = run_backtest(
            start_dt=start_dt, end_dt=end_dt,
            starting_bal=STARTING_BAL, notes=f"sanity_{var_id}",
            preloaded_candle_data=preloaded, use_cache=True, quiet=True,
        )
    finally:
        _saf.get_structure_stop = _ORIG_STOP_FN
        _sc.ATR_MIN_MULTIPLIER  = _ORIG_ATR_MIN_MULT

    trades = bt.trades or []
    rs = [t.get("r", 0.0) for t in trades]
    n   = len(trades)
    wr  = sum(1 for r in rs if r >= 0) / n if n else 0.0
    sr  = sum(rs)
    avg = sr / n if n else 0.0
    return WinResult(
        window=window_name, n=n, wr=wr, sum_r=sr,
        ret_pct=bt.return_pct, max_dd=bt.max_dd_pct, avg_r=avg,
        trades=trades, candle_data=getattr(bt, "_candle_data", None),
    )


VARIANTS = [
    ("A",  None,          False),   # baseline
    ("C",  _make_c_stop(),  True),  # struct + 3×ATR ceiling, no floor
    ("C8", _make_c8_stop(), True),  # struct + 3×ATR ceiling + 8-pip hard floor
]


def run_all() -> Dict[str, List[WinResult]]:
    results: Dict[str, List[WinResult]] = {v[0]: [] for v in VARIANTS}
    for win_name, start_dt, end_dt in WINDOWS:
        print(f"\n{'='*56}\nWindow: {win_name}\n{'='*56}")
        cached: Optional[dict] = None
        for var_id, stop_fn, patch_floor in VARIANTS:
            print(f"  Var {var_id} ...", end=" ", flush=True)
            wr = _run_window(var_id, stop_fn, patch_floor,
                             start_dt, end_dt, win_name, cached)
            if wr.candle_data and cached is None:
                cached = wr.candle_data
            results[var_id].append(wr)
            sr_sign = "+" if wr.sum_r >= 0 else ""
            print(f"T={wr.n:2d}  WR={wr.wr*100:.0f}%  "
                  f"SumR={sr_sign}{wr.sum_r:.2f}R  Ret={wr.ret_pct:+.1f}%  DD={wr.max_dd:.1f}%")
    return results


# ── Trade-key helpers ─────────────────────────────────────────────────────────
def _tkey(t: dict) -> Tuple[str, str, str]:
    return (t.get("pair", ""), t.get("direction", ""), str(t.get("entry_ts", ""))[:13])

def _unlocked(base: List[dict], new: List[dict]) -> List[dict]:
    bkeys = {_tkey(t) for t in base}
    return [t for t in new if _tkey(t) not in bkeys]


# ── Analysis functions ────────────────────────────────────────────────────────
def _target_pips(t: dict) -> float:
    """Estimate target pips from stop_pips × planned_rr."""
    sp  = t.get("initial_stop_pips", 0.0) or 0.0
    rr  = t.get("planned_rr", 0.0) or 0.0
    return sp * rr


def _atr_ratio(t: dict) -> float:
    """stop_pips / ATR_1H_pips — computed from entry_risk/ATR if available.
    Returns 0.0 when unavailable."""
    # We don't store ATR directly in trade dict; use initial_risk vs entry_risk_dollars
    # as a proxy.  Return 0.0 if we can't compute it.
    return 0.0


def outlier_audit(unlocked_c: List[dict]) -> List[dict]:
    """Return top 10 unlocked C trades sorted by R descending with full detail."""
    top = sorted(unlocked_c, key=lambda t: t.get("r", 0.0), reverse=True)[:10]
    return top


def tiny_stop_counts(all_c_trades: List[dict]) -> Dict[str, int]:
    """Count C trades with stop_pips <= threshold."""
    thresholds = [3, 5, 8, 10]
    return {
        f"stop_le_{th}p": sum(1 for t in all_c_trades
                              if 0 < (t.get("initial_stop_pips") or 0) <= th)
        for th in thresholds
    }


def realism_flags(unlocked_c: List[dict], window_name: str, window_sum_r: float) -> List[str]:
    """Return list of realism warning strings for this window's unlocked C trades."""
    warnings: List[str] = []
    for t in unlocked_c:
        sp  = t.get("initial_stop_pips") or 0.0
        r   = t.get("r", 0.0)
        pair = t.get("pair", "?")
        ts   = str(t.get("entry_ts", ""))[:10]
        pat  = t.get("pattern", "?")

        if 0 < sp < 8:
            warnings.append(f"⚠ TINY_STOP {pair} {ts} {pat}: stop={sp:.1f}p < 8p floor")
        if r > 8:
            warnings.append(f"⚠ EXTREME_R {pair} {ts} {pat}: R={r:.2f}R  stop={sp:.1f}p")
        if window_sum_r > 0 and r / window_sum_r > 0.20:
            pct = r / window_sum_r * 100
            warnings.append(
                f"⚠ CONCENTRATION {pair} {ts} {pat}: "
                f"single trade = {pct:.0f}% of window SumR={window_sum_r:.2f}R"
            )
    return warnings


def robustness(all_c_trades: List[dict], all_a_trades: List[dict]) -> Dict[str, float]:
    """Recompute C SumR ex-top-1 and ex-top-2 highest-R trades."""
    sorted_by_r = sorted(all_c_trades, key=lambda t: t.get("r", 0.0), reverse=True)
    a_sum = sum(t.get("r", 0.0) for t in all_a_trades)
    c_sum = sum(t.get("r", 0.0) for t in all_c_trades)
    ex1   = sum(t.get("r", 0.0) for t in sorted_by_r[1:])   # drop top-1
    ex2   = sum(t.get("r", 0.0) for t in sorted_by_r[2:])   # drop top-2
    # top trade details
    top1  = sorted_by_r[0] if sorted_by_r else {}
    top2  = sorted_by_r[1] if len(sorted_by_r) > 1 else {}
    return {
        "a_sum_r":     a_sum,
        "c_sum_r":     c_sum,
        "c_ex_top1":   ex1,
        "c_ex_top2":   ex2,
        "c_beats_a_ex1": ex1 > a_sum,
        "c_beats_a_ex2": ex2 > a_sum,
        "top1_r":      top1.get("r", 0.0),
        "top1_pair":   top1.get("pair", "—"),
        "top1_stop":   top1.get("initial_stop_pips", 0.0),
        "top1_ts":     str(top1.get("entry_ts", ""))[:10],
        "top1_pat":    top1.get("pattern", "—"),
        "top2_r":      top2.get("r", 0.0),
        "top2_pair":   top2.get("pair", "—"),
        "top2_stop":   top2.get("initial_stop_pips", 0.0),
        "top2_ts":     str(top2.get("entry_ts", ""))[:10],
        "top2_pat":    top2.get("pattern", "—"),
    }


# ── Report builder ────────────────────────────────────────────────────────────
def _fmt_r(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:+.2f}R"

def _fmt_pct(v: float) -> str:
    return f"{v:+.1f}%"

def _fmt_wr(v: float) -> str:
    return f"{v*100:.0f}%"


def build_report(results: Dict[str, List[WinResult]],
                 windows: Optional[List] = None) -> str:
    if windows is None:
        windows = WINDOWS
    import pytz
    now_et = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d %H:%M ET")
    lines: List[str] = []
    W = lines.append

    W("# Variant C Sanity Confirmation")
    W(f"Generated: {now_et}\n")
    W("Offline only. Compares Baseline (A) vs Variant C (struct + 3×ATR_1H ceil)"
      " vs Variant C8 (C + 8-pip hard floor).\n")

    # ── 1. Per-window summary A vs C vs C8 ───────────────────────────────────
    W("## 1. Per-Window Summary")
    W("")
    W("| Window | Var | T | WR | SumR | Ret% | MaxDD | AvgR |")
    W("|--------|-----|:-:|:--:|:----:|:----:|:-----:|:----:|")
    for i, (win_name, _, _) in enumerate(windows):
        for var_id in ["A", "C", "C8"]:
            wr = results[var_id][i]
            bold = "**" if var_id != "A" else ""
            W(f"| {win_name if var_id=='A' else ''} | {bold}{var_id}{bold} | {wr.n} | "
              f"{_fmt_wr(wr.wr)} | {_fmt_r(wr.sum_r)} | {_fmt_pct(wr.ret_pct)} | "
              f"{_fmt_pct(wr.max_dd)} | {_fmt_r(wr.avg_r)} |")
        W("| | | | | | | | |")

    # Aggregates
    W("## 2. Aggregate Totals")
    W("")
    W("| Var | Trades | WR | SumR | Total Ret% | AvgDD | AvgR |")
    W("|-----|:------:|:--:|:----:|:----------:|:-----:|:----:|")
    for var_id in ["A", "C", "C8"]:
        all_t = [t for wr in results[var_id] for t in wr.trades]
        n  = len(all_t)
        rs = [t.get("r", 0.0) for t in all_t]
        sr = sum(rs)
        wrf = sum(1 for r in rs if r >= 0) / n if n else 0.0
        avg = sr / n if n else 0.0
        tot_ret = sum(wr.ret_pct for wr in results[var_id])
        avg_dd  = sum(wr.max_dd  for wr in results[var_id]) / len(windows)
        W(f"| **{var_id}** | {n} | {_fmt_wr(wrf)} | {_fmt_r(sr)} | "
          f"{_fmt_pct(tot_ret)} | {_fmt_pct(avg_dd)} | {_fmt_r(avg)} |")
    W("")

    # ── 2. Outlier audit — top 10 unlocked C trades ───────────────────────────
    all_a = [t for wr in results["A"]  for t in wr.trades]
    all_c = [t for wr in results["C"]  for t in wr.trades]
    all_unlocked = _unlocked(all_a, all_c)

    W("## 3. Outlier Audit — Top 10 Unlocked C Trades by R")
    W("")
    W("| # | Date | Pair | Dir | Pattern | Stop | Target | R | MAE | MFE | Stop<8p? | StopType |")
    W("|---|------|------|-----|---------|:----:|:------:|:-:|:---:|:---:|:--------:|----------|")
    top10 = outlier_audit(all_unlocked)
    for rank, t in enumerate(top10, 1):
        sp  = t.get("initial_stop_pips") or 0.0
        rr  = t.get("planned_rr") or 0.0
        tgt = sp * rr
        sub8 = "🚩 YES" if 0 < sp < 8 else "no"
        W(
            f"| {rank} | {str(t.get('entry_ts',''))[:10]} "
            f"| {t.get('pair','?')} | {t.get('direction','?')} "
            f"| {t.get('pattern','?')} "
            f"| {sp:.1f}p | {tgt:.1f}p | {_fmt_r(t.get('r'))} "
            f"| {_fmt_r(t.get('mae_r'))} | {_fmt_r(t.get('mfe_r'))} "
            f"| {sub8} | {t.get('stop_type','?')} |"
        )
    W("")

    # ── 3. Tiny-stop audit ────────────────────────────────────────────────────
    W("## 4. Tiny-Stop Audit (All C Trades)")
    W("")
    counts = tiny_stop_counts(all_c)
    W("| Threshold | Count | % of C trades |")
    W("|-----------|:-----:|:-------------:|")
    n_c = len(all_c)
    for th in [3, 5, 8, 10]:
        k = f"stop_le_{th}p"
        cnt = counts[k]
        pct = cnt / n_c * 100 if n_c else 0
        flag = " 🚩" if th <= 5 and cnt > 0 else ""
        W(f"| stop ≤ {th}p | {cnt} | {pct:.0f}%{flag} |")
    W("")

    # Also: list all C trades with stop < 8 pips
    sub8_trades = [t for t in all_c if 0 < (t.get("initial_stop_pips") or 0) < 8]
    if sub8_trades:
        W("### C Trades with Stop < 8 Pips (potential realism risk)")
        W("")
        W("| Date | Pair | Dir | Pattern | Stop | R | MAE | MFE |")
        W("|------|------|-----|---------|:----:|:-:|:---:|:---:|")
        for t in sorted(sub8_trades, key=lambda x: x.get("r", 0.0), reverse=True):
            sp = t.get("initial_stop_pips") or 0.0
            W(f"| {str(t.get('entry_ts',''))[:10]} | {t.get('pair','?')} "
              f"| {t.get('direction','?')} | {t.get('pattern','?')} "
              f"| {sp:.1f}p | {_fmt_r(t.get('r'))} "
              f"| {_fmt_r(t.get('mae_r'))} | {_fmt_r(t.get('mfe_r'))} |")
        W("")
    else:
        W("*No C trades have stop < 8 pips.*\n")

    # ── 4. Realism audit ──────────────────────────────────────────────────────
    W("## 5. Realism Audit (Per Window, Unlocked Trades)")
    W("")
    any_flags = False
    for i, (win_name, _, _) in enumerate(windows):
        a_trades = results["A"][i].trades
        c_trades = results["C"][i].trades
        win_unlocked = _unlocked(a_trades, c_trades)
        win_sum_r = sum(t.get("r", 0.0) for t in c_trades)
        flags = realism_flags(win_unlocked, win_name, win_sum_r)
        if flags:
            any_flags = True
            W(f"**{win_name}**")
            for f in flags:
                W(f"  - {f}")
            W("")
    if not any_flags:
        W("*No realism flags raised across all windows.*\n")

    # ── 5. Robustness ─────────────────────────────────────────────────────────
    W("## 6. Robustness Check")
    W("")
    rb = robustness(all_c, all_a)
    W("| Scenario | SumR | Beats A? |")
    W("|----------|:----:|:--------:|")
    W(f"| **A baseline** | {_fmt_r(rb['a_sum_r'])} | — |")
    W(f"| **C full** | {_fmt_r(rb['c_sum_r'])} | {'✅ Yes' if rb['c_sum_r'] > rb['a_sum_r'] else '❌ No'} |")
    W(f"| **C ex top-1** ({rb['top1_pair']} {rb['top1_ts']} {rb['top1_stop']:.0f}p {_fmt_r(rb['top1_r'])}) "
      f"| {_fmt_r(rb['c_ex_top1'])} | {'✅ Yes' if rb['c_beats_a_ex1'] else '❌ No'} |")
    W(f"| **C ex top-2** (also {rb['top2_pair']} {rb['top2_ts']} {rb['top2_stop']:.0f}p {_fmt_r(rb['top2_r'])}) "
      f"| {_fmt_r(rb['c_ex_top2'])} | {'✅ Yes' if rb['c_beats_a_ex2'] else '❌ No'} |")
    W("")

    # Outlier concentration
    W("### Top-2 Outlier Concentration")
    W("")
    top2_sum = rb['top1_r'] + rb['top2_r']
    pct_of_c = top2_sum / rb['c_sum_r'] * 100 if rb['c_sum_r'] else 0
    pct_of_delta = top2_sum / (rb['c_sum_r'] - rb['a_sum_r']) * 100 if (rb['c_sum_r'] - rb['a_sum_r']) else 0
    W(f"- Top-2 trades contribute **{_fmt_r(top2_sum)}** out of **{_fmt_r(rb['c_sum_r'])}** total C SumR ({pct_of_c:.0f}%)")
    W(f"- Top-2 trades contribute **{pct_of_delta:.0f}%** of the C vs A SumR delta ({_fmt_r(rb['c_sum_r'] - rb['a_sum_r'])})")
    W(f"- {'⚠ HIGH concentration — delta is largely driven by two outlier trades' if pct_of_delta > 60 else '✅ Concentration acceptable — delta not dominated by outliers'}")
    W("")

    # ── 6. C8 (8-pip floor) analysis ─────────────────────────────────────────
    W("## 7. Variant C8 — 8-Pip Hard Floor Applied")
    W("")
    all_c8 = [t for wr in results["C8"] for t in wr.trades]
    all_unlocked_c8 = _unlocked(all_a, all_c8)
    rs_c8 = [t.get("r", 0.0) for t in all_c8]
    n_c8  = len(all_c8)
    sr_c8 = sum(rs_c8)
    wr_c8 = sum(1 for r in rs_c8 if r >= 0) / n_c8 if n_c8 else 0.0
    avg_c8 = sr_c8 / n_c8 if n_c8 else 0.0

    W("*Same as Variant C but enforces stop ≥ 8 pips. "
      "Trades whose structural stop is < 8 pips are bumped to 8 pips. "
      "The backtester ATR-floor gate is still bypassed to isolate the effect.*\n")
    W(f"- **Total trades**: {n_c8}")
    W(f"- **Win rate**: {_fmt_wr(wr_c8)}")
    W(f"- **SumR**: {_fmt_r(sr_c8)}")
    W(f"- **Avg R**: {_fmt_r(avg_c8)}")
    W(f"- **Total Ret%**: {_fmt_pct(sum(wr.ret_pct for wr in results['C8']))}")
    W(f"- **Avg MaxDD**: {_fmt_pct(sum(wr.max_dd for wr in results['C8']) / len(windows))}")
    W(f"- **Unlocked vs A**: {len(all_unlocked_c8)} trades")
    W(f"- **Beats A?**: {'✅ Yes' if sr_c8 > rb['a_sum_r'] else '❌ No'}  "
      f"(A={_fmt_r(rb['a_sum_r'])}  C8={_fmt_r(sr_c8)})")
    W("")

    W("### C8 Per-Window")
    W("")
    W("| Window | T | WR | SumR | Ret% | MaxDD | vs A |")
    W("|--------|:-:|:--:|:----:|:----:|:-----:|:----:|")
    for i, (win_name, _, _) in enumerate(windows):
        a_wr  = results["A"][i]
        c8_wr = results["C8"][i]
        delta = c8_wr.sum_r - a_wr.sum_r
        flag  = "✅" if delta >= 0 else "❌"
        W(f"| {win_name} | {c8_wr.n} | {_fmt_wr(c8_wr.wr)} | {_fmt_r(c8_wr.sum_r)} "
          f"| {_fmt_pct(c8_wr.ret_pct)} | {_fmt_pct(c8_wr.max_dd)} "
          f"| {flag} {_fmt_r(delta)} |")
    W("")

    # ── 7. Recommendation ─────────────────────────────────────────────────────
    W("## 8. Recommendation")
    W("")

    # Evaluate
    c_beats_a      = rb["c_sum_r"]    > rb["a_sum_r"]
    c_ex1_beats_a  = rb["c_beats_a_ex1"]
    c_ex2_beats_a  = rb["c_beats_a_ex2"]
    c8_beats_a     = sr_c8            > rb["a_sum_r"]
    any_sub8       = counts["stop_le_8p"] > 0
    extreme_conc   = pct_of_delta > 60

    W("| Check | Result |")
    W("|-------|--------|")
    W(f"| C full beats A | {'✅' if c_beats_a else '❌'} |")
    W(f"| C ex top-1 beats A | {'✅' if c_ex1_beats_a else '❌'} |")
    W(f"| C ex top-2 beats A | {'✅' if c_ex2_beats_a else '❌'} |")
    W(f"| C8 (8-pip floor) beats A | {'✅' if c8_beats_a else '❌'} |")
    W(f"| Sub-8-pip stops present | {'🚩 Yes' if any_sub8 else '✅ No'} |")
    W(f"| Top-2 outliers >60% of delta | {'🚩 Yes' if extreme_conc else '✅ No'} |")
    W("")

    # Verdict
    if c_ex2_beats_a and c8_beats_a and not extreme_conc:
        verdict = "**✅ PROMOTE WITH GUARDRAIL — Variant C8 (8-pip floor enforced)**"
        detail  = (
            "Outperformance is real and robust: C beats A even ex-top-2 outliers, "
            "and C8 (with 8-pip floor) still beats A. "
            "Promote Variant C with the 8-pip hard floor applied "
            "to prevent fill-unrealistic micro-stops from inflating returns in live."
        )
    elif c_ex2_beats_a and c8_beats_a and extreme_conc:
        verdict = "**⚠ PROMOTE WITH GUARDRAIL + MONITOR — top-2 concentration is elevated**"
        detail  = (
            "C8 beats A robustly, but top-2 outliers explain >60% of the delta. "
            "Promote C8 (8-pip floor) and monitor first 20 live trades closely."
        )
    elif c_ex1_beats_a and not c_ex2_beats_a:
        verdict = "**⚠ NEEDS GUARDRAILS FIRST — collapses ex-top-2**"
        detail  = (
            "C beats A ex-top-1 but not ex-top-2. "
            "The delta is driven by two outlier trades. "
            "Enforce 8-pip floor AND a max single-trade-R cap before promoting."
        )
    else:
        verdict = "**❌ REJECT — outperformance does not survive outlier removal**"
        detail  = (
            "C's edge collapses when outliers are removed. "
            "Do not promote. Investigate stop geometry further."
        )

    W(f"### Verdict: {verdict}")
    W("")
    W(detail)
    W("")
    W("---")
    W("*Promotion decision deferred to Mike.*")

    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="backtesting/results/variant_c_sanity.md")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║  Variant C Sanity Confirmation  (OFFLINE ONLY)  ║")
    print("║  A vs C vs C8 (8-pip floor)                     ║")
    print("╚══════════════════════════════════════════════════╝\n")

    results = run_all()

    report = build_report(results)

    out_path = Path(_ROOT) / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\n✅ Report written to: {out_path}")

    # Quick console summary
    all_a   = [t for wr in results["A"]  for t in wr.trades]
    all_c   = [t for wr in results["C"]  for t in wr.trades]
    all_c8  = [t for wr in results["C8"] for t in wr.trades]
    print("\n" + "="*50)
    for var_id, all_t in [("A", all_a), ("C", all_c), ("C8", all_c8)]:
        rs = [t.get("r", 0.0) for t in all_t]
        n  = len(rs)
        sr = sum(rs)
        wr = sum(1 for r in rs if r >= 0) / n if n else 0
        print(f"  Var {var_id}: {n:2d}T  WR={wr*100:.0f}%  SumR={sr:+.2f}R")


if __name__ == "__main__":
    main()
