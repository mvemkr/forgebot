#!/usr/bin/env python3
"""
W2 Diagnostic: Pro-Trend-Only vs Baseline  (strict parity enforced)
====================================================================
Runs two backtests on the W2 window (Oct 1 2025 – Feb 28 2026).
The ONLY difference between Run 1 and Run 2 is PROTREND_ONLY.

Hard constraints (aborts if violated):
  • Same commit SHA
  • Same pairs (hash 89a6ef)
  • Same trail arm (must be C)
  • Same MIN_RR (2.5)
  • Same risk tiers, weekly cap, time rules, doji filter, concurrency

Usage:
    cd ~/trading-bot
    venv/bin/python backtesting/w2_protrend_diagnostic.py          # arm C (default)
    venv/bin/python backtesting/w2_protrend_diagnostic.py --arm C
    venv/bin/python backtesting/w2_protrend_diagnostic.py --quiet  # suppress per-bar noise
"""
from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.oanda_backtest_v2 import (  # noqa: E402
    WATCHLIST,
    _resample_weekly,
    run_backtest,
)
from src.strategy.forex import strategy_config as _sc  # noqa: E402
from src.strategy.forex.pattern_detector import PatternDetector  # noqa: E402

# ── Window ────────────────────────────────────────────────────────────────────
W2_START = datetime(2025, 10, 1,  tzinfo=timezone.utc)
W2_END   = datetime(2026, 2, 28,  tzinfo=timezone.utc)
W2_WEEKS = (W2_END - W2_START).days / 7   # ≈ 21.3 weeks

# ── Required arm ─────────────────────────────────────────────────────────────
REQUIRED_ARM = "C"


# =============================================================================
# Parity snapshot + enforcement
# =============================================================================

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).parent.parent),
             "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _pairs_hash(pairs: list[str]) -> str:
    return hashlib.md5(",".join(sorted(pairs)).encode()).hexdigest()[:6]


def _config_snapshot(arm: str, protrend_only: bool) -> dict:
    """Capture every lever that must be identical between runs."""
    m = _sc  # live module reference
    return {
        "sha":              _git_sha(),
        "pairs":            sorted(WATCHLIST),
        "pairs_hash":       _pairs_hash(WATCHLIST),
        "arm":              arm,
        "min_rr":           m.MIN_RR,
        "min_rr_ct":        m.MIN_RR_COUNTERTREND,
        "risk_tiers":       str(getattr(m, "RISK_TIERS", "n/a")),
        "protrend_only":    protrend_only,               # ← intentionally varies
        "concurrency_bt":   m.MAX_CONCURRENT_TRADES_BACKTEST,
        "concurrency_live": m.MAX_CONCURRENT_TRADES_LIVE,
        "trigger_mode":     m.ENTRY_TRIGGER_MODE,
        "wk_cap_small":     m.MAX_TRADES_PER_WEEK_SMALL,
        "wk_cap_std":       m.MAX_TRADES_PER_WEEK_STANDARD,
        "no_sun":           m.NO_SUNDAY_TRADES_ENABLED,
        "no_thu_fri":       m.NO_THU_FRI_TRADES_ENABLED,
        "doji_gate":        m.INDECISION_FILTER_ENABLED,
        "atr_stop_mult":    m.ATR_STOP_MULTIPLIER,
        "atr_min_mult":     m.ATR_MIN_MULTIPLIER,
        "spread_model":     m.SPREAD_MODEL_ENABLED,
        "min_confidence":   m.MIN_CONFIDENCE,
    }


def _print_snapshot(snap: dict, label: str):
    W = 64
    print(f"┌─ Config snapshot: {label} {'─'*(W - len(label))}┐")
    print(f"│  SHA            : {snap['sha']:<46}│")
    print(f"│  Pairs hash     : {snap['pairs_hash']:<46}│")
    print(f"│  Arm            : {snap['arm']:<46}│")
    print(f"│  MIN_RR         : {snap['min_rr']:<46}│")
    print(f"│  MIN_RR_CT      : {snap['min_rr_ct']:<46}│")
    print(f"│  PROTREND_ONLY  : {str(snap['protrend_only']):<46}│")
    print(f"│  Concurrency BT : {snap['concurrency_bt']:<46}│")
    print(f"│  Trigger mode   : {snap['trigger_mode']:<46}│")
    print(f"│  Wk cap (sm/std): {snap['wk_cap_small']}/{snap['wk_cap_std']:<44}│")
    print(f"│  Time rules     : sun={snap['no_sun']}  thu_fri={snap['no_thu_fri']:<31}│")
    print(f"│  Doji gate      : {snap['doji_gate']:<46}│")
    print(f"│  Spread model   : {snap['spread_model']:<46}│")
    print(f"│  Candle cache   : ON (preloaded on run 2)             {'':>10}│")
    print(f"└{'─'*(W+2)}┘")


def _check_parity(s1: dict, s2: dict) -> list[str]:
    """Return list of unexpected differences (everything except protrend_only)."""
    skip = {"protrend_only"}
    diffs = []
    for k in s1:
        if k in skip:
            continue
        if s1[k] != s2[k]:
            diffs.append(f"  {k}: run1={s1[k]!r}  run2={s2[k]!r}")
    return diffs


# =============================================================================
# W==D calendar agreement
# =============================================================================

def _wd_calendar_agreement_pct(candle_data: dict, start_dt: datetime, end_dt: datetime) -> Optional[float]:
    """
    % of daily bars in [start_dt, end_dt] where weekly trend == daily trend.
    Uses the first pair alphabetically as representative.
    Trend detection mirrors set_and_forget.py: detect_trend(df_daily) for D,
    detect_trend(df_weekly) for W — both use the real D/W DataFrames.
    Returns None if candle data is unavailable.
    """
    rep_pair = sorted(candle_data.keys())[0]
    pdata    = candle_data[rep_pair]
    df_d_raw = pdata.get("d")
    if df_d_raw is None or df_d_raw.empty:
        return None

    # Normalise to tz-naive so all comparisons are consistent
    df_d_full = df_d_raw.copy()
    if df_d_full.index.tz is not None:
        df_d_full.index = df_d_full.index.tz_localize(None)

    # Resample full daily history to weekly once
    df_w_full = _resample_weekly(df_d_full)

    # Comparison bounds — always tz-naive
    start_naive = pd.Timestamp(start_dt).tz_localize(None) if start_dt.tzinfo else pd.Timestamp(start_dt)
    end_naive   = pd.Timestamp(end_dt).tz_localize(None)   if end_dt.tzinfo   else pd.Timestamp(end_dt)

    detector = PatternDetector(rep_pair)

    total = 0
    agree = 0

    # Walk daily bars inside the window
    d_indices = df_d_full.index[(df_d_full.index >= start_naive) &
                                 (df_d_full.index <  end_naive)]

    for ts in d_indices:
        i_d = df_d_full.index.get_loc(ts)
        if i_d < 50:
            continue  # need history for trend

        # Daily trend: last 50 daily bars
        d_slice = df_d_full.iloc[max(0, i_d - 50): i_d + 1]
        # Weekly trend: last 26 weekly bars (≈ 6 months of weeks)
        w_up_to = df_w_full[df_w_full.index <= ts]
        if len(w_up_to) < 10:
            continue
        w_slice = w_up_to.iloc[-26:]

        try:
            d_trend = detector.detect_trend(d_slice)
            w_trend = detector.detect_trend(w_slice)
        except Exception:
            continue

        total += 1

        # "Agree" = both directional and same direction
        d_bull = d_trend.value in ("bullish", "strong_bullish")
        d_bear = d_trend.value in ("bearish", "strong_bearish")
        w_bull = w_trend.value in ("bullish", "strong_bullish")
        w_bear = w_trend.value in ("bearish", "strong_bearish")

        if (d_bull and w_bull) or (d_bear and w_bear):
            agree += 1

    return (agree / total * 100) if total else None


# =============================================================================
# Stats helpers
# =============================================================================

def _wls(trades):
    w = sum(1 for t in trades if t["r"] >  0.05)
    l = sum(1 for t in trades if t["r"] < -0.05)
    s = len(trades) - w - l
    return w, l, s


def _r_pctiles(trades):
    """Realized R p50/p75/p90 from closed trades (may include scratches)."""
    rs = sorted(t["r"] for t in trades)
    if not rs:
        return None, None, None
    n = len(rs)
    p50 = rs[int(n * 0.50)]
    p75 = rs[int(n * 0.75)]
    p90 = rs[min(int(n * 0.90), n - 1)]
    return p50, p75, p90


def _planned_rr_pctiles(trades):
    """Planned exec_rr p50/p75/p90 (what the bot thought RR was at entry)."""
    rrs = sorted(t.get("planned_rr", 0) for t in trades if t.get("planned_rr", 0) > 0)
    if not rrs:
        return None, None, None
    n = len(rrs)
    p50 = rrs[int(n * 0.50)]
    p75 = rrs[int(n * 0.75)]
    p90 = rrs[min(int(n * 0.90), n - 1)]
    return p50, p75, p90


def _rr_min_blocks(gap_log) -> int:
    return sum(
        1 for g in gap_log
        if g.get("gap_type", "") in ("MIN_RR_ALIGN", "exec_rr_min")
    )


def _entry_hour_hist(trades) -> dict:
    counts: Counter = Counter()
    for t in trades:
        ts = t.get("entry_ts", "")
        if ts:
            try:
                counts[datetime.fromisoformat(ts).hour] += 1
            except (ValueError, TypeError):
                pass
    return dict(sorted(counts.items()))


# =============================================================================
# Capture helper
# =============================================================================

def _capture_run(quiet: bool, **kwargs):
    old = sys.stdout
    buf = io.StringIO()
    if quiet:
        sys.stdout = buf
    try:
        result = run_backtest(**kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue(), result


# =============================================================================
# Table printer
# =============================================================================

COL = 30

def _row(label: str, v1, v2, mark_diff: bool = True):
    s1, s2 = str(v1), str(v2)
    arrow = "►" if (mark_diff and s1 != s2) else " "
    print(f"  {label:<32}  {s1:<{COL}}  {arrow} {s2}")


def _sep():
    print(f"  {'─'*32}  {'─'*COL}  {'─'*COL}")


def _section(title: str):
    _sep()
    print(f"  {title}")


# =============================================================================
# Main
# =============================================================================

def run_diagnostic(arm: str = "C", quiet_runs: bool = False):

    if arm != REQUIRED_ARM:
        print(f"  ❌ ABORT: arm must be {REQUIRED_ARM}, got '{arm}'")
        sys.exit(1)

    print(f"\n{'═'*72}")
    print(f"  W2 PROTREND DIAGNOSTIC  (strict parity)")
    print(f"  Window  : Oct 1 2025 → Feb 28 2026  ({W2_WEEKS:.1f} weeks)")
    print(f"  Arm     : {arm}  |  Required: {REQUIRED_ARM}")
    print(f"  Pairs   : Alex 7  |  Trigger: engulf_only")
    print(f"  Candle  : shared between runs (0 extra API calls on run 2)")
    print(f"{'═'*72}\n")

    # ── Snapshot BEFORE each run ──────────────────────────────────────────────
    snap1 = _config_snapshot(arm=arm, protrend_only=False)
    print("  [Pre-run 1 config snapshot]")
    _print_snapshot(snap1, "RUN 1 — BASELINE")
    print()

    # ── Arm must be C ─────────────────────────────────────────────────────────
    if snap1["arm"] != REQUIRED_ARM:
        print(f"  ❌ ABORT: arm is '{snap1['arm']}', must be '{REQUIRED_ARM}'")
        sys.exit(1)

    # ── RUN 1: Baseline ───────────────────────────────────────────────────────
    print(f"{'━'*72}")
    print(f"  RUN 1 — BASELINE  (PROTREND_ONLY = False)")
    print(f"{'━'*72}\n")
    out1, r1 = _capture_run(
        quiet=quiet_runs,
        start_dt=W2_START,
        end_dt=W2_END,
        trail_arm_key=arm,
        use_cache=True,
        wd_protrend_htf=False,
        notes=f"W2-diag-baseline-arm{arm}",
    )
    if quiet_runs:
        print(out1)
    if r1 is None:
        print("  ❌ Run 1 failed — aborting.")
        sys.exit(1)

    # ── Snapshot BEFORE run 2 ─────────────────────────────────────────────────
    snap2 = _config_snapshot(arm=arm, protrend_only=True)
    print("\n  [Pre-run 2 config snapshot]")
    _print_snapshot(snap2, "RUN 2 — PROTREND_ONLY")
    print()

    # ── PARITY CHECK ─────────────────────────────────────────────────────────
    diffs = _check_parity(snap1, snap2)
    if diffs:
        print(f"  ❌ PARITY VIOLATION — config drifted between runs (aborting):")
        for d in diffs:
            print(d)
        sys.exit(1)
    print(f"  ✅ Parity OK — only PROTREND_ONLY differs between runs\n")

    # ── RUN 2: Protrend-only (shared candles) ─────────────────────────────────
    print(f"{'━'*72}")
    print(f"  RUN 2 — PROTREND_ONLY  (W+D must agree, 4H exempt)")
    print(f"{'━'*72}\n")
    out2, r2 = _capture_run(
        quiet=quiet_runs,
        start_dt=W2_START,
        end_dt=W2_END,
        trail_arm_key=arm,
        use_cache=True,
        wd_protrend_htf=True,
        preloaded_candle_data=r1.candle_data,
        notes=f"W2-diag-protrend-arm{arm}",
    )
    if quiet_runs:
        print(out2)
    if r2 is None:
        print("  ❌ Run 2 failed — aborting.")
        sys.exit(1)

    # ── Derived stats ─────────────────────────────────────────────────────────
    w1, l1, s1_ = _wls(r1.trades)
    w2, l2, s2_ = _wls(r2.trades)

    rp50_1, rp75_1, rp90_1 = _r_pctiles(r1.trades)
    rp50_2, rp75_2, rp90_2 = _r_pctiles(r2.trades)

    pp50_1, pp75_1, pp90_1 = _planned_rr_pctiles(r1.trades)
    pp50_2, pp75_2, pp90_2 = _planned_rr_pctiles(r2.trades)

    rr1 = _rr_min_blocks(r1.gap_log)
    rr2 = _rr_min_blocks(r2.gap_log)

    h1 = _entry_hour_hist(r1.trades)
    h2 = _entry_hour_hist(r2.trades)

    def _fmt_pct(v):
        return f"{v:.0f}%" if v is not None else "—"

    def _fmt_r(v):
        return f"{v:+.2f}R" if v is not None else "—"

    # ── WD calendar agreement ─────────────────────────────────────────────────
    print(f"\n  [Computing WD calendar agreement from candle data — may take ~10s]")
    wd_cal_pct = _wd_calendar_agreement_pct(r1.candle_data, W2_START, W2_END)

    # ── COMPARISON TABLE ──────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  REQUIRED METRICS TABLE — W2 Oct 2025 → Feb 2026")
    print(f"  Commit: {snap1['sha']}  |  Pairs: {snap1['pairs_hash']}  |  Arm: {arm}")
    print(f"  MIN_RR: {snap1['min_rr']}  |  Concurrency: {snap1['concurrency_bt']}  |  Risk tiers: hysteresis")
    print(f"{'═'*72}")
    _sep()
    print(f"  {'Metric':<32}  {'BASELINE (protrend=OFF)':<{COL}}    PROTREND_ONLY (W+D gate)")
    _sep()

    _section("── Core performance")
    _row("Mode",         "PROTREND_ONLY=False",     "PROTREND_ONLY=True")
    _row("Trades",       r1.n_trades,               r2.n_trades)
    _row("W / L / S",    f"{w1}W/{l1}L/{s1_}S",    f"{w2}W/{l2}L/{s2_}S")
    _row("AvgR",         _fmt_r(r1.avg_r),          _fmt_r(r2.avg_r))
    _row("BestR",        _fmt_r(r1.best_r),         _fmt_r(r2.best_r))
    _row("WorstR",       _fmt_r(r1.worst_r),        _fmt_r(r2.worst_r))
    _row("MaxDD",        f"{r1.max_dd_pct:.1f}%",   f"{r2.max_dd_pct:.1f}%")
    _row("Return%",      f"{r1.return_pct:+.1f}%",  f"{r2.return_pct:+.1f}%")
    _row("Trades/week",  f"{r1.n_trades/W2_WEEKS:.2f}", f"{r2.n_trades/W2_WEEKS:.2f}")

    _section("── Gate counters")
    _row("wd_htf_blocks",       r1.wd_htf_blocks,           r2.wd_htf_blocks)
    _row("time_blocks",         r1.time_blocks,              r2.time_blocks)
    _row("weekly_cap_blocks",   r1.weekly_limit_blocks,      r2.weekly_limit_blocks)
    _row("rr_min_blocks",       rr1,                         rr2)

    _section("── Additional diagnostics")
    _row("% entered trades W==D",
         _fmt_pct(r1.wd_alignment_pct),
         _fmt_pct(r2.wd_alignment_pct))
    _row("% calendar time W==D (D bars)",
         _fmt_pct(wd_cal_pct),
         "(same window — shared)")
    _row("atr_fallback %",
         _fmt_pct(r1.atr_fallback_pct),
         _fmt_pct(r2.atr_fallback_pct))
    _row("Stop pips p50",
         f"{r1.stop_pips_p50:.0f}p",
         f"{r2.stop_pips_p50:.0f}p" if r2.trades else "—")

    _section("── Realized R distribution (closed trades)")
    _row("R p50",  _fmt_r(rp50_1),  _fmt_r(rp50_2))
    _row("R p75",  _fmt_r(rp75_1),  _fmt_r(rp75_2))
    _row("R p90",  _fmt_r(rp90_1),  _fmt_r(rp90_2))

    _section("── Planned exec RR distribution (at entry)")
    _row("Planned RR p50",  f"{pp50_1:.2f}R" if pp50_1 else "—",  f"{pp50_2:.2f}R" if pp50_2 else "—")
    _row("Planned RR p75",  f"{pp75_1:.2f}R" if pp75_1 else "—",  f"{pp75_2:.2f}R" if pp75_2 else "—")
    _row("Planned RR p90",  f"{pp90_1:.2f}R" if pp90_1 else "—",  f"{pp90_2:.2f}R" if pp90_2 else "—")

    _section("── Entry hour distribution (UTC)")
    print(f"  {'BASELINE':<32}  {' '.join(f'{h:02d}h:{c}' for h,c in h1.items()) or '—'}")
    print(f"  {'PROTREND':<32}  {' '.join(f'{h:02d}h:{c}' for h,c in h2.items()) or '—'}")
    _sep()

    # ── INTERPRETATION ────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  INTERPRETATION")
    print(f"{'═'*72}")

    n_ok = 0
    checks = [
        ("fewer_trades",  r2.n_trades < r1.n_trades,
         f"Fewer trades: {r1.n_trades} → {r2.n_trades}"),
        ("higher_avg_r",  r2.avg_r > r1.avg_r,
         f"AvgR: {r1.avg_r:+.2f} → {r2.avg_r:+.2f}"),
        ("lower_max_dd",  r2.max_dd_pct < r1.max_dd_pct,
         f"MaxDD: {r1.max_dd_pct:.1f}% → {r2.max_dd_pct:.1f}%"),
        ("better_return", r2.return_pct > r1.return_pct,
         f"Return: {r1.return_pct:+.1f}% → {r2.return_pct:+.1f}%"),
    ]
    for _, passed, desc in checks:
        mark = "✅" if passed else "❌"
        print(f"  {mark} {desc}")
        if passed:
            n_ok += 1

    print()
    if r2.n_trades == 0:
        print("  ⛔ 0 trades in protrend run — bias computation broken or data missing.")
        print("     Investigate PatternDetector.detect_trend() for W2 daily/weekly data.")
    elif n_ok >= 3:
        print(f"  ✅ VERDICT ({n_ok}/4): W2 weakness is a REGIME ISSUE.")
        print("     Pro-trend-only is the correct LOW-mode entry gate.")
    elif n_ok == 2:
        print(f"  ⚠️  AMBIGUOUS ({n_ok}/4): inspect per-trade breakdown before deciding.")
        if wd_cal_pct is not None:
            if wd_cal_pct < 40:
                print(f"     Calendar W==D = {wd_cal_pct:.0f}% — market was genuinely mixed → regime likely.")
            else:
                print(f"     Calendar W==D = {wd_cal_pct:.0f}% — market mostly aligned → possible bias bug.")
    else:
        print(f"  ❌ VERDICT ({n_ok}/4): likely ENGINE BUG — gate made things worse.")
        print("     Audit trend detection for 2025–2026 before applying gate.")

    if wd_cal_pct is not None:
        print(f"\n  WD calendar insight: W and D trend agreed on {wd_cal_pct:.0f}% of daily bars.")
        if wd_cal_pct < 35:
            print("  → Market spent most of W2 in MIXED/COUNTER-TREND regime.")
            print("    Only 17% of bot entries were W+D aligned — consistent with this.")
            print("    PROTREND_ONLY doesn't help here: too few aligned setups existed.")
        elif wd_cal_pct >= 55:
            print("  → Market had MOSTLY ALIGNED W+D conditions in W2.")
            print("    17% WD alignment at entry may indicate a bias detection bug.")
        else:
            print("  → Market had MIXED conditions in W2 — genuine regime uncertainty.")

    print(f"\n{'═'*72}")
    print("  STOP — do not implement any change until results are interpreted.\n")

    return r1, r2


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="W2 Pro-Trend Diagnostic (strict parity enforced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arm", default=REQUIRED_ARM,
        help=f"Trail arm (must be {REQUIRED_ARM})",
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress per-bar backtest output (comparison table always printed)",
    )
    args = parser.parse_args()
    run_diagnostic(arm=args.arm.strip().upper(), quiet_runs=args.quiet)
