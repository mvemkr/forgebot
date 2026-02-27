#!/usr/bin/env python3
"""
W2 Diagnostic: Pro-Trend-Only vs Baseline
==========================================
Goal: Determine whether W2 (Oct 2025–Feb 2026) weakness is a market regime
issue (expected — counter-trend setups just didn't materialise) or an engine/
backtest bug (unexpected — something is broken in bias computation or alignment).

Runs TWO backtests on the same W2 window sharing cached candle data:
  Run 1 — BASELINE       (PROTREND_ONLY = False, Alex's full style)
  Run 2 — PROTREND_ONLY  (wd_protrend_htf=True, W+D must agree, 4H exempt)

All other config is IDENTICAL: trail Arm C, hysteresis tiers, engulf_only,
time rules ON, weekly cap ON, doji gate ON, concurrency=1.

Interpretation:
  If PROTREND_ONLY → fewer trades + higher AvgR + lower MaxDD + better return
    → W2 weakness is a REGIME ISSUE — protrend gate is correct LOW-mode behaviour.
  If PROTREND_ONLY → 0 trades or worse
    → Investigate bias computation or timeframe alignment before applying gate.

Usage:
    cd ~/trading-bot
    venv/bin/python backtesting/w2_protrend_diagnostic.py
    venv/bin/python backtesting/w2_protrend_diagnostic.py --arm C
    venv/bin/python backtesting/w2_protrend_diagnostic.py --quiet  # no per-trade noise
"""
from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.oanda_backtest_v2 import run_backtest  # noqa: E402

# ── Window constants ──────────────────────────────────────────────────────────
W2_START = datetime(2025, 10, 1,  tzinfo=timezone.utc)
W2_END   = datetime(2026, 2, 28,  tzinfo=timezone.utc)
W2_WEEKS = (W2_END - W2_START).days / 7   # ≈ 21.3 weeks

COL_W = 36   # width of each data column in comparison table


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _capture_run(quiet: bool, **kwargs):
    """
    Run run_backtest(**kwargs).  Returns (stdout_text, result).
    If quiet=True the per-bar noise is swallowed; the comparison table is always
    printed by the script itself so nothing is lost.
    """
    old = sys.stdout
    buf = io.StringIO()
    if quiet:
        sys.stdout = buf
    try:
        result = run_backtest(**kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue(), result


def _wls(trades):
    """(wins, losses, scratches) from trade list.  Scratch = |r| <= 0.05R."""
    w = sum(1 for t in trades if t["r"] >  0.05)
    l = sum(1 for t in trades if t["r"] < -0.05)
    s = len(trades) - w - l
    return w, l, s


def _entry_hour_hist(trades) -> dict:
    """UTC entry hour → trade count, sorted."""
    counts: Counter = Counter()
    for t in trades:
        ts = t.get("entry_ts", "")
        if ts:
            try:
                counts[datetime.fromisoformat(ts).hour] += 1
            except (ValueError, TypeError):
                pass
    return dict(sorted(counts.items()))


def _rr_min_blocks(gap_log) -> int:
    """Count entries blocked by the MIN_RR_ALIGN gate (exec_rr_min shortfall)."""
    return sum(
        1 for g in gap_log
        if g.get("gap_type", "") in ("MIN_RR_ALIGN", "exec_rr_min")
        or "exec_rr_min" in str(g.get("failed_filters", ""))
    )


def _hist_str(hist: dict) -> str:
    if not hist:
        return "—"
    return "  ".join(f"{h:02d}h:{c}" for h, c in sorted(hist.items()))


def _stop_dist(stop_type_counts: dict) -> str:
    if not stop_type_counts:
        return "—"
    total = sum(stop_type_counts.values())
    parts = []
    for k in ("structure", "neckline_retest_swing", "shoulder_anchor",
              "retest_swing", "structural_anchor", "atr_fallback", "unknown"):
        n = stop_type_counts.get(k, 0)
        if n:
            parts.append(f"{k}:{n} ({n/total*100:.0f}%)")
    # catch-all for any unexpected keys
    shown = set(("structure", "neckline_retest_swing", "shoulder_anchor",
                 "retest_swing", "structural_anchor", "atr_fallback", "unknown"))
    for k, n in stop_type_counts.items():
        if k not in shown and n:
            parts.append(f"{k}:{n} ({n/total*100:.0f}%)")
    return "  ".join(parts) if parts else "—"


# ─────────────────────────────────────────────────────────────────────────────
# Table printer
# ─────────────────────────────────────────────────────────────────────────────

def _hdr(label: str, b_val: str, p_val: str):
    print(f"  {'─'*28}  {'─'*COL_W}  {'─'*COL_W}")
    print(f"  {label:<28}  {b_val:<{COL_W}}  {p_val}")


def _row(label: str, b_val, p_val, bold_diff: bool = False):
    bv = str(b_val)
    pv = str(p_val)
    if bold_diff and bv != pv:
        pv = f"► {pv}"   # highlight changes in right column
    print(f"  {label:<28}  {bv:<{COL_W}}  {pv}")


def _sep():
    print(f"  {'─'*28}  {'─'*COL_W}  {'─'*COL_W}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic(arm: str = "C", quiet_runs: bool = False):
    print(f"\n{'═'*72}")
    print(f"  W2 PROTREND DIAGNOSTIC")
    print(f"  Window : Oct 1 2025 → Feb 28 2026  ({W2_WEEKS:.1f} weeks)")
    print(f"  Arm    : {arm}  |  Pairs: Alex 7 (89a6ef)  |  Trigger: engulf_only")
    print(f"  Gate   : time=ON  weekly-cap=ON  doji=ON  concurrency=1")
    print(f"{'═'*72}\n")

    # ── RUN 1: Baseline ───────────────────────────────────────────────────────
    print(f"{'━'*72}")
    print(f"  RUN 1 — BASELINE  (PROTREND_ONLY = False, Alex full style)")
    print(f"{'━'*72}\n")
    out1, r1 = _capture_run(
        quiet=quiet_runs,
        start_dt=W2_START,
        end_dt=W2_END,
        trail_arm_key=arm,
        use_cache=True,
        wd_protrend_htf=False,
        notes="W2-diagnostic-baseline",
    )
    if quiet_runs:
        print(out1)   # replay captured output

    if r1 is None:
        print("  ❌ Run 1 failed — cannot continue.")
        return None, None

    # ── RUN 2: Protrend-only  (share candle data — no extra API calls) ────────
    print(f"\n{'━'*72}")
    print(f"  RUN 2 — PROTREND_ONLY  (W+D must agree, 4H exempt)")
    print(f"{'━'*72}\n")
    out2, r2 = _capture_run(
        quiet=quiet_runs,
        start_dt=W2_START,
        end_dt=W2_END,
        trail_arm_key=arm,
        use_cache=True,
        wd_protrend_htf=True,
        preloaded_candle_data=r1.candle_data,   # ← shared, 0 extra API calls
        notes="W2-diagnostic-protrend-only",
    )
    if quiet_runs:
        print(out2)

    if r2 is None:
        print("  ❌ Run 2 failed — cannot continue.")
        return r1, None

    # ── Derived stats ─────────────────────────────────────────────────────────
    w1, l1, s1 = _wls(r1.trades)
    w2, l2, s2 = _wls(r2.trades)
    h1 = _entry_hour_hist(r1.trades)
    h2 = _entry_hour_hist(r2.trades)
    rr1 = _rr_min_blocks(r1.gap_log)
    rr2 = _rr_min_blocks(r2.gap_log)

    trades_pw1 = r1.n_trades / W2_WEEKS if W2_WEEKS else 0
    trades_pw2 = r2.n_trades / W2_WEEKS if W2_WEEKS else 0

    # ── COMPARISON TABLE ──────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  COMPARISON TABLE — W2 Oct 2025 → Feb 2026")
    print(f"{'═'*72}")
    print(f"  {'Metric':<28}  {'BASELINE (protrend=OFF)':<{COL_W}}  PROTREND_ONLY (W+D gate)")
    _sep()

    # Performance
    _hdr("── Performance", "", "")
    _row("Trades",            r1.n_trades,                  r2.n_trades,          bold_diff=True)
    _row("W / L / S",         f"{w1}W / {l1}L / {s1}S",   f"{w2}W / {l2}L / {s2}S", bold_diff=True)
    _row("Win rate",           f"{r1.win_rate*100:.0f}%",   f"{r2.win_rate*100:.0f}%", bold_diff=True)
    _row("Avg R",              f"{r1.avg_r:+.2f}R",         f"{r2.avg_r:+.2f}R",  bold_diff=True)
    _row("Best R",             f"{r1.best_r:+.2f}R",        f"{r2.best_r:+.2f}R", bold_diff=True)
    _row("Worst R",            f"{r1.worst_r:+.2f}R",       f"{r2.worst_r:+.2f}R",bold_diff=True)
    _row("Max DD",             f"{r1.max_dd_pct:.1f}%",     f"{r2.max_dd_pct:.1f}%", bold_diff=True)
    _row("Return %",           f"{r1.return_pct:+.1f}%",    f"{r2.return_pct:+.1f}%", bold_diff=True)
    _row("Trades / week",      f"{trades_pw1:.2f}",          f"{trades_pw2:.2f}",  bold_diff=True)
    _row("WD alignment %",     f"{r1.wd_alignment_pct:.0f}%",f"{r2.wd_alignment_pct:.0f}%")

    # Stop quality
    _hdr("── Stop distribution", "", "")
    _row("atr_fallback %",     f"{r1.atr_fallback_pct:.0f}%", f"{r2.atr_fallback_pct:.0f}%", bold_diff=True)
    _row("Stop pips p50",      f"{r1.stop_pips_p50:.0f}p",  f"{r2.stop_pips_p50:.0f}p")
    for stype in ("structure", "neckline_retest_swing", "shoulder_anchor",
                  "retest_swing", "structural_anchor", "atr_fallback", "unknown"):
        n1 = r1.stop_type_counts.get(stype, 0)
        n2 = r2.stop_type_counts.get(stype, 0)
        if n1 or n2:
            _row(f"  {stype[:26]}", n1, n2, bold_diff=True)

    # Gate counters
    _hdr("── Gate counters", "", "")
    _row("WD_HTF blocks",      r1.wd_htf_blocks,     r2.wd_htf_blocks,     bold_diff=True)
    _row("Time blocks",        r1.time_blocks,        r2.time_blocks)
    _row("Weekly cap blocks",  r1.weekly_limit_blocks,r2.weekly_limit_blocks)
    _row("RR_min blocks",      rr1,                   rr2,                  bold_diff=True)
    _row("Countertrend blocks",r1.countertrend_htf_blocks, r2.countertrend_htf_blocks, bold_diff=True)

    # Entry hour histogram
    _sep()
    print(f"  {'Entry hour (UTC)':<28}  {_hist_str(h1)}")
    print(f"  {'':28}  {'PROTREND ↓':}")
    print(f"  {'':28}  {_hist_str(h2)}")

    # ── INTERPRETATION ────────────────────────────────────────────────────────
    _sep()
    print(f"\n  INTERPRETATION")
    print(f"  {'─'*68}")

    criteria = {
        "fewer_trades":  r2.n_trades < r1.n_trades,
        "higher_avg_r":  r2.avg_r    > r1.avg_r,
        "lower_max_dd":  r2.max_dd_pct < r1.max_dd_pct,
        "better_return": r2.return_pct > r1.return_pct,
    }
    labels = {
        "fewer_trades":  f"Fewer trades ({r1.n_trades} → {r2.n_trades})",
        "higher_avg_r":  f"Higher AvgR ({r1.avg_r:+.2f} → {r2.avg_r:+.2f})",
        "lower_max_dd":  f"Lower MaxDD ({r1.max_dd_pct:.1f}% → {r2.max_dd_pct:.1f}%)",
        "better_return": f"Better return ({r1.return_pct:+.1f}% → {r2.return_pct:+.1f}%)",
    }
    for key, passed in criteria.items():
        mark = "✅" if passed else "❌"
        print(f"  {mark} {labels[key]}")

    passing = sum(criteria.values())
    print()

    if r2.n_trades == 0:
        verdict = (
            "⛔ PROTREND_ONLY produced 0 trades.\n"
            "     Bias computation is likely broken or trend data missing for 2025–2026.\n"
            "     Investigate PatternDetector.detect_trend() on H4/D/W for this window\n"
            "     before applying any gate."
        )
    elif passing >= 3:
        verdict = (
            f"✅ VERDICT ({passing}/4 criteria met): W2 weakness is a REGIME ISSUE.\n"
            f"     Pro-trend-only is the correct LOW-mode entry gate — apply it.\n"
            f"     Counter-trend setups in W2 lacked edge; restricting to W+D aligned\n"
            f"     trades reduces noise and improves quality."
        )
    elif passing == 2:
        verdict = (
            f"⚠️  AMBIGUOUS ({passing}/4 criteria met): mixed signal.\n"
            f"     Possible partial regime + partial bug.\n"
            f"     Inspect per-trade breakdown for W2 counter-trend losses before deciding."
        )
    else:
        verdict = (
            f"❌ VERDICT ({passing}/4 criteria met): likely ENGINE/BACKTEST BUG.\n"
            f"     W+D gate made things worse — the bias signal may be inverted or\n"
            f"     timeframe alignment is off for the 2025–2026 data.\n"
            f"     Do NOT apply PROTREND_ONLY until alignment code is audited."
        )

    for line in verdict.split("\n"):
        print(f"  {line}")

    print(f"\n{'═'*72}\n")
    return r1, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="W2 Pro-Trend-Only vs Baseline Diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--arm", default="C",
        help="Trail arm A|B|C (default: C — production Arm C)",
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress verbose per-trade output from each run "
             "(comparison table always printed)",
    )
    args = parser.parse_args()

    _arm = args.arm.strip().upper()
    if _arm not in ("A", "B", "C"):
        parser.error(f"--arm must be A, B, or C — got '{args.arm}'")

    run_diagnostic(arm=_arm, quiet_runs=args.quiet)
