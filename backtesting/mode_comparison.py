#!/usr/bin/env python3
"""
Risk Mode Comparison: AUTO vs Pinned LOW/MEDIUM/HIGH
=====================================================
Runs W1 (Jul–Oct 2024) and W2 (Oct 2025–Feb 2026) under five mode configs:
  AUTO    — dynamic compute_risk_mode() per entry (production behaviour)
  LOW     — forced 0.5× for every entry
  MEDIUM  — forced 1.0× for every entry
  HIGH    — forced 1.5× for every entry (requires W==D in AUTO — here bypassed)

Candle data is fetched once per window and shared across all mode runs.

Primary questions:
  1. What is time_in_mode_pct + avgR per mode for W1 and W2 under AUTO?
  2. Does AUTO stay conservative (LOW/MEDIUM dominant) in W2?
  3. Does AUTO escalate (HIGH/EXTREME dominant) in W1?
  4. Does forcing HIGH in W2 produce worse results than AUTO?

Usage:
    cd ~/trading-bot
    venv/bin/python backtesting/mode_comparison.py          # both windows
    venv/bin/python backtesting/mode_comparison.py --window w1
    venv/bin/python backtesting/mode_comparison.py --window w2
"""
from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.oanda_backtest_v2 import run_backtest  # noqa: E402

# ── Windows ───────────────────────────────────────────────────────────────────
WINDOWS = {
    "W1": {
        "label":  "W1 Jul–Oct 2024  (Alex challenge window)",
        "start":  datetime(2024, 7,  1, tzinfo=timezone.utc),
        "end":    datetime(2024, 10, 31, tzinfo=timezone.utc),
    },
    "W2": {
        "label":  "W2 Oct 2025–Feb 2026",
        "start":  datetime(2025, 10, 1, tzinfo=timezone.utc),
        "end":    datetime(2026, 2,  28, tzinfo=timezone.utc),
    },
}

MODES_TO_RUN = ["AUTO", "LOW", "MEDIUM", "HIGH"]

ARM = "C"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _capture(quiet: bool, **kwargs):
    old = sys.stdout
    buf = io.StringIO()
    if quiet:
        sys.stdout = buf
    try:
        result = run_backtest(**kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue(), result


def _avgr_by_mode(trades: list) -> dict:
    """Dict {mode: avgR} grouped by risk_mode_at_entry."""
    bucket: dict = defaultdict(list)
    for t in trades:
        bucket[t.get("risk_mode_at_entry", "MEDIUM")].append(t["r"])
    return {m: sum(rs) / len(rs) for m, rs in bucket.items() if rs}


def _wl(trades):
    w = sum(1 for t in trades if t["r"] >  0.05)
    l = sum(1 for t in trades if t["r"] < -0.05)
    return w, l


# ── Table ─────────────────────────────────────────────────────────────────────

def _hdr_row():
    print(f"\n  {'Window':<6}  {'Mode':<8}  "
          f"{'Trades':>7}  {'W/L':>7}  "
          f"{'AvgR':>7}  {'MaxDD':>7}  {'Ret%':>8}  "
          f"{'tim_LOW':>8}  {'tim_MED':>8}  {'tim_HIGH':>9}  {'tim_EXT':>8}  "
          f"{'aR_LOW':>8}  {'aR_MED':>8}  {'aR_HIGH':>9}")
    print("  " + "─" * 145)


def _data_row(win_key: str, mode: str, r, avgr_by_mode: dict):
    if r is None:
        print(f"  {win_key:<6}  {mode:<8}  {'ERROR':>7}")
        return
    w, l = _wl(r.trades)
    print(
        f"  {win_key:<6}  {mode:<8}  "
        f"{r.n_trades:>7}  {w}W/{l}L  "
        f"{r.avg_r:>+6.2f}R  {r.max_dd_pct:>6.1f}%  {r.return_pct:>+7.1f}%  "
        f"{r.time_in_mode_pct_low:>7.0f}%  {r.time_in_mode_pct_medium:>7.0f}%  "
        f"{r.time_in_mode_pct_high:>8.0f}%  {r.time_in_mode_pct_extreme:>7.0f}%  "
        f"{avgr_by_mode.get('LOW',   float('nan')):>+7.2f}R  "
        f"{avgr_by_mode.get('MEDIUM',float('nan')):>+7.2f}R  "
        f"{avgr_by_mode.get('HIGH',  float('nan')):>+8.2f}R"
    )


def _mode_detail(win_key: str, mode: str, r, avgr_by_mode: dict):
    """Print per-mode breakdown for AUTO runs."""
    from backtesting.oanda_backtest_v2 import RISK_MODE_PARAMS
    print(f"\n  ── AUTO mode distribution ({win_key}) ─────────────────────────────────")
    print(f"    {'Mode':<10}  {'N':>4}  {'%entries':>9}  {'%h4time':>8}  "
          f"{'AvgR':>7}  {'BestR':>7}  {'WorstR':>7}")
    print(f"    {'─'*10}  {'─'*4}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}")
    tim = {
        "LOW":     r.time_in_mode_pct_low,
        "MEDIUM":  r.time_in_mode_pct_medium,
        "HIGH":    r.time_in_mode_pct_high,
        "EXTREME": r.time_in_mode_pct_extreme,
    }
    for m in ["LOW", "MEDIUM", "HIGH", "EXTREME"]:
        bucket = [t for t in r.trades if t.get("risk_mode_at_entry") == m]
        n_pct  = len(bucket) / r.n_trades * 100 if r.n_trades else 0
        avgr   = sum(t["r"] for t in bucket) / len(bucket) if bucket else float("nan")
        bestr  = max((t["r"] for t in bucket), default=float("nan"))
        worstr = min((t["r"] for t in bucket), default=float("nan"))
        mult   = RISK_MODE_PARAMS.get(m, {}).get("risk_mult", 1.0)
        n_str  = str(len(bucket)) if bucket else "—"
        print(
            f"    {m:<10}  {n_str:>4}  {n_pct:>8.0f}%  {tim[m]:>7.0f}%  "
            f"{avgr:>+6.2f}R  {bestr:>+6.2f}R  {worstr:>+6.2f}R  "
            f"[{mult}×]"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def run_comparison(windows_to_run: list[str], quiet_runs: bool = True):
    print(f"\n{'═'*80}")
    print(f"  RISK MODE COMPARISON — AUTO vs pinned LOW/MEDIUM/HIGH")
    print(f"  Arm: {ARM}  |  Pairs: Alex 7  |  Trigger: engulf_only")
    print(f"  Hypothesis: AUTO stays LOW/MEDIUM in W2, escalates in W1")
    print(f"{'═'*80}")

    all_results: dict = {}   # (win_key, mode) → BacktestResult

    for win_key in windows_to_run:
        cfg = WINDOWS[win_key]
        print(f"\n{'━'*80}")
        print(f"  {cfg['label']}")
        print(f"{'━'*80}")

        shared_candles: Optional[dict] = None

        for mode in MODES_TO_RUN:
            force = None if mode == "AUTO" else mode
            label = f"{win_key}:{mode}"
            print(f"\n  ▶ {label} {'(dynamic compute_risk_mode)' if not force else f'(FORCED={force})'}")

            out, result = _capture(
                quiet=quiet_runs,
                start_dt=cfg["start"],
                end_dt=cfg["end"],
                trail_arm_key=ARM,
                use_cache=True,
                force_risk_mode=force,
                preloaded_candle_data=shared_candles,
                notes=f"mode_cmp-{win_key}-{mode}",
            )

            if quiet_runs and out.strip():
                # Print just the RESULTS block, not per-bar noise
                lines = out.split("\n")
                in_results = False
                for ln in lines:
                    if "RESULTS — v2" in ln:
                        in_results = True
                    if in_results:
                        print("    " + ln)
                    if in_results and ln.startswith("  Return:"):
                        break

            if result is None:
                print(f"  ❌ {label} FAILED")
                all_results[(win_key, mode)] = None
                continue

            # Share candles after first fetch
            if shared_candles is None and result.candle_data:
                shared_candles = result.candle_data

            all_results[(win_key, mode)] = result

            if mode == "AUTO":
                _mode_detail(win_key, mode, result, _avgr_by_mode(result.trades))

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'═'*80}")
    print(f"  SUMMARY TABLE")
    print(f"  Columns: Trades | W/L | AvgR | MaxDD | Return% | "
          f"tim_LOW% | tim_MED% | tim_HIGH% | tim_EXT% | aR per mode")
    print(f"{'═'*80}")
    _hdr_row()

    for win_key in windows_to_run:
        first = True
        for mode in MODES_TO_RUN:
            r = all_results.get((win_key, mode))
            if first:
                print()
            first = False
            _data_row(win_key, mode, r, _avgr_by_mode(r.trades) if r else {})

    # ── Interpretation ────────────────────────────────────────────────────────
    print(f"\n\n{'═'*80}")
    print(f"  INTERPRETATION")
    print(f"{'═'*80}")

    for win_key in windows_to_run:
        r_auto = all_results.get((win_key, "AUTO"))
        r_low  = all_results.get((win_key, "LOW"))
        r_high = all_results.get((win_key, "HIGH"))
        if not r_auto:
            continue

        auto_dominant = "LOW" if r_auto.time_in_mode_pct_low > 35 else \
                        "HIGH/EXTREME" if (r_auto.time_in_mode_pct_high +
                                           r_auto.time_in_mode_pct_extreme) > 40 else "MEDIUM"

        print(f"\n  {win_key}: AUTO dominant mode = {auto_dominant}")
        print(f"     time_in_mode: LOW={r_auto.time_in_mode_pct_low:.0f}%  "
              f"MED={r_auto.time_in_mode_pct_medium:.0f}%  "
              f"HIGH={r_auto.time_in_mode_pct_high:.0f}%  "
              f"EXT={r_auto.time_in_mode_pct_extreme:.0f}%")

        if r_low and r_high:
            print(f"     AUTO return: {r_auto.return_pct:+.1f}%  "
                  f"vs LOW: {r_low.return_pct:+.1f}%  "
                  f"vs HIGH: {r_high.return_pct:+.1f}%")

        if win_key == "W2":
            if r_auto.time_in_mode_pct_low + r_auto.time_in_mode_pct_medium > 70:
                print(f"     ✅ AUTO correctly stays conservative in W2 (mixed regime)")
            else:
                print(f"     ⚠️  AUTO escalated in W2 — W==D gate may need tuning")
        elif win_key == "W1":
            if r_auto.time_in_mode_pct_high + r_auto.time_in_mode_pct_extreme > 30:
                print(f"     ✅ AUTO correctly escalates in W1 (trending regime)")
            else:
                print(f"     ⚠️  AUTO stayed conservative in W1 — escalation gate may be too tight")

    print(f"\n{'═'*80}\n")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Risk Mode Comparison: AUTO vs pinned LOW/MEDIUM/HIGH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--window", default="both",
        choices=["w1", "w2", "both"],
        help="Which window to run: w1 | w2 | both (default: both)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Print full per-bar backtest output (default: results summary only)",
    )
    args = parser.parse_args()

    _wins = (
        ["W1", "W2"] if args.window == "both"
        else ["W1"] if args.window == "w1"
        else ["W2"]
    )
    run_comparison(windows_to_run=_wins, quiet_runs=not args.verbose)
