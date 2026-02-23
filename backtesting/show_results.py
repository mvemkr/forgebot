#!/usr/bin/env python3
"""
show_results.py — Print backtest result history from logs/backtest_results.jsonl

Usage:
  python3 backtesting/show_results.py              # all results
  python3 backtesting/show_results.py --last 10    # most recent N
  python3 backtesting/show_results.py --window alex  # filter by window keyword
  python3 backtesting/show_results.py --window jan
  python3 backtesting/show_results.py --commit 2d980a4  # filter by commit
"""

import json
import argparse
from pathlib import Path

LOG = Path(__file__).parent.parent / "logs" / "backtest_results.jsonl"

def load(window_filter=None, commit_filter=None, last=None):
    if not LOG.exists():
        print(f"No results log found at {LOG}")
        return []
    with open(LOG) as f:
        records = [json.loads(l) for l in f if l.strip()]
    if window_filter:
        kw = window_filter.lower()
        records = [r for r in records
                   if kw in r["window_start"] or kw in r["window_end"]
                   or ("alex" in kw and "2024" in r["window_start"])
                   or ("jan" in kw and "2026-01" in r["window_start"])]
    if commit_filter:
        records = [r for r in records if commit_filter in r["commit"]]
    if last:
        records = records[-last:]
    return records

def fmt(r):
    res = r["results"]
    cfg = r["config"]
    win = r["window_start"][:10] + " → " + r["window_end"][:10]
    pnl_sign = "+" if res["net_pnl"] >= 0 else ""
    ret_sign = "+" if res["return_pct"] >= 0 else ""
    status = "✅" if res["return_pct"] >= 0 else "❌"
    gates = []
    if cfg.get("BLOCK_ENTRY_WHILE_WINNER_RUNNING"):
        gates.append(f"winner>{cfg.get('WINNER_THRESHOLD_R', '?')}R")
    if not gates:
        gates.append("no-gate")
    gate_str = ", ".join(gates)

    lines = [
        f"\n{'─'*70}",
        f"  {status}  {r['run_dt'][:16]}  commit={r['commit']}",
        f"     Window:  {win}",
        f"     Return:  {ret_sign}{res['return_pct']:.1f}%   "
        f"Net P&L: {pnl_sign}${res['net_pnl']:,.2f}   "
        f"Final: ${res['final_bal']:,.2f}",
        f"     Trades:  {res['n_trades']}  "
        f"({res['n_wins']}W / {res['n_losses']}L)  "
        f"WR={res['win_rate']:.0f}%",
        f"     Config:  ATR_MIN={cfg['ATR_MIN_MULTIPLIER']}  "
        f"ATR_MAX={cfg['ATR_STOP_MULTIPLIER']}  "
        f"MIN_CONF={cfg['MIN_CONFIDENCE']}  "
        f"gate={gate_str}",
    ]

    if r.get("trades"):
        lines.append(f"     Trades:")
        for t in r["trades"]:
            s = "✅" if t["pnl"] >= 0 else "❌"
            pnl_s = "+" if t["pnl"] >= 0 else ""
            theme = f"  🎯 {t['macro_theme']}" if t.get("macro_theme") else ""
            lines.append(
                f"       {s} {t['pair']:<10} {t['direction']:<6} "
                f"entry={t['entry']:.5f}  "
                f"{'+' if t['r'] >= 0 else ''}{t['r']:.1f}R  "
                f"{pnl_s}${t['pnl']:,.2f}  [{t['reason']}]  "
                f"{t.get('pattern','')[:30]}{theme}"
            )

    if r.get("gap_summary"):
        gaps = {k: v for k, v in r["gap_summary"].items()
                if k not in ("news_filter_skipped",) and v > 0}
        if gaps:
            lines.append(f"     Gaps:    " + "  ".join(f"{k}={v}" for k, v in sorted(gaps.items())))

    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--last", type=int, help="Show last N results")
    p.add_argument("--window", type=str, help="Filter by window keyword (alex, jan, 2024, etc.)")
    p.add_argument("--commit", type=str, help="Filter by commit hash substring")
    args = p.parse_args()

    records = load(window_filter=args.window, commit_filter=args.commit, last=args.last)

    if not records:
        print("No matching results.")
        return

    print(f"\n{'═'*70}")
    print(f"  BACKTEST HISTORY  ({len(records)} run{'s' if len(records)!=1 else ''})")
    print(f"{'═'*70}")
    for r in records:
        print(fmt(r))
    print(f"\n{'─'*70}")

if __name__ == "__main__":
    main()
