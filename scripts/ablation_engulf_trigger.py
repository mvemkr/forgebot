#!/usr/bin/env python3
"""
Engulfing confirmation timing ablation study — offline replay only.

Variants
--------
A  engulf_only          lb=2   Baseline — current production (engulf in last 2 bars)
B  engulf+reject        lb=2   Engulf OR strict-rejection candle (same window)
C  engulf_only          lb=3   Engulf within last 3 bars (extended window)
D  engulf+reject        lb=3   Broadest — rejection OR engulf within 3 bars

"Rejection candle" = engulf_or_strict_pin_at_level mode:
  strict hammer/shooting star: wick ≥ 3× body, body ≤ 25% range,
  opposite wick ≤ 10% range, close in outer 35%, level touched recently.

Same window / pairs as prior ablation studies:
  Feb 01 – Mar 04 2026, 7 Alex pairs, $8,000 capital, H1 cadence.

Safety
------
  atexit() resets ENTRY_TRIGGER_MODE and ENGULF_CONFIRM_LOOKBACK_BARS to
  production defaults even on crash.
"""

from __future__ import annotations

import atexit
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

import src.strategy.forex.strategy_config as _sc

# ── atexit guard ──────────────────────────────────────────────────────────────
_ORIG_TRIGGER_MODE = getattr(_sc, "ENTRY_TRIGGER_MODE",         "engulf_only")
_ORIG_LB           = getattr(_sc, "ENGULF_CONFIRM_LOOKBACK_BARS", 2)

def _reset_engulf_config():
    _sc.ENTRY_TRIGGER_MODE          = _ORIG_TRIGGER_MODE
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS = _ORIG_LB

atexit.register(_reset_engulf_config)

# ── backtester import ─────────────────────────────────────────────────────────
from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult   # noqa

# ──────────────────────────────────────────────────────────────────────────────
WINDOW_START = datetime(2026, 2, 1, tzinfo=timezone.utc)
WINDOW_END   = datetime(2026, 3, 4, tzinfo=timezone.utc)
CAPITAL      = 8_000.0
REPORT_PATH  = REPO / "backtesting/results/ablation_engulf_trigger.md"

# (label, trigger_mode, lookback, short_desc)
VARIANTS: List[Tuple[str, str, int, str]] = [
    ("A", "engulf_only",                   2, "Baseline — engulf only, lb=2 (current production)"),
    ("B", "engulf_or_strict_pin_at_level", 2, "Engulf OR strict-rejection candle, lb=2"),
    ("C", "engulf_only",                   3, "Engulf only, lb=3 (+1 bar window)"),
    ("D", "engulf_or_strict_pin_at_level", 3, "Engulf OR strict-rejection, lb=3 (broadest)"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _trade_key(t: Dict) -> str:
    ts = t.get("entry_ts") or t.get("open_ts") or ""
    if hasattr(ts, "strftime"):
        ts = ts.strftime("%Y%m%d%H")
    else:
        ts = str(ts)[:13].replace("-","").replace("T","").replace(":","").replace(" ","")
    return f"{t.get('pair','?')}|{ts}"


def _r(t: Dict) -> float:
    for k in ("r", "realised_r", "result_r"):
        v = t.get(k)
        if v is not None:
            try: return float(v)
            except (TypeError, ValueError): pass
    return 0.0


def _is_win(t: Dict) -> bool:
    return _r(t) > 0.0


def _session(ts_str: str) -> str:
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        h  = dt.hour
        if 8  <= h < 12: return "London"
        if 12 <= h < 13: return "London_NY_Overlap"
        if 13 <= h < 17: return "NY"
        return "off_session"
    except Exception:
        return "unknown"


def _week(ts_str: str) -> str:
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        if dt <= datetime(2026, 2, 14, tzinfo=timezone.utc): return "W1 (Feb 1–14)"
        if dt <= datetime(2026, 2, 28, tzinfo=timezone.utc): return "W2 (Feb 15–28)"
        return "Live-parity (Mar 1–4)"
    except Exception:
        return "unknown"


def _pct(v: float) -> str:
    return f"{v:+.1f}%" if v != 0 else "0%"


def _rs(v: float) -> str:
    return f"{v:+.2f}R"


def _wr(trades: List[Dict]) -> str:
    if not trades: return "—"
    wins = sum(1 for t in trades if _is_win(t))
    return f"{int(wins/len(trades)*100)}%"


def _avg_r(trades: List[Dict]) -> str:
    if not trades: return "—"
    return _rs(sum(_r(t) for t in trades) / len(trades))


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_variant(
    label: str,
    trigger_mode: str,
    lookback: int,
    desc: str,
    preloaded: Optional[Dict] = None,
    verbose: bool = False,
) -> Tuple[BacktestResult, Optional[Dict]]:
    print(f"\n{'─'*70}")
    print(f"  Variant {label}: {desc}")
    print(f"  trigger_mode = {trigger_mode!r}  |  lookback = {lookback}")
    print(f"{'─'*70}\n")

    _sc.ENTRY_TRIGGER_MODE          = trigger_mode
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS = lookback

    t0 = time.time()
    result = run_backtest(
        start_dt              = WINDOW_START,
        end_dt                = WINDOW_END,
        starting_bal          = CAPITAL,
        notes                 = f"ablation_engulf_{label}",
        trail_arm_key         = f"engulf_{label}",
        preloaded_candle_data = preloaded,
        use_cache             = True,
        quiet                 = not verbose,
    )
    elapsed = time.time() - t0

    # Reset to production defaults after each variant
    _sc.ENTRY_TRIGGER_MODE          = _ORIG_TRIGGER_MODE
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS = _ORIG_LB

    print(f"\n  ✓ {result.n_trades} trades | ret={result.return_pct:+.1f}% | "
          f"WR={getattr(result,'win_rate',0.0):.0%} | "
          f"maxDD={result.max_dd_pct:.1f}% | {elapsed:.1f}s")

    return result, getattr(result, "candle_data", None)


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def build_report(results: Dict[str, BacktestResult]) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ws = WINDOW_START.strftime("%Y-%m-%d")
    we = WINDOW_END.strftime("%Y-%m-%d")
    lines: List[str] = []
    a = lines.extend

    def _m(r: BacktestResult):
        n    = r.n_trades
        wr   = int(round(getattr(r, "win_rate", 0.0) * 100))
        avg  = getattr(r, "avg_r", 0.0) or 0.0
        best = getattr(r, "best_r",  0.0) or 0.0
        worst= getattr(r, "worst_r", 0.0) or 0.0
        rs   = [_r(t) for t in r.trades]
        w3r  = sum(sorted(rs)[:3])
        return n, wr, avg, r.return_pct, r.max_dd_pct, w3r, best, worst

    ra = results["A"]
    na, wra, avga, reta, dda, w3ra, besta, worsta = _m(ra)
    baseline_keys = {_trade_key(t) for t in ra.trades}

    # ── header ────────────────────────────────────────────────────────────────
    a([
        f"# Engulfing Confirmation Timing Ablation Study",
        f"Window: {ws} → {we} | Pairs: 7 (Alex universe)",
        f"Generated: {now_utc}",
        "",
        "## Variant Definitions",
        "",
        "| Label | Trigger mode | Lookback | Description |",
        "|-------|-------------|---------|-------------|",
    ])
    for lbl, tm, lb, desc in VARIANTS:
        a([f"| {lbl} | `{tm}` | {lb} | {desc} |"])
    a([
        "",
        "**Rejection candle** (`engulf_or_strict_pin_at_level`):",
        "- Strict hammer / shooting star: wick ≥ 3× body, body ≤ 25% range,",
        "  opposite wick ≤ 10% range, close in outer 35%",
        "- Requires level to have been touched within last 5 H1 bars",
        "- At-level check uses neckline as the zone anchor",
        "",
        "**Extended lookback (lb=3):** EntrySignalDetector scans last 3 H1 bars",
        "(vs lb=2 baseline). If any bar in the window is a valid trigger, entry fires.",
        "",
        "---",
        "",
        "## Primary Metrics",
        "",
    ])

    rows = []
    for lbl, _, _, _ in VARIANTS:
        r = results[lbl]
        n, wr, avg, ret, dd, w3r, best, worst = _m(r)
        rows.append((lbl, n, wr, avg, ret, dd, w3r, best, worst))

    # Build table
    header = "| Metric                   |" + "".join(f" {'Var '+r[0]:>20} |" for r in rows)
    sep    = "|" + "---|" * (len(rows) + 1)
    a([header, sep])
    a([f"| Total trades             |" + "".join(f" {r[1]:>20} |" for r in rows)])
    a([f"| Win rate                 |" + "".join(f" {r[2]:>19}% |" for r in rows)])
    a([f"| Average R                |" + "".join(f" {_rs(r[3]):>20} |" for r in rows)])
    a([f"| Return %                 |" + "".join(f" {r[4]:>19.1f}% |" for r in rows)])
    a([f"| Max drawdown %           |" + "".join(f" {r[5]:>19.1f}% |" for r in rows)])
    a([f"| Worst 3-loss cluster R   |" + "".join(f" {_rs(r[6]):>20} |" for r in rows)])
    a([f"| Best trade R             |" + "".join(f" {_rs(r[7]):>20} |" for r in rows)])
    a([f"| Worst trade R            |" + "".join(f" {_rs(r[8]):>20} |" for r in rows)])
    a([f"| Δ trades vs A            |" + "".join(f" {r[1]-na:>+20} |" for r in rows)])
    a([f"| Δ return vs A            |" + "".join(f" {_pct(r[4]-reta):>20} |" for r in rows)])
    a([f"| Δ max DD vs A            |" + "".join(f" {_pct(r[5]-dda):>20} |" for r in rows)])
    a([""])

    # ── per-variant unlock detail ─────────────────────────────────────────────
    a(["---", "", "## Unlock Analysis", ""])
    for lbl, tm, lb, desc in VARIANTS[1:]:
        rv = results[lbl]
        ul = [t for t in rv.trades if _trade_key(t) not in baseline_keys]
        lo = [t for t in ra.trades if _trade_key(t) not in {_trade_key(u) for u in rv.trades}]

        a([f"### Variant {lbl} vs A — {desc}", ""])
        if not ul and not lo:
            a(["*No change to trade list.*", ""])
            continue

        if ul:
            a([f"**{len(ul)} newly unlocked trade(s):**", ""])
            a(["| # | Pair | Dir | Entry | Pattern | Trigger | R | Exit |",
               "|---|------|-----|-------|---------|---------|---|------|"])
            for i, t in enumerate(ul, 1):
                ts_s = str(t.get("entry_ts", t.get("entry_time","")) )[:16]
                pat  = str(t.get("pattern",""))[:28]
                sig  = str(t.get("entry_signal", t.get("signal_type", "")))[:20]
                a([f"| {i} | {t.get('pair','')} | {t.get('direction','')} | {ts_s} "
                   f"| {pat} | {sig} | {_rs(_r(t))} | {t.get('exit_reason','')} |"])
            a([""])

        if lo:
            a([f"**{len(lo)} trade(s) present in A but absent in {lbl}:**", ""])
            for t in lo:
                ts_s = str(t.get("entry_ts",""))[:16]
                a([f"- {t.get('pair','')} {t.get('direction','')} @ {ts_s}  R={_rs(_r(t))}"])
            a([""])

    # ── aggregate unlock summary ──────────────────────────────────────────────
    a(["---", "", "## Aggregate Unlock Summary", ""])
    a(["| Metric                          | B vs A | C vs A | D vs A |",
       "|---------------------------------|-------:|-------:|-------:|"])
    for metric_name, fn in [
        ("Newly unlocked entries",         lambda ul, lo: len(ul)),
        ("Unlocked wins",                  lambda ul, lo: sum(1 for t in ul if _is_win(t))),
        ("Unlocked losses",                lambda ul, lo: sum(1 for t in ul if not _is_win(t))),
        ("Win rate unlocked",              lambda ul, lo: _wr(ul)),
        ("Avg R unlocked",                 lambda ul, lo: _avg_r(ul)),
        ("Gate-protected losses (A only)", lambda ul, lo: sum(1 for t in lo if not _is_win(t))),
    ]:
        cells = []
        for var_lbl in ("B", "C", "D"):
            rv = results[var_lbl]; vk = {_trade_key(t) for t in rv.trades}
            ul = [t for t in rv.trades if _trade_key(t) not in baseline_keys]
            lo = [t for t in ra.trades if _trade_key(t) not in vk]
            cells.append(str(fn(ul, lo)))
        a([f"| {metric_name:<31} | {cells[0]:>6} | {cells[1]:>6} | {cells[2]:>6} |"])
    a([""])

    # ── MAE / MFE for unlocked trades ─────────────────────────────────────────
    a(["---", "", "## MAE / MFE — Unlocked Trades", ""])
    for lbl in ("B", "C", "D"):
        rv = results[lbl]
        ul = [t for t in rv.trades if _trade_key(t) not in baseline_keys]
        a([f"**Variant {lbl}** ({len(ul)} unlocked):"])
        if not ul:
            a(["*(none)*", ""]); continue
        mfes = [float(t.get("mfe_r", 0.0)) for t in ul]
        maes = [float(t.get("mae_r", 0.0)) for t in ul]
        a([f"- Avg MFE: {sum(mfes)/len(mfes):+.2f}R  "
           f"(min {min(mfes):+.2f}R, max {max(mfes):+.2f}R)"])
        a([f"- Avg MAE: {sum(maes)/len(maes):+.2f}R  "
           f"(worst {min(maes):+.2f}R)"])
        a([""])

    # ── pattern breakdown of unlocked entries ─────────────────────────────────
    a(["---", "", "## Pattern Breakdown — Unlocked Entries (B+C+D combined, de-duped)", ""])
    seen: set = set()
    all_ul_dedup = []
    for lbl in ("B", "C", "D"):
        for t in results[lbl].trades:
            k = _trade_key(t)
            if k not in baseline_keys and k not in seen:
                seen.add(k)
                all_ul_dedup.append({**t, "_first_unlock": lbl})

    if not all_ul_dedup:
        a(["*(no unlocked trades)*", ""])
    else:
        # Pattern counts
        by_pat: Dict[str, list] = {}
        for t in all_ul_dedup:
            p = str(t.get("pattern", "unknown"))
            by_pat.setdefault(p, []).append(t)
        a(["| Pattern | Count | Wins | WR | Avg R | First Unlock |",
           "|---------|------:|-----:|---:|------:|-------------|"])
        for pat, ts_list in sorted(by_pat.items(), key=lambda x: -len(x[1])):
            w = sum(1 for t in ts_list if _is_win(t))
            ar = sum(_r(t) for t in ts_list) / len(ts_list)
            fu_labels = list(dict.fromkeys(t["_first_unlock"] for t in ts_list))
            a([f"| {pat} | {len(ts_list)} | {w} | {_wr(ts_list)} | {_rs(ar)} | {', '.join(fu_labels)} |"])
        a([""])

        # By signal type (trigger candle type)
        by_sig: Dict[str, list] = {}
        for t in all_ul_dedup:
            s = str(t.get("entry_signal", t.get("signal_type","unknown")))
            by_sig.setdefault(s, []).append(t)
        a(["### By trigger candle type", ""])
        a(["| Trigger type | Count | WR | Avg R |",
           "|-------------|------:|---:|------:|"])
        for sig, ts_list in sorted(by_sig.items(), key=lambda x: -len(x[1])):
            a([f"| {sig} | {len(ts_list)} | {_wr(ts_list)} | {_avg_r(ts_list)} |"])
        a([""])

        # Top 20 by abs(R)
        a(["### Top-20 unlocked by |R|", ""])
        a(["| # | Unlock | Pair | Dir | Entry | Pattern | R |",
           "|---|--------|------|-----|-------|---------|---|"])
        for i, t in enumerate(sorted(all_ul_dedup, key=lambda t: abs(_r(t)), reverse=True)[:20], 1):
            ts_s = str(t.get("entry_ts",""))[:16]
            pat  = str(t.get("pattern",""))[:28]
            a([f"| {i} | {t['_first_unlock']} | {t.get('pair','')} | "
               f"{t.get('direction','')} | {ts_s} | {pat} | {_rs(_r(t))} |"])
        a([""])

    # ── temporal breakdown ────────────────────────────────────────────────────
    a(["---", "", "## Temporal Breakdown — Unlocked Trades", ""])

    for dim, label_fn, labels in [
        ("week",    _week,    ["W1 (Feb 1–14)", "W2 (Feb 15–28)", "Live-parity (Mar 1–4)"]),
        ("session", _session, ["London", "London_NY_Overlap", "NY", "off_session"]),
    ]:
        a([f"### By {dim}", ""])
        col = "| {{:<24}} |".format()
        a([f"| {dim.title():<24} | B ul | B W | B avgR | C ul | C W | C avgR | D ul | D W | D avgR |"])
        a(["|" + "---|" * 10])
        for lbl_row in labels:
            cells = []
            for var_lbl in ("B", "C", "D"):
                rv = results[var_lbl]
                bucket = [t for t in rv.trades
                          if _trade_key(t) not in baseline_keys
                          and label_fn(t.get("entry_ts", t.get("entry_time",""))) == lbl_row]
                cells += [str(len(bucket)),
                          str(sum(1 for t in bucket if _is_win(t))),
                          _avg_r(bucket)]
            a([f"| {lbl_row:<24} | {' | '.join(cells)} |"])
        a([""])

    # ── risk control insight ──────────────────────────────────────────────────
    a(["---", "", "## Risk Control Insight", ""])
    a(["*Trades present in A but absent in B/C/D AND realised_r < 0:*", ""])
    any_protected = False
    for lbl in ("B", "C", "D"):
        rv = results[lbl]; vk = {_trade_key(t) for t in rv.trades}
        lo = [t for t in ra.trades if _trade_key(t) not in vk and not _is_win(t)]
        if lo:
            any_protected = True
            a([f"**Variant {lbl}** — {len(lo)} loss(es) present in A but blocked in {lbl}:"])
            for t in lo:
                ts_s = str(t.get("entry_ts",""))[:16]
                a([f"  - {t.get('pair','')} {t.get('direction','')} @ {ts_s}  R={_rs(_r(t))}"])
    if not any_protected:
        a(["No cases where A had a loss that B/C/D blocked."])
    a([""])

    # ── engulf block counts from funnel ───────────────────────────────────────
    a(["---", "", "## Entry Funnel Context", ""])
    a(["Block counts are from the backtester's per-run verbose output.",
       "NO_ENGULF decreases as the trigger is relaxed; downstream gates absorb remainder.", ""])
    # We can't pull structured filter counts from BacktestResult; document directionally.
    a(["| Variant | Trigger mode | LB | Δ trades vs A |",
       "|---------|-------------|----|:-------------:|"])
    for lbl, tm, lb, _ in VARIANTS:
        rv = results[lbl]
        dt = rv.n_trades - na
        a([f"| {lbl} | `{tm}` | {lb} | {dt:+} |"])
    a([""])

    # ── findings ──────────────────────────────────────────────────────────────
    a(["---", "", "## Findings Summary", ""])

    rb = results["B"]; rc = results["C"]; rd = results["D"]
    ul_b = [t for t in rb.trades if _trade_key(t) not in baseline_keys]
    ul_c = [t for t in rc.trades if _trade_key(t) not in baseline_keys]
    ul_d = [t for t in rd.trades if _trade_key(t) not in baseline_keys]

    yn = lambda cond: "Yes" if cond else "No"
    a(["| Question | Answer |",
       "|----------|--------|",
       f"| Does adding rejection candle (B) unlock entries? | {yn(bool(ul_b))} ({len(ul_b)} trades) |",
       f"| Does lb=3 engulf window (C) unlock entries?       | {yn(bool(ul_c))} ({len(ul_c)} trades) |",
       f"| Does broadest variant (D) unlock entries?         | {yn(bool(ul_d))} ({len(ul_d)} trades) |",
       f"| B return delta vs A                               | {_pct(rb.return_pct - reta)} |",
       f"| C return delta vs A                               | {_pct(rc.return_pct - reta)} |",
       f"| D return delta vs A                               | {_pct(rd.return_pct - reta)} |",
       f"| B win rate (unlocked)                             | {_wr(ul_b)} |",
       f"| C win rate (unlocked)                             | {_wr(ul_c)} |",
       f"| D win rate (unlocked)                             | {_wr(ul_d)} |",
       f"| Primary bottleneck after zone+engulf              | Downstream gates (weekly_limit, exec_rr, confidence) |",
       f"| Is engulf-only trigger suppressing valid entries? | See unlock table — if WR of unlocked < 50%, gate is warranted |",
       ""])

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(verbose: bool = False) -> None:
    print("=" * 70)
    print(" Engulfing Confirmation Timing Ablation Study")
    _ws = WINDOW_START.strftime("%Y-%m-%d")
    _we = WINDOW_END.strftime("%Y-%m-%d")
    print(f" Window: {_ws} → {_we} | 7 pairs")
    print(" Variants: A (baseline) / B (reject) / C (lb=3) / D (reject+lb=3)")
    print("=" * 70)

    results: Dict[str, BacktestResult] = {}
    candle_cache: Optional[Dict] = None

    for lbl, tm, lb, desc in VARIANTS:
        result, cache = run_variant(lbl, tm, lb, desc,
                                    preloaded=candle_cache, verbose=verbose)
        results[lbl] = result
        if lbl == "A" and candle_cache is None:
            candle_cache = cache

    # Final safety reset
    _reset_engulf_config()

    print(f"\n{'=' * 70}")
    print(f" Generating report → {REPORT_PATH}")
    print(f"{'=' * 70}\n")

    report = build_report(results)
    print(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)

    ra = results["A"]
    bkeys = {_trade_key(t) for t in ra.trades}
    for lbl in ("B", "C", "D"):
        rv = results[lbl]
        ul = sum(1 for t in rv.trades if _trade_key(t) not in bkeys)
        print(f" Variant {lbl}: {rv.n_trades} trades | ret={rv.return_pct:+.1f}% | unlocked={ul}")

    print(f"\n Report: {REPORT_PATH}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    main(verbose=verbose)
