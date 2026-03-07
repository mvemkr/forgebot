#!/usr/bin/env python3
"""
Trend-Alignment Gate Ablation Study
=====================================
Offline-only research. Runs the full backtester three times over the same
window with one controlled variable: TREND_ALIGNMENT_GATE_MODE.

Variant A — Baseline   : "full"           (production behaviour)
Variant B — Disabled   : "disabled"       (gate removed entirely)
Variant C — Rev-bypass : "reversal_bypass" (DT/DB/H&S/IH&S bypass only)

Outputs a structured Markdown report:
  backtesting/results/ablation_trend_alignment.md

Constraints honoured:
  • start_dt / end_dt match the HTF alignment study (Feb 01 → Mar 04 2026)
  • Same 7 Alex pairs
  • dry_run semantics — no orders, no state mutations
  • master / production behaviour is NEVER changed at runtime
  • TREND_ALIGNMENT_GATE_MODE is reset to "full" on exit (even on crash)

Usage:
    python scripts/ablation_trend_alignment.py [--quiet]
"""
from __future__ import annotations

import argparse
import atexit
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

# ── Strategy config import (must come before backtester to patch mode early) ──
from src.strategy.forex import strategy_config as _sc

# ── Backtester ────────────────────────────────────────────────────────────────
from backtesting.oanda_backtest_v2 import run_backtest

# ── Constants ────────────────────────────────────────────────────────────────
STUDY_START = datetime(2026, 2, 1, tzinfo=timezone.utc)
STUDY_END   = datetime(2026, 3, 4, tzinfo=timezone.utc)
STARTING_BAL = 8_000.0

ALEX_PAIRS_OANDA = [
    "GBP/JPY", "USD/JPY", "USD/CHF", "GBP/CHF",
    "USD/CAD", "EUR/USD", "GBP/USD",
]

# Session windows in UTC hours (entry_ts.hour)
SESSION_MAP = {
    "London":          range(8, 13),
    "London_NY_Overlap": range(13, 17),
    "NY":              range(17, 22),
}

# Calendar windows
W1_END   = datetime(2026, 2, 14, tzinfo=timezone.utc)  # Feb 1–14
W2_END   = datetime(2026, 2, 28, tzinfo=timezone.utc)  # Feb 15–28
LIVE_END = STUDY_END                                    # Mar 1–4

REVERSAL_TYPES = frozenset({
    "double_top", "double_bottom", "head_and_shoulders", "inverted_head_and_shoulders",
})

VARIANTS = [
    ("A", "full",            "Baseline — gate active as implemented"),
    ("B", "disabled",        "Gate disabled — all patterns enter regardless of alignment"),
    ("C", "reversal_bypass", "Reversal-bypass — DT/DB/H&S/IH&S bypass; others still gated"),
]

# ── Safety: always restore gate mode on exit ──────────────────────────────────
_ORIGINAL_GATE_MODE = getattr(_sc, "TREND_ALIGNMENT_GATE_MODE", "full")

def _restore_gate():
    _sc.TREND_ALIGNMENT_GATE_MODE = _ORIGINAL_GATE_MODE

atexit.register(_restore_gate)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_variant(
    label: str,
    gate_mode: str,
    description: str,
    preloaded: Optional[Dict] = None,
    quiet: bool = True,
) -> Tuple[Any, Dict]:
    """Run one variant. Returns (BacktestResult, preloaded_candle_data)."""
    print(f"\n{'─'*62}")
    print(f"  Variant {label}: {description}")
    print(f"  gate_mode = {gate_mode!r}")
    print(f"{'─'*62}")

    _sc.TREND_ALIGNMENT_GATE_MODE = gate_mode
    t0 = time.time()

    result = run_backtest(
        start_dt              = STUDY_START,
        end_dt                = STUDY_END,
        starting_bal          = STARTING_BAL,
        notes                 = f"ablation_{label}_{gate_mode}",
        trail_arm_key         = f"ablation_{label}",
        preloaded_candle_data = preloaded,
        use_cache             = True,
        quiet                 = quiet,
    )

    elapsed = time.time() - t0
    print(f"  ✓ {result.n_trades} trades | ret={result.return_pct:+.1f}% | "
          f"WR={result.win_rate:.0%} | maxDD={result.max_dd_pct:.1f}% | {elapsed:.1f}s")

    _sc.TREND_ALIGNMENT_GATE_MODE = "full"  # reset after each run
    return result, result.candle_data or {}


# ── Trade helpers ─────────────────────────────────────────────────────────────

def _trade_key(t: dict) -> str:
    """Stable identifier for a trade: pair + open H1 bar."""
    ts = t.get("entry_ts") or t.get("open_ts") or ""
    if hasattr(ts, "strftime"):
        ts = ts.strftime("%Y%m%d%H")
    else:
        ts = str(ts)[:13].replace("-", "").replace("T", "").replace(":", "").replace(" ", "")
    return f"{t.get('pair', '?')}|{ts}"


def _is_win(t: dict) -> bool:
    r = t.get("realised_r") or t.get("result_r") or t.get("r") or 0.0
    return float(r) > 0


def _realised_r(t: dict) -> float:
    for k in ("realised_r", "result_r", "r", "realised_R", "R"):
        if k in t:
            return float(t[k])
    return 0.0


def _session(t: dict) -> str:
    ts = t.get("entry_ts")
    if ts is None:
        return "unknown"
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except Exception:
            return "unknown"
    h = ts.hour if hasattr(ts, "hour") else 0
    for name, r in SESSION_MAP.items():
        if h in r:
            return name
    return "off_session"


def _window(t: dict) -> str:
    ts = t.get("entry_ts")
    if ts is None:
        return "unknown"
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except Exception:
            return "unknown"
    if not hasattr(ts, "tzinfo") or ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if ts < W1_END:
        return "W1 (Feb 1–14)"
    if ts < W2_END:
        return "W2 (Feb 15–28)"
    return "Live-parity (Mar 1–4)"


def _pattern(t: dict) -> str:
    p = t.get("pattern") or t.get("pattern_type") or "unknown"
    return str(p)


def _is_reversal(t: dict) -> bool:
    return any(k in _pattern(t) for k in REVERSAL_TYPES)


def _worst_3_cluster(trades: List[dict]) -> Tuple[float, List[dict]]:
    """Find the worst 3-trade consecutive losing cluster by sum R."""
    if len(trades) < 3:
        return 0.0, []
    sorted_by_ts = sorted(trades, key=lambda x: str(x.get("entry_ts", "")))
    worst_sum = 0.0
    worst_cluster: List[dict] = []
    for i in range(len(sorted_by_ts) - 2):
        cluster = sorted_by_ts[i:i+3]
        s = sum(_realised_r(t) for t in cluster)
        if s < worst_sum:
            worst_sum = s
            worst_cluster = cluster
    return worst_sum, worst_cluster


def _max_dd(trades: List[dict]) -> float:
    """Compute peak-to-trough cumulative R drawdown from trade list."""
    if not trades:
        return 0.0
    rs = [_realised_r(t) for t in sorted(trades, key=lambda x: str(x.get("entry_ts", "")))]
    peak = 0.0; trough = 0.0; dd = 0.0; cum = 0.0
    for r in rs:
        cum += r
        if cum > peak:
            peak = cum
        dd_now = peak - cum
        if dd_now > dd:
            dd = dd_now
    return dd


# ── Report generator ──────────────────────────────────────────────────────────

def generate_report(
    results: Dict[str, Any],
    trades:  Dict[str, List[dict]],
    out_path: Path,
) -> str:

    lines: List[str] = []
    a = lines.append

    a("# Trend-Alignment Gate Ablation Study")
    a(f"Window: {STUDY_START.date()} → {STUDY_END.date()} | Pairs: {len(ALEX_PAIRS_OANDA)} (Alex universe)")
    a(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    a("")
    a("---")
    a("")

    # ── Primary metrics table ─────────────────────────────────────────────────
    a("## Primary Metrics")
    a("")
    a("| Metric | Variant A (Baseline) | Variant B (Disabled) | Variant C (Rev-bypass) |")
    a("|--------|---------------------:|---------------------:|-----------------------:|")

    def _fmt(r, attr, fmt):
        v = getattr(r, attr, None)
        if v is None:
            return "—"
        return fmt.format(v)

    rA = results["A"]; rB = results["B"]; rC = results["C"]
    tA = trades["A"];  tB = trades["B"];  tC = trades["C"]

    rows = [
        ("Total trades",    "n_trades",    "{:d}"),
        ("Win rate",        "win_rate",    "{:.0%}"),
        ("Average R",       "avg_r",       "{:+.2f}R"),
        ("Return %",        "return_pct",  "{:+.1f}%"),
        ("Max drawdown %",  "max_dd_pct",  "{:.1f}%"),
        ("Best trade R",    "best_r",      "{:+.2f}R"),
        ("Worst trade R",   "worst_r",     "{:+.2f}R"),
    ]
    for label, attr, fmt in rows:
        a(f"| {label} | {_fmt(rA, attr, fmt)} | {_fmt(rB, attr, fmt)} | {_fmt(rC, attr, fmt)} |")

    # Worst 3-loss cluster
    w3A, _ = _worst_3_cluster(tA)
    w3B, _ = _worst_3_cluster(tB)
    w3C, _ = _worst_3_cluster(tC)
    a(f"| Worst 3-loss cluster R | {w3A:+.2f}R | {w3B:+.2f}R | {w3C:+.2f}R |")
    a("")

    # ── Unlock analysis ───────────────────────────────────────────────────────
    def _unlocked(base: List[dict], variant: List[dict]) -> List[dict]:
        """Trades in variant that are NOT in base (newly unlocked)."""
        base_keys = {_trade_key(t) for t in base}
        return [t for t in variant if _trade_key(t) not in base_keys]

    def _blocked_good(base: List[dict], variant: List[dict]) -> List[dict]:
        """Losing trades in variant-only (trades the gate correctly prevented in base)."""
        return [t for t in _unlocked(base, variant) if not _is_win(t)]

    unlocked_B = _unlocked(tA, tB)
    unlocked_C = _unlocked(tA, tC)
    protected_B = _blocked_good(tA, tB)   # losses gate prevented in A that B allows
    protected_C = _blocked_good(tA, tC)

    a("## Unlock Analysis")
    a("")
    a("| Metric | B vs A | C vs A |")
    a("|--------|-------:|-------:|")
    a(f"| Newly unlocked entries | {len(unlocked_B)} | {len(unlocked_C)} |")
    a(f"| Unlocked wins  | {sum(1 for t in unlocked_B if _is_win(t))} | {sum(1 for t in unlocked_C if _is_win(t))} |")
    a(f"| Unlocked losses | {len(protected_B)} | {len(protected_C)} |")
    a(f"| Win rate (unlocked only) | {sum(1 for t in unlocked_B if _is_win(t))/max(len(unlocked_B),1):.0%} | {sum(1 for t in unlocked_C if _is_win(t))/max(len(unlocked_C),1):.0%} |")
    unlocked_B_rs = [_realised_r(t) for t in unlocked_B]
    unlocked_C_rs = [_realised_r(t) for t in unlocked_C]
    a(f"| Avg R (unlocked) | {np.mean(unlocked_B_rs):+.2f}R | {np.mean(unlocked_C_rs):+.2f}R |" if unlocked_B_rs and unlocked_C_rs else
      f"| Avg R (unlocked) | — | — |")
    a(f"| Risk control: gate correctly prevented losses in A | {len(protected_B)} | {len(protected_C)} |")
    a("")

    # Top 20 newly unlocked patterns (B and C separately)
    for var_label, unlocked in [("B", unlocked_B), ("C", unlocked_C)]:
        if not unlocked:
            a(f"### Top unlocked patterns — Variant {var_label}: *(none)*")
            a("")
            continue
        a(f"### Top 20 unlocked patterns — Variant {var_label}")
        a("")
        a("| # | Pair | Pattern | Dir | entry_ts | R | Win? | Session | Window |")
        a("|---|------|---------|-----|----------|---|------|---------|--------|")
        sorted_by_r = sorted(unlocked, key=lambda t: -_realised_r(t))
        for i, t in enumerate(sorted_by_r[:20], 1):
            ts_str = str(t.get("entry_ts", "?"))[:16]
            a(f"| {i} | {t.get('pair','?')} | {_pattern(t)} | {t.get('direction','?')} | {ts_str} | "
              f"{_realised_r(t):+.2f}R | {'✓' if _is_win(t) else '✗'} | "
              f"{_session(t)} | {_window(t)} |")
        a("")

    # ── Trade quality — MAE/MFE of unlocked ──────────────────────────────────
    a("## Trade Quality — Unlocked Trades")
    a("")
    for var_label, unlocked in [("B", unlocked_B), ("C", unlocked_C)]:
        if not unlocked:
            a(f"### Variant {var_label} unlocked: *(no unlocked trades to analyse)*")
            a("")
            continue
        mfes = [t.get("mfe_r", 0.0) for t in unlocked]
        maes = [abs(t.get("mae_r", 0.0)) for t in unlocked]
        a(f"### Variant {var_label} — MAE/MFE distribution (n={len(unlocked)})")
        a(f"| Metric | MAE (adverse, pips × R) | MFE (favourable, pips × R) |")
        a(f"|--------|------------------------:|---------------------------:|")
        a(f"| Mean   | {np.mean(maes):+.2f}R | {np.mean(mfes):+.2f}R |")
        a(f"| Median | {np.median(maes):+.2f}R | {np.median(mfes):+.2f}R |")
        a(f"| P75    | {np.percentile(maes,75):+.2f}R | {np.percentile(mfes,75):+.2f}R |")
        a(f"| Max    | {max(maes):+.2f}R | {max(mfes):+.2f}R |")
        a("")

        # Pattern breakdown
        a(f"#### By pattern type")
        a(f"| Pattern | Count | Wins | WR | Avg R |")
        a(f"|---------|------:|-----:|---:|------:|")
        pat_grp = defaultdict(list)
        for t in unlocked:
            pat_grp[_pattern(t)].append(t)
        for pt, ts in sorted(pat_grp.items(), key=lambda x: -len(x[1])):
            wins = sum(1 for t in ts if _is_win(t))
            avg_r = np.mean([_realised_r(t) for t in ts])
            a(f"| {pt} | {len(ts)} | {wins} | {wins/len(ts):.0%} | {avg_r:+.2f}R |")
        a("")

    # ── Temporal breakdown ────────────────────────────────────────────────────
    a("## Temporal Breakdown — Unlocked Trades")
    a("")
    a("### By window")
    a("")
    a("| Window | B unlocked | B wins | B avg R | C unlocked | C wins | C avg R |")
    a("|--------|----------:|-------:|--------:|-----------:|-------:|--------:|")
    for win_name in ["W1 (Feb 1–14)", "W2 (Feb 15–28)", "Live-parity (Mar 1–4)"]:
        Bu = [t for t in unlocked_B if _window(t) == win_name]
        Cu = [t for t in unlocked_C if _window(t) == win_name]
        Bw = sum(1 for t in Bu if _is_win(t))
        Cw = sum(1 for t in Cu if _is_win(t))
        Br = np.mean([_realised_r(t) for t in Bu]) if Bu else 0.0
        Cr = np.mean([_realised_r(t) for t in Cu]) if Cu else 0.0
        a(f"| {win_name} | {len(Bu)} | {Bw} | {Br:+.2f}R | {len(Cu)} | {Cw} | {Cr:+.2f}R |")
    a("")

    a("### By session")
    a("")
    a("| Session | B unlocked | B wins | B avg R | C unlocked | C wins | C avg R |")
    a("|---------|----------:|-------:|--------:|-----------:|-------:|--------:|")
    for sess_name in ["London", "London_NY_Overlap", "NY", "off_session"]:
        Bu = [t for t in unlocked_B if _session(t) == sess_name]
        Cu = [t for t in unlocked_C if _session(t) == sess_name]
        Bw = sum(1 for t in Bu if _is_win(t))
        Cw = sum(1 for t in Cu if _is_win(t))
        Br = np.mean([_realised_r(t) for t in Bu]) if Bu else 0.0
        Cr = np.mean([_realised_r(t) for t in Cu]) if Cu else 0.0
        a(f"| {sess_name} | {len(Bu)} | {Bw} | {Br:+.2f}R | {len(Cu)} | {Cw} | {Cr:+.2f}R |")
    a("")

    # ── Risk control insight ──────────────────────────────────────────────────
    a("## Risk Control Insight")
    a("")
    a("*Baseline losing trades that trend_alignment CORRECTLY prevented (i.e.*")
    a("*trade exists in B/C but NOT in A, and it was a loss)*")
    a("")
    a("| Variant | Protected losses | Unprotected wins (opportunity cost) |")
    a("|---------|----------------:|------------------------------------:|")
    a(f"| B (disabled)       | {len(protected_B)} | {sum(1 for t in unlocked_B if _is_win(t))} |")
    a(f"| C (reversal-bypass)| {len(protected_C)} | {sum(1 for t in unlocked_C if _is_win(t))} |")
    a("")
    # Detail: worst protected losses
    for var_label, protected in [("B", protected_B), ("C", protected_C)]:
        if not protected:
            continue
        worst = sorted(protected, key=lambda t: _realised_r(t))[:5]
        a(f"### Worst protected losses — Variant {var_label} (top 5)")
        a(f"| Pair | Pattern | Dir | R | Session | Window |")
        a(f"|------|---------|-----|---|---------|--------|")
        for t in worst:
            a(f"| {t.get('pair','?')} | {_pattern(t)} | {t.get('direction','?')} | "
              f"{_realised_r(t):+.2f}R | {_session(t)} | {_window(t)} |")
        a("")

    # ── Conclusion ────────────────────────────────────────────────────────────
    a("## Findings Summary")
    a("")
    a("| Question | Answer |")
    a("|----------|--------|")

    # Is gate net positive?
    b_improvement = rB.return_pct - rA.return_pct
    c_improvement = rC.return_pct - rA.return_pct
    b_dd_delta    = rB.max_dd_pct - rA.max_dd_pct
    c_dd_delta    = rC.max_dd_pct - rA.max_dd_pct

    a(f"| Does removing the gate (B) improve returns? | {'Yes' if b_improvement > 0 else 'No'} ({b_improvement:+.1f}%) |")
    a(f"| Does B change max drawdown? | {b_dd_delta:+.1f}% DD delta |")
    a(f"| Does reversal-bypass (C) improve returns? | {'Yes' if c_improvement > 0 else 'No'} ({c_improvement:+.1f}%) |")
    a(f"| Does C change max drawdown? | {c_dd_delta:+.1f}% DD delta |")
    a(f"| Newly unlocked trades (B): win rate | {sum(1 for t in unlocked_B if _is_win(t))/max(len(unlocked_B),1):.0%} ({len(unlocked_B)} trades) |")
    a(f"| Newly unlocked trades (C): win rate | {sum(1 for t in unlocked_C if _is_win(t))/max(len(unlocked_C),1):.0%} ({len(unlocked_C)} trades) |")
    a(f"| Gate protective value (losses blocked in A): B | {len(protected_B)} |")
    a(f"| Gate protective value (losses blocked in A): C | {len(protected_C)} |")
    a(f"| Is trend_alignment gate net beneficial for reversal engine? | {'Yes — keeps out more losses than wins' if len(protected_B) > sum(1 for t in unlocked_B if _is_win(t)) else 'Ambiguous — see trade quality table'} |")
    a("")

    report = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="Trend-Alignment Gate Ablation Study")
    parser.add_argument("--quiet",   action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out",     type=Path,
                        default=_ROOT / "backtesting/results/ablation_trend_alignment.md")
    args = parser.parse_args(argv)
    quiet = not args.verbose

    W = 62
    print(f"\n{'='*W}")
    print(f" Trend-Alignment Gate Ablation Study")
    print(f" Window : {STUDY_START.date()} → {STUDY_END.date()}")
    print(f" Pairs  : {len(ALEX_PAIRS_OANDA)} (Alex universe)")
    print(f" Output : {args.out}")
    print(f"{'='*W}")

    results = {}
    trade_lists = {}
    preloaded = None

    for label, gate_mode, description in VARIANTS:
        result, candle_data = run_variant(
            label, gate_mode, description,
            preloaded=preloaded,
            quiet=quiet,
        )
        results[label]     = result
        trade_lists[label] = list(result.trades)
        if preloaded is None and candle_data:
            preloaded = candle_data   # reuse data for B and C

    # Restore gate mode (atexit will also do this)
    _sc.TREND_ALIGNMENT_GATE_MODE = _ORIGINAL_GATE_MODE

    print(f"\n{'='*W}")
    print(f" Generating report → {args.out}")
    print(f"{'='*W}")

    report = generate_report(results, trade_lists, args.out)
    rlines = report.split("\n")
    print("\n" + "\n".join(rlines[:100]))
    if len(rlines) > 100:
        print(f"\n[... {len(rlines)-100} more lines → {args.out}]")

    print(f"\n{'='*W}")
    print(f" Report: {args.out}")

    # Summarise counts
    base_keys = {_trade_key(t) for t in trade_lists["A"]}
    uB = [t for t in trade_lists["B"] if _trade_key(t) not in base_keys]
    uC = [t for t in trade_lists["C"] if _trade_key(t) not in base_keys]
    print(f" A: {results['A'].n_trades} trades | B: {results['B'].n_trades} | C: {results['C'].n_trades}")
    print(f" Newly unlocked B: {len(uB)} | C: {len(uC)}")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
