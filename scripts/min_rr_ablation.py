"""
min_rr_ablation.py — Offline MIN_RR ablation study.

Three variants across 8 windows, using the current B-Prime LIVE_PAPER trigger config:
  A) Baseline  MIN_RR = 2.5  (production)
  B) Moderate  MIN_RR = 2.0
  C) Lower     MIN_RR = 1.5

Goal: determine whether 2.0 or 1.5 is structurally better than 2.5.

Usage:
    python3 scripts/min_rr_ablation.py
    python3 scripts/min_rr_ablation.py --windows Q1-2025 Q4-2025
    python3 scripts/min_rr_ablation.py --variants A B

OFFLINE ONLY.  No live changes.  No master merge.  atexit resets all config.
"""
from __future__ import annotations

import atexit
import argparse
import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── repo root on path ─────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

import src.strategy.forex.strategy_config as _sc
from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult, STARTING_BAL

# ── Capture production values at import time ──────────────────────────────
_ORIG_MIN_RR              = _sc.MIN_RR
_ORIG_MIN_RR_STANDARD     = _sc.MIN_RR_STANDARD
_ORIG_MIN_RR_COUNTERTREND = _sc.MIN_RR_COUNTERTREND
_ORIG_MIN_RR_SMALL        = _sc.MIN_RR_SMALL_ACCOUNT
# B-Prime trigger state (must survive resets)
_ORIG_TRIGGER_MODE        = _sc.ENTRY_TRIGGER_MODE
_ORIG_WL                  = _sc.STRICT_PIN_PATTERN_WHITELIST
_ORIG_LB                  = _sc.ENGULF_CONFIRM_LOOKBACK_BARS


def _reset_config() -> None:
    """Restore all mutated config vars to production values."""
    _sc.MIN_RR              = _ORIG_MIN_RR
    _sc.MIN_RR_STANDARD     = _ORIG_MIN_RR_STANDARD
    _sc.MIN_RR_COUNTERTREND = _ORIG_MIN_RR_COUNTERTREND
    _sc.MIN_RR_SMALL_ACCOUNT= _ORIG_MIN_RR_SMALL
    # Restore B-Prime trigger config
    _sc.ENTRY_TRIGGER_MODE          = _ORIG_TRIGGER_MODE
    _sc.STRICT_PIN_PATTERN_WHITELIST= _ORIG_WL
    _sc.ENGULF_CONFIRM_LOOKBACK_BARS= _ORIG_LB
    _sc.ENGULFING_ONLY              = (_sc.ENTRY_TRIGGER_MODE == "engulf_only")


atexit.register(_reset_config)


def _set_rr(min_rr: float) -> None:
    """Set all four RR vars to min_rr. Trigger config is left at B-Prime."""
    _sc.MIN_RR               = min_rr
    _sc.MIN_RR_STANDARD      = min_rr
    _sc.MIN_RR_COUNTERTREND  = min_rr
    _sc.MIN_RR_SMALL_ACCOUNT = min_rr


# ── Variants ──────────────────────────────────────────────────────────────
VARIANTS: List[Tuple[str, float, str]] = [
    ("A", 2.5, "Baseline  (2.5R)"),
    ("B", 2.0, "Moderate  (2.0R)"),
    ("C", 1.5, "Lower     (1.5R)"),
]

# ── Windows ───────────────────────────────────────────────────────────────
_UTC = timezone.utc
WINDOWS: List[Tuple[str, datetime, datetime]] = [
    ("Q1-2025",      datetime(2025, 1,  1, tzinfo=_UTC), datetime(2025, 3, 31, tzinfo=_UTC)),
    ("Q2-2025",      datetime(2025, 4,  1, tzinfo=_UTC), datetime(2025, 6, 30, tzinfo=_UTC)),
    ("Q3-2025",      datetime(2025, 7,  1, tzinfo=_UTC), datetime(2025, 9, 30, tzinfo=_UTC)),
    ("Q4-2025",      datetime(2025, 10, 1, tzinfo=_UTC), datetime(2025, 12,31, tzinfo=_UTC)),
    ("Jan-Feb-2026", datetime(2026, 1,  1, tzinfo=_UTC), datetime(2026, 2, 28, tzinfo=_UTC)),
    ("W1",           datetime(2024, 7,  1, tzinfo=_UTC), datetime(2024, 7, 31, tzinfo=_UTC)),
    ("W2",           datetime(2024, 8,  1, tzinfo=_UTC), datetime(2024, 8, 31, tzinfo=_UTC)),
    ("live-parity",  datetime(2026, 2, 28, tzinfo=_UTC), datetime(2026, 3,  6, tzinfo=_UTC)),
]

CAPITAL = STARTING_BAL   # 8_000.0 — matches backtester default

# Informational only — backtester uses its internal WATCHLIST.
# Alex's 7 pairs are enforced via logs/whitelist_backtest.json when enabled.
PAIRS_NOTE = ["GBP/JPY", "USD/JPY", "USD/CHF", "GBP/CHF",
              "USD/CAD", "EUR/USD", "GBP/USD"]

# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    """Single variant result for one window."""
    variant:  str
    min_rr:   float
    label:    str
    window:   str
    result:   Optional[BacktestResult]
    error:    str = ""

    # ── derived metrics ───────────────────────────────────────────────────
    @property
    def n(self) -> int:
        return self.result.n_trades if self.result else 0

    @property
    def ret(self) -> float:
        return self.result.return_pct if self.result else 0.0

    @property
    def wr(self) -> float:
        return self.result.win_rate if self.result else 0.0

    @property
    def avg_r(self) -> float:
        return self.result.avg_r if self.result else 0.0

    @property
    def max_dd(self) -> float:
        return self.result.max_dd_pct if self.result else 0.0

    @property
    def trades(self) -> List[dict]:
        return self.result.trades if self.result else []

    @property
    def total_r(self) -> float:
        return sum(t.get("r", 0.0) for t in self.trades)

    @property
    def expectancy(self) -> float:
        """Sum_R / n_trades (raw expectancy in R)."""
        return self.total_r / self.n if self.n else 0.0

    @property
    def worst3(self) -> List[float]:
        rs = sorted(t.get("r", 0.0) for t in self.trades)
        return rs[:3]

    @property
    def mae_r_list(self) -> List[float]:
        return [t.get("mae_r", 0.0) for t in self.trades if t.get("mae_r") is not None]

    @property
    def mfe_r_list(self) -> List[float]:
        return [t.get("mfe_r", 0.0) for t in self.trades if t.get("mfe_r") is not None]


@dataclass
class WindowTriple:
    """Results for all three variants on one window."""
    window:    str
    start:     datetime
    end:       datetime
    result_a:  WindowResult
    result_b:  WindowResult
    result_c:  WindowResult

    @property
    def results(self) -> List[WindowResult]:
        return [self.result_a, self.result_b, self.result_c]

    def by_variant(self, v: str) -> WindowResult:
        return {"A": self.result_a, "B": self.result_b, "C": self.result_c}[v]


# ── Trade identity helpers ────────────────────────────────────────────────

def _trade_key(t: dict) -> Tuple[str, str, str]:
    """(pair, direction, entry_ts rounded to hour) — enough to identify same trade."""
    ts = str(t.get("entry_ts", ""))[:13]   # "2025-01-15T09"
    return (t.get("pair", ""), t.get("direction", ""), ts)


def _find_unlocked(
    base_trades: List[dict],
    new_trades:  List[dict],
) -> List[dict]:
    """Return trades in new_trades whose key does not appear in base_trades."""
    base_keys = {_trade_key(t) for t in base_trades}
    return [t for t in new_trades if _trade_key(t) not in base_keys]


def _find_removed(
    base_trades: List[dict],
    new_trades:  List[dict],
) -> List[dict]:
    """Return trades in base_trades that are absent from new_trades (regression check)."""
    new_keys = {_trade_key(t) for t in new_trades}
    return [t for t in base_trades if _trade_key(t) not in new_keys]


# ── Distribution helpers ──────────────────────────────────────────────────

def _pair_dist(trades: List[dict]) -> Dict[str, int]:
    d: Dict[str, int] = defaultdict(int)
    for t in trades:
        d[t.get("pair", "?")] += 1
    return dict(sorted(d.items(), key=lambda x: -x[1]))


def _pattern_dist(trades: List[dict]) -> Dict[str, int]:
    d: Dict[str, int] = defaultdict(int)
    for t in trades:
        p = t.get("pattern") or t.get("pattern_type") or "?"
        d[p] += 1
    return dict(sorted(d.items(), key=lambda x: -x[1]))


def _dir_dist(trades: List[dict]) -> Dict[str, int]:
    d: Dict[str, int] = defaultdict(int)
    for t in trades:
        d[t.get("direction", "?")] += 1
    return dict(d)


def _wr(trades: List[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("r", 0.0) > 0)
    return wins / len(trades)


def _avg_r(trades: List[dict]) -> float:
    if not trades:
        return 0.0
    return sum(t.get("r", 0.0) for t in trades) / len(trades)


def _sum_r(trades: List[dict]) -> float:
    return sum(t.get("r", 0.0) for t in trades)


def _avg_mae(trades: List[dict]) -> Optional[float]:
    vals = [t.get("mae_r") for t in trades if t.get("mae_r") is not None]
    return sum(vals) / len(vals) if vals else None


def _avg_mfe(trades: List[dict]) -> Optional[float]:
    vals = [t.get("mfe_r") for t in trades if t.get("mfe_r") is not None]
    return sum(vals) / len(vals) if vals else None


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


# ── Core run ──────────────────────────────────────────────────────────────

def _run_variant(
    variant:  str,
    min_rr:   float,
    label:    str,
    window:   str,
    start:    datetime,
    end:      datetime,
    preloaded_candle_data: Optional[dict] = None,
) -> Tuple[WindowResult, Optional[dict]]:
    """
    Set config, run backtest, restore RR, return (WindowResult, candle_data).
    Trigger config (B-Prime) is left unchanged throughout.
    """
    _set_rr(min_rr)
    candle_data = None
    try:
        result: BacktestResult = run_backtest(
            start_dt              = start,
            end_dt                = end,
            starting_bal          = CAPITAL,
            notes                 = f"minrr_{variant}_{window}",
            trail_arm_key         = f"rr_{variant}_{window}",
            preloaded_candle_data = preloaded_candle_data,
            use_cache             = True,
            quiet                 = True,
        )
        candle_data = getattr(result, "candle_data", None)
        wr = WindowResult(
            variant=variant, min_rr=min_rr, label=label,
            window=window, result=result,
        )
    except Exception as exc:                         # noqa: BLE001
        wr = WindowResult(
            variant=variant, min_rr=min_rr, label=label,
            window=window, result=None, error=str(exc),
        )
    finally:
        _reset_config()
    return wr, candle_data


def run_ablation(
    windows_filter:  Optional[List[str]] = None,
    variants_filter: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[WindowTriple]:
    """
    Run the full ablation study.  Returns list of WindowTriple (one per window).
    """
    windows_to_run  = WINDOWS  if not windows_filter  else [w for w in WINDOWS  if w[0] in windows_filter]
    variants_to_run = VARIANTS if not variants_filter else [v for v in VARIANTS if v[0] in variants_filter]

    triples: List[WindowTriple] = []

    for win_name, start, end in windows_to_run:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Window: {win_name}  ({start.date()} → {end.date()})")
            print(f"{'='*60}")

        results_by_v: Dict[str, WindowResult] = {}
        shared_cache: Optional[dict] = None

        for var_id, min_rr, label in variants_to_run:
            if verbose:
                print(f"  Variant {var_id} ({min_rr}R) … ", end="", flush=True)
            wr, cache = _run_variant(
                variant=var_id, min_rr=min_rr, label=label,
                window=win_name, start=start, end=end,
                preloaded_candle_data=shared_cache,
            )
            if shared_cache is None and cache:
                shared_cache = cache
            results_by_v[var_id] = wr
            if verbose:
                status = f"{wr.n} trades  {_fmt_pct(wr.ret)}  {_fmt_wr(wr.wr)} WR" if not wr.error else f"ERROR: {wr.error[:60]}"
                print(status)

        # Build triple — fill missing variants with zero result
        def _get(v: str) -> WindowResult:
            if v in results_by_v:
                return results_by_v[v]
            return WindowResult(variant=v, min_rr={"A":2.5,"B":2.0,"C":1.5}[v],
                                label="skipped", window=win_name, result=None)

        triple = WindowTriple(
            window=win_name, start=start, end=end,
            result_a=_get("A"),
            result_b=_get("B"),
            result_c=_get("C"),
        )
        triples.append(triple)

    return triples


# ── Report builder ────────────────────────────────────────────────────────

def _build_report(triples: List[WindowTriple]) -> str:
    lines: List[str] = []
    W = lines.append

    W("# MIN_RR Ablation Study")
    W(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    W("")
    W("## Config")
    W("| Setting | Value |")
    W("|---------|-------|")
    W(f"| ENTRY_TRIGGER_MODE | `{_ORIG_TRIGGER_MODE}` |")
    W(f"| STRICT_PIN_PATTERN_WHITELIST | `{_ORIG_WL}` |")
    W(f"| ENGULF_CONFIRM_LOOKBACK_BARS | `{_ORIG_LB}` |")
    W(f"| MIN_CONFIDENCE | `{_sc.MIN_CONFIDENCE}` |")
    W(f"| MIN_PIP_EQUITY | `{_sc.MIN_PIP_EQUITY}` |")
    W(f"| RECOVERY_MIN_RR (unchanged) | `{_sc.RECOVERY_MIN_RR}` |")
    W("")
    W("## Variants")
    W("| ID | MIN_RR | Label |")
    W("|----|--------|-------|")
    for var_id, min_rr, label in VARIANTS:
        W(f"| {var_id} | {min_rr} | {label} |")
    W("")

    # ── Per-window table ─────────────────────────────────────────────────
    W("## Per-Window Results")
    W("")
    hdr = "| Window | Var | MinRR | Trades | WR | AvgR | SumR | Ret% | MaxDD | Expectancy | Worst3 |"
    sep = "|--------|-----|-------|:------:|:--:|:----:|:----:|:----:|:-----:|:----------:|--------|"
    W(hdr); W(sep)

    for triple in triples:
        for wr in triple.results:
            if wr.result is None:
                W(f"| {triple.window} | {wr.variant} | {wr.min_rr} | — | — | — | — | — | — | — | {wr.error[:30] or 'skipped'} |")
                continue
            worst_str = "  ".join(_fmt_r(r) for r in wr.worst3) if wr.worst3 else "—"
            W(f"| {triple.window} | **{wr.variant}** | {wr.min_rr} "
              f"| {wr.n} "
              f"| {_fmt_wr(wr.wr)} "
              f"| {_fmt_r(wr.avg_r)} "
              f"| {_fmt_r(wr.total_r)} "
              f"| {_fmt_pct(wr.ret)} "
              f"| {_fmt_pct(wr.max_dd)} "
              f"| {_fmt_r(wr.expectancy)} "
              f"| {worst_str} |")
        W(f"| | | | | | | | | | | |")

    W("")

    # ── Aggregates ───────────────────────────────────────────────────────
    W("## Aggregate Totals (all windows)")
    W("")
    W("| Var | MinRR | Total Trades | Total SumR | Avg Ret%/win | Wins | Losses | WR | AvgDD |")
    W("|-----|-------|:------------:|:----------:|:------------:|:----:|:------:|:--:|:-----:|")

    for var_id, min_rr, _ in VARIANTS:
        all_trades = [t for tri in triples for t in tri.by_variant(var_id).trades]
        n = len(all_trades)
        wins   = sum(1 for t in all_trades if t.get("r", 0) > 0)
        losses = n - wins
        wr_agg = wins / n if n else 0.0
        sum_r  = _sum_r(all_trades)
        rets   = [tri.by_variant(var_id).ret for tri in triples if tri.by_variant(var_id).result]
        avg_ret = sum(rets) / len(rets) if rets else 0.0
        dds     = [tri.by_variant(var_id).max_dd for tri in triples if tri.by_variant(var_id).result]
        avg_dd  = sum(dds) / len(dds) if dds else 0.0
        W(f"| **{var_id}** | {min_rr} | {n} | {_fmt_r(sum_r)} | {_fmt_pct(avg_ret)} "
          f"| {wins} | {losses} | {_fmt_wr(wr_agg)} | {_fmt_pct(avg_dd)} |")

    W("")

    # ── Unlock analysis ──────────────────────────────────────────────────
    W("## Unlock Analysis")
    W("")
    W("### Bucket: 2.0–2.49R  (trades in B not in A)")
    W("")
    all_b_unlocked: List[dict] = []
    for tri in triples:
        unlocked = _find_unlocked(tri.result_a.trades, tri.result_b.trades)
        all_b_unlocked.extend(unlocked)

    _write_unlock_section(lines, all_b_unlocked, "A → B", triples, "A", "B")

    W("")
    W("### Bucket: 1.5–1.99R  (trades in C not in B)")
    W("")
    all_c_unlocked: List[dict] = []
    for tri in triples:
        unlocked = _find_unlocked(tri.result_b.trades, tri.result_c.trades)
        all_c_unlocked.extend(unlocked)

    _write_unlock_section(lines, all_c_unlocked, "B → C", triples, "B", "C")

    W("")

    # ── Regression check ─────────────────────────────────────────────────
    W("## Regression Check  (trades present in A but removed in B or C)")
    W("")
    all_b_removed: List[dict] = []
    all_c_removed: List[dict] = []
    for tri in triples:
        all_b_removed.extend(_find_removed(tri.result_a.trades, tri.result_b.trades))
        all_c_removed.extend(_find_removed(tri.result_a.trades, tri.result_c.trades))

    W(f"- Removed A→B: **{len(all_b_removed)}** trades")
    W(f"- Removed A→C: **{len(all_c_removed)}** trades")
    if all_b_removed:
        W("")
        W("Removed in B (not in A):")
        for t in all_b_removed[:10]:
            W(f"  - {t.get('entry_ts','')[:10]} {t.get('pair','')} {t.get('direction','')} "
              f"{t.get('pattern','')}  r={_fmt_r(t.get('r'))}")
    W("")

    # ── Per-window unlock table ───────────────────────────────────────────
    W("## Per-Window Unlock Breakdown")
    W("")
    W("| Window | B_unlocked | B_unlocked WR | B_unlocked SumR | C_unlocked | C_unlocked WR | C_unlocked SumR |")
    W("|--------|:----------:|:-------------:|:---------------:|:----------:|:-------------:|:---------------:|")

    for tri in triples:
        bu = _find_unlocked(tri.result_a.trades, tri.result_b.trades)
        cu = _find_unlocked(tri.result_b.trades, tri.result_c.trades)
        bw = _fmt_wr(_wr(bu)) if bu else "—"
        cw = _fmt_wr(_wr(cu)) if cu else "—"
        bs = _fmt_r(_sum_r(bu)) if bu else "—"
        cs = _fmt_r(_sum_r(cu)) if cu else "—"
        W(f"| {tri.window} | {len(bu)} | {bw} | {bs} | {len(cu)} | {cw} | {cs} |")

    W("")

    # ── Pattern distribution ──────────────────────────────────────────────
    W("## Pattern Distribution of Unlocked Trades")
    W("")
    W("### B-unlocked (2.0–2.49R bucket)")
    _write_dist_table(lines, _pattern_dist(all_b_unlocked), all_b_unlocked)
    W("")
    W("### C-unlocked (1.5–1.99R bucket)")
    _write_dist_table(lines, _pattern_dist(all_c_unlocked), all_c_unlocked)
    W("")

    # ── Pair distribution ─────────────────────────────────────────────────
    W("## Pair Distribution of Unlocked Trades")
    W("")
    W("### B-unlocked")
    _write_dist_table(lines, _pair_dist(all_b_unlocked), all_b_unlocked, key="pair")
    W("")
    W("### C-unlocked")
    _write_dist_table(lines, _pair_dist(all_c_unlocked), all_c_unlocked, key="pair")
    W("")

    # ── MAE / MFE of unlocked trades ─────────────────────────────────────
    W("## MAE / MFE — Unlocked Trades")
    W("")
    W("| Bucket | N | Avg MAE | Avg MFE | WR | AvgR | SumR |")
    W("|--------|:-:|:-------:|:-------:|:--:|:----:|:----:|")
    for label, trades in [("B-unlocked (2.0–2.49R)", all_b_unlocked),
                           ("C-unlocked (1.5–1.99R)", all_c_unlocked)]:
        n = len(trades)
        if n == 0:
            W(f"| {label} | 0 | — | — | — | — | — |")
            continue
        mae = _avg_mae(trades)
        mfe = _avg_mfe(trades)
        W(f"| {label} | {n} "
          f"| {_fmt_r(mae)} "
          f"| {_fmt_r(mfe)} "
          f"| {_fmt_wr(_wr(trades))} "
          f"| {_fmt_r(_avg_r(trades))} "
          f"| {_fmt_r(_sum_r(trades))} |")
    W("")

    # ── Stop-width inflation check ────────────────────────────────────────
    W("## Stop-Width Inflation Check")
    W("")
    W("Trades where avg_MAE < avg_MFE suggest the stop was correctly placed.")
    W("Trades where avg_MAE ≈ avg_MFE suggest the stop may be too wide (inflated).")
    W("If most unlocked trades have MAE close to −1R and MFE > 1R, stop is fine.")
    W("If MAE ≈ −1R and MFE ≈ 0R, the stop is eating into natural excursion.")
    W("")
    for label, trades in [("B-unlocked", all_b_unlocked),
                           ("C-unlocked", all_c_unlocked)]:
        if not trades:
            W(f"- **{label}**: no trades")
            continue
        mae = _avg_mae(trades) or 0.0
        mfe = _avg_mfe(trades) or 0.0
        spread = mfe - abs(mae)
        verdict = "✅ stop OK (MFE >> |MAE|)" if spread > 0.3 else ("⚠️ tight (MFE ≈ |MAE|)" if spread > 0 else "❌ inflated stop (|MAE| > MFE)")
        W(f"- **{label}**: avg MAE={_fmt_r(mae)}  avg MFE={_fmt_r(mfe)}  spread={_fmt_r(spread)}  → {verdict}")
    W("")

    # ── Total R comparison ────────────────────────────────────────────────
    W("## Total R Captured vs Baseline")
    W("")
    W("| Window | A SumR | B SumR | B vs A | C SumR | C vs A |")
    W("|--------|:------:|:------:|:------:|:------:|:------:|")
    for tri in triples:
        ar = tri.result_a.total_r
        br = tri.result_b.total_r
        cr = tri.result_c.total_r
        b_delta = _fmt_r(br - ar)
        c_delta = _fmt_r(cr - ar)
        W(f"| {tri.window} | {_fmt_r(ar)} | {_fmt_r(br)} | {b_delta} | {_fmt_r(cr)} | {c_delta} |")

    total_a = sum(tri.result_a.total_r for tri in triples if tri.result_a.result)
    total_b = sum(tri.result_b.total_r for tri in triples if tri.result_b.result)
    total_c = sum(tri.result_c.total_r for tri in triples if tri.result_c.result)
    W(f"| **TOTAL** | **{_fmt_r(total_a)}** | **{_fmt_r(total_b)}** | **{_fmt_r(total_b - total_a)}** | **{_fmt_r(total_c)}** | **{_fmt_r(total_c - total_a)}** |")
    W("")

    # ── Verdict ───────────────────────────────────────────────────────────
    W("## Verdict")
    W("")

    b_unlocked_wr = _wr(all_b_unlocked)
    c_unlocked_wr = _wr(all_c_unlocked)
    b_unlocked_sum = _sum_r(all_b_unlocked)
    c_unlocked_sum = _sum_r(all_c_unlocked)
    degraded_b = sum(1 for tri in triples
                     if tri.result_b.result and tri.result_b.ret < tri.result_a.ret - 2.0)
    degraded_c = sum(1 for tri in triples
                     if tri.result_c.result and tri.result_c.ret < tri.result_a.ret - 2.0)

    W(f"- **B (2.0R)**: {len(all_b_unlocked)} new trades unlocked, "
      f"{_fmt_wr(b_unlocked_wr)} WR, {_fmt_r(b_unlocked_sum)} SumR, "
      f"{degraded_b} degraded windows (>2% below A)")
    W(f"- **C (1.5R)**: {len(all_c_unlocked)} new trades unlocked (on top of B), "
      f"{_fmt_wr(c_unlocked_wr)} WR, {_fmt_r(c_unlocked_sum)} SumR, "
      f"{degraded_c} degraded windows (>2% below A)")
    W("")
    W("Promote B if: B unlocked WR ≥ 50%, SumR positive, 0 degraded windows.")
    W("Promote C if: C unlocked WR ≥ 50%, SumR positive, 0 degraded windows (on top of B).")
    W("Do NOT promote if: any degraded window or unlocked WR < 40%.")
    W("")
    W("*No promotion from this script. Results inform the next decision.*")

    return "\n".join(lines)


def _write_unlock_section(
    lines: List[str],
    unlocked: List[dict],
    label: str,
    triples: List[WindowTriple],
    var_from: str,
    var_to: str,
) -> None:
    W = lines.append
    n = len(unlocked)
    if n == 0:
        W(f"  No trades unlocked ({label}).")
        return
    W(f"Total unlocked ({label}): **{n}** trades")
    W(f"- WR: {_fmt_wr(_wr(unlocked))}")
    W(f"- AvgR: {_fmt_r(_avg_r(unlocked))}")
    W(f"- SumR: {_fmt_r(_sum_r(unlocked))}")
    W(f"- Avg MAE: {_fmt_r(_avg_mae(unlocked))}")
    W(f"- Avg MFE: {_fmt_r(_avg_mfe(unlocked))}")
    W("")
    W("| Entry | Pair | Dir | Pattern | R | MAE | MFE |")
    W("|-------|------|-----|---------|:-:|:---:|:---:|")
    for t in sorted(unlocked, key=lambda x: x.get("entry_ts",""))[:20]:
        W(f"| {str(t.get('entry_ts',''))[:10]} "
          f"| {t.get('pair','')} "
          f"| {t.get('direction','')} "
          f"| {t.get('pattern','?')} "
          f"| {_fmt_r(t.get('r'))} "
          f"| {_fmt_r(t.get('mae_r'))} "
          f"| {_fmt_r(t.get('mfe_r'))} |")
    if n > 20:
        W(f"*… {n-20} more rows omitted*")


def _write_dist_table(
    lines: List[str],
    dist: Dict[str, int],
    trades: List[dict],
    key: str = "pattern",
) -> None:
    W = lines.append
    if not dist:
        W("  (no trades)")
        return
    total = len(trades)
    W(f"| {key.title()} | Count | % | WR | AvgR |")
    W(f"|{'------'*(1 if key=='pair' else 2)}|:-----:|:-:|:--:|:----:|")
    for name, cnt in list(dist.items())[:10]:
        subset = [t for t in trades if (t.get(key) or t.get("pattern_type","")) == name]
        pct = cnt / total * 100
        W(f"| {name} | {cnt} | {pct:.0f}% | {_fmt_wr(_wr(subset))} | {_fmt_r(_avg_r(subset))} |")


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MIN_RR ablation study — offline only")
    parser.add_argument("--windows",  nargs="+", help="Filter windows by name")
    parser.add_argument("--variants", nargs="+", help="Filter variants by id (A B C)")
    parser.add_argument("--out",      default="", help="Output file path (default: auto)")
    args = parser.parse_args()

    print("MIN_RR Ablation Study")
    print(f"B-Prime trigger active: {_sc.ENTRY_TRIGGER_MODE}")
    print(f"Whitelist: {_sc.STRICT_PIN_PATTERN_WHITELIST}")
    print(f"Pairs: backtester WATCHLIST (Alex 7 via whitelist_backtest.json when enabled)")
    print(f"Variants: {[v[0] for v in VARIANTS]}")
    print(f"Windows:  {[w[0] for w in WINDOWS]}")
    print()

    triples = run_ablation(
        windows_filter=args.windows,
        variants_filter=args.variants,
        verbose=True,
    )

    report = _build_report(triples)

    out_path = args.out
    if not out_path:
        results_dir = REPO / "backtesting" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(results_dir / "ablation_min_rr.md")

    Path(out_path).write_text(report)
    print(f"\nReport written → {out_path}")
    print("\n" + "="*60)
    # Print summary table to stdout
    for line in report.split("\n"):
        if line.startswith("| ") and ("Window" in line or "TOTAL" in line or
                                       any(w[0] in line for w in WINDOWS)):
            print(line)


if __name__ == "__main__":
    main()
