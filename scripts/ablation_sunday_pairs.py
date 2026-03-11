#!/usr/bin/env python3
"""
Sunday London + Watchlist Expansion Ablation — offline replay only.

Goal
----
  1. Does opening Sunday 03:00–09:00 ET (London window) add quality setups?
  2. Does expanding to 26 pairs fill signal gaps, especially Q3 2025?
  3. Combined effect of both changes?

Variants
--------
  A  Baseline        12 pairs   Sunday blocked   (production intent)
  B  Sunday London   12 pairs   Sunday 03:00–09:00 ET allowed
  C  Pairs++         26 pairs   Sunday blocked
  D  Both changes    26 pairs   Sunday 03:00–09:00 ET allowed

12 current pairs:
  USD/JPY, GBP/CHF, USD/CHF, USD/CAD, GBP/JPY, EUR/USD, GBP/USD,
  NZD/USD, GBP/NZD, EUR/GBP, AUD/USD, NZD/JPY

26 total pairs (12 + 14 new):
  + EUR/JPY, EUR/CHF, EUR/CAD, EUR/AUD, EUR/NZD, GBP/AUD, GBP/CAD,
    AUD/JPY, AUD/CAD, AUD/CHF, AUD/NZD, CAD/JPY, CHF/JPY, NZD/CAD

Safety
------
  atexit() restores whitelist JSON, WATCHLIST, and SessionFilter.is_hard_blocked.
  No strategy_config patches — NO_SUNDAY_TRADES_ENABLED not touched.
"""

from __future__ import annotations

import atexit
import json
import sys
from collections import defaultdict
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

import pytz
from src.strategy.forex.session_filter import SessionFilter
import src.strategy.forex.strategy_config as _sc

ET = pytz.timezone("America/New_York")

# ── backtester import ─────────────────────────────────────────────────────────
import backtesting.oanda_backtest_v2 as _bt
from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult  # noqa

# ──────────────────────────────────────────────────────────────────────────────
WHITELIST_PATH = REPO / "logs" / "whitelist_backtest.json"
REPORT_PATH    = REPO / "backtesting/results/ablation_sunday_pairs.md"
UTC = timezone.utc

PAIRS_12: List[str] = [
    "USD/JPY", "GBP/CHF", "USD/CHF", "USD/CAD", "GBP/JPY",
    "EUR/USD", "GBP/USD", "NZD/USD", "GBP/NZD", "EUR/GBP",
    "AUD/USD", "NZD/JPY",
]

PAIRS_NEW_14: List[str] = [
    "EUR/JPY", "EUR/CHF", "EUR/CAD", "EUR/AUD", "EUR/NZD",
    "GBP/AUD", "GBP/CAD",
    "AUD/JPY", "AUD/CAD", "AUD/CHF", "AUD/NZD",
    "CAD/JPY", "CHF/JPY", "NZD/CAD",
]

PAIRS_26: List[str] = sorted(set(PAIRS_12 + PAIRS_NEW_14))

WINDOWS: List[Tuple[str, datetime, datetime]] = [
    ("Q1-2025",      datetime(2025, 1, 1,  tzinfo=UTC), datetime(2025, 3, 31, tzinfo=UTC)),
    ("Q2-2025",      datetime(2025, 4, 1,  tzinfo=UTC), datetime(2025, 6, 30, tzinfo=UTC)),
    ("Q3-2025",      datetime(2025, 7, 1,  tzinfo=UTC), datetime(2025, 9, 30, tzinfo=UTC)),
    ("Q4-2025",      datetime(2025, 10, 1, tzinfo=UTC), datetime(2025, 12, 31, tzinfo=UTC)),
    ("Jan-Feb-2026", datetime(2026, 1, 1,  tzinfo=UTC), datetime(2026, 2, 28, tzinfo=UTC)),
    ("W1",           datetime(2026, 2, 17, tzinfo=UTC), datetime(2026, 2, 21, tzinfo=UTC)),
    ("W2",           datetime(2026, 2, 24, tzinfo=UTC), datetime(2026, 2, 28, tzinfo=UTC)),
    ("live-parity",  datetime(2026, 3, 2,  tzinfo=UTC), datetime(2026, 3, 8,  tzinfo=UTC)),
]

# (label, pairs_list, sunday_london_allowed, description)
VARIANTS: List[Tuple[str, List[str], bool, str]] = [
    ("A", PAIRS_12, False, "Baseline — 12 pairs, Sunday blocked"),
    ("B", PAIRS_12, True,  "Sunday London — 12 pairs, Sun 03:00–09:00 ET"),
    ("C", PAIRS_26, False, "Pairs++ — 26 pairs, Sunday blocked"),
    ("D", PAIRS_26, True,  "Both — 26 pairs, Sun 03:00–09:00 ET"),
]

# ── atexit guard ──────────────────────────────────────────────────────────────
_ORIG_WHITELIST_DATA  = json.loads(WHITELIST_PATH.read_text()) if WHITELIST_PATH.exists() else None
_ORIG_WATCHLIST       = list(_bt.WATCHLIST)
_ORIG_IS_HARD_BLOCKED = SessionFilter.is_hard_blocked

def _restore_all():
    SessionFilter.is_hard_blocked = _ORIG_IS_HARD_BLOCKED
    _bt.WATCHLIST[:] = _ORIG_WATCHLIST
    if _ORIG_WHITELIST_DATA is not None:
        WHITELIST_PATH.write_text(json.dumps(_ORIG_WHITELIST_DATA, indent=2))
    elif WHITELIST_PATH.exists():
        WHITELIST_PATH.unlink()

atexit.register(_restore_all)


# ──────────────────────────────────────────────────────────────────────────────
# Patching helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_sunday_patch():
    """Allow Sunday 03:00–09:00 ET; keep all other Sunday hours blocked."""
    _orig = _ORIG_IS_HARD_BLOCKED

    def _patched(self, dt=None):
        if dt is None:
            dt = self.now_et()
        dt_et = dt.astimezone(ET)
        weekday = dt_et.weekday()
        t       = dt_et.time()
        if weekday == 6 and time(3, 0) <= t < time(9, 0):
            return False, ""   # London window — allow
        return _orig(self, dt)

    SessionFilter.is_hard_blocked = _patched


def _remove_sunday_patch():
    SessionFilter.is_hard_blocked = _ORIG_IS_HARD_BLOCKED


def _apply_pairs(pairs: List[str]):
    """Write whitelist JSON and ensure all pairs are in WATCHLIST."""
    # Extend WATCHLIST with any new pairs
    existing = set(_bt.WATCHLIST)
    for p in pairs:
        if p not in existing:
            _bt.WATCHLIST.append(p)
    # Write whitelist file
    WHITELIST_PATH.write_text(json.dumps({
        "scope":      "backtest",
        "enabled":    True,
        "pairs":      pairs,
        "updated_at": datetime.now(UTC).isoformat(),
        "updated_by": "ablation_sunday_pairs",
        "reason":     f"ablation variant — {len(pairs)} pairs",
    }, indent=2))


def _restore_pairs():
    _bt.WATCHLIST[:] = _ORIG_WATCHLIST
    if _ORIG_WHITELIST_DATA is not None:
        WHITELIST_PATH.write_text(json.dumps(_ORIG_WHITELIST_DATA, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_sunday_trade(t: Dict) -> bool:
    ts = t.get("entry_ts")
    if not ts:
        return False
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    et = ts.astimezone(ET)
    return et.weekday() == 6


def _trade_key(t: Dict) -> str:
    ts = t.get("entry_ts") or ""
    ts_s = str(ts)[:13].replace("-","").replace("T","").replace(":","")
    return f"{t.get('pair','?')}_{ts_s}_{t.get('direction','?')}"


def _new_pair_trades(result_a: BacktestResult, result_x: BacktestResult) -> List[Dict]:
    """Trades in result_x on pairs NOT in PAIRS_12."""
    return [t for t in result_x.trades if t.get("pair") not in set(PAIRS_12)]


def _find_unlocked(result_a: BacktestResult, result_x: BacktestResult) -> List[Dict]:
    keys_a = {_trade_key(t) for t in result_a.trades}
    return [t for t in result_x.trades if _trade_key(t) not in keys_a]


def _find_displaced(result_a: BacktestResult, result_x: BacktestResult) -> List[Dict]:
    keys_x = {_trade_key(t) for t in result_x.trades}
    return [t for t in result_a.trades if _trade_key(t) not in keys_x]


def _r(v) -> str:
    return "—" if v is None else f"{v:+.2f}R"


def _wl(t: Dict) -> str:
    return "✅ W" if t.get("r", 0) > 0 else "❌ L"


def _fmt_ts(ts) -> str:
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.astimezone(ET).strftime("%a %Y-%m-%d %H:%M ET")
        except Exception:
            return ts[:16]
    return str(ts)[:16] if ts else "—"


def _pair_quality(trades: List[Dict]) -> Dict[str, Dict]:
    """Per-pair aggregate: count, wins, sumR."""
    out: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "sumr": 0.0})
    for t in trades:
        p = t.get("pair", "?")
        out[p]["n"]    += 1
        out[p]["sumr"] += t.get("r", 0)
        if t.get("r", 0) > 0:
            out[p]["wins"] += 1
    return dict(out)


def _htf_aligned(t: Dict) -> str:
    """Try to extract HTF alignment from trade notes — return Yes/No/?"""
    # The trade dict doesn't carry HTF info directly; mark as unknown
    return "?"


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation(verbose: bool = False) -> Dict[str, Dict[str, BacktestResult]]:
    results: Dict[str, Dict[str, BacktestResult]] = {}

    for win_name, win_start, win_end in WINDOWS:
        results[win_name] = {}
        preloaded_12: Optional[Dict] = None
        preloaded_26: Optional[Dict] = None

        print(f"\n  ── Window: {win_name} ──────────────────────────────")
        for vlabel, pairs, sunday_on, _ in VARIANTS:
            # Apply patches
            _apply_pairs(pairs)
            if sunday_on:
                _apply_sunday_patch()
            else:
                _remove_sunday_patch()

            # Choose preloaded cache bucket by pair count
            preloaded = preloaded_26 if len(pairs) > 12 else preloaded_12

            print(f"    Variant {vlabel} ({len(pairs)}p, sun={'on' if sunday_on else 'off'})…",
                  end=" ", flush=True)
            result = run_backtest(
                win_start, win_end,
                starting_bal=8_000.0,
                notes=f"sun_pairs_{vlabel}_{win_name}",
                trail_arm_key="A",
                preloaded_candle_data=preloaded,
                use_cache=True,
                quiet=not verbose,
            )
            results[win_name][vlabel] = result

            # Cache per pair-set
            if len(pairs) > 12:
                if preloaded_26 is None and result.candle_data:
                    preloaded_26 = result.candle_data
            else:
                if preloaded_12 is None and result.candle_data:
                    preloaded_12 = result.candle_data

            sumr = sum(t.get("r", 0) for t in result.trades)
            n_sun = sum(1 for t in result.trades if _is_sunday_trade(t))
            n_new = len(_new_pair_trades(result, result)) if vlabel in ("C","D") else 0
            if vlabel in ("C","D"):
                n_new = len(_new_pair_trades(result, result))
            print(f"{result.n_trades}T  SumR={sumr:+.2f}R  sun={n_sun}  newpairs={n_new}")

        # Reset after window
        _restore_all()
        atexit.register(_restore_all)  # re-register after _restore_all unregisters implicitly

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(results: Dict[str, Dict[str, BacktestResult]]) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    L = lines.append

    L("# Sunday London + Pairs Expansion Ablation")
    L(f"\nGenerated: {now}  |  Branch: `feat/sunday-pairs-ablation`")
    L("")
    L("## Setup")
    L("")
    L("| | |")
    L("|---|---|")
    L("| Capital | $8,000 |")
    L("| Stop | C8 (structural + 3×ATR_1H + 8-pip floor) |")
    L("| Trigger | `engulf_or_strict_pin_at_level` (B-Prime) |")
    L("| MIN_RR | 2.5 | MIN_CONFIDENCE | 0.77 | Weekly cap | 1 |")
    L("")
    L("### Variants")
    L("")
    L("| Variant | Pairs | Sunday | Description |")
    L("|---------|:-----:|:------:|-------------|")
    for vlabel, pairs, sun, desc in VARIANTS:
        prod = " **(baseline)**" if vlabel == "A" else ""
        sun_s = "03:00–09:00 ET" if sun else "blocked"
        L(f"| **{vlabel}** | {len(pairs)}{prod} | {sun_s} | {desc} |")
    L("")
    L("### New 14 pairs (Variants C & D)")
    L("")
    L(", ".join(f"`{p}`" for p in PAIRS_NEW_14))
    L("")

    # ── 1. Per-window ──────────────────────────────────────────────────────────
    L("## 1. Per-Window Breakdown")
    L("")
    L("| Window | Var | T | WR | SumR | AvgR | MaxDD | WkBlk | Worst3L |")
    L("|--------|-----|:--:|:--:|:----:|:----:|:-----:|:-----:|---------|")

    agg: Dict[str, Dict] = {v[0]: {"trades": [], "sumr": 0.0, "dd": []} for v in VARIANTS}

    for win_name, _, _ in WINDOWS:
        for vi, (vlabel, pairs, sun, _) in enumerate(VARIANTS):
            r  = results[win_name][vlabel]
            ts = r.trades
            sumr = sum(t.get("r", 0) for t in ts)
            avgr = sumr / len(ts) if ts else 0.0
            dd   = r.max_dd_pct or 0.0
            worst = sorted(t.get("r", 0) for t in ts)[:3]
            w3   = ", ".join(f"{x:+.2f}R" for x in worst) if worst else "—"
            L(f"| {win_name if vi == 0 else ''} | {vlabel} "
              f"| {len(ts)} | {r.win_rate*100:.0f}% "
              f"| {sumr:+.2f}R | {avgr:+.3f}R "
              f"| {dd:.1f}% | {r.weekly_limit_blocks} | {w3} |")
            agg[vlabel]["trades"].extend(ts)
            agg[vlabel]["sumr"]  += sumr
            agg[vlabel]["dd"].append(dd)
        L("| | | | | | | | | |")
    L("")

    # ── 2. Aggregate summary ───────────────────────────────────────────────────
    L("## 2. Aggregate Summary")
    L("")
    L("| Variant | Pairs | Sunday | Trades | WR | SumR | vs A | Avg MaxDD |")
    L("|---------|:-----:|:------:|:------:|:--:|:----:|:----:|:---------:|")
    a_sumr = agg["A"]["sumr"]
    for vlabel, pairs, sun, _ in VARIANTS:
        a  = agg[vlabel]
        ts = a["trades"]
        s  = a["sumr"]
        wr = sum(1 for t in ts if t.get("r", 0) > 0) / len(ts) * 100 if ts else 0.0
        ad = sum(a["dd"]) / len(a["dd"]) if a["dd"] else 0.0
        vs = "—" if vlabel == "A" else f"{s - a_sumr:+.2f}R"
        sun_s = "✅" if sun else "—"
        L(f"| **{vlabel}** | {len(pairs)} | {sun_s} | {len(ts)} | {wr:.0f}% "
          f"| {s:+.2f}R | {vs} | {ad:.1f}% |")
    L("")

    # ── 3. Sunday trade analysis ───────────────────────────────────────────────
    L("## 3. Sunday Trade Analysis (Variants B and D)")
    L("")

    for vlabel, pairs, sun, _ in VARIANTS:
        if not sun:
            continue
        all_sun = [t for win_name, _, _ in WINDOWS
                   for t in results[win_name][vlabel].trades
                   if _is_sunday_trade(t)]
        if not all_sun:
            L(f"### Variant {vlabel} — No Sunday trades across all 8 windows.")
            continue
        sun_sumr = sum(t.get("r", 0) for t in all_sun)
        sun_wr   = sum(1 for t in all_sun if t.get("r", 0) > 0) / len(all_sun) * 100
        L(f"### Variant {vlabel} — {len(all_sun)} Sunday trade(s) total")
        L(f"WR={sun_wr:.0f}%  SumR={sun_sumr:+.2f}R  AvgR={sun_sumr/len(all_sun):+.3f}R")
        L("")
        L("| Window | Pair | Pattern | Dir | Entry (ET) | Session | R | MAE | MFE | W/L |")
        L("|--------|------|---------|-----|------------|:-------:|:--:|:---:|:---:|:---:|")
        for win_name, _, _ in WINDOWS:
            for t in results[win_name][vlabel].trades:
                if not _is_sunday_trade(t):
                    continue
                sf = SessionFilter()
                ts = t.get("entry_ts", "")
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z","+00:00"))
                    sess, qual = sf.session_quality(dt)
                    sess_s = f"{sess}({qual:.1f})"
                except Exception:
                    sess_s = "?"
                L(f"| {win_name} | {t.get('pair','?')} | {t.get('pattern','?')} "
                  f"| {t.get('direction','?')} | {_fmt_ts(ts)} "
                  f"| {sess_s} | {t.get('r',0):+.2f}R "
                  f"| {_r(t.get('mae_r'))} | {_r(t.get('mfe_r'))} | {_wl(t)} |")
        L("")

    # ── 4. New pair contribution (C and D) ─────────────────────────────────────
    L("## 4. New Pair Contribution Analysis (Variants C and D)")
    L("")

    for vlabel, pairs, sun, _ in VARIANTS:
        if len(pairs) <= 12:
            continue
        all_new = [t for win_name, _, _ in WINDOWS
                   for t in results[win_name][vlabel].trades
                   if t.get("pair") not in set(PAIRS_12)]
        all_existing = [t for win_name, _, _ in WINDOWS
                        for t in results[win_name][vlabel].trades
                        if t.get("pair") in set(PAIRS_12)]

        L(f"### Variant {vlabel} — New pair trades vs existing pair trades")
        L("")
        if all_new:
            new_sumr = sum(t.get("r",0) for t in all_new)
            new_wr   = sum(1 for t in all_new if t.get("r",0) > 0) / len(all_new) * 100
            L(f"**New pairs ({len(PAIRS_NEW_14)} pairs)**: {len(all_new)} trades, "
              f"WR={new_wr:.0f}%, SumR={new_sumr:+.2f}R")
        else:
            L(f"**New pairs ({len(PAIRS_NEW_14)} pairs)**: 0 trades")
        if all_existing:
            ex_sumr = sum(t.get("r",0) for t in all_existing)
            ex_wr   = sum(1 for t in all_existing if t.get("r",0) > 0) / len(all_existing) * 100
            L(f"**Existing 12 pairs**: {len(all_existing)} trades, "
              f"WR={ex_wr:.0f}%, SumR={ex_sumr:+.2f}R")
        L("")

        if all_new:
            pq = _pair_quality(all_new)
            L("#### Per-new-pair breakdown")
            L("")
            L("| Pair | Trades | WR | SumR | AvgR |")
            L("|------|:------:|:--:|:----:|:----:|")
            for pair, d in sorted(pq.items(), key=lambda x: -x[1]["n"]):
                wr_p = d["wins"] / d["n"] * 100 if d["n"] else 0.0
                ar_p = d["sumr"] / d["n"] if d["n"] else 0.0
                L(f"| {pair} | {d['n']} | {wr_p:.0f}% | {d['sumr']:+.2f}R | {ar_p:+.3f}R |")
            L("")

            L("#### New pair trade detail")
            L("")
            L("| Window | Pair | Pattern | Dir | Entry (ET) | R | MAE | MFE | W/L |")
            L("|--------|------|---------|-----|------------|:--:|:---:|:---:|:---:|")
            for win_name, _, _ in WINDOWS:
                for t in results[win_name][vlabel].trades:
                    if t.get("pair") not in set(PAIRS_12):
                        L(f"| {win_name} | {t.get('pair','?')} | {t.get('pattern','?')} "
                          f"| {t.get('direction','?')} | {_fmt_ts(t.get('entry_ts',''))} "
                          f"| {t.get('r',0):+.2f}R "
                          f"| {_r(t.get('mae_r'))} | {_r(t.get('mfe_r'))} | {_wl(t)} |")
            L("")

    # ── 5. Q3 dead zone ────────────────────────────────────────────────────────
    L("## 5. Q3 2025 Dead Zone — Does Expansion Help?")
    L("")
    L("| Variant | Trades | WR | SumR | Sunday? | New pairs? |")
    L("|---------|:------:|:--:|:----:|:-------:|:----------:|")
    for vlabel, pairs, sun, _ in VARIANTS:
        r   = results["Q3-2025"][vlabel]
        ts  = r.trades
        sumr = sum(t.get("r",0) for t in ts)
        wr   = r.win_rate * 100
        n_sun = sum(1 for t in ts if _is_sunday_trade(t))
        n_new = sum(1 for t in ts if t.get("pair") not in set(PAIRS_12))
        L(f"| {vlabel} | {len(ts)} | {wr:.0f}% | {sumr:+.2f}R "
          f"| {n_sun} | {n_new} |")
    L("")

    # ── 6. Cascade displacement ────────────────────────────────────────────────
    L("## 6. Cascade Displacement Analysis")
    L("")
    for vlabel, pairs, sun, _ in VARIANTS:
        if vlabel == "A":
            continue
        L(f"### Variant {vlabel} vs A")
        L("")
        L("| Window | Displaced | Disp R | Replacement | Repl R | Net ΔR |")
        L("|--------|-----------|:------:|-------------|:------:|:------:|")
        total_delta = 0.0
        any_row = False
        for win_name, _, _ in WINDOWS:
            r_a = results[win_name]["A"]
            r_x = results[win_name][vlabel]
            unlocked  = _find_unlocked(r_a, r_x)
            displaced = _find_displaced(r_a, r_x)
            for disp, repl in zip(displaced, unlocked[:len(displaced)]):
                dr = disp.get("r", 0)
                rr = repl.get("r", 0)
                delta = rr - dr
                total_delta += delta
                L(f"| {win_name} | {disp.get('pair','?')} {disp.get('pattern','?')}"
                  f" | {dr:+.2f}R | {repl.get('pair','?')} {repl.get('pattern','?')}"
                  f" | {rr:+.2f}R | {delta:+.2f}R |")
                any_row = True
        if not any_row:
            L("| (none) | — | — | — | — | — |")
        L("")
        L(f"**Net displacement: {total_delta:+.2f}R**")
        L("")

    # ── 7. ATR floor ───────────────────────────────────────────────────────────
    L("## 7. ATR Floor Check (C8 8-pip minimum)")
    L("")
    L("| Window | A | B | C | D |")
    L("|--------|:-:|:-:|:-:|:-:|")
    total_v = {v[0]: 0 for v in VARIANTS}
    for win_name, _, _ in WINDOWS:
        row = f"| {win_name}"
        for vlabel, _, _, _ in VARIANTS:
            r  = results[win_name][vlabel]
            vv = sum(1 for t in r.trades if (t.get("initial_stop_pips") or 999) < 8.0)
            total_v[vlabel] += vv
            row += f" | {vv}{'✅' if vv==0 else '⚠️'}"
        L(row + " |")
    t_row = "| **Total**"
    for vlabel, _, _, _ in VARIANTS:
        vv = total_v[vlabel]
        t_row += f" | **{vv}{'✅' if vv==0 else '⚠️'}**"
    L(t_row + " |")
    L("")

    # ── 8. Verdict ─────────────────────────────────────────────────────────────
    L("## 8. Verdict")
    L("")
    L("### Summary")
    L("")
    L("| Variant | Trades | SumR | vs A | Sunday T | New Pair T |")
    L("|---------|:------:|:----:|:----:|:--------:|:----------:|")
    for vlabel, pairs, sun, _ in VARIANTS:
        a   = agg[vlabel]
        ts  = a["trades"]
        s   = a["sumr"]
        vs  = "—" if vlabel == "A" else f"{s - a_sumr:+.2f}R"
        n_sun = sum(1 for t in ts if _is_sunday_trade(t))
        n_new = sum(1 for t in ts if t.get("pair") not in set(PAIRS_12))
        L(f"| {vlabel} | {len(ts)} | {s:+.2f}R | {vs} | {n_sun} | {n_new} |")
    L("")
    L("### Per-variant decision")
    L("")
    for vlabel, pairs, sun, _ in VARIANTS:
        if vlabel == "A":
            L(f"**A (baseline)**: {a_sumr:+.2f}R — production.")
            continue
        a  = agg[vlabel]
        s  = a["sumr"]
        delta = s - a_sumr
        ts = a["trades"]
        wr = sum(1 for t in ts if t.get("r",0)>0) / len(ts) * 100 if ts else 0.0
        if delta > 0.5:
            verb = f"CONSIDER — +{delta:.2f}R improvement. Review unlocked trade quality."
        elif -0.5 <= delta <= 0.5:
            verb = f"NEUTRAL — {delta:+.2f}R. No clear signal."
        else:
            verb = f"REJECT — {delta:+.2f}R regression."
        L(f"**{vlabel}** ({pairs[0][:3]}… {len(pairs)}p, sun={'on' if sun else 'off'}): {verb}")
        L("")

    L("_Report generated by `scripts/ablation_sunday_pairs.py`._")
    L("_Offline replay only — no live changes, no master merge._")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  SUNDAY LONDON + PAIRS EXPANSION ABLATION")
    print("  Offline replay — no live changes")
    print("═"*60)
    print(f"  Variants: {len(VARIANTS)} | Windows: {len(WINDOWS)} | Runs: {len(VARIANTS)*len(WINDOWS)}")
    print(f"  12-pair baseline → 26-pair expansion ({len(PAIRS_NEW_14)} new pairs)")

    try:
        results = run_ablation(verbose=args.verbose)
        print("\n  Generating report…")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        report = generate_report(results)
        REPORT_PATH.write_text(report)
        print(f"  Report → {REPORT_PATH}")
        print("\n✅  Done.")
    finally:
        _restore_all()
        print(f"  Whitelist restored: {[p for p in json.loads(WHITELIST_PATH.read_text()).get('pairs',[])]}")
        print(f"  Sunday patch: {'active' if SessionFilter.is_hard_blocked is not _ORIG_IS_HARD_BLOCKED else 'restored'}")


if __name__ == "__main__":
    main()
