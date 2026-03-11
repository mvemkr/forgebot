#!/usr/bin/env python3
"""
Session filter ablation study — offline replay only.

Part A — Q3 2025 Dead Zone Diagnosis
--------------------------------------
  Runs Q3 2025 (Jul–Sep 2025) with current baseline and analyses ALL block
  reasons from the decision log to determine whether the zero-trade quarter
  was a market condition issue or a filter issue.

Part B — Session Filter Ablation
----------------------------------
  Variants (8 windows × 3 variants = 24 runs):
    A  Baseline      THU_CUTOFF=09h ET,  Mon hard-block ends 08:00 ET (current prod)
    B  Extended Thu  THU_CUTOFF=12h ET,  Mon unchanged
    C  Thu+Mon       THU_CUTOFF=12h ET,  Mon hard-block ends 07:00 ET

  Windows (same 8 as all prior ablation studies):
    Q1-2025, Q2-2025, Q3-2025, Q4-2025,
    Jan-Feb-2026, W1, W2, live-parity

  Everything else locked:
    MIN_CONFIDENCE=0.77, MIN_RR_STANDARD=2.5, C8 structural stop,
    B-Prime trigger, weekly cap, zone-touch, all quality gates.

Safety
------
  atexit() resets THU_ENTRY_CUTOFF_HOUR_ET and SessionFilter.MONDAY_HARD_BLOCK_END
  to production defaults even on crash.
"""

from __future__ import annotations

import atexit
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

import src.strategy.forex.strategy_config as _sc
from src.strategy.forex.session_filter import SessionFilter
import pytz

# ── atexit guard ──────────────────────────────────────────────────────────────
_ORIG_THU_CUTOFF     = getattr(_sc, "THU_ENTRY_CUTOFF_HOUR_ET", 9)
_ORIG_MON_BLOCK_END  = SessionFilter.MONDAY_HARD_BLOCK_END   # time(8, 0)

def _reset_session_config():
    _sc.THU_ENTRY_CUTOFF_HOUR_ET     = _ORIG_THU_CUTOFF
    SessionFilter.MONDAY_HARD_BLOCK_END = _ORIG_MON_BLOCK_END

atexit.register(_reset_session_config)

# ── backtester import ─────────────────────────────────────────────────────────
from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult  # noqa

# ──────────────────────────────────────────────────────────────────────────────
CAPITAL     = 8_000.0
REPORT_PATH = REPO / "backtesting/results/ablation_session_filter.md"
DECISION_LOG = REPO / "logs/backtest_v2_decisions.json"

UTC = timezone.utc
ET  = pytz.timezone("America/New_York")

WINDOWS: List[Tuple[str, datetime, datetime]] = [
    ("Q1-2025",      datetime(2025,1,1,  tzinfo=UTC), datetime(2025,3,31, tzinfo=UTC)),
    ("Q2-2025",      datetime(2025,4,1,  tzinfo=UTC), datetime(2025,6,30, tzinfo=UTC)),
    ("Q3-2025",      datetime(2025,7,1,  tzinfo=UTC), datetime(2025,9,30, tzinfo=UTC)),
    ("Q4-2025",      datetime(2025,10,1, tzinfo=UTC), datetime(2025,12,31,tzinfo=UTC)),
    ("Jan-Feb-2026", datetime(2026,1,1,  tzinfo=UTC), datetime(2026,2,28, tzinfo=UTC)),
    ("W1",           datetime(2026,2,17, tzinfo=UTC), datetime(2026,2,21, tzinfo=UTC)),
    ("W2",           datetime(2026,2,24, tzinfo=UTC), datetime(2026,2,28, tzinfo=UTC)),
    ("live-parity",  datetime(2026,3,2,  tzinfo=UTC), datetime(2026,3,8,  tzinfo=UTC)),
]

# (label, thu_cutoff_hour, mon_block_end_hour, short_desc)
VARIANTS: List[Tuple[str, int, int, str]] = [
    ("A", 9,  8, "Baseline — Thu cutoff 09:00 ET, Mon hard-block ends 08:00 ET"),
    ("B", 12, 8, "Extended Thu — cutoff 12:00 ET (noon), Mon unchanged"),
    ("C", 12, 7, "Thu+Mon — cutoff 12:00 ET + Mon hard-block ends 07:00 ET"),
]

# Flags for unlocked trade diagnostics
_HIGH_CONC_PCT = 0.30   # flag if |trade_r / window_sumr| ≥ 30%
_LOSS_CONF_THR = 0.72   # flag if conf < this AND loss (not applicable here but kept)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _trade_key(t: Dict) -> str:
    ts = t.get("entry_ts") or t.get("open_ts") or ""
    if hasattr(ts, "strftime"):
        ts = ts.strftime("%Y%m%d%H")
    else:
        ts = str(ts)[:13].replace("-","").replace("T","").replace(":","").replace(" ","")
    return f"{t.get('pair','?')}_{ts}_{t.get('direction','?')}"


def _r(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:+.2f}R"
    return str(v)


def _pct(v) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}%"


def _wl(t: Dict) -> str:
    r = t.get("r", 0)
    return "✅ W" if r > 0 else "❌ L"


def _entry_time_et(t: Dict) -> str:
    """Return entry time in ET as HH:MM string."""
    ts = t.get("entry_ts")
    if not ts:
        return "—"
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return "—"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    et = ts.astimezone(ET)
    return et.strftime("%a %H:%M")


def _session_quality_at(ts_str: str) -> Tuple[str, float]:
    """Return session name + quality for a given ISO timestamp."""
    sf = SessionFilter()
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return "?", 0.0
    return sf.session_quality(ts)


def _load_decisions() -> List[Dict]:
    """Load decisions from the last-written backtest_v2_decisions.json."""
    if not DECISION_LOG.exists():
        return []
    with open(DECISION_LOG) as f:
        try:
            data = json.load(f)
        except Exception:
            return []
    return data.get("decisions", [])


def _filter_distribution(decisions: List[Dict]) -> Counter:
    """Count all filter codes across all decisions."""
    counts: Counter = Counter()
    for d in decisions:
        dec = d.get("decision", "")
        if dec == "BLOCKED":
            for f in (d.get("failed_filters") or []):
                counts[f] += 1
            # Also count reason text as fallback
            reason = d.get("reason", "")
            if not d.get("failed_filters") and reason:
                # extract first token of reason as filter code
                code = reason.split(":")[0].strip().replace(" ", "_")
                counts[code] += 1
        elif dec == "WAIT":
            for f in (d.get("failed_filters") or []):
                counts["WAIT_" + f] += 1
    return counts


def _block_breakdown(decisions: List[Dict]) -> Dict[str, int]:
    """High-level block category counts."""
    session_codes  = {"NO_SUNDAY_TRADES","NO_THU_FRI_TRADES","MONDAY_WICK_GUARD",
                      "LOW_QUALITY_SESSION","SESSION_BLOCK","MARKET_CLOSED"}
    htf_codes      = {"HTF_BLOCKED","htf_not_aligned","htf_block"}
    conf_codes     = {"low_confidence","MIN_CONFIDENCE"}
    zone_codes     = {"NO_ZONE_TOUCH","zone_touch"}
    weekly_codes   = {"WEEKLY_TRADE_LIMIT"}
    rr_codes       = {"MIN_RR_ALIGN","MIN_RR","exec_rr_min"}
    theme_codes    = {"theme_direction_conflict","theme_contradiction","THEME_CONFLICT"}
    stop_codes     = {"stop_too_tight","stop_too_wide","DYN_PIP_EQUITY"}
    cooldown_codes = {"STOP_COOLDOWN","stop_cooldown"}
    concurrent_codes = {"MAX_CONCURRENT","open_position"}
    trigger_codes  = {"NO_ENGULFING","engulf","trigger"}

    cats = {
        "session_filter":      session_codes,
        "htf_alignment":       htf_codes,
        "low_confidence":      conf_codes,
        "zone_touch":          zone_codes,
        "weekly_cap":          weekly_codes,
        "min_rr":              rr_codes,
        "theme_conflict":      theme_codes,
        "stop_size":           stop_codes,
        "cooldown":            cooldown_codes,
        "concurrent_limit":    concurrent_codes,
        "trigger":             trigger_codes,
    }

    totals: Dict[str, int] = {k: 0 for k in cats}
    totals["other"] = 0
    total_counted = 0

    for d in decisions:
        filters = list(d.get("failed_filters") or [])
        reason  = d.get("reason", "")
        if d.get("decision") not in ("BLOCKED","WAIT"):
            continue
        matched = False
        for cat, codes in cats.items():
            for f in filters:
                if f in codes or any(c.lower() in f.lower() for c in codes):
                    totals[cat] += 1
                    matched = True
                    break
            if matched:
                break
        if not matched:
            # Try reason text
            for cat, codes in cats.items():
                for code in codes:
                    if code.lower() in reason.lower():
                        totals[cat] += 1
                        matched = True
                        break
                if matched:
                    break
        if not matched:
            totals["other"] += 1
        total_counted += 1

    totals["TOTAL"] = total_counted
    return totals


def _find_unlocked_trades(
    result_a: BacktestResult,
    result_x: BacktestResult,
) -> List[Dict]:
    """Trades in result_x that are NOT in result_a (unlocked by looser filter)."""
    keys_a = {_trade_key(t) for t in result_a.trades}
    return [t for t in result_x.trades if _trade_key(t) not in keys_a]


def _find_displaced_trades(
    result_a: BacktestResult,
    result_x: BacktestResult,
    unlocked: List[Dict],
) -> List[Dict]:
    """Baseline trades in result_a that are NOT in result_x (displaced by unlocked)."""
    keys_x = {_trade_key(t) for t in result_x.trades}
    return [t for t in result_a.trades if _trade_key(t) not in keys_x]


def _run_variant(label: str, thu_cutoff: int, mon_end: int,
                 window_name: str, start: datetime, end: datetime,
                 preloaded: Optional[Dict], verbose: bool = False) -> Tuple[BacktestResult, Optional[Dict]]:
    """Patch session config, run backtest, return (result, preloaded_data)."""
    _sc.THU_ENTRY_CUTOFF_HOUR_ET      = thu_cutoff
    SessionFilter.MONDAY_HARD_BLOCK_END = time(mon_end, 0)

    result = run_backtest(
        start, end,
        starting_bal=CAPITAL,
        notes=f"sess_ablation_{label}_{window_name}",
        trail_arm_key="A",
        preloaded_candle_data=preloaded,
        use_cache=True,
        quiet=not verbose,
    )
    return result, result.candle_data if preloaded is None else preloaded


# ──────────────────────────────────────────────────────────────────────────────
# Q3 Dead Zone Diagnosis
# ──────────────────────────────────────────────────────────────────────────────

def run_q3_diagnosis(verbose: bool = False) -> Dict:
    """
    Run Q3 2025 baseline and return a breakdown of why 0 trades happened.
    Returns dict with: filter_breakdown, total_blocked, top_killers,
    any_patterns_formed, verdict.
    """
    print("\n" + "═"*60)
    print("  PART A — Q3 2025 DEAD ZONE DIAGNOSIS")
    print("═"*60)

    q3_start = datetime(2025, 7, 1, tzinfo=UTC)
    q3_end   = datetime(2025, 9, 30, tzinfo=UTC)

    # Reset to baseline before running
    _sc.THU_ENTRY_CUTOFF_HOUR_ET      = _ORIG_THU_CUTOFF
    SessionFilter.MONDAY_HARD_BLOCK_END = _ORIG_MON_BLOCK_END

    result = run_backtest(
        q3_start, q3_end,
        starting_bal=CAPITAL,
        notes="sess_diagnosis_Q3-2025_baseline",
        trail_arm_key="A",
        use_cache=True,
        quiet=not verbose,
    )

    decisions = _load_decisions()
    breakdown = _block_breakdown(decisions)
    filt_dist = _filter_distribution(decisions)
    top_killers = filt_dist.most_common(10)

    # Count ENTER decisions (made it past evaluate) vs BLOCKED (inside evaluate)
    n_enter_raw  = sum(1 for d in decisions if d.get("decision") == "ENTER")
    n_blocked    = sum(1 for d in decisions if d.get("decision") == "BLOCKED")
    n_wait       = sum(1 for d in decisions if d.get("decision") == "WAIT")
    n_pre_cand   = sum(1 for d in decisions if d.get("event") == "PRE_CANDIDATE")
    n_cand_wait  = sum(1 for d in decisions if "CANDIDATE_WAIT" in (d.get("event","") or ""))

    # Session blocks specifically
    session_blocks = sum(1 for d in decisions
                         for f in (d.get("failed_filters") or [])
                         if f in ("NO_SUNDAY_TRADES","NO_THU_FRI_TRADES",
                                  "MONDAY_WICK_GUARD","LOW_QUALITY_SESSION",
                                  "SESSION_BLOCK","MARKET_CLOSED"))

    # htf blocks
    htf_blocks = sum(1 for d in decisions
                     for f in (d.get("failed_filters") or [])
                     if "htf" in f.lower() or "HTF" in f)

    print(f"\n  Q3 decision summary:")
    print(f"    Total decisions logged:   {len(decisions)}")
    print(f"    BLOCKED (in evaluate):    {n_blocked}")
    print(f"    WAIT (CANDIDATE_WAIT):    {n_wait}")
    print(f"    ENTER (past evaluate):    {n_enter_raw}")
    print(f"    PRE_CANDIDATE events:     {n_pre_cand}")
    print(f"    Session blocks:           {session_blocks}")
    print(f"    HTF blocks:               {htf_blocks}")
    print(f"    Gap log entries:          {len(result.gap_log or [])}")
    print(f"\n  Top block filters:")
    for f, cnt in top_killers[:8]:
        print(f"    {f:<40} {cnt:>4}x")

    # Gap summary from result
    gap_sum = {}
    for g in (result.gap_log or []):
        gt = g.get("gap_type","?")
        gap_sum[gt] = gap_sum.get(gt,0) + 1

    print(f"\n  Gap summary (post-evaluate blocks):")
    for gt, cnt in sorted(gap_sum.items(), key=lambda x: -x[1]):
        print(f"    {gt:<40} {cnt:>4}x")

    # Verdict logic
    if n_enter_raw == 0 and n_blocked > 0:
        # All blocks happened inside evaluate() — session or HTF or confidence
        if session_blocks > htf_blocks and session_blocks > sum(1 for d in decisions
                for f in (d.get("failed_filters") or [])
                if "confidence" in f.lower() or f == "low_confidence"):
            verdict = "FILTER_ISSUE: Session filter is primary killer in Q3"
        elif htf_blocks > session_blocks:
            verdict = "FILTER_ISSUE: HTF alignment gate is primary killer in Q3"
        else:
            verdict = "FILTER_ISSUE: Confidence / multiple gates killing Q3 setups"
    elif n_enter_raw > 0 and result.n_trades == 0:
        # Patterns formed, got to ENTER, then blocked post-evaluate
        top_post = sorted(gap_sum.items(), key=lambda x: -x[1])
        primary  = top_post[0][0] if top_post else "unknown"
        verdict  = f"FILTER_ISSUE: Patterns formed but post-evaluate gate '{primary}' blocked all"
    elif n_blocked == 0 and n_enter_raw == 0:
        verdict = "MARKET_CONDITION: No patterns detected at all (market too choppy / no structure)"
    else:
        verdict = f"MIXED: {n_blocked} blocked in-evaluate, {n_enter_raw} post-evaluate ENTER decisions"

    print(f"\n  VERDICT: {verdict}\n")

    return {
        "result":          result,
        "decisions":       decisions,
        "breakdown":       breakdown,
        "filter_dist":     filt_dist,
        "top_killers":     top_killers,
        "n_enter_raw":     n_enter_raw,
        "n_blocked":       n_blocked,
        "n_wait":          n_wait,
        "session_blocks":  session_blocks,
        "htf_blocks":      htf_blocks,
        "gap_sum":         gap_sum,
        "verdict":         verdict,
        "candle_data":     result.candle_data,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Ablation runner
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation(verbose: bool = False):
    """Run all 8 windows × 3 variants. Returns nested results dict."""
    print("\n" + "═"*60)
    print("  PART B — SESSION FILTER ABLATION (8w × 3v = 24 runs)")
    print("═"*60)

    results: Dict[str, Dict[str, BacktestResult]] = {}   # results[window][variant]

    for win_name, win_start, win_end in WINDOWS:
        results[win_name] = {}
        preloaded: Optional[Dict] = None

        print(f"\n  ── Window: {win_name} ──────────────────────────────")
        for vlabel, thu_cut, mon_end, _ in VARIANTS:
            print(f"    Variant {vlabel} (THU={thu_cut}h, MON={mon_end}h)…", flush=True)
            result, preloaded = _run_variant(
                vlabel, thu_cut, mon_end,
                win_name, win_start, win_end,
                preloaded, verbose=verbose,
            )
            results[win_name][vlabel] = result
            print(f"      → {result.n_trades}T  {result.win_rate:.0f}%WR  "
                  f"SumR={sum(t.get('r',0) for t in result.trades):+.2f}R")

        # Reset after each window
        _sc.THU_ENTRY_CUTOFF_HOUR_ET      = _ORIG_THU_CUTOFF
        SessionFilter.MONDAY_HARD_BLOCK_END = _ORIG_MON_BLOCK_END

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(
    q3_diag:  Dict,
    results:  Dict[str, Dict[str, BacktestResult]],
) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    L = lines.append

    L("# Session Filter Ablation Study")
    L(f"\nGenerated: {now}  |  Branch: `feat/session-filter-ablation`")
    L("")
    L("## Setup")
    L("")
    L("| | |")
    L("|---|---|")
    L("| Capital | $8,000 |")
    L("| Stop logic | C8 (structural + 3×ATR_1H ceiling + 8-pip floor) |")
    L("| Trigger mode | `engulf_or_strict_pin_at_level` (B-Prime) |")
    L("| ENGULF_CONFIRM_LOOKBACK_BARS | 2 |")
    L("| STRICT_PIN_PATTERN_WHITELIST | head_and_shoulders, inverted_head_and_shoulders |")
    L("| MIN_RR_STANDARD | 2.5 |")
    L("| MIN_CONFIDENCE | 0.77 |")
    L("")
    L("### Variants")
    L("")
    L("| Variant | THU_CUTOFF_ET | MON_HARD_BLOCK_END | Description |")
    L("|---------|:-------------:|:------------------:|-------------|")
    for vlabel, thu_cut, mon_end, desc in VARIANTS:
        prod = " (production)" if vlabel == "A" else ""
        L(f"| **{vlabel}** | {thu_cut:02d}:00 | {mon_end:02d}:00{prod} | {desc} |")
    L("")
    L("### Windows")
    L("")
    for w, ws, we in WINDOWS:
        L(f"- **{w}**: {ws.date()} → {we.date()}")
    L("")
    L("---")
    L("")

    # ── Part A: Q3 Dead Zone Diagnosis ──────────────────────────────────────
    L("## 1. Q3 2025 Dead Zone Diagnosis")
    L("")
    L("Q3 2025 (Jul–Sep 2025) produced **0 trades** across every prior ablation study.")
    L("This section diagnoses why.")
    L("")
    diag = q3_diag
    L("### Decision Funnel — Q3 2025 Baseline")
    L("")
    L("| Stage | Count |")
    L("|-------|------:|")
    L(f"| Total decisions logged | {len(diag['decisions'])} |")
    L(f"| BLOCKED (inside evaluate) | {diag['n_blocked']} |")
    L(f"| WAIT / CANDIDATE_WAIT | {diag['n_wait']} |")
    L(f"| ENTER (past evaluate) | {diag['n_enter_raw']} |")
    L(f"| Actual trades | {diag['result'].n_trades} |")
    L("")
    L("### Top Block Filters")
    L("")
    L("| Filter | Count |")
    L("|--------|------:|")
    for f, cnt in diag["top_killers"][:10]:
        L(f"| `{f}` | {cnt} |")
    L("")
    L("### Post-evaluate Block Distribution (Gap Log)")
    L("")
    L("| Gap type | Count |")
    L("|----------|------:|")
    gap_sorted = sorted(diag["gap_sum"].items(), key=lambda x: -x[1])
    for gt, cnt in gap_sorted:
        L(f"| `{gt}` | {cnt} |")
    if not diag["gap_sum"]:
        L("| (none) | — |")
    L("")
    L(f"**Session blocks**: {diag['session_blocks']}  |  "
      f"**HTF blocks**: {diag['htf_blocks']}")
    L("")
    L(f"### Verdict")
    L("")
    L(f"> {diag['verdict']}")
    L("")
    L("---")
    L("")

    # ── Part B: Per-window breakdown ─────────────────────────────────────────
    L("## 2. Per-Window Breakdown")
    L("")
    hdr = ("| Window | Var | T | WR | SumR | AvgR | MaxDD | Worst3L |")
    sep = ("|--------|-----|:--:|:--:|:----:|:----:|:-----:|---------|")
    L(hdr)
    L(sep)

    # Aggregate totals per variant
    agg: Dict[str, Dict] = {v[0]: {"trades":[],"sumr":0.0,"dd_list":[]} for v in VARIANTS}

    for win_name, _, _ in WINDOWS:
        for vi, (vlabel, _, _, _) in enumerate(VARIANTS):
            r = results[win_name][vlabel]
            trades = r.trades
            sumr   = sum(t.get("r",0) for t in trades)
            avgr   = sumr / len(trades) if trades else 0.0
            dd     = r.max_dd_pct or 0.0
            worst3 = sorted([t.get("r",0) for t in trades])[:3]
            w3str  = ", ".join(f"{x:+.2f}R" for x in worst3) if worst3 else "—"

            row = (f"| {win_name if vi==0 else ''} | {vlabel} "
                   f"| {len(trades)} | {r.win_rate:.0f}% "
                   f"| {sumr:+.2f}R | {avgr:+.3f}R "
                   f"| {dd:.1f}% | {w3str} |")
            L(row)

            agg[vlabel]["trades"].extend(trades)
            agg[vlabel]["sumr"]  += sumr
            agg[vlabel]["dd_list"].append(dd)
        L("| | | | | | | | |")

    L("")

    # ── Aggregate summary ────────────────────────────────────────────────────
    L("## 3. Aggregate Summary")
    L("")
    L("| Variant | Threshold | Trades | WR | SumR | AvgR | Avg MaxDD |")
    L("|---------|:---------:|:------:|:--:|:----:|:----:|:---------:|")
    for vlabel, thu_cut, mon_end, _ in VARIANTS:
        a = agg[vlabel]
        ts = a["trades"]
        sumr = a["sumr"]
        wr   = (sum(1 for t in ts if t.get("r",0)>0)/len(ts)*100) if ts else 0.0
        avgr = sumr/len(ts) if ts else 0.0
        avdd = sum(a["dd_list"])/len(a["dd_list"]) if a["dd_list"] else 0.0
        vs_a = ""
        if vlabel != "A":
            delta = sumr - agg["A"]["sumr"]
            vs_a  = f" ({delta:+.2f}R vs A)"
        L(f"| **{vlabel}** | Thu≤{thu_cut}h/Mon≥{mon_end}h | {len(ts)} | {wr:.0f}% "
          f"| {sumr:+.2f}R{vs_a} | {avgr:+.3f}R | {avdd:.1f}% |")
    L("")

    # ── Unlocked trade diagnostics ───────────────────────────────────────────
    L("## 4. Unlocked Trade Analysis")
    L("")

    for vlabel, thu_cut, mon_end, _ in VARIANTS:
        if vlabel == "A":
            continue
        L(f"### Variant {vlabel} vs A — Newly Unlocked Trades")
        L("")
        all_unlocked: List[Dict] = []

        for win_name, _, _ in WINDOWS:
            r_a = results[win_name]["A"]
            r_x = results[win_name][vlabel]
            unlocked = _find_unlocked_trades(r_a, r_x)
            if not unlocked:
                continue
            sumr_x = sum(t.get("r",0) for t in r_x.trades)
            L(f"#### {win_name} — {len(unlocked)} unlocked trade(s)")
            L("")
            L("| Pair | Pattern | Dir | Stop(p) | Entry(ET) | Session | R | MAE | MFE | W/L |")
            L("|------|---------|-----|:-------:|:---------:|:-------:|:--:|:---:|:---:|:---:|")
            for t in unlocked:
                pair    = t.get("pair","?")
                pat     = t.get("pattern","?")
                d       = t.get("direction","?")
                stop    = t.get("initial_stop_pips")
                stop_s  = f"{stop:.0f}p" if stop else "—"
                r_val   = t.get("r", 0.0)
                mae     = t.get("mae_r", None)
                mfe     = t.get("mfe_r", None)
                et_time = _entry_time_et(t)
                ts      = t.get("entry_ts","")
                sess, qual = _session_quality_at(str(ts)) if ts else ("?", 0.0)
                sess_s  = f"{sess}({qual:.1f})"
                L(f"| {pair} | {pat} | {d} | {stop_s} | {et_time} | {sess_s} "
                  f"| {r_val:+.2f}R | {_r(mae)} | {_r(mfe)} | {_wl(t)} |")
                all_unlocked.append(t)
                # Flags
                if sumr_x != 0 and abs(r_val / sumr_x) >= _HIGH_CONC_PCT:
                    L(f"> ⚠️  HIGH_CONCENTRATION: {pair} {pat} = "
                      f"{abs(r_val/sumr_x)*100:.0f}% of window SumR "
                      f"({r_val:+.2f}R / {sumr_x:+.2f}R)")
            L("")

        # Unlocked summary
        if all_unlocked:
            u_sumr = sum(t.get("r",0) for t in all_unlocked)
            u_wr   = sum(1 for t in all_unlocked if t.get("r",0)>0)/len(all_unlocked)*100
            L(f"**All windows — {len(all_unlocked)} unlocked trade(s)**: "
              f"WR={u_wr:.0f}%  SumR={u_sumr:+.2f}R  AvgR={u_sumr/len(all_unlocked):+.3f}R")
        else:
            L(f"No unlocked trades for Variant {vlabel}.")
        L("")

    # ── Cascade displacement ─────────────────────────────────────────────────
    L("## 5. Cascade Displacement Analysis")
    L("")
    L("Baseline (A) trades displaced by weekly cap when unlocked trades consumed the slot.")
    L("")

    for vlabel, _, _, _ in VARIANTS:
        if vlabel == "A":
            continue
        L(f"### Variant {vlabel} displacements vs A")
        L("")
        L("| Window | Displaced trade | Displaced R | Replacement | Replacement R | Net ΔR |")
        L("|--------|----------------|:-----------:|-------------|:-------------:|:------:|")
        total_delta = 0.0
        any_row = False
        for win_name, _, _ in WINDOWS:
            r_a = results[win_name]["A"]
            r_x = results[win_name][vlabel]
            unlocked   = _find_unlocked_trades(r_a, r_x)
            displaced  = _find_displaced_trades(r_a, r_x, unlocked)
            if not displaced:
                continue
            for disp, repl in zip(displaced, unlocked[:len(displaced)]):
                d_r = disp.get("r", 0)
                r_r = repl.get("r", 0)
                delta = r_r - d_r
                total_delta += delta
                disp_desc = f"{disp.get('pair','?')} {disp.get('pattern','?')}"
                repl_desc = f"{repl.get('pair','?')} {repl.get('pattern','?')}"
                L(f"| {win_name} | {disp_desc} | {d_r:+.2f}R "
                  f"| {repl_desc} | {r_r:+.2f}R | {delta:+.2f}R |")
                any_row = True
        if not any_row:
            L("| (none) | — | — | — | — | — |")
        L("")
        L(f"**Net displacement delta (all windows): {total_delta:+.2f}R**")
        L("")

    # ── ATR floor check ──────────────────────────────────────────────────────
    L("## 6. ATR Floor Check (C8 8-pip floor)")
    L("")
    L("| Window | A violations | B violations | C violations |")
    L("|--------|:-----------:|:-----------:|:-----------:|")
    total_viol = {"A": 0, "B": 0, "C": 0}
    for win_name, _, _ in WINDOWS:
        row_parts = [f"| {win_name}"]
        for vlabel, _, _, _ in VARIANTS:
            r  = results[win_name][vlabel]
            vv = sum(1 for t in r.trades if (t.get("initial_stop_pips") or 999) < 8.0)
            total_viol[vlabel] += vv
            row_parts.append(f" {vv} {'✅' if vv==0 else '⚠️'}")
        L(" |".join(row_parts) + " |")
    row_t = "| **Total**"
    for vlabel, _, _, _ in VARIANTS:
        vv = total_viol[vlabel]
        row_t += f" | **{vv} {'✅' if vv==0 else '⚠️'}**"
    L(row_t + " |")
    L("")

    # ── Verdict ──────────────────────────────────────────────────────────────
    L("## 7. Verdict")
    L("")
    a_sumr = agg["A"]["sumr"]
    rows_v = []
    for vlabel, thu_cut, mon_end, _ in VARIANTS:
        a = agg[vlabel]
        ts = a["trades"]
        sumr = a["sumr"]
        delta = sumr - a_sumr if vlabel != "A" else 0.0
        rows_v.append((vlabel, thu_cut, mon_end, len(ts), sumr, delta))

    L("### Summary comparison")
    L("")
    L("| Variant | Thu/Mon config | Trades | SumR | vs A |")
    L("|---------|:---------------:|:------:|:----:|:----:|")
    for vlabel, thu_cut, mon_end, n, sumr, delta in rows_v:
        vs_str = "—" if vlabel == "A" else f"{delta:+.2f}R"
        L(f"| {vlabel} | Thu≤{thu_cut}h / Mon≥{mon_end}h | {n} | {sumr:+.2f}R | {vs_str} |")
    L("")
    L("### Decision")
    L("")
    for vlabel, thu_cut, mon_end, n, sumr, delta in rows_v:
        if vlabel == "A":
            L(f"**Variant A (baseline)**: {sumr:+.2f}R across {n} trades — production default.")
            continue
        if delta > 0.5:
            decision_str = f"**Variant {vlabel}: CONSIDER** — SumR improved {delta:+.2f}R vs baseline. Review unlocked trades and cascade displacement before promoting."
        elif -0.5 <= delta <= 0.5:
            decision_str = f"**Variant {vlabel}: NEUTRAL** — SumR within ±0.5R of baseline ({delta:+.2f}R). No strong signal either way."
        else:
            decision_str = f"**Variant {vlabel}: REJECT** — SumR regressed vs baseline ({delta:+.2f}R)."
        L(decision_str)
        L("")
    L("_Report generated by `scripts/ablation_session_filter.py`._")
    L("_Offline replay only — no live changes, no master merge._")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(verbose: bool = False):
    import argparse
    parser = argparse.ArgumentParser(description="Session filter ablation study")
    parser.add_argument("--verbose", action="store_true", help="Show backtester output")
    args = parser.parse_args()
    verbose = args.verbose

    print("\n" + "═"*60)
    print("  SESSION FILTER ABLATION STUDY")
    print("  Offline replay — no live changes")
    print("═"*60)

    # Part A: Q3 dead zone diagnosis
    q3_diag = run_q3_diagnosis(verbose=verbose)

    # Part B: 8w × 3v ablation
    results = run_ablation(verbose=verbose)

    # Generate report
    print("\n  Generating report…")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_text = generate_report(q3_diag, results)
    REPORT_PATH.write_text(report_text)
    print(f"  Report written to: {REPORT_PATH}")

    # Final reset (belt + suspenders on top of atexit)
    _sc.THU_ENTRY_CUTOFF_HOUR_ET      = _ORIG_THU_CUTOFF
    SessionFilter.MONDAY_HARD_BLOCK_END = _ORIG_MON_BLOCK_END

    print("\n✅  Done. Config reset to production defaults.")
    print(f"    THU_ENTRY_CUTOFF_HOUR_ET      = {_sc.THU_ENTRY_CUTOFF_HOUR_ET}")
    print(f"    SessionFilter.MONDAY_HARD_BLOCK_END = {SessionFilter.MONDAY_HARD_BLOCK_END}")


if __name__ == "__main__":
    main()
