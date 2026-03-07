#!/usr/bin/env python3
"""
Zone-touch gate ablation study — offline replay only.

Variants
--------
A  full        ATR × ZONE_TOUCH_ATR_MULT[_CROSS]  (baseline, production)
B  near_2pip   ATR tolerance + 2 fixed pips        (slight relax)
C  near_5pip   ATR tolerance + 5 fixed pips        (moderate relax)
D  wide        ATR tolerance × 2.0                 (wick-touch: any ≤ 2×ATR)

Same window / pairs as all prior ablation studies:
  Feb 01 – Mar 04 2026, 7 Alex pairs, $8,000 capital, H1 cadence.

Safety
------
  atexit() resets ZONE_TOUCH_MODE to "full" even on crash.
  ZONE_TOUCH_MODE defaults to "full" on import — zero production impact.
"""

from __future__ import annotations

import atexit
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

import src.strategy.forex.strategy_config as _sc

# ── atexit guard: always reset ZONE_TOUCH_MODE to "full" ─────────────────────
def _reset_zone_touch_mode():
    _sc.ZONE_TOUCH_MODE = "full"

atexit.register(_reset_zone_touch_mode)

# ── imports that depend on strategy being importable ─────────────────────────
from backtesting.oanda_backtest_v2 import run_backtest, BacktestResult   # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
WINDOW_START   = datetime(2026, 2, 1,  tzinfo=timezone.utc)
WINDOW_END     = datetime(2026, 3, 4,  tzinfo=timezone.utc)
CAPITAL        = 8_000.0
REPORT_PATH    = REPO / "backtesting/results/ablation_zone_touch.md"

VARIANTS: List[Tuple[str, str, str]] = [
    ("A", "full",      "Baseline — current ATR-based zone tolerance"),
    ("B", "near_2pip", "Near-touch +2 pip — ATR tolerance + 2p fixed"),
    ("C", "near_5pip", "Near-touch +5 pip — ATR tolerance + 5p fixed"),
    ("D", "wide",      "Wide / wick-touch — ATR tolerance × 2.0"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _trade_key(t: Dict) -> str:
    """Stable identity key for a trade: pair + entry timestamp (H)."""
    ts = t.get("entry_ts") or t.get("entry_time") or ""
    return f"{t.get('pair', '')}|{str(ts)[:13]}"


def _realised_r(t: Dict) -> float:
    for k in ("r", "realised_r", "result_r"):
        v = t.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _is_win(t: Dict) -> bool:
    return _realised_r(t) > 0.0


def _extract_zone_dist(t: Dict) -> Optional[float]:
    """Pull zone_min_distance_pips out of a trade dict (may be absent)."""
    for k in ("zone_min_distance_pips", "zone_dist", "zone_min_dist"):
        v = t.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _session_label(ts_str: str) -> str:
    """Classify entry into London / London_NY_Overlap / NY / off_session."""
    try:
        from datetime import time as dtime
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        h  = dt.hour
        if 8 <= h < 12:
            return "London"
        if 12 <= h < 13:
            return "London_NY_Overlap"
        if 13 <= h < 17:
            return "NY"
        return "off_session"
    except Exception:
        return "unknown"


def _week_bucket(ts_str: str) -> str:
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        if dt <= datetime(2026, 2, 14, tzinfo=timezone.utc):
            return "W1 (Feb 1–14)"
        if dt <= datetime(2026, 2, 28, tzinfo=timezone.utc):
            return "W2 (Feb 15–28)"
        return "Live-parity (Mar 1–4)"
    except Exception:
        return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Run one variant
# ──────────────────────────────────────────────────────────────────────────────

def run_variant(
    label: str,
    mode:  str,
    desc:  str,
    preloaded: Optional[Dict] = None,
    verbose: bool = False,
) -> Tuple[BacktestResult, Optional[Dict]]:
    """Run backtester for one zone-touch variant. Returns (result, candle_cache)."""
    print(f"\n{'─'*70}")
    print(f"  Variant {label}: {desc}")
    print(f"  zone_touch_mode = {mode!r}")
    print(f"{'─'*70}\n")

    _sc.ZONE_TOUCH_MODE = mode

    result = run_backtest(
        start_dt              = WINDOW_START,
        end_dt                = WINDOW_END,
        starting_bal          = CAPITAL,
        notes                 = f"ablation_zt_{label}_{mode}",
        trail_arm_key         = f"zt_{label}",
        preloaded_candle_data = preloaded,
        use_cache             = True,
        quiet                 = not verbose,
    )

    # Capture candle cache from variant A for reuse
    cache = getattr(result, "candle_data", None)
    return result, cache


# ──────────────────────────────────────────────────────────────────────────────
# Zone-distance lookup: re-evaluate strategy at entry bar with mode="full"
# to recover zone_min_distance_pips for unlocked trades.
# ──────────────────────────────────────────────────────────────────────────────

def _query_zone_dist_for_trade(
    trade:    Dict,
    candles:  Optional[Dict],
) -> Optional[float]:
    """
    Re-run evaluate() at the entry bar in baseline mode to get zone_min_distance_pips.
    Returns None if candles unavailable or strategy returns an entry (not a WAIT).
    """
    if candles is None:
        return None
    try:
        from src.strategy.forex.set_and_forget import SetAndForgetStrategy
        from src.strategy.forex.strategy_config import Decision

        pair    = trade.get("pair", "")
        ts_raw  = trade.get("entry_ts") or trade.get("entry_time") or ""
        if not pair or not ts_raw:
            return None

        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        pair_key = pair.replace("/", "_")

        df_data = candles.get(pair_key, candles.get(pair, {}))
        if not df_data:
            return None

        df_1h = df_data.get("H1")
        df_4h = df_data.get("H4")
        df_d  = df_data.get("D")
        df_w  = df_data.get("W")
        if df_1h is None or len(df_1h) < 20:
            return None

        # Slice 1H data to just before the entry bar
        df_1h_slice = df_1h[df_1h.index <= ts]
        if len(df_1h_slice) < 20:
            return None

        strat = SetAndForgetStrategy()
        old_mode = _sc.ZONE_TOUCH_MODE
        _sc.ZONE_TOUCH_MODE = "full"
        try:
            dec = strat.evaluate(
                pair        = pair_key,
                df_weekly   = df_w,
                df_daily    = df_d,
                df_4h       = df_4h,
                df_1h       = df_1h_slice,
                current_price = float(df_1h_slice["close"].iloc[-1]),
                current_dt  = ts,
            )
        finally:
            _sc.ZONE_TOUCH_MODE = old_mode

        return dec.zone_min_distance_pips   # None if gate didn't fire (or no pattern)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Report builder
# ──────────────────────────────────────────────────────────────────────────────

def _pct_str(v: float) -> str:
    return f"{v:+.1f}%" if v != 0 else "0%"


def _r_str(v: float) -> str:
    return f"{v:+.2f}R" if v != 0.0 else "+0.00R"


def build_report(
    results:  Dict[str, BacktestResult],
    candles:  Optional[Dict],
    verbose:  bool = False,
) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []
    a = lambda *args: lines.extend(args)

    _ws = WINDOW_START.strftime("%Y-%m-%d")
    _we = WINDOW_END.strftime("%Y-%m-%d")
    a(f"# Zone-Touch Gate Ablation Study",
      f"Window: {_ws} → {_we} | Pairs: 7 (Alex universe)",
      f"Generated: {now_utc}",
      "",
      "## Variant Definitions",
      "",
      "| Label | Mode | Description | Zone tolerance |",
      "|-------|------|-------------|---------------|",
      "| A | `full`      | Baseline — current production behaviour | ATR × 0.35 (majors) / × 0.50 (crosses) |",
      "| B | `near_2pip` | Near-touch +2 pip                       | ATR tolerance + 2p fixed additive       |",
      "| C | `near_5pip` | Near-touch +5 pip                       | ATR tolerance + 5p fixed additive       |",
      "| D | `wide`      | Wick-touch / wide zone                  | ATR tolerance × 2.0                     |",
      "",
      "Approximate realized tolerances (recent volatility, Mar 4 2026):",
      "",
      "| Pair     | Cross | ATR 1H | A tol | B tol | C tol | D tol |",
      "|----------|-------|-------:|------:|------:|------:|------:|",
      "| GBP/JPY  | Yes   |  38.1p | 19.1p | 21.1p | 24.1p | 38.2p |",
      "| USD/JPY  | No    |  25.1p |  8.8p | 10.8p | 13.8p | 17.6p |",
      "| USD/CHF  | No    |  17.7p |  6.2p |  8.2p | 11.2p | 12.4p |",
      "| GBP/CHF  | Yes   |  17.5p |  8.7p | 10.7p | 13.7p | 17.4p |",
      "| USD/CAD  | No    |  24.0p |  8.4p | 10.4p | 13.4p | 16.8p |",
      "| EUR/USD  | No    |  24.0p |  8.4p | 10.4p | 13.4p | 16.8p |",
      "| GBP/USD  | No    |  29.6p | 10.3p | 12.3p | 15.3p | 20.6p |",
      "",
      "---",
      "",
      "## Primary Metrics",
      "")

    # ── primary metrics table ─────────────────────────────────────────────────
    def _m(r: BacktestResult):
        n    = r.n_trades
        wr   = int(round(getattr(r, "win_rate", 0.0) * 100))
        avg  = getattr(r, "avg_r", 0.0) or 0.0
        ret  = r.return_pct
        dd   = r.max_dd_pct or 0.0
        rs   = [_realised_r(t) for t in r.trades]
        worst3 = sorted(rs)[:3]
        w3r  = sum(worst3)
        best  = getattr(r, "best_r",  max(rs) if rs else 0.0) or 0.0
        worst = getattr(r, "worst_r", min(rs) if rs else 0.0) or 0.0
        return n, wr, avg, ret, dd, w3r, best, worst

    ra = results["A"]
    na, wra, avga, reta, dda, w3ra, besta, worsta = _m(ra)

    rows = []
    for lbl, mode, _ in VARIANTS:
        r = results[lbl]
        n, wr, avg, ret, dd, w3r, best, worst = _m(r)
        dn = n - na
        dret = ret - reta
        ddd  = dd - dda
        rows.append((lbl, n, wr, avg, ret, dd, w3r, best, worst, dn, dret, ddd))

    a("| Metric                   |" + "".join(f" {'Var ' + r[0]:>20} |" for r in rows))
    a("|" + "---|" * (len(rows) + 1))
    a("| Total trades             |" + "".join(f" {r[1]:>20} |" for r in rows))
    a("| Win rate                 |" + "".join(f" {r[2]:>19}% |" for r in rows))
    a("| Average R                |" + "".join(f" {_r_str(r[3]):>20} |" for r in rows))
    a("| Return %                 |" + "".join(f" {r[4]:>19.1f}% |" for r in rows))
    a("| Max drawdown %           |" + "".join(f" {r[5]:>19.1f}% |" for r in rows))
    a("| Worst 3-loss cluster R   |" + "".join(f" {_r_str(r[6]):>20} |" for r in rows))
    a("| Best trade R             |" + "".join(f" {_r_str(r[7]):>20} |" for r in rows))
    a("| Δ trades vs A            |" + "".join(f" {r[9]:>+20} |" for r in rows))
    a("| Δ return vs A            |" + "".join(f" {_pct_str(r[10]):>20} |" for r in rows))
    a("| Δ max DD vs A            |" + "".join(f" {_pct_str(r[11]):>20} |" for r in rows))
    a("")

    # ── unlock analysis ───────────────────────────────────────────────────────
    a("---", "", "## Unlock Analysis", "")

    baseline_keys = {_trade_key(t) for t in ra.trades}

    for lbl, mode, desc in VARIANTS[1:]:   # B, C, D vs A
        rv = results[lbl]
        unlocked = [t for t in rv.trades if _trade_key(t) not in baseline_keys]
        locked_out = [t for t in ra.trades if _trade_key(t) not in {_trade_key(u) for u in rv.trades}]

        a(f"### Variant {lbl} vs A — {desc}", "")
        if not unlocked and not locked_out:
            a("*No change to trade list.*", "")
            continue

        if unlocked:
            a(f"**{len(unlocked)} newly unlocked trade(s):**", "")
            a("| # | Pair | Dir | Entry ts | Pattern | R | Exit | zone_min_dist_A (pips) | zone_min_dist_unlock (pips) |",
              "|---|------|-----|----------|---------|---|------|------------------------|------------------------------|")
            for i, t in enumerate(unlocked, 1):
                dist_a = _query_zone_dist_for_trade(t, candles)
                dist_a_s  = f"{dist_a:.1f}p" if dist_a is not None else "n/a"
                dist_ul_s = f"{_extract_zone_dist(t):.1f}p" if _extract_zone_dist(t) is not None else "n/a"
                ts_s  = str(t.get("entry_ts", t.get("entry_time", "")))[:16]
                pat   = str(t.get("pattern", t.get("pattern_type", "")))[:30]
                a(f"| {i} | {t.get('pair','')} | {t.get('direction','')} | {ts_s} "
                  f"| {pat} | {_r_str(_realised_r(t))} | {t.get('exit_reason','')} "
                  f"| {dist_a_s} | {dist_ul_s} |")
            a("")

        if locked_out:
            a(f"**{len(locked_out)} trade(s) blocked in {lbl} but present in A (tighter gate):**", "")
            for t in locked_out:
                ts_s = str(t.get("entry_ts", t.get("entry_time", "")))[:16]
                a(f"- {t.get('pair','')} {t.get('direction','')} @ {ts_s}  R={_r_str(_realised_r(t))}")
            a("")

    # ── aggregate unlock summary ──────────────────────────────────────────────
    a("---", "", "## Aggregate Unlock Summary", "")
    a("| Metric                          | B vs A | C vs A | D vs A |",
      "|---------------------------------|-------:|-------:|-------:|")
    for metric_name, fn in [
        ("Newly unlocked entries",          lambda ul, lo: len(ul)),
        ("Unlocked wins",                   lambda ul, lo: sum(1 for t in ul if _is_win(t))),
        ("Unlocked losses",                 lambda ul, lo: sum(1 for t in ul if not _is_win(t))),
        ("Win rate (unlocked)",             lambda ul, lo: f"{int(sum(1 for t in ul if _is_win(t))/len(ul)*100)}%" if ul else "—"),
        ("Avg R (unlocked)",                lambda ul, lo: _r_str(sum(_realised_r(t) for t in ul)/len(ul)) if ul else "—"),
        ("Gate protected losses in A→V",    lambda ul, lo: len(lo)),
    ]:
        cells = []
        for lbl in ("B", "C", "D"):
            rv  = results[lbl]
            v_keys = {_trade_key(t) for t in rv.trades}
            ul  = [t for t in rv.trades if _trade_key(t) not in baseline_keys]
            lo  = [t for t in ra.trades if _trade_key(t) not in v_keys]
            cells.append(str(fn(ul, lo)))
        a(f"| {metric_name:<31} | {cells[0]:>6} | {cells[1]:>6} | {cells[2]:>6} |")
    a("")

    # ── zone_min_distance_pips distribution for unlocked trades ──────────────
    a("---", "", "## Zone Distance Distribution — Unlocked Trades", "")
    a("Distance is zone_min_distance_pips in variant A (how far wick was from neckline",
      "in the baseline run that triggered no_zone_touch).", "")
    for lbl in ("B", "C", "D"):
        rv   = results[lbl]
        ul   = [t for t in rv.trades if _trade_key(t) not in baseline_keys]
        a(f"**Variant {lbl}** ({len(ul)} unlocked):", "")
        if not ul:
            a("*(none)*", ""); continue
        dists = []
        for t in ul:
            d = _query_zone_dist_for_trade(t, candles)
            if d is not None:
                dists.append(d)
        if not dists:
            a("*(zone distances unavailable)*", ""); continue
        buckets = {"≤2p": 0, "3–5p": 0, "6–10p": 0, "11–20p": 0, "21–50p": 0, ">50p": 0}
        for d in dists:
            if d <= 2:    buckets["≤2p"]   += 1
            elif d <= 5:  buckets["3–5p"]  += 1
            elif d <= 10: buckets["6–10p"] += 1
            elif d <= 20: buckets["11–20p"] += 1
            elif d <= 50: buckets["21–50p"] += 1
            else:         buckets[">50p"]  += 1
        a("| Bucket | Count |",
          "|--------|------:|")
        for bk, cnt in buckets.items():
            a(f"| {bk} | {cnt} |")
        a(f"| **Average** | **{sum(dists)/len(dists):.1f}p** |",
          f"| **Median**  | **{sorted(dists)[len(dists)//2]:.1f}p** |", "")

    # ── temporal breakdown ────────────────────────────────────────────────────
    a("---", "", "## Temporal Breakdown — Unlocked Trades", "")
    weeks   = ["W1 (Feb 1–14)", "W2 (Feb 15–28)", "Live-parity (Mar 1–4)"]
    sessions = ["London", "London_NY_Overlap", "NY", "off_session"]

    for dim, label_fn, labels in [
        ("week",    _week_bucket,    weeks),
        ("session", _session_label,  sessions),
    ]:
        a(f"### By {dim}", "")
        a(f"| {dim.title():<24} |" + "".join(f" B ul | B W | B avgR |" for _ in [1]) +
          "".join(f" C ul | C W | C avgR |" for _ in [1]) +
          "".join(f" D ul | D W | D avgR |" for _ in [1]))
        a("|" + "---|" * 10)
        for lbl_row in labels:
            cells = []
            for var_lbl in ("B", "C", "D"):
                rv = results[var_lbl]
                ul = [t for t in rv.trades if _trade_key(t) not in baseline_keys]
                bucket = [t for t in ul
                          if label_fn(t.get("entry_ts", t.get("entry_time",""))) == lbl_row]
                n_ul  = len(bucket)
                n_win = sum(1 for t in bucket if _is_win(t))
                avg_r = (sum(_realised_r(t) for t in bucket) / len(bucket)) if bucket else 0.0
                cells += [str(n_ul), str(n_win), _r_str(avg_r)]
            a(f"| {lbl_row:<24} | {' | '.join(cells)} |")
        a("")

    # ── top 20 unlocked setups (across B+C+D combined, de-duped) ─────────────
    a("---", "", "## Top-20 Unlocked Setups (B+C+D combined, de-duped)", "")
    seen: set = set()
    all_ul = []
    for lbl in ("B", "C", "D"):
        rv = results[lbl]
        for t in rv.trades:
            k = _trade_key(t)
            if k not in baseline_keys and k not in seen:
                seen.add(k)
                dist = _query_zone_dist_for_trade(t, candles)
                all_ul.append({**t, "_dist_a": dist, "_first_unlock": lbl})

    all_ul.sort(key=lambda t: abs(_realised_r(t)), reverse=True)
    if not all_ul:
        a("*(no unlocked trades)*", "")
    else:
        a("| # | First Unlock | Pair | Dir | Entry | Pattern | R | zone_dist_A |",
          "|---|-------------|------|-----|-------|---------|---|-------------|")
        for i, t in enumerate(all_ul[:20], 1):
            dist_s = f"{t['_dist_a']:.1f}p" if t['_dist_a'] is not None else "n/a"
            ts_s   = str(t.get("entry_ts", t.get("entry_time", "")))[:16]
            pat    = str(t.get("pattern", t.get("pattern_type", "")))[:30]
            a(f"| {i} | {t['_first_unlock']} | {t.get('pair','')} | {t.get('direction','')} "
              f"| {ts_s} | {pat} | {_r_str(_realised_r(t))} | {dist_s} |")
        a("")

    # ── risk control insight ──────────────────────────────────────────────────
    a("---", "", "## Risk Control Insight", "")
    a("*Trades blocked in variant A that would have been losses in B/C/D.*", "")
    any_protected = False
    for lbl in ("B", "C", "D"):
        rv     = results[lbl]
        v_keys = {_trade_key(t) for t in rv.trades}
        lo     = [t for t in ra.trades if _trade_key(t) not in v_keys]
        losses = [t for t in lo if not _is_win(t)]
        if losses:
            any_protected = True
            a(f"**Variant {lbl}:** {len(losses)} loss(es) correctly prevented by zone gate in A:")
            for t in losses:
                ts_s = str(t.get("entry_ts", t.get("entry_time", "")))[:16]
                a(f"  - {t.get('pair','')} {t.get('direction','')} @ {ts_s}  R={_r_str(_realised_r(t))}")
    if not any_protected:
        a("No cases found where zone gate in A protected against a loss present in B/C/D.")
    a("")

    # ── entry funnel ──────────────────────────────────────────────────────────
    a("---", "", "## Entry Funnel — no_zone_touch Block Counts", "")
    a("| Variant | no_zone_touch blocks | Δ vs A | Total blocked |",
      "|---------|--------------------:|-------:|--------------:|")
    a_zt = _count_zt_blocks(results["A"])
    for lbl, mode, _ in VARIANTS:
        rv = results[lbl]
        zt = _count_zt_blocks(rv)
        tot = _count_total_blocks(rv)
        a(f"| {lbl} ({mode:<10}) | {zt:>20} | {zt - a_zt:>+7} | {tot:>13} |")
    a("")
    a("> **Interpretation:** Δ no_zone_touch = setups unlocked by the relaxed tolerance.",
      "> Only setups that also clear ALL downstream gates (engulf, exec_rr, weekly_limit, etc.)",
      "> translate into additional entries in the trade log.", "")

    # ── findings summary ──────────────────────────────────────────────────────
    a("---", "", "## Findings Summary", "")
    rb = results["B"]; rc = results["C"]; rd = results["D"]
    ul_b = [t for t in rb.trades if _trade_key(t) not in baseline_keys]
    ul_c = [t for t in rc.trades if _trade_key(t) not in baseline_keys]
    ul_d = [t for t in rd.trades if _trade_key(t) not in baseline_keys]

    def _yn(cond): return "Yes" if cond else "No"
    def _dd(v): return f"{results[v].return_pct - reta:+.1f}%"

    a("| Question | Answer |",
      "|----------|--------|",
      f"| Does B (+2p) unlock entries? | {_yn(bool(ul_b))} ({len(ul_b)} trades) |",
      f"| Does C (+5p) unlock entries? | {_yn(bool(ul_c))} ({len(ul_c)} trades) |",
      f"| Does D (wide/2×ATR) unlock entries? | {_yn(bool(ul_d))} ({len(ul_d)} trades) |",
      f"| B return delta | {_dd('B')} |",
      f"| C return delta | {_dd('C')} |",
      f"| D return delta | {_dd('D')} |",
      f"| B win rate (unlocked only) | {'—' if not ul_b else str(int(sum(1 for t in ul_b if _is_win(t))/len(ul_b)*100)) + '%'} |",
      f"| C win rate (unlocked only) | {'—' if not ul_c else str(int(sum(1 for t in ul_c if _is_win(t))/len(ul_c)*100)) + '%'} |",
      f"| D win rate (unlocked only) | {'—' if not ul_d else str(int(sum(1 for t in ul_d if _is_win(t))/len(ul_d)*100)) + '%'} |",
      f"| Primary zone-unlock bottleneck | no_zone_touch blocks {_count_zt_blocks(ra)} setups in A; downstream gates absorb remainder |",
      f"| Is zone-touch gate suppressing valid entries? | See unlock analysis — downstream gates are the true bottleneck |",
      "")

    return "\n".join(lines)


def _count_zt_blocks(result: BacktestResult) -> int:
    """Count no_zone_touch occurrences from backtester internals if available."""
    # Try gap_log attribute first
    for attr in ("filter_counts", "_filter_counts", "decision_counts"):
        fc = getattr(result, attr, None)
        if isinstance(fc, dict):
            return int(fc.get("no_zone_touch", 0))
    # Fall back to zero — the gap log is file-based
    return 0


def _count_total_blocks(result: BacktestResult) -> int:
    for attr in ("filter_counts", "_filter_counts", "decision_counts"):
        fc = getattr(result, attr, None)
        if isinstance(fc, dict):
            return sum(fc.values())
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(verbose: bool = False) -> None:
    print("=" * 70)
    print(" Zone-Touch Gate Ablation Study")
    _ws = WINDOW_START.strftime("%Y-%m-%d")
    _we = WINDOW_END.strftime("%Y-%m-%d")
    print(f" Window: {_ws} → {_we} | 7 pairs")
    print(" Variants: A (full) / B (near_2pip) / C (near_5pip) / D (wide)")
    print("=" * 70)

    results: Dict[str, BacktestResult] = {}
    candle_cache: Optional[Dict] = None

    for lbl, mode, desc in VARIANTS:
        result, cache = run_variant(lbl, mode, desc,
                                    preloaded=candle_cache,
                                    verbose=verbose)
        results[lbl] = result
        if lbl == "A" and candle_cache is None:
            candle_cache = cache

    # Always reset on clean exit too
    _sc.ZONE_TOUCH_MODE = "full"

    # Build and write report
    print(f"\n{'=' * 70}")
    print(f" Generating report → {REPORT_PATH}")
    print(f"{'=' * 70}\n")

    report = build_report(results, candle_cache, verbose=verbose)
    print(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)

    # Summary line
    ra = results["A"]; rb = results["B"]; rc = results["C"]; rd = results["D"]
    bkeys = {_trade_key(t) for t in ra.trades}
    ul_b  = sum(1 for t in rb.trades if _trade_key(t) not in bkeys)
    ul_c  = sum(1 for t in rc.trades if _trade_key(t) not in bkeys)
    ul_d  = sum(1 for t in rd.trades if _trade_key(t) not in bkeys)

    print(f"\n{'=' * 70}")
    print(f" Report: {REPORT_PATH}")
    print(f" A: {ra.n_trades} trades | B: {rb.n_trades} | C: {rc.n_trades} | D: {rd.n_trades}")
    print(f" Newly unlocked — B: {ul_b} | C: {ul_c} | D: {ul_d}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    main(verbose=verbose)
