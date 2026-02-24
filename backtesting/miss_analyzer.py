#!/usr/bin/env python3
"""
Miss Analyzer — post-backtest gap analysis against Alex's known trades.

After every backtest run this shows:
  - Which Alex trades the bot caught vs missed
  - For each miss: WHY (from decision log)
  - Alex peak pip potential vs bot captured
  - Trend analysis across multiple runs

Usage:
  python3 backtesting/miss_analyzer.py [--runs N]
"""

import json, sys, os, argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

LOGS      = Path.home() / "trading-bot" / "logs"
DECISIONS = LOGS / "backtest_v2_decisions.json"
MISS_LOG  = LOGS / "miss_analysis.jsonl"

# ── Alex's ground truth trades ────────────────────────────────────────────────
# Each entry: week label, pair, direction, entry_price, window start, window end,
# peak_pips (max favorable move to Oct31 from price data),
# should_skip (Alex's losing trades the bot should NOT take)

ALEX_TRADES = [
    {
        "label":       "Wk1",
        "pair":        "GBP/JPY",
        "direction":   "short",
        "entry_price": 204.994,
        "window_start":"2024-07-14",
        "window_end":  "2024-07-19",
        "peak_pips":   2489,
        "should_skip": False,
        "notes":       "H&S at 205, 4H EMA confluence, London session engulfing",
    },
    {
        "label":       "Wk2",
        "pair":        "USD/JPY",
        "direction":   "short",
        "entry_price": 157.500,
        "window_start":"2024-07-20",
        "window_end":  "2024-07-26",
        "peak_pips":   1792,
        "should_skip": False,
        "notes":       "H&S neckline at 157.5, 1H engulfing + EMA rejection",
    },
    {
        "label":       "Wk3",
        "pair":        "USD/CHF",
        "direction":   "short",
        "entry_price": 0.889,
        "window_start":"2024-07-28",
        "window_end":  "2024-08-16",
        "peak_pips":   515,
        "should_skip": False,
        "notes":       "Double top / break-retest at 0.889",
    },
    {
        "label":       "Wk4",
        "pair":        "EUR/USD",
        "direction":   "short",
        "entry_price": 0.0,
        "window_start":"2024-08-10",
        "window_end":  "2024-08-16",
        "peak_pips":   0,
        "should_skip": True,
        "notes":       "Alex's rule violation — counter-trend, no valid setup. Bot should skip.",
    },
    {
        "label":       "Wk5",
        "pair":        None,
        "direction":   None,
        "entry_price": 0.0,
        "window_start":"2024-08-17",
        "window_end":  "2024-08-23",
        "peak_pips":   0,
        "should_skip": True,
        "notes":       "Alex waited — no valid setup. Bot should skip.",
    },
    {
        "label":       "Wk6",
        "pair":        "GBP/CHF",
        "direction":   "short",
        "entry_price": 1.125,
        "window_start":"2024-08-20",
        "window_end":  "2024-09-15",
        "peak_pips":   226,
        "should_skip": False,
        "notes":       "Double top at 1.125, neckline = weekly EMA, engulfing confirmation",
    },
    {
        "label":       "Wk7",
        "pair":        "GBP/CHF",
        "direction":   "short",
        "entry_price": 1.110,
        "window_start":"2024-09-02",
        "window_end":  "2024-09-13",
        "peak_pips":   76,
        "should_skip": False,
        "notes":       "GBP/CHF continuation short, ~30p target",
    },
    {
        "label":       "Wk8",
        "pair":        "USD/JPY",
        "direction":   "short",
        "entry_price": 144.0,
        "window_start":"2024-08-23",
        "window_end":  "2024-09-06",
        "peak_pips":   442,
        "should_skip": False,
        "notes":       "H&S at 144, body close + retest + engulfing",
    },
    {
        "label":       "Wk9",
        "pair":        "NZD/CAD",
        "direction":   "short",
        "entry_price": 0.0,
        "window_start":"2024-09-07",
        "window_end":  "2024-09-13",
        "peak_pips":   0,
        "should_skip": True,
        "notes":       "Alex's loss — bad setup, bot should skip.",
    },
    {
        "label":       "Wk10",
        "pair":        "USD/CAD",
        "direction":   "short",
        "entry_price": 1.350,
        "window_start":"2024-09-20",
        "window_end":  "2024-09-30",
        "peak_pips":   80,
        "should_skip": False,
        "notes":       "Break + retest at 1.35, EMA, bearish engulfing. Peak limited in window.",
    },
    {
        "label":       "Wk11",
        "pair":        "USD/CAD",
        "direction":   "short",
        "entry_price": 1.350,
        "window_start":"2024-10-04",
        "window_end":  "2024-10-11",
        "peak_pips":   0,
        "should_skip": False,
        "notes":       "Wk10 re-entry at same level. USD/CAD went against in this window.",
    },
    {
        "label":       "Wk12a",
        "pair":        "USD/JPY",
        "direction":   "short",
        "entry_price": 0.0,
        "window_start":"2024-10-04",
        "window_end":  "2024-10-11",
        "peak_pips":   0,
        "should_skip": True,
        "notes":       "Alex's Diddy loss — bad setup, bot should skip.",
    },
    {
        "label":       "Wk12b",
        "pair":        "GBP/CHF",
        "direction":   "short",
        "entry_price": 1.125,
        "window_start":"2024-10-04",
        "window_end":  "2024-10-18",
        "peak_pips":   143,
        "should_skip": False,
        "notes":       "4H engulfing at consolidation break. 600p+ Alex captured beyond Oct31 window.",
    },
    {
        "label":       "Wk13",
        "pair":        "USD/CHF",
        "direction":   "long",
        "entry_price": 0.857,
        "window_start":"2024-10-12",
        "window_end":  "2024-10-25",
        "peak_pips":   142,
        "should_skip": False,
        "notes":       "Bullish break + retest at consolidation level",
    },
]


def load_decisions():
    if not DECISIONS.exists():
        return []
    with open(DECISIONS) as f:
        data = json.load(f)
    return data.get("decisions", [])


def load_trades():
    if not DECISIONS.exists():
        return []
    with open(DECISIONS) as f:
        data = json.load(f)
    return data.get("trades", [])


def get_run_meta():
    if not DECISIONS.exists():
        return {}
    with open(DECISIONS) as f:
        data = json.load(f)
    return {
        "run_dt":      data.get("run_dt", ""),
        "notes":       data.get("notes", ""),
        "return_pct":  data.get("return_pct", 0),
        "n_trades":    data.get("n_trades", 0),
        "win_rate":    data.get("win_rate", 0),
    }


def find_bot_trade(trades, pair, direction, window_start, window_end):
    """Find if bot entered this pair+direction within (or before) the window.

    A trade counts as 'caught' if:
      (a) entry was within the window, OR
      (b) entry was before the window AND exit was after window_start
          (i.e. the position was still open during the window — same trade context)
    This handles cases where the bot enters slightly early (e.g. Jul 16 vs Jul 20
    window start) but takes the same directional trade Alex took in that week.
    """
    for t in trades:
        if (t.get("pair", "").replace("_", "/") == pair.replace("_", "/")
                and t.get("direction") == direction):
            entry_ts = t.get("entry_ts", t.get("entry_dt", ""))[:10]
            exit_ts  = t.get("exit_ts",  t.get("exit_dt",  ""))[:10]
            # In-window entry
            if window_start <= entry_ts <= window_end:
                return t
            # Pre-window entry but still open during the window
            if entry_ts < window_start and (exit_ts >= window_start or exit_ts == ""):
                return t
    return None


def _bot_pips_from_trade(t, pair, direction):
    """Compute pips from a bot trade dict."""
    entry = t.get("entry", t.get("entry_price", 0))
    exit_ = t.get("exit",  t.get("exit_price",  0))
    if not entry or not exit_:
        return 0
    m   = 100 if "JPY" in pair else 10000
    raw = (entry - exit_) if direction == "short" else (exit_ - entry)
    return raw * m


def get_miss_reason(decisions, pair, window_start, window_end):
    """
    Scan decision log for the pair in the window.
    Returns the most informative reason the bot didn't enter.
    """
    pair_norm = pair.replace("_", "/")
    window_decisions = [
        d for d in decisions
        if d.get("pair") == pair_norm
        and window_start <= d.get("ts", "")[:10] <= window_end
    ]

    if not window_decisions:
        return "NOT_EVALUATED", "Pair never evaluated in window — likely blocked before strategy ran (currency overlap, max_concurrent, session filter)"

    enter_attempts = [d for d in window_decisions if d.get("decision") == "ENTER"]
    if enter_attempts:
        # Was blocked after ENTER decision — shouldn't happen but check
        return "ENTER_BLOCKED", f"ENTER signal fired but was blocked: {enter_attempts[0].get('reason','')[:120]}"

    # Look for highest-confidence WAIT reason
    wait_decisions = [d for d in window_decisions if d.get("decision") == "WAIT"]
    if wait_decisions:
        best = max(wait_decisions, key=lambda d: d.get("confidence", 0))
        conf = best.get("confidence", 0)
        reason = best.get("reason", "")[:150]
        filters = best.get("filters_failed", [])

        if "WAITING FOR ENGULFING" in reason:
            return "NO_ENGULFING", f"Pattern found (conf={conf:.0%}), no engulfing candle fired. {reason[:100]}"
        if "neckline_not_at_level" in filters or "not near a round number" in reason:
            return "LEVEL_MISMATCH", f"Pattern found but level not at round number. {reason[:100]}"
        if "not at a major round" in reason or "structural" in reason.lower():
            return "LEVEL_MISMATCH", f"Level check failed. {reason[:100]}"
        if "R:R too low" in reason:
            return "LOW_RR", f"R:R too low. {reason[:100]}"
        if "No qualifying pattern" in reason:
            return "NO_PATTERN", f"No pattern detected near price. Last seen: {best.get('ts','')[:10]}"
        if "weekly_candle_opposing" in filters:
            return "WEEKLY_OPPOSING", f"Weekly candle blocked entry. {reason[:100]}"
        if "Trend" in reason or "trend" in reason:
            return "TREND_BLOCK", f"Trend filter blocked. {reason[:100]}"
        return "WAIT_OTHER", f"(conf={conf:.0%}) {reason[:120]}"

    return "ALL_BLOCKED", f"All {len(window_decisions)} decisions were BLOCKED (session/currency/concurrent). Never reached strategy evaluation."


def analyze(verbose=True):
    decisions = load_decisions()
    trades    = load_trades()
    meta      = get_run_meta()

    if verbose:
        print(f"\n{'='*70}")
        print(f"  MISS ANALYZER — {meta.get('notes','?')}  |  Return: {meta.get('return_pct',0):+.1f}%  |  {meta.get('run_dt','')[:10]}")
        print(f"{'='*70}")
        print(f"{'Wk':6} {'Pair':8} {'Dir':5} {'Bot?':6} {'Peak':>7}  Miss reason")
        print("-" * 70)

    results    = []
    total_peak = 0
    total_bot  = 0

    for trade in ALEX_TRADES:
        label  = trade["label"]
        pair   = trade["pair"]
        direct = trade["direction"]
        skip   = trade["should_skip"]
        peak   = trade["peak_pips"]
        ws, we = trade["window_start"], trade["window_end"]

        if skip or pair is None:
            result = {
                "label": label, "pair": pair, "direction": direct,
                "status": "CORRECT_SKIP", "bot_pips": 0, "peak_pips": 0,
                "miss_reason": None, "miss_detail": None,
            }
            if verbose:
                print(f"{label:6} {'—':8} {'—':5} {'✅ skip':6} {'—':>7}  Correctly avoided (Alex's loss/no-trade)")
            results.append(result)
            continue

        bot_trade = find_bot_trade(trades, pair, direct, ws, we)
        total_peak += max(peak, 0)

        if bot_trade:
            bot_pips = _bot_pips_from_trade(bot_trade, pair, direct)
            total_bot += bot_pips
            result = {
                "label": label, "pair": pair, "direction": direct,
                "status": "CAUGHT", "bot_pips": round(bot_pips),
                "peak_pips": peak, "miss_reason": None, "miss_detail": None,
            }
            if verbose:
                print(f"{label:6} {pair:8} {direct:5} {'✅':6} {peak:>+6}p  Caught — bot: {bot_pips:+.0f}p")
        else:
            reason_code, reason_detail = get_miss_reason(decisions, pair, ws, we)
            result = {
                "label": label, "pair": pair, "direction": direct,
                "status": "MISSED", "bot_pips": 0, "peak_pips": peak,
                "miss_reason": reason_code, "miss_detail": reason_detail,
            }
            if verbose:
                print(f"{label:6} {pair:8} {direct:5} {'❌':6} {peak:>+6}p  [{reason_code}] {reason_detail[:55]}")

        results.append(result)

    if verbose:
        print("-" * 70)
        miss_trades = [r for r in results if r["status"] == "MISSED"]
        caught = [r for r in results if r["status"] == "CAUGHT"]
        print(f"\nCaught:  {len(caught)} / {len([r for r in results if not r['status']=='CORRECT_SKIP'])} Alex winning trades")
        print(f"Alex peak pips available : {total_peak:+.0f}p")
        print(f"Bot pips from matched    : {total_bot:+.0f}p")
        print(f"Gap                      : {total_peak - total_bot:+.0f}p")

        if miss_trades:
            print(f"\nMiss pattern breakdown:")
            codes = Counter(r["miss_reason"] for r in miss_trades)
            for code, count in codes.most_common():
                trades_with_code = [r["label"] for r in miss_trades if r["miss_reason"] == code]
                print(f"  {code:20} ×{count}  ({', '.join(trades_with_code)})")

    # Append to persistent miss log
    log_entry = {
        "ts":          datetime.utcnow().isoformat(),
        "run_notes":   meta.get("notes", ""),
        "return_pct":  meta.get("return_pct", 0),
        "total_peak":  total_peak,
        "total_bot":   total_bot,
        "gap":         total_peak - total_bot,
        "results":     results,
    }
    with open(MISS_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="Show last N runs from miss log")
    args = parser.parse_args()

    if args.runs > 1:
        print(f"\nLast {args.runs} runs from miss log:")
        if MISS_LOG.exists():
            runs = [json.loads(l) for l in open(MISS_LOG) if l.strip()]
            for run in runs[-args.runs:]:
                ts   = run["ts"][:16]
                note = run["run_notes"][:30]
                gap  = run["gap"]
                ret  = run["return_pct"]
                codes = Counter(r["miss_reason"] for r in run["results"]
                                if r["status"] == "MISSED")
                top = ", ".join(f"{c}×{n}" for c,n in codes.most_common(3))
                print(f"  {ts}  {note:30}  ret={ret:+.1f}%  gap={gap:+.0f}p  [{top}]")
        else:
            print("  No miss log yet — run a backtest first.")
    else:
        analyze()
