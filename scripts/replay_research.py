#!/usr/bin/env python3
"""
Replay Research Harness — 5-minute cadence zone-touch analysis

Research only. NO strategy thresholds, rules, or entry logic changed.
Loads historical candles, evaluates strategy at 5m cadence, records:
  a) PRE_CANDIDATE count (below 0.4 clarity floor)
  b) CANDIDATE_WAIT count by reason
  c) NO_ZONE_TOUCH events + zone_min_distance_pips distribution
  d) Time-to-touch: minutes from pattern recognition to first zone touch
  e) Would-have-entered: zone touches detected on M5 bar highs/lows
  f) VIRTUAL_ENTRY_MISSED_BY_HOURLY: zone touches between hourly evaluations

Usage:
    python scripts/replay_research.py \\
        --from 2026-02-01 --to 2026-03-04 \\
        --cadence 5 \\
        --pairs GBP_JPY USD_JPY USD_CHF GBP_CHF USD_CAD EUR_USD GBP_USD \\
        --out backtesting/results/replay_research_2026-02-01_to_2026-03-04.md \\
        [--use-cache]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time as _time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Bootstrap path ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from backtesting.oanda_backtest_v2 import _fetch_range, _resample_weekly, _oanda_client
from src.strategy.forex.set_and_forget import SetAndForgetStrategy
from src.strategy.forex.session_filter import SessionFilter, ET
from src.strategy.forex import strategy_config as _cfg

# ── M5 paginated fetch ────────────────────────────────────────────────────────
# _fetch_range in oanda_backtest_v2 defaults to 1H delta for unknown granularities.
# For M5 over 45 days (~12 960 bars) that causes a single over-limit request that
# OANDA rejects.  This function paginates correctly with a 5-minute step.

import requests as _requests
import time as _sleep_mod

def _fetch_m5_range(
    pair: str,
    from_dt: datetime,
    to_dt: datetime,
) -> Optional[pd.DataFrame]:
    """
    Paginated M5 candle fetch.  Steps forward by 5-minute delta so each page
    stays within OANDA's 5 000-bar limit.
    """
    from backtesting.oanda_backtest_v2 import INSTRUMENT_MAP  # noqa: PLC0415
    instrument = INSTRUMENT_MAP.get(pair, pair)
    delta = timedelta(minutes=5)
    page_bars = 4900          # stay under 5 000-bar hard limit with margin

    all_rows: List[dict] = []
    cur = from_dt
    final = to_dt

    while cur < final:
        tentative_end = cur + delta * page_bars
        if tentative_end >= final:
            # last page — use from+to (OK because ≤ page_bars)
            params = {
                "granularity": "M5",
                "from":  cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to":    final.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "price": "M",
            }
        else:
            params = {
                "granularity": "M5",
                "from":  cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "count": page_bars,
                "price": "M",
            }

        try:
            resp = _requests.get(
                f"{_oanda_client.base}/v3/instruments/{instrument}/candles",
                headers=_oanda_client.headers,
                params=params,
                timeout=30,
            )
            if resp.status_code != 200:
                print(f"    ⚠ M5 fetch HTTP {resp.status_code} for {pair}: {resp.text[:200]}")
                break
            candles = resp.json().get("candles", [])
            if not candles:
                break
            for c in candles:
                mid = c.get("mid", {})
                all_rows.append({
                    "time":   pd.Timestamp(c["time"]).tz_localize(None),
                    "open":   float(mid.get("o", 0)),
                    "high":   float(mid.get("h", 0)),
                    "low":    float(mid.get("l", 0)),
                    "close":  float(mid.get("c", 0)),
                    "volume": int(c.get("volume", 0)),
                })
            last_ts = pd.Timestamp(candles[-1]["time"]).tz_localize(None)
            if "to" in params or len(candles) < page_bars:
                break
            cur = (last_ts.to_pydatetime() + delta).replace(tzinfo=timezone.utc)
            _sleep_mod.sleep(0.25)
        except Exception as exc:
            print(f"    ⚠ M5 fetch error {pair}: {exc}")
            break

    if not all_rows:
        return None
    df = pd.DataFrame(all_rows).set_index("time")
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df


# ── Constants ──────────────────────────────────────────────────────────────────
ALEX_PAIRS = [
    "GBP_JPY", "USD_JPY", "USD_CHF", "GBP_CHF",
    "USD_CAD", "EUR_USD", "GBP_USD",
]
OANDA_TO_SLASH = {p: p.replace("_", "/") for p in ALEX_PAIRS}

REPLAY_CACHE_PATH = Path("/tmp/replay_candle_cache.pkl")

# Strategy needs ~180 days of H1 history for reliable pattern detection
LOOKBACK_DAYS = 180
# M5 lookback — only need a couple weeks before replay start for warm-up
M5_LOOKBACK_DAYS = 14

DIST_BUCKET_EDGES = [2, 5, 10, 25, 50]
DIST_BUCKET_LABELS = ["0-2p", "2-5p", "5-10p", "10-25p", "25-50p", ">50p"]
TTT_BUCKET_LABELS = ["<30m", "30-60m", "1-4h", "4-24h", ">24h/never"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def pip_size(pair: str) -> float:
    """0.01 for JPY crosses, 0.0001 for everything else."""
    return 0.01 if "JPY" in pair else 0.0001


def zone_atr_mult(pair: str) -> float:
    """Match strategy_config: CROSS uses higher multiplier."""
    ccys = pair.replace("_", "/").split("/")
    is_cross = "USD" not in ccys
    return _cfg.ZONE_TOUCH_ATR_MULT_CROSS if is_cross else _cfg.ZONE_TOUCH_ATR_MULT


def compute_atr14_zone_tol(pair: str, df_1h: pd.DataFrame) -> float:
    """
    Compute 14-bar ATR from df_1h and multiply by zone_atr_mult.
    Returns zone tolerance in price terms (same unit as price).
    Returns 0.0 if not enough data.
    """
    if len(df_1h) < 15:
        return 0.0
    h = df_1h["high"].values[-15:]
    lo = df_1h["low"].values[-15:]
    c = df_1h["close"].values[-15:]
    tr = np.maximum(
        h[1:] - lo[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])),
    )
    atr = float(np.mean(tr)) if len(tr) > 0 else 0.0
    return atr * zone_atr_mult(pair)


def dist_bucket(pips: Optional[float]) -> str:
    if pips is None:
        return "unknown"
    for edge, label in zip(DIST_BUCKET_EDGES, DIST_BUCKET_LABELS):
        if pips <= edge:
            return label
    return ">50p"


def touch_bucket(mins: Optional[int]) -> str:
    if mins is None:
        return ">24h/never"
    if mins < 30:
        return "<30m"
    if mins < 60:
        return "30-60m"
    if mins < 240:
        return "1-4h"
    if mins < 1440:
        return "4-24h"
    return ">24h/never"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_candles(
    pairs: List[str],
    from_dt: datetime,
    to_dt: datetime,
    use_cache: bool = False,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch M5, H1, H4, D candles for each pair.
    H1/H4/D include a full LOOKBACK_DAYS history before from_dt.
    M5 includes M5_LOOKBACK_DAYS before from_dt.
    Returns {oanda_pair: {"M5": df, "H1": df, "H4": df, "D": df, "W": df}}.
    """
    if use_cache and REPLAY_CACHE_PATH.exists():
        print(f"  ↩ Loading cache: {REPLAY_CACHE_PATH}")
        with open(REPLAY_CACHE_PATH, "rb") as fh:
            return pickle.load(fh)

    htf_start = (from_dt - timedelta(days=LOOKBACK_DAYS)).replace(tzinfo=timezone.utc)
    m5_start  = (from_dt - timedelta(days=M5_LOOKBACK_DAYS)).replace(tzinfo=timezone.utc)
    end       = to_dt.replace(tzinfo=timezone.utc)

    candles: Dict[str, Dict[str, pd.DataFrame]] = {}

    for pair in pairs:
        print(f"  {pair}:")
        pair_data: Dict[str, pd.DataFrame] = {}
        ok = True

        for tf, start in [("H1", htf_start), ("H4", htf_start), ("D", htf_start)]:
            _time.sleep(0.35)
            df = _fetch_range(pair, tf, from_dt=start, to_dt=end)
            if df is not None and len(df) >= 50:
                pair_data[tf] = df
                print(f"    {tf}: {len(df)} bars")
            else:
                n = len(df) if df is not None else 0
                print(f"    {tf}: ⚠ only {n} bars — skipping pair")
                if tf == "H1":
                    ok = False
                    break

        if not ok:
            continue

        _time.sleep(0.35)
        df_m5 = _fetch_m5_range(pair, from_dt=m5_start, to_dt=end)
        if df_m5 is not None and len(df_m5) > 0:
            pair_data["M5"] = df_m5
            print(f"    M5: {len(df_m5)} bars")
        else:
            print(f"    M5: ⚠ no data — skipping pair")
            continue

        if "D" in pair_data:
            pair_data["W"] = _resample_weekly(pair_data["D"])
            print(f"    W:  {len(pair_data['W'])} bars (resampled)")

        candles[pair] = pair_data

    with open(REPLAY_CACHE_PATH, "wb") as fh:
        pickle.dump(candles, fh)
    print(f"\n  Cache saved: {REPLAY_CACHE_PATH}")
    return candles


# ── Forward zone-touch scan ────────────────────────────────────────────────────

def scan_forward_touch(
    neckline: float,
    zone_tol: float,
    m5_df: pd.DataFrame,
    from_ts,
    max_bars: int = 288,   # 24 hours × 12 bars/hr
) -> Optional[int]:
    """
    Scan M5 bars after from_ts to find first zone touch.
    Touch condition (matches strategy): low ≤ neckline + tol AND high ≥ neckline − tol
    Returns minutes-to-touch, or None if no touch within max_bars.
    """
    if m5_df.empty:
        return None
    future = m5_df[m5_df.index > from_ts].iloc[:max_bars]
    for ts, row in future.iterrows():
        if float(row["low"]) <= neckline + zone_tol and float(row["high"]) >= neckline - zone_tol:
            delta = (ts - from_ts).total_seconds() / 60.0
            return int(round(delta))
    return None


# ── Replay loop ────────────────────────────────────────────────────────────────

def run_replay(
    candles: Dict[str, Dict[str, pd.DataFrame]],
    from_dt: datetime,
    to_dt: datetime,
    cadence_mins: int = 5,
) -> List[dict]:
    """
    Evaluate strategy at cadence_mins intervals over the replay window.
    Analysis only — open_positions kept empty throughout.
    Returns list of event dicts.
    """
    sf = SessionFilter()
    events: List[dict] = []

    from_naive = from_dt.replace(tzinfo=None)
    to_naive   = to_dt.replace(tzinfo=None)

    for oanda_pair, pair_data in candles.items():
        slash_pair = OANDA_TO_SLASH[oanda_pair]
        pip = pip_size(oanda_pair)

        # One fresh strategy instance per pair — keeps _TFBarCache working across
        # bars; open_positions intentionally stays empty (analysis only).
        strategy = SetAndForgetStrategy(account_balance=8000.0, risk_pct=15.0)

        m5_df = pair_data["M5"]
        h1_df = pair_data["H1"]
        h4_df = pair_data.get("H4", pd.DataFrame())
        d_df  = pair_data.get("D",  pd.DataFrame())
        w_df  = pair_data.get("W",  pd.DataFrame())

        # Replay window — only M5 bars inside [from, to]
        m5_window = m5_df[
            (m5_df.index >= from_naive) & (m5_df.index <= to_naive)
        ]

        n_sess_block = 0
        n_evaluated  = 0
        t_start = _time.time()

        for ts in m5_window.index:
            # Only evaluate at cadence boundaries
            if ts.minute % cadence_mins != 0:
                continue

            ts_utc = ts.to_pydatetime().replace(tzinfo=timezone.utc)

            # Fast path: session check before expensive DataFrame slicing
            allowed, sess_reason = sf.is_entry_allowed(ts_utc)
            if not allowed:
                n_sess_block += 1
                # Record only at hourly boundaries to keep event list manageable
                if ts.minute == 0:
                    events.append({
                        "ts": ts,
                        "pair": slash_pair,
                        "decision": "SESSION_BLOCKED",
                        "wait_reasons": [sess_reason],
                        "pattern": None,
                        "confidence": None,
                        "neckline": None,
                        "zone_min_distance_pips": None,
                        "zone_touch_type": None,
                        "zone_tol": None,
                        "is_hourly_boundary": True,
                        "pre_candidate_count": 0,
                    })
                continue

            # Slice each timeframe to history available at ts
            df_1h_s = h1_df[h1_df.index <= ts]
            df_4h_s = h4_df[h4_df.index <= ts] if not h4_df.empty else pd.DataFrame()
            df_d_s  = d_df[d_df.index   <= ts] if not d_df.empty  else pd.DataFrame()
            df_w_s  = w_df[w_df.index   <= ts] if not w_df.empty  else pd.DataFrame()

            if len(df_1h_s) < 50:
                continue  # insufficient history

            # Current price from M5 bar close
            cur_price = float(m5_window.loc[ts, "close"])

            try:
                decision = strategy.evaluate(
                    slash_pair,
                    df_w_s, df_d_s, df_4h_s, df_1h_s,
                    current_price=cur_price,
                    current_dt=ts_utc,
                )
            except Exception as exc:
                events.append({
                    "ts": ts,
                    "pair": slash_pair,
                    "decision": "ERROR",
                    "wait_reasons": [str(exc)[:120]],
                    "pattern": None,
                    "confidence": None,
                    "neckline": None,
                    "zone_min_distance_pips": None,
                    "zone_touch_type": None,
                    "zone_tol": None,
                    "is_hourly_boundary": ts.minute == 0,
                    "pre_candidate_count": 0,
                })
                continue

            n_evaluated += 1

            # Neckline + zone tolerance (needed for forward scan in post-process)
            neckline = None
            zone_tol = None
            if decision.pattern is not None:
                raw_nl = getattr(decision.pattern, "neckline", None)
                if raw_nl is not None:
                    neckline = float(raw_nl)
                    zone_tol = compute_atr14_zone_tol(oanda_pair, df_1h_s)

            # PRE_CANDIDATE: sub-threshold patterns collected as side-effect in evaluate()
            pre_cand_list = (
                getattr(strategy, "_pre_candidate_data", {}).get(slash_pair, [])
            )
            pre_candidate_count = len(pre_cand_list)

            events.append({
                "ts":    ts,
                "pair":  slash_pair,
                "decision": decision.decision.value,
                "wait_reasons": list(decision.failed_filters),
                "pattern":    str(decision.pattern.pattern_type) if decision.pattern else None,
                "confidence": decision.confidence,
                "neckline":   neckline,
                "zone_min_distance_pips": decision.zone_min_distance_pips,
                "zone_touch_type": decision.zone_touch_type_seen,
                "zone_tol": zone_tol,
                "is_hourly_boundary": ts.minute == 0,
                "pre_candidate_count": pre_candidate_count,
            })

        elapsed = _time.time() - t_start
        print(
            f"  {slash_pair}: {len(m5_window):,} M5 bars | "
            f"{n_sess_block:,} session-blocked | "
            f"{n_evaluated:,} evaluated | {elapsed:.1f}s"
        )

    return events


# ── Post-processing ────────────────────────────────────────────────────────────

def add_time_to_touch(
    events: List[dict],
    candles: Dict[str, Dict[str, pd.DataFrame]],
) -> List[dict]:
    """
    For each NO_ZONE_TOUCH event with a known neckline + zone_tol, scan forward
    in M5 data to find time-to-touch.  Adds "time_to_touch_mins" field.
    """
    pair_m5 = {OANDA_TO_SLASH[k]: v["M5"] for k, v in candles.items()}

    for evt in events:
        if "no_zone_touch" not in evt.get("wait_reasons", []):
            continue
        if evt.get("neckline") is None or evt.get("zone_tol") is None:
            evt["time_to_touch_mins"] = None
            continue
        m5 = pair_m5.get(evt["pair"])
        if m5 is None:
            evt["time_to_touch_mins"] = None
            continue
        evt["time_to_touch_mins"] = scan_forward_touch(
            neckline=evt["neckline"],
            zone_tol=evt["zone_tol"],
            m5_df=m5,
            from_ts=evt["ts"],
        )

    return events


def add_virtual_missed(events: List[dict]) -> List[dict]:
    """
    VIRTUAL_ENTRY_MISSED_BY_HOURLY:
    An hourly evaluation has NO_ZONE_TOUCH AND a zone touch happens within
    the next 60 minutes (on a 5m bar the hourly bot can't see).
    The hourly bot would have evaluated, seen no touch, and the touch occurred
    between its hourly scans — exactly the cadence gap we are measuring.
    """
    for evt in events:
        is_vm = (
            evt.get("is_hourly_boundary")
            and "no_zone_touch" in evt.get("wait_reasons", [])
            and evt.get("time_to_touch_mins") is not None
            and evt["time_to_touch_mins"] < 60
            and evt["time_to_touch_mins"] > 0   # touch must be AFTER hourly bar, not on it
        )
        evt["virtual_entry_missed"] = bool(is_vm)

    return events


# ── Report generation ──────────────────────────────────────────────────────────

def generate_report(
    events: List[dict],
    from_dt: datetime,
    to_dt: datetime,
    cadence_mins: int,
    pairs: List[str],
    out_path: Path,
) -> str:
    lines: List[str] = []
    a = lines.append

    slash_pairs = [OANDA_TO_SLASH[p] for p in pairs if p in OANDA_TO_SLASH]

    # Split event pools
    sess_blocked   = [e for e in events if e.get("decision") == "SESSION_BLOCKED"]
    errors         = [e for e in events if e.get("decision") == "ERROR"]
    real_events    = [e for e in events if e.get("decision") not in ("SESSION_BLOCKED", "ERROR")]
    cand_waits     = [e for e in real_events if e.get("decision") == "WAIT" and e.get("pattern")]
    no_zone        = [e for e in real_events if "no_zone_touch" in e.get("wait_reasons", [])]
    pre_cands      = [e for e in real_events if e.get("pre_candidate_count", 0) > 0]
    virtual_missed = [e for e in events if e.get("virtual_entry_missed")]
    zone_touched   = [e for e in no_zone if e.get("time_to_touch_mins") is not None]

    # ── Header ─────────────────────────────────────────────────────────────────
    a("# Replay Research Report")
    a(f"Window: {from_dt.date()} → {to_dt.date()} | Cadence: {cadence_mins}m | Pairs: {len(slash_pairs)}")
    a(f"Pairs: {', '.join(slash_pairs)}")
    a("")
    a("---")
    a("")

    # ── Summary ────────────────────────────────────────────────────────────────
    a("## Summary")
    a(f"| Metric | Value |")
    a(f"|--------|------:|")
    a(f"| Total 5m evaluations (session-allowed) | {len(real_events):,} |")
    a(f"| Session-blocked (hourly records only) | {len(sess_blocked):,} |")
    a(f"| Evaluation errors | {len(errors):,} |")
    a(f"| CANDIDATE_WAIT (pattern present) | {len(cand_waits):,} |")
    a(f"| NO_ZONE_TOUCH events | {len(no_zone):,} |")
    a(f"| PRE_CANDIDATE (sub-threshold, clarity < 0.4) | {len(pre_cands):,} |")
    a(f"| Zone touches detected (any M5 bar, ≤24h) | {len(zone_touched):,} |")
    a(f"| VIRTUAL_ENTRY_MISSED_BY_HOURLY | {len(virtual_missed):,} |")
    if no_zone:
        pct = 100.0 * len(virtual_missed) / len(no_zone)
        a(f"| % of NO_ZONE_TOUCH missed by hourly | {pct:.1f}% |")
    a("")

    # ── A) CANDIDATE_WAIT by reason ────────────────────────────────────────────
    a("## A) CANDIDATE_WAIT by reason")
    reason_counts: Dict[str, int] = defaultdict(int)
    for e in real_events:
        if e.get("decision") == "WAIT":
            reasons = e.get("wait_reasons") or ["(no reason)"]
            for r in reasons:
                reason_counts[r] += 1
    total_w = sum(reason_counts.values()) or 1
    a(f"| Reason | Count | % of WAIT reasons |")
    a(f"|--------|------:|------------------:|")
    for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        a(f"| `{reason}` | {cnt:,} | {100*cnt/total_w:.1f}% |")
    a("")

    # ── B) NO_ZONE_TOUCH — zone_min_distance_pips distribution ─────────────────
    a("## B) NO_ZONE_TOUCH — zone_min_distance_pips distribution")
    buckets:      Dict[str, int] = defaultdict(int)
    wick_counts:  Dict[str, int] = defaultdict(int)
    body_counts:  Dict[str, int] = defaultdict(int)
    for e in no_zone:
        b = dist_bucket(e.get("zone_min_distance_pips"))
        buckets[b] += 1
        tt = e.get("zone_touch_type") or ""
        if tt == "wick":
            wick_counts[b] += 1
        elif tt == "body":
            body_counts[b] += 1
    total_nz = len(no_zone) or 1
    ordered = DIST_BUCKET_LABELS + ["unknown"]   # DIST_BUCKET_LABELS already ends with >50p
    a(f"| Bucket | Count | % | Wick | Body |")
    a(f"|--------|------:|--:|-----:|-----:|")
    for b in ordered:
        cnt = buckets.get(b, 0)
        if cnt == 0:
            continue
        a(f"| {b} | {cnt:,} | {100*cnt/total_nz:.1f}% | {wick_counts.get(b,0)} | {body_counts.get(b,0)} |")
    a("")

    # ── C) Time-to-touch distribution ──────────────────────────────────────────
    a("## C) Time-to-touch distribution")
    ttt: Dict[str, int] = defaultdict(int)
    for e in no_zone:
        ttt[touch_bucket(e.get("time_to_touch_mins"))] += 1
    a(f"| Bucket | Count | % of NO_ZONE_TOUCH |")
    a(f"|--------|------:|------------------:|")
    for b in TTT_BUCKET_LABELS:
        cnt = ttt.get(b, 0)
        a(f"| {b} | {cnt:,} | {100*cnt/total_nz:.1f}% |")
    a("")

    # ── D) Per-pair / per-pattern top-10 near-miss table ───────────────────────
    a("## D) Per-pair / per-pattern near-miss table (top 10)")
    pp_map: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for e in no_zone:
        pp_map[(e["pair"], e.get("pattern") or "unknown")].append(e)
    pp_rows = []
    for (pair, pat), evts in pp_map.items():
        dists = [e["zone_min_distance_pips"] for e in evts if e.get("zone_min_distance_pips") is not None]
        ttts  = [e["time_to_touch_mins"] for e in evts if e.get("time_to_touch_mins") is not None]
        med_d = float(np.median(dists)) if dists else None
        med_t = int(np.median(ttts)) if ttts else None
        pp_rows.append((pair, pat, len(evts), med_d, med_t))
    pp_rows.sort(key=lambda x: -x[2])
    a(f"| Pair | Pattern | NO_ZONE_TOUCH | Median dist (pips) | Median time-to-touch |")
    a(f"|------|---------|-------------:|-------------------:|---------------------:|")
    for pair, pat, cnt, med_d, med_t in pp_rows[:10]:
        ds = f"{med_d:.1f}p" if med_d is not None else "—"
        ts = f"{med_t}m"    if med_t is not None else "—"
        a(f"| {pair} | {pat} | {cnt:,} | {ds} | {ts} |")
    a("")

    # ── E) Hourly vs 5m cadence comparison ─────────────────────────────────────
    a("## E) Hourly vs 5m cadence comparison")
    hourly_nz   = [e for e in no_zone if e.get("is_hourly_boundary")]
    intra_nz    = [e for e in no_zone if not e.get("is_hourly_boundary")]
    hourly_zt   = [e for e in hourly_nz if e.get("time_to_touch_mins") is not None]
    intra_zt    = [e for e in intra_nz  if e.get("time_to_touch_mins") is not None]
    total_zt    = len(zone_touched) or 1
    a(f"| Metric | Count | % |")
    a(f"|--------|------:|--:|")
    a(f"| Total zone touches detected (any scan, ≤24h) | {len(zone_touched):,} | 100% |")
    a(f"| Zone touches at hourly boundary | {len(hourly_zt):,} | {100*len(hourly_zt)/total_zt:.1f}% |")
    a(f"| Zone touches at non-hourly 5m bar | {len(intra_zt):,} | {100*len(intra_zt)/total_zt:.1f}% |")
    a(f"| VIRTUAL_ENTRY_MISSED_BY_HOURLY (touch within 60m of hourly eval) | {len(virtual_missed):,} | {100*len(virtual_missed)/total_zt:.1f}% |")
    a("")
    a("### Per-pair breakdown")
    a(f"| Pair | NO_ZONE_TOUCH | Zone touches ≤24h | At hourly | Non-hourly | Virtual missed |")
    a(f"|------|-------------:|------------------:|----------:|-----------:|---------------:|")
    for sp in slash_pairs:
        p_nz   = [e for e in no_zone     if e["pair"] == sp]
        p_zt   = [e for e in p_nz        if e.get("time_to_touch_mins") is not None]
        p_h    = [e for e in p_zt        if e.get("is_hourly_boundary")]
        p_ih   = [e for e in p_zt        if not e.get("is_hourly_boundary")]
        p_vm   = [e for e in virtual_missed if e["pair"] == sp]
        a(f"| {sp} | {len(p_nz):,} | {len(p_zt):,} | {len(p_h):,} | {len(p_ih):,} | {len(p_vm):,} |")
    a("")

    # ── F) Daily counts ─────────────────────────────────────────────────────────
    a("## F) Daily counts")
    daily: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "evals": 0, "cw": 0, "nz": 0, "zt": 0, "vm": 0
    })
    for e in real_events:
        d = str(e["ts"].date())
        daily[d]["evals"] += 1
        if e.get("decision") == "WAIT" and e.get("pattern"):
            daily[d]["cw"] += 1
        if "no_zone_touch" in e.get("wait_reasons", []):
            daily[d]["nz"] += 1
            if e.get("time_to_touch_mins") is not None:
                daily[d]["zt"] += 1
        if e.get("virtual_entry_missed"):
            daily[d]["vm"] += 1
    a(f"| Date | Evals | CANDIDATE_WAIT | NO_ZONE_TOUCH | Zone touches ≤24h | Virtual missed |")
    a(f"|------|------:|---------------:|--------------:|------------------:|---------------:|")
    for date in sorted(daily.keys()):
        r = daily[date]
        a(f"| {date} | {r['evals']:,} | {r['cw']:,} | {r['nz']:,} | {r['zt']:,} | {r['vm']:,} |")
    a("")

    # ── Totals ──────────────────────────────────────────────────────────────────
    a("## Totals")
    a(f"- Total NO_ZONE_TOUCH misses: **{len(no_zone):,}**")
    a("")

    by_pair: Dict[str, int] = defaultdict(int)
    by_pat:  Dict[str, int] = defaultdict(int)
    for e in no_zone:
        by_pair[e["pair"]] += 1
        by_pat[e.get("pattern") or "unknown"] += 1

    a("### Misses per pair")
    a(f"| Pair | Count |")
    a(f"|------|------:|")
    for pair, cnt in sorted(by_pair.items(), key=lambda x: -x[1]):
        a(f"| {pair} | {cnt:,} |")
    a("")

    a("### Misses per pattern")
    a(f"| Pattern | Count |")
    a(f"|---------|------:|")
    for pat, cnt in sorted(by_pat.items(), key=lambda x: -x[1]):
        a(f"| {pat} | {cnt:,} |")
    a("")

    # % of all zone touches missed by hourly
    intra_hour_zt = len([
        e for e in no_zone
        if not e.get("is_hourly_boundary")
        and e.get("time_to_touch_mins") is not None
        and e["time_to_touch_mins"] < 60
    ])
    hourly_zt_cnt = len([
        e for e in no_zone
        if e.get("is_hourly_boundary")
        and e.get("time_to_touch_mins") is not None
    ])
    total_with_touch = intra_hour_zt + hourly_zt_cnt
    if total_with_touch > 0:
        pct_missed = 100.0 * intra_hour_zt / total_with_touch
        a("### % of all zone touches missed by hourly")
        a(f"Zone touches on non-hourly M5 bars (within 60m): "
          f"**{intra_hour_zt:,} / {total_with_touch:,} ({pct_missed:.1f}%)**")
        a("")

    report_str = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_str, encoding="utf-8")
    return report_str


# ── CLI entry point ────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Replay research harness — 5m cadence zone-touch analysis (research only)"
    )
    parser.add_argument("--from",    dest="from_date", default="2026-02-01",
                        help="Replay window start (YYYY-MM-DD)")
    parser.add_argument("--to",      dest="to_date",   default="2026-03-04",
                        help="Replay window end (YYYY-MM-DD)")
    parser.add_argument("--cadence", type=int, default=5,
                        help="Minutes between evaluations (default: 5)")
    parser.add_argument("--pairs",   nargs="+", default=ALEX_PAIRS,
                        help="OANDA pair codes (e.g. GBP_JPY)")
    parser.add_argument("--out", type=Path,
                        default=Path("backtesting/results/replay_research_2026-02-01_to_2026-03-04.md"),
                        help="Output markdown report path")
    parser.add_argument("--use-cache", action="store_true",
                        help=f"Load candle data from cache ({REPLAY_CACHE_PATH})")
    args = parser.parse_args(argv)

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d")
    to_dt   = datetime.strptime(args.to_date,   "%Y-%m-%d").replace(hour=23, minute=59)

    _W = 60
    print(f"\n{'='*_W}")
    print(f" Replay Research Harness")
    print(f" Window : {from_dt.date()} → {to_dt.date()}")
    print(f" Cadence: {args.cadence}m")
    print(f" Pairs  : {', '.join(args.pairs)}")
    print(f" Cache  : {'YES (--use-cache)' if args.use_cache else 'NO — will fetch from OANDA'}")
    print(f"{'='*_W}\n")

    # ── Step 1: candles ────────────────────────────────────────────────────────
    print("Step 1/4  Loading candles...")
    candles = load_candles(args.pairs, from_dt, to_dt, use_cache=args.use_cache)
    if not candles:
        print("ERROR: no candle data loaded — check OANDA credentials")
        sys.exit(1)
    print(f"  Loaded {len(candles)} pairs\n")

    # ── Step 2: replay ─────────────────────────────────────────────────────────
    print(f"Step 2/4  Running {args.cadence}m replay...")
    t_replay = _time.time()
    events = run_replay(candles, from_dt, to_dt, cadence_mins=args.cadence)
    print(f"  Done: {len(events):,} events recorded in {_time.time()-t_replay:.1f}s\n")

    # ── Step 3: post-processing ────────────────────────────────────────────────
    print("Step 3/4  Post-processing...")
    events = add_time_to_touch(events, candles)
    events = add_virtual_missed(events)
    n_nz  = len([e for e in events if "no_zone_touch" in e.get("wait_reasons", [])])
    n_zt  = len([e for e in events if e.get("time_to_touch_mins") is not None])
    n_vm  = len([e for e in events if e.get("virtual_entry_missed")])
    print(f"  NO_ZONE_TOUCH: {n_nz:,}  |  zone touches found: {n_zt:,}  |  virtual missed: {n_vm:,}\n")

    # ── Step 4: report ─────────────────────────────────────────────────────────
    print(f"Step 4/4  Generating report → {args.out}")
    report = generate_report(events, from_dt, to_dt, args.cadence, args.pairs, args.out)
    rlines = report.split("\n")
    # Print first 50 lines for quick review
    print("\n" + "\n".join(rlines[:50]))
    if len(rlines) > 50:
        print(f"\n[... {len(rlines)-50} more lines — see {args.out} for full report ...]")
    print(f"\n{'='*_W}")
    print(f" Report saved: {args.out}")
    print(f"{'='*_W}\n")


if __name__ == "__main__":
    main()
