#!/usr/bin/env python3
"""
near_miss_analysis.py
=====================
Extract and report CANDIDATE_WAIT + CANDIDATE_BLOCKED records from
decision_log.jsonl for a given time window.

Usage
-----
  python3 scripts/near_miss_analysis.py \\
      --from 2026-03-03T13:00:00Z \\
      --to   2026-03-03T14:00:00Z

  # Or just the last N hours:
  python3 scripts/near_miss_analysis.py --last-hours 1

Options
-------
  --log PATH       path to decision_log.jsonl  [default: logs/decision_log.jsonl]
  --from TS        ISO-8601 start timestamp (inclusive)
  --to   TS        ISO-8601 end timestamp   (inclusive)
  --last-hours N   analyse the last N hours (overrides --from/--to)
  --top N          top-N closest misses     [default: 20]

Output sections
---------------
  D  CANDIDATE_WAIT + CANDIDATE_BLOCKED counts by reason / pair
  E  Proximity buckets (conf gap, RR gap)
  F  Top-N closest misses sorted by gap ascending

Notes
-----
  - CANDIDATE_WAIT is the primary data source (fires when pattern found but
    decision=WAIT — captures near-miss data before the ENTER stage).
  - CANDIDATE_BLOCKED supplements (fires when decision=ENTER but downstream
    gate rejected it).
  - Throttle dedup (1h/same key) is enforced by the logger; this script reports
    raw counts as-is.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path


def _ts(r: dict) -> str:
    return r.get("timestamp_utc", r.get("ts", ""))


def load_records(log_path: Path, ts_from: str, ts_to: str) -> list:
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") not in ("CANDIDATE_WAIT", "CANDIDATE_BLOCKED"):
                continue
            ts = _ts(r)
            if ts and ts_from <= ts <= ts_to:
                records.append(r)
    return records


def section_d(records: list) -> None:
    wait_recs    = [r for r in records if r["event"] == "CANDIDATE_WAIT"]
    blocked_recs = [r for r in records if r["event"] == "CANDIDATE_BLOCKED"]

    print("══════════════════════════════════════════════")
    print("D) CANDIDATE SUMMARY")
    print("══════════════════════════════════════════════")
    print(f"  CANDIDATE_WAIT    : {len(wait_recs)}")
    print(f"  CANDIDATE_BLOCKED : {len(blocked_recs)}")
    print(f"  Total             : {len(records)}")

    # By reason — WAIT uses wait_reasons (list), BLOCKED uses block_reasons (list)
    all_reasons: Counter = Counter()
    for r in wait_recs:
        for reason in r.get("wait_reasons", []):
            all_reasons[f"WAIT/{reason}"] += 1
    for r in blocked_recs:
        for reason in r.get("block_reasons", []):
            all_reasons[f"BLOCKED/{reason}"] += 1

    print("\n  By reason (desc):")
    if all_reasons:
        for reason, cnt in all_reasons.most_common():
            print(f"    {reason:<42} {cnt}")
    else:
        print("    (none)")

    # Top pairs by WAIT count
    wait_pair_ctr: Counter = Counter(r.get("pair", "?") for r in wait_recs)
    print("\n  Top 10 pairs by WAIT count:")
    if wait_pair_ctr:
        for pair, cnt in wait_pair_ctr.most_common(10):
            print(f"    {pair:<14} {cnt}")
    else:
        print("    (none)")


def section_e(records: list) -> None:
    print("\n══════════════════════════════════════════════")
    print("E) PROXIMITY BUCKETS")
    print("══════════════════════════════════════════════")

    # CONFIDENCE proximity — WAIT uses conf_gap directly; BLOCKED derives it
    def get_conf_gap(r: dict):
        if r["event"] == "CANDIDATE_WAIT":
            return r.get("conf_gap")
        # CANDIDATE_BLOCKED: derive from fields
        thr  = r.get("confidence_threshold")
        cand = r.get("candidate_confidence")
        return (float(thr) - float(cand)) if thr is not None and cand is not None else None

    conf_recs = [r for r in records
                 if (r["event"] == "CANDIDATE_WAIT" and "CONFIDENCE_BELOW_MIN" in r.get("wait_reasons", []))
                 or (r["event"] == "CANDIDATE_BLOCKED" and "CONFIDENCE_BLOCK" in r.get("block_reasons", []))]
    bc = {"[0–0.02]": 0, "(0.02–0.05]": 0, "(0.05–0.10]": 0, ">0.10": 0}
    for r in conf_recs:
        d = get_conf_gap(r)
        if d is None:
            continue
        if   d <= 0.02: bc["[0–0.02]"]    += 1
        elif d <= 0.05: bc["(0.02–0.05]"] += 1
        elif d <= 0.10: bc["(0.05–0.10]"] += 1
        else:           bc[">0.10"]        += 1
    print(f"  CONFIDENCE gap (threshold − confidence)  [n={len(conf_recs)}]")
    for b, cnt in bc.items():
        print(f"    {b:<14} {cnt}")

    # RR proximity — WAIT uses rr_gap directly; BLOCKED derives it
    def get_rr_gap(r: dict):
        if r["event"] == "CANDIDATE_WAIT":
            return r.get("rr_gap")
        thr  = r.get("min_rr_threshold")
        cand = r.get("candidate_rr")
        return (float(thr) - float(cand)) if thr is not None and cand is not None else None

    rr_recs = [r for r in records
               if (r["event"] == "CANDIDATE_WAIT" and r.get("rr_gap") is not None)
               or (r["event"] == "CANDIDATE_BLOCKED" and "RR_BLOCK" in r.get("block_reasons", []))]
    br = {"[0–0.2]": 0, "(0.2–0.5]": 0, "(0.5–1.0]": 0, ">1.0": 0}
    for r in rr_recs:
        d = get_rr_gap(r)
        if d is None:
            continue
        if   d <= 0.2: br["[0–0.2]"]   += 1
        elif d <= 0.5: br["(0.2–0.5]"] += 1
        elif d <= 1.0: br["(0.5–1.0]"] += 1
        else:          br[">1.0"]       += 1
    print(f"\n  RR gap (threshold − rr)                  [n={len(rr_recs)}]")
    for b, cnt in br.items():
        print(f"    {b:<14} {cnt}")


def section_f(records: list, top_n: int = 20) -> None:
    print("\n══════════════════════════════════════════════")
    print("F) TOP CLOSEST MISSES")
    print("══════════════════════════════════════════════")

    # ── CONFIDENCE closest misses ─────────────────────────────────────────────
    def conf_gap_val(r: dict):
        if r["event"] == "CANDIDATE_WAIT":
            v = r.get("conf_gap")
        else:
            thr  = r.get("confidence_threshold")
            cand = r.get("candidate_confidence")
            v = (float(thr) - float(cand)) if thr is not None and cand is not None else None
        return v if v is not None else float("inf")

    conf_pool = [r for r in records
                 if (r["event"] == "CANDIDATE_WAIT" and "CONFIDENCE_BELOW_MIN" in r.get("wait_reasons", []))
                 or (r["event"] == "CANDIDATE_BLOCKED" and "CONFIDENCE_BLOCK" in r.get("block_reasons", []))]
    # Keep only true misses (gap > 0 means confidence is below threshold)
    conf_pool = [r for r in conf_pool if conf_gap_val(r) > 0]
    conf_sorted = sorted(conf_pool, key=conf_gap_val)[:top_n]

    print(f"\n  Confidence closest misses — top {top_n} (n={len(conf_pool)} total)")
    if conf_sorted:
        hdr = f"  {'pair':<12} {'pattern':<22} {'dir':<6} {'conf':>6} {'thr':>6} {'gap':>7} {'evt':<8}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for r in conf_sorted:
            pair = r.get("pair", "?")
            pat  = r.get("pattern", "?")
            dirn = r.get("direction", "?")
            thr  = r.get("confidence_threshold") or 0
            cand = r.get("candidate_confidence") or 0
            gap  = conf_gap_val(r)
            evt  = "WAIT" if r["event"] == "CANDIDATE_WAIT" else "BLOCKED"
            print(f"  {pair:<12} {str(pat):<22} {dirn:<6} {cand:>5.0%}  {thr:>5.0%}  {gap:>+6.3f}  {evt}")
    else:
        print("  (none)")

    # ── RR closest misses ─────────────────────────────────────────────────────
    def rr_gap_val(r: dict):
        if r["event"] == "CANDIDATE_WAIT":
            v = r.get("rr_gap")
        else:
            thr  = r.get("min_rr_threshold")
            cand = r.get("candidate_rr")
            v = (float(thr) - float(cand)) if thr is not None and cand is not None else None
        return v if v is not None else float("inf")

    rr_pool = [r for r in records
               if (r["event"] == "CANDIDATE_WAIT" and r.get("rr_gap") is not None)
               or (r["event"] == "CANDIDATE_BLOCKED" and "RR_BLOCK" in r.get("block_reasons", []))]
    # Keep only true misses (gap > 0 means RR is below threshold)
    rr_pool = [r for r in rr_pool if rr_gap_val(r) > 0]
    rr_sorted = sorted(rr_pool, key=rr_gap_val)[:top_n]

    print(f"\n  RR closest misses — top {top_n} (n={len(rr_pool)} total)")
    if rr_sorted:
        hdr = f"  {'pair':<12} {'pattern':<22} {'dir':<6} {'rr':>6} {'thr':>6} {'gap':>7} {'evt':<8}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for r in rr_sorted:
            pair = r.get("pair", "?")
            pat  = r.get("pattern", "?")
            dirn = r.get("direction", "?")
            thr  = r.get("min_rr_threshold") or 0
            cand = r.get("candidate_rr") or 0
            gap  = rr_gap_val(r)
            evt  = "WAIT" if r["event"] == "CANDIDATE_WAIT" else "BLOCKED"
            print(f"  {pair:<12} {str(pat):<22} {dirn:<6} {cand:>5.2f}   {thr:>5.2f}  {gap:>+6.3f}  {evt}")
    else:
        print("  (none)")

    print("\n══════════════════════════════════════════════")


def main() -> None:
    parser = argparse.ArgumentParser(description="Near-miss distribution analysis")
    parser.add_argument("--log",        default="logs/decision_log.jsonl",
                        help="Path to decision_log.jsonl")
    parser.add_argument("--from",       dest="ts_from", default=None,
                        help="ISO-8601 start (inclusive)")
    parser.add_argument("--to",         dest="ts_to",   default=None,
                        help="ISO-8601 end (inclusive)")
    parser.add_argument("--last-hours", type=float,     default=None,
                        help="Analyse last N hours (overrides --from/--to)")
    parser.add_argument("--top",        type=int,       default=20,
                        help="Top-N closest misses (default 20)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = Path(__file__).parent.parent / log_path

    now_utc = datetime.now(timezone.utc)
    if args.last_hours is not None:
        ts_to   = now_utc.isoformat()
        ts_from = (now_utc - timedelta(hours=args.last_hours)).isoformat()
    elif args.ts_from and args.ts_to:
        ts_from = args.ts_from
        ts_to   = args.ts_to
    else:
        parser.error("Provide --last-hours OR both --from and --to")
        return

    print(f"\nWindow : {ts_from} → {ts_to}")
    print(f"Log    : {log_path}\n")

    records = load_records(log_path, ts_from, ts_to)
    section_d(records)
    section_e(records)
    section_f(records, top_n=args.top)


if __name__ == "__main__":
    main()
