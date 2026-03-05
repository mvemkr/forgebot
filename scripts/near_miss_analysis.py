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
  S  SCAN_SUMMARY  — aggregate scan stats (SCAN_HEARTBEAT events); shown
                     separately, NEVER mixed into candidate totals.
                     Pairs with no_pattern appear ONLY here.
  D  CANDIDATE_WAIT + CANDIDATE_BLOCKED counts by reason / pair
     *** Counts derived ONLY from CANDIDATE_WAIT / CANDIDATE_BLOCKED events.
     *** Records with no_pattern cannot appear here: the orchestrator emits
         CANDIDATE_WAIT only when decision.pattern is not None.
  E  Proximity buckets (conf gap, RR gap)
     rr_unavailable count shown separately (entry not reached → exec_rr=0.0).
  F  Top-N closest misses sorted by gap ascending
     Negative-gap rows excluded (gap must be > 0).
     rr_unavailable rows excluded from RR closest misses.

Notes
-----
  - CANDIDATE_WAIT fires when pattern found but decision=WAIT, BEFORE ENTER
    stage.  Zero broker risk.
  - CANDIDATE_BLOCKED fires when decision=ENTER but a post-signal gate blocked
    it.
  - rr_unavailable=True means the WAIT record has a pattern but RR was not
    computed (entry signal absent → _calculate_entry never ran → exec_rr=0.0
    default).  These appear in CONFIDENCE proximity but NOT in RR proximity.
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
    """Load only CANDIDATE_WAIT / CANDIDATE_BLOCKED events in window.

    Invariant: these events are emitted by the orchestrator only when
    decision.pattern is not None, so no_pattern rows can never appear here.
    """
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
                # Defensive: drop any record that somehow has no pattern
                # (should never happen, but guard analysis correctness).
                if r.get("pattern") is None:
                    continue
                records.append(r)
    return records


def load_scan_records(log_path: Path, ts_from: str, ts_to: str) -> list:
    """Load SCAN_HEARTBEAT events in window (for SCAN_SUMMARY section only)."""
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
            if r.get("event") != "SCAN_HEARTBEAT":
                continue
            ts = _ts(r)
            if ts and ts_from <= ts <= ts_to:
                records.append(r)
    return records


def load_pre_candidate_records(log_path: Path, ts_from: str, ts_to: str) -> list:
    """Load PRE_CANDIDATE events in window.

    PRE_CANDIDATE fires when pattern detector found a forming pattern below
    the recognition floor (clarity < min_pattern_clarity=0.4) AND
    decision.pattern is None (formal recognition never occurred).
    These are distinct from CANDIDATE_WAIT which requires decision.pattern != None.
    """
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
            if r.get("event") != "PRE_CANDIDATE":
                continue
            ts = _ts(r)
            if ts and ts_from <= ts <= ts_to:
                records.append(r)
    return records


def section_s(scan_records: list) -> None:
    """SCAN_SUMMARY — aggregate totals from SCAN_HEARTBEAT events.

    Displayed BEFORE candidate sections.  Pairs with no_pattern appear ONLY here.
    These numbers are never mixed into D/E/F candidate counts.
    """
    print("══════════════════════════════════════════════")
    print("S) SCAN SUMMARY  (SCAN_HEARTBEAT — not candidate data)")
    print("══════════════════════════════════════════════")
    if not scan_records:
        print("  No SCAN_HEARTBEAT events in window.")
        return

    total_scans   = len(scan_records)
    total_pairs   = sum(r.get("pairs_scanned", 0) for r in scan_records)
    total_wait    = sum(r.get("wait_count",    0) for r in scan_records)
    total_enter   = sum(r.get("enter_count",   0) for r in scan_records)
    total_blocked = sum(r.get("blocked_count", 0) for r in scan_records)

    print(f"  Scans in window       : {total_scans}")
    print(f"  Pair-scans total      : {total_pairs}  (includes no_pattern placeholders)")
    print(f"  Aggregate WAIT        : {total_wait}")
    print(f"  Aggregate ENTER       : {total_enter}")
    print(f"  Aggregate BLOCKED     : {total_blocked}")
    print()
    print("  ⚠ SCAN totals include no_pattern pairs (conf=0.30 placeholders).")
    print("    True near-miss candidates are in section D below (CANDIDATE events only).")


def section_p(pre_records: list) -> None:
    """P) PRE_CANDIDATE SUMMARY — patterns below recognition floor.

    These are DISTINCT from CANDIDATE_WAIT:
      PRE_CANDIDATE  → decision.pattern is None (clarity < recognition_floor)
      CANDIDATE_WAIT → decision.pattern is not None (formally recognized, but WAIT)

    The gap_to_floor buckets mirror section E's conf_gap buckets so the two
    populations can be visually compared.
    """
    print("══════════════════════════════════════════════")
    print("P) PRE_CANDIDATE SUMMARY  (below recognition floor)")
    print("══════════════════════════════════════════════")
    if not pre_records:
        print("  PRE_CANDIDATE events : 0  (no sub-threshold patterns in window)")
        return

    print(f"  PRE_CANDIDATE events : {len(pre_records)}")
    print()

    # ── gap_to_floor buckets ──────────────────────────────────────────
    pb = {"[0–0.02]": [], "(0.02–0.05]": [], "(0.05–0.10]": [], ">0.10": []}
    for r in pre_records:
        g = r.get("confidence_gap_to_floor")
        if g is None or g <= 0:
            continue
        if   g <= 0.02: pb["[0–0.02]"].append(r)
        elif g <= 0.05: pb["(0.02–0.05]"].append(r)
        elif g <= 0.10: pb["(0.05–0.10]"].append(r)
        else:           pb[">0.10"].append(r)

    print("  gap_to_floor buckets (recognition_floor − raw_confidence):")
    for bucket, items in pb.items():
        pairs_str = ", ".join(f"{r.get('pair','?')}({r.get('confidence_gap_to_floor',0):.3f})"
                              for r in items[:5])
        print(f"    {bucket:<14} {len(items):3d}  {pairs_str}")
    print()

    # ── Top 20 closest PRE_CANDIDATE (smallest gap > 0) ──────────────
    pool = [r for r in pre_records
            if r.get("confidence_gap_to_floor") is not None
            and r.get("confidence_gap_to_floor") > 0]
    pool_sorted = sorted(pool, key=lambda r: r["confidence_gap_to_floor"])[:20]

    print(f"  Top {min(20, len(pool_sorted))} closest PRE_CANDIDATE  (n={len(pool)} total, gap>0)")
    if pool_sorted:
        hdr = (f"  {'pair':<12} {'pattern_type':<22} {'dir':<6} "
               f"{'raw_conf':>8} {'floor':>6} {'gap':>7} {'session':<6}")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for r in pool_sorted:
            pair  = r.get("pair", "?")
            ptype = str(r.get("pattern_type") or "?")
            dirn  = str(r.get("direction") or "?")
            rc    = r.get("raw_confidence", 0)
            fl    = r.get("recognition_floor", 0.4)
            gap   = r.get("confidence_gap_to_floor", 0)
            sess  = "yes" if r.get("session_allowed") else "no"
            print(f"  {pair:<12} {ptype:<22} {dirn:<6} "
                  f"{rc:>7.4f}  {fl:>5.2f}  {gap:>+6.4f}  {sess}")
    else:
        print("  (none)")


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

    # RR proximity — WAIT uses rr_gap directly; BLOCKED derives it.
    # rr_unavailable=True → entry not reached, exec_rr was 0.0 default.
    # These rows MUST be excluded from RR buckets — they have no meaningful gap.
    def get_rr_gap(r: dict):
        if r["event"] == "CANDIDATE_WAIT":
            if r.get("rr_unavailable", False):
                return None   # exclude: RR not computed
            return r.get("rr_gap")
        thr  = r.get("min_rr_threshold")
        cand = r.get("candidate_rr")
        return (float(thr) - float(cand)) if thr is not None and cand is not None else None

    # Count rr_unavailable separately before building the bucket pool
    rr_unavailable_count = sum(
        1 for r in records
        if r["event"] == "CANDIDATE_WAIT" and r.get("rr_unavailable", False)
    )

    rr_recs = [r for r in records
               if (r["event"] == "CANDIDATE_WAIT"
                   and not r.get("rr_unavailable", False)
                   and r.get("rr_gap") is not None)
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
    if rr_unavailable_count:
        print(f"    rr_unavailable    {rr_unavailable_count}  "
              f"(entry not reached → exec_rr=0.0; excluded from buckets)")


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
            if r.get("rr_unavailable", False):
                return float("inf")   # not computed — sort to the back / exclude
            v = r.get("rr_gap")
        else:
            thr  = r.get("min_rr_threshold")
            cand = r.get("candidate_rr")
            v = (float(thr) - float(cand)) if thr is not None and cand is not None else None
        return v if v is not None else float("inf")

    rr_pool = [r for r in records
               if (r["event"] == "CANDIDATE_WAIT"
                   and not r.get("rr_unavailable", False)   # exclude uncalculated RR
                   and r.get("rr_gap") is not None)
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

    scan_records = load_scan_records(log_path, ts_from, ts_to)
    records      = load_records(log_path, ts_from, ts_to)
    pre_records  = load_pre_candidate_records(log_path, ts_from, ts_to)

    section_s(scan_records)
    print()
    section_p(pre_records)
    print()
    section_d(records)
    section_e(records)
    section_f(records, top_n=args.top)


if __name__ == "__main__":
    main()
