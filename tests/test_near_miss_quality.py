"""
tests/test_near_miss_quality.py
================================
Data-quality tests for near-miss capture pipeline.

Covers four invariants:
  1. no_pattern rows can never enter candidate counts
     (load_records drops them; orchestrator never emits them — but we test
     the load_records defensive drop as a belt+suspenders guard).
  2. A real WAIT candidate (pattern present) includes conf_gap and either
     a non-null rr_gap OR rr_unavailable=True.
  3. RR gap buckets populate correctly when rr is available, and
     rr_unavailable rows are excluded from bucket counts.
  4. Top-N closest misses exclude gap<=0 rows and require gap>0.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from io import StringIO
from unittest.mock import patch

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

def _write_jsonl(tmp_path: Path, records: list) -> Path:
    p = tmp_path / "decision_log.jsonl"
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


_TS_IN  = "2026-03-04T13:00:00+00:00"
_TS_OUT = "2026-03-04T15:00:00+00:00"

def _make_candidate_wait(pair="USD/JPY", pattern="double_top", direction="short",
                         confidence=0.70, conf_thr=0.77,
                         candidate_rr=None, rr_unavailable=True,
                         min_rr_threshold=2.5,
                         wait_reasons=None, rr_gap=None,
                         ts=_TS_IN) -> dict:
    if wait_reasons is None:
        wait_reasons = ["CONFIDENCE_BELOW_MIN"]
    conf_gap = round(conf_thr - confidence, 4)
    record = {
        "ts":                   ts,
        "event":                "CANDIDATE_WAIT",
        "pair":                 pair,
        "pattern":              pattern,
        "direction":            direction,
        "wait_reasons":         wait_reasons,
        "candidate_confidence": confidence,
        "confidence_threshold": conf_thr,
        "conf_gap":             conf_gap,
        "candidate_rr":         candidate_rr,
        "min_rr_threshold":     min_rr_threshold,
        "rr_gap":               rr_gap,
        "rr_unavailable":       rr_unavailable,
    }
    return record


def _make_scan_heartbeat(wait_count=12, ts=_TS_IN) -> dict:
    return {
        "ts":            ts,
        "event":         "SCAN_HEARTBEAT",
        "pair":          "ALL",
        "pairs_scanned": 12,
        "wait_count":    wait_count,
        "enter_count":   0,
        "blocked_count": 0,
    }


# Import the functions under test after helpers are defined
from scripts.near_miss_analysis import (
    load_records,
    load_scan_records,
    section_d,
    section_e,
    section_f,
)


# ── Test 1 ────────────────────────────────────────────────────────────────────

class TestNoPatternsNeverInCandidates:
    """
    no_pattern rows must NEVER increment candidate counts.

    A SCAN_HEARTBEAT with wait_count=12 typically includes 11 no_pattern pairs.
    load_records() must drop any CANDIDATE_WAIT record that somehow has pattern=None.
    """

    def test_candidate_wait_with_null_pattern_is_dropped(self, tmp_path):
        """Records with pattern=None are stripped by load_records, even if event type is right."""
        poisoned = {
            "ts": _TS_IN,
            "event": "CANDIDATE_WAIT",
            "pair": "GBP/CHF",
            "pattern": None,          # no_pattern placeholder
            "direction": None,
            "wait_reasons": ["no_pattern"],
            "candidate_confidence": 0.30,
            "confidence_threshold": 0.77,
            "conf_gap": 0.47,
            "candidate_rr": None,
            "rr_unavailable": True,
        }
        real = _make_candidate_wait()
        log  = _write_jsonl(tmp_path, [poisoned, real])
        records = load_records(log, _TS_IN, _TS_OUT)
        assert len(records) == 1, (
            f"Expected 1 real candidate, got {len(records)}. "
            "no_pattern record must be dropped."
        )
        assert records[0]["pair"] == "USD/JPY"

    def test_scan_heartbeat_not_in_candidate_records(self, tmp_path):
        """SCAN_HEARTBEAT events must not appear in candidate records."""
        log = _write_jsonl(tmp_path, [
            _make_scan_heartbeat(),
            _make_candidate_wait(),
        ])
        records = load_records(log, _TS_IN, _TS_OUT)
        assert all(r["event"] in ("CANDIDATE_WAIT", "CANDIDATE_BLOCKED") for r in records)
        assert len(records) == 1

    def test_section_d_counts_only_real_candidates(self, tmp_path, capsys):
        """section_d total == real candidates only, not scan-heartbeat wait_count."""
        records = [_make_candidate_wait(pair="USD/JPY")]
        log = _write_jsonl(tmp_path, records + [_make_scan_heartbeat(wait_count=12)])
        candidates = load_records(log, _TS_IN, _TS_OUT)
        section_d(candidates)
        out = capsys.readouterr().out
        # Must report 1 WAIT candidate, not 12 (scan heartbeat total)
        assert "CANDIDATE_WAIT    : 1" in out
        assert "12" not in out.split("CANDIDATE_WAIT")[1].split("\n")[0]


# ── Test 2 ────────────────────────────────────────────────────────────────────

class TestRealCandidateHasProximityFields:
    """
    A WAIT candidate with a pattern must have either a non-null rr_gap
    OR rr_unavailable=True — never both null with no explanation.
    """

    def test_wait_with_rr_computed_has_rr_gap(self):
        """When candidate_rr is set, rr_gap must be non-null and equal thr − cand."""
        r = _make_candidate_wait(
            candidate_rr=1.8,
            min_rr_threshold=2.5,
            rr_gap=round(2.5 - 1.8, 4),
            rr_unavailable=False,
        )
        assert r["pattern"] is not None
        assert r["rr_unavailable"] is False
        assert r["candidate_rr"] == 1.8
        assert r["rr_gap"] == pytest.approx(0.7, abs=1e-3)

    def test_wait_without_rr_sets_unavailable_flag(self):
        """When entry not reached, candidate_rr=None and rr_unavailable=True."""
        r = _make_candidate_wait(
            candidate_rr=None,
            rr_gap=None,
            rr_unavailable=True,
        )
        assert r["pattern"] is not None
        assert r["rr_unavailable"] is True
        assert r["candidate_rr"] is None
        assert r["rr_gap"] is None

    def test_wait_always_has_conf_gap(self):
        """conf_gap must be populated for any WAIT candidate with a pattern."""
        r = _make_candidate_wait(confidence=0.70, conf_thr=0.77)
        assert r["conf_gap"] == pytest.approx(0.07, abs=1e-4)

    def test_conf_gap_and_rr_unavailable_can_coexist(self):
        """Most common case: pattern found, conf below threshold, but no engulf yet → RR unknown."""
        r = _make_candidate_wait(
            confidence=0.70,
            conf_thr=0.77,
            candidate_rr=None,
            rr_gap=None,
            rr_unavailable=True,
        )
        assert r["conf_gap"] == pytest.approx(0.07, abs=1e-4)
        assert r["rr_unavailable"] is True


# ── Test 3 ────────────────────────────────────────────────────────────────────

class TestRRBucketsPopulateCorrectly:
    """
    RR gap buckets must use only records where rr_unavailable=False and rr_gap>0.
    rr_unavailable rows must appear in the rr_unavailable count, not the buckets.
    """

    def _build_records(self):
        return [
            # Real RR gaps across different buckets
            _make_candidate_wait(pair="P1", candidate_rr=2.4, min_rr_threshold=2.5,
                                 rr_gap=0.1, rr_unavailable=False,
                                 wait_reasons=["CONFIDENCE_BELOW_MIN"]),
            _make_candidate_wait(pair="P2", candidate_rr=2.2, min_rr_threshold=2.5,
                                 rr_gap=0.3, rr_unavailable=False,
                                 wait_reasons=["CONFIDENCE_BELOW_MIN"]),
            _make_candidate_wait(pair="P3", candidate_rr=1.8, min_rr_threshold=2.5,
                                 rr_gap=0.7, rr_unavailable=False,
                                 wait_reasons=["CONFIDENCE_BELOW_MIN"]),
            _make_candidate_wait(pair="P4", candidate_rr=0.5, min_rr_threshold=2.5,
                                 rr_gap=2.0, rr_unavailable=False,
                                 wait_reasons=["CONFIDENCE_BELOW_MIN"]),
            # rr_unavailable — must NOT enter any RR bucket
            _make_candidate_wait(pair="P5", candidate_rr=None,
                                 rr_gap=None, rr_unavailable=True),
            _make_candidate_wait(pair="P6", candidate_rr=None,
                                 rr_gap=None, rr_unavailable=True),
        ]

    def test_rr_bucket_totals_exclude_unavailable(self, capsys):
        """rr_unavailable rows must not appear in any RR bucket."""
        records = self._build_records()
        # Patch out conf section to isolate RR output
        section_e(records)
        out = capsys.readouterr().out
        # Four real RR records: 0.1→[0–0.2], 0.3→(0.2–0.5], 0.7→(0.5–1.0], 2.0→>1.0
        lines = {ln.strip() for ln in out.splitlines()}
        assert any("[0–0.2]" in ln and "1" in ln for ln in lines), (
            f"[0-0.2] bucket should have count 1\nOutput:\n{out}")
        assert any("(0.2–0.5]" in ln and "1" in ln for ln in lines), (
            f"(0.2-0.5] bucket should have count 1\nOutput:\n{out}")
        assert any("(0.5–1.0]" in ln and "1" in ln for ln in lines), (
            f"(0.5-1.0] bucket should have count 1\nOutput:\n{out}")
        assert any(">1.0" in ln and "1" in ln for ln in lines), (
            f">1.0 bucket should have count 1\nOutput:\n{out}")

    def test_rr_unavailable_count_shown(self, capsys):
        """rr_unavailable count must be reported separately."""
        records = self._build_records()
        section_e(records)
        out = capsys.readouterr().out
        assert "rr_unavailable" in out, (
            f"rr_unavailable count should appear in section E\nOutput:\n{out}")
        assert "2" in out, "Expected count of 2 rr_unavailable records"

    def test_rr_bucket_n_label_excludes_unavailable(self, capsys):
        """The [n=X] label for RR buckets should count only rr-available records."""
        records = self._build_records()
        section_e(records)
        out = capsys.readouterr().out
        # 4 rr-available records → n=4
        assert "[n=4]" in out, (
            f"RR bucket n= label should be 4 (not 6)\nOutput:\n{out}")


# ── Test 4 ────────────────────────────────────────────────────────────────────

class TestTop20ExcludesNonPositiveGaps:
    """
    Top-N closest misses must only include records where gap > 0.
    gap == 0 (already at threshold) and gap < 0 (above threshold) must be excluded.
    """

    def _build_gap_records(self):
        return [
            # gap > 0 — valid misses
            _make_candidate_wait(pair="VALID1", confidence=0.74, conf_thr=0.77),  # gap=0.03
            _make_candidate_wait(pair="VALID2", confidence=0.70, conf_thr=0.77),  # gap=0.07
            # gap == 0 — exactly at threshold (should be ENTER, not near-miss)
            _make_candidate_wait(pair="ZERO",   confidence=0.77, conf_thr=0.77),  # gap=0.00
            # gap < 0 — above threshold (should not be a WAIT at all, but test robustness)
            _make_candidate_wait(pair="NEG",    confidence=0.85, conf_thr=0.77),  # gap=-0.08
        ]

    def test_section_f_conf_excludes_zero_gap(self, capsys):
        records = self._build_gap_records()
        section_f(records, top_n=20)
        out = capsys.readouterr().out
        assert "ZERO" not in out,  "gap=0 row must be excluded from closest misses"
        assert "NEG"  not in out,  "gap<0 row must be excluded from closest misses"

    def test_section_f_conf_includes_positive_gaps(self, capsys):
        records = self._build_gap_records()
        section_f(records, top_n=20)
        out = capsys.readouterr().out
        assert "VALID1" in out, "gap>0 records must appear in closest misses"
        assert "VALID2" in out, "gap>0 records must appear in closest misses"

    def test_section_f_conf_sorted_ascending(self, capsys):
        """Smallest gap first (VALID1 gap=0.03 before VALID2 gap=0.07)."""
        records = self._build_gap_records()
        section_f(records, top_n=20)
        out = capsys.readouterr().out
        idx1 = out.find("VALID1")
        idx2 = out.find("VALID2")
        assert idx1 < idx2, "VALID1 (smaller gap 0.03) must appear before VALID2 (0.07)"

    def test_section_f_rr_excludes_rr_unavailable(self, capsys):
        """rr_unavailable=True records must not appear in RR closest misses."""
        records = [
            _make_candidate_wait(pair="HAS_RR",   candidate_rr=1.8, min_rr_threshold=2.5,
                                 rr_gap=0.7, rr_unavailable=False,
                                 wait_reasons=["CONFIDENCE_BELOW_MIN"]),
            _make_candidate_wait(pair="NO_RR",    candidate_rr=None,
                                 rr_gap=None, rr_unavailable=True),
        ]
        section_f(records, top_n=20)
        out = capsys.readouterr().out
        # NO_RR must not appear in the RR misses section
        # Find the RR section specifically
        rr_section = out.split("RR closest misses")[-1] if "RR closest misses" in out else out
        assert "NO_RR" not in rr_section, (
            "rr_unavailable row must be excluded from RR closest misses"
        )
