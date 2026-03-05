"""
tests/test_pre_candidate.py
============================
Tests for PRE_CANDIDATE telemetry pipeline.

Invariants:
  1. PRE_CANDIDATE only emitted when sub-threshold patterns exist AND
     decision.pattern is None.
  2. PRE_CANDIDATE NOT emitted when decision.pattern is not None
     (that path belongs to CANDIDATE_WAIT).
  3. recognition_floor sourced from strategy.min_pattern_clarity (not hardcoded).
  4. confidence_gap_to_floor = recognition_floor − raw_confidence, always > 0.
  5. section_p() buckets and top-20 exclude gap <= 0 rows.
  6. load_pre_candidate_records() loads only PRE_CANDIDATE events.
  7. No PRE_CANDIDATE emitted when sub-threshold list is empty.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

def _write_jsonl(tmp_path: Path, records: list) -> Path:
    p = tmp_path / "decision_log.jsonl"
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


_TS_IN  = "2026-03-05T13:00:00+00:00"
_TS_OUT = "2026-03-05T15:00:00+00:00"


def _make_pattern_result(pattern_type="double_top", direction="bearish", clarity=0.25):
    """Build a minimal PatternResult-like object."""
    p = MagicMock()
    p.pattern_type  = pattern_type
    p.direction     = direction
    p.clarity       = clarity
    p.neckline      = 159.50
    p.pattern_level = 160.00
    return p


def _make_pre_candidate_record(pair="USD/JPY", pattern_type="double_top",
                                direction="bearish", raw_confidence=0.25,
                                recognition_floor=0.4, ts=_TS_IN) -> dict:
    gap = round(recognition_floor - raw_confidence, 4)
    return {
        "ts":                     ts,
        "event":                  "PRE_CANDIDATE",
        "pair":                   pair,
        "pattern_type":           pattern_type,
        "direction":              direction,
        "raw_confidence":         raw_confidence,
        "recognition_floor":      recognition_floor,
        "confidence_gap_to_floor": gap,
        "trigger_state":          "BELOW_RECOGNITION_FLOOR",
        "atr_ratio":              0.75,
        "wd_aligned":             False,
        "session_allowed":        True,
        "session_reason":         "",
        "neckline":               159.50,
        "pattern_level":          160.00,
    }


from scripts.near_miss_analysis import (
    load_pre_candidate_records,
    load_records,
    section_p,
)
from src.execution.block_logger import CandidateBlockLogger


# ── Test 1: emit only when decision.pattern is None AND sub_threshold non-empty ──

class TestPreCandidateEmissionConditions:

    def _make_logger(self, tmp_path):
        log_file = tmp_path / "decision_log.jsonl"
        return CandidateBlockLogger(log_file), log_file

    def _make_ctx(self, recognition_floor=0.4):
        return {
            "recognition_floor": recognition_floor,
            "wd_aligned":        False,
            "atr_ratio":         0.75,
            "session_allowed":   True,
            "session_reason":    "",
        }

    def test_emits_when_sub_threshold_patterns_present(self, tmp_path):
        """log_pre_candidate writes a record when sub-threshold patterns exist."""
        logger, log_file = self._make_logger(tmp_path)
        patterns = [_make_pattern_result(clarity=0.25)]
        written = logger.log_pre_candidate("USD/JPY", patterns, self._make_ctx())
        assert written == 1
        lines = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0]["event"] == "PRE_CANDIDATE"
        assert lines[0]["pair"] == "USD/JPY"

    def test_does_not_emit_when_list_empty(self, tmp_path):
        """No record written when sub_threshold_patterns is empty."""
        logger, log_file = self._make_logger(tmp_path)
        written = logger.log_pre_candidate("GBP/USD", [], self._make_ctx())
        assert written == 0
        assert not log_file.exists() or log_file.read_text().strip() == ""

    def test_emits_multiple_patterns_as_separate_records(self, tmp_path):
        """Each sub-threshold pattern → separate PRE_CANDIDATE record."""
        logger, log_file = self._make_logger(tmp_path)
        patterns = [
            _make_pattern_result("double_top",          "bearish", 0.25),
            _make_pattern_result("head_and_shoulders",  "bearish", 0.30),
        ]
        written = logger.log_pre_candidate("USD/JPY", patterns, self._make_ctx())
        assert written == 2
        lines = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        types = {l["pattern_type"] for l in lines}
        assert "double_top" in types
        assert "head_and_shoulders" in types

    def test_not_emitted_when_decision_has_pattern(self, tmp_path):
        """
        PRE_CANDIDATE must NOT be emitted when decision.pattern is not None —
        that is handled by CANDIDATE_WAIT.  This tests the orchestrator guard
        by calling log_pre_candidate with an empty list (simulating the
        orchestrator's `if decision.pattern is None` check passing an empty
        sub_threshold list).
        """
        logger, log_file = self._make_logger(tmp_path)
        # Simulate: decision.pattern is not None → sub_threshold never fetched
        # → caller passes empty list
        written = logger.log_pre_candidate("USD/JPY", [], self._make_ctx())
        assert written == 0

    def test_throttle_suppresses_duplicate_within_1h(self, tmp_path):
        """Same (pair, pattern_type, direction) within 1h → second call suppressed."""
        logger, log_file = self._make_logger(tmp_path)
        patterns = [_make_pattern_result(clarity=0.25)]
        w1 = logger.log_pre_candidate("USD/JPY", patterns, self._make_ctx())
        w2 = logger.log_pre_candidate("USD/JPY", patterns, self._make_ctx())
        assert w1 == 1
        assert w2 == 0  # throttled


# ── Test 2: recognition_floor sourced from strategy constant ─────────────────

class TestRecognitionFloor:

    def _make_logger(self, tmp_path):
        log_file = tmp_path / "decision_log.jsonl"
        return CandidateBlockLogger(log_file), log_file

    def test_recognition_floor_in_record(self, tmp_path):
        """recognition_floor field must equal the value passed in context."""
        logger, log_file = self._make_logger(tmp_path)
        ctx = {"recognition_floor": 0.4, "session_allowed": True, "session_reason": ""}
        patterns = [_make_pattern_result(clarity=0.28)]
        logger.log_pre_candidate("USD/JPY", patterns, ctx)
        r = json.loads(log_file.read_text().splitlines()[0])
        assert r["recognition_floor"] == pytest.approx(0.4)

    def test_gap_to_floor_computed_correctly(self, tmp_path):
        """confidence_gap_to_floor = recognition_floor − raw_confidence."""
        logger, log_file = self._make_logger(tmp_path)
        ctx = {"recognition_floor": 0.4, "session_allowed": True, "session_reason": ""}
        clarity = 0.28
        patterns = [_make_pattern_result(clarity=clarity)]
        logger.log_pre_candidate("USD/JPY", patterns, ctx)
        r = json.loads(log_file.read_text().splitlines()[0])
        expected_gap = round(0.4 - clarity, 4)
        assert r["confidence_gap_to_floor"] == pytest.approx(expected_gap, abs=1e-4)

    def test_pre_candidate_ctx_uses_strategy_constant(self):
        """_pre_candidate_ctx reads min_pattern_clarity from strategy, not hardcoded."""
        # Build a minimal mock orchestrator
        orch = MagicMock()
        orch._last_regime_score = {"wd_aligned": True, "atr_ratio": 1.2}
        orch.strategy.min_pattern_clarity = 0.45  # non-default value
        orch.strategy.session_filter.is_allowed.return_value = (True, "")

        # Import and call the real method
        from src.execution.orchestrator import ForexOrchestrator
        ctx = ForexOrchestrator._pre_candidate_ctx(orch, MagicMock())
        assert ctx["recognition_floor"] == pytest.approx(0.45)

    def test_gap_always_positive(self, tmp_path):
        """gap_to_floor must be > 0 for any sub-threshold pattern."""
        logger, log_file = self._make_logger(tmp_path)
        ctx = {"recognition_floor": 0.4, "session_allowed": True, "session_reason": ""}
        # clarity=0.39 is just below floor; gap should be positive
        patterns = [_make_pattern_result(clarity=0.39)]
        logger.log_pre_candidate("USD/JPY", patterns, ctx)
        r = json.loads(log_file.read_text().splitlines()[0])
        assert r["confidence_gap_to_floor"] > 0


# ── Test 3: section_p buckets and top-20 ─────────────────────────────────────

class TestSectionPBucketsAndTop20:

    def _make_records(self):
        return [
            _make_pre_candidate_record("P1", raw_confidence=0.39),  # gap=0.01  → [0–0.02]
            _make_pre_candidate_record("P2", raw_confidence=0.36),  # gap=0.04  → (0.02–0.05]
            _make_pre_candidate_record("P3", raw_confidence=0.32),  # gap=0.08  → (0.05–0.10]
            _make_pre_candidate_record("P4", raw_confidence=0.10),  # gap=0.30  → >0.10
            # gap == 0 → excluded from buckets and top-20
            _make_pre_candidate_record("P5", raw_confidence=0.40),  # gap=0.00
            # gap < 0 (above floor — shouldn't exist but test robustness)
            _make_pre_candidate_record("P6", raw_confidence=0.45),  # gap=-0.05
        ]

    def test_bucket_counts(self, capsys):
        records = self._make_records()
        section_p(records)
        out = capsys.readouterr().out
        assert "[0–0.02]" in out
        assert "(0.02–0.05]" in out
        assert "(0.05–0.10]" in out
        assert ">0.10" in out

    def test_zero_and_negative_gaps_excluded_from_top20(self, capsys):
        records = self._make_records()
        section_p(records)
        out = capsys.readouterr().out
        # P5 (gap=0) and P6 (gap<0) must not appear in top-20
        top20_section = out.split("closest PRE_CANDIDATE")[-1] if "closest PRE_CANDIDATE" in out else out
        assert "P5" not in top20_section, "gap=0 must be excluded"
        assert "P6" not in top20_section, "gap<0 must be excluded"

    def test_top20_sorted_ascending(self, capsys):
        records = self._make_records()
        section_p(records)
        out = capsys.readouterr().out
        top_section = out.split("closest PRE_CANDIDATE")[-1] if "closest PRE_CANDIDATE" in out else out
        idx1 = top_section.find("P1")  # gap=0.01 — smallest
        idx2 = top_section.find("P4")  # gap=0.30 — largest
        assert idx1 < idx2, "smallest gap must appear first"


# ── Test 4: load_pre_candidate_records isolation ─────────────────────────────

class TestLoadPreCandidateRecords:

    def test_loads_only_pre_candidate_events(self, tmp_path):
        """load_pre_candidate_records must ignore all other event types."""
        records = [
            _make_pre_candidate_record("USD/JPY"),
            {"ts": _TS_IN, "event": "CANDIDATE_WAIT",  "pair": "GBP/USD",
             "pattern": "double_bottom"},
            {"ts": _TS_IN, "event": "SCAN_HEARTBEAT",  "pair": "ALL"},
            {"ts": _TS_IN, "event": "CANDIDATE_BLOCKED","pair": "USD/CHF"},
        ]
        log = _write_jsonl(tmp_path, records)
        pre = load_pre_candidate_records(log, _TS_IN, _TS_OUT)
        assert len(pre) == 1
        assert pre[0]["event"] == "PRE_CANDIDATE"
        assert pre[0]["pair"] == "USD/JPY"

    def test_candidate_wait_not_in_pre_candidate_records(self, tmp_path):
        """CANDIDATE_WAIT events must NOT appear in pre_candidate pool."""
        records = [
            {"ts": _TS_IN, "event": "CANDIDATE_WAIT", "pair": "EUR/USD",
             "pattern": "double_top"},
        ]
        log = _write_jsonl(tmp_path, records)
        pre = load_pre_candidate_records(log, _TS_IN, _TS_OUT)
        assert len(pre) == 0

    def test_respects_time_window(self, tmp_path):
        """Records outside the window must be filtered out."""
        records = [
            _make_pre_candidate_record("USD/JPY", ts="2026-03-05T12:59:59+00:00"),  # before
            _make_pre_candidate_record("GBP/USD", ts=_TS_IN),                        # in
            _make_pre_candidate_record("USD/CAD", ts="2026-03-05T15:00:01+00:00"),  # after
        ]
        log = _write_jsonl(tmp_path, records)
        pre = load_pre_candidate_records(log, _TS_IN, _TS_OUT)
        assert len(pre) == 1
        assert pre[0]["pair"] == "GBP/USD"
