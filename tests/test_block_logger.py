"""
tests/test_block_logger.py
==========================
Tests for CandidateBlockLogger (src/execution/block_logger.py).

Invariant classes:
  1. Record structure — all required fields present and typed correctly
  2. Block reason enum normalisation (aliases + canonical values)
  3. Unknown reason handled without raising
  4. Throttle — suppress same (pair, pattern) within 1 hour
  5. Throttle reset on block_reason change
  6. Throttle reset after 1 hour has passed
  7. CONFIDENCE_BLOCK record values vs thresholds
  8. RR_BLOCK record values vs thresholds
  9. SESSION_BLOCK record fields
  10. PAUSE_BLOCK effective_paused derivation
  11. CHOP_SHIELD_BLOCK / RECOVERY_RULES_BLOCK context fields
  12. fail-open — write failure does not raise
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import pytest

from src.execution.block_logger import (
    CandidateBlockLogger,
    BLOCK_REASONS,
    _REASON_ALIASES,
    _THROTTLE_SECS,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _logger(tmp_path):
    """Return a CandidateBlockLogger writing to a temp file."""
    f = tmp_path / "decision_log.jsonl"
    return CandidateBlockLogger(decisions_file=f), f


def _read_records(path: Path) -> list:
    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _ctx(
    pattern="double_top",
    direction="short",
    candidate_confidence=0.60,
    confidence_threshold=0.65,
    candidate_rr=1.8,
    min_rr_threshold=1.5,
    session_allowed=True,
    session_reason="",
    bot_mode="active",
    pause_new_entries=False,
    pause_expiry_ts=None,
    loss_streak=0,
    paused_by_chop=False,
    recovery_rules_active=False,
    htf_aligned=True,
    **kwargs,
):
    d = dict(
        pattern=pattern,
        direction=direction,
        candidate_confidence=candidate_confidence,
        confidence_threshold=confidence_threshold,
        candidate_rr=candidate_rr,
        min_rr_threshold=min_rr_threshold,
        session_allowed=session_allowed,
        session_reason=session_reason,
        bot_mode=bot_mode,
        pause_new_entries=pause_new_entries,
        pause_expiry_ts=pause_expiry_ts,
        loss_streak=loss_streak,
        paused_by_chop=paused_by_chop,
        recovery_rules_active=recovery_rules_active,
        htf_aligned=htf_aligned,
    )
    d.update(kwargs)
    return d


# ── Class 1: Record structure ──────────────────────────────────────────────

class TestRecordStructure:

    REQUIRED_FIELDS = [
        "ts", "event", "pair", "pattern", "direction",
        "block_reasons",
        "candidate_confidence", "confidence_threshold",
        "candidate_rr", "min_rr_threshold",
        "session_allowed", "session_reason",
        "weekly_cap_remaining", "weekly_cap_limit",
        "bot_mode", "pause_new_entries", "pause_expiry_ts", "effective_paused",
        "loss_streak", "paused_by_chop", "recovery_rules_active",
        "htf_aligned",
    ]

    def test_all_required_fields_present(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("GBP/JPY", "CONFIDENCE_BLOCK", _ctx())
        records = _read_records(f)
        assert len(records) == 1
        r = records[0]
        for field in self.REQUIRED_FIELDS:
            assert field in r, f"Missing field: {field}"

    def test_event_is_candidate_blocked(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("USD/JPY", "RR_BLOCK", _ctx())
        r = _read_records(f)[0]
        assert r["event"] == "CANDIDATE_BLOCKED"

    def test_block_reasons_is_list(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("USD/CHF", "CONFIDENCE_BLOCK", _ctx())
        r = _read_records(f)[0]
        assert isinstance(r["block_reasons"], list)
        assert len(r["block_reasons"]) == 1

    def test_ts_is_valid_iso(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("EUR/USD", "PAUSE_BLOCK", _ctx())
        r = _read_records(f)[0]
        # Should parse without error
        datetime.fromisoformat(r["ts"])

    def test_pair_is_preserved(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("GBP/CHF", "SESSION_BLOCK", _ctx())
        r = _read_records(f)[0]
        assert r["pair"] == "GBP/CHF"


# ── Class 2: Block reason normalisation ───────────────────────────────────

class TestReasonNormalisation:

    @pytest.mark.parametrize("alias,expected", [
        ("TIME_BLOCK",          "SESSION_BLOCK"),
        ("RR_MIN",              "RR_BLOCK"),
        ("RECOVERY_MIN_RR",     "RECOVERY_RULES_BLOCK"),
        ("RECOVERY_CONF",       "RECOVERY_RULES_BLOCK"),
        ("RECOVERY_WEEKLY_CAP", "RECOVERY_RULES_BLOCK"),
        ("WEEKLY_TRADE_LIMIT",  "WEEKLY_CAP_BLOCK"),
        ("CONF",                "CONFIDENCE_BLOCK"),
        ("CONFIDENCE",          "CONFIDENCE_BLOCK"),
        ("WINNER_RUNNING",      "WINNER_RUNNING_BLOCK"),
        ("THEME_CONFLICT",      "THEME_CONFLICT_BLOCK"),
        ("STOP_WIDE",           "STOP_WIDE_BLOCK"),
        ("STOP_TIGHT",          "STOP_TIGHT_BLOCK"),
        ("PAUSED",              "PAUSE_BLOCK"),
        ("RECONCILE",           "RECONCILE_PAUSE_BLOCK"),
    ])
    def test_alias_resolved(self, tmp_path, alias, expected):
        bl, f = _logger(tmp_path)
        bl.log_block("USD/JPY", alias, _ctx())
        r = _read_records(f)[0]
        assert r["block_reasons"] == [expected]

    @pytest.mark.parametrize("canonical", sorted(BLOCK_REASONS))
    def test_canonical_reason_preserved(self, tmp_path, canonical):
        bl, f = _logger(tmp_path)
        bl.log_block("GBP/JPY", canonical, _ctx())
        r = _read_records(f)[0]
        assert r["block_reasons"] == [canonical]


# ── Class 3: Unknown reason handled ───────────────────────────────────────

class TestUnknownReason:

    def test_unknown_reason_does_not_raise(self, tmp_path):
        bl, f = _logger(tmp_path)
        result = bl.log_block("EUR/USD", "TOTALLY_UNKNOWN_GATE", _ctx())
        # Should succeed (or fail open) — must not raise
        assert isinstance(result, bool)

    def test_unknown_reason_written_as_is(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("EUR/USD", "CUSTOM_GATE_XYZ", _ctx())
        records = _read_records(f)
        if records:  # written (not suppressed)
            assert records[0]["block_reasons"] == ["CUSTOM_GATE_XYZ"]


# ── Class 4: Throttle — suppress same (pair, pattern, direction, reasons) ─

class TestThrottleSuppression:

    def test_duplicate_within_hour_suppressed(self, tmp_path):
        """Identical key (pair+pattern+direction+reasons) within 1h → suppressed."""
        bl, f = _logger(tmp_path)
        r1 = bl.log_block("GBP/USD", "CONFIDENCE_BLOCK", _ctx())
        r2 = bl.log_block("GBP/USD", "CONFIDENCE_BLOCK", _ctx())
        assert r1 is True
        assert r2 is False
        assert len(_read_records(f)) == 1

    def test_different_pair_not_suppressed(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("GBP/USD", "CONFIDENCE_BLOCK", _ctx())
        r2 = bl.log_block("EUR/USD", "CONFIDENCE_BLOCK", _ctx())
        assert r2 is True
        assert len(_read_records(f)) == 2

    def test_different_pattern_not_suppressed(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("GBP/USD", "CONFIDENCE_BLOCK", _ctx(pattern="double_top"))
        r2 = bl.log_block("GBP/USD", "CONFIDENCE_BLOCK", _ctx(pattern="head_and_shoulders"))
        assert r2 is True
        assert len(_read_records(f)) == 2

    def test_different_direction_not_suppressed(self, tmp_path):
        """Direction is part of the throttle key — long vs short → both logged."""
        bl, f = _logger(tmp_path)
        bl.log_block("GBP/USD", "CONFIDENCE_BLOCK", _ctx(direction="short"))
        r2 = bl.log_block("GBP/USD", "CONFIDENCE_BLOCK", _ctx(direction="long"))
        assert r2 is True
        assert len(_read_records(f)) == 2


# ── Class 5: Throttle — different block_reason = different key ────────────

class TestThrottleResetOnReasonChange:

    def test_new_reason_is_different_key(self, tmp_path):
        """block_reasons is part of the throttle key — different reason → different key → logged."""
        bl, f = _logger(tmp_path)
        bl.log_block("USD/JPY", "CONFIDENCE_BLOCK", _ctx())
        r2 = bl.log_block("USD/JPY", "RR_BLOCK", _ctx())
        assert r2 is True
        assert len(_read_records(f)) == 2

    def test_same_reason_reverted_suppressed_within_hour(self, tmp_path):
        """
        CONFIDENCE → RR_BLOCK → CONFIDENCE within 1h:
        Third call has same key as first → suppressed (key was cached at attempt 1).
        """
        bl, f = _logger(tmp_path)
        bl.log_block("USD/JPY", "CONFIDENCE_BLOCK", _ctx())   # key A → logged
        bl.log_block("USD/JPY", "RR_BLOCK", _ctx())            # key B → logged
        r3 = bl.log_block("USD/JPY", "CONFIDENCE_BLOCK", _ctx())  # key A again, <1h → suppressed
        assert r3 is False
        assert len(_read_records(f)) == 2


# ── Class 6: Throttle expiry after 1 hour ────────────────────────────────

class TestThrottleExpiry:

    def test_same_key_written_after_throttle_expires(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("GBP/CHF", "CONFIDENCE_BLOCK", _ctx())
        # Throttle key now maps to a datetime directly
        key = ("GBP/CHF", "double_top", "short", frozenset({"CONFIDENCE_BLOCK"}))
        bl._throttle[key] = (
            datetime.now(timezone.utc) - timedelta(seconds=_THROTTLE_SECS + 1)
        )
        r2 = bl.log_block("GBP/CHF", "CONFIDENCE_BLOCK", _ctx())
        assert r2 is True
        assert len(_read_records(f)) == 2


# ── Class 7: CONFIDENCE_BLOCK values ──────────────────────────────────────

class TestConfidenceBlockValues:

    def test_candidate_confidence_and_threshold_written(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("USD/CHF", "CONFIDENCE_BLOCK", _ctx(
            candidate_confidence=0.58,
            confidence_threshold=0.65,
        ))
        r = _read_records(f)[0]
        assert r["candidate_confidence"] == pytest.approx(0.58)
        assert r["confidence_threshold"] == pytest.approx(0.65)
        assert r["block_reasons"] == ["CONFIDENCE_BLOCK"]

    def test_candidate_confidence_below_threshold(self, tmp_path):
        bl, f = _logger(tmp_path)
        ctx = _ctx(candidate_confidence=0.50, confidence_threshold=0.65)
        bl.log_block("GBP/USD", "CONFIDENCE_BLOCK", ctx)
        r = _read_records(f)[0]
        assert r["candidate_confidence"] < r["confidence_threshold"]


# ── Class 8: RR_BLOCK values + auto-threshold derivation ──────────────────

class TestRRBlockValues:

    def test_explicit_min_rr_threshold_wins(self, tmp_path):
        """Explicit ctx override of min_rr_threshold is written as-is."""
        bl, f = _logger(tmp_path)
        bl.log_block("USD/CAD", "RR_BLOCK", _ctx(
            candidate_rr=0.9,
            min_rr_threshold=2.5,
        ))
        r = _read_records(f)[0]
        assert r["candidate_rr"] == pytest.approx(0.9)
        assert r["min_rr_threshold"] == pytest.approx(2.5)
        assert r["block_reasons"] == ["RR_BLOCK"]

    def test_htf_aligned_written_with_rr_block(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("EUR/USD", "RR_BLOCK", _ctx(htf_aligned=False))
        r = _read_records(f)[0]
        assert r["htf_aligned"] is False

    def test_min_rr_auto_derived_from_strategy_config_htf_true(self, tmp_path):
        """No explicit threshold → derived from strategy_config.MIN_RR_STANDARD."""
        from src.strategy.forex import strategy_config as _sc
        expected = float(getattr(_sc, "MIN_RR_STANDARD", 2.5))
        bl, f = _logger(tmp_path)
        # no min_rr_threshold in ctx → auto-derive
        ctx = _ctx(htf_aligned=True)
        ctx.pop("min_rr_threshold", None)
        bl.log_block("GBP/CHF", "RR_BLOCK", ctx)
        r = _read_records(f)[0]
        assert r["min_rr_threshold"] == pytest.approx(expected)

    def test_min_rr_auto_derived_from_strategy_config_htf_false(self, tmp_path):
        """No explicit threshold + htf_aligned=False → MIN_RR_COUNTERTREND."""
        from src.strategy.forex import strategy_config as _sc
        expected = float(getattr(_sc, "MIN_RR_COUNTERTREND", 2.5))
        bl, f = _logger(tmp_path)
        ctx = _ctx(htf_aligned=False)
        ctx.pop("min_rr_threshold", None)
        bl.log_block("EUR/CHF", "RR_BLOCK", ctx)
        r = _read_records(f)[0]
        assert r["min_rr_threshold"] == pytest.approx(expected)

    def test_recovery_min_rr_explicit_override(self, tmp_path):
        """RECOVERY_RULES_BLOCK passes RECOVERY_MIN_RR (3.0) explicitly."""
        from src.strategy.forex import strategy_config as _sc
        rec_rr = float(getattr(_sc, "RECOVERY_MIN_RR", 3.0))
        bl, f = _logger(tmp_path)
        bl.log_block("USD/JPY", "RECOVERY_RULES_BLOCK", _ctx(
            candidate_rr=1.8,
            min_rr_threshold=rec_rr,
            recovery_rules_active=True,
        ))
        r = _read_records(f)[0]
        assert r["min_rr_threshold"] == pytest.approx(rec_rr)
        assert r["recovery_rules_active"] is True


# ── Class 9: SESSION_BLOCK fields ─────────────────────────────────────────

class TestSessionBlockFields:

    def test_session_allowed_false_in_session_block(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("GBP/JPY", "SESSION_BLOCK", _ctx(
            session_allowed=False,
            session_reason="OUTSIDE_LONDON_SESSION",
        ))
        r = _read_records(f)[0]
        assert r["session_allowed"] is False
        assert r["session_reason"] == "OUTSIDE_LONDON_SESSION"
        assert r["block_reasons"] == ["SESSION_BLOCK"]


# ── Class 10: PAUSE_BLOCK effective_paused ────────────────────────────────

class TestPauseBlockEffectivePaused:

    def test_effective_paused_true_when_pne_true(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("USD/JPY", "PAUSE_BLOCK", _ctx(
            pause_new_entries=True,
            bot_mode="active",
        ))
        r = _read_records(f)[0]
        assert r["effective_paused"] is True
        assert r["pause_new_entries"] is True

    def test_effective_paused_true_when_bot_mode_paused(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("USD/JPY", "PAUSE_BLOCK", _ctx(
            pause_new_entries=False,
            bot_mode="paused",
        ))
        r = _read_records(f)[0]
        assert r["effective_paused"] is True

    def test_effective_paused_true_when_chop_expiry_active(self, tmp_path):
        future_ts = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        bl, f = _logger(tmp_path)
        bl.log_block("GBP/CHF", "CHOP_SHIELD_BLOCK", _ctx(
            pause_new_entries=False,
            bot_mode="active",
            pause_expiry_ts=future_ts,
        ))
        r = _read_records(f)[0]
        assert r["effective_paused"] is True

    def test_effective_paused_false_when_all_clear(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("EUR/USD", "CONFIDENCE_BLOCK", _ctx(
            pause_new_entries=False,
            bot_mode="active",
            pause_expiry_ts=None,
        ))
        r = _read_records(f)[0]
        assert r["effective_paused"] is False


# ── Class 11: CHOP_SHIELD / RECOVERY_RULES context fields ─────────────────

class TestChopShieldContext:

    def test_loss_streak_written(self, tmp_path):
        bl, f = _logger(tmp_path)
        bl.log_block("GBP/USD", "RECOVERY_RULES_BLOCK", _ctx(
            loss_streak=3,
            recovery_rules_active=True,
            min_rr_threshold=3.0,
        ))
        r = _read_records(f)[0]
        assert r["loss_streak"] == 3
        assert r["recovery_rules_active"] is True

    def test_paused_by_chop_written(self, tmp_path):
        bl, f = _logger(tmp_path)
        future_ts = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        bl.log_block("USD/CHF", "CHOP_SHIELD_BLOCK", _ctx(
            paused_by_chop=True,
            pause_expiry_ts=future_ts,
            loss_streak=3,
        ))
        r = _read_records(f)[0]
        assert r["paused_by_chop"] is True
        assert r["pause_expiry_ts"] == future_ts


# ── Class 12: fail-open on write failure ──────────────────────────────────

class TestFailOpen:

    def test_write_failure_does_not_raise(self, tmp_path):
        """Logger must never raise even if the file can't be written."""
        # Point the log file at a path whose parent is an existing regular FILE
        # (not a directory), so mkdir + open both fail.
        blocker = tmp_path / "is_a_file"
        blocker.write_text("x")
        bad_path = blocker / "sub" / "decision_log.jsonl"
        bl = CandidateBlockLogger(decisions_file=bad_path)
        result = bl.log_block("USD/JPY", "CONFIDENCE_BLOCK", _ctx())
        # Should return False (write failed) but must not raise
        assert result is False

    def test_bad_context_values_do_not_raise(self, tmp_path):
        bl, f = _logger(tmp_path)
        bad_ctx = {"pause_expiry_ts": "NOT_A_DATE", "candidate_confidence": "banana"}
        result = bl.log_block("GBP/USD", "CONFIDENCE_BLOCK", bad_ctx)
        assert isinstance(result, bool)
