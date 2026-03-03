"""
CandidateBlockLogger
====================
Writes structured CANDIDATE_BLOCKED records to decision_log.jsonl when a
pre-gate ENTER signal exists but is rejected by an existing gate.

Contract:
  - Does NOT change any gate logic, threshold, or execution flow.
  - One record per gate-rejection per (pair, pattern_key) per hour,
    unless the block_reason set changes (any new reason → log immediately).
  - Fails silently — never raises, never blocks entry path.
  - Writes to decision_log.jsonl (decision feed), NOT paper_journal.jsonl.

Block reason enums (string):
  CONFIDENCE_BLOCK      candidate_confidence < confidence_threshold
  RR_BLOCK              candidate_rr < min_rr_threshold
  SESSION_BLOCK         outside allowed session window (TIME_BLOCK alias)
  WEEKLY_CAP_BLOCK      weekly trade count >= cap
  CHOP_SHIELD_BLOCK     entries paused by chop-shield AUTO_PAUSE
  RECOVERY_RULES_BLOCK  recovery-mode RR/conf/weekly-cap gate
  PAUSE_BLOCK           effective_paused=True (manual or chop expiry)
  STOP_WIDE_BLOCK       stop distance > ATR_STOP_MULTIPLIER × ATR
  STOP_TIGHT_BLOCK      stop distance < ATR_MIN_MULTIPLIER × ATR
  WINNER_RUNNING_BLOCK  open position already up >= WINNER_THRESHOLD_R
  THEME_CONFLICT_BLOCK  macro theme direction conflicts with signal
  RECONCILE_PAUSE_BLOCK pre-order reconciler triggered pause
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DECISIONS_FILE = Path.home() / "trading-bot" / "logs" / "decision_log.jsonl"

# All valid block-reason enum values
BLOCK_REASONS = frozenset({
    "CONFIDENCE_BLOCK",
    "RR_BLOCK",
    "SESSION_BLOCK",
    "WEEKLY_CAP_BLOCK",
    "CHOP_SHIELD_BLOCK",
    "RECOVERY_RULES_BLOCK",
    "PAUSE_BLOCK",
    "STOP_WIDE_BLOCK",
    "STOP_TIGHT_BLOCK",
    "WINNER_RUNNING_BLOCK",
    "THEME_CONFLICT_BLOCK",
    "RECONCILE_PAUSE_BLOCK",
})

# Alias map: internal gate names → canonical enum values
_REASON_ALIASES: Dict[str, str] = {
    "TIME_BLOCK":        "SESSION_BLOCK",
    "RR_MIN":            "RR_BLOCK",
    "RECOVERY_MIN_RR":   "RECOVERY_RULES_BLOCK",
    "RECOVERY_CONF":     "RECOVERY_RULES_BLOCK",
    "RECOVERY_WEEKLY_CAP": "RECOVERY_RULES_BLOCK",
    "WEEKLY_TRADE_LIMIT":"WEEKLY_CAP_BLOCK",
    "CONF":              "CONFIDENCE_BLOCK",
    "CONFIDENCE":        "CONFIDENCE_BLOCK",
    "WINNER_RUNNING":    "WINNER_RUNNING_BLOCK",
    "THEME_CONFLICT":    "THEME_CONFLICT_BLOCK",
    "STOP_WIDE":         "STOP_WIDE_BLOCK",
    "STOP_TIGHT":        "STOP_TIGHT_BLOCK",
    "PAUSED":            "PAUSE_BLOCK",
    "CHOP_PAUSE":        "CHOP_SHIELD_BLOCK",
    "RECONCILE":         "RECONCILE_PAUSE_BLOCK",
}

_THROTTLE_SECS = 3600  # suppress duplicate (pair, pattern_key, reasons) within 1 hour


class CandidateBlockLogger:
    """
    Log one CANDIDATE_BLOCKED record per gate-rejection, throttled per
    (pair, pattern_key) to at most once per hour unless block_reason changes.
    """

    def __init__(self, decisions_file: Optional[Path] = None):
        self._file = decisions_file or _DECISIONS_FILE
        # key: (pair, pattern_key) → {"ts": datetime, "reasons": frozenset}
        self._throttle: Dict[tuple, Dict] = {}

    # ------------------------------------------------------------------
    def log_block(
        self,
        pair: str,
        block_reason: str,
        context: Dict[str, Any],
    ) -> bool:
        """
        Emit one CANDIDATE_BLOCKED record.

        Parameters
        ----------
        pair          : e.g. "USD/JPY"
        block_reason  : one of BLOCK_REASONS (or an alias — will be normalised)
        context       : dict containing any available gate-context fields
                        (see _build_record for recognised keys)

        Returns True if the record was written, False if throttled.
        """
        try:
            reason_norm = self._normalise(block_reason)
            pattern_key = str(context.get("pattern") or "")
            throttle_key = (pair, pattern_key)
            reasons_set  = frozenset([reason_norm])

            now = datetime.now(timezone.utc)
            cached = self._throttle.get(throttle_key)
            if cached:
                age = (now - cached["ts"]).total_seconds()
                if age < _THROTTLE_SECS and cached["reasons"] == reasons_set:
                    return False  # same reason, within 1h — suppress

            record = self._build_record(pair, reason_norm, context, now)
            ok = self._write(record)
            if ok:
                self._throttle[throttle_key] = {"ts": now, "reasons": reasons_set}
            return ok
        except Exception as e:
            logger.debug(f"CandidateBlockLogger.log_block failed (non-fatal): {e}")
            return False

    # ------------------------------------------------------------------
    def _normalise(self, reason: str) -> str:
        r = _REASON_ALIASES.get(reason, reason)
        if r not in BLOCK_REASONS:
            logger.debug(f"CandidateBlockLogger: unknown reason {reason!r} → kept as-is")
        return r

    # ------------------------------------------------------------------
    def _build_record(
        self,
        pair: str,
        reason_norm: str,
        ctx: Dict[str, Any],
        now: datetime,
    ) -> dict:
        # Resolve effective_paused from components if not explicitly provided
        bot_mode       = ctx.get("bot_mode", "unknown")
        pne            = bool(ctx.get("pause_new_entries", False))
        expiry_ts      = ctx.get("pause_expiry_ts")
        chop_active    = False
        if expiry_ts:
            try:
                chop_active = now < datetime.fromisoformat(expiry_ts)
            except Exception:
                pass
        effective_paused = ctx.get("effective_paused",
                                   (bot_mode == "paused") or pne or chop_active)

        return {
            "ts":                    now.isoformat(),
            "event":                 "CANDIDATE_BLOCKED",
            "pair":                  pair,
            "pattern":               ctx.get("pattern"),
            "direction":             ctx.get("direction"),
            # block reason
            "block_reasons":         [reason_norm],
            # confidence gate
            "candidate_confidence":  ctx.get("candidate_confidence"),
            "confidence_threshold":  ctx.get("confidence_threshold"),
            # RR gate
            "candidate_rr":          ctx.get("candidate_rr"),
            "min_rr_threshold":      ctx.get("min_rr_threshold"),
            # session gate
            "session_allowed":       ctx.get("session_allowed"),
            "session_reason":        ctx.get("session_reason", ""),
            # weekly cap gate
            "weekly_cap_remaining":  ctx.get("weekly_cap_remaining"),
            "weekly_cap_limit":      ctx.get("weekly_cap_limit"),
            # control plane
            "bot_mode":              bot_mode,
            "pause_new_entries":     pne,
            "pause_expiry_ts":       expiry_ts,
            "effective_paused":      effective_paused,
            # chop shield state
            "loss_streak":           ctx.get("loss_streak"),
            "paused_by_chop":        ctx.get("paused_by_chop", False),
            "recovery_rules_active": ctx.get("recovery_rules_active", False),
            # HTF alignment (informational only)
            "htf_aligned":           ctx.get("htf_aligned"),
            # stop gate (informational)
            "stop_distance_pips":    ctx.get("stop_distance_pips"),
            "atr_max_pips":          ctx.get("atr_max_pips"),
            "atr_min_pips":          ctx.get("atr_min_pips"),
        }

    # ------------------------------------------------------------------
    def _write(self, record: dict) -> bool:
        """Write record; return True on success, False on any error."""
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file, "a") as f:
                f.write(json.dumps(record) + "\n")
            return True
        except Exception as e:
            logger.warning(f"CandidateBlockLogger._write failed: {e}")
            return False
