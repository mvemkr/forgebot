"""
CandidateBlockLogger
====================
Writes structured CANDIDATE_BLOCKED records to decision_log.jsonl when a
pre-gate ENTER signal exists but is rejected by an existing gate.

Contract:
  - Does NOT change any gate logic, threshold, or execution flow.
  - One record per (pair, pattern, direction, block_reason) per hour.
    Throttle key: (pair, pattern, direction, frozenset(block_reasons)).
    Same key within 1 hour → suppressed.  Different key → logged immediately.
  - Fails silently — never raises, never blocks entry path.
  - Writes to decision_log.jsonl (decision feed) ONLY.
    NOT paper_journal.jsonl (entries/exits only).

Output path:
  Default: Path.home() / "trading-bot" / "logs" / "decision_log.jsonl"
  Follows the same LOG_DIR convention as orchestrator.py / dashboard/app.py.
  Callers should pass the module-level DECISIONS_FILE constant from orchestrator
  so both share exactly one path definition.

Block reason enums (12 total):
  CONFIDENCE_BLOCK      candidate_confidence < confidence_threshold
  RR_BLOCK              candidate_rr < min_rr_threshold (2.5R/3.0R from alex_policy)
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

Threshold fields:
  confidence_threshold  → MIN_CONFIDENCE from strategy_config (0.65)
  min_rr_threshold      → MIN_RR_STANDARD (2.5R) if htf_aligned else
                           MIN_RR_COUNTERTREND (2.5R / 3.0R per config);
                          auto-derived from strategy_config in _build_record.
                          For RECOVERY_RULES_BLOCK: RECOVERY_MIN_RR (3.0R).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default output path — same LOG_DIR convention as orchestrator.py / dashboard/app.py.
# Orchestrator passes its own DECISIONS_FILE constant so both share one definition.
_LOG_DIR        = Path.home() / "trading-bot" / "logs"
_DECISIONS_FILE = _LOG_DIR / "decision_log.jsonl"

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
    "TIME_BLOCK":          "SESSION_BLOCK",
    "RR_MIN":              "RR_BLOCK",
    "RECOVERY_MIN_RR":     "RECOVERY_RULES_BLOCK",
    "RECOVERY_CONF":       "RECOVERY_RULES_BLOCK",
    "RECOVERY_WEEKLY_CAP": "RECOVERY_RULES_BLOCK",
    "WEEKLY_TRADE_LIMIT":  "WEEKLY_CAP_BLOCK",
    "CONF":                "CONFIDENCE_BLOCK",
    "CONFIDENCE":          "CONFIDENCE_BLOCK",
    "WINNER_RUNNING":      "WINNER_RUNNING_BLOCK",
    "THEME_CONFLICT":      "THEME_CONFLICT_BLOCK",
    "STOP_WIDE":           "STOP_WIDE_BLOCK",
    "STOP_TIGHT":          "STOP_TIGHT_BLOCK",
    "PAUSED":              "PAUSE_BLOCK",
    "CHOP_PAUSE":          "CHOP_SHIELD_BLOCK",
    "RECONCILE":           "RECONCILE_PAUSE_BLOCK",
}

_THROTTLE_SECS = 3600  # suppress identical (pair, pattern, direction, reasons) within 1 hour


class CandidateBlockLogger:
    """
    Log one CANDIDATE_BLOCKED record per gate-rejection, throttled per
    (pair, pattern, direction, frozenset(block_reasons)) to at most once per hour.

    Throttle key includes direction so a short and long signal on the same pair/
    pattern are treated as distinct candidates.  Including block_reasons means a
    reason change (CONFIDENCE_BLOCK → RR_BLOCK) is always logged immediately.
    """

    def __init__(self, decisions_file: Optional[Path] = None):
        # Callers should pass the module-level DECISIONS_FILE from orchestrator.py
        # so both components share a single path definition.
        self._file = decisions_file or _DECISIONS_FILE
        # throttle: (pair, pattern, direction, frozenset(block_reasons)) → last write datetime
        self._throttle: Dict[tuple, datetime] = {}

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
        context       : dict with gate-context fields (see _build_record)

        Returns True if written, False if throttled or write failed.
        """
        try:
            reason_norm  = self._normalise(block_reason)
            reasons_set  = frozenset([reason_norm])
            pattern      = str(context.get("pattern") or "")
            direction    = str(context.get("direction") or "")
            throttle_key = (pair, pattern, direction, reasons_set)

            now    = datetime.now(timezone.utc)
            cached = self._throttle.get(throttle_key)
            if cached is not None:
                age = (now - cached).total_seconds()
                if age < _THROTTLE_SECS:
                    return False  # same key within 1h — suppress

            record = self._build_record(pair, reason_norm, context, now)
            ok = self._write(record)
            if ok:
                self._throttle[throttle_key] = now
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
    def _rr_threshold(self, ctx: Dict[str, Any]) -> Optional[float]:
        """
        Derive the applicable min_rr_threshold from strategy_config.
        Uses htf_aligned to select STANDARD vs COUNTERTREND threshold.
        For RECOVERY_RULES_BLOCK the caller should pass min_rr_threshold explicitly.
        Falls back to explicit ctx value if config import fails.
        """
        # Explicit override wins (e.g. recovery mode passes RECOVERY_MIN_RR)
        explicit = ctx.get("min_rr_threshold")
        if explicit is not None:
            return explicit
        try:
            from ..strategy.forex import strategy_config as _sc
            htf = ctx.get("htf_aligned", True)
            if htf:
                return float(getattr(_sc, "MIN_RR_STANDARD", 2.5))
            else:
                return float(getattr(_sc, "MIN_RR_COUNTERTREND", 2.5))
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _build_record(
        self,
        pair: str,
        reason_norm: str,
        ctx: Dict[str, Any],
        now: datetime,
    ) -> dict:
        bot_mode     = ctx.get("bot_mode", "unknown")
        pne          = bool(ctx.get("pause_new_entries", False))
        expiry_ts    = ctx.get("pause_expiry_ts")
        chop_active  = False
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
            # RR gate — threshold auto-derived from strategy_config (MIN_RR_STANDARD /
            # MIN_RR_COUNTERTREND) based on htf_aligned; explicit ctx value wins
            "candidate_rr":          ctx.get("candidate_rr"),
            "min_rr_threshold":      self._rr_threshold(ctx),
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
            # HTF alignment (informational only — also drives min_rr_threshold selection)
            "htf_aligned":           ctx.get("htf_aligned"),
            # stop gate
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
