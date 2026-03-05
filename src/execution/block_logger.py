"""
CandidateBlockLogger
====================
Writes two structured event types to decision_log.jsonl:

  CANDIDATE_WAIT     — strategy found a pattern but returned WAIT before ENTER
                        (logged at pattern-detection stage; zero broker risk)
  CANDIDATE_BLOCKED  — strategy returned ENTER but a post-signal gate rejected it

Contract:
  - Does NOT change any gate logic, threshold, or execution flow.
  - One record per (pair, pattern, direction, frozenset(reasons)) per hour.
    Throttle key: (pair, pattern, direction, frozenset(reasons)).
    Same key within 1 hour → suppressed.  Different key → logged immediately.
  - Fails silently — never raises, never blocks entry path.
  - Writes to decision_log.jsonl (decision feed) ONLY.
    NOT paper_journal.jsonl (entries/exits only).

Output path:
  Default: Path.home() / "trading-bot" / "logs" / "decision_log.jsonl"
  Follows the same LOG_DIR convention as orchestrator.py / dashboard/app.py.
  Callers should pass the module-level DECISIONS_FILE constant from orchestrator
  so both share exactly one path definition.

CANDIDATE_WAIT reason codes (9 total):
  CONFIDENCE_BELOW_MIN  candidate_confidence < confidence_threshold
  NO_ZONE_TOUCH         price not in neckline zone on 1H
  AWAITING_TRIGGER      pattern+zone ok but no engulfing entry signal yet
  HTF_NOT_ALIGNED       weekly/daily trend opposes signal direction
  SESSION_CLOSED        outside allowed session window
  NEWS_BLACKOUT         high-impact news window
  MAX_CONCURRENT        already at max open positions
  CURRENCY_OVERLAP      currency already exposed in open position
  STOP_COOLDOWN         recently stopped out on this pair

CANDIDATE_BLOCKED reason codes (12 total):
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

# ── CANDIDATE_BLOCKED reason enums ───────────────────────────────────────────
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

# Alias map: internal gate names → canonical CANDIDATE_BLOCKED enum values
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

# ── CANDIDATE_WAIT reason enums ───────────────────────────────────────────────
WAIT_REASONS = frozenset({
    "CONFIDENCE_BELOW_MIN",  # confidence < threshold (pattern found but not strong enough)
    "NO_ZONE_TOUCH",         # price not within neckline zone on 1H
    "AWAITING_TRIGGER",      # pattern + zone ok, waiting for engulfing candle
    "HTF_NOT_ALIGNED",       # weekly/daily trend opposes signal direction
    "SESSION_CLOSED",        # outside allowed session window
    "NEWS_BLACKOUT",         # high-impact news window active
    "MAX_CONCURRENT",        # already at max open positions
    "CURRENCY_OVERLAP",      # currency already exposed in an open position
    "STOP_COOLDOWN",         # recently stopped out on this pair
})

# Map strategy failed_filter codes → canonical CANDIDATE_WAIT reason codes
_WAIT_FILTER_MAP: Dict[str, str] = {
    "no_zone_touch":    "NO_ZONE_TOUCH",
    "no_entry_signal":  "AWAITING_TRIGGER",
    "trend_alignment":  "HTF_NOT_ALIGNED",
    "COUNTERTREND_HTF": "HTF_NOT_ALIGNED",
    "session":          "SESSION_CLOSED",
    "news_blackout":    "NEWS_BLACKOUT",
    "max_concurrent":   "MAX_CONCURRENT",
    "currency_overlap": "CURRENCY_OVERLAP",
    "stop_cooldown":    "STOP_COOLDOWN",
    "winner_rule":      "MAX_CONCURRENT",   # winner-running = another position open
}

_THROTTLE_SECS = 3600  # suppress identical (pair, pattern, direction, reasons) within 1 hour

# ── PRE_CANDIDATE fields ──────────────────────────────────────────────────────
# Emitted when pattern detector found a forming pattern below the recognition
# floor (PatternResult.clarity < SetAndForgetStrategy.min_pattern_clarity = 0.4).
# Fields added to every PRE_CANDIDATE record:
#   recognition_floor       → the actual min_pattern_clarity constant in use
#   raw_confidence          → PatternResult.clarity (raw detector score, 0-1)
#   confidence_gap_to_floor → recognition_floor − raw_confidence (always > 0)
#   trigger_state           → "BELOW_RECOGNITION_FLOOR" (always; entry not reached)
#   atr_ratio, wd_aligned   → from last regime score
#   session_allowed, session_reason → from session filter


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
        # Cumulative count of _write failures (JSON errors + IO errors).
        # Readable by tests and monitoring; never resets across instance lifetime.
        self.candidate_log_failures: int = 0

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
    def log_wait(
        self,
        pair: str,
        failed_filters: list,
        context: Dict[str, Any],
    ) -> bool:
        """
        Emit one CANDIDATE_WAIT record when the strategy found a pattern but
        returned decision=WAIT before reaching the ENTER stage.

        Only call this when decision.pattern is not None — callers are
        responsible for that guard; log_wait does not re-check.

        Parameters
        ----------
        pair           : e.g. "USD/JPY"
        failed_filters : decision.failed_filters list from the strategy
        context        : dict with all available gate-context fields

        Returns True if written, False if throttled or write failed.
        """
        try:
            wait_reasons = self._map_wait_reasons(failed_filters, context)
            reasons_set  = frozenset(wait_reasons) if wait_reasons else frozenset({"BUILDING_SETUP"})
            pattern      = str(context.get("pattern") or "")
            direction    = str(context.get("direction") or "")
            throttle_key = (pair, pattern, direction, reasons_set)

            now    = datetime.now(timezone.utc)
            cached = self._throttle.get(throttle_key)
            if cached is not None:
                age = (now - cached).total_seconds()
                if age < _THROTTLE_SECS:
                    return False  # same key within 1h — suppress

            record = self._build_wait_record(pair, list(reasons_set), context, now)
            ok = self._write(record)
            if ok:
                self._throttle[throttle_key] = now
            return ok
        except Exception as e:
            logger.debug(f"CandidateBlockLogger.log_wait failed (non-fatal): {e}")
            return False

    # ------------------------------------------------------------------
    def log_pre_candidate(
        self,
        pair: str,
        sub_threshold_patterns: list,
        context: Dict[str, Any],
    ) -> int:
        """
        Emit one PRE_CANDIDATE record per sub-threshold pattern in
        sub_threshold_patterns.  Throttled per (pair, pattern_type, direction)
        at 1h — same window as CANDIDATE_WAIT.

        Parameters
        ----------
        pair                  : e.g. "USD/JPY"
        sub_threshold_patterns: list of PatternResult objects with
                                clarity < recognition_floor
        context               : dict from orchestrator._pre_candidate_ctx()

        Returns number of records actually written (0-N).
        """
        written = 0
        try:
            now = datetime.now(timezone.utc)
            for pat in sub_threshold_patterns:
                try:
                    pattern_type = getattr(pat, "pattern_type", str(pat))
                    direction    = getattr(pat, "direction", None)
                    throttle_key = (pair, pattern_type, direction, frozenset(["PRE_CANDIDATE"]))

                    cached = self._throttle.get(throttle_key)
                    if cached is not None and (now - cached).total_seconds() < _THROTTLE_SECS:
                        continue

                    record = self._build_pre_candidate_record(pair, pat, context, now)
                    ok = self._write(record)
                    if ok:
                        self._throttle[throttle_key] = now
                        written += 1
                except Exception as e:
                    logger.debug(f"log_pre_candidate inner failed for {pair}: {e}")
        except Exception as e:
            logger.debug(f"CandidateBlockLogger.log_pre_candidate failed (non-fatal): {e}")
        return written

    # ------------------------------------------------------------------
    def _build_pre_candidate_record(
        self,
        pair: str,
        pat,          # PatternResult
        ctx: Dict[str, Any],
        now: datetime,
    ) -> dict:
        recognition_floor  = ctx.get("recognition_floor", 0.4)
        raw_confidence     = float(getattr(pat, "clarity", 0.0))
        gap_to_floor       = round(recognition_floor - raw_confidence, 4)
        return {
            "ts":                     now.isoformat(),
            "event":                  "PRE_CANDIDATE",
            "pair":                   pair,
            "pattern_type":           getattr(pat, "pattern_type", None),
            "direction":              getattr(pat, "direction", None),
            # proximity to recognition floor
            "raw_confidence":         round(raw_confidence, 4),
            "recognition_floor":      recognition_floor,
            "confidence_gap_to_floor": gap_to_floor,
            # trigger state — always BELOW_RECOGNITION_FLOOR at this layer;
            # entry signal check is never reached for sub-threshold patterns.
            "trigger_state":          "BELOW_RECOGNITION_FLOOR",
            # regime context
            "atr_ratio":              ctx.get("atr_ratio"),
            "wd_aligned":             ctx.get("wd_aligned"),
            # session context
            "session_allowed":        ctx.get("session_allowed"),
            "session_reason":         ctx.get("session_reason", ""),
            # pattern geometry (informational)
            "neckline":               getattr(pat, "neckline", None),
            "pattern_level":          getattr(pat, "pattern_level", None),
        }

    # ------------------------------------------------------------------
    def _map_wait_reasons(
        self, failed_filters: list, context: Dict[str, Any]
    ) -> list:
        """
        Convert strategy failed_filters + computed confidence check into
        canonical WAIT_REASONS codes. Order: confidence first, then filters.
        """
        reasons: list = []

        # Computed confidence check — runs even if not in failed_filters
        thr  = context.get("confidence_threshold")
        cand = context.get("candidate_confidence")
        if thr is not None and cand is not None and float(cand) < float(thr):
            reasons.append("CONFIDENCE_BELOW_MIN")

        # Map each failed_filter to a canonical wait reason
        seen: set = set()
        for filt in (failed_filters or []):
            mapped = _WAIT_FILTER_MAP.get(filt)
            if mapped and mapped not in seen:
                reasons.append(mapped)
                seen.add(mapped)

        return reasons

    # ------------------------------------------------------------------
    def _build_wait_record(
        self,
        pair: str,
        wait_reasons: list,
        ctx: Dict[str, Any],
        now: datetime,
    ) -> dict:
        thr  = ctx.get("confidence_threshold")
        cand = ctx.get("candidate_confidence")
        conf_gap = (float(thr) - float(cand)) if (thr is not None and cand is not None) else None

        rr_thr        = self._rr_threshold(ctx)
        rr_cand       = ctx.get("candidate_rr")          # None when rr_unavailable
        rr_unavailable = ctx.get("rr_unavailable", rr_cand is None)

        # rr_gap: only compute when both values exist (i.e. RR was actually computed).
        # Never synthesise a gap from a null/zero RR — that would pollute proximity buckets.
        if not rr_unavailable and rr_thr is not None and rr_cand is not None:
            rr_gap: Optional[float] = float(rr_thr) - float(rr_cand)
        else:
            rr_gap = None

        return {
            "ts":                    now.isoformat(),
            "event":                 "CANDIDATE_WAIT",
            "pair":                  pair,
            "pattern":               ctx.get("pattern"),
            "direction":             ctx.get("direction"),
            "wait_reasons":          wait_reasons,
            # confidence proximity
            "candidate_confidence":  cand,
            "confidence_threshold":  thr,
            "conf_gap":              round(conf_gap, 4) if conf_gap is not None else None,
            # RR proximity — rr_unavailable=True means entry was never reached so
            # exec_rr was 0.0 (default).  candidate_rr and rr_gap will be null.
            # Consumers MUST exclude rr_unavailable=True rows from RR gap buckets.
            "candidate_rr":          rr_cand,
            "min_rr_threshold":      rr_thr,
            "rr_gap":                round(rr_gap, 4) if rr_gap is not None else None,
            "rr_unavailable":        rr_unavailable,
            # zone
            "zone_touch":            ctx.get("zone_touch", False),
            # HTF flags
            "htf_aligned":           ctx.get("htf_aligned"),
            "trend_weekly":          ctx.get("trend_weekly"),
            "trend_daily":           ctx.get("trend_daily"),
            "trend_4h":              ctx.get("trend_4h"),
            # regime
            "wd_aligned":            ctx.get("wd_aligned"),
            "atr_ratio":             ctx.get("atr_ratio"),
            # session
            "session_allowed":       ctx.get("session_allowed"),
            "session_reason":        ctx.get("session_reason", ""),
            # control plane / pause state
            "pause_new_entries":     ctx.get("pause_new_entries", False),
            "effective_paused":      ctx.get("effective_paused", False),
            # chop shield
            "loss_streak":           ctx.get("loss_streak"),
            "paused_by_chop":        ctx.get("paused_by_chop", False),
        }

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
        """Write record; return True on success, False on any error.

        Failures are never silent:
          - WARNING logged with event type, pair, and error detail.
          - candidate_log_failures counter incremented so callers and
            tests can assert failure visibility.
        """
        event = record.get("event", "UNKNOWN")
        pair  = record.get("pair",  "UNKNOWN")
        try:
            # Validate serializability before touching the file — raises
            # TypeError immediately if any value is a non-primitive object.
            payload = json.dumps(record)
            self._file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file, "a") as f:
                f.write(payload + "\n")
            return True
        except Exception as e:
            self.candidate_log_failures += 1
            logger.warning(
                f"CandidateBlockLogger._write failed "
                f"[event={event} pair={pair}]: {e}"
            )
            return False
