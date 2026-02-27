"""
control_state.py
================
Persistent control plane for the live bot.

Manages runtime_state/control.json — the single file that carries:
  • pause_new_entries  : bool  — block NEW entries; open trade management continues
  • last_updated       : ISO timestamp
  • updated_by         : "dashboard" | "api" | "telegram" | "startup"
  • reason             : free-form reason string

Design principles:
  • File is WRITTEN on change, READ on every startup + every orchestrator cycle.
  • Missing file → default (pause_new_entries = False, bot runs normally).
  • Never auto-reset — only explicit set_pause() / set_resume() calls change it.
  • Backtester NEVER reads or writes this file (isolated by is_backtest guard).
  • Thread-safe: writes are atomic (write to temp, rename).

Usage:
    from src.execution.control_state import ControlState

    ctrl = ControlState()               # loads current state (or default)
    ctrl.pause("Waiting for NFP")       # set pause, persist
    ctrl.resume()                       # clear pause, persist
    if ctrl.pause_new_entries:          # check before entry
        ...
    d = ctrl.to_dict()                  # for /api/status
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── File location ──────────────────────────────────────────────────────────
_DEFAULT_DIR  = Path(__file__).resolve().parents[2] / "runtime_state"
CONTROL_FILE  = _DEFAULT_DIR / "control.json"

_DEFAULT_STATE: dict = {
    "pause_new_entries": False,
    "last_updated":      None,
    "updated_by":        "startup",
    "reason":            "",
}


class ControlState:
    """
    Thin wrapper around runtime_state/control.json.
    Instantiate once per process; call reload() to refresh from disk.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        is_backtest: bool = False,
    ) -> None:
        self._path        = Path(path) if path else CONTROL_FILE
        self._is_backtest = is_backtest
        self._state: dict = dict(_DEFAULT_STATE)

        if not is_backtest:
            self.reload()

    # ── Read ───────────────────────────────────────────────────────────

    @property
    def pause_new_entries(self) -> bool:
        return bool(self._state.get("pause_new_entries", False))

    @property
    def reason(self) -> str:
        return str(self._state.get("reason", ""))

    @property
    def last_updated(self) -> Optional[str]:
        return self._state.get("last_updated")

    @property
    def updated_by(self) -> str:
        return str(self._state.get("updated_by", "unknown"))

    def to_dict(self) -> dict:
        return {
            "pause_new_entries": self.pause_new_entries,
            "last_updated":      self.last_updated,
            "updated_by":        self.updated_by,
            "reason":            self.reason,
        }

    # ── Write ──────────────────────────────────────────────────────────

    def pause(self, reason: str = "", updated_by: str = "api") -> None:
        """Block new entries. Open positions continue to be managed."""
        if self._is_backtest:
            return   # backtester never modifies live control file
        self._update(pause_new_entries=True, reason=reason, updated_by=updated_by)
        logger.info(f"⏸  pause_new_entries=True  by={updated_by}  reason={reason!r}")

    def resume(self, reason: str = "", updated_by: str = "api") -> None:
        """Allow new entries again."""
        if self._is_backtest:
            return
        self._update(pause_new_entries=False, reason=reason, updated_by=updated_by)
        logger.info(f"▶  pause_new_entries=False  by={updated_by}  reason={reason!r}")

    def reload(self) -> None:
        """Re-read control.json from disk (called each orchestrator cycle)."""
        if self._is_backtest:
            return
        if not self._path.exists():
            self._state = dict(_DEFAULT_STATE)
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._state = {**_DEFAULT_STATE, **raw}
        except Exception as exc:
            logger.warning(f"control_state: failed to read {self._path}: {exc}")
            # Keep current in-memory state rather than wiping

    # ── Internal ───────────────────────────────────────────────────────

    def _update(self, pause_new_entries: bool, reason: str, updated_by: str) -> None:
        self._state = {
            "pause_new_entries": pause_new_entries,
            "last_updated":      datetime.now(timezone.utc).isoformat(),
            "updated_by":        updated_by,
            "reason":            reason,
        }
        self._persist()

    def _persist(self) -> None:
        """Atomic write: temp file → rename, so a crash mid-write is safe."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp = tempfile.mkstemp(
                dir=self._path.parent, suffix=".tmp", prefix="control_"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as exc:
            logger.error(f"control_state: failed to persist {self._path}: {exc}")
