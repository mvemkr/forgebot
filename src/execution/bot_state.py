"""
Bot State Manager — Persistence layer for crash recovery and dashboard.

Writes a rich JSON state file every cycle:
  ~/trading-bot/logs/bot_state.json

On startup, reads this file + reconciles with OANDA to restore:
  - Open positions (pairs, entry, stop, direction)
  - Account balance
  - Kill switch status
  - Last scan results and pair analysis
  - Recent decisions

This means if the power goes out, comes back up, and the bot restarts:
it knows exactly where it left off, what was open, and why.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

STATE_FILE = Path.home() / "trading-bot" / "logs" / "bot_state.json"
DECISIONS_FILE = Path.home() / "trading-bot" / "logs" / "decision_log.jsonl"


class BotState:
    """
    Manages persistent bot state for crash recovery and dashboard.
    """

    def __init__(self, path: Path = STATE_FILE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Any] = {}

    # ── Write ──────────────────────────────────────────────────────────

    def save(
        self,
        *,
        account_balance: float,
        dry_run: bool,
        halted: bool,
        halt_reason: Optional[str],
        risk_pct: float,
        open_positions: Dict,
        pair_analysis: Dict,
        recent_decisions: List,
        last_scan_time: Optional[str] = None,
        top_setups: List = None,
        watching: List = None,
        stats: Dict = None,
        mode: str = "running",
    ):
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "dry_run": dry_run,
            "account_balance": account_balance,
            "risk_pct": risk_pct,
            "halted": halted,
            "halt_reason": halt_reason,
            "open_positions": open_positions,
            "pair_analysis": pair_analysis,
            "top_setups": top_setups or [],
            "watching": watching or [],
            "last_scan_time": last_scan_time,
            "stats": stats or {},
            "recent_decisions": recent_decisions[-20:],  # last 20 decisions
        }
        self._state = state
        try:
            self.path.write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save bot state: {e}")

    def log_decision(self, event: str, pair: str, details: dict):
        """Append a decision to the running decision log."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "pair": pair,
            **details,
        }
        try:
            with open(DECISIONS_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        return entry

    # ── Read / Recovery ────────────────────────────────────────────────

    def load(self) -> Dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except Exception as e:
            logger.error(f"Failed to load bot state: {e}")
            return {}

    def get_recent_decisions(self, n: int = 30) -> List[dict]:
        if not DECISIONS_FILE.exists():
            return []
        lines = []
        try:
            with open(DECISIONS_FILE) as f:
                lines = f.readlines()
        except Exception:
            return []
        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line.strip()))
            except Exception:
                pass
        return list(reversed(entries))

    def reconcile_with_oanda(self, oanda_client, strategy) -> Dict:
        """
        On startup, reconcile our saved state with actual OANDA open trades.
        Returns a summary of what was recovered.
        """
        saved = self.load()
        if not saved:
            logger.info("No saved state found — starting fresh")
            return {"recovered": False}

        age_seconds = 0
        try:
            saved_ts = datetime.fromisoformat(saved["timestamp"])
            age_seconds = (datetime.now(timezone.utc) - saved_ts).total_seconds()
            age_min = int(age_seconds / 60)
            logger.info(f"Found saved state from {age_min}m ago")
        except Exception:
            pass

        try:
            oanda_trades = oanda_client.get_open_trades()
            oanda_open = {t["instrument"].replace("_", "/"): t for t in oanda_trades}
        except Exception as e:
            logger.error(f"OANDA reconciliation failed: {e}")
            return {"recovered": False, "error": str(e)}

        recovered_positions = {}
        saved_positions = saved.get("open_positions", {})

        for pair, pos in saved_positions.items():
            if pair in oanda_open:
                # Trade still open on OANDA — restore full context from saved state.
                # Start with everything we saved (the full "why" — pattern, trends,
                # confidence, entry reason, tier) then overlay live OANDA values for
                # fields that may have changed (stop, units, trade ID).
                oanda_trade = oanda_open[pair]
                restored = dict(pos)  # full saved context — pattern, trends, reason, etc.
                restored["oanda_trade_id"] = oanda_trade["id"]
                restored["units"]          = oanda_trade["units"]
                # Trust OANDA's stop as ground truth (may have been moved to breakeven)
                live_stop = oanda_trade.get("stop_loss")
                if live_stop:
                    restored["stop"] = live_stop
                strategy.open_positions[pair] = restored
                recovered_positions[pair] = restored
                logger.info(
                    f"Recovered open position: {pair} {restored.get('direction','?')} "
                    f"entry={restored.get('entry','?')} stop={restored.get('stop','?')} "
                    f"pattern={restored.get('pattern_type','?')} "
                    f"conf={restored.get('confidence', 0):.0%}"
                )
            else:
                logger.info(f"Saved position {pair} not found on OANDA — may have been closed")

        return {
            "recovered": True,
            "state_age_seconds": age_seconds,
            "recovered_positions": recovered_positions,
            "oanda_positions": list(oanda_open.keys()),
            "prev_balance": saved.get("account_balance"),
            "was_halted": saved.get("halted", False),
            "halt_reason": saved.get("halt_reason"),
        }
