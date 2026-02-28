"""
Trade Journal — Structured event log for every decision the bot makes.

Format: JSON Lines (.jsonl) — one JSON object per line.
File: ~/trading-bot/logs/trade_journal.jsonl

Every meaningful bot event is logged:
  SETUP_DETECTED    — alignment found, waiting for entry signal
  TRADE_ENTERED     — limit order placed
  TRADE_EXITED      — position closed (stop hit or manual close)
  EXIT_SIGNAL       — exit signal fired (pin bar, engulfing against trade)
  BREAKEVEN_MOVED   — SL moved to breakeven
  KILL_SWITCH       — circuit breaker triggered
  SCAN_COMPLETE     — weekly scanner finished (summary)
  BLOCKED           — trade blocked with reason

This is your record. Over time the stats here tell you whether the strategy
is working, what the actual win rate / R:R is, and where the edge lives.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

JOURNAL_PATH = Path.home() / "trading-bot" / "logs" / "trade_journal.jsonl"


class TradeJournal:
    """
    Append-only trade journal. Thread-safe for single-process use.
    """

    def __init__(self, path: Path = JOURNAL_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ── Write ─────────────────────────────────────────────────────────

    def _write(self, entry: dict):
        entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"[JOURNAL] {entry['event']} | {entry.get('pair','?')} | {entry.get('notes','')}")

    def log_setup_detected(
        self,
        pair: str,
        direction: str,
        key_level: float,
        level_score: int,
        pattern: str,
        pattern_clarity: float,
        trend_weekly: str,
        trend_daily: str,
        trend_4h: str,
        notes: str = "",
    ):
        self._write({
            "event": "SETUP_DETECTED",
            "pair": pair,
            "direction": direction,
            "key_level": key_level,
            "level_score": level_score,
            "pattern": pattern,
            "pattern_clarity": pattern_clarity,
            "trend_weekly": trend_weekly,
            "trend_daily": trend_daily,
            "trend_4h": trend_4h,
            "notes": notes,
        })

    def log_trade_entered(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        lot_size: float,
        units: int,
        risk_pct: float,
        risk_dollars: float,
        account_balance: float,
        pattern: str,
        pattern_clarity: float,
        signal_type: str,
        signal_strength: float,
        level_score: int,
        trend_weekly: str,
        trend_daily: str,
        trend_4h: str,
        session: str,
        confidence: float,
        oanda_trade_id: Optional[str] = None,
        dry_run: bool = False,
        notes: str = "",
    ):
        stop_pips = abs(entry_price - stop_loss) * (100 if "JPY" in pair else 10000)
        self._write({
            "event": "TRADE_ENTERED",
            "pair": pair,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "stop_pips": round(stop_pips, 1),
            "lot_size": lot_size,
            "units": units,
            "risk_pct": risk_pct,
            "risk_dollars": risk_dollars,
            "account_balance_before": account_balance,
            "pattern": pattern,
            "pattern_clarity": pattern_clarity,
            "signal_type": signal_type,
            "signal_strength": signal_strength,
            "level_score": level_score,
            "trend_weekly": trend_weekly,
            "trend_daily": trend_daily,
            "trend_4h": trend_4h,
            "session": session,
            "confidence": confidence,
            "oanda_trade_id": oanda_trade_id,
            "dry_run": dry_run,
            "exit_price": None,
            "exit_reason": None,
            "pnl_pips": None,
            "pnl_dollars": None,
            "rr_achieved": None,
            "notes": notes,
        })

    def log_trade_exited(
        self,
        pair: str,
        oanda_trade_id: Optional[str],
        exit_price: float,
        exit_reason: str,   # 'stop_hit', 'exit_signal_manual', 'manual_close', 'breakeven_stop'
        entry_price: float,
        stop_loss: float,
        direction: str,
        units: int,
        account_balance_after: float,
        notes: str = "",
    ):
        pip_mult = 100 if "JPY" in pair else 10000
        if direction == "long":
            pnl_pips = (exit_price - entry_price) * pip_mult
        else:
            pnl_pips = (entry_price - exit_price) * pip_mult

        risk_pips = abs(entry_price - stop_loss) * pip_mult
        rr = pnl_pips / risk_pips if risk_pips > 0 else 0

        self._write({
            "event": "TRADE_EXITED",
            "pair": pair,
            "oanda_trade_id": oanda_trade_id,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_loss": stop_loss,
            "exit_reason": exit_reason,
            "units": units,
            "pnl_pips": round(pnl_pips, 1),
            # Pip value → USD conversion.
            # For JPY pairs the raw calculation gives JPY, not USD.
            # Divide by approximate USD/JPY rate (150) to convert.
            # For non-JPY pairs 0.0001 pip × units ≈ USD directly.
            "pnl_dollars": round(
                pnl_pips * units * (0.01 / 150 if "JPY" in pair else 0.0001), 2
            ),
            "rr_achieved": round(rr, 2),
            "account_balance_after": account_balance_after,
            "notes": notes,
        })

    def log_exit_signal(
        self,
        pair: str,
        signal_type: str,   # e.g. 'bearish_engulfing_against', 'pin_bar_against'
        current_price: float,
        entry_price: float,
        direction: str,
        oanda_trade_id: Optional[str] = None,
        notes: str = "",
    ):
        """
        Exit signal detected — bot alerts Mike but does NOT auto-close.
        Mike reviews and decides whether to close or hold.
        """
        self._write({
            "event": "EXIT_SIGNAL",
            "pair": pair,
            "signal_type": signal_type,
            "current_price": current_price,
            "entry_price": entry_price,
            "direction": direction,
            "oanda_trade_id": oanda_trade_id,
            "notes": f"⚠️ EXIT SIGNAL — review and decide. {notes}",
        })

    def log_breakeven_moved(
        self,
        pair: str,
        oanda_trade_id: Optional[str],
        entry_price: float,
        current_price: float,
        direction: str,
        notes: str = "",
    ):
        self._write({
            "event": "BREAKEVEN_MOVED",
            "pair": pair,
            "oanda_trade_id": oanda_trade_id,
            "entry_price": entry_price,
            "current_price": current_price,
            "direction": direction,
            "new_stop": entry_price,
            "notes": notes,
        })

    def log_kill_switch(self, reason: str, action_taken: str, account_balance: float, notes: str = ""):
        self._write({
            "event": "KILL_SWITCH",
            "pair": "ALL",
            "reason": reason,
            "action_taken": action_taken,
            "account_balance": account_balance,
            "notes": notes,
        })

    def log_blocked(self, pair: str, reason: str, failed_filters: list, notes: str = ""):
        self._write({
            "event": "BLOCKED",
            "pair": pair,
            "reason": reason,
            "failed_filters": failed_filters,
            "notes": notes,
        })

    def log_scan_complete(self, pairs_scanned: int, prime_setups: list, watching: list, notes: str = ""):
        self._write({
            "event": "SCAN_COMPLETE",
            "pair": "SCAN",
            "pairs_scanned": pairs_scanned,
            "prime_setups": prime_setups,
            "watching": watching,
            "notes": notes,
        })

    def log_reconcile_event(
        self,
        pair: str,
        event: str,   # "EXTERNAL_CLOSE" | "RECOVERED_POSITION" | "EXTERNAL_MODIFY" | "RECONCILE_PAUSE"
        data: dict,
        notes: str = "",
    ):
        """Record a broker reconciliation event (state integrity, not a trade signal)."""
        self._write({
            "event":   f"RECONCILE_{event}",
            "pair":    pair,
            "notes":   notes or data.get("notes", ""),
            **{k: v for k, v in data.items() if k != "notes"},
        })

    # ── Read / Stats ──────────────────────────────────────────────────

    def _load_all(self) -> List[dict]:
        if not self.path.exists():
            return []
        entries = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def get_open_trades(self) -> List[dict]:
        """Return all TRADE_ENTERED events that don't have a matching TRADE_EXITED."""
        entries = self._load_all()
        entered = {e["oanda_trade_id"]: e for e in entries
                   if e.get("event") == "TRADE_ENTERED" and e.get("oanda_trade_id")}
        exited_ids = {e.get("oanda_trade_id") for e in entries
                      if e.get("event") == "TRADE_EXITED"}
        return [v for k, v in entered.items() if k not in exited_ids]

    def get_recent_trades(self, n: int = 10) -> List[dict]:
        """Last N completed trades (TRADE_EXITED events)."""
        entries = self._load_all()
        exits = [e for e in entries if e.get("event") == "TRADE_EXITED"]
        return exits[-n:]

    def get_all_trades(self) -> List[dict]:
        """
        All completed trades as normalized dicts for the trade analyzer.
        Merges TRADE_ENTERED + TRADE_EXITED records into single trade objects.
        """
        entries = self._load_all()
        entered = {}
        for e in entries:
            if e.get("event") == "TRADE_ENTERED":
                key = e.get("oanda_trade_id") or e.get("pair", "") + str(e.get("timestamp",""))
                entered[key] = e

        trades = []
        for e in entries:
            if e.get("event") != "TRADE_EXITED":
                continue
            key = e.get("oanda_trade_id") or ""
            entry_data = entered.get(key, {})
            trades.append({
                "pair":            e.get("pair"),
                "direction":       e.get("direction") or entry_data.get("direction"),
                "entry_price":     entry_data.get("entry_price", 0),
                "exit_price":      e.get("exit_price", 0),
                "stop_loss":       entry_data.get("stop_loss", 0),
                "entry_ts":        entry_data.get("timestamp", ""),
                "exit_ts":         e.get("timestamp", ""),
                "exit_reason":     e.get("exit_reason", ""),
                "pnl":             e.get("pnl_dollars", 0),
                "rr":              e.get("rr_achieved", 0),
                "pattern_type":    entry_data.get("pattern_type", ""),
                "psych_level":     entry_data.get("psych_level", 0),
                "key_level_score": entry_data.get("key_level_score", 0),
                "signal_strength": entry_data.get("signal_strength", 0),
                "session":         entry_data.get("session", ""),
                "risk_pct":        entry_data.get("risk_pct", 0),
                "risk_dollars":    entry_data.get("risk_dollars", 0),
                "bars_held":       e.get("bars_held", 0),
                "confidence":      entry_data.get("confidence", 0),
            })
        return trades

    def get_stats(self) -> Dict[str, Any]:
        """Win rate, avg R:R, total P&L, trade count."""
        entries = self._load_all()
        exits = [e for e in entries if e.get("event") == "TRADE_EXITED"]

        if not exits:
            return {"trades": 0, "message": "No completed trades yet"}

        wins = [e for e in exits if (e.get("rr_achieved") or 0) > 0]
        losses = [e for e in exits if (e.get("rr_achieved") or 0) <= 0]

        rr_values = [e["rr_achieved"] for e in exits if e.get("rr_achieved") is not None]
        pnl_values = [e["pnl_dollars"] for e in exits if e.get("pnl_dollars") is not None]

        best = max(exits, key=lambda e: e.get("rr_achieved", -99), default=None)
        worst = min(exits, key=lambda e: e.get("rr_achieved", 99), default=None)

        return {
            "trades": len(exits),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(exits) * 100, 1) if exits else 0,
            "avg_rr": round(sum(rr_values) / len(rr_values), 2) if rr_values else 0,
            "total_pnl": round(sum(pnl_values), 2) if pnl_values else 0,
            "best_trade": f"{best['pair']} {best['rr_achieved']:+.1f}R (${best['pnl_dollars']:+.0f})" if best else None,
            "worst_trade": f"{worst['pair']} {worst['rr_achieved']:+.1f}R (${worst['pnl_dollars']:+.0f})" if worst else None,
        }

    def get_weekly_pnl(self) -> Dict[str, float]:
        """P&L grouped by ISO week string (e.g. '2026-W08')."""
        entries = self._load_all()
        exits = [e for e in entries if e.get("event") == "TRADE_EXITED"]
        weekly: Dict[str, float] = {}
        for e in exits:
            try:
                dt = datetime.fromisoformat(e["timestamp"])
                week_key = f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"
                weekly[week_key] = weekly.get(week_key, 0) + (e.get("pnl_dollars") or 0)
            except Exception:
                pass
        return dict(sorted(weekly.items()))

    def get_trades_this_week(self, as_of: "datetime | None" = None) -> int:
        """
        Return the number of trades *entered* (TRADE_OPENED events) in the
        current ISO calendar week up to *as_of* (defaults to now UTC).

        Used by the live orchestrator to enforce the weekly punch-card gate
        (alex_policy.check_weekly_trade_limit).
        """
        if as_of is None:
            as_of = datetime.now(timezone.utc)
        iso_year, iso_week, _ = as_of.isocalendar()
        entries = self._load_all()
        count = 0
        for e in entries:
            if e.get("event") != "TRADE_OPENED":
                continue
            try:
                dt = datetime.fromisoformat(e["timestamp"])
                y, w, _ = dt.isocalendar()
                if y == iso_year and w == iso_week:
                    count += 1
            except Exception:
                pass
        return count

    def get_current_week_stats(self, as_of: "datetime | None" = None) -> dict:
        """
        Return closed-trade stats for the *current* ISO calendar week.

        Returns dict with keys:
          trades_this_week  : int
          wins_this_week    : int
          losses_this_week  : int
          pnl_this_week     : float  (sum of pnl_dollars for TRADE_EXITED events)

        Used by standings display to show W/L counts independently of AccountState
        (which tracks week_pnl but not win/loss breakdown).
        """
        if as_of is None:
            as_of = datetime.now(timezone.utc)
        iso_year, iso_week, _ = as_of.isocalendar()
        entries = self._load_all()

        exits_this_week = []
        for e in entries:
            if e.get("event") != "TRADE_EXITED":
                continue
            try:
                dt = datetime.fromisoformat(e["timestamp"])
                y, w, _ = dt.isocalendar()
                if y == iso_year and w == iso_week:
                    exits_this_week.append(e)
            except Exception:
                pass

        wins   = [e for e in exits_this_week if (e.get("rr_achieved") or 0) > 0]
        losses = [e for e in exits_this_week if (e.get("rr_achieved") or 0) <= 0]
        pnl    = sum(e.get("pnl_dollars", 0) for e in exits_this_week)

        return {
            "trades_this_week":  len(exits_this_week),
            "wins_this_week":    len(wins),
            "losses_this_week":  len(losses),
            "pnl_this_week":     round(pnl, 2),
        }
