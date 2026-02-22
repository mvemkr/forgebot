"""
Forge Bot Dashboard — Flask API + Web UI
Serves the Tailwind/DaisyUI/Alpine.js dashboard at http://localhost:5001
"""
import json, sys, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from flask import Flask, render_template, jsonify

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.exchange.oanda_client import OandaClient
from src.execution.trade_journal import TradeJournal

LOG_DIR        = Path.home() / "trading-bot" / "logs"
HEARTBEAT_FILE = LOG_DIR / "forex_orchestrator.heartbeat"
STATE_FILE     = LOG_DIR / "bot_state.json"
DECISIONS_FILE = LOG_DIR / "decision_log.jsonl"

app = Flask(__name__)
logging.basicConfig(level=logging.WARNING)

_oanda: Optional[OandaClient]   = None
_journal: Optional[TradeJournal] = None

def get_oanda():
    global _oanda
    if _oanda is None:
        try: _oanda = OandaClient()
        except Exception: pass
    return _oanda

def get_journal():
    global _journal
    if _journal is None:
        _journal = TradeJournal()
    return _journal

def _age_label(s):
    if s < 60: return f"{int(s)}s ago"
    if s < 3600: return f"{int(s/60)}m ago"
    return f"{int(s/3600)}h ago"

def load_heartbeat():
    if not HEARTBEAT_FILE.exists():
        return {"status": "not_started", "timestamp": None}
    try:
        d = json.loads(HEARTBEAT_FILE.read_text())
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(d["timestamp"])).total_seconds()
        d["age_seconds"] = int(age)
        d["age_label"]   = _age_label(age)
        d["status_emoji"] = "🟢" if age < 120 else "🟡" if age < 600 else "🔴"
        return d
    except Exception as e:
        return {"status": "error", "error": str(e)}

def load_bot_state():
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}

def load_recent_decisions(n=40):
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
        try: entries.append(json.loads(line.strip()))
        except Exception: pass
    return list(reversed(entries))

def get_scan_state():
    """Last 48h of pair events from the trade journal."""
    journal = get_journal()
    entries = journal._load_all()
    cutoff  = datetime.now(timezone.utc) - timedelta(hours=48)
    pair_state = {}
    for e in entries:
        try:
            ts = datetime.fromisoformat(e["timestamp"])
            if ts < cutoff: continue
        except Exception:
            continue
        pair = e.get("pair", "?")
        if pair in ("ALL", "SCAN"): continue
        if e.get("event") in ("SETUP_DETECTED","TRADE_ENTERED","BLOCKED","EXIT_SIGNAL","BREAKEVEN_MOVED"):
            pair_state[pair] = {
                "pair":         pair,
                "event":        e.get("event",""),
                "timestamp":    e.get("timestamp",""),
                "direction":    e.get("direction","?"),
                "pattern":      e.get("pattern",""),
                "level_score":  e.get("level_score",0),
                "notes":        e.get("notes", e.get("reason","")),
                "trend_weekly": e.get("trend_weekly","?"),
                "trend_daily":  e.get("trend_daily","?"),
                "trend_4h":     e.get("trend_4h","?"),
            }
    return sorted(pair_state.values(), key=lambda x: x["timestamp"], reverse=True)

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/status")
def api_status():
    heartbeat     = load_heartbeat()
    bot_state     = load_bot_state()
    journal       = get_journal()
    recent_decs   = load_recent_decisions(40)

    account = {}; open_trades = []
    try:
        oanda = get_oanda()
        if oanda:
            account     = oanda.get_account_summary()
            open_trades = oanda.get_open_trades()
    except Exception as e:
        account = {"error": str(e)}

    stats         = journal.get_stats()
    recent_trades = journal.get_recent_trades(15)
    scan_state    = get_scan_state()

    # Merge heartbeat + state for richer payload
    merged_hb = {**heartbeat, **{k: v for k, v in bot_state.items()
                                 if k not in heartbeat or bot_state.get(k) is not None}}

    # Risk / mode enrichment from bot_state stats block
    bot_stats   = bot_state.get("stats", {})
    risk_pct    = merged_hb.get("risk_pct") or bot_stats.get("risk_pct", 0)
    mode        = merged_hb.get("mode") or bot_stats.get("mode", "unknown")
    tier_label  = bot_stats.get("tier", "—")
    peak_bal    = bot_stats.get("peak_balance", 0)
    regroup_ends= bot_stats.get("regroup_ends")
    pattern_ct  = len(bot_stats.get("traded_pattern_keys", []))
    dry_run     = merged_hb.get("dry_run", True)
    halted      = merged_hb.get("halted", False)

    # Win rate from stats
    wr      = stats.get("win_rate", 0) if isinstance(stats, dict) else 0
    tot_tr  = stats.get("trades", 0)   if isinstance(stats, dict) else 0
    tot_pnl = stats.get("total_pnl", 0) if isinstance(stats, dict) else 0

    # Drawdown from peak
    bal = account.get("balance", 0) if isinstance(account, dict) else 0
    dd_pct = round((peak_bal - bal) / peak_bal * 100, 1) if peak_bal > 0 and bal < peak_bal else 0.0

    return jsonify({
        "heartbeat":          merged_hb,
        "account":            account,
        "open_trades":        open_trades,
        "stats":              stats,
        "recent_trades":      recent_trades,
        "scan_state":         scan_state,
        "recent_decisions":   recent_decs,
        "top_setups":         bot_state.get("top_setups", []),
        "server_time":        datetime.now(timezone.utc).isoformat(),
        # enriched fields for dashboard display
        "mode":               mode,
        "dry_run":            dry_run,
        "halted":             halted,
        "risk_pct":           risk_pct,
        "tier_label":         tier_label,
        "peak_balance":       peak_bal,
        "drawdown_pct":       dd_pct,
        "regroup_ends":       regroup_ends,
        "pattern_memory_count": pattern_ct,
        "account_balance":    bal,
        "win_rate":           round(wr * 100, 1),
        "total_trades":       tot_tr,
        "total_pnl":          tot_pnl,
    })

@app.route("/api/weekly_pnl")
def api_weekly_pnl():
    return jsonify(get_journal().get_weekly_pnl())

@app.route("/api/journal")
def api_journal():
    return jsonify({"entries": get_journal()._load_all()[-50:]})

@app.route("/api/news")
def api_news():
    """
    Return upcoming Tier 1 news events for the next 72 hours,
    filtered to currencies in our watchlist.
    Each event includes whether it currently blocks entry on any pair.
    """
    try:
        from src.strategy.forex.news_filter import NewsFilter
        from src.execution.orchestrator import WATCHLIST
        from datetime import datetime, timezone, timedelta

        nf = NewsFilter()
        nf.refresh_if_needed()

        now     = datetime.now(timezone.utc)
        cutoff  = now + timedelta(hours=72)

        # Collect all currencies in the watchlist
        watchlist_ccys = set()
        for pair in WATCHLIST:
            watchlist_ccys.update(pair.replace("_", "/").upper().split("/"))

        events = []
        for evt in nf._events:
            evt_dt = evt.get("dt_utc")
            if not evt_dt:
                continue
            if not (now - timedelta(hours=2) <= evt_dt <= cutoff):
                continue
            if evt.get("currency", "").upper() not in watchlist_ccys:
                continue

            blocked, reason = nf.is_entry_blocked(now)
            mins_away = int((evt_dt - now).total_seconds() / 60)

            events.append({
                "title":      evt.get("title", "Unknown"),
                "currency":   evt.get("currency", "?"),
                "dt_utc":     evt_dt.isoformat(),
                "mins_away":  mins_away,
                "is_past":    evt_dt < now,
                "blocks_now": blocked and abs(mins_away) <= 90,
                "affected_pairs": [
                    p for p in WATCHLIST
                    if evt.get("currency", "").upper() in p.upper()
                ],
            })

        # Sort: soonest first
        events.sort(key=lambda e: e["mins_away"])
        return jsonify({"events": events, "count": len(events)})

    except Exception as e:
        return jsonify({"events": [], "error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
