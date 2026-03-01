"""
Forge Bot Dashboard — Flask API + Web UI
Serves the Tailwind/DaisyUI/Alpine.js dashboard at http://localhost:5001
"""
import json, sys, logging, subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from flask import Flask, render_template, jsonify, request

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.exchange.oanda_client import OandaClient
from src.execution.trade_journal import TradeJournal
from src.strategy.forex.strategy_config import (
    MAX_CONCURRENT_TRADES_LIVE,
    MAX_CONCURRENT_TRADES_BACKTEST,
    ENTRY_TRIGGER_MODE,
)

LOG_DIR        = Path.home() / "trading-bot" / "logs"
HEARTBEAT_FILE = LOG_DIR / "forex_orchestrator.heartbeat"
STATE_FILE     = LOG_DIR / "bot_state.json"
DECISIONS_FILE = LOG_DIR / "decision_log.jsonl"
CONTROL_FILE   = LOG_DIR / "bot_control.json"

# Persistent control plane (pause_new_entries) — separate from one-shot bot_control.json
RUNTIME_STATE_DIR   = Path.home() / "trading-bot" / "runtime_state"
RUNTIME_CONTROL_FILE = RUNTIME_STATE_DIR / "control.json"
# Two independent whitelist files — live never touches backtest and vice versa
WHITELIST_LIVE_FILE      = LOG_DIR / "whitelist_live.json"
WHITELIST_BACKTEST_FILE  = LOG_DIR / "whitelist_backtest.json"
# Legacy path alias (for any code that may still reference the old name)
WHITELIST_FILE           = WHITELIST_LIVE_FILE

# ── Pair universe ──────────────────────────────────────────────────────────
ALL_KNOWN_PAIRS = [
    "GBP/JPY", "USD/JPY", "USD/CHF", "GBP/CHF", "USD/CAD",
    "EUR/USD", "GBP/USD", "NZD/USD", "GBP/NZD", "EUR/GBP",
    "AUD/USD", "NZD/JPY", "EUR/CAD", "EUR/AUD", "AUD/JPY",
    "EUR/JPY", "GBP/AUD", "GBP/CAD", "AUD/CAD", "AUD/NZD",
    "EUR/NZD", "EUR/CHF", "NZD/CAD",
]

# ── Named presets ──────────────────────────────────────────────────────────
WHITELIST_PRESETS = {
    "alex":    ["GBP/JPY", "USD/JPY", "USD/CHF", "GBP/CHF", "USD/CAD", "EUR/USD", "GBP/USD"],
    "majors":  ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD", "AUD/USD", "NZD/USD"],
    "jpy":     ["GBP/JPY", "USD/JPY", "EUR/JPY", "AUD/JPY", "NZD/JPY"],
    "chf":     ["USD/CHF", "GBP/CHF", "EUR/CHF"],
    "wide":    ALL_KNOWN_PAIRS,
}
# Back-compat alias
ALEX_PAIRS = WHITELIST_PRESETS["alex"]

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


def get_last_decision_ts() -> Optional[str]:
    """Timestamp of the most recent entry in decision_log.jsonl."""
    if not DECISIONS_FILE.exists():
        return None
    try:
        with open(DECISIONS_FILE, "rb") as f:
            # Seek to end, scan backwards for last newline-terminated JSON line
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return None
            f.seek(max(0, size - 512))
            tail = f.read().decode(errors="ignore")
        for line in reversed(tail.strip().splitlines()):
            try:
                return json.loads(line).get("ts")
            except Exception:
                continue
    except Exception:
        pass
    return None


def get_engine_sha() -> str:
    """Current git commit SHA (short) for the trading-bot repo."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path.home() / "trading-bot"),
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"

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
    return render_template("dashboard.html", page="live")

@app.route("/backtests")
def page_backtests():
    return render_template("dashboard.html", page="backtests")

@app.route("/whitelist")
def page_whitelist():
    return render_template("dashboard.html", page="whitelist")

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
    recent_trades     = journal.get_recent_trades(15)
    scan_state        = get_scan_state()
    last_decision_ts  = get_last_decision_ts()
    engine_sha        = get_engine_sha()

    # Merge heartbeat + state for richer payload
    merged_hb = {**heartbeat, **{k: v for k, v in bot_state.items()
                                 if k not in heartbeat or bot_state.get(k) is not None}}

    # Risk / mode enrichment from bot_state stats block
    bot_stats         = bot_state.get("stats", {})
    risk_pct          = merged_hb.get("risk_pct") or bot_stats.get("final_risk_pct", 0)
    mode              = merged_hb.get("mode") or bot_stats.get("mode", "unknown")
    tier_label        = bot_stats.get("tier", "—")
    peak_bal          = bot_stats.get("peak_balance", 0)
    regroup_ends      = bot_stats.get("regroup_ends")
    pattern_ct        = len(bot_stats.get("traded_pattern_keys", []))
    dry_run           = merged_hb.get("dry_run", True)
    halted            = merged_hb.get("halted", False)
    # Risk decomposition — all caps
    base_risk_pct      = bot_stats.get("base_risk_pct",      merged_hb.get("base_risk_pct", 0))
    final_risk_pct     = bot_stats.get("final_risk_pct",     merged_hb.get("final_risk_pct", risk_pct))
    dd_flag            = bot_stats.get("dd_flag",            merged_hb.get("dd_flag", ""))
    active_cap_label   = bot_stats.get("active_cap_label",   merged_hb.get("active_cap_label", ""))
    final_risk_dollars = bot_stats.get("final_risk_dollars", merged_hb.get("final_risk_dollars", 0))
    consecutive_losses = bot_stats.get("consecutive_losses", merged_hb.get("consecutive_losses", 0))
    drawdown_pct_saved = bot_stats.get("drawdown_pct",       merged_hb.get("drawdown_pct", 0))
    paused             = bot_stats.get("paused",             merged_hb.get("paused", False)) or (mode == "paused")
    paused_since       = bot_stats.get("paused_since",       merged_hb.get("paused_since"))
    peak_source        = bot_stats.get("peak_source",        merged_hb.get("peak_source", "broker"))
    session_allowed    = bot_stats.get("session_allowed",    merged_hb.get("session_allowed", True))
    session_reason     = bot_stats.get("session_reason",     merged_hb.get("session_reason", ""))
    next_session       = bot_stats.get("next_session",       merged_hb.get("next_session", "London"))
    next_session_mins  = bot_stats.get("next_session_mins",  merged_hb.get("next_session_mins", 0))

    # Win rate from stats
    wr      = stats.get("win_rate", 0) if isinstance(stats, dict) else 0
    tot_tr  = stats.get("trades", 0)   if isinstance(stats, dict) else 0
    tot_pnl = stats.get("total_pnl", 0) if isinstance(stats, dict) else 0

    # ── Equity / NAV: AccountState is authoritative, broker dict is secondary ──
    # For LIVE_PAPER the broker dict has $0 (unfunded demo account) — never use it.
    # For LIVE_REAL with unknown equity the bot_state carries None explicitly.
    account_mode_val = bot_stats.get("account_mode", merged_hb.get("account_mode", "live_paper"))
    is_live_paper    = account_mode_val == "live_paper"

    # Prefer bot_state equity (from AccountState); fall back to broker only for LIVE_REAL
    bot_equity  = bot_stats.get("account_equity",   merged_hb.get("account_equity"))
    broker_bal  = (account.get("balance") if isinstance(account, dict) and not is_live_paper else None)
    bal         = bot_equity if bot_equity is not None else broker_bal   # None = UNKNOWN

    # For drawdown we need a float; use safe fallback but mark UNKNOWN separately
    bal_for_dd  = bal if bal is not None else 0.0
    dd_pct      = round((peak_bal - bal_for_dd) / peak_bal * 100, 1) if peak_bal > 0 and bal_for_dd < peak_bal else (drawdown_pct_saved or 0.0)

    # NAV: from bot_state (mode-aware, computed in orchestrator); never from broker for paper
    nav_val        = bot_stats.get("nav", merged_hb.get("nav"))
    unrealized_val = bot_stats.get("unrealized_pnl", merged_hb.get("unrealized_pnl", 0.0))

    # Confluence state: dict keyed by pair, sorted by decision priority
    raw_confluence = bot_state.get("confluence_state", {})
    confluence_sorted = sorted(
        raw_confluence.values(),
        key=lambda x: (
            0 if x.get("decision") == "ENTER" else
            1 if x.get("decision") == "WAIT"  else 2,
            -x.get("confidence", 0),
        )
    )

    # ── Control-plane derived fields ──────────────────────────────────────
    _ctrl_fields        = _load_control_state()   # pause_new_entries, risk_mode(pin), pause_expiry_ts, etc.
    _pne                = _ctrl_fields.get("pause_new_entries", False)
    _expiry_ts          = _ctrl_fields.get("pause_expiry_ts")
    _chop_active        = False
    if _expiry_ts:
        try:
            _chop_active = datetime.now(timezone.utc) < datetime.fromisoformat(_expiry_ts)
        except Exception:
            pass
    _bot_mode_paused    = (mode == "paused")
    _effective_paused   = _bot_mode_paused or _pne or _chop_active

    # Drift: the two planes disagree — one says stop, the other says go
    if not _bot_mode_paused and _pne:
        _control_drift = "ACTIVE_BUT_PNE_SET"    # BotMode running but soft gate still blocking
    elif _bot_mode_paused and not _pne:
        _control_drift = "PAUSED_BUT_PNE_CLEAR"  # BotMode stopped but soft gate is open
    else:
        _control_drift = None

    return jsonify({
        "heartbeat":          merged_hb,
        "account":            account,
        "open_trades":        open_trades,
        "stats":              stats,
        "recent_trades":      recent_trades,
        "scan_state":         scan_state,
        "recent_decisions":   recent_decs,
        "top_setups":         bot_state.get("top_setups", []),
        "confluence":         confluence_sorted,
        "server_time":        datetime.now(timezone.utc).isoformat(),
        "last_scan_ts":       bot_state.get("last_scan_time") or merged_hb.get("timestamp"),
        "last_decision_ts":   last_decision_ts,
        "last_state_write_ts": bot_state.get("timestamp"),
        "engine_sha":         engine_sha,
        # enriched fields for dashboard display
        "mode":               mode,
        "dry_run":            dry_run,
        "halted":             halted,
        "paused":             paused,
        "paused_since":       paused_since,
        "peak_source":        peak_source,
        "session_allowed":    session_allowed,
        "session_reason":     session_reason,
        "next_session":       next_session,
        "next_session_mins":  next_session_mins,
        # risk decomposition
        "risk_pct":           final_risk_pct,
        "base_risk_pct":      base_risk_pct,
        "final_risk_pct":     final_risk_pct,
        "dd_flag":            dd_flag,
        "active_cap_label":   active_cap_label,
        "final_risk_dollars": final_risk_dollars,
        "consecutive_losses": consecutive_losses,
        # equity / drawdown
        "tier_label":         tier_label,
        "peak_balance":       peak_bal,
        "drawdown_pct":       dd_pct,
        "regroup_ends":       regroup_ends,
        "pattern_memory_count": pattern_ct,
        "account_balance":    bal,          # None = UNKNOWN (never 0 for live_paper)
        "nav":                nav_val,      # None = UNKNOWN; equity+unrealized for paper
        "unrealized_pnl":     unrealized_val,
        "win_rate":           round(wr * 100, 1),
        "total_trades":       tot_tr,
        "total_pnl":          tot_pnl,
        # whitelist — both scopes surfaced for dashboard
        **{f"whitelist_live_{k}": v
           for k, v in _load_whitelist_state("live").items()
           if k in ("enabled", "pairs", "updated_at")},
        **{f"whitelist_backtest_{k}": v
           for k, v in _load_whitelist_state("backtest").items()
           if k in ("enabled", "pairs", "updated_at")},
        # strategy config — surfaced so UI can show active mode without reading files
        "entry_trigger_mode":           ENTRY_TRIGGER_MODE,
        "max_concurrent_live":          MAX_CONCURRENT_TRADES_LIVE,
        "max_concurrent_backtest":      MAX_CONCURRENT_TRADES_BACKTEST,
        "regime_score":                 bot_stats.get("regime_score"),  # dict or None
        # ── Control plane ─────────────────────────────────────────────────
        **_ctrl_fields,          # pause_new_entries, risk_mode (pin), last_updated, reason, pause_expiry_ts
        "bot_mode":             mode,          # raw BotMode string: active/paused/regroup/sleeping
        "effective_paused":     _effective_paused,
        "control_drift":        _control_drift,
        # ── Regime / risk mode ────────────────────────────────────────────
        "risk_mode":                    bot_stats.get("risk_mode"),
        "risk_mode_mult":               bot_stats.get("risk_mode_mult"),
        "regime_weekly_caps":           bot_stats.get("regime_weekly_caps"),
        # ── Risk tier index (for status panel) ───────────────────────────
        "tier_label":                   tier_label,
        # ── Account mode + equity source ─────────────────────────────────
        "account_mode":                 bot_stats.get("account_mode",           merged_hb.get("account_mode", "live_paper")),
        "account_mode_label":           bot_stats.get("account_mode_label",     merged_hb.get("account_mode_label", "LIVE PAPER")),
        "equity_source":                bot_stats.get("account_equity_source",  merged_hb.get("equity_source", "SIM")),
        "equity_unknown":               bot_stats.get("account_is_unknown",     merged_hb.get("equity_unknown", False)),
        "equity_display":               bot_stats.get("account_equity_display", merged_hb.get("equity_display", "—")),
        "realized_session_pnl":         bot_stats.get("account_realized_session_pnl", 0.0),
        "broker_fetch_failures":        bot_stats.get("account_broker_fetch_failures", 0),
    })

_SENTINEL = object()   # used as default sentinel in _write_control_state

def _load_control_state() -> dict:
    """Read runtime_state/control.json; return safe default if missing/corrupt."""
    default = {"pause_new_entries": False, "risk_mode": None,
               "last_updated": None, "updated_by": "system", "reason": "",
               "pause_expiry_ts": None}
    if not RUNTIME_CONTROL_FILE.exists():
        return default
    try:
        return {**default, **json.loads(RUNTIME_CONTROL_FILE.read_text())}
    except Exception:
        return default


def _write_control_state(pause: bool, reason: str, updated_by: str = "dashboard",
                          risk_mode: str | None = _SENTINEL,
                          pause_expiry_ts: str | None = _SENTINEL) -> dict:
    """Write runtime_state/control.json atomically; return new state.
    risk_mode=_SENTINEL (default) preserves existing risk_mode from disk.
    risk_mode=None clears the pin (AUTO).  risk_mode="HIGH" pins the mode.
    pause_expiry_ts=_SENTINEL preserves existing value.  None clears it.
    """
    import os, tempfile
    # Preserve existing fields unless explicitly passed
    existing = _load_control_state()
    if risk_mode is _SENTINEL:
        risk_mode = existing.get("risk_mode")   # type: ignore[assignment]
    _VALID_RISK_MODES = {"LOW", "MEDIUM", "HIGH", "EXTREME"}
    if risk_mode not in _VALID_RISK_MODES:
        risk_mode = None
    if pause_expiry_ts is _SENTINEL:
        pause_expiry_ts = existing.get("pause_expiry_ts")  # type: ignore[assignment]
    state = {
        "pause_new_entries": pause,
        "risk_mode":         risk_mode,
        "last_updated":      datetime.now(timezone.utc).isoformat(),
        "updated_by":        updated_by,
        "reason":            reason,
        "pause_expiry_ts":   pause_expiry_ts,
    }
    RUNTIME_STATE_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=RUNTIME_STATE_DIR, suffix=".tmp", prefix="control_")
    with os.fdopen(fd, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, RUNTIME_CONTROL_FILE)
    return state


def _write_control_audit(command: str, reason: str):
    """Append a pause/resume audit entry to decision_log.jsonl."""
    entry = json.dumps({
        "ts":      datetime.now(timezone.utc).isoformat(),
        "event":   f"CONTROL_{command.upper()}",
        "pair":    "ALL",
        "source":  "dashboard",
        "command": command,
        "reason":  reason,
    })
    try:
        with open(DECISIONS_FILE, "a") as f:
            f.write(entry + "\n")
    except Exception:
        pass


def _do_pause(reason: str, updated_by: str = "dashboard") -> dict:
    """
    Authoritative pause: sets BotMode=PAUSED (via bot_control.json one-shot)
    AND sets pause_new_entries=True in runtime_state/control.json.
    Both planes are always updated together to prevent drift.
    """
    _write_control_audit("pause", reason)
    CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_FILE.write_text(json.dumps({"command": "pause", "reason": reason}))
    state = _write_control_state(pause=True, reason=reason, updated_by=updated_by)
    return state


def _do_resume(reason: str, updated_by: str = "dashboard") -> dict:
    """
    Authoritative resume: sets BotMode=ACTIVE (via bot_control.json one-shot)
    AND sets pause_new_entries=False AND clears pause_expiry_ts in control.json.
    Both planes are always updated together to prevent drift.
    """
    _write_control_audit("resume", reason)
    CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_FILE.write_text(json.dumps({"command": "resume", "reason": reason}))
    # Clear pause_expiry_ts (chop-shield expiry) on any manual resume
    state = _write_control_state(pause=False, reason=reason, updated_by=updated_by,
                                  pause_expiry_ts=None)
    return state


@app.route("/api/pause", methods=["POST"])
def api_pause():
    """
    Authoritative pause — sets BotMode=PAUSED AND pause_new_entries=True.
    Orchestrator picks up BotMode change on the next tick (~1 min).
    """
    data   = request.get_json() or {}
    reason = data.get("reason", "Dashboard pause request")
    try:
        state = _do_pause(reason)
        return jsonify({"status": "ok", "command": "pause", "reason": reason,
                        "pause_new_entries": True, "last_updated": state["last_updated"]})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/resume", methods=["POST"])
def api_resume():
    """
    Authoritative resume — sets BotMode=ACTIVE AND pause_new_entries=False
    AND clears pause_expiry_ts.
    Orchestrator picks up BotMode change on the next tick (~1 min).
    """
    data   = request.get_json() or {}
    reason = data.get("reason", "Dashboard resume request")
    try:
        state = _do_resume(reason)
        return jsonify({"status": "ok", "command": "resume", "reason": reason,
                        "pause_new_entries": False, "last_updated": state["last_updated"]})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/pause_entries", methods=["POST"])
def api_pause_entries():
    """
    DEPRECATED — now delegates to /api/pause (authoritative).
    Kept for backward compatibility; callers should migrate to /api/pause.
    """
    data   = request.get_json() or {}
    reason = data.get("reason", "Dashboard pause request")
    try:
        state = _do_pause(reason)
        return jsonify({"status": "ok", "pause_new_entries": True, "reason": reason,
                        "last_updated": state["last_updated"],
                        "deprecated": "use /api/pause instead"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/resume_entries", methods=["POST"])
def api_resume_entries():
    """
    DEPRECATED — now delegates to /api/resume (authoritative).
    Kept for backward compatibility; callers should migrate to /api/resume.
    """
    data   = request.get_json() or {}
    reason = data.get("reason", "Dashboard resume")
    try:
        state = _do_resume(reason)
        return jsonify({"status": "ok", "pause_new_entries": False, "reason": reason,
                        "last_updated": state["last_updated"],
                        "deprecated": "use /api/resume instead"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/control_state", methods=["GET"])
def api_control_state():
    """Return current persistent control state (pause_new_entries, risk_mode, etc.)."""
    ctrl = _load_control_state()
    # Merge in live risk_mode data from bot_state
    try:
        bot_state = load_bot_state()
        bot_stats = bot_state.get("stats", {})
        ctrl["risk_mode_effective"] = bot_stats.get("risk_mode")
        ctrl["risk_mode_source"]    = bot_stats.get("risk_mode_source", "AUTO")
        ctrl["risk_mode_mult"]      = bot_stats.get("risk_mode_mult")
        ctrl["effective_risk_pct"]  = bot_stats.get("effective_risk_pct")
        ctrl["regime_weekly_caps"]  = bot_stats.get("regime_weekly_caps")
        ctrl["regime_score"]        = bot_stats.get("regime_score")
    except Exception:
        pass
    return jsonify(ctrl)


@app.route("/api/set_risk_mode", methods=["POST"])
def api_set_risk_mode():
    """
    Pin or clear the risk mode.

    Body: {"mode": "LOW"|"MEDIUM"|"HIGH"|"EXTREME"|null}
      null / missing → clear pin → AUTO (dynamic computation)

    The orchestrator reads control.json on every cycle and calls
    risk.set_regime_mode() with the pinned value (or dynamic value if null).
    """
    _VALID = {"LOW", "MEDIUM", "HIGH", "EXTREME"}
    data   = request.get_json() or {}
    mode   = data.get("mode")   # None = clear pin

    if mode is not None and mode not in _VALID:
        return jsonify({
            "status": "error",
            "error":  f"Invalid mode {mode!r}. Valid: {sorted(_VALID)} or null for AUTO"
        }), 400

    try:
        existing = _load_control_state()
        import os, tempfile
        state = {
            "pause_new_entries": existing.get("pause_new_entries", False),
            "risk_mode":         mode,   # None clears pin
            "last_updated":      datetime.now(timezone.utc).isoformat(),
            "updated_by":        "dashboard",
            "reason":            existing.get("reason", ""),
        }
        RUNTIME_STATE_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=RUNTIME_STATE_DIR, suffix=".tmp", prefix="control_")
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, RUNTIME_CONTROL_FILE)

        label = mode if mode else "AUTO (dynamic)"
        _write_control_audit("set_risk_mode", f"risk_mode={label}")
        return jsonify({"status": "ok", "risk_mode": mode,
                        "label": label, "last_updated": state["last_updated"]})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


def _whitelist_file(scope: str) -> Path:
    """Return the correct whitelist file for scope='live'|'backtest'."""
    return WHITELIST_BACKTEST_FILE if scope == "backtest" else WHITELIST_LIVE_FILE


def _load_whitelist_state(scope: str = "live") -> dict:
    """Read scoped whitelist file; return safe default if missing/corrupt."""
    wl_file = _whitelist_file(scope)
    default_pairs = list(WHITELIST_PRESETS["alex"])
    if wl_file.exists():
        try:
            data = json.loads(wl_file.read_text())
            return {
                "scope":      scope,
                "enabled":    bool(data.get("enabled", False)),
                "pairs":      list(data.get("pairs", default_pairs)),
                "updated_at": data.get("updated_at"),
                "updated_by": data.get("updated_by", "unknown"),
            }
        except Exception:
            pass
    return {
        "scope":      scope,
        "enabled":    False,
        "pairs":      default_pairs,
        "updated_at": None,
        "updated_by": None,
    }


def _write_whitelist_audit(enabled: bool, pairs: list, scope: str, reason: str = "UI update"):
    """Append a WHITELIST_UPDATE entry to decision_log.jsonl."""
    entry = {
        "ts":      datetime.now(timezone.utc).isoformat(),
        "event":   "WHITELIST_UPDATE",
        "pair":    "ALL",
        "source":  "dashboard",
        "scope":   scope,
        "enabled": enabled,
        "pairs":   pairs,
        "reason":  reason,
    }
    try:
        DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DECISIONS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        import logging
        logging.getLogger("app").warning(f"Could not write whitelist audit: {e}")


@app.route("/api/whitelist", methods=["GET"])
def api_whitelist_get():
    """
    Return whitelist state for a scope.
    ?scope=live (default) | backtest
    Also returns all pair groups and presets for the UI.
    """
    scope = request.args.get("scope", "live")
    state = _load_whitelist_state(scope)
    state["all_known_pairs"] = ALL_KNOWN_PAIRS
    state["presets"]         = WHITELIST_PRESETS
    return jsonify(state)


@app.route("/api/whitelist", methods=["POST"])
def api_whitelist_post():
    """
    Update whitelist for a scope.
    Query:  ?scope=live (default) | backtest
    Body:   {"enabled": bool, "pairs": [...], "reason": "..."}
    - Writes to whitelist_{scope}.json
    - Appends WHITELIST_UPDATE to decision_log.jsonl with scope field
    """
    scope   = request.args.get("scope", "live")
    data    = request.get_json() or {}
    enabled = bool(data.get("enabled", False))
    pairs   = [str(p) for p in data.get("pairs", WHITELIST_PRESETS["alex"])]
    reason  = data.get("reason", "Dashboard update")

    wl_file = _whitelist_file(scope)
    payload = {
        "scope":      scope,
        "enabled":    enabled,
        "pairs":      pairs,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": "dashboard",
        "reason":     reason,
    }
    try:
        wl_file.parent.mkdir(parents=True, exist_ok=True)
        wl_file.write_text(json.dumps(payload, indent=2))
        _write_whitelist_audit(enabled, pairs, scope, reason)
        return jsonify({
            "status":  "ok",
            "scope":   scope,
            "enabled": enabled,
            "pairs":   pairs,
            "message": f"[{scope.upper()}] Whitelist {'ENABLED' if enabled else 'DISABLED'} — "
                       f"{len(pairs)} pairs active.",
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


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

RESULTS_LOG  = LOG_DIR / "backtest_results.jsonl"
VENV_PYTHON  = Path.home() / "trading-bot" / "venv" / "bin" / "python"
BOT_ROOT     = Path.home() / "trading-bot"

@app.route("/api/backtest_results")
def api_backtest_results():
    if not RESULTS_LOG.exists():
        return jsonify([])
    records = []
    try:
        with open(RESULTS_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        return jsonify({"error": str(e)})
    records.reverse()   # newest first
    return jsonify(records)


_backtest_proc = None   # track last background backtest

@app.route("/api/run_backtest", methods=["POST"])
def run_backtest_api():
    global _backtest_proc
    data  = request.get_json() or {}
    start = data.get("start", "2024-07-01")
    end   = data.get("end",   "2024-10-31")
    notes = data.get("notes", "")

    # Don't spawn if one is already running
    if _backtest_proc and _backtest_proc.poll() is None:
        return jsonify({"status": "already_running", "pid": _backtest_proc.pid})

    cmd = [
        str(VENV_PYTHON), "-m", "backtesting.oanda_backtest_v2",
        "--start", start, "--end", end,
    ]
    if notes:
        cmd += ["--notes", notes]

    _backtest_proc = subprocess.Popen(
        cmd,
        cwd=str(BOT_ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(BOT_ROOT)},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return jsonify({"status": "started", "pid": _backtest_proc.pid})


@app.route("/api/backtest_status")
def backtest_status():
    global _backtest_proc
    if _backtest_proc is None:
        return jsonify({"running": False, "pid": None, "exit_code": None})
    code = _backtest_proc.poll()
    return jsonify({
        "running":   code is None,
        "pid":       _backtest_proc.pid,
        "exit_code": code,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
