"""
Bot Command Relay — Used by Forge to relay Mike's Telegram commands to the running bot.

The bot checks ~/trading-bot/logs/bot_control.json every 60s.
Forge writes a command here; the bot picks it up next tick and clears the file.

Usage (from Python / Forge agent):
    from scripts.bot_command import relay_command
    relay_command("pause",   reason="Mike requested pause via Telegram")
    relay_command("resume",  reason="Mike said resume trading")
    relay_command("extend_cooldown", days=7)

Supported commands:
    pause            — Enter indefinite PAUSED mode (Mike must resume)
    resume           — Resume trading (works from REGROUP or PAUSED)
    extend_cooldown  — Add N more days to regroup cooldown
    status           — (no-op here; read bot_state.json for status)
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

CONTROL_FILE = Path.home() / "trading-bot" / "logs" / "bot_control.json"
STATE_FILE   = Path.home() / "trading-bot" / "logs" / "bot_state.json"


def relay_command(command: str, reason: str = "Relayed from Forge", **kwargs):
    """Write a command to the bot control file."""
    payload = {
        "command":    command,
        "reason":     reason,
        "issued_at":  datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_FILE.write_text(json.dumps(payload, indent=2))
    print(f"✅ Command written: {command} — bot will pick it up within 60s")
    return payload


def get_bot_status() -> dict:
    """Read current bot state from state file."""
    if not STATE_FILE.exists():
        return {"error": "Bot state file not found — bot may not be running"}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception as e:
        return {"error": str(e)}


def status_summary() -> str:
    """Return a human-readable status string."""
    s = get_bot_status()
    if "error" in s:
        return f"⚠️ {s['error']}"

    mode    = s.get("mode", "?")
    bal     = s.get("account_balance", 0)
    dry_run = s.get("dry_run", True)
    halted  = s.get("halted", False)
    pos     = s.get("open_positions", {})
    stats   = s.get("stats", {})
    tier    = stats.get("tier", "?")
    peak    = stats.get("peak_balance", 0)
    regroup = stats.get("regroup_ends")

    mode_emoji = {"active": "🟢", "regroup": "🟡", "paused": "⏸"}.get(mode, "❓")
    lines = [
        f"{mode_emoji} Mode: {mode.upper()}" + (f" (dry run)" if dry_run else ""),
        f"Balance: ${bal:,.2f}" + (f" | Peak: ${peak:,.2f}" if peak else ""),
        f"Risk tier: {tier}",
    ]
    if pos:
        lines.append(f"Open: {', '.join(pos.keys())}")
    else:
        lines.append("No open positions")
    if regroup:
        lines.append(f"Regroup ends: {regroup}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(status_summary())
        sys.exit(0)

    cmd = sys.argv[1].lower()
    reason = sys.argv[2] if len(sys.argv) > 2 else "CLI command"

    if cmd == "status":
        print(status_summary())
    elif cmd in ("pause", "resume"):
        relay_command(cmd, reason=reason)
    elif cmd == "extend":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        relay_command("extend_cooldown", days=days, reason=f"Extended by {days} days")
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python bot_command.py [status|pause|resume|extend <days>]")
