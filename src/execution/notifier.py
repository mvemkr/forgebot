"""
Notifier — Sends alerts and summaries to Mike via Telegram.

Configure in ~/trading-bot/.env:
  TELEGRAM_BOT_TOKEN=<bot token from @BotFather>
  TELEGRAM_CHAT_ID=8265912344

Message types:
  send()               — raw message
  send_trade_entry()   — new position opened (overnight or daytime)
  send_standings()     — regular P&L standings update
  send_regroup_alert() — kill switch regroup entered
  send_regroup_resume()— cooldown over, trading resumed
  send_daily_brief()   — morning summary
  send_weekly_brief()  — Sunday scan results
  send_exit_signal()   — exit signal detected, Mike decides
"""
import os
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv(Path.home() / "trading-bot" / ".env")

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class Notifier:
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id:   Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id   = chat_id   or os.getenv("TELEGRAM_CHAT_ID", "8265912344")
        self._ok       = bool(self.bot_token and self.chat_id)

        if self._ok:
            logger.info(f"Notifier: Telegram ready (chat={self.chat_id})")
        else:
            logger.warning(
                "Notifier: TELEGRAM_BOT_TOKEN not set — alerts will only log to console."
            )

    # ── Core Send ─────────────────────────────────────────────────────

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        logger.info(f"[ALERT] {message}")
        if not self._ok:
            return False
        try:
            resp = requests.post(
                f"{TELEGRAM_API}/bot{self.bot_token}/sendMessage",
                json={
                    "chat_id":    self.chat_id,
                    "text":       message,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return True
            logger.error(f"Telegram send failed: {resp.status_code} {resp.text[:200]}")
            return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    # ── Trade Notifications ───────────────────────────────────────────

    def send_trade_entry(
        self,
        pair:         str,
        direction:    str,
        entry_price:  float,
        stop_price:   float,
        risk_amount:  float,
        risk_pct:     float,
        pattern:      str,
        key_level:    float,
        overnight:    bool = False,
        dry_run:      bool = False,
    ):
        """Fires the moment a trade is placed — especially important for overnight entries."""
        arrow   = "⬆️ LONG" if direction == "long" else "⬇️ SHORT"
        pips    = abs(entry_price - stop_price) * (100 if "JPY" in pair else 10_000)
        mode    = "🌙 OVERNIGHT AUTO-EXECUTE" if overnight else "🎯 TRADE ENTERED"
        dry_tag = " [DRY RUN]" if dry_run else ""

        self.send(
            f"<b>{mode}{dry_tag}</b>\n\n"
            f"  Pair:    <b>{pair}</b>  {arrow}\n"
            f"  Entry:   {entry_price:.5f}\n"
            f"  Stop:    {stop_price:.5f}  ({pips:.1f} pips)\n"
            f"  Pattern: {pattern} @ {key_level:.5f}\n"
            f"  Risk:    ${risk_amount:.2f} ({risk_pct:.0f}% of account)\n"
            f"  No TP — set and forget 🔒\n\n"
            + (
                "☀️ <i>You were asleep — this trade auto-executed per your London "
                "session pre-approval. Check it when you wake up.</i>"
                if overnight else
                "Monitoring position. Will alert on breakeven move or exit signal."
            )
        )

    def send_exit_signal(
        self,
        pair:        str,
        direction:   str,
        entry_price: float,
        current_price: float,
        signal_desc: str,
        pnl_r:       float,
        pnl_dollars: float,
    ):
        """Exit signal detected — Mike decides whether to close."""
        arrow   = "⬆️" if direction == "long" else "⬇️"
        pnl_tag = f"+{pnl_r:.1f}R (+${pnl_dollars:,.0f})" if pnl_dollars >= 0 else f"{pnl_r:.1f}R (-${abs(pnl_dollars):,.0f})"

        self.send(
            f"<b>⚡ EXIT SIGNAL — {pair} {arrow}</b>\n\n"
            f"  Entry: {entry_price:.5f} → Now: {current_price:.5f}\n"
            f"  P&amp;L: <b>{pnl_tag}</b>\n"
            f"  Signal: {signal_desc}\n\n"
            f"<b>Bot will NOT auto-close.</b> You decide:\n"
            f"  ✅ Close now and bank it\n"
            f"  🔒 Hold — let it run further\n"
            f"  👁 Watch another candle"
        )

    def send_breakeven_moved(self, pair: str, direction: str, entry: float, new_stop: float):
        arrow = "⬆️" if direction == "long" else "⬇️"
        self.send(
            f"<b>🛡️ STOP → BREAKEVEN  {pair} {arrow}</b>\n"
            f"  Stop moved to entry: {new_stop:.5f}\n"
            f"  Trade is now risk-free. Let it run. 🔒"
        )

    # ── Standings ─────────────────────────────────────────────────────

    def send_standings(
        self,
        account_balance:  float,
        nav,              # float | None — None shown as N/A (broker unknown)
        unrealized_pnl:   float,
        weekly_pnl:       float,
        peak_balance:     float,
        risk_pct:         float,
        tier_label:       str,
        open_positions:   dict,
        trades_this_week: int,
        wins_this_week:   int,
        losses_this_week: int,
        mode:             str = "active",
        regroup_ends:     Optional[datetime] = None,
    ):
        """
        Regular standings update — send every 6 hours so Mike always
        knows where things stand, even if trades happened while he slept.
        """
        drawdown_from_peak = ((peak_balance - account_balance) / peak_balance * 100) if peak_balance > 0 else 0
        now_et = datetime.now(timezone.utc)  # caller can format for ET if needed

        # Mode indicator
        if mode == "regroup":
            mode_tag = f"🟡 REGROUP MODE"
            ends_tag = f" (resumes {regroup_ends.strftime('%b %d') if regroup_ends else '?'})"
        elif mode == "paused":
            mode_tag = "⏸ PAUSED"
            ends_tag = ""
        else:
            mode_tag = "🟢 ACTIVE"
            ends_tag = ""

        # Open positions block
        if open_positions:
            pos_lines = []
            for pair, pos in open_positions.items():
                be = " 🛡️BE" if pos.get("at_breakeven") else ""
                pos_lines.append(
                    f"    • {pair} {pos.get('direction','?').upper()} "
                    f"@ {pos.get('entry', 0):.5f}{be}"
                )
            pos_block = "\n".join(pos_lines)
        else:
            pos_block = "    None"

        weekly_tag  = f"+${weekly_pnl:,.0f}" if weekly_pnl >= 0 else f"-${abs(weekly_pnl):,.0f}"
        dd_tag      = f"{drawdown_from_peak:.1f}% from peak" if drawdown_from_peak > 1 else "✅ Near peak"

        self.send(
            f"<b>📊 Standings Update</b>\n\n"
            f"  Mode:      {mode_tag}{ends_tag}\n"
            f"  Balance:   <b>${account_balance:,.2f}</b>\n"
            f"  NAV:       {('$'+f'{nav:,.2f}') if nav is not None else 'N/A'}  (unrealized: ${unrealized_pnl:+,.2f})\n"
            f"  This week: {weekly_tag}  |  {wins_this_week}W / {losses_this_week}L  "
            f"({trades_this_week} trade{'s' if trades_this_week != 1 else ''})\n"
            f"  Drawdown:  {dd_tag}\n"
            f"  Risk/trade: {risk_pct:.0f}%  ({tier_label})\n\n"
            f"  <b>Open positions:</b>\n{pos_block}"
        )

    # ── Regroup Alerts ────────────────────────────────────────────────

    def send_regroup_alert(
        self,
        reason:       str,
        balance:      float,
        cooldown_days: int,
        resume_date:  datetime,
    ):
        self.send(
            f"<b>🟡 REGROUP MODE — Taking a step back</b>\n\n"
            f"  Triggered: {reason}\n"
            f"  Balance: ${balance:,.2f}\n"
            f"  Cooldown: {cooldown_days} days\n"
            f"  Auto-resume: {resume_date.strftime('%b %d, %Y')}\n\n"
            f"<b>What's happening:</b>\n"
            f"  • No new trades will be opened\n"
            f"  • Existing positions still monitored\n"
            f"  • Bot will observe + analyze daily\n"
            f"  • Coherence rebuild in progress 🔄\n\n"
            f"This is how Alex handled drawdowns — step back, study "
            f"the market, wait for a clear A+ setup before re-engaging.\n\n"
            f"<i>To resume early: tell Forge 'resume trading'</i>"
        )

    def send_regroup_resume(self, balance: float, days_in_regroup: int):
        self.send(
            f"<b>✅ Regroup Complete — Resuming Live Trading</b>\n\n"
            f"  Days in cooldown: {days_in_regroup}\n"
            f"  Current balance: ${balance:,.2f}\n\n"
            f"Back to scanning. Next trade only when an A+ setup appears. 🎯"
        )

    def send_regroup_observation(self, observations: List[str], days_remaining: int):
        """Daily during regroup — bot shares what it's seeing in the market."""
        lines = [
            f"<b>🔍 Regroup — Daily Observation  ({days_remaining}d remaining)</b>\n"
        ]
        for obs in observations:
            lines.append(f"  • {obs}")
        lines.append("\n<i>Watching. Not trading. Building coherence.</i>")
        self.send("\n".join(lines))

    # ── Weekly & Daily Briefs ─────────────────────────────────────────

    def send_weekly_brief(self, setups: list, account_balance: float, risk_pct: float):
        lines = [
            f"<b>📊 Weekly Forex Setup Brief</b>",
            f"Account: ${account_balance:,.2f} | Risk/trade: {risk_pct:.0f}%\n",
        ]
        prime    = [s for s in setups if s.status.value == "PRIME"]
        watching = [s for s in setups if s.status.value == "WATCHING"]

        if prime:
            lines.append("🎯 <b>PRIME SETUPS — at level, watching for signal:</b>")
            for s in prime:
                arrow = "⬆️" if s.direction == "long" else "⬇️"
                lines.append(
                    f"  {arrow} {s.pair} — Level: {s.key_level:.5f} "
                    f"(score={s.key_level_score}) | {s.notes}"
                )
            lines.append("")

        if watching:
            lines.append("👁 <b>WATCHING — setup forming:</b>")
            for s in watching[:5]:
                arrow = "⬆️" if s.direction == "long" else "⬇️"
                lines.append(f"  {arrow} {s.pair} → Alert @ {s.alert_price:.5f}")
            lines.append("")

        lines.append("Remember: price at level ≠ entry. Wait for the engulfing. 🕯️")
        self.send("\n".join(lines))

    def send_daily_brief(
        self,
        open_positions: dict,
        account_summary: dict,
        recent_signals:  list,
    ):
        balance    = account_summary.get("balance", 0)
        nav        = account_summary.get("nav", 0)
        unrealized = account_summary.get("unrealized_pnl", 0)

        lines = [
            f"<b>☀️ Daily Brief</b>",
            f"Balance: ${balance:,.2f} | NAV: ${nav:,.2f}",
            f"Unrealized: ${unrealized:+,.2f}\n",
        ]

        if open_positions:
            lines.append("<b>Open Positions:</b>")
            for pair, pos in open_positions.items():
                be = " 🛡️(BE)" if pos.get("at_breakeven") else ""
                lines.append(
                    f"  • {pair} {pos['direction'].upper()} "
                    f"@ {pos['entry']:.5f} | SL: {pos['stop']:.5f}{be}"
                )
        else:
            lines.append("No open positions.")

        if recent_signals:
            lines.append("\n<b>Recent Signals (24h):</b>")
            for sig in recent_signals[-3:]:
                lines.append(f"  • {sig}")

        self.send("\n".join(lines))
