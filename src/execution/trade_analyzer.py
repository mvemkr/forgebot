"""
Post-Trade Analyzer — Lessons Learned Engine

Fires after every trade closes (win or loss) and:
  1. Breaks down exactly what happened vs what was expected
  2. Extracts specific lessons tied to trade attributes
  3. Identifies patterns across the trade history (what's working, what isn't)
  4. Sends Mike a full debrief via Telegram
  5. Appends to lessons_learned.jsonl for long-term pattern tracking
  6. Generates periodic performance reports vs simulation baseline

The goal: win rate improves from 55% → 65%+ as the system learns which
setups are truly A+ and which are marginal. At 65% WR with $500K in the
account, $75-150K months become the median outcome.
"""
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LESSONS_FILE    = Path.home() / "trading-bot" / "logs" / "lessons_learned.jsonl"
ANALYTICS_FILE  = Path.home() / "trading-bot" / "logs" / "trade_analytics.json"


class TradeAnalyzer:
    """
    Post-trade learning engine.

    Usage (call from position_monitor or orchestrator when a trade closes):
        analyzer = TradeAnalyzer(notifier)
        analyzer.analyze_closed_trade(trade_record, all_trades_history)
    """

    def __init__(self, notifier=None):
        self.notifier = notifier
        LESSONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── Main Entry Point ───────────────────────────────────────────────

    def analyze_closed_trade(
        self,
        trade: Dict,
        trade_history: List[Dict],
    ):
        """
        Full post-trade analysis. Call every time a position closes.

        trade keys expected:
            pair, direction, entry_price, exit_price, stop_loss,
            entry_ts, exit_ts, exit_reason, pnl, rr,
            pattern_type, psych_level, key_level_score,
            signal_strength, session, risk_pct, risk_dollars, units,
            bars_held, confidence
        """
        outcome   = "WIN" if trade.get("pnl", 0) > 0 else "LOSS"
        rr        = trade.get("rr", 0)
        pnl       = trade.get("pnl", 0)
        pair      = trade.get("pair", "?")
        pattern   = trade.get("pattern_type", "?")
        direction = trade.get("direction", "?")
        reason    = trade.get("exit_reason", "?")
        sig_str   = trade.get("signal_strength", 0)
        lvl_score = trade.get("key_level_score", 0)
        session   = trade.get("session", "?")
        bars      = trade.get("bars_held", 0)
        risk_pct  = trade.get("risk_pct", 0)

        lessons   = self._extract_lessons(trade, trade_history)
        stats     = self._portfolio_stats(trade_history)
        edge_data = self._edge_breakdown(trade_history)

        # Save lesson to file
        self._save_lesson(trade, outcome, lessons)

        # Send Telegram debrief
        if self.notifier:
            self._send_debrief(trade, outcome, lessons, stats, edge_data)

        # Recompute and save analytics file
        self._update_analytics(trade_history)

        return lessons

    # ── Lesson Extraction ──────────────────────────────────────────────

    def _extract_lessons(self, trade: Dict, history: List[Dict]) -> List[str]:
        """
        Generate specific, actionable lessons from this trade.
        Compares this trade's attributes against historical performance.
        """
        lessons = []
        outcome = "WIN" if trade.get("pnl", 0) > 0 else "LOSS"
        rr      = trade.get("rr", 0)
        reason  = trade.get("exit_reason", "")
        sig_str = trade.get("signal_strength", 0)
        bars    = trade.get("bars_held", 0)
        pair    = trade.get("pair", "")
        pattern = trade.get("pattern_type", "")

        # --- Exit reason lessons ---
        if reason == "stop_hit":
            if sig_str < 0.50:
                lessons.append(
                    f"WEAK SIGNAL AT ENTRY — signal strength was {sig_str:.2f}. "
                    f"Stops tend to get hit when signal strength < 0.50. "
                    f"Consider raising MIN_SIGNAL_STRENGTH."
                )
            if bars < 24:
                lessons.append(
                    f"STOPPED OUT FAST ({bars}h held) — trade never got going. "
                    f"Check if entry was too early (before neckline fully confirmed)."
                )

        if reason in ("exit_signal", "exit_2d_reversal"):
            if rr < 2.0:
                lessons.append(
                    f"EXITED TOO EARLY — only {rr:.1f}R captured. "
                    f"Alex's avg winning trade is 5.5R+. "
                    f"Was the exit signal genuinely strong or just noise at a minor level?"
                )
            elif rr >= 5.0:
                lessons.append(
                    f"GREAT EXIT — {rr:.1f}R captured on exit signal. "
                    f"This is exactly the target profile. Signal threshold is well-calibrated."
                )

        if reason == "max_hold":
            lessons.append(
                f"MAX HOLD HIT — trade held 30 days without a clean exit signal. "
                f"Consider whether the pattern was valid or if this was a ranging market."
            )
            if trade.get("pnl", 0) > 0:
                lessons.append(
                    f"Profitable max-hold exit (+{rr:.1f}R). "
                    f"The trend ran without a reversal signal — correct to hold."
                )

        if reason == "open_at_end":
            lessons.append("Trade still open at analysis cutoff — no lessons yet on exit.")

        # --- Pattern lessons ---
        pair_pattern_trades = [
            t for t in history
            if t.get("pair") == pair and t.get("pattern_type") == pattern
            and t.get("exit_reason") not in ("open_at_end",)
        ]
        if len(pair_pattern_trades) >= 3:
            wr = sum(1 for t in pair_pattern_trades if t.get("pnl", 0) > 0) / len(pair_pattern_trades)
            if wr < 0.40:
                lessons.append(
                    f"LOW WIN RATE WARNING — {pattern} on {pair}: "
                    f"{wr:.0%} WR over {len(pair_pattern_trades)} trades. "
                    f"This combination may not be producing quality setups. Monitor closely."
                )
            elif wr >= 0.65:
                lessons.append(
                    f"HIGH-EDGE COMBO — {pattern} on {pair}: "
                    f"{wr:.0%} WR over {len(pair_pattern_trades)} trades. "
                    f"This is an A+ setup profile. Prioritize these."
                )

        # --- R:R lessons ---
        if outcome == "WIN" and rr >= 7.0:
            lessons.append(
                f"BIG WINNER +{rr:.1f}R — Analyse what made this setup exceptional: "
                f"pattern={pattern}, signal={trade.get('signal_strength',0):.2f}, "
                f"level_score={trade.get('key_level_score',0)}. Replicate these conditions."
            )

        if outcome == "WIN" and rr < 1.5:
            lessons.append(
                f"SMALL WIN — only +{rr:.1f}R. Risk was correct but exit may have fired "
                f"prematurely. Was this at a genuinely new major level or the same zone as entry?"
            )

        # --- Session lessons ---
        session = trade.get("session", "")
        session_trades = [t for t in history if t.get("session") == session
                          and t.get("exit_reason") not in ("open_at_end",)]
        if len(session_trades) >= 4:
            s_wr = sum(1 for t in session_trades if t.get("pnl",0) > 0) / len(session_trades)
            if outcome == "LOSS" and s_wr < 0.45:
                lessons.append(
                    f"SESSION WARNING — {session} session: {s_wr:.0%} WR across "
                    f"{len(session_trades)} trades. Consider tightening entry criteria "
                    f"for this session or weighting toward London open only."
                )

        # --- General good habits to reinforce ---
        if outcome == "WIN" and rr >= 3.0:
            if trade.get("key_level_score", 0) >= 3:
                lessons.append(
                    f"HIGH-SCORE LEVEL DELIVERED — level score {trade.get('key_level_score')} "
                    f"produced a +{rr:.1f}R winner. High confluence levels continue to be "
                    f"the best predictor of large moves."
                )

        if not lessons:
            if outcome == "WIN":
                lessons.append(f"Clean trade execution. +{rr:.1f}R. No adjustments needed.")
            else:
                lessons.append(
                    f"Stop hit. Part of the game at {trade.get('risk_pct',0):.0f}% risk. "
                    f"Review setup quality: was this an A+ signal or marginal?"
                )

        return lessons

    # ── Portfolio Stats ────────────────────────────────────────────────

    def _portfolio_stats(self, history: List[Dict]) -> Dict:
        closed = [t for t in history if t.get("exit_reason") not in ("open_at_end",)]
        if not closed:
            return {}

        wins   = [t for t in closed if t.get("pnl", 0) > 0]
        losses = [t for t in closed if t.get("pnl", 0) <= 0]
        rr_vals = [t.get("rr", 0) for t in closed]

        win_rr  = [t.get("rr", 0) for t in wins]
        loss_rr = [t.get("rr", 0) for t in losses]
        total_pnl = sum(t.get("pnl", 0) for t in closed)

        return {
            "total_trades":   len(closed),
            "wins":           len(wins),
            "losses":         len(losses),
            "win_rate":       len(wins) / len(closed) if closed else 0,
            "avg_win_rr":     statistics.mean(win_rr)  if win_rr  else 0,
            "avg_loss_rr":    statistics.mean(loss_rr) if loss_rr else 0,
            "avg_rr_all":     statistics.mean(rr_vals) if rr_vals else 0,
            "total_pnl":      total_pnl,
            "profit_factor":  (
                sum(t.get("pnl",0) for t in wins) /
                max(abs(sum(t.get("pnl",0) for t in losses)), 1)
            ),
            "best_trade_rr":  max(rr_vals) if rr_vals else 0,
            "worst_trade_rr": min(rr_vals) if rr_vals else 0,
            "consecutive_losses": self._max_consecutive_losses(closed),
        }

    def _max_consecutive_losses(self, trades: List[Dict]) -> int:
        max_streak = streak = 0
        for t in trades:
            if t.get("pnl", 0) <= 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    def _edge_breakdown(self, history: List[Dict]) -> Dict:
        """Break down performance by pair, pattern, session, level score."""
        closed = [t for t in history if t.get("exit_reason") not in ("open_at_end",)]
        if not closed:
            return {}

        def group_stats(key_fn):
            groups = {}
            for t in closed:
                k = key_fn(t)
                if k not in groups:
                    groups[k] = []
                groups[k].append(t)
            return {
                k: {
                    "trades": len(v),
                    "wr": sum(1 for t in v if t.get("pnl",0)>0)/len(v),
                    "avg_rr": statistics.mean([t.get("rr",0) for t in v]),
                    "pnl": sum(t.get("pnl",0) for t in v),
                }
                for k, v in groups.items() if len(v) >= 2
            }

        return {
            "by_pair":    group_stats(lambda t: t.get("pair", "?")),
            "by_pattern": group_stats(lambda t: t.get("pattern_type", "?")),
            "by_session": group_stats(lambda t: t.get("session", "?")),
        }

    # ── Telegram Debrief ──────────────────────────────────────────────

    def _send_debrief(
        self,
        trade: Dict,
        outcome: str,
        lessons: List[str],
        stats: Dict,
        edge_data: Dict,
    ):
        pair      = trade.get("pair", "?")
        direction = trade.get("direction", "?")
        rr        = trade.get("rr", 0)
        pnl       = trade.get("pnl", 0)
        reason    = trade.get("exit_reason", "?")
        bars      = trade.get("bars_held", 0)
        entry     = trade.get("entry_price", 0)
        exit_p    = trade.get("exit_price", 0)
        entry_ts  = str(trade.get("entry_ts",""))[:16]
        exit_ts   = str(trade.get("exit_ts",""))[:16]

        emoji = "✅" if outcome == "WIN" else "❌"
        arrow = "⬆️" if direction == "long" else "⬇️"
        days_held = bars // 24 if bars else 0

        lines = [
            f"<b>{emoji} TRADE CLOSED — {pair} {arrow}</b>\n",
            f"  Result:   <b>{outcome}  {rr:+.1f}R  (${pnl:+,.0f})</b>",
            f"  Entry:    {entry:.5f}  ({entry_ts})",
            f"  Exit:     {exit_p:.5f}  ({exit_ts})",
            f"  Held:     {days_held}d {bars%24}h  |  Reason: {reason}",
            f"  Pattern:  {trade.get('pattern_type','?')} @ {trade.get('psych_level',0):.5f}",
            f"  Signal:   strength={trade.get('signal_strength',0):.2f}  "
            f"level_score={trade.get('key_level_score',0)}",
        ]

        # Overall stats block
        if stats:
            wr_pct = stats['win_rate'] * 100
            lines.append(f"\n<b>📊 Running Stats ({stats['total_trades']} trades)</b>")
            lines.append(
                f"  WR: {wr_pct:.0f}%  |  Avg win: +{stats['avg_win_rr']:.1f}R  |  "
                f"Total P&amp;L: ${stats['total_pnl']:+,.0f}"
            )
            lines.append(f"  Profit factor: {stats['profit_factor']:.2f}x")

            # Win rate trend (vs simulation baseline of 55%)
            if wr_pct > 60:
                lines.append(f"  🔥 WR {wr_pct:.0f}% — beating baseline (55%). Edge is sharpening.")
            elif wr_pct < 45:
                lines.append(f"  ⚠️ WR {wr_pct:.0f}% — below baseline. Review setup quality.")

        # Best performing setups
        by_pair = edge_data.get("by_pair", {})
        if by_pair:
            top = sorted(by_pair.items(), key=lambda x: x[1]["wr"], reverse=True)[:2]
            if top:
                lines.append(f"\n<b>🎯 Strongest setups so far:</b>")
                for pair_name, data in top:
                    if data["trades"] >= 2:
                        lines.append(
                            f"  {pair_name}: {data['wr']:.0%} WR, "
                            f"avg {data['avg_rr']:+.1f}R ({data['trades']} trades)"
                        )

        # Lessons
        lines.append(f"\n<b>📝 Lessons from this trade:</b>")
        for i, lesson in enumerate(lessons[:3], 1):
            lines.append(f"  {i}. {lesson}")

        self.notifier.send("\n".join(lines))

    # ── Monthly Report ─────────────────────────────────────────────────

    def send_monthly_report(self, trade_history: List[Dict], account_balance: float):
        """Send a monthly performance report vs simulation baseline."""
        month_trades = [
            t for t in trade_history
            if t.get("exit_reason") not in ("open_at_end",)
        ]
        stats    = self._portfolio_stats(month_trades)
        edge     = self._edge_breakdown(month_trades)
        baseline = {"win_rate": 0.55, "avg_win_rr": 5.5, "expected_monthly_trades": 3.4}

        if not stats or not self.notifier:
            return

        wr_vs_base  = (stats["win_rate"] - baseline["win_rate"]) * 100
        rr_vs_base  = stats["avg_win_rr"] - baseline["avg_win_rr"]
        wr_dir      = "+" if wr_vs_base >= 0 else ""
        rr_dir      = "+" if rr_vs_base >= 0 else ""

        lines = [
            f"<b>📈 Monthly Performance Report</b>\n",
            f"  Balance:       ${account_balance:,.2f}",
            f"  Trades:        {stats['total_trades']}  "
            f"(baseline: ~{baseline['expected_monthly_trades']:.0f}/month)",
            f"  Win rate:      {stats['win_rate']:.0%}  "
            f"(baseline 55%:  {wr_dir}{wr_vs_base:.1f}pp)",
            f"  Avg win R:R:   +{stats['avg_win_rr']:.1f}R  "
            f"(baseline +5.5R:  {rr_dir}{rr_vs_base:.1f}R)",
            f"  Profit factor: {stats['profit_factor']:.2f}x",
            f"  Total P&amp;L:     ${stats['total_pnl']:+,.0f}",
            f"  Max consec. L: {stats['consecutive_losses']}",
        ]

        # Assessment
        if stats["win_rate"] >= 0.60 and stats["avg_win_rr"] >= 5.0:
            lines.append("\n🔥 Strategy is outperforming baseline — edge is real and sharpening.")
        elif stats["win_rate"] >= 0.55:
            lines.append("\n✅ Performing at or above baseline. Stay the course.")
        else:
            lines.append(
                "\n⚠️ Below baseline WR. Review recent setups — "
                "are entries still meeting all 5 filters?"
            )

        # Top patterns
        by_pattern = edge.get("by_pattern", {})
        if by_pattern:
            top = sorted(by_pattern.items(), key=lambda x: x[1]["pnl"], reverse=True)
            lines.append("\n<b>🎯 Best performing patterns this month:</b>")
            for name, data in top[:3]:
                lines.append(
                    f"  {name}: {data['wr']:.0%} WR  avg {data['avg_rr']:+.1f}R  "
                    f"${data['pnl']:+,.0f}"
                )

        # Forward guidance
        lines.append(f"\n<b>📌 Next month focus:</b>")
        if stats["win_rate"] < 0.50:
            lines.append("  • Tighten entry — only trade level score ≥4 setups")
            lines.append("  • Consider waiting for stronger signal (strength ≥0.80)")
        elif stats["avg_win_rr"] < 3.0:
            lines.append("  • Winners exiting too early — check exit level separation")
            lines.append("  • Confirm price has traveled ≥2R before any exit fires")
        else:
            lines.append("  • Keep executing. The edge is working.")
            lines.append("  • Stay patient — let the compounding work.")

        self.notifier.send("\n".join(lines))

    # ── Persistence ────────────────────────────────────────────────────

    def _save_lesson(self, trade: Dict, outcome: str, lessons: List[str]):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair":      trade.get("pair"),
            "direction": trade.get("direction"),
            "outcome":   outcome,
            "rr":        trade.get("rr"),
            "pnl":       trade.get("pnl"),
            "exit_reason": trade.get("exit_reason"),
            "pattern":   trade.get("pattern_type"),
            "signal_strength": trade.get("signal_strength"),
            "key_level_score": trade.get("key_level_score"),
            "session":   trade.get("session"),
            "bars_held": trade.get("bars_held"),
            "lessons":   lessons,
        }
        with open(LESSONS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _update_analytics(self, history: List[Dict]):
        stats    = self._portfolio_stats(history)
        edge     = self._edge_breakdown(history)
        all_lessons = self._load_all_lessons()

        analytics = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "overall":      stats,
            "edge":         edge,
            "lesson_count": len(all_lessons),
            "recurring_issues": self._find_recurring_issues(all_lessons),
        }
        ANALYTICS_FILE.write_text(json.dumps(analytics, indent=2))

    def _load_all_lessons(self) -> List[Dict]:
        if not LESSONS_FILE.exists():
            return []
        lessons = []
        for line in LESSONS_FILE.read_text().strip().splitlines():
            try:
                lessons.append(json.loads(line))
            except Exception:
                pass
        return lessons

    def _find_recurring_issues(self, lessons: List[Dict]) -> List[str]:
        """Identify issues that keep appearing across multiple trades."""
        from collections import Counter
        # Count lesson keywords across all trades
        keywords = [
            "WEAK SIGNAL", "EXITED TOO EARLY", "STOPPED OUT FAST",
            "LOW WIN RATE", "SESSION WARNING", "SMALL WIN"
        ]
        counts = Counter()
        for record in lessons:
            for lesson in record.get("lessons", []):
                for kw in keywords:
                    if kw in lesson:
                        counts[kw] += 1

        return [
            f"{kw}: seen {n} times — needs parameter review"
            for kw, n in counts.most_common(3)
            if n >= 3
        ]
