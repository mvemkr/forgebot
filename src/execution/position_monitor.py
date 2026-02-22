"""
Position Monitor — Watches open trades for exit signals and stop management.

This implements the "set and forget" EXIT strategy from the research:

  - Every 4H: check if price has moved 1:1 → move SL to breakeven (automatic)
  - On daily close: check for reversal signals (pin bar, engulfing against trade)
      → ALERT Mike (do NOT auto-close — he decides)
  - If OANDA reports stop was hit → log the exit, close position in our tracker

The bot is responsible for:
  ✅ Moving SL to breakeven (automatic — always safe)
  ✅ Detecting exit signals and notifying Mike
  ✅ Logging when stop is hit
  
Mike is responsible for:
  🧠 Deciding whether to close on exit signals or hold
  🧠 Final exit decision on large runners

This matches how Alex operated: he exited when the CHART told him to,
not when a number told him to.
"""
import logging
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ..exchange.oanda_client import OandaClient
from ..strategy.forex.set_and_forget import SetAndForgetStrategy
from ..strategy.forex.entry_signal import EntrySignalDetector, EntrySignal
from .trade_journal import TradeJournal
from .notifier import Notifier

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Monitors all open positions for exit signals and stop management.

    Parameters
    ----------
    strategy : SetAndForgetStrategy
        Running strategy instance (holds open_positions dict)
    oanda : OandaClient
        Live OANDA connection for current prices and trade data
    journal : TradeJournal
        Trade journal for logging exits/signals
    notifier : Notifier
        Sends alerts to Mike
    dry_run : bool
        If True, stop moves are simulated (not sent to OANDA)
    """

    def __init__(
        self,
        strategy: SetAndForgetStrategy,
        oanda: OandaClient,
        journal: TradeJournal,
        notifier: "Notifier",
        dry_run: bool = True,
    ):
        from .trade_analyzer import TradeAnalyzer
        self.strategy = strategy
        self.oanda = oanda
        self.journal = journal
        self.notifier = notifier
        self.dry_run = dry_run
        self._signal_detector = EntrySignalDetector(min_body_ratio=0.45)
        self._breakeven_moved: set = set()   # track which trades already moved to BE
        self.analyzer = TradeAnalyzer(notifier=notifier)

    # ── Main Check ─────────────────────────────────────────────────────

    def check_all(self, candle_data: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Run all position checks. Call every 4H during active sessions.

        Parameters
        ----------
        candle_data : dict
            {pair: {'daily': df_d, '4h': df_4h, '1h': df_1h}}
            Only pairs in strategy.open_positions need to be included.
        """
        if not self.strategy.open_positions:
            return

        # First: sync with OANDA to catch any stops that were hit
        self._sync_stops_from_oanda()

        # Then check each open position
        for pair, pos in list(self.strategy.open_positions.items()):
            data = candle_data.get(pair, {})
            df_d  = data.get("daily")
            df_4h = data.get("4h")
            df_1h = data.get("1h")

            if df_d is None or df_4h is None:
                logger.warning(f"{pair}: No candle data for position monitor")
                continue

            current_price = df_1h["close"].iloc[-1] if df_1h is not None else df_d["close"].iloc[-1]

            # 1. Check breakeven move
            self._check_breakeven(pair, pos, current_price)

            # 2. Check daily exit signals
            self._check_exit_signals(pair, pos, df_d, current_price)

    # ── Stop Hit Detection ─────────────────────────────────────────────

    def _sync_stops_from_oanda(self):
        """
        Fetch open trades from OANDA. If a position we're tracking
        is no longer on OANDA, the stop was hit — log and close.
        """
        try:
            oanda_trades = self.oanda.get_open_trades()
            oanda_instruments = {t["instrument"].replace("_", "/") for t in oanda_trades}

            for pair in list(self.strategy.open_positions.keys()):
                if pair not in oanda_instruments:
                    pos = self.strategy.open_positions[pair]
                    logger.info(f"📌 {pair}: No longer in OANDA open trades — stop hit or closed")

                    # Get final account balance
                    try:
                        summary = self.oanda.get_account_summary()
                        balance_after = summary.get("balance", 0)
                    except Exception:
                        balance_after = 0

                    # Log the exit — we don't know exact exit price, use stop as approximation
                    self.journal.log_trade_exited(
                        pair=pair,
                        oanda_trade_id=pos.get("oanda_trade_id"),
                        exit_price=pos["stop"],
                        exit_reason="stop_hit",
                        entry_price=pos["entry"],
                        stop_loss=pos["stop"],
                        direction=pos["direction"],
                        units=pos.get("units", 0),
                        account_balance_after=balance_after,
                        notes="Stop hit detected via OANDA sync",
                    )

                    self.strategy.close_position(pair, pos["stop"])
                    self.strategy.record_stop_out(pair)   # start cooldown — no re-entry for 5 days

                    self.notifier.send(
                        f"🔴 {pair} STOP HIT\n"
                        f"Entry: {pos['entry']:.5f} → Stop: {pos['stop']:.5f}\n"
                        f"Direction: {pos['direction'].upper()}\n"
                        f"Account balance: ${balance_after:,.2f}"
                    )

                    # Post-trade analysis — lessons learned
                    trade_record = {
                        "pair":            pair,
                        "direction":       pos.get("direction"),
                        "entry_price":     pos.get("entry"),
                        "exit_price":      pos.get("stop"),
                        "stop_loss":       pos.get("stop"),
                        "entry_ts":        pos.get("entry_ts", ""),
                        "exit_ts":         datetime.now(timezone.utc).isoformat(),
                        "exit_reason":     "stop_hit",
                        "pnl":             -(pos.get("risk_dollars", 0)),
                        "rr":              -1.0,
                        "pattern_type":    pos.get("pattern_type", ""),
                        "psych_level":     pos.get("psych_level", 0),
                        "key_level_score": pos.get("key_level_score", 0),
                        "signal_strength": pos.get("signal_strength", 0),
                        "session":         pos.get("session", ""),
                        "risk_pct":        pos.get("risk_pct", 0),
                        "risk_dollars":    pos.get("risk_dollars", 0),
                        "bars_held":       0,
                        "confidence":      pos.get("confidence", 0),
                    }
                    try:
                        all_trades = self.journal.get_all_trades() if hasattr(self.journal, "get_all_trades") else []
                        self.analyzer.analyze_closed_trade(trade_record, all_trades)
                    except Exception as ae:
                        logger.warning(f"Trade analysis failed: {ae}")

        except Exception as e:
            logger.error(f"OANDA sync failed: {e}")

    # ── Breakeven Move ─────────────────────────────────────────────────

    def _check_breakeven(self, pair: str, pos: dict, current_price: float):
        """
        Move SL to breakeven when price has moved 1:1 in our favor.
        This is automatic — always safe, never moves SL further from entry.
        """
        trade_key = f"{pair}_{pos.get('oanda_trade_id', 'noid')}"
        if trade_key in self._breakeven_moved:
            return

        entry = pos["entry"]
        stop  = pos["stop"]
        direction = pos["direction"]

        risk = abs(entry - stop)
        if risk == 0:
            return

        # Has price moved 1:1 in our favor?
        if direction == "long":
            moved_1r = current_price >= entry + risk
        else:
            moved_1r = current_price <= entry - risk

        if not moved_1r:
            return

        # Already at breakeven?
        if stop == entry:
            self._breakeven_moved.add(trade_key)
            return

        logger.info(f"✅ {pair}: Price moved 1:1 — moving stop to breakeven at {entry:.5f}")

        oanda_trade_id = pos.get("oanda_trade_id")
        result = self.oanda.move_stop_to_breakeven(
            trade_id=oanda_trade_id or "unknown",
            entry_price=entry,
            dry_run=self.dry_run,
        )

        # Update local position tracker
        pos["stop"] = entry

        self._breakeven_moved.add(trade_key)

        self.journal.log_breakeven_moved(
            pair=pair,
            oanda_trade_id=oanda_trade_id,
            entry_price=entry,
            current_price=current_price,
            direction=direction,
        )

        self.notifier.send(
            f"✅ {pair}: Stop moved to breakeven\n"
            f"Entry: {entry:.5f} | Current: {current_price:.5f}\n"
            f"Trade is now risk-free 🎉"
        )

    # ── Exit Signal Detection ──────────────────────────────────────────

    def _check_exit_signals(
        self,
        pair: str,
        pos: dict,
        df_daily: pd.DataFrame,
        current_price: float,
    ):
        """
        Check the daily chart for reversal signals AGAINST our trade direction.
        
        If found: ALERT Mike. Do NOT auto-close.
        Mike reviews the chart and decides.
        
        Alex's rule: "If daily candle closes showing rejection (doji, pin bar,
        engulfing against you): consider exit."
        """
        direction = pos["direction"]
        entry = pos["entry"]
        opposite_direction = "short" if direction == "long" else "long"

        # Look for reversal signals in the opposite direction
        has_signal, signal = self._signal_detector.has_signal(df_daily, opposite_direction)

        if not has_signal or signal is None:
            return

        # Only alert on meaningful signals
        if signal.strength < 0.50:
            return

        # Calculate current R:R
        stop = pos["stop"]
        risk = abs(entry - stop)
        if direction == "long":
            current_rr = (current_price - entry) / risk if risk > 0 else 0
        else:
            current_rr = (entry - current_price) / risk if risk > 0 else 0

        logger.warning(
            f"⚠️ {pair}: Exit signal detected — {signal.signal_type} "
            f"(strength={signal.strength:.2f}) | Current R:R = {current_rr:+.1f}"
        )

        self.journal.log_exit_signal(
            pair=pair,
            signal_type=signal.signal_type,
            current_price=current_price,
            entry_price=entry,
            direction=direction,
            oanda_trade_id=pos.get("oanda_trade_id"),
            notes=(
                f"Signal strength: {signal.strength:.2f} | "
                f"Current R:R: {current_rr:+.1f} | "
                f"Review chart — Mike decides whether to close or hold"
            ),
        )

        # Direction qualifiers for the message
        price_emoji = "📈" if current_rr > 0 else "📉"

        self.notifier.send(
            f"⚠️ {pair} EXIT SIGNAL\n"
            f"Signal: {signal.signal_type} (strength {signal.strength:.0%})\n"
            f"Direction: {direction.upper()} | Current: {current_price:.5f}\n"
            f"{price_emoji} Current R:R: {current_rr:+.1f}\n\n"
            f"Bot is NOT closing — YOU decide.\n"
            f"If daily is showing rejection at a major level → consider closing.\n"
            f"If trend still intact → consider holding."
        )
