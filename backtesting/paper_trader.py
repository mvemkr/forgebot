"""
Paper Trading Engine — live market data, simulated execution with full fee modeling.
Runs the SOL momentum strategy and reports results to logs + Telegram.
"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

from src.exchange.coinbase_client import CoinbaseClient
from src.risk.risk_manager import RiskManager, MarketType, TradeVerdict
from src.risk.portfolio_tracker import PortfolioTracker, Position
from src.strategy.sol_momentum import SOLMomentumStrategy, PRODUCT_ID, TIMEFRAME, CANDLES_NEEDED

# --- Logging setup ---
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'paper_trader.log')),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('paper_trader')

STARTING_CAPITAL = 1000.0
POLL_INTERVAL_SECONDS = 60 * 15  # Check every 15 minutes (one candle)
CANDLE_LOOKBACK_HOURS = 24        # How much history to pull per cycle


class PaperTrader:
    """
    Runs the SOL momentum strategy on live data with simulated execution.
    """

    def __init__(self):
        self.client = CoinbaseClient()
        self.risk = RiskManager(initial_capital=STARTING_CAPITAL)
        self.portfolio = PortfolioTracker(starting_capital=STARTING_CAPITAL, paper=True)
        self.strategy = SOLMomentumStrategy()
        self.cycle = 0

        # Pre-populate portfolio state in risk manager
        self.risk.state.spot_available = STARTING_CAPITAL
        self.risk.state.futures_available_margin = STARTING_CAPITAL
        self.risk.state.total_value = STARTING_CAPITAL

        logger.info("=" * 60)
        logger.info("[PAPER] Paper Trading Engine starting")
        logger.info(f"[PAPER] Product: {PRODUCT_ID} | Timeframe: {TIMEFRAME}")
        logger.info(f"[PAPER] Starting capital: ${STARTING_CAPITAL:,.2f}")
        logger.info("=" * 60)

    def fetch_candles(self) -> list[dict]:
        """Fetch recent candles for strategy analysis."""
        now = int(time.time())
        lookback = CANDLE_LOOKBACK_HOURS * 3600
        candles = self.client.get_candles(
            product_id=PRODUCT_ID,
            granularity=TIMEFRAME,
            start=now - lookback,
            end=now,
        )
        logger.info(f"Fetched {len(candles)} candles for {PRODUCT_ID}")
        return candles

    def get_current_price(self) -> float:
        """Get the current SOL price."""
        product = self.client.get_product(PRODUCT_ID)
        price = product.get('price', 0)
        if price == 0:
            # Fallback to best bid/ask mid
            bbo = self.client.get_best_bid_ask(PRODUCT_ID)
            price = bbo.get('mid', 0)
        return price

    def check_open_positions(self, current_price: float):
        """Check if any open positions have hit TP or SL."""
        for product_id, position in list(self.portfolio.open_positions.items()):
            if position.hit_take_profit(current_price):
                trade = self.portfolio.close_position(
                    product_id=product_id,
                    exit_price=position.take_profit,
                    exit_reason='take_profit',
                    fee_rate=0.0005,
                )
                if trade:
                    self._log_trade_close(trade, current_price)

            elif position.is_stopped_out(current_price):
                trade = self.portfolio.close_position(
                    product_id=product_id,
                    exit_price=position.stop_loss,
                    exit_reason='stop_loss',
                    fee_rate=0.0005,
                )
                if trade:
                    self._log_trade_close(trade, current_price)

    def _log_trade_close(self, trade, current_price: float):
        result = "WIN ✅" if trade.net_pnl > 0 else "LOSS ❌"
        logger.info(
            f"[PAPER] Trade closed — {result} | {trade.side} {trade.product_id} "
            f"| Entry: ${trade.entry_price:.4f} Exit: ${trade.exit_price:.4f} "
            f"| Net P&L: ${trade.net_pnl:+.4f} | Fees: ${trade.entry_fee + trade.exit_fee:.4f} "
            f"| Reason: {trade.exit_reason} | Duration: {trade.hold_duration_minutes:.0f}min"
        )
        self._log_decisions({
            'event': 'trade_closed',
            'result': 'win' if trade.net_pnl > 0 else 'loss',
            'net_pnl': round(trade.net_pnl, 4),
            'fees': round(trade.entry_fee + trade.exit_fee, 4),
            'exit_reason': trade.exit_reason,
            'duration_min': round(trade.hold_duration_minutes, 1),
        })

    def run_cycle(self):
        """Run one full strategy cycle."""
        self.cycle += 1
        logger.info(f"\n--- Cycle {self.cycle} | {datetime.now(timezone.utc).isoformat()} ---")

        # Get current price
        current_price = self.get_current_price()
        if current_price == 0:
            logger.warning("Could not get current price — skipping cycle")
            self.risk.record_api_error()
            return

        self.risk.record_data_update()
        logger.info(f"Current {PRODUCT_ID} price: ${current_price:,.4f}")

        # Check existing positions first
        self.check_open_positions(current_price)

        # Update risk manager state
        port_value = self.portfolio.total_value({PRODUCT_ID: current_price})
        daily_pnl = self.portfolio.realized_pnl()
        daily_fees = self.portfolio.total_fees_paid()
        alerts = self.risk.update_portfolio(
            total_value=port_value,
            spot_available=self.portfolio.cash,
            futures_available_margin=self.portfolio.cash,
            open_positions=len(self.portfolio.open_positions),
            daily_pnl=daily_pnl,
            daily_fees_paid=daily_fees,
        )
        for alert in alerts:
            logger.critical(f"ALERT: {alert}")

        # Skip strategy if kill switch active
        if self.risk.state.any_kill_switch_active:
            logger.warning("[PAPER] Kill switch active — no new positions")
            return

        # Skip if already have a position in this product
        if PRODUCT_ID in self.portfolio.open_positions:
            pos = self.portfolio.open_positions[PRODUCT_ID]
            unrealized = pos.unrealized_pnl(current_price)
            logger.info(
                f"[PAPER] Holding {pos.side} position | "
                f"Entry: ${pos.entry_price:.4f} | "
                f"Unrealized P&L: ${unrealized:+.4f}"
            )
            return

        # Fetch candles and analyze
        candles = self.fetch_candles()
        if len(candles) < CANDLES_NEEDED:
            logger.warning(f"[PAPER] Not enough candles ({len(candles)})")
            return

        signal = self.strategy.analyze(candles)
        logger.info(
            f"[PAPER] Signal: {signal.side} | RSI={signal.rsi:.1f} "
            f"MACD={signal.macd_hist:.4f} | Reason: {signal.reason}"
        )

        if signal.side == 'NONE':
            self._log_decisions({'event': 'no_signal', 'reason': signal.reason})
            return

        # Risk evaluation
        evaluation = self.risk.evaluate_trade(
            product_id=PRODUCT_ID,
            entry_price=signal.entry_price,
            stop_loss_price=signal.stop_loss,
            market_type=MarketType.FUTURES,
            side=signal.side,
        )

        self._log_decisions({
            'event': 'signal_evaluated',
            'signal_side': signal.side,
            'entry': signal.entry_price,
            'stop': signal.stop_loss,
            'verdict': evaluation.verdict.value,
            'reason': evaluation.reason,
            'position_size': round(evaluation.position_size, 2),
            'contracts': evaluation.contracts,
            'net_rr': round(evaluation.net_rr_ratio, 2),
            'fee_ratio': round(evaluation.fee_ratio, 4),
            'total_fees': round(evaluation.total_fees, 4),
        })

        if evaluation.verdict != TradeVerdict.ACCEPT:
            logger.info(f"[PAPER] Trade rejected: {evaluation.reason}")
            return

        # Open paper position
        entry_fee = evaluation.position_size * 0.0005
        position = Position(
            product_id=PRODUCT_ID,
            side=signal.side,
            entry_price=evaluation.entry_price,
            size=float(evaluation.contracts) if evaluation.contracts > 0 else evaluation.position_size / evaluation.entry_price,
            stop_loss=evaluation.stop_loss_price,
            take_profit=evaluation.take_profit_price,
            market_type='futures',
            entry_fee=entry_fee,
            paper=True,
        )

        if self.portfolio.open_position(position):
            logger.info(
                f"[PAPER] ✅ ENTERED {signal.side} {evaluation.contracts} contracts {PRODUCT_ID} "
                f"@ ${evaluation.entry_price:.4f} | "
                f"SL: ${evaluation.stop_loss_price:.4f} | "
                f"TP: ${evaluation.take_profit_price:.4f} | "
                f"Risk: ${evaluation.actual_risk:.2f} | "
                f"Net R:R: {evaluation.net_rr_ratio:.2f}:1"
            )

    def status_report(self, current_price: float):
        """Log a full status report."""
        summary = self.portfolio.performance_summary({PRODUCT_ID: current_price})
        risk_status = self.risk.status_report()
        logger.info("\n" + "=" * 60)
        logger.info("[PAPER] STATUS REPORT")
        logger.info(f"  Capital:       ${summary['current_value']:,.2f} (started ${summary['starting_capital']:,.2f})")
        logger.info(f"  Return:        {summary['total_return_pct']:+.2f}%")
        logger.info(f"  Realized P&L:  ${summary['realized_pnl']:+.2f}")
        logger.info(f"  Total Fees:    ${summary['total_fees_paid']:.2f} ({summary['fees_as_pct_of_capital']:.2f}% of capital)")
        logger.info(f"  Trades:        {summary['total_trades']} closed | Win rate: {summary['win_rate']:.1f}%")
        logger.info(f"  Open:          {summary['open_positions']} position(s)")
        logger.info(f"  Drawdown:      {risk_status['drawdown_pct']:.2f}% from peak")
        logger.info("=" * 60)
        return summary

    def _log_decisions(self, data: dict):
        """Append a decision log entry."""
        entry = {'ts': datetime.now(timezone.utc).isoformat(), **data}
        with open(os.path.join(LOG_DIR, 'decisions.log'), 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def run(self, cycles: int = 0):
        """
        Main loop. cycles=0 means run indefinitely.
        """
        logger.info(f"[PAPER] Starting run. {'Indefinite' if cycles == 0 else f'{cycles} cycles'}")
        cycle_count = 0

        try:
            while True:
                self.run_cycle()
                cycle_count += 1

                # Status report every 4 cycles (~1 hour on 15-min candles)
                if cycle_count % 4 == 0:
                    price = self.get_current_price()
                    self.status_report(price)

                if cycles > 0 and cycle_count >= cycles:
                    break

                logger.info(f"[PAPER] Sleeping {POLL_INTERVAL_SECONDS}s until next candle...")
                time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("[PAPER] Interrupted by user.")
        finally:
            price = self.get_current_price() or 85.0
            summary = self.status_report(price)
            logger.info("[PAPER] Paper trading session ended.")
            return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SOL Paper Trader")
    parser.add_argument('--cycles', type=int, default=0, help='Number of cycles to run (0=infinite)')
    parser.add_argument('--test', action='store_true', help='Run one cycle and exit')
    args = parser.parse_args()

    trader = PaperTrader()
    if args.test:
        trader.run_cycle()
        price = trader.get_current_price() or 85.0
        trader.status_report(price)
    else:
        trader.run(cycles=args.cycles)
