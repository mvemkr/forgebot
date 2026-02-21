"""
Risk Manager — THE non-negotiable layer.
All trading decisions are subordinate to these rules.
Fee-aware position sizing. Hard kill switches. No exceptions.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class MarketType(Enum):
    SPOT = "spot"
    FUTURES = "futures"


class TradeVerdict(Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    KILL_SWITCH = "KILL_SWITCH"


@dataclass
class TradeEvaluation:
    verdict: TradeVerdict
    reason: str
    product_id: str = ""
    market_type: MarketType = MarketType.SPOT
    position_size: float = 0.0
    entry_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    total_fees: float = 0.0
    net_profit_if_win: float = 0.0
    actual_risk: float = 0.0
    net_rr_ratio: float = 0.0
    fee_ratio: float = 0.0
    contracts: int = 0


@dataclass
class PortfolioState:
    total_value: float = 0.0
    spot_available: float = 0.0
    futures_available_margin: float = 0.0
    deployed_pct: float = 0.0
    open_positions: int = 0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_value: float = 0.0
    drawdown_pct: float = 0.0
    daily_fees_paid: float = 0.0

    # Kill switch state
    daily_loss_triggered: bool = False
    weekly_loss_triggered: bool = False
    max_drawdown_triggered: bool = False
    api_error_triggered: bool = False
    stale_data_triggered: bool = False
    below_minimum_triggered: bool = False

    @property
    def any_kill_switch_active(self) -> bool:
        return any([
            self.daily_loss_triggered,
            self.weekly_loss_triggered,
            self.max_drawdown_triggered,
            self.api_error_triggered,
            self.stale_data_triggered,
            self.below_minimum_triggered,
        ])


class RiskManager:
    """
    Hard risk rules. All values match the master directive.
    This class must be consulted before every trade.
    """

    # --- HARD RULES (never change without Mike's explicit approval) ---
    MAX_RISK_PER_TRADE_PCT = 0.02        # 2% of portfolio per trade
    MIN_RR_RATIO = 3.0                   # 1:3 minimum R:R AFTER FEES
    MAX_OPEN_POSITIONS = 3               # simultaneous positions
    MAX_DAILY_LOSS_PCT = 0.03            # 3% daily loss limit
    MAX_WEEKLY_LOSS_PCT = 0.07           # 7% weekly loss limit
    MAX_DRAWDOWN_PCT = 0.15              # 15% drawdown from peak
    MAX_POSITION_SIZE_PCT = 0.40         # 40% of portfolio per position
    MAX_DEPLOYED_PCT = 0.60              # 60% total deployed at once
    MINIMUM_PORTFOLIO = 300.0            # halt trading below $300
    MAX_FEE_RATIO = 0.20                 # fees must be <20% of gross profit
    STALE_DATA_SECONDS = 60              # websocket stale threshold
    API_ERROR_THRESHOLD = 5              # errors in 60s before halt
    STALE_ORDER_MINUTES = 15             # cancel unfilled limit orders after 15 min
    MARGIN_HEALTH_WARN_PCT = 50.0        # reduce futures positions below this
    MARGIN_HEALTH_EMERGENCY_PCT = 30.0   # close all futures below this

    # Fee rates
    SPOT_MAKER_FEE = 0.006     # 0.60% spot maker (entry tier)
    SPOT_TAKER_FEE = 0.012     # 1.20% spot taker
    FUTURES_FEE = 0.0005       # 0.05% per side, futures
    FUTURES_MIN_FEE = 0.15     # $0.15 minimum per contract

    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital
        self.state = PortfolioState(
            total_value=initial_capital,
            peak_value=initial_capital,
        )
        self._api_errors: list[float] = []  # timestamps of recent errors
        self._last_data_timestamp: float = time.time()
        self._session_start_value: float = initial_capital
        self._week_start_value: float = initial_capital
        logger.info(f"RiskManager initialized. Capital: ${initial_capital:,.2f}")

    # ------------------------------------------------------------------ #
    # Core Trade Evaluation                                                #
    # ------------------------------------------------------------------ #

    def evaluate_trade(
        self,
        product_id: str,
        entry_price: float,
        stop_loss_price: float,
        market_type: MarketType = MarketType.FUTURES,
        side: str = 'BUY',
    ) -> TradeEvaluation:
        """
        Evaluate whether a trade is acceptable under all risk rules.
        Returns a TradeEvaluation with verdict and full P&L breakdown.
        """
        portfolio_value = self.state.total_value

        # Kill switch check first
        if self.state.any_kill_switch_active:
            active = [k for k, v in {
                'daily_loss': self.state.daily_loss_triggered,
                'weekly_loss': self.state.weekly_loss_triggered,
                'max_drawdown': self.state.max_drawdown_triggered,
                'api_errors': self.state.api_error_triggered,
                'stale_data': self.state.stale_data_triggered,
                'below_minimum': self.state.below_minimum_triggered,
            }.items() if v]
            return TradeEvaluation(
                verdict=TradeVerdict.KILL_SWITCH,
                reason=f"Kill switch active: {', '.join(active)}",
                product_id=product_id,
            )

        # Position count check
        if self.state.open_positions >= self.MAX_OPEN_POSITIONS:
            return TradeEvaluation(
                verdict=TradeVerdict.REJECT,
                reason=f"Max open positions reached ({self.MAX_OPEN_POSITIONS})",
                product_id=product_id,
            )

        # Deployment check
        if self.state.deployed_pct >= self.MAX_DEPLOYED_PCT * 100:
            return TradeEvaluation(
                verdict=TradeVerdict.REJECT,
                reason=f"Portfolio {self.state.deployed_pct:.1f}% deployed — max {self.MAX_DEPLOYED_PCT*100:.0f}%",
                product_id=product_id,
            )

        # Minimum portfolio check
        if portfolio_value < self.MINIMUM_PORTFOLIO:
            self.state.below_minimum_triggered = True
            return TradeEvaluation(
                verdict=TradeVerdict.KILL_SWITCH,
                reason=f"Portfolio ${portfolio_value:.2f} below minimum ${self.MINIMUM_PORTFOLIO}",
                product_id=product_id,
            )

        # Fee rate selection
        fee_rate = self.FUTURES_FEE if market_type == MarketType.FUTURES else self.SPOT_MAKER_FEE

        # Stop distance
        if entry_price <= 0 or stop_loss_price <= 0:
            return TradeEvaluation(
                verdict=TradeVerdict.REJECT,
                reason="Invalid entry or stop price",
                product_id=product_id,
            )

        stop_distance_pct = abs(entry_price - stop_loss_price) / entry_price
        if stop_distance_pct <= 0:
            return TradeEvaluation(
                verdict=TradeVerdict.REJECT,
                reason="Stop loss must differ from entry",
                product_id=product_id,
            )

        # Position sizing from 2% portfolio risk
        risk_budget = portfolio_value * self.MAX_RISK_PER_TRADE_PCT
        position_size = risk_budget / stop_distance_pct
        position_size = min(position_size, portfolio_value * self.MAX_POSITION_SIZE_PCT)

        # Available capital check
        available = (
            self.state.futures_available_margin
            if market_type == MarketType.FUTURES
            else self.state.spot_available
        )
        if position_size > available * 0.95:
            position_size = available * 0.95
            if position_size < 10:
                return TradeEvaluation(
                    verdict=TradeVerdict.REJECT,
                    reason=f"Insufficient available capital: ${available:.2f}",
                    product_id=product_id,
                )

        # Fee calculation
        fee_per_side = max(position_size * fee_rate, self.FUTURES_MIN_FEE if market_type == MarketType.FUTURES else 0)
        total_fees = fee_per_side * 2  # entry + exit

        # R:R calculation (1:3 AFTER fees)
        target_distance_pct = stop_distance_pct * self.MIN_RR_RATIO
        gross_profit = position_size * target_distance_pct
        net_profit = gross_profit - total_fees

        actual_risk = (position_size * stop_distance_pct) + total_fees
        net_rr = net_profit / actual_risk if actual_risk > 0 else 0

        # Fee ratio gate
        fee_ratio = total_fees / gross_profit if gross_profit > 0 else 1.0
        if fee_ratio > self.MAX_FEE_RATIO:
            return TradeEvaluation(
                verdict=TradeVerdict.REJECT,
                reason=f"Fees consume {fee_ratio:.0%} of gross profit — max {self.MAX_FEE_RATIO:.0%}. Move size or use larger timeframe.",
                product_id=product_id,
                market_type=market_type,
                fee_ratio=fee_ratio,
            )

        # Net R:R gate
        if net_rr < self.MIN_RR_RATIO:
            return TradeEvaluation(
                verdict=TradeVerdict.REJECT,
                reason=f"Net R:R after fees is {net_rr:.2f}:1 — minimum {self.MIN_RR_RATIO:.0f}:1",
                product_id=product_id,
                market_type=market_type,
                net_rr_ratio=net_rr,
            )

        # Calculate take profit price
        if side.upper() == 'BUY':
            take_profit_price = entry_price * (1 + target_distance_pct)
            stop_loss_final = stop_loss_price
        else:
            take_profit_price = entry_price * (1 - target_distance_pct)
            stop_loss_final = stop_loss_price

        # For futures, calculate contracts
        contracts = 0
        if market_type == MarketType.FUTURES:
            contracts = max(1, int(position_size / entry_price))
            position_size = contracts * entry_price  # Adjust to exact contract count

        return TradeEvaluation(
            verdict=TradeVerdict.ACCEPT,
            reason="Trade passes all risk checks",
            product_id=product_id,
            market_type=market_type,
            position_size=position_size,
            entry_price=entry_price,
            stop_loss_price=stop_loss_final,
            take_profit_price=take_profit_price,
            total_fees=total_fees,
            net_profit_if_win=net_profit,
            actual_risk=actual_risk,
            net_rr_ratio=net_rr,
            fee_ratio=fee_ratio,
            contracts=contracts,
        )

    # ------------------------------------------------------------------ #
    # Kill Switches                                                         #
    # ------------------------------------------------------------------ #

    def update_portfolio(
        self,
        total_value: float,
        spot_available: float,
        futures_available_margin: float,
        open_positions: int,
        daily_pnl: float,
        daily_fees_paid: float,
    ) -> list[str]:
        """
        Update portfolio state and check all kill switch conditions.
        Returns list of triggered alerts.
        """
        alerts = []
        self.state.total_value = total_value
        self.state.spot_available = spot_available
        self.state.futures_available_margin = futures_available_margin
        self.state.open_positions = open_positions
        self.state.daily_pnl = daily_pnl
        self.state.daily_fees_paid = daily_fees_paid

        # Update peak value
        if total_value > self.state.peak_value:
            self.state.peak_value = total_value

        # Drawdown from peak
        drawdown = (self.state.peak_value - total_value) / self.state.peak_value if self.state.peak_value > 0 else 0
        self.state.drawdown_pct = drawdown * 100

        # === KILL SWITCH CHECKS ===

        # 1. Below minimum
        if total_value < self.MINIMUM_PORTFOLIO and not self.state.below_minimum_triggered:
            self.state.below_minimum_triggered = True
            msg = f"🚨 KILL SWITCH: Portfolio ${total_value:.2f} below minimum ${self.MINIMUM_PORTFOLIO}. All trading halted."
            alerts.append(msg)
            logger.critical(msg)

        # 2. Daily loss
        daily_loss_pct = abs(daily_pnl) / self._session_start_value if daily_pnl < 0 else 0
        if daily_loss_pct >= self.MAX_DAILY_LOSS_PCT and not self.state.daily_loss_triggered:
            self.state.daily_loss_triggered = True
            msg = f"🚨 KILL SWITCH: Daily loss {daily_loss_pct:.1%} (${abs(daily_pnl):.2f}) hit {self.MAX_DAILY_LOSS_PCT:.0%} limit."
            alerts.append(msg)
            logger.critical(msg)

        # 3. Weekly loss
        weekly_pnl = total_value - self._week_start_value
        weekly_loss_pct = abs(weekly_pnl) / self._week_start_value if weekly_pnl < 0 else 0
        if weekly_loss_pct >= self.MAX_WEEKLY_LOSS_PCT and not self.state.weekly_loss_triggered:
            self.state.weekly_loss_triggered = True
            msg = f"🚨 KILL SWITCH: Weekly loss {weekly_loss_pct:.1%} (${abs(weekly_pnl):.2f}) hit {self.MAX_WEEKLY_LOSS_PCT:.0%} limit."
            alerts.append(msg)
            logger.critical(msg)

        # 4. Max drawdown
        if drawdown >= self.MAX_DRAWDOWN_PCT and not self.state.max_drawdown_triggered:
            self.state.max_drawdown_triggered = True
            msg = f"🚨 KILL SWITCH: Drawdown {drawdown:.1%} from peak ${self.state.peak_value:.2f} hit {self.MAX_DRAWDOWN_PCT:.0%} limit."
            alerts.append(msg)
            logger.critical(msg)

        if alerts:
            logger.critical(f"Kill switches triggered: {len(alerts)}")

        return alerts

    def record_api_error(self) -> bool:
        """
        Record an API error. Returns True if kill switch threshold exceeded.
        """
        now = time.time()
        self._api_errors.append(now)
        # Keep only errors in the last 60 seconds
        self._api_errors = [t for t in self._api_errors if now - t <= 60]
        if len(self._api_errors) >= self.API_ERROR_THRESHOLD and not self.state.api_error_triggered:
            self.state.api_error_triggered = True
            msg = f"🚨 KILL SWITCH: {len(self._api_errors)} API errors in 60 seconds. Trading paused."
            logger.critical(msg)
            return True
        return False

    def record_data_update(self):
        """Call this whenever a valid WebSocket message arrives."""
        self._last_data_timestamp = time.time()
        if self.state.stale_data_triggered:
            self.state.stale_data_triggered = False
            logger.info("Stale data kill switch cleared — data flowing again.")

    def check_stale_data(self) -> bool:
        """Returns True and triggers kill switch if data is stale."""
        age = time.time() - self._last_data_timestamp
        if age > self.STALE_DATA_SECONDS and not self.state.stale_data_triggered:
            self.state.stale_data_triggered = True
            msg = f"🚨 KILL SWITCH: No WebSocket data for {age:.0f}s. Trading paused."
            logger.critical(msg)
            return True
        return False

    def reset_daily(self, current_value: float):
        """Call at the start of each trading day."""
        self._session_start_value = current_value
        self.state.daily_loss_triggered = False
        self.state.daily_pnl = 0.0
        self.state.daily_fees_paid = 0.0
        self._api_errors.clear()
        logger.info(f"Daily reset. Starting value: ${current_value:.2f}")

    def reset_weekly(self, current_value: float):
        """Call at the start of each trading week."""
        self._week_start_value = current_value
        self.state.weekly_loss_triggered = False
        logger.info(f"Weekly reset. Starting value: ${current_value:.2f}")

    def manual_resume(self, mike_approved: bool = False):
        """Mike can manually resume after weekly loss or drawdown kill switch."""
        if not mike_approved:
            logger.warning("manual_resume requires explicit Mike approval")
            return
        self.state.weekly_loss_triggered = False
        self.state.max_drawdown_triggered = False
        self.state.api_error_triggered = False
        self.state.stale_data_triggered = False
        logger.info("Kill switches manually cleared by Mike.")

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def status_report(self) -> dict:
        """Return a snapshot of the current risk state."""
        return {
            'portfolio_value': self.state.total_value,
            'peak_value': self.state.peak_value,
            'drawdown_pct': round(self.state.drawdown_pct, 2),
            'daily_pnl': round(self.state.daily_pnl, 2),
            'daily_fees': round(self.state.daily_fees_paid, 2),
            'open_positions': self.state.open_positions,
            'deployed_pct': round(self.state.deployed_pct, 1),
            'kill_switches': {
                'daily_loss': self.state.daily_loss_triggered,
                'weekly_loss': self.state.weekly_loss_triggered,
                'max_drawdown': self.state.max_drawdown_triggered,
                'api_errors': self.state.api_error_triggered,
                'stale_data': self.state.stale_data_triggered,
                'below_minimum': self.state.below_minimum_triggered,
            },
            'any_kill_switch_active': self.state.any_kill_switch_active,
        }
