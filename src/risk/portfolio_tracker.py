"""
Portfolio tracker — tracks P&L, positions, and fees in real time.
Used by both paper trading and live trading.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    product_id: str
    side: str              # 'BUY' or 'SELL'
    entry_price: float
    size: float            # In base units (or contracts for futures)
    stop_loss: float
    take_profit: float
    market_type: str       # 'spot' or 'futures'
    opened_at: float = field(default_factory=time.time)
    entry_fee: float = 0.0
    paper: bool = True     # Paper trade or live
    order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None

    @property
    def notional(self) -> float:
        return self.entry_price * self.size

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == 'BUY':
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size

    def is_stopped_out(self, current_price: float) -> bool:
        if self.side == 'BUY':
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def hit_take_profit(self, current_price: float) -> bool:
        if self.side == 'BUY':
            return current_price >= self.take_profit
        return current_price <= self.take_profit


@dataclass
class ClosedTrade:
    product_id: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    market_type: str
    entry_fee: float
    exit_fee: float
    gross_pnl: float
    net_pnl: float
    exit_reason: str       # 'take_profit', 'stop_loss', 'manual'
    opened_at: float
    closed_at: float
    paper: bool = True

    @property
    def hold_duration_minutes(self) -> float:
        return (self.closed_at - self.opened_at) / 60


class PortfolioTracker:
    """
    Tracks open positions, closed trades, fees, and P&L.
    Works for both paper and live trading.
    """

    def __init__(self, starting_capital: float = 1000.0, paper: bool = True):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.paper = paper
        self.open_positions: dict[str, Position] = {}  # product_id -> Position
        self.closed_trades: list[ClosedTrade] = []
        self._session_start = time.time()

    # ------------------------------------------------------------------ #
    # Position Management                                                  #
    # ------------------------------------------------------------------ #

    def open_position(self, position: Position) -> bool:
        """
        Record a new position. Deducts notional + fee from cash.
        Returns False if insufficient cash.
        """
        required = position.notional + position.entry_fee
        if self.cash < required:
            logger.warning(
                f"open_position: Insufficient cash. Need ${required:.2f}, have ${self.cash:.2f}"
            )
            return False

        if position.product_id in self.open_positions:
            logger.warning(f"open_position: Already have position in {position.product_id}")
            return False

        self.cash -= required
        self.open_positions[position.product_id] = position
        logger.info(
            f"{'[PAPER] ' if self.paper else ''}Opened {position.side} {position.size} "
            f"{position.product_id} @ ${position.entry_price:,.4f} "
            f"| SL=${position.stop_loss:,.4f} TP=${position.take_profit:,.4f} "
            f"| Fee=${position.entry_fee:.4f}"
        )
        return True

    def close_position(
        self,
        product_id: str,
        exit_price: float,
        exit_reason: str,
        fee_rate: float = 0.0005,
    ) -> Optional[ClosedTrade]:
        """
        Close an open position. Returns the ClosedTrade or None if not found.
        """
        position = self.open_positions.get(product_id)
        if not position:
            logger.warning(f"close_position: No open position for {product_id}")
            return None

        exit_fee = position.notional * fee_rate
        if position.side == 'BUY':
            gross_pnl = (exit_price - position.entry_price) * position.size
        else:
            gross_pnl = (position.entry_price - exit_price) * position.size

        net_pnl = gross_pnl - exit_fee - position.entry_fee

        trade = ClosedTrade(
            product_id=product_id,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            market_type=position.market_type,
            entry_fee=position.entry_fee,
            exit_fee=exit_fee,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            opened_at=position.opened_at,
            closed_at=time.time(),
            paper=position.paper,
        )

        # Return cash
        exit_notional = exit_price * position.size
        self.cash += exit_notional - exit_fee
        del self.open_positions[product_id]
        self.closed_trades.append(trade)

        result_emoji = "✅" if net_pnl > 0 else "❌"
        logger.info(
            f"{'[PAPER] ' if self.paper else ''}{result_emoji} Closed {position.side} {product_id} "
            f"@ ${exit_price:,.4f} | Reason: {exit_reason} "
            f"| Net P&L: ${net_pnl:+.4f} | Fees: ${position.entry_fee + exit_fee:.4f}"
        )
        return trade

    # ------------------------------------------------------------------ #
    # State Queries                                                        #
    # ------------------------------------------------------------------ #

    def total_value(self, current_prices: dict[str, float]) -> float:
        """Total portfolio value: cash + open position values."""
        position_value = sum(
            pos.entry_price * pos.size + pos.unrealized_pnl(current_prices.get(pid, pos.entry_price))
            for pid, pos in self.open_positions.items()
        )
        return self.cash + position_value

    def unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        return sum(
            pos.unrealized_pnl(current_prices.get(pid, pos.entry_price))
            for pid, pos in self.open_positions.items()
        )

    def realized_pnl(self) -> float:
        return sum(t.net_pnl for t in self.closed_trades)

    def total_fees_paid(self) -> float:
        closed_fees = sum(t.entry_fee + t.exit_fee for t in self.closed_trades)
        open_fees = sum(p.entry_fee for p in self.open_positions.values())
        return closed_fees + open_fees

    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.net_pnl > 0)
        return wins / len(self.closed_trades)

    def performance_summary(self, current_prices: dict[str, float]) -> dict:
        """Full performance snapshot."""
        closed = self.closed_trades
        wins = [t for t in closed if t.net_pnl > 0]
        losses = [t for t in closed if t.net_pnl <= 0]
        total_fees = self.total_fees_paid()
        realized = self.realized_pnl()
        unrealized = self.unrealized_pnl(current_prices)
        port_value = self.total_value(current_prices)

        avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0
        profit_factor = abs(avg_win * len(wins)) / abs(avg_loss * len(losses)) if losses and avg_loss != 0 else float('inf')

        return {
            'starting_capital': self.starting_capital,
            'current_value': round(port_value, 2),
            'cash': round(self.cash, 2),
            'total_return_pct': round((port_value - self.starting_capital) / self.starting_capital * 100, 2),
            'realized_pnl': round(realized, 2),
            'unrealized_pnl': round(unrealized, 2),
            'total_fees_paid': round(total_fees, 2),
            'fees_as_pct_of_capital': round(total_fees / self.starting_capital * 100, 2),
            'total_trades': len(closed),
            'open_positions': len(self.open_positions),
            'win_rate': round(self.win_rate() * 100, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'mode': 'PAPER' if self.paper else 'LIVE',
        }
