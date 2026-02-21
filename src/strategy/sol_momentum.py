"""
SOL Perpetual Momentum Strategy — v1
Timeframe: 15-minute candles
Market: SLP-20DEC30 (SOL perpetual futures)

Entry logic:
  - RSI(14) crosses above 50 from below (bullish) or below 50 from above (bearish)
  - MACD histogram flipping positive (buy) or negative (sell)
  - Price above 20 EMA for buys, below for sells
  - 15-min candle close confirms direction

Exit logic:
  - Take profit: 3x stop distance (minimum 1:3 R:R)
  - Stop loss: below recent swing low (buy) / above recent swing high (sell)
"""

import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

PRODUCT_ID = "SLP-20DEC30-CDE"  # SOL perpetual futures
TIMEFRAME = "FIFTEEN_MINUTE"
CANDLES_NEEDED = 50  # Minimum candles for indicator calculation


@dataclass
class Signal:
    side: str           # 'BUY' or 'SELL' or 'NONE'
    entry_price: float
    stop_loss: float
    confidence: float   # 0-1
    reason: str
    rsi: float = 0.0
    macd_hist: float = 0.0
    ema_20: float = 0.0


class SOLMomentumStrategy:
    """
    15-minute momentum strategy for SOL perpetual futures.
    Uses RSI + MACD + EMA for signal generation.
    Requires fee-aware R:R validation from RiskManager.
    """

    # Strategy parameters
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 65
    RSI_OVERSOLD = 35
    RSI_MIDLINE = 50
    EMA_SHORT = 9
    EMA_MED = 20
    EMA_LONG = 50
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    STOP_SWING_LOOKBACK = 5  # Candles to look back for swing low/high

    def __init__(self):
        self._prev_rsi: Optional[float] = None
        self._prev_macd_hist: Optional[float] = None
        self._signal_count = 0

    def analyze(self, candles: list[dict]) -> Signal:
        """
        Analyze a list of OHLCV candles and return a trading signal.
        candles: list of dicts with keys: timestamp, open, high, low, close, volume
        Expects candles sorted oldest-first.
        """
        if len(candles) < CANDLES_NEEDED:
            return Signal(
                side='NONE',
                entry_price=0,
                stop_loss=0,
                confidence=0,
                reason=f"Insufficient candles: {len(candles)} < {CANDLES_NEEDED}",
            )

        df = pd.DataFrame(candles)
        df = df.sort_values('timestamp').reset_index(drop=True)
        closes = df['close']
        highs = df['high']
        lows = df['low']

        # --- Indicators ---
        rsi = self._rsi(closes, self.RSI_PERIOD)
        ema_20 = closes.ewm(span=self.EMA_MED, adjust=False).mean()
        ema_9 = closes.ewm(span=self.EMA_SHORT, adjust=False).mean()
        macd_line, signal_line, macd_hist = self._macd(closes)

        current_close = closes.iloc[-1]
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        current_macd_hist = macd_hist.iloc[-1]
        prev_macd_hist = macd_hist.iloc[-2]
        current_ema20 = ema_20.iloc[-1]
        current_ema9 = ema_9.iloc[-1]

        # --- Signal Logic ---
        buy_signal = (
            prev_rsi < self.RSI_MIDLINE and current_rsi >= self.RSI_MIDLINE  # RSI crossing above 50
            and current_rsi < self.RSI_OVERBOUGHT                             # Not overbought
            and prev_macd_hist < 0 and current_macd_hist >= 0                 # MACD flip positive
            and current_close > current_ema20                                  # Above 20 EMA
            and current_ema9 > current_ema20                                   # Short EMA above medium
        )

        sell_signal = (
            prev_rsi > self.RSI_MIDLINE and current_rsi <= self.RSI_MIDLINE  # RSI crossing below 50
            and current_rsi > self.RSI_OVERSOLD                               # Not oversold
            and prev_macd_hist > 0 and current_macd_hist <= 0                 # MACD flip negative
            and current_close < current_ema20                                  # Below 20 EMA
            and current_ema9 < current_ema20                                   # Short EMA below medium
        )

        entry_price = current_close

        if buy_signal:
            # Stop below recent swing low
            swing_low = lows.iloc[-self.STOP_SWING_LOOKBACK:].min()
            stop_loss = swing_low * 0.998  # Small buffer below swing low
            stop_distance_pct = (entry_price - stop_loss) / entry_price

            # Require minimum stop distance (avoid noise trades)
            if stop_distance_pct < 0.005:
                return Signal(
                    side='NONE', entry_price=entry_price, stop_loss=0,
                    confidence=0, reason=f"Stop too tight: {stop_distance_pct:.2%}",
                    rsi=current_rsi, macd_hist=current_macd_hist, ema_20=current_ema20,
                )

            confidence = self._confidence_score(
                rsi=current_rsi, macd_hist=current_macd_hist,
                price_vs_ema=(current_close - current_ema20) / current_ema20,
                side='BUY'
            )

            return Signal(
                side='BUY',
                entry_price=entry_price,
                stop_loss=stop_loss,
                confidence=confidence,
                reason=(
                    f"RSI {prev_rsi:.1f}→{current_rsi:.1f} crossed 50, "
                    f"MACD hist flipped +, above EMA20"
                ),
                rsi=current_rsi,
                macd_hist=current_macd_hist,
                ema_20=current_ema20,
            )

        elif sell_signal:
            # Stop above recent swing high
            swing_high = highs.iloc[-self.STOP_SWING_LOOKBACK:].max()
            stop_loss = swing_high * 1.002
            stop_distance_pct = (stop_loss - entry_price) / entry_price

            if stop_distance_pct < 0.005:
                return Signal(
                    side='NONE', entry_price=entry_price, stop_loss=0,
                    confidence=0, reason=f"Stop too tight: {stop_distance_pct:.2%}",
                    rsi=current_rsi, macd_hist=current_macd_hist, ema_20=current_ema20,
                )

            confidence = self._confidence_score(
                rsi=current_rsi, macd_hist=current_macd_hist,
                price_vs_ema=(current_close - current_ema20) / current_ema20,
                side='SELL'
            )

            return Signal(
                side='SELL',
                entry_price=entry_price,
                stop_loss=stop_loss,
                confidence=confidence,
                reason=(
                    f"RSI {prev_rsi:.1f}→{current_rsi:.1f} crossed below 50, "
                    f"MACD hist flipped -, below EMA20"
                ),
                rsi=current_rsi,
                macd_hist=current_macd_hist,
                ema_20=current_ema20,
            )

        return Signal(
            side='NONE',
            entry_price=entry_price,
            stop_loss=0,
            confidence=0,
            reason=f"No signal. RSI={current_rsi:.1f}, MACD={current_macd_hist:.4f}, vs EMA20={((current_close-current_ema20)/current_ema20*100):.2f}%",
            rsi=current_rsi,
            macd_hist=current_macd_hist,
            ema_20=current_ema20,
        )

    # ------------------------------------------------------------------ #
    # Indicator Helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rsi(closes: pd.Series, period: int) -> pd.Series:
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def _macd(closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _confidence_score(rsi: float, macd_hist: float, price_vs_ema: float, side: str) -> float:
        """Score 0-1 based on indicator alignment strength."""
        score = 0.0
        if side == 'BUY':
            score += min((rsi - 50) / 15, 1.0) * 0.4      # RSI distance from midline
            score += min(abs(macd_hist) * 100, 1.0) * 0.3  # MACD momentum
            score += min(price_vs_ema * 10, 1.0) * 0.3     # EMA distance
        else:
            score += min((50 - rsi) / 15, 1.0) * 0.4
            score += min(abs(macd_hist) * 100, 1.0) * 0.3
            score += min(-price_vs_ema * 10, 1.0) * 0.3
        return max(0.0, min(1.0, score))
