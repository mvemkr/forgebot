"""
Forex Data Provider — yfinance

Provides OHLCV candles for Forex pairs across all timeframes.
Free, no auth required, covers all major/minor pairs.

Schwab API does not expose Forex historical data — this is the
designated data source for the Set & Forget Forex strategy.

Pair naming: standard "EUR/USD" format → auto-converted to yfinance symbols.
"""
import pandas as pd
import logging
from typing import Dict, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Standard pair → yfinance ticker symbol
PAIR_SYMBOLS = {
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "GBP/USD": "GBPUSD=X",
    "USD/CHF": "CHF=X",
    "USD/CAD": "CAD=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "GBP/JPY": "GBPJPY=X",
    "GBP/CHF": "GBPCHF=X",
    "GBP/NZD": "GBPNZD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/AUD": "EURAUD=X",
    "EUR/JPY": "EURJPY=X",
    "NZD/JPY": "NZDJPY=X",
    "NZD/CAD": "NZDCAD=X",
    "AUD/CAD": "AUDCAD=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/CAD": "EURCAD=X",
    "EUR/CHF": "EURCHF=X",
    "USD/SGD": "SGD=X",
    "USD/HKD": "HKD=X",
}

# Timeframe → yfinance (period, interval) params
# period: how far back to fetch ("1d","5d","1mo","3mo","6mo","1y","2y","5y")
# interval: candle size ("1m","5m","15m","30m","60m","1h","1d","1wk","1mo")
TIMEFRAME_MAP = {
    "1m":  ("7d",  "1m"),
    "5m":  ("60d", "5m"),
    "15m": ("60d", "15m"),
    "30m": ("60d", "30m"),
    "1h":  ("2y",  "1h"),
    "4h":  ("2y",  "1h"),    # fetch 1H, resample to 4H
    "1D":  ("5y",  "1d"),
    "1W":  ("5y",  "1wk"),
}


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV to 4H."""
    return df.resample("4h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()


class ForexData:
    """
    Forex market data via yfinance.

    Usage:
        fx = ForexData()
        df = fx.get_candles("EUR/USD", "1D", lookback=200)
        data = fx.get_multi_timeframe("USD/JPY")
    """

    def __init__(self):
        try:
            import yfinance
            self._yf = yfinance
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

    def get_candles(
        self,
        pair: str,
        timeframe: str,
        lookback: int = 200,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for a Forex pair.

        Parameters
        ----------
        pair : str        e.g. "EUR/USD"
        timeframe : str   "1m","5m","15m","30m","1h","4h","1D","1W"
        lookback : int    max candles to return

        Returns
        -------
        pd.DataFrame: open, high, low, close, volume — datetime-indexed (UTC)
        """
        symbol = PAIR_SYMBOLS.get(pair)
        if not symbol:
            # Try auto-format: "EUR/USD" → "EURUSD=X"
            symbol = pair.replace("/", "") + "=X"
            logger.warning(f"Pair {pair} not in PAIR_SYMBOLS, trying {symbol}")

        tf_params = TIMEFRAME_MAP.get(timeframe)
        if not tf_params:
            raise ValueError(f"Unknown timeframe: {timeframe}. Use: {list(TIMEFRAME_MAP.keys())}")

        period, interval = tf_params
        resample_4h = (timeframe == "4h")

        try:
            ticker = self._yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)

            if df.empty:
                logger.warning(f"No data returned for {pair} ({symbol}) {timeframe}")
                return pd.DataFrame()

            # Normalize columns
            df.columns = [c.lower() for c in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()

            # Ensure UTC index
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            df = df.sort_index()

            # Resample 1H → 4H if needed
            if resample_4h:
                df = _resample_4h(df)

            return df.iloc[-lookback:] if len(df) > lookback else df

        except Exception as e:
            logger.error(f"Error fetching {pair} {timeframe}: {e}")
            return pd.DataFrame()

    def get_multi_timeframe(
        self,
        pair: str,
        lookbacks: Optional[Dict[str, int]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all timeframes needed for Set & Forget strategy analysis.

        Returns dict: {'weekly': df, 'daily': df, '4h': df, '1h': df}
        """
        if lookbacks is None:
            lookbacks = {
                'weekly': 52,
                'daily':  200,
                '4h':     200,
                '1h':     100,
            }

        tf_map = {
            'weekly': '1W',
            'daily':  '1D',
            '4h':     '4h',
            '1h':     '1h',
        }

        result = {}
        for key, tf in tf_map.items():
            lb = lookbacks.get(key, 100)
            df = self.get_candles(pair, tf, lookback=lb)
            if df.empty:
                logger.warning(f"Empty {key} data for {pair}")
            result[key] = df

        return result

    def get_current_price(self, pair: str) -> Optional[float]:
        """Get the latest close price for a pair."""
        df = self.get_candles(pair, '1h', lookback=1)
        if not df.empty:
            return float(df['close'].iloc[-1])
        return None

    def scan_pairs(
        self,
        pairs: Optional[list] = None,
        timeframe: str = '1D',
        lookback: int = 200,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch candles for a list of pairs. Returns dict of pair → df.
        Skips pairs with no data.
        """
        if pairs is None:
            pairs = list(PAIR_SYMBOLS.keys())

        result = {}
        for pair in pairs:
            df = self.get_candles(pair, timeframe, lookback=lookback)
            if not df.empty:
                result[pair] = df
            else:
                logger.warning(f"Skipping {pair} — no data")
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fx = ForexData()

    watchlist = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "GBP/JPY"]
    print("Fetching multi-timeframe data for watchlist...\n")
    for pair in watchlist:
        data = fx.get_multi_timeframe(pair)
        w = data['weekly']
        d = data['daily']
        h4 = data['4h']
        h1 = data['1h']
        price = d['close'].iloc[-1] if not d.empty else 0
        print(f"{pair:12} | price={price:.5f} | W:{len(w)} D:{len(d)} 4H:{len(h4)} 1H:{len(h1)} candles")
