"""
Schwab Client — Market data + account management via schwab-py

Wraps the schwab-py library to provide:
  - OHLCV candles (multi-timeframe) for Forex and equities
  - Account balance and positions
  - Order placement (limit orders only — no market orders)
  - Token refresh (auto-handled by schwab-py)

Usage:
    client = SchwabClient()  # loads token from .schwab_token.json
    df = client.get_candles("EUR/USD", "1D", lookback=200)
"""
import os, json, logging
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load env from trading-bot root
_ROOT = Path(__file__).parents[2]
load_dotenv(_ROOT / ".env")

APP_KEY    = os.getenv("SCHWAB_APP_KEY")
APP_SECRET = os.getenv("SCHWAB_APP_SECRET")
TOKEN_PATH = _ROOT / ".schwab_token.json"

# Schwab uses its own pair naming convention
PAIR_MAP = {
    # Standard → Schwab symbol format
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
    "GBP/USD": "GBP/USD",
    "USD/CHF": "USD/CHF",
    "USD/CAD": "USD/CAD",
    "AUD/USD": "AUD/USD",
    "NZD/USD": "NZD/USD",
    "GBP/JPY": "GBP/JPY",
    "GBP/CHF": "GBP/CHF",
    "GBP/NZD": "GBP/NZD",
    "EUR/GBP": "EUR/GBP",
    "EUR/AUD": "EUR/AUD",
    "NZD/JPY": "NZD/JPY",
    "NZD/CAD": "NZD/CAD",
    "AUD/CAD": "AUD/CAD",
    "AUD/JPY": "AUD/JPY",
}

# Schwab frequency types for candle requests
FREQUENCY_MAP = {
    "1m":  {"frequencyType": "minute",  "frequency": 1},
    "5m":  {"frequencyType": "minute",  "frequency": 5},
    "15m": {"frequencyType": "minute",  "frequency": 15},
    "30m": {"frequencyType": "minute",  "frequency": 30},
    "1h":  {"frequencyType": "minute",  "frequency": 60},
    "4h":  {"frequencyType": "minute",  "frequency": 240},
    "1D":  {"frequencyType": "daily",   "frequency": 1},
    "1W":  {"frequencyType": "weekly",  "frequency": 1},
}


class SchwabClient:
    """
    Schwab API client wrapping schwab-py for trading and market data.
    """

    def __init__(self, token_path: Optional[str] = None):
        self.token_path = Path(token_path) if token_path else TOKEN_PATH
        self._client = None
        self._connect()

    def _connect(self):
        """Load token and initialize schwab-py client."""
        if not APP_KEY or not APP_SECRET:
            raise ValueError("SCHWAB_APP_KEY and SCHWAB_APP_SECRET must be set in .env")

        if not self.token_path.exists():
            raise FileNotFoundError(
                f"Schwab token not found at {self.token_path}. "
                "Run: python scripts/schwab_auth_manual.py"
            )

        try:
            import schwab
            self._client = schwab.auth.client_from_token_file(
                token_path=str(self.token_path),
                api_key=APP_KEY,
                app_secret=APP_SECRET,
            )
            logger.info("Schwab client connected")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Schwab: {e}")

    def is_connected(self) -> bool:
        try:
            resp = self._client.get_account_numbers()
            return resp.status_code == 200
        except:
            return False

    # ------------------------------------------------------------------ #
    # Market Data
    # ------------------------------------------------------------------ #

    def get_candles(
        self,
        pair: str,
        timeframe: str,
        lookback: int = 200,
    ) -> pd.DataFrame:
        """
        Get OHLCV candles for a Forex pair.

        Parameters
        ----------
        pair : str
            e.g. "EUR/USD", "USD/JPY"
        timeframe : str
            "1m", "5m", "15m", "30m", "1h", "4h", "1D", "1W"
        lookback : int
            Number of candles to return

        Returns
        -------
        pd.DataFrame with columns: open, high, low, close, volume
        Indexed by datetime (UTC).
        """
        symbol = PAIR_MAP.get(pair, pair)
        freq_params = FREQUENCY_MAP.get(timeframe)
        if not freq_params:
            raise ValueError(f"Unknown timeframe: {timeframe}. Use: {list(FREQUENCY_MAP.keys())}")

        # Calculate date range
        end_dt = datetime.utcnow()
        if freq_params["frequencyType"] == "weekly":
            start_dt = end_dt - timedelta(weeks=lookback * 2)
            period_type = "year"
            period = max(1, lookback // 52 + 1)
        elif freq_params["frequencyType"] == "daily":
            start_dt = end_dt - timedelta(days=lookback * 2)
            period_type = "month"
            period = max(1, lookback // 22 + 1)
        else:
            start_dt = end_dt - timedelta(minutes=freq_params["frequency"] * lookback * 2)
            period_type = "day"
            period = max(1, (freq_params["frequency"] * lookback) // (24 * 60) + 2)

        try:
            resp = self._client.get_price_history(
                symbol=symbol,
                period_type=period_type,
                period=period,
                frequency_type=freq_params["frequencyType"],
                frequency=freq_params["frequency"],
                need_extended_hours_data=False,
            )

            if resp.status_code != 200:
                logger.error(f"Schwab candle request failed: {resp.status_code} {resp.text[:200]}")
                return pd.DataFrame()

            data = resp.json()
            candles = data.get("candles", [])
            if not candles:
                logger.warning(f"No candles returned for {pair} {timeframe}")
                return pd.DataFrame()

            df = pd.DataFrame(candles)
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
            df = df.set_index('datetime')
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df = df.sort_index()

            # Return last N candles
            return df.iloc[-lookback:]

        except Exception as e:
            logger.error(f"Error fetching candles for {pair} {timeframe}: {e}")
            return pd.DataFrame()

    def get_multi_timeframe(
        self, pair: str, lookbacks: Optional[Dict[str, int]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all timeframes needed for strategy analysis.

        Returns dict with keys: 'weekly', 'daily', '4h', '1h'
        """
        if lookbacks is None:
            lookbacks = {
                'weekly': 52,   # 1 year of weekly
                'daily':  200,  # ~10 months of daily
                '4h':     200,  # ~33 days of 4H
                '1h':     100,  # ~4 days of 1H
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
                logger.warning(f"No data for {pair} {key} ({tf})")
            result[key] = df

        return result

    def get_quote(self, pair: str) -> Optional[float]:
        """Get current bid/ask midpoint for a pair."""
        symbol = PAIR_MAP.get(pair, pair)
        try:
            resp = self._client.get_quote(symbol)
            if resp.status_code == 200:
                data = resp.json()
                quote = data.get(symbol, {}).get('quote', {})
                bid = quote.get('bidPrice', 0)
                ask = quote.get('askPrice', 0)
                return (bid + ask) / 2 if bid and ask else None
        except Exception as e:
            logger.error(f"Quote fetch failed for {pair}: {e}")
        return None

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #

    def get_account_balance(self, account_hash: Optional[str] = None) -> Dict:
        """
        Returns account balance info.
        """
        if account_hash is None:
            account_hash = os.getenv("SCHWAB_ACCOUNT_HASH")

        try:
            if account_hash:
                resp = self._client.get_account(account_hash, fields=['positions'])
            else:
                resp = self._client.get_accounts(fields=['positions'])

            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    data = data[0] if data else {}
                account = data.get('securitiesAccount', {})
                balance = account.get('currentBalances', {})
                return {
                    'cash':        balance.get('cashBalance', 0),
                    'equity':      balance.get('liquidationValue', 0),
                    'buying_power': balance.get('buyingPower', 0),
                    'margin_balance': balance.get('marginBalance', 0),
                }
        except Exception as e:
            logger.error(f"Account balance fetch failed: {e}")
        return {}

    def get_positions(self, account_hash: Optional[str] = None) -> list:
        """Return list of open positions."""
        if account_hash is None:
            account_hash = os.getenv("SCHWAB_ACCOUNT_HASH")
        try:
            resp = self._client.get_account(account_hash, fields=['positions'])
            if resp.status_code == 200:
                data = resp.json()
                account = data.get('securitiesAccount', {})
                return account.get('positions', [])
        except Exception as e:
            logger.error(f"Positions fetch failed: {e}")
        return []

    # ------------------------------------------------------------------ #
    # Orders (Forex — limit only)
    # ------------------------------------------------------------------ #

    def place_forex_limit_order(
        self,
        pair: str,
        direction: str,       # 'long' or 'short'
        quantity: float,      # lot size
        limit_price: float,
        stop_price: float,
        account_hash: Optional[str] = None,
        dry_run: bool = True,
    ) -> Dict:
        """
        Place a Forex limit order with attached stop loss.
        dry_run=True (default) — logs the order but does NOT submit.

        IMPORTANT: dry_run must be explicitly set to False to send live orders.
        This is a safety feature — live orders require deliberate opt-in.
        """
        if account_hash is None:
            account_hash = os.getenv("SCHWAB_ACCOUNT_HASH")

        symbol = PAIR_MAP.get(pair, pair)
        instruction = "BUY" if direction == "long" else "SELL"

        order = {
            "orderType": "LIMIT",
            "session": "NORMAL",
            "duration": "GOOD_TILL_CANCEL",
            "orderStrategyType": "TRIGGER",
            "price": str(round(limit_price, 5)),
            "orderLegCollection": [{
                "instruction": instruction,
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol,
                    "assetType": "FOREX",
                },
            }],
            "childOrderStrategies": [{
                "orderType": "STOP",
                "session": "NORMAL",
                "duration": "GOOD_TILL_CANCEL",
                "orderStrategyType": "SINGLE",
                "stopPrice": str(round(stop_price, 5)),
                "orderLegCollection": [{
                    "instruction": "SELL" if direction == "long" else "BUY",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "FOREX",
                    },
                }],
            }],
        }

        log_msg = (
            f"FOREX ORDER: {instruction} {quantity} lots {symbol} "
            f"@ LIMIT {limit_price:.5f}  SL={stop_price:.5f}"
        )

        if dry_run:
            logger.info(f"[DRY RUN] {log_msg}")
            return {"dry_run": True, "order": order}

        logger.info(f"[LIVE] Submitting: {log_msg}")
        try:
            resp = self._client.place_order(account_hash, order)
            if resp.status_code in (200, 201):
                logger.info(f"Order placed successfully")
                return {"success": True, "status": resp.status_code}
            else:
                logger.error(f"Order failed: {resp.status_code} {resp.text[:300]}")
                return {"success": False, "status": resp.status_code, "error": resp.text}
        except Exception as e:
            logger.error(f"Order exception: {e}")
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        client = SchwabClient()
        print(f"Connected: {client.is_connected()}")
        balance = client.get_account_balance()
        print(f"Balance: {balance}")
        df = client.get_candles("EUR/USD", "1D", lookback=10)
        print(f"EUR/USD daily candles:\n{df.tail()}")
    except Exception as e:
        print(f"Error: {e}")
