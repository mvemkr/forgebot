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
def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV dataframe to a higher timeframe (e.g. '1h', '4h')."""
    resampled = df.resample(rule).agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }).dropna()
    return resampled


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

import schwab as _schwab
_PH = _schwab.client.Client.PriceHistory

# Period enum aliases (values are ints: 1,2,3,4,5,10,6,15,20)
# When used with PeriodType.DAY  → 1..10 days
# When used with PeriodType.YEAR → 1..20 years
_P = _PH.Period

# Maps timeframe → (period_type, period_enum, frequency_type, frequency_enum, resample_rule)
# resample_rule: if set, fetch 30m data and resample to this pandas offset string
FREQUENCY_MAP = {
    "1m":  (_PH.PeriodType.DAY,  _P.THREE_DAYS,   _PH.FrequencyType.MINUTE, _PH.Frequency.EVERY_MINUTE,          None),
    "5m":  (_PH.PeriodType.DAY,  _P.FIVE_DAYS,    _PH.FrequencyType.MINUTE, _PH.Frequency.EVERY_FIVE_MINUTES,    None),
    "15m": (_PH.PeriodType.DAY,  _P.TEN_DAYS,     _PH.FrequencyType.MINUTE, _PH.Frequency.EVERY_FIFTEEN_MINUTES, None),
    "30m": (_PH.PeriodType.DAY,  _P.TEN_DAYS,     _PH.FrequencyType.MINUTE, _PH.Frequency.EVERY_THIRTY_MINUTES,  None),
    "1h":  (_PH.PeriodType.DAY,  _P.TEN_DAYS,     _PH.FrequencyType.MINUTE, _PH.Frequency.EVERY_THIRTY_MINUTES,  "1h"),
    "4h":  (_PH.PeriodType.DAY,  _P.TEN_DAYS,     _PH.FrequencyType.MINUTE, _PH.Frequency.EVERY_THIRTY_MINUTES,  "4h"),
    "1D":  (_PH.PeriodType.YEAR, _P.ONE_DAY,      _PH.FrequencyType.DAILY,  None,                                None),
    "1W":  (_PH.PeriodType.YEAR, _P.FIVE_DAYS,    _PH.FrequencyType.WEEKLY, None,                                None),
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
        freq_entry = FREQUENCY_MAP.get(timeframe)
        if not freq_entry:
            raise ValueError(f"Unknown timeframe: {timeframe}. Use: {list(FREQUENCY_MAP.keys())}")

        period_type, period, freq_type, freq, resample_rule = freq_entry

        is_intraday = (freq is not None)

        try:
            if is_intraday:
                resp = self._client.get_price_history(
                    symbol=symbol,
                    period_type=period_type,
                    period=period,
                    frequency_type=freq_type,
                    frequency=freq,
                    need_extended_hours_data=False,
                )
            else:
                # Daily/weekly: frequency enum not required
                resp = self._client.get_price_history(
                    symbol=symbol,
                    period_type=period_type,
                    period=period,
                    frequency_type=freq_type,
                    need_extended_hours_data=False,
                )

            if resp.status_code != 200:
                logger.error(f"Schwab candle request failed {pair} {timeframe}: {resp.status_code} {resp.text[:300]}")
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

            # Resample for 1H / 4H (built from 30m data)
            if resample_rule:
                df = _resample_ohlcv(df, resample_rule)

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

        import schwab as _s
        _Fields = _s.client.Client.Account.Fields
        try:
            if account_hash:
                resp = self._client.get_account(account_hash, fields=[_Fields.POSITIONS])
            else:
                resp = self._client.get_accounts(fields=[_Fields.POSITIONS])

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
        import schwab as _s
        _Fields = _s.client.Client.Account.Fields
        try:
            resp = self._client.get_account(account_hash, fields=[_Fields.POSITIONS])
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


    # ------------------------------------------------------------------ #
    # ES / MES Futures
    # ------------------------------------------------------------------ #

    # Active front-month contracts — update each rollover
    ES_SYMBOL  = "/ESM26"
    MES_SYMBOL = "/MESM26"

    # Futures-compatible timeframe → (frequencyType, frequency) pairs
    _FUTURES_FREQ = {
        "5m":  (_PH.FrequencyType.MINUTE, _PH.Frequency.EVERY_FIVE_MINUTES),
        "15m": (_PH.FrequencyType.MINUTE, _PH.Frequency.EVERY_FIFTEEN_MINUTES),
        "1h":  (_PH.FrequencyType.MINUTE, _PH.Frequency.EVERY_THIRTY_MINUTES),  # resample
        "1D":  (_PH.FrequencyType.DAILY,  None),
    }

    def get_es_candles(
        self,
        timeframe: str = "15m",
        lookback:  int = 100,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for /MESM26 (Micro ES).

        Parameters
        ----------
        timeframe : "5m" | "15m" | "1h" | "1D"
        lookback  : number of candles to return

        Returns
        -------
        pd.DataFrame  columns: open, high, low, close, volume  (UTC index)
        """
        resample = None
        if timeframe == "1h":
            resample = "1h"
            freq_type, freq = self._FUTURES_FREQ["15m"]  # fetch 15m then resample
        elif timeframe in self._FUTURES_FREQ:
            freq_type, freq = self._FUTURES_FREQ[timeframe]
        else:
            raise ValueError(f"Unsupported futures timeframe: {timeframe}. Use 5m/15m/1h/1D")

        period_type = (_PH.PeriodType.DAY if timeframe != "1D" else _PH.PeriodType.YEAR)
        period      = (_P.TEN_DAYS        if timeframe != "1D" else _P.ONE_DAY)

        try:
            if freq is not None:
                resp = self._client.get_price_history(
                    symbol=self.MES_SYMBOL,
                    period_type=period_type,
                    period=period,
                    frequency_type=freq_type,
                    frequency=freq,
                    need_extended_hours_data=True,   # futures trade nearly 24h
                )
            else:
                resp = self._client.get_price_history(
                    symbol=self.MES_SYMBOL,
                    period_type=period_type,
                    period=period,
                    frequency_type=freq_type,
                    need_extended_hours_data=True,
                )

            if resp.status_code != 200:
                logger.error(f"ES candle request failed {timeframe}: {resp.status_code}")
                return pd.DataFrame()

            data    = resp.json()
            candles = data.get("candles", [])
            if not candles:
                logger.warning(f"No ES candles for {timeframe}")
                return pd.DataFrame()

            df = pd.DataFrame(candles)
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
            df = df.set_index("datetime")
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df = df.sort_index()

            if resample:
                df = _resample_ohlcv(df, resample)

            return df.iloc[-lookback:]

        except Exception as e:
            logger.error(f"Error fetching ES candles ({timeframe}): {e}")
            return pd.DataFrame()

    def get_es_quote(self) -> Dict:
        """
        Get current bid/ask/mid/last for /ESM26.

        Returns
        -------
        dict  {"bid": float, "ask": float, "mid": float, "last": float}
              Values are 0.0 on failure — never raises.
        """
        try:
            # get_quote() (singular) returns 404 for futures — must use get_quotes()
            resp = self._client.get_quotes([self.ES_SYMBOL])
            if resp.status_code == 200:
                data  = resp.json()
                quote = data.get(self.ES_SYMBOL, {}).get("quote", {})
                bid  = float(quote.get("bidPrice",  0) or 0)
                ask  = float(quote.get("askPrice",  0) or 0)
                last = float(quote.get("lastPrice", 0) or 0)
                mid  = (bid + ask) / 2 if (bid and ask) else last
                return {"bid": bid, "ask": ask, "mid": mid, "last": last}
        except Exception as e:
            logger.error(f"ES quote fetch failed: {e}")
        return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "last": 0.0}

    def place_futures_order(
        self,
        direction:    str,                    # "long" or "short"
        quantity:     int,                    # number of contracts
        order_type:   str = "LIMIT",          # "LIMIT" or "MARKET"
        limit_price:  Optional[float] = None,
        stop_price:   Optional[float] = None,
        account_hash: Optional[str]   = None,
        dry_run:      bool             = True,
    ) -> Dict:
        """
        Place a futures order for /ESM26 or /MESM26 with optional attached stop.

        Parameters
        ----------
        direction   : "long"  → BUY to open, SELL to stop
                      "short" → SELL to open, BUY to stop
        quantity    : integer number of contracts
        order_type  : "LIMIT" or "MARKET"
        limit_price : required when order_type="LIMIT"
        stop_price  : if provided, attached as childOrderStrategy stop
        dry_run     : True (default) — logs without submitting

        Safety
        ------
        dry_run=True by default.  Must be explicitly False to submit live.
        """
        if account_hash is None:
            account_hash = os.getenv("SCHWAB_ACCOUNT_HASH")

        import os as _os
        use_mes = _os.getenv("ES_USE_MES", "true").lower() != "false"
        symbol  = self.MES_SYMBOL if use_mes else self.ES_SYMBOL

        instruction       = "BUY_TO_OPEN"  if direction == "long"  else "SELL_TO_OPEN"
        stop_instruction  = "SELL_TO_CLOSE" if direction == "long"  else "BUY_TO_CLOSE"

        leg = {
            "instruction": instruction,
            "quantity": quantity,
            "instrument": {
                "symbol":    symbol,
                "assetType": "FUTURE",
            },
        }

        order: Dict = {
            "orderType":         order_type,
            "session":           "NORMAL",
            "duration":          "DAY",
            "orderStrategyType": "SINGLE" if stop_price is None else "TRIGGER",
            "orderLegCollection": [leg],
        }

        if order_type == "LIMIT":
            if limit_price is None:
                raise ValueError("limit_price required for LIMIT orders")
            order["price"] = str(round(limit_price, 2))

        if stop_price is not None:
            order["orderStrategyType"] = "TRIGGER"
            order["childOrderStrategies"] = [{
                "orderType":         "STOP",
                "session":           "NORMAL",
                "duration":          "DAY",
                "orderStrategyType": "SINGLE",
                "stopPrice":         str(round(stop_price, 2)),
                "orderLegCollection": [{
                    "instruction": stop_instruction,
                    "quantity":    quantity,
                    "instrument":  {
                        "symbol":    symbol,
                        "assetType": "FUTURE",
                    },
                }],
            }]

        log_msg = (
            f"FUTURES ORDER: {instruction} {quantity}× {symbol}  "
            f"type={order_type}  "
            + (f"limit={limit_price:.2f}  " if limit_price else "")
            + (f"stop={stop_price:.2f}" if stop_price else "")
        )

        if dry_run:
            logger.info(f"[DRY RUN] {log_msg}")
            return {"dry_run": True, "order": order, "symbol": symbol}

        logger.info(f"[LIVE] Submitting: {log_msg}")
        try:
            resp = self._client.place_order(account_hash, order)
            if resp.status_code in (200, 201):
                logger.info("Futures order placed successfully")
                return {"success": True, "status": resp.status_code, "order": order}
            else:
                logger.error(f"Futures order failed: {resp.status_code} {resp.text[:300]}")
                return {"success": False, "status": resp.status_code, "error": resp.text}
        except Exception as e:
            logger.error(f"Futures order exception: {e}")
            return {"success": False, "error": str(e)}

    def get_futures_positions(self, account_hash: Optional[str] = None) -> list:
        """
        Return open futures positions only (assetType == 'FUTURE').

        Filters get_positions() to futures instruments.
        """
        all_positions = self.get_positions(account_hash=account_hash)
        return [
            p for p in all_positions
            if p.get("instrument", {}).get("assetType", "").upper() == "FUTURE"
        ]


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
