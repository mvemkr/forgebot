"""
OANDA REST API Client (v20)

Handles all Forex execution:
  - Live bid/ask quotes
  - Market + limit order placement (limit orders ONLY for strategy)
  - Stop loss attachment
  - Position monitoring and closing
  - Account balance and open trade tracking

Live account: 001-001-20761942-001
Leverage: 50:1 (US retail cap)
Base URL: https://api-fxtrade.oanda.com

SAFETY: All order methods default to dry_run=True.
        Must explicitly pass dry_run=False to place live orders.
"""
import os, requests, logging
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv
from datetime import datetime

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[2]
load_dotenv(_ROOT / ".env")

LIVE_BASE     = "https://api-fxtrade.oanda.com"
PRACTICE_BASE = "https://api-fxpractice.oanda.com"

# Standard pair → OANDA instrument name
INSTRUMENT_MAP = {
    "EUR/USD": "EUR_USD",
    "USD/JPY": "USD_JPY",
    "GBP/USD": "GBP_USD",
    "USD/CHF": "USD_CHF",
    "USD/CAD": "USD_CAD",
    "AUD/USD": "AUD_USD",
    "NZD/USD": "NZD_USD",
    "GBP/JPY": "GBP_JPY",
    "GBP/CHF": "GBP_CHF",
    "GBP/NZD": "GBP_NZD",
    "EUR/GBP": "EUR_GBP",
    "EUR/AUD": "EUR_AUD",
    "EUR/JPY": "EUR_JPY",
    "NZD/JPY": "NZD_JPY",
    "NZD/CAD": "NZD_CAD",
    "AUD/CAD": "AUD_CAD",
    "AUD/JPY": "AUD_JPY",
    "EUR/CAD": "EUR_CAD",
    "EUR/CHF": "EUR_CHF",
}


class OandaClient:
    """
    OANDA v20 REST API client for Forex execution.

    Parameters
    ----------
    env : str
        'live' or 'practice'
    account_id : str
        OANDA account ID (overrides .env if provided)
    api_key : str
        OANDA API token (overrides .env if provided)
    """

    def __init__(
        self,
        env: Optional[str] = None,
        account_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.api_key    = api_key    or os.getenv("OANDA_API_KEY")
        self.account_id = account_id or os.getenv("OANDA_ACCOUNT_ID")
        self.env        = env        or os.getenv("OANDA_ENV", "live")
        self.base       = LIVE_BASE if self.env == "live" else PRACTICE_BASE

        if not self.api_key:
            raise ValueError("OANDA_API_KEY not set in .env")
        if not self.account_id:
            raise ValueError("OANDA_ACCOUNT_ID not set in .env")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

    def _get(self, path: str, params: dict = None) -> dict:
        resp = requests.get(
            f"{self.base}{path}", headers=self.headers,
            params=params, timeout=10
        )
        if resp.status_code != 200:
            logger.error(f"GET {path} → {resp.status_code}: {resp.text[:300]}")
            return {}
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        resp = requests.post(
            f"{self.base}{path}", headers=self.headers,
            json=body, timeout=10
        )
        if resp.status_code not in (200, 201):
            logger.error(f"POST {path} → {resp.status_code}: {resp.text[:300]}")
            return {"error": resp.text, "status": resp.status_code}
        return resp.json()

    def _put(self, path: str, body: dict) -> dict:
        resp = requests.put(
            f"{self.base}{path}", headers=self.headers,
            json=body, timeout=10
        )
        return resp.json()

    # ── Account ──────────────────────────────────────────────────────

    def get_account_summary(self) -> Dict:
        data = self._get(f"/v3/accounts/{self.account_id}/summary")
        acct = data.get("account", {})
        return {
            "balance":        float(acct.get("balance", 0)),
            "nav":            float(acct.get("NAV", 0)),
            "unrealized_pnl": float(acct.get("unrealizedPL", 0)),
            "margin_used":    float(acct.get("marginUsed", 0)),
            "margin_avail":   float(acct.get("marginAvailable", 0)),
            "open_trades":    int(acct.get("openTradeCount", 0)),
            "leverage":       f"{1/float(acct.get('marginRate', 0.02)):.0f}:1",
            "currency":       acct.get("currency", "USD"),
        }

    def get_open_trades(self) -> List[Dict]:
        data = self._get(f"/v3/accounts/{self.account_id}/openTrades")
        trades = []
        for t in data.get("trades", []):
            trades.append({
                "id":          t["id"],
                "instrument":  t["instrument"],
                "direction":   "long" if float(t["currentUnits"]) > 0 else "short",
                "units":       abs(float(t["currentUnits"])),
                "entry":       float(t["price"]),
                "current":     float(t.get("unrealizedPL", 0)),
                "stop_loss":   float(t["stopLossOrder"]["price"]) if "stopLossOrder" in t else None,
                "take_profit": float(t["takeProfitOrder"]["price"]) if "takeProfitOrder" in t else None,
                "open_time":   t.get("openTime"),
            })
        return trades

    # ── Market Data ───────────────────────────────────────────────────

    def get_quote(self, pair: str) -> Dict:
        """Get live bid/ask for a pair."""
        instrument = INSTRUMENT_MAP.get(pair, pair.replace("/", "_"))
        data = self._get(
            f"/v3/accounts/{self.account_id}/pricing",
            params={"instruments": instrument}
        )
        prices = data.get("prices", [])
        if not prices:
            return {}
        p = prices[0]
        bids = p.get("bids", [{}])
        asks = p.get("asks", [{}])
        bid = float(bids[0].get("price", 0))
        ask = float(asks[0].get("price", 0))
        return {
            "pair":   pair,
            "bid":    bid,
            "ask":    ask,
            "mid":    (bid + ask) / 2,
            "spread": ask - bid,
            "tradeable": p.get("tradeable", False),
        }

    def get_candles(
        self,
        pair: str,
        granularity: str = "D",
        count: int = 200,
    ) -> list:
        """
        Get OHLCV candles from OANDA.
        Granularity: S5,S10,S15,S30,M1,M2,M4,M5,M10,M15,M30,H1,H2,H3,H4,H6,H8,H12,D,W,M
        """
        instrument = INSTRUMENT_MAP.get(pair, pair.replace("/", "_"))
        data = self._get(
            f"/v3/instruments/{instrument}/candles",
            params={"granularity": granularity, "count": count, "price": "M"}
        )
        candles = []
        for c in data.get("candles", []):
            mid = c.get("mid", {})
            candles.append({
                "time":   c["time"],
                "open":   float(mid.get("o", 0)),
                "high":   float(mid.get("h", 0)),
                "low":    float(mid.get("l", 0)),
                "close":  float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })
        return candles

    # ── Orders ────────────────────────────────────────────────────────

    def place_limit_order(
        self,
        pair: str,
        direction: str,        # 'long' or 'short'
        units: int,            # number of units (not lots — OANDA uses units)
        limit_price: float,
        stop_loss: float,
        take_profit: Optional[float] = None,  # None = no TP (set & forget)
        dry_run: bool = True,
    ) -> Dict:
        """
        Place a limit order with attached stop loss.
        OANDA units: 1 standard lot = 100,000 units; 1 mini = 10,000; 1 micro = 1,000

        dry_run=True (default) — logs without submitting. 
        Set dry_run=False deliberately to go live.
        """
        instrument = INSTRUMENT_MAP.get(pair, pair.replace("/", "_"))
        signed_units = units if direction == "long" else -units
        decimals = 3 if "JPY" in pair else 5

        order_body = {
            "order": {
                "type":       "LIMIT",
                "instrument": instrument,
                "units":      str(signed_units),
                "price":      str(round(limit_price, decimals)),
                "timeInForce": "GTC",
                "stopLossOnFill": {
                    "price": str(round(stop_loss, decimals)),
                    "timeInForce": "GTC",
                },
            }
        }

        if take_profit is not None:
            order_body["order"]["takeProfitOnFill"] = {
                "price": str(round(take_profit, decimals)),
                "timeInForce": "GTC",
            }

        log_msg = (
            f"OANDA LIMIT ORDER: {direction.upper()} {units:,} units {pair} "
            f"@ {limit_price:.{decimals}f}  "
            f"SL={stop_loss:.{decimals}f}  "
            f"TP={'None (set&forget)' if take_profit is None else f'{take_profit:.{decimals}f}'}"
        )

        if dry_run:
            logger.info(f"[DRY RUN] {log_msg}")
            return {"dry_run": True, "order": order_body, "message": log_msg}

        logger.info(f"[LIVE] Submitting: {log_msg}")
        result = self._post(f"/v3/accounts/{self.account_id}/orders", order_body)

        if "orderFillTransaction" in result:
            fill = result["orderFillTransaction"]
            logger.info(f"Order filled at {fill.get('price')} — trade ID {fill.get('tradeOpened', {}).get('tradeID')}")
        elif "orderCreateTransaction" in result:
            logger.info(f"Limit order created — waiting for fill")
        elif "error" in result:
            logger.error(f"Order failed: {result}")

        return result

    def place_market_order(
        self,
        pair: str,
        direction: str,
        units: int,
        stop_loss: float,
        take_profit: Optional[float] = None,
        dry_run: bool = True,
    ) -> Dict:
        """
        Market order for immediate fill.
        NOTE: Strategy prefers limit orders. Use market only for emergency closes.
        """
        instrument = INSTRUMENT_MAP.get(pair, pair.replace("/", "_"))
        signed_units = units if direction == "long" else -units
        decimals = 3 if "JPY" in pair else 5

        order_body = {
            "order": {
                "type":       "MARKET",
                "instrument": instrument,
                "units":      str(signed_units),
                "timeInForce": "FOK",
                "stopLossOnFill": {
                    "price": str(round(stop_loss, decimals)),
                    "timeInForce": "GTC",
                },
            }
        }
        if take_profit:
            order_body["order"]["takeProfitOnFill"] = {
                "price": str(round(take_profit, decimals)),
                "timeInForce": "GTC",
            }

        if dry_run:
            logger.info(f"[DRY RUN] MARKET {direction.upper()} {units} {pair} SL={stop_loss:.{decimals}f}")
            return {"dry_run": True, "order": order_body}

        return self._post(f"/v3/accounts/{self.account_id}/orders", order_body)

    def close_trade(self, trade_id: str, dry_run: bool = True) -> Dict:
        """Close a specific open trade by ID."""
        if dry_run:
            logger.info(f"[DRY RUN] Would close trade {trade_id}")
            return {"dry_run": True, "trade_id": trade_id}
        return self._put(
            f"/v3/accounts/{self.account_id}/trades/{trade_id}/close",
            {"units": "ALL"}
        )

    def move_stop_to_breakeven(self, trade_id: str, entry_price: float, dry_run: bool = True) -> Dict:
        """
        Move stop loss to entry price (breakeven).
        Only call after trade is at 1:1 profit.
        NEVER move stop further from entry — that was the $50K lesson.
        """
        decimals = 5
        if dry_run:
            logger.info(f"[DRY RUN] Moving SL to breakeven ({entry_price:.{decimals}f}) for trade {trade_id}")
            return {"dry_run": True}

        return self._put(
            f"/v3/accounts/{self.account_id}/trades/{trade_id}/orders",
            {"stopLoss": {"price": str(round(entry_price, decimals)), "timeInForce": "GTC"}}
        )

    # ── Position Sizing ───────────────────────────────────────────────

    def calculate_units(
        self,
        pair: str,
        account_balance: float,
        risk_pct: float,
        stop_pips: float,
    ) -> int:
        """
        Calculate units to trade given risk parameters.

        For most pairs: 1 pip = 0.0001 price movement
        For JPY pairs:  1 pip = 0.01 price movement
        1 standard lot = 100,000 units → pip value ≈ $10/pip

        Formula: units = (account * risk_pct) / (stop_pips * pip_value_per_unit)
        """
        risk_dollars = account_balance * (risk_pct / 100)
        is_jpy = "JPY" in pair
        pip_size = 0.01 if is_jpy else 0.0001

        # Approximate pip value per unit in USD
        # For USD-quoted pairs (EUR/USD, GBP/USD): $0.0001 per unit
        # For USD-base pairs (USD/JPY, USD/CHF): depends on rate (approx $0.0001)
        pip_value_per_unit = pip_size  # simplified; accurate for USD-quoted pairs

        if stop_pips <= 0:
            return 1000  # fallback micro lot

        units = int(risk_dollars / (stop_pips * pip_value_per_unit))

        # Enforce minimums and maximums
        units = max(1000, units)    # min 1 micro lot
        units = min(units, 100000) # max 1 standard lot until account grows

        return units


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = OandaClient()
    print("Testing OANDA connection...\n")

    summary = client.get_account_summary()
    print("Account Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nLive Quotes:")
    for pair in ["EUR/USD", "USD/JPY", "GBP/USD", "GBP/JPY"]:
        q = client.get_quote(pair)
        if q:
            print(f"  {pair}: bid={q['bid']:.5f} ask={q['ask']:.5f} spread={q['spread']*10000:.1f} pips tradeable={q['tradeable']}")

    print("\nPosition sizing example (1% risk, 20 pip stop, $4,000 balance):")
    units = client.calculate_units("EUR/USD", 4000, 1.0, 20)
    print(f"  EUR/USD: {units:,} units ({units/100000:.2f} lots)")
