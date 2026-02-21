"""
Coinbase Advanced Trade API client wrapper.
Handles REST + WebSocket for both spot and futures.
"""

import os
import logging
import time
from typing import Optional
from dotenv import load_dotenv
from coinbase.rest import RESTClient

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

logger = logging.getLogger(__name__)


class CoinbaseClient:
    """
    Thin wrapper around the Coinbase Advanced Trade REST API.
    All methods return plain dicts. No business logic here.
    """

    def __init__(self):
        key_name = os.getenv("COINBASE_API_KEY_NAME")
        private_key = os.getenv("COINBASE_API_PRIVATE_KEY")
        if not key_name or not private_key:
            raise ValueError("Missing COINBASE_API_KEY_NAME or COINBASE_API_PRIVATE_KEY in .env")
        self._client = RESTClient(api_key=key_name, api_secret=private_key)
        logger.info("CoinbaseClient initialized")

    # ------------------------------------------------------------------ #
    # Accounts & Balances                                                  #
    # ------------------------------------------------------------------ #

    def get_accounts(self) -> list[dict]:
        """Return all accounts with non-zero balances."""
        raw = self._client.get_accounts(limit=250)['accounts']
        result = []
        for a in raw:
            d = a.__dict__
            result.append({
                'uuid': d.get('uuid'),
                'currency': d.get('currency'),
                'available': float(d['available_balance']['value']),
                'hold': float(d['hold']['value']),
                'type': d.get('type'),
            })
        return result

    def get_balance(self, currency: str) -> dict:
        """Return available and hold for a specific currency."""
        for a in self.get_accounts():
            if a['currency'] == currency:
                return a
        return {'currency': currency, 'available': 0.0, 'hold': 0.0}

    def get_futures_balance(self) -> dict:
        """Return the futures account balance summary."""
        try:
            raw = self._client.get_futures_balance_summary()
            d = raw.__dict__ if hasattr(raw, '__dict__') else {}
            summary = d.get('balance_summary', {})
            if hasattr(summary, '__dict__'):
                summary = summary.__dict__

            def _val(obj):
                if isinstance(obj, dict):
                    return float(obj.get('value', 0))
                if hasattr(obj, '__dict__'):
                    return float(obj.__dict__.get('value', 0))
                return 0.0

            return {
                'futures_buying_power': _val(summary.get('futures_buying_power', {})),
                'total_usd_balance': _val(summary.get('total_usd_balance', {})),
                'cbi_usd_balance': _val(summary.get('cbi_usd_balance', {})),
                'cfm_usd_balance': _val(summary.get('cfm_usd_balance', {})),
                'available_margin': _val(summary.get('available_margin', {})),
                'used_margin': _val(summary.get('initial_margin', {})),
                'unrealized_pnl': _val(summary.get('unrealized_pnl', {})),
                'liquidation_buffer_pct': float(summary.get('liquidation_buffer_percentage', 1000)),
            }
        except Exception as e:
            logger.error(f"get_futures_balance failed: {e}")
            return {}

    # ------------------------------------------------------------------ #
    # Products                                                             #
    # ------------------------------------------------------------------ #

    def get_product(self, product_id: str) -> dict:
        """Return product details."""
        p = self._client.get_product(product_id).__dict__
        return {
            'product_id': p.get('product_id'),
            'price': float(p.get('price') or 0),
            'status': p.get('status'),
            'base_currency': p.get('base_currency_id'),
            'quote_currency': p.get('quote_currency_id'),
            'base_min_size': float(p.get('base_min_size') or 0),
            'base_increment': float(p.get('base_increment') or 0),
            'quote_increment': float(p.get('quote_increment') or 0),
            'product_type': p.get('product_type'),
            'volume_24h': float(p.get('volume_24h') or 0),
            'is_disabled': p.get('is_disabled', False),
            'trading_disabled': p.get('trading_disabled', False),
        }

    def get_spot_products(self, quote_currency: str = 'USD') -> list[dict]:
        """Return all online spot products for a given quote currency, sorted by volume."""
        raw = self._client.get_products(product_type='SPOT')['products']
        result = []
        for p in raw:
            d = p.__dict__
            if d.get('quote_currency_id') == quote_currency and d.get('status') == 'online':
                result.append({
                    'product_id': d.get('product_id'),
                    'price': float(d.get('price') or 0),
                    'volume_24h_usd': float(d.get('approximate_quote_24h_volume') or 0),
                    'base_min_size': float(d.get('base_min_size') or 0),
                    'base_increment': float(d.get('base_increment') or 0),
                    'quote_increment': float(d.get('quote_increment') or 0),
                })
        return sorted(result, key=lambda x: x['volume_24h_usd'], reverse=True)

    def get_futures_products(self) -> list[dict]:
        """Return all futures contracts with their current prices."""
        raw = self._client.get_products(product_type='FUTURE')['products']
        result = []
        for p in raw:
            d = p.__dict__
            fcm = d.get('fcm_trading_session_details') or {}
            if isinstance(fcm, dict):
                is_open = fcm.get('is_session_open', False)
            else:
                is_open = getattr(fcm, 'is_session_open', False)
            result.append({
                'product_id': d.get('product_id'),
                'price': float(d.get('price') or 0),
                'base_min_size': float(d.get('base_min_size') or 1),
                'quote_increment': float(d.get('quote_increment') or 0.01),
                'is_session_open': is_open,
            })
        return sorted(result, key=lambda x: x['price'])

    # ------------------------------------------------------------------ #
    # Market Data                                                          #
    # ------------------------------------------------------------------ #

    def get_best_bid_ask(self, product_id: str) -> dict:
        """Return current best bid/ask."""
        try:
            r = self._client.get_best_bid_ask(product_ids=[product_id])
            if hasattr(r, '__dict__'):
                pricebooks = r.__dict__.get('pricebooks', [])
            else:
                pricebooks = r.get('pricebooks', [])
            for pb in pricebooks:
                d = pb.__dict__ if hasattr(pb, '__dict__') else pb
                if d.get('product_id') == product_id:
                    bids = d.get('bids', [])
                    asks = d.get('asks', [])
                    best_bid = float(bids[0].__dict__.get('price', 0) if hasattr(bids[0], '__dict__') else bids[0].get('price', 0)) if bids else 0.0
                    best_ask = float(asks[0].__dict__.get('price', 0) if hasattr(asks[0], '__dict__') else asks[0].get('price', 0)) if asks else 0.0
                    return {'bid': best_bid, 'ask': best_ask, 'mid': (best_bid + best_ask) / 2}
        except Exception as e:
            logger.warning(f"get_best_bid_ask({product_id}) failed: {e}")
        return {'bid': 0.0, 'ask': 0.0, 'mid': 0.0}

    def get_candles(self, product_id: str, granularity: str, start: int, end: int) -> list[dict]:
        """
        Fetch OHLCV candles.
        granularity: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, FOUR_HOUR, ONE_DAY
        start/end: Unix timestamps
        """
        try:
            raw = self._client.get_candles(
                product_id=product_id,
                start=str(start),
                end=str(end),
                granularity=granularity
            )
            candles = raw.get('candles', []) if isinstance(raw, dict) else raw.__dict__.get('candles', [])
            result = []
            for c in candles:
                d = c.__dict__ if hasattr(c, '__dict__') else c
                result.append({
                    'timestamp': int(d.get('start', 0)),
                    'open': float(d.get('open', 0)),
                    'high': float(d.get('high', 0)),
                    'low': float(d.get('low', 0)),
                    'close': float(d.get('close', 0)),
                    'volume': float(d.get('volume', 0)),
                })
            return sorted(result, key=lambda x: x['timestamp'])
        except Exception as e:
            logger.error(f"get_candles({product_id}, {granularity}) failed: {e}")
            return []

    def get_market_trades(self, product_id: str, limit: int = 100) -> list[dict]:
        """Return recent market trades."""
        try:
            raw = self._client.get_market_trades(product_id=product_id, limit=limit)
            trades_raw = raw.get('trades', []) if isinstance(raw, dict) else raw.__dict__.get('trades', [])
            result = []
            for t in trades_raw:
                d = t.__dict__ if hasattr(t, '__dict__') else t
                result.append({
                    'trade_id': d.get('trade_id'),
                    'price': float(d.get('price', 0)),
                    'size': float(d.get('size', 0)),
                    'side': d.get('side'),
                    'time': d.get('time'),
                })
            return result
        except Exception as e:
            logger.error(f"get_market_trades({product_id}) failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Orders                                                               #
    # ------------------------------------------------------------------ #

    def get_orders(self, product_id: Optional[str] = None, status: str = 'OPEN') -> list[dict]:
        """Return orders, optionally filtered by product and status."""
        try:
            kwargs = {'order_status': [status]}
            if product_id:
                kwargs['product_id'] = product_id
            raw = self._client.list_orders(**kwargs)
            orders_raw = raw.get('orders', []) if isinstance(raw, dict) else raw.__dict__.get('orders', [])
            result = []
            for o in orders_raw:
                d = o.__dict__ if hasattr(o, '__dict__') else o
                result.append({
                    'order_id': d.get('order_id'),
                    'product_id': d.get('product_id'),
                    'side': d.get('side'),
                    'status': d.get('status'),
                    'order_type': d.get('order_type'),
                    'filled_size': float(d.get('filled_size') or 0),
                    'filled_value': float(d.get('filled_value') or 0),
                    'average_filled_price': float(d.get('average_filled_price') or 0),
                    'created_time': d.get('created_time'),
                })
            return result
        except Exception as e:
            logger.error(f"get_orders failed: {e}")
            return []

    def place_limit_order(
        self,
        product_id: str,
        side: str,
        size: str,
        limit_price: str,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """
        Place a limit order. ALWAYS use limit orders (maker).
        side: 'BUY' or 'SELL'
        size: base quantity as string
        limit_price: price as string
        """
        import uuid
        coid = client_order_id or f"forge-{uuid.uuid4().hex[:12]}"
        try:
            result = self._client.limit_order_gtc(
                client_order_id=coid,
                product_id=product_id,
                side=side,
                base_size=size,
                limit_price=limit_price,
            )
            d = result.__dict__ if hasattr(result, '__dict__') else result
            success = d.get('success', False)
            order = d.get('order_id') or d.get('success_response', {})
            logger.info(f"limit_order: {side} {size} {product_id} @ {limit_price} → success={success} id={order}")
            return {'success': success, 'order_id': order, 'client_order_id': coid, 'raw': d}
        except Exception as e:
            logger.error(f"place_limit_order failed: {e}")
            return {'success': False, 'error': str(e), 'client_order_id': coid}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            result = self._client.cancel_orders(order_ids=[order_id])
            d = result.__dict__ if hasattr(result, '__dict__') else {}
            results = d.get('results', [])
            if results:
                r = results[0].__dict__ if hasattr(results[0], '__dict__') else results[0]
                return r.get('success', False)
            return False
        except Exception as e:
            logger.error(f"cancel_order({order_id}) failed: {e}")
            return False

    def get_order(self, order_id: str) -> dict:
        """Get a single order by ID."""
        try:
            raw = self._client.get_order(order_id=order_id)
            d = raw.__dict__ if hasattr(raw, '__dict__') else {}
            order = d.get('order', {})
            o = order.__dict__ if hasattr(order, '__dict__') else order
            return {
                'order_id': o.get('order_id'),
                'product_id': o.get('product_id'),
                'side': o.get('side'),
                'status': o.get('status'),
                'filled_size': float(o.get('filled_size') or 0),
                'filled_value': float(o.get('filled_value') or 0),
                'average_filled_price': float(o.get('average_filled_price') or 0),
            }
        except Exception as e:
            logger.error(f"get_order({order_id}) failed: {e}")
            return {}
