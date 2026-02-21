"""
Schwab Advanced Trade API client.
Handles token auto-refresh, quotes, options chains, account data, and order placement.
Note: Access token expires every 30min. Refresh token expires every 7 days.
"""

import os, json, time, base64, logging
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
logger = logging.getLogger(__name__)

TOKEN_PATH   = Path(__file__).parent.parent.parent / ".schwab_token.json"
TRADER_BASE  = "https://api.schwabapi.com/trader/v1"
MARKET_BASE  = "https://api.schwabapi.com/marketdata/v1"
TOKEN_URL    = "https://api.schwabapi.com/v1/oauth/token"


class SchwabClient:
    """
    Schwab API client with automatic token refresh.
    Provides quotes, options chains, account balances, and order management.
    """

    def __init__(self):
        self.app_key      = os.getenv("SCHWAB_APP_KEY")
        self.app_secret   = os.getenv("SCHWAB_APP_SECRET")
        self.account_hash = os.getenv("SCHWAB_ACCOUNT_HASH")
        self._token       = self._load_token()
        self._token_expiry = time.time() + self._token.get('expires_in', 1800) - 60
        logger.info("SchwabClient initialized")

    # ------------------------------------------------------------------ #
    # Token Management                                                     #
    # ------------------------------------------------------------------ #

    def _load_token(self) -> dict:
        if not TOKEN_PATH.exists():
            raise FileNotFoundError(
                f"Schwab token not found at {TOKEN_PATH}. Run scripts/schwab_direct_auth.py first."
            )
        return json.loads(TOKEN_PATH.read_text())

    def _save_token(self, token: dict):
        self._token = token
        TOKEN_PATH.write_text(json.dumps(token, indent=2))
        TOKEN_PATH.chmod(0o600)

    def _refresh_token(self) -> bool:
        """Exchange refresh token for a new access token."""
        rt = self._token.get('refresh_token')
        if not rt:
            logger.error("No refresh token available — need to re-authorize")
            return False
        creds = base64.b64encode(f"{self.app_key}:{self.app_secret}".encode()).decode()
        try:
            resp = requests.post(TOKEN_URL, headers={
                "Authorization": f"Basic {creds}",
                "Content-Type": "application/x-www-form-urlencoded",
            }, data={"grant_type": "refresh_token", "refresh_token": rt})
            if resp.status_code == 200:
                new_token = resp.json()
                # Preserve account info
                new_token['_accounts'] = self._token.get('_accounts', [])
                self._save_token(new_token)
                self._token_expiry = time.time() + new_token.get('expires_in', 1800) - 60
                logger.info("Schwab token refreshed successfully")
                return True
            else:
                logger.error(f"Token refresh failed: {resp.status_code} {resp.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"Token refresh exception: {e}")
            return False

    def _headers(self) -> dict:
        """Return auth headers, refreshing token if needed."""
        if time.time() >= self._token_expiry:
            logger.info("Schwab token expiring — refreshing...")
            self._refresh_token()
        return {"Authorization": f"Bearer {self._token['access_token']}"}

    def _get(self, base: str, path: str, params: dict = None) -> requests.Response:
        resp = requests.get(f"{base}{path}", headers=self._headers(), params=params)
        if resp.status_code == 401:
            logger.warning("401 Unauthorized — attempting token refresh")
            self._refresh_token()
            resp = requests.get(f"{base}{path}", headers=self._headers(), params=params)
        return resp

    def _post(self, base: str, path: str, json_body: dict) -> requests.Response:
        resp = requests.post(f"{base}{path}", headers={
            **self._headers(), "Content-Type": "application/json"
        }, json=json_body)
        if resp.status_code == 401:
            self._refresh_token()
            resp = requests.post(f"{base}{path}", headers={
                **self._headers(), "Content-Type": "application/json"
            }, json=json_body)
        return resp

    # ------------------------------------------------------------------ #
    # Account                                                              #
    # ------------------------------------------------------------------ #

    def get_account(self) -> dict:
        """Return account balances and positions."""
        resp = self._get(TRADER_BASE, f"/accounts/{self.account_hash}",
                         params={"fields": "positions"})
        if resp.status_code != 200:
            logger.error(f"get_account: {resp.status_code} {resp.text[:200]}")
            return {}
        data = resp.json().get('securitiesAccount', {})
        bal  = data.get('currentBalances', {})
        return {
            'account_number': data.get('accountNumber'),
            'type':           data.get('type'),
            'cash':           float(bal.get('cashBalance', 0)),
            'buying_power':   float(bal.get('buyingPower',
                                bal.get('cashAvailableForTrading', 0))),
            'equity':         float(bal.get('liquidationValue',
                                bal.get('totalCash', 0))),
            'positions':      data.get('positions', []),
        }

    # ------------------------------------------------------------------ #
    # Market Data                                                          #
    # ------------------------------------------------------------------ #

    def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Return latest quotes for a list of symbols."""
        resp = self._get(MARKET_BASE, "/quotes",
                         params={"symbols": ",".join(symbols), "indicative": "false"})
        if resp.status_code != 200:
            logger.error(f"get_quotes: {resp.status_code} {resp.text[:200]}")
            return {}
        result = {}
        for sym, data in resp.json().items():
            q = data.get('quote', {})
            result[sym] = {
                'last':   float(q.get('lastPrice', q.get('last', 0))),
                'bid':    float(q.get('bidPrice', q.get('bid', 0))),
                'ask':    float(q.get('askPrice', q.get('ask', 0))),
                'volume': int(q.get('totalVolume', 0)),
                'change_pct': float(q.get('netPercentChangeInDouble',
                                q.get('percentChange', 0))),
                'high':   float(q.get('highPrice', q.get('high52Week', 0))),
                'low':    float(q.get('lowPrice', q.get('low52Week', 0))),
            }
        return result

    def get_vix(self) -> float:
        """Return the current VIX level."""
        quotes = self.get_quotes(['^VIX', 'VIX'])
        for sym in ['^VIX', 'VIX']:
            v = quotes.get(sym, {}).get('last', 0)
            if v > 0:
                return v
        # Fallback: use VIXY ETF as proxy
        quotes2 = self.get_quotes(['VIXY'])
        return quotes2.get('VIXY', {}).get('last', 0)

    def get_price_history(self, symbol: str, period_type: str = 'day',
                          period: int = 1, frequency_type: str = 'minute',
                          frequency: int = 5) -> list[dict]:
        """Return OHLCV candles for a symbol."""
        resp = self._get(MARKET_BASE, f"/pricehistory", params={
            'symbol': symbol,
            'periodType': period_type,
            'period': period,
            'frequencyType': frequency_type,
            'frequency': frequency,
            'needExtendedHoursData': 'false',
        })
        if resp.status_code != 200:
            logger.error(f"get_price_history({symbol}): {resp.status_code}")
            return []
        candles = resp.json().get('candles', [])
        return [{
            'timestamp': c['datetime'] // 1000,
            'open':   float(c['open']),
            'high':   float(c['high']),
            'low':    float(c['low']),
            'close':  float(c['close']),
            'volume': float(c['volume']),
        } for c in candles]

    def get_options_chain(self, symbol: str, contract_type: str = 'ALL',
                          strike_count: int = 10,
                          expiration_date: str = None) -> dict:
        """Return the full options chain for a symbol."""
        params = {
            'symbol': symbol,
            'contractType': contract_type,
            'strikeCount': strike_count,
            'includeUnderlyingQuote': 'true',
        }
        if expiration_date:
            params['expirationDate'] = expiration_date
        resp = self._get(MARKET_BASE, "/chains", params=params)
        if resp.status_code != 200:
            logger.error(f"get_options_chain({symbol}): {resp.status_code} {resp.text[:200]}")
            return {}
        chain = resp.json()
        return {
            'symbol':           chain.get('symbol'),
            'underlying_price': float(chain.get('underlyingPrice', 0)),
            'volatility':       float(chain.get('volatility', 0)),
            'call_exp_map':     chain.get('callExpDateMap', {}),
            'put_exp_map':      chain.get('putExpDateMap', {}),
            'status':           chain.get('status'),
        }

    def get_market_hours(self, markets: list[str] = None) -> dict:
        """Check if markets are currently open."""
        markets = markets or ['equity', 'option', 'future']
        resp = self._get(TRADER_BASE, "/markets",
                         params={"markets": ",".join(markets)})
        if resp.status_code != 200:
            return {}
        return resp.json()

    # ------------------------------------------------------------------ #
    # Cross-Market Signal                                                  #
    # ------------------------------------------------------------------ #

    def get_market_regime(self) -> dict:
        """
        Assess current market regime using equity signals.
        Returns context to inform crypto trading decisions.
        """
        quotes = self.get_quotes(['SPY', 'QQQ', 'IWM', 'VIXY'])
        spy  = quotes.get('SPY', {})
        qqq  = quotes.get('QQQ', {})
        vixy = quotes.get('VIXY', {})

        # Regime assessment
        spy_change  = spy.get('change_pct', 0)
        qqq_change  = qqq.get('change_pct', 0)
        vixy_change = vixy.get('change_pct', 0)

        # Risk-off: equities falling + volatility rising
        risk_off = spy_change < -0.5 and vixy_change > 3.0
        # Risk-on: equities rising + volatility falling
        risk_on  = spy_change > 0.3 and vixy_change < -2.0
        # Choppy: mixed signals
        choppy   = not risk_off and not risk_on

        regime = 'RISK_OFF' if risk_off else ('RISK_ON' if risk_on else 'NEUTRAL')

        return {
            'regime':       regime,
            'spy_chg_pct':  round(spy_change, 2),
            'qqq_chg_pct':  round(qqq_change, 2),
            'vixy_chg_pct': round(vixy_change, 2),
            'spy_price':    spy.get('last', 0),
            'avoid_longs':  risk_off,    # Don't buy crypto when equities dumping
            'avoid_shorts': risk_on,     # Don't short crypto when equities ripping
            'note': ('Equities selling off — avoid new crypto longs' if risk_off else
                     'Risk-on environment — crypto longs favored' if risk_on else
                     'Neutral equity conditions'),
        }
