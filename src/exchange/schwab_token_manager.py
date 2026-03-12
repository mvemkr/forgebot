"""
Schwab token lifecycle manager.
- Refreshes access token every 25 minutes (before 30-min expiry)
- Tracks refresh token age (expires after 7 days)
- Sends proactive Telegram alerts when re-auth is approaching
- Runs as a background thread
"""

import os, json, time, base64, logging, threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
logger = logging.getLogger(__name__)

TOKEN_PATH       = Path(__file__).parent.parent.parent / ".schwab_token.json"
TOKEN_META_PATH  = Path(__file__).parent.parent.parent / ".schwab_token_meta.json"
TOKEN_URL        = "https://api.schwabapi.com/v1/oauth/token"
REFRESH_INTERVAL  = 25 * 60         # Refresh access token every 25 min
REFRESH_TOKEN_TTL = 7 * 24 * 3600   # Schwab refresh token lives ~7 days
WARN_3DAY         = 3 * 24 * 3600   # Start warning 3 days before expiry
WARN_1DAY         = 1 * 24 * 3600   # Hourly alerts inside 24 hours
HOURLY_ALERT_INTERVAL = 3600        # How often to re-alert when < 1 day


class SchwabTokenManager:
    """
    Manages the Schwab token lifecycle in a background thread.
    Instantiate once at startup and call start().
    """

    def __init__(self, notifier=None):
        self.app_key    = os.getenv("SCHWAB_APP_KEY")
        self.app_secret = os.getenv("SCHWAB_APP_SECRET")
        self._thread    = None
        self._stop      = threading.Event()
        self._meta      = self._load_meta()
        self._notifier  = notifier          # optional Notifier instance
        self._last_alert_ts: float = 0.0    # epoch of last Telegram alert
        self._last_refresh_ts: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Metadata                                                             #
    # ------------------------------------------------------------------ #

    def _load_meta(self) -> dict:
        if TOKEN_META_PATH.exists():
            try:
                return json.loads(TOKEN_META_PATH.read_text())
            except Exception:
                pass
        meta = {'refresh_token_issued_at': time.time()}
        TOKEN_META_PATH.write_text(json.dumps(meta))
        return meta

    def _save_meta(self):
        TOKEN_META_PATH.write_text(json.dumps(self._meta))

    def refresh_token_age_seconds(self) -> float:
        """Seconds since the current refresh token was issued."""
        # Prefer the precise ISO timestamp stored in the token file
        try:
            tok = json.loads(TOKEN_PATH.read_text())
            expires_at = tok.get('refresh_token_expires_at')
            if expires_at:
                exp = datetime.fromisoformat(expires_at)
                now = datetime.now(timezone.utc)
                remaining = (exp - now).total_seconds()
                return REFRESH_TOKEN_TTL - remaining
        except Exception:
            pass
        return time.time() - self._meta.get('refresh_token_issued_at', 0)

    def refresh_token_expires_in(self) -> float:
        """Seconds until the refresh token expires (negative = already expired)."""
        return REFRESH_TOKEN_TTL - self.refresh_token_age_seconds()

    # ------------------------------------------------------------------ #
    # Token Refresh                                                        #
    # ------------------------------------------------------------------ #

    def do_refresh(self) -> bool:
        """Perform an access token refresh. Returns True on success."""
        try:
            token = json.loads(TOKEN_PATH.read_text())
            rt    = token.get('refresh_token')
            if not rt:
                logger.error("TokenManager: No refresh token found — manual re-auth required")
                return False

            creds = base64.b64encode(f"{self.app_key}:{self.app_secret}".encode()).decode()
            resp  = requests.post(TOKEN_URL, headers={
                "Authorization": f"Basic {creds}",
                "Content-Type":  "application/x-www-form-urlencoded",
            }, data={"grant_type": "refresh_token", "refresh_token": rt}, timeout=15)

            if resp.status_code == 200:
                new_token = resp.json()
                new_token['_accounts'] = token.get('_accounts', [])

                # If Schwab issued a new refresh token, reset clock and timestamps
                if new_token.get('refresh_token') and new_token['refresh_token'] != rt:
                    self._meta['refresh_token_issued_at'] = time.time()
                    self._save_meta()
                    logger.info("TokenManager: New refresh token issued — clock reset")
                    # Stamp new expiry on the token file
                    new_token['refresh_token_expires_at'] = (
                        datetime.now(timezone.utc) + timedelta(seconds=REFRESH_TOKEN_TTL)
                    ).isoformat()
                else:
                    # Preserve existing expiry stamp if present
                    if token.get('refresh_token_expires_at'):
                        new_token['refresh_token_expires_at'] = token['refresh_token_expires_at']

                # Always stamp last_refreshed
                now_iso = datetime.now(timezone.utc).isoformat()
                new_token['last_refreshed_at'] = now_iso
                self._last_refresh_ts = time.time()

                TOKEN_PATH.write_text(json.dumps(new_token, indent=2))
                TOKEN_PATH.chmod(0o600)
                logger.info("TokenManager: Access token refreshed ✅")
                return True

            else:
                logger.error(f"TokenManager: Refresh failed {resp.status_code}: {resp.text[:200]}")
                return False

        except Exception as e:
            logger.error(f"TokenManager: Refresh exception: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Refresh-token health alerts                                          #
    # ------------------------------------------------------------------ #

    def _check_refresh_token_health(self):
        """Log and optionally Telegram-alert as refresh token nears expiry."""
        expires_in = self.refresh_token_expires_in()
        days  = expires_in / 86400
        hours = expires_in / 3600

        if expires_in <= 0:
            msg = ("🔴 <b>Schwab token EXPIRED</b> — bot running blind.\n"
                   "Re-auth immediately: run <code>python scripts/schwab_direct_auth.py url</code>")
            logger.critical(msg)
            self._maybe_alert(msg, force=True)

        elif expires_in < WARN_1DAY:
            msg = (f"🔴 <b>Schwab refresh token expires in {hours:.1f} hours!</b>\n"
                   "Re-auth now: run <code>python scripts/schwab_direct_auth.py url</code>")
            logger.critical(msg)
            self._maybe_alert(msg, cooldown=HOURLY_ALERT_INTERVAL)

        elif expires_in < WARN_3DAY:
            msg = (f"⚠️ <b>Schwab refresh token expires in {days:.1f} days</b>\n"
                   "Plan your re-auth: run <code>python scripts/schwab_direct_auth.py url</code>")
            logger.warning(msg)
            self._maybe_alert(msg, cooldown=WARN_1DAY)   # once per day

        else:
            logger.debug(f"TokenManager: Refresh token healthy — {days:.1f}d remaining")

    def _maybe_alert(self, msg: str, cooldown: float = 0, force: bool = False):
        """Send Telegram alert respecting cooldown period."""
        if self._notifier is None:
            return
        now = time.time()
        if force or (now - self._last_alert_ts) >= cooldown:
            try:
                self._notifier.send(msg)
                self._last_alert_ts = now
            except Exception as e:
                logger.error(f"TokenManager: Alert send failed: {e}")

    # ------------------------------------------------------------------ #
    # Background loop                                                      #
    # ------------------------------------------------------------------ #

    def _run(self):
        logger.info("TokenManager: Background refresh loop started")
        while not self._stop.wait(REFRESH_INTERVAL):
            self.do_refresh()
            self._check_refresh_token_health()

    def start(self):
        """Start the background refresh thread. Refreshes immediately on start."""
        self.do_refresh()
        self._check_refresh_token_health()
        self._thread = threading.Thread(target=self._run, daemon=True, name="SchwabTokenMgr")
        self._thread.start()
        logger.info("TokenManager: Started — will refresh every 25 minutes")

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------ #
    # Status (used by dashboard API)                                       #
    # ------------------------------------------------------------------ #

    def status(self) -> dict:
        """Return a serialisable health snapshot for the dashboard."""
        expires_in  = self.refresh_token_expires_in()
        days_left   = expires_in / 86400

        # Access token age from file
        access_age_s  = None
        access_exp_s  = None
        last_refresh  = None
        rt_expires_at = None
        try:
            tok = json.loads(TOKEN_PATH.read_text())
            last_refresh  = tok.get('last_refreshed_at')
            rt_expires_at = tok.get('refresh_token_expires_at')
            if last_refresh:
                lr_dt      = datetime.fromisoformat(last_refresh)
                access_age_s = (datetime.now(timezone.utc) - lr_dt).total_seconds()
                access_exp_s = max(0, 1800 - access_age_s)
        except Exception:
            pass

        if expires_in <= 0:
            health = "expired"
        elif expires_in < WARN_1DAY:
            health = "critical"
        elif expires_in < WARN_3DAY:
            health = "warning"
        else:
            health = "ok"

        return {
            'health':                    health,
            'refresh_token_expires_in_s': round(expires_in),
            'refresh_token_days_left':   round(days_left, 2),
            'refresh_token_expires_at':  rt_expires_at,
            'access_token_age_s':        round(access_age_s) if access_age_s is not None else None,
            'access_token_expires_in_s': round(access_exp_s) if access_exp_s is not None else None,
            'last_refreshed_at':         last_refresh,
            'needs_reauth':              expires_in < WARN_1DAY,
        }
