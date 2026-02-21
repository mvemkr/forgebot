"""
Schwab token lifecycle manager.
- Refreshes access token every 25 minutes (before 30-min expiry)
- Tracks refresh token age (expires after 7 days)
- Alerts Mike via logging when re-auth is needed
- Runs as a background thread
"""

import os, json, time, base64, logging, threading
from pathlib import Path
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
logger = logging.getLogger(__name__)

TOKEN_PATH       = Path(__file__).parent.parent.parent / ".schwab_token.json"
TOKEN_META_PATH  = Path(__file__).parent.parent.parent / ".schwab_token_meta.json"
TOKEN_URL        = "https://api.schwabapi.com/v1/oauth/token"
REFRESH_INTERVAL = 25 * 60        # Refresh access token every 25 min
REFRESH_TOKEN_TTL = 7 * 24 * 3600 # Refresh token lives ~7 days
WARN_BEFORE_EXPIRY = 24 * 3600    # Warn 24 hours before refresh token dies


class SchwabTokenManager:
    """
    Manages the Schwab token lifecycle in a background thread.
    Instantiate once at startup and call start().
    """

    def __init__(self):
        self.app_key    = os.getenv("SCHWAB_APP_KEY")
        self.app_secret = os.getenv("SCHWAB_APP_SECRET")
        self._thread    = None
        self._stop      = threading.Event()
        self._meta      = self._load_meta()

    # ------------------------------------------------------------------ #
    # Metadata (tracks when refresh token was issued)                     #
    # ------------------------------------------------------------------ #

    def _load_meta(self) -> dict:
        if TOKEN_META_PATH.exists():
            try:
                return json.loads(TOKEN_META_PATH.read_text())
            except Exception:
                pass
        # First time — initialize from current token
        meta = {'refresh_token_issued_at': time.time()}
        TOKEN_META_PATH.write_text(json.dumps(meta))
        return meta

    def _save_meta(self):
        TOKEN_META_PATH.write_text(json.dumps(self._meta))

    def refresh_token_age_seconds(self) -> float:
        return time.time() - self._meta.get('refresh_token_issued_at', 0)

    def refresh_token_expires_in(self) -> float:
        return REFRESH_TOKEN_TTL - self.refresh_token_age_seconds()

    # ------------------------------------------------------------------ #
    # Token Refresh                                                        #
    # ------------------------------------------------------------------ #

    def do_refresh(self) -> bool:
        """Perform a token refresh. Returns True on success."""
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
                TOKEN_PATH.write_text(json.dumps(new_token, indent=2))
                TOKEN_PATH.chmod(0o600)

                # If Schwab issued a new refresh token, reset the clock
                if new_token.get('refresh_token') != rt:
                    self._meta['refresh_token_issued_at'] = time.time()
                    self._save_meta()
                    logger.info("TokenManager: New refresh token issued — clock reset")

                logger.info("TokenManager: Access token refreshed ✅")
                return True
            else:
                logger.error(f"TokenManager: Refresh failed {resp.status_code}: {resp.text[:200]}")
                return False

        except Exception as e:
            logger.error(f"TokenManager: Refresh exception: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Background loop                                                      #
    # ------------------------------------------------------------------ #

    def _run(self):
        logger.info("TokenManager: Background refresh loop started")
        while not self._stop.wait(REFRESH_INTERVAL):
            self.do_refresh()
            self._check_refresh_token_health()

    def _check_refresh_token_health(self):
        expires_in = self.refresh_token_expires_in()
        if expires_in < WARN_BEFORE_EXPIRY:
            hours = expires_in / 3600
            msg = (f"⚠️  SCHWAB REFRESH TOKEN EXPIRES IN {hours:.1f} HOURS. "
                   f"Mike must re-authorize: run python scripts/schwab_direct_auth.py url")
            logger.critical(msg)
            # This will bubble up to the alert system when integrated
        elif expires_in < 3 * 24 * 3600:
            logger.warning(f"TokenManager: Refresh token expires in {expires_in/3600:.0f}h")

    def start(self):
        """Start the background refresh thread."""
        self.do_refresh()  # Refresh immediately on start
        self._thread = threading.Thread(target=self._run, daemon=True, name="SchwabTokenMgr")
        self._thread.start()
        logger.info("TokenManager: Started — will refresh every 25 minutes")

    def stop(self):
        self._stop.set()

    def status(self) -> dict:
        expires_in = self.refresh_token_expires_in()
        return {
            'refresh_token_age_hours':    round(self.refresh_token_age_seconds() / 3600, 1),
            'refresh_token_expires_in_h': round(expires_in / 3600, 1),
            'needs_reauth':               expires_in < WARN_BEFORE_EXPIRY,
        }
