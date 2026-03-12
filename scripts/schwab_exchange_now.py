"""
Raw HTTP token exchange — no schwab-py library, no state validation.
Usage: python scripts/schwab_exchange_now.py "<redirect_url>"
"""
import os, sys, json, base64, requests
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

APP_KEY      = os.getenv("SCHWAB_APP_KEY")
APP_SECRET   = os.getenv("SCHWAB_APP_SECRET")
CALLBACK_URL = "https://127.0.0.1"
TOKEN_PATH   = Path(os.path.dirname(os.path.dirname(__file__))) / ".schwab_token.json"
TOKEN_URL    = "https://api.schwabapi.com/v1/oauth/token"

if len(sys.argv) < 2:
    print("Usage: python scripts/schwab_exchange_now.py \"<redirect_url>\"")
    sys.exit(1)

received_url = sys.argv[1].strip()
code = parse_qs(urlparse(received_url).query).get("code", [None])[0]
if not code:
    print("❌ No code found in redirect URL"); sys.exit(1)

print(f"Code: {code[:30]}...")
print("Exchanging...")

creds = base64.b64encode(f"{APP_KEY}:{APP_SECRET}".encode()).decode()
resp  = requests.post(TOKEN_URL,
    headers={"Authorization": f"Basic {creds}", "Content-Type": "application/x-www-form-urlencoded"},
    data={"grant_type": "authorization_code", "code": code, "redirect_uri": CALLBACK_URL},
    timeout=15)

if resp.status_code != 200:
    print(f"❌ {resp.status_code}: {resp.text}"); sys.exit(1)

token = resp.json()
# Wrap in schwab-py's expected nested format
from datetime import datetime, timezone, timedelta
wrapped = {
    "creation_timestamp":       int(__import__("time").time()),
    "token":                    token,
    "refresh_token_expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
    "last_refreshed_at":        datetime.now(timezone.utc).isoformat(),
}
TOKEN_PATH.write_text(json.dumps(wrapped, indent=2))
TOKEN_PATH.chmod(0o600)
print(f"✅ Token saved! Expires in {token.get('expires_in')}s")

# Verify
verify = requests.get("https://api.schwabapi.com/trader/v1/accounts/accountNumbers",
    headers={"Authorization": f"Bearer {token['access_token']}"})
if verify.status_code == 200:
    accounts = verify.json()
    print(f"✅ Connected — {len(accounts)} account(s):")
    for a in accounts:
        print(f"   {a.get('accountNumber')}  hash={a.get('hashValue')}")
    token['_accounts'] = accounts
    TOKEN_PATH.write_text(json.dumps(token, indent=2))
else:
    print(f"⚠️  Verify: {verify.status_code}")
