"""
Step 2 of Schwab OAuth: exchange the redirect URL for a token.
Usage: python schwab_token_exchange.py "https://127.0.0.1/?code=...&state=..."
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
import schwab

APP_KEY      = os.getenv("SCHWAB_APP_KEY")
APP_SECRET   = os.getenv("SCHWAB_APP_SECRET")
CALLBACK_URL = "https://127.0.0.1"
TOKEN_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".schwab_token.json")

if len(sys.argv) < 2:
    print("Usage: python schwab_token_exchange.py \"<redirect_url>\"")
    sys.exit(1)

redirect_url = sys.argv[1]
print(f"Exchanging token for redirect URL...")
print(f"Token will be saved to: {TOKEN_PATH}")

client = schwab.auth.client_from_received_url(
    api_key=APP_KEY,
    app_secret=APP_SECRET,
    callback_url=CALLBACK_URL,
    received_url=redirect_url,
    token_path=TOKEN_PATH,
)

print("✅ Token saved!")
resp = client.get_account_numbers()
if resp.status_code == 200:
    accounts = resp.json()
    print(f"✅ Connected! {len(accounts)} account(s):")
    for a in accounts:
        print(f"   Account: {a.get('accountNumber')}  Hash: {a.get('hashValue')}")
else:
    print(f"⚠️  Status {resp.status_code}: {resp.text[:300]}")
