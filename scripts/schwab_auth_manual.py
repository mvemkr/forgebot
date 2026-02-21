"""
Manual Schwab OAuth setup — fallback if browser auto-capture doesn't work.
Prints the URL, you paste back the redirect URL after approving.
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

if not APP_KEY or not APP_SECRET:
    print("ERROR: missing SCHWAB_APP_KEY / SCHWAB_APP_SECRET in .env")
    sys.exit(1)

print("=== Schwab Manual OAuth ===")
print()

client = schwab.auth.client_from_manual_flow(
    api_key=APP_KEY,
    app_secret=APP_SECRET,
    callback_url=CALLBACK_URL,
    token_path=TOKEN_PATH,
)

print()
print("✅ Token saved to:", TOKEN_PATH)

resp = client.get_account_numbers()
if resp.status_code == 200:
    accounts = resp.json()
    print(f"✅ Connected! {len(accounts)} account(s):")
    for a in accounts:
        print(f"   {a.get('accountNumber')}  (hash: {a.get('hashValue')})")
else:
    print(f"⚠️  Status {resp.status_code}: {resp.text[:200]}")
