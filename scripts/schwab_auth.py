"""
One-time Schwab OAuth setup.
Run this script while sitting at the machine with a browser open.
It will open the Schwab login page, you approve, and the token is saved automatically.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

import schwab

APP_KEY    = os.getenv("SCHWAB_APP_KEY")
APP_SECRET = os.getenv("SCHWAB_APP_SECRET")
CALLBACK_URL = "https://127.0.0.1"
TOKEN_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".schwab_token.json")

if not APP_KEY or not APP_SECRET:
    print("ERROR: SCHWAB_APP_KEY and SCHWAB_APP_SECRET must be set in .env")
    sys.exit(1)

print(f"App Key:     {APP_KEY}")
print(f"Token will be saved to: {TOKEN_PATH}")
print(f"Callback URL: {CALLBACK_URL}")
print()
print("A browser window will open. Log in to Schwab and approve the app.")
print("After approval you'll be redirected — the script captures it automatically.")
print()

try:
    client = schwab.auth.client_from_login_flow(
        api_key=APP_KEY,
        app_secret=APP_SECRET,
        callback_url=CALLBACK_URL,
        token_path=TOKEN_PATH,
    )
    print()
    print("✅ Authorization successful! Token saved.")
    print(f"   Token file: {TOKEN_PATH}")

    # Quick test — get accounts
    print("\nTesting connection...")
    resp = client.get_account_numbers()
    if resp.status_code == 200:
        accounts = resp.json()
        print(f"✅ Connected! Accounts found: {len(accounts)}")
        for a in accounts:
            print(f"   Account: {a.get('accountNumber', '?')}  "
                  f"Hash: {a.get('hashValue', '?')}")
    else:
        print(f"⚠️  Auth worked but account fetch returned {resp.status_code}")

except Exception as e:
    print(f"\n❌ Auth failed: {e}")
    print()
    print("If the browser didn't open, try the manual flow instead:")
    print("  python scripts/schwab_auth_manual.py")
