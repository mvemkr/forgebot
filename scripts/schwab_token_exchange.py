"""
Schwab OAuth Token Exchange — Fixed
Extracts state from the redirect URL and completes token exchange.

Usage:
    python schwab_token_exchange.py "https://127.0.0.1/?code=...&state=..."

If your redirect URL has expired, re-run schwab_auth_manual.py to get a fresh one.
"""
import os, sys, json
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

import schwab
from schwab.auth import AuthContext, client_from_received_url

APP_KEY      = os.getenv("SCHWAB_APP_KEY")
APP_SECRET   = os.getenv("SCHWAB_APP_SECRET")
CALLBACK_URL = "https://127.0.0.1"
TOKEN_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".schwab_token.json")

if not APP_KEY or not APP_SECRET:
    print("❌ SCHWAB_APP_KEY or SCHWAB_APP_SECRET not found in .env")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python schwab_token_exchange.py \"<redirect_url>\"")
    print("\nRedirect URL looks like:")
    print("  https://127.0.0.1/?code=C0.b64.xxxx&state=xxxx")
    sys.exit(1)

received_url = sys.argv[1].strip()

# Extract state from the redirect URL
parsed = urlparse(received_url)
params = parse_qs(parsed.query)

if 'state' not in params:
    print("❌ No 'state' parameter found in redirect URL.")
    print("   Make sure you're pasting the FULL redirect URL including ?code=...&state=...")
    sys.exit(1)

if 'code' not in params:
    print("❌ No 'code' parameter found in redirect URL.")
    print("   The authorization code may have expired. Re-run schwab_auth_manual.py for a fresh URL.")
    sys.exit(1)

state = params['state'][0]
print(f"✅ Extracted state: {state[:12]}...")
print(f"✅ Authorization code found")
print(f"📁 Token will be saved to: {TOKEN_PATH}")
print(f"🔄 Exchanging token with Schwab...")

# Reconstruct auth context using the state from the redirect URL
# This ensures the state validation passes
auth_context = schwab.auth.get_auth_context(APP_KEY, CALLBACK_URL, state=state)

def token_write_func(token):
    with open(TOKEN_PATH, 'w') as f:
        json.dump(token, f, indent=2)
    os.chmod(TOKEN_PATH, 0o600)
    print(f"✅ Token written to {TOKEN_PATH}")

try:
    client = client_from_received_url(
        api_key=APP_KEY,
        app_secret=APP_SECRET,
        auth_context=auth_context,
        received_url=received_url,
        token_write_func=token_write_func,
    )
    print("✅ Token exchange successful!")
except Exception as e:
    print(f"❌ Token exchange failed: {e}")
    print("\n💡 The authorization code expires quickly (~30 seconds).")
    print("   If you get a 'bad_request' or 'invalid_grant' error, run:")
    print("   python scripts/schwab_auth_manual.py")
    print("   to generate a fresh auth URL, then immediately paste the redirect.")
    sys.exit(1)

# Verify connection
print("\n🔍 Verifying Schwab connection...")
try:
    resp = client.get_account_numbers()
    if resp.status_code == 200:
        accounts = resp.json()
        print(f"✅ Connected! {len(accounts)} account(s):")
        for a in accounts:
            print(f"   Account: {a.get('accountNumber')}  Hash: {a.get('hashValue')}")
    else:
        print(f"⚠️  Connected but API returned status {resp.status_code}: {resp.text[:300]}")
except Exception as e:
    print(f"⚠️  Token saved but verification failed: {e}")
    print("   The token may still be valid — try running schwab_client.py directly.")
