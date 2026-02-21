"""
Direct Schwab OAuth token exchange — no interactive prompts, no state mismatch.
Usage:
  Step 1: python schwab_direct_auth.py url   → prints auth URL
  Step 2: python schwab_direct_auth.py token <redirect_url> → exchanges for token
"""

import os, sys, json, base64, urllib.parse, secrets
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

import requests

APP_KEY      = os.getenv("SCHWAB_APP_KEY")
APP_SECRET   = os.getenv("SCHWAB_APP_SECRET")
CALLBACK_URL = "https://127.0.0.1"
TOKEN_PATH   = Path(os.path.dirname(os.path.dirname(__file__))) / ".schwab_token.json"
STATE_PATH   = Path(os.path.dirname(os.path.dirname(__file__))) / ".schwab_state.txt"

AUTH_URL  = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"


def cmd_url():
    """Generate and print the authorization URL, saving state for step 2."""
    state = secrets.token_urlsafe(22)
    STATE_PATH.write_text(state)

    params = {
        "response_type": "code",
        "client_id": APP_KEY,
        "redirect_uri": CALLBACK_URL,
        "scope": "readonly",
        "state": state,
    }
    url = AUTH_URL + "?" + urllib.parse.urlencode(params)
    print(f"\nOpen this URL in your browser:\n\n  {url}\n")
    print("After approving, paste the FULL redirect URL (starting with https://127.0.0.1) back here.")


def cmd_token(redirect_url: str):
    """Exchange auth code for token and save it."""
    # Parse code and state from redirect URL
    parsed = urllib.parse.urlparse(redirect_url)
    params = urllib.parse.parse_qs(parsed.query)

    code  = params.get("code", [None])[0]
    state = params.get("state", [None])[0]

    if not code:
        print(f"❌ No 'code' found in redirect URL.\nGot: {redirect_url}")
        sys.exit(1)

    # Verify state if we saved one
    if STATE_PATH.exists():
        saved_state = STATE_PATH.read_text().strip()
        if state != saved_state:
            print(f"⚠️  State mismatch (saved={saved_state}, got={state}). Continuing anyway...")
    
    print(f"Code found: {code[:20]}...")
    print("Exchanging for token...")

    # Basic auth header: base64(app_key:app_secret)
    creds  = f"{APP_KEY}:{APP_SECRET}"
    b64    = base64.b64encode(creds.encode()).decode()
    headers = {
        "Authorization": f"Basic {b64}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": CALLBACK_URL,
    }

    resp = requests.post(TOKEN_URL, headers=headers, data=data)
    if resp.status_code != 200:
        print(f"❌ Token exchange failed: {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    token = resp.json()
    TOKEN_PATH.write_text(json.dumps(token, indent=2))
    if STATE_PATH.exists():
        STATE_PATH.unlink()

    print(f"✅ Token saved to {TOKEN_PATH}")
    print(f"   Access token: {token.get('access_token','?')[:30]}...")
    print(f"   Expires in:   {token.get('expires_in')}s")
    print(f"   Scope:        {token.get('scope')}")

    # Quick account test using the new token
    print("\nTesting account access...")
    test_resp = requests.get(
        "https://api.schwabapi.com/trader/v1/accounts/accountNumbers",
        headers={"Authorization": f"Bearer {token['access_token']}"},
    )
    if test_resp.status_code == 200:
        accounts = test_resp.json()
        print(f"✅ Connected! {len(accounts)} account(s):")
        for a in accounts:
            print(f"   Account: {a.get('accountNumber')}  Hash: {a.get('hashValue')}")
        # Save hash for future use
        token['_accounts'] = accounts
        TOKEN_PATH.write_text(json.dumps(token, indent=2))
    else:
        print(f"⚠️  Account fetch: {test_resp.status_code} — {test_resp.text[:200]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python schwab_direct_auth.py url              # Generate auth URL")
        print("  python schwab_direct_auth.py token <url>      # Exchange code for token")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "url":
        cmd_url()
    elif cmd == "token" and len(sys.argv) >= 3:
        cmd_token(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
