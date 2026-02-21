#!/usr/bin/env python3
"""Quick connection test — verifies API credentials and lists account balances."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

from coinbase.rest import RESTClient

def test_connection():
    key_name = os.getenv("COINBASE_API_KEY_NAME")
    private_key = os.getenv("COINBASE_API_PRIVATE_KEY")

    if not key_name or not private_key:
        print("ERROR: Missing API credentials in .env")
        return False

    print(f"Key name: {key_name[:50]}...")
    print("Connecting to Coinbase Advanced Trade API...")

    try:
        client = RESTClient(api_key=key_name, api_secret=private_key)

        # Test 1: Get accounts
        print("\n[1] Fetching accounts...")
        accounts = client.get_accounts()
        print(f"    Accounts returned: {len(accounts['accounts'])}")
        for acct in accounts['accounts']:
            bal = float(acct['available_balance']['value'])
            if bal > 0:
                print(f"    {acct['currency']}: {bal:.6f} (hold: {acct['hold']['value']})")

        # Test 2: Get BTC-USD ticker
        print("\n[2] Fetching BTC-USD product...")
        btc = client.get_product("BTC-USD")
        print(f"    BTC-USD status: {btc['status']}")
        print(f"    Quote currency: {btc['quote_currency_id']}")

        # Test 3: List products (count only)
        print("\n[3] Counting available products...")
        products = client.get_products()
        print(f"    Total products: {len(products['products'])}")

        print("\n✅ Connection successful!")
        return True

    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
