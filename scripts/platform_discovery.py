#!/usr/bin/env python3
"""
Full Coinbase platform discovery.
Categorizes all products, audits accounts, checks futures eligibility.
"""

import os, sys, json
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

from coinbase.rest import RESTClient

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/research")
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = RESTClient(
    api_key=os.getenv("COINBASE_API_KEY_NAME"),
    api_secret=os.getenv("COINBASE_API_PRIVATE_KEY")
)

def p2d(obj):
    """Convert SDK object to dict via __dict__."""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return {}

def discover_products():
    print("=== PRODUCT DISCOVERY ===")
    raw = client.get_products()['products']
    print(f"Total products from API: {len(raw)}")

    categories = defaultdict(list)
    for p in raw:
        d = p2d(p)
        ptype = d.get('product_type', 'UNKNOWN')
        categories[ptype].append(d)

    print("\nProduct types found:")
    for ptype, items in sorted(categories.items()):
        print(f"  {ptype}: {len(items)}")

    # Look for anything that's not SPOT or FUTURE
    other = {k: v for k, v in categories.items() if k not in ('SPOT', 'FUTURE', 'UNKNOWN')}
    if other:
        print("\n⚠️  Other product types detected (possible stocks/ETFs):")
        for k, items in other.items():
            for item in items[:5]:
                print(f"  {item.get('product_id')} — {k}")
    else:
        print("\n  No stock/equity products found via API.")

    return categories

def audit_accounts():
    print("\n=== ACCOUNT BALANCES ===")
    raw = client.get_accounts()['accounts']
    
    has_funds = []
    for acct in raw:
        d = p2d(acct)
        avail_obj = d.get('available_balance', {})
        hold_obj = d.get('hold', {})
        if hasattr(avail_obj, '__dict__'):
            avail_obj = p2d(avail_obj)
            hold_obj = p2d(hold_obj)
        avail = float(avail_obj.get('value', 0) if isinstance(avail_obj, dict) else 0)
        hold = float(hold_obj.get('value', 0) if isinstance(hold_obj, dict) else 0)
        total = avail + hold
        if total > 0.0001:
            has_funds.append({
                'currency': d.get('currency', '?'),
                'available': avail,
                'hold': hold,
                'total': total,
                'type': d.get('type', 'unknown'),
            })

    if has_funds:
        print(f"  Accounts with balances:")
        for a in has_funds:
            print(f"  {a['currency']:10s}  avail={a['available']:.6f}  hold={a['hold']:.6f}  type={a['type']}")
    else:
        print("  ⚠️  No funded accounts found (zero balances or restricted key).")

    return has_funds

def check_futures():
    print("\n=== FUTURES / DERIVATIVES CHECK ===")
    try:
        summary = client.get_futures_balance_summary()
        d = p2d(summary) if hasattr(summary, '__dict__') else {}
        print(f"  ✅ Futures account accessible!")
        print(f"  Raw summary: {d}")
        return True, d
    except Exception as e:
        err = str(e)
        print(f"  ❌ Futures not accessible")
        print(f"  Error: {err[:300]}")
        return False, None

def sample_spot_products(categories):
    print("\n=== SPOT PRODUCT DETAILS ===")
    spot = categories.get('SPOT', [])
    usd_pairs = [p for p in spot if p.get('quote_currency_id') == 'USD' and p.get('status') == 'online']
    usdc_pairs = [p for p in spot if p.get('quote_currency_id') == 'USDC' and p.get('status') == 'online']
    
    print(f"  Online USD pairs:  {len(usd_pairs)}")
    print(f"  Online USDC pairs: {len(usdc_pairs)}")

    # Sort USD pairs by 24h volume descending
    usd_pairs_sorted = sorted(usd_pairs, key=lambda x: float(x.get('volume_24h') or 0), reverse=True)
    print(f"\n  Top 10 USD spot pairs by 24h volume:")
    for p in usd_pairs_sorted[:10]:
        vol = float(p.get('approximate_quote_24h_volume') or p.get('volume_24h') or 0)
        print(f"  {p.get('product_id'):15s}  vol=${vol:,.0f}  min={p.get('base_min_size')}  price_inc={p.get('quote_increment')}")

    # Check a few specific products in detail
    print(f"\n  Detailed check: BTC-USD, ETH-USD, BTC-USDC, ETH-USDC")
    for pid in ['BTC-USD', 'ETH-USD', 'BTC-USDC', 'ETH-USDC']:
        try:
            p = p2d(client.get_product(pid))
            price = float(p.get('price') or 0)
            print(f"  {pid}: price=${price:,.2f}  min={p.get('base_min_size')}  status={p.get('status')}")
        except Exception as e:
            print(f"  {pid}: Error — {e}")

    return usd_pairs, usdc_pairs, usd_pairs_sorted

def sample_futures(categories):
    print("\n=== FUTURES CONTRACTS ===")
    futures = categories.get('FUTURE', [])
    if not futures:
        print("  No futures products in product list.")
        return []
    
    print(f"  Total futures contracts visible: {len(futures)}")
    futures_active = [f for f in futures if f.get('status') == 'online']
    print(f"  Active (online) contracts: {len(futures_active)}")
    print(f"\n  Sample contracts:")
    for f in futures_active[:15]:
        print(f"  {f.get('product_id'):35s}  min={f.get('base_min_size')}  inc={f.get('quote_increment')}")
    
    return futures

def main():
    print(f"Coinbase Platform Discovery — {datetime.utcnow().isoformat()}\n")
    
    categories = discover_products()
    balances = audit_accounts()
    futures_ok, futures_summary = check_futures()
    usd_pairs, usdc_pairs, top_usd = sample_spot_products(categories)
    futures = sample_futures(categories)

    # Save results
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_products": sum(len(v) for v in categories.values()),
            "spot_count": len(categories.get('SPOT', [])),
            "futures_count": len(categories.get('FUTURE', [])),
            "futures_account_accessible": futures_ok,
            "funded_accounts": len(balances),
        },
        "balances": balances,
        "top_usd_pairs": top_usd[:20],
        "futures_contracts": futures[:50],
    }
    path = os.path.join(OUTPUT_DIR, "platform_discovery.json")
    with open(path, 'w') as fh:
        json.dump(results, fh, indent=2, default=str)
    
    print(f"\n{'='*50}")
    print("DISCOVERY COMPLETE")
    print(f"{'='*50}")
    print(f"  Spot products:    {results['summary']['spot_count']}")
    print(f"  Futures:          {results['summary']['futures_count']}")
    print(f"  Futures eligible: {'YES ✅' if futures_ok else 'NO ❌'}")
    print(f"  Funded accounts:  {results['summary']['funded_accounts']}")
    print(f"  Results saved to: {path}")

    return results

if __name__ == "__main__":
    main()
