# News Event Data

Source: TradingView Economic Calendar API (economic-calendar.tradingview.com)
Currencies: USD, EUR, GBP, JPY, AUD, NZD, CAD, CHF
Importance: high only (imp=1 = TradingView high-impact)

## Files
- `high_impact_events.csv` — Jul 2024–Oct 2024 + Jan 2026
  Fields: datetime_utc, currency, importance, event, hour_utc, in_london_window

## Refresh
To update or extend the date range:
  python3 scripts/fetch_news.py --start 2024-07-01 --end 2026-01-31
