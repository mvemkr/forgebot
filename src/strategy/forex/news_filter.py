"""
News Filter — Tier 1 Economic Event Blackout

Fetches the ForexFactory economic calendar weekly and blocks entries
during high-impact news windows.

Two-layer protection:
  1. Entry blackout: no new trades 30 min before → 90 min after a Tier 1 event.
     After 60 min (one full 1H candle has closed), entry is allowed early
     if the post-news candle is "clean" (body ≥ 33% of range, not a spike).
     This means we enter on the first genuine post-news candle rather than
     waiting a fixed clock window.
  2. Candle tagging: 1H candles that open inside a news window are
     flagged so the signal detector ignores them as entry triggers.
     A 200-pip NFP spike looks like a perfect engulfing candle — it isn't.

Tier 1 events tracked (by currency pair exposure):
  USD: NFP, FOMC Rate Decision, CPI, GDP, Retail Sales, ISM
  GBP: BOE Rate Decision, CPI, GDP
  EUR: ECB Rate Decision, CPI, GDP
  JPY: BOJ Rate Decision, Tokyo CPI
  CAD: BOC Rate Decision, Employment Change
  AUD: RBA Rate Decision, Employment Change
  NZD: RBNZ Rate Decision
  CHF: SNB Rate Decision

Source: ForexFactory calendar JSON (free, no API key needed)
Cached locally for the week — fetched every Sunday or on first run.
"""
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_FILE = Path.home() / "trading-bot" / "logs" / "news_calendar_cache.json"

# ForexFactory free calendar endpoint
FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Currencies we care about (maps to pairs we trade)
TRACKED_CURRENCIES = {"USD", "GBP", "EUR", "JPY", "CHF", "CAD", "AUD", "NZD"}

# Only events with these keywords in the title qualify as Tier 1
TIER1_KEYWORDS = [
    "Non-Farm",
    "Nonfarm",
    "Fed Funds Rate",
    "FOMC",
    "Interest Rate",
    "Rate Decision",
    "Monetary Policy",
    "CPI",
    "Consumer Price Index",
    "GDP",
    "Gross Domestic Product",
    "Retail Sales",
    "Employment Change",
    "Unemployment Rate",
    "Claimant",
    "ISM Manufacturing",
    "ISM Services",
    "PMI",
    "Tokyo CPI",
    "BOJ",
    "BOE",
    "ECB",
    "SNB",
    "RBA",
    "RBNZ",
    "BOC",
]

# Blackout windows (minutes)
BLACKOUT_BEFORE_MINS   = 30   # 30 min before event — spreads widen, pre-positioning
BLACKOUT_AFTER_MINS    = 90   # 90 min after event — hard ceiling
POST_NEWS_MIN_WAIT     = 60   # min wait before checking clean bar (one full 1H candle)


class NewsFilter:
    """
    Manages Tier 1 news event detection and entry blackouts.

    Usage:
        nf = NewsFilter()
        nf.refresh_if_needed()   # call weekly or on startup

        # In session filter / orchestrator (pass the last closed 1H candle for early release):
        blocked, reason = nf.is_entry_blocked(
            datetime.now(timezone.utc),
            post_news_candle=last_closed_candle,   # optional dict with open/high/low/close
        )

        # In signal detector — skip candles that opened during a news window:
        is_news = nf.is_news_candle(candle_open_timestamp_utc)

        # Check if a candle looks like genuine price action (not a spike):
        clean = NewsFilter.is_clean_bar(candle_dict)
    """

    def __init__(self):
        self._events: List[dict] = []
        self._last_fetch: Optional[datetime] = None
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._load_cache()

    # ── Public Interface ──────────────────────────────────────────────

    def refresh_if_needed(self, force: bool = False) -> bool:
        """
        Fetch and cache the weekly calendar if it's stale (>6 days old).
        Call on startup and every Sunday. Returns True if refreshed.
        """
        if not force and self._last_fetch:
            age_hours = (datetime.now(timezone.utc) - self._last_fetch).total_seconds() / 3600
            if age_hours < 144:   # 6 days — fresh enough
                return False

        logger.info("NewsFilter: Fetching Tier 1 calendar from ForexFactory...")
        try:
            resp = requests.get(FF_CALENDAR_URL, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
            self._events = self._parse_events(raw)
            self._last_fetch = datetime.now(timezone.utc)
            self._save_cache()
            logger.info(f"NewsFilter: Loaded {len(self._events)} Tier 1 events for this week")
            return True
        except Exception as e:
            logger.warning(f"NewsFilter: Calendar fetch failed: {e} — using cached data")
            return False

    def is_entry_blocked(
        self,
        dt_utc: datetime,
        post_news_candle: Optional[dict] = None,
    ) -> Tuple[bool, str]:
        """
        Returns (blocked, reason) for the given timestamp.
        Block if we're within the blackout window of any Tier 1 event.

        Post-event logic:
          - Hard block: 0–90 min after event.
          - Early release: after 60 min, if `post_news_candle` is provided
            and passes is_clean_bar(), the block is lifted immediately.
            This means we enter on the first genuine post-news 1H candle
            rather than waiting a fixed 90-min clock.

        Args:
            dt_utc:           Current UTC timestamp to evaluate.
            post_news_candle: The first 1H candle that closed after the news
                              event (open, high, low, close floats). Optional —
                              if omitted the hard 90-min clock applies.
        """
        for event in self._events:
            event_dt = event.get("dt_utc")
            if not event_dt:
                continue

            before_window = event_dt - timedelta(minutes=BLACKOUT_BEFORE_MINS)
            after_window  = event_dt + timedelta(minutes=BLACKOUT_AFTER_MINS)

            if not (before_window <= dt_utc <= after_window):
                continue

            mins_since = (dt_utc - event_dt).total_seconds() / 60
            mins_to    = (event_dt - dt_utc).total_seconds() / 60

            # ── Pre-event block ──────────────────────────────────────────
            if dt_utc < event_dt:
                reason = (
                    f"⛔ NEWS BLACKOUT: {event['title']} ({event['currency']}) "
                    f"in {int(mins_to)} min — "
                    f"no entries {BLACKOUT_BEFORE_MINS}min before Tier 1 events."
                )
                return True, reason

            # ── Post-event: early release on clean bar ───────────────────
            if mins_since >= POST_NEWS_MIN_WAIT and post_news_candle:
                if self.is_clean_bar(post_news_candle):
                    return False, ""   # one clean post-news candle — safe to enter

            # ── Standard post-event cooldown ─────────────────────────────
            reason = (
                f"⛔ NEWS COOLDOWN: {event['title']} ({event['currency']}) "
                f"fired {int(mins_since)} min ago. "
                f"Waiting {BLACKOUT_AFTER_MINS}min or one clean 1H bar (>{POST_NEWS_MIN_WAIT}min)."
            )
            return True, reason

        return False, ""

    @staticmethod
    def is_clean_bar(candle: dict) -> bool:
        """
        Returns True if a 1H candle looks like genuine price action (not a news spike).

        A clean bar has a real body ≥ 33% of its total high-low range.
        Spike candles (huge wicks, tiny body) caused by news releases fail this test.

        Accepts either raw keys (open/high/low/close) or OANDA mid-price keys
        (mid_o / mid_h / mid_l / mid_c).
        """
        try:
            o = float(candle.get("open",  candle.get("mid_o", 0)))
            h = float(candle.get("high",  candle.get("mid_h", 0)))
            l = float(candle.get("low",   candle.get("mid_l", 0)))
            c = float(candle.get("close", candle.get("mid_c", 0)))

            total_range = h - l
            if total_range == 0:
                return True   # doji — no spike, treat as clean

            body_ratio = abs(c - o) / total_range
            return body_ratio >= 0.33   # body at least a third of the total range
        except Exception:
            return True   # can't assess → don't block

    def is_news_candle(self, candle_dt_utc: datetime, pair: str = "") -> bool:
        """
        Returns True if this 1H candle opens inside a news blackout window.
        Signal detector calls this to skip candles that fired during news.

        A news candle looks like a genuine signal (large body, clear direction)
        but is actually a data release spike — not tradeable price action.

        Uses the same BLACKOUT_BEFORE_MINS / BLACKOUT_AFTER_MINS constants as
        is_entry_blocked() so candle exclusion and entry blocking stay in sync.
        """
        for event in self._events:
            event_dt = event.get("dt_utc")
            if not event_dt:
                continue
            mins_offset = (candle_dt_utc - event_dt).total_seconds() / 60
            # Candle opens in pre-event or post-event blackout window
            if -BLACKOUT_BEFORE_MINS <= mins_offset <= BLACKOUT_AFTER_MINS:
                return True
        return False

    def upcoming_events(self, hours_ahead: int = 48) -> List[dict]:
        """Return Tier 1 events in the next N hours — for dashboard/briefings."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        return [
            e for e in self._events
            if e.get("dt_utc") and now <= e["dt_utc"] <= cutoff
        ]

    def format_upcoming(self, hours_ahead: int = 48) -> str:
        """Human-readable string of upcoming Tier 1 events."""
        events = self.upcoming_events(hours_ahead)
        if not events:
            return "No Tier 1 events in the next 48 hours — clear to trade."
        lines = [f"⚠️ Upcoming Tier 1 events (next {hours_ahead}h):"]
        for e in events:
            dt_str = e["dt_utc"].strftime("%a %b %d %H:%M UTC") if e.get("dt_utc") else "?"
            lines.append(f"  • {dt_str} — {e['currency']} {e['title']}")
        return "\n".join(lines)

    # ── Calendar Parsing ──────────────────────────────────────────────

    def _parse_events(self, raw: list) -> List[dict]:
        """
        Filter ForexFactory JSON to Tier 1 events on tracked currencies.
        Only keeps HIGH impact events matching Tier 1 keywords.

        API format (as of 2026): events have a single 'date' field in ISO 8601
        format with timezone offset, e.g. '2026-02-17T08:30:00-05:00'.
        No separate 'time' field.
        """
        tier1 = []
        for item in raw:
            if item.get("impact", "").lower() != "high":
                continue

            currency = item.get("country", "").upper()
            if currency not in TRACKED_CURRENCIES:
                continue

            title = item.get("title", "")
            is_tier1 = any(kw.lower() in title.lower() for kw in TIER1_KEYWORDS)
            if not is_tier1:
                continue

            dt_utc = self._parse_dt(item.get("date", ""))
            if not dt_utc:
                continue

            tier1.append({
                "title":    title,
                "currency": currency,
                "dt_utc":   dt_utc,
                "impact":   "HIGH",
                "forecast": item.get("forecast", ""),
                "previous": item.get("previous", ""),
            })

        tier1.sort(key=lambda x: x["dt_utc"])
        return tier1

    def _parse_dt(self, date_str: str) -> Optional[datetime]:
        """
        Parse ISO 8601 timestamp (with or without timezone offset) into UTC datetime.
        API returns format like: '2026-02-17T08:30:00-05:00'
        """
        if not date_str:
            return None
        try:
            # Python 3.7+ handles ISO 8601 with timezone offset natively
            dt = datetime.fromisoformat(date_str)
            # Convert to UTC
            return dt.astimezone(timezone.utc).replace(tzinfo=timezone.utc)
        except Exception:
            pass
        try:
            # Fallback: strip timezone and treat as UTC
            clean = date_str[:19]  # '2026-02-17T08:30:00'
            dt = datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    # ── Cache ─────────────────────────────────────────────────────────

    def _save_cache(self):
        try:
            serializable = []
            for e in self._events:
                ev = dict(e)
                if ev.get("dt_utc"):
                    ev["dt_utc"] = ev["dt_utc"].isoformat()
                serializable.append(ev)
            CACHE_FILE.write_text(json.dumps({
                "fetched_at": self._last_fetch.isoformat() if self._last_fetch else None,
                "events":     serializable,
            }, indent=2))
        except Exception as ex:
            logger.warning(f"NewsFilter: Cache save failed: {ex}")

    def _load_cache(self):
        if not CACHE_FILE.exists():
            return
        try:
            data = json.loads(CACHE_FILE.read_text())
            fetched = data.get("fetched_at")
            if fetched:
                self._last_fetch = datetime.fromisoformat(fetched)
            events = []
            for e in data.get("events", []):
                if e.get("dt_utc"):
                    e["dt_utc"] = datetime.fromisoformat(e["dt_utc"])
                events.append(e)
            self._events = events
            logger.info(f"NewsFilter: Loaded {len(self._events)} cached events")
        except Exception as ex:
            logger.warning(f"NewsFilter: Cache load failed: {ex}")
