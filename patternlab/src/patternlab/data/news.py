"""Economic-calendar news from the ForexFactory weekly JSON feed.

Same grounding contract as the detection engine: this module does the deterministic
fetch/filter/time math, and the LLM only ever sees the structured events it returns.
"""
from __future__ import annotations

import datetime as dt
import threading

_CACHE_TTL = dt.timedelta(minutes=15)      # feed updates slowly and 429s aggressive pollers
_RETRY_COOLDOWN = dt.timedelta(minutes=2)  # after a failure, don't hammer the feed

_lock = threading.Lock()
_cache: tuple[dt.datetime, list[dict]] | None = None
_last_failure: dt.datetime | None = None

_IMPACT_RANK = {"Holiday": 2, "Low": 1, "Medium": 2, "High": 3}


def fetch_calendar() -> list[dict]:
    """This week's events, UTC-normalized, cached for _CACHE_TTL. On fetch failure serves
    the last good copy however stale (the feed rate-limits per IP — stale beats nothing)
    and backs off for _RETRY_COOLDOWN. Blocking (httpx) — call through run_in_threadpool
    from async handlers."""
    global _cache, _last_failure
    import httpx

    from patternlab.config import settings

    now = dt.datetime.now(dt.timezone.utc)
    with _lock:
        cached = _cache
        if cached is not None and now - cached[0] < _CACHE_TTL:
            return cached[1]
        if _last_failure is not None and now - _last_failure < _RETRY_COOLDOWN:
            return cached[1] if cached else []
    try:
        r = httpx.get(settings.news_feed_url, timeout=15.0,
                      headers={"User-Agent": "patternlab/0.1"})
        r.raise_for_status()
        raw = r.json()
    except Exception:
        with _lock:
            _last_failure = now
        if cached is not None:
            return cached[1]
        raise
    events = []
    for e in raw:
        try:
            when = dt.datetime.fromisoformat(e["date"]).astimezone(dt.timezone.utc)
        except (KeyError, TypeError, ValueError):
            continue
        events.append({"time_utc": when, "currency": e.get("country", ""),
                       "impact": e.get("impact", ""), "title": e.get("title", ""),
                       "forecast": e.get("forecast", ""), "previous": e.get("previous", "")})
    with _lock:
        _cache = (now, events)
        _last_failure = None
    return events


def _symbol_currencies(symbol: str) -> set[str]:
    s = symbol.upper().replace("_", "").replace("/", "")
    # XAU/XAG halves match no calendar currency, leaving the USD side — which is what
    # actually drives metals.
    return {s[:3], s[3:6]} if len(s) >= 6 else {s}


def relevant_events(symbol: str, now: dt.datetime | None = None,
                    hours_back: float = 8.0, hours_ahead: float = 24.0,
                    min_impact: str = "Medium") -> list[dict]:
    """Calendar events for the symbol's currencies within [now-hours_back, now+hours_ahead],
    at or above min_impact (bank holidays count as Medium — they gut liquidity). Adds
    status + minutes_until so the LLM never does time math."""
    now = now or dt.datetime.now(dt.timezone.utc)
    ccys = _symbol_currencies(symbol)
    lo, hi = now - dt.timedelta(hours=hours_back), now + dt.timedelta(hours=hours_ahead)
    floor = _IMPACT_RANK.get(min_impact, 2)
    out = []
    for e in fetch_calendar():
        if e["currency"] not in ccys or _IMPACT_RANK.get(e["impact"], 0) < floor:
            continue
        if not lo <= e["time_utc"] <= hi:
            continue
        minutes = round((e["time_utc"] - now).total_seconds() / 60)
        out.append({**e, "time_utc": e["time_utc"].isoformat(),
                    "minutes_until": minutes,
                    "status": "upcoming" if minutes > 0 else "released"})
    return sorted(out, key=lambda e: e["minutes_until"])
