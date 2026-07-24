"""TradingView data source via the unofficial tvDatafeed websocket client.

tvDatafeed is synchronous (blocking websocket) — API handlers must call
`fetch_tradingview` through `run_in_threadpool`, never directly on the event loop.

Anonymous access works for most symbols (~5000 bar cap). Set TV_USERNAME/TV_PASSWORD
in .env to use a logged-in session. Bar timestamps come back naive in the machine's
local timezone (tvDatafeed uses `datetime.fromtimestamp`); containers/EC2 run UTC so
`normalize()` localizing to UTC is correct there.

Unofficial API caveat: no SLA, can break without notice if TradingView changes the
websocket protocol. The official fallback path is data/loader.py::fetch_oanda.
"""
from __future__ import annotations

import threading
import time

import pandas as pd
from tvDatafeed import Interval, TvDatafeed

from patternlab.config import settings
from patternlab.data.loader import normalize

# "1", "3", "5", "15", "30", "45", "1H", "2H", "4H", "1D", "1W", "1M"
INTERVALS: dict[str, Interval] = {i.value: i for i in Interval}

# Short TTL: the frontend polls every few seconds for live updates; the cache only
# needs to absorb duplicate/concurrent requests, not throttle the poll itself.
CACHE_TTL_S = 4.0

_client: TvDatafeed | None = None
_lock = threading.Lock()  # one shared websocket; serialize fetches
_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}


def _get_client() -> TvDatafeed:
    global _client
    if _client is None:
        _client = TvDatafeed(settings.tv_username or None, settings.tv_password or None)
    return _client


def _get_hist(symbol: str, exchange: str, interval: Interval, n_bars: int) -> pd.DataFrame | None:
    """Raw tvDatafeed call, isolated so tests can monkeypatch it."""
    return _get_client().get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)


def fetch_tradingview(symbol: str = "XAUUSD", exchange: str = "OANDA",
                      interval: str = "1", n_bars: int = 1500) -> pd.DataFrame:
    """Fetch OHLCV from TradingView into the canonical frame (see loader.normalize).

    Results are cached for CACHE_TTL_S per (symbol, exchange, interval, n_bars) so the
    frontend re-analyzing doesn't hammer TradingView's websocket.
    """
    if interval not in INTERVALS:
        raise ValueError(f"Unknown interval {interval!r}; one of {sorted(INTERVALS)}")
    key = (symbol.upper(), exchange.upper(), interval, n_bars)
    with _lock:
        now = time.monotonic()
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL_S:
            return hit[1]
        raw = _get_hist(key[0], key[1], INTERVALS[interval], n_bars)
        if raw is None or raw.empty:
            raise RuntimeError(f"TradingView returned no data for {exchange}:{symbol}")
        df = normalize(raw.drop(columns=["symbol"], errors="ignore"))
        _cache[key] = (now, df)
        return df
