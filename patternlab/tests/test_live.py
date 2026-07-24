import sys
sys.path.insert(0, "src")

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from patternlab.data import tradingview
from patternlab.data.loader import synthetic_session


def _raw_tv_frame(n: int = 600) -> pd.DataFrame:
    """Shape tvDatafeed actually returns: naive microsecond-resolution index,
    extra 'symbol' column."""
    df = synthetic_session("grind_break", n=n, seed=7)
    df.index = df.index.tz_localize(None).as_unit("us")
    df.insert(0, "symbol", "OANDA:XAUUSD")
    return df


@pytest.fixture
def mock_tv(monkeypatch):
    calls = {"n": 0}

    def fake_get_hist(symbol, exchange, interval, n_bars):
        calls["n"] += 1
        return _raw_tv_frame()

    monkeypatch.setattr(tradingview, "_get_hist", fake_get_hist)
    tradingview._cache.clear()
    return calls


@pytest.fixture
def no_db(monkeypatch):
    """Ignore any DATABASE_URL from a local .env — endpoint tests must be hermetic."""
    from patternlab.config import settings
    monkeypatch.setattr(settings, "database_url", "")


def test_fetch_normalizes_and_caches(mock_tv):
    df = tradingview.fetch_tradingview("XAUUSD", "OANDA", "1", 600)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert "symbol" not in df.columns

    tradingview.fetch_tradingview("XAUUSD", "OANDA", "1", 600)
    assert mock_tv["n"] == 1  # second hit served from TTL cache

    tradingview.fetch_tradingview("XAUUSD", "OANDA", "5", 600)
    assert mock_tv["n"] == 2  # different interval -> new fetch


def test_fetch_rejects_unknown_interval():
    with pytest.raises(ValueError):
        tradingview.fetch_tradingview(interval="7")


def test_live_analyze_endpoint(mock_tv, no_db):
    from patternlab.api.main import app

    with TestClient(app) as client:
        r = client.get("/live/analyze", params={"symbol": "XAUUSD", "exchange": "OANDA",
                                                "interval": "1", "n_bars": 600})
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "XAUUSD"
    assert len(body["bars"]) == 600
    assert {"time", "open", "high", "low", "close", "volume"} <= set(body["bars"][0])
    assert body["bars"][0]["time"] > 1_500_000_000  # unix seconds, not a unit artifact
    assert body["analysis"]["n_bars"] == 600
    assert body["matches"] is None  # no DATABASE_URL in tests

    levels = body["analysis"]["levels"]
    assert levels and all("events" in l for l in levels)
    ev = levels[0]["events"][0]
    assert {"time", "price", "kind"} <= set(ev)
    assert ev["kind"] in ("H", "L")
    assert ev["time"] == body["bars"][0]["time"] or ev["time"] > 1_500_000_000


def test_live_analyze_lookback_extends_levels_window(monkeypatch, no_db):
    from patternlab.api.main import app

    def fake_get_hist(symbol, exchange, interval, n_bars):
        return _raw_tv_frame(n_bars)

    monkeypatch.setattr(tradingview, "_get_hist", fake_get_hist)
    tradingview._cache.clear()

    with TestClient(app) as client:
        r = client.get("/live/analyze", params={"interval": "1", "n_bars": 300,
                                                "lookback_bars": 600})
    assert r.status_code == 200
    body = r.json()
    assert body["lookback_bars"] == 600
    assert len(body["bars"]) == 300  # chart shows only the display window
    assert body["analysis"]["n_bars"] == 300  # regimes/context scoped to display window


def test_live_analyze_bad_interval_is_422(mock_tv, no_db):
    from patternlab.api.main import app

    with TestClient(app) as client:
        r = client.get("/live/analyze", params={"interval": "7"})
    assert r.status_code == 422


def test_live_narrate_endpoint(mock_tv, no_db, monkeypatch):
    from patternlab.agent import graph
    from patternlab.api.main import app

    seen = {}

    def fake_narrate(analysis, matches=None):
        seen["analysis"] = analysis
        return "structural read here"

    monkeypatch.setattr(graph, "narrate_live", fake_narrate)
    with TestClient(app) as client:
        r = client.get("/live/narrate", params={"n_bars": 600})
    assert r.status_code == 200
    body = r.json()
    assert body["narrative"] == "structural read here"
    assert seen["analysis"]["n_bars"] == 600  # LLM got the deterministic analysis, not raw bars


def _oanda_page(start_min: int, n: int) -> list[dict]:
    base = pd.Timestamp("2026-07-24 00:00", tz="UTC")
    return [{"time": (base + pd.Timedelta(minutes=start_min + i)).isoformat(),
             "mid": {"o": "4000.1", "h": "4001.2", "l": "3999.3", "c": "4000.7"},
             "volume": 100, "complete": True} for i in range(n)]


def test_fetch_oanda_paginates_and_normalizes(monkeypatch):
    from patternlab.config import settings
    from patternlab.data import loader

    monkeypatch.setattr(settings, "oanda_api_token", "test-token")
    calls = []

    def fake_candles(instrument, params):
        calls.append(params)
        if "to" not in params:
            return _oanda_page(5000, params["count"])  # newest page
        return _oanda_page(0, 5001)  # older page, 1-candle overlap at the seam

    monkeypatch.setattr(loader, "_oanda_candles", fake_candles)
    df = loader.fetch_oanda("XAU_USD", "M1", 8000)
    assert len(df) == 8000
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert calls[0]["count"] == 5000 and "to" in calls[1]

    with pytest.raises(ValueError):
        loader.fetch_oanda("XAU_USD", "M3", 100)


def test_fetch_oanda_day_requests_utc_day_range(monkeypatch):
    import datetime as dt

    from patternlab.config import settings
    from patternlab.data import loader

    monkeypatch.setattr(settings, "oanda_api_token", "test-token")
    seen = {}

    def fake_candles(instrument, params):
        seen.update(params)
        return _oanda_page(0, 1376)

    monkeypatch.setattr(loader, "_oanda_candles", fake_candles)
    df = loader.fetch_oanda_day("XAU_USD", dt.date(2026, 7, 21))
    assert len(df) == 1376
    assert seen["from"] == "2026-07-21T00:00:00+00:00"
    assert seen["to"] == "2026-07-22T00:00:00+00:00"
    assert "count" not in seen  # from/to and count are mutually exclusive in the v20 API


def test_live_analyze_falls_back_to_oanda(monkeypatch, no_db):
    from patternlab.api import main as api_main
    from patternlab.config import settings

    def tv_down(*a, **k):
        raise RuntimeError("websocket dead")

    monkeypatch.setattr(tradingview, "_get_hist", tv_down)
    tradingview._cache.clear()
    monkeypatch.setattr(settings, "oanda_api_token", "test-token")
    monkeypatch.setattr(api_main, "fetch_oanda",
                        lambda instrument, granularity, count: normalize_oanda_stub(instrument, granularity, count))

    def normalize_oanda_stub(instrument, granularity, count):
        assert instrument == "XAU_USD"  # XAUUSD mapped to OANDA naming
        assert granularity == "M1"
        df = synthetic_session("grind_break", n=count, seed=3)
        return df

    with TestClient(api_main.app) as client:
        r = client.get("/live/analyze", params={"symbol": "XAUUSD", "n_bars": 600})
    assert r.status_code == 200
    assert r.json()["source"] == "oanda"
    assert len(r.json()["bars"]) == 600


def test_live_sessions_requires_db(mock_tv, no_db):
    from patternlab.api.main import app

    with TestClient(app) as client:
        r = client.post("/live/sessions", params={"symbol": "XAUUSD"})
    assert r.status_code == 503
