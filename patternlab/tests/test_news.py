import datetime as dt

import pytest

from patternlab.data import news

NOW = dt.datetime(2026, 7, 24, 12, 0, tzinfo=dt.timezone.utc)


def _event(title, currency, impact, hours_from_now):
    return {"time_utc": NOW + dt.timedelta(hours=hours_from_now), "currency": currency,
            "impact": impact, "title": title, "forecast": "1.0%", "previous": "0.5%"}


@pytest.fixture()
def mock_calendar(monkeypatch):
    events = [
        _event("Core CPI m/m", "USD", "High", 2),
        _event("Unemployment Claims", "USD", "High", -3),
        _event("Bank Holiday", "USD", "Holiday", 5),
        _event("ECB Press Conference", "EUR", "High", 1),      # wrong currency
        _event("Crude Oil Inventories", "USD", "Low", 1),      # below min impact
        _event("FOMC Statement", "USD", "High", 48),           # outside window
        _event("Retail Sales m/m", "USD", "High", -20),        # too far back
    ]
    monkeypatch.setattr(news, "fetch_calendar", lambda: events)


def test_relevant_events_filters_and_sorts(mock_calendar):
    out = news.relevant_events("XAUUSD", now=NOW)
    assert [e["title"] for e in out] == ["Unemployment Claims", "Core CPI m/m", "Bank Holiday"]
    released, cpi, holiday = out
    assert released["status"] == "released" and released["minutes_until"] == -180
    assert cpi["status"] == "upcoming" and cpi["minutes_until"] == 120
    assert holiday["impact"] == "Holiday"  # holidays count as Medium — liquidity matters
    assert all(isinstance(e["time_utc"], str) for e in out)  # JSON-safe


def test_relevant_events_matches_both_symbol_halves(mock_calendar):
    out = news.relevant_events("EURUSD", now=NOW)
    assert "ECB Press Conference" in [e["title"] for e in out]
