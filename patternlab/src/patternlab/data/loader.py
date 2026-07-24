"""Data loading. CSV path works today; OANDA fetch is the Phase-2 TODO.

Canonical frame contract used everywhere in this repo:
  - DatetimeIndex, UTC, ascending, no duplicates
  - columns: open, high, low, close, volume (float)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("time", "timestamp", "datetime", "date"):
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand], utc=True)
                df = df.set_index(cand)
                break
        else:
            raise ValueError("No datetime index or time column found")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[REQUIRED + ["volume"]].astype(float)


def load_csv(path: str) -> pd.DataFrame:
    """Load any TradingView/broker CSV export into the canonical frame."""
    return normalize(pd.read_csv(path))


def fetch_oanda(instrument: str = "XAU_USD", granularity: str = "M1",
                count: int = 5000) -> pd.DataFrame:
    """Fetch candles from OANDA v20 REST API.

    TODO(alexandre): implement with real credentials — needs practice-account token
    (settings.oanda_api_token) and account id. Endpoint:
      GET https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles
          ?granularity=M1&count=5000&price=M
      Authorization: Bearer <token>
    Parse response["candles"] -> rows (time, mid.o/h/l/c, volume) -> normalize().
    Paginate with `from`/`to` for backfills > 5000 candles.
    """
    raise NotImplementedError("Phase 2 — see TODO above and CLAUDE.md build order")


def synthetic_session(kind: str, n: int = 900, base: float = 4000.0,
                      seed: int = 0) -> pd.DataFrame:
    """Synthetic M1 sessions for testing/demo. Two archetypes mirroring the reference
    XAUUSD analysis:

    - "grind_break":  base -> stair-step rally -> long sideways grind under the key level
                      with multiple rejections -> breakout. (fresh-base analog)
    - "v_recovery":   rally to a high -> selloff -> sharp near-vertical recovery leg into
                      the same key level, first tap. (recovery-leg analog)
    """
    rng = np.random.default_rng(seed)
    noise = lambda scale, m=n: rng.normal(0, scale, m)  # noqa: E731

    if kind == "grind_break":
        seg = [
            np.full(200, 0.0) + np.cumsum(noise(0.6, 200)) * 0.3,            # basing
            np.linspace(0, 45, 250) + np.cumsum(noise(0.8, 250)) * 0.4,      # stair-step rally
            np.full(350, 45.0) + np.cumsum(noise(0.7, 350)) * 0.25
            + 3 * np.sin(np.linspace(0, 9 * np.pi, 350)),                    # grind under level, rejections
            np.linspace(45, 75, 100) + np.cumsum(noise(0.9, 100)) * 0.3,     # breakout
        ]
    elif kind == "v_recovery":
        seg = [
            np.linspace(0, 55, 200) + np.cumsum(noise(0.8, 200)) * 0.3,      # rally to high
            np.linspace(55, 15, 250) + np.cumsum(noise(0.9, 250)) * 0.35,    # selloff
            np.full(250, 15.0) + np.cumsum(noise(0.6, 250)) * 0.25,          # demand-zone base
            np.linspace(15, 46, 200) + np.cumsum(noise(0.7, 200)) * 0.2,     # steep recovery, first tap
        ]
    else:
        raise ValueError(kind)

    path = base + np.concatenate(seg)[:n]
    m = len(path)
    close = path
    open_ = np.roll(close, 1); open_[0] = close[0]
    spread = np.abs(rng.normal(0.8, 0.4, m)) + 0.2
    high = np.maximum(open_, close) + spread * rng.random(m)
    low = np.minimum(open_, close) - spread * rng.random(m)
    idx = pd.date_range("2026-07-24 00:00", periods=m, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(50, 500, m).astype(float)}, index=idx)
