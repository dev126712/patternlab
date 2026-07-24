"""ATR-scaled zigzag swing detection + ATR + momentum-candle metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from patternlab.config import DetectionParams


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR. df needs high/low/close."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def momentum_candles(df: pd.DataFrame, p: DetectionParams) -> pd.Series:
    """Boolean series: strong directional candle (large body relative to range AND to ATR)."""
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    a = atr(df, p.atr_period)
    return ((body / rng >= p.momentum_body_ratio) & (body >= p.momentum_body_atr * a)).fillna(False)


def zigzag_pivots(df: pd.DataFrame, p: DetectionParams) -> pd.DataFrame:
    """Swing highs/lows: a pivot is confirmed when price reverses by >= swing_reversal_atr * ATR.

    Returns DataFrame with columns [idx, price, kind] where kind in {"H","L"},
    idx is the integer position in df.
    """
    a = atr(df, p.atr_period).to_numpy()
    highs, lows, closes = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(df)
    if n < p.atr_period * 2:
        return pd.DataFrame(columns=["idx", "price", "kind"])

    pivots: list[tuple[int, float, str]] = []
    trend = 0  # +1 tracking a high, -1 tracking a low
    ext_i, ext_p = 0, closes[0]

    for i in range(1, n):
        thr = p.swing_reversal_atr * (a[i] if np.isfinite(a[i]) else a[~np.isnan(a)][0])
        if trend >= 0:
            if highs[i] > ext_p or trend == 0:
                if trend == 0:
                    trend = 1
                if highs[i] >= ext_p:
                    ext_i, ext_p = i, highs[i]
            if ext_p - lows[i] >= thr:            # reversal down -> confirm swing high
                pivots.append((ext_i, ext_p, "H"))
                trend, ext_i, ext_p = -1, i, lows[i]
        if trend < 0:
            if lows[i] < ext_p:
                ext_i, ext_p = i, lows[i]
            if highs[i] - ext_p >= thr:           # reversal up -> confirm swing low
                pivots.append((ext_i, ext_p, "L"))
                trend, ext_i, ext_p = 1, i, highs[i]

    return pd.DataFrame(pivots, columns=["idx", "price", "kind"])
