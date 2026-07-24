"""Regime segmentation: consolidation/basing vs impulsive rally vs neutral.

Core signal: Kaufman Efficiency Ratio over a rolling window
    ER = |close_t - close_{t-w}| / sum(|close_i - close_{i-1}|)
ER near 0 -> choppy sideways (consolidation candidate, confirmed by tight rolling range).
ER near 1 -> one-directional (impulse candidate, confirmed by net move in ATR terms and
momentum-candle density).
Adjacent same-label bars are merged into segments; segments shorter than min_regime_bars
are dropped to neutral.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from patternlab.config import DetectionParams
from patternlab.features.swings import atr, momentum_candles


@dataclass
class Segment:
    start: int
    end: int          # inclusive
    kind: str         # "consolidation" | "impulse_up" | "impulse_down" | "neutral"
    net_move_atr: float
    er: float
    momentum_frac: float

    def to_dict(self) -> dict:
        return {
            "start": self.start, "end": self.end, "bars": self.end - self.start + 1,
            "kind": self.kind, "net_move_atr": round(self.net_move_atr, 2),
            "er": round(self.er, 2), "momentum_candle_frac": round(self.momentum_frac, 2),
        }


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    change = (close - close.shift(window)).abs()
    vol = close.diff().abs().rolling(window).sum()
    return (change / vol.replace(0, np.nan)).clip(0, 1)


def label_bars(df: pd.DataFrame, p: DetectionParams) -> pd.Series:
    close = df["close"]
    a = atr(df, p.atr_period)
    er = efficiency_ratio(close, p.regime_window)
    roll_hi = df["high"].rolling(p.regime_window).max()
    roll_lo = df["low"].rolling(p.regime_window).min()
    roll_range_atr = (roll_hi - roll_lo) / a
    net_move_atr = (close - close.shift(p.regime_window)) / a

    labels = pd.Series("neutral", index=df.index)
    labels[(er <= p.er_consolidation_max) & (roll_range_atr <= p.consol_range_atr_max)] = "consolidation"
    up = (er >= p.er_impulse_min) & (net_move_atr >= p.impulse_net_move_atr_min)
    dn = (er >= p.er_impulse_min) & (net_move_atr <= -p.impulse_net_move_atr_min)
    labels[up] = "impulse_up"
    labels[dn] = "impulse_down"
    return labels


def segment_regimes(df: pd.DataFrame, p: DetectionParams) -> list[Segment]:
    labels = label_bars(df, p).to_numpy()
    close = df["close"].to_numpy()
    a = atr(df, p.atr_period).to_numpy()
    mom = momentum_candles(df, p).to_numpy()

    segs: list[Segment] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            end = i - 1
            kind = labels[start]
            if kind != "neutral" and end - start + 1 >= p.min_regime_bars:
                sl = slice(start, end + 1)
                med_atr = np.nanmedian(a[sl]) or 1.0
                net = (close[end] - close[start]) / med_atr
                path = np.nansum(np.abs(np.diff(close[sl]))) or 1.0
                segs.append(Segment(
                    start=start, end=end, kind=kind,
                    net_move_atr=float(net),
                    er=float(min(abs(close[end] - close[start]) / path, 1.0)),
                    momentum_frac=float(mom[sl].mean()),
                ))
            start = i
    return segs
