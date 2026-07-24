"""Horizontal S/R zone extraction.

Cluster swing-pivot prices with agglomerative clustering where the linkage distance
is expressed in ATR units. Each cluster becomes a zone [lo, hi] with a weight =
touch count * recency decay. This is what turns raw pivots into the
"4026/4030/4032 -> 4045 -> 4051-4053 -> 4061/4067 -> 4077" ladder a trader sees.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from patternlab.config import DetectionParams
from patternlab.features.swings import atr, zigzag_pivots


@dataclass
class Level:
    lo: float
    hi: float
    center: float
    touches: int
    weight: float
    last_touch_idx: int
    # per-touch history: [{"time": unix_s, "price": float, "kind": "H"|"L"}], chronological.
    # "H" = swing high (tested as resistance and rejected), "L" = swing low (held as support).
    events: list[dict] = field(default_factory=list)

    def contains(self, price: float, tol: float = 0.0) -> bool:
        return self.lo - tol <= price <= self.hi + tol

    def to_dict(self) -> dict:
        return {
            "lo": round(self.lo, 3), "hi": round(self.hi, 3), "center": round(self.center, 3),
            "touches": self.touches, "weight": round(self.weight, 2), "events": self.events,
        }


def extract_levels(df: pd.DataFrame, p: DetectionParams) -> list[Level]:
    piv = zigzag_pivots(df, p)
    if len(piv) < 2:
        return []
    a = float(atr(df, p.atr_period).median())
    prices = piv["price"].to_numpy().reshape(-1, 1)

    Z = linkage(prices, method="average")
    labels = fcluster(Z, t=p.level_cluster_atr * a, criterion="distance")

    n = len(df)
    levels: list[Level] = []
    for lab in np.unique(labels):
        mask = labels == lab
        pts = piv.loc[mask]
        touches = len(pts)
        if touches < p.level_min_touches:
            continue
        # recency-decayed weight so stale levels fade
        ages = n - pts["idx"].to_numpy()
        decay = np.exp(-np.log(2) * ages / p.level_recency_half_life_bars)
        events = sorted(
            ({"time": int(df.index[int(i)].timestamp()), "price": round(float(pr), 3), "kind": str(k)}
             for i, pr, k in zip(pts["idx"], pts["price"], pts["kind"])),
            key=lambda e: e["time"],
        )
        levels.append(Level(
            lo=float(pts["price"].min()),
            hi=float(pts["price"].max()),
            center=float(pts["price"].mean()),
            touches=touches,
            weight=float(touches * decay.mean() * 10),
            last_touch_idx=int(pts["idx"].max()),
            events=events,
        ))
    return sorted(levels, key=lambda l: l.center)
