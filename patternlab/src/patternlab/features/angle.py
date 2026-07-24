"""Angle of approach into a level + time spent at a level.

"Angle" on a chart is meaningless in raw units (depends on axis scaling), so we define it
as OLS slope of close over the approach window, normalized to ATR-per-bar:
    slope_atr = OLS_slope(close[-N:]) / median_ATR
>= steep_slope_atr  -> "steep" (near-vertical trendline, first-tap risk of rejection)
<= grinding_slope_atr -> "grinding" (energy-building consolidation drift)
otherwise -> "moderate".

Time-at-level counts bars whose high prints within +/- tol*ATR of the level before the
current bar, and groups them into distinct rejection events (gap >= rejection_min_gap_bars).
This is what distinguishes "consolidated under 4077 for hours, multiple rejections"
from "first tap".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from patternlab.config import DetectionParams
from patternlab.features.swings import atr


def approach_angle(df: pd.DataFrame, p: DetectionParams, end_idx: int | None = None) -> dict:
    end = len(df) if end_idx is None else end_idx + 1
    window = df.iloc[max(0, end - p.approach_lookback):end]
    if len(window) < 5:
        return {"slope_atr_per_bar": 0.0, "classification": "unknown", "r2": 0.0}

    y = window["close"].to_numpy()
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = ((y - y.mean()) ** 2).sum() or 1.0
    r2 = 1 - (resid ** 2).sum() / ss_tot

    med_atr = float(atr(df, p.atr_period).iloc[max(0, end - p.approach_lookback):end].median()) or 1.0
    slope_atr = slope / med_atr

    if abs(slope_atr) >= p.steep_slope_atr:
        cls = "steep"
    elif abs(slope_atr) <= p.grinding_slope_atr:
        cls = "grinding"
    else:
        cls = "moderate"
    return {
        "slope_atr_per_bar": round(float(slope_atr), 3),
        "direction": "up" if slope > 0 else "down",
        "classification": cls,
        "r2": round(float(r2), 3),
        "lookback_bars": len(window),
    }


def time_at_level(df: pd.DataFrame, level: float, p: DetectionParams,
                  end_idx: int | None = None) -> dict:
    end = len(df) if end_idx is None else end_idx + 1
    a = atr(df, p.atr_period)
    tol = float(a.iloc[:end].median()) * p.at_level_tol_atr

    highs = df["high"].to_numpy()[:end]
    at = np.where(np.abs(highs - level) <= tol)[0]
    if len(at) == 0:
        return {"bars_at_level": 0, "rejection_events": 0, "first_tap": True}

    # group contiguous-ish touches into distinct rejection events
    events = 1
    for i in range(1, len(at)):
        if at[i] - at[i - 1] >= p.rejection_min_gap_bars:
            events += 1

    # exclude the touches belonging to the current (latest) event when deciding "first tap"
    last_event_start = at[-1]
    for i in range(len(at) - 1, 0, -1):
        if at[i] - at[i - 1] >= p.rejection_min_gap_bars:
            break
        last_event_start = at[i - 1]
    prior_touches = int((at < last_event_start).sum())

    return {
        "bars_at_level": int(len(at)),
        "rejection_events": int(events),
        "first_tap": prior_touches == 0,
        "current_event_bars": int((at >= last_event_start).sum()),
    }
