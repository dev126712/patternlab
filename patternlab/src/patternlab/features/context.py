"""Context engine: is the current advance a fresh-base breakout or a recovery leg?

Mechanism: locate the lookback high, then the lowest low printed AFTER it — that is the
origin of the current rally if a selloff occurred. Ask how deep that origin sits in the
prior structure:

  dd_frac = (lookback_high - rally_origin_price) / (lookback_high - lookback_low)

  high is at/near the current bar (no prior selloff to recover from) -> "fresh_base"
  dd_frac <= fresh_base_dd_max                                        -> "fresh_base"
  dd_frac >= recovery_leg_dd_min                                      -> "recovery_leg"
                                                                          (trapped supply
                                                                           overhead)
  otherwise                                                           -> "mixed"

Also reports remaining overhead structure (distance to lookback high in ATR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from patternlab.config import DetectionParams
from patternlab.features.swings import atr


def classify_context(df: pd.DataFrame, p: DetectionParams) -> dict:
    look = df.iloc[-p.context_lookback_bars:] if len(df) > p.context_lookback_bars else df
    base = len(df) - len(look)
    highs, lows = look["high"].to_numpy(), look["low"].to_numpy()
    hi, lo = float(highs.max()), float(lows.min())
    rng = hi - lo or 1.0
    a = float(atr(df, p.atr_period).median()) or 1.0
    n = len(look)

    hi_idx = int(np.argmax(highs))
    high_is_recent = hi_idx >= n - max(int(n * 0.05), 5)

    if high_is_recent or hi_idx >= n - 2:
        # price is making/at the highs of the lookback -> nothing overhead to recover through
        origin_idx = int(np.argmin(lows))
        origin_price = lo
        dd_frac = 0.0
        kind = "fresh_base"
    else:
        after = lows[hi_idx:]
        origin_rel = int(np.argmin(after))
        origin_idx = hi_idx + origin_rel
        origin_price = float(after[origin_rel])
        dd_frac = (hi - origin_price) / rng
        if dd_frac <= p.fresh_base_dd_max:
            kind = "fresh_base"
        elif dd_frac >= p.recovery_leg_dd_min:
            kind = "recovery_leg"
        else:
            kind = "mixed"

    close = float(df["close"].iloc[-1])
    return {
        "kind": kind,
        "rally_origin_price": round(origin_price, 3),
        "rally_origin_idx": base + origin_idx,
        "lookback_high": round(hi, 3),
        "lookback_low": round(lo, 3),
        "origin_drawdown_frac": round(float(dd_frac), 2),
        "overhead_to_lookback_high_atr": round((hi - close) / a, 2),
        "trapped_supply_overhead": kind == "recovery_leg" and close < hi,
    }
