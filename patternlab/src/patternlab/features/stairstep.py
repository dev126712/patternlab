"""Stair-step detection: sequential level breaks with shallow pullbacks that respect
prior resistance as support.

A stair-step through levels L1 < L2 < ... exists when, in order:
  1. price closes above L_k (break),
  2. the subsequent pullback retraces < stairstep_max_pullback of the leg that broke it,
  3. the pullback low holds >= L_k.hi - hold_tol (broken resistance acts as support),
  4. price then breaks L_{k+1}.
The result is the programmatic version of "impulsive rally that stair-steps through
4026/4030/4032 -> 4045 -> 4051-4053 -> 4061/4067 with shallow pullbacks".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from patternlab.config import DetectionParams
from patternlab.features.levels import Level
from patternlab.features.swings import atr


def detect_stairstep(df: pd.DataFrame, levels: list[Level], p: DetectionParams,
                     start_idx: int = 0) -> dict:
    """Scan df[start_idx:] for an upward stair-step through `levels` (sorted ascending)."""
    a = float(atr(df, p.atr_period).median())
    hold_tol = p.stairstep_hold_tol_atr * a
    closes = df["close"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)

    steps: list[dict] = []
    leg_origin = float(lows[start_idx:].min()) if start_idx < n else float("nan")
    li = 0
    lvls = [l for l in levels if l.center > closes[start_idx]]
    i = start_idx

    while i < n and li < len(lvls):
        lvl = lvls[li]
        # 1. find break: close above zone hi
        brk = np.argmax(closes[i:] > lvl.hi) if (closes[i:] > lvl.hi).any() else -1
        if brk == -1:
            break
        brk_i = i + int(brk)

        # 2/3. examine pullback after break until next level breaks (or end)
        nxt_hi = lvls[li + 1].hi if li + 1 < len(lvls) else np.inf
        j = brk_i + 1
        pull_low = closes[brk_i]
        held = True
        while j < n and closes[j] <= nxt_hi:
            pull_low = min(pull_low, lows[j])
            j += 1
        leg = closes[brk_i] - leg_origin
        retrace = (closes[brk_i] - pull_low) / leg if leg > 0 else 1.0
        held = pull_low >= lvl.hi - hold_tol and retrace < p.stairstep_max_pullback

        steps.append({
            "level": lvl.to_dict(),
            "break_idx": brk_i,
            "pullback_low": round(float(pull_low), 3),
            "retrace_frac": round(float(min(retrace, 1.0)), 2),
            "held_as_support": bool(held),
        })
        leg_origin = pull_low
        i = brk_i + 1
        li += 1

    clean = [s for s in steps if s["held_as_support"]]
    return {
        "steps_broken": len(steps),
        "steps_held_as_support": len(clean),
        "is_stairstep": len(clean) >= 2,
        "steps": steps,
    }
