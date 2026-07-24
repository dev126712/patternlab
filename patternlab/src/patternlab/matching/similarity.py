"""Fractal-rhyme matching.

Two complementary signals, combined:

1. Structural fingerprint (weight 0.6) — a fixed-length vector of the *interpreted*
   structure: regime mix, stair-step quality, approach angle, time-at-key-level, context
   class, momentum-candle density, level-ladder shape. Cosine similarity. This is what
   captures "same story" even when the raw shapes differ in duration/amplitude.

2. DTW shape similarity (weight 0.4) — Dynamic Time Warping distance between z-normalized,
   downsampled close series. Captures raw geometric resemblance and tolerates time
   stretching (a 3-hour version of a 6-hour pattern still matches).

Phase 4 upgrade path: replace/augment with a learned 1D-CNN encoder + pgvector ANN search
when the fingerprint DB grows past ~50k segments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from patternlab.config import DetectionParams
from patternlab.features.angle import approach_angle, time_at_level
from patternlab.features.context import classify_context
from patternlab.features.levels import extract_levels
from patternlab.features.regimes import segment_regimes
from patternlab.features.stairstep import detect_stairstep
from patternlab.features.swings import atr, momentum_candles

_CTX = {"fresh_base": 0.0, "mixed": 0.5, "recovery_leg": 1.0}
_ANGLE = {"grinding": 0.0, "moderate": 0.5, "steep": 1.0, "unknown": 0.5}


def analyze_session(df: pd.DataFrame, p: DetectionParams,
                    levels_df: pd.DataFrame | None = None) -> dict:
    """Full structural read of one session. This dict is also what the LLM agent receives.

    levels_df: optional longer-history frame ending at the same bar as df, used only for
    S/R level extraction — touch counts/weights then span more history than the analyzed
    window while regimes/context/approach stay scoped to df.
    """
    levels = extract_levels(levels_df if levels_df is not None else df, p)
    segs = segment_regimes(df, p)
    ctx = classify_context(df, p)
    close = float(df["close"].iloc[-1])

    key_level = None
    above = [l for l in levels if l.center >= close]
    below = [l for l in levels if l.center < close]
    if above:
        key_level = min(above, key=lambda l: l.center - close)
    elif levels:
        key_level = max(levels, key=lambda l: l.weight)

    stair = detect_stairstep(df, levels, p, start_idx=ctx["rally_origin_idx"] if ctx["rally_origin_idx"] < len(df) - 5 else 0)
    ang = approach_angle(df, p)
    tal = time_at_level(df, key_level.center, p) if key_level else {"bars_at_level": 0, "rejection_events": 0, "first_tap": True}
    mom_frac = float(momentum_candles(df, p).tail(120).mean())

    return {
        "n_bars": len(df),
        "close": round(close, 3),
        "atr": round(float(atr(df, p.atr_period).median()), 3),
        "levels": [l.to_dict() for l in levels],
        "key_level": key_level.to_dict() if key_level else None,
        "regimes": [s.to_dict() for s in segs],
        "context": ctx,
        "stairstep": {k: v for k, v in stair.items() if k != "steps"} | {"steps": stair["steps"][:6]},
        "approach": ang,
        "time_at_key_level": tal,
        "momentum_candle_frac_recent": round(mom_frac, 3),
    }


def fingerprint(analysis: dict) -> np.ndarray:
    """Fixed-length structural vector from analyze_session output."""
    segs = analysis["regimes"]
    n = max(len(segs), 1)
    consol = sum(1 for s in segs if s["kind"] == "consolidation") / n
    imp_up = sum(1 for s in segs if s["kind"] == "impulse_up") / n
    imp_dn = sum(1 for s in segs if s["kind"] == "impulse_down") / n
    stair = analysis["stairstep"]
    tal = analysis["time_at_key_level"]
    return np.array([
        consol, imp_up, imp_dn,
        min(stair["steps_held_as_support"], 5) / 5,
        1.0 if stair["is_stairstep"] else 0.0,
        _ANGLE.get(analysis["approach"]["classification"], 0.5),
        np.clip(abs(analysis["approach"]["slope_atr_per_bar"]) / 0.6, 0, 1),
        _CTX.get(analysis["context"]["kind"], 0.5),
        np.clip(tal["bars_at_level"] / 120, 0, 1),
        np.clip(tal["rejection_events"] / 5, 0, 1),
        0.0 if tal["first_tap"] else 1.0,
        np.clip(analysis["momentum_candle_frac_recent"] / 0.3, 0, 1),
        np.clip(len(analysis["levels"]) / 10, 0, 1),
    ], dtype=float)


def _znorm(x: np.ndarray) -> np.ndarray:
    s = x.std() or 1.0
    return (x - x.mean()) / s


def _resample(x: np.ndarray, m: int) -> np.ndarray:
    if len(x) <= m:
        return x
    idx = np.linspace(0, len(x) - 1, m)
    return np.interp(idx, np.arange(len(x)), x)


def dtw_distance(a: np.ndarray, b: np.ndarray, band_frac: float = 0.1) -> float:
    """Classic DTW with Sakoe-Chiba band. O(n*band). Inputs already normalized."""
    n, m = len(a), len(b)
    w = max(int(max(n, m) * band_frac), abs(n - m) + 1)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def compare_sessions(df_a: pd.DataFrame, df_b: pd.DataFrame, p: DetectionParams) -> dict:
    an_a, an_b = analyze_session(df_a, p), analyze_session(df_b, p)
    fa, fb = fingerprint(an_a), fingerprint(an_b)
    cos = float(fa @ fb / ((np.linalg.norm(fa) * np.linalg.norm(fb)) or 1.0))

    ca = _znorm(_resample(df_a["close"].to_numpy(), p.dtw_downsample_to))
    cb = _znorm(_resample(df_b["close"].to_numpy(), p.dtw_downsample_to))
    d = dtw_distance(ca, cb)
    dtw_sim = float(np.exp(-3.0 * d))  # map distance -> (0,1]

    score = p.weight_fingerprint * cos + p.weight_dtw * dtw_sim
    return {
        "similarity_score": round(score, 3),
        "fingerprint_cosine": round(cos, 3),
        "dtw_similarity": round(dtw_sim, 3),
        "session_a": an_a,
        "session_b": an_b,
    }
