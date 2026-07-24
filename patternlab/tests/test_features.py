import sys
sys.path.insert(0, "src")

import numpy as np
from patternlab.config import DetectionParams
from patternlab.data.loader import synthetic_session
from patternlab.features.swings import zigzag_pivots
from patternlab.features.levels import extract_levels
from patternlab.features.regimes import segment_regimes
from patternlab.features.angle import approach_angle
from patternlab.features.context import classify_context
from patternlab.matching.similarity import compare_sessions, fingerprint

P = DetectionParams()


def test_synthetic_frames_are_canonical():
    df = synthetic_session("grind_break")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


def test_pivots_and_levels():
    df = synthetic_session("grind_break", seed=7)
    piv = zigzag_pivots(df, P)
    assert len(piv) >= 4
    levels = extract_levels(df, P)
    assert len(levels) >= 2
    assert all(l.touches >= P.level_min_touches for l in levels)


def test_regimes_detects_structure():
    df = synthetic_session("grind_break", seed=7)
    kinds = {s.kind for s in segment_regimes(df, P)}
    assert "consolidation" in kinds or "impulse_up" in kinds


def test_angle_direction():
    a = approach_angle(synthetic_session("v_recovery", seed=11), P)
    assert a["direction"] == "up"
    assert a["slope_atr_per_bar"] > 0


def test_context_classification_differs():
    a = classify_context(synthetic_session("grind_break", seed=7), P)
    b = classify_context(synthetic_session("v_recovery", seed=11), P)
    assert b["kind"] in ("recovery_leg", "mixed")
    assert a["kind"] in ("fresh_base", "mixed")


def test_compare_end_to_end():
    r = compare_sessions(synthetic_session("grind_break", seed=7),
                         synthetic_session("v_recovery", seed=11), P)
    assert 0 <= r["similarity_score"] <= 1.001
    fa = fingerprint(r["session_a"])
    assert fa.shape[0] == 13 and np.isfinite(fa).all()
