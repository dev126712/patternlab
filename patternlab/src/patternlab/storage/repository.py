"""Persist session analyses and rank a session against the stored history.

Similarity here is the identical weighted cosine(fingerprint) + DTW(closes) combo as
matching/similarity.py::compare_sessions, just evaluated against every stored fingerprint
instead of a single counterpart. Computed in Python (not pgvector) since the fingerprint
table is small (thousands, not millions, of sessions) — see CLAUDE.md Phase 4 for the
learned-embedding/ANN upgrade path if that changes.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from patternlab.config import DetectionParams
from patternlab.matching.similarity import _resample, _znorm, analyze_session, dtw_distance, fingerprint
from patternlab.storage.models import FingerprintRecord, SessionRecord


async def save_session(
    db: AsyncSession, df: pd.DataFrame, instrument: str, session_date: dt.date, p: DetectionParams
) -> SessionRecord:
    """Analyze df and upsert (instrument, session_date) with its fingerprint + full analysis."""
    analysis = analyze_session(df, p)
    vec = fingerprint(analysis).tolist()
    closes = _znorm(_resample(df["close"].to_numpy(), p.dtw_downsample_to)).tolist()

    result = await db.execute(
        select(SessionRecord).where(
            SessionRecord.instrument == instrument, SessionRecord.session_date == session_date
        )
    )
    record = result.scalar_one_or_none()

    if record is None:
        record = SessionRecord(instrument=instrument, session_date=session_date, n_bars=len(df))
        db.add(record)
        await db.flush()
        db.add(FingerprintRecord(session_id=record.id, vector=vec, closes_znorm=closes, analysis=analysis))
    else:
        record.n_bars = len(df)
        fp_result = await db.execute(
            select(FingerprintRecord).where(FingerprintRecord.session_id == record.id)
        )
        fp = fp_result.scalar_one()
        fp.vector, fp.closes_znorm, fp.analysis = vec, closes, analysis

    await db.commit()
    await db.refresh(record)
    return record


async def get_session(
    db: AsyncSession, session_id: int
) -> tuple[SessionRecord, FingerprintRecord] | None:
    stmt = select(SessionRecord, FingerprintRecord).join(
        FingerprintRecord, FingerprintRecord.session_id == SessionRecord.id
    ).where(SessionRecord.id == session_id)
    row = (await db.execute(stmt)).first()
    return (row[0], row[1]) if row else None


async def list_sessions(db: AsyncSession, instrument: str | None = None) -> list[SessionRecord]:
    stmt = select(SessionRecord).order_by(SessionRecord.session_date.desc())
    if instrument:
        stmt = stmt.where(SessionRecord.instrument == instrument)
    result = await db.execute(stmt)
    return list(result.scalars())


async def find_matches(
    db: AsyncSession,
    df: pd.DataFrame,
    p: DetectionParams,
    instrument: str | None = None,
    top_k: int = 5,
    exclude_session_id: int | None = None,
    exclude_session_date: dt.date | None = None,
) -> list[dict]:
    """Rank stored sessions by fingerprint-cosine + DTW similarity against df, best first."""
    analysis = analyze_session(df, p)
    vec = fingerprint(analysis)
    closes = _znorm(_resample(df["close"].to_numpy(), p.dtw_downsample_to))

    stmt = select(SessionRecord, FingerprintRecord).join(
        FingerprintRecord, FingerprintRecord.session_id == SessionRecord.id
    )
    if instrument:
        stmt = stmt.where(SessionRecord.instrument == instrument)
    if exclude_session_id is not None:
        stmt = stmt.where(SessionRecord.id != exclude_session_id)
    if exclude_session_date is not None:
        stmt = stmt.where(SessionRecord.session_date != exclude_session_date)
    result = await db.execute(stmt)
    rows = result.all()

    def _score_all() -> list[dict]:
        scored: list[dict] = []
        for rec, fp in rows:
            cand_vec = np.array(fp.vector, dtype=float)
            cos = float(vec @ cand_vec / ((np.linalg.norm(vec) * np.linalg.norm(cand_vec)) or 1.0))
            d = dtw_distance(closes, np.array(fp.closes_znorm, dtype=float))
            dtw_sim = float(np.exp(-3.0 * d))
            score = p.weight_fingerprint * cos + p.weight_dtw * dtw_sim
            scored.append({
                "session_id": rec.id,
                "instrument": rec.instrument,
                "session_date": rec.session_date.isoformat(),
                "similarity_score": round(score, 3),
                "fingerprint_cosine": round(cos, 3),
                "dtw_similarity": round(dtw_sim, 3),
                "analysis": fp.analysis,
            })
        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]

    # Pure-Python DTW over the whole library is CPU-bound — with ~100 stored sessions it's
    # around a second, which must not stall the event loop on every live refresh.
    return await run_in_threadpool(_score_all)
