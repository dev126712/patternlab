"""SQLAlchemy models for session/fingerprint persistence.

One row per (instrument, session_date) in `sessions`, with its structural fingerprint,
DTW-ready closes, and full analysis JSON in `fingerprints` — this is what lets /match
rank a session against the whole stored history without recomputing anything.

`vector` / `closes_znorm` / `analysis` use the generic JSON type (not Postgres ARRAY/JSONB)
so the same schema works against Postgres in prod and SQLite in tests.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "sessions"
    __table_args__ = (UniqueConstraint("instrument", "session_date", name="uq_session_instrument_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument: Mapped[str] = mapped_column(String(20), index=True)
    session_date: Mapped[dt.date]
    n_bars: Mapped[int]
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class FingerprintRecord(Base):
    __tablename__ = "fingerprints"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), unique=True, index=True
    )
    vector: Mapped[list] = mapped_column(JSON)         # 13-dim structural fingerprint
    closes_znorm: Mapped[list] = mapped_column(JSON)   # downsampled z-normalized closes, for DTW
    analysis: Mapped[dict] = mapped_column(JSON)       # full analyze_session() output
