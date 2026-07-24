import sys
sys.path.insert(0, "src")

import datetime as dt

import pytest_asyncio

from patternlab.config import DetectionParams
from patternlab.data.loader import synthetic_session
from patternlab.storage import repository
from patternlab.storage.db import Database

P = DetectionParams()


@pytest_asyncio.fixture
async def db():
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.init_models()
    async with database.session() as session:
        yield session
    await database.dispose()


async def test_save_and_list_sessions(db):
    df = synthetic_session("grind_break", seed=7)
    record = await repository.save_session(db, df, "XAU_USD", dt.date(2026, 7, 1), P)
    assert record.id is not None

    sessions = await repository.list_sessions(db, "XAU_USD")
    assert len(sessions) == 1
    assert sessions[0].n_bars == len(df)


async def test_save_session_is_idempotent_upsert(db):
    df = synthetic_session("grind_break", seed=7)
    await repository.save_session(db, df, "XAU_USD", dt.date(2026, 7, 1), P)
    await repository.save_session(db, df, "XAU_USD", dt.date(2026, 7, 1), P)

    sessions = await repository.list_sessions(db, "XAU_USD")
    assert len(sessions) == 1


async def test_find_matches_ranks_same_archetype_higher():
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.init_models()
    async with database.session() as session:
        same = synthetic_session("grind_break", seed=8)
        different = synthetic_session("v_recovery", seed=11)
        await repository.save_session(session, same, "XAU_USD", dt.date(2026, 7, 1), P)
        await repository.save_session(session, different, "XAU_USD", dt.date(2026, 7, 2), P)

        query = synthetic_session("grind_break", seed=7)
        matches = await repository.find_matches(session, query, P, instrument="XAU_USD", top_k=2)

        assert len(matches) == 2
        assert matches[0]["session_date"] == "2026-07-01"
        assert matches[0]["similarity_score"] >= matches[1]["similarity_score"]
    await database.dispose()
