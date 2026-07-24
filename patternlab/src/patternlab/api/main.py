"""FastAPI service.

POST /analyze  — one session (CSV upload) -> structural analysis JSON
POST /compare  — two sessions -> similarity + both analyses (+ optional LLM narrative)
POST /sessions — persist one session's fingerprint + analysis (upsert on instrument+date)
GET  /sessions — list stored sessions
POST /match    — one session (CSV upload) -> ranked fingerprint/DTW matches against history
GET  /sessions/{id}/bars — stored session re-hydrated from OANDA (bars + stored analysis)
GET  /live/analyze  — fetch from TradingView (tvDatafeed) -> bars + analysis (+ matches if DB)
GET  /live/compare  — live session vs one stored session (+ optional LLM narrative)
GET  /live/narrate  — LLM read of the current live session
POST /live/sessions — fetch from TradingView and persist as a session
GET  /health   — liveness (log-suppressed)

Persistence endpoints (/sessions, /match, /live/sessions) require DATABASE_URL; they 503
without it. /live/analyze works without a DB (matches just come back null).
"""
from __future__ import annotations

import datetime as dt
import io
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from patternlab.config import params, settings
from patternlab.data.loader import fetch_oanda, fetch_oanda_day, normalize
from patternlab.data.tradingview import fetch_tradingview
from patternlab.matching.similarity import analyze_session, compare_sessions
from patternlab.storage import repository
from patternlab.storage.db import get_database


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if settings.database_url:
        await get_database().init_models()
    yield


app = FastAPI(title="PatternLab", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": settings.effective_llm_provider,
            "persistence": bool(settings.database_url)}


async def _read_csv(f: UploadFile) -> pd.DataFrame:
    raw = await f.read()
    return normalize(pd.read_csv(io.BytesIO(raw)))


async def _get_db() -> AsyncIterator[AsyncSession]:
    if not settings.database_url:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")
    async with get_database().session() as session:
        yield session


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    df = await _read_csv(file)
    return analyze_session(df, params)


@app.post("/compare")
async def compare(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    narrate: bool = Query(False, description="Run the LLM agent for a narrative comparison"),
) -> dict:
    df_a, df_b = await _read_csv(file_a), await _read_csv(file_b)
    result = compare_sessions(df_a, df_b, params)
    if narrate:
        # NOTE: for large histories move this to a worker (Celery) — don't block the loop.
        from patternlab.agent.graph import build_graph
        out = build_graph().invoke({"df_a": df_a, "df_b": df_b})
        result["narrative"] = out.get("narrative", "")
    return result


@app.post("/sessions")
async def create_session(
    file: UploadFile = File(...),
    instrument: str = Query(..., description="e.g. XAU_USD"),
    session_date: dt.date = Query(..., description="Date this session/segment represents"),
    db: AsyncSession = Depends(_get_db),
) -> dict:
    df = await _read_csv(file)
    record = await repository.save_session(db, df, instrument, session_date, params)
    return {"session_id": record.id, "instrument": record.instrument,
            "session_date": record.session_date.isoformat(), "n_bars": record.n_bars}


@app.get("/sessions")
async def get_sessions(
    instrument: str | None = Query(None),
    db: AsyncSession = Depends(_get_db),
) -> list[dict]:
    records = await repository.list_sessions(db, instrument)
    return [{"session_id": r.id, "instrument": r.instrument,
             "session_date": r.session_date.isoformat(), "n_bars": r.n_bars} for r in records]


@app.post("/match")
async def match(
    file: UploadFile = File(...),
    instrument: str | None = Query(None, description="Restrict matches to this instrument"),
    top_k: int = Query(5, ge=1, le=50),
    db: AsyncSession = Depends(_get_db),
) -> dict:
    df = await _read_csv(file)
    matches = await repository.find_matches(db, df, params, instrument=instrument, top_k=top_k)
    return {"query_analysis": analyze_session(df, params), "matches": matches}


async def _hydrate_session(db: AsyncSession, session_id: int):
    """Stored session row + its raw bars re-fetched from OANDA (the DB has no bars)."""
    row = await repository.get_session(db, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    rec, fp = row
    oanda_name = _oanda_instrument(rec.instrument.split(":")[-1])
    try:
        df = await run_in_threadpool(fetch_oanda_day, oanda_name, rec.session_date)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OANDA fetch for {rec.session_date} failed: {e}")
    return rec, fp, df


@app.get("/sessions/{session_id}/bars")
async def session_bars(session_id: int, db: AsyncSession = Depends(_get_db)) -> dict:
    """Re-hydrated bars + stored analysis for one stored session (for the compare view)."""
    rec, fp, df = await _hydrate_session(db, session_id)
    return {"session_id": rec.id, "instrument": rec.instrument,
            "session_date": rec.session_date.isoformat(),
            "bars": _bars_json(df), "analysis": fp.analysis}


@app.get("/live/compare")
async def live_compare(
    session_id: int = Query(..., description="Stored session to compare the live session against"),
    symbol: str = Query("XAUUSD"),
    exchange: str = Query("OANDA"),
    interval: str = Query("1"),
    n_bars: int = Query(1500, ge=100, le=5000),
    narrate: bool = Query(True, description="Run the LLM agent for a narrative comparison"),
    db: AsyncSession = Depends(_get_db),
) -> dict:
    """Compare the current live session against a stored one. On-demand only — not part
    of the auto-refresh loop (DTW + optional LLM call)."""
    rec, _, df_hist = await _hydrate_session(db, session_id)
    df_live, _ = await _fetch_live(symbol, exchange, interval, n_bars)

    def _run() -> dict:
        if narrate:
            from patternlab.agent.graph import build_graph
            state = build_graph().invoke({"df_a": df_hist, "df_b": df_live})
            return state["comparison"] | {"narrative": state.get("narrative", "")}
        return compare_sessions(df_hist, df_live, params)

    try:
        # CPU-bound DTW + blocking LLM call — threadpool, never on the loop.
        result = await run_in_threadpool(_run)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Comparison failed: {e}")
    return {"session_id": rec.id, "session_date": rec.session_date.isoformat(),
            "instrument": rec.instrument,
            "similarity_score": result["similarity_score"],
            "fingerprint_cosine": result["fingerprint_cosine"],
            "dtw_similarity": result["dtw_similarity"],
            "narrative": result.get("narrative")}


def _bars_json(df: pd.DataFrame) -> list[dict]:
    """Bars as lightweight-charts feed format (time = unix seconds, UTC)."""
    # .timestamp() per bar: index resolution varies by source (ns from CSV, us from
    # tvdatafeed), so int64-view arithmetic is unit-unsafe.
    ts = [int(t.timestamp()) for t in df.index]
    cols = [df[c].tolist() for c in ("open", "high", "low", "close", "volume")]
    return [{"time": t, "open": o, "high": h, "low": lo, "close": c, "volume": v}
            for t, o, h, lo, c, v in zip(ts, *cols)]


# TradingView interval -> OANDA granularity (TV's 3/45-minute have no OANDA equivalent)
_OANDA_GRANULARITY = {"1": "M1", "5": "M5", "15": "M15", "30": "M30",
                      "1H": "H1", "2H": "H2", "4H": "H4", "1D": "D", "1W": "W", "1M": "M"}


def _oanda_instrument(symbol: str) -> str:
    s = symbol.upper()
    return f"{s[:3]}_{s[3:]}" if "_" not in s and len(s) == 6 else s


async def _fetch_live(symbol: str, exchange: str, interval: str, n_bars: int) -> tuple[pd.DataFrame, str]:
    """Returns (bars, source). tvDatafeed is blocking — always fetch on the threadpool,
    never on the loop. If TradingView fails (unofficial API, no SLA) and OANDA credentials
    are configured, falls back to the official OANDA feed for OANDA-exchange symbols."""
    try:
        return await run_in_threadpool(fetch_tradingview, symbol, exchange, interval, n_bars), "tradingview"
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        gran = _OANDA_GRANULARITY.get(interval)
        if settings.oanda_api_token and exchange.upper() == "OANDA" and gran:
            try:
                df = await run_in_threadpool(fetch_oanda, _oanda_instrument(symbol), gran, n_bars)
                return df, "oanda"
            except Exception as e2:
                raise HTTPException(status_code=502,
                                    detail=f"TradingView failed ({e}); OANDA fallback failed: {e2}")
        raise HTTPException(status_code=502, detail=f"TradingView fetch failed: {e}")


async def _live_analysis(symbol: str, exchange: str, interval: str, n_bars: int,
                         lookback_bars: int | None, top_k: int) -> dict:
    """Shared fetch + analyze pipeline for /live/analyze and /live/narrate."""
    full, source = await _fetch_live(symbol, exchange, interval, max(n_bars, lookback_bars or 0))
    df = full.tail(n_bars) if len(full) > n_bars else full
    levels_df = full if len(full) > len(df) else None
    matches = None
    if settings.database_url:
        async with get_database().session() as db:
            # A stored session for the current day is (partly) the query itself — noise.
            matches = await repository.find_matches(db, df, params, top_k=top_k,
                                                    exclude_session_date=df.index[-1].date())
    return {"symbol": symbol.upper(), "exchange": exchange.upper(), "interval": interval,
            "source": source,
            "lookback_bars": len(levels_df) if levels_df is not None else len(df),
            "bars": _bars_json(df), "analysis": analyze_session(df, params, levels_df=levels_df),
            "matches": matches}


@app.get("/live/analyze")
async def live_analyze(
    symbol: str = Query("XAUUSD"),
    exchange: str = Query("OANDA"),
    interval: str = Query("1", description="TradingView interval: 1,3,5,15,30,45,1H,2H,4H,1D,1W,1M"),
    n_bars: int = Query(1500, ge=100, le=5000),
    lookback_bars: int | None = Query(None, ge=100, le=5000,
                                      description="Extract S/R levels from this many bars while charting/analyzing only n_bars"),
    top_k: int = Query(5, ge=1, le=50),
) -> dict:
    return await _live_analysis(symbol, exchange, interval, n_bars, lookback_bars, top_k)


@app.get("/live/narrate")
async def live_narrate(
    symbol: str = Query("XAUUSD"),
    exchange: str = Query("OANDA"),
    interval: str = Query("1"),
    n_bars: int = Query(1500, ge=100, le=5000),
    lookback_bars: int | None = Query(None, ge=100, le=5000),
    top_k: int = Query(5, ge=1, le=50),
) -> dict:
    """LLM read of the current live session. On-demand only — the frontend must NOT call
    this on its auto-refresh loop (slow, costs tokens)."""
    result = await _live_analysis(symbol, exchange, interval, n_bars, lookback_bars, top_k)
    from patternlab.agent.graph import narrate_live
    try:
        # Sync LLM call — threadpool, never on the event loop.
        narrative = await run_in_threadpool(narrate_live, result["analysis"], result["matches"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM narration failed: {e}")
    return {"narrative": narrative, "llm_provider": settings.effective_llm_provider,
            "as_of": result["bars"][-1]["time"] if result["bars"] else None}


@app.post("/live/sessions")
async def save_live_session(
    symbol: str = Query("XAUUSD"),
    exchange: str = Query("OANDA"),
    interval: str = Query("1"),
    n_bars: int = Query(1500, ge=100, le=5000),
    db: AsyncSession = Depends(_get_db),
) -> dict:
    df, _ = await _fetch_live(symbol, exchange, interval, n_bars)
    instrument = f"{exchange}:{symbol}".upper()
    record = await repository.save_session(db, df, instrument, df.index[-1].date(), params)
    return {"session_id": record.id, "instrument": record.instrument,
            "session_date": record.session_date.isoformat(), "n_bars": record.n_bars}
