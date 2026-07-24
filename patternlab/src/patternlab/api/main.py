"""FastAPI service.

POST /analyze  — one session (CSV upload) -> structural analysis JSON
POST /compare  — two sessions -> similarity + both analyses (+ optional LLM narrative)
GET  /health   — liveness (log-suppressed)
"""
from __future__ import annotations

import io
import logging

import pandas as pd
from fastapi import FastAPI, File, Query, UploadFile

from patternlab.config import params, settings
from patternlab.data.loader import normalize
from patternlab.matching.similarity import analyze_session, compare_sessions


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

app = FastAPI(title="PatternLab", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": settings.effective_llm_provider}


async def _read_csv(f: UploadFile) -> pd.DataFrame:
    raw = await f.read()
    return normalize(pd.read_csv(io.BytesIO(raw)))


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
