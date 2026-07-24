# CLAUDE.md — PatternLab

## What this project is

PatternLab is a quantitative pattern-recognition engine for financial time series
(1-minute OHLCV, primary instrument XAUUSD). It programmatically replicates the kind of
structural chart analysis a discretionary trader does by eye:

- Detect **basing/consolidation phases** vs **impulsive rallies**
- Extract **horizontal S/R level clusters** (e.g. 4026/4030/4032 → 4045 → 4051–4053 → 4061/4067 → 4077)
- Detect **stair-stepping** through levels (break → shallow pullback → prior resistance holds as support)
- Measure **angle of approach** into a level (steep/vertical vs grinding)
- Measure **time spent at resistance** (bars within a band before break/reject)
- Classify **context**: fresh-base breakout vs recovery leg after a selloff (trapped supply overhead)
- **Match sessions**: find "fractal rhymes" between the current session and historical sessions
  (structural fingerprint + DTW shape similarity), and produce a narrative comparison via LLM agent.

Reference behavior: given two sessions, output an analysis like
"both stair-step through the same level cluster, both press 4077.285; session A ground
sideways under it for hours before breaking, session B is tapping it for the first time
after a near-vertical approach off the 4045–4051 demand zone → expect chop/pullback more
likely than immediate continuation."

## Architecture

```
CSV / broker feed ──▶ data/loader.py ──▶ pandas OHLCV (UTC index)
                                            │
              ┌─────────────────────────────┼──────────────────────────────┐
              ▼                             ▼                              ▼
   features/regimes.py            features/levels.py             features/swings.py
   (consolidation/impulse         (S/R zone clustering            (ATR-scaled zigzag
    segmentation, Kaufman ER)      from swing pivots)              pivots)
              │                             │                              │
              └───────────────┬─────────────┴───────────┬──────────────────┘
                              ▼                         ▼
                 features/angle.py, candles.py,   features/context.py
                 stairstep + time-at-level        (fresh base vs recovery leg)
                              │
                              ▼
                matching/fingerprint.py  ← one vector per session/segment
                matching/dtw_matcher.py  ← shape similarity on normalized closes
                matching/similarity.py   ← weighted combined score, ranked matches
                              │
                              ▼
                agent/  (LangGraph StateGraph: analyze → compare → narrate)
                agent/llm_factory.py (provider-agnostic, same pattern as KubeAIOps)
                              │
                              ▼
                api/main.py (FastAPI)  +  scripts/demo.py
```

**Deterministic core, LLM on top.** All detection is pure numpy/pandas/scipy — reproducible,
testable, fast. The LLM agent NEVER computes numbers; it only receives structured JSON from
the tools and writes the comparative narrative. This is non-negotiable: keep math out of
prompts and prose out of detectors.

## Tech stack & conventions

- Python 3.12, **`uv`** for packaging (NOT pip/poetry). `uv sync`, `uv run`. Dockerfile uses uv.
- FastAPI + (later) async SQLAlchemy + PostgreSQL for storing session fingerprints.
- numpy / pandas / scipy only for the core. **No TA-Lib** (C dep pain on EC2, everything we
  need is ~20 lines of pandas). PyTorch only if/when we add embedding-based matching (phase 4).
- LangGraph agent, provider-agnostic `llm_factory.py`:
  `LLM_PROVIDER` ∈ {anthropic, groq, ollama, openai}. Convention from other projects:
  `DEPLOYMENT_ENV=dev` → Groq (fast, free tier), `DEPLOYMENT_ENV=prod` → Ollama on the EC2 box
  (llama3.1:8b-instruct or qwen2.5:14b — tool-calling capable models only).
- Config via pydantic-settings, `.env` file, never hardcode keys.
- bcrypt directly if auth is added later (not passlib). `HealthCheckFilter` on uvicorn logs.
- All timestamps UTC. All price thresholds ATR-normalized (never absolute $ — must work on
  XAUUSD at 4000 and EURUSD at 1.08).

## Key domain definitions (the "subjective → programmatic" mapping)

| Trader concept | Programmatic definition | Where |
|---|---|---|
| Consolidation / basing | Rolling Kaufman Efficiency Ratio < `er_lo` AND rolling range < `k_atr`·ATR for ≥ `min_bars` | `features/regimes.py` |
| Impulsive rally | ER > `er_hi`, net move > `k_imp`·ATR, majority momentum candles | `features/regimes.py` |
| Momentum candle | body/range ≥ 0.6 AND body ≥ 1.2·ATR(14) | `features/candles.py` |
| S/R level cluster | Agglomerative clustering (scipy linkage, distance = fraction of ATR) of zigzag pivot prices; zone = [min,max] of cluster, weight = touch count × recency decay | `features/levels.py` |
| Stair-stepping | Ordered sequence of level breaks where each pullback retraces < 50% of the leg AND low of pullback ≥ broken level − tolerance | `features/stairstep.py` |
| Angle of approach | OLS slope of close over last N bars into the level, in ATR-per-bar units; steep ≥ 0.30, grinding ≤ 0.12 | `features/angle.py` |
| Time at resistance | Count of bars with high within ±tol·ATR of level before break/reject; also number of distinct rejection events | `features/angle.py` |
| Fresh base vs recovery leg | Where did the rally originate relative to prior structure: drawdown from lookback high at rally origin < 0.25·range → fresh base; > 0.5 → recovery leg (trapped supply overhead) | `features/context.py` |
| Fractal rhyme | cosine similarity of structural fingerprints (0.6 weight) + DTW similarity of z-normalized closes (0.4 weight) | `matching/similarity.py` |

All thresholds live in `config.py` → `DetectionParams`. Tune them there, never inline.

## Build order (MVP roadmap)

1. ✅ **Phase 1 — deterministic core** (this repo): swings → levels → regimes → angle/context
   → fingerprint → DTW match. Validated on synthetic sessions in `scripts/demo.py`.
2. **Phase 2 — real data**: implement `data/loader.py::fetch_oanda()` (TODO markers in file),
   backfill 90 days of XAUUSD M1 into Postgres, nightly fingerprint job (Celery or cron).
3. **Phase 3 — agent + API hardening**: wire `agent/graph.py` to live tools, `/compare`
   endpoint returns narrative, add auth if exposed publicly.
4. **Phase 4 (optional)** — learned embeddings: small 1D-CNN/transformer encoder over
   normalized OHLCV windows, ANN search (pgvector) to replace/augment DTW at scale.

## Commands

```bash
uv sync                          # install
uv run python scripts/demo.py    # end-to-end demo on synthetic sessions
uv run pytest                    # tests
uv run uvicorn patternlab.api.main:app --reload   # API on :8000
docker compose up -d             # postgres + ollama + api (EC2)
```

## EC2 deployment notes

- t3.large is fine for the deterministic core + Groq. If running Ollama locally on the box,
  you need ≥ 16 GB RAM (g5.xlarge if you want GPU inference; qwen2.5:14b-q4 fits on 12 GB VRAM).
- `docker compose --profile ollama up -d` starts the local LLM sidecar; first run pulls the model.
- Keep the detection API and Ollama on the same private network; only expose FastAPI via
  nginx/ALB. Fernet-encrypt any broker API tokens at rest (same pattern as PostBridge).

## Things Claude should NOT do in this repo

- Don't add TA-Lib, backtrader, or heavyweight frameworks — core stays numpy/pandas/scipy.
- Don't let the LLM produce numbers that aren't in tool output (hallucinated levels = worthless).
- Don't use absolute price thresholds anywhere; always ATR-relative.
- Don't block the FastAPI event loop with detection on large histories — offload to a worker.

## TODO (needs real repo/account inspection)

- [ ] `data/loader.py::fetch_oanda` — needs OANDA practice-account token + account id (TODO marker in file)
- [ ] Postgres persistence for fingerprints (`storage/` module, async SQLAlchemy) — schema sketched in README
- [ ] Decide Ollama model after benchmarking tool-calling reliability on the EC2 instance
- [ ] Threshold calibration: run `scripts/demo.py --calibrate` idea against 90d of real M1 data
