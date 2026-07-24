# PatternLab

A quantitative engine that replicates discretionary structural chart analysis on 1-minute
OHLCV data (built around XAUUSD). It detects consolidation/basing phases, impulsive
stair-step rallies through clustered S/R levels, measures the angle of approach into a
level and the time spent at it, classifies context (fresh-base breakout vs recovery leg
with trapped supply overhead), and matches "fractal rhymes" between sessions using a
structural fingerprint combined with Dynamic Time Warping. A LangGraph agent turns the
structured output into a trader-style narrative comparison — the LLM never computes
numbers, it only narrates what the deterministic engine found.

## Quick start

```bash
uv sync
uv run python scripts/demo.py            # full pipeline on two synthetic sessions
uv run python scripts/demo.py --narrate  # + LLM narrative (needs GROQ_API_KEY or Ollama)
uv run pytest
uv run uvicorn patternlab.api.main:app --reload
```

The demo builds two synthetic sessions mirroring the reference XAUUSD analysis: one that
grinds under a key level for hours with repeated rejections before breaking out, and one
that sells off and then recovers steeply into the same area on a first tap. The engine
produces, for each, the level ladder, regime segmentation, stair-step quality, approach
classification, time-at-level statistics, and context class, then a combined similarity
score. Try `POST /compare?narrate=true` with two TradingView CSV exports to run the same
comparison on real data.

## How subjective trader concepts become code

The engine expresses every price threshold in ATR multiples, so the same parameters work
on gold at 4,000 and EURUSD at 1.08. A consolidation is a rolling window whose Kaufman
Efficiency Ratio falls below 0.25 while its range stays inside 3 ATR — lots of travel,
no displacement. An impulse is the opposite: ER above 0.55 with net displacement beyond
3 ATR, and its character is measured by the fraction of momentum candles (body at least
60% of range and larger than 1.2 ATR). S/R levels come from clustering zigzag swing
pivots hierarchically with an 0.8-ATR linkage distance; each cluster becomes a zone
weighted by touch count and recency decay, which is what reconstructs the
4026/4030/4032 → 4045 → 4051–4053 → 4061/4067 → 4077 ladder from raw pivots.
Stair-stepping is an ordered sequence of zone breaks where each pullback retraces under
half the leg and its low respects the broken zone as support. The angle of approach is
the OLS slope of close over the last 30 bars in ATR-per-bar units — 0.30 and above is
steep/vertical, 0.12 and below is grinding — with R² reported so you know how clean the
trendline is. Time at resistance counts bars printing within 0.6 ATR of the level and
groups them into distinct rejection events, which is precisely the difference between
"consolidated under 4077 for hours, multiple rejections" and "first tap". Context comes
from where the rally originates relative to the lookback structure: if the current
advance starts from a low printed after the lookback high, and that origin sits more
than half the range below the high, it is a recovery leg with trapped supply overhead;
if price is pushing out of a base at its highs, it is a fresh-base move.

Session matching combines two signals. The structural fingerprint is a 13-dimensional
vector of the interpreted read (regime mix, stair-step quality, angle class and
magnitude, context class, time-at-level, rejection count, first-tap flag, momentum
density, ladder richness) compared by cosine similarity — this captures "same story"
even when durations differ. DTW on z-normalized, downsampled closes with a Sakoe–Chiba
band captures raw geometric resemblance and tolerates time stretching. The combined
score weights them 0.6/0.4. All thresholds live in `config.py::DetectionParams`.

## Repository layout

| Path | Purpose |
|---|---|
| `src/patternlab/config.py` | Every tunable threshold + env settings |
| `src/patternlab/data/loader.py` | CSV loader, synthetic sessions, OANDA fetch (TODO) |
| `src/patternlab/features/` | swings, levels, regimes, angle, stairstep, context |
| `src/patternlab/matching/similarity.py` | fingerprint, DTW, combined comparison |
| `src/patternlab/agent/` | llm_factory (provider-agnostic) + LangGraph StateGraph |
| `src/patternlab/api/main.py` | FastAPI: /analyze, /compare, /health |
| `scripts/demo.py` | End-to-end demo |
| `CLAUDE.md` | Context bridge for Claude Code sessions |

## Roadmap

Phase 1 (this repo) is the deterministic core validated on synthetic data. Phase 2 wires
real data: implement `fetch_oanda()` (TODO markers in the file), backfill ~90 days of
XAUUSD M1 into Postgres, and run a nightly job that fingerprints every session so
`/compare` can rank the current session against the whole history instead of one
counterpart. Phase 3 hardens the agent and API for the EC2 deployment. Phase 4, only if
the fingerprint store grows large, replaces DTW with a small 1D-CNN encoder over
normalized OHLCV windows and pgvector ANN search.

Sketch for the Phase-2 schema: a `sessions` table (id, instrument, date, n_bars) and a
`fingerprints` table (session_id FK, vector float[13], analysis JSONB), unique on
(instrument, date).

## EC2 deployment

```bash
scp -r patternlab ec2:/opt/ && ssh ec2
cd /opt/patternlab && cp .env.example .env   # fill keys
docker compose up -d --build                  # api + postgres
docker compose --profile ollama up -d         # optional local LLM (prod)
docker exec -it patternlab-ollama-1 ollama pull qwen2.5:14b
```

Set `DEPLOYMENT_ENV=dev` to use Groq (fast, free tier, good for iteration) or
`DEPLOYMENT_ENV=prod` to route the agent to the local Ollama sidecar. The deterministic
core runs fine on a t3.large; only run Ollama on the box if it has at least 16 GB RAM
(qwen2.5:14b-q4 wants ~12 GB, llama3.1:8b-instruct fits smaller instances — verify tool
calling actually works on whichever model you pick before trusting narratives). Expose
only FastAPI through nginx or an ALB; keep Ollama and Postgres on the private network.

Nothing in this repository is trading advice; it measures structure, it does not predict.
