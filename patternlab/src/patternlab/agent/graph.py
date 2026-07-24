"""LangGraph analysis agent: three-node StateGraph (analyze -> compare -> narrate).

Design rule (see CLAUDE.md): the LLM never computes numbers. Nodes 1-2 are deterministic
Python calling the detection engine; only node 3 invokes the LLM, and its prompt contains
the full structured JSON so every number in the narrative is grounded in tool output.

Usage:
    from patternlab.agent.graph import build_graph
    graph = build_graph()
    out = graph.invoke({"df_a": df_a, "df_b": df_b})
    print(out["narrative"])
"""
from __future__ import annotations

import json
from typing import Any, TypedDict

import pandas as pd

from patternlab.config import params
from patternlab.matching.similarity import analyze_session, compare_sessions


class AgentState(TypedDict, total=False):
    df_a: pd.DataFrame
    df_b: pd.DataFrame
    analysis_a: dict
    analysis_b: dict
    comparison: dict
    narrative: str
    narrative_error: str


SYSTEM_PROMPT = """You are an expert quantitative price-action analyst. You receive a JSON
comparison of two trading sessions produced by a deterministic detection engine. Write a
concise structural comparison in the style of a professional trader's read:

1. "The shared pattern" — common structure (consolidation -> impulse, stair-stepping
   through the level ladder, momentum character).
2. "Where they converge" — the shared key level and how each session behaves at it.
3. "Key differences" — numbered: angle of approach (steep vs grinding, cite slope values),
   context (fresh base vs recovery leg / trapped supply), time at resistance
   (rejection_events, first_tap).
4. One-sentence implication of the historical analog, hedged appropriately.

STRICT RULES: every number you mention MUST appear in the JSON. Never invent levels,
slopes, or counts. If a field is missing, omit that point. No trading advice beyond the
structural implication."""


LIVE_SYSTEM_PROMPT = """You are an expert quantitative price-action analyst. You receive the
structural analysis JSON of the CURRENT live session, produced by a deterministic detection
engine, plus (optionally) its closest historical matches. Write a concise professional
trader's read:

1. "Structure so far" — the regime sequence (consolidation vs impulse), stair-stepping
   through the level ladder, momentum character.
2. "At the key level" — where price sits relative to the key level, angle of approach
   (cite the slope value), time spent there (bars_at_level, rejection_events, first_tap).
3. "Context" — fresh base vs recovery leg, trapped supply overhead.
4. If matches are present: the closest historical rhyme (cite its similarity_score) and
   how its structure compares.
5. "News context" — only if the news array is present and non-empty: the relevant
   released and upcoming calendar events (cite title, impact, and minutes_until from the
   JSON; negative minutes_until = already released, compare actual context via
   forecast/previous), and how proximity to a high-impact release typically interacts
   with the structure (chop, fakeouts, spread widening into the print). Skip this
   section entirely when the array is empty or absent.
6. One-sentence structural implication, hedged appropriately.
7. "A professional's read (15m / 2h)" — close with how a professional trader watching this
   same session on a 15-minute and a 2-hour chart would frame it: what the 1-minute
   structure compresses into on each timeframe (e.g. a stair-step ladder becomes a single
   impulse leg on 2h; a long grind at a level becomes one or two indecision candles on 15m),
   which levels from the JSON would stand out as the obvious higher-timeframe references,
   and how such a trader would typically frame entries and invalidation around those exact
   levels. Hedged, scenario-based ("if it holds X... / if it loses Y..."), never a directive.

STRICT RULES: every number you mention MUST appear in the JSON. Never invent levels,
slopes, counts, or news events — this includes section 7: the higher-timeframe framing
must reference only levels present in the JSON, never new ones. If a field is missing,
omit that point. No trading advice beyond the structural implication and the scenario
framing of section 7."""


CHAT_SYSTEM_PROMPT = """You are an expert quantitative price-action analyst answering a
trader's follow-up questions about the CURRENT live session. You are given the structural
analysis JSON produced by a deterministic detection engine (plus optionally its closest
historical matches). That JSON is your ONLY source of numbers.

Typical questions map to JSON fields:
- "solid / strong highs and lows", "key support/resistance" -> the levels array (highest
  weight and most touches = most solid), plus context.lookback_high / lookback_low.
- "where is price" -> close relative to key_level; approach; time_at_key_level.
- "what's the structure / trend" -> regimes, stairstep, context, momentum_candle_frac.
- "any news / what's coming up / why so choppy" -> the news array: economic-calendar
  events with title, impact, time_utc, minutes_until (negative = already released),
  forecast, previous. If the array is empty or absent, say no relevant calendar events
  are in the data — never invent one.

Answer directly and concisely, citing the specific values. STRICT RULES: every number you
mention MUST appear in the JSON. Never invent levels, slopes, or counts. If the JSON cannot
answer the question, say so plainly instead of guessing. Frame any forward-looking view as
hedged scenarios around JSON levels ("if it holds X / if it loses Y"), never a directive
to buy or sell."""


# Levels beyond the heaviest N add nothing to the narrative but grow the prompt on long
# lookbacks (5000-bar level extraction). Prompt-size knob, not a detection threshold.
_MAX_LEVELS_FOR_LLM = 12


def _slim_level(lvl: dict | None) -> dict | None:
    """Per-touch event lists are for the UI's level drill-down — pure token bloat for the LLM."""
    return {k: v for k, v in lvl.items() if k != "events"} if lvl else None


def _slim_analysis(analysis: dict) -> dict:
    """Compact an analysis dict for an LLM prompt: drop event lists everywhere they appear
    (top-level levels, key_level, AND the level embedded in each stairstep step) and keep
    only the heaviest levels. Groq free tier allows 12k tokens/min — unslimmed payloads
    with matches blow past it (413)."""
    levels = [_slim_level(l) for l in analysis.get("levels", [])]
    if len(levels) > _MAX_LEVELS_FOR_LLM:
        levels = sorted(sorted(levels, key=lambda l: l["weight"], reverse=True)[:_MAX_LEVELS_FOR_LLM],
                        key=lambda l: l["center"])
    stair = analysis.get("stairstep", {})
    steps = [s | {"level": _slim_level(s.get("level"))} for s in stair.get("steps", [])]
    return analysis | {
        "levels": levels,
        "key_level": _slim_level(analysis.get("key_level")),
        "stairstep": stair | {"steps": steps} if stair else stair,
    }


def narrate_live(analysis: dict, matches: list | None = None,
                 news: list | None = None) -> str:
    """Single-session narrative for the live page. Blocking (sync LLM call) — API handlers
    must run this through run_in_threadpool."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from patternlab.agent.llm_factory import get_llm

    slim_matches = [m | {"analysis": _slim_analysis(m["analysis"])} if "analysis" in m else m
                    for m in (matches or [])[:3]]
    payload = json.dumps({"analysis": _slim_analysis(analysis), "matches": slim_matches,
                          "news": news or []}, default=str)
    msg = get_llm().invoke([
        SystemMessage(content=LIVE_SYSTEM_PROMPT),
        HumanMessage(content=f"Live session JSON:\n{payload}"),
    ])
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def chat_live(analysis: dict, matches: list | None, messages: list[dict],
              news: list | None = None) -> str:
    """Grounded follow-up chat about the live session. `messages` is the whole conversation
    so far as [{"role": "user"|"assistant", "content": ...}], ending with the new user
    question — stateless, the caller resends history each turn. Blocking (sync LLM call) —
    API handlers must run this through run_in_threadpool."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from patternlab.agent.llm_factory import get_llm

    slim_matches = [m | {"analysis": _slim_analysis(m["analysis"])} if "analysis" in m else m
                    for m in (matches or [])[:3]]
    payload = json.dumps({"analysis": _slim_analysis(analysis), "matches": slim_matches,
                          "news": news or []}, default=str)
    history = [AIMessage(content=m["content"]) if m["role"] == "assistant"
               else HumanMessage(content=m["content"]) for m in messages]
    msg = get_llm().invoke([
        SystemMessage(content=f"{CHAT_SYSTEM_PROMPT}\n\nLive session JSON:\n{payload}"),
        *history,
    ])
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def _node_analyze(state: AgentState) -> AgentState:
    return {
        "analysis_a": analyze_session(state["df_a"], params),
        "analysis_b": analyze_session(state["df_b"], params),
    }


def _node_compare(state: AgentState) -> AgentState:
    return {"comparison": compare_sessions(state["df_a"], state["df_b"], params)}


def _node_narrate(state: AgentState) -> AgentState:
    from langchain_core.messages import HumanMessage, SystemMessage
    from patternlab.agent.llm_factory import get_llm

    llm = get_llm()
    comparison = state["comparison"] | {
        "session_a": _slim_analysis(state["comparison"]["session_a"]),
        "session_b": _slim_analysis(state["comparison"]["session_b"]),
    }
    payload = json.dumps(comparison, default=str)
    try:
        msg = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Comparison JSON:\n{payload}"),
        ])
    except Exception as e:
        # A provider failure (quota, network, bad key) must not discard the deterministic
        # comparison — callers degrade by reading narrative_error instead.
        return {"narrative": "", "narrative_error": f"{type(e).__name__}: {e}"}
    return {"narrative": msg.content}


def build_graph() -> Any:
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(AgentState)
    g.add_node("analyze", _node_analyze)
    g.add_node("compare", _node_compare)
    g.add_node("narrate", _node_narrate)
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "compare")
    g.add_edge("compare", "narrate")
    g.add_edge("narrate", END)
    return g.compile()
