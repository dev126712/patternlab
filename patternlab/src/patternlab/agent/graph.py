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
5. One-sentence structural implication, hedged appropriately.

STRICT RULES: every number you mention MUST appear in the JSON. Never invent levels,
slopes, or counts. If a field is missing, omit that point. No trading advice beyond the
structural implication."""


def _strip_level_events(analysis: dict) -> dict:
    """Per-touch event lists are for the UI's level drill-down — pure token bloat for the LLM."""
    slim = lambda lvl: {k: v for k, v in lvl.items() if k != "events"}  # noqa: E731
    kl = analysis.get("key_level")
    return analysis | {"levels": [slim(l) for l in analysis.get("levels", [])],
                       "key_level": slim(kl) if kl else None}


def narrate_live(analysis: dict, matches: list | None = None) -> str:
    """Single-session narrative for the live page. Blocking (sync LLM call) — API handlers
    must run this through run_in_threadpool."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from patternlab.agent.llm_factory import get_llm

    slim_matches = [m | {"analysis": _strip_level_events(m["analysis"])} if "analysis" in m else m
                    for m in (matches or [])[:3]]
    payload = json.dumps({"analysis": _strip_level_events(analysis), "matches": slim_matches},
                         default=str)
    msg = get_llm().invoke([
        SystemMessage(content=LIVE_SYSTEM_PROMPT),
        HumanMessage(content=f"Live session JSON:\n{payload}"),
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
        "session_a": _strip_level_events(state["comparison"]["session_a"]),
        "session_b": _strip_level_events(state["comparison"]["session_b"]),
    }
    payload = json.dumps(comparison, default=str)
    msg = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Comparison JSON:\n{payload}"),
    ])
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
