"""End-to-end demo: build two synthetic sessions mirroring the reference XAUUSD analysis
(grind-then-break vs V-recovery into the same level), run the full detection stack, and
print the structured comparison. Run:  uv run python scripts/demo.py [--narrate]

--narrate additionally invokes the LangGraph agent (needs GROQ_API_KEY or Ollama running).
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from patternlab.config import params
from patternlab.data.loader import synthetic_session
from patternlab.matching.similarity import compare_sessions


def brief(a: dict, name: str) -> None:
    ctx, ap, tal, st = a["context"], a["approach"], a["time_at_key_level"], a["stairstep"]
    print(f"\n── {name} ──")
    print(f"  key level        : {a['key_level']}")
    print(f"  level ladder     : {[l['center'] for l in a['levels']]}")
    print(f"  regimes          : {[(s['kind'], s['bars']) for s in a['regimes']]}")
    print(f"  stair-step       : broken={st['steps_broken']} held={st['steps_held_as_support']} is_stairstep={st['is_stairstep']}")
    print(f"  approach         : {ap['classification']} ({ap['slope_atr_per_bar']} ATR/bar, r2={ap['r2']})")
    print(f"  time at level    : bars={tal['bars_at_level']} rejections={tal['rejection_events']} first_tap={tal['first_tap']}")
    print(f"  context          : {ctx['kind']} (origin dd={ctx['origin_drawdown_frac']}, trapped supply={ctx['trapped_supply_overhead']})")


def main() -> None:
    df_a = synthetic_session("grind_break", seed=7)   # earlier session: grind under level, then break
    df_b = synthetic_session("v_recovery", seed=11)   # current session: steep recovery leg, first tap

    result = compare_sessions(df_a, df_b, params)
    brief(result["session_a"], "Session A (grind → break)")
    brief(result["session_b"], "Session B (V-recovery, first tap)")

    print("\n── Similarity ──")
    print(f"  combined score       : {result['similarity_score']}")
    print(f"  fingerprint cosine   : {result['fingerprint_cosine']}")
    print(f"  DTW shape similarity : {result['dtw_similarity']}")

    if "--narrate" in sys.argv:
        from patternlab.agent.graph import build_graph
        out = build_graph().invoke({"df_a": df_a, "df_b": df_b})
        print("\n── LLM narrative ──\n")
        print(out["narrative"])
    else:
        print("\n(add --narrate to generate the LLM comparison narrative)")

    with open("demo_comparison.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\nFull JSON written to demo_comparison.json")


if __name__ == "__main__":
    main()
