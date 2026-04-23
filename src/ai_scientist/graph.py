"""LangGraph StateGraph assembly.

Topology (3 functional nodes + 1 executor + 1 verdict + 1 finalize):

    START ─► scientist ─┬─► tools ─► scientist   (ReAct inner loop)
                        │
                        └─► writer ─► reviewer ─► verdict ─┬─► writer  (REJECT)
                                                           │
                                                           └─► finalize ─► END
                                                                (PASS / MAX_ROUNDS / STAGNATED)

All node-to-node routing lives here. No node decides its own successor.
"""

from __future__ import annotations

import os
from typing import Any

from langgraph.graph import END, START, StateGraph

from .agents.reviewer import reviewer_node
from .agents.scientist import (
    scientist_node,
    scientist_should_continue,
    tool_executor_node,
)
from .agents.writer import writer_node
from .schemas import AgentState, Verdict
from .verdict import route_from_verdict, verdict_node


def _finalize_node(state: AgentState) -> dict[str, Any]:
    """Attach a final header to the draft depending on verdict."""
    draft = state.get("draft_report") or ""
    v = state.get("verdict") or Verdict.PASS.value
    reason = state.get("reject_reason") or ""
    header_map = {
        Verdict.PASS.value: "# 靶点初评报告  (verdict: PASS)",
        Verdict.MAX_ROUNDS.value: "# 靶点初评报告  (verdict: MAX_ROUNDS · 已达最大修订轮数)",
        Verdict.STAGNATED.value: "# 靶点初评报告  (verdict: STAGNATED · 连续停滞)",
    }
    header = header_map.get(v, f"# 靶点初评报告  (verdict: {v})")
    footer = f"\n\n---\n*Verdict reason:* {reason}\n"
    return {"final_report": f"{header}\n\n{draft}{footer}"}


def build_graph():
    g: StateGraph = StateGraph(AgentState)
    g.add_node("scientist", scientist_node)
    g.add_node("tools", tool_executor_node)
    g.add_node("writer", writer_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("verdict", verdict_node)
    g.add_node("finalize", _finalize_node)

    g.add_edge(START, "scientist")
    g.add_conditional_edges(
        "scientist",
        scientist_should_continue,
        {"tools": "tools", "writer": "writer"},
    )
    g.add_edge("tools", "scientist")
    g.add_edge("writer", "reviewer")
    g.add_edge("reviewer", "verdict")
    g.add_conditional_edges(
        "verdict",
        route_from_verdict,
        {"writer": "writer", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)

    return g.compile()


def initial_state(user_query: str) -> dict[str, Any]:
    return {
        "user_query": user_query,
        "messages": [],
        "tool_outputs": {},
        "tool_calls_log": [],
        "draft_report": "",
        "claims": [],
        "issues": [],
        "prev_open_issue_count": None,
        "stagnation_count": 0,
        "revision_count": 0,
        "verdict": "",
        "reject_reason": "",
        "final_report": "",
        "max_revisions": int(os.getenv("MAX_REVISIONS") or 2),
        "stagnation_threshold": int(os.getenv("STAGNATION_THRESHOLD") or 2),
    }
