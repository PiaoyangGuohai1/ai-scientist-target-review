"""Scientist node — autonomous evidence-gathering planner.

Responsibilities:
- Decide which tool to call next, given the user's query and the evidence
  gathered so far. Tool selection is LLM-driven; there is no hard-coded
  order.
- Persist each tool result into ``state['tool_outputs']`` so downstream
  nodes (Writer / Reviewer) use structured ground truth directly rather
  than re-parsing message history.
- Terminate its own loop by emitting an AIMessage with no tool_calls.

Routing lives in ``graph.py`` via a conditional edge that inspects the
last message's ``tool_calls``.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..llm import get_chat_model
from ..schemas import AgentState
from ..tools import ALL_TOOLS, TOOL_BY_NAME

SCIENTIST_SYSTEM = """You are "AI Scientist", an autonomous research agent \
for AIDD (AI-driven drug discovery) target triage.

Given a user query about a target–disease hypothesis, decide which biology \
tools to call, in what order, to gather enough evidence to judge the \
target's feasibility and risk. You have three tools:

  1. query_gwas_data(gene_symbol)         — genetic / phenotype associations
  2. query_expression_atlas(gene_symbol)  — tissue expression (on-/off-target)
  3. search_pubmed_literature(query)      — prior-art literature

Rules:
- Do NOT invent biological facts. Only use information returned by the tools.
- Call each of the three tools at most once, unless a later call materially \
depends on a previous one.
- When you have enough evidence to brief a writer, respond WITHOUT any \
tool_call. Emit a short paragraph titled "Evidence brief:" summarising the \
facts the Writer should use. Do NOT draft the report yourself.
"""


def scientist_node(state: AgentState) -> dict[str, Any]:
    """Let the LLM pick the next tool (or terminate with a brief)."""
    llm = get_chat_model().bind_tools(ALL_TOOLS)

    messages = list(state.get("messages") or [])
    if not messages:
        # First call — seed the conversation and persist system + user
        # into state so subsequent turns (which see state['messages']) still
        # carry the user query. Previously only the AIMessage was returned,
        # causing the LLM's second invocation to receive only tool history
        # and trip Qwen-style chat templates that require a user turn.
        initial = [
            SystemMessage(content=SCIENTIST_SYSTEM),
            HumanMessage(content=state["user_query"]),
        ]
        ai_msg: AIMessage = llm.invoke(initial)
        return {"messages": initial + [ai_msg]}

    ai_msg = llm.invoke(messages)
    return {"messages": [ai_msg]}


def tool_executor_node(state: AgentState) -> dict[str, Any]:
    """Execute every tool_call in the last AIMessage, persisting outputs.

    We deliberately do not use ``langgraph.prebuilt.ToolNode`` because we
    want the side effect of recording results into ``tool_outputs`` so the
    Reviewer can do claim-level grounding without re-parsing messages.
    """
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage):
        return {}

    tool_outputs: dict[str, Any] = dict(state.get("tool_outputs") or {})
    tool_log: list[dict[str, Any]] = list(state.get("tool_calls_log") or [])
    new_messages: list[Any] = []

    for call in getattr(last, "tool_calls", None) or []:
        name = call["name"]
        args = call.get("args") or {}
        tool = TOOL_BY_NAME.get(name)
        if tool is None:
            result: Any = {"error": f"unknown tool: {name}"}
        else:
            try:
                result = tool.invoke(args)
            except Exception as exc:  # pragma: no cover - defensive
                result = {"error": f"{type(exc).__name__}: {exc}"}

        tool_outputs[name] = result
        tool_log.append({"tool": name, "args": args, "result": result})
        new_messages.append(
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=call.get("id", name),
                name=name,
            )
        )

    return {
        "messages": new_messages,
        "tool_outputs": tool_outputs,
        "tool_calls_log": tool_log,
    }


def scientist_should_continue(state: AgentState) -> str:
    """Conditional-edge router for the Scientist tool-calling inner loop."""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "writer"
