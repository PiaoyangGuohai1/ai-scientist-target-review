"""Writer node — drafts (and revises) the target-triage report.

Contract:
- First call  → draft from the Scientist's evidence brief + raw tool outputs.
- Later calls → revise, consuming ``state['issues']`` as a change-list, and
                marking each fixed issue id in the prose.

The Writer is instructed to label every factual claim with a source tag
like ``[from query_expression_atlas]``. This lets the Reviewer resolve
each claim back to a single tool return during grounding.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_chat_model
from ..schemas import AgentState

WRITER_SYSTEM = """You are the "Target Triage Writer". You produce a \
concise feasibility & risk briefing on a drug target in Markdown.

Hard constraints:
- Every factual sentence MUST carry a source tag like ``[from <tool_name>]``. \
Allowed tool names: query_gwas_data, query_expression_atlas, \
search_pubmed_literature, inference (only for conclusions derived from \
already-tagged facts; inference sentences MUST NOT introduce new numbers \
or qualitative attributes that were not stated in an upstream tool-tagged \
sentence).
- Do NOT fabricate numbers. If a number was not returned by a tool, do not \
cite it.
- Structure:
  1. ## Genetic evidence
  2. ## Expression & on-/off-target rationale
  3. ## Prior-art & druggability
  4. ## Overall verdict (risk / feasibility)
- Keep it under ~300 words.

If you receive REVISION FEEDBACK with a list of issues, address each \
issue id explicitly and keep all other unchanged content stable.
"""
def writer_node(state: AgentState) -> dict[str, Any]:
    revision_count = state.get("revision_count") or 0
    prev_draft = state.get("draft_report") or ""
    issues = state.get("issues") or []

    llm = get_chat_model()
    tool_outputs = state.get("tool_outputs") or {}

    user_parts: list[str] = [f"User query: {state['user_query']}", ""]
    user_parts.append("## Tool outputs (ground truth — do not contradict)")
    user_parts.append("```json")
    user_parts.append(json.dumps(tool_outputs, ensure_ascii=False, indent=2, default=str))
    user_parts.append("```")

    if issues:
        user_parts.append("")
        user_parts.append("## REVISION FEEDBACK — address every issue below")
        for i in issues:
            user_parts.append(
                f"- [{i.get('id')}] ({i.get('type')}/{i.get('severity')}) "
                f"claim: {i.get('claim_text')}\n  evidence: {i.get('evidence_note')}\n"
                f"  suggestion: {i.get('suggestion')}"
            )

    msg = [SystemMessage(content=WRITER_SYSTEM), HumanMessage(content="\n".join(user_parts))]
    resp = llm.invoke(msg)
    draft = (resp.content if hasattr(resp, "content") else str(resp)).strip()

    return {
        "draft_report": draft,
        "revision_count": revision_count + (0 if not prev_draft else 1),
        # Clear issues — they get re-populated by the Reviewer next.
        "issues": [],
    }
