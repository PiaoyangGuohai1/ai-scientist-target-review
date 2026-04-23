"""Pydantic data models & LangGraph state for the target-review workflow.

Design choices:
- Evidence is REQUIRED on every claim. If a claim cannot produce a valid
  Evidence pointer, it is marked UNSUPPORTED (fail-closed default).
- Verdict is an enum judged by pure code (see verdict.py), never by an LLM.
- State uses TypedDict so LangGraph can merge partial updates naturally.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Tool identifiers (single source of truth)
# ---------------------------------------------------------------------------
ToolName = Literal[
    "query_gwas_data",
    "query_expression_atlas",
    "search_pubmed_literature",
]

ALL_TOOL_NAMES: tuple[str, ...] = (
    "query_gwas_data",
    "query_expression_atlas",
    "search_pubmed_literature",
)


# ---------------------------------------------------------------------------
# Reviewer data classes
# ---------------------------------------------------------------------------
class Evidence(BaseModel):
    """Pointer from a claim to a specific field inside tool_outputs."""

    tool_name: Literal[
        "query_gwas_data",
        "query_expression_atlas",
        "search_pubmed_literature",
        "inference",
        "none",
    ] = Field(description="Which tool the claim is sourced from. 'none' ⇒ UNSUPPORTED.")
    field_path: str = Field(
        default="",
        description="JSON-path-style pointer, e.g. 'expression[0].level'. Empty for inference.",
    )
    snippet_or_value: str = Field(
        default="",
        description="Actual field value or literature snippet cited.",
    )
    comparison_result: Literal["支持", "矛盾", "无关", "无法定位"] = Field(
        default="无法定位",
        description="Relationship between claim and evidence after lookup.",
    )


class Claim(BaseModel):
    """An atomic factual proposition extracted from the draft report."""

    text: str = Field(description="Exact span from the draft (do not paraphrase).")
    normalized_proposition: str = Field(
        description="A minimal declarative rewording, used only for matching.",
    )
    source_tool: Literal[
        "query_gwas_data",
        "query_expression_atlas",
        "search_pubmed_literature",
        "inference",
    ]
    evidence: Evidence


class IssueType(str, Enum):
    FABRICATION = "FABRICATION"
    CONTRADICTION = "CONTRADICTION"
    UNSUPPORTED = "UNSUPPORTED"


class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class Issue(BaseModel):
    """Grounding problem discovered by the Reviewer."""

    id: str
    type: IssueType
    severity: IssueSeverity = IssueSeverity.MAJOR
    claim_text: str
    evidence_note: str
    suggestion: str


class Verdict(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"
    MAX_ROUNDS = "MAX_ROUNDS"
    STAGNATED = "STAGNATED"


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    """The shared state across every node in the workflow graph."""

    # Inputs
    user_query: str

    # Scientist node scratchpad (LLM message history + tool evidence)
    messages: Annotated[list, add_messages]
    tool_outputs: dict[str, Any]
    tool_calls_log: list[dict[str, Any]]

    # Writer & Reviewer artefacts
    draft_report: str
    claims: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    prev_open_issue_count: int
    stagnation_count: int

    # Control flow
    revision_count: int
    verdict: str  # Verdict enum value
    reject_reason: str
    final_report: str

    # Config echoes (for log readability)
    max_revisions: int
    stagnation_threshold: int
