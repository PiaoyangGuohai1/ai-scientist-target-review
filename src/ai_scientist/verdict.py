"""Verdict logic — three deterministic termination locks, pure code.

This module is intentionally LLM-free. It exists so that every control-flow
decision is testable and reviewable in isolation, and so that reviewers can
audit the guardrail without running a model.
"""

from __future__ import annotations

import os
from typing import Any

from .schemas import AgentState, Verdict


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def decide_verdict(state: AgentState) -> Verdict:
    """Compute the Reviewer-stage verdict.

    Precedence (important — do NOT reorder casually):
        1. PASS          — no open issues.
        2. MAX_ROUNDS    — revision budget exhausted.
        3. STAGNATED     — open-issue count plateau (configurable threshold).
        4. REJECT        — default when issues remain and budget is left.
    """
    issues = state.get("issues") or []
    if not issues:
        return Verdict.PASS

    revision_count = state.get("revision_count") or 0
    max_revisions = state.get("max_revisions") or _int_env("MAX_REVISIONS", 2)
    if revision_count >= max_revisions:
        return Verdict.MAX_ROUNDS

    stagnation_count = state.get("stagnation_count") or 0
    stagnation_threshold = (
        state.get("stagnation_threshold") or _int_env("STAGNATION_THRESHOLD", 2)
    )
    if stagnation_count >= stagnation_threshold:
        return Verdict.STAGNATED

    return Verdict.REJECT


def verdict_node(state: AgentState) -> dict[str, Any]:
    """Graph node that simply writes the verdict into state."""
    v = decide_verdict(state)
    reason = ""
    if v == Verdict.PASS:
        reason = "No open issues after Reviewer grounding."
    elif v == Verdict.MAX_ROUNDS:
        reason = f"Revision budget exhausted (revision_count={state.get('revision_count')})."
    elif v == Verdict.STAGNATED:
        reason = (
            f"Stagnation ≥ threshold (count={state.get('stagnation_count')}, "
            f"open_issues={len(state.get('issues') or [])})."
        )
    else:
        reason = f"{len(state.get('issues') or [])} open issue(s) — Writer must revise."

    return {"verdict": v.value, "reject_reason": reason}


def route_from_verdict(state: AgentState) -> str:
    """Conditional-edge selector used in graph.py."""
    v = state.get("verdict")
    if v == Verdict.REJECT.value:
        return "writer"
    return "finalize"
