"""Verdict logic must be code-owned and deterministic.

These tests pin down the precedence order (PASS ▸ MAX_ROUNDS ▸ STAGNATED ▸
REJECT) so nobody later 'improves' it into an LLM call.
"""

from __future__ import annotations

from ai_scientist.schemas import Verdict
from ai_scientist.verdict import decide_verdict


def _state(**overrides):
    base = {
        "issues": [],
        "revision_count": 0,
        "stagnation_count": 0,
        "max_revisions": 2,
        "stagnation_threshold": 2,
    }
    base.update(overrides)
    return base


def test_pass_when_no_issues():
    assert decide_verdict(_state()) == Verdict.PASS


def test_reject_when_issues_and_budget_left():
    s = _state(issues=[{"id": "R-001", "type": "CONTRADICTION"}], revision_count=0)
    assert decide_verdict(s) == Verdict.REJECT


def test_max_rounds_when_budget_exhausted():
    s = _state(
        issues=[{"id": "R-001", "type": "CONTRADICTION"}],
        revision_count=2,
        max_revisions=2,
    )
    assert decide_verdict(s) == Verdict.MAX_ROUNDS


def test_stagnated_when_plateau_reached():
    s = _state(
        issues=[{"id": "R-001"}],
        revision_count=1,
        stagnation_count=2,
        stagnation_threshold=2,
    )
    assert decide_verdict(s) == Verdict.STAGNATED


def test_pass_precedes_max_rounds():
    # Even if we're at the budget ceiling, zero issues => PASS.
    s = _state(revision_count=99, max_revisions=2)
    assert decide_verdict(s) == Verdict.PASS


def test_max_rounds_precedes_stagnated():
    # Budget exhausted => MAX_ROUNDS, even if stagnation also tripped.
    s = _state(
        issues=[{"id": "R-001"}],
        revision_count=5,
        max_revisions=2,
        stagnation_count=5,
        stagnation_threshold=2,
    )
    assert decide_verdict(s) == Verdict.MAX_ROUNDS
