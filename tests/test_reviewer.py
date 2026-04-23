"""Reviewer guardrail regression tests.

Two responsibilities:

1. Pin the E2E reject→revise→PASS loop — using ``patched_writer`` and
   ``patched_scientist`` fixtures so the graph exercises real Reviewer /
   Verdict / tool-executor code but deterministic LLM-free Writer and
   Scientist stubs.
2. Pin the ``[from inference]`` bypass fix — a writer cannot smuggle a
   contradictory number or qualitative attribute by tagging it
   ``inference``.

No test in this module contacts a real LLM.
"""

from __future__ import annotations


def _run(initial_query: str = "评估一下 TP53 作为一个心血管疾病靶点的潜在风险和可行性"):
    from ai_scientist.graph import build_graph, initial_state

    graph = build_graph()
    state = initial_state(initial_query)
    final: dict = {}
    for step in graph.stream(state, stream_mode="updates"):
        for _node, update in step.items():
            final.update(update)
    return final


def test_reviewer_catches_bad_draft_then_converges(patched_scientist, patched_writer):
    """E2E: bad draft → Reviewer rejects → Writer flips to good → PASS."""
    final = _run()

    assert final.get("final_report"), "graph must reach the finalize node"
    assert (final.get("revision_count") or 0) >= 1, (
        "writer must have been forced to revise at least once"
    )
    assert final.get("verdict") == "PASS", (
        f"expected PASS after revision, got {final.get('verdict')!r}"
    )
    # Good draft says heart expression is low, not high.
    fr = final.get("final_report") or ""
    assert "highly expressed in cardiac" not in fr
    assert "low" in fr.lower() or "低" in fr


def test_final_report_has_verdict_header(patched_scientist, patched_writer):
    final = _run()
    assert "verdict:" in (final.get("final_report") or "").lower()


# ---------------------------------------------------------------------------
# Direct reviewer_node tests — no graph, no LLM, pure guardrail logic
# ---------------------------------------------------------------------------
def test_untagged_substantive_sentence_is_flagged_unsupported():
    from ai_scientist.agents.reviewer import reviewer_node

    cases = [
        (
            "## Expression & on-/off-target rationale\n"
            "TP53 is highly druggable in heart tissue without support.\n"
            "TP53 is high in heart.\n",
            {
                "query_expression_atlas": {
                    "gene": "TP53",
                    "expression": [{"tissue": "heart", "tpm": 2.1, "level": "low"}],
                }
            },
        ),
        ("TP53 high in heart.\n", {}),
    ]

    for draft, tool_outputs in cases:
        out = reviewer_node({"draft_report": draft, "tool_outputs": tool_outputs})
        assert out["issues"], "untagged factual sentences must be flagged"
        assert any(i["type"] == "UNSUPPORTED" for i in out["issues"])


def test_inference_tag_cannot_bypass_contradictory_expression():
    """Regression for the [from inference] bypass loophole.

    If a writer writes 'TP53 is highly expressed in heart [from inference]'
    while the mock says heart level=low, the Reviewer MUST flag a
    CONTRADICTION. Before the fix this returned 0 issues.
    """
    from ai_scientist.agents.reviewer import reviewer_node

    draft = "TP53 is highly expressed in heart and strongly associated with CAD [from inference].\n"
    state = {
        "draft_report": draft,
        "tool_outputs": {
            "query_expression_atlas": {
                "gene": "TP53",
                "expression": [{"tissue": "heart", "tpm": 2.1, "level": "low"}],
            },
            "query_gwas_data": {
                "gene": "TP53",
                "associations": [
                    {"phenotype": "Coronary artery disease", "p_value": 0.42}
                ],
            },
        },
    }
    out = reviewer_node(state)
    types = {i["type"] for i in out["issues"]}
    assert "CONTRADICTION" in types, (
        f"inference-tagged contradictory claim must surface CONTRADICTION; "
        f"got types={types}"
    )


def test_benign_inference_passes():
    """Counterexample: a genuinely derivative conclusion with no
    groundable topic hits must NOT be flagged."""
    from ai_scientist.agents.reviewer import reviewer_node

    draft = "Overall this target merits cautious further evaluation [from inference].\n"
    out = reviewer_node({"draft_report": draft, "tool_outputs": {}})
    assert out["issues"] == [], (
        f"benign inference should pass; got issues={out['issues']}"
    )
