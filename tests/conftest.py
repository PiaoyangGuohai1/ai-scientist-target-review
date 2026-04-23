"""Shared pytest fixtures.

Two jobs:

1. Make the ``ai_scientist`` package importable when pytest is run from
   the repo root without ``pip install -e``. This is a developer-
   ergonomics fix; in CI / production we still install the package.

2. Provide deterministic ``patched_writer`` / ``patched_scientist``
   fixtures so the E2E regression tests can exercise the graph's
   reject→revise→PASS cycle without hitting a real LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# --- make the `ai_scientist` package importable from repo root --------------
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture
def patched_writer(monkeypatch):
    """Replace ``writer_node`` so revision 0 returns BAD_DRAFT and
    every subsequent call returns GOOD_DRAFT. Reviewer / Verdict /
    Scientist / tools all still run their real code paths."""
    from tests.fixtures.drafts import BAD_DRAFT, GOOD_DRAFT

    bad, good = BAD_DRAFT, GOOD_DRAFT

    def _fake_writer_node(state):
        rev = state.get("revision_count") or 0
        issues = state.get("issues") or []
        prev = state.get("draft_report") or ""
        draft = bad if not issues else good
        return {
            "draft_report": draft,
            "revision_count": rev + (0 if not prev else 1),
            "issues": [],
        }

    import ai_scientist.graph as graph_mod

    monkeypatch.setattr(graph_mod, "writer_node", _fake_writer_node)
    return _fake_writer_node


@pytest.fixture
def patched_scientist(monkeypatch):
    """Replace ``scientist_node`` to call all three tools in a fixed
    order (GWAS → Expression → PubMed) without invoking any LLM. Used
    by the E2E regression tests to keep the focus on the Reviewer
    guardrail rather than on LLM reasoning."""
    from langchain_core.messages import AIMessage

    def _fake_scientist_node(state):
        messages = state.get("messages") or []
        called = set()
        for m in messages:
            if isinstance(m, AIMessage):
                for tc in getattr(m, "tool_calls", None) or []:
                    called.add(tc["name"])

        plan = ["query_gwas_data", "query_expression_atlas", "search_pubmed_literature"]
        for tool in plan:
            if tool not in called:
                args = {"query": "TP53 cardiovascular druggability"} if tool == "search_pubmed_literature" else {"gene_symbol": "TP53"}
                return {
                    "messages": [
                        AIMessage(
                            content=f"[test-fixture] next tool = {tool}",
                            tool_calls=[{"name": tool, "args": args, "id": f"fx-{tool}"}],
                        )
                    ]
                }
        # All tools called — emit a brief.
        return {"messages": [AIMessage(content="Evidence brief: gathered GWAS + expression + literature.")]}

    import ai_scientist.graph as graph_mod

    monkeypatch.setattr(graph_mod, "scientist_node", _fake_scientist_node)
    return _fake_scientist_node


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Force tests to never hit a real LLM.

    The Reviewer's claim extractor is LLM-first with a rule-based
    fallback; by unsetting the API key we guarantee the fallback kicks
    in, which is the deterministic code path we want to regression-test.
    Tests that specifically want to exercise the LLM extractor should
    override this with their own ``monkeypatch.setenv``.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MAX_REVISIONS", "2")
    monkeypatch.setenv("STAGNATION_THRESHOLD", "3")
    yield
