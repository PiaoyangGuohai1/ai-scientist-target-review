"""Tools are mocks, but their contracts still matter."""

from __future__ import annotations

from ai_scientist.tools import (
    query_gwas_data,
    query_expression_atlas,
    search_pubmed_literature,
    TOOL_BY_NAME,
    ALL_TOOLS,
)


def test_gwas_tp53_no_significant_cv_association():
    result = query_gwas_data("TP53")
    assert result["gene"] == "TP53"
    cv = [
        a
        for a in result["associations"]
        if "coronary" in a["phenotype"].lower() or "cardiovascular" in a["phenotype"].lower()
    ]
    assert cv, "mock data must include a CV phenotype for the demo contrast"
    # Intentional signal: CV association is NOT significant.
    assert all(a["p_value"] > 5e-8 for a in cv)


def test_expression_tp53_low_in_heart_high_in_tumor():
    result = query_expression_atlas("TP53")
    levels = {row["tissue"]: row["level"] for row in result["expression"]}
    assert levels.get("heart") == "low", "TP53 heart must be LOW (demo contrast)"
    assert levels.get("tumor_microenvironment") == "high"


def test_pubmed_returns_hits():
    r = search_pubmed_literature("TP53 cardiovascular druggability")
    assert r["hits"], "expected at least one abstract"
    for h in r["hits"]:
        assert "pmid" in h and "title" in h and "abstract" in h


def test_every_tool_exposes_structured_schema():
    # LLM function calling needs the args_schema present.
    for tool in ALL_TOOLS:
        assert tool.args_schema is not None
        assert tool.name in TOOL_BY_NAME
