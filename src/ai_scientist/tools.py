"""Three mock biology tools.

Mock data intentionally embeds a contrast signal for TP53 so that the
forced-failure demo can produce a Reviewer-catchable hallucination:

    TP53 is LOW in heart, HIGH in tumor microenvironment.

The tools are exposed as LangChain StructuredTools so the Scientist node
can bind them to an LLM via `bind_tools(...)`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Mock knowledge base
# ---------------------------------------------------------------------------
_GWAS_DB: dict[str, dict[str, Any]] = {
    "TP53": {
        "gene": "TP53",
        "associations": [
            {"phenotype": "Li-Fraumeni syndrome", "p_value": 3.1e-42, "odds_ratio": 8.4},
            {"phenotype": "Breast cancer", "p_value": 2.7e-18, "odds_ratio": 1.6},
            {"phenotype": "Lung adenocarcinoma", "p_value": 9.4e-12, "odds_ratio": 1.9},
            # No significant cardiovascular association — intentional.
            {"phenotype": "Coronary artery disease", "p_value": 0.42, "odds_ratio": 1.02},
        ],
        "summary": (
            "TP53 shows strong cancer-predisposition signal but no significant "
            "cardiovascular association (P=0.42 for CAD)."
        ),
    },
    "PCSK9": {
        "gene": "PCSK9",
        "associations": [
            {"phenotype": "LDL cholesterol", "p_value": 5.2e-210, "odds_ratio": 2.8},
            {"phenotype": "Coronary artery disease", "p_value": 1.1e-45, "odds_ratio": 1.9},
        ],
        "summary": "PCSK9 is a validated cardiovascular target (CAD P=1.1e-45).",
    },
}

_EXPRESSION_DB: dict[str, dict[str, Any]] = {
    "TP53": {
        "gene": "TP53",
        "expression": [
            {"tissue": "heart", "tpm": 2.1, "level": "low"},
            {"tissue": "liver", "tpm": 11.7, "level": "medium"},
            {"tissue": "tumor_microenvironment", "tpm": 86.3, "level": "high"},
            {"tissue": "immune_cells", "tpm": 24.8, "level": "medium"},
        ],
    },
    "PCSK9": {
        "gene": "PCSK9",
        "expression": [
            {"tissue": "liver", "tpm": 52.4, "level": "high"},
            {"tissue": "heart", "tpm": 3.2, "level": "low"},
        ],
    },
}

_PUBMED_DB: list[dict[str, Any]] = [
    {
        "pmid": "30123456",
        "title": "TP53 as a central tumor-suppressor: revisiting druggability.",
        "year": 2023,
        "abstract": (
            "Despite decades of effort, direct pharmacological restoration of "
            "wild-type TP53 function remains extremely challenging. Most "
            "candidate molecules target mutant conformations via reactivators "
            "(e.g. APR-246). TP53 is generally regarded as a poor cardiovascular "
            "target with limited tissue rationale."
        ),
    },
    {
        "pmid": "30876543",
        "title": "Expression atlas of tumor suppressors across adult tissues.",
        "year": 2024,
        "abstract": (
            "Comprehensive profiling confirms that TP53 exhibits low basal "
            "expression in cardiac tissue and is enriched in tumor "
            "microenvironments, consistent with its role in cellular stress "
            "response."
        ),
    },
    {
        "pmid": "31002233",
        "title": "PCSK9 inhibitors: a decade of cardiovascular benefit.",
        "year": 2024,
        "abstract": (
            "PCSK9 monoclonal antibodies and siRNA therapies provide durable "
            "LDL-C reduction and major adverse cardiovascular event "
            "attenuation."
        ),
    },
]


# ---------------------------------------------------------------------------
# Tool input schemas (used by LLM to pick & call correctly)
# ---------------------------------------------------------------------------
class _GeneInput(BaseModel):
    gene_symbol: str = Field(
        description="HGNC official gene symbol, e.g. 'TP53'. Case-insensitive.",
    )


class _LiteratureInput(BaseModel):
    query: str = Field(
        description=(
            "Free-text query, e.g. 'TP53 cardiovascular druggability'. Include the "
            "gene symbol and the disease/mechanism keywords."
        ),
    )


# ---------------------------------------------------------------------------
# Tool implementations (pure, deterministic)
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def query_gwas_data(gene_symbol: str) -> dict[str, Any]:
    """Mock GWAS lookup — returns phenotype associations with P-values / ORs."""
    key = (gene_symbol or "").strip().upper()
    record = _GWAS_DB.get(
        key,
        {
            "gene": key,
            "associations": [],
            "summary": f"No GWAS record available for {key}.",
        },
    )
    return {
        "data_source": "mock_gwas_catalog_v0",
        "fetched_at": _now(),
        **record,
    }


def query_expression_atlas(gene_symbol: str) -> dict[str, Any]:
    """Mock single-cell / bulk expression atlas lookup."""
    key = (gene_symbol or "").strip().upper()
    record = _EXPRESSION_DB.get(
        key,
        {"gene": key, "expression": []},
    )
    return {
        "data_source": "mock_gtex_hpa_v0",
        "fetched_at": _now(),
        **record,
    }


def search_pubmed_literature(query: str) -> dict[str, Any]:
    """Mock PubMed RAG retrieval — returns the most relevant abstracts."""
    q = (query or "").lower()
    tokens = [t for t in q.replace(",", " ").split() if len(t) > 1]

    def _score(doc: dict[str, Any]) -> int:
        hay = (doc["title"] + " " + doc["abstract"]).lower()
        return sum(1 for t in tokens if t in hay)

    scored = sorted(_PUBMED_DB, key=_score, reverse=True)
    top = [d for d in scored if _score(d) > 0][:3] or _PUBMED_DB[:2]
    return {
        "data_source": "mock_pubmed_v0",
        "fetched_at": _now(),
        "query": query,
        "hits": top,
    }


# ---------------------------------------------------------------------------
# LangChain StructuredTool wrappers — this is what the LLM actually sees.
# ---------------------------------------------------------------------------
TOOL_GWAS = StructuredTool.from_function(
    func=query_gwas_data,
    name="query_gwas_data",
    description=(
        "Query a GWAS catalogue for disease/phenotype associations of a single "
        "gene. Returns a list of (phenotype, p_value, odds_ratio) entries plus "
        "a one-line summary. Call this FIRST when genetic evidence is needed."
    ),
    args_schema=_GeneInput,
)

TOOL_EXPR = StructuredTool.from_function(
    func=query_expression_atlas,
    name="query_expression_atlas",
    description=(
        "Query a tissue / single-cell expression atlas for a single gene. "
        "Returns per-tissue TPM values and a categorical level "
        "(low / medium / high). Use this to reason about on-target tissue "
        "rationale and off-target (e.g. cardiac) safety risk."
    ),
    args_schema=_GeneInput,
)

TOOL_PUBMED = StructuredTool.from_function(
    func=search_pubmed_literature,
    name="search_pubmed_literature",
    description=(
        "Retrieve up to 3 relevant PubMed abstracts for a free-text query. "
        "Use this to gather prior-art evidence on druggability, clinical "
        "precedent, and mechanism."
    ),
    args_schema=_LiteratureInput,
)

ALL_TOOLS: list[StructuredTool] = [TOOL_GWAS, TOOL_EXPR, TOOL_PUBMED]

TOOL_BY_NAME: dict[str, StructuredTool] = {t.name: t for t in ALL_TOOLS}
