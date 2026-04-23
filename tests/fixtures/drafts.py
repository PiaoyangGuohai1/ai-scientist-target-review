"""Deterministic draft fixtures for E2E reject→revise→PASS regression.

These strings used to live inside ``src/ai_scientist/agents/writer.py`` as
``_OFFLINE_BAD_DRAFT`` / ``_OFFLINE_GOOD_DRAFT``. Relocated here so the
production writer always goes through a real LLM, while tests that need a
deterministic adversarial input can inject them via ``monkeypatch``.

The bad draft is hand-crafted to contain exactly three Reviewer-catchable
issues against the bundled TP53 mock data:

  1. GWAS CV significance — mock says P=0.42 (non-significant), draft
     claims P=3.2e-18 (strong signal).
  2. Cardiac expression level — mock says level=low (TPM≈2.1), draft
     says "highly expressed in cardiac tissue (TPM≈92.0)".
  3. PubMed literature grounding — no abstract mentions "late-stage
     cardiovascular trials" of TP53.

It also contains one ``[from inference]`` sentence with an unsupported
qualitative claim ("TP53 is a promising cardiovascular target"), which
must now also fail the guardrail (this is the regression test for the
``[from inference]`` bypass fix).
"""

BAD_DRAFT = """## Genetic evidence
TP53 shows a strong GWAS signal for cardiovascular disease with \
P=3.2e-18 [from query_gwas_data]. Variants in TP53 have been linked to \
coronary artery disease risk [from query_gwas_data].

## Expression & on-/off-target rationale
TP53 is highly expressed in cardiac tissue (TPM ≈ 92.0) [from \
query_expression_atlas], supporting a strong cardiac on-target \
rationale. It is also elevated in tumor microenvironments [from \
query_expression_atlas].

## Prior-art & druggability
Multiple small-molecule activators of wild-type TP53 are in late-stage \
cardiovascular trials [from search_pubmed_literature].

## Overall verdict (risk / feasibility)
TP53 is a promising cardiovascular target with strong cardiac \
expression and robust CAD genetic support [from inference].
"""


GOOD_DRAFT = """## Genetic evidence
TP53 shows a strong cancer-predisposition signal (e.g. Li-Fraumeni \
syndrome P=3.1e-42, OR=8.4) but **no** significant cardiovascular \
association (Coronary artery disease P=0.42) [from query_gwas_data].

## Expression & on-/off-target rationale
TP53 is **low** in heart tissue (TPM ≈ 2.1, level=low) and **high** in \
tumor microenvironment (TPM ≈ 86.3, level=high) [from \
query_expression_atlas]. This pattern indicates weak cardiac on-target \
rationale but is consistent with an oncology positioning [from inference].

## Prior-art & druggability
Direct pharmacological restoration of wild-type TP53 remains highly \
challenging; most candidates (e.g. APR-246) target mutant conformations \
[from search_pubmed_literature]. TP53 is generally regarded as a poor \
cardiovascular target with limited tissue rationale [from \
search_pubmed_literature].

## Overall verdict (risk / feasibility)
Given the non-significant CAD association, low cardiac expression, and \
absent cardiovascular-trial literature, repositioning toward oncology \
is a more productive direction [from inference].
"""
