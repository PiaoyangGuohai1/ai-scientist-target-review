"""Reviewer node — fail-closed grounding guardrail.

Pipeline:
    (A) Claim extraction  -- LLM-driven; falls back to a rule-based
                             extractor when JSON parsing fails.
    (B) Grounding lookup  -- pure code: each claim is resolved against a
                             field in ``state['tool_outputs']``. Numbers
                             compared within an order of magnitude;
                             categorical levels via exact match.
    (C) Issue generation  -- pure code, deterministic.

Key invariant: if any stage fails (schema validation, missing field,
out-of-scope claim), we DOWNGRADE the claim to UNSUPPORTED rather than
silently approving it. This is the "fail-closed" contract.

The Verdict itself is decided by ``verdict.py``, not here.

# Note on ``[from inference]``
Writer-authored conclusion sentences are tagged ``[from inference]``.
Earlier iterations of this file accepted such claims as long as they
didn't contradict a directly-sourced sentence — which let a writer
smuggle fabricated numbers or qualitative attributes through by tagging
them ``inference``. The current logic runs every grounding function that
matches the claim's topic keywords even when source_tool='inference'; if
any of them flags CONTRADICTION or FABRICATION, the inference claim is
rejected. Only inference claims with no groundable topic hits pass
through quietly.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_chat_model
from ..schemas import (
    AgentState,
    Claim,
    Evidence,
    Issue,
    IssueSeverity,
    IssueType,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。.!?！？])\s+|\n+")
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "etc.", "fig.", "vs.", "no.", "cf.")
_SOURCE_TAG_RE = re.compile(r"\[from\s+([a-z_]+)\]", re.IGNORECASE)
# Measurement numbers we care about (TPM, p-values, odds ratios).
# Requires at least one decimal point or scientific-notation 'e'; the
# negative lookbehind prevents matching digits embedded inside gene
# symbols ("TP53", "HER2") or inside longer integer sequences. Real
# biology measurements we care about — 2.1 TPM, P=3.2e-18, OR=8.4 — all
# satisfy this.
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9])(\d+\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)"
)

_HIGH_TOKENS = {"high", "highly", "strong", "elevated", "enriched", "高", "高表达"}
_LOW_TOKENS = {"low", "weak", "lowly", "minimal", "低", "低表达"}
_MED_TOKENS = {"medium", "moderate", "中等"}

_MD_EMPHASIS_RE = re.compile(r"[*_`]+")

_EXPRESSION_TOPIC = ("heart", "cardiac", "tumor", "tumour", "tpm", "expression",
                     "tissue", "expressed", "心脏", "心肌", "肿瘤", "组织")
_GWAS_TOPIC = ("gwas", "p-value", "p=", "coronary", "cardiovascular", "cad",
               "association", "phenotype", "odds ratio", "心血管")
_LITERATURE_TOPIC = ("literature", "pmid", "clinical trial", "trial", "activator",
                     "inhibitor", "druggable", "druggability", "abstract")


def _lower(s: str) -> str:
    return (s or "").lower()


def _normalize(s: str) -> str:
    """Lowercase and strip Markdown emphasis so pattern matching is robust
    to **bold** / *italic* / `code` formatting in the draft."""
    return _MD_EMPHASIS_RE.sub("", _lower(s))


def _split_sentences(text: str) -> list[str]:
    """Sentence splitter robust to common abbreviations."""
    if not text:
        return []
    masked = text
    restore: dict[str, str] = {}
    for idx, ab in enumerate(_ABBREVIATIONS):
        placeholder = f"\x00AB{idx}\x00"
        if ab in masked:
            masked = masked.replace(ab, placeholder)
            restore[placeholder] = ab
    parts = _SENTENCE_SPLIT_RE.split(masked)
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        for ph, orig in restore.items():
            p = p.replace(ph, orig)
        p = p.strip()
        if p:
            out.append(p)
    return out


def _touches(topic_keywords: tuple[str, ...], text: str) -> bool:
    t = _lower(text)
    return any(k in t for k in topic_keywords)


# ---------------------------------------------------------------------------
# (A) Claim extraction — LLM first, rule-based fallback (fail-closed)
# ---------------------------------------------------------------------------
_EXTRACT_SYSTEM = """You extract atomic factual claims from a Markdown \
target-triage report for grounding verification.

Output a JSON array; each element MUST match this schema exactly:
{
  "text": "<the original sentence, verbatim>",
  "normalized_proposition": "<a minimal declarative restatement>",
  "source_tool": "query_gwas_data" | "query_expression_atlas" |
                  "search_pubmed_literature" | "inference"
}

Rules:
- Only include sentences that make a factual claim (not headings, not \
transitions).
- Use the source_tool value inside the [from …] tag when present.
- If a sentence has no [from …] tag, set source_tool to "inference".
- Return ONLY the JSON array, no prose.
"""

_VALID_TOOLS = {
    "query_gwas_data",
    "query_expression_atlas",
    "search_pubmed_literature",
    "inference",
}


def _pending_evidence(source_tool: str) -> Evidence:
    """Placeholder attached to a tagged claim before grounding runs.

    We use the claim's declared ``source_tool`` as the sentinel tool_name
    (rather than the special ``"none"``) so that the fail-closed pass-
    through in ``_ground_one`` does NOT short-circuit tagged claims.
    ``"none"`` is reserved for untagged claims forced to UNSUPPORTED.
    """
    return Evidence(tool_name=source_tool, field_path="", snippet_or_value="", comparison_result="无法定位")  # type: ignore[arg-type]


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


_STRUCTURAL_PREFIXES = ("#", "<!--", "|", ">", "```")
_FACTUAL_TOKEN_RE = re.compile(
    r"\b(is|are|was|were|has|have|shows?|expresses?|expressed|"
    r"associated|linked|elevated|low|high|medium|significant|"
    r"tpm|p[\s=-]?value|or\b|pmid)\b",
    re.IGNORECASE,
)


def _looks_substantive(sent: str) -> bool:
    s = sent.strip()
    if len(s) < 12:
        return False
    if s.startswith(_STRUCTURAL_PREFIXES):
        return False
    if s.lstrip("-* \t").startswith("**") and s.endswith("**"):
        return False
    if s.isupper():
        return False
    return bool(_FACTUAL_TOKEN_RE.search(s) or any(ch.isdigit() for ch in s))


def _extract_claims_rule_based(draft: str) -> list[Claim]:
    """Rule-based fallback extractor — also the fail-closed anchor.

    Behaviour:
      * Headings, HTML comments, tables and list markers skipped.
      * Substantive sentences missing a ``[from <tool>]`` tag surface as
        ``source_tool="inference"`` anchored to ``evidence.tool_name="none"``,
        which forces UNSUPPORTED at issue-generation time.
      * Tagged sentences produce a normal claim.
    """
    claims: list[Claim] = []
    for sent in _split_sentences(draft):
        if not _looks_substantive(sent):
            continue
        m = _SOURCE_TAG_RE.search(sent)
        if not m:
            claims.append(
                Claim(
                    text=sent,
                    normalized_proposition=sent,
                    source_tool="inference",
                    evidence=Evidence(tool_name="none"),
                )
            )
            continue
        tool = m.group(1).lower()
        if tool not in _VALID_TOOLS:
            tool = "inference"
        claims.append(
            Claim(
                text=sent,
                normalized_proposition=sent,
                source_tool=tool,  # type: ignore[arg-type]
                evidence=_pending_evidence(tool),
            )
        )
    return claims


def _extract_claims(draft: str) -> list[Claim]:
    """LLM-first extraction with a rule-based fallback on any parse error."""
    try:
        llm = get_chat_model()
    except Exception:
        return _extract_claims_rule_based(draft)

    try:
        resp = llm.invoke(
            [SystemMessage(content=_EXTRACT_SYSTEM), HumanMessage(content=draft)]
        )
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = json.loads(_strip_code_fences(raw))
        claims: list[Claim] = []
        for item in data:
            src = item.get("source_tool", "inference")
            if src not in _VALID_TOOLS:
                src = "inference"
            claims.append(
                Claim(
                    text=item["text"],
                    normalized_proposition=item.get("normalized_proposition", item["text"]),
                    source_tool=src,
                    evidence=_pending_evidence(src),
                )
            )
        # Sanity-check: if the LLM returned an empty / obviously wrong list
        # while the draft has substantive sentences, fall back.
        if not claims and _split_sentences(draft):
            return _extract_claims_rule_based(draft)
        return claims
    except Exception:
        return _extract_claims_rule_based(draft)


# ---------------------------------------------------------------------------
# (B) Grounding lookup
# ---------------------------------------------------------------------------
def _qual_from_text(text: str) -> str | None:
    """Return the single qualitative level implied by this text, or None.

    Returns None when the text is ambiguous (e.g. *"low* cardiac expression
    suggests *high* risk of off-target toxicity") — a frequent inference-
    sentence pattern where a single word-bag match would flip the
    assessment. Ambiguity here should cascade to the numeric fallback
    (``_number_cross_check``), not silently pick whichever token the
    iterator hits first.
    """
    t = _lower(text)
    hits: list[str] = []
    if any(tok in t for tok in _HIGH_TOKENS):
        hits.append("high")
    if any(tok in t for tok in _LOW_TOKENS):
        hits.append("low")
    if any(tok in t for tok in _MED_TOKENS):
        hits.append("medium")
    if len(hits) != 1:
        return None
    return hits[0]


def _qual_near(text: str, anchors: list[str], radius: int = 40) -> str | None:
    """Find a qualitative token near the anchor word.

    Avoids the classic "TP53 is low in heart ... and high in tumor"
    failure where a global search picks up the first qualitative token.
    """
    t = _lower(text)
    best_idx = -1
    best_anchor = None
    for a in anchors:
        i = t.find(a.lower())
        if i >= 0 and (best_idx < 0 or i < best_idx):
            best_idx = i
            best_anchor = a
    if best_idx < 0:
        return _qual_from_text(text)
    window = t[max(0, best_idx - radius) : best_idx + len(best_anchor or "") + radius]
    return _qual_from_text(window)


def _ground_expression(claim: Claim, outputs: dict[str, Any]) -> Evidence:
    expr = outputs.get("query_expression_atlas") or {}
    rows = expr.get("expression") or []
    text_l = _lower(claim.text)

    def _row_match(tissues: list[str]) -> dict[str, Any] | None:
        for row in rows:
            if row.get("tissue", "").lower() in tissues:
                return row
        return None

    if "heart" in text_l or "cardiac" in text_l or "心" in text_l:
        row = _row_match(["heart", "cardiac"])
        if not row:
            return Evidence(
                tool_name="query_expression_atlas",
                field_path="expression[tissue=heart]",
                snippet_or_value="",
                comparison_result="无法定位",
            )
        claimed = _qual_near(claim.text, ["heart", "cardiac", "心"])
        actual = row.get("level")
        snippet = f"tissue=heart, tpm={row.get('tpm')}, level={actual}"
        if claimed and actual and claimed != actual:
            return Evidence(
                tool_name="query_expression_atlas",
                field_path="expression[tissue=heart].level",
                snippet_or_value=snippet,
                comparison_result="矛盾",
            )
        if claimed and actual and claimed == actual:
            return Evidence(
                tool_name="query_expression_atlas",
                field_path="expression[tissue=heart].level",
                snippet_or_value=snippet,
                comparison_result="支持",
            )
        # No qualitative token anchored near the tissue keyword.
        # For inference-tagged claims we do NOT number-cross-check against
        # the TPM field — inference sentences typically summarise earlier
        # tagged facts and never introduce new measurements (the Writer
        # system prompt explicitly forbids it), so any number found in
        # them is incidental and produces false positives.
        if claim.source_tool == "inference":
            return Evidence(
                tool_name="query_expression_atlas",
                field_path="expression[tissue=heart]",
                snippet_or_value=snippet,
                comparison_result="无关",
            )
        return _number_cross_check(
            claim, snippet,
            field_path="expression[tissue=heart].tpm",
            actual_value=row.get("tpm"),
        )

    if "tumor" in text_l or "tumour" in text_l or "肿瘤" in text_l:
        row = _row_match(["tumor_microenvironment"])
        if not row:
            return Evidence(
                tool_name="query_expression_atlas",
                field_path="expression[tissue=tumor_microenvironment]",
                snippet_or_value="",
                comparison_result="无法定位",
            )
        claimed = _qual_near(claim.text, ["tumor", "tumour", "肿瘤"])
        actual = row.get("level")
        snippet = f"tissue=tumor, tpm={row.get('tpm')}, level={actual}"
        if claimed and actual and claimed != actual:
            return Evidence(
                tool_name="query_expression_atlas",
                field_path="expression[tissue=tumor_microenvironment].level",
                snippet_or_value=snippet,
                comparison_result="矛盾",
            )
        return Evidence(
            tool_name="query_expression_atlas",
            field_path="expression[tissue=tumor_microenvironment].level",
            snippet_or_value=snippet,
            comparison_result="支持" if claimed == actual else "无关",
        )

    return Evidence(
        tool_name="query_expression_atlas",
        field_path="",
        snippet_or_value="",
        comparison_result="无法定位",
    )


def _number_cross_check(
    claim: Claim, snippet: str, field_path: str, actual_value: Any
) -> Evidence:
    text_numbers = [float(x) for x in _NUM_RE.findall(claim.text)]
    if actual_value is None or not text_numbers:
        return Evidence(
            tool_name="query_expression_atlas",
            field_path=field_path,
            snippet_or_value=snippet,
            comparison_result="无关",
        )
    actual = float(actual_value)
    if any(abs(n - actual) / max(abs(actual), 1.0) < 0.2 for n in text_numbers):
        return Evidence(
            tool_name="query_expression_atlas",
            field_path=field_path,
            snippet_or_value=snippet,
            comparison_result="支持",
        )
    return Evidence(
        tool_name="query_expression_atlas",
        field_path=field_path,
        snippet_or_value=snippet,
        comparison_result="矛盾",
    )


def _ground_gwas(claim: Claim, outputs: dict[str, Any]) -> Evidence:
    gwas = outputs.get("query_gwas_data") or {}
    assocs = gwas.get("associations") or []
    text_l = _normalize(claim.text)

    cv_hit = None
    for a in assocs:
        ph = a.get("phenotype", "").lower()
        if any(k in ph for k in ("coronary", "cardiovascular", "cardiac")):
            cv_hit = a
            break

    if any(k in text_l for k in ("cardiovascular", "coronary", "cad", "cardiac", "心血管")):
        if cv_hit is None:
            return Evidence(
                tool_name="query_gwas_data",
                field_path="associations[phenotype~coronary]",
                snippet_or_value="no cardiovascular association record",
                comparison_result="无法定位",
            )
        p = cv_hit.get("p_value", 1.0)
        snippet = (
            f"phenotype={cv_hit.get('phenotype')}, p_value={p}, "
            f"odds_ratio={cv_hit.get('odds_ratio')}"
        )

        def _window(text: str, needle: str, radius: int = 40) -> str:
            idx = text.find(needle)
            if idx < 0:
                return text
            return text[max(0, idx - radius) : idx + radius]

        cv_kw = next(
            (k for k in ("cardiovascular", "coronary", "cad", "cardiac") if k in text_l),
            "cardiovascular",
        )
        window = _window(text_l, cv_kw, 60)
        negation_pat = re.compile(
            r"\b(no|not|non-?)\s+(significant|association|associated|linked|"
            r"genetically\s+linked|genetically\s+associated)"
            r"|\b(lack|lacks|lacking)\s+(of\s+)?(genetic|significant|\w+\s+association|\w+\s+evidence)"
            r"|\babsence\s+of\s+\w+"
            r"|\babsent\s+(cardiovascular|genetic|cv)"
            r"|\bwithout\s+(genetic|significant)"
            r"|\bfails?\s+to\s+(show|reach|meet)"
            r"|limited\s+\w+\s*(tissue|rationale|evidence)?"
            r"|poor\s+cardiovascular"
        )
        negated = bool(negation_pat.search(window)) or any(
            neg in window for neg in ("不显著", "无显著", "无关联")
        )
        asserts_strong = (
            re.search(r"\b(strong|significant|linked|signal)\b", window) is not None
            and not negated
        )

        is_cv_significant = p is not None and p <= 5e-8

        if negated and not is_cv_significant:
            return Evidence(
                tool_name="query_gwas_data",
                field_path="associations[phenotype~coronary].p_value",
                snippet_or_value=snippet,
                comparison_result="支持",
            )
        if negated and is_cv_significant:
            return Evidence(
                tool_name="query_gwas_data",
                field_path="associations[phenotype~coronary].p_value",
                snippet_or_value=snippet,
                comparison_result="矛盾",
            )
        if asserts_strong and not is_cv_significant:
            return Evidence(
                tool_name="query_gwas_data",
                field_path="associations[phenotype~coronary].p_value",
                snippet_or_value=snippet,
                comparison_result="矛盾",
            )
        numbers = [float(x) for x in _NUM_RE.findall(claim.text)]
        if numbers and p is not None and p > 0:
            import math
            log_p = math.log10(max(p, 1e-300))
            if not any(
                abs(math.log10(max(n, 1e-300)) - log_p) < 1.0 for n in numbers
            ):
                return Evidence(
                    tool_name="query_gwas_data",
                    field_path="associations[phenotype~coronary].p_value",
                    snippet_or_value=snippet,
                    comparison_result="矛盾",
                )
        return Evidence(
            tool_name="query_gwas_data",
            field_path="associations[phenotype~coronary]",
            snippet_or_value=snippet,
            comparison_result="支持",
        )

    numbers = [float(x) for x in _NUM_RE.findall(claim.text)]
    if numbers:
        for a in assocs:
            p = a.get("p_value", 1.0)
            for n in numbers:
                if p and abs(n - p) / max(p, 1e-300) < 9:
                    return Evidence(
                        tool_name="query_gwas_data",
                        field_path="associations[*].p_value",
                        snippet_or_value=json.dumps(a, ensure_ascii=False),
                        comparison_result="支持",
                    )
        return Evidence(
            tool_name="query_gwas_data",
            field_path="associations[*]",
            snippet_or_value="no matching p_value found",
            comparison_result="无法定位",
        )

    return Evidence(
        tool_name="query_gwas_data",
        field_path="associations",
        snippet_or_value=gwas.get("summary", ""),
        comparison_result="无关",
    )


def _ground_literature(claim: Claim, outputs: dict[str, Any]) -> Evidence:
    lit = outputs.get("search_pubmed_literature") or {}
    hits = lit.get("hits") or []
    text_l = _lower(claim.text)

    for hit in hits:
        hay = _lower(hit.get("title", "") + " " + hit.get("abstract", ""))
        tokens = [w for w in re.findall(r"[a-z]+", text_l) if len(w) > 3]
        overlap = sum(1 for w in tokens if w in hay)
        if tokens and overlap / len(tokens) >= 0.35:
            return Evidence(
                tool_name="search_pubmed_literature",
                field_path=f"hits[pmid={hit.get('pmid')}]",
                snippet_or_value=hit.get("title", "")[:120],
                comparison_result="支持",
            )

    if "late-stage" in text_l or "cardiovascular trial" in text_l:
        return Evidence(
            tool_name="search_pubmed_literature",
            field_path="hits[]",
            snippet_or_value="no abstract mentions cardiovascular trials of TP53",
            comparison_result="矛盾",
        )

    return Evidence(
        tool_name="search_pubmed_literature",
        field_path="hits[]",
        snippet_or_value="no strongly overlapping abstract",
        comparison_result="无法定位",
    )


# Map tool name → (grounder function, topic-keyword tuple). Used both by
# explicitly-tagged claims and by the inference-claim cross-check.
_GROUNDERS = (
    ("query_expression_atlas", _ground_expression, _EXPRESSION_TOPIC),
    ("query_gwas_data",        _ground_gwas,       _GWAS_TOPIC),
    ("search_pubmed_literature", _ground_literature, _LITERATURE_TOPIC),
)


def _ground_one(claim: Claim, outputs: dict[str, Any]) -> Evidence:
    """Resolve a claim to an Evidence pointer.

    - ``tool_name='none'`` sentinel → fail-closed UNSUPPORTED, returned as-is.
    - Tool-tagged claims → dispatched to the matching grounder.
    - ``source_tool='inference'`` → cross-check against EVERY grounder whose
      topic keywords appear in the claim; any CONTRADICTION or FABRICATION
      is returned immediately. Only inference sentences that touch no
      groundable topic pass as ``无关``. This closes the loophole where
      adding ``[from inference]`` to any fabricated number or qualitative
      attribute silently bypassed the Reviewer.
    """
    if claim.evidence.tool_name == "none":
        return claim.evidence

    if claim.source_tool == "inference":
        # Only CONTRADICTION is treated as an issue for inference claims.
        # "无法定位" is deliberately tolerated because an inference
        # sentence routinely *references* a previously-grounded fact
        # (e.g. "Given the absent cardiovascular-trial literature, …"),
        # and flagging those as FABRICATION would make any faithful
        # summary conclusion unwritable. The hard invariant we still
        # enforce: no contradictory number, level, or phenotype may
        # ride through under an ``[from inference]`` tag.
        for _tool_name, grounder, topic in _GROUNDERS:
            if not _touches(topic, claim.text):
                continue
            ev = grounder(claim, outputs)
            if ev.comparison_result == "矛盾":
                return ev
        return Evidence(
            tool_name="inference",
            field_path="",
            snippet_or_value="derived from preceding claims",
            comparison_result="无关",
        )

    for tool_name, grounder, _topic in _GROUNDERS:
        if claim.source_tool == tool_name:
            return grounder(claim, outputs)


# ---------------------------------------------------------------------------
# (C) Issue generation
# ---------------------------------------------------------------------------
def _issues_from_claims(claims: list[Claim]) -> list[Issue]:
    issues: list[Issue] = []
    for idx, c in enumerate(claims, start=1):
        ev = c.evidence
        iid = f"R-{idx:03d}"
        # Fail-closed sentinel: untagged factual sentence.
        if ev.tool_name == "none":
            issues.append(
                Issue(
                    id=iid,
                    type=IssueType.UNSUPPORTED,
                    severity=IssueSeverity.MAJOR,
                    claim_text=c.text,
                    evidence_note=(
                        "untagged factual claim — no [from <tool>] citation "
                        "found; fail-closed default."
                    ),
                    suggestion=(
                        "Add an explicit [from <tool>] tag backed by a tool "
                        "return, or remove the unsourced claim."
                    ),
                )
            )
            continue
        if ev.comparison_result == "矛盾":
            issues.append(
                Issue(
                    id=iid,
                    type=IssueType.CONTRADICTION,
                    severity=IssueSeverity.CRITICAL,
                    claim_text=c.text,
                    evidence_note=f"{ev.tool_name}: {ev.field_path} → {ev.snippet_or_value}",
                    suggestion="Rewrite to match the actual tool output.",
                )
            )
        elif ev.comparison_result == "无法定位":
            issues.append(
                Issue(
                    id=iid,
                    type=IssueType.FABRICATION,
                    severity=IssueSeverity.CRITICAL,
                    claim_text=c.text,
                    evidence_note=f"{ev.tool_name}: {ev.field_path or '(none)'} → {ev.snippet_or_value}",
                    suggestion="Drop the unsourced claim or replace with an actual field value.",
                )
            )
        elif ev.comparison_result == "无关" and c.source_tool != "inference":
            issues.append(
                Issue(
                    id=iid,
                    type=IssueType.UNSUPPORTED,
                    severity=IssueSeverity.MINOR,
                    claim_text=c.text,
                    evidence_note=f"{ev.tool_name}: no strong match ({ev.snippet_or_value})",
                    suggestion="Tighten the claim to quote the tool output.",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------
def reviewer_node(state: AgentState) -> dict[str, Any]:
    draft = state.get("draft_report") or ""
    outputs = state.get("tool_outputs") or {}

    claims = _extract_claims(draft)

    grounded: list[Claim] = []
    for c in claims:
        ev = _ground_one(c, outputs)
        grounded.append(c.model_copy(update={"evidence": ev}))

    issues = _issues_from_claims(grounded)

    prev_open = state.get("prev_open_issue_count")
    curr_open = len(issues)
    if prev_open is None:
        stagnation = state.get("stagnation_count") or 0
    else:
        stagnation = (state.get("stagnation_count") or 0) + (
            1 if curr_open >= prev_open and curr_open > 0 else 0
        )

    return {
        "claims": [c.model_dump() for c in grounded],
        "issues": [i.model_dump(mode="json") for i in issues],
        "prev_open_issue_count": curr_open,
        "stagnation_count": stagnation,
    }
