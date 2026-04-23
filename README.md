# AI Scientist — Target Triage Agent

A Python reference implementation for the **AIDD Target 评估** brief: given a free-text query like *"评估 TP53 作为心血管靶点的潜在风险和可行性"*, an LLM-driven Agent plans its own tool calls over three mock biology tools, drafts a target-triage report, and submits the draft to a **fail-closed Reviewer** that grounds every claim against the raw tool returns. If any claim contradicts a tool return — including one smuggled in under an ``[from inference]`` tag — the Writer is forced to revise, within a deterministic revision budget.

- **Framework:** LangGraph `StateGraph` (Scientist → tools → Writer → Reviewer → Verdict → ... )
- **Guardrail:** claim extraction → field-level grounding → typed `Issue` → pure-code `Verdict` (PASS / REJECT / MAX_ROUNDS / STAGNATED)
- **Production path only:** the agent always goes through a real LLM. No "offline" mode. Tests use monkeypatched stubs for determinism; the delivered demo log is from a real LLM run (see § 2.3).

---

## 1. Layout

```
target-review-agent/
├── .env.example                      # LLM_PROVIDER / keys / tunables
├── pyproject.toml                    # installable package + CLI entry
├── README.md
├── src/ai_scientist/
│   ├── schemas.py                    # Pydantic: Claim / Evidence / Issue / Verdict / AgentState
│   ├── tools.py                      # 3 mock biology tools + StructuredTool wrappers
│   ├── llm.py                        # openai / anthropic provider factory
│   ├── verdict.py                    # pure-code verdict decision (never LLM)
│   ├── graph.py                      # LangGraph topology & routing
│   ├── logging_utils.py              # JSONL trace + Rich console rendering
│   ├── cli.py                        # `ai-scientist "<query>"`
│   └── agents/
│       ├── scientist.py              # tool-calling planner + tool executor
│       ├── writer.py                 # draft / revision prose generation
│       └── reviewer.py               # fail-closed grounding pipeline
├── tests/
│   ├── conftest.py                   # fake-LLM monkeypatch fixtures
│   ├── fixtures/drafts.py            # deterministic bad/good draft fixtures
│   ├── test_verdict.py               # pin verdict precedence
│   ├── test_reviewer.py              # E2E + [from inference] bypass regression
│   └── test_tools.py                 # mock-data contract assertions
├── logs/                             # JSONL traces land here (gitignored)
└── examples/
    ├── sample_run.log                # console transcript of a real LLM run
    ├── sample_trace.jsonl            # machine-readable trace of the same run
    └── final-report.md               # the post-revision report the agent produced
```

---

## 2. Quick start

### 2.1 Install

```bash
cd target-review-agent
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

`pip install -e ".[dev]"` also works if you do not have `uv`.

### 2.2 Configure the LLM

Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`. The default points at **SiliconFlow hosted Qwen3-30B-A3B** — the same endpoint that produced the committed `examples/sample_run.log` (≈3s per LLM call, full run in under a minute):

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...                   # your SiliconFlow key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

Any OpenAI-compatible endpoint also works — swap `OPENAI_BASE_URL` / `OPENAI_MODEL` for api.openai.com, a local OMLX / Ollama server, DeepSeek, Moonshot, 智谱 BigModel, etc. Anthropic (`LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=...`) is also supported; see `.env.example`.

### 2.3 Run the canonical demo

```bash
ai-scientist "评估一下 TP53 作为一个心血管疾病靶点的潜在风险和可行性" \
    --save examples/final-report.md
```

On the SiliconFlow path this finishes in ~40s. The committed artefacts `examples/sample_run.log` / `sample_trace.jsonl` / `final-report.md` capture one such run:

1. Scientist autonomously calls `query_gwas_data` → `query_expression_atlas` → `search_pubmed_literature` (order decided by the LLM, not hard-coded);
2. Writer drafts revision 0, with every factual sentence tagged `[from <tool>]`;
3. Reviewer extracts 6 claims, grounds each against `tool_outputs`, finds **0 issues** — Writer's draft is consistent with the mock data;
4. Verdict → **PASS** — final report saved.

**On the reject→revise cycle.** The committed run converges in one pass because Qwen3-30B-A3B happens to produce a faithful first draft for this query. The guardrail's ability to catch a bad draft, force a rewrite, and re-check is pinned as a deterministic regression test — see `tests/test_reviewer.py::test_reviewer_catches_bad_draft_then_converges`. That test injects a draft containing three seeded errors (contradictory cardiac expression, fabricated CAD p-value, non-existent PubMed claim) and asserts the full REJECT → revise → PASS cycle, so the cycle is continuously exercised in CI even when a given LLM run passes first try. Weaker or more loquacious models (Qwen3-8B, gpt-3.5-turbo, …) will naturally trigger the cycle in production runs without the injection.

### 2.4 Run the tests

```bash
pytest -q        # 15 passed
```

All tests are deterministic; none contacts a real LLM. `tests/conftest.py` unsets the API key so the Reviewer's LLM-first claim extractor falls back to its rule-based extractor (the same fail-closed branch that guards against malformed LLM JSON in production).

---

## 3. Architecture choice — why LangGraph

### 3.1 Three hard constraints

1. **Reviewer → Writer is an explicit loop edge.** The brief requires the Reviewer to send the report back on failure. This is a cycle, not a chat.
2. **Reviewer must directly inspect tool ground-truth.** Its job is to say "this claim contradicts `tool_outputs[X].field`". Fishing tool outputs out of message history and re-parsing JSON every round is lossy and slow.
3. **State transitions must be code-deterministic.** "Issue list non-empty ⇒ revise" is a business rule, not a taste call. Letting an LLM decide the next hop dilutes the guardrail.

### 3.2 Ruled-out alternatives

| Alternative | Why it loses |
|---|---|
| Hard-coded `if/else` pipeline | Explicitly forbidden by the brief. |
| Single agent with tool-calling (ReAct) | Cycles via message history are opaque; no natural place for a separate Reviewer. |
| Plan-and-Execute | Reasonable, but you still need to glue the Reviewer cycle on top. No saving in complexity. |
| AutoGen GroupChat | Hands turn-taking to an LLM. Constraint 3 is violated — you cannot guarantee the Reviewer always runs after the Writer and vice versa. |

### 3.3 Why LangGraph fits

- `StateGraph` gives a shared TypedDict (`AgentState`), so the Reviewer reads `tool_outputs` directly as a dict rather than re-parsing messages. (Constraint 2)
- `add_conditional_edges` is the right primitive for **code-owned** routing — the Reviewer's output routes via `verdict.route_from_verdict`, a pure function. (Constraint 3)
- Cycles are first-class: `verdict → writer` is just another edge. (Constraint 1)
- It is **not** a chat framework, so control flow never accidentally ends up embedded in prose.

### 3.4 Topology

```
START ─► scientist ─┬─► tools ─► scientist      (ReAct inner loop for evidence)
                    │
                    └─► writer ─► reviewer ─► verdict ─┬─► writer   (REJECT)
                                                       │
                                                       └─► finalize ─► END
                                                           (PASS / MAX_ROUNDS / STAGNATED)
```

Three functional nodes (`scientist`, `writer`, `reviewer`), one tool executor (`tools`), one pure-code decision node (`verdict`), one finalizer. The Scientist / tools inner loop exposes every tool_call and every tool return as discrete state updates, which is what makes the JSONL trace useful for debugging.

---

## 4. Anti-hallucination strategy (this demo)

The system organises three separated roles into a **Generator / Critic / Judge** pattern:

| Role | Node | What it does |
|---|---|---|
| **Generator** (attack surface) | `writer` | Produces the draft. Hallucinations naturally originate here. |
| **Critic** (defence) | `reviewer` | Treats the draft as untrusted input. Grounds every claim against the raw `tool_outputs` captured in state; emits typed `Issue`s. |
| **Judge** (arbitration) | `verdict` | A pure-code node that reads the Critic's issues and deterministically decides `PASS / REJECT / MAX_ROUNDS / STAGNATED`. Never an LLM call — the accept/re-queue decision is auditable, testable, diff-reviewable. |

The Reviewer is a three-stage pipeline, **fail-closed at every stage**:

### 4.1 Stage A — claim extraction (LLM-first, rule-based fallback)

The primary path asks the LLM for a JSON array of `{text, normalized_proposition, source_tool}` objects. If the JSON fails to parse (malformed output, rate-limit, network glitch), the extractor falls back to a sentence-level rule-based extractor. Either path emits an **`UNSUPPORTED`** claim for any substantive sentence missing a ``[from <tool>]`` source tag — so an untagged hallucinated sentence cannot silently slip through a failed LLM call.

### 4.2 Stage B — grounding lookup (pure code)

For each claim the Reviewer resolves a field pointer against `state['tool_outputs']`:

- categorical levels (`low/medium/high`) via exact match, using a proximity-bounded search so `"low in heart ... high in tumor"` is split into two independent judgements, not one ambiguous one;
- p-values compared within an order of magnitude (log-space);
- absence of a matching record ⇒ ``无法定位``;
- explicit negation handling (`"no significant"`, `"lack of association"`, `"absence of ... "`) so a correctly-phrased negative claim is classified as `支持`, not `矛盾`.

### 4.3 Stage C — issue generation (pure code)

| Grounding result | Issue |
|---|---|
| `矛盾` | `CONTRADICTION` / critical |
| `无法定位` | `FABRICATION` / critical |
| `无关` on a tool-sourced claim | `UNSUPPORTED` / minor |
| extraction or sentinel failure | `UNSUPPORTED` / major (fail-closed) |

### 4.4 The `[from inference]` back-door (and how it's closed)

An earlier iteration of this file made a subtle mistake: any claim tagged ``[from inference]`` was given a free pass in the grounding step. A writer could then smuggle arbitrary fabrications — *"TP53 is highly expressed in heart and strongly associated with CAD [from inference]"* — without tripping a single issue, because the grounder only dispatched to a real tool-check when `source_tool != "inference"`.

The current logic treats ``[from inference]`` as follows: **if the sentence's topic keywords match any of the three tool domains (tissue names, GWAS phenotypes, literature markers), the corresponding grounder is run and any `CONTRADICTION` is surfaced.** Only inference sentences that touch no groundable topic pass through. `无法定位` on inference claims is tolerated — a faithful summary line legitimately references facts the Reviewer cannot always anchor, and flagging those would make any derivative conclusion impossible to write.

The regression is pinned as `tests/test_reviewer.py:: test_inference_tag_cannot_bypass_contradictory_expression`, plus a counter-test that a benign derivative sentence is not flagged.

### 4.5 Verdict precedence (`verdict.py`)

Pinned by `tests/test_verdict.py`:

1. `PASS` — no open issues.
2. `MAX_ROUNDS` — `revision_count >= MAX_REVISIONS` (default 2).
3. `STAGNATED` — open-issue count plateau ≥ threshold (default 2).
4. `REJECT` — otherwise.

**The Verdict never consults an LLM.** This is the single most important property of the guardrail.

---

## 5. From Demo to industrial pipeline

This prototype implements the skeleton. Four highest-ROI extensions for a real AIDD pipeline, in the order I'd ship them:

### 5.1 Replace the regex grounder with a bounded LLM judge + verified code check

The Stage-B grounder in this demo is regex and keyword heuristics. That is tractable for the three-tool mock but scales poorly to real `variant × tissue × pathway × paper` claims. The production move is:

- a small "judge" LLM that takes one claim + one candidate tool field and returns exactly one of `{支持, 矛盾, 无关}` — no prose, no explanation — so the decision is still code-routable;
- every judge call is checked by a second-pass deterministic verifier (numeric comparison, vocabulary whitelist) which can **override** the judge to `无关` if the judge's `支持` is not independently justified. This is the generator/critic pattern applied recursively one layer deeper.

### 5.2 Retrieval-grounded literature with citation anchors

Replace mock `search_pubmed_literature` with a real PubMed RAG where every hit carries `pmid + sentence_offset + embedding_id`. The Reviewer's claim-to-evidence pointer collapses to a citation check: each literature claim must quote or paraphrase a specific sentence. Unsupported abstracts become the main class of FABRICATION in production, so anchoring literature properly is where guardrail recall improves most.

### 5.3 Golden-set evals + nightly regression

A versioned set of queries with known-good reports and known-bad injected hallucinations. Nightly CI computes faithfulness, coverage, and hallucination-catch-rate and alerts on regression. The ``[from inference]`` back-door fix was exactly the kind of regression that would have slipped through a manually-reviewed PR but would have been caught by an adversarial golden-set test — and so that regression is now a pinned unit test.

### 5.4 HITL escalation with feedback capture

When the graph terminates on `STAGNATED` or `MAX_ROUNDS`, don't just return the draft — route to a human reviewer queue, capture the human verdict, and feed the rationale back as supervised data for the Stage-B LLM judge. Humans stay in the loop for the cases the system cannot resolve, but they never have to read a draft the system already knows is good.

Multi-reviewer ensembles, provenance tracking (`data_source + schema_version + fetched_at + upstream_etag`), and knowledge-graph anchoring are plausible next steps beyond this list but are not what moves the needle first.

---

## 6. Environment variables

See `.env.example` for the full list.

| Var | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` (any OpenAI-compatible endpoint) or `anthropic` |
| `OPENAI_API_KEY` | — | required when `LLM_PROVIDER=openai` |
| `OPENAI_BASE_URL` | `https://api.siliconflow.cn/v1` | any OpenAI-compatible endpoint; swap for api.openai.com / Ollama / OMLX / etc. |
| `OPENAI_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | model name (provider-specific) |
| `ANTHROPIC_API_KEY` | — | required when `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Claude model name |
| `MAX_REVISIONS` | `2` | revision budget before `MAX_ROUNDS` |
| `STAGNATION_THRESHOLD` | `2` | plateau counter before `STAGNATED` |
| `LOG_DIR` | `logs` | JSONL trace directory (gitignored) |

### 6.1 Proxy behaviour

When `OPENAI_BASE_URL` points at a localhost endpoint or a known domestic Chinese gateway (SiliconFlow, DeepSeek, DashScope, 智谱 BigModel, 通义, Moonshot, 火山 Volces), the CLI automatically clears any globally-set `HTTP_PROXY` / `ALL_PROXY` vars before any HTTP client is constructed. This avoids the common failure mode where `ALL_PROXY=socks5://...` tries to tunnel to 127.0.0.1 through a SOCKS proxy and raises an `ImportError` from `httpx` about the missing `socksio` package. For geo-restricted endpoints (api.openai.com, api.anthropic.com) the user's proxy config is left alone.
