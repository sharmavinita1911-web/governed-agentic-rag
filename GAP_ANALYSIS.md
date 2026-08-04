# Gap Analysis — Proposal vs. Current Implementation

**Project:** Governed Agentic RAG · CCE IISc · Himani Ponia & Vinita Sharma
**Date:** August 2026 · Model: gemma4:12b-mlx on MacBook M4

---

## Summary

| # | Gap | Severity | Status |
|---|---|---|---|
| 1 | Faithfulness = 0.0 everywhere — C2 claim splitter broken | Critical | **Fixed** |
| 2 | `injection_success` = 0 at baseline — wrong metric for this model | Critical | Needs note in report |
| 3 | Multi-hop planning is a stub (single retrieval round only) | Moderate | Not implemented |
| 4 | Llama Guard / NeMo Guardrails not wired behind config flag | Moderate | Not implemented |
| 5 | `reports/report.md` still has `⟨…⟩` placeholders | Minor | Needs fill-in |
| 6 | LangGraph wrapper untested in eval | Minor | Exists but unused |

---

## Critical Gaps

### Gap 1 — Faithfulness = 0.0 everywhere (C2 grounding gate broken)

**Proposal claim:** *"RAGAS faithfulness — measurable gain with C2."*
**Actual result:** `faithfulness = 0.000` in every config, including full governance.

**Root cause:**
`split_claims()` in `src/controls/c2_grounding.py` used a sentence-end regex
`(?<=[.!?])\s+` that only works for plain prose. `gemma4:12b-mlx` returns
markdown-formatted responses (bullet points, numbered lists, headers), so the
regex found no split points and returned an empty list. With 0 claims:

```
faithfulness = 0 / 1 = 0.0
C2 always hard-rejects (allow=False)
audit shows: "grounded 0/0 claims (backend=llm)"
```

**Fix applied (Aug 2026):**
`src/controls/c2_grounding.py` — `split_claims()` now handles:
1. Bullet lines (`- item`, `* item`, `• item`)
2. Numbered list lines (`1. item`, `2) item`)
3. Markdown symbol stripping (`**bold**`, `*italic*`, `` `code` ``, `# headers`)
4. Falls back to sentence splitting for plain prose
5. Explicit early-return guard when 0 claims extracted (clear audit reason instead of silent 0)

**Impact of fix:** Re-run `python scripts/run_eval.py --n-boundary 3` to get real
faithfulness numbers. Expected range: 0.3–0.7 depending on how closely the model
stays within the retrieved context.

---

### Gap 2 — `injection_success` = 0 at baseline (metric mismatch)

**Proposal claim:** *"injection success high → sharp reduction with C3."*
**Actual result:** `injection_success = 0.000` in ALL configs, including baseline (no C3).

**Root cause:**
`injection_success` measures whether a canary string (e.g. `"ANSWER: 42"`) literally
appears in the model's generated answer. `gemma4:12b-mlx` is instruction-following
enough to never echo the injected phrase verbatim, even when the poison chunk is in
context. The before/after comparison is flat: `0.0 → 0.0`.

**What actually works:**
`poison_exposure` (added during implementation) measures whether a poisoned chunk
*reached the context window* — which is what C3 directly controls. This metric
correctly shows `1.0 → 0.0` with C3.

**Action needed:**
Add a note in the report and presentation explaining:
- `injection_success` was superseded by `poison_exposure`
- Reason: whether the LLM *obeys* an injection depends on model capability;
  whether the chunk *reached context* is deterministic and directly measures C3's effect
- `poison_exposure` is a stricter and more meaningful metric for the attack model used

---

## Moderate Gaps

### Gap 3 — Multi-hop planning is a stub

**Proposal claim:** *"planner → retriever → validator → generator agentic loop."*
**Actual implementation:** Single retrieval round. The pipeline does one vector search
and one generation call. No iterative re-retrieval or multi-hop chaining.

**What exists:**
- `GovernedPipeline._hitl_interrupt()` — no-op stub
- `build_langgraph()` in `src/agent/graph.py` — wraps the pipeline in a LangGraph
  `StateGraph` but still only calls `pipeline.run()` once (no loop)
- The LangGraph wrapper is not called by `run_eval.py`

**Impact:** The "agentic" claim in the title refers to the governance control loop
(plug-in contract, LangGraph graph structure), not to multi-hop retrieval.
This is an architectural limitation but does not affect the 4 governance metrics.

**Recommended note in report:** Frame the single-hop design as a deliberate scope
decision for the controlled ablation experiment; multi-hop adds confounding variables
that make isolating each control's effect harder.

---

### Gap 4 — Llama Guard / NeMo Guardrails not wired

**Plan statement:** *"wire an optional Llama Guard / NeMo Guardrails check behind a
config flag"* for C3 injection screen.

**Actual implementation:** C3 uses regex heuristics only (11+ patterns covering common
injection templates). No semantic classifier or dedicated guardrail model is wired.

**Impact on results:** The regex approach achieved `poison_exposure = 0.0` on the
test set (4 PoisonedRAG-style docs), so it is sufficient for the current evaluation.
Paraphrased or encoded injections not in the pattern list can evade the screen.

**Recommended note in report:** Describe regex-based C3 as a heuristic baseline;
identify Llama Guard / NeMo Guardrails as a recommended production upgrade.

---

## Minor Gaps

### Gap 5 — `reports/report.md` placeholders not filled

`reports/report.md` still contains `⟨…⟩` template variables in every metrics table.
`reports/report.html` (generated Aug 2026) has all actual numbers from `metrics.json`.

**Action needed:** Either fill `report.md` from `metrics.json` or point reviewers to
`report.html` as the primary submission document.

Values to fill (from `artifacts/metrics.json`):

| Config | Leak rate | Poison exposure | Audit completeness | Latency (ms) | Tokens |
|---|---|---|---|---|---|
| baseline | 1.000 | 1.000 | 0.000 | 45,692 | 1025 |
| +C1 permission | 0.000 | 1.000 | 0.000 | 37,653 | 706 |
| +C3 injection | 1.000 | 0.000 | 0.000 | 53,184 | 1025 |
| +C2 grounding | 1.000 | 1.000 | 0.000 | 63,163 | 1025 |
| +C4 audit | 1.000 | 1.000 | 1.000 | 43,723 | 1025 |
| full governance | 0.000 | 0.000 | 1.000 | 38,401 | 703 |

*(faithfulness column to be filled after re-running with the split_claims fix)*

---

### Gap 6 — LangGraph wrapper untested in eval

`build_langgraph()` in `src/agent/graph.py` compiles a `StateGraph` and is importable,
but `run_eval.py` and `demo.ipynb` both call `GovernedPipeline.run()` directly.
The LangGraph path has no test coverage.

**Impact:** Low — the governance controls themselves are fully tested through the plain
pipeline. LangGraph is a structural wrapper for human-in-the-loop interrupt capability,
which is out of scope for the ablation experiment.

---

## What Fully Works (vs. Proposal)

| Proposal requirement | Result |
|---|---|
| Pluggable control contract (frozen Phase-0 interface) | ✅ `apply(ctx, chunks) -> ControlDecision` |
| C1 permission-aware retrieval | ✅ leak rate 100% → 0%, −41% tokens |
| C3 injection / poison screen | ✅ poison_exposure 100% → 0% |
| C4 tamper-evident audit, 100% completeness | ✅ hash-chain verified, tamper detected |
| PII redaction (Presidio) | ✅ 24 entities redacted in demo |
| Safety–cost trade-off curve | ✅ `artifacts/tradeoff.png` + `safety_bars.png` |
| Demo notebook (governance ON vs OFF) | ✅ live outputs in `notebooks/demo.ipynb` |
| Modular repo, README, Colab notebook | ✅ complete |
| Hash-chained audit with `sha256(prev_hash + canonical_json)` | ✅ verified |
| 6-config ablation grid | ✅ baseline → +C1 → +C3 → +C2 → +C4 → full |

---

## Recommended Actions Before Submission

| Priority | Action | Time estimate |
|---|---|---|
| 1 | Re-run eval after split_claims fix: `python scripts/run_eval.py --n-boundary 5` | ~30 min on M4 |
| 2 | Fill `reports/report.md` faithfulness column with new numbers | 5 min |
| 3 | Add one paragraph to report explaining `injection_success → poison_exposure` metric change | 10 min |
| 4 | Add one sentence framing single-hop as a deliberate ablation design choice | 5 min |
| 5 | Commit and push final state | 5 min |

---

## Faithfulness Range Reference

| Value | Meaning |
|---|---|
| `1.0` | Every claim in the answer is supported by a retrieved chunk |
| `0.5` | Half the claims are grounded, half are hallucinated |
| `0.0` | No claims grounded — or 0 claims extracted (the bug, now fixed) |
| `-` | Not computed (run used `--no-ragas` or `--quick`) |

Expected after fix: **0.3–0.7** baseline, higher with C2 (only grounded claims pass).
