# Governed Agentic RAG — Measuring the Safety–Cost Trade-off of a Governance Layer

An agentic RAG pipeline (planner → retriever → validator → generator) wrapped in a
**governance layer of four pluggable controls**, plus a governance evaluation harness
that quantifies *how much safety each control buys and what it costs* in latency and tokens
— reported as a reproducible **safety–cost trade-off curve**.

CCE IISc · *LLMs – A Hands-on Approach* · Himani Ponia & Vinita Sharma.

## The four controls

| # | Control | Where it acts | Module |
|---|---|---|---|
| C1 | **Permission-aware retrieval** — ACL filter at the vector layer | pre-context | `src/controls/c1_permission.py` |
| C2 | **Grounding / citation gate** — Self-RAG-style, rejects unsupported claims | post-generation | `src/controls/c2_grounding.py` |
| C3 | **Injection / poisoning screen** — drops chunks with embedded instructions | pre-context | `src/controls/c3_injection.py` |
| C4 | **Tamper-evident audit logging** — hash-chained record of every step | cross-cutting | `src/controls/c4_audit.py` |

All controls implement one frozen contract (`src/controls/base.py`):
`apply(ctx, chunks) -> ControlDecision(allow, reasons, filtered_chunks, metadata)`.
A PII redaction step (Presidio, with a regex fallback) is available as an optional screen.

## Architecture

```
HotpotQA + synthetic ACL overlay + PoisonedRAG docs
      │  (src/ingest)
      ▼
  Qdrant index (on-disk, local)
      │  (src/retrieval)
      ▼
  GovernedPipeline  ── C1 ─→ C3/PII ─→ generate ─→ C2 ─→ (C4 logs every step)
      │  (src/agent/graph.py, Ollama Gemma 3n E4B GGUF)
      ▼
  Eval harness → leak rate · injection success · faithfulness · audit completeness · overhead
      │  (src/eval) → artifacts/metrics.json, tradeoff.png
```

## Setup

```bash
# Python 3.12 (ragas/LangChain 0.3.x need <3.13). With uv:
uv venv --python 3.12 && .venv\Scripts\activate   # macOS/Linux: source .venv/bin/activate
uv sync                                           # or: pip install -r requirements.txt
```

Local generator + judge — Ollama pulls the Unsloth GGUF of Gemma 3n E4B-it straight from HuggingFace:

```bash
ollama pull hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M
```

Change the quant tag for the speed/quality trade-off: `Q4_K_M` (balanced) → `IQ4_XS`
→ `Q3_K_M` (faster, rougher). The config's `llm.model` must match the tag you pull.
(If chat output ever looks malformed, the alternative `models/Modelfile.gemma3n-e4b`
imports a local `.gguf` with the Gemma chat template forced.)

> No CUDA GPU? This ~4B model runs on CPU at tens of seconds per call — use
> `run_eval.py --quick` or a small `--n-boundary`. For the full ablation at speed,
> run on a GPU or swap `llm.model` for a smaller model (e.g. `gemma3:1b`).

## Reproduce the results

```bash
# 1. Build the local index (downloads HotpotQA + bge-small on first run)
python scripts/build_index.py

# 2. Run the governance ablation → metrics + plots + a sample audit log
python scripts/run_eval.py --n-boundary 10

# 3. Or walk through it interactively
jupyter notebook notebooks/demo.ipynb
```

Outputs land in `artifacts/`: `metrics.json`, `tradeoff.png`, `safety_bars.png`, `audit_sample.json`.
Everything is seeded (`configs/default.yaml → corpus.seed`) for reproducibility.

## Configuration

All knobs live in `configs/default.yaml`: corpus subset size, embedding model, roles,
the Ollama model, which controls are active, and the faithfulness backend
(`ragas` local judge, or `nli` entailment fallback if RAGAS-local is too slow).

## Metrics (targets from the proposal)

| Metric | Baseline | Governance on |
|---|---|---|
| Unauthorised-disclosure (leak) rate | 30–60% | ~0% (C1) |
| Injection / poisoning success | high | sharp reduction (C3) |
| RAGAS faithfulness | base score | measurable gain (C2) |
| Audit completeness | partial/none | 100% (C4) |
| Overhead (latency ms, tokens) | — | reported as the trade-off curve |

## Team

- **Himani** — agent loop, grounding (C2), audit (C4), evaluation, report.
- **Vinita** — corpus/ACL/poison, permission-aware retrieval (C1), injection screen (C3), reproducibility.

## Notes

- Runs fully locally: on-disk Qdrant (no server) + Ollama. One 16–24 GB GPU or CPU with the small quantised model.
- The generator is **Gemma 3n E4B** (Unsloth GGUF `hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M` via Ollama, ~4B class). The NLI faithfulness fallback is kept for when the local judge is weak.
