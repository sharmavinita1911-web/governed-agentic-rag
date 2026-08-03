# Governed Agentic RAG — Measuring the Safety–Cost Trade-off of a Governance Layer

An agentic RAG pipeline (**planner → retriever → validator → generator**) wrapped in a
**governance layer of four pluggable controls**, plus an evaluation harness that measures
*how much safety each control buys and what it costs* in latency and tokens — reported as a
reproducible **safety–cost trade-off curve**.

CCE IISc · *LLMs – A Hands-on Approach* · Himani Ponia & Vinita Sharma.

This README is a complete, step-by-step guide. Pick **one** path:
- **[Part A — Run locally](#part-a--run-locally)** (your own machine; CPU is fine but slow)
- **[Part B — Run on Google Colab (free T4 GPU)](#part-b--run-on-google-colab-t4-gpu)** (recommended; much faster)

---

## What it does

| # | Control | What it does | Module |
|---|---|---|---|
| C1 | **Permission-aware retrieval** | filters chunks by ACL/role at the vector layer | `src/controls/c1_permission.py` |
| C2 | **Grounding / citation gate** | rejects answer claims not supported by evidence | `src/controls/c2_grounding.py` |
| C3 | **Injection / poisoning screen** | drops retrieved chunks carrying injected instructions | `src/controls/c3_injection.py` |
| C4 | **Tamper-evident audit log** | hash-chains every retrieval + decision | `src/controls/c4_audit.py` |

All controls share one contract (`src/controls/base.py`):
`apply(ctx, chunks) -> ControlDecision(allow, reasons, filtered_chunks, metadata)`.
A PII redaction step (`src/controls/pii.py`, Presidio with a regex fallback) is an optional screen.

```
HotpotQA + synthetic ACL overlay + PoisonedRAG docs        (src/ingest)
      ▼
  Qdrant index (on-disk, local — no server)                (src/retrieval)
      ▼
  GovernedPipeline: C1 → C3/PII → generate → C2 → (C4 logs every step)   (src/agent)
      ▼
  Eval harness → leak rate · injection success · poison exposure ·
                 faithfulness · audit completeness · latency/token overhead   (src/eval)
      ▼
  artifacts/  metrics.json · tradeoff.png · safety_bars.png · audit_sample.json
```

## Repo layout

```
configs/default.yaml     all settings (model, paths, controls, metrics)
src/ingest/              load HotpotQA, assign ACLs, inject poison, build index
src/retrieval/           Qdrant wrapper + permission filter
src/controls/            C1–C4 + PII (the frozen contract lives in base.py)
src/agent/               Ollama LLM wrapper + the governed pipeline
src/eval/                test suite, metrics, ablation runner, plots
scripts/                 build_index.py · run_eval.py · index_status.py
notebooks/               demo.ipynb · colab_run.ipynb
reports/report.md        the write-up (fill the tables from metrics.json)
```

---

## Prerequisites (both paths)

- **Python 3.11 or 3.12** (NOT 3.13+ — `ragas`/LangChain need `<3.13`).
- **[Ollama](https://ollama.com/download)** — serves the local LLM.
- **git**.
- ~5 GB free disk (models + index). A GPU is optional but makes generation many times faster.

---

## Part A — Run locally

### A1. Clone the repo
```bash
git clone <your-repo-url> governed-agentic-rag
cd governed-agentic-rag
```

### A2. Create a Python 3.12 environment and install deps

Using **[uv](https://docs.astral.sh/uv/)** (recommended — handles the Python version for you):
```bash
uv python install 3.12
uv venv --python 3.12
# activate it:
#   Windows PowerShell:  .venv\Scripts\Activate.ps1
#   Windows cmd:         .venv\Scripts\activate.bat
#   macOS/Linux:         source .venv/bin/activate
uv sync
```

Or with plain `pip` (make sure `python --version` shows 3.11/3.12 first):
```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> **Verify you're on the right interpreter** (a common gotcha):
> ```bash
> python -c "import sys; print(sys.executable)"
> ```
> It must point at `...\governed-agentic-rag\.venv\...`. If not, activate the venv.

### A3. Start Ollama and pull the model
Ollama runs as a background service after install. Pull the generator (a Gemma GGUF, quantized to run on CPU):
```bash
ollama pull hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M
```
This tag must match `llm.model` in `configs/default.yaml` (it already does). For more speed on a
weak CPU, change **both** the pull tag and the config to a smaller quant (`IQ4_XS`, `Q3_K_M`) or a
smaller model (`ollama pull gemma3:1b` and set `llm.model: gemma3:1b`).

### A4. Build the search index (one time, ~3–5 min on CPU)
```bash
python scripts/build_index.py
```
Downloads HotpotQA + the embedding model on first run, then builds the on-disk Qdrant index
(~5,150 chunks). Progress is logged to `artifacts/logs/build_index.log`.

Check it any time:
```bash
python scripts/index_status.py
```

### A5. Run the governance ablation
Start with a fast sanity pass:
```bash
python scripts/run_eval.py --quick
```
Then the real run (CPU is slow — keep N small):
```bash
python scripts/run_eval.py --n-boundary 5
```
Results go to `artifacts/` and progress to `artifacts/logs/run_eval.log`.

> **CPU is slow** (tens of seconds per generation). For the full 6-config run with RAGAS
> faithfulness, use **Part B (Colab)** instead. Locally the harness uses the NLI faithfulness
> backend by default, which is fast and needs no LLM judge.

### A6. See the demo (optional)
```bash
jupyter notebook notebooks/demo.ipynb
```
Shows the same query with governance **off vs on** — a blocked cross-role leak and a defeated
poisoned document, plus the audit chain and a mini trade-off plot.

---

## Part B — Run on Google Colab (T4 GPU)

Recommended: on the free T4, generation is seconds (not ~40 s), so the full ablation with the
stronger **`gemma4:12b`** judge and real RAGAS faithfulness is practical. There is a ready-made
notebook — **`notebooks/colab_run.ipynb`** — with every step as a cell. The same steps, to paste
into any Colab notebook:

### B0. Set the runtime to GPU
Colab menu → **Runtime → Change runtime type → T4 GPU**.

### B1. Clone + install (Colab-tuned deps)
```python
!git clone <your-repo-url> governed-agentic-rag
%cd governed-agentic-rag
# remove Colab's preinstalled langgraph 1.x (unused here, conflicts with the langchain-core ragas needs)
!pip uninstall -y -q langgraph langgraph-prebuilt langgraph-checkpoint langgraph-sdk
!pip install -q -r requirements-colab.txt
```
`requirements-colab.txt` deliberately omits `torch/numpy/pandas/matplotlib` (Colab already has
them — reinstalling torch can break the GPU) and `langgraph` (the eval doesn't use it). Any
leftover `requests`/`cryptography` version warnings from Colab's own packages are harmless.

### B2. Install + start Ollama, pull the model
```python
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time
subprocess.Popen(["ollama", "serve"], stdout=open("/content/ollama.log", "w"), stderr=subprocess.STDOUT)
time.sleep(5)
!ollama pull gemma4:12b
!ollama list
```
(Logging `ollama serve` to `/content/ollama.log` lets you inspect it: `!tail -n 80 /content/ollama.log`.)

### B3. Choose storage + model
```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["GRAG_PROFILE"] = "colab"     # paths → Google Drive, model → gemma4:12b (persists across sessions)

# No Drive? Use an ephemeral session store instead:
# os.environ["GRAG_BASE_DIR"] = "/content/governed_rag"
# os.environ["GRAG_MODEL"]    = "gemma4:12b"
```
Environment variables set here are inherited by the `!python ...` cells below.

### B4. Build the index (once; persists on Drive)
```python
!python scripts/index_status.py     # if it already shows 5150 chunks, skip the build
!python scripts/build_index.py
```

### B5. Run the ablation
```python
!python scripts/run_eval.py --n-boundary 5     # bump to 10–20 for smoother numbers on the GPU
```

### B6. View results
```python
import json, os, sys; sys.path.insert(0, ".")
from IPython.display import Image, display
from src.config import load_config
art = load_config()["paths"]["artifacts_dir"]
print(json.dumps(json.load(open(os.path.join(art, "metrics.json"))), indent=2))
display(Image(os.path.join(art, "tradeoff.png")))
display(Image(os.path.join(art, "safety_bars.png")))
```

---

## What you get (outputs)

All under `artifacts/` (or your Drive folder on Colab):

| File | Contents |
|---|---|
| `metrics.json` | per-config metrics (leak rate, injection success, poison exposure, faithfulness, audit completeness, overhead) |
| `tradeoff.png` | the safety-vs-cost trade-off curve |
| `safety_bars.png` | per-config bar chart of each metric |
| `audit_sample.json` | one full-governance hash-chained audit log |
| `logs/*.log` | timestamped run logs |

## Command reference

```bash
python scripts/build_index.py          # build/rebuild the Qdrant index
python scripts/index_status.py         # print index chunk count + ACL breakdown
python scripts/run_eval.py [flags]     # run the governance ablation
```
`run_eval.py` flags:
- `--quick` — fast sanity run: 2 configs, tiny N, no RAGAS.
- `--n-boundary N` — number of permission-boundary (leak-test) queries per config (default 10).
- `--no-ragas` — skip the faithfulness computation entirely.

## Configuration

Everything is in **`configs/default.yaml`**: corpus size, embedding model, roles, the Ollama
model, which controls are active, and the faithfulness backend. Override per-machine with
environment variables (no code edits):

| Env var | Purpose | Example |
|---|---|---|
| `GRAG_PROFILE` | select a profile block (`local` \| `colab`) — sets paths (and model) | `colab` |
| `GRAG_BASE_DIR` | put data/qdrant/artifacts under one directory | `/content/governed_rag` |
| `GRAG_MODEL` | override the Ollama model tag | `gemma3:12b` |
| `GRAG_OLLAMA_HOST` | point at a specific Ollama daemon | `http://127.0.0.1:11434` |

The compute device is **auto-detected** (CUDA if available), so a GPU is used automatically — no flag needed.

### Choosing a model vs your hardware

| Hardware | Suggested model | Note |
|---|---|---|
| CPU only | `hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M` (default) or `gemma3:1b` | tens of s/call; keep N small, use `--quick` |
| T4 (16 GB, free Colab) | `gemma4:12b` | ~8 GB at Q4, fits with room; **27B does NOT fit** |
| A100/L4 (Colab Pro) | `gemma3:27b` | needs 24–40 GB VRAM |

### Faithfulness backend
`configs/default.yaml → evaluation.faithfulness_backend`:
- `nli` — deterministic entailment scorer (small, fast, CPU-friendly). **Best locally.**
- `ragas` — LLM-as-judge. **Only meaningful with a strong judge** (e.g. `gemma4:12b` on Colab);
  with a 4B/CPU judge it returns 0/NaN, so the harness auto-falls back to NLI.

## Metrics (targets from the proposal)

| Metric | Baseline (governance off) | Governance on |
|---|---|---|
| Unauthorised-disclosure (leak) rate | high (~1.0) | ~0 (C1) |
| Poison exposure / injection success | high | ~0 (C3) |
| Faithfulness (RAGAS or NLI) | base | measurable gain (C2) |
| Audit completeness | 0 | 1.0 (C4) |
| Overhead (latency, tokens) | — | the trade-off curve |

---

## Troubleshooting

**`Microsoft Visual C++ 14.0 or greater is required` / `scikit-network` fails to build** — your
Python is likely 3.13+ (no prebuilt wheels). Use Python 3.12: `uv venv --python 3.12`. On Windows,
also ensure `.python-version` says `3.12` and `pyproject.toml` has `requires-python = ">=3.11,<3.13"`.

**`ModuleNotFoundError: No module named 'qdrant_client'` (or any dep)** — the venv isn't active.
Run `python -c "import sys; print(sys.executable)"`; it must point inside `.venv`. Activate it, or
call the venv Python directly: `.venv\Scripts\python.exe scripts/...`.

**`Storage folder ... data/qdrant is already accessed by another instance`** — the on-disk Qdrant
is single-writer. Make sure only one script touches it at a time (don't run `build_index.py`,
`run_eval.py`, or `index_status.py` simultaneously); close stale Python processes.

**`ModuleNotFoundError: langchain_community.chat_models.vertexai` (ragas import)** — a LangChain
1.x/ragas version clash. Use the pinned stack: `pip install -r requirements.txt` locally, or on
Colab `requirements-colab.txt` after uninstalling `langgraph` (see B1).

**`faithfulness` is `NaN` or `0.0`** — RAGAS with a weak/CPU judge. Locally use
`faithfulness_backend: nli`; for RAGAS use `gemma4:12b` on Colab. The harness auto-falls back to
NLI when RAGAS returns NaN.

**Everything is very slow / `latency ~70000 ms`** — you're on CPU. Use a smaller model
(`gemma3:1b`), `--quick`, a small `--n-boundary`, or run on Colab (Part B).

**Colab `pip` dependency-conflict errors** — use `requirements-colab.txt` and the `langgraph`
uninstall in step B1; the residual `requests`/`cryptography` warnings are harmless.

**Ollama seems stuck / check GPU offload** — inspect the server log (`!tail -n 100 /content/ollama.log`
on Colab). Look for `offloaded N/M layers to GPU`; `out of memory` means the model is too big for
the GPU — use a smaller model/quant.

**`git pull` says "local changes would be overwritten: notebooks/colab_run.ipynb"** — Colab wrote
cell outputs into the tracked notebook. Discard them and pull:
`git checkout -- notebooks/colab_run.ipynb && git pull`. To avoid it recurring:
`git update-index --skip-worktree notebooks/colab_run.ipynb`.

---

## Team
- **Himani** — agent loop, grounding (C2), audit (C4), evaluation, report.
- **Vinita** — corpus/ACL/poison, permission-aware retrieval (C1), injection screen (C3), reproducibility.
