# Vinita Handoff - Retrieval Governance

## Current Branch

`vinita-retrieval-governance`

## What Is Implemented

- HotpotQA distractor subset loading in Colab.
- Corpus creation from HotpotQA context paragraphs.
- Synthetic ACL/RBAC metadata overlay.
- Synthetic poisoned documents.
- `BAAI/bge-small-en-v1.5` embeddings.
- Local Qdrant vector index in Colab/Drive.
- Reusable retrieval function with governance on/off mode.
- Role-based Qdrant metadata filtering using `allow_<role>`.
- Unauthorized retrieval sanity check.
- Baseline vs governed leak-rate evaluation.
- Rule-based prompt-injection / poisoning detector.
- Simple regex-based PII detector and redactor.
- CSV output for ACL evaluation.

## Key Result

Current ACL evaluation result:

```text
Baseline query leak rate: 1.0
Baseline chunk unauthorized rate: 0.6
Governed query leak rate: 0.0
Governed chunk unauthorized rate: 0.0
```

Interpretation:

Without governance, every test query retrieved at least one unauthorized chunk. With Qdrant metadata filtering enabled, unauthorized retrieval dropped to zero on the current test set.

## Poisoning Screen Result

The poisoning test retrieved two synthetic poisoned chunks. The rule-based injection screen blocked both before final context construction.

Important insight:

ACL filtering and injection screening solve different problems. Public poisoned documents can still pass ACL checks, so an injection screen is needed after retrieval.

## Files

- Notebook: `notebooks/IISC_Project.ipynb`
- ACL evaluation CSV: `reports/results/retrieval_acl_eval.csv`

## APIs Himani Can Use

```python
retrieve(query, user_role=None, governed=True, top_k=5)
```

Purpose:

- `governed=False`: baseline retrieval, no ACL filter.
- `governed=True`: role-filtered retrieval using Qdrant metadata.

```python
check_role_access(points, role)
```

Purpose:

- Returns retrieved chunks that the caller role should not access.

```python
screen_retrieved_points(points)
```

Purpose:

- Splits retrieved chunks into safe allowed chunks and blocked injection-like chunks.

```python
detect_pii(text)
redact_pii(text)
```

Purpose:

- Detects/redacts simple synthetic PII examples.

## Suggested Next Integration

Himani can plug `retrieve()` into the LangGraph planner/retriever/validator/generator flow.

Recommended order:

1. Use `retrieve(..., governed=False)` for baseline agentic RAG.
2. Use `retrieve(..., governed=True)` for governed agentic RAG.
3. Run `screen_retrieved_points()` before passing chunks to the generator.
4. Use `check_role_access()` in evaluation.
5. Use the CSV result as the first evidence table in the report.
