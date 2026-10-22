# CodeGen: Open Source LLMs for Code Generation, SQL Handling, and RAG

College capstone project exploring open-source LLMs for code generation, with an initial focus on translating natural language into MongoDB queries (MQL).

## Mentor deliverable: Spider → MQL dataset + model comparison (current)

- **Dataset**: `data/processed/spider1_nl_mql.jsonl`, 279 validated, unique Spider 1.0 (dev)
  NL → MQL examples. Each gold MQL is executed on MongoDB Atlas (`codegen_spider1`) and
  matches the official gold SQL result. 267 of the 279 also match on 3 official
  test-suite DB variants. See `docs/spider1_dataset.md`.
- **Secondary set**: `data/processed/nl_mql_dataset.jsonl`, 68 validated Spider 2.0-Lite
  examples (much harder, local SQLite subset only). See `docs/conversion_report.md`.
- **Comparison**: Qwen2.5-Coder-Instruct (3B and 7B, local CPU via llama.cpp) vs
  openai/gpt-oss-120b (a large open-weight model accessed through Groq, not a closed
  frontier model). Same prompt, same examples, same Atlas execution and the same comparator
  for every model.
  - Both Qwen models were evaluated on all 279 examples.
  - All three models were evaluated on all 279 (gpt-oss-120b was resumed on 2026-10-15 for
    the last 68). Strict execution accuracy: Qwen-3B 132/279 = 47.3%, Qwen-7B 181/279 = 64.9%,
    gpt-oss-120b 232/279 = 83.2%.
  - See `docs/spider1_model_comparison.md` and `outputs/evaluation_spider1/`.
  - Week 2 verification: every verdict reproduced by re-executing the saved MQL on Atlas.
    See `docs/week2_report.md` (`python scripts/spider1/rescore.py [--reexecute]`).

```powershell
python scripts/spider1/build_dataset.py                     # re-validate + write dataset
python scripts/spider1/run_eval.py --provider reference     # harness self-check (gold replay)
python scripts/spider1/run_eval.py --provider qwen --label qwen2.5-coder-3b   # baseline
python scripts/spider1/run_eval.py --provider qwen7b --label qwen2.5-coder-7b
python scripts/spider1/run_eval.py --provider frontier_groq --label gpt-oss-120b   # makes Groq calls; NOT needed for the reports
python scripts/spider1/compare.py qwen2.5-coder-3b qwen2.5-coder-7b gpt-oss-120b   # reports only, no model calls
python scripts/spider1/error_analysis.py qwen2.5-coder-3b qwen2.5-coder-7b gpt-oss-120b
python -m pytest tests -q
```

## Week 3: RAG baseline (expanded 455 benchmark)

Retrieval-augmented Qwen2.5-Coder-3B, piloted on the **validation** partition only. The test
partition is untouched. See `docs/week3_rag_report.md`.

```powershell
python scripts/spider1/make_splits_expanded.py         # verify database-disjoint 318/72/65 split
python scripts/rag/build_corpus.py --check             # corpus rebuilds byte-for-byte (Atlas read-only)
python scripts/rag/retrieval_eval.py                   # offline retriever selection (no model calls)
python scripts/rag/run_rag_eval.py --experiment zs_rerun      # then random_k3, hybrid_k3, hybrid_k3_g3, hybrid_k3_fk
python scripts/rag/run_rag_eval.py --experiment hybrid_k3 --partition train   # pre-registered train runs: see report §12
python scripts/rag/rescore_rag.py                      # re-execute saved RAG outputs (no model calls)
python scripts/rag/rag_report.py                       # outputs/rag_week3/report.{json,md}
python scripts/rag/rag_report.py --partition train     # outputs/rag_week3/report_train.{json,md}
python -m pytest -q
```

## Baseline Model

- **Qwen-2.5-Coder-Instruct** (zero-shot baseline initially; fine-tuning later)

## Current Status: Week 1

Week 1 goal: define an **NL → MongoDB/MQL** task derived from **Spider 2.0-Lite**, including:

- Schema and query-type definitions
- Conversion of Spider examples into MongoDB/MQL format
- Creation and validation of an initial dataset

Raw Spider 2.0-Lite files live in `data/raw/spider2/` and are **never modified**.

## Planned Roadmap

1. Week 1 — Task definition, schema/query types, initial dataset creation and validation
2. Zero-shot baseline with Qwen-2.5-Coder-Instruct
3. RAG augmentation (schema/docs retrieval)
4. Fine-tuning
5. Evaluation and ablations
6. Deployment

## Project Structure

| Directory | Purpose |
|---|---|
| `data/raw/` | Original Spider 2.0-Lite and Spider 1.0 files (read-only, never modified) |
| `data/interim/` | Intermediate converted/transformed data |
| `data/processed/` | Final validated NL → MongoDB/MQL dataset |
| `data/splits/` | Database-disjoint train/validation/test split of the 279 Spider 1.0 examples (retrospective; see `docs/week2_report.md` §9.2) |
| `src/data/` | Dataset loading and preparation utilities |
| `src/conversion/` | Logic for adapting Spider examples to MongoDB/MQL |
| `src/validation/` | Logic for validating/executing MQL against MongoDB |
| `src/models/` | Model loading and inference (later weeks) |
| `src/rag/` | Retrieval-augmented generation components (later weeks) |
| `src/evaluation/` | Metrics and evaluation harness (later weeks) |
| `scripts/` | Runnable scripts for inspection, conversion, validation |
| `notebooks/` | Exploratory analysis notebooks |
| `tests/` | Unit tests for conversion and validation |
| `configs/` | Dataset paths and MongoDB connection settings |
| `docs/` | Task definitions, schema notes, query-type definitions |
| `outputs/` | Predictions, evaluations, logs (generated; not committed) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your MongoDB URI
```

## Rules

1. Never modify files in `data/raw/`.
2. Keep raw, interim, and processed data separated.
3. No secrets in code — MongoDB URI/database config comes from environment variables (`.env`, gitignored).
4. Datasets and model weights are downloaded by documented commands into gitignored folders (`data/raw/`, `models/`) and never committed.
