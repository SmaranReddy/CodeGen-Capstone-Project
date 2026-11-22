# CodeGen: Open Source LLMs for Code Generation, SQL Handling, and RAG

College capstone project exploring open-source LLMs for code generation.

## Project objective (confirmed 2026-10-29)

**Primary task: natural language → MongoDB query (NL→MQL).** Given a question and a MongoDB
schema, the model generates a read-only MQL query (`aggregate` pipeline or `find`). The query is
executed on **MongoDB Atlas**, and it is correct only if its executed result matches the gold
result. MQL text is never compared with SQL text.

The plan requires:

1. a zero-shot Qwen2.5-Coder-3B baseline;
2. a RAG baseline;
3. fine-tuning Qwen on MQL, a query language it was not specialised for;
4. a comparison against a frontier LLM;
5. error analysis and ablations;
6. a final integrated demonstration.

The NL→SQL work (spider-sql-v1, SQL-0001) is a **separate auxiliary experiment**. Its numbers are
SQL-only and are never MQL performance (see "Auxiliary SQL track" below).

Every quantitative result is recorded in `docs/experiment_accuracy_ledger.md` (NL→MQL results are
in section B as `MQL-xxxx`). The prioritised roadmap and its gates are in
`docs/nl2mql_next_steps.md`.

## Eight-week plan and status

Weeks 1–3 are done. Weeks 4–8 below are the **proposed** sequence from
`docs/nl2mql_next_steps.md`, not a completed record.

Earlier schedule, kept for reference: the original roadmap was (1) task definition and dataset,
(2) zero-shot Qwen baseline, (3) RAG, (4) fine-tuning, (5) evaluation and ablations,
(6) deployment. `docs/week3_rag_report.md` §11 placed fine-tuning and the one-time test run in
Week 4. The 2026-10-25 converter audit found known-wrong labels in the fine-tuning set, so a
data-readiness gate now comes first. That moves fine-tuning to Week 6 and the final test run to
Week 8.

| Week | Focus | Status (2026-10-29) | Evidence |
|---|---|---|---|
| 1 | Task definition, schemas, query types, first validated NL→MQL set | Done | `docs/dataset_task_definition.md`; Spider 2.0-Lite 68; Spider 1.0 v1 279 |
| 2 | Zero-shot Qwen 3B (and 7B) vs a large model, Atlas execution | Done | Ledger MQL-0001 (279), MQL-0002 (455) |
| 3 | RAG baseline | Done (pilot; gain not significant after Holm) | Ledger MQL-0003, MQL-0004, MQL-0005 |
| 4 | Data-readiness gate: local MongoDB, `sql2mql-v2`, 300-item conversion pilot, native-set re-execution, licence answers | **Proposed. Not started** | `docs/nl2mql_next_steps.md` P1–P4 |
| 5 | Build the versioned MQL training corpus (`ft-v2`); freeze the evaluation protocol | Proposed | P5–P6 |
| 6 | QLoRA fine-tuning (3B pilot, then 7B) + fine-tuned RAG; selection on validation (72) only | Proposed | P7–P8 |
| 7 | Frontier comparison on the same protocol, error analysis, ablations | Proposed | P7, P9 |
| 8 | One-time test-partition (65) evaluation, integrated demo, final report | Proposed | P9 |

### Current NL→MQL results (all Atlas execution; from the ledger)

| Benchmark | Qwen2.5-Coder-3B | Qwen2.5-Coder-7B | gpt-oss-120b |
|---|---|---|---|
| v1, 279 items (authored gold MQL) — MQL-0001 | 132/279 (47.3%) | 181/279 (64.9%) | 232/279 (83.2%) |
| Expanded, 455 items (MQL result vs gold-SQL result) — MQL-0002 | 217/455 (47.7%) | 287/455 (63.1%) | 385/455 (84.6%) |

- Both benchmarks are derived from Spider **dev**; the 455 includes the 279.
- The Qwen runs used CPU llama.cpp GGUF q4_k_m; gpt-oss-120b ran through Groq.
- **gpt-oss-120b is a large open-weight model, not a closed frontier model.** Whether it satisfies
  the plan's "frontier LLM" comparison is an open question for the mentor.
- RAG (Qwen-3B, clean Kaggle GPU control, 318 train-partition items): zero-shot 160, random-k3 155,
  hybrid-k3 172. The primary contrast, hybrid vs random, is +5.3 points, McNemar p = 0.021, Holm
  0.064: **not significant**. See ledger MQL-0005.
- A QLoRA pilot adapter (3B) has been trained but not yet scored on Atlas (see "Fine-tuning status").

## Mentor deliverable: Spider → MQL dataset + model comparison (Week 2)

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

## Fine-tuning status (pilot run done, no accuracy number yet)

### QLoRA pilot on the approved v2.1 corpus (2026-11-19)

Notebook: `notebooks/kaggle/codegen_finetune.ipynb` (Unsloth, runs on Kaggle or Colab T4).
Outputs are kept in the notebook so the run can be read without re-running it.

- **Model:** `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit` (4-bit QLoRA).
- **Data:** SFT files built from the approved v2.1 corpus, 3,220 train / 225 dev. The dev split
  holds out whole databases. Builder: `scripts/finetune_pilot/build_pilot_sft.py`
  (see `scripts/finetune_pilot/README.md`). The `.jsonl` files are gitignored; only
  `data/finetune_pilot/pilot_v0/manifest.json` is committed.
- **Settings:** LoRA r=16, alpha=32, dropout 0, all 7 attention + MLP projections
  (29.9M trainable params, 0.96%); 1 epoch, batch 2 x grad-accum 4, lr 2e-4, cosine, warmup 3%,
  seed 3407, max length 2048. Loss is on the assistant answer only.
- **Run:** 403 steps, 43.7 min on one T4, peak GPU memory 2.7 GB, final train loss 0.0456.
- **Dev loss (100 dev examples):** 0.723 before, 0.026 after.
- **Before/after demo (3 dev questions):** the base model wraps its answers in markdown fences
  and uses its own style. After fine-tuning it returns bare JSON in the corpus style, and one
  demo answer matches gold exactly. The join demo is still wrong (wrong collection, `$lookup`
  missing).
- **Not an accuracy result.** Lower dev loss and valid JSON only show the pipeline works.
  Execution accuracy on Atlas with the project scorer is still pending; nothing has been
  added to the ledger beyond the provisional entry MQL-0009.
- **Next:** score the adapter on Atlas; try r 8/32, lr 1e-4, 2 epochs, and the 7B model; use
  both Kaggle T4s to cut training time.

### ft-v1 (not run)

- `docs/fine_tuning_plan.md` (ft-v1) and `scripts/finetune_v1/train_qlora.py` exist, but ft-v1
  has never been trained.
- **Do not run ft-v1 as written.** `data/finetune/ft_v1/` still contains the 88 converter labels
  the 2026-10-25 audit found wrong (86 in `train.jsonl`, 2 in `dev_internal.jsonl`;
  `docs/sql2mql_independent_audit.md`), plus 51 labels with SQL-level date-equality errors (45 in
  train, 6 in dev_internal). The data-readiness gate in `docs/nl2mql_next_steps.md` comes first.
- Native NL→MQL datasets (AptMQL-Bench, SM3, TEND-v1) have been researched
  (`docs/native_nl2mql_dataset_research.md`) but **none is on disk or re-executed**.

## Auxiliary SQL track (SQL-only; not MQL performance)

- Dataset: spider-sql-v1 (`docs/spider_sql_v1_dataset.md`), train 7,773 / val_internal 797 /
  dev_full 1,034 / test_full 2,147.
- SQL-0001: zero-shot Qwen2.5-Coder-3B on `dev_full`, official EX 446/1,034 (43.1%), EM 378/1,034
  (36.6%), test-suite EX 630/1,034 (60.9%). Ledger section A.
- These numbers measure SQL generation, use a different metric and runtime, and must never be
  shown next to an MQL number as a comparison.

## Week 1 history

Week 1 defined the **NL → MongoDB/MQL** task on **Spider 2.0-Lite** (schema and query-type
definitions, conversion of examples to MQL, a validated set of 68). The 135 offline-runnable
Spider 2.0-Lite ids could not support 200–300 examples, so the mentor deliverable moved to
Spider 1.0. Only gold-replay harness checks have run on the 68 (no model baseline).

Raw Spider 2.0-Lite files live in `data/raw/spider2/` and are **never modified**.

## Project Structure

| Directory | Purpose |
|---|---|
| `data/raw/` | Original Spider 2.0-Lite and Spider 1.0 files (read-only, never modified) |
| `data/interim/` | Intermediate converted/transformed data |
| `data/processed/` | Final validated NL → MongoDB/MQL dataset |
| `data/splits/` | Database-disjoint train/validation/test split of the 279 Spider 1.0 examples (retrospective; see `docs/week2_report.md` §9.2) |
| `data/splits_expanded/` | Database-disjoint 318/72/65 train/validation/test split of the 455 expanded benchmark. The 65-item test partition is used once, at the final evaluation |
| `data/finetune/ft_v1/` | ft-v1 MQL SFT candidate (not training-ready; see "Fine-tuning status") |
| `data/finetune_pilot/` | SFT pilot data manifests (`.jsonl` files gitignored; see "Fine-tuning status") |
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
