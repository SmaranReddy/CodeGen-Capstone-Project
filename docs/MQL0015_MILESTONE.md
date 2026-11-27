# MQL-0015 milestone: hybrid RAG + fine-tuned Qwen2.5-Coder-3B, 59.8 % execution accuracy

This branch (`research/mql0015-embedding-ft-r3v`) is a source snapshot of the NL→MQL system as it stood when the
**MQL-0015** experiment was completed (2026-11-25). Later work (MQL-0016 Stage 0, the FT2-COV six-run fine-tuning
programme) is deliberately **not** included.

## 1. Result

1,000 questions sampled from the Spider 1.0 `test_full` split (40 databases), one registered run, scored once.

| System | Description | Correct / 1,000 | Accuracy |
|---|---|---:|---:|
| A0 | zero-shot Qwen2.5-Coder-3B-Instruct | 471 | 47.1 % |
| A0V | A0 + schema validator with one feedback retry | 497 | 49.7 % |
| R3V | hybrid RAG (R3, top-3 demonstrations) + validator/retry | 544 | 54.4 % |
| **FT-R3V** | **LoRA fine-tuned model + R3 + validator/retry** | **598** | **59.8 %** |

- The primary registered contrast was FT-R3V vs R3V: +5.4 pp, paired bootstrap 95 % CI [2.7, 8.1], exact McNemar p = 1.4 × 10⁻⁴.
- The full report is in `reports/mql0015/results_report.md`, with the analysis in `analysis.json` and provenance hashes in `RUN_MANIFEST.json`.

### How accuracy is measured

There are **no hand-written gold MQL pipelines**. Each generated MongoDB aggregation pipeline is executed on MongoDB Atlas. Its result rows are compared with the rows produced by running the **reference Spider SQL** on the original SQLite database (cross-engine execution agreement).

The comparison uses the unchanged evaluator in `scripts/nl2mql/dev1034_core.py`:
- order-aware when the SQL has ORDER BY;
- value normalisation;
- explicit outcome taxonomy: correct, incorrect, invalid, execution failure, timeout, not comparable.

**Every outcome other than `correct` counts as wrong.** MQL text is never compared with SQL text.

## 2. End-to-end workflow

```
natural-language question
  -> schema / context construction        Atlas-form collection schema + sample documents
  -> hybrid retrieval (R3)                 BM25 + BGE-small dense (Qdrant) -> RRF -> structural Jaccard -> top 3
  -> prompt construction                   zero-shot template + "solved examples from other databases" block
  -> Qwen2.5-Coder-3B generation           bf16, greedy, max 1,024 new tokens (base model, or base + LoRA adapter)
  -> MQL validation                        JSON shape, collections / fields against the schema (SchemaValidator)
  -> retry if needed                       one schema-feedback retry; the retry is kept only if it has fewer issues
  -> MongoDB execution                     read-only Atlas user, aggregate() with a timeout
  -> result comparison                     against the SQLite execution of the reference Spider SQL
```

## 3. Where each part lives

| Component | Code |
|---|---|
| Spider schema and prompt template (zero-shot) | `scripts/spider1/prompt_spider1.py`, `scripts/spider1/spider1_kit.py` |
| Atlas import, the representation of each Spider table as a collection | `scripts/spider1/import_atlas.py` |
| Offline Atlas-form context derivation | `scripts/nl2mql_mql0009/build_mql0009_package.py` (`derive_context`) |
| Generation, checkpointed passes, assembly and export (Kaggle runtime) | `kaggle/nl2mql_mql0015_final_1000/src/mql0015_core.py`, `.../mql0008_core.py` (`make_qwen_generator`), `.../mql0009_core.py` |
| Structural cues (SQL / question patterns) | `scripts/rag/patterns.py` (runtime copy: `kaggle/.../src/patterns.py`) |
| BM25 + structure-aware retriever (R0) | `scripts/rag/retrieval.py`, `scripts/nl2mql_mql0010/src/retriever_v2.py` |
| **Dense embeddings (BAAI/bge-small-en-v1.5 @ `5c38ec7c`)** | `scripts/nl2mql_mql0011/src/embedder.py` (pinned revision, file sha check) |
| **Embedding artifact and Qdrant index construction** | `scripts/nl2mql_mql0011/build_embedding_artifact.py`, `scripts/nl2mql_mql0011/ingest_vector_store.py` |
| **Qdrant store** (embedded mode, exact search) | `scripts/nl2mql_mql0011/src/qdrant_store.py`, `scripts/nl2mql_mql0011/src/vector_store.py` |
| **Hybrid retrieval: RRF (k = 60) + Jaccard rerank, leave-target-DB-out** | `scripts/nl2mql_mql0011/src/retriever_dense.py` (mode `rrf_fused` = R3) |
| Planning: top-3 selection under a token budget, exclusions | `scripts/nl2mql_mql0011/src/planner_dense.py`, `scripts/nl2mql_mql0010/src/prompt_v2.py`, `kaggle/.../src/planner.py` |
| Prompt block with retrieved examples | `kaggle/.../src/rag_block.py`, `.../prompt_v2.py` |
| Validation and retry | `scripts/nl2mql_mql0010/src/validator.py`, `scripts/nl2mql_mql0010/src/recovery.py` |
| Execution and scoring | `scripts/nl2mql/dev1034_core.py`, `scripts/nl2mql_mql0015/mql0015_scoring.py` |
| MQL-0015 analysis (McNemar, bootstrap, slices) | `scripts/nl2mql_mql0015/src/mql0015_analysis.py` |
| MQL-0015 sampling and registration | `scripts/nl2mql_final_1000/` |
| MQL-0015 package build and notebook | `scripts/nl2mql_mql0015/build_mql0015_package.py`, `make_mql0015_notebook.py`, `kaggle/nl2mql_mql0015_final_1000/` |
| Fine-tuning (QLoRA / Unsloth), data preparation, inference | `finetuning/q3b_r16_q10/` (see its README) |
| Earlier milestones | MQL-0009 (retrieval control), MQL-0010 (validator / retry, holdout), MQL-0011 (dense retrieval), MQL-0012 (offline retrieval confirmation), MQL-0014 (R3 recheck): `scripts/nl2mql_mql00xx/` and `kaggle/` |

**MQL-0013** (`scripts/nl2mql_mql0013/`) was registered but superseded and never executed. Its plan builder is included only because the MQL-0015 R3 plans were byte copies of its frozen plan rows (see `build_mql0015_package.py`).

## 4. What you must supply to reproduce it (not in this repository)

| Item | Why it's excluded | How to obtain |
|---|---|---|
| Spider 1.0 (SQLite databases, train / dev / test JSON) | dataset licence and size; contains the test questions and gold SQL | the official Spider release; see `scripts/spider1/` for the build steps |
| Processed datasets (`data/processed/**`), the MQL-0015 1,000-item sample, prompts and plans | evaluation data and gold references | rebuild with `scripts/spider1/`, `scripts/nl2mql_final_1000/` and the package builders |
| Approved RAG corpus v2.1 (3,445 records) | derived training data | built by the sql2mql pipeline (`scripts/finetune_v2/`) |
| `Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1ff808d1d3d0ba301aef8c11461451ec5` | model weights | Hugging Face (downloaded by the notebook at that revision) |
| `BAAI/bge-small-en-v1.5` @ `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | model weights | Hugging Face; `embedder.py` checks the file hash |
| LoRA adapter `q3b_r16_q10` (adapter_model.safetensors sha256 `7096760a…`) | model weights | retrain with `finetuning/q3b_r16_q10/`; metadata is in `adapter_metadata/` |
| Qdrant index `mql0011__v2_1_view2912__bge_small_en_v1_5__5c38ec7c__masked_q` | generated artifact | `build_embedding_artifact.py`, then `ingest_vector_store.py` |
| MongoDB Atlas cluster with the evaluation collections, and a read-only user | external service and credentials | provision with `scripts/spider1/import_atlas.py`; set `MONGODB_URI` in `.env` (see `.env.example`) |
| Kaggle (T4 GPU) for generation | compute | upload the package as a private dataset and run the notebook manually |

**Full numerical reproduction is therefore not possible from this repository alone.** It needs the datasets, the model and adapter weights, the index and an Atlas cluster above. The code, configurations, pinned versions and hashes are included so that each of those artifacts can be rebuilt and checked.

## 5. Setup (code-level)

```bash
git clone -b research/mql0015-embedding-ft-r3v https://github.com/SmaranReddy/CodeGen-Capstone-Project.git
cd CodeGen-Capstone-Project
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-mql0015.txt                    # retrieval stack: install torch first (CPU wheel is enough)
cp .env.example .env                                       # then set MONGODB_URI (read-only user) etc.; never commit .env
pytest -q tests/test_mql0011_dense.py tests/test_mql0011_store_contract.py tests/test_mql0015_sampling.py
```

Many tests check frozen experiment artifacts (packages, registrations, datasets). Without those local artifacts they **skip or fail by design**. The release report lists which tests run on a clean clone.

## 6. Known limitations at this milestone

- **Coverage gap.** The fine-tuning and RAG corpus contain almost no anti-join, set-operation, nested or 3+-table queries. On those slices FT-R3V is weak; see `reports/mql0015/results_report.md` "By structure".
- **Metric caveats.** The metric is cross-engine execution agreement, not semantic equivalence. Degenerate gold results (for example empty sets) can make a wrong query count as correct. The sensitivity analyses are in the report.
- **One registered run, one seed per arm.** The 3B model is evaluated on a single Kaggle T4 configuration (bf16 emulated).
