# MQL-0009: NL→MQL retrieval-control experiment (Kaggle package, generate only)

**Status: PACKAGE BUILT, NOT EXECUTED.** No prediction and no score exist. Scoring is **BLOCKED** until the 16 held-out
databases are provisioned on Atlas (see "Scoring" below). Generation does not depend on Atlas.

**Question.** Does the MQL-0008 hybrid-RAG gain (415 → 457 of 892) come from *relevant* retrieval, or simply from giving
`Qwen/Qwen2.5-Coder-3B-Instruct` three demonstrations?

| run | demonstrations | selection |
|---|---|---|
| `A_zero_shot` | 0 | none (frozen `nl-mql-spider1-v1` prompt) |
| `B_random_s1` … `B_random_s5` | 3 | `retrieval.Retriever(method="random", seed=s)`, seeds 20261181 … 20261185 |
| `C_fixed` | 3 | `spider_train_02033`, `spider_train_01967`, `spider_train_03829` for every question |
| `D_hybrid` | 3 | the unchanged MQL-0008 `hybrid_masked` retriever (BM25 k1 1.5, b 0.75 + 1.0 × pattern Jaccard) |

8 runs × 639 questions = **5,112 predictions**.

## What is fixed across arms

- Same model and revision (`488639f1ff808d1d3d0ba301aef8c11461451ec5`), **bfloat16**, greedy, `max_new_tokens` 1024,
  batch 8, seed 42, one Tesla T4.
- Same system prompt, schema text and question. The RAG prompt is the zero-shot prompt with one reference block inserted
  before the final `Question:` line. Tests check this for every question of every run.
- Same eligible corpus view, the same 1,500-token context cap, and the same budget loop. Over-budget examples are skipped,
  never truncated.
- K = 3 for every question in every demonstration run (checked; no budget drops occurred).
- The same evaluation-specific exclusions in every demonstration run: 17 (question, corpus example) pairs. They come from
  the question-level leakage gates: exact or normalised question match, or token Jaccard ≥ 0.8.
- `src/mql0008_core.py`, `retrieval.py`, `patterns.py`, `rag_block.py` are byte-identical to the MQL-0008 package. The
  hybrid plan equals `mql0008_core.build_plan` row for row.

## Evaluation set and leakage safety

- **Items.** 639 questions from 16 whole Spider-train databases: spider-sql-v1 `val_internal`, a database-level split
  made by seeded shuffle with seed 20261027. Membership is from the audited spider-pool-v3 `val_internal`.
- **No database overlap.** None of the 16 databases is among the 123 corpus databases, so the eligible view is the full
  approved corpus. That is 3,445 records, SHA-256 `96552f06…dd199bf`, and the corpus file is unmodified.
- **Package contents.** Only `id`, `db_id` and `question` are shipped. Gold SQL, gold rows, hardness and query-family
  labels stay on Windows.
- **Protected sets.** The 892-item MQL-0008 Spider-dev set and the 65-item test partition are not used.
- **Leakage report.** See `data/leakage_report.json`. SQL-based overlap checks were computed locally (report-only) and
  are not shipped.

## Precision and hardware (resolved discrepancy)

MQL-0006, MQL-0007 and MQL-0008 recorded `bfloat16` on a **Tesla T4**. The T4 has compute capability 7.5 and no native
bf16, so PyTorch *emulates* bf16 there; `torch.cuda.is_bf16_supported()` returns True with emulation.

This package keeps bf16 for comparability with MQL-0008, and handles it as follows:
- It accepts only the registered GPU (`Tesla T4`).
- It records `compute_capability` and `bf16_native` in every run config and prediction.
- It never substitutes fp16 or fp32.
- All 8 runs must share GPU type, dtype, torch and transformers. A resume in another environment is refused.

## Kaggle execution

1. **Dataset:** *Datasets → New Dataset* → upload `nl2mql_mql0009_retrieval_control_bundle.zip`. Keep it **private**.
   Kaggle extracts it under `/kaggle/input/<your-dataset>/nl2mql_mql0009_retrieval_control/`; no slug is hard-coded.
2. **Notebook:** *File → Import Notebook* → `nl2mql_mql0009_retrieval_control.ipynb`. Then *Add data* → your dataset.
3. **Settings:**
   - Accelerator **GPU T4 x2**; one card is used.
   - **Internet On**, to download the public model at the pinned revision. No token is needed.
   - Persistence *Files only*.
   - Attach **no** secrets.
4. **Session 1:** leave `SESSION = 1`, `RESUME_FROM = None`. Then *Save Version → Save & Run All (Commit)*.
   - It runs `A_zero_shot`, `D_hybrid`, `B_random_s1`, estimated at about 6.5 h on a T4.
   - Then set *Settings → Environment → Pin to original environment*.
5. **Download** `/kaggle/working/mql0009_predictions.zip` and keep it.
6. **Session 2:** *Add data → Notebook Output* of session 1's version. Set `SESSION = 2`, `RESUME_FROM = "auto"`, and run.
   - It runs `C_fixed`, `B_random_s2`, `B_random_s3`, about 7.3 h.
   - Earlier runs are hash-verified and carried forward unchanged.
7. **Session 3:** the same, with `SESSION = 3`, attaching session 2's output. It runs `B_random_s4`–`s5`, about 4.9 h.
   - The final zip must show `"status": "complete"` and 5,112 completed predictions in `MQL0009_PREDICTIONS_MANIFEST.json`.

**Total** is about 18.6 GPU-hours, based on the MQL-0008 T4 rates. That is about 60% of a 30 h weekly quota.

**Time limits and interruptions.**
- The notebook stops starting new batches after `TIME_BUDGET_HOURS = 11` and still exports.
- If a version stops for any reason, attach its output and run it again with the same `SESSION` and `RESUME_FROM = "auto"`.
  Completed questions are never regenerated. Generation failures are retried. A different configuration or environment
  is refused.
- **If the pinned environment is lost** (a resume is refused for a torch/transformers/GPU change), do not mix: start a new
  output and regenerate all 8 runs in the new environment.

## Output (`mql0009_outputs/`)

- `predictions/<run>/predictions.jsonl`: one record per question. Schema: `configs/prediction_record.schema.json`.
  - Identity: `prediction_id` = `MQL-0009:<run>:<id>`, plus `arm`, `arm_family`, `random_seed`, `id`, `db_id`.
  - Output: `raw_text`, `normalized_mql` and `parse_status` (informational; the Windows evaluator re-parses `raw_text`),
    plus `status` and `generation_failure`.
  - Retrieval: retrieved ids, ranks, scores, source DBs, exclusions, context tokens, config fingerprint.
  - Generation: `prompt_sha256`, token counts, model id and revision, `precision`, `generation_config`, `runtime`
    (GPU, compute capability, bf16_native, torch, transformers), `config_sha256`, `fingerprint`.
- `predictions/<run>/run_config.json`, `retrieval_log.jsonl` and `checkpoints/predictions.partial.jsonl`: append-only and
  fsynced.
- `MQL0009_STATUS.json` and `MQL0009_PREDICTIONS_MANIFEST.json`: expected versus completed counts per run, checksums.
- `preflight/preflight_session<N>.json` and `SHA256SUMS.txt`.

## Verify locally

```
python -m pytest tests -q
MQL0009_TOKENIZER_DIR=<pinned tokenizer dir> python -m pytest tests -q     # + full plan re-derivation
```

## Scoring (Windows, separate; not part of this package)

`scripts/nl2mql_mql0009/mql0009_scoring.py` (`readiness | verify | score | analyze`) uses the unchanged MQL-0006
evaluator with the read-only Atlas account. Full parity with the provisioning contract is mandatory.

**Blocker:** the 16 databases (69 collections, 16,885 documents) are not on Atlas. The prompt context in this package was
derived offline with the `import_atlas.py` value mapping. After provisioning, scoring refuses any Atlas collection whose
type-tagged row digest differs from that contract.

**Do NOT provision with `scripts/spider1/import_atlas.py`.** That script drops and recreates collections in
`codegen_spider1`, the evaluation database of MQL-0006/0007/0008. It also rewrites the repo's Atlas inventory file. Use a
non-destructive, dry-run-first seeder: `scripts/nl2mql/seed_atlas_spider_dev.py` has the right `seed()` logic (never drops,
refuses the read-only URI, digest check after insert, `--target-db`). Its CLI only accepts the Spider-dev bundle, so it
needs a thin wrapper for this contract.
