# MQL-0006: zero-shot Qwen2.5-Coder-3B-Instruct, NL→MQL on Spider dev (892 items, held-out test databases excluded)

## Status: **COMPLETED — VERIFIED (2026-10-30)**

- **Prepared:** 2026-10-29/30. **Scope:** single model (Qwen only).
- **Generated** on Kaggle in two sessions under one fingerprint. The first session stopped at 716 / 892 because of CUDA OOM; the resumed session generated the other 176 (§3).
- **Scored** locally on Windows against Atlas, as the read-only user.
- **Verified:** `local_atlas_scoring.py verify` reports `all_checks_passed: true`, `complete: true`.

## Executive summary

- **Result:** zero-shot Qwen2.5-Coder-3B-Instruct gets **421 / 892 (47.2%, Wilson 95% CI 43.9–50.5%)** on the primary metric, **cross-engine execution accuracy**: its MQL, executed on MongoDB Atlas, returns the same result as Spider's gold SQL on SQLite. This is not native MQL exact-match accuracy.
- **The model almost always produces usable MQL.**
  - 99.66% of outputs are well-formed JSON MQL.
  - 99.55% pass the read-only safety gate. One output used a forbidden write stage and was blocked before execution.
  - 97.87% execute on Atlas without error.
- **Failures are semantic, not syntactic.** Of the 413 incorrect answers:
  - 230 return the wrong number of rows;
  - 111 return the wrong columns;
  - 71 have wrong values.
- **Difficulty is the dominant factor.**
  - easy 72.0%;
  - medium 59.2%;
  - hard 18.2%;
  - extra-hard 10.0%.

  Multi-join, nested and set-operation questions are where a 3B model breaks down.
- **The run was clean.**
  - 0 infrastructure failures and 0 generation failures.
  - 39 items (4.4%) could not be judged, because the gold `ORDER BY … LIMIT` cuts through a tie. They count against the model in the denominator.
  - 96 items were generated at batch size 1 after GPU out-of-memory, which is disclosed in §3.

| Item | Value |
|---|---|
| Model | **Qwen/Qwen2.5-Coder-3B-Instruct** (single model) |
| Frontier-model comparison | **Deferred** to a later experiment. The Groq-hosted open-weight `openai/gpt-oss-120b` is not part of MQL-0006 |
| Evaluation set | 892 items, 17 databases, 70 MongoDB collections |
| Metric | Cross-engine execution accuracy |
| Architecture | Kaggle generates only: Qwen on the GPU, with no Atlas access, no external API and no secrets. All SQLite-vs-Atlas scoring runs on Windows (`scripts/nl2mql/local_atlas_scoring.py`), as the read-only `codegen_eval_readonly` |

Run guide: `docs/kaggle_nl2mql_spider_dev_1034.md`. Ledger: `docs/experiment_accuracy_ledger.md` MQL-0006
(verified, note (e)). Presentation report: `outputs/nl2mql_spider_dev_1034_model_comparison/v1/scoring/results_report.md`.

> **Limitation (applies to every number in this document).** The metric measures cross-engine execution
> equivalence between generated MQL on MongoDB Atlas and Spider gold SQL on SQLite. It is not native MQL
> exact-match accuracy. Results depend on the fidelity of the database mapping and result-normalization procedure.
> Matching on one database instance does not prove general semantic correctness, and the gold SQL labels have not
> been independently human-verified for every question.
>
> `results_report.md` carries the same disclosure, generated at scoring time. Its wording differs by one phrase
> ("the chosen normalization policy"). The file is hash-protected, so it was not edited.

## 1. Objective

The task is to measure how well a small open code model does at the project's primary task:
natural-language question + MongoDB schema → read-only MQL, executed on MongoDB Atlas.

- **Setting:** zero-shot, with no fine-tuning, RAG or examples.
- **Item set:** every Spider dev question not on a held-out test database.

This experiment is separate from:
- MQL-0001 (279) and MQL-0002 (455), whose Qwen runs were CPU GGUF;
- MQL-0003..0005 (RAG);
- SQL-0001 (SQL-only).

Never compare its numbers with those directly (§9).

## 2. Dataset and provenance

| Item | Value |
|---|---|
| Source | `data/processed/spider_sql_v1/dev_full.jsonl`: 1,034 Spider dev records in `dev.json` order. Its sha256 is checked against the spider-sql-v1 manifest. The file itself is unchanged |
| Held-out exclusion | The 65-item MQL test partition is database-disjoint (`employee_hire_evaluation`, `pets_1`, `tvshow`). Those databases come from the ID-only `data/splits_expanded/split_manifest.json`; **the partition file was never opened**. All 142 dev records on those DBs are excluded: the 65 held-out questions, plus 77 others, of which 65 are paraphrases with identical gold SQL |
| **Evaluation set** | **892 records, 17 databases, 70 collections**, in original order. Ids sha256 `21d1478b…0197` |
| Per DB | world_1 120, car_1 92, cre_Doc_Template_Mgt 84, dog_kennels 82, flight_2 80, student_transcripts_tracking 78, wta_1 62, network_1 56, concert_singer 45, orchestra 40, poker_player 40, course_teach 30, singer 30, museum_visit 18, battle_death 16, voter_1 15, real_estate_properties 4 |
| Hardness (official Spider labels) | easy 214 · medium 380 · hard 148 · extra 150 |
| Gold reference | Spider gold SQL executed read-only on SQLite 3.40.1 at bundle-build time. Every record reproduces its spider-sql-v1 result hashes. The rows ship in the bundle. On Windows, every gold SQL is re-executed before scoring: a changed answer fails the run, while SQLite-version noise only produces a warning |
| Prior curation | 424 of the 892 were audited for the 455 benchmark (390 included, 34 excluded). **468 have never been audited.** All 892 count |
| Bundle | `kaggle/nl2mql_spider_dev_1034_bundle_v2.zip`, sha256 `e9986dbe…0760`. Unchanged, and reused for the Qwen-only run |
| Licence | Spider: CC BY-SA 4.0 |

Reference status (computed offline; it decides how each item is judged):

| Status | Items | Policy |
|---|---:|---|
| `ok` | 846 | Normal comparison; ORDER BY ties may appear in any order |
| `tie_at_limit` | 44 | `ORDER BY … LIMIT` cuts through a tie. A match counts as correct; a mismatch counts as **not_comparable**, never incorrect |
| `order_ties_unverified` | 2 | `SELECT DISTINCT … ORDER BY`. "Same rows, different order" counts as not_comparable |
| `utf8_dependent` / `reference_error` | 0 / 0 | — |

## 3. Model and generation

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-3B-Instruct` (open-weight, 3B), HF revision `488639f1ff808d1d3d0ba301aef8c11461451ec5` (as SQL-0001) |
| Runtime | Kaggle GPU (Tesla T4, 14.56 GiB), Hugging Face transformers 5.16.1, torch 2.11.0+cu128. **dtype: bfloat16**, as recorded in `run_config.json` (`"dtype": "bfloat16"`). `QWEN_DTYPE = "auto"` picks bf16 whenever `torch.cuda.is_bf16_supported()` is true, and on this T4 it was, through emulation. (Corrected 2026-10-30: earlier versions of this document said fp16.) The resumed session must keep bf16, because the dtype is part of the run fingerprint |
| Decoding | Greedy (`do_sample=False`), `max_new_tokens` 1,024, batch 8 with left padding, seed 42. A batch that hits CUDA out-of-memory is retried one prompt at a time with the same settings (see below) |
| Prompt | Frozen `nl-mql-spider1-v1` (template sha256 `7401bf14…`, as MQL-0001..0005). It contains the question, the item's database schema (collections, field types, foreign keys, 2 sample documents per collection), instructions and the JSON MQL output format |
| Input isolation | Prompts are built from `(db_id, question)` only, through a `PromptInput` object. They contain no gold SQL, gold MQL or reference result. The Kaggle preflight re-derives all 892 and asserts byte equality and no gold leak |
| Checkpointing | Every batch is checkpointed and the run is resumable through `RESUME_FROM`. Errors are retried; finished ids are never regenerated. A fingerprint guard stops a run if settings, GPU type or library versions change. Before anything is copied, the resume source is verified: SHA256SUMS, fingerprint recompute, configuration, no id completed twice, and checkpoint equal to `predictions.jsonl`. Every earlier completed prediction must come out byte-identical, or the run is partial |
| CUDA OOM fallback (added 2026-10-30) | `torch.cuda.OutOfMemoryError` on a batch of 8: the batch's tensors are released, then `gc.collect()` and `torch.cuda.empty_cache()` run, and each prompt of that batch is retried alone. The prompt, `max_new_tokens`, dtype, revision and greedy decoding are unchanged, and nothing is truncated. A prompt that also runs out of memory alone is recorded as a generation error, with no further retry that session. Other exceptions are never treated as OOM. Each new prediction records `configured_batch_size`, `effective_batch_size`, `oom_recovery` and `oom_retries`. `PREDICTIONS_STATUS.json` → `models.<label>.generation` counts normal-batch versus fallback completions, remaining failures, OOM events and total generation time |

**Generation history (2026-10-30).**
- **First session:** 716 of 892 generated in batches of 8, in 6,624.8 s of generation wall time. The other 176 failed with CUDA out-of-memory: 22 batches, mostly the long-schema databases `dog_kennels` (82) and `student_transcripts_tracking` (78), plus `poker_player` 7, `singer` 6 and `battle_death` 3. That run is `partial` and is not scored.
- **Resumed session (done 2026-10-30):** generated only those 176, in 2,887.4 s, and kept the 716 byte for byte.
  - The resume source was verified, and the fingerprint (`184a7215a8a53394`) was unchanged.
  - 12 batches of 8 hit OOM again. Their **96 prompts were generated one at a time**, with none failing alone; the other 80 completed in normal batches of 8.
  - **Total:** 796 predictions in normal batches, 96 through the fallback, 0 failures; total generation wall time 9,512.2 s.
- **Disclosure:** predictions generated at batch size 1 can differ slightly in numerics from what batch 8 would have produced (different padding and kernel shapes), even with greedy decoding. Their number is reported in the final status.

## 4. Scoring protocol (Windows)

1. **Verify before Atlas.** The predictions zip is checked for:
   - checksums and bundle identity;
   - exactly the 892 ids, in order, with no duplicates;
   - model id, revision and prompt hash;
   - the fingerprint and per-item prompt hashes;
   - only the Qwen model being present;
   - no held-out content and no credentials.

   It is then imported without overwriting.
2. **Reference.** Gold SQL is re-executed on the bundled SQLite files.
3. **Safety gate.** Each prediction passes through:
   - `parse_output`;
   - `validate_mql` + `forbidden_anywhere`: byte copies of the MQL-0001..0005 harness, which reject `$out`, `$merge`, `$function`, `$accumulator`, `$where` and admin stages at any depth;
   - a no-cross-database rule;
   - size and depth limits.

   Only collections of the item's own database are allowed.
4. **Execution.** Atlas `codegen_spider1`, as the read-only user. The preflight fails if any write privilege is granted. Each query gets `maxTimeMS` 30 s and a bounded cursor; more than 25,000 rows is reported as `result_too_large`.
5. **Comparison.** `spider1_kit.exec_match`, unchanged. Its rules:
   - an ObjectId or null `_id` is dropped;
   - nested objects are flattened;
   - column names and order are ignored;
   - numbers are compared to 4 decimal places;
   - strings must match exactly, and the comparison is type-sensitive;
   - NULL equals a missing field;
   - rows are compared as a multiset;
   - order matters only with an ORDER BY, and is tie-aware.

   A secondary verdict normalising BSON types is reported alongside.

| Outcome | Meaning | Attributed to |
|---|---|---|
| `correct` | Executed, and matches the gold result | — |
| `incorrect` | Executed and compared, but mismatched (values, row/column count, order, more than 25,000 rows) | model |
| `invalid` | `format_error`, `invalid_mql` or `unsafe_mql`; never executed | model |
| `timeout` | Atlas `maxTimeMS` exceeded | model |
| `failed` / model | Atlas rejected the validated pipeline | model |
| `failed` / infrastructure | Generation error or Atlas unreachable. Retried by re-running `score`, with no regeneration. A run with any of these left is `partial` | infrastructure |
| `not_comparable` | The reference cannot judge this output (§2) | reference |

## 5. Database mapping (SQLite → Atlas; full detail in `data/atlas_mapping.json`)

| Aspect | Mapping |
|---|---|
| Collections | One per table, named `<db_id>__<table>`; 70 collections for the 17 DBs. Case is preserved |
| Fields and types | Every column keeps its SQLite name. Storage class is kept per value: INTEGER → int32/int64, REAL → double, TEXT → string, NULL → explicit null |
| Dates / numbers stored as text | Kept as strings |
| Duplicates | Preserved |
| `_id` | Generated ObjectId |
| Relationships | Flat; foreign keys are listed in the prompt |
| UTF-8 | `wta_1__players` has 2 U+FFFD-replaced strings; no gold result depends on them |
| Evidence | Import 2026-10: 54.4 MB data + 9.6 MB index (M0 limit 512 MB). Parity check 2026-10-20: identical. Read-only user verified 2026-10-20 |
| At run time | `local_atlas_scoring.py preflight` runs before any Kaggle GPU use. It checks privileges, 70/70 collections, counts and the full row digest. `score` repeats the privilege, coverage and count checks. **Not verified at preparation time** |

## 6. Metrics (all with denominator 892; nothing silently excluded)

- **Primary:** cross-engine correct / 892, with a Wilson 95% interval.
- **Validity and execution:** format validity, structural validity, read-only validation, Atlas execution success, comparison coverage (correct + incorrect), and accuracy on compared items.
- **Breakdowns:** every outcome and detail-category count, with model and infrastructure failures separated, and accuracy by hardness with a Wilson CI per level.
- **Cost:** latency (mean, median, p90, p95, p99, max; amortised per item over the batch), generation and evaluation wall time, and input/output tokens. There is no API cost, since the model runs on a local GPU.
- **Sensitivity counts:** reference-ok items (846) and items without the 34 prior-audit exclusions (858). These can be recomputed from `evaluation_results.jsonl` and never replace the primary metric.
- **Paired model comparison:** none in MQL-0006, because there is a single model. A frontier comparison, when run, will need its own experiment ID, generated on the same 892 ids.

## 7. Comparability notes

- The MQL-0001/0002 Qwen-3B numbers came from a CPU GGUF q4_k_m runtime on different item sets. They are context only, not a baseline for this run.
- **Historical results stay separate:**
  - MQL-0001: 132/279, 47.3%. A curated 279-item benchmark, authored gold MQL.
  - MQL-0002: 217/455, 47.7%. A curated 455-item set that includes the 65 held-out test items.
  - MQL-0006: 421/892, 47.2%.

  The three percentages look alike, but the item sets, curation, reference type and runtime (4-bit GGUF on CPU vs bf16 transformers on GPU) all differ. They are not a replication and must not be read as "no change".
- SQL-0001 is an NL→SQL result (SQL-only) and is not shown beside this MQL result.

## 8. Results (verified 2026-10-30)

Source: `outputs/nl2mql_spider_dev_1034_model_comparison/v1/scoring/models/qwen2.5-coder-3b-instruct/baseline_summary.json`, with generation counts from `v1/predictions/PREDICTIONS_STATUS.json`. Verification: `v1/scoring.verification.json`, `all_checks_passed: true`, `complete: true`.

| Metric (/892) | Qwen2.5-Coder-3B-Instruct |
|---|---|
| Run status | complete, verified |
| **Cross-engine correct** (primary: cross-engine execution accuracy) | **421 / 892** |
| Accuracy | **47.2%** |
| Wilson 95% CI | 43.9–50.5% |
| Format validity | 889 / 892 (99.66%) |
| Structural validity | 889 / 892 (99.66%) |
| Read-only validation passed | 888 / 892 (99.55%) |
| Atlas execution success | 873 / 892 (97.87%) |
| Comparison coverage (correct + incorrect) | 834 / 892 (93.5%); accuracy on compared items 421 / 834 (50.48%) |
| correct / incorrect / invalid / timeout / failed (model) / failed (infrastructure) / not_comparable | 421 / 413 / 4 / 2 / 13 / **0** / 39 |
| Model generation failures | 0 |
| easy (214) / medium (380) / hard (148) / extra (150) | 154 (71.96%, CI 65.6–77.6) / 225 (59.21%, 54.2–64.0) / 27 (18.24%, 12.9–25.2) / 15 (10.0%, 6.2–15.8) |
| Latency mean / median / p95 (s) | 10.43 / 9.42 / 19.09 (max 49.49), per item, amortised over the batch |
| Total generation time | 9,512.2 s (session 1: 6,624.8 s; session 2: 2,887.4 s) on one Tesla T4, bf16 |
| Evaluation time (local scoring) | 139.1 s; Atlas exec median 16 ms, p95 93 ms, max 30,047 ms |
| Tokens in / out | 1,109,483 / 95,891 |
| Secondary (BSON-normalised) correct | 421 / 892 (47.2%), identical to the primary |

**Failure detail:**
- `incorrect` 413: wrong_row_count 230, wrong_column_count 111, wrong_values 71, order_mismatch 1.
- `invalid` 4: format_error 3, unsafe_mql 1.
- `failed` (model) 13: execution_error 10, plus 3 recorded as `atlas_tier_restriction`. Inspection shows those 3 are model errors: stages written without the `$` prefix (`{"match": …}`, `{"project": …}`), which Atlas rejects with AtlasError 8000. They are correctly attributed to the model; only the detail label is misleading.
- `not_comparable` 39: every one is a `tie_at_limit` mismatch, where the gold `ORDER BY … LIMIT` cuts through a tie.

**By generation path** (descriptive, not a controlled comparison: the fallback items are concentrated on the long-schema DBs `dog_kennels` and `student_transcripts_tracking`):
- session 1, batch 8: 348 / 716 correct;
- session 2, batch 8: 34 / 80;
- batch-1 fallback: 39 / 96.

**Reporting notes:**
- `results_report.md` and `baseline_summary.json` show `generation_wall_s` = 2,887.4 s. That is the resumed session only; the total is 9,512.2 s.
- Mean latency mixes per-item latency amortised over batches of 8 with single-prompt latency for the 96 fallback items.

**Key findings for the presentation:**
1. A 3B open model, zero-shot, answers 47.2% (CI 43.9–50.5%) of 892 Spider dev questions correctly as executable MongoDB queries, judged by result equivalence against gold SQL.
2. Syntax and safety are essentially solved at this scale: 99.7% well-formed and 97.9% executable. The remaining gap is semantic: the queries run but return the wrong rows (230), the wrong columns (111) or the wrong values (71).
3. Accuracy falls steeply with query complexity: 72% easy, 59% medium, 18% hard and 10% extra. The headroom for fine-tuning and RAG is in hard and extra-hard questions.
4. The pipeline is leakage-safe and reproducible:
   - the held-out test DBs are excluded;
   - Kaggle generated and Windows scored as a read-only user;
   - every artifact is hash-verified and every verdict re-scored offline.

## 9. Limitations and comparability

- The limitation quoted at the top of this document applies to every number.
- **Not comparable with:**
  - MQL-0001/0002 (different items, CPU GGUF runtime);
  - SQL-0001 (SQL-only);
  - published "full Spider dev" figures (3 of 20 dev databases are excluded here).
- **Unaudited items.** 468 items were never manually audited; the earlier audit removed 7.7% (38/493).
- **Empty golds.** 49 of the 1,034 source records have empty gold results. A wrong query that returns nothing can match one.
- **Contamination.** Spider dev has been public since 2018, so the model may have seen it in pre-training.
- **Single model.** This experiment supports no claim about relative model quality.

## 10. Reproducibility

- **Bundle:** `scripts/nl2mql/prepare_spider_dev_1034_bundle.py` builds `kaggle/nl2mql_spider_dev_1034_bundle_v2.zip`; it is unchanged and is checked with `--verify`.
- **Notebook:** `scripts/nl2mql/make_dev1034_notebook.py` builds `notebooks/kaggle/nl2mql_spider_dev_1034_model_comparison.ipynb`. The notebook is Qwen-only and generate-only, and writes `nl2mql_spider_dev_predictions.zip`.
- **Shared code:** `scripts/nl2mql/dev1034_core.py`, shipped in the bundle.
- **Local preflight, scoring and verification:** `scripts/nl2mql/local_atlas_scoring.py` (`preflight`, `score`, `verify`).
- **Offline verifier:** `scripts/nl2mql/verify_kaggle_spider_dev_1034.py` (`predictions`, `scoring`).
- **Results:** `outputs/nl2mql_spider_dev_1034_model_comparison/v1/{predictions,scoring}/`, plus `scoring/results_report.md`.
- **Tests:** `tests/test_nl2mql_dev1034.py`, 62 tests. The main ones:
  - the notebook contains no Groq, API, secret or Atlas code;
  - a Qwen-only end-to-end run covers generation errors → partial, a crash → partial, and a resume → complete;
  - predictions verification catches tampering, an extra model and duplicate or missing ids;
  - local scoring with a fake Atlas handles an infrastructure retry and never re-scores or overwrites;
  - partial or unverified bundles are refused before Atlas is touched;
  - the preflight is mocked, and the URI is never printed;
  - the Kaggle preflight blocks model loading;
  - the held-out exclusion never opens the partition file.
