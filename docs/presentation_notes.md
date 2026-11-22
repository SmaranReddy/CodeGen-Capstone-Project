# Presentation notes: experiment log

This is a running record of experiments and results for the final presentation. It was started on 2026-10-27.
Each entry gives the date, what was done, the numbers with their denominators, and the source file.
Results that have not been run yet are marked **PENDING**. Do not quote a number from this file unless it has a source.

## Project framing (as of 2026-10-27)

- Primary task: **NL→SQL** on Spider 1.0. NL→MQL is secondary.
- Dataset: **spider-sql-v1** (`docs/spider_sql_v1_dataset.md`).

  | Split | Records | DBs | Note |
  |---|---:|---:|---|
  | train | 7,773 | 130 | |
  | val_internal | 797 | 16 | DB-disjoint |
  | dev_full | 1,034 | 20 | headline benchmark |
  | test_full | 2,147 | 40 | used once, at the end |

- Train was filtered from 8,659 records; 89 were rejected:
  - 40 questions leaked from dev/test;
  - 38 gold-label defects;
  - 8 exact duplicates;
  - 3 gold SQL execution errors.

**Correction 2026-10-29:**
- The project's target task is **NL→MQL**. The framing above is kept as a record of 2026-10-27; see `docs/target_language_scope_review.md`.
- Every number in §1–§1c is an **NL→SQL result only**. Never present it as MQL performance, and never show it next to the MQL baselines (for example, 3B SQL EX 446/1,034 against 3B MQL 217/455) as a comparison.
- The MQL baselines live in `docs/spider1_expanded_benchmark.md` and `docs/week3_rag_report.md`.

## 1. Zero-shot baseline: Qwen2.5-Coder-3B-Instruct on Spider dev (1,034)

Guide: `docs/kaggle_spider_sql_v1_baseline.md`. Status: **run on Kaggle on 2026-10-28 and verified locally** (ledger entry SQL-0001 in `docs/experiment_accuracy_ledger.md`).

**Setup**

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-3B-Instruct` @ HF `488639f1`, no fine-tuning |
| Hardware | Kaggle T4, fp16 |
| Decoding | Greedy, `max_new_tokens` 256, batch 8 |
| Prompt | `spider-sql-zs-v1`: question + CREATE TABLE DDL only, no examples, no sample rows |
| Scoring | Official Spider `evaluation.py` (taoyds/spider @ b7b5b8c): execution accuracy and exact-set match |
| Secondary scoring | Parse/compile/execute rates, an unofficial result-set match |

**Local validation (2026-10-27)**

- Gold SQL fed through the same scoring path as if it were the model's output scores **1,034/1,034** on every metric. The official exact-match printout agrees.
- One finding worth a slide: **the official Spider evaluator fails on 2 dev examples under Python 3, even when the prediction is the gold query.**
  - Cause: `wta_1` contains non-UTF-8 bytes (`spider_dev_0455`, `0456`).
  - Without a fix, the EX ceiling would be 1,032/1,034.
  - Fix: a lenient UTF-8 decoder, the same one the dataset builder uses. It is disclosed in the report.
- The slowest gold query takes 1.24 s (`wta_1`). Predictions get a 30 s limit and a read-only connection: the original evaluator opens the DB read-write and has no time limit.
- 229 tests pass.

**Results (Kaggle, 2026-10-28; verified locally with the `verify` CLI, `"problems": []`)**

Source: `outputs/spider_sql_v1_zs_3b/v1/spider_sql_v1_zs_qwen25coder3b_v1/baseline_summary.json`

| Metric (denominator 1,034) | Value |
|---|---|
| Execution accuracy (official) | **446/1,034 (43.1%)** |
| Exact-set match (official) | **378/1,034 (36.6%)** |
| Official parser success | 573/1,034 (55.4%) |
| SQLite compile success (EXPLAIN) | 913/1,034 (88.3%) |
| SQLite execution success | 913/1,034 (88.3%) |
| Result-set match (unofficial) | 654/1,034 (63.2%) |
| EX by hardness: easy 248 / medium 446 / hard 174 / extra 166 | 188 (75.8%) / 172 (38.6%) / 62 (35.6%) / 24 (14.5%) |
| Generation failures / timeouts / truncations | 0 / 0 / 0 |
| Mean latency per example; total generation time | 0.5518 s; 570.56 s in 130 batches (1× T4) |

One finding worth a slide: 461 predictions fail the official parser, so they score 0 on EX and EM. 185 of those still return the gold result. This is why the unofficial result-set match (63.2%) is well above official EX (43.1%).

Caveat for slides: official EX at b7b5b8c is **not** the test-suite EX that most papers report. Compare with published Qwen2.5-Coder numbers only with that stated.

### 1b. Parser-rejection audit of the baseline (2026-10-28, diagnostic)

Source: `docs/sql_v1_zero_shot_parser_audit.md`

- **The local re-score reproduces the Kaggle run exactly.** All 6 metrics and all 1,034 per-example records match.
- **None of the 461 parser rejections is a syntax error.**
  - 117 are invalid SQL for the schema; SQLite rejects them too.
  - 344 are valid SQLite that the 2019 official parser cannot read: `INNER JOIN` (217), column or table aliases (94), multi-literal `IN` lists (26), others (7).
- **Slide-worthy finding: the official metric depends on SQL style.** Rewrites that change only the surface form, and leave the result rows unchanged, would raise official EX from 446 to 604 and EM from 378 to 498. These are diagnostic numbers, not results. A fine-tuned model will pick up Spider style, so its official EX gain must be shown next to a parser-independent execution metric.
- **The 185 rejected-but-matching predictions are not shown to be correct.**
  - 24 of them return different rows from gold on the test-suite variants of the dev DBs.
  - For comparison, 36 of the 446 official EX hits are distinguishable in the same way.

### 1c. Test-suite execution accuracy of the baseline (2026-10-28, secondary metric)

Source: `docs/sql_v1_test_suite_eval.md`; `outputs/spider_sql_v1_zs_3b/v1/testsuite_eval_v1/predictions/testsuite_summary.json`

- **Test-suite EX is 630/1,034 (60.9%)**, against official EX of 446/1,034 (43.1%). Official EX and EM are unchanged and remain primary.
  - Method: the official Spider test-suite method (Zhong et al. 2020), code @ e97acc5 unmodified, default settings.
  - Each prediction must match gold on the original DB **and** on all its distilled variants: 695 DB files across the 20 dev DBs, 25–60 per DB.
  - With `--keep_distinct` the score is 603/1,034 (58.3%).
- By hardness: easy 208/248 (83.9%), medium 286/446 (64.1%), hard 86/174 (49.4%), extra 50/166 (30.1%).
- **Slide-worthy:** the two metrics disagree on 248 examples.
  - 216 pass the test suite but score 0 on official EX. Of these, 177 were rejected by the 2019 parser.
  - 32 pass official EX but fail the test suite. 31 of them are coincidental matches on the single original DB. Example: `spider_dev_0208` hard-codes the airport code `'ABE'` instead of joining on the city name.
- Sanity checks:
  - gold as the prediction scores 1,034/1,034;
  - 0 examples are not evaluable;
  - a re-run is byte-identical.
- **For the fine-tuning comparison,** report official EX, official EM, test-suite EX (plus `keep_distinct`), parser acceptance and SQLite execution side by side.

## Earlier work (pointers only; numbers to be filled from the source documents)

- Week 2: NL→MQL model comparison (`docs/week2_report.md`, `docs/spider1_model_comparison.md`)
- Week 3: RAG for NL→MQL (`docs/week3_rag_report.md`)
- Dataset research and audits: `docs/dataset_research_report.md`, `docs/sql2mql_independent_audit.md`, `docs/current_dataset_inventory.md`

**Added 2026-10-29: NL→MQL numbers (primary task), back-filled into the ledger as MQL-0001 to MQL-0005.** No new runs; counts recomputed from the saved outputs.
- MQL-0001, v1 279, zero-shot, Atlas execution: Qwen-3B 132/279 (47.3%), Qwen-7B 181/279 (64.9%), gpt-oss-120b 232/279 (83.2%).
- MQL-0002, expanded 455, zero-shot: Qwen-3B 217/455 (47.7%), Qwen-7B 287/455 (63.1%), gpt-oss-120b 385/455 (84.6%).
- MQL-0003, RAG, validation 72, CPU: zs_rerun 33, random_k3 37, hybrid_k3 43 (Holm 0.16, not significant).
- MQL-0005, clean RAG control, train 318, Kaggle T4: zero-shot 160, random_k3 155, hybrid_k3 172; hybrid vs random +5.3, p 0.021, Holm 0.064 (not significant).
- gpt-oss-120b is open-weight (Groq), not a closed frontier model. Roadmap: `docs/nl2mql_next_steps.md`.

## 2. MQL-0006: Qwen-3B vs Groq gpt-oss-120b, zero-shot NL→MQL on all 1,034 Spider dev questions — PENDING

Status: **PREPARED — RESULTS PENDING** (prepared 2026-10-29/30; nothing has been run). Write-up: `docs/nl2mql_spider_dev_1034_model_comparison.md`; run guide: `docs/kaggle_nl2mql_spider_dev_1034.md`; ledger entry MQL-0006 (planned).

- **What it measures:** generated MQL is executed on Atlas and its result is compared with the Spider gold SQL result on SQLite, over all 1,034 dev questions (not the curated 455). The metric is cross-engine execution equivalence; it is not native MQL exact-match accuracy.
- **The two models:**
  - Qwen2.5-Coder-3B @ `488639f1`, run with transformers on a Kaggle T4 GPU, greedy decoding;
  - `openai/gpt-oss-120b` on Groq. It is **open-weight**; the provider was chosen by the user.
- **What is shared:** both models get the same prompt `nl-mql-spider1-v1`, the same safety gate and the same comparator.
- **Preparation findings worth a slide:**
  - All 1,034 gold SQL results reproduce their recorded hashes.
  - 44 gold answers are not unique: a LIMIT cuts through a tie. A mismatch on these counts as `not_comparable`, never as incorrect.
  - 0 of the 62 wta_1 items depend on its 2 non-UTF-8 strings.
  - 541 of the 1,034 items were never manually audited. The earlier audit removed 38 of 493 items (7.7%).
- **Results:** PENDING. Report correct/1,034 with a Wilson CI, outcome buckets, results by hardness, latency, tokens and cost, and the paired difference with exact McNemar and a bootstrap CI.
- **Do not compare** these numbers with MQL-0001/0002 (different items, and CPU GGUF Qwen) or with SQL-0001 (SQL-only).

**Correction 2026-10-30 (before any run).** MQL-0006 now runs on **892 items / 17 DBs**: Spider dev minus all 142 records on the 3 databases of the held-out MQL test partition (`employee_hire_evaluation`, `pets_1`, `tvshow`).
- The databases were identified from the ID-only split manifest; the test file was never opened.
- Excluding only the 65 overlapping questions would have kept 65 same-SQL paraphrases.
- New counts: reference ok 846, tie_at_limit 44; 468 items unaudited.
- The 1,034 figures in the section above are superseded for this experiment.
- Atlas network access for Kaggle is pending user approval.
- gpt-oss-120b remains described as **open-weight, Groq-hosted**, not a closed frontier model.

**Update 2026-10-30 (b): architecture.**
- Kaggle now only **generates** predictions:
  - Qwen-3B on the GPU;
  - Groq-hosted open-weight gpt-oss-120b, which is not a closed frontier model.
- **Scoring runs on Windows** against Atlas, using the read-only account (`scripts/nl2mql/local_atlas_scoring.py`):
  - an Atlas preflight runs before any GPU or API use;
  - the predictions bundle is verified before scoring;
  - the scoring outputs are verified offline afterwards.
- Kaggle never touches Atlas, so no Atlas network-access change is needed.
- The metric is still **cross-engine execution accuracy** on 892 items. It is not native MQL exact match. Results are PENDING.

**Update 2026-10-30 (c): MQL-0006 is now Qwen-only.**
- **What runs:** zero-shot Qwen2.5-Coder-3B-Instruct on the leakage-safe 892-item set (17 DBs, 70 collections). Kaggle only generates (GPU, no API, no secrets, no Atlas), and Windows scores against Atlas as the read-only user.
- **Deferred:** the Groq-hosted open-weight gpt-oss-120b comparison moves to a later experiment.
- **Metric:** cross-engine execution accuracy (correct/892, Wilson 95% CI, outcome buckets, hardness, latency). It is not native MQL exact match.
- **Results:** PENDING.
- **Superseded:** the two-model items above, for this experiment.

**Update 2026-10-30 (d): MQL-0006 first Kaggle run was partial; fix ready.**
- **Partial run:** 716 of 892 predictions were generated. The other 176 ran out of GPU memory on the T4, in batches of 8 with long schema prompts (mostly `dog_kennels` and `student_transcripts_tracking`). Nothing has been scored, and results are still PENDING.
- **Fix:** an out-of-memory batch is now retried one prompt at a time, with the same prompt, length limit, precision and greedy decoding. The resumed run generates only the 176 missing items and keeps the 716 unchanged.
- **Disclosure for the slides:** the number of predictions made at batch size 1. Batch-size differences can cause minor numerical variation even with deterministic decoding.
- **Correction:** the model ran in **bfloat16**, not fp16 (Kaggle's T4 reports bf16 support, and the notebook picks bf16 automatically).

**Update 2026-10-30 (e): MQL-0006 RESULT, verified.** Zero-shot Qwen2.5-Coder-3B-Instruct (HF `488639f1…`) on 892 Spider dev questions (17 DBs, 70 Atlas collections; the held-out test DBs are excluded).

| Slide item | Value |
|---|---|
| **Cross-engine execution accuracy** (primary; not native MQL exact match) | **421 / 892 = 47.2%**, Wilson 95% CI 43.9–50.5% |
| Valid MQL / passes read-only gate / executes on Atlas | 99.66% / 99.55% / 97.87% |
| correct / incorrect / invalid / timeout / failed (model) / failed (infra) / not comparable | 421 / 413 / 4 / 2 / 13 / 0 / 39 |
| easy / medium / hard / extra | 72.0% / 59.2% / 18.2% / 10.0% |
| Hardware / time | Kaggle Tesla T4, bf16; generation 9,512 s over two sessions (96 items at batch size 1 after OOM); scoring 139 s locally |

- **Talking points:**
  - Syntax and safety are near-solved (97.9% executes); errors are semantic: wrong rows 230, wrong columns 111, wrong values 71.
  - Accuracy collapses on hard and extra-hard questions, which is where fine-tuning and RAG must help.
- **Must say on the slide:** the metric measures cross-engine execution equivalence between generated MQL on MongoDB Atlas and Spider gold SQL on SQLite. It is not native MQL exact-match accuracy. Results depend on the fidelity of the database mapping and result-normalization procedure. Matching on one database instance does not prove general semantic correctness, and the gold SQL labels have not been independently human-verified for every question.
- **Do not compare directly** with the 279 (MQL-0001, 47.3%) or 455 (MQL-0002, 47.7%) results: different items, curation and runtime (4-bit CPU GGUF). Do not put SQL-0001 beside it.
- **Artifacts:** `outputs/nl2mql_spider_dev_1034_model_comparison/v1/` (predictions, scoring, both verification files).

## 3. Planned MQL-0007 (RAG + Qwen-3B on the 892 items): BLOCKED, 2026-10-30

Status: **BLOCKED, NOT PREPARED.** Nothing was built or run. Write-up: `docs/nl2mql_spider_dev_1034_rag_experiment.md`.

- **Slide-worthy finding:** the Week 3 retrieval corpus (280 examples) is made entirely of Spider-dev items. The 892-item set *is* Spider dev, so all 280 examples are evaluation questions.
  - The 280 Spider rows of ft-v1 are the same items. Fine-tuning on ft-v1 and then scoring on the 892 would be contaminated.
- **Label quality:** the other 1,584 ft-v1 rows carry unremediated sql2mql-v1 labels (88 known-wrong), so no reliable training pairs remain. Eligible supervised pairs: 0 of 2,144.
- **Why the earlier RAG results were fine:** they used leave-one-database-out *within* Spider dev, on a different item set (318). That design does not transfer, and those numbers are not comparable to the 892.
- **Unblock options:**
  - sql2mql-v2 plus Spider-train conversion;
  - re-executing AptMQL/SM3;
  - docs-only RAG (a new configuration).

## 4. sql2mql-v2 converter validation (Phase 1), 2026-10-30: NOT PASSED

Write-up: `docs/sql2mql_v2_quality_assessment.md`. All measurements are local (SQLite + MongoDB 8.0.34 in Docker), not Atlas.

- **What was built:** a conservative SQLite-semantics SQL→MQL converter, plus an independent differential verifier. The verifier runs the original SQL on SQLite and the MQL on MongoDB, on the real database and on 24 paired adversarial fixture families.
- **Slide-worthy results:**
  - 0 false accepts in a 139-case adversarial suite;
  - every v1 bug class covered (26/26 matrix);
  - 72% of a 500-item Spider-train dev sample converted and verified, with 0 mismatches.
- **The honest headline:** a mutant study, which feeds deliberately wrong MQL to the verifier, found it still accepts some wrong translations on multi-join queries.
  - 13.5% of labels in iteration 1;
  - 4.35% on a pre-registered held-out set in iteration 2 (CI up to 8.7%);
  - this misses our frozen ≤ 5% bar, so no training data has been produced yet.
- **Method point worth a slide:** the acceptance criteria and the pilot sample were frozen before any code existed, and the second measurement was pre-registered on labels never used for debugging.

## 5. sql2mql-v2 verifier-3 and round-3 G4 (2026-10-31): PASSED

Write-up: `docs/sql2mql_v2_round3_report.md`. All measurements are local (SQLite + MongoDB 8.0.34 in Docker), not Atlas.

- **What changed:** a new verifier version (verifier-3) that keeps every earlier check and adds targeted test data:
  - reordered rows (catches a dropped sort before `LIMIT 1`);
  - in-place unmatched join keys (catches inner → left join on multi-table chains);
  - "all other conditions true" boundaries (catches `> 2000` vs `> 2001` inside an AND);
  - a lookup-field existence check.
- **Headline:** on a new pre-registered sample of 740 Spider-train records (542 accepted labels), the verifier misses a deliberately wrong translation on an estimated **2.4%** of labels (95% CI 1.4–4.1%). That is under the ≤ 5% bar, and the unweighted figure passes too (upper bound 4.5%), so Phase 1 passes. The bar was frozen in advance; the population and the weighting were pre-registered for this round.
- **Honest caveat:** three-or-more-table joins are still the weak spot, at 7 of 104 labels (CI up to 13%). The overall figure is population-weighted, because the hard queries were deliberately oversampled.
- **Method slide:**
  - sample, strata, weights and pass threshold were frozen before the new verifier existed, and the code was frozen before the run;
  - one shot, with no tuning on the sample;
  - an independent recompute matched every figure;
  - checks before the freeze (0/200 in-sample) proved over-optimistic compared with the held-out 2.4%.
- **Next:** the 300-example pilot with an independent human review. It has not started yet.

## 6. Independent audit of `spider_pool.zip` (2026-11-05): CONDITIONAL PASS

Write-up: `docs/spider_pool_independent_audit.md`. SQL-side dataset audit, measured locally; not an NL→MQL model result.

- **What it is:** an externally built "cleaned Spider" package meant as the source for SQL→MQL conversion.
- **Headline:** the training file is mechanically excellent: 6,258 / 6,258 train and validation queries run on the official databases and reproduce their recorded answers. Executing is not the same as being right, though: 57 confirmed label defects survive, such as answers exactly 5× too large from a join fan-out.
- **The real problem:** the package also contains all 1,034 Spider dev records, including gold SQL for all 65 held-out test items. The train side has 0 overlap with them, but the zip cannot be used as a whole.
- **Provenance:** every input hash and every training row traces back to Spider, but the cleaning script was not supplied, so the process cannot be reproduced.
- **Method slide:** dev and test fields discarded at parse time; ID-level leakage checks only; rejection rules replayed independently; scan-order reversal and nested tie probes found what a plain re-execution cannot.
- **Next:** decide how the frozen 300-example pilot relates to this pool (276 of the 300 are in it). The pilot still has not started.

## 7. Reproducibility audit of `spider_pool.zip` (2026-11-05): CONDITIONAL PASS

SQL-side dataset audit only; no model metric. Full report: `docs/spider_pool_reproducibility_audit.md`.

- **Headline:** the cleaned data is reproducible. Running the original script gave byte-identical data files, with 0 record mismatches in 9,604.
- **Environment matters:** that needs Linux line endings and a recent SQLite. On SQLite 3.40.1, 4 extra rows are kept (5,518 train); the zip file itself has no build script.
- **Reproducible is not clean:** all 57 confirmed defects, 37 row-choice rows and 42 review rows are still in the output, because the script has no logic that could catch them.
- **The 5,430 / 734 figures:** these are Tier A and B removed, Tier C kept. They are not files that exist yet.
- **Leakage:** train and val hold no protected IDs, but the output still contains all 1,034 dev records by design, so it must never be loaded whole.
- **Status:** usable as a SQL source through `train.jsonl` after exclusions. The pilot still has not started.

## 8. spider-pool-v2 prepared (2026-11-06)

SQL-side dataset preparation; no model metric. Report: `docs/spider_pool_v2_preparation_report.md`.

- **Result:** 5,387 training and 734 validation examples. 94 known-defective rows were removed and 43 held back for a human decision.
- **Nothing rewritten:** every kept example is copied byte for byte from the audited package, and every removal names its reason and source line.
- **Safety:** 0 development, test or held-out IDs; train and validation share no database. 38 automated checks pass, and deliberately corrupted copies were caught.
- **Caveat for the slide:** kept does not mean correct. The labels still have to pass conversion verification and human review.

## 9. sql2mql-v2 300-example pilot with verifier-3 (2026-11-06): run, gate pending human review

Label validation on local MongoDB 8.0.34, not a model metric. Report: `docs/sql2mql_v2_pilot_v3_report.md`.

- **Result:** verifier-3 accepted 213 / 300 frozen pilot examples (71.0%). 76 were outside the converter's scope; 11 had ORDER BY … LIMIT ties with no unique answer. 0 semantic mismatches, 0 timeouts, 0 crashes.
- **Difficulty gradient:** easy 96% → medium 89% → hard 47% → extra 20%. Subqueries and set operations are not converted at all.
- **Integrity:** a new runner refuses to start unless all 19 frozen code hashes match, and it has no verifier-2 fallback. It was frozen before touching any pilot item. A second execution reproduced every outcome.
- **Caution:** accepted is not proven correct. Three-or-more-table joins (20 accepted) are held out of any corpus, and the pilot gate still needs an independent human review of 120 labels.

## 10. AI cross-check of the 120 pilot review labels (2026-11-06): 6 wrong labels found

AI review (Claude Opus 5.5), not the human review the pilot gate needs. Local SQLite + MongoDB. Report: `outputs/sql2mql_v2/pilot_v3/run01/cross_review/cross_review_report.md`.

- **Result:** 111 / 120 correct, 6 incorrect, 3 uncertain. Re-running every SQL and MQL query in full gave identical answers on all 120: the SQL→MQL conversion made no errors.
- **The errors are in Spider's original SQL:** e.g. 'oldest first' sorted on dates stored as DD.MM.YYYY text, 'most memberships' sorted as text ('91' > '1418'), a teacher's first and last name swapped (answer 0 instead of 7), 'most recent' sorted ascending.
- **Why it matters:** the verifier compares SQL with MQL, so a wrong SQL label passes; only reading schemas and data catches it. The earlier AI review missed 4 of the 6.
- **Caution:** still needs a named human reviewer before any corpus is built.

## 11. Semantic audit of the 5,387 retained Spider training labels (2026-11-06)

SQL-side dataset audit; no model metric. AI detectors and AI review, not human. Report: `docs/spider_pool_semantic_audit.md`.

- **Result:** 30 confirmed wrong labels (0.56%), 58 likely (1.08%), 254 need review (4.72%), 5,045 with nothing detected (93.65%).
- **What kind of errors:** wrong sort direction for "highest/most expensive" (16), boundary operators like "at least" written as `>` (6), question and SQL value disagree or the question is cut off (6), numbers stored as text and sorted alphabetically (39 likely).
- **Honest limit:** the checks found only 24% of already-known-bad rows at likely-or-above level, so "nothing detected" is not "correct". The earlier human-style sample (5% wrong) is the better guide.
- **Decision for the slide:** do not approve a final training corpus yet. Drop the 30, hold the 312 flagged, then get a human reviewer on a fresh random sample.

## 12. Provisional RAG corpus v1 built (2026-11-06)

Label-validation on local MongoDB 8.0.34, not a model metric. Report: `outputs/sql2mql_v2/provisional_corpus_v1/conversion_report.md`.

- **Result:** of 5,045 audit-clean Spider training labels, the frozen converter handled 3,962 (78.5%) and verifier-3 accepted all of them. 1,083 are unsupported (mostly subqueries and set operations). After holding out 3+-table joins and a few other flags, the corpus has 3,589 entries.
- **Difficulty gradient:** easy 98.7% accepted, medium 90.4%, hard 54.7%, extra 36.7%.
- **Integrity:** one frozen pipeline for every record, 19 code hashes checked first, no leakage into any evaluation file, every entry marked PROVISIONAL and NOT_REVIEWED.
- **Caution for the slide:** verifier-3 proves SQL and MQL agree, not that the SQL answers the question. The pilot found about 5% wrong source labels, so expect a similar rate. The corpus is biased toward easy and medium queries.

## 2026-11-06: MQL-0007 preflight (no experiment run yet)
- Built a multi-arm local scorer (A zero-shot, B random, C relevant RAG + 1-item sensitivity arm), pre-registered the statistics (C vs A, C vs B, Holm; Wilson; McNemar) before any prediction exists.
- One near-duplicate pair (spider_train_00650 vs two dev questions) is excluded from retrieval for the two items; only 1 item's prompt actually changes.
- Zero-shot reuse is conditional on the Kaggle environment equalling MQL-0006's; otherwise arm A is regenerated.

## 13. Multi-agent AI audit of the 120 pilot review labels (2026-11-12)

Four independent AI reviewers checked the 120 labels: a source-SQL auditor, a translation auditor, an adversarial auditor, and an adjudicator who ruled on the 32 disputed items with fresh evidence and no majority vote. They ran on local SQLite and MongoDB. This is not the human gate. Report: `outputs/sql2mql_v2/pilot_v3/run01/manual_human_review/ai_multiagent_review/consensus_report.md`.

- **Source SQL wrong for the question:** 10 / 120 (8.3%). Four are certain: dates stored as DD.MM.YYYY text, amounts sorted as text, first and last name swapped, "most recent" sorted ascending. Six depend on how the question is read, for example "all X" questions that miss groups with no matching rows.
- **Translation:**
  - On clean data (values match their column types), 118 / 120 MQL queries match the SQL. The two exceptions, R040 and R082, involve a Unicode case-folding edge case in `$regex`.
  - If numeric columns can hold text, 18 more diverge: `$gt` has no type guard, and `$avg`/`$sum` ignore strings.
  - None of these 20 shows up on the real data.
- **Valid training pairs:** 108 / 120 on clean data; 90 / 120 under the strict any-value standard.
- **Why it matters:** the earlier dataset cleaning is not enough. Wrong Spider gold SQL passes the verifier, which compares SQL with MQL and never SQL with the question. The project also needs to decide which data domain the MQL labels must be correct on.
- **Caution:** all four reviewers are the same model and saw one line summarising the earlier review, so their agreement is weaker evidence than it looks. A human review is still required.

## 14. Plan for the next converter and dataset revision (2026-11-12)

This is a plan only; nothing has been implemented. Document: `docs/pilot_v3_followup_implementation_plan.md`. The target contract is the type-conformant domain.

- **Contract (TC-1):**
  - every field holds one stored type, either numeric or text, and that type is frozen in a manifest;
  - Atlas must be re-checked against the manifest at every scoring run.
  - The rule is based on the stored type, not the declared type: Spider keeps 195 date columns as text, which is harmless.
- **LIKE fix:**
  - write `[kK]`-style character classes instead of using the regex case-insensitive flag;
  - this was tested locally and matches SQLite exactly, including the Kelvin-sign and long-s traps;
  - no current result changes, because no training database contains those characters.
- **Source labels:**
  - quarantine or human-adjudicate all 10 findings and their paraphrase twins;
  - never auto-rewrite gold SQL;
  - six targeted detectors (text-stored numbers, sort direction, "all X", entity counts, truncated questions, unsupported literals) flag records for human review only.
- **Versions:** converter sql2mql-v2.1, verifier 3.1, spider-pool-v3 and provisional corpus v2. All are new paths; nothing frozen is edited.
- **Gates:** source-label accuracy and translation correctness are reported separately, from a fresh human-reviewed sample.
- **Caution:** the run01 human gate (P2) is still pending.

## 15. Converter v2.1, AI review panel and the new dataset versions (2026-11-13 to 2026-11-16)

The plan in section 14 is implemented. Nothing frozen was changed (217 file hashes verified). No model was trained and no RAG experiment was run.

- **Converter sql2mql-v2.1 and verifier 3.1:**
  - LIKE is now written as `[kK]`-style character classes;
  - on the 300 pilot labels the accepted set is unchanged (213), all 205 non-LIKE labels are byte-identical, and the 8 LIKE labels differ only in the regex.
- **Atlas preflight:** the type contract holds on Atlas (70 collections, 0 violations) and the LIKE fix matches SQLite there. No label has been run on Atlas yet.
- **AI review panel (in place of the human review):**
  - four reviewer roles in isolated contexts; every disagreement adjudicated with fresh execution evidence; no majority vote;
  - calibration: 60 / 60 planted source errors and 60 / 60 planted translation errors caught, 0 / 60 clean controls flagged;
  - repeat run: 113 / 116 source and 98 / 98 translation verdicts agree.
- **G-AIR gate (200 fresh accepted labels):**
  - translation: 0 errors of 200, 95% upper bound 1.9%, bound 5%; passed;
  - source labels: 7 of 200 flagged (6 wrong, 1 unsure); reported, no verdict.
- **spider-pool-v3:** train 5,387 -> 4,790; validation 734 -> 639. 26 records removed as panel-confirmed defects, 667 held for review as detector hits. No gold SQL was rewritten.
- **Provisional RAG corpus v2:** 3,447 records (v1 had 3,589); 3,962 of 5,045 candidates converted and verified locally.
- **Cautions:**
  - all of this is AI-reviewed, not human-reviewed. The run01 human gate (P2) is still not passed;
  - the reviewers are the same model family, so their agreement does not establish true accuracy, and 200 items do not prove an error rate below 5%;
  - the detectors are imprecise (about 15-20% of sampled hits confirmed for most), so most of the 667 held records are probably fine;
  - corpus v2 was verified on local MongoDB only; the Atlas check is pending;
  - 3 of 581 tests fail, none on converter or verifier behaviour (ledger DA-2026-11-16-CORPUS-V2).

## 16. Pre-training sign-off of corpus v2 (2026-11-16)

Result: **blocked**. Fine-tuning has not started and is not authorized yet. Report: `docs/corpus_v2_signoff_report_2026-11-16.md`.

- **Why it is blocked:**
  - the corpus labels query 516 collections from 123 Spider training databases;
  - Atlas holds only the 80 evaluation collections, and the project account is read-only;
  - so none of the 3,447 labels could be executed on Atlas. All label verification so far is on local MongoDB.
- **What was checked on live Atlas:** the account is read-only, and the evaluation collections still satisfy the type contract (70 collections, 0 violations).
- **Test suite:** 591 of 591 pass (was 578 of 581). The three failures were test and tooling faults, not converter or verifier faults.
- **Six records on mixed-type fields:** 2 removed (they group on the field), 4 kept (they only output it).
- **Candidate corpus:** 3,445 records, marked not approved.
- **Review protocol:** the AI-only gate (G-AIR) is the review gate for this dataset version. The human review gate (P2) is still not passed.
- **Next decision:** load the training databases into a scratch Atlas database (needs write access), or accept local verification for training labels and record that as a limitation.

## 17. Training corpus approved under the local validation policy (2026-11-16)

Status: **conditional go**. Fine-tuning may begin; it has not been started.

- **Owner decision:** training labels are validated on local MongoDB 8.0.34; all reported evaluation metrics must still come from Atlas.
- **Approved corpus:** 3,445 records in `outputs/sql2mql_v2_1/approved_corpus_v2_1/`.
- **Checks:** every record has a complete local verification record; a live re-check of 64 records reproduced all 64; 594 of 594 tests pass.
- **Cautions:**
  - the corpus labels were never executed on Atlas (the training collections are not there);
  - the data is AI-reviewed, not human-reviewed, and the human review gate (P2) is not passed;
  - the 7 of 200 flagged source labels are a sample observation, not a corpus-wide error rate.

## 18. MQL-0008 package: zero-shot vs hybrid RAG on the approved corpus (2026-11-16)

Status: **package built, not yet executed**. There is no MQL-0008 result.

- **Question:** does RAG built from the approved 3,445-record corpus beat zero-shot on the 892 questions?
- **Design:** two arms only, same model and settings as MQL-0006 and MQL-0007; the retrieval settings are the frozen MQL-0007 ones, only the corpus changes.
- **Leakage audit, repeated for the new corpus:** no exact or normalised question match and no shared database; the same two template near-duplicates as before, excluded from retrieval for those two questions.
- **Validation:** 594 of 594 project tests pass; one notebook resume test still has to be re-run (the test process was stopped for low memory).
- **Next:** run the notebook on Kaggle, then score locally against Atlas (a two-arm scoring adapter is still to be written).
- **Historical only:** MQL-0006 zero-shot 421/892; MQL-0007 hybrid RAG on corpus v1 437/892.
- **Update (2026-11-16):** the outstanding notebook resume test was re-run and passes; all 34 package tests pass. The package is ready for the Kaggle run.
- **Update (2026-11-16): scoring is ready.** The two-arm scoring adapter and the analysis plan were written and registered before any prediction exists. One primary comparison (hybrid RAG vs zero-shot, exact McNemar, with Wilson and bootstrap intervals); RAG is called better only if the test is significant. 626 of 626 project tests pass. Still to do: the Kaggle run, then scoring on Atlas.

## 19. MQL-0008 result: RAG on the approved corpus beats zero-shot (2026-11-17)

- **Zero-shot:** 415 of 892 (46.5%). **Hybrid RAG, approved corpus v2.1:** 457 of 892 (51.2%).
- **Difference:** +42 questions, +4.7 points (95% interval +1.9 to +7.6); exact McNemar p = 0.0018, so the gain is statistically significant.
- RAG fixed 108 zero-shot errors and introduced 66.
- **For comparison only:** RAG on the old corpus v1 (MQL-0007) gained 16 questions over zero-shot and was not significant (p = 0.25). That run was scored on another day, so the two are not a controlled comparison.
- **Caveats:** the zero-shot score is 415 here and 421 in earlier runs with identical model output, because 6 tie-dependent queries returned a different tied row on Atlas; the corpus is AI-reviewed, not human-reviewed; RAG produced more execution errors (29 vs 13).
- Metric is cross-engine execution accuracy on Atlas, not exact match. Details: ledger entry MQL-0008.


## 20. First fine-tuning run: QLoRA pilot on Qwen-3B (2026-11-19)

Status: **training done, not yet evaluated**. There is no fine-tuning accuracy number yet. Ledger entry MQL-0009 (provisional).

- **What:** a LoRA adapter for Qwen2.5-Coder-3B (4-bit, Unsloth), trained on the approved corpus v2.1 SFT data: 3,220 training and 225 dev examples, dev on held-out databases.
- **Cost:** 1 epoch, 43.7 minutes on one free Kaggle T4, 2.7 GB GPU memory. Only 0.96% of the model's weights are trained.
- **Dev loss:** 0.72 before, 0.026 after.
- **Before/after on 3 dev questions:** the base model wraps answers in markdown and uses its own query style. The fine-tuned model returns bare JSON in the corpus style: one answer matches gold exactly, one differs only in a field-name capital letter, and the join question is still wrong.
- **Caveat:** lower loss and well-formed JSON show the training pipeline works. They do not show the queries are correct. That needs execution on Atlas with the MQL-0008 scorer.
- **Next:** score the adapter on the 892 questions; try r 8/32, lr 1e-4, 2 epochs and the 7B model; try using both T4s to cut training time.
- Notebook with outputs: `notebooks/kaggle/codegen_finetune.ipynb`.
