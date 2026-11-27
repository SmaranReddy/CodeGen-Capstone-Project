# MQL-0010: experiment plan (registered 2026-11-19, NOT executed)

> **Superseded in part by amendment A4 (2026-11-19).** It adds two arms (EP, D0G), batch-1 delta and retry passes, enforced holdout-view checks, a new test sequence and a practical criterion.
>
> The authoritative execution plan is now `docs/mql0010_final_execution_plan.md`. The governing hashes are in `outputs/nl2mql_mql0010/preflight/registration_index.json`. This document is kept as the original registration record.

**Registration (immutable; amend only with a new dated file)**

| Item | Path | sha256 |
|---|---|---|
| Config | `scripts/nl2mql_mql0010/configs/mql0010_experiment.json` | `1c9e502e…` |
| Analysis plan | `outputs/nl2mql_mql0010/preflight/analysis_plan.json` | `a6294743…` |
| Split and leakage audit | `outputs/nl2mql_mql0010/preflight/split_registration.json` (amendment A1) | `fd0c8e18…` |
| Amendment A2 (dev proxy recomputed on the registered dev split; registration unchanged) | `outputs/nl2mql_mql0010/preflight/amendment_A2_2026-11-19.json` | — |
| **Amendment A3 (test order; supersedes the order in the analysis plan)** | `outputs/nl2mql_mql0010/preflight/amendment_A3_2026-11-19.json` | — |
| Source files | `scripts/nl2mql_mql0010/src/*.py` | hashes pinned in the config; `tests/test_mql0010_planner.py::test_registered_config_is_consistent` fails on drift |

**Not yet done:**
- No prediction, score, Kaggle run or Atlas provisioning exists for MQL-0010.
- No ledger entry was made, because this task restricted ledger updates to MQL-0009.

---

## 1. Question

On new databases, does the improved architecture (§2) beat the unchanged MQL-0008/0009 hybrid RAG in cross-engine execution accuracy, and which component is responsible? The architecture adds:
- **V**, schema-grounded validation with one controlled retry;
- **E**, signature-diversified example selection;
- **F**, notes for type-incompatible join keys.

## 2. Hypotheses (registered; test order as amended by A3)

**Amendment A3 (2026-11-19, before any prediction) is the order that holds.** It is a fixed sequence at family-wise α = 0.05. Each step is an exact two-sided McNemar test with a bootstrap CI:
1. **D0V vs D0** (V alone).
2. If step 1 rejects: **FULL vs D0**.
3. If step 2 rejects: **E vs D0** and **EF vs E**, Holm over the two.

Reason: power. D0V vs D0 discordance comes only from kept retries (MDE ≈ 3 pp), while FULL vs D0 inherits E's prompt changes (MDE ≈ 4.2 pp). A neutral E must not block the test of the best-evidenced component. The table below is the originally registered order, kept for the record.


| ID | Comparison | Test | Status |
|---|---|---|---|
| **H1 (primary)** | FULL vs D0 | exact two-sided McNemar, α = 0.05; paired bootstrap 95% CI over questions (10,000 resamples, seed 20261119) | confirmatory |
| H2 | D0V vs D0 (V alone) | McNemar + CI; **tested only if H1 rejects**; Holm over H2–H4 | gatekept secondary |
| H3 | E vs D0 (E alone) | as H2; **expected to be underpowered** (MDE ≈ 4 pp) | gatekept secondary |
| H4 | EF vs E (F alone) | as H2; prompts differ only on 2 of 18 holdout DBs, so effectively descriptive | gatekept secondary |

**Descriptive only (no claim):**
- A0 vs D0 (replication of the RAG effect);
- A0V vs A0 (recovery without retrieval);
- recovery statistics;
- outcome taxonomy;
- per-family and per-DB accuracy;
- retrieval traces.

**Decision rules (A3):**
- Adopt V if step 1 rejects.
- Adopt the full architecture if steps 1 and 2 reject.
- Adopt E only if step 3 rejects E vs D0; otherwise keep the MQL-0008 selection order.
- F is descriptive unless EF vs E rejects.

## 3. Arms (7 derived from 3 full generation passes)

| Arm | Kind | Definition | Generation |
|---|---|---|---|
| A0 | generate | zero-shot `nl-mql-spider1-v1`, unchanged | full pass |
| A0V | retry | A0's first-pass outputs + V | retries for flagged items only |
| D0 | generate | MQL-0008/0009 hybrid RAG, unchanged (k = 3, `hybrid_masked`, w = 1.0) | full pass |
| D0V | retry | D0's first-pass outputs + V | retries only |
| E | generate | D0 ranking + `select_diverse` (top-50 pool) | full pass |
| EF | prompt_delta | E + F notes; items whose prompt hash equals E's **reuse E's output** | only the changed prompts |
| FULL | retry | EF's first-pass outputs + V | retries only |

**Why reuse.** With greedy decoding the same prompt gives the same output, so reuse is exact.
- Each derived comparison is then attributable: a D0V vs D0 difference can only come from a kept retry, and EF vs E only from a changed prompt.
- Batch-composition noise is removed from those comparisons. In MQL-0009, 28% of items flipped across random seeds and the OOM fallback differed by run.
- Every derived record carries `source: reused:<arm>`. Reuse is verified by prompt hash.

**Random arm.** No random-retrieval arm: MQL-0009 already established hybrid > random on held-out databases (H1, Holm p = 0.008). One random seed would add noise without testing a new hypothesis.

**Fine-tuning** is a separate comparison and not part of MQL-0010.

## 4. Data split and leakage audit (`preflight/make_split.py`)

**Universe.** The 123 databases of approved corpus v2.1. It is disjoint by construction from:
- Spider dev (MQL-0006/0008, 20 DBs);
- the 16 MQL-0009 evaluation DBs.

Both are verified in the script.

**Draws.** Both were made with fixed seeds before any item, question or outcome was inspected:

| Set | Seed | Databases | Items |
|---|---|---|---:|
| **Holdout** | 20261119 | allergy_1, behavior_monitoring, cre_Theme_park, customers_and_addresses, e_learning, game_1, insurance_and_eClaims, local_govt_and_lot, mountain_photos, news_report, program_share, race_track, railway, school_player, shop_membership, store_product, student_1, workshop_paper (18) | **716** |
| **Dev** | 20261120, from the remaining 105 | climbing, college_1, college_2, customer_complaints, device, dorm_1, flight_company, hospital_1, icfp_1, insurance_policies, local_govt_in_alabama, music_4, pilot_record, station_weather, store_1 (15) | 779 |
| Build | — | the other 90 DBs: retrieval only | — |

**Item eligibility:**
- spider-sql-v1 `train.jsonl`, tier `exec_ok`;
- no `empty_result` or `all_null_result` flag;
- **amendment A1** (2026-11-19, before any MQL-0010 prediction): membership in spider-pool-v3, the label-audit filter that built the MQL-0009 639. Before A1 the holdout had 859 items and dev 971.
- All query families are kept, including subqueries and set operations.

**Leakage checks:**

| Level | Check | Result |
|---|---|---|
| L1 | DB overlap between holdout and the retrieval view | 0 (view = corpus minus the 18 holdout DBs: 2,912 records, 105 DBs; a view, the corpus file is unchanged) |
| L2 | id overlap | 0 |
| L3/L4/L5 | exact, normalised or token-Jaccard ≥ 0.8 question matches between holdout and view | **13 pairs**, registered as question-level retrieval exclusions applied identically in every RAG arm |
| L6 | identical normalised gold SQL in the view | 14 items, report only (sensitivity S3) |

- **Structure-predictor training rows** (used only for evidence; component S is not adopted): train minus holdout DBs minus 11 near-duplicates of holdout questions = 6,869.
- **Gold answers** are never used to build a retrieval prompt or a retry message. The validator sees only the schema context.
- **Protected sets.** Neither the MQL-0009 639 nor the Spider-dev 892 is used for any MQL-0010 decision. The 639 is design-informed (PEA §8) and is not a confirmatory set.

**Limitation.** The holdout DBs are corpus DBs: their questions were converted and AI-reviewed during corpus construction. No model was trained on them and the retrieval view excludes them. Their gold SQL was used only to derive corpus labels. The fully external alternative is Spider test, but it is reserved for one final use by project policy. Using it is the user's decision; it is not adopted here.

## 5. Development protocol (before the holdout)

The configuration is frozen from offline evidence:

| Evidence | Measured on | Result |
|---|---|---|
| Structure predictor | train-side cross-validation | `dev_evidence/structure_predictor_cv.json` |
| Retrieval proxy | the 779 registered dev items, leave-own-DB-out | baseline hit@3 57.1, E 65.9 (`dev_evidence/retrieval_dev_proxy.json`, recomputed after A1; recorded in amendment A2, `preflight/amendment_A2_2026-11-19.json`) |
| Validator false alarms | corpus gold MQL | 0/3,445 |

**Optional dev rehearsal (recommended, about 3 T4-hours on a 200-item seeded dev subset).**
- Run A0 / D0 / D0V / E end to end to check the notebook, the reuse logic and the retry formatting.
- Report static statistics only: retry rate, retries ending with 0 issues, parse rates.
- Dev *accuracy* would need local MongoDB scoring. It may be computed only if the user approves, labelled development-only, never reported (CLAUDE.md).
- No hypothesis, threshold or flag may change after the rehearsal without a dated amendment written before any holdout generation.

## 6. Metrics

**Primary.** Cross-engine execution accuracy, correct / 716, computed with:
- the registered MQL-0006..0009 evaluator (`dev1034_core`);
- read-only Atlas user, full-parity preflight;
- nothing excluded.

**Secondary rates per arm:**
- invalid;
- execution error;
- executable but incorrect, split into empty result / non-empty wrong row count / wrong column count / wrong values;
- timeout.

**Recovery:**
- items retried;
- retries kept;
- retries ending with 0 issues;
- outcome transitions caused by kept retries (fix / break).

**Retrieval:**
- distinct structure signatures shown;
- ranks of the selected examples;
- score plateaus;
- share of items with an F note.

**Cost:**
- input and output tokens per arm;
- generation wall time;
- retry share.

## 7. Statistical plan

- Paired per-question analysis only. Seeds are never pooled; there are none in MQL-0010.
- **Order (A3):** D0V vs D0, then FULL vs D0, then {E vs D0, EF vs E} with Holm. Exact McNemar, two-sided, plus bootstrap CIs; fixed-sequence gatekeeping at α = 0.05.
- **Sensitivity:**
  - S1: reference status `ok` only;
  - S2: without the 13 L3–L5 exclusion items;
  - S3: without the 14 L6 items.
- **Power** (normal approximation, α = 0.05, power 0.8, n = 716):
  - with 16% discordant pairs (MQL-0009 D vs C), MDE ≈ 2.8·√(0.16/716) ≈ **4.2 pp**;
  - ~~for D0V vs D0, discordance is bounded by the retried share (about 20%). If about half the retries change the outcome, MDE ≈ 3 pp.~~ **Corrected 2026-11-19 (amendment A4, audit R4):** with about 21% of items flagged and near-zero breaks, the exact McNemar test rejects at about **+1 pp** (power 0.79 at a 5% repair rate). Step 1 is therefore called *practically meaningful* only if the 95% CI lower bound is ≥ +1.0 pp.
  - **E's expected effect (≤ 2–3 pp) is below the MDE; a null H3 is not evidence of no effect.**
- **Tie order.** A single scoring pass; tie-dependent items are reported.

## 8. Model and inference configuration (as MQL-0006..0009)

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1ff808d1d3d0ba301aef8c11461451ec5` |
| Precision and decoding | bfloat16 (emulated on the T4), greedy, `max_new_tokens` 1024 (retry: the same) |
| Batching | batch 8, seed 42; OOM fallback to batch 1, recorded |
| Hardware | Kaggle Tesla T4 only |
| Prompts | zero-shot `nl-mql-spider1-v1`; RAG `nl-mql-spider1-rag-v1`; schema context derived offline from SQLite with the `import_atlas.py` mapping (as MQL-0009) |
| Retrieval budget | k = 3, `max_context_tokens` 1500, `n_ctx` 8192, margin 64 |
| Retry turn | same system prompt; `[user: first prompt, assistant: first answer, user: issues + RETRY_SUFFIX]` |

## 9. Prerequisites before any generation (not done; need user authorisation)

1. **Kaggle package and notebook.** Build them from `scripts/nl2mql_mql0010/src`, reusing the MQL-0009 package builder pattern:
   - frozen prompts for the 716 holdout items;
   - plans for A0 / D0 / E / EF;
   - a manifest;
   - a resume-safe first pass followed by the retry pass;
   - export with hashes.
   
   The components and driver are implemented and tested, but **the package and notebook are not built**.
2. **Holdout schema context.** Derive it offline (as MQL-0009's `derive_context`), then freeze and hash it.
3. **Atlas provisioning** of the 18 holdout databases into a separate database, for example `codegen_spider1_holdout_mql0010`:
   - insert-only wrapper, as in `provision_heldout_atlas.py`;
   - a read-only `find` / `listCollections` / `killCursors` grant for `codegen_eval_readonly`;
   - full-parity contract.
4. **A scoring adapter** that scores all 7 arms in one session against one preflight. It must refuse partial exports and verify reuse by hash.

## 10. Compute

From MQL-0009 run configs (9.9–12.8 s per item):

| Step | T4-hours |
|---|---:|
| A0 | 2.0 |
| D0 | 2.6 |
| E | 2.6 |
| EF delta (about 52 items on 2 DBs) | 0.3 |
| Retries for A0V / D0V / FULL (≈ 20% of items × 3) | about 2.0 |
| **Total** | **≈ 9.5** |

This fits in 1–2 Kaggle sessions. Atlas scoring takes about 30 minutes locally.

## 11. Artifact and checkpoint contract

| Artifact | Contents |
|---|---|
| `predictions/<arm>/predictions.jsonl` | one record per item: `raw_text` (scored), `first_raw_text`, `source`, `prompt_sha256`, `recovery`, `experiment` |
| `predictions/<arm>/plan.jsonl` | the planner rows: retrieval decisions, notes, hashes, flags |
| `predictions/<arm>/run_config.json` | model, revision, environment, flags, source hashes |
| Checkpoints | append-only JSONL keyed by id, fingerprinted (the MQL-0009 `Checkpoint` class); a retry checkpoint is separate from the first pass |
| Export | `MQL0010_PREDICTIONS_MANIFEST.json` + `SHA256SUMS.txt`; the scorer refuses unpinned or partial exports |

## 12. Reproducibility rules

- Deterministic retrieval: stable sort keys, no randomness except fixed seeds.
- Every prompt is rebuildable from the plan and hash-checked (`planner.user_text_for`).
- Registered files are immutable; a change needs a dated amendment written before the affected predictions.
- The MQL-0008 / MQL-0009 code, predictions and scores, the approved corpus v2.1 and its converter are not modified. Tests assert the corpus and package hashes.
- With every flag off, the planner reproduces the registered MQL-0009 D plan for 639/639 items with byte-identical prompts.
