# sql2mql-v2 Phase-2 pilot with verifier-3 (run01)

Date: 2026-11-06. All figures come from the local SQLite + MongoDB **8.0.34** Docker container (`codegen-mongo-8034`, image digest `sha256:b9924c33…fa142`, `127.0.0.1:27018`). They are **label-validation measurements, not model metrics**. Nothing connected to Atlas.

**Status: the pilot was executed on all 300 frozen examples. The pilot gate is NOT passed yet.**
- P1 (automated equivalence checks on accepted labels) and P4 (every failure classified) are complete.
- P2 (independent human review) and P3 (no systematic error) need the named human reviewer. The review package is built, with 0 judgements filled.

**Verifier acceptance is not proven semantic correctness.** An accepted label passed every verifier-3 check, but round 3 measured a verifier miss rate of 2.38% (weighted, upper bound 4.12%), and 6.7% on ≥3-table joins. No label in this pilot is approved for training. Corpus candidacy is decided only after P2.

## 1. Runner

New file `scripts/finetune_v2/run_pilot_v3.py` (`sql2mql-v2-pilot-runner-v3.1`, sha256 `daf42dfa6d62e3b485436ff16f2d4807565cc442e9027b8b5e85d66c949ce5f5`). `run_v2.py pilot`, which runs verifier-2, was not used or changed.

| Property | How it is enforced |
|---|---|
| **Frozen code, checked before import** | The 19 files in `data/interim/finetune_v2/round3_code_freeze.json` are hashed **before** any of them is imported. The manifest's own sha256 (`db320443…`) is pinned. It must list exactly those 19 files and name `sql2mql-v2-verifier-3` |
| **Fails closed** | A missing file, a changed byte, a missing or extra manifest entry, a wrong verifier name, an import failure, a module resolving outside `scripts/finetune_v2`, or a non-verifier-3 verification record stops the run before anything executes |
| **No verifier-2 fallback** | The runner never imports `verifier.py` or `run_v2.py`. `verifier3.py` itself runs verifier-2's checks first, by design. Every stored record must carry `sql2mql-v2-verifier-3`, or resume stops |
| **Pinned inputs** | Pinned and checked: the pilot freeze (`freeze_sha256` `fb1affb9…`, recomputed), the round-3 pre-registration (`944e9063…`), `spider_sql_v1/train.jsonl` (`40c1a7c1…`), `review_package.py` (`616903b8…`), and the spider_pool_v2 manifest and all four of its files |
| **Converter and verifier unchanged** | Conversion calls the frozen `run_v3.convert`; verification calls `verifier3.verify` with its default seed. The runner only classifies and records |
| **Read-only safety** | Before any execution, `eval_kit.validate_mql` (unchanged) runs on each converted MQL, restricted to the database's own collections, plus a recursive scan for `$out`, `$merge`, `$function`, `$accumulator` and `$where`. Unsafe MQL is never executed |
| **One outcome per example** | `accepted`, `conversion_rejected`, `conversion_error`, `safety_rejected`, `timeout` (MongoDB `maxTimeMS` or a SQLite interrupt), `verification_failed` (sub-class semantic / structure / precondition / other) or `verifier_error` |
| **Infrastructure loss** | A lost MongoDB connection aborts the run **without writing a record**, so a resume re-runs that example. Infrastructure faults never become pilot data |
| **Resume** | One flushed record is appended per finished example. On resume:<br>• a cut trailing line is dropped and logged;<br>• a duplicate, an ID outside the 300, or a record from another runner version stops the run.<br>`--budget` stops cleanly between examples |
| **Frozen before execution** | `data/interim/finetune_v2/pilot_v3_run01_runner_freeze.json` was written before any pilot example was converted. It holds the runner hash, the 19 hashes, the inputs, the MongoDB digest, the configuration and the classification rules. A changed runner refuses this run name |

**Two runner fixes were made before the freeze, both caught by tests:**
- `summary.json` depended on record order. It is now sorted by ID.
- The `os` import was removed, because an existing test forbids `os` in this package as an Atlas guard. Records are flushed but not fsynced.

A third pre-freeze fix: Python could not find the `docker` CLI, so the standard install path was added as a fallback.

**Files added:**
- `scripts/finetune_v2/run_pilot_v3.py`;
- `tests/test_finetune_v2_pilot_v3.py`;
- `data/interim/finetune_v2/pilot_v3_run01_runner_freeze.json`;
- `outputs/sql2mql_v2/pilot_v3/`;
- this report.

No existing file was modified apart from the ledger and presentation-notes appends.

## 2. The 19 frozen hashes

All 19 match, checked before the run and again after it.

| File | sha256 (first 12) | | File | sha256 (first 12) |
|---|---|---|---|---|
| `sql2mql_v2.py` | `07ea2f0391b8` | | `index_real.py` | `b87f20eefd74` |
| `schema_profile.py` | `9f8bdedbdb50` | | `v2_common.py` | `292c85202495` |
| `verifier3.py` | `8660753acb3d` | | `oracle_v2.py` | `bd4c2b1a55f4` |
| `fixtures3.py` | `0aa78fbd127c` | | `mutants_v2.py` | `3c2bbc3c4386` |
| `verifier.py` | `b28515f73b19` | | `run_v3.py` | `a118d398fd5b` |
| `v2_fixtures.py` | `53067b5c207f` | | `mutant_study_v3.py` | `66bb2b444cf4` |
| `comparator.py` | `315352369539` | | `round3_stats.py` | `b65435718778` |
| `mongo_import.py` | `663383297182` | | `round3_report.py` | `31e31c7ec1f8` |
| `freeze_round3.py` | `1e1c8bcdc7a8` | | `freeze_round3_code.py` | `08a23bdeaadb` |
| `scripts/evaluation/eval_kit.py` | `e479b5a8fc6a` | | | |

Full hashes are in the runner freeze and in `run_manifest.json`.

## 3. Tests

`python -B -m pytest -q tests/test_finetune_v2.py tests/test_finetune_v2_round3.py tests/test_finetune_v2_pilot_v3.py` gives **99 passed, 0 skipped, 0 failed**, with MongoDB up, so the live tests ran.

- **New (33):**
  - **Hashes:** all 19 match.
  - **Fail-closed checks:**
    - a missing verifier3 is rejected, in-process and through the CLI (non-zero exit, nothing run);
    - 5 frozen files modified by one byte are each rejected;
    - an incomplete, extended or verifier-2 manifest is rejected;
    - a verifier3 import failure is rejected;
    - a wrong `VERIFIER_VERSION` is rejected;
    - a verifier-2 record on resume is rejected.
  - **No fallback:** an AST check confirms the runner imports neither verifier-2 nor `run_v2`.
  - **Sample:** exactly the 300 frozen IDs, rows equal to the source, disjoint from DEV 500 and the round-3 740; a tampered pilot freeze is rejected; membership is recomputed as 271 / 1 / 4 / 24.
  - **Classification:** outcome classes, including both timeout kinds.
  - **Three-or-more-table rule:** top-level join, UNION, subquery, `$lookup` count.
  - **Corpus status:** kept separate from pilot inclusion.
  - **Read-only scan:** `$out`, `$merge`, `$where`, `$function`, a foreign `$lookup` and a foreign collection are all rejected.
  - **Unsafe MQL:** never verified.
  - **Timeout and verifier-error separation.**
  - **Lost MongoDB:** aborts with no record.
  - **Interruption:** a KeyboardInterrupt mid-run plus a cut line resumes to exactly 6 unique records, with no re-processing; budget stop works.
  - **Duplicate, foreign-ID or corrupt-line records:** stop the run.
  - **Outputs:** every example is accounted for, manifest hashes are correct, outputs are identical regardless of record order, and the summary refuses incomplete accounting.
  - **Live, on DEV IDs only:** real conversion plus verifier-3 gives accept, converter rejection and `tie_cut` as expected; a 3-table DEV item is accepted but quarantined, not dropped; a re-run is identical.
- **Existing V2 regressions:** `test_finetune_v2.py` 46 and `test_finetune_v2_round3.py` 20, all passing.

## 4. Pilot results: all 300 examples

The run took 2 chunks. The first stopped cleanly at its budget after 294 examples; the resume log shows 294 kept and 0 cut lines. There are 300 records with 300 unique IDs, equal to the frozen set. Every converted example was verified by `sql2mql-v2-verifier-3` (224 / 224).

| Outcome | Examples | Rate |
|---|---:|---:|
| **Accepted by verifier-3** | **213 / 300** | **71.00%** (Wilson 65.63–75.84%) |
| Converter rejection (unsupported) | 76 / 300 | 25.33% |
| Conversion error (crash) | 0 / 300 | 0.00% |
| Safety rejection (runner scan) | 0 / 300 | 0.00% |
| Timeout | 0 / 300 | 0.00% |
| Verification failure | 11 / 300 | 3.67% |
| Verifier error | 0 / 300 | 0.00% |

- **Coverage:** 224 / 300 converted (74.67%). Accepted of converted: 213 / 224 (95.09%).
- **Converter rejections (76):**

  | Code | Count |
  |---|---:|
  | `not_a_single_select` | 24 |
  | `construct` | 21 |
  | `bare_column` | 18 |
  | `aggregate_type` | 5 |
  | `join_key_types` | 5 |
  | `self_join` | 1 |
  | `join_condition` | 1 |
  | `arithmetic_type` | 1 |

- **Verification failures (11):** all are `tie_cut`, meaning ORDER BY … LIMIT cuts through a tie on the real database, so the SQL itself has no unique answer. They are untestable preconditions, not semantic mismatches.
- **Semantic mismatches:** 0 L1, 0 fixture, 0 reorder and 0 nondeterminism rejections.
- **P1:** 213 / 213 accepted labels carry passing results for:
  - L1 under explicit and absent NULL;
  - V6 double execution;
  - both v3 reorder runs;
  - `v3_lookup_keys`;
  - V8 (`validate_mql`, plus the runner scan);
  - every fixture family.

  V5 is enforced by the `tie_cut` rejection.
- **P4:** every non-accepted example has a classified cause (0 unclassified). No accepted label is *known* wrong, but that is not evidence that none is.

## 5. Results by prepared-pool membership

Membership was recomputed by ID against `data/processed/spider_pool_v2`. Every one of the 300 was processed and reported, whatever its membership.

| Membership | Examples | Converted | Accepted | Other outcomes | Corpus status |
|---|---:|---:|---:|---|---|
| In v2 `train.jsonl` | 271 | 207 | **207 (76.38%)** | 64 converter rejections | 188 candidates pending P2; 19 quarantined (≥3 tables); 64 not candidates |
| Quarantined (`04586`) | 1 | 1 | 1 | none | **Not a candidate** (stays quarantined) |
| Excluded (`01259`, `02225`, `06102`, `06104`) | 4 | 0 | 0 | 4 converter rejections (`bare_column`, `aggregate_type`, `arithmetic_type`, `construct`) | Not candidates |
| Not in pool | 24 | 16 | 5 (20.83%) | 8 converter rejections, **11 `tie_cut`** | Not candidates |
| In v2 `val_internal` | 0 | — | — | — | — |

- **Quarantined `04586`:** its audit reason is "= literal matches no stored value (OR or UNION branch never fires)". Verifier-3 accepted the SQL→MQL translation, but that does not settle whether the Spider SQL answers the question. It was not promoted.
- **The 24 out-of-pool examples:** all 11 pilot `tie_cut` failures fall here, which is consistent with the pool builder's own row-choice rejects.
- **One in-pool pilot ID differs in text between sources:** `spider_train_01837`. `spider_pool_v2` writes the question without the space before "?", and its `question_raw` keeps the original. The pilot used the frozen source text unchanged.

## 6. Results by SQL difficulty and query pattern

**Spider hardness**

| Hardness | Examples | Converted | Accepted | Rejected by converter | `tie_cut` |
|---|---:|---:|---:|---:|---:|
| Easy | 72 | 69 | 69 (95.83%) | 3 | 0 |
| Medium | 118 | 106 | 105 (88.98%) | 12 | 1 |
| Hard | 64 | 35 | 30 (46.88%) | 29 | 5 |
| Extra | 46 | 14 | 9 (19.57%) | 32 | 5 |

**Round-3 stratum** (`round3_stats.stratum`, top-level SELECT)

| Stratum | Examples | Converted | Accepted | Rejected by converter | `tie_cut` |
|---|---:|---:|---:|---:|---:|
| J2 (≥2 joins) | 23 | 20 | 20 (86.96%) | 3 | 0 |
| OL (ORDER BY … LIMIT) | 52 | 39 | 28 (53.85%) | 13 | 11 |
| J1 (one join) | 58 | 43 | 43 (74.14%) | 15 | 0 |
| G (other) | 167 | 122 | 122 (73.05%) | 45 | 0 |

**SQL pattern** (sqlglot; one example can have several patterns)

| Pattern | Examples | Accepted |
|---|---:|---:|
| Simple select | 38 | 38 (100.00%) |
| Join | 112 | 65 (58.04%) |
| Aggregate | 140 | 87 (62.14%) |
| GROUP BY | 78 | 41 (52.56%) |
| HAVING | 19 | 10 (52.63%) |
| ORDER BY | 76 | 47 (61.84%) |
| LIMIT | 56 | 28 (50.00%) |
| DISTINCT | 29 | 25 (86.21%) |
| LIKE | 9 | 8 (88.89%) |
| Subquery | 21 | 0 (0.00%) |
| Set operation | 24 | 0 (0.00%) |

Subqueries and set operations are out of the converter's scope by design. Accepted labels therefore under-represent hard and extra queries.

## 7. Three-or-more-table joins

The rule was frozen before execution. An example is flagged if any of these holds:
- its SQL is in J2;
- its SQL names ≥3 distinct tables anywhere;
- its MQL has ≥2 `$lookup` stages.

Flagged examples stay in every pilot count above.

| | Examples | Accepted | Rejected by converter |
|---|---:|---:|---:|
| Flagged (≥3 tables) | **28** | **20 (71.43%)** | 8 |
| Not flagged | 272 | 193 (70.96%) | 68 (plus 11 `tie_cut`) |

- **Indicators:** J2 stratum 23; ≥3 distinct tables 28; ≥2 `$lookup` 20.
- **Disagreements:** the indicators disagree on 5 examples (`02209`, `03852`, `03991`, `05021`, `05959`). Each names ≥3 tables outside the top-level join chain, through a subquery or set operation. All 5 were rejected by the converter.
- **Corpus effect:** the 20 accepted flagged labels (19 in v2 train, 1 not in pool) are **quarantined from corpus candidacy**, as listed in `multi_table_3plus_quarantine.jsonl`. Their verifier acceptance is reported as is.
- **Why the quarantine:** round 3 measured J2 at 7/104 false accepts (Wilson 3.30–13.25%). Releasing them needs a J2-specific check on records never used, or a supplementary human review. That is your decision.
- **P2 coverage:** the frozen P2 sample contains 11 flagged labels, selected by the unchanged rule.

## 8. Human-review package

**Location:** `outputs/sql2mql_v2/pilot_v3/run01/review_package/`. It holds:
- `review_sheet.csv` (UTF-8 BOM, empty reviewer columns);
- `review_packet.md`;
- `review_items.jsonl`;
- `package_manifest.json` (`judgements_filled: 0`).

**How it was built:**
- Built with the **unchanged** frozen `review_package.build` (sha `616903b8…`), using its frozen P2 selection: min(120, accepted), stratified, seed 20261101.
- **Selection:** 120 of 213 accepted labels, by stratum J2 11, OL 16, J1 24, G 69. By membership, 118 are v2 train and 2 are not in pool.
- **Location note:** `docs/sql2mql_v2_human_review.md` names `outputs/sql2mql_v2/review_pilot/` as the output. That folder does not exist; the package lives inside the versioned run folder instead. The template and selection rule are identical.

**Next step:** a named, independent human reviewer fills `review_sheet.csv` as `docs/sql2mql_v2_human_review.md` describes. P2 passes if the Wilson upper bound of wrong / reviewed is ≤ 5%. P3 fails if any discrepancy type occurs ≥ 2 times.

## 9. Reproducibility and integrity

- **Second execution:** the frozen runner and data produced `outputs/sql2mql_v2/pilot_v3/run01_repro/`.
- **Strict comparison:** the runner's own comparison (`run01/reproducibility.json`) found **22 / 300 records differing, and exited 1**.
- **What differs:** every difference is in one informational field, `verification.checks.v6_determinism`. In both runs, V6's two executions returned an equal result. The field records only whether they also returned documents in the same order, which MongoDB does not guarantee without a sort.
- **After excluding that note and timings:** 300 / 300 records are identical. Outcomes, verifier statuses, MQL text, corpus status and the accepted set are all identical (`run01/reproducibility_diff_analysis.json`, produced by `pilot_v3/repro_diff_analysis.py`).
- **Integrity:**
  - **Frozen code:** all 19 hashes match after the run.
  - **Inputs unchanged:** `spider_pool.zip` (`31848241…`), the pilot freeze (file `df46c580…`) and spider_pool_v2 (manifest `cb80a118…`).
  - **Manifest:** `run_manifest.json` output hashes match the files.
  - **Independent recount** from raw `records.jsonl`: 213 accepted, 76 converter rejections, 11 `tie_cut`.
  - **Text preserved:** question, SQL, `db_id` and hardness equal the source for all 300.

**Outputs** (`outputs/sql2mql_v2/pilot_v3/run01/`):

| File | Contents |
|---|---|
| `records.jsonl` | Raw, append order |
| `results_per_example.jsonl` | 300 records, sorted |
| `accepted.jsonl` | 213 records |
| `rejected.jsonl` | 87 records |
| `classifications.jsonl` | Membership, outcome, multi-table flags, corpus status; `pilot_evaluation_included` true for all 300 |
| `corpus_ineligible.jsonl` | 93 records |
| `multi_table_3plus_quarantine.jsonl` | 28 records |
| `summary.json` | Counts, rates and gate status |
| `run_manifest.json` | Sample IDs, versions, hashes, configuration |
| `resume_log.jsonl` | Resume events |
| `reproducibility.json` | The runner's strict comparison |
| `reproducibility_diff_analysis.json` | Field-level breakdown of the differences |
| `review_package/` | The P2 human-review package |

No corpus file was written.

**Files written after `finalize`.** These are not covered by `run_manifest.json`, and `finalize` was not re-run, so the manifest was not overwritten. Their sha256 values:

| File | sha256 |
|---|---|
| `data/interim/finetune_v2/pilot_v3_run01_runner_freeze.json` | `0e941a308342e802cd5a10fe37baa0437a1f9f6f8e71daa7bd6455a84fad8681` |
| `run01/reproducibility.json` | `e330314b9440b5b3f755e5251178cabad08bc1d931567779853f2962ca142c18` |
| `run01/reproducibility_diff_analysis.json` | `98bad37bf73a85fef3b7879c43734a3ad57e0bd77982c08f849218fb4c550ff8` |
| `pilot_v3/repro_diff_analysis.py` | `ea4d200bb3be92c1e54797bdef952f620390401ca7a1e35bb28d3036501c19f9` |
| `run01_repro/records.jsonl` | `c33f1b19dd84839bc790e554d73700de88146774f6a35e1ce3733810d9413752` |
| `run01/review_package/package_manifest.json` | `affabd6273a63ff9087c7bb9920adcc2e810fb82285d31e88c90c0422cacef00` |
| `run01/review_package/review_sheet.csv` | `deecc2fa67bc5c70d7304a7a3bf36c914ca08c74f1890430963348ba56a6f3d1` |
| `run01/review_package/review_packet.md` | `585997a0effe407da60bfd5ac0f23bf72444a1197d2da3fb9b5691f93c7bba19` |
| `run01/review_package/review_items.jsonl` | `f2ddda16e1651c262c2bd35d4a162db571baae5520b3c9c44f4a0acf049b1833` |

**`accepted.jsonl` is not a corpus.** 25 of its 213 records are not candidates:
- 19 are ≥3-table rows in v2 train;
- 1 is the quarantined `04586`;
- 5 are not in the pool.

`corpus_status` in `classifications.jsonl` is the field to use.

**Open text decision for a later corpus build:** `spider_train_01837`'s question differs between spider-sql-v1 and spider_pool_v2 by one space before "?". The pilot used the frozen spider-sql-v1 text.

## 10. Blockers and limitations

1. **The pilot gate is not passed.** P2 and P3 need a named independent human reviewer. An AI-written converter and verifier cannot review their own output.
2. **Acceptance is not correctness.** It is bounded by verifier-3's measured sensitivity: a weighted 2.38% false-accept rate (upper bound 4.12%), and 6.7% on J2. G4 is also a lower bound, since single-edit mutants only.
3. **SQL labels are not certified.** Verifier-3 checks SQL↔MQL equivalence, not whether Spider's SQL answers the question. Gold-SQL problems are what P2's `nl_sql_agree` column is for.
4. **The strict reproducibility check failed on an informational field.** Substantive reproducibility holds (300 / 300). The runner was not changed after its freeze; a future runner should leave the V6 order note out of its comparison.
5. **Coverage is limited.** Subqueries and set operations are 0% converted, and hard/extra acceptance is 47% / 20%. A corpus built from this converter will be skewed toward simpler queries.
6. **Corpus candidacy (188) is provisional.** No training or RAG corpus was created. Releasing candidates needs P2, and releasing the 19 quarantined 3-table labels needs your J2 decision.
7. **Shared environment.** I started Docker Desktop and the existing `codegen-mongo-8034` container (stopped since the last session). No other container was touched, though Docker Desktop's own restart policies auto-started some unrelated containers. Verifier-3 writes only to its own fixture and reorder databases, as in round 3.
8. **Local only.** Nothing here is an Atlas or model metric.
