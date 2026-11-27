# Corpus v2 pre-training sign-off report (2026-11-16)

**Current status (updated 2026-11-16): CONDITIONAL GO — APPROVED FOR TRAINING UNDER LOCAL VALIDATION POLICY.** The project owner accepted local MongoDB verification for training labels; see the addendum at the end. Training-corpus execution on Atlas was NOT PERFORMED, and evaluation metrics must still be generated on Atlas.

The text below, up to the addendum, is the report as first written, when the status was BLOCKED. It is kept unchanged as the record of that state, apart from one dated wording correction in section 6.

**Status when first written: BLOCKED.** Fine-tuning is not authorized.

The Atlas execution audit could not execute any corpus-v2 label: none of the collections the corpus reads exist on Atlas, and the only credential is read-only. Every other item in this task is complete. The corpus is **AI-reviewed, not human-reviewed**, and the run01 human P2 gate is **not passed**.

## The blocker

- Corpus v2 is built from 123 Spider *training* databases (516 collections).
- Atlas holds one database, `codegen_spider1`, with 80 collections: the *evaluation* databases. The corpus is disjoint from them by design (the leakage rule; `db_overlap_with_spider_dev` is empty).
- Result of the live check: 0 of 516 corpus collections are on Atlas, so 0 of 3,447 records can be executed there.
- Verifier 3.1 also needs to create 47 fixture collections per record, which a read-only account cannot do.
- The earlier preflight covered the evaluation collections and the LIKE regex only. It is not evidence about the corpus labels, and it is not presented as such.

Ways to remove the blocker (a decision for the project owner):

| Option | What it needs | What it would establish |
|---|---|---|
| A. Load the training databases into a separate Atlas database and run verifier 3.1's real-data check there | A write-capable credential for a scratch database, and an explicit decision to allow writes outside `codegen_spider1`. Source SQLite size is 366 MB; the Atlas tier's storage limit must allow it. | Each label returns the SQLite reference result on Atlas, on real data. Fixture checks would also need write access. |
| B. Run each label on the Atlas server with the rows supplied inline (`$documents`), read-only | No new access. Joins (826 records) would need their `$lookup` source replaced by inline rows, which changes the query under test. This is a new method and was not used here because the task ruled out inventing one. | The Atlas server version executes the label and returns the reference rows. It does not use Atlas collections. |
| C. Accept local verification for training labels | A recorded decision. CLAUDE.md already allows label validation on a local MongoDB; only reported model metrics must come from Atlas. | Nothing new. The status would become CONDITIONAL GO with the limitation stated. |

## Checklist

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Test suite | Pass: 591 collected, 591 passed, 0 failed, 0 skipped | `python -B -m pytest tests -q -p no:cacheprovider -rs` (before: 581 collected, 578 passed, 3 failed) |
| 2 | Six nonconformant-field decisions | Done: 2 quarantined, 4 included | `data/interim/corpus_v2_adjudication_v1.jsonl` |
| 3 | Atlas execution validation | **BLOCKED**: 0 attempted, 0 validated | `outputs/corpus_v2_atlas_audit/run_20261116T183448/report.json` |
| 4 | Protected holdout integrity | Intact: never opened; size and modification time match the baseline | same report; `hash_baseline.py verify signoff` |
| 5 | G-AIR protocol documentation | Done | `docs/g_air_protocol_decision_2026-11-16.md` |
| 6 | 667 quarantined records | Retained in quarantine, not re-reviewed, provenance kept | `data/processed/spider_pool_v3/exclusions_v3.jsonl` |
| 7 | Approved corpus manifest and hashes | **Not approved.** A candidate exists with status `CANDIDATE_NOT_APPROVED` | `outputs/sql2mql_v2_1/corpus_v2_1_candidate/manifest.json` |
| 8 | Remaining limitations | Listed below | |
| 9 | Fine-tuning authorized | **No** | |

## 1. Test fixes

- **`test_no_atlas_credentials_are_read`** (`tests/test_finetune_v2.py`): the text search is replaced by a syntax-tree scan. It flags attribute or name access to `environ`, `getenv`, `putenv` and the dotenv loaders, imports of them, and string literals that name them or contain an Atlas URI marker. Comments and docstrings are ignored. The forbidden-import check (`os`, `dotenv`, `mongo_conn`) is unchanged. `scripts/finetune_v2/run_pilot_v4.py` was not edited; its hash is unchanged. All 38 files in `scripts/finetune_v2/` pass the scan.
  - New regression test `test_credential_scan_ignores_comments_but_catches_real_access`: a comment containing "environment" passes; eight forms of real access are caught.
- **Kaggle bundle tests** (`tests/test_kaggle_sql_v1_baseline.py`):
  - Cause: the ZIP and the bundle as built are clean (the builder copies an allowlist). The two `.pyc` files appeared on 2026-10-29 when Python imported the evaluator out of the bundle directory.
  - Fix: `tests/conftest.py` turns off bytecode writing for the test session; `scripts/sql_v1/prepare_kaggle_dev_bundle.py` gains `purge_python_caches` (run on every build and available as `--purge-caches`). `verify_dir` is as strict as before and still rejects cache files.
  - The two stale `.pyc` files were removed from the unpacked bundle directory. The ZIP was not touched (same SHA-256 before and after) and no new bundle version was needed.
  - New tests: the bundle and ZIP hold no cache files; `verify_dir` rejects a cache file and the purge removes exactly those; tests do not write bytecode.

## 2. The six nonconformant-field records

Rule, fixed before the outcomes were examined: quarantine if any type-dependent operation reads the field; include only if the field is copied to the output and nothing else.

| Record | Collection and field | Stored values | Use in the query | Decision |
|---|---|---|---|---|
| `spider_train_00145` | `bike_1__weather.cloud_cover` | 3,664 integers, 1 empty string | `$group` key (top-3 most common) | Quarantine |
| `spider_train_00146` | same | same | `$group` key | Quarantine |
| `spider_train_05887` | `cre_Theme_park__Hotels.price_range` | 14 reals, 6 empty strings | projected only | Include |
| `spider_train_05888` | same | same | projected only | Include |
| `spider_train_05895` | same | same | projected only (filter is on `star_rating_code`, a conforming text field) | Include |
| `spider_train_05896` | same | same | projected only | Include |

- Whether Atlas data satisfies the query's assumptions could not be checked for any of the six: these collections are not on Atlas.
- For the four included records no assumption about the field's type is needed, and verifier 3.1 accepted them on the real data and fixtures (local MongoDB).
- No SQL or MQL was rewritten. The decisions were made by rule, by an AI, not by a human.

## 3. Atlas execution audit

What was actually executed on live Atlas (server 8.0.37, 3-node topology, not a local host):

- connection and authentication; privilege check: 1 user, 0 write or admin actions granted;
- collection listing: 80 collections;
- TC-1 re-check of the 70 evaluation collections: 0 violations, 0 missing, no unknown nonconformant field.

What was only inspected, offline:

- all 3,447 labels passed the read-only safety gate (`validate_mql` plus a scan for `$out`, `$merge`, `$where`, `$function`, `$accumulator`): 3,447 passed, 0 rejected.

Outcome counts:

| Outcome | Records |
|---|---|
| Discovered | 3,447 |
| Attempted on Atlas | 0 |
| Completed | 0 |
| Skipped: collection absent on Atlas | 3,447 |
| Validated | 0 |
| Incorrect result | 0 |
| Invalid query | 0 |
| Execution error | 0 |
| Timeout | 0 |
| Not comparable | 0 |
| Infrastructure failure | 0 |

- The six nonconformant-field records: not executed, same reason.
- Interruptions and retries: none.
- Atlas was not modified: the only commands sent were ping, buildInfo, connectionStatus, listCollections and read-only `$type` census aggregates.
- The protected holdout was not opened.

## 4. Candidate corpus (not approved)

- `outputs/sql2mql_v2_1/corpus_v2_1_candidate/corpus.jsonl`: 3,445 records, SHA-256 `96552f06…dd199bf`.
- It is provisional corpus v2 (3,447 records, SHA-256 `764f92f9…b618c604af`) minus the 2 quarantined records; every remaining line is byte-identical.
- The manifest records the adjudication decisions, converter and verifier hashes, type-contract hashes, the Atlas audit run, the G-AIR reference and limitations, and the test result. Its status is `CANDIDATE_NOT_APPROVED` and `fine_tuning_authorized` is false.
- Provisional corpus v2 is untouched (`records.jsonl` hash equals the manifest value).

## 5. Frozen artifacts

`hash_baseline.py verify signoff`: 217 recorded files unchanged, no file added inside a frozen directory, protected holdout unchanged and not opened.

## 6. Remaining limitations

- **No corpus label has been executed on Atlas.** All verification used local MongoDB 8.0.34; Atlas runs 8.0.37.
- AI-only review, one model family, partial reviewer isolation, a model change in the stability run. See `docs/g_air_protocol_decision_2026-11-16.md`.
- The 0 / 200 translation result does not establish a population error rate below 5%, and the 60 / 60 calibration result does not show that unknown defects are caught.
- In the G-AIR sample the panel flagged 7 of 200 source labels (6 NO, 1 UNSURE), and those 7 records are quarantined. This is an observed result for that sample, as judged by an AI panel. It is not a reliable estimate of the share of wrong source labels in the whole corpus. (Corrected 2026-11-16: an earlier version of this line said a similar share should be expected in the corpus; that inference is not supported.)
- Verification checks SQL against MQL, not SQL against the question, so wrong source labels that no detector or reviewer flagged can still be in the corpus. How many is unknown.
- The corpus records were not individually reviewed.
- 667 detector-hit records are excluded although most are probably correct.

## 7. Can Qwen2.5-Coder-3B fine-tuning start now?

**No.** The required Atlas validation is blocked and the corpus is not approved. Choose one of options A to C above; after that the decision can be revisited.

---

## Addendum (2026-11-16): owner decision and approval

**Final status: CONDITIONAL GO — APPROVED FOR TRAINING UNDER LOCAL VALIDATION POLICY.** This supersedes the BLOCKED status above. The Atlas audit result itself is unchanged: it stays BLOCKED, with 0 corpus records executed on Atlas.

- **Owner decision (project owner, 2026-11-16):** Option C above (presented to the owner as Option 1). Local MongoDB verification is accepted for training labels, under the existing policy that labels may be validated locally while reported evaluation metrics must come from Atlas.
- **Validation policy:**
  - training-label validation: local MongoDB 8.0.34, verifier 3.1;
  - Atlas: evaluation environment and the only source of reported benchmark metrics;
  - Atlas preflight: passed for the 70 evaluation collections;
  - training-corpus execution on Atlas: **NOT PERFORMED**;
  - the absence of the 516 training collections from Atlas is an accepted infrastructure limitation, not a successful Atlas execution audit.
- **Approved corpus:** `outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl`, 3,445 records, SHA-256 `96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf`. Manifest `manifest.json` (SHA-256 `1c5e01ba…94711049`); checks in `approval_checks.json`. Provisional corpus v2 and the candidate were not modified.
- **Checks run by `scripts/corpus_v2_signoff/approve_corpus.py` (all 14 passed):**
  - the corpus is byte-identical to provisional corpus v2 minus the 2 quarantined records, and to the candidate;
  - 3,445 / 3,445 records have a complete verifier 3.1 record: accepted, all 12 checks present, real-data and fixture comparisons matching, MQL identical to the verified MQL. Some fixture families were skipped by the verifier as degenerate or for storage-class drift; counts are in `approval_checks.json`;
  - local MongoDB version is exactly 8.0.34 now, and the runner enforces it at every start;
  - a seeded sample of 64 records (60 random plus the 4 included nonconformant-field records) was re-verified live with the frozen runner: 64 / 64 reproduced;
  - every record is in spider-pool-v3 train and in no exclusion list; code freeze, type contract and pool hashes unchanged;
  - 217 frozen files unchanged; holdout not opened; Atlas not contacted by the approval script.
- **Six records:** 2 quarantined, 4 included (output-only use); no SQL or MQL rewritten.
- **Review status:** G-AIR passed under its AI-only procedure. Human P2 is NOT PASSED. The data is AI-reviewed, not human-reviewed. All limitations in `docs/g_air_protocol_decision_2026-11-16.md` stand.
- **667 detector-hit records:** still quarantined; not reintroduced.
- **Tests:** `python -B -m pytest tests -q -p no:cacheprovider -rs`: 594 collected, 594 passed, 0 failed, 0 skipped.
- **Fine-tuning:** Qwen2.5-Coder-3B fine-tuning on the approved corpus may begin. It was not started by this task. Evaluation metrics must be generated on Atlas.
