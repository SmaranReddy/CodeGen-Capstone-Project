> **SUPERSEDED 2026-11-23** by the user's request to use the MQL-0010 holdout instead (MQL-0014, `outputs/nl2mql_mql0010_r3_execution_recheck/`). Nothing was generated or provisioned for MQL-0013, and test_full remains unused. The MQL-0013 files are kept unmodified as a record.

# MQL-0013 execution runbook: R0 vs R3 on Spider test_full (Atlas execution accuracy)

Status 2026-11-22: **registered, package built, dry-run passed. Nothing has been generated or scored yet.**

MQL-0013 is one controlled comparison:

- **R0:** BM25 hybrid retrieval (MQL-0010 D0).
- **R3:** RRF(BM25, dense bge-small via Qdrant) + Jaccard, K=3 (MQL-0011 R3, unchanged).
- **Both arms** go through the MQL-0010 D0V recipe: first answer, static schema validator, then one schema-feedback retry.
- **Same everywhere:** Qwen2.5-Coder-3B @ 488639f1, bf16 on a T4, greedy decoding, the same prompts and the same scorer.

## Benchmark

- **Source:** Spider `test_full`, 2,147 questions over 40 databases (`data/processed/spider_sql_v1/test_full.jsonl`). It was chosen by the user on 2026-11-22.
- **Never used before:** no NL→MQL experiment has used these databases.
- **Disjointness**, asserted at build, by database and by id, from:
  - the approved corpus;
  - the MQL-0010 holdout, dev and build databases;
  - X0;
  - spider-sql-v1 train, val_internal and dev_full;
  - the expanded-split manifest.
- **This experiment uses up test_full** for NL→MQL.
- **No filtering:** the denominator is 2,147 for every arm.

## Artifacts

| What | Path | sha256 |
|---|---|---|
| Registration (write-once) | `outputs/nl2mql_mql0013/registration/mql0013_registration.json` | aa64661e… |
| Package zip | `kaggle/nl2mql_mql0013_test_full_bundle.zip` | 5cd00340… |
| Package manifest | `kaggle/nl2mql_mql0013_test_full/package_manifest.json` | b43c5625… |
| Package config | `kaggle/nl2mql_mql0013_test_full/configs/experiment.json` | 54f2525f… |
| Atlas contract | `outputs/nl2mql_mql0013/scoring_contract/atlas_mapping.json` | 32cbbc4d… |
| Build report | `outputs/nl2mql_mql0013/work/build_report.json` | |
| Mocked end-to-end dry run | `outputs/nl2mql_mql0013/dryrun/dry_run_report.json` | |

## Package facts

- **Shared prompts:** R0 and R3 prompts are byte-identical on only 165/2,147 items. The other 1,982 items differ in at least one example or in example order.
- **Budget drops:** 0 in either arm.
- **Leakage exclusions:** 24 L3–L5 pairs, applied to both arms.
- **Plans** are byte-identical across two independent builds.
- **Qdrant storage** is byte-identical before and after the build (read-only).
- **Schema info for masking and cues:** schema info comes from the Atlas-form context, as in the MQL-0010 D0 package.
  - Masked questions, and therefore the dense queries, are identical on all 2,147 items to the tables.json masking that MQL-0011/0012 used.
  - The regex cues differ on 17 items. That difference applies to both arms equally.
- **Mocked end-to-end dry run** (fake generator and fake evaluator; real plans, reference and scoring code):
  - It ran a kill and resume across 2 sessions.
  - 562 items per arm were retried and about 300 retries were kept.
  - The 45 shared items that needed a retry in both arms were generated once, giving 1,079 distinct retry prompts.
  - Export gate and analysis: OK.
- **Known limitation:** the export gate refuses any record whose generation failed even at batch size 1. Such an item would block the export rather than count as wrong. All 61 out-of-memory fallbacks in MQL-0010 succeeded at batch 1.

## Step 1: provision Atlas (user action, authorised 2026-11-22)

1. **Create a provisioning user.** In the Atlas UI, create a user (for example `mql0013_provisioner`) with a custom role on database `codegen_spider1_test_mql0013` only.
   - It needs exactly: `find`, `insert`, `listCollections`, `createCollection`.
   - It must have no write action on any other database (MQL-0010's `mql0010_provisioner` role must not be reused unchanged).
   - The wrapper refuses before writing if this does not hold.
2. **Set the URI and run the provisioning.** Put the user's URI in `$MQL0013_PROVISION_URI` (never `MONGODB_URI`). Then run:
   ```
   python -B scripts/nl2mql_mql0013/provision_test_atlas.py --execute --target-db codegen_spider1_test_mql0013 --confirm-target codegen_spider1_test_mql0013
   ```
   - It loads 180 collections, 245,574 documents in all; aan_1 alone holds 243k of them.
   - It verifies the count and digest of every collection.
   - It creates no indexes.
3. **Grant the scoring user.** Give the read-only scoring user `codegen_eval_readonly` the actions `find`, `listCollections` and `killCursors` on `codegen_spider1_test_mql0013`. A missing grant blocked MQL-0009 scoring for a day.

## Step 2: generate on Kaggle (user action)

- **Setup:**
  - Upload `kaggle/nl2mql_mql0013_test_full_bundle.zip` as a dataset.
  - Open `nl2mql_mql0013_test_full.ipynb`.
  - Settings: GPU T4 x2 (one card used), Internet on, no secrets.
- **Expected cost** (from MQL-0010's measured 13.9 s/item at batch 8 and 16 s per retry): about 16 T4-hours of first passes (165 + 2 × 1,982 prompts) plus about 3–4 h of retries. That is **about 20 T4-hours, so two or three sessions.**
- **Session 1:** run all cells. After 11 h it exports a partial `mql0013_predictions.zip`.
- **Each later session:**
  - Attach the previous version's output.
  - Set `RESUME_FROM = "auto"` and `SESSION = 2`, then 3.
  - Pin the environment after session 1. Export refuses passes run in different environments.
- **Finish:** download the final zip once it reports `MQL-0013 STATUS: complete`, and put it at the repo root.

## Step 3: score locally (I can run this)

```
python -B scripts/nl2mql_mql0013/mql0013_scoring.py verify --predictions-zip mql0013_predictions.zip
python -B scripts/nl2mql_mql0013/mql0013_scoring.py score  --predictions-zip mql0013_predictions.zip --run-name run_final
```

Outputs go to `outputs/nl2mql_mql0013/scoring_runs/run_final/scoring/`: `analysis.json`, `results_report.md` and `RUN_MANIFEST.json`.

## Registered analysis

- **Primary:** R3V vs R0V, exact two-sided McNemar (α 0.05), plus a paired bootstrap 95% CI. It is the single hypothesis.
- **Descriptive:**
  - R3 vs R0;
  - recovery effect per arm;
  - the 60% target (R3V ≥ 1,289/2,147), reported as a yes/no fact;
  - outcome taxonomy: incorrect, invalid, execution failure, timeout, generation failure, not comparable;
  - accuracy by Spider hardness, by gold-SQL structure label and by reference status.
- **Scoring dedup:** each distinct (item, answer text) is executed once and its outcome is shared across arms.
