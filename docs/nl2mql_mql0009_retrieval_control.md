# MQL-0009: NL→MQL retrieval-control experiment (registered plan, package, scoring readiness)

**Status (2026-11-18): PARTIALLY READY.**

| Part | State |
|---|---|
| Kaggle inference package | **Ready for manual upload and execution.** Built, hash-pinned, tested. Nothing has been run. |
| Scoring | **BLOCKED** until the 16 held-out databases are provisioned on Atlas. |
| Predictions and scores | None exist. No metric has been produced. |

## 1. Question and design

Does the MQL-0008 gain (zero-shot 415/892 → hybrid RAG 457/892, +4.71 points, McNemar p = 0.0018) come from **relevant**
retrieval, or simply from showing Qwen three demonstrations? MQL-0008 had no random or fixed control.

| Run | Demonstrations | Selection |
|---|---|---|
| `A_zero_shot` | 0 | none |
| `B_random_s1`…`s5` | 3 | `retrieval.Retriever(method="random", seed=s)`, seeds 20261181–20261185 (MQL-0005/0007 random retriever, unchanged) |
| `C_fixed` | 3 | `spider_train_02033` (gas_company), `spider_train_01967` (products_for_hire), `spider_train_03829` (e_learning), same order, every question |
| `D_hybrid` | 3 | MQL-0008 `hybrid_masked` (BM25 k1 1.5, b 0.75 + 1.0 × cue/pattern Jaccard), unchanged |

That is 8 runs × 639 questions = 5,112 predictions. The only intended difference between the arms is how the three
demonstrations are chosen. Everything else is shared:
- the same model (`Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1…`, bfloat16, greedy, `max_new_tokens` 1024, batch 8,
  seed 42);
- the same system prompt, schema text and question;
- the same reference-block renderer and insertion point;
- the same eligible corpus view, 1,500-token context cap, budget loop and evaluation-specific exclusions;
- the same output parser on Windows.

None of the following is introduced: validation or retry, schema warnings, new MongoDB rules, AST retrieval,
structure-aware gating, converter changes or corpus changes.

**Fixed-example selection:**
- Procedure: a seeded draw (seed 20261186) over the sorted eligible ids, one example per database.
- It uses only corpus attributes, plus a question-level leakage gate.
- Zero draws were skipped by the gate, so the selection did not depend on the evaluation questions at all.
- Evidence: `outputs/nl2mql_mql0009_retrieval_control/preflight/fixed_selection.json`.

## 2. Evaluation split and leakage audit

**Split.** The split is an existing, approved database-level one: spider-sql-v1 `val_internal`.
- It has 16 whole Spider-train DBs, picked by a seeded shuffle of db_ids with seed 20261027.
- Membership comes from the audited spider-pool-v3 `val_internal`.
- Records (question text, gold SQL, execution hashes) come from spider-sql-v1, the same source as the MQL-0006/0008 protocol.

**Funnel:** 797 (spider-sql-v1) → 744 (spider-pool v1) → 734 (v2: Tier A 4 + Tier B 6) → **639** (v3: 4 adjudicated
semantic defects + 91 detector hits). All 639 have tier `exec_ok` with execution hashes. Four questions differ from v3 only
in whitespace or punctuation; the spider-sql-v1 text is used.

**By database:**

| DB | n | DB | n | DB | n | DB | n |
|---|---:|---|---:|---|---:|---|---:|
| hr_1 | 85 | chinook_1 | 71 | cre_Docs_and_Epenses | 66 | manufactory_1 | 63 |
| inn_1 | 58 | small_bank_1 | 44 | gymnast | 35 | tracking_software_problems | 35 |
| ship_1 | 32 | tracking_share_transactions | 30 | machine_repair | 28 | swimming | 27 |
| election_representative | 19 | epinions_1 | 18 | phone_market | 14 | company_employee | 14 |

**By hardness:** easy 181, medium 260, hard 111, extra 87.

**Protected sets:**
- No evaluation DB is a Spider-dev DB or an MQL-0008 evaluation DB.
- No evaluation id is a dev id.
- Spider dev and test question and SQL files were not opened.
- The 65-item test partition is untouched.

**Leakage gates.** These were registered before the audit ran. The audit compared the 639 questions against the 3,445
corpus records.

| Gate | Action | Result |
|---|---|---|
| L1 database overlap | STOP | 0 |
| L2 id overlap / Spider source-index overlap | STOP | 0 / 0 |
| L3 exact raw question | exclusion | 4 pairs |
| L4 normalised question | exclusion | 5 pairs |
| L5 token Jaccard ≥ 0.8 (MQL-0007/0008 rule) | exclusion | 17 pairs |
| L6 normalised gold SQL = corpus SQL | report only | 7 items (generic SQL on shared table names in other DBs) |
| L7 masked SQL skeleton match | report only | 383 of 639 items have ≥ 1 structural twin |

**How the gates were applied:**
- L3–L5 give **17 (question, corpus example) exclusion pairs across 13 questions**.
- They are applied after ranking, identically in every demonstration run. The corpus is unchanged.
- SQL-based gates are report-only, because gold must never drive retrieval.
- The 7 L6 items get a pre-registered sensitivity analysis, S3.

**Eligible corpus view.** No corpus database is an evaluation database, so the filtered view equals the full approved
corpus: 3,445 records, view ids sha256 `966830c1…`, corpus sha256 `96552f06…dd199bf`, file unmodified.

**Plan checks:**
- K = 3 holds for every question in every demonstration run.
- 0 over-budget drops.
- Longest prompt is about 3.36k tokens, plus 1,024 new tokens, within the 8,192-token context.

**Evidence:** `outputs/nl2mql_mql0009_retrieval_control/preflight/{eval_split_registration,leakage_audit}.json`.

## 3. Precision and hardware (discrepancy resolved)

MQL-0008's run configs record `dtype` bfloat16, GPU Tesla T4, torch 2.11.0+cu130, transformers 5.20.2 and
`bf16_supported: true`. A T4 has compute capability 7.5 and no native bf16, so PyTorch reports bf16 support through
**emulation**.

**Decision:** keep bfloat16, because D must reproduce MQL-0008 and fp16 would break comparability.
- The notebook accepts only a Tesla T4.
- It records `compute_capability` and `bf16_native` (= false) in every run config and prediction.
- It never substitutes another precision.
- All 8 runs must share GPU type, dtype, torch and transformers. A resume in another environment is refused.

MQL-0008 recorded `resolved_revision: null`. MQL-0009 adds an independent check: every Hugging Face download-metadata file
must name commit `488639f1…`, and `HfApi().model_info(...).sha` must equal it.

## 4. Kaggle package

**Package files:**
- Directory: `kaggle/nl2mql_mql0009_retrieval_control/` (33 files).
- Upload archive: **`kaggle/nl2mql_mql0009_retrieval_control_bundle.zip`**, 1,501,208 bytes, SHA-256
  `2ed81c46874c828b139b0eac87ea087f9b0a2f97396fac9307fd9319a71515ce`.

**Frozen identities:**
- `package_manifest.json`: SHA-256 `2a4f22e0eb526540e70f62f561d0314b1b2faa5c12917196a953d9f98aecface`. The scoring
  adapter pins this.
- `configs/experiment.json`: `3f052e98…`.
- Analysis plan: `ecb7479f7909a65f726fadce1e49dda4f6cc472cbcf9c2eaa3f2d59eb2bb4144`.
- Evaluation ids: `6f1425f7…`.
- Frozen zero-shot prompts: `7e765120…`.
- Per-run plan hashes: in the config.

**Contents:**
- the notebook;
- `src/`: `mql0009_core.py`, plus byte-identical MQL-0008 `mql0008_core.py`, `retrieval.py`, `patterns.py`, `rag_block.py`;
- the frozen config, including seeds, fixed ids, exclusions, session plan and analysis-plan hash;
- the approved corpus and its features, and the eligible-view ids;
- evaluation inputs (`id`, `db_id`, `question` only) and frozen zero-shot prompts;
- Atlas-form schema context;
- 8 frozen plans;
- the question-level leakage report;
- the prediction-record JSON schema;
- `requirements-kaggle.txt`, README, self-tests, provenance copies of the builder, the notebook maker and stats;
- `package_manifest.json` and `SHA256SUMS.txt`.

**Never in the package:** gold SQL, gold rows, hardness, query families, SQLite files, credentials, `.env` files, model
weights, Spider-dev data.

**Schema context.** The held-out DBs are not on Atlas, so the `nl-mql-spider1-v1` context was derived offline. It reads
the SQLite files with exactly the `import_atlas.py` value mapping: one collection per table, columns in PRAGMA order, rows
in rowid order, the same `bson_type()`, and the first two rows as sample documents. That makes it a **provisioning
contract**: `outputs/nl2mql_mql0009_retrieval_control/scoring_contract/atlas_mapping.json` holds 69 collections, 16,885
documents, and a type-tagged row digest and SQLite hash per collection.

**Offline context = Atlas path (evidence).** `derive_context` was run on the 16 Spider-dev DBs whose context was
originally read from live Atlas by `import_atlas.py` and `build_samples`. wta_1 (2 invalid-UTF-8 strings) and the 3
held-out test DBs were left out. Result: 67/67 collections are identical in fields and field order, declared types,
document counts, and both sample documents with their key order. Test:
`tests/test_nl2mql_mql0009.py::test_offline_context_equals_the_atlas_derived_context_on_dev_dbs`.

### Exact Kaggle steps

1. **Create the dataset:** *Datasets → New Dataset*, upload the bundle zip, keep it **private**. No slug is hard-coded.
2. **Import the notebook:** *File → Import Notebook* → `nl2mql_mql0009_retrieval_control.ipynb`, then *Add data* → the
   dataset.
3. **Settings:**
   - **GPU T4 x2** (one card used);
   - **Internet On** (public model download at the pinned revision; no token);
   - Persistence *Files only*;
   - no secrets attached.
4. **Session 1:** `SESSION = 1`, `RESUME_FROM = None` → *Save & Run All (Commit)*.
   - Runs `A_zero_shot`, `D_hybrid`, `B_random_s1`, about 6.5 h. D and one random seed share a session, so the primary
     contrast has a same-session pair.
   - Afterwards: *Pin to original environment*, then download `mql0009_predictions.zip`.
5. **Session 2:** *Add data → Notebook Output* of session 1, `SESSION = 2`, `RESUME_FROM = "auto"`.
   - Runs `C_fixed`, `B_random_s2`, `B_random_s3`, about 7.3 h.
6. **Session 3:** the same, with `SESSION = 3`, attaching session 2's output.
   - Runs `B_random_s4`–`s5`, about 4.9 h.
   - The final zip must say `"status": "complete"` with 5,112 completed predictions.

**Total** is about 18.6 T4-hours, based on MQL-0008 rates; that is about 60% of a 30 h weekly quota.
- Each session stops starting new batches after 11 h and still exports.
- Any interrupted version is resumed with the same `SESSION` and `RESUME_FROM = "auto"`. Completed questions are never
  regenerated.
- If the pinned environment is lost (a resume is refused), regenerate all 8 runs in one new environment under a new
  output. Never mix environments.

## 5. Prediction artifact contract

`mql0009_outputs/predictions/<run>/predictions.jsonl` holds one record per question; the JSON schema is in the package.

**Identity:**
- `prediction_id` (`MQL-0009:<run>:<id>`), `experiment`, `arm`, `arm_family`, `random_seed`, `id`, `db_id`.

**Model output:**
- `raw_text`: the only scored field.
- `normalized_mql` and `parse_status`: informational.
- `status`, `generation_failure`, `error`.

**Retrieval provenance:**
- strategy, retrieved ids, ranks, scores, source DBs, exclusions, over-budget skips, context tokens, corpus and view
  hashes, retrieval fingerprint.

**Generation and runtime:**
- `prompt_sha256` and token counts;
- model id and revision, `precision`, `generation_config`;
- `runtime`: GPU, compute capability, bf16_native, torch, transformers;
- `config_sha256` and `fingerprint`.

**How outputs are written:**
- Checkpoints are append-only and fsynced. Final files are written atomically.
- A completed prediction is never overwritten or dropped; `safe_write_predictions` refuses.
- `MQL0009_PREDICTIONS_MANIFEST.json` gives expected versus completed counts per run and checksums.
- `MQL0009_STATUS.json` reports environment consistency.

## 6. Windows scoring

The adapter is `scripts/nl2mql_mql0009/mql0009_scoring.py`. It is a thin wrapper around the **unchanged** MQL-0006
evaluator: `compute_reference`, `run_evaluation` / `evaluate_prediction`, `atlas_preflight`, and the read-only
`local_atlas_scoring` client. The outcome classes are unchanged.

**What `score` does:**
- verifies the export against the pinned package;
- computes the gold reference from local SQLite once per run directory and freezes it by hash;
- runs the fail-closed read-only Atlas preflight with **full parity mandatory**;
- scores all 8 runs on the same items and the same snapshot;
- writes paired per-question results and the analysis.

**Pre-registered analysis** (`preflight/analysis_plan.json`, sha `ecb7479f…`):

| Level | Comparisons and tests |
|---|---|
| Primary H1 | D vs the per-question mean of the 5 random seeds: paired bootstrap 95% CI and sign-flip test |
| Primary H2 | D vs C: exact McNemar and bootstrap |
| Primary correction | Holm over H1 and H2 |
| Secondary | D, C and each seed vs A; D vs each seed; R̄ vs A |
| Exploratory | per-arm rates (invalid, execution error, timeout, executable-but-incorrect, not comparable, generation and infrastructure failures); query family from local gold SQL only; hardness; database; transition matrices; fixes and regressions; seed mean, SD and pairwise agreement; prompt tokens |
| Sensitivity | S1 reference ok; S2 without exclusion items; S3 without the 7 L6 items |

Seeds are never pooled as independent items.

### Scoring readiness (offline, `readiness` command)

`outputs/nl2mql_mql0009_retrieval_control/scoring_readiness/scoring_readiness.json` reports **BLOCKED**, with exactly one
blocker.

- **OK:**
  - the package pin;
  - 639 local gold records;
  - the provisioning contract (69 collections, 16,885 documents, SQLite hashes unchanged);
  - pymongo;
  - the registered analysis plan.
- **Blocker, data provisioning:**
  - The last recorded Atlas inventory (`data/interim/spider1/atlas_collections.json`) lists 80 collections for 20 Spider-dev
    DBs, and **0 of the 69 required collections**.
  - Live Atlas was not contacted (forbidden in this task).
  - Storage is not a concern: about 1.4 MB as SQLite, against 54.4 of 512 MB recorded usage.

**Needed before scoring.** These are user decisions and actions, not done here.
1. **Insert the 16 DBs** into a **separate** Atlas database with a writable account; never use the evaluation account.
   - Use the same value mapping as `import_atlas.py`: INTEGER→int, REAL→double, TEXT→string, NULL→null, rowid order,
     `<db>__<table>`.
   - The proposed database is `codegen_spider1_heldout`; any name works via `--atlas-db`.
   - **Do not use `scripts/spider1/import_atlas.py`.** It drops and recreates collections in `codegen_spider1`, the MQL-0006
     to 0008 evaluation database, and rewrites `data/interim/spider1/atlas_collections.json`, which `prompt_spider1` and
     `readiness` read.
   - The safe route is `scripts/nl2mql/seed_atlas_spider_dev.seed()`. It never drops or overwrites, aborts if any target
     collection exists, refuses the read-only URI, and checks the digest after inserting.
   - Its CLI only accepts the Spider-dev bundle (`verify_bundle`), so this contract needs a thin wrapper, dry run by default.
     That wrapper has not been written; it is a user decision.
2. **Grant `find`** on that database to the read-only role.
3. **Run `score`.** Its preflight refuses any count or digest mismatch against the contract.

### Commands for later (Windows, repo root)

```
python -B scripts/nl2mql_mql0009/mql0009_scoring.py readiness
python -B scripts/nl2mql_mql0009/mql0009_scoring.py verify --predictions-zip <path>\mql0009_predictions.zip
python -B scripts/nl2mql_mql0009/mql0009_scoring.py score  --predictions-zip <path>\mql0009_predictions.zip --run-name run01 --atlas-db codegen_spider1_heldout
python -B scripts/nl2mql_mql0009/mql0009_scoring.py analyze --run-dir outputs/nl2mql_mql0009_retrieval_control/scoring_runs/run01
```

- `MONGODB_URI` must be the read-only account. `score` aborts if the user has any write or admin privilege.
- If the three sessions' zips are given together, every run must come from complete, identical sources.

## 7. Verification evidence

- **Package self-tests:** `kaggle/nl2mql_mql0009_retrieval_control/tests`, 7/7 with the pinned tokenizer. That includes
  full re-derivation of all 8 plans, which matched the frozen plans.
- **Builder check:** the D_hybrid plan equals `mql0008_core.build_plan` row for row on all 639 items.
- **Repo tests:** `tests/test_nl2mql_mql0009.py`, 37 test cases (32 functions, one parametrised over 6 inputs), including the offline-context = Atlas-context check; full project suite 2026-11-18: 650 passed, 13 skipped, 0 failed. They include:
  - split and leakage gates;
  - view stability, random reproducibility and uniformity, fixed consistency, hybrid = MQL-0008;
  - K=3 and budget shortfall, prompt equivalence;
  - parser verbatim, persistence and checkpoint safety;
  - precision and revision checks, manifest tampering, no gold or secrets in the zip;
  - the stats unit of analysis.
- **Mocked notebook dry run** (fake GPU, fake generator with one simulated OOM, real pinned tokenizer through a
  subprocess):
  - session 1 is interrupted, so its export is partial;
  - the resume completes 5,112/5,112 with zero regenerations and carries earlier records forward unchanged;
  - an unregistered GPU and a different torch version are refused;
  - export → Windows `verify` passes, and a tampered zip and a partial zip are rejected;
  - fake read-only Atlas scoring with full parity (documents served from SQLite) completes all 8 runs, and offline
    `analyze` reproduces it;
  - a parity mismatch fails closed.

## 8. Limitations and open items

- **Scoring is blocked** on Atlas provisioning, as in section 6.
- **GPU budget.** About 18.6 T4-hours over three sessions. Estimates come from MQL-0008 rates; chinook_1 and hr_1 have
  long schemas.
- **Batch effects.** Batch composition and the OOM fallback can change greedy outputs slightly under bf16, as in MQL-0008.
  It is unavoidable when prompts differ by arm, and it is recorded per record.
- **Atlas tie order.** Same-day scoring of all 8 runs is required: tie-order noise across days was seen in MQL-0008, with
  6 items.
- **Retrieval caveats:**
  - Random retrieval is uniform over the view after dedup, budget and exclusion filters, which removed nothing in practice.
  - The fixed set is a single seeded draw: three order_by/join+where examples, no group_by. It is one realisation, not an
    average over fixed sets.
- **Leakage caveats:**
  - L6: 7 items have gold-SQL twins in other databases. Their questions differ and they are not excluded, by policy. S3
    reports without them.
  - L7: structural twins are expected and are exactly what relevant retrieval should exploit.
- **Corpus and model caveats:**
  - Corpus labels are AI-reviewed, not human-reviewed, and were validated locally, never on Atlas.
  - Qwen pre-training exposure to Spider affects all arms equally.
- **Evaluation-set caveat.** The held-out items are Spider-train paraphrase pairs, so two questions often share a gold SQL.
  That is fine within one arm, because the unit is the question. This is a different item set from MQL-0008, so absolute
  accuracies are not comparable with 415/457.
