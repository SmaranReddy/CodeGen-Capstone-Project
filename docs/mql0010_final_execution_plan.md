# MQL-0010: final execution plan (after audit corrections R1–R5)

Date: 2026-11-19.

**Status: REGISTERED (amendment A4), NOT EXECUTED. Stopped for human review.**
- No Kaggle run, model inference or Atlas connection has happened.
- Approved corpus v2.1 and every MQL-0008 / MQL-0009 artifact are unchanged.

---

## 1. Governing registration

`outputs/nl2mql_mql0010/preflight/registration_index.json` (sha256 `11b56c79…`) pins every document and source file. Precedence is **A4 > A3 > analysis plan > config**, for any aspect two documents both address.

| Document | sha256 | Governs |
|---|---|---|
| `scripts/nl2mql_mql0010/configs/mql0010_experiment.json` | `1c9e502e…` | Model, prompts, corpus, split, context budget. Its arms, compute, source hashes and test order are superseded. |
| `outputs/nl2mql_mql0010/preflight/analysis_plan.json` | `a6294743…` | Metric, sensitivity analyses S1–S3, integrity rules. Its test order is superseded. |
| `outputs/nl2mql_mql0010/preflight/split_registration.json` | `fd0c8e18…` | Holdout and dev, leakage audit (A1 embedded) |
| `…/amendment_A2_2026-11-19.json` | pinned | Dev proxy recomputed (no registration change) |
| `…/amendment_A3_2026-11-19.json` | pinned | Step 1 only |
| **`…/amendment_A4_2026-11-19.json`** | `d463a399…` | **Arms, batching, test sequence, practical criterion, compute (R1–R5)** |
| Source `scripts/nl2mql_mql0010/src/*.py` (9 files) | pinned in the index | `tests/test_mql0010_registration.py` fails on any drift |

---

## 2. Arms (9)

| Arm | How produced | Batch | Prompt / second turn |
|---|---|---:|---|
| A0 | full first pass | 8 | frozen zero-shot `nl-mql-spider1-v1` |
| D0 | full first pass | 8 | MQL-0008/0009 hybrid block, k = 3 |
| E | full first pass | 8 | D0 ranking, signature-diversified selection (pool 50) |
| **EP** | matched delta (control) | 1 | E's prompt, regenerated for exactly the items whose EF prompt differs; all other items reuse E |
| EF | matched delta | 1 | E + type-incompatible join-key notes. Changed items are generated in the same pass as EP; all others reuse E |
| A0V | retry on A0 | 1 | schema feedback (validator issues) |
| D0V | retry on D0 | 1 | schema feedback |
| **D0G** | retry on D0 | 1 | **generic feedback:** "Your previous query may contain mistakes. Review it and return a corrected query." Same trigger, items, acceptance rule, budget, model and decoding as D0V. |
| FULL | retry on EF | 1 | schema feedback |

**Shared settings for every arm:**

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1…` |
| Precision and decoding | bf16 (emulated on the T4), greedy |
| Output | `max_new_tokens` 1024 |
| Seed | 42 |
| Hardware | Kaggle T4 only |
| Retry | one retry at most |
| System prompt | the same in both turns |

**Retry rules:**
- **Trigger:** the evaluator's parser fails, or ≥ 1 static schema issue is found.
- **Acceptance:** the retry is kept only if it has strictly fewer issues.
- **Unparseable first answers** get the same format message in D0V and D0G.

---

## 3. Registered tests (A4)

Serial gatekeeping at family-wise α = 0.05. Each test is an exact two-sided McNemar test on per-question correctness, with a question-level bootstrap 95% CI (10,000 resamples, seed 20261119).

| Step | Gate | Hypotheses | Estimand |
|---|---|---|---|
| **1** | — | **D0V vs D0** | Effect of validator-triggered retry with schema feedback, on identical first answers |
| | | **Practical criterion (R4):** called *practically meaningful* only if it rejects **and** the CI lower bound is ≥ **+1.0 pp** | |
| **2** | step 1 rejects | Holm over {**FULL vs D0**, **D0V vs D0G**} | Package effect; effect of schema-specific feedback *content* |
| **3** | both step-2 hypotheses reject | Holm over {**E vs D0**, **EF vs EP**} | Diversified selection; join-key notes, batch-matched |

**Descriptive only:**
- D0 vs A0;
- A0V vs A0;
- D0G vs D0;
- per retry arm: retried / kept / zero-issue retries / fixed and broken by retry;
- outcome taxonomy;
- per-database and per-family accuracy.

**Power** (simulated from MQL-0009 development evidence; `docs/mql0010_pre_execution_audit.md` §3–4, §9):

| Comparison | Power |
|---|---|
| Step 1 | 0.79 at a 5% repair rate (+1.1 pp); about 1.0 at ≥ 10% |
| D0V vs D0G, at the Holm worst case α/2 | 0.53–0.99 for repair rates of 20% vs 10%; 0.21–0.66 for 10% vs 5% |
| E vs D0 | 0.24–0.43 at a true +2 pp: a null result is inconclusive |
| EF vs EP | ≤ 0.71 even at +8 net: descriptive in practice |

---

## 4. Holdout and retrieval view (R3)

- **Holdout:** 716 items over 18 databases (`holdout_items.jsonl`, ids sha256 `186b080a…`).
- **Retrieval view:** approved corpus v2.1 minus the 18 holdout DBs = 2,912 records over 105 DBs (ids sha256 `cfde5e9c…`).

**The view is enforced three times:**
1. `holdout_view.load_view` checks the corpus hash, the registered view id hash and count, and that no holdout DB is present.
2. `holdout_view.assert_docs_exclude` re-checks the built documents.
3. `planner.build_plan(..., forbidden_db_ids=<18 DBs>)` refuses any holdout-DB document before an index is built.

The 13 question-level L3–L5 retrieval exclusions apply identically in every RAG arm. Those 10 items **stay in the evaluation**; sensitivity S2 drops them.

---

## 5. Exact execution sequence (every step needs your explicit authorisation)

**1. Build the Kaggle package (local, CPU).**
- Derive the holdout context exactly as in `preflight/dry_run.py`.
- Build the frozen prompts with the gold-leak check.
- Build plans A0 / D0 / E / EF from `load_view` docs with `forbidden_db_ids`, using the **pinned Qwen tokenizer** instead of len/4.
- Write a manifest and SHA256SUMS. Add a notebook that implements:
  - first passes at batch 8 with the OOM fallback;
  - the matched delta pass;
  - the four retry passes at batch 1, through `driver.py`;
  - a resume-safe checkpoint per pass;
  - an export with hashes.
- Re-run all tests.

**2. Kaggle generation** (≈ 10.0 T4-hours; 1–2 sessions):

| Pass | Generations | T4-hours |
|---|---|---:|
| First passes A0 / D0 / E | 3 × 716 | 7.2 |
| Matched delta | 104 (52 EP + 52 EF) | 0.4 |
| Retries for A0V, D0V, D0G, FULL | ≈ 620 (dry-run mock: 689) | 2.4 |

**3. Atlas provisioning** (separate authorisation):
- Load the 18 holdout DBs with the insert-only wrapper pattern of `provision_heldout_atlas.py` into a new database, under a new contract pin.
- Grant `find` / `listCollections` / `killCursors` to `codegen_eval_readonly` on that database. This is a manual Atlas UI step for you.

**4. Scoring** (local Windows, about 30–40 minutes, one session):
- one full-parity, read-only preflight;
- `scoring.verify_exports`, which fails closed on any reuse or contract violation;
- `scoring.score_all`, through `scoring.atlas_evaluator` wrapping the unchanged `dev1034_core.evaluate_prediction`;
- `scoring.analyze` in the A4 order.

**5. Records:** a ledger entry (verified only after hash and recount checks) and presentation notes.

---

## 6. Evidence that the pipeline works (no inference)

**Tests: 50 passed** (`outputs/nl2mql_mql0010/test_results_A4.txt`):
- `tests/test_mql0010_components.py`: validator, retriever, recovery, driver;
- `tests/test_mql0010_corrections.py`: R2, R5, the scoring adapter, the practical criterion, Holm;
- `tests/test_mql0010_registration.py`: R1 hash pinning, A4 ↔ code consistency, R3 view enforcement and tamper tests;
- `tests/test_mql0010_planner.py`: 639/639 reproduction of the MQL-0009 D plan with byte-identical prompts.

**Mocked dry run** (`preflight/dry_run.py` → `outputs/nl2mql_mql0010/dryrun/dry_run_report.json`). The real registered holdout and view were used, with a fake model and a fake evaluator:

| Check | Result |
|---|---|
| Items | 716 |
| View | 2,912 records; hash verified; 0 holdout-DB documents |
| Budget drops (len/4 counter) | 0 |
| Matched delta | 52 items (race_track, shop_membership) → 104 generation calls |
| Same retried items in D0V and D0G | yes (168 = 168 in the mock) |
| `verify_exports` | ok |
| Analysis | ran through all gates |

The mock "accuracies" are meaningless by construction.

---

## 7. Not done (deliberately)

- The Kaggle package and notebook are not built (step 1).
- No Atlas provisioning or grant (step 3).
- No ledger entry for MQL-0010 or its split audit; offered, not made.
- Audit items R6–R9 are documentation recommendations; they are not implemented beyond the R4 correction.
