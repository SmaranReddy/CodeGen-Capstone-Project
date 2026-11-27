# MQL-0010 final results: schema-feedback retry, signature diversity and FK type notes

**Status: VERIFIED. Scored 2026-11-21 on Atlas with the registered evaluator.**

| | |
|---|---|
| Task | NL→MQL |
| Questions | 716 held-out Spider-train questions from 18 corpus databases (amendment A1 holdout) |
| Model | Qwen/Qwen2.5-Coder-3B-Instruct, revision 488639f1ff808d1d3d0ba301aef8c11461451ec5, bf16 (emulated on a Kaggle T4), greedy, max_new_tokens 1024 |
| Execution | Read-only, cross-engine, on Atlas `codegen_spider1_holdout_mql0010` as `codegen_eval_readonly` |
| Analysis | Registration precedence A4 > A3 > plan > config (`registration_index.json` sha256 11b56c79…). The plan was not changed after the results were seen. |

## 1. Provenance and verification

**Archive.** `mql0010_predictions.zip` at the repo root, sha256 d18e7a85751151d0efff9b9432932dcbd37a62376be921cc0ec31945c0facebe.
- It is the only MQL-0010 prediction archive on disk. The other zip in `kaggle/` is the input bundle.
- Status is `complete`, with 8/8 passes:
  - first passes A0, D0 and E: 716 each;
  - matched delta: 104 calls (52 items × EP+EF);
  - retries: A0V 118, D0V 129, D0G 129, FULL 125.

**`mql0010_scoring.py verify`: `ok`.** It confirmed:
- the SHA256SUMS of the export;
- the package manifest is 87bea695… (the frozen package);
- the registration index is 11b56c79…;
- the exported plans equal the frozen plans;
- the per-arm prediction hashes equal the manifest;
- `verify_exports`:
  - 9 arms × 716 ids, equal to the registered holdout;
  - no duplicates;
  - every generation has status ok;
  - retry arms reuse their source's first answers byte for byte;
  - D0V and D0G retried the identical 129 items;
  - EP and EF were generated in the matched delta for exactly the 52 changed-prompt items and reuse E elsewhere;
  - prompt hashes match the plans.

**Configuration check (post hoc, `posthoc_config_check.json`): 60/60 checks pass.** No configuration was changed. It confirmed:
- the run's `config_sha256` 0f80cb87… equals the frozen package config;
- the config's model, revision, decoding, max_new_tokens and seed equal the A4 inference block;
- every pass fingerprint, recomputed from the frozen package, equals the export. The fingerprint covers model, revision, dtype, decoding, max_new_tokens, seed, batch size (8 for first passes, 1 for delta and retries), manifest, config and registration;
- first-pass and delta payloads equal the frozen plans;
- retry payloads, recomputed from the per-record retry prompts, equal the export;
- every record carries the frozen revision, config hash and its pass fingerprint;
- D0G's generic text is verbatim R5;
- the feedback mode and source of every retry arm match A4;
- delta records ran at batch size 1. First-pass batch sizes are 8, 4 (the final partial batch: 716 mod 8 = 4) and 1 (the OOM fallback).

**Kaggle run notes.** None of these blocks the results.
- All 8 passes ran in **one** Kaggle session (planned: 2). The environment was consistent and the model revision was checked (12 files).
- There were 61 batch-OOM events in the batch-8 first passes. Those prompts used the registered batch-1 fallback, which is recorded. There were 0 single-prompt OOMs.
- bf16 was emulated (T4, compute capability 7.5), as in MQL-0009.

**Atlas preflight (full parity, fail-closed): ok.**
- Server 8.0.39.
- 99/99 collections and 1,280 documents.
- Every collection's type-tagged digest equals the SQLite source.
- The account holds no write or admin action (`offending_actions = []`).
- Before scoring, the configured URI's user was checked to be `codegen_eval_readonly`. Credentials were not printed. Nothing was written to Atlas.

**Gold reference.** Computed once from local SQLite: 716/716 with status `ok`, so every item is comparable (sha256 1ff619f2…).

**Infrastructure failures: 0 in every arm.** Every non-correct outcome is attributable to the model.

**Post-hoc recount** (`descriptive_recount_posthoc.json`):
- the per-arm correct counts equal `analysis.json`;
- each arm has 716 evaluation records;
- the prediction hashes equal `RUN_MANIFEST.json`.

**Artifacts** (`outputs/nl2mql_mql0010/scoring_runs/run_final/`):
- `input_validation_report.json`
- `predictions/<arm>/`
- `reference/`
- `scoring/preflight/atlas_check.json`
- `scoring/arms/<arm>/evaluation_results.jsonl`
- `scoring/analysis.json`
- `scoring/RUN_MANIFEST.json`
- `scoring/results_report.md`
- `scoring/descriptive_recount_posthoc.json`
- `scoring/posthoc_noise_sensitivity.json`
- `scoring/posthoc_config_check.json`

**Post-hoc scripts.** All three are read-only; rerun them with `python -I -B <script> <run_dir|export_root> <out>`:
- `scripts/nl2mql_mql0010/analysis/posthoc_descriptive_recount.py`
- `scripts/nl2mql_mql0010/analysis/posthoc_noise_sensitivity.py`
- `scripts/nl2mql_mql0010/analysis/posthoc_config_check.py`

## 2. Accuracy (cross-engine execution match, correct / 716)

Every arm is registered over all 716 questions (`per_arm.n = 716`, required by `verify_exports`). EP and EF differ from E only on the 52 matched-delta items and reuse E's record on the other 664.

| Experimental arm | What it is | Correct | Total | Accuracy | Wilson 95% |
|---|---|---:|---:|---:|---:|
| A0 | zero-shot | 397 | 716 | 55.45% | 51.8–59.0% |
| A0V | A0 + schema-feedback retry | 415 | 716 | 57.96% | 54.3–61.5% |
| D0 | hybrid RAG k=3 (MQL-0009 D) | 413 | 716 | 57.68% | 54.0–61.2% |
| D0V | D0 + schema-feedback retry | 430 | 716 | 60.06% | 56.4–63.6% |
| D0G | D0 + generic retry (control) | 414 | 716 | 57.82% | 54.2–61.4% |
| E | D0 + signature diversity | 424 | 716 | 59.22% | 55.6–62.8% |
| EP | E prompt regenerated at batch 1 on 52 items (control) | 426 | 716 | 59.50% | 55.9–63.0% |
| EF | E + FK type notes on 52 items | 421 | 716 | 58.80% | 55.2–62.4% |
| **FULL** | **EF + schema-feedback retry** | **434** | **716** | **60.61%** | **57.0–64.1%** |

**52-item matched-delta subset (descriptive).** E 28/52, EP 28/52, EF 25/52.

### Outcome taxonomy (share of 716)

How the outcome categories of the unchanged evaluator (`dev1034_core`) map onto the columns:
- **invalid pipeline:** `invalid`, meaning no JSON, a structurally invalid pipeline, or an unsafe pipeline rejected before execution;
- **MongoDB execution error:** `failed` with failure_source model (`execution_error` or `atlas_tier_restriction`), plus `timeout` (0 in every arm);
- **executable but incorrect:** `incorrect`;
- **not comparable:** `not_comparable` (0, because every reference is `ok`).

| Arm | Invalid pipeline | Execution error | Executable but incorrect | Not comparable | Correct |
|---|---:|---:|---:|---:|---:|
| A0 | 13 (1.82%) | 9 (1.26%) | 297 (41.48%) | 0 | 397 (55.45%) |
| A0V | 11 (1.54%) | 9 (1.26%) | 281 (39.25%) | 0 | 415 (57.96%) |
| D0 | 6 (0.84%) | 15 (2.09%) | 282 (39.39%) | 0 | 413 (57.68%) |
| D0V | 5 (0.70%) | 19 (2.65%) | 262 (36.59%) | 0 | 430 (60.06%) |
| D0G | 6 (0.84%) | 15 (2.09%) | 281 (39.25%) | 0 | 414 (57.82%) |
| E | 4 (0.56%) | 19 (2.65%) | 269 (37.57%) | 0 | 424 (59.22%) |
| EP | 4 (0.56%) | 19 (2.65%) | 267 (37.29%) | 0 | 426 (59.50%) |
| EF | 4 (0.56%) | 21 (2.93%) | 270 (37.71%) | 0 | 421 (58.80%) |
| FULL | 3 (0.42%) | 22 (3.07%) | 257 (35.89%) | 0 | 434 (60.61%) |

**Where the errors are.** The dominant failure is a wrong result from an executable query: 36–41% of items in every arm.
- In D0, the incorrect results break down as 168 wrong row count, 62 wrong column count and 50 wrong values.
- On hybrid RAG (D0 → D0V), the schema-feedback retry cuts both error types it can see:
  - wrong row count falls 168 → 157 (−11);
  - wrong column count falls 62 → 53 (−9);
  - wrong values are unchanged (50 → 50).
- On the full stack (EF → FULL), most of the reduction is in wrong column count (51 → 41); wrong row count falls only 165 → 163.
- The retry slightly raises execution errors: 15 → 19 in D0V. Kept retries can turn a wrong-but-runnable query into a failing one. This costs no correct answers, because those items were already wrong.

## 3. Registered analysis (amendment A4, serial gatekeeping, family-wise α = 0.05)

The statistics below are taken verbatim from `scoring/analysis.json`:
- the significance test is the exact two-sided McNemar test;
- confidence intervals are question-level paired bootstrap 95% (10,000 resamples, seed 20261119);
- "a only / b only" are discordant pairs: questions one arm gets right and the other gets wrong.

| Step | Comparison | a only / b only | Δ (pp) | 95% CI (pp) | p | Holm p | Registered result |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | D0V vs D0 | 17 / 0 | +2.37 | [+1.26, +3.63] | 1.5e-5 | — | **Reject. Practically meaningful:** the CI lower bound of +1.26 is ≥ +1.0 pp. **PASS** |
| 2 | FULL vs D0 | 42 / 21 | +2.93 | [+0.84, +5.17] | 0.0111 | 0.0111 | **Reject. PASS** |
| 2 | D0V vs D0G | 16 / 0 | +2.23 | [+1.26, +3.35] | 3.1e-5 | 6.1e-5 | **Reject. PASS** |
| 3 | E vs D0 | 30 / 19 | +1.54 | [−0.28, +3.49] | 0.152 | 0.250 | Not rejected. **Inconclusive** (A4 power 0.24–0.43 at +2 pp) |
| 3 | EF vs EP | 1 / 6 | −0.70 | [−1.54, 0.00] | 0.125 | 0.250 | Not rejected. **Descriptive only** (52 items, power ≤ 0.71) |

Step 2 was opened because step 1 rejected. Step 3 was opened because **both** step-2 hypotheses rejected. All discordant pairs for EF vs EP necessarily come from the 52 delta items.

**Descriptive comparisons (registered as "no claim").**

| Comparison | a only / b only | Δ (pp) | 95% CI | p |
|---|---:|---:|---:|---:|
| D0 vs A0 (RAG vs zero-shot) | 98 / 82 | +2.23 | [−1.40, +5.87] | 0.263 |
| A0V vs A0 (retry on zero-shot) | 20 / 2 | +2.51 | [+1.26, +3.77] | 1.2e-4 |
| D0G vs D0 (generic retry) | 2 / 1 | +0.14 | [−0.28, +0.70] | 1.000 |

### Recovery statistics (R4: regressions from kept retries reported explicitly)

| Retry arm | Retried | Retry kept | Kept retry with 0 issues | Fixed by retry | **Broken by retry** | Net |
|---|---:|---:|---:|---:|---:|---:|
| A0V (source A0) | 118 | 55 | 50 | 20 | **0** | +20 |
| D0V (source D0) | 129 | 52 | 44 | 17 | **0** | +17 |
| D0G (source D0, generic) | 129 | 5 | 5 | 2 | **0** | +2 |
| FULL (source EF) | 125 | 49 | 43 | 13 | **1** | +12 |

Only 6 of the 129 flagged D0 items were correct before the retry. The validator trigger is precise.
- The schema-feedback retry raised the flagged set to 23/129 correct.
- The generic retry raised it only to 7/129.
- With the same trigger, the same items and the same acceptance rule, the generic text produced an acceptable retry (strictly fewer issues) on only 5 items, against 52 for the schema text.

### Fixes and regressions against the appropriate baseline (whole 716)

| Arm vs baseline | Fixes | Regressions | Net |
|---|---:|---:|---:|
| A0V vs A0 | 20 | 2 | +18 |
| D0V vs D0 | 17 | 0 | +17 |
| D0G vs D0 | 2 | 1 | +1 |
| D0V vs D0G | 16 | 0 | +16 |
| E vs D0 | 30 | 19 | +11 |
| EF vs EP | 1 | 6 | −5 |
| FULL vs EF | 14 | 1 | +13 |
| FULL vs D0 | 42 | 21 | +21 |
| D0 vs A0 | 98 | 82 | +16 |

Some whole-716 counts differ from the recovery table:
- A0V has 2 regressions and D0G has 1, against `broken_by_retry` 0;
- FULL vs EF has 14 fixes, against `fixed_by_retry` 13 (the extra one is `spider_train_01527`).

The difference is evaluator noise on byte-identical queries (§4), not retries.

## 4. Evaluation-noise check (post hoc, descriptive)

Reused records have byte-identical query text, so they should receive the outcome of their source. **10 of the reused-record comparisons disagree:**
- EP vs E: 4 on the 664 non-delta items;
- EF vs E: 2;
- A0V vs A0: 2;
- D0G vs D0: 1;
- FULL vs EF: 1;
- D0V vs D0: **0**.

All of them come from re-executing an identical pipeline on Atlas. The comparison flips between an "ordered match (tie-aware)" and wrong-values or order-mismatch, on items `spider_train_03801`, `06109`, `06110`, `06112`, `06132` and `01527`. This is the tie-order/document-order nondeterminism already noted for MQL-0008, and its size is about ±2 items per arm.

**Sensitivity** (`posthoc_noise_sensitivity.json`). Each reused record was given its source's outcome and the registered `analyze` was re-run:
- A0V 417, D0G 415, EP 424, FULL 432; other arms unchanged.
- Step 1 is identical.
- Step 2: FULL vs D0 is +2.65 pp, Holm p 0.023; D0V vs D0G is +2.09 pp, Holm p 1.2e-4. Both still reject.
- Step 3: E vs D0 is +1.54 pp, Holm p 0.30; EF vs EP is −0.42 pp, p 0.25.

**No registered conclusion changes.** The registered numbers above remain the official result. This sensitivity analysis is post hoc and descriptive.

## 5. What the results say about the improved RAG architecture

1. **The validator-triggered, schema-feedback retry (component V) is the one clearly effective improvement.**
   - On top of hybrid RAG it adds +2.37 pp (17 fixes, 0 regressions). It passes both the significance test and the pre-registered +1 pp practical bar.
   - It works on zero-shot too: +2.51 pp, descriptive.
   - It is cheap: about 18% of items get one extra batch-1 call.
2. **The gain comes from the schema-specific content, not from "asking again".**
   - The generic-retry control (D0G) used the same trigger, items, acceptance rule and budget, and gained only +0.14 pp.
   - D0V beats D0G by +2.23 pp, Holm p 6e-5.
   - The retry works because the validator's concrete schema-issue list gives the model something specific to correct.
3. **The full stack (FULL = diversity + FK notes + retry) is the best configuration at 434/716 (60.61%)** and beats the D0 baseline by +2.93 pp (Holm p 0.011).
   - Its gain over D0V (430) is only +4 net (31 fixes, 27 regressions). That gap is within noise and was not a registered test.
   - In practice, FULL ≈ D0V. The extra components add no demonstrated value beyond the retry.
4. **Signature diversity (E) is inconclusive.** It gives +1.54 pp with a CI spanning 0. Given the registered power of 0.24–0.43, this null is not evidence of no effect, but it cannot be claimed.
5. **FK type notes (EF vs EP) show no benefit and lean negative** (−0.70 pp; 1 fix vs 6 regressions on the 52 affected items). The comparison is descriptive, and about 2 of the −5 net are attributable to the noise in §4. There is no evidence of benefit, so the notes are not recommended as currently written.
6. **Retrieval itself (D0 vs A0) is +2.23 pp on this holdout and not significant** (descriptive). The holdout DBs' own corpus records were excluded from the retrieval view by design (R3). Absolute numbers are **not comparable** with MQL-0008 (892 items) or MQL-0009 (639 items), because the question sets differ.
7. **What remains unsolved.**
   - About 36% of items return executable but wrong results, mostly wrong row counts (filter, join or grouping semantics).
   - The validator cannot see these errors, because it checks schema conformance, not semantics.
   - Further gains need semantic signals (for example, result-shape checks or empty-result feedback), not more schema feedback.

**Recommendation.** Adopt **D0V (hybrid RAG + schema-feedback retry)** as the default pipeline. It reaches essentially the best accuracy, has the strongest evidence and has the fewest moving parts. Keep FULL only if a later, powered test shows that diversity adds value. The FK type notes are not recommended in their current form: no evidence of benefit, and the comparison is descriptive.

## 6. Exact commands run

```
python -B scripts/nl2mql_mql0010/mql0010_scoring.py verify --predictions-zip mql0010_predictions.zip
python -B scripts/nl2mql_mql0010/mql0010_scoring.py score  --predictions-zip mql0010_predictions.zip --run-name run_final
```

Nothing was regenerated. Kaggle was not launched. Unchanged: the ZIP, the approved corpus, the MQL-0008/0009 artifacts, the A4 registration and the Atlas data.
