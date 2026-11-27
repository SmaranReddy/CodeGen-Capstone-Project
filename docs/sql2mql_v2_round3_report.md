# sql2mql-v2: verifier-3 and the round-3 G4 re-measurement

**Status (2026-10-31): round-3 G4 PASSED on the pre-registered measure.** The population-weighted false-accept rate is 2.38%, with a Wilson 95% interval of 1.37–4.12%.

- **Robustness check:** the unweighted figure also passes, at 15/542 with an upper bound of 4.52%. It over-represents the hard strata, so it is not the population rate. The pass therefore does not depend on the choice of estimator.
- **Phase 1 status:** G1–G3 and G5–G6 pass as frozen on 2026-10-30. G4 passes the round-3 criterion you specified, pre-registered before the run: a new stratified population and a weighted Wilson bound, replacing the original "up to 200 DEV labels". On that basis Phase 1 is passed.
- **The pilot has NOT been started.** It waits for your decision (§7).

**Multi-table joins (J2) remain weak:** 7/104 labels with a false accept (6.7%, Wilson 3.3–13.3%). J2 is about 9% of the estimated accepted population (≈357 of ≈3,832). Read §8 before relying on J2 labels.

All measurements are local: SQLite, and MongoDB **8.0.34** in Docker on 127.0.0.1:27018 (`mongo:8.0.34`, digest `sha256:b9924c33…fa142`). They are label-validation measurements, not model metrics. Nothing connected to Atlas.

What this round did not touch:
- the 300 pilot ids (not converted, verified or read);
- the 892-item MQL evaluation set and the 65-item test partition (not read);
- training or retrieval corpora (none built);
- inference and fine-tuning (none run);
- converter `sql2mql_v2.py` (unchanged);
- every V1 artefact and every verifier-2 artefact (unchanged).

## 1. What this round is for

Phase 1 gate **G4** measures the verifier's sensitivity: the share of accepted labels for which the verifier accepts a deliberately wrong single-edit mutant that a stronger, differently designed oracle battery shows is not equivalent. It passes when the Wilson 95% upper bound is ≤ 5%. G4 is not the converter's error rate. See `docs/sql2mql_v2_quality_assessment.md` §5.4.

| Round | Verifier | Labels | Labels with a false accept | Wilson 95% | Gate |
|---|---|---|---|---|---|
| 1 (diagnosis set) | verifier-1 | 200 | 27 (13.5%) | 9.45–18.93% | failed |
| 2 (pre-registered holdout) | verifier-2 | 161 | 7 (4.35%) | 2.12–8.70% | failed |
| **3 (new pre-registered stratified sample)** | **verifier-3** | 542 | 15 (weighted 2.38%; unweighted 2.77%) | weighted 1.37–4.12% (unweighted 1.68–4.52%) | **passed** (§6) |

Per the round-2 stop rule, the round-2 holdout was not re-used. Its `mutants.jsonl` was never opened after its run. Two things from it were seen: its class-level summary (multi-join `inner_to_left` / `lookup_field_swap` / `cmp_ne_to_eq` / `null_guard_drop`, and one LIMIT-1 `sort_drop`), and the failing-label id list printed in its log. Neither was used to design verifier-3. The tools refuse those labels (`mutant_study_v3.holdout2_label_ids()`; `run_v3.py dev3` excludes them).

## 2. Diagnosis data (what verifier-3 was developed against)

Only the following were used:

1. **The 200 iteration-1 diagnosis labels** (DEV; `outputs/sql2mql_v2/mutant_study/diagnosis_labels.json`), re-measured with verifier-2. This is the in-sample run that had been killed at 60/200 on 2026-10-30. Resumable now, it ran to completion: `outputs/sql2mql_v2/diag_v2_insample/`.
2. Hand-built synthetic databases shaped like the reported escape classes (`synthetic_v3.py`, 14 queries, 105 mutants).
3. The existing adversarial suite and negative controls (G2/G3).

## 3. Verifier changes (verifier-3, `sql2mql-v2-verifier-3`)

Verifier-3 lives in new files: `verifier3.py`, `fixtures3.py` and `index_real.py`.
- `verifier.py` and `v2_fixtures.py` are unchanged; their hashes match `outputs/sql2mql_v2/rename_manifest.json`.
- Verifier-3 imports and re-runs **every** verifier-2 check and fixture family, with the same seeds. Nothing was relaxed: no tolerance, comparator rule, family or rejection rule of verifier-2 was changed.
- Its fixture and reorder collections live in their own databases (`sql2mql_v3_fixtures*`, `sql2mql_v3_reorder*`), so verifier-2 and verifier-3 runs never share collections.

| Addition | What it does | Escape class it targets |
|---|---|---|
| `v3_reorder` | Real data again, with each referenced table inserted in **reversed** order (explicit null) and in a **seeded shuffled** order (absent-field NULLs). The SQL truth is order-free apart from tie order, which the comparison already treats as free | A dropped `$sort` before `$limit 1` passing because natural order equals sort order; `$first`/`$last` dependence |
| `<fam>_rev`, `<fam>_shuf` | Six verifier-2 families (`sample_a`, `sample_b`, `dup`, `literal_hits`, `boundary_equal`, `orphans`) with the **stored** rows inserted into MongoDB reversed / shuffled | Same, on fixture data. Reordering before the SQLite insert would have been undone for `INTEGER PRIMARY KEY` tables (rowid = key); a unit test found this, and the reorder is applied after read-back |
| `dangle<i>`, `dangle_null<i>` | **In-place** orphaning of one join-key column at a time on every other witness row: the value becomes absent from every joined table, or NULL. No copy, no fresh primary key | Multi-join `inner_to_left`. Verifier-2's `orphans` copied rows with fresh keys, so in a chain A→B→C the orphaned middle-table copies were unreachable from A |
| `dangle_eq<i>`, `dangle_null_eq<i>` | The same edits on `boundary_equal` rows, where the WHERE literals hold | Queries whose WHERE matches no real row, so they have no witnesses (e.g. the `club_1` count) |
| `boundary_{minus,equal,plus}_alltrue` | Every predicate atom of a table is set **true**, except one cycling target atom at literal−1 / literal / literal+1 | `x > 2000` vs `> 2001` inside a conjunction. Verifier-2's boundary rows kept the other atoms unchanged or false, so `director = 'James Cameron' AND year = 2001` never occurred |
| `keyshift` | Integer join-key values shifted by a per-join-group offset, consistently on both sides of every edge, so ids of different tables stop coinciding | Joins on the wrong column pair that agree only because ids are 1..n everywhere |
| `null_keys_both` (+ absent-field run) | NULL join keys on both sides of every edge in the same rows | `$lookup` matching null/missing to null/missing (`null_guard_drop`, `$ne: null` → `$eq: null`) |
| `v3_lookup_keys` | Every `$lookup` `localField` must exactly (case-sensitively) name a column of the start table or `<as>.<column>` of an earlier lookup, and `foreignField` must be a column of the `from` table | `lookup_field_swap` onto a field that does not exist, which MongoDB treats as a silent null = null cross join |
| Key indexes (`index_real.py`, `RealDbs.get`) | Single-field indexes on PK / FK / `*id*` columns of collections with more than 2,000 documents in the local real-data import | Speed only. An index never changes an aggregation result; it removes timeouts on large `$lookup`s |

### 3.1 Checks before freezing (in-sample or synthetic; NOT gate evidence)

| Check | Verifier-2 | Verifier-3 |
|---|---|---|
| Mutant study on the 200 diagnosis labels (in-sample) | 3/200 labels with a false accept: `literal_plus1` in a conjunction; multi-join `inner_to_left` ×2 labels | **0/200** (Wilson 0–1.88%). All 4 escaping mutants are killed (`boundary_minus_alltrue`, `dangle_eq2`, `dangle_eq0`, `dangle3`), and the 3 original labels are still accepted |
| Synthetic suite (14 queries, 105 mutants) | 0 false accepts | 0 false accepts (**did not separate the two verifiers**) |
| G2 adversarial suite (110 converted cases, MQL byte-identical) | 0 false accepts | 0 false accepts; 108/110 correct accepted, as with verifier-2 (A1b, O8 conservatively rejected) |
| G3 v1 wrong MQL | 33/33 rejected | 33/33 rejected |
| Dev items, minus the 161 holdout-2 labels (217 converted) | 200 accepted | 200 accepted. 0 extra rejections (no false-reject cost). 2 items verifier-2 had rejected as `mongo_timeout` (flight_4 GROUP BY over a 67k-row join) now finish because of the indexes and pass every semantic check |

The in-sample 0/200 was used only as a go/no-go signal before spending the single round-3 run. Rounds 1–2 showed that in-sample results overstate held-out sensitivity.

## 4. Round-3 pre-registration

Both stages were written before any round-3 result existed. The round-3 scripts refuse to run unless both stages are present and every hash matches (`freeze_round3_code.check_frozen`).

| Stage | File | Written | Content |
|---|---|---|---|
| 1 | `data/interim/finetune_v2/round3_prereg.json` (body sha256 `944e9063c64e9a6d315c01793f05b86ce61daee6b3a180ef3edd23249738d3e6`) | 2026-10-31T15:50:16, before any verifier-3 code existed | Population, strata, sample ids, inclusion probabilities, weights, estimator, gate, stop rule, power table, disclosure of what was seen of holdout-2 |
| 2 | `data/interim/finetune_v2/round3_code_freeze.json` | 2026-10-31T16:57:34 | SHA-256 of 19 files: converter, verifier-3 plus the verifier-2 code it runs, oracle, mutants, runners, statistics, report, both freeze scripts, `eval_kit.py`. Also the MongoDB image digest and the hashes of the pre-freeze evidence |

**Acceptance criterion (fixed before results, unchanged):** round 3 passes **iff the weighted Wilson 95% upper bound, unrounded, is ≤ 0.05**. Comparing the bound unrounded is stricter than rounding first.

**Stop rule:** one run; if it fails, stop and report, and no verifier change is re-measured on this sample.

## 5. Sampling methodology

- **Population:** the frozen pilot population (tier `exec_ok`, not leaked; 6,029) minus the 300 PILOT and 500 DEV ids, leaving **5,229 records**.
- **Strata:** computed from the SQL text only (`round3_stats.stratum`, top-level SELECT), with precedence J2 > OL > J1 > G.

| Stratum | Definition | Population N_h | Sampled n_h | Inclusion prob. π_h | Design weight N_h/n_h |
|---|---|---|---|---|---|
| J2 | ≥ 2 JOINs (≥ 3 tables) | 446 | 130 | 0.2915 | 3.431 |
| OL | ORDER BY … LIMIT (incl. LIMIT 1), < 2 joins | 751 | 130 | 0.1731 | 5.777 |
| J1 | exactly 1 JOIN | 1,090 | 130 | 0.1193 | 8.385 |
| G | everything else | 2,942 | 350 | 0.1190 | 8.406 |
| **Total** | | **5,229** | **740** | | |

- **Design:** stratified simple random sampling without replacement. Ids are sorted within each stratum, then drawn with `random.Random(20261031).sample` in the order J2, OL, J1, G. J2 and OL are oversampled.
- **Sample size:** chosen from a power table computed from verifier-2's per-stratum dev acceptance rates (planning only):
  - expected n_eff ≈ 492;
  - up to ≈ 15 weighted escapes still pass;
  - P(pass) is 0.96 if the true rate is 2%, 0.82 at 2.5%, 0.59 at 3%, and 0.09 at round 2's 4.35%.
- **Estimator:** the unit is an accepted label, y = 1 if any verifier-accepted mutant is shown non-equivalent by the oracle.
  - p_w = Σw·y / Σw (a ratio estimator over accepted labels);
  - Kish n_eff = (Σw)² / Σw²;
  - Wilson 95% on (p_w·n_eff, n_eff), the rounds-1/2 Wilson method. With equal weights this reduces exactly to k/n Wilson (unit-tested).
- **Procedure:** each step ran once.
  - **Step 1:** `run_v3.py round3`, converting and verifying all 740 records.
  - **Step 2:** `mutant_study_v3.py --out round3_mutants` on **every** accepted label, with no sub-sampling.
  - **Step 3:** `round3_report.py`.

## 6. Results

### 6.1 Coverage and acceptance (step 1; local; acceptance is not a correctness rate)

| Stratum | Records | Converted | Accepted | Rejected after conversion |
|---|---|---|---|---|
| J2 | 130 | 108 (83.1%) | 104 (80.0%) | 2 tie_cut, 2 empty table |
| OL | 130 | 106 (81.5%) | 78 (60.0%) | 26 tie_cut, 2 empty table |
| J1 | 130 | 91 (70.0%) | 90 (69.2%) | 1 empty table |
| G | 350 | 270 (77.1%) | 270 (77.1%) | — |
| **All** | **740** | **575 (77.7%)** | **542 (73.2%)** | 28 tie_cut, 5 empty referenced table |

- **Converter rejections** (165): `construct` 47, `not_a_single_select` 40, `bare_column` 39, `join_key_types` 18, `aggregate_type` 7, `join_without_on` 7, `join_condition` 3, `self_join` 3, `literal_type_mismatch` 1. There were 0 crashes.
- **Estimated population acceptance** (stratified expansion): 73.3%, about 3,832 accepted labels out of 5,229.
- **V8 / G6:** 542/542 accepted labels pass `eval_kit.validate_mql`.

### 6.2 G4 (step 2)

There were 2,741 mutants over 542 labels (22 labels produced no mutant; none hit the 30-mutant cap):
- killed by verifier-3: 1,969;
- accepted and judged non-equivalent by the oracle (**false accepts**): 21 mutants on **15 labels**;
- accepted and not separated by the oracle (**unkilled**): 751.

| | k / n | Point estimate | Wilson 95% |
|---|---|---|---|
| **Weighted (the gate)** | k_eff 11.95 / n_eff 502.2 | **2.38%** | **1.37% – 4.12%** (unrounded upper 0.04118) |
| Unweighted (NOT the population rate: strata were oversampled) | 15 / 542 | 2.77% | 1.68% – 4.52% |

**Gate G4 (round 3): PASSED** (0.04118 ≤ 0.05).

By stratum (reported, not gated):

| Stratum | Labels with a false accept | Rate | Wilson 95% | Escaping operators (mutants) |
|---|---|---|---|---|
| J2 | 7 / 104 | 6.73% | 3.30 – 13.25% | `cmp_gt_to_gte` 3, `null_guard_drop` 3, `inner_to_left` 2, `match_atom_drop` 1 |
| OL | 0 / 78 | 0.00% | 0.00 – 4.69% | — |
| J1 | 3 / 90 | 3.33% | 1.14 – 9.35% | `match_atom_drop` 2, and 1 each of `count_nonnull_to_star`, `sum_guard_drop`, `cmp_ne_to_eq`, `null_guard_drop`, `inner_to_left` |
| G | 5 / 270 | 1.85% | 0.79 – 4.26% | `cmp_lt_to_lte` 2, `cmp_gte_to_gt` 1, `cmp_gt_to_gte` 1, `distinct_to_all` 1 |

Verifier kills (reported, not gated):
- semantic: 1,831 (real-data L1 1,341, fixtures 469, reordered real data 15, nondeterminism 6);
- structural: 138 (`v3_lookup_keys`);
- non-semantic (timeouts, too few fixtures): 0.

**Independent recompute.** A separate script read only the raw `records.jsonl`, `mutants.jsonl` and `no_mutants.jsonl` and recomputed the weights, p_w, n_eff and Wilson. It reproduced every gate figure (p_w 0.0238, n_eff 502.23, upper 0.04118; unweighted 15/542, upper 0.04516). It also confirmed:
- all 19 frozen hashes match;
- 740 unique records;
- every accepted label has a mutant result;
- every false-accept row was verifier-accepted and carries separating oracle families.

**Interruptions** (logged in `round3/resume_log.jsonl` and `round3_mutants/resume_log.jsonl`). The Claude Code session's low-memory reaper stopped:
- step 1 once, at 530/740;
- step 2 twice, at 220 and 367 labels.

Each resume was made at your request and used the identical frozen code (every resume re-runs `check_frozen`). The last part of step 2 ran in the foreground in about 9-minute chunks (`timeout 560`). All 5 resume-log lines (1 for step 1, 4 for step 2) show **0 incomplete labels and 0 cut lines dropped**. Results are written once per finished label, so a label cut mid-run leaves nothing behind and is simply re-run.

## 7. Decision

- **Phase 1:**
  - G1 (converter unchanged);
  - G2 (verifier-3: 0 false accepts);
  - G3 (verifier-3: 33/33);
  - G4 (round 3, weighted upper 4.12%);
  - G5 (all families present, as a superset);
  - G6 (542/542).

  G1–G3 and G5–G6 pass as frozen. G4 passes the round-3 criterion you specified and that was pre-registered; the original unweighted Wilson form also passes, at 4.52%. **Phase 1 is passed on that basis.**
- **The 300-example pilot was not started** (this phase stops here). Proceeding needs your decision and:
  1. a verifier-3 pilot runner (`run_v2.py pilot` still uses verifier-2 and its pre-rename hash list);
  2. a named independent human reviewer for P2 (`docs/sql2mql_v2_human_review.md`).
- **Recommended precaution for J2** (your call; the gate and P2's frozen selection rule are unchanged): either
  - add a supplementary J2-only human review on top of P2's frozen min(120, accepted) sample, or
  - keep ≥ 3-table labels out of the eventual corpus until a J2-specific check passes on records never used.

## 8. Remaining limitations

1. **G4 measures verifier sensitivity, not converter accuracy.** It counts single-edit mutants the verifier accepts that the oracle battery separates. It is a **lower bound** on the verifier miss rate: 751 survivors were not separated, and round 2's manual adjudication found 1 of 12 such survivors non-equivalent. A seeded sample of 20 survivors is listed in `outputs/sql2mql_v2/round3_report.json`; it has **not** been adjudicated.
2. **J2 (≥ 3 tables) is the weakest class:** 6.7%, upper bound 13.3%. Escapes include `>` vs `>=` boundaries where the conjuncts sit in different joined tables (the `_alltrue` families make atoms true per table, not across the join), NULL-key guards on multi-hop lookups, and `inner_to_left` on edges the dangling families did not reach.
3. **In-sample optimism:** verifier-3 had 0/200 in-sample but 2.38% on the new sample. The synthetic suite did not separate verifier-2 from verifier-3.
4. **Shared environment change:** the key indexes were added to the shared local import (`spider_train_v2*`), which verifier-2 also reads. The two verifier-2 `mongo_timeout` dev rejections no longer reproduce. Results are unaffected; only timeouts are.
5. **Single-edit mutants** do not cover multi-edit or structurally different wrong MQL. G2/G3 and the pending human review address other parts of that risk.
6. **The 740 round-3 records are now spent for gating.** They must not be used for a future gate measurement.
7. **Local only:** all figures are from SQLite + MongoDB 8.0.34 in Docker. None are Atlas or model metrics.

## 9. Artefacts and reproduction

| What | Path |
|---|---|
| Verifier-3 | `scripts/finetune_v2/verifier3.py`, `fixtures3.py`, `index_real.py` |
| Round-3 tooling | `scripts/finetune_v2/freeze_round3.py`, `freeze_round3_code.py`, `round3_stats.py`, `run_v3.py`, `mutant_study_v3.py`, `round3_report.py` |
| Pre-freeze evidence | `outputs/sql2mql_v2/diag_v2_insample/`, `diag_v3_insample/`, `dev3/`, `adversarial_v3/`, `synthetic_v3/` (`adversarial_v3.py`, `synthetic_v3.py`) |
| Round 3 | `outputs/sql2mql_v2/round3/`, `round3_mutants/`, `round3_report.json`, logs `round3.log`, `round3_mutants.log` |
| Human review | `docs/sql2mql_v2_human_review.md`, `scripts/finetune_v2/review_package.py`, `outputs/sql2mql_v2/review_dryrun/` |
| Tests | `tests/test_finetune_v2_round3.py` (20), `tests/test_finetune_v2.py` (46) |

```
cd scripts/finetune_v2
python -B round3_report.py      # refuses if any frozen file changed; recomputes from the raw files
cd ../..
python -m pytest -q tests/test_finetune_v2_round3.py tests/test_finetune_v2.py
```
