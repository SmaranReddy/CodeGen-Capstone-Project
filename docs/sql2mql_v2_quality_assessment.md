# sql2mql-v2 quality assessment

**Status (2026-10-30): Phase 1 NOT PASSED. V2 is NOT ready for the pilot or for full-corpus conversion.**

> **Dated update, 2026-10-31.** The 2026-10-30 status above is kept as written. A third round on a new pre-registered sample of 740 Spider-train records, with a new separately versioned verifier-3, **passed G4**: weighted false-accept rate 2.38%, Wilson 95% upper bound 4.12% ≤ 5%. G2 and G3 were re-run with verifier-3 and still pass. G1–G3 and G5–G6 pass as frozen. G4 passes the round-3 criterion you specified and that was pre-registered (a new stratified population, weighted Wilson); the original unweighted form also passes at 4.52%. **Phase 1 is passed on that basis.**
>
> - **Weak class:** multi-table joins (≥ 3 tables) remain weak, at 7/104 labels (upper bound 13.3%).
> - **The pilot has not been started.**
>
> See §9 below and `docs/sql2mql_v2_round3_report.md`. Nothing in §§1–8 was changed.

| Gate | Result |
|---|---|
| G1 coverage matrix | passed |
| G2 adversarial suite | passed |
| G3 negative controls | passed |
| G5 fixture families | passed |
| G6 / V8 safety | passed |
| **G4 mutant study (V2)** | **failed** on the pre-registered held-out measurement: 7/161 labels, Wilson 95% upper bound 8.7% > 5% |

- No pilot item has been converted, verified or inspected. No full-corpus data exists, and no training or RAG data was created.
- V1 code and data are unchanged, as are the 892-item evaluation set, the 65-item test partition and Atlas.

Everything below was measured locally on SQLite and on MongoDB **8.0.34**, run as a Docker container on 127.0.0.1:27018. Image `mongo:8.0.34`, digest `sha256:b9924c33…fa142`. These are label-validation measurements, not model metrics. Nothing connected to Atlas.

## 1. What was built

All code is in `scripts/finetune_v2/`. V1 (`scripts/finetune_v1/`, `scripts/audit_sql2mql_v1/`) was not edited.

| Side | Modules | Role |
|---|---|---|
| Converter | `sql2mql_v2.py`, `schema_profile.py` | Conservative SQLite-semantics translator; type-aware from the observed storage classes |
| Verifier | `verifier.py`, `comparator.py`, `v2_fixtures.py`, `mongo_import.py` | Differential verification. The oracle is the **original SQL run verbatim on SQLite** |
| Measurement only | `oracle_v2.py`, `mutants_v2.py`, `mutant_study.py`, `adversarial_v2.py`, `adversarial_cases_v2.py`, `suite_v2.py`, `coverage_matrix.py` | Negative controls and gates; never used to accept a label |
| Plumbing | `v2_common.py`, `freeze_pilot.py`, `run_v2.py`, `rename_modules.py` | Paths, local-only Mongo guard, freeze, runner |

**Independence.** Converter and verifier never import each other, and the verifier never imports the oracle or the mutants. This is enforced by tests. The verifier translates nothing: it executes the SQL on SQLite, executes the candidate MQL on MongoDB, and compares the results with its own strict comparator.

**Local-only MongoDB.** `v2_common.local_client` refuses any host other than 127.0.0.1:27018. No module reads `MONGODB_URI`, `.env` or the environment (tested).

**Representation.** The local import uses the benchmark's Atlas contract (`scripts/spider1/import_atlas.py`):
- one collection `<db>__<table>` per table;
- SQLite storage class → BSON type;
- explicit `null`;
- rowid order.

A second copy stores NULL as an absent field (F15). SQLite↔Mongo multiset parity is checked on every import. The P1 exit check was passed on `department_management` and `college_2`, and on every database imported since.

Two module files were renamed after the measurements (`common.py` → `v2_common.py`, `fixtures.py` → `v2_fixtures.py`) because their names collided with other script directories in the full pytest run. Only import lines changed. Before/after hashes are in `outputs/sql2mql_v2/rename_manifest.json`.

## 2. Semantics contract of the converter

The contract is stated in full in the `sql2mql_v2.py` docstring. In summary:

- **SQLite semantics** throughout: integer division, ASCII-only case-insensitive LIKE, BINARY collation for `=` and ORDER BY, column-affinity literal conversion.
- **NULL.** Three-valued logic: a NULL predicate is false. Every expression is safe whether NULL is stored as an explicit null or as an absent field.
- **Types.** A column's observed storage classes must be homogeneous wherever its type matters; otherwise the query is rejected.
- **Rejected by design:**
  - subqueries, set operations, CTEs;
  - CASE, CAST, COALESCE and every scalar function, including ROUND (F14);
  - self-joins, non-LEFT outer joins, joins without a single ON equality;
  - bare columns not determined by a grouped non-NULL primary key;
  - LIMIT without ORDER BY.

### Coverage matrix (G1)

`python -B scripts/finetune_v2/coverage_matrix.py` produces `outputs/sql2mql_v2/coverage_matrix.json`. It covers **26 items**, every one with a passing unit test and/or adversarial case.

| Item | Handling | Evidence (adversarial cases) |
|---|---|---|
| F1 empty-input aggregate (B1) | implemented: `$facet` + default row (COUNT 0, others NULL) | V_F1a, A1, A2, A1b; live family test |
| F2 SUM of all-NULL (B2) | implemented: non-NULL count guard | V_F2a, A6, A6b, V_HAV1, V_CD1 |
| F3 DISTINCT + GROUP BY | implemented (dedupe stage); rejected when combined with ORDER BY | V_F3a |
| F4 arithmetic on text/dates | rejected | V_F4a |
| F5 identifier case | implemented: canonical spelling; SQLite double-quoted-string fallback | V_F5a, V_F5b, V_DQ1 |
| F6 implicit casts | implemented (affinity rules) + rejected (mixed-class columns, non-numeric text vs number) | V_F6a–f, W17, V_NUM1 |
| F7 mixed-type join keys | rejected | V_F7a, J8 |
| F8 division | implemented: zero divisor → NULL; `$trunc` when both operands are integers | V_F8a–c, N1, N1b, N3, V_AVG1 |
| F9 ORDER BY ordinal/alias | implemented | V_F9a/b, O4, O4c, O5 |
| F10 NOT bare column | implemented: `{col: 0}` | V_F10a, W14 |
| F11 LEFT JOIN guards | implemented: missing-aware | V_F11a, J4, J4b |
| F12 lookup name collision | implemented: reserved `__j<k>` | V_F12a, J6 |
| F13 LIKE | implemented: options `is`, absolute `\z` anchor, non-ASCII pattern rejected | V_F13a–d, W10d, W10e |
| F14 ROUND | rejected | V_F14a |
| F15 NULL contract | explicit null in the import; every label verified under both representations | W19, W19b, W27, A5, A5b, A13, J3, J5 |
| D4, A3, C4@missing (clock), bare column, LIMIT without ORDER BY, DISTINCT+GROUP BY, date subtraction, identifier case, implicit cast, int-vs-text join | see matrix | |

## 3. Verifier

For every candidate the verifier checks, in order:
1. structure, plus `eval_kit.validate_mql` unchanged (V8);
2. non-empty referenced tables and valid UTF-8;
3. **L1 on the real database** under explicit null **and** absent-field NULL (V4);
4. ORDER BY ... LIMIT tie cut → reject (V5);
5. repeated execution (V6);
6. **paired fixture databases.**

Fixtures are built from each table's original `CREATE TABLE` DDL, filled with family-specific rows and read back as stored. Exactly those rows are imported into MongoDB. If the real result is empty, a non-empty fixture result is required.

**Comparator.** It is positional (no column permutation) and keeps strings and numbers distinct (`'10' != 10`). Relative tolerance is 1e-9. Within ORDER BY tie blocks, ordering is tie-aware.

Families in `sql2mql-v2-verifier-2`:
- The nine required by V3: empty, zero_match, all_null, dup, case_variants, shuffle (distinct values per GROUP BY key), zero_div, text_numerics, newline.
- Also: null_mix, literal_hits and two samples.
- Added in iteration 2: boundary_minus/equal/plus/case, orphans, small_orphans, null_atoms_even/odd, null_keys_even/odd, all_null_keep.
- Six families also run with absent-field NULL.

Every sample starts from the **witness rows**: real rows that satisfy the query's joins and WHERE.

**Gold sanity check.** For 500/500 dev items, the verifier's SQLite oracle reproduces the gold result hash recorded in `data/processed/spider_sql_v1/train.jsonl`.

## 4. Pre-registration

`data/interim/finetune_v2/pilot_freeze.json` (sha `fb1affb9…`) was written **before any converter or verifier code existed**. It contains:
- the V9 screen over all 7,773 train records:
  - 0 records on the 20 Spider-dev databases;
  - 0 questions with character-trigram Jaccard ≥ 0.8 against the 892 evaluation questions;
  - `test.jsonl` and the held-out test-database questions were not read;
- the pilot population: the 6,029 records of tier `exec_ok`, verified on the official populated DB. The 1,744 records on the seven all-empty DBs are excluded;
- **PILOT**: 300 ids, stratified by hardness. **Not used.**
- **DEV**: 500 disjoint ids. These are the only Spider items used for development;
- the acceptance criteria quoted below.

`data/interim/finetune_v2/g4_remeasure_prereg.json` (sha `fe27924f…`) was written **before the iteration-2 measurement**. It fixes:
- the held-out label rule;
- the unchanged gate;
- the code hashes of the oracle and the mutant generator, byte-identical to iteration 1;
- the stop rule.

Frozen Phase 1 gate:
- **G1** every F1–F15 item and audit defect implemented or rejected, with a passing regression test.
- **G2** 0 adversarial false accepts.
- **G3** 100% of v1 wrong MQL and hand controls rejected.
- **G4** labels with ≥ 1 verifier-accepted, oracle-non-equivalent mutant: Wilson 95% upper bound ≤ 0.05.
- **G5** all nine V3 families present and unit-tested.
- **G6** V8 = 100%.

## 5. Results

### 5.1 Adversarial regression suite (V1 / G2): passed

`python -B scripts/finetune_v2/adversarial_v2.py suite` writes `outputs/sql2mql_v2/adversarial/suite.{jsonl,summary.json}`.

| | Cases |
|---|---|
| Cases (103 from the 2026-10-25 audit + 36 new v2 cases) | 139 |
| Converted | 110, **all correct** under explicit null **and** absent-field NULL |
| Rejected by the converter (all 13 expected rejections included) | 29 |
| **False accepts** (verifier accepts an MQL whose result differs from the expectation) | **0** |
| Correct translations conservatively rejected by the verifier | 2: A1b (empty table), O8 (tie at LIMIT) |

Expectations are the audit's hand expectations. Three are MySQL semantics and were re-derived by hand for SQLite integer division:
- N1: `7/2 = 3`;
- N1b: the same, with no dialect note in the audit;
- N3: no row satisfies `a/b > 3`.

The audit's W16, A10 and O11 already follow SQLite BINARY collation. Eleven cases use MySQL-only functions (NOW, YEAR, …) that SQLite lacks; for those, rejection is the correct v2 outcome. After the overrides, SQLite's native result matches the expectation in every case.

### 5.2 Negative controls (G3): passed

The 33 wrong translations recorded for v1 (the audit's 20 explicit-null failures plus 13 absent-field failures) were replayed through the v2 verifier, without running any v1 code.
- **33/33 rejected.**
- 28 rejections are discriminating: a result mismatch, or a MongoDB error where SQL gives NULL.
- 5 are incidental: 4 MySQL-only functions and 1 empty table.

There is also a live test: `COUNT(*)` without the F1 default row passes L1 on non-empty data, and must be rejected by a fixture family. It is.

### 5.3 Dev sample (500 items, 117 databases): coverage, execution, equivalence

`run_v2.py dev` produces `outputs/sql2mql_v2/dev_iter{1,2}/`.

| Measure | Verifier-1 | **Verifier-2** |
|---|---|---|
| Converted (coverage) | 378 / 500 (75.6%) | 378 / 500 (75.6%) |
| MQL executed on the real DB | 362 | 362 |
| **Accepted** (all checks) | 360 (72.0%) | **361 (72.2%)** |
| Rejected: ORDER BY ... LIMIT tie cut (V5) | 14 | 14 |
| Rejected: MongoDB timeout (30 s, large `$lookup`, `flight_4`) | 2 | 2 |
| Rejected: non-discriminating (empty everywhere) | 2 | 1 |
| L1 or fixture **mismatch** of a converted label | 0 | 0 |
| Labels lost going to the stronger verifier (false-reject cost) | | **0** |

Accepted by hardness, under verifier-2:

| Hardness | Accepted |
|---|---|
| easy | 117 / 120 |
| medium | 174 / 197 |
| hard | 53 / 106 |
| extra | 17 / 77 |

Unsupported, by reason:

| Reason | Items |
|---|---|
| set operation / not a single SELECT | 37 |
| subquery or other construct | 35 |
| bare column | 28 |
| join-key type mismatch | 10 |
| other | 12 |

Accepted labels passed:
- V6 determinism;
- V8, at 100% (`validate_mql` is a verifier check);
- V9, by construction of the freeze.

**Caveat.** "0 mismatches" does not show that the labels are correct. It only shows that no check found a difference. How sensitive the checks are is what G4 measures.

### 5.4 Mutant study (V2 / G4): FAILED

Mutants are single-edit wrong MQLs from 26 operator classes. They include a reversal of every v2 fix, and are applied to accepted **dev** labels. A survivor (a mutant accepted by the verifier) is judged by `oracle_v2.py`. That oracle is a stronger, differently designed battery, ported from the audit's synthetic small-domain families: 12 random families × 3 seeds, plus empty, nullcols, dup and absent-field runs. A mutant counts as **non-equivalent** (a false accept) only on a family where the original label passes.

| Run | Labels | Mutants | Killed by verifier | Survivors | **Non-equivalent survivors** | Labels with a false accept | Wilson 95% | G4 |
|---|---|---|---|---|---|---|---|---|
| Iteration 1 (verifier-1), `mutant_study/` | 200 | 862 | 608 | 254 | 41 | **27 (13.5%)** | 9.5–18.9% | fail |
| **Iteration 2 (verifier-2), held-out**, `mutant_study_iter2_heldout/` | **161** | 673 | 472 | 201 | 9 | **7 (4.35%)** | **2.1–8.7%** | **fail** |
| Iteration 2, in-sample (secondary) | — | — | — | — | — | **not completed** | | n/a |

The secondary in-sample run was stopped by Claude Code's low-memory reaper at 60/200 labels and wrote no result. It was not restarted.

**Iteration-1 diagnosis.** On the 27 failing labels:
- 24 of 53 table instances had **degenerate copy-adding families**. Copies of rows in tables with a text or composite primary key were silently dropped by `INSERT OR IGNORE`.
- No family put values on a literal boundary, created orphan join rows, or placed NULLs specifically in predicate or join-key columns.

Verifier-2 fixes these, as listed in §3, with a degeneracy check. An ad-hoc in-sample diagnostic during development found it kills 37 of iteration 1's 41 non-equivalent mutants while still accepting all 27 original labels. That figure is diagnostic only, not a gate figure.

**What still escapes verifier-2** (held-out, 9 mutants on 7 labels):
- `inner_to_left` ×3 and `lookup_field_swap` ×1 on 2–3-way joins with GROUP BY / HAVING;
- `cmp_ne_to_eq` ×2 and `null_guard_drop` ×2 on 3-way joins with COUNT;
- `sort_drop` ×1 on `ORDER BY price DESC LIMIT 1`.

All need small synthetic tables where the joins only partly match. The real-row-derived families rarely produce those.

**Survivors not separated by either battery** (192 held-out). A manual adjudication of a seeded random sample of 12 found:
- **11 equivalent**:
  - 4 `ifnull_drop` on group or distinct keys, equivalent under a uniform NULL representation;
  - 5 `inner_to_left` where a later filter or join removes the unmatched rows;
  - 2 `null_guard_drop` on INTEGER PRIMARY KEY join keys;
- **1 non-equivalent**: `sum_guard_drop` under `ORDER BY sum(..) DESC LIMIT 1`, in an all-NULL-group edge case neither battery generated.

The measured false-accept rates are therefore **lower bounds**.

**Integrity of the re-measurement.**
- An independent recompute from the per-mutant JSONL reproduces every summary figure:
  - iteration 1: 27/200;
  - iteration 2: 7/161;
  - both Wilson intervals.
- The `verifier.py`, `fixtures.py`, `oracle_v2.py` and `mutants_v2.py` hashes recorded by the held-out run equal those in the pre-registration.
- The oracle also calls verifier/fixture functions: `order_plan`, `sqlite_truth`, `compare`, `run_mql`, `literals`, `referenced_tables`, `DbInfo` and `load`. Per this session's edit history, none of them changed between iterations; only the verifier's fixture *building* changed. This could not be byte-checked, because iteration-1 copies of those files were not kept.

**Comparison with v1** (10.5% in the audit). It is not like-for-like:
- the oracle here is considerably stronger;
- the labels are Spider-train items, not Kaggle rows.

### 5.5 V-checks summary

| Check | Status |
|---|---|
| V1 adversarial | passed (0 false accepts / 139) |
| V2 mutants | **failed** (held-out upper bound 8.7%) |
| V3 families | passed (24 families: the nine required + 15 more, unit-tested) |
| V4 L1 on real DB, both NULL representations | implemented; every accepted dev label |
| V5 ties | implemented; 14 dev rejections, 0 accepted tie-cut |
| V6 determinism | implemented; every accepted dev label |
| V7 human review | **not run**: it applies to the pilot only. A reviewer must be chosen; an AI reviewing its own converter is not independent |
| V8 safety | 100% |
| V9 leakage | 0 / 7,773 |

## 6. Limitations

- The verifier is a **differential tester, not a prover**. Passing it means equivalence on the real database plus 24 paired fixture families, not general semantic equivalence. G4 shows that its sensitivity on multi-join queries is not yet at the frozen bar.
- The pilot population excludes the 1,744 records on all-empty DBs. Coverage is about 76% on dev, and much lower for hard/extra queries. The corpus will under-represent subqueries, set operations and functions.
- Labels are correct with respect to the gold SQL, not the question. V7 must cover the question ↔ SQL link.
- `exec_match` compatibility of accepted labels was not separately recomputed. The v2 comparator is stricter: it is positional, and does not round to 4 decimal places.
- The oracle reuses the verifier's comparator and ORDER BY tie logic. Only the data design is independent.

## 7. Decision and options

**V2 is not ready for the pilot.** The pilot (Phase 2) was not started, and the full corpus (Phase 3) is blocked behind it. Per the pre-registered stop rule, there is no further iteration against the same 161-label holdout.

Options, for the user to decide:
1. **Iteration 3 with a fresh holdout.** Draw an additional, pre-registered dev sample from the 5,229 population records outside the pilot and dev samples. Add families targeting partly-matching small multi-join tables and LIMIT-1 orderings, then re-measure G4 on labels never used for diagnosis.
2. **Scope restriction.** Exclude the query class where the verifier is weak (for example, three or more joins) and re-measure G4 within the remaining class. This changes the gate's population, so it needs an explicit decision.
3. Accept the current verifier with documented risk. **Not recommended**; it would contradict the frozen gate.

## 8. Reproduction

```
docker run -d --name codegen-mongo-8034 -p 127.0.0.1:27018:27017 -v ~/codegen_data/mongo_v2:/data/db mongo:8.0.34
cd scripts/finetune_v2
python -B adversarial_v2.py negative
python -B adversarial_v2.py suite
python -B coverage_matrix.py
python -B run_v2.py dev --out <dir>
python -B mutant_study.py --records <dir>/records.jsonl --exclude-labels ../../outputs/sql2mql_v2/mutant_study/diagnosis_labels.json --out <new dir>
cd ../..
python -m pytest tests/test_finetune_v2.py -q        # 46 tests; 'live' tests need the container
```

## 9. Addendum (2026-10-31): verifier-3 and round 3

Full report: `docs/sql2mql_v2_round3_report.md`.

**Verifier-3** (`verifier3.py`, `fixtures3.py`, `index_real.py`) is new code. Verifier-2 and every §1–8 artifact are unchanged. Verifier-3 runs every verifier-2 check, then adds:
- real data in reversed and shuffled storage order;
- reversed and shuffled fixture rows;
- in-place dangling join keys;
- conjunction boundaries with the other atoms true;
- shifted join ids;
- NULL keys on both sides;
- a lookup-field existence check;
- result-invariant key indexes.

**Round 3** was pre-registered in two stages: `data/interim/finetune_v2/round3_prereg.json` (stage 1, written before verifier-3 existed) and `round3_code_freeze.json` (stage 2). It used 740 records stratified by query complexity, drawn from the 5,229 population records outside pilot and dev.

| Gate | Verifier-3 result |
|---|---|
| G2 adversarial | 0 false accepts (110 converted cases; 108 correct accepted, as with verifier-2) |
| G3 negative controls | 33/33 rejected |
| **G4 round 3 (weighted, pre-registered)** | **p_w 2.38%, n_eff 502.2, Wilson 95% 1.37–4.12% → PASSED** |
| G4 round 3 unweighted (not the population rate) | 15/542, 2.77%, Wilson 1.68–4.52% |
| G4 by stratum | J2 7/104 (6.7%, up to 13.3%); OL 0/78; J1 3/90 (3.3%); G 5/270 (1.9%) |
| G6 / V8 | 542/542 |

**Environment note (dated 2026-10-31).** Single-field key indexes were added to the shared local import `spider_train_v2` / `spider_train_v2_missing` (`index_real.py`). Indexes do not change aggregation results. However, the two dev-iteration-2 rejections recorded as `mongo_timeout` in §5.3 (flight_4) no longer time out. Re-verified with verifier-3, they pass every semantic check (`outputs/sql2mql_v2/dev3/`).
