# Test-suite execution accuracy: zero-shot SQL baseline (SQL-0001), Spider dev

Date: 2026-10-28. **Development split only:** the 1,034 records of `data/processed/spider_sql_v1/dev_full.jsonl`.
No held-out test questions, labels or predictions were opened, and no inference was run.
The SQL-0001 predictions and metrics are unchanged.

## 1. Summary

| Metric (denominator 1,034) | Value | Status |
|---|---|---|
| **Test-suite execution accuracy** (Zhong et al. 2020; official defaults: values predicted, DISTINCT removed) | **630 / 1,034 (60.9%)** | secondary metric, new |
| Test-suite EX, sensitivity setting `--keep_distinct` | 603 / 1,034 (58.3%) | sensitivity only |
| Official EX, `evaluation.py` @ b7b5b8c (SQL-0001, unchanged) | 446 / 1,034 (43.1%) | primary |
| Official exact-set match (SQL-0001, unchanged) | 378 / 1,034 (36.6%) | primary |
| SQLite execution success (SQL-0001) | 913 / 1,034 (88.3%) | execution only |
| Unofficial result-set match, original DB only (SQL-0001) | 654 / 1,034 (63.2%) | unofficial |

- **Coverage is complete.** The suite covers all 20 dev databases with 695 SQLite files: the 20 originals plus 675 variants, 25–60 files per database. That includes 33 files for `wta_1`, which earlier work had not extracted.
  - All 1,034 examples were evaluated.
  - 0 examples lack variants, 0 gold queries fail and 0 queries time out.
- **Gold as the prediction scores 1,034 / 1,034** on every file.
- **Methodology.** The verdict uses the official test-suite code, unmodified: `taoyds/test-suite-sql-eval` @ `e97acc5`.
  - The unmodified official `evaluation.py`, run on the same inputs, prints the same accuracy overall and at every hardness level (§4).
- **What the 630 covers.** The test-suite metric does not depend on the 2019 parser. The 630 includes 177 predictions that the official parser rejected. It also excludes 32 official EX hits, 31 of which are coincidental: they match on the original DB and fail on a variant.

## 2. Assets and how the test suite works

| Asset | Location | Verified |
|---|---|---|
| Test-suite databases | `data/raw/spider1_testsuite_download/testsuite_database.zip` | sha256 `9ec24ea8debc6bd04abfe137b5f1a739b5a8836f32c0464e4dfc94eb7f41da96`. This matches the value in `docs/spider1_dataset.md`. |
| Official evaluator | `scripts/sql_v1/testsuite_official_e97acc5/` | `exec_eval.py`, `parse.py`, `evaluation.py`, `process_sql.py`, `LICENSE` (Apache-2.0) and `README.md`, fetched from GitHub at the pinned commit. They are unmodified; sha256 values are in `SOURCE_COMMIT.txt` and are checked at import. |
| Extracted dev suite | `%USERPROFILE%\codegen_data\spider1_testsuite_e97acc5\database\<db_id>\` | Outside the OneDrive-synced repo, because `wta_1` alone is 3.46 GB. The location can be overridden with `--db-root` or `SPIDER_TS_DB_ROOT`. The manifest (`dev_manifest.json`) records size and sha256 for all 695 files. A copy is in each output folder. |
| Earlier 3-variant extraction | `data/raw/spider1/testsuite/` | Not used and not changed. It belongs to the MQL track. |

**How the official method works**

- The source is the README and `exec_eval.eval_exec_match`.
- For each example, gold and prediction run on **every** `.sqlite` file in `database/<db_id>/`. That is the original DB plus all of its distilled variants.
- A prediction is correct only if its denotation equals gold's on every file. Gold and prediction always run on the same file.
- **Equality rule (`result_eq`):**
  - Rows are compared as bags (multisets).
  - Row order is enforced only if the gold contains `order by`.
  - Column order is ignored: any column permutation is allowed.
- **Before execution:**
  - `evaluate()` replaces the literal `value` with `1` in predictions.
  - `postprocess` fixes spaced operators such as `> =`.
  - Without `--keep_distinct`, every `DISTINCT` token is removed from both queries.
- **Options:**
  - `--plug_value` is meant for systems that do not predict values. It is off here, because the model writes literal values.
  - The README says `--keep_distinct` should be left off "to make a fair comparison with the original exact set match metric". So the primary setting leaves it off, which is also the default.

**Mapping and coverage (verified on the files, not inferred from names)**

- **Mapping to db_id.** Variant files sit in `database/<db_id>/`, named `<db_id>v<tag>.sqlite`. Each dev `db_id` maps to its own directory.
- **Originals.** All 20 originals in the zip are byte-identical (sha256) to `data/raw/spider1/spider_data/database/<db_id>/<db_id>.sqlite`.
- **Readability.** Every one of the 695 files opened read-only and listed its tables.
- **Not extracted.** The zip also holds `__MACOSX/` resource forks. These would be wrongly picked up by the official `'.sqlite' in basename` filter. It also holds 8 non-dev databases (`academic`, `atis`, `advising`, `geography`, `imdb`, `restaurants`, `scholar`, `yelp`). Neither was extracted.
- **Additional extraction was required.** Before this task only 3 variants per DB existed, and none for `wta_1`.

| db_id | dev examples | files (incl. original) | TS EX |
|---|---:|---:|---:|
| battle_death | 16 | 28 | 9 |
| car_1 | 92 | 60 | 33 |
| concert_singer | 45 | 32 | 27 |
| course_teach | 30 | 25 | 12 |
| cre_Doc_Template_Mgt | 84 | 25 | 50 |
| dog_kennels | 82 | 48 | 51 |
| employee_hire_evaluation | 38 | 27 | 31 |
| flight_2 | 80 | 56 | 48 |
| museum_visit | 18 | 35 | 13 |
| network_1 | 56 | 29 | 38 |
| orchestra | 40 | 28 | 36 |
| pets_1 | 42 | 27 | 26 |
| poker_player | 40 | 31 | 38 |
| real_estate_properties | 4 | 32 | 1 |
| singer | 30 | 26 | 25 |
| student_transcripts_tracking | 78 | 55 | 44 |
| tvshow | 62 | 25 | 41 |
| voter_1 | 15 | 27 | 12 |
| world_1 | 120 | 46 | 51 |
| wta_1 | 62 | 33 | 44 |
| **total** | **1,034** | **695** | **630** |

## 3. Method

**Script:** `scripts/sql_v1/evaluate_test_suite_dev.py`

| Subcommand | What it does |
|---|---|
| `extract` | Idempotent; never overwrites |
| `evaluate` | Scores the run |
| `parity` | Runs the unmodified official evaluator |
| `compare` | Builds the cross-tabs |

**Inputs**
- Predictions are the SQL-0001 `eval_sql` strings, which are the exact strings the official evaluator saw.
- Gold comes from `dev_full.jsonl`. Order, ids, db_id and gold must all match the run, or the script stops.

**Imported from the official code, unmodified:** `postprocess`, `remove_distinct`, `result_eq`, `replace_cur_year`. The script also reproduces `evaluate()`'s `replace("value", "1")`. It changed 0 predictions.

**Disclosed deviations.** They change robustness, not the decision rule.

1. **Read-only connections.** The script opens databases with `mode=ro`; the official code opens them read-write. It decodes text with `errors="ignore"`, as the official code does.
2. **A real 60 s timeout.**
   - The official code sets `TIMEOUT = 60`, but its `asyncio.wait_for` wraps a synchronous `cursor.execute`, so the timeout can never fire.
   - This script interrupts SQLite through a progress handler instead.
   - A prediction that times out is counted as incorrect. None did.
3. **Gold failures are recorded, not asserted.** The official code stops the whole run if gold fails on any file. Here it is recorded as `gold_error` or `gold_timeout` and reported as "not evaluable". There were none.
4. **Deterministic order and seed.**
   - Files are checked original first, then sorted, and checking stops at the first failing file, as in the official loop. Because the verdict requires every file to pass, the order cannot change it.
   - `random` is seeded before each comparison. `get_constraint_permutation` samples rows only to prune column permutations that are impossible anyway, so the seed affects speed, never the verdict.
   - A second run produced byte-identical `testsuite_results.jsonl`.

**Not used:** the e97acc5 `process_sql` parser, and exact match. Hardness comes from `dev_full.jsonl`.

**Environment:** Python 3.10.11, SQLite 3.40.1, sqlparse 0.5.3 (installed for this task; the official `parse.py` needs it), nltk 3.9.3 (needed only by the parity run).

**Outputs** are in `outputs/spider_sql_v1_zs_3b/v1/testsuite_eval_v1/`:

| Path | Contents |
|---|---|
| `predictions/` | Primary run: per-example `testsuite_results.jsonl`, `testsuite_summary.json` (config, versions, coverage, aggregates) and `dev_manifest.json` |
| `predictions_keep_distinct/` | Sensitivity run |
| `gold_as_pred/` | Gold queries scored as predictions (sanity check) |
| `official_parity_run3/` | Inputs, command and stdout of the unmodified official `evaluation.py` (the completed parity run) |
| `official_parity/`, `official_parity_run2/` | Stalled attempts, kept as evidence of the Windows `asyncio` hang (§4) |
| `comparison.json` | All cross-tabs below, with example ids |

Each per-example record holds: `status` (pass / mismatch / pred_error / pred_timeout / pred_preprocess_error / gold_error / gold_timeout / missing_variants), `ts_exec` (1 / 0, or null if not evaluable), `n_files`, `n_checked`, the first failing file, a short detail, and `orig_db_pass`.

## 4. Parity with the unmodified official evaluator

**Method.** `evaluate_test_suite_dev.py parity` writes `gold.txt` (`gold \t db_id`) and `pred.txt` from the same strings. It then runs the vendored `evaluation.py --etype exec` unmodified, as `__main__` via `runpy`. The run used the default settings and the same 695-file DB directory.

**Output (`official_parity_run3/official_stdout.txt`):**

```
                     easy                 medium               hard                 extra                all
count                248                  446                  174                  166                  1034
execution            0.839                0.641                0.494                0.301                0.609
```

**Agreement.** Each printed fraction maps to exactly one integer numerator, and those are the numerators of this script's run:
- easy 208/248 = 0.8387;
- medium 286/446 = 0.6413;
- hard 86/174 = 0.4943;
- extra 50/166 = 0.3012;
- all 630/1,034 = 0.6093.

The official hardness counts, computed with e97acc5's own `process_sql`, also equal the `dev_full` counts. Parity is established at the aggregate and per-hardness level. The official script prints no per-example verdicts.

**Two harness problems on Windows, both disclosed**

1. **The event loop hangs.** `exec_eval` calls `asyncio.run()` twice per DB file. Each call builds a new Windows Proactor event loop, which opens a loopback socket pair.
   - **Run 2 hung within its first 10 examples.** It never printed "Evaluating 10th prediction". The faulthandler stack dump (`official_parity_run2/official_stderr.txt`) shows `asyncio.run` → new Proactor loop → `socketpair` → `socket.accept()` blocked.
   - **Run 1 (`official_parity/`) also stalled,** with flat CPU and no output. Its stdout was buffered, so its stall point was not captured.
   - The root cause of the blocked `accept()` is not established.
   - Run 3 points `asyncio.run` at one persistent event loop in the bootstrap. The coroutines and both official files are unchanged.
2. **Opened read-write.** The official code opens DBs read-write. After the run, all 695 files still matched their manifest sha256, and no extra files had appeared.

Run 3 produced no stderr and no gold-assert failure.

## 5. Results

**By hardness** (the denominator is all examples at that level):

| Hardness | n | TS EX | Official EX (SQL-0001) |
|---|---:|---:|---:|
| easy | 248 | 208 (83.9%) | 188 (75.8%) |
| medium | 446 | 286 (64.1%) | 172 (38.6%) |
| hard | 174 | 86 (49.4%) | 62 (35.6%) |
| extra | 166 | 50 (30.1%) | 24 (14.5%) |
| **all** | **1,034** | **630 (60.9%)** | **446 (43.1%)** |

**Why the other 404 fail**
- **Wrong result, 283.** 205 already differ on the original DB; 78 match on the original DB and differ only on a variant.
- **SQLite error, 121.** These are exactly the 121 SQLite execution failures from SQL-0001.
- **Timeout, 0.**

## 6. Comparison with the existing metrics

These metrics measure different things, and none of them is redefined here.

**Official EX × test-suite EX**

| | TS pass | TS fail |
|---|---:|---:|
| Official EX = 1 | 414 | **32** |
| Official EX = 0 | **216** | 372 |

**Accepted by official EX but failing the test suite: 32**
- **31 are coincidental matches.**
  - They pass on the original DB and fail on at least one variant.
  - Example: `spider_dev_0208` hard-codes `SourceAirport = 'ABE'` where gold joins on `City = "Aberdeen"`. On the original DB the two coincide; on `flight_2v5080` they don't.
  - These are genuine false positives of single-database EX.
- **1 (`spider_dev_0154`) fails already on the original DB under the default setting.**
  - Gold is `select distinct year … between 3000 and 4000`, and the prediction is an `INTERSECT` of two `SELECT DISTINCT`.
  - With DISTINCT removed, gold returns 107 rows but the `INTERSECT` still de-duplicates to 13.
  - With `--keep_distinct` it matches on the original DB but fails on the variant `car_1v5080` (14 vs 20 rows). The prediction is semantically different: years with *some* car under 4000 and *some* car over 3000, rather than one car between the two.
  - It is not credited under either setting.
- **All 32 fail under both settings,** default and `--keep_distinct`.

**Rejected by the official parser but passing the test suite: 177 of 461**

The categories are those of the parser audit, unchanged.

| Audit category | n | TS pass | TS fail |
|---|---:|---:|---:|
| Parser limitation (`INNER/LEFT JOIN`, `IN` lists, …) | 250 | 123 | 127 |
| Alias / formatting | 94 | 54 | 40 |
| Invalid for the schema (SQLite also rejects) | 116 | 0 | 116 |
| Other SQLite error | 1 | 0 | 1 |
| **total** | **461** | **177** | **284** |

- Of the 185 rejected predictions that matched gold on the original DB, **152 pass** the test suite and 33 fail. Those 33 are not semantically equivalent to gold under this criterion.
- The other 39 of the 216 official-EX = 0 / test-suite-pass cases parse but score 0 under official EX.
  - **31 also match gold on the original DB under the unofficial check.** They fall in the parser audit's "parses, rows match, EX = 0" group, whose established causes are column-id keying of SELECT expressions (e.g. `COUNT(col)` vs `COUNT(*)`) and row order without gold `ORDER BY`. The test-suite comparison keys nothing by expression.
  - **8 do not match under the unofficial check.** 7 of them pass only because DISTINCT is removed; 1 passes even with DISTINCT kept.

**Failing under both: 372.** This includes all 121 SQLite errors.

**Exact match:** 370 of the 378 EM hits pass the test suite. 8 EM hits fail it, and 260 test-suite passes are not EM hits.

**Unofficial result-set match (original DB only) × test-suite EX**

| | TS pass | TS fail |
|---|---:|---:|
| Unofficial match = 1 | 576 | 78 |
| Unofficial match = 0 | 54 | 326 |

- **The 78 (match, fail)** are 74 predictions distinguished only by a variant, plus 4 that fail on the original DB through DISTINCT removal. Example: `spider_dev_0087`: `COUNT(DISTINCT Continent)` becomes `COUNT(Continent)`.
- **The 54 (no match, pass):**
  - 19 pass only because DISTINCT is removed.
  - 35 pass even with DISTINCT kept. That fits the column-permutation invariance of `result_eq`, which the column-order-sensitive unofficial match lacks. I did not check this example by example.

**Earlier 3-variant check from the parser audit, against the full suite**

The audit's check is column-order-sensitive and keeps DISTINCT, so the two checks are not expected to agree exactly.

| Audit group | Audit verdict | TS pass | TS fail |
|---|---|---:|---:|
| Official EX hits (446) | all 3 match (374) | 363 | 11 |
| | distinguished (36) | 16 | 20 |
| | no variants, `wta_1` (36) | 35 | 1 |
| Rejected but result-matching (185) | all 3 match (155) | 147 | 8 |
| | distinguished (24) | 2 | 22 |
| | no variants, `wta_1` (6) | 3 | 3 |

- **The 19 "distinguished but TS pass" cases** are explained as follows:
  - 16 official EX hits pass `result_eq` with DISTINCT kept on the same 3 variants, so the audit flagged them only for column order.
  - The other 3 (2 rejected, 1 parsed with EX = 0) pass only after DISTINCT removal.
- **The 20 "all 3 match but TS fail" cases** (11 + 8 + 1) were caught by variants beyond the 3 sampled ones, or by DISTINCT removal.
- **The parser audit's conclusions stand:**
  - The 185 were never proven correct, and 33 now fail.
  - Official EX includes coincidental hits: 31 are confirmed.

**DISTINCT sensitivity**
- With `--keep_distinct`, the score is 603 / 1,034. 602 predictions pass under both settings.
- 28 pass only under the default:
  - in 13, only gold has `DISTINCT` (e.g. `spider_dev_0075`);
  - in 13, only the prediction has it;
  - in 2, both have it.
- 1 passes only under `--keep_distinct` (`spider_dev_0451`).
- 131 of the 1,034 examples have `DISTINCT` in the gold or the prediction (gold 87, prediction 109).

## 7. Tests

`tests/test_sql_v1_testsuite_eval.py` has 20 tests, using tiny synthetic SQLite DBs only. They cover:

- the vendored official files' hashes;
- `db_id` → variant mapping, with the original first;
- extraction that skips `__MACOSX` and non-requested DBs, is idempotent and refuses to overwrite a changed file;
- missing directory or missing original → `missing_variants`;
- a coincidental match on the original that fails on a variant;
- column-permutation invariance;
- row order only with gold `ORDER BY`;
- DISTINCT removed vs kept;
- the official `value` → `1` step;
- invalid SQL and empty predictions → incorrect;
- a recursive-CTE timeout → `pred_timeout`;
- gold error or timeout → not evaluable;
- gold and prediction always run on the same file;
- read-only connections;
- the dev-only guards: non-`spider_dev_` ids, `partition != dev`, `test_full`-named inputs, and test ids in the run file;
- refusal to write into a non-empty output folder.

Results:
- New tests: **20 passed**.
- Full suite (`python -m pytest`): **249 passed, 0 failed**.
- One clean-up was needed before the full suite passed:
  - The first full run had 2 failures in `tests/test_kaggle_sql_v1_baseline.py`. Their cause was two `__pycache__/*.pyc` files inside `kaggle/spider_sql_v1_dev_bundle_v1/code/official_eval/`, which made the bundle tree differ from its manifest.
  - Those files were written at 15:37 today by the earlier parser-audit run, not by this task's code.
  - I removed them. The bundle again matches its manifest, and the full suite then passed.

## 8. Is it reliable enough as a secondary semantic-correctness metric?

**Yes, as a secondary metric, with the disclosures below.**

**Why it is reliable**
- It is the official, current Spider methodology, and the code is unmodified.
- Coverage is complete: 20/20 DBs, 695 files and 1,034/1,034 examples, with 0 not evaluable.
- Gold as the prediction scores 1,034/1,034.
- It is deterministic.
- It does not depend on the 2019 parser. A fine-tuned model's Spider-style gains (`JOIN` instead of `INNER JOIN`, no aliases) cannot raise it by format alone.
- It removes coincidental single-DB matches (31 here).

**Limitations to disclose**
1. **It is an upper bound on semantic accuracy, not proof.** Zhong et al. call it a "tighter upper bound". A query can still pass all variants by coincidence, and `real_estate_properties` has only 4 examples.
2. **Default DISTINCT removal** strips every `DISTINCT`, including inside `COUNT(DISTINCT …)`.
   - It moves 29 verdicts here: 28 gained, 1 lost, net +27 (630 vs 603).
   - Report the setting, and `--keep_distinct` as sensitivity.
3. **Value strictness.** Values must be predicted exactly; `--plug_value` is off.
4. **Column order is ignored, and row order is enforced only when gold has `ORDER BY`.** This is more lenient than the SQL-0001 unofficial match.
5. **The timeout is a deviation.** The script enforces the 60 s timeout; the official code never does. Connections are read-only. No timeouts occurred.
6. **Not comparable with the primary SQL-0001 metric.** It is not official EX at b7b5b8c. Report it next to that metric, never in its place, and keep "Current best results" keyed to official EX/EM until a decision is documented.
7. **The variant DBs live outside the repo.** Reproducing the score requires `extract` from the pinned zip, which the manifest verifies.

**Report side by side for the baseline and every fine-tuned model** (same dev split, same 1,034 denominator)

| # | Metric | Role |
|---|---|---|
| 1 | Official EX (b7b5b8c) | primary, unchanged |
| 2 | Official EM (b7b5b8c) | primary, unchanged |
| 3 | Test-suite EX (e97acc5 defaults) | secondary semantic metric |
| 4 | Test-suite EX with `--keep_distinct` | sensitivity |
| 5 | Official-parser acceptance | shows how much of a gain is format |
| 6 | SQLite execution success | |
| 7 | Breakdown of the change between models | official-only gains vs test-suite gains; see below |

**What to break down.** Split the official EX/EM change into: newly parseable predictions; predictions that were already parseable and became correct; and the test-suite change. This shows how much of an official EX gain is style adaptation. The normalised style diagnostic from `docs/sql_v1_zero_shot_parser_audit.md` stays a diagnostic.

**Re-running for a fine-tuned run**

```
python -I scripts/sql_v1/evaluate_test_suite_dev.py extract                    # once; verifies/extracts 695 files
python -I scripts/sql_v1/evaluate_test_suite_dev.py evaluate --run <RUN_DIR> --out <NEW_DIR>/predictions
python -I scripts/sql_v1/evaluate_test_suite_dev.py evaluate --run <RUN_DIR> --out <NEW_DIR>/predictions_keep_distinct --keep-distinct
python -I scripts/sql_v1/evaluate_test_suite_dev.py compare  --run <RUN_DIR> --out <NEW_DIR>
```

`<RUN_DIR>` must hold an `evaluation_results.jsonl` with `eval_sql` and `gold_sql` for the 1,034 dev ids. Any non-dev id is refused.
