# Week 2 — Atlas-based evaluation of model-generated MQL (verified)

**Task.** Spider 1.0 (dev) natural-language questions → MongoDB aggregation pipelines (MQL).
Models: Qwen2.5-Coder-3B/7B-Instruct (local CPU) vs openai/gpt-oss-120b (Groq). Correctness
is decided **only by executing the generated MQL on MongoDB Atlas and comparing the result
rows with the validated expected result.** This follows the mentor's guidance.

**Status (2026-10-15): all three models evaluated and verified on all 279 examples.**
- **Split caveat.** A database-disjoint train/validation/test split (195/42/42) now exists in
  `data/splits/`. It was created **after** this evaluation, so none of the results below is a
  held-out test score (§9.2). The `$unionWith` validator gap is closed (§9.1).
- The project owner reversed the earlier decision and asked for full coverage. The
  gpt-oss-120b run was therefore **resumed** on the Groq free tier, for exactly the 68
  missing ids. The 211 existing results were kept (they are content-identical after the
  resume). There were 0 provider errors and $0 spend.
- A live re-execution on Atlas then reproduced every verdict:
  - **gold MQL 279/279**;
  - **model verdicts 279/279 for each of Qwen-3B, Qwen-7B and gpt-oss-120b.**
- Qwen predictions, gold MQL and the dataset are unchanged; their SHA-256 hashes were
  checked before and after.
- **Final strict execution accuracy (n = 279 each):**

  | Model | Correct |
  |---|---|
  | Qwen-3B | **132/279 = 47.3%** |
  | Qwen-7B | **181/279 = 64.9%** |
  | gpt-oss-120b | **232/279 = 83.2%** |

> The earlier 211-example partial state (gpt-oss 176/211 = 83.4%) is superseded. It is kept
> for audit in §5.3 and in `outputs/evaluation_spider1/gpt-oss-120b/history/`.

## 1. Pipeline audit: question → MQL → Atlas → result comparison → verdict

The code path for every example and every model (`scripts/spider1/run_eval.py`):

1. **Prompt.** `prompt_spider1.messages(db_id, question)` builds the prompt (version
   `nl-mql-spider1-v1`; its SHA-256 is stored in every record). It contains the schema of
   that example's collections, foreign keys, 2 sample documents per collection, and the
   question. It never contains gold SQL, gold MQL or results.
2. **Generation.** The provider returns `raw_text`, which is saved verbatim in
   `executions.jsonl`.
3. **Parse.** `run_nl_mql_eval.parse_output` reads the JSON object
   `{"collection", "pipeline"}`. If no object can be parsed, the record is `format_error`.
4. **Read-only validation** (`eval_kit.validate_mql`). These are rejected as `invalid_mql`:
   - a top-level (or `$facet`) stage named `$out` or `$merge`, a server-admin stage, or
     `$where`, `$function` or `$accumulator`;
   - a target collection outside the example's database;
   - any `from` or `coll` reference (`$lookup`, or the document form of `$unionWith`) to a
     collection outside it;
   - (added in the Week 2 finalisation, §9) any `$unionWith`, at any depth, whose target is
     not one of the example's collections, in **either** form: the short form
     `{"$unionWith": "<coll>"}` or the long form `{"$unionWith": {"coll": …, "pipeline": […]}}`;
   - a malformed `$unionWith`: an empty or blank name, a non-string/non-object value, a long
     form without a non-blank `coll`, extra keys such as `db`, or a non-list `pipeline`.

   Remaining known gap: JavaScript operators nested inside a stage are not detected. It did
   not affect any score (see §7).
5. **Atlas execution.**
   `client["codegen_spider1"][collection].aggregate(pipeline, maxTimeMS=30000, allowDiskUse=True)`.
   - A server error or timeout gives `execution_error`.
   - More than 100,000 rows also gives `execution_error` ("result too large").
6. **Result comparison** (`spider1_kit.docs_to_rows` + `exec_match`). The output documents
   become rows, which are compared with the example's **`gold_rows`**: the result of the
   official gold SQL on the Spider SQLite DB, which the gold MQL also reproduces on Atlas.
   - Column names and column order are ignored, and the column count must match.
   - Numbers are compared at 4 decimal places.
   - Rows are compared as a multiset, unless the gold has ORDER BY. Then order matters,
     except among tied rows.
7. **Verdict.**
   - `correct` only if `exec_match` succeeds.
   - Otherwise the record gets `wrong_row_count`, `wrong_column_count`, `wrong_values` or
     `order_mismatch`, with up to 10 predicted rows saved.
   - Wrong answers are never retried.

**Accuracy is not text similarity.** No step compares the predicted MQL with the gold MQL
text. A query that differs from the gold but returns the same rows is correct, which is
covered by a unit test (`tests/test_spider1_atlas_eval.py`).

**Example → Atlas mapping.**
- Each example carries its Spider `db_id`.
- Every SQLite table of that DB was imported to Atlas DB `codegen_spider1` as collection
  `<db_id>__<table>` (`spider1_kit.collection_name`). There are 20 databases.
- The allowed collections for an example are the Atlas collections with prefix
  `<db_id>__`. The model must pick one of them, and any `$lookup` must stay inside them.
- Test-suite variants used to validate the dataset live in `codegen_spider1_ts1..3`. They
  are not used for model scoring.

**What the audit changed.** The audit found no missing execution or validation step, so the
pipeline was preserved. The only code change is a behaviour-preserving refactor: steps 3–7
moved into `run_eval.score_output()`. The live runner and the new verifier now call one
identical scoring path.

## 2. Dataset and validation methodology

- **Dataset.** `data/processed/spider1_nl_mql.jsonl` holds **279** validated examples from 20
  Spider dev databases (300 sampled, 21 excluded with recorded reasons). See
  `docs/spider1_dataset.md`.
  - Hardness: easy 71, medium 126, hard 44, extra 38.
  - 50 examples have an ORDER BY gold.
- **Trusted expected result** = `gold_rows`.
  - The gold SQL's result on the official DB is accepted only when the gold MQL executed on
    Atlas reproduces it under the same comparator.
  - 267/279 also match on 3 official test-suite DB variants. The other 12 are dev-only:
    11 wta_1 and `0667`.
  - 13 examples are flagged `ambiguous`. Robustness slices that exclude them are reported in
    `outputs/evaluation_spider1/comparison.md`.
- **Qwen-3B scope.** Qwen-3B was run before the 21 exclusions, on the full 300-item sample.
  - Its `summary.json` therefore shows 134/300.
  - The 279-example figure is that run restricted to the dataset ids: 132/279.
  - For all 279 ids, `gold_rows`, `order_by` and `tie_blocks` in the old sample file are
    **identical** to the final dataset. So no 3B verdict was scored against a superseded
    target. Re-execution confirms this (§4).

## 3. Atlas execution and verification procedure (Week 2)

`scripts/spider1/rescore.py` verifies saved results without calling any model.

| Level | Command | What it proves |
|---|---|---|
| offline | `python scripts/spider1/rescore.py` | Recounts strict accuracy and the paired table from raw records. Checks record integrity: unique ids, one model id and one prompt hash per run, and that every `correct` has `executed=True` and a saved MQL. Re-parses all format errors. Re-scores every mismatch whose full result was saved (≤10 rows) against the current gold rows. **It cannot confirm "correct" verdicts, because their rows are not stored.** |
| live, read-only | `python scripts/spider1/rescore.py --reexecute` | (a) **Gold replay:** every gold MQL goes through the same score path on Atlas; anything below 279/279 would mean the data drifted. (b) Every saved model `raw_text` is re-parsed, re-validated, **re-executed on Atlas** and re-scored. Each verdict is compared with the recorded one. |

- The original `executions.jsonl` and `summary.json` files are never written. Recorded
  verdicts stay canonical.
- The Atlas URI is read from `.env` (`MONGODB_URI`) through `spider1_kit.mongo_client()`.
  It is never printed. The outputs were checked and contain no credentials.

## 4. Verification results

| Check | Qwen-3B | Qwen-7B | gpt-oss-120b |
|---|---|---|---|
| raw records (in dataset) | 300 (279) | 279 (279) | 279 (279) |
| duplicate ids / provider errors | 0 / 0 | 0 / 0 | 0 / 0 |
| model ids / prompt hashes per run | 1 / 1 | 1 / 1 | 1 / 1 |
| `correct` without execution evidence | 0 | 0 | 0 |
| offline: format errors re-parsed (still fail) | 2 (2) | 3 (3) | 8 (8) |
| offline: mismatches re-scored (still mismatch) | 108 (108) | 74 (74) | 23 (23) |
| **live Atlas re-execution: correct** | **132** | **181** | **232** |
| **live verdict agreement with recorded** | **279/279** | **279/279** | **279/279** |

- **Gold MQL replay on Atlas: 279/279 correct.**
- The live re-execution was performed on 2026-10-15 against `codegen_spider1`: 19:36 local on
  the 211-item state, and again at 19:58 after the resume. The second run gives the figures above.
  The evidence is in `outputs/evaluation_spider1/week2_verification/`.

## 5. Results: strict execution accuracy

Strict execution accuracy (EX) means the generated MQL executes on Atlas **and** its result
matches `gold_rows`. Format errors, invalid MQL, execution failures and mismatches all count
as wrong. 95% Wilson CIs are in parentheses.

### 5.1 Primary: all three models on the same 279 examples

| Model | Evaluated | EX (n = 279) | 95% CI | easy (71) | medium (126) | hard (44) | extra (38) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-3B | 279/279 | **132/279 = 47.3%** | 41.5–53.2% | 47 | 74 | 7 | 4 |
| Qwen2.5-Coder-7B | 279/279 | **181/279 = 64.9%** | 59.1–70.2% | 58 | 98 | 16 | 9 |
| **gpt-oss-120b** | 279/279 | **232/279 = 83.2%** | 78.3–87.1% | 64 | 107 | 37 | 24 |

| Pair (n = 279) | both correct | only first | only second | neither | McNemar exact p |
|---|---:|---:|---:|---:|---|
| Qwen-3B vs Qwen-7B | 123 | 9 | 58 | 89 | 6.8e-10 |
| Qwen-3B vs gpt-oss | 119 | 13 | 113 | 34 | 4.5e-21 |
| Qwen-7B vs gpt-oss | 163 | 18 | 69 | 29 | 3.3e-08 |

Robustness slices (3B / 7B / gpt-oss):
- unflagged items (266): 49.2% / 66.5% / 85.7%;
- test-suite-backed items (267): 47.6% / 66.3% / 82.4%.

### 5.2 Type-conversion breakdown (`slice_type_conversion.json`)

The gold needs an explicit type conversion (`$toString`, `$toInt`, …) on 16 of the 279.

| Model | needs conversion (16) | other (263) |
|---|---|---|
| Qwen-3B | 0/16 | 132/263 = 50.2% |
| Qwen-7B | 1/16 | 180/263 = 68.4% |
| gpt-oss-120b | 13/16 | 219/263 = 83.3% |

### 5.3 Historical: 211-example partial comparison (superseded)

Before the resume, gpt-oss had covered dataset positions 1–211 only. On those paired ids:
- Qwen-3B 97/211 = 46.0%;
- Qwen-7B 133/211 = 63.0%;
- gpt-oss 176/211 = 83.4%.

On the 68 ids added on 2026-10-15, gpt-oss scored 56/68 = 82.4% (Qwen-3B 35/68, Qwen-7B
48/68). The full-dataset figure is therefore close to the partial one.

## 6. Error categories (all 279 ids per model)

| Outcome | Qwen-3B | Qwen-7B | gpt-oss-120b |
|---|---:|---:|---:|
| correct | 132 | 181 | 232 |
| malformed output: truncated at the 1,024-token cap | 1 | 0 | 0 |
| malformed output: invalid JSON | 1 | 3 | 8 |
| invalid MQL (rejected by the read-only validator) | 0 | 0 | 0 |
| execution failure (Atlas error / timeout) | 15 | 4 | 5 |
| result mismatch: wrong row count | 74 | 54 | 22 |
| result mismatch: wrong column count | 37 | 25 | 5 |
| result mismatch: wrong values | 19 | 12 | 7 |
| result mismatch: order only | 0 | 0 | 0 |
| missing prediction / provider error | 0 | 0 | 0 |
| **total** | 279 | 279 | 279 |

Notes:
- **gpt-oss-120b's 8 malformed outputs share one defect.** Each is a short, otherwise valid
  answer with one extra `]` before the final `}` (e.g. `…{"$limit":1}]]}`), and every
  response ended normally (`stop_reason = stop`).
  - Two of the 8 (`0800`, `1002`) are in the resumed 68.
  - Under the strict metric they are wrong, and they stay wrong.
  - A separate sensitivity check (`sensitivity_repair.py`) removes that one character. Then
    7 of the 8 become correct, giving 239/279 = 85.7%. This is not the reported metric.
- **Execution failures.**
  - Qwen-3B: 10 of its 15 are Atlas rejecting a misused operator name ("… is not allowed or
    the syntax is incorrect": 7 `match`, 1 `sort`, 1 `group`, 1 `$distinct`). There is also 1 timeout
    (`0437`).
  - gpt-oss: 3 "unknown top level operator" errors and 2 plan-executor errors, all in the
    first 211; the resumed 68 had none.
- **Root causes.** The dominant root-cause tags (`error_analysis_<model>.json`) are extra
  projected columns, empty results from type-mismatched `$lookup`/`$match` (string vs
  number keys), and missing `$group`/`$limit`.
- No model ever produced a forbidden or out-of-database stage (`invalid_mql` = 0). No model
  used the `find` form, so the validator's dropping of `find`-only `sort`/`limit` keys
  affected nothing.

## 7. Coverage limitations and remaining caveats

- **Coverage is complete:** 279/279 for every model, with no missing predictions or provider
  errors.
- **Two generation sessions for gpt-oss.** gpt-oss-120b's answers were generated in two
  sessions five days apart: positions 1–211 by 2026-10-10, and the 68 resumed ids on
  2026-10-15. Both used the same model id, prompt and settings. Groq exposes no model
  revision, so a silent server-side change cannot be ruled out. Accuracy on the two parts is
  similar (83.4% vs 82.4%).
- **"Frontier" caveat.** gpt-oss-120b is a large open-weight model served by Groq, not a
  closed frontier model.
- **Natural order.** A few golds rely on MongoDB natural (insertion) order for ties, which is
  not guaranteed. The 2026-10-15 re-execution gave identical verdicts, but this could change
  if collections are re-imported.
- **Generalisation.** Correct verdicts are checked on the dev DB contents only. Test-suite
  variants validate the gold MQL, not model predictions. A prediction can be right on this
  data by coincidence.
- **Contamination.** Spider dev is public, so contamination cannot be ruled out.
- **Validator gaps.** These have no effect on the reported numbers.
  - Across all saved outputs, the only use of `$unionWith`, `$where`, `$function` or
    `$accumulator` is Qwen-7B `0926`. It is a string-form `$unionWith` to a collection of the
    same database (`dog_kennels__Owners`), and Atlas rejected it (`execution_error`).
  - The string-form check has now been added (§9.1). It passes `0926` (allowed target), so its
    recorded verdict is unchanged. Nested JavaScript operators are still not detected.
- **Retrospective split.** The train/validation/test split (§9.2) was created after all
  three models were evaluated on all 279 examples. See §9.2 for what that means.

## 8. Reproduce from saved predictions (no model calls)

```powershell
python -m pytest -q                                          # unit tests (mocked Atlas; no network)
python scripts/spider1/make_splits.py                        # re-derive the split from its seed and verify data/splits/
python scripts/spider1/rescore.py                            # offline audit of saved records
python scripts/spider1/rescore.py --reexecute                # + gold replay and re-execution on Atlas (read-only)
python scripts/spider1/compare.py qwen2.5-coder-3b qwen2.5-coder-7b gpt-oss-120b   # paired report (overwrites comparison.md/json)
python scripts/spider1/error_analysis.py qwen2.5-coder-3b qwen2.5-coder-7b gpt-oss-120b
python scripts/spider1/slice_type_conversion.py qwen2.5-coder-3b qwen2.5-coder-7b gpt-oss-120b
python scripts/spider1/slice_type_conversion.py --paired qwen2.5-coder-3b qwen2.5-coder-7b gpt-oss-120b
python scripts/spider1/sensitivity_repair.py qwen2.5-coder-3b qwen2.5-coder-7b gpt-oss-120b
```

Outputs:
- `outputs/evaluation_spider1/week2_verification/`:
  - `audit_offline.{md,json}`;
  - `verification.{md,json}` (live run);
  - `gold_replay.jsonl`;
  - `reexec_<model>.jsonl`, one per model, with per-id recorded vs re-executed verdicts.
- `outputs/evaluation_spider1/comparison.{md,json}` (three-way report, regenerated on all 279).
- `outputs/evaluation_spider1/gpt-oss-120b/history/` (archived 211-example partial state).
- Raw evidence: `outputs/evaluation_spider1/<model>/executions.jsonl`.

## 9. Week 2 finalisation: validator fix and dataset split (2026-10-15)

No model was re-run, no API was called, and no prediction, gold MQL or dataset record was
changed. The source dataset hash is unchanged
(`e727b7591556ae46f92efff603652dbfe7087443109cb42b2101ba8204bdf7b0`). So are the SHA-256
hashes of the Qwen-3B, Qwen-7B, gpt-oss-120b and reference `executions.jsonl` files and of
every `data/interim/spider1/gold_mql/*.json` file, checked before and after. The headline results stay **Qwen-3B 132/279
(47.3%), Qwen-7B 181/279 (64.9%), gpt-oss-120b 232/279 (83.2%)**.

### 9.1 `$unionWith` target validation

`scripts/evaluation/eval_kit.py`:
- The new `union_with_targets(pipeline)` walks the whole pipeline, including `$facet`,
  `$lookup.pipeline` and `$unionWith.pipeline`. It returns every `$unionWith` target, or an
  error for a malformed one.
- `validate_mql` adds these targets to the reference check. So every target must be in the
  example's allowed set: the Atlas collections with prefix `<db_id>__`, as built by
  `run_eval.py`.

| `$unionWith` value | Result |
|---|---|
| `"<db_id>__<table>"` of this example | accepted |
| `{"coll": "<db_id>__<table>"}`, with an optional `"pipeline": [...]` | accepted |
| a collection of another database, or `system.*` (either form, any depth) | `invalid_mql`: references collections outside the example |
| `""` or a blank string | `invalid_mql`: empty target |
| a number, `null`, `true` or a list | `invalid_mql`: malformed |
| `{}`, or `coll` empty, blank or not a string | `invalid_mql`: no collection name |
| a collection-less `{"pipeline": [...]}` | `invalid_mql` |
| extra keys, e.g. `{"coll": …, "db": "admin"}` | `invalid_mql`: unsupported keys |
| `pipeline` that is not a list | `invalid_mql` |

- Malformed values are rejected even when no allow-list is passed.
- With an empty allow-list, reference checks stay disabled, as before. This only applies to
  legacy callers; the Spider 1.0 runner always passes a non-empty set.

**Behaviour preserved.** The old and new validators were both replayed offline over 1,416
records:
- 279 gold MQL;
- every saved Qwen-3B (300), Qwen-7B, gpt-oss-120b and reference prediction.

There were **0 differences**. The 13 records that never parse are unaffected. The only
string-form use, Qwen-7B `0926`, targets an allowed collection, so it still passes validation
and keeps its recorded `execution_error`. The 68-example reference outputs that use
`$unionWith` (`local022`, `local301`) also still pass.

### 9.2 Train / validation / test split

Files are in `data/splits/`: `train.jsonl`, `validation.jsonl`, `test.jsonl` and
`split_manifest.json`. They are produced by `scripts/spider1/make_splits.py`.

| Partition | Examples | Share | Databases | easy / medium / hard / extra |
|---|---:|---:|---|---|
| train | 195 | 69.9% | car_1, concert_singer, cre_Doc_Template_Mgt, dog_kennels, flight_2, orchestra, poker_player, singer, student_transcripts_tracking, voter_1, world_1, wta_1 | 50 / 89 / 27 / 29 |
| validation | 42 | 15.1% | battle_death, course_teach, museum_visit, network_1, real_estate_properties | 11 / 18 / 8 / 5 |
| test | 42 | 15.1% | employee_hire_evaluation, pets_1, tvshow | 10 / 19 / 9 / 4 |

**The split is database-disjoint.** All examples of a Spider `db_id` are in one partition,
so no schema or collection is shared across partitions. The 20 databases range from 1 to 34
examples, which made the 70/15/15 targets (195/42/42) reachable exactly.

**Method.** The method is fully determined by seed `20261015` and 10,000 draws:
1. Sort the distinct `db_id`s.
2. For each draw, shuffle them with `random.Random(seed)`.
3. Fill validation up to 42, then test up to 42, and put the rest in train.
4. Keep the draw with the smallest count deviation. Break ties on the smallest deviation of
   the hardness mix from the overall mix, and then by the earliest draw.

The chosen draw was 1003, with count deviation 0. Only `db_id` and the official Spider
hardness labels are used; no model output or score is.

Split files contain the source lines **byte-for-byte**, in source order. The manifest
records:
- the seed, draw count, procedure and command;
- the source path, SHA-256 and size;
- each partition's ids, count, file hash, databases and hardness mix;
- the database → partition map;
- a per-example `{id, db_id, partition}` list covering all 279 examples;
- the retrospective note.

**Verification.** Both `python scripts/spider1/make_splits.py` and
`tests/test_spider1_split.py` re-derive the split from the recorded seed and method. They
check that the files and manifest are reproduced exactly, and that:
- every one of the 279 source ids is in exactly one partition;
- no id overlaps;
- no database overlaps;
- counts, ids and hashes match the manifest;
- records equal the source.

The split was also reproduced under `PYTHONHASHSEED` 0, 1 and 12345.

> **Limitation: the split is retrospective.** Qwen2.5-Coder-3B, Qwen2.5-Coder-7B and
> gpt-oss-120b were all evaluated on all 279 examples **before** this split existed. No model
> was trained or tuned on any partition, and the prompts are zero-shot. Still, the results
> above are **not** an untouched held-out evaluation, and neither is any per-partition
> subset of those runs. In particular, the 42 "test" examples were seen during the Week 2
> evaluation and error analysis. The split is intended for future work (few-shot example
> selection, fine-tuning, prompt tuning), which must use only train/validation and touch
> test once.

### 9.3 Tests

`python -m pytest`: **87 passed, 0 failed**. Before this change there were 46 tests.

- `tests/test_mql_validator.py`, 29 tests (new):
  - permitted short-form and long-form targets, including inside `$facet`, `$lookup` and
    `$unionWith` pipelines;
  - unauthorized collections, at top level and nested;
  - 14 malformed values;
  - malformed values rejected without an allow-list;
  - existing `$lookup`, `$out` and collection checks unchanged;
  - all 279 gold MQL pass.
- `tests/test_spider1_atlas_eval.py`: 3 cases added (unauthorized string form, empty
  string, unauthorized long form). Each is rejected as `invalid_mql` **before** any
  execution.
- `tests/test_spider1_split.py`, 9 tests (new): source hash, exactly-one-partition, database
  disjointness, manifest consistency, the ≈70/15/15 ratio, unchanged records, seed
  reproducibility, input-order independence, and that the retrospective note is recorded.
