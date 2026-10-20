# Spider 1.0 → MQL, expanded benchmark (455 examples, no gold MQL)

This is a versioned expansion of the 279-example benchmark (`docs/spider1_dataset.md`,
`docs/spider1_model_comparison.md`). Those files, the 279-example dataset
(`data/processed/spider1_nl_mql.jsonl`, sha256 `e727b759…`), and their predictions and reports
are unchanged. Everything here lives in separate, versioned paths.

**Headline (n = 455, strict execution accuracy, Wilson 95% CI):**

| model | correct | accuracy |
|---|---:|---|
| Qwen2.5-Coder-3B-Instruct (GGUF q4_k_m, CPU) | 217 | **47.7%** [43.1, 52.3] |
| Qwen2.5-Coder-7B-Instruct (GGUF q4_k_m, CPU) | 287 | **63.1%** [58.6, 67.4] |
| openai/gpt-oss-120b (Groq free tier) | 385 | **84.6%** [81.0, 87.6] |

All three models were scored on exactly the same 455 examples. Every example has a recorded
outcome for every model: no missing items, no provider errors and no duplicates. Each pairwise
difference is significant (exact McNemar, p < 1e-6).

## 1. What changed vs the 279-example benchmark

| | v1 (279) | expanded (455) |
|---|---|---|
| candidates | seed-42 stratified sample of 300 from the 493-item pool | **all 493** pool items |
| reference answer | gold-SQL result on SQLite | same |
| evidence that Atlas can answer the question | hand-authored gold MQL, matched on the dev DB and 3 test-suite variants | **table-level SQLite↔Atlas data parity** (no gold MQL) |
| scorer | `spider1_kit.exec_match` | same function, same settings |
| prompt / generation config | `nl-mql-spider1-v1`, temp 0, seed 42, 1024 new tokens | identical (reuse verified, §4) |

No gold MQL was written or used for the expanded set. Scoring was always "model MQL result on
Atlas vs SQL result on SQLite". Gold MQL only ever served as evidence that the Atlas data could
produce the answer. In the expanded set, that role is filled by a direct parity check of the data.

## 2. Pipeline

| step | script | output |
|---|---|---|
| 1. re-derive the candidate pool with the original pre-registered filters | `build_pool.py --out-dir <scratch>` | byte-identical to `data/interim/spider1/pool.jsonl` (sha256 `d3b1de85…`, 493 items) |
| 2. SQLite↔Atlas parity, every table of all 20 DBs | `parity_check.py` | `data/interim/spider1/expanded/parity.json` |
| 3. NL/SQL audit of the 193 pool items outside the v1 sample | manual review + SQLite probes | `data/interim/spider1/expanded/audit_new_items.json` |
| 4. build the dataset: re-execute every SQL, apply curation and parity gates | `build_expanded.py` | `data/processed/spider1_nl_mql_expanded.jsonl` (sha256 `59b6c1f3…`), `expanded/status.jsonl`, `expanded/dataset_summary.json` |
| 5. reuse eligible v1 predictions (re-executed on Atlas) | `seed_expanded.py` | `outputs/evaluation_spider1_expanded/reuse_manifest.json` |
| 6. generate the missing predictions (resumable, quota-aware) | `run_until_complete.py` → `run_eval.py` | `outputs/evaluation_spider1_expanded/<model>/executions.jsonl` |
| 7. re-execute every saved prediction on Atlas | `rescore_expanded.py` | `outputs/evaluation_spider1_expanded/rescore_log.json` |
| 8. report | `report_expanded.py` | `outputs/evaluation_spider1_expanded/report.{json,md}` |

### 2.1 Data parity (step 2)
For every table, the SQLite rows (read-only) are compared with the Atlas documents
(`find` only). The checks are:
- row count;
- field names;
- **multiset of type-tagged rows**, so TEXT `'3'` ≠ INTEGER `3` ≠ REAL `3.0`, and NULL ≠ missing;
- natural order vs rowid order.

Result: 19 of 20 databases are identical in every table, and their natural order equals rowid
order. **wta_1** has a single difference: two `players` names are not valid UTF-8 in SQLite.
BSON can only hold valid UTF-8, so Atlas stores the U+FFFD-replaced text. Rather than drop the
database, `build_expanded.py` re-runs each wta_1 SQL on an in-memory copy where those two strings
are replaced exactly as on Atlas. An item is kept only if its result is unchanged; all 23 are
unchanged.

### 2.2 Curation (steps 3–4)
- **300 v1-sample items:** the v1 curation stands. The 279 validated items are included, and
  `build_expanded.py` asserts that their question, SQL, gold rows, ordering and tie data are
  byte-identical to v1. The 21 v1 exclusions keep their written reasons.
- **193 new items:** I reviewed every question, SQL and result. Targeted SQLite probes checked
  numbers stored as TEXT, whitespace padding, case sensitivity, NULL ordering, ties inside
  subqueries, and NL-vs-SQL deviations. Policy:
  - **exclude** when the gold result is not the correct answer to the question on this database,
    or is nondeterministic;
  - **keep and flag `ambiguous`** when the SQL deviates from or under-determines the question but
    its result is still a defensible answer on this data.
- Every expected result was re-executed now and must equal the frozen pool result.

| | count |
|---|---:|
| eligible pool | 493 |
| excluded: v1 adjudication | 21 |
| excluded: NL/SQL audit of new items | 17 |
| excluded: expected-result or parity gate | 0 |
| **final dataset** | **455** (279 from v1 + 176 new) |
| flagged `ambiguous` but kept | 20 (13 v1 + 7 new) |

By hardness: easy 115, medium 204, hard 72, extra 64. All 20 dev DBs are covered, and all 455
questions are unique.

**The 17 new exclusions** (full reasons are in `audit_new_items.json` and `expanded/status.jsonl`):

| id | db | kind | reason |
|---|---|---|---|
| 0125 | car_1 | data quirk | `horsepower > 150` on a TEXT column is a string comparison: gold 281, numeric answer 49 |
| 0159 | car_1 | data quirk | TEXT horsepower with `'null'` placeholders: ORDER BY picks a `'null'` row; gold 39, intended 399 |
| 0161 | car_1 | annotation mismatch | "how many countries…": SQL returns per-country counts, not the number of countries |
| 0169 | car_1 | data quirk | `max(mpg)` on TEXT returns the string `'null'` (intended 43.1) |
| 0203, 0205, 0207, 0211, 0215, 0217 | flight_2 | data quirk | airport codes and cities are whitespace-padded, so exact matches give 0; trimmed answers are 11, 7, 21, 2, 1, 3 (same rule as v1 exclusions 0209/0219/0255) |
| 0533 | student_transcripts | annotation mismatch | "2 degree programs in one semester": SQL counts 2 enrolments in total |
| 0583 | student_transcripts | annotation mismatch | `country = 'haiti'` vs stored `'Haiti'` (case-sensitive `=`) drops 3 of 4 answers |
| 0641 | tvshow | annotation mismatch | asks for all channel ids; GROUP BY country returns one bare id |
| 0687 | voter_1 | annotation mismatch | "how many states" counts 305 area-code rows; there are 51 states |
| 0942 | dog_kennels | annotation mismatch | "at least two types of treatments": SQL counts treatments, not types |
| 0954 | dog_kennels | nondeterministic | "rarest breed" subquery LIMIT 1 falls on a tie (ESK = HUS = 4) |
| 0960 | dog_kennels | annotation mismatch | "youngest dog": SQL uses `max(age)` |

**The 7 new flagged-but-kept items:**
- 0037, 0065: duplicate rows vs a distinct reading;
- 0241: "at least 10" vs `> 10`, identical here because every airline has 100 flights;
- 0701: case-insensitive LIKE vs the quoted `'Al'`;
- 0742: rows vs distinct languages, with the same answer here;
- 0774: "less than any" read as `< max`;
- 1032: under-specified wording.

## 3. Scoring

`run_eval.score_output` performs four steps:
1. **Parse:** the raw output is parsed as JSON.
2. **Validate:** `eval_kit.validate_mql` checks collections and references are restricted to the
   item's database and blocks forbidden stages. A new **deep scan**
   (`run_eval.forbidden_anywhere`) also rejects `$out`/`$merge`/`$function`/`$where`/
   `$accumulator`/admin stages at any depth, including inside expressions and
   `$lookup`/`$unionWith` sub-pipelines.
3. **Execute:** only an `aggregate` is run on Atlas `codegen_spider1`, with `maxTimeMS` = 30 s.
4. **Compare:** `exec_match` compares the result with the SQLite result.

The comparator is unchanged from v1:
- it compares row multisets, and is order-sensitive only for a top-level ORDER BY, with rows
  tied on the sort key allowed in any order;
- the column count must match, but column names and order are ignored;
- numbers are rounded to 4 decimals, so `3 == 3.0`, but strings must match exactly, so `'9'` ≠ `9`;
- NULL equals a missing field.

Outcomes are recorded separately: `malformed_output` (truncated or not), `validation_rejected`,
`execution_error`, `timeout` (new; previously folded into execution errors) and
`result_mismatch` (row count, column count, values, or order).

Each record persists:
- the raw output and the validated query;
- the execution status and the actual result rows (up to 1,000; gold has at most 1,000);
- the expected rows and the verdict;
- tokens, latency and request ids.

**Scorer fix applied after generation.** `validate_mql` rejected an empty pipeline
(`{"pipeline": []}`), although that is valid read-only MQL that returns every document. It is
now run as its exact no-op equivalent `[{"$match": {}}]`. The fix changes one item (0291,
`SELECT * FROM hiring`) for 3B and gpt-oss, from rejected to correct. No v1 record was affected.
`rescore_expanded.py` then re-executed all 1,365 saved predictions on Atlas without model calls:
1,363 verdicts reproduced unchanged, and the 2 fixes are logged with the old verdicts kept.

**Atlas safety.**
- Only aggregate reads run on Atlas, and only after validation.
- The connection string comes from `.env` and is never printed or logged.
- No model call reached a paid tier: Groq free tier, local GGUF.
- **Server-side read-only access (verified 2026-10-20):** `MONGODB_URI` in `.env` now
  authenticates as **`codegen_eval_readonly`**. Atlas `connectionStatus` (with
  `showPrivileges`) reports a single role, `codegen_spider1_read`, whose privileges are
  read-only actions (`find`, `listCollections`, `listIndexes`, `collStats`, `dbStats` and
  similar) on `codegen_spider1` only, with no insert, update, remove, drop or admin actions.
  The client-side checks above still apply on top of these permissions.
- Under this user, `rescore_expanded.py --dry-run` reproduced all 455 verdicts for each of the
  three models with zero changes, and all 94 tests pass.
- The test-suite databases `codegen_spider1_ts1`, `codegen_spider1_ts2` and
  `codegen_spider1_ts3` are outside this user's permissions. The expanded benchmark does not use
  them, but the v1 test-suite checks (`testsuite_check.py`) would need `read` on them added to
  the role.
- The generation runs and the earlier rescores were made while the URI still authenticated as
  `atlasAdmin`, so at that time read-only behaviour was enforced only by the client-side checks.

## 4. Prediction reuse
`seed_expanded.py` reused v1 predictions only when all of the following held:
- the model and generation config blocks were identical;
- the model id, prompt version and template hash matched;
- the prompt-context files (`atlas_collections.json`, `prompt_samples.json`) had the same hash and
  were last modified before the v1 runs;
- the GGUF sha256 equalled the pinned value (3B `724fb256…`, 7B `509287f7…`);
- the record was not a provider error, and its question and db matched.

All checks passed for all three runs, so 279 records per model were reused. The 21 extra 3B
records for v1-excluded items were not reused. The reused raw outputs were **re-executed** on
Atlas, which reproduced every v1 correctness verdict. The only change was one 3B
`execution_error` that is now classified `timeout`. The remaining 176 items per model were
generated fresh with the same config, 528 generations in all. A finished id is never requested
again. Groq answered 175 requests with HTTP 429 (rate limit), and the provider retried them after
`Retry-After`; a rejected request produces no generation. At most one in-flight request could
have been re-sent for each of the two memory-pressure process kills.

## 5. Results

| | Qwen-3B | Qwen-7B | gpt-oss-120b |
|---|---:|---:|---:|
| **execution accuracy, all 455** | 217 = 47.7% [43.1, 52.3] | 287 = 63.1% [58.6, 67.4] | 385 = 84.6% [81.0, 87.6] |
| easy (115) | 66.1% [57.0, 74.1] | 80.0% [71.8, 86.3] | 90.4% [83.7, 94.6] |
| medium (204) | 58.3% [51.5, 64.9] | 75.0% [68.6, 80.4] | 85.8% [80.3, 89.9] |
| hard (72) | 20.8% [13.1, 31.6] | 36.1% [26.0, 47.6] | 87.5% [77.9, 93.3] |
| extra (64) | 10.9% [5.4, 20.9] | 25.0% [16.0, 36.8] | 67.2% [55.0, 77.4] |
| v1 279 subset | 132 = 47.3% (= v1) | 181 = 64.9% (= v1) | 232 = 83.2% (= v1) |
| 176 new items | 85 = 48.3% [41.0, 55.6] | 106 = 60.2% [52.8, 67.2] | 153 = 86.9% [81.2, 91.1] |
| without 20 `ambiguous` items (435) | 49.4% | 64.8% | 86.9% |

**Output validity is not accuracy.** These counts show how many outputs were well-formed and
executable, regardless of correctness:

| | Qwen-3B | Qwen-7B | gpt-oss-120b |
|---|---:|---:|---:|
| valid JSON | 453 | 452 | 444 |
| passed read-only validation | 453 | 452 | 444 |
| executed without error or timeout | 430 | 441 | 439 |

**Outcome categories (sum = 455):**

| | Qwen-3B | Qwen-7B | gpt-oss-120b |
|---|---:|---:|---:|
| correct | 217 | 287 | 385 |
| malformed output (truncated / other) | 1 / 1 | 0 / 3 | 0 / 11 |
| validation rejected | 0 | 0 | 0 |
| execution error | 22 | 10 | 5 |
| timeout | 1 | 1 | 0 |
| result mismatch: row count / column count / values / order | 124 / 62 / 27 / 0 | 85 / 45 / 23 / 1 | 35 / 7 / 11 / 1 |

All 11 gpt-oss malformed outputs are the same defect: a complete pipeline followed by a
stray `]` (`…}]]}`), with a normal `stop` finish. They are scored strictly, as in v1.

**Paired (same 455 items):**

| pair | both correct | only first | only second | neither | McNemar exact p |
|---|---:|---:|---:|---:|---:|
| 3B vs 7B | 196 | 21 | 91 | 147 | < 1e-6 |
| 3B vs gpt-oss | 195 | 22 | 190 | 48 | < 1e-6 |
| 7B vs gpt-oss | 261 | 26 | 124 | 44 | < 1e-6 |

**Comparison with the 279-example benchmark.**
- The ranking and the gaps hold on the larger set:
  - 3B: 47.3% → 47.7%
  - 7B: 64.9% → 63.1%
  - gpt-oss: 83.2% → 84.6%
- On the 176 new items alone:
  - gpt-oss and 3B are within about 1–4 points of their v1 scores;
  - 7B is 4.7 points lower: 60.2% vs 64.9%, with overlapping CIs.
- CIs are about 1.25× narrower than at n = 279; for gpt-oss the half-width falls from ±4.4 to
  ±3.3 points.
- The frontier advantage is concentrated on hard and extra questions. gpt-oss holds 87.5% on
  hard, where 7B is at 36.1%. The Qwen models' dominant failure is the wrong row set (wrong
  joins, a missing or extra filter, or TEXT vs number id joins) and the wrong number of output
  columns.

### Sensitivity (does not replace the primary metric)
| | Qwen-3B | Qwen-7B | gpt-oss-120b |
|---|---:|---:|---:|
| numeric TEXT counted as equal to a number (`'9'` == `9`) | 47.7% | 63.1% | 85.0% (+2) |
| without 7 post-hoc questionable items (n = 448) | 48.4% | 64.1% | 85.9% |

**Post-hoc questionable items.** I found these 7 while reviewing gpt-oss failures on the new
items; this was not a systematic re-audit. The dataset was kept frozen rather than edited after
seeing model outputs. All three models get every one of them wrong, so excluding them does not
change the ranking.
- **0035, 0367, 0369, 0884:** "for each X" questions where the inner-join SQL omits zero counts.
- **0419:** "visited some museums more than once" has two readings.
- **0537:** "most students" counts enrolments; counting distinct students gives a 5–5 tie.
- **0778:** countries are taken only from `countrylanguage`.

## 6. Limitations
- **NL/SQL audit:** an LLM (Claude) audited the 193 new items, with data probes; a human did not.
  The 279 v1 items keep their v1 curation, which had stronger gold-MQL plus test-suite evidence.
  The 176 new items rely on data parity only. Their reference answers are SQL results on the dev
  DB, without test-suite cross-checks.
- **Strict comparator:** it keeps v1 semantics so the two benchmarks stay comparable. Numbers
  stored as TEXT must be returned as strings, as the schema in the prompt states. The sensitivity
  row above shows this changes at most 2 verdicts.
- **gpt-oss nondeterminism:** gpt-oss-120b on Groq is not bit-reproducible, even at temperature 0.
  Its saved raw outputs are the record of the run; re-scoring them is deterministic.
- **Contamination:** Spider dev is public, so contamination cannot be ruled out (as in v1).
- **Retrospective splits:** the database-disjoint splits in `data/splits/` cover the 279 set only
  and were not rebuilt.

## 7. Reproduce
```powershell
# data (read-only checks; Atlas data was imported for v1 and is not modified)
python scripts/spider1/build_pool.py --out-dir $env:TEMP\pool_check   # must equal data/interim/spider1/pool.jsonl
python scripts/spider1/parity_check.py
python scripts/spider1/build_expanded.py
# predictions: reuse v1 where eligible, then generate the rest (resumable; waits out Groq daily quota)
python scripts/spider1/seed_expanded.py
python scripts/spider1/run_until_complete.py --config configs/evaluation_spider1_expanded.yaml --provider qwen          --label qwen2.5-coder-3b
python scripts/spider1/run_until_complete.py --config configs/evaluation_spider1_expanded.yaml --provider qwen7b        --label qwen2.5-coder-7b
python scripts/spider1/run_until_complete.py --config configs/evaluation_spider1_expanded.yaml --provider frontier_groq --label gpt-oss-120b
# verification + report (no model calls)
python scripts/spider1/rescore_expanded.py
python scripts/spider1/report_expanded.py
python -m pytest -q
```
Run the two Qwen models one after the other, not together. The 7B GGUF needs about 5 GB of RAM,
and running it alongside other memory-heavy work got the run killed twice; it resumed with no
lost or duplicated items.
