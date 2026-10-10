# Dataset / Task Definition (Week 1)

> Status: Week 1 specification, informed by the validated `local041` pilot (`docs/pilot_conversion.md`) and a
> pattern survey of all 135 local Spider 2.0-Lite examples (`data/interim/spider2_local_query_patterns.jsonl`,
> built by `scripts/build_query_pattern_table.py`). **No conversion beyond the pilot has been done.**
> Related: `schema_notes.md` (relational → MongoDB mapping), `query_types.md` (categories and SQL → MQL idioms).

## 1. Task

Given a natural-language question and a MongoDB schema, generate an **MQL read query** (an aggregation
pipeline, or `find` where trivial) whose **executed result** matches the Spider gold result. Write operations
(insert/update/delete) are **out of scope**: Spider is read-only.

## 2. Source and what is actually available

- Spider 2.0-Lite, `data/raw/spider2/` (never modified). Only the **135 `local*` ids** run offline on SQLite;
  the other 412 of 547 need BigQuery/Snowflake and are excluded.
- **The 135 are the ceiling for this source.** A 200-300 example dataset cannot be built from them one-to-one
  (see §9, question 1).
- Of the 135: **24 have gold SQL, 111 do not**; **91 have more than one gold CSV**; **13 reference an
  external-knowledge document**; **31 touch a single table**.

## 3. Inputs / outputs per example

Inputs: `id`, `question`, `db_id`, MongoDB `schema` (collections, fields, types; see `schema_notes.md`),
optional external-knowledge text. Outputs: `query` (MQL), `query_type` (a `query_types.md` category),
`validation_status` and the executed result.

## 4. Query-pattern survey (all 135)

Classified into `data/interim/spider2_local_query_patterns.jsonl` (one row per id). Fields: tables, constructs,
joins, category, flags, notes, gold shapes. **`basis` is recorded per row:**
- `gold_sql` (24 rows): tables and constructs come from the shipped SQL, then were hand-reviewed.
- `inferred_from_question` (111 rows): tables and constructs are **my reading of the question and the schema.
  They are estimates, not facts**; the true SQL may differ. Re-check each when it is actually converted.

Counts (computed by the script, not hand-tallied):

| Category | Meaning | Count | of which gold-SQL basis |
|---|---|---|---|
| 1 `direct` | `$match/$group/$project/$sort/$limit`, at most one simple `$lookup` | **2** | 0 |
| 2 `special_transform` | feasible and deterministic, needs lookup chains, pivots, windows, parsing, per-group top-N, etc. | **75** | 20 |
| 3 `difficult_ambiguous` | underspecified, procedural, tie-dependent, external-doc, or heavily multi-shaped golds | **55** | 1 |
| 4 `not_faithful` | cannot be certified faithful: gold CSV not reproducible from the shipped gold SQL at value level | **3** | 3 |

The two category-1 ids are `local041` (pilot) and `local133`. Category 1 is tiny because Spider 2.0 is built
around multi-table analytics, not single-table lookups; expect most items to need pipelines with `$lookup`, `$group`,
`$setWindowFields` or `$reduce`.

Orthogonal flags (a flag is not a category): `multi_gold_shapes` 47 (computed from the CSVs: golds differ in row/column
count), `needs_mentor` 32, `big_table` 18 (a table >100k rows), `ext_knowledge` 13, `gold_unreliable` 3,
`gold_sql_disagrees` 3, `view_dependent` 4 confirmed + `view_possible` 8 (includes those 4; the rest are my guess).

Most frequent constructs (an id can have several; mostly inferred): aggregation 126, GROUP BY 114, join 98, filter 51,
sort 43, date/time 40, window 36, limit 31, CASE 25, subquery 24, per-group top-N 23, CTE 19, set operation 14,
string parsing 13, non-trivial math 11, running/rolling 9, HAVING 9, bucketing 6, calendar densification 6,
gaps-and-islands 5, median 3, recursion 3, FIFO allocation 3, pivot 2, graph/self-join 2.

## 5. Validation rules (what "correct" means)

Correctness is judged **only by the executed result**, never by comparing MQL text to SQL text.

1. Execute the MQL on Atlas and put the rows in a DataFrame. Use the **official** `compare_pandas_table` from
   `evaluation_suite/evaluate.py` (extract its source as `scripts/pilot_local041.py` does; do not reimplement).
2. Semantics (verified in `spider2_lite_inspection.md` §9.1): each required gold column must equal **some**
   prediction column; names and column order are ignored; extra columns are allowed; NaN/null → 0; numbers match with
   absolute tolerance 0.01; strings are exact, case- and whitespace-sensitive; a number never equals a string.
   `ignore_order` is true for **all 135**, so each column is sorted independently and row alignment across columns
   is **not** checked. `condition_cols` (when non-empty) restricts which gold columns are required.
3. **Multiple gold CSVs:** the example passes if **any one** matches (`compare_multi_pandas_table`). A flat
   `condition_cols` is reused for every gold; a list of lists is applied per gold. Record *which* gold matched and
   keep the others in the artifact. **Do not assume every gold is reproducible**: pilot `local041_b` is not (it is
   0.0117 away from the value its own interpretation gives).
4. **Gold shapes can differ** (47 ids): a prediction must be shaped for at least one gold, not for all.
5. Type discipline matters because of rule 2: numbers must come back as numbers and text as text. Check that the
   MQL does not turn a numeric SQLite value into a string, or vice versa.
6. Because `ignore_order` hides row alignment, a pass is weaker evidence than it looks for multi-column results:
   additionally keep the full result in the artifact and spot-check one row end-to-end.

## 6. Examples without gold SQL (111 of 135)

1. Write a **reference SQL** (clearly labelled ours, not Spider's) and run it on the read-only SQLite file.
2. Score the reference SQL's result against the gold CSVs with the same comparator. **It must match at least one gold
   before any MQL is written.** If no candidate SQL reproduces any gold, mark the id `gold_unreproducible`,
   do not convert it, and add it to the mentor list.
3. Then write the MQL and validate it against the gold CSV (§5), not against the reference SQL.
4. Record in the artifact: reference SQL, the gold that matched, and every interpretation choice (denominator, ties,
   rounding, NULLs).

## 7. Examples with gold SQL (24 of 135)

Shipped gold SQL is **advisory, not authoritative**: replaying it showed it does not always produce the gold CSV
(`spider2_lite_inspection.md` §9). Known cases: `local003`, `local066`, `local309` (value-level mismatch → category 4);
`local029`, `local131`, `local210` (shape mismatch; the CSV is still reproducible by a different query → flag
`gold_sql_disagrees`). Always validate against the CSV; if the shipped SQL disagrees, say so in the artifact.

## 8. Environment findings that constrain conversion (Atlas probe, run on the pilot collection)

- MongoDB **8.0.34**. The tier is not directly visible, but `$function` returns *"not allowed in this atlas tier"*
  (code 8000), which is consistent with a free/shared tier. **Assume no server-side JavaScript.**
- Available and working: `$setWindowFields` (`$rank`, running sums), `$densify`, `$sortArray`, `$median` and
  `$percentile` (approximate method only), `$graphLookup`, `$dateFromString`, `$regexMatch` with `i`.
- `$round` is **round-half-even** (`0.5→0`, `2.5→2`, `0.125,2→0.12`); SQLite `round()` rounds half away from zero.
  `$divide` is always floating point (`5/2 = 2.5`); SQLite integer division truncates.
- Size: the used tables total about **10.3 million rows** (estimated from my table lists; 18 tables exceed 100k rows,
  largest `olist_geolocation` at 1,000,163). The pilot's 690,626 three-field docs measured 51 MB data / 16 MB storage.
  Importing *every* needed table at once may approach a 512 MB free-tier quota (**unverified: I could not read the quota**);
  see `schema_notes.md` §9 for the proposed lifecycle. Aggregation memory and `$lookup` on 1M-row collections are untested.
- Not found: `docs/conversion_notes.md` and `docs/validation_notes.md` (referenced from `docs/README.md`) **do not exist
  yet**. The validation and conversion-decision content now lives in §5-§7 here and in `schema_notes.md`/`query_types.md`.

## 9. Mentor clarification needed

1. **Target size.** The task says 200-300 examples, but the local source has 135 ids (and realistically about 77
   categories 1-2). Options: accept ~77-100; add several questions per database (not in Spider); or add the
   non-local ids by re-hosting their data (they need BigQuery/Snowflake exports).
2. **Category 4 / unreliable golds** (`local003`, `local066`, `local309`; also `local029`, `local131`, `local210`):
   drop them, or hand-derive a procedure that reproduces the gold CSV?
3. **Ambiguous questions with several differently-shaped golds** (32 ids flagged `needs_mentor`, listed in the JSONL):
   include with the any-gold rule, pin the interpretation we choose, or exclude? Notable: `local002`, `local007`,
   `local015`, `local020`, `local032`, `local040`, `local067`, `local132`, `local229`, `local286`, `local358`
   (`local358` needs an unstated reference date for ages and its two golds give different counts).
4. **Hosting.** Confirm the Atlas tier and quota, and whether we may use one database per Spider db
   (`codegen_capstone_<db>`) and drop our own collections after validating (see `schema_notes.md` §9).
5. **Views.** Spider DDL omits SQLite views; some examples likely depend on them (EU_soccer `match_view`, stacking
   `stack_ok`, oracle_sql `monthly_orders`). Materialize their data as collections?
6. **Scope of "faithful".** Is a fixed-depth `$lookup` unrolling (valid for this data) acceptable for the nested
   packaging examples (`local269`, `local270`), given MQL has no general recursion with per-path arithmetic?

## 10. Dataset splits

Undecided. If only ~77-100 items survive, hold out by **database** (not by row) to avoid leaking schemas
between train and test; revisit once the target size (question 1) is settled.
