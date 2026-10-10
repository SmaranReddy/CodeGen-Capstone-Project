# Spider 1.0 → gold MQL authoring protocol

This protocol applies to every gold MQL entry in `data/interim/spider1/gold_mql/<db_id>.json`.
The gold MQL entries were written by Claude (an LLM) using the Spider gold SQL and question
as the reference. They were accepted only after passing execution checks.

## Inputs (read-only)
- `data/interim/spider1/sample.jsonl`: the frozen seed-42 sample of 300 items. Each item has
  `id`, `db_id`, `question`, `gold_sql`, `gold_rows`, `order_by`, `limit`, and `tie_blocks`.
- Atlas database `codegen_spider1` holds one collection per table, named `<db_id>__<table>`.
  Every column is kept, and each value keeps its SQLite storage class. Many Spider columns
  store numbers as TEXT. Check the observed types.
- `python scripts/spider1/show_items.py <db_id>` prints the schema with observed BSON types,
  foreign keys, sample documents, and the items.

## Output format
`data/interim/spider1/gold_mql/<db_id>.json`:
```json
{"spider1_dev_0000": {"collection": "concert_singer__singer",
                      "pipeline": [{"$count": "count"}],
                      "notes": "optional: non-obvious decisions"}}
```
- Each entry is one `aggregate` pipeline run on `collection`. Every referenced collection
  (`$lookup.from`, `$unionWith`) must be a `<same db_id>__*` collection.
- Each output document is one result row. The columns are its fields. An ObjectId `_id`, or an `_id` that is null in every row, is
  ignored. Any other `_id` counts as a column, so project it away (`"_id": 0`) unless it is
  an answer column. Nested objects are flattened.
- The number of output columns must equal the gold SQL's. Column names and column order do
  not matter.

## Acceptance (both are required)
1. `python scripts/spider1/validate_gold.py --db <db_id> --show` must print PASS. This is an
   execution match against the gold SQL result on the dev DB. It is order-sensitive only
   when the SQL has a top-level ORDER BY, and rows tied on the ORDER BY key may come in any
   order.
2. `python scripts/spider1/testsuite_check.py --db <db_id>` must print PASS or INCONCLUSIVE,
   never FAIL. It runs the same MQL and gold SQL on 3 official test-suite database variants
   (`codegen_spider1_ts1..3`), which catches coincidental matches. wta_1 has no variants,
   so it reports `not_checked`.

## Semantics: reproduce the gold SQL exactly as SQLite evaluates it
- **Faithfulness.** The pipeline must compute the same thing as the SQL in general, not
  just return the same output on this data.
  - Never hard-code result values or IDs that are not literals in the SQL or question.
  - Never use `$documents`, `$where`, `$function`, `$accumulator`, `$out`, or `$merge`.
- **Type affinity.** SQLite compares TEXT columns with numeric literals by converting them.
  In MQL, match the stored type (for example `"2014"`), or convert with
  `$toString`/`$toInt`/`$toDouble`.
  - Joins across columns whose stored types differ need `$lookup` with `let`/`pipeline`
    and a conversion.
- **Division.** `int / int` in SQLite is integer division (truncation toward zero). Use
  `$trunc` of `$divide` where the SQL does that.
- **Aggregates.** `avg` returns a float. `count(col)` and `avg/sum/min/max` ignore NULLs.
  `count(*)` counts rows. `count(DISTINCT x)` ignores NULLs.
- **LIKE.** LIKE is case-insensitive for ASCII. Translate it to an anchored, escaped
  `$regex` with `"$options": "i"`: `%` becomes `.*` and `_` becomes `.`.
- **Bare columns.** `SELECT name, max(x)` returns `name` from the row with the max.
  `SELECT a, count(*) ... GROUP BY b` returns `a` from some row of the group, typically
  the last one scanned. Reproduce this with `$sort` plus `$first`/`$last`, or with
  `$top`/`$bottom`.
- **Joins.** JOIN semantics are inner. `$unwind` drops non-matching rows, which matches
  inner-join behavior. Use `preserveNullAndEmptyArrays` only for LEFT JOIN.
- **Set operations.** INTERSECT, EXCEPT, and UNION are set operations, so they return
  DISTINCT rows. UNION ALL keeps duplicates. `NOT IN (subquery)` returns no rows if the
  subquery yields a NULL.
- **ORDER BY.** Ties may come in any order, but non-tied order must match. SQLite sorts
  NULLs first in ASC order. Text sorts by binary comparison, and Mongo's default string
  sort is also binary.
- **DISTINCT** applies to whole output rows.
- **Correlated or scalar subqueries.** Use `$lookup` pipelines, `$setWindowFields`, or
  `$facet`.

## When the gold is wrong or ambiguous
- **Still author MQL that reproduces the gold SQL.** Do this even when the gold SQL does not
  match the question, for example when the SQL omits a filter the question states.
- **Record the problem.** Add the item to `data/interim/spider1/gold_mql/<db_id>.issues.json`
  as `{"<id>": {"kind": "annotation_mismatch" | "ambiguous" | "cannot_convert", "detail": "..."}}`.
  The curator decides whether the item enters the validated dataset.
- **Unconvertible items.** If an item cannot be faithfully converted after a serious effort,
  leave it out of the gold file and record `cannot_convert` with the reason.

## Constraints
- Write only your assigned `gold_mql/<db_id>.json` and `.issues.json` files.
- Never modify scripts, `data/raw`, `sample.jsonl`, or any Atlas data. Only aggregate
  (read) queries are allowed.
- Do not read `outputs/`.
