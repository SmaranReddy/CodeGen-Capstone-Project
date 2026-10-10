# Schema Notes (Week 1)

> Status: Week 1 specification. Rules marked **[verified]** were confirmed on the `local041` pilot or by the Atlas
> probe; **[from SQLite semantics]** are documented SQLite/MongoDB behaviour not yet exercised here; **[proposed]** need
> approval. Companion docs: `dataset_task_definition.md` (validation), `query_types.md` (categories, idioms).

Purpose: define how a Spider 2.0-Lite SQLite database becomes MongoDB collections **without changing what a query over
it returns**. The guiding fact: SQL results are computed over a *bag of rows*, so the Mongo copy must be a bag of
documents that is row-for-row equal for every column a query touches.

## 1. Table → collection, row → document

1. One table → one collection; one row → one document; **no embedding, no denormalisation**. Foreign keys stay
   as plain fields and are joined with `$lookup` (or `$group` on a shared key). Embedding would change the multiplicity
   of rows (a one-to-many embed hides parent rows that have no children and merges duplicate parents), and the Spider
   questions are written against the flat schema. **[proposed, supported by pilot]**
2. Collection name: `<instance_or_db>_<table>` inside a dedicated database (pilot: `codegen_capstone_pilot.local041_trees`).
   Never write into pre-existing databases (`library_management`, `sample_mflix`, ...).
3. Field name = column name, unchanged (case preserved). MongoDB accepts the odd names here (`loss_rate_%`,
   `qty_sold(kg)`, `unit_selling_px_rmb/kg`, `sale/return`, `discount(%)`, `whsle_px_rmb-kg`): none contains `.` or starts
   with `$`. In pipelines use `{$getField: "loss_rate_%"}` rather than `"$loss_rate_%"`. 2 of 135 examples are affected
   (`local284`, `local285`). The `ERD` table in `imdb_movies` has columns like `Unnamed: 2`: not used by any question.
4. **Views.** Spider's DDL omits SQLite views, and some examples probably depend on them (confirmed in table lists: EU_soccer
   `match_view` → `local219`; stacking `stack_ok` → `local262`/`local263`; oracle_sql `monthly_orders` → `local275`/`local279`).
   Rule **[proposed]**: run `SELECT * FROM <view>` on SQLite and import the result as a collection named `view_<name>`,
   flagged `materialized_view` in the artifact; do not re-implement view logic in MQL unless the question needs it.
   `oracle_sql.emp_hire_periods_with_name` is a **broken view** (references a missing column); no question uses it.

## 2. Types **[from SQLite semantics; `trees` pilot confirms ints/text/null]**

| SQLite | BSON | Note |
|---|---|---|
| INTEGER | int32/int64 (driver chooses) | |
| REAL | double | same IEEE-754 as SQLite |
| TEXT | string | **keep as stored**, including all date/time columns |
| NULL | `null` (field present) | see §3 |
| BLOB | binary | none used by the questions |

SQLite is dynamically typed: a column declared `INTEGER` may hold text. Import each value **with the type SQLite stores
it as** (check `typeof(col)` per column; flag mixed-type columns). Do not coerce on import.
**Dates are stored as TEXT** (e.g. `order_purchase_timestamp`, `birth_date`, `week_date`). Default **[proposed]**: keep the
string and parse inside the pipeline (`$dateFromString`) when arithmetic is needed. Converting to BSON `Date` on import
would change comparison semantics and force a timezone choice; if ever done, treat everything as UTC and record it.
Non-ISO or mixed date formats must be inspected per column before conversion (not yet surveyed) and parsed with an explicit `format`.

## 3. NULL handling

1. Import SQL NULL as an **explicit `null`**, never omit the field. **[verified: 31,935 NULL `health` values in the pilot]**
2. SQL three-valued logic differs from MQL. `WHERE col <> 'x'` excludes NULL rows; `{col: {$ne: 'x'}}` **includes** `null`
   and missing. Add `{col: {$ne: null}}` (or `$expr` with `$ne` + null check). Same for `NOT IN`, `NOT LIKE`.
3. `COUNT(*)` → `{$sum: 1}`. `COUNT(col)` → count where `col` is not `null`. The pilot answer hinges on this: `COUNT(*)`
   gave 78.1473 (gold A); `COUNT(health)` gave 82.6377.
4. `SUM`/`AVG`/`MIN`/`MAX` ignore NULL in both SQL and `$sum/$avg/$min/$max`. **But** an all-NULL group gives SQL `NULL` and
   `$sum` gives `0`: use `$cond`/`$ifNull` when that distinction reaches the output.
5. Sort order: NULL sorts lowest in ascending order in both systems **[from semantics]**.
6. The evaluator maps null/NaN to `0`, so SQL `NULL` and `0` are interchangeable *for scoring*; still reproduce the
   NULL faithfully.

## 4. Duplicate rows

1. **Never de-duplicate on import.** Pilot: `trees` has 690,626 rows but only 683,788 distinct `tree_id`; the gold counts rows.
   **[verified]**
2. SQL `DISTINCT` / `COUNT(DISTINCT x)` / `UNION` → explicit `$group` (preferably a two-stage `$group` rather than
   `$addToSet`, which is limited by the 16 MB document size and 100 MB stage memory). `UNION ALL` → `$unionWith`
   without dedup; `UNION` → `$unionWith` + `$group`.
3. Joins multiply rows exactly as in SQL: `$lookup` + `$unwind` reproduces an inner join including fan-out. Use
   `preserveNullAndEmptyArrays: true` for LEFT JOIN.
4. Spider tables **declare no primary keys** in the DDL we have, so "key" columns (`tree_id`, `idx`) are *not*
   guaranteed unique. Verify uniqueness (`COUNT(*) = COUNT(DISTINCT col)`) before treating anything as an identifier.

## 5. `_id` rules **[proposed, supported by pilot]**

1. Use the default `ObjectId`. **Do not** reuse a SQL column as `_id` unless uniqueness was verified (pilot proved
   `tree_id` and `idx` are not unique; using them would silently drop or reject rows).
2. Do not add synthetic fields by default ("do not invent unnecessary fields").
3. **Exception to decide with the mentor:** SQLite's implicit order (`rowid`) breaks ties in `ORDER BY ... LIMIT`,
   `ROW_NUMBER()` without a unique order key, and "first row per group" queries. Where a question depends on such ties
   (e.g. `local035`, `local360`, `local019`), add a single `_rid` field = SQLite `rowid` and **document it**. Off by default.

## 6. SQL semantics that do not carry over **[from SQLite semantics unless noted]**

| SQLite | MQL | Rule |
|---|---|---|
| `LIKE` is **case-insensitive** for ASCII | `$regex` case-sensitive | add `options: "i"`; escape `%`/`_` → `.*` / `.` (`local039`) |
| `a/b` on integers truncates | `$divide` always double **[verified: 5/2 → 2.5]** | `$trunc` when the SQL divides integers; use `100.0*` forms as the SQL does |
| `round(x, n)` half away from zero | `$round` half-to-even **[verified: 0.5→0, 2.5→2, 0.125→0.12]** | implement `floor(x*10^n + 0.5)/10^n` for positive values; matters at the 0.01 tolerance edge |
| `strftime`, `julianday`, `date()` | `$dateFromString`, `$dateToString`, `$dateDiff` | parse text first; document the format string |
| `substr`, `instr`, `replace`, `group_concat` | `$substrCP`, `$indexOfCP`, `$replaceAll`, `$reduce`+`$concat` | `$split` replaces recursive-CTE string splitting (`local065`, `local066`, `local073`) |
| String comparison BINARY | default Mongo comparison is binary | same; do not add a collation |
| Bare non-aggregated column in `GROUP BY` | none | arbitrary row in SQLite; ambiguous, flag for mentor |
| Column affinity (`'5' = 5` true for INTEGER columns) | no coercion | cast explicitly with `$toInt`/`$toDouble` |
| `AVG` of ints | double | same |
| Median / stddev (no builtin in SQLite) | `$median` is approximate-only **[verified]**; stddev via `$stdDevPop/$stdDevSamp` | exact median: `$sortArray` + index; stddev population vs sample must follow the gold (`local284`) |
| Window functions | `$setWindowFields` **[verified]** | `LAG/LEAD` → `$shift`, `ROW_NUMBER` → `$documentNumber`, `RANK` → `$rank`, `NTILE` → no builtin, derive from rank/count |
| Recursive CTE | `$graphLookup` **[verified present]** or `$reduce` over an array | `$graphLookup` cannot carry per-path arithmetic (packaging quantities) |
| User-defined functions | `$function` **[verified: blocked on this tier]** | do not use JavaScript |

## 7. What must be imported to preserve query semantics

For each example, import:
1. **All rows** of every table the query reads (never sample, never pre-filter, even if the question names one value;
   the pilot imports all five boroughs so the `$match` is real).
2. **Every column the query touches**: join keys, filter columns, grouped/sorted columns, columns in expressions,
   tie-break columns, plus identifier columns needed to render the answer (names). Drop the rest.
3. The **base tables** behind any view the example uses (or the materialised view, §1.4).
4. For a question with no gold SQL, the column set is taken from the **reference SQL** that reproduced a gold CSV
   (`dataset_task_definition.md` §6), not from guesses.
5. Rows are loaded **bit-for-bit** (values and stored types); no trimming, case-folding, or date conversion.

Examples that need the largest imports (rows): `olist_geolocation` 1,000,163 (`local035`), `hardware_fact_sales_monthly`
971,631 (`local058`/`local059`), `sales` 918,843 (complex_oracle family), `veg_txn_df` 878,503 (`local285`), `trees` 690,626
(`local040`, pilot), `lap_positions` 613,112 (f1 overtakes), WWE `Matches` 540,800 (`local019`).

## 8. Schema description format for the model input

```json
{
  "db_id": "modern_data",
  "collections": [
    {"name": "trees", "fields": [
        {"name": "boroname", "type": "string", "nullable": false},
        {"name": "health",   "type": "string", "nullable": true},
        {"name": "tree_id",  "type": "int",    "nullable": false, "unique": false}]}],
  "relationships": [{"from": "orders.customer_id", "to": "customers.customer_id", "kind": "foreign_key (undeclared)"}],
  "notes": ["dates are strings 'YYYY-MM-DD HH:MM:SS'", "health may be null"]
}
```
Include **only imported fields**, nullability observed from data, `unique:false` where verified non-unique, and plain-text
notes for date formats. Decision still open: whether to include 2-3 sample documents (helps date formats; costs tokens).
Foreign keys are *not declared* in Spider's DDL, so `relationships` must be derived (shared column names) and checked.

## 9. Proposed hosting lifecycle (needs approval; quota unverified)

The pilot database is `codegen_capstone_pilot`. For the rest: one dedicated database per Spider db
(`codegen_capstone_<db>`), import → validate → record artifact → drop **our own** collections, because importing all
~10.3 M needed rows at once may exceed a 512 MB free-tier cap (not confirmed). Create indexes only on `$lookup` foreign
fields of large collections (indexes also use quota). Never touch other databases.
