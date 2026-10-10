# Conversion prototypes (controlled run, 6 examples + the `local041` pilot)

Goal: test the pilot's conversion rules on different query patterns **before** automating anything. Each prototype was
done by hand: inspect → relational semantics → MongoDB mapping → minimal import → MQL → execute → score with the
**official** Spider comparator (`compare_pandas_table` extracted unchanged from `evaluate.py`; pass = any one gold CSV).
Nothing was auto-generated; there is no converter. Not run: Qwen or any other LLM. Not touched: raw Spider files,
the 32 `needs_mentor` ids, other MongoDB databases.

Code: `scripts/prototypes/harness.py` (plumbing only: read-only SQLite, column-list import, comparator, artifact writer) and
one `proto_<id>.py` per example (mapping + MQL are written out by hand in each). Artifacts:
`data/interim/pilot/<id>.json`. Data lives in `codegen_capstone_pilot`, collections named `<id>__<table>`.

## 1. Results

| Id | Pattern | Tables→collections (docs) | Gold SQL reproduces gold CSV? | MQL vs gold |
|---|---|---|---|---|
| `local041` (pilot) | filter + ratio | 1 (690,626) | n/a (no SQL) | **pass** (gold A) |
| `local163` | single-table aggregate, per-group extremum with ties | 1 (8) | yes | **pass** |
| `local038` | 6-table inner join, filters, group, top-1 | 6 (7,684) | yes | **pass** (gold A; gold B has a different shape and does not match) |
| `local022` | CTEs, composite-key join, GROUP BY/HAVING, derived key, DISTINCT | 5 (279,492) | yes | **pass** (2.4 s) |
| `local039` | 8-table chain, text timestamps→hours, case-insensitive LIKE | 8 (24,443) | yes (gold B only) | **pass** both golds (fractional variant); truncated variant passes gold B only |
| `local197` | window function (LAG), top-N, max, rounding | 2 (16,648) | yes | **pass** all 3 golds |
| `local075` | multi-table report, CASE-sums, EXISTS / NOT EXISTS, 3-CTE join | 2 (32,747) | yes | **pass** both golds |

6/6 new prototypes validated, 0 failures. Storage cost of everything in the pilot database now: 84.0 MB data, 24.2 MB storage,
**32.4 MB indexes** (indexes are not negligible next to compressed data; quota still unverified).

How strong is a "pass"? Weaker than it looks, because the official comparator ignores names and order and checks
only `condition_cols`. Examples: `local197` is scored on the difference value only (77.83); the customer and month are
not checked, so I confirmed them by eye (customer 148 ELEANOR HUNT, 2005-07 match all gold files). `local075` is scored
on the four count columns, not on page ids. Keep full results in the artifacts and spot-check, as done here.

## 2. Per-example notes

### local163: faculty closest to their rank's average salary
- Semantics: `AVG(FacSalary)` per `FacRank`, `ABS` difference, `MIN` per rank, keep **all** rows equal to the min (ties kept).
- Mapping: `university_faculty` → one collection, 4 columns, all 8 rows.
- MQL: `$setWindowFields` (partition by rank, `$avg`) → `$addFields` diff → second `$setWindowFields` (`$min`) → `$match $expr $eq`.
- Result: 4 rows including the two tied ASSC rows. Float equality is safe only because salaries are integers.

### local038: actor with the most G/PG English children's films, 2000-2010, ≤120 min
- Semantics: inner join of 6 tables, count films per actor, `ORDER BY COUNT DESC LIMIT 1`.
- Pre-check recorded: top count 3 vs runner-up 2, so `LIMIT 1` is not tie-dependent.
- Pitfall found and verified: `film.release_year` is TEXT; `BETWEEN 2000 AND 2010` is therefore a text comparison in SQLite.
  In MongoDB numeric bounds matched **0 of 1000** films and string bounds **1000 of 1000**.
- Join order: started from `film` (selective filters), `$lookup`+`$unwind` for each join, indexed `film_actor.film_id`.

### local022: players with ≥100 runs in a match while on the losing team
- Semantics: join `ball_by_ball` ⨝ `batsman_scored` on **four** columns, sum per (striker, match), `HAVING ≥ 100`,
  loser = `team_1` unless it equals the winner, join to `player_match` on (player, match, loser), `DISTINCT` names.
- Pre-checks: both 4-column keys are unique, no NULL winners (so the CASE is total).
- `$lookup` takes one key, so the composite join is built as `$unionWith` + `$group` (push each side's values per key) +
  `$unwind` both arrays. This is exact for non-unique keys too (per-key cross product).
- Imported only 5 of 8 tables and only the needed columns (e.g. 5 of 10 `ball_by_ball` columns).

### local039: category with the most rental hours in cities starting with "A" or containing "-"
- Semantics: rental→inventory→film→film_category→category and rental→customer→address→city, LIKE filter, per-rental hours from
  text timestamps, sum per category.
- **Ambiguity kept visible (not silently resolved):** the gold SQL truncates each rental's hours to INTEGER (`CAST(... AS INTEGER)`)
  and its result (16,790) equals neither number in gold A. Gold A (16,858.166667) is the *fractional* sum; gold B is just `Sports`.
  Primary MQL = fractional (natural reading; passes A and B); the truncated variant is also stored and passes B only. Both pick `Sports`.
- LIKE: SQLite's LIKE is case-insensitive → `$regex "^a"` with `i`; the script asserts the selected city set is identical in
  SQLite and MongoDB.
- NULL `return_date` (183 rentals) → `$dateDiff` gives null → `$sum` ignores it, same as SQL.

### local197: largest month-over-month payment change among the top-10 payers
- Semantics: top 10 customers by total, per-customer monthly sums, difference to the previous month, global max, rounded to 2 decimals.
- **Ambiguity kept visible:** the gold SQL uses `LAG(...) OVER (PARTITION BY customer_id)` with **no ORDER BY** and groups
  by `strftime('%m')` only, so which row is "previous" depends on scan order and could pair non-adjacent months (data runs
  2005-05…2005-08 and 2006-02). The MQL sorts by month and compares only calendar-adjacent months, following the question.
  A calendar-adjacent SQLite reference gives the same winner (148, 2005-07, 77.83; runner-up 76.85; no tie at the max or at rank 10).
- `strftime('%Y-%m', ts)` on ISO text is the first 7 characters; pre-check confirmed every `payment_date` has the ISO prefix.
- `amount` is stored as REAL (16,025 rows) and INTEGER (24); imported as stored.

### local075: product views / cart adds / abandoned / purchased
- Semantics: product pages only; `SUM(CASE ...)` counters; purchases = cart adds in visits that contain an `event_type = 3`
  row (`EXISTS`), abandoned = cart adds in visits that do not (`NOT EXISTS`); three CTEs inner-joined.
- MQL: `$setWindowFields` partitioned by `visit_id` with `$max` over the whole partition replaces the correlated
  EXISTS; one `$group` with four `$cond` sums replaces the three CTEs. Pre-check: the SQL's 3-way inner join keeps all 9 product pages,
  so a single `$group` is equivalent.

## 3. Conversion patterns

Validated (rules proven on real data above):

| SQL pattern | MQL realisation | Proven by |
|---|---|---|
| Filter, ratio, `COUNT(*)` vs `COUNT(col)` | `$match` + `$group`/`$cond` | pilot |
| Inner join (N tables) | `$lookup` + `$unwind`; start from the selective/fact collection; index foreign fields | 038, 039 |
| Multi-column join | `$unionWith` + `$group` + `$unwind` (or `$lookup` let/pipeline on already-reduced rows) | 022 |
| `GROUP BY` / `HAVING` / `DISTINCT` | `$group` / `$match` after it / `$group` on the value | 022 |
| CTE chain | consecutive stages | 022, 075 |
| Per-group extremum with ties | `$setWindowFields` (partition aggregate) + `$match $expr` | 163 |
| `LAG` | `$setWindowFields` + `$shift` with explicit `sortBy` | 197 |
| `EXISTS` / `NOT EXISTS` on same table | per-key flag via `$setWindowFields` `$max` | 075 |
| `SUM(CASE ...)` | `$sum` + `$cond` | 075 |
| text timestamps → duration | `$dateFromString` (explicit format) + `$dateDiff` | 039 |
| `strftime('%Y-%m')` on ISO text | `$substrCP` prefix | 197 |
| Case-insensitive `LIKE` | `$regex` + `i` | 039 |
| Top-N then detail | `$group` → `$sort` → `$limit` → `$lookup` with sub-pipeline | 197 |
| `IS NOT NULL`, `NOT IN` | `{$ne: null}`, `$nin` | 075 |

Need special handling (worked, but only with a human decision or a data check each time):
- **Type affinity** (038): comparisons against TEXT-stored numbers.
- **Ties / `LIMIT`** (038, 163): requires a data pre-check; a tie makes the item ambiguous.
- **Unordered window** (197) and **truncation vs gold CSV** (039): the gold SQL itself is questionable.
- **Rounding** (197): `$round` is half-to-even (see `schema_notes.md` §6).
- **Dropping pass-through tables** (039 keeps `film`): only safe if referential integrity is verified.

## 4. Rules for the eventual converter

1. Import only the listed columns, all rows, values exactly as stored (int/real/text, NULL → `null`); default `_id`; collections `<id>__<table>` in the pilot database; drop only the same prefix on re-run.
2. Before writing MQL, run the reference SQL on SQLite and require it to reproduce a gold CSV (all 6 did).
3. Record per-example pre-checks in the artifact: key uniqueness for any join key, tie check for `LIMIT`/top-1, NULL counts for tested columns, storage class of compared columns.
4. Compare against TEXT-stored values with strings; cast explicitly when the SQL relied on affinity.
5. Use `$sortArray`/window stages with an explicit order; never reproduce an unordered SQL window.
6. Keep every variant that arises from ambiguity (e.g. truncated vs fractional) and report which gold each passes; do not pick silently.
7. Score only with the official comparator and keep the full result in the artifact, because the comparator can pass on a column subset.
8. Create indexes only on foreign fields used by `$lookup` and record them.

## 5. What prevents safe automation across the other examples

1. **111 of 135 have no gold SQL.** The reference SQL has to be authored per question; that is interpretation, not translation.
2. **Gold SQL is not trustworthy as a specification** (039 truncates, 197 has an unordered `LAG`, and earlier work found value-level mismatches in `local003/066/309`). A mechanical SQL→MQL translation would reproduce the defect.
3. **Multiple golds with different shapes/semantics**: the converter must decide which one it targets; the choice changes the query (039 A vs B).
4. **Data-dependent safety** (ties, uniqueness, NULL denominators, TEXT-stored numbers) can only be established by querying the data, so the converter needs a pre-check phase whose outcome can stop the example.
5. **Weak evaluator**: ignore_order + column subset means a pass is not proof of full correctness (197, 075); a human or secondary check is required.
6. **Tier limits**: no `$function`, `$median` is approximate, `$round` differs from SQLite; the pilot database already holds 84 MB data + 32 MB indexes and the quota is unconfirmed, while the full set needs ~10M rows.
7. **Views and unusual names** (e.g. `match_view`, `loss_rate_%`) are not covered by these prototypes and need their own rules.
8. These six all have small-to-moderate data. Not yet tested: `$lookup` over 1M-row collections, `$setWindowFields` over ~600k rows, gaps-and-islands, densified calendars, FIFO allocation.

## 6. Reproduce

`python scripts/prototypes/proto_<id>.py` (needs `.env`; re-imports its own `<id>__*` collections, then validates and rewrites the artifact).
