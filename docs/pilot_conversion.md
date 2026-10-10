# Pilot conversion: Spider2-Lite `local041` → MongoDB/MQL

One example only. Script: `scripts/pilot_local041.py`. Artifact: `data/interim/pilot/local041.json`.
Raw Spider files were only read. Mongo writes went only to the dedicated database `codegen_capstone_pilot` (it did not exist before; other databases untouched). The URI is read from `.env` and is not recorded anywhere.

## 1. Source example
| | |
|---|---|
| instance_id | `local041` |
| db | `modern_data` (SQLite; only table `trees` is used) |
| question | "What percentage of trees in the Bronx have a health status of Good?" |
| external knowledge | none |
| gold SQL | not shipped. Reference SQL below is **ours**, not Spider's |
| gold CSVs | `local041_a.csv`: `PCT_GOOD = 78.1473`; `local041_b.csv`: `pct_healthy = 82.649376` |
| eval config | `condition_cols=[]` (all gold columns), `ignore_order=true` |

Reference SQL (ours, run read-only on SQLite → 78.14726564770152):
```sql
SELECT 100.0 * SUM(CASE WHEN health = 'Good' THEN 1 ELSE 0 END) / COUNT(*) AS PCT_GOOD
FROM trees WHERE boroname = 'Bronx';
```

Why chosen: local*, no external knowledge, single table, one filter + one ratio, no JOIN/subquery, one-cell gold. It still contains two real conversion traps (NULLs and duplicate keys), which makes it a useful first test.

## 2. Mapping
`trees` has 16 columns and 690,626 rows. The question needs only `boroname` (filter) and `health` (tested value).

| Relational | MongoDB (`codegen_capstone_pilot`) |
|---|---|
| table `trees` | collection `local041_trees` |
| one row | one document, 690,626 total, **duplicates preserved** |
| `boroname`, `health` | same-named fields; `health` SQL NULL → BSON `null` (31,935 docs) |
| `tree_id` | kept for traceability only (**not** `_id`, not unique) |
| other 13 columns | not imported (not needed) |
| `_id` | default ObjectId |

Example document: `{tree_id, boroname: "Bronx", health: "Good"}` (plus `_id`).
All five boroughs were imported (not just the Bronx), so the `$match` is a real filter and the data is a faithful slice of the table.

## 3. MQL
```js
db.local041_trees.aggregate([
  { $match: { boroname: "Bronx" } },
  { $group: { _id: null,
              good:  { $sum: { $cond: [{ $eq: ["$health", "Good"] }, 1, 0] } },
              total: { $sum: 1 } } },
  { $project: { _id: 0, PCT_GOOD: { $multiply: [100, { $divide: ["$good", "$total"] }] } } }
])
```

## 4. Result and validation
- MongoDB result: `[{PCT_GOOD: 78.14726564770152}]` (Bronx: 67,234 Good of 86,035 docs). Same as the SQLite reference SQL.
- Validation used the **official `compare_pandas_table`**, its source extracted from `evaluation_suite/evaluate.py` and executed unchanged. Semantics: gold columns must each match some prediction column, names/order ignored, abs tolerance 0.01, NaN→0, `ignore_order`; passes if **any** gold file matches.
- `local041_a.csv`: **match** (|78.14727 − 78.1473| ≈ 3e-5). `local041_b.csv`: no match. **Overall: PASS** (any-one-gold rule).

## 5. Assumptions and problems found
1. **Denominator.** The gold `_a` divides by *all* Bronx rows, so NULL `health` counts as "not Good". Dividing only by non-NULL `health` gives 82.6377%.
2. **Gold `_b` is not reproducible.** 82.649376 differs from the non-NULL-denominator value (82.63766) by 0.0117, which exceeds the 0.01 tolerance, so even the "right" alternative interpretation fails against `_b`. `_b` appears to come from a slightly different data version; it is irrelevant here because `_a` suffices, but multi-gold files are not guaranteed to be reproducible from the shipped SQLite.
3. **Duplicate keys.** `tree_id` and `idx` each have only 683,788 distinct values across 690,626 rows. The gold counts rows, so we must not dedupe or key by `tree_id`.
4. The reference SQL is ours (no gold SQL exists). Correctness is judged only by the result vs gold CSV, never by comparing query text.
5. The MongoDB result is a float computed in Mongo; Python-side the artifact stores the full double.

## 6. Rules learned for the remaining examples
1. Map table→collection and row→document 1:1; **never dedupe, and never use a SQL "key" as `_id` without checking uniqueness**: SQL result semantics are over rows.
2. Keep SQL NULLs as explicit `null` and translate `COUNT(*)` vs `COUNT(col)` deliberately; the denominator choice decides the answer.
3. Import only the columns the question needs, but all rows of the table, so filters stay real.
4. Always validate by running the official comparator on the result, accepting any one of the gold CSVs; do not assume every gold alternative is reproducible.
5. Where no gold SQL exists, write and run a reference SQL on SQLite first and confirm it matches a gold CSV before writing the MQL.
