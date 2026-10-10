# Query Types (Week 1)

> Status: Week 1 specification, from the pattern survey of all 135 local Spider 2.0-Lite examples
> (`data/interim/spider2_local_query_patterns.jsonl`, `scripts/build_query_pattern_table.py`).
> **Caveat:** only 24 rows are classified from gold SQL; 111 are classified from the question text and schema
> (`basis: inferred_from_question`) and are estimates.

## 1. Conversion categories (difficulty / faithfulness)

| ID | Category | Rubric | Count |
|---|---|---|---|
| 1 | `direct` | `$match/$group/$project/$sort/$limit`, at most one simple `$lookup` | **2** |
| 2 | `special_transform` | deterministic and feasible, but needs a `$lookup` chain, `$unwind`, pivot, window/bucket stages, date or string parsing, per-group top-N, set logic, exact median | **75** |
| 3 | `difficult_ambiguous` | underspecified or tie/LIMIT-dependent, external-knowledge doc, procedural (gaps-and-islands, retention, rolling over densified calendars, FIFO allocation, graph walks), or golds of many shapes | **55** |
| 4 | `not_faithful` | cannot be certified faithful: gold CSV not reproducible from the shipped gold SQL at value level | **3** |

Gold-SQL basis: category 2 has 20 of 75, category 3 has 1 (`local219`), category 4 has all 3. Category 4 is deliberately
narrow: with `$reduce`, `$setWindowFields` and `$densify` almost any procedure can be *expressed* in MQL. The real limit is
whether we can **verify** faithfulness (no `$function` on this tier; see `dataset_task_definition.md` §8).
Gold-reliability and similar properties are kept as **flags**, not categories.

### Examples by category

- **1 direct (2):** `local041` (pilot: filter + ratio), `local133` (weighted score per style).
- **2 special transform (75), typical:** `local028` (month × year pivot), `local198` (median via `$sortArray`),
  `local054` (5-way `$lookup` chain), `local004` (timestamp text → lifespan weeks), `local201` (anagram signature via sorted chars),
  `local065` (recursive string split → `$split`), `local078` (rank + union of top/bottom 10), `local081` (range-join to thresholds).
- **3 difficult (55), the hardest:**
  - *Gaps-and-islands / consecutive dates:* `local070`, `local071`, `local072`, `local098`, `local355`.
  - *Densified calendars + running/rolling state:* `local074`, `local299`, `local300`, `local169`, `local170`, `local171`.
  - *Sequential allocation (FIFO):* `local272`, `local273`, `local274`.
  - *Recursion with per-path arithmetic:* `local269`, `local270` (fixed-depth unrolling or `$reduce`; `$graphLookup` can't multiply along paths).
  - *Event-sequence classification (F1 overtakes, external doc):* `local344`, `local336`, `local356`.
  - *Dozens of metrics per entity with an external glossary:* `local258`, `local259`.
  - *Forecast/regression procedures from an external doc:* `local002`, `local061`, `local050`, `local275`, `local277`.
  - *Combinatorial string/set logic:* `local073` (topping lists), `local132` (style pair matching), `local100` (two-hop actor graph).
  - *Huge / many-join:* `local219` (26 joins in the shipped SQL), `local220` (22 player columns), `local035` (1,000,163-row ordered LAG).
- **4 not faithful (3):** `local003` (RFM; gold CSV has labels/segments the shipped SQL does not produce, and that SQL
  ran >25 min), `local066` (pizza ingredient totals; values differ), `local309` (champions per year; 8 of 75 years differ).

### Flags (not categories)

`multi_gold_shapes` 47 · `needs_mentor` 32 · `big_table` 18 · `ext_knowledge` 13 · `gold_unreliable` 3 ·
`gold_sql_disagrees` 3 (`local029`, `local131`, `local210`: SQL shape ≠ CSV, but the CSV is reproducible) ·
`view_dependent` 4 (+4 more `view_possible`) · `odd_field_names` 2 · 91 ids have more than one gold CSV · 31 touch a single table.

## 2. Construct frequency (all 135; multi-label; mostly inferred)

aggregation 126 · GROUP BY 114 · join 98 · filter 51 · sort 43 · date/time 40 · window 36 · limit 31 · CASE 25 ·
subquery 24 · per-group top-N 23 · CTE 19 · set operation 14 · string parsing 13 · non-trivial math 11 ·
running/rolling 9 · HAVING 9 · bucketing 6 · calendar densification 6 · gaps-and-islands 5 · median 3 · recursion 3 ·
FIFO allocation 3 · pivot 2 · graph/self-join 2.

## 3. SQL construct → MQL mapping (to use when converting)

| SQL | MQL |
|---|---|
| `WHERE` | `$match` (use `$expr` for field-vs-field or computed conditions) |
| `SELECT cols` | `$project` / `$addFields` (output only the columns the gold needs, plus any extras are harmless) |
| `ORDER BY ... LIMIT n` | `$sort` + `$limit` (add a deterministic tie-break key) |
| `GROUP BY` + aggregates | `$group` (`$sum/$avg/$min/$max/$push/$addToSet`) |
| `HAVING` | `$match` after `$group` |
| `JOIN` | `$lookup` (+ `$unwind`; `preserveNullAndEmptyArrays` for LEFT) — index the foreign field on big collections |
| subquery in `FROM` / CTE | earlier pipeline stages, or `$lookup` with `pipeline`, or `$facet` for two derived sets |
| scalar subquery / `IN (SELECT ...)` | `$lookup`+`$match`, or two-step: compute the set in a first stage and `$match` with `$in` |
| `UNION [ALL]` | `$unionWith` (+ `$group` for dedup) |
| `INTERSECT` / `EXCEPT` | `$group` + count of sources / `$lookup` anti-join |
| `CASE WHEN` | `$cond` / `$switch` |
| window (`LAG`, `ROW_NUMBER`, `RANK`, running sum) | `$setWindowFields` (`$shift`, `$documentNumber`, `$rank`, `$sum` over `window`) |
| `NTILE(n)` | derive from `$rank` and count in a second `$setWindowFields`/`$bucketAuto` — ties differ; check |
| `DISTINCT` / `COUNT(DISTINCT)` | `$group` then `$count`/`$size` |
| date/time functions | `$dateFromString`, `$dateDiff`, `$dateToString`, `$dateTrunc` |
| string functions | `$substrCP`, `$split`, `$regexFind`, `$replaceAll`, `$toUpper`, `$concat` |
| calendar fill (generate series) | `$densify` or `$range` |
| median / percentile | `$sortArray` + index (exact); `$median` is approximate-only |
| recursive CTE | `$graphLookup` for reachability; `$reduce` for sequential state; fixed-depth `$lookup` unrolling for path products |
| pivot | `$group` with `$cond` sums, or `$arrayToObject` |

## 4. Labels for dataset rows (`query_type`)

For fine-tuning/evaluation breakdowns each converted example carries `query_type` = the **primary pipeline pattern**
(`filter`, `projection`, `sort_limit`, `aggregate`, `join`, `window`, `subquery_multistage`, `set_op`, `date_logic`,
`string_logic`, `procedural`), plus the category (1-4) and flags above. `write` operations are **out of scope**
(Spider is read-only).

## 5. Recommended conversion strategy

1. **Wave 1 (highest confidence):** category 1-2 ids that have gold SQL (20 + `local133`) and whose golds share one shape.
   These can be checked against both shipped SQL and CSV.
2. **Wave 2:** the remaining category-2 ids (55 inferred). For each: reference SQL first, matched to a gold CSV, then MQL
   (`dataset_task_definition.md` §6).
3. **Wave 3 (only with mentor sign-off):** category 3 where the question is well-defined (e.g. gaps-and-islands with a clear spec).
4. **Never auto-convert:** category 4 and `gold_unreproducible` ids.
5. Convert **one database at a time** (shared collections, shared schema description), validate each example immediately
   with the official comparator, then drop our own collections for that database (`schema_notes.md` §9).
6. Do not build a generic SQL→MQL translator: the survey shows that most items need judgement (windows, bucketing, ties,
   NULL denominators), and the pilot needed a per-question interpretation of the denominator.
