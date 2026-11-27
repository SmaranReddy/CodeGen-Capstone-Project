# Independent accuracy audit of `sql2mql-v1`

Audit date: 2026-10-25. Auditor: Claude Code (`claude-opus-5-5`).

| | |
|---|---|
| Audited code | `scripts/finetune_v1/{sql2mql,sqlsem,fixtures,verify,build_ft}.py`, `tests/test_finetune_v1.py` (read and executed, **not modified**) |
| Audit code | `scripts/audit_sql2mql_v1/` (new) |
| Audit outputs | `outputs/audit_sql2mql_v1/` (new); every figure below is recomputed by `summarize.py` into `summary.json` |
| Labels audited | the 1,584 converter-produced ft-v1 labels: 1,504 in `train.jsonl` + 80 in `dev_internal.jsonl` |

Nothing under `scripts/finetune_v1/`, `data/`, the benchmark, its splits, scores or the model
pipeline was changed. The 65-item held-out test partition was not read. The Spider census
asserts in code that none of its databases occur in `data/splits_expanded/{validation,test}`.

## 0. Summary

**The converter is accurate on what it supports, but "fixture-equivalence verified" overstates
the evidence.** Every row difference found in the 1,584 training labels falls into two
deterministic converter bugs, both in edge cases the existing fixtures never generate.

1. **Defects in training labels (explicit-NULL contract): 88 of 1,584 (5.6%, 95% CI 4.5–6.8%).**
   All 88 come from exactly two bugs. Re-running each failure and classifying its rows left
   no unexplained case.
   - **B1, empty-input aggregates (84 labels).** An ungrouped `COUNT/SUM/AVG/MIN/MAX` over
     zero qualifying rows returns *no document*. SQL returns one row (`0` or `NULL`).
   - **B2, `SUM` over only NULLs (20 labels, 16 also B1).** The label returns `0` where SQL
     returns `NULL`.

   On data where at least one row qualifies and no `SUM` group is all-NULL, every one of the
   1,584 labels matched the reference SQL on all of the audit's independent fixture families.
2. **The existing verifier accepts wrong MQL.**
   - **Adversarial suite.** Of 20 wrong translations, it accepted 10 as
     `fixture_equivalence_verified`: B1, B2, division by zero, a numeric literal against a
     text column, and LIKE across a newline.
   - **Mutant study.** On 200 sampled labels it accepted 30 deliberately wrong MQL variants
     that the audit's fixtures showed are not equivalent. 21 of 200 labels (10.5%, CI
     7.0–15.5%) have a plausible wrong MQL the verifier cannot tell apart from the label.
     Examples: swapped GROUP BY columns, a 31-day instead of 30-day interval, a lost
     case-insensitive flag, a dropped NULL guard.
3. **NULL representation is an unstated contract.**
   - All labels assume SQL NULL is stored as an **explicit `null`**.
   - If NULL is an **absent field** (common in real MongoDB), 62 labels (3.9%, CI 3.1–5.0%)
     give wrong results or raise errors. `$expr` treats a missing field as different from
     `null`.
   - The verifier cannot see this: its fixtures always write every field.
4. **Labels can be wrong even when the MQL is faithful.**
   - A semantic review of 120 labels (fresh random sample) found the MQL faithful to its
     SQL in 120/120, except the B1/B2 edge in 7.
   - But the *SQL* does not answer the question in 5 of 120 (4.2%, CI 1.8–9.4%). All 5 are
     a `datetime` column compared for equality with a date-only value.
   - A static scan finds that pattern in 51 of the 1,584 labels. The verifier cannot catch
     it by construction, because it only checks MQL against the SQL.
5. **Original-database test: Spider train, original SQLite databases (§5).**
   - **Coverage.** 3,438 of the 4,731 unique queries convert and run. 26% are rejected as
     unsupported.
   - **Clean data.** On queries whose columns have consistent stored types, 7.2% of results
     are wrong (CI 6.4–8.1%):
     - B1 accounts for 5.5%, mostly from seven Spider databases whose tables are all empty;
     - missing implicit type coercion (TEXT `'40'` vs `50`, numbers stored as text)
       accounts for 1.7%;
     - pure operator-logic errors are **2 of 3,036 (0.07%)**. Both are new bugs: DISTINCT
       is dropped when GROUP BY is present, and date-string subtraction raises an error.
   - **Type-flagged data.** Where join keys or columns mix int and text, 58% are wrong.
   - **Identifiers.** 55% of the Spider queries needed identifier-case canonicalisation
     before conversion. The converter copies names verbatim, and MongoDB field names are
     case-sensitive.

**Verdict (§9):**

| Use | Safe? |
|---|---|
| Generating candidate MQL | Yes |
| Training labels after automated verification | Yes, after B1/B2 are fixed or the 88 labels are removed, and with the NULL contract stated |
| High-confidence labels without independent review | No |

## 1. Scope, constraints and independence

| Question | Evidence used | Independence from the audited code |
|---|---|---|
| What does the code claim to support? | Reading the five modules and the 30 tests (only ~5 tests exercise the converter directly) | Static reading |
| Is each feature translated correctly? | 103 adversarial cases with **hand-written expected rows** (`adversarial_cases.py`) | Expectations written from SQL semantics before running anything. SQLite is used only as a cross-check and agreed on 103/103. |
| Would the existing verifier notice? | The same 103 cases run through `verify.verify_pair` | Uses the existing verifier as the object under test |
| Is SQLite the right oracle for the clock functions? | 11 `sqlsem` UDF checks + 3 MongoDB date-arithmetic checks against hand-computed MySQL 8 results | 14/14 agree (`adversarial_udf.json`) |
| Can the verifier tell wrong MQL from the label? | MQL-mutant kill study, 200 labels, 1,122 mutants | Mutants are of the MQL. Survivors are re-checked with the audit's own fixtures. |
| Are the 1,584 labels right? | Census with the audit's own fixture families (`audit_fixtures.py`): 4 random, empty, zero-match, all-NULL, duplicated, plus a missing-field re-run | Own generator, own comparator (`common.py`). Gold = reference SQL on SQLite via `sqlsem.to_sqlite`, the only way to run MySQL clock functions on SQLite; its UDFs were checked as above. |
| Does the label answer the question? | Semantic review of a fresh seeded sample of 120 | Reviewer is the same model family as the converter's author (disclosed limitation) |
| Does the converter work on genuine data? | Spider train census on the original SQLite databases | Real data, real gold SQL, no synthetic fixtures |

**Limits that could not be removed:**
- **The Kaggle examples have no schema or database.** `data/interim/finetune_v1/schema.json`
  is inferred from the SQL itself, and the converter, the old fixtures and the audit
  fixtures all share it. Errors in the inferred schema (wrong types, missing columns) are
  untestable here. That covers whether `user_id` is really text and whether a `*_date`
  column is really `datetime`.
- **The Kaggle examples are evidence-labelled `audit_multi_fixture`, never "verified
  against the source database".**
- **MySQL oracle.** MySQL 8.0 and PostgreSQL 18 are running locally, but the audit had no
  credentials, and probing for them was not appropriate. MySQL-specific behaviour (`=`
  collation, PAD SPACE, implicit casts) is therefore argued from documentation (§2), not
  executed. Running the reference SQL on MySQL would need credentials added to `.env` by
  the user.
- **MongoDB.** Atlas 8.0.34, read-only user. All data was shipped inline with `$documents`;
  nothing was written to Atlas.

## 2. Coverage matrix

Status: **S** supported and correct in every test · **P** partially correct (named failure
mode) · **U** unsupported (rejected by `sqlsem.analyze` or `sql2mql.convert`, so no label
is produced). Case IDs refer to `outputs/audit_sql2mql_v1/adversarial.jsonl`. "Kaggle
labels" counts the 1,584 audited labels using the feature.

| Feature | Status | Implementation path | Evidence | Kaggle labels |
|---|---|---|---|---|
| Column projection, aliases | S | `convert` non-grouped branch → `$project` (`out_name` builds names) | P1, P6; census | most |
| Literal in SELECT | S | `value_expr` → `$literal` for non-strings | P3, P4 | few |
| `+ - *` (NULL-propagating) | S | `value_expr` → `$add/$subtract/$multiply` | P2, P5, N4 | few |
| `/` | **P** | `value_expr` → `$divide` | N1, N3: **division by zero raises a MongoDB error** (SQL returns NULL), and the verifier accepts it because its fixtures never contain a 0 divisor. Uses real division (MySQL); Spider/SQLite uses integer division. | 0 |
| `SELECT *` | S (single table) | `$project {_id: 0}` | P6 | 1,020 |
| `SELECT *` with JOIN, qualified `t.*` | U | `analyze`: `select_star_with_join`, `qualified_star` | — | 0 |
| LOWER / UPPER / LENGTH / DATE / YEAR / MONTH / DAY of a column | S under explicit NULL; **P** under missing fields | `value_expr` + `nullsafe` (`$cond` on `== null`) | W21, W22, W27–W29 pass. With absent fields: `$toInt("")` and `$strLenCP(missing)` raise errors, and `LOWER(missing)` becomes `""`. | 39 |
| CASE, IF, COALESCE, CAST, CONCAT, ROUND, other functions | U | `analyze` (`case`, `if`, `select_expr:*`) | — | 0 |
| Column vs literal: `= <> < <= > >=` | S | `predicate` → query operators; `<>` → `$nin [v, null]` | W1, W2 | most |
| Numeric literal vs text column (implicit cast) | **P** | `predicate` emits `{col: 10}` | W17: MySQL and SQLite both cast and match `'10' = 10`, but MongoDB matches nothing. The verifier accepts it because the old fixtures insert an *int* into the text column. | depends on the inferred schema |
| Column vs column | S under explicit NULL; **P** under missing fields | `expr_compare` with `$ne: null` guards | W19, W19b. A missing field passes `$ne: [x, null]` (verified on Atlas: `$ne: [missing, null]` is `true`). | few |
| AND / OR / NOT, three-valued logic | S | De Morgan in `predicate(negate=…)`; every negated atom excludes NULL | W2–W4b, W15 | 1,083 WHERE |
| `NOT <bare boolean column>` | **P (bug)** | `predicate`, `exp.Column` with negate → `$in [0, null]` | W14: **includes NULL rows**, whereas SQL `NOT NULL` is unknown. The verifier rejects it, so no label is affected. | 0 |
| IN / NOT IN literal lists | S | `$in`; `$nin [..., null]` | W5, W5b. A NULL inside the list is rejected by the analyzer. | 61 |
| IN (subquery), EXISTS, correlated, scalar subqueries | U | `analyze`: `subquery` | — | 0 |
| BETWEEN / NOT BETWEEN | S | `predicate` → `$gte/$lte`; NOT → `$or` of `$lt/$gt` | W6, W6b | 85 |
| LIKE / NOT LIKE / ILIKE | **P** | `like_regex`: `%` → `.*`, `_` → `.`, anchored, `$options: i` | W7–W10c, W10f pass. W10d/W10e: `_` and `%` do **not match a newline**. W10e is accepted by the verifier. LIKE … ESCAPE is U; MySQL's default backslash escape is not modelled. | 153 |
| IS NULL / IS NOT NULL | S | `{f: null}` / `{f: {$ne: null}}` | W12, W12b; correct for absent fields too | 126 |
| `HAVING agg IS NULL` | U | `analyze`: `having_is_form` | A15 | 0 |
| Bare boolean column, `= TRUE` | S | `{f: {$nin: [0, null]}}`, TRUE → 1 | W13, W30. Assumes booleans are stored as 0/1, not BSON `true`. | few |
| String equality, case | S (SQLite) | query equality is binary | W16 matches SQLite/ISO. **MySQL's default collation is case-insensitive and PAD SPACE**, so the source-dialect answer differs. | 227 |
| DISTINCT | S | `$group` on all select columns | D1–D3, D5; NULL kept once | 69 |
| DISTINCT together with GROUP BY | **P (bug)** | the grouped branch of `convert` never reads `distinct` | Spider `train_spider:3451`. Reproduced: `SELECT DISTINCT grp FROM t GROUP BY grp, is_active` returns duplicates. | 0 |
| Arithmetic on date/text columns | **P (gap)** | `_ok_value_expr` accepts `-` on any column | Spider `train_spider:6101`: `$subtract` on strings raises an error | 0 |
| ORDER BY, ASC/DESC, multi-key, NULL placement | S | `$sort` (key order preserved) | O1–O3, O6, O7, O10, O12. NULL first ASC / last DESC, as in MySQL and SQLite. | 170 |
| ORDER BY ordinal (`ORDER BY 2`) | **P (bug)** | `sort_target` returns the *output* name, but `$sort` runs **before** `$project` | O4, O4c, O5, D4: the sort key does not exist, so the order is arbitrary. Correct only when the output name equals the source field (O4b). The verifier rejects these, so no label is affected. | 0 |
| String sort / MIN / MAX on text | S (binary) | `$sort`, `$min`, `$max` | O11, A10 match SQLite; MySQL ci collation orders differently | few |
| LIMIT / OFFSET | S | `$skip`, `$limit` | O10 | 58 |
| LIMIT without ORDER BY | nondeterministic | insertion order | O9: any k rows is a correct SQL answer; the label returns the first k | 22 |
| Ties at a LIMIT boundary | nondeterministic | — | O8 | some |
| COUNT(*), COUNT(col), COUNT(DISTINCT col) | S on non-empty input; **P** | `accumulator` | A5, A7, A7b pass. **A1, A1b: no row on empty input (B1).** With absent fields, COUNT(col) counts missing values (A5@missing). | 122 + 3 |
| SUM | **P (bug)** | `accumulator` → `$sum` | **A6, A6b, J4b: an all-NULL group gives `0`, not NULL (B2). A2: empty input gives no row (B1).** The verifier accepts both. | 21 |
| AVG, MIN, MAX | S on non-empty input; **P** | `$avg/$min/$max` | A4, A10, A11, A14 pass. A3: empty input gives no row (B1). | 165 |
| GROUP BY (columns, YEAR(col)), HAVING (agg or alias) | S | `$group` → `$match` on accumulators | A8, A9, A9b, A13; NULL forms one group | 207 |
| INNER equi-JOIN | S when key types are identical | `$lookup localField/foreignField` + `$unwind`; local NULL keys pre-filtered | J1, J2, J7 pass. **J8: text `'1'` vs int `1` keys never match in MongoDB** (SQLite/MySQL coerce them); see the Spider census. | 30 |
| `$lookup "as"` = joined table name | **P (bug)** | `convert`: `as` is the bare table name | J6: if the base table has a column with that name, it is overwritten. The verifier rejects it. | 0 seen |
| LEFT JOIN rows, anti-join, WHERE on the right table | S under explicit NULL | `$lookup` pipeline + `$unwind preserveNullAndEmptyArrays` | J3, J5, J9, J10. With absent fields, a NULL key matches a missing key (J3/J5@missing). | 0 |
| LEFT JOIN + COUNT(right.col) / SUM(right.col) | **P (bug)** | the unmatched side is an absent field, so the `$cond: [{$ne: [x, null]}]` guard is true | **J4: counts unmatched rows as 1** (the verifier rejects it). **J4b: SUM gives 0** (the verifier accepts it). | 0 |
| RIGHT / FULL / CROSS / comma / USING / self / non-equi joins | U | `analyze` (`join:*`) | — | 0 |
| UNION, CTE, window functions | U | `analyze` | — | 0 |
| Date strings, clock functions (NOW, CURDATE, ± INTERVAL, DATE_SUB/ADD) | S under stated assumptions | `clock_expr` → `$$NOW`, `$dateTrunc`, `$dateAdd/$dateSubtract`, `$dateToString` | C1–C9 pass; 14/14 UDF and Mongo date checks; month-end clamping matches MySQL. Assumes dates are stored as `'YYYY-MM-DD[ HH:MM:SS]'` strings (YEAR() needs that) and that the clock is UTC; MySQL uses the session time zone. Labels are time-relative. | 278 |
| Identifier case | **P** | names copied verbatim | MongoDB field names are case-sensitive; SQL identifiers are not. Spider SQL needed canonicalisation (§5). | — |
| Missing-field NULL representation | **P (contract)** | every `$expr`/`$cond` NULL guard | 13 adversarial cases (W19, W19b, W21, W22, W27, W28, A5, A5b, A13, J3, J5, C4, C7) and 62 labels fail | 62 |

## 3. Adversarial suite (103 cases)

`python -B scripts/audit_sql2mql_v1/adversarial.py` → `adversarial.jsonl`. Every case was run
on Atlas twice (NULL as `null`, and NULL as an absent field) and through the existing
`verify.verify_pair`.

| Result | Count |
|---|---|
| Cases / converted / unsupported | 103 / 102 / 1 (A15) |
| Hand expectation = SQLite (`sqlsem`) result | 103 / 103 |
| Converter correct, explicit null | 82 / 102 |
| Converter correct, absent fields | 69 / 102 |
| **Wrong, and the existing verifier says `fixture_equivalence_verified`** | **10**: W10e, W17, A1, A1b, A2, A6, A6b, J4b, N1, N3 |
| Wrong, and the existing verifier rejects or cannot discriminate | 10: W10d, W14, A3, D4, O4, O4c, O5, J4, J6, J8 |
| Correct with explicit null, wrong with absent fields | 13 |

### Reproduced counterexamples (minimal failing examples, correct expected behaviour)

| ID | SQL | Data | Expected (SQL) | Converter MQL result | Verifier |
|---|---|---|---|---|---|
| A1 | `SELECT COUNT(*) FROM t WHERE x > 100` | 5 rows, x ≤ 3 | `[(0,)]` | `[]` | **accepted** |
| A1b | `SELECT COUNT(*) FROM e` | empty table | `[(0,)]` | `[]` | **accepted** |
| A2 | `SELECT SUM(x) FROM t WHERE x > 100` | as A1 | `[(NULL,)]` | `[]` | **accepted** |
| A6 | `SELECT grp, SUM(x) FROM t GROUP BY grp` | group `b` has x = NULL, NULL | `('b', NULL)` | `('b', 0)` | **accepted** |
| J4b | `SELECT c.name, SUM(o.amt) FROM customers c LEFT JOIN orders o ON c.id = o.cust_id GROUP BY c.name` | Ben has no orders | `('Ben', NULL)` | `('Ben', 0)` | **accepted** |
| N1 | `SELECT id, a / b FROM nums` | one row has b = 0 | `(2, NULL)` | MongoDB error `can't $divide by zero` | **accepted** |
| W17 | `SELECT id FROM t WHERE code = 10` (`code` TEXT) | `code` = '10' | ids 1, 5 | `[]` | **accepted** |
| W10e | `SELECT id FROM l WHERE name LIKE 'a%d'` | `'ab\ncd'` | id 7 | `[]` (`.` does not match `\n`) | **accepted** |
| W14 | `SELECT id FROM t WHERE NOT is_active` | is_active = NULL in row 3 | ids 2, 5 | ids 2, 3, 5 | rejected |
| O4 | `SELECT x AS val FROM t ORDER BY 1` | — | NULL, NULL, 1, 2, 3 | insertion order (sorts on nonexistent `val`) | rejected |
| O5 | `SELECT grp, COUNT(*) FROM t GROUP BY grp ORDER BY 2 DESC, 1` | — | (a,2), (b,2), (NULL,1) | unsorted | rejected |
| J4 | `SELECT c.name, COUNT(o.id) … LEFT JOIN … GROUP BY c.name` | Ben, Eve unmatched | Ben 0, Eve 0 | Ben 1, Eve 1 | rejected |
| J6 | `SELECT c.name, c.orders, o.amt FROM customers c JOIN orders o ON …` | `customers.orders` = 2 | `('Ann', 2, 5.0)` | `c.orders` is replaced by the joined document | rejected |
| J8 | `… FROM orders_t o JOIN customers c ON o.cust_id = c.id` (text vs int) | '1' vs 1 | 2 rows | `[]` | not discriminating |
| C4@missing | `SELECT id FROM ev WHERE ts < NOW()` | ts absent in one row | that row excluded | included (`$lt: [missing, x]` is true) | not visible |

**Cause of B1:** `convert` emits `{$group: {_id: null, …}}`, which produces no document from
empty input. **Cause of B2:** `$sum` ignores NULL and returns 0. Both are standard MongoDB
behaviours; the converter never compensates for them. Correct pattern for B1 (not applied,
per the audit brief): `$group` + `$unionWith` a default document, or `$facet`; for B2,
`$cond` on a non-NULL count.

## 4. Does the verifier catch wrong MQL? (mutant kill study)

`python -B scripts/audit_sql2mql_v1/mql_mutants.py --n 200`

- **Sample.** A seeded simple random sample (seed 20261025) of 200 of the 1,584 labels.
- **Fixtures.** The existing fixture set was regenerated exactly as `verify_pair` builds it.
  The label itself passed it in 200/200.
- **Mutants.** 1,122 mutants (at most 30 per label) from 26 operators.
- **Kill test.** A mutant counts as killed if the pipeline's own comparator
  (`spider1_kit.docs_to_rows` + `fixtures.compare`) reports a mismatch on any kept variant.
- **Survivors.** Each survivor was re-run on the audit's fixture families. A kill there
  counts only on a family where the unmutated label itself passes, so B1/B2 failures of the
  label are not credited to the mutant.

| | Count |
|---|---|
| Mutants | 1,122 |
| Killed by the existing fixtures | 998 (88.9%) |
| Survivors | 124 |
| ... shown non-equivalent by the audit fixtures (wrong MQL the verifier would accept) | **30** |
| ... not killed by either (equivalent, or undetected by both) | 94 |
| Labels with at least one accepted non-equivalent mutant | **21 / 200 = 10.5% (CI 7.0–15.5%)** |

Operators where the existing fixtures are weakest:

| Operator | Mutants | Killed by existing | Survivors killed by audit | Example of an accepted wrong MQL |
|---|---|---|---|---|
| `null_guard_drop` | 64 | 17 (27%) | 2 | a NULL-key inner join, or `due_date <= CURDATE()+10` that now includes NULL rows |
| `project_swap` | 47 | 33 (70%) | 13 | `facility_id` and `incident_type_id` swapped in a two-key GROUP BY |
| `regex_case` | 28 | 15 (54%) | 6 | `LIKE 'ABC%'` made case-sensitive |
| `date_amount`, `literal_plus1` | 23, 65 | 14, 56 | 4, 4 | `INTERVAL 30 DAY` emitted as 31 days |
| `cmp_boundary` | 70 | 50 (71%) | 0 | 20 survivors, mostly `>` vs `>=` against a clock value (equivalent at second resolution) |
| `distinct_to_all` | 1 | 0 | 1 | `COUNT(DISTINCT x)` emitted as `COUNT(x)` |

The reason is structural. `fixtures.adequate_set` requires every *SQL* mutant (WHERE atom
drop/negate/op-swap, ORDER flip, LIMIT/OFFSET/DISTINCT drop) to be witnessed. It does not
target aggregation, projection order, NULL handling, case sensitivity or interval arithmetic,
and its GROUP BY columns share one small value pool. Kill rates are lower bounds: some of
the 94 unkilled survivors may be non-equivalent in ways neither fixture set exercises.

## 5. Original-database test: Spider train census

`python -B scripts/audit_sql2mql_v1/spider_census.py`, then `classify_spider_failures.py`.

**Setup:**
- **Queries.** Every unique (db, SQL) pair in Spider 1.0 `train_spider` + `train_others`:
  4,731 pairs over 146 databases.
- **Held-out guard.** The script asserts that none of these databases appear in Spider dev
  or the benchmark's validation/test partitions. Nothing from the 455-item benchmark was
  read.
- **Gold.** The **original SQL text on the original SQLite database** (read-only).
- **MongoDB side.** The converter's MQL over the same database's rows, shipped with
  `$documents` and SQLite's stored types preserved.

**Evidence label: `audit_original_db`.** These figures describe the Spider-train
distribution. They are not an estimate for the Kaggle labels: Spider has almost no clock
queries, and its data has real type noise.

| | Queries |
|---|---|
| Unique (db, SQL) | 4,731 |
| Unsupported (rejected; no MQL) | 1,238 (26.2%). Top reasons: subquery 424, "grouped expression not available" 305, INTERSECT 136, unqualified column in a join 127, EXCEPT 113, self-join 49, UNION 33 |
| Skipped: database too large for inline `$documents` (`baseball_1`, `soccer_1`, `wta_1`) | 55 |
| **Tested** | **3,438** |
| SQL identifiers needed case canonicalisation first | 1,905 (55%). The converter copies names verbatim, MongoDB field names are case-sensitive, and SQLite's are not. **Run raw on Spider, the converter would emit wrong field names for more than half the queries.** |
| Tested with no data-type anomaly in the referenced columns ("clean") | 3,316 |
| Tested with mixed-type columns or join keys of different stored types ("type-flagged") | 122 |

Results, explicit-NULL representation. The absent-field run differs by at most 5 queries in
any cell, because Spider queries rarely hit the `$expr` NULL paths.

| Stratum | Pass | Wrong | Inconclusive (ties) | Harness timeout |
|---|---|---|---|---|
| Clean | 3,034 | 235 | 32 | 15 |
| Type-flagged | 50 | 69 | 3 | — |

Every wrong result was classified from the stored data:

| Class | Clean | Flagged | Nature |
|---|---|---|---|
| B1 empty-input aggregate (no row qualifies on the SQL side; checked with a COUNT on the real database) | 179 | 0 | **Converter bug.** 160 of these come from seven Spider databases whose tables are **all empty** (checked: academic 50, yelp 36, geo 24, scholar 24, imdb 15, restaurants 6, music_2 5). There the gold answer is `0`/NULL and the MQL returns nothing. |
| Literal type vs stored type (e.g. TEXT `'40'` vs `50`) | 31 | 4 | SQLite coerces by column affinity; MongoDB compares types strictly. The converter does not model implicit casts. |
| SUM/AVG/MIN/MAX over numbers stored as text | 23 | — | `$avg` and `$sum` ignore strings; SQLite converts them. |
| Join keys of different stored types (int vs text) | — | 55 | `$lookup` is strictly typed; SQLite and MySQL coerce. |
| Other mixed-type column effects | — | 10 | e.g. SQLite's `'' > 85` is true |
| **DISTINCT combined with GROUP BY** | 1 | — | **New converter bug.** The grouped branch ignores DISTINCT. Reproduced: `SELECT DISTINCT grp FROM t GROUP BY grp, is_active` returns `a, b, a, b, NULL`. 0 Kaggle labels use it. |
| **Arithmetic on date strings** (`avg(active_to_date - active_from_date)`) | 1 | — | **Converter gap.** The analyzer accepts `-` on text columns; `$subtract` on strings raises an error. SQLite coerces the text to numbers. |

**Rates on clean queries (3,269 decided, excluding ties and timeouts):**

| Measure | Wrong | Rate (95% CI) |
|---|---|---|
| All causes | 235 | **7.2% (6.4–8.1%)** |
| B1 alone | 179 | 5.5% (4.8–6.3%) |
| Implicit type coercion | 54 | 1.7% (1.3–2.2%) |
| Neither (pure logic) | 2 of 3,036 | **0.07% (0.02–0.24%)** |

On type-flagged queries, 69 of 119 decided (58%) are wrong.

**Conclusion from genuine data:**
- For the SQL it accepts, the converter's operator logic is almost always right; the two
  exceptions are listed above.
- On real databases, its dominant failure modes are the empty-input bug (B1) and the absence
  of SQL's implicit type coercion.
- None of these is visible to fixture equivalence. The old fixtures are generated
  type-consistent from the same inferred schema, and never contain empty tables.

## 6. Kaggle labels: independent multi-fixture census (all 1,584)

`python -B scripts/audit_sql2mql_v1/kaggle_census.py` tests the stored `target_mql`.
Re-converting the SQL reproduced it byte-for-byte in 1,584/1,584. Gold is the reference SQL
on SQLite. There are 8 explicit-NULL families per label, plus 4 random families re-run
with NULLs stored as absent fields.

**Evidence label: `audit_multi_fixture` on the inferred schema. These are not
source-database results.**

| | Labels | Rate (95% Wilson CI) |
|---|---|---|
| Pass, explicit null | 1,495 | |
| **Fail, explicit null** | **88** | **5.6% (4.5–6.8%)** |
| Inconclusive (only tied LIMIT boundaries) | 1 | |
| Fail, absent-field representation | 62 | 3.9% (3.1–5.0%) |

Failure classes were assigned by re-running every failing (label, family) pair and comparing
rows (`classify_kaggle_failures.py`). B1 additionally requires that zero rows qualify on the
SQL side of that fixture.

| Class | Labels (explicit null) | Mechanism |
|---|---|---|
| B1 empty-input aggregate | 84 | `$group {_id: null}` emits nothing; SQL emits one row (`0` / `NULL`) |
| B2 `SUM` over only NULLs | 20 (16 also B1) | `$sum` → 0, SQL → NULL |
| Anything else | **0** | — |
| Absent-field errors (`$toInt("")`, `$strLenCP(missing)`) | 28 (absent fields only) | `nullsafe` checks `== null`, which is false for a missing field |
| Absent-field wrong rows | 28 (absent fields only) | `$expr` guards/comparisons treat missing ≠ null |

**How often the failures occur.** Only 6 of the 88 failing labels also failed on a random
family (ids in `summary.json`), and each did so because that random fixture happened to
qualify no rows or contain an all-NULL group; the others fail only on the empty, zero-match or all-NULL families. In practice
these labels are wrong only when nothing qualifies (B1) or a summed column is entirely NULL
(B2). That is rare on populated data, but routine for questions like "total revenue in the
last 24 hours", which several of these labels are.

## 7. Kaggle labels: semantic review (120 labels)

- **Sample.** A fresh seeded sample (seed `semantic-20261025`), independent of
  `ft_v1/semantic_audit_sample.json`. Judgements are in `semantic_review.jsonl`.
- **Reviewer.** Claude (`claude-opus-5-5`), the same model family that wrote the converter.
  Read these as a second opinion, not as human review.

| Judgement | Result |
|---|---|
| MQL faithful to its SQL | 113 fully; 7 except the B1/B2 empty or all-NULL edge (all 7 are ungrouped aggregates); 0 otherwise unfaithful (rule-of-three upper bound 2.5%) |
| SQL answers the question | 98 yes; 16 yes with a minor same-day `date`-vs-`datetime` boundary difference; 1 arguable; **5 no** |
| **Label correct overall** | **98 correct + 16 correct-with-minor-boundary; 1 arguable; 5 incorrect = 4.2% (CI 1.8–9.4%)** |

- **The 5 incorrect labels** (`idx` 5, 50, 69, 113, 115) compare a `datetime` column for
  equality with a date-only value. Examples: `date_created = '2023-01-01'` for "created on
  2023-01-01", and `trial_ends_at = CURDATE()` for "ending today". In MySQL and in the
  label, these match only timestamps at exactly midnight.
- **A static scan** finds this pattern in **51 of the 1,584 labels (3.2%)**, conditional on
  the inferred schema typing those columns `datetime`. The ids are listed in
  `outputs/audit_sql2mql_v1/datetime_date_equality.jsonl`.
- **These are source-dataset SQL errors, not converter errors.** Fixture equivalence cannot
  detect them by design.
- **The arguable label** (idx 2) hard-codes "at least three types of transportation (e.g.
  car, train, plane)" as `car AND train AND plane`.

## 8. Accuracy estimates

| Quantity | Population / method | Estimate |
|---|---|---|
| Converter output = SQL, on populated data | 1,584 labels, 4 random + duplicate families | 1,584 / 1,584 pass on non-degenerate random families, apart from 6 labels whose random data happened to hit B1/B2 |
| Converter output = SQL, all audited edge cases, explicit null | 1,584 labels (census, no sampling) | **94.4% correct; 88 wrong (5.6%, CI 4.5–6.8%)**, all B1/B2 |
| Same, NULL stored as an absent field | 1,584 labels | 3.9% wrong (CI 3.1–5.0%), on top of B1/B2 for most of those labels |
| SQL itself answers the question | 120-label random sample, single reviewer | 4.2% incorrect (CI 1.8–9.4%); 0.8% arguable |
| Labels the verifier cannot tell apart from a plausible wrong MQL | 200-label random sample | 10.5% (CI 7.0–15.5%) |
| Converter on genuine databases, consistently typed data | Spider train census, 3,269 decided queries (census) | 7.2% wrong (CI 6.4–8.1%): B1 5.5%, implicit-cast gaps 1.7%, logic 0.07% |
| Converter on genuine databases, type-heterogeneous data | 119 decided queries | 58% wrong (join-key and column type mixing) |
| Could not be verified conclusively | 1 Kaggle label (ties only) | — |

**Interpreting these numbers:**
- **The pipeline's own pass rate is not an accuracy estimate.** 2,561 of 2,621 eligible
  pairs passed. The labels were *selected* by the verifier, so errors found among them
  measure the verifier's false acceptances, not the converter's raw accuracy.
- **The 88 wrong labels are a census count, not a sample estimate.** The CI is given only
  as a guide to how the rate would generalise to similar data.
- **The label-level error rate combines several effects.** Some combination of B1/B2 (5.6%),
  the SQL date-equality pattern (about 3.2%) and other question-level problems (about 1%
  from the review) makes the true rate of labels that are wrong for some input roughly
  **8–11%**. The overlap between these sources was not measured, so this is an
  approximate sum. Most of these errors appear only on edge-case data.

## 9. Recommendations and verdict

**1. Safe for generating candidate MQL? Yes.**
- On everything it supports, the output matched the reference SQL on populated data.
- It refuses what it cannot translate: subqueries, CASE, CAST, outer joins other than LEFT,
  and more.
- The known bugs (B1, B2, division by zero, ordinals, `NOT bare_col`, LEFT JOIN + COUNT/SUM,
  `$lookup` name collision, LIKE with newlines, DISTINCT + GROUP BY, date subtraction) all
  have small, mechanical fixes.
- **Before using it on Spider train** (the next step recommended by
  `dataset_research_report.md`), it needs four additions:
  - identifier-case canonicalisation (55% of queries need it);
  - the B1 fix;
  - type-aware literals and join keys, or verification against the real database. With
    real-database execution, the coercion and key-type failures in §5 are rejected rather
    than shipped, so L1 verification on Spider is essential, not optional.

**2. Safe for training labels after automated verification? Yes, conditionally.** Before the
next build, do the following:
- Fix B1 and B2 in a versioned `sql2mql-v2`, or drop the 88 labels listed in
  `kaggle_failure_classes.jsonl`. As they stand, they teach the model that "how many X…"
  returns nothing when there are no X.
- Add the audit's families to the verifier: empty tables, zero-match WHERE, all-NULL
  aggregate columns, duplicate rows, case variants for LIKE, distinct values per GROUP BY
  key, and a zero divisor.
- State the NULL contract (explicit `null`, all fields present, dates as strings, booleans as
  0/1) in the prompt or the dataset card. Alternatively, emit missing-safe MQL
  (`$ifNull` / `$type` guards) and verify it under both representations.
- Drop or repair the 51 labels with `datetime = date` equality (`datetime_date_equality.jsonl`). They are SQL errors that no
  execution check can find.
- Keep the status name honest: `fixture_equivalence_verified` means "equal to the reference
  SQL on synthetic data". It says nothing about the question or the source database.

**3. Safe for high-confidence labels without independent review? No.**
- The verifier accepted wrong MQL in 10 of 20 adversarial cases.
- It cannot distinguish a plausible wrong MQL for 10.5% of labels.
- It cannot see question/SQL mismatches, which affected 4.2% of the reviewed sample.
- High-confidence status needs at least one of: execution on a real database (possible for
  Spider, not for Kaggle), the strengthened fixture families above, or a human semantic
  review sample with a stated error bound.

### Demonstrated vs. assumed vs. untested

- **Demonstrated** (executed, reproducible from `outputs/audit_sql2mql_v1/`):
  - every Pass/Fail in §2–§6;
  - the Spider failure classes;
  - B1 and B2 are the only explicit-NULL failure modes in the 1,584 labels under the audit's
    families;
  - the verifier's blind spots in §4.
- **Assumed:**
  - The inferred Kaggle schema is right: column existence, types, date-string storage.
  - MySQL semantics where they differ from SQLite (`=` collation, PAD SPACE, implicit casts).
    These were argued from documentation, not executed.
  - A UTC clock.
  - Booleans stored as 0/1.
- **Untested:**
  - MySQL execution of the reference SQL.
  - Behaviour on any real Kaggle-domain database (none exists).
  - Unsupported SQL features (no labels are produced for them).
  - Mutants of kinds not generated by the 26 operators.
  - Whether the 94 unkilled mutant survivors are all equivalent.

## Reproduction

```
python -B scripts/audit_sql2mql_v1/adversarial.py               # §3  (~10 min, Atlas)
python -B scripts/audit_sql2mql_v1/kaggle_census.py             # §6  (~25 min)
python -B scripts/audit_sql2mql_v1/classify_kaggle_failures.py  # §6
python -B scripts/audit_sql2mql_v1/mql_mutants.py --n 200       # §4  (~30 min)
python -B scripts/audit_sql2mql_v1/spider_census.py             # §5  (~1 h)
python -B scripts/audit_sql2mql_v1/classify_spider_failures.py  # §5
python -B scripts/audit_sql2mql_v1/semantic_sample.py           # §7  sample (judgements: semantic_review.jsonl)
python -B scripts/audit_sql2mql_v1/summarize.py                 # summary.json
```

`-B` keeps bytecode out of `scripts/finetune_v1/`. The scripts only read project data and
write under `outputs/audit_sql2mql_v1/`. They never call `verify.main()` or `build_ft.py`.
