# Independent audit of `spider_pool.zip`

Audit date: 2026-11-05. **Read-only: the archive, the datasets, the converter, verifier-3, the pilot and the G4 artifacts were not changed. MongoDB and Atlas were not touched.** Only SQL-side data was audited, so every figure here is a dataset-audit figure and none is an NL→MQL model metric.

Archive audited: `spider_pool.zip` at the repo root, sha256 `31848241861988ebb0619e1802c38fb5d90cfb62f818a65a53cf601340888a6b`, 973,318 bytes. It is the only copy found in the repo and the user folders searched (Downloads, Desktop, Documents, OneDrive, `~/codegen_data`). A whole-disk search had found no other copy when it was stopped unfinished, so copies elsewhere on the disk cannot be ruled out. The sha256 was re-checked after the audit and is unchanged.

Evidence: `outputs/spider_pool_audit_v1/` (scripts `scripts/a01`–`a09`, stage JSON files, record-level files in `records/`).

## A. Executive verdict

**CONDITIONAL PASS: `train.jsonl` only, after the exclusions below. The package as delivered must not be used as a training artifact.**

| Part | Result |
|---|---|
| `train.jsonl`, mechanical quality | Strong. All 5,514 rows execute and match their hashes. SQL, `db_id` and `question_raw` match Spider byte for byte; `question` differs in whitespace only, in 34 train/val rows |
| Label correctness | Not clean. 57 confirmed label defects survive (53 train, 4 val), and a hand sample found 3/80 clear errors |
| Package as "training-side only" | **Fails.** It holds all 1,034 Spider dev records, including all 65 held-out MQL test items, with gold SQL |
| Provenance | Inputs verified, process not. The builder script is in neither the archive nor the repo |

Conditions: (1) use only `train.jsonl` (and `val_internal.jsonl` for checkpoint selection); (2) keep `dev.jsonl` and the dev rows of `rejected.jsonl` away from every pipeline; (3) exclude Tier A and Tier B (section F); (4) obtain `build_spider_pool.py`, or record that the cleaning is unreproducible.

## B. Dataset inventory

| File | Records | Unique IDs | Distinct queries (`group`) | Databases | Audit depth |
|---|---:|---:|---:|---:|---|
| `train.jsonl` | 5,514 | 5,514 | 3,150 | 123 (659 tables) | Full |
| `val_internal.jsonl` | 744 | 744 | 440 | 16 | Full |
| `dev.jsonl` | 899 | 899 | 482 | 20 | IDs and metadata only (protected) |
| `rejected.jsonl` | 2,447 | 2,447 | n/a | n/a | Train/val rows 2,312 in full; dev rows 135 metadata only |
| **Total** | **9,604** | **9,604** | | | |

- Train + val together: 139 databases and 728 tables. Collections do not exist yet, because nothing has been converted.
- All 5,514 train rows come from `train_spider.json`; `train_others.json` contributes 0, because all its databases are empty officially.
- Archive: 7 entries (1 directory, 6 files), 0 CRC errors, 0 duplicate paths, 0 unsafe paths. **It holds 0 SQLite files, 0 schema files and 0 scripts.**

| Funnel (spider-sql-v1 → pool) | Train | Val | Dev (ID-level only) |
|---|---:|---:|---:|
| Input | 7,773 | 797 | 1,034 |
| R1 weak verification (empty official DB) | 1,744 | 0 | 0 |
| R3 result mismatch | 4 | 0 | 0 |
| R4 empty / all-NULL | 159 | 18 | 53 |
| R5 single 0 | 39 | 5 | 24 |
| R6 over 1,000 rows | 38 | 0 | 12 |
| R7 LIMIT tie | 225 | 19 | 44 |
| R8 order not verifiable | 50 | 9 | 2 |
| R9 duplicate question | 0 | 2 | 0 |
| **Kept** | **5,514** | **744** | **899** |
| Silently omitted | 0 | 0 | 0 |
| In both kept and rejected | 0 | 0 | 0 |

Before spider-sql-v1, the Spider source had 8,659 train rows; 89 were dropped there, and 0 of those 89 appear anywhere in this package.

## C. Quality results

Population: every kept train and val row (6,258). Percentages use that base. Dev was not included. A flag is a screen hit, not necessarily an error; section F says which are errors.

**Checks that found nothing (0 / 6,258, 0.00%):** missing, null or wrong-type fields; empty questions; control characters; mojibake; PTB quotes; empty SQL; non-SELECT or write keywords; unbalanced parentheses; truncated SQL; placeholder literals and `"null"` strings; `= NULL`; `!= ` on columns holding NULLs (104 rows checked); double-quoted literals resolving as columns; official parser failures; sqlglot failures; prepare failures; broken table or column references; exact duplicates; same question with different SQL in one DB; `tie_blocks` errors (1,246 recomputed from sort keys, all match).

| Group | Check | Train | Val | Total |
|---|---|---:|---:|---:|
| Questions | Under 4 words (all are valid short questions) | 4 | 0 | 4 / 6,258 (0.06%) |
| Questions | No terminal punctuation | 73 | 6 | 79 / 6,258 (1.26%) |
| Questions | Starts lowercase | 73 | 38 | 111 / 6,258 (1.77%) |
| Questions | Leftover tokenised spacing | 4 | 7 | 11 / 6,258 (0.18%) |
| Questions | Non-ASCII characters (0 mojibake) | 6 | 5 | 11 / 6,258 (0.18%) |
| Questions | Placeholder regex hits (all 5 read: false positives on 'value') | 5 | 0 | 5 / 6,258 (0.08%) |
| SQL text | Trailing semicolon (harmless) | 290 | 54 | 344 / 6,258 (5.50%) |
| SQL text | Literal with tokenised spacing (read: those checked match stored values) | 6 | 0 | 6 / 6,258 (0.10%) |
| SQL text | Literal with outer whitespace (05635, Tier A) | 1 | 0 | 1 / 6,258 (0.02%) |
| Grounding | `=` literal matches no stored value | 13 | 0 | 13 / 6,258 (0.21%) |
| Grounding | `=` literal matches only after case folding | 2 | 0 | 2 / 6,258 (0.03%) |
| Grounding | SQL literal absent from question (screen; many are valid, such as LA / Los Angeles) | 82 | 5 | 87 / 6,258 (1.39%) |
| Joins | JOIN with no own ON (screen; most are chained ONs on a later JOIN) | 24 | 2 | 26 / 6,258 (0.42%) |
| Joins | ON compares an alias with itself | 8 | 0 | 8 / 6,258 (0.13%) |
| Joins | Join between integer and text stored values | 136 | 5 | 141 / 6,258 (2.25%) |
| Grouping | GROUP BY with a bare column (screen) | 415 | 50 | 465 / 6,258 (7.43%) |
| Grouping | Aggregate + bare column, no GROUP BY | 9 | 0 | 9 / 6,258 (0.14%) |
| NULL | `NOT IN (subquery)`, NULL-sensitive construct | 167 | 26 | 193 / 6,258 (3.08%) |
| NULL | Expected output contains NULL | 33 | 4 | 37 / 6,258 (0.59%) |
| Types | Compare on a column mixing numbers and text | 12 | 0 | 12 / 6,258 (0.19%) |
| Types | Numeric stored values vs string literal | 18 | 0 | 18 / 6,258 (0.29%) |
| Types | Text stored values vs numeric literal | 52 | 8 | 60 / 6,258 (0.96%) |
| Dates | Subtraction on stored date columns | 6 | 0 | 6 / 6,258 (0.10%) |
| Dates | Range compare on date-like columns (includes integer-year false positives) | 22 | 10 | 32 / 6,258 (0.51%) |
| Construct | `LIKE` (case-insensitive in SQLite, matters for MQL) | 135 | 32 | 167 / 6,258 (2.67%) |
| Confirmation | Disconnected join graph (true cartesian) | 8 | 0 | 8 / 6,258 (0.13%) |
| Confirmation | GROUP BY bare column with several values in a group | 24 | 10 | 34 / 6,258 (0.54%) |
| Confirmation | Result changes with scan order | 2 | 0 | 2 / 6,258 (0.03%) |
| Confirmation | Nested LIMIT tie (6 change the answer, 4 do not) | 10 | 0 | 10 / 6,258 (0.16%) |
| Confirmation | Many-to-many join | 42 (train+val) | | 42 / 6,258 (0.67%) |
| Confirmation | `!=` literal matches no stored value | 5 | 0 | 5 / 6,258 (0.08%) |
| Confirmation | AND/OR precedence | 2 | 0 | 2 / 6,258 (0.03%) |
| Confirmation | Number in SQL differs from question (screen, read by hand) | 10 | 0 | 10 / 6,258 (0.16%) |

**Duplicates:** near-duplicate pairs (same DB, different query, token-sort score ≥ 92): 34, all with different results. The ones at score ≥ 95 were read. Most are legitimately different questions (grade 4 vs 5, AND vs OR, have vs have not). Two exceptions: 01802 and 01814 ask the same thing with swapped output columns (harmless), and 03782 says "distinct courses" while its SQL is `count(course_id)` with no DISTINCT (questionable, not tiered). The list is `records/near_duplicate_questions.jsonl`. 18 questions recur across databases, which is not a conflict. Paraphrase groups are intact: 0 groups span a database, have more than one normalised SQL, or have more than one result.

**Question edits:** 34 rows differ from `question_raw`, all whitespace only. The README's claim holds. All 8,570 training-side rows match spider-sql-v1 and raw Spider on SQL, question, DB and source index (0 differences).

**Hand review (not a gate).** I read a seeded sample of 80 train rows (seed 20261105), drawn only from rows outside pilot 300, V2 dev 500 and round-3 740: 3 clear errors (Wilson 95% 1.28–10.45%) and 2 questionable. One LLM reader, no second rater, and 2 of the 3 errors share one SQL query, so treat this as a rough indication only. The interval describes the pool before exclusions. It cannot estimate the rate after them, because its findings fed Tier A; the pilot's P2 review remains the real check. Results: `records/manual_review_results.jsonl`.

**Databases.** Local files match all 159 manifest hashes. `PRAGMA integrity_check` is ok on all 139 train/val databases. Weaknesses in the source data:
- Empty tables: `formula_1` (pitStops, lapTimes), `sakila_1` (film_text, language, staff, store).
- Foreign-key violations in 6 databases, 39,523 in total (`sakila_1` 38,273, `flight_4` 1,240, `hr_1` 6, `college_1` 2, `allergy_1` 1, `hospital_1` 1). 3 more (`baseball_1`, `dorm_1`, `soccer_1`) declare foreign keys SQLite cannot check.
- `bike_1.weather` has blank strings in numeric columns (54 in `mean_humidity`, 4 in `max_temperature_f`). In SQLite, blanks pass `> 50`.
- `tables.json` differs only for `soccer_1` and `store_1`, by the internal `sqlite_sequence` table.

## D. Leakage results

Method: ID sets and `db_id` / opaque `group` values only. Dev and test question, SQL and output fields were discarded at parse time, unread. `spider1_dev_NNNN` was mapped to `spider_dev_NNNN`; the mapping agrees on `db_id` for 455/455 and 279/279 items. `data/splits_expanded/test.jsonl` was never opened.

| Overlap with `train.jsonl` + `val_internal.jsonl` | Count |
|---|---:|
| Spider dev IDs | **0** |
| Spider test IDs (2,147) | **0** |
| 65 held-out MQL IDs | **0** |
| Train vs val IDs | **0** |
| Databases: train–val, train–dev, val–dev, train–test, val–test | **0 each** |
| Query groups: train–val, train–dev, val–dev | **0 each** |
| spider-sql-v1 rejects (incl. 40 question-leak rows) | **0** |

| Protected set **inside the package** | Size | In `dev.jsonl` | In `rejected.jsonl` | Total present |
|---|---:|---:|---:|---:|
| Official Spider dev | 1,034 | 899 | 135 | **1,034 (100%)** |
| **Held-out MQL test** | **65** | **64** | **1** (`spider_dev_0051`) | **65 (100%)** |
| Expanded validation | 72 | 72 | 0 | 72 (100%) |
| Expanded train | 318 | 316 | 2 | 318 (100%) |
| MQL-0006 evaluation set | 892 | 763 | 129 | 892 (100%) |
| Spider test | 2,147 | 0 | 0 | 0 |

The 65 IDs and their locations are in `records/heldout65_overlap_ids.jsonl`. The train side is clean; the package is not.

Limits of this check:
- ID overlap is weak by construction, because each split has its own ID prefix. The stronger evidence is the zero database overlap.
- Question-text and SQL-text overlap with dev or test was **not** re-measured, because that would mean reading protected fields. Train inherits spider-sql-v1's rule T3 (40 rows dropped; confirmed absent here) and the earlier V9 trigram screen.

## E. Execution verification

Everything ran on the official SQLite file for its own database, read-only and immutable, with a 30 s limit, on SQLite 3.40.1 (the package was built on 3.53.1). No database was substituted.

| Level | Kept train | Kept val | Rejected train/val | Dev (899 + 135) |
|---|---:|---:|---:|---:|
| **Full execution on a populated DB** | 5,514 | 744 | 568 | 0 |
| **Schema-only** (runs on an empty official DB) | 0 | 0 | 1,744 | 0 |
| **Parser-only** | 0 | 0 | 0 | 0 |
| **Unverified by this audit** | 0 | 0 | 0 | **1,034 (protected)** |

| Result for kept rows | Count |
|---|---:|
| Parser failures: official `process_sql` / sqlglot | 0 / 0 |
| Prepare failures (schema validity) | 0 |
| Execution errors / timeouts / unsupported | 0 / 0 / 0 |
| Recomputed hash = `result_sha256` | 6,258 / 6,258 (100.00%) |
| Recomputed hash = spider-sql-v1 hash | 6,258 / 6,258 (100.00%) |
| `expected_output` hashes to `result_sha256` | 6,258 / 6,258 (100.00%) |
| `n_rows`, `columns` correct | 6,258 / 6,258 (100.00%) |
| `spider1_kit.exec_match`-style 4-decimal compare equal | 6,256 / 6,258 (99.97%) |

- The 2 misses (02225, 02226) are floating-point noise of at most 8.5e-14 crossing a rounding boundary between SQLite versions. They are in Tier A for other reasons.
- Slowest query: 0.64 s.

**Rule replay** (rejected rows): R1 1,744/1,744, R4 177/177, R5 44/44, R6 38/38 and R7 244/244 confirmed. Every kept row passes R1 and R4–R6; R7 holds at top level (768 clean cuts, 0 ties, 16 with no cut). R3: all 4 match spider-sql-v1 on this SQLite version, which fits the README's explanation but does not prove it, as 3.53.1 was not run. R8 (59 rows) could not be replayed exactly without the script; those rows are rejected, so this does not affect the kept data.

**This does not prove semantic correctness.** Spider has no independent answer key. `result_sha256`, `expected_output` and the spider-sql-v1 hash all come from running the same SQL on the same database, so a match shows stability only. Every Tier A row below passed every execution check.

## F. Issues and severity

### Critical

**C1. The package contains protected evaluation data.** `dev.jsonl` has 899 Spider dev records with `question`, `sql` and `expected_output` fields, and `rejected.jsonl` has 135 more with `question` and `sql`. Together that is all 1,034 dev records, including gold SQL for all 65 held-out items. Loading the zip or `rejected.jsonl` without a split filter would leak them. The README documents `dev.jsonl`, so this was a design choice and not an accident, but it conflicts with this project's training-side requirement. IDs: every `spider_dev_*` row (DBs listed in `records/heldout65_overlap_ids.jsonl` for the 65). Exposure is shown; actual misuse is not.

### High

**H1. The cleaning cannot be reproduced.** `scripts/build_spider_pool.py` (`code_sha256` `45e1e28e50c56b91…`) is not in the archive or the repo. The README's "byte-identical, checked 3×" claim is unverifiable. No log or reject-rule code ships with it.

**H2. 57 confirmed label defects remain (29 queries, Tier A): 53 in `train.jsonl` and 4 in `val_internal.jsonl` (the `chinook_1` rows).** All pass every execution check. In each, the SQL does not do what the question asks; a few may still return the right rows on this particular data, but they would teach the wrong query.

| Defect | DB | `spider_train_` IDs |
|---|---|---|
| SQL literal 'Tony Award' contradicts the question's 'Bob Fosse' | `musical` | 00272, 00273 |
| AND/OR precedence: sex filter applies to Milk branch only | `allergy_1` | 00509, 00510 |
| wrong table: question asks for employee names, SQL returns the name of one of the customers (Peacock, Park, Johnson are the real employees) | `chinook_1` | 00869, 00870, 00871, 00872 |
| no GROUP BY for a per-account question: 1 row with arbitrary id, changes with scan order | `customers_and_invoices` | 01579, 01580 |
| SQL uses 2014, question says 2004 | `formula_1` | 02223, 02224, 02225, 02226 |
| fan-out inflation: team has 5 rows named Boston Red Stockings; answer is exactly 5x the join-free value | `baseball_1` | 03667, 03668, 03671, 03672, 03673, 03674, 03677, 03678, 03679, 03680, 03693, 03694, 03699, 03700 |
| join to team with no year: returns 'Philadelphia Blue Jays' for 2008, when team holds 'Philadelphia Phillies' for that year | `baseball_1` | 03669, 03670 |
| literal 'OTHA' misses stored ' OTHA': EXCEPT removes nothing | `student_1` | 04076, 04077 |
| name columns swapped and teacher firstname stored with leading space: filter vacuous | `student_1` | 04078, 04079 |
| literal 'Austin' misses stored 'austin': NOT IN removes nothing | `network_2` | 04480, 04481 |
| text date subtraction casts to leading year: elapsed time is not computed | `customers_and_addresses`, `department_store` | 04749, 04750, 06101, 06102, 06103, 06104 |
| literal 'Handful' misses padded stored value: != removes nothing | `products_gen_characteristics` | 05588, 05589 |
| literal 'Australia<TAB>' misses stored 'Australia': != removes nothing, answer 12 is Australia's | `railway` | 05635 |
| subquery projects outer T2.dormid: IN is vacuous, returns all 31 residents | `dorm_1` | 05756, 05757 |
| SQL uses 10000000, question says 8000000 | `city_record` | 06298, 06299 |
| tautological join (ON compares an alias with itself, so a true cartesian); 06795-06800 also return lname for 'first names' | `activity_1` | 06795, 06796, 06797, 06798, 06799, 06800, 06801, 06802 |

**H3. Adopting the pool changes frozen V2 sets.** ID membership only; no gate statistic was computed.

| Frozen set | Size | In pool train | Rejected by pool | In Tier A / B |
|---|---:|---:|---:|---:|
| Pilot | 300 | 276 | 24 | 3 / 1 |
| V2 dev | 500 | 459 | 41 | 3 / 5 |
| Round-3 G4 | 740 | 670 | 70 | 4 / 4 |

- G4 was measured on the 5,229-record spider-sql-v1 frame. The pool narrows that frame unevenly: the round-3 OL stratum loses 38/130 (29%), against 7/130 for J1. G4's result therefore does not carry over to the pool with the same weights.
- Pilot rows rejected by the pool: `records/frozen_pilot_rejected_ids.jsonl`. Pilot rows in Tier A/B: 01259, 02225, 06102, 06104.

**H4. `dev.jsonl` is not a substitute benchmark.** It drops 135 official dev rows, including 129 of the 892 MQL-0006 items, and adds 136 rows from the held-out databases. Scores on it are not comparable with MQL-0006. Filtering dev also conflicts with the rule that `dev_full` stays unfiltered.

### Medium

**M1. 37 rows depend on engine-specific or unspecified row choice (Tier B).** `apartment_rentals`: 01258, 01259; `architecture`: 06947; `baseball_1`: 03641, 03642, 03657, 03658, 03659, 03660; `bike_1`: 00155, 00156; `candidate_poll`: 02416, 02417; `department_store`: 04773, 04774; `document_management`: 04521, 04522; `e_government`: 06316, 06317; `hr_1`: 03515, 03516; `inn_1`: 02582, 02583; `insurance_fnol`: 00897, 00898; `manufactory_1`: 05350, 05351; `movie_1`: 02462, 02463, 02466, 02467, 02504, 02505, 02506, 02507; `store_1`: 00567, 00568.
- Causes: bare GROUP BY columns with several values, 6 nested LIMIT ties that change the answer, and aggregate-plus-bare-column queries.
- MongoDB has no equivalent row choice, so a converted label cannot be reliably checked against these answers. 7 of them (00567, 00568, 02462, 02463, 02466, 02467, 06947) use the aggregate form and are less risky, but are still non-standard SQL.
- Tier B was not read row by row against the question, so it may hold more wrong labels. Reading part of it moved 00869–00872 and 03669–03670 to Tier A.

**M2. Rule R7 appears to cover the top level only.** Nested `ORDER BY … LIMIT` ties survive in 10 kept rows: 6 change the answer (Tier B) and 4 do not (01517, 01518 in Tier C; 06103, 06104 in Tier A for date subtraction). 2 more (05756, 05757) could not be probed because the subquery is correlated; they are in Tier A.

**M3. 42 rows need a human decision (Tier C).** `baseball_1`: 03631, 03632, 03665, 03666, 03697, 03698; `bike_1`: 00109, 00110, 00137, 00138, 00153, 00154, 00157, 00158, 00195, 00196; `book_2`: 00233; `coffee_shop`: 00791; `college_2`: 01450, 01451; `cre_Theme_park`: 05913, 05914; `csu_1`: 02380, 02381; `formula_1`: 02207; `hospital_1`: 03932, 03933, 03934, 03935; `insurance_and_eClaims`: 01517, 01518, 01533, 01534; `movie_1`: 02488, 02489; `sakila_1`: 02944, 02945; `shop_membership`: 05410, 05411; `solvency_ii`: 04586; `tracking_grants_for_research`: 04362, 04363. Reasons per row are in `records/recommended_exclusions.jsonl`: many-to-many joins, blank strings in numeric columns, and literals that never match. In `baseball_1`, `team` has one row per year, so a join without a year can multiply rows; every count or sum on `Boston Red Stockings` tested this way was exactly 5× too large and is in Tier A (`records/baseball_team_fanout_confirmation.jsonl`). 03631, 03632, 03665 and 03666 stay here because their avg and max match the join-free answer. My join-free check for 03697 and 03698 was inconclusive, so they also stay here. Tier C screens were not all read row by row. For the literals, `Person` vs stored `Pearson` and `Harford` vs `Hartford` copy the question correctly, but that OR branch is never tested.

**M4. No databases or schemas ship with the package.** It relies on the local `spider_data.zip`. The hashes all match here, so the impact is low on this machine, but the package is not self-contained.

**M5. Unexplained external reference.** The README mentions "our 145-example eval set" built on Spider-DK copies of `concert_singer`, `orchestra` and `pets_1`. No project doc mentions it, and `pets_1` is a held-out database. Who holds that set, and what it was used for, needs asking.

**M6. Difficulty shift.** Kept share of the `exec_ok` population: easy 96.9%, medium 95.3%, hard 86.2%, extra 80.3%. The pool is somewhat easier than its source, so any model trained on it sees fewer hard queries.

### Low

- **R3 rejects are environment artefacts:** 00547, 00548 (`store_1`), 05935, 05936 (`cre_Theme_park`). They would pass on SQLite 3.40.1.
- **Cosmetic:** 344 trailing semicolons; a few questions with leftover spacing or no end punctuation (section C; `records/quality_flags.jsonl`).
- **Conversion cautions, not defects:** LIKE 167, `NOT IN` 193, integer-vs-text joins 141, text/number compares 78. SQLite coerces these silently and MongoDB does not, so expect converter rejections.

## G. Final recommendation

1. **Is `spider_pool.zip` genuinely cleaned?** Mechanically yes, semantically no. Its stated rules were really applied: I reproduced them from the data for every rule except R8. Those rules test whether a query runs and has a usable answer, not whether it answers the question, so 57 confirmed label defects remain. Spider's annotation noise is still present.
2. **Can we trust its provenance?** Partly. Inputs and record lineage are proven: all input hashes match, and every training-side row's SQL, `db_id` and `question_raw` match raw Spider byte for byte. The process is not, because the builder is missing.
3. **Is it safe as our SQL source training dataset?** `train.jsonl` is, after exclusions. The zip as a whole is not, because of C1.
4. **Which records should be excluded?**
   - Tier A: 57 rows, always.
   - Tier B: 37 rows (31 train, 6 val), recommended.
   - Tier C: 42 rows, after a human decision.
   - All dev rows, everywhere.
   - Result after A and B: train 5,430, val 734. List: `records/recommended_exclusions.jsonl`. Nothing has been removed; this is a recommendation.
5. **What limitations remain?**
   - Semantic errors outside the screens are not enumerated; the hand sample suggests roughly 1–10% before exclusions, with low precision.
   - Dev rows were not audited, by your restriction.
   - Text-level leakage against dev/test was not re-measured.
   - R8 and the builder are unverified, and R3's explanation was not tested on SQLite 3.53.1.
   - SQL labels are not MQL labels: this audit says nothing about conversion quality.
6. **Can we proceed to the V2 pilot using this dataset?** **Not yet, but the dataset is not the main blocker.** Needed first:
   - **Your decision on the pilot set:** keep the frozen 300 unchanged, with pool membership reported only, or pre-register an amended set. I recommend keeping the frozen 300. Its hash pins spider-sql-v1, its P1 criteria already reject tie cuts, and the 24 pool rejects and 4 Tier A/B rows can be reported as a slice. Silently moving to 276 would break the freeze.
   - **Prerequisites that already existed:** a verifier-3 pilot runner, a named independent human reviewer, and your go-ahead.
   - **Package handling:** a `train.jsonl`-only intake, with `dev.jsonl` unavailable to the pipeline.

## Audit conduct

- **Not opened:** `dev.jsonl` and dev-row content fields, `spider_sql_v1/dev_*` and `test_*` content fields, `dev.json` and `test.json` (hashed only), `dev_gold.sql`, `test_gold.sql`, `data/splits_expanded/test.jsonl`. The first three were parsed in memory by scripts that kept only `id`, `db_id`, `split`, `group`, `source` and `reject_reason`. `kaggle/nl2mql_spider_dev_1034_bundle_v2/data/eval_records.jsonl` (the MQL-0006 dev records) was parsed the same way, keeping only `id`. No dev SQL was executed.
- **Not run:** converter, verifier-3, pilot, G4, RAG, fine-tuning, MongoDB.
- **The 740 round-3 records:** they were in the full-population data scans, but no gate statistic was computed on them and they were excluded from the hand sample. Pilot and round-3 rows did appear among the flagged rows I read (see H3 for the counts). A later pilot reviewer should be told that these were seen.
- **Screens have blind spots.** A clean count means that screen found nothing, not that the problem cannot exist. Date handling and type coercion were screened by declared type and stored type, not judged row by row.
