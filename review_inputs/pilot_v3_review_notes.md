# AI-assisted technical review of the 120-row Verifier-3 pilot review sheet

## Scope and limitation

Reviewed all 120 rows in the uploaded `review_sheet.csv`. This is an AI-assisted static technical review, **not an independent human sign-off**. The supplied CSV contains the question, SQL, generated MQL and result previews, but not the database files, full schemas/constraints, or complete result sets. I therefore did not invent a human reviewer identity or time-spent values, and marked uncertainty where the missing context matters.

## Automated integrity checks

- 120/120 MQL payloads parse as JSON.
- 120/120 have a collection name prefixed by the row's `db_id`.
- All 30 SQL `ORDER BY` cases inspected have matching MQL sort keys and directions.
- The displayed SQL/MongoDB preview rows agree as unordered row values in 115/120 rows. The five exceptions are explained by output ordering where SQL does not define a tie/order (R009, R023, R105, R108, R113); the previews are truncated for these, so they do not prove full-result equality.

## Row-level decisions

### Definite natural-language / SQL mismatch

- **R119 — `spider_train_03948` (`hospital_1`)**: The question asks for the most recent treatment, but SQL uses `ORDER BY dateundergoes LIMIT 1` without `DESC`. Ascending is the default, so it selects the earliest date. MQL uses `$sort: {DateUndergoes: 1}`, faithfully reproducing the SQL error. Decision: NL↔SQL = N; SQL↔MQL = Y; MQL answers question = N. The source label should be quarantined for human adjudication, not silently repaired.

### Incomplete / underspecified question

- **R012 — `spider_train_04685` (`college_3`)**: The question text ends “course that has course name” and never supplies a course name; SQL filters `CName = "COMPUTER LITERACY"`. MQL mirrors SQL. Decision: NL↔SQL = U; SQL↔MQL = Y; MQL answers question = U. Check the authoritative original question before accepting this example.

### Schema/cardinality-dependent uncertainty

- **R090 — `spider_train_00515` (`allergy_1`)**: SQL/MQL group `Has_allergy` by allergy and count rows. The question asks for number of students affected. This is equivalent only if the table's row grain guarantees at most one row per student/allergy (or one row per affected student per allergy). The schema/uniqueness constraints were not supplied. Decision: NL↔SQL = U; SQL↔MQL = Y; MQL answers question = U.

- **R106 — `spider_train_04532` (`document_management`)**: SQL compares `user_login` to numeric `1`; MQL compares to string `"1"`. Both supplied previews return 7. This is likely consistent with text affinity/string storage, but the field type is not included in the sheet. Kept Y based on matching preview and conversion intent, with confidence 3 and a type-mapping caveat.

### Preview ordering notes, not identified semantic defects

- **R009**: DISTINCT output has no ORDER BY; differing preview order is immaterial.
- **R023, R108, R113**: GROUP BY outputs have no ORDER BY; differing group order is immaterial.
- **R105**: SQL and MQL sort only by Year; row order within equal-year ties is unspecified. Different preview ordering inside ties is not by itself a mismatch.

- **R116**: Question spells “Interanation”, while SQL/MQL search for “International”. Treated as an obvious typo; confirm if literal wording was intended.

## Summary of decisions

| Field | Y | N | U |
|---|---:|---:|---:|
| NL–SQL agreement | 117 | 1 | 2 |
| SQL–MQL equivalence | 120 | 0 | 0 |
| MQL answers question | 117 | 1 | 2 |

## Important interpretation

The SQL–MQL Y decisions are based on static translation inspection and supplied execution previews. They are not a substitute for re-running all 120 full queries against the authoritative SQLite and MongoDB databases and comparing complete results, especially where previews are truncated. This review sheet must not be represented as completed independent human P2/P3 sign-off. An independent human reviewer should adjudicate R012, R090 and R119, and check the full database evidence before final corpus promotion.
