# sql2mql-v2: instructions for the independent human review (pilot gate P2)

**Who reviews:** the mentor or another independent human. The converter and verifier were written by an AI assistant, so they cannot review their own output; no review decision in this package is pre-filled or generated.

**When:** only after Phase 2 (the 300-example pilot) has run. Phase 2 has not started; it waits for the Phase 1 gate (G4) to pass. The package generator and template are ready now. A **dry run** on 24 development items (not pilot items) is provided so the format can be checked in advance. Its judgements, if you fill any in, do not count toward any gate.

## What gets reviewed

The pilot gate P2 (frozen 2026-10-30 in `data/interim/finetune_v2/pilot_freeze.json`) reviews a stratified random sample of **min(120, accepted)** accepted pilot labels.

The selection rule is frozen in `scripts/finetune_v2/review_package.py` (`select`):

- accepted labels only;
- strata J2 (≥2 joins), OL (ORDER BY … LIMIT), J1 (one join) and G (the rest);
- proportional allocation by largest remainder, at least 1 per non-empty stratum;
- seed 20261101.

`review_package.py` sha256 at the time of writing (2026-10-31): `616903b80144f3cef97737bf1b6fe3406eae12dd8bd192aec0c24cd3365b23c2`. If the file changes before the pilot review, record the new hash and the reason here.

| Command | Output |
|---|---|
| `python -B scripts/finetune_v2/review_package.py pilot --records outputs/sql2mql_v2/pilot/records.jsonl` | `outputs/sql2mql_v2/review_pilot/` (after Phase 2) |
| `python -B scripts/finetune_v2/review_package.py dryrun` | `outputs/sql2mql_v2/review_dryrun/` (exists now; DRY RUN) |

## Files in a package

| File | Use |
|---|---|
| `review_packet.md` | One section per item: question, original SQL, converted MQL, `CREATE TABLE` text and row counts of the referenced tables, SQLite result of the verbatim SQL, MongoDB result of the MQL (first 15 rows + row count), converter decision, verifier decision |
| `review_sheet.csv` | The same items, one row each, plus **empty** reviewer columns. Fill this file. It opens in Excel or Google Sheets (UTF-8 with BOM) |
| `review_items.jsonl` | Machine-readable copy of the packet |
| `package_manifest.json` | Selection rule, per-stratum counts, file hashes, `judgements_filled: 0` |

MongoDB results come from the local MongoDB 8.0.34 import of the Spider-train SQLite files, with NULL stored as explicit `null`. A value shown as `<absent>` means the output document had no such field.

## How to judge an item

Answer three questions separately, so that a mistake in Spider's gold SQL is not counted as a converter error:

1. **`nl_sql_agree`:** does the original SQL answer the natural-language question? (Y / N / U for unsure)
2. **`sql_mql_equivalent`:** does the MQL return the same result as the SQL on **every** database with this schema, not just the rows shown? (Y / N / U)
   - SQLite semantics are the reference: integer division truncates, `LIKE` is ASCII-case-insensitive, comparisons with NULL are not true, an `INNER JOIN` drops unmatched rows.
   - Column order matters. Row order matters only under `ORDER BY`, and rows that tie on the ORDER BY key may come in any order.
3. **`mql_answers_question`:** would the MQL be an acceptable answer to the question? (Y / N / U)

Then fill in:

- **`discrepancy_type`:** one of `none`, `wrong_filter`, `wrong_join`, `missing_or_extra_rows`, `null_handling`, `aggregation`, `grouping`, `ordering_or_limit`, `distinct`, `projection_or_columns`, `type_or_case_semantics`, `gold_sql_wrong_for_question`, `other`.
- **`semantic_discrepancy`:** a concrete description: what input would make SQL and MQL differ, and how.
- **`comments`**, **`confidence`** (1 = guess, 5 = certain), **`minutes_spent`**, and **`reviewer_id`** (initials).

Matching result previews are **not** proof of equivalence. Many differences appear only on other data, such as NULLs, ties, empty tables or unmatched join rows. Please judge from the query text.

## How the results are used (frozen, P2/P3)

- A label is **wrong** if `sql_mql_equivalent = N`.
- P2 passes if the Wilson 95% upper bound of wrong / reviewed is ≤ 0.05.
- P3 fails if any error pattern (`discrepancy_type`) occurs ≥ 2 times.
- `U` answers are reported separately and resolved by discussion before the gate is computed. They are never silently counted as correct.
- `nl_sql_agree = N` with `sql_mql_equivalent = Y` is a Spider gold-SQL issue. It is recorded but is not a converter error.
