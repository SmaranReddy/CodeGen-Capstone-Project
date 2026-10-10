# Spider 2.0-Lite → MongoDB/MQL Conversion Report

This report summarises the batch conversion of the **local** Spider 2.0-Lite examples
(135 SQLite instances) into validated NL → MongoDB (MQL) examples, executed against the
Atlas database `codegen_capstone_pilot`.

## 1. Scope and result

| Status | Count |
| --- | ---: |
| `existing_validated` (prototype round, reused) | 7 |
| `passed` (newly converted, matched an official gold CSV) | 61 |
| `deferred` | 62 |
| `blocked` | 3 |
| `not_attempted` | 2 |
| **Total** | **135** |

Validated dataset: **66 rows** written to `data/processed/nl_mql_dataset.jsonl`
(7 pilots + 59 new). "_Validated_" means the hand-written MQL was run on Atlas and its
result passed the **official** Spider comparator against at least one gold CSV.

## 2. Method

Per example (`scripts/conversions/conv_<id>.py`):

1. Read the manifest record (`data/interim/spider2_local_manifest.jsonl`): source DB,
   question, gold CSV paths, and `gold_eval_config` (`condition_cols`, `ignore_order`).
2. Author a labelled **reference SQL** for the question and verify it reproduces a gold
   CSV with the official comparator (`score`). This separates "I understood the question"
   from "the MQL is correct".
3. Import only the needed columns of the needed tables into `<id>__<table>` collections
   (`import_table`, tracking every collection in `_atlas_collections.json`).
4. Write the MQL by hand and run it on Atlas; score the result against every gold CSV.
5. Persist the artifact (`data/interim/conversions/<id>.json`), status, and rebuild the
   dataset (`scripts/run_conversions.py`).

The comparator is the **verbatim** official `compare_pandas_table` from
`data/raw/spider2/evaluation_suite/evaluate.py`: `condition_cols` selects the gold columns
compared, column names/order are ignored, extra prediction columns are allowed,
`ignore_order` sorts each column independently, NULL is treated as 0, numbers match within
0.01, and the predicted column length must equal the gold row length.

## 3. Newly converted examples (61)

Multi-table joins, windows and set logic were the bulk of the work; representative
mappings:

| id | DB | Semantics (MQL pattern) |
| --- | --- | --- |
| local004 | EU_soccer | home/away reliability ratio (branch + divide) |
| local008 | EU_soccer | max over grouped branches |
| local017 | EU_soccer | per-season winner aggregation |
| local018 | California_traffic | pcf share 2011 − 2021 (windows for yearly totals) |
| local019 | WWE | shortest NXT match, `MM:SS` → seconds, argmin |
| local021 | IPL | over-run composite-key join + >50 strikers average |
| local024 | IPL | top-5 country batting average |
| local025 | IPL | per-match highest over + across-match average |
| local026 | IPL | global top-3 match-max overs |
| local028 | chinook | month × year pivot |
| local030 | chinook | 5 lowest-city averages |
| local031 | chinook | peak month in weakest year |
| local034 | chinook | avg per-category max payment |
| local037 | chinook | top-3 category payments |
| local049 | modern_data | avg unicorns/year (gold `_a`) |
| local054 | chinook | best artist under $1 |
| local055 | chinook | abs avg diff (gold `_b`) |
| local056 | sakila | mean absolute month-over-month change |
| local064 | bank | month-end balances, positive-customer argmax/argmin |
| local065 | sakila | per-customer top spend |
| local068 | city_legislation | running totals + YoY growth (windows + `$shift`) |
| local077 | bank | monthly top interest, rolling avg + lags |
| local078 | sakila | category spend share |
| local081 | northwind | spending groups |
| local085 | northwind | late-order % |
| local096 | Db-IMDB | exclusively-female % by year (`$trim` on `M_Cast.PID`) |
| local097 | Db-IMDB | 10-year window |
| local099 | sakila | country rental counts |
| local128 | BowlingLeague | teams winning at all venues |
| local133 | sakila | weighted style score + abs diff |
| local141 | AdventureWorks | annual sales − quota per salesperson (inner join, gold `_b`) |
| local152 | IMDB | per-director rating stats + mean inter-movie gap (half-up rounding) |
| local156 | bank_sales_trading | per-region avg BTC buy price, rank + YoY % (gold `_a`) |
| local193 | sakila | 7/30-day LTV shares |
| local195 | sakila | top-5 actors' rental percentage (gold `_a`) |
| local196 | sakila | spend/subsequent rentals by first-rental rating |
| local198 | chinook | median 249.53 |
| local199 | chinook | per-customer latest invoice |
| local202 | city_legislation | friendly vs hostile states |
| local218 | EU_soccer | median highest season goals |
| local221 | EU_soccer | top-10 wins |
| local230 | imdb | top-4 directors in top-3 genres |
| local283 | EU_soccer | explode matches → 3-1-0 champion per season/league |
| local298 | bank | signed amounts, cumulative pre-month balance |
| local301 | sakila | multi-year aggregation |
| local302 | bank | windowed before/after sales, worst attribute type |
| local310 | f1 | 3-year smallest driver+constructor points |
| local311 | f1 | constructor + best-driver combined points (gold `_a`) |
| local329 | log | registration count |
| local330 | log | landing/exit (gold `_c`) |
| local335 | f1 | constructors with fewest point-scoring drivers' seasons |

Wave 5 (larger DBs / `big_table`) added eight more:

| id | DB | Semantics (MQL pattern) |
| --- | --- | --- |
| local058 | education_business | unique products per segment (big fact table, fiscal_year) |
| local059 | education_business | per-division top-3 avg quantity (calendar year) |
| local114 | education_business | web sales report via multi-hop `$lookup` (gold `_d`) |
| local062 | complex_oracle | Italian customers, 10 profit buckets (profits-view replication) |
| local285 | bank_sales_trading | per-category-year veg wholesale financials (gold `_a`) |
| local201 | modern_data | 4–5 char 'r' words with anagrams, top-10 (gold `_a`) |
| local209 | delivery_center | top store's delivered-order ratio (gold `_a`) |
| local212 | delivery_center | top-5 drivers by average daily deliveries (gold `_c`) |

The reconciliation batch (wave 6) resolved two previously-deferred `special_transform`
catalogue entries:

| id | DB | Semantics (MQL pattern) |
| --- | --- | --- |
| local023 | IPL | season-5 per-match runs (`$lookup` on composite key) → top-5 by avg runs + batting average (`$lookup` dismissals, `$divide`/`$max`) |
| local297 | bank_sales_trading | monthly net → cumulative closing (`$setWindowFields` sum) → latest-month growth (`$shift`/`$documentNumber`) → percentage |

The 7 `existing_validated` pilots (`local022, local038, local039, local041, local075,
local163, local197`) were carried over unchanged.

## 4. Non-validated examples

- **deferred (62)** — not converted in this batch:
  - `needs_mentor` (29): underspecified question or differently-shaped gold outputs.
  - `difficult_ambiguous` (22): procedural/underspecified; deferred to a later wave.
  - `ext_knowledge` (9): depend on an external-knowledge document (out of scope).
  - `view_dependent` (2): depend on a SQL view absent from the DDL (materialisation rule).
- **blocked (3)** — `local003, local066, local309`: the gold CSV is not reproducible from
  the shipped gold SQL at value level (`not_faithful`).
- **not_attempted (2)** — remaining catalogue entries, both `special_transform`, pending a
  mentor decision on their gold:
  - `local168` — PARTIAL: of the 5 golds only gold `_d` (`106900.0`) is reproducible
    (top-3 named skills `python, sas, javascript` over `job_title_short='Data Analyst'`,
    non-null `salary_year_avg`, `job_work_from_home=1`). The shipped `skills_dim` is
    truncated (26 rows) and 311,795 `skills_job_dim` rows have no skill name, so variants
    a/b/c/e cannot be reproduced. Not added to the dataset (≥1-gold-match policy not forced).
  - `local331` — UNRESOLVED: a 16-variant session-sequencing search matched none of the 3
    golds; `activity_log` contains 7 duplicated session streams (rowid offset 32) that make
    ordering ambiguous. No reference procedure is known.
  - (`local023` and `local297`, previously listed here, were resolved — see §4.1.)

### 4.1 Reconciliation findings (local023, local297)

- **local023** — the shipped gold SQL does **not** reproduce `local023_a.csv` at value
  level. A reconstructed reference (runs `= SUM(runs)` from `ball_by_ball ⋈ batsman_scored`
  KEY `(match_id, over_id, ball_id, innings_no)`; dismissals from `wicket_taken`; both
  filtered to `season_id=5`; `avg_runs_per_match = SUM(runs)/COUNT(match_id)`,
  `batting_average = SUM(runs)/dismissals`) reproduces the gold exactly. Converted and
  scored = 1.
- **local297 — ambiguous gold.** This single question has **two** official accepted answers:
  - `local297_a.csv = 44.869215` (223/497): group **all** transactions, purchases contribute 0;
  - `local297_b.csv = 53.002070` (256/483): restrict to `txn_type IN ('deposit','withdrawal')`
    **before** grouping, so purchase-only months drop out.
  The prompt ("group all deposits and withdrawals") reads as variant B, which we implement;
  both variants reproduce. The ambiguity is retained here and in the artifact
  (`alternate_interpretation`); the eval harness passes either way (it accepts a match
  against any gold).

## 5. Atlas operations

- Engine 8.0 (M0 free tier, 512 MB limit measured as **data + index size**).
- Collection naming `<id>__<table>`; every created collection is registered in
  `data/interim/conversions/_atlas_collections.json` so cleanup only ever drops our own
  data.
- Free tier is tight: after the batch the pilot DB held ~104 collections. Collections of
  already-validated examples were dropped to reclaim space and are re-importable on demand.
- Operational notes discovered on M0:
  - `allowDiskUse` does **not** relieve the 32 MB blocking-operator limit; avoid global
    `$sort`/`$group` over large collections. Prefer `$top`/`$first` accumulators, indexed
    `$lookup`s, and pre-filtering before sorting.
  - Composite ball keys `match_id|over_id|ball_id|innings_no` keep `$lookup`s indexable.
  - Some source columns are TEXT where a joined key is INTEGER (`Matches.title_id` vs
    `Belts.id`); normalise with `$toString` / typed literal lists.

## 6. Reproduce

```powershell
python scripts/seed_status.py                 # initialise status for all 135
python scripts/run_conversions.py <id> [<id>...]   # convert specific examples
python scripts/run_conversions.py --all       # convert every module present
```

`run_conversions.py` writes `data/interim/conversions/<id>.json`, updates
`data/interim/conversion_status.jsonl`, and rebuilds `data/processed/nl_mql_dataset.jsonl`.

## 7. Key artifacts

- `scripts/conversion_kit.py` — read-only SQLite access, official comparator, Atlas import
  and registry, artifact/status/dataset writers.
- `scripts/run_conversions.py` — batch driver.
- `scripts/feasibility_inventory.py` — reproducible 547-record feasibility inventory +
  summary (`data/interim/feasibility_inventory.jsonl`, `feasibility_summary.json`).
- `scripts/conversions/conv_<id>.py` — 61 per-example conversions (the 7 pilots live in
  `scripts/prototypes/`).
- `data/interim/conversions/<id>.json` — per-example artifact (reference SQL, MQL,
  result, validation).
- `data/interim/conversion_status.jsonl` — consolidated status for all 135.
- `data/processed/nl_mql_dataset.jsonl` — 68 validated NL → MQL examples.

## 8. Feasibility inventory of the full pool (547)

`scripts/feasibility_inventory.py` covers the **entire** official Spider 2.0-Lite pool
(`spider2-lite-current.jsonl`, 547 records, commit `cafb8673…`) and emits
`data/interim/feasibility_inventory.jsonl` + `feasibility_summary.json`. Each row records
source provenance, backend, required tables, availability of data/schema/gold SQL/gold CSV/
eval config/external-knowledge doc, whether a reference SQL is verifiable, MQL complexity,
source DB size + an estimated Atlas footprint, and a verification-based eligibility class.

Backend split: **sqlite 135 / bigquery 205 / snowflake 207** — the 412 cloud records ship
**no executable data** (schema metadata only, no BigQuery/Snowflake credentials), so they
cannot be locally converted or verified.

| Eligibility class | Count |
| --- | ---: |
| `already_validated` | 68 |
| `deferred_decision` (need a mentor call) | 62 |
| `partial` (local168) | 1 |
| `blocked` (local003/066/309 + local331) | 4 |
| `unavailable` (cloud) | 412 |

**200-target verdict: not achievable from Spider 2.0-Lite alone.** The local pool is capped
at 135 records (68 already validated; 67 more only if every deferred/partial/blocked decision
is resolved); the other 412 require cloud credentials. Reaching 200 therefore needs either
cloud access *or* mentor approval to draw on additional sources.
