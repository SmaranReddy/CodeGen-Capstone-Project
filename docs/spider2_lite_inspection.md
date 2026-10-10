# Spider 2.0-Lite Inspection Report

Scope: what is in our local `data/raw/spider2/spider2-lite.jsonl`, and what the official repository provides around it.
No raw data was modified; no conversion, inference, or schema/taxonomy decisions were made.

**Evidence labels**
- **[LOCAL]**: computed from our local file.
- **[OFFICIAL]**: verified against `github.com/xlang-ai/Spider2` (`main`, repo last pushed 2026-08-12; checked 2026-10-08) via the GitHub API / raw files. Nothing was downloaded into the project; official files were only read into a scratchpad.
- **[UNVERIFIED]**: could not be checked.

Official base path: <https://github.com/xlang-ai/Spider2/tree/main/spider2-lite> (MIT license).

## 1. Local file vs. official dataset

| Property | Local | Current official (`main`) |
|---|---|---|
| Records | 260 | **547** |
| SHA-256 | `595eaf4c412cf81ac66a88fa23f853a66ada71cdd03ca350ab53bbc4f34b67ac` | `4ba48916576fbd60311a2478c6d4550b5d8cf3fcbc512457ea493b5941ca009d` |
| Fields | `instance_id, db, question, external_knowledge` (+ optional `temporal` in 15) | same fields (+ `temporal` in 37) |
| Prefix counts (`bq`/`sf_bq`/`local`/`ga`/`sf`) | 89 / 85 / 64 / 14 / 8 | 180 / 189 / **135** / 25 / 18 |
| `external_knowledge` non-null | 39 | 107 |

**Why 260 instead of 547: verified.** Our file is **byte-identical** (same SHA-256) to `spider2-lite/spider2-lite.jsonl` at official commit **`f8be2c664d6ebefc3da0af4d5e495003ef9c377f`** ("update spider2lite", 2024-11-10), the *first* version, which had 260 examples. The next commit `ef559e1c` (2024-12-12, "upload spider2 full examples and answer") expanded it to 547 records. The file has since been revised ~28 times (30 commits touch it; latest content change 2025-08-05, `b92f19e7`). **Our file is the stale, pre-release early version, not a random subset.**

Comparison against current `main` [OFFICIAL vs LOCAL]:
- 254 of our 260 `instance_id`s still exist; **6 were removed** (`bq015, bq078, bq350, bq379, bq451, bq452`) (these ids first appear missing in commit `ab2b4519`, 2025-05-22; not investigated further).
- Of the 254 shared ids, only some records are unchanged. Differences: `question` text changed for 136 ids over the whole 260 (rewritten/clarified), `external_knowledge` changed for 16, `db` changed for 7. For the 64 `local*` ids we hold, **29 are identical to current, 30 have a changed question** (the rest are consistent).
- `db` differences: `local056/194/195/199` (`SQLITE_SAKILA` to `sqlite-sakila`), `local099` (`DB_IMDB` to `Db-IMDB`), `sf_bq295` (`GITHUB_REPOS` to `GITHUB_REPOS_DATE`), `sf_bq358` (`NOAA_DATA` to `NEW_YORK_CITIBIKE_1`).
- Some current official `db` values are full cloud paths in the README example (`bigquery-public-data.ga4_obfuscated_sample_ecommerce`); the `main` JSONL I fetched uses short names (e.g. `ga4`) for the first rows. Not checked exhaustively.

**Conclusion:** the gold SQL and gold execution results in `main` correspond to the *current* questions. Pairing them with our *old* questions would be silently wrong for ~half of records. **We should replace our JSONL with the current official file** (as a new file/version in `data/raw/`, keeping the old one for provenance; do not overwrite in place).

## 2. Official `spider2-lite/` layout [OFFICIAL]

```
spider2-lite/
  README.md
  spider2-lite.jsonl               # 547 examples (README calls it spider2-lite.json; the actual file is .jsonl)
  resource/
    databases/
      sqlite/<db>/                 # 30 dirs, 462 files (~1.0 MB total): DDL.csv + <table>.json sample files
      bigquery/                    # 5,651 files, ~462 MB  (schema JSON/metadata, NOT queryable data)
      snowflake/                   # 2,447 files, ~171 MB  (same)
    documents/                     # ~70 .md external-knowledge docs (referenced by external_knowledge)
    syntax/                        # BigQuery docs and function refs (not needed for us)
    interface/                     # sqlite.py, bigquery.py, snowflake.py helper scripts
  evaluation_suite/
    evaluate.py, evaluate_utils.py, README.md, *_credential.json
    gold/spider2lite_eval.jsonl    # 547 rows: instance_id, condition_cols, ignore_order, toks (+ temporal)
    gold/sql/                      # 256 gold SQL files (NOT all 547)
    gold/exec_result/              # 1,544 CSV files (~259 MB), covering all 547 ids (`<id>_a.csv`, `_b.csv`...)
  baselines/                       # DAIL-SQL, DIN-SQL, CodeS code (not needed)
```

### Resource map

| Need | Official resource | Mapped by |
|---|---|---|
| Schema / tables | `resource/databases/sqlite/<db>/DDL.csv` (columns `table_name, DDL` = `CREATE TABLE` text) and `<table>.json` (keys `table_name, table_fullname, column_names, column_types, description, sample_rows`) | directory name = `db` |
| Actual DB contents (local) | **Not in the git repo** (no `.sqlite`/`.db` file exists anywhere in it). README: a Google Drive zip of `.sqlite` files, <https://drive.usercontent.google.com/download?id=1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG&export=download&authuser=0>. | `<db>.sqlite` filename presumably matches `db` [UNVERIFIED] |
| Gold SQL | `evaluation_suite/gold/sql/<instance_id>.sql` | `instance_id` |
| Gold results | `evaluation_suite/gold/exec_result/<instance_id>[_a/_b/...].csv` | `instance_id` (suffixes = alternative acceptable answers) |
| Comparison config | `evaluation_suite/gold/spider2lite_eval.jsonl` | `instance_id` |
| External knowledge | `resource/documents/<external_knowledge value>` | filename in `external_knowledge` |

There is no separate index file mapping `instance_id` to resources; the mapping is by naming convention (`db` to folder, `instance_id` to gold filename, `external_knowledge` to document filename).

The SQLite `json` files contain **only sample rows** (small, ~1 MB total across all 30 dbs), not the full tables. Full data is only in the Drive zip. The size of that zip could not be determined (Drive returns an HTML interstitial) [UNVERIFIED].

## 3. What is usable locally without paid cloud access

| Subset | Questions | Schema | Data | Gold SQL | Gold results | Verdict |
|---|---|---|---|---|---|---|
| `local*` (SQLite) | **135** across **30 dbs** | Yes (DDL.csv, free, in git) | Yes via Drive zip (free, unverified size/contents) | **Only 24** of 135 | **All 135** | **Usable.** |
| `bq*`, `ga*` (BigQuery) | 205 | Metadata JSON in git | Requires a BigQuery account/credentials (Google Cloud; the repo's own guideline asks you to sign up). Free tier limits not verified | 134+14 | Yes | **Not usable without cloud access**; skip. |
| `sf_bq*`, `sf*` (Snowflake) | 207 | Metadata JSON in git | Requires a Spider2-granted Snowflake account (Google Form request, per README) | 76+8 | Yes | **Not usable**; skip. |

Total with gold SQL: 256 of 547 (the rest have results only).

### `local*` details
- 135 current local questions; 13 have `external_knowledge`, 0 have `temporal`.
- Gold SQL exists only for: `local003, 004, 008, 017, 019, 022, 023, 029, 038, 039, 058, 065, 066, 075, 078, 099, 131, 163, 197, 199, 210, 219, 301, 309` (24). Of those, 16 are in our old file.
- Gold CSVs for all 135 local ids: 328 files, **~0.95 MB total**, trivially small.
- Local dbs and question counts (current): bank_sales_trading 15, IPL 11, city_legislation 10, f1 9, Brazilian_E_Commerce 8, oracle_sql 8, modern_data 7, sqlite-sakila 7, complex_oracle 6, education_business 5, Db-IMDB 5, EU_soccer 5, log 5, then 17 dbs with 1 to 3 each (E_commerce, California_Traffic_Collision, chinook, EntertainmentAgency, delivery_center, stacking, Baseball, Airlines, Pagila, northwind, imdb_movies, WWE, BowlingLeague, school_scheduling, AdventureWorks, music, electronic_sales).
- **Naming inconsistency in the official repo:** current JSONL uses `sqlite-sakila` and `Db-IMDB`, but the resource directories are `SQLITE_SAKILA` and `DB_IMDB`. Needs a manual mapping.
- Gold results are SQL outputs that depend on the **SQLite data**, so they can only validate an MQL answer if our MongoDB data is loaded from the same SQLite data.
- Local-example SQL is SQLite dialect (e.g. window functions `NTILE() OVER`, as in `local003`), so many questions are analytics-style rather than simple CRUD.

## 4. Fields (all verified against `main`)

- `instance_id`: unique id; prefix correlates with backend (`bq`/`ga` BigQuery, `sf`/`sf_bq` Snowflake, `local` SQLite) per README folder structure and the data resource used. 
- `db`: database id; for local, it names a folder in `resource/databases/sqlite/`.
- `question`: NL question.
- `external_knowledge`: filename in `resource/documents/` (README: "filenames of external knowledge, documentation, and information required to answer this question") or null.
- `temporal`: optional `"Yes"` (37 in `main`). **Not defined in the README**; meaning remains unverified.
- `gold/spider2lite_eval.jsonl` fields `condition_cols`, `ignore_order`, `toks`: evaluator configuration for result comparison (column subset to compare, order sensitivity, a token-count-like value). Semantics read from names only; confirm in `evaluate.py` before relying on them [UNVERIFIED].

## 5. Earlier local-only findings that still hold [LOCAL]
UTF-8 no BOM, LF, trailing newline; 261 physical lines, 1 blank; no duplicate ids; 114 distinct `db`; 9 questions with non-ASCII characters (read/write with explicit UTF-8, Windows cp1252 consoles fail).

## 6. Implications for NL to MongoDB/MQL
1. Realistic validated pool is the **`local*` subset: 135 questions, 30 dbs**, not 547 and not our 260. Reaching 200-300 examples from Spider's local questions alone is **not possible** (max 135 before any MQL-feasibility filtering). The target needs to be reconsidered, or examples need to come from additional sources (several questions per db; derived/variant questions; or non-Spider data), which is a mentor decision.
2. Cloud-backed questions (~75%) can't be executed by us, so they are only usable as NL-text inspiration, without validation, if at all.
3. For validation by execution, ground truth = gold CSVs (all 135) and/or running gold SQL on the SQLite data (only 24 have SQL). 
4. Using the Drive SQLite files, one can export tables into MongoDB; then equivalence to gold CSVs is checkable. Schema design for Mongo (flat collections vs embedded) remains a decision for later.
5. Many questions need window functions, regression, or geo math (e.g. `local002` regression, haversine docs); MQL feasibility varies and is not assumed.
6. Use the **current** `question` text, never the stale ones in our file.

## 7. Open questions for the mentor
1. Is the 200-300 example target meant to be Spider-derived only? The executable local pool is 135 questions (30 dbs), fewer after MQL-feasibility filtering. Are new/derived questions allowed?
2. Is it acceptable to spend time on cloud BigQuery/Snowflake access (BigQuery signup; Snowflake needs an access request), or do we restrict to SQLite?
3. Validation criterion: result equality with the SQL gold CSV, execution-only, or manual review? How to handle questions with multiple acceptable CSVs (`_a`, `_b`)?
4. "Atlassian database" in the brief: I assumed MongoDB Atlas (`.env.example` uses a MongoDB URI). Confirm (including whether Atlas free tier/512 MB limits matter; local SQLite data is small but sizes unverified).
5. Meaning/handling of `temporal: "Yes"`.
6. Treatment of `external_knowledge` documents: include them as model input, or restrict to questions without them (122 of 135 local questions have none)?
7. Doc inconsistencies in our own repo: `docs/dataset_task_definition.md` references nonexistent `conversion_notes.md` and `validation_notes.md`, and uses `id`/`db_id` where Spider uses `instance_id`/`db`.

## 8. Downloaded resources and manifest (added after step 3)

Pinned source: `xlang-ai/Spider2` commit `cafb867313aab4e674652054198f383cf4018943` (also in `data/raw/spider2/SOURCE_COMMIT.txt`). Fetched by `scripts/download_spider2_local.py`; manifest built by `scripts/build_local_manifest.py` into `data/interim/spider2_local_manifest.jsonl` (135 rows).

- `data/raw/spider2/spider2-lite-current.jsonl`: SHA-256 `4ba48916...ca009d`, 547 rows. The original 260-row `spider2-lite.jsonl` is unchanged (`595eaf4c...`).
- `data/raw/spider2/resource/databases/spider2-localdb/`: 30 `.sqlite` files + `local-map.jsonl` (instance_id to db, matches all 135 manifest rows). From the official Google Drive zip `local_sqlite.zip` (435 MB, 1.68 GB extracted; Drive could not virus-scan it, zip integrity test passed, contains only `.sqlite`/`.jsonl`).
- `resource/databases/sqlite/<db>/` (DDL.csv + sample-row JSON), `evaluation_suite/gold/{sql,exec_result}` for local ids, `resource/documents/` (the 13 referenced), `evaluation_suite/{gold/spider2lite_eval.jsonl,evaluate.py,evaluate_utils.py}`.
- Nothing from BigQuery/Snowflake/baselines/credentials was fetched.

Checks run: all 135 manifest rows have an openable SQLite file, DDL, at least one gold CSV, a gold eval config, and (if referenced) the document. Gold SQL replayed read-only on SQLite for the 24 that have it: 22 returned the same row count as the gold CSV; `local003` hit my 60 s cap (not completed); `local131` (20 rows vs gold 25) and `local309` (75 vs 74/74) differ in row count, cause not investigated (could be CSV parsing of multi-line cells). Row counts only; values not compared. DDL.csv omits some objects present in SQLite (views `customer_list`, `film_list`, `match_view`, etc.; `stacking` extra tables).

## 9. Evaluator logic, gold-replay anomalies, schema audit, environment (verified)

Supersedes the row-count claims at the end of section 8, which were only a weak check. Everything below was run against the pinned commit `cafb8673…`.

### 9.1 How gold results are compared (`evaluation_suite/evaluate.py`, read in full)
Only `evaluate.py` is used. `evaluate_utils.py` is a stale older copy: it imports Snowflake, has its own older comparison code, and calls an undefined `evaluate_spider2sql`; `evaluate.py` never imports it.

1. **Prediction to table.** In `sql` mode the SQL is run on SQLite (the db is first copied into memory), written to CSV in chunks and re-read with `pd.read_csv`. Values therefore pass through a CSV round trip and pandas type inference. In `exec_result` mode a predicted CSV is read directly. Fenced SQL (a sql code fence) is stripped. The `--timeout` argument is accepted but **explicitly unused for SQLite** (`del timeout`).
2. **Gold files.** `resolve_gold_paths`: if `<id>.csv` exists it is the only gold. Otherwise all `<id>.csv` / `<id>_<a-z>.csv` files are gold alternatives. Gold CSVs are read with `pd.read_csv`.
3. **Multiple golds.** `compare_multi_pandas_table` passes if **any one** gold file matches. A flat `condition_cols` list with more than one gold is reused for every gold; a list of lists is applied per gold (12 of our 135 configs are nested).
4. **`condition_cols`.** Integer column positions into the **gold** table (`gold.iloc[:, cols]`), used only if non-empty. Only those gold columns have to be matched. Empty means all gold columns must be matched. Applies per gold file.
5. **Column matching.** Tables are transposed into column vectors. Each required gold column must equal **some** prediction column. Column names are ignored, column order is ignored, extra prediction columns are allowed, and several gold columns may match the same prediction column.
6. **`ignore_order`.** If true, each column vector is sorted **independently** (key: is-None, `str(x)`, is-number), so row alignment across columns is not checked. **All 135 local configs have `ignore_order: true`.** If false, row order within each column is compared.
7. **Value rules.** `NaN`/null is converted to **0** before comparing (so null equals 0). Numbers match with absolute tolerance `0.01`. Everything else uses `==` (case- and whitespace-sensitive strings; a number never equals a string).
8. **`toks`.** Present in every config as a string, but **`evaluate.py` never reads it** (grep finds no use). It looks like metadata; its meaning is unverified.
9. **Scoring.** 0/1 per instance. The "real score" divides by a hard-coded 547.
10. **Path inconsistency.** `evaluate.py` looks for SQLite files in `resource/databases/<db>.sqlite` and reads `spider2-lite.jsonl` from the `spider2-lite/` root, but the README says to put the `.sqlite` files in `resource/databases/spider2-localdb`. Our layout follows the README and the Drive zip.

Implication for our own validation: result-set equivalence is lenient about column names, column order, row order and numeric precision, and strict about string values and about the required gold columns being present.

### 9.2 Gold-replay anomalies
Gold SQL was run on the shipped SQLite files and scored with a verbatim copy of `compare_multi_pandas_table`. Environment: Python 3.10 `sqlite3` = SQLite 3.40.1, pandas 2.3.3, database copied into memory like the evaluator.

For the 24 local ids with gold SQL: **16 pass, 8 fail** (`local003, 023, 029, 066, 131, 210, 219, 309`). None of the 8 is an execution error: each query runs.

| Id | Finding | Cause category |
|---|---|---|
| `local003` | My earlier failure was my own 60 s cap, **not** a data problem. The unmodified query was still running after 25+ minutes (killed). With the CTEs marked `MATERIALIZED` (experiment only; the gold file is untouched) it finishes in about 420 s and returns 11 rows. That output does not match the gold CSV (9 rows): labels differ (`Potentital Loyalists` in the SQL vs `Potential Loyalists` in the gold; `Price Sensitive` and `Promising` are absent from the gold) and values differ (e.g. Champions 250.86 vs 301.07). | **Performance** (the official evaluator has no SQLite timeout, so it would hang) **plus gold SQL ≠ gold CSV** |
| `local131` | SQL returns 20 rows × 4 columns, filtered by `HAVING … > 0`. The gold CSV has 25 rows × 5 columns (adds `StyleID`) and includes styles with zero counts, which that `HAVING` makes impossible. | **Gold SQL does not produce the gold CSV** (not timeout, version or ordering) |
| `local309` | SQL returns 75 × 3. The three gold alternatives are 75×3, 74×5 and 74×3, so my earlier "75 vs 74" was not an anomaly; the alternatives just have different shapes. Versus gold `_a` the SQL disagrees on 8 of 75 years (1958, 1964, 1965, 1970, 1972, 1973, 1974, 1988). It sums points per year, while the gold looks like official champions (e.g. 1964 gold = John Surtees; from my background knowledge, not verified from the data). No NaN and no duplicate years, so not an ordering or null issue. | **Gold SQL semantics differ from gold CSV** |

The other failures are the same kind (SQL output differs from the gold CSV in shape or values): `local023` (gold has 3 columns with a different second metric), `local029` (gold has `customer_unique_id`, `delivered_orders`… that the SQL does not output), `local066` (different values, e.g. Bacon 14 vs 12), `local210` (SQL outputs only `hub_name`; gold has 6 columns including `pct_increase`), `local219` (SQL lacks the `TOTAL_WINS` column).

For the value-level mismatches (`local066`, `local309`) I could not tell whether the cause is the SQL or the data version. The shape and column mismatches (`local131`, `local210`, `local029`) are definitely query-level. **Conclusion: the shipped gold SQL is not a reliable generator of the gold CSVs. The gold CSVs are the evaluator's ground truth; the gold SQL (24 of 135 only) is advisory.** Not ruled out: SQLite version differences (not tested on another version) and NTILE tie ordering (`local003`).

### 9.3 Schema audit: actual SQLite objects vs `DDL.csv` (30 dbs, all 135 questions)
- **Tables:** every table in `DDL.csv` exists in its SQLite file, and no real table is missing from `DDL.csv`. `DDL.csv` also lists SQLite internals `sqlite_sequence` (BowlingLeague, EU_soccer, WWE, school_scheduling, chinook) and `sqlite_stat1` (chinook).
- **Views missing from `DDL.csv` (25, in 5 dbs):**
  - Pagila and sqlite-sakila: `customer_list, film_list, sales_by_film_category, sales_by_store, staff_list`
  - EU_soccer: `match_view`
  - complex_oracle: `cal_month_sales_mv, fweek_pscat_sales_mv, profits`
  - oracle_sql: `brewery_products, customer_order_products, customer_order_products_obj, emp_hire_periods_with_name, inventory_totals, inventory_with_dims, monthly_orders, purchases_with_dims, total_sales, web_page_counter_hist, yearly_sales`
  - stacking: `solution_ext, stack_ok, stack_ok_score` (my earlier note called these "tables"; they are views)
- **Do they affect the 135 questions?** No verified dependence. I searched question text, gold SQL and linked documents for these names. `local219`'s gold SQL mentions `match_view`, but only as a CTE it defines itself, not the db view. `local062` and `local067` (complex_oracle) use the word "profits"; I could not tell whether that refers to the `profits` view. No other view name appears. This is **unverified for the 111 questions without gold SQL**, since a hidden dependence cannot be ruled out.
- **Other findings:** the DDL text is not always executable SQLite. Columns named `index`, `group` or `table` are unquoted, so `CREATE TABLE` fails (AdventureWorks, Brazilian_E_Commerce, Db-IMDB, bank_sales_trading, education_business, electronic_sales, imdb_movies). For the tables whose DDL does execute, column names match `PRAGMA table_info`. 12 of 30 databases define indexes, which `DDL.csv` does not list. Empty tables: Pagila and sqlite-sakila `film_text`, 4 BowlingLeague archive tables, log `action_log_with_noise`, 2 northwind demographic tables, 5 oracle_sql tables.

### 9.4 MongoDB availability on this machine
- **No usable MongoDB.** No `mongod`, `mongosh` or `mongo` on PATH; no MongoDB Windows service; nothing listening on port 27017; no `.env` file (only `.env.example`); no `MONGODB_URI` or `MONGODB_DB` environment variable.
- `C:\Program Files\MongoDB\Server\8.0\` exists but holds only `data/` and `log/` (no `bin/`): leftovers of an installation, not a runnable server.
- Docker CLI 29.3.1 is installed but the Docker daemon is not running.
- No credentials were read or printed. Whether an Atlas account exists is unknown.
- `pymongo` is not installed in the system Python. The replays used a throwaway venv in the scratch directory (pandas, pymongo); nothing was installed system-wide.

### 9.5 What one local Spider example contains and depends on
| Part | Source | Per example or shared | Coverage (of 135 local) |
|---|---|---|---|
| `instance_id`, `question` | current JSONL | per example | 135 |
| `db` | current JSONL, confirmed by `local-map.jsonl` | shared (30 dbs) | 135 |
| Schema (`CREATE TABLE`, types, descriptions, sample rows) | `resource/databases/sqlite/<folder>/` | shared per db; folder name differs from `db` for 2 dbs; omits views | 135 |
| Database contents | `spider2-localdb/<db>.sqlite` (30 files, 1.68 GB) | shared per db | 135 |
| External knowledge | `resource/documents/<file>.md` | per example, optional | 13 |
| Gold result(s) | `exec_result/<id>[_x].csv` (1 to 12 alternatives) | per example | 135 |
| Comparison config | `spider2lite_eval.jsonl` (`condition_cols`, `ignore_order`; `toks` unused) | per example | 135 |
| Gold SQL | `gold/sql/<id>.sql` | per example | 24 (16 reproduce the gold result) |
| `temporal` flag | JSONL | per example | 0 |

A question cannot be answered or checked from its text alone. It depends on the database contents, the gold result with its comparison config, and (for 13) a document. Gold SQL is optional and, per 9.2, not authoritative.

### 9.6 Recommended minimal canonical record for the FUTURE NL to MQL dataset
A proposal for discussion, not an implemented decision. It does not settle the open mentor questions (scope, correctness criterion, use of documents, target size).

```json
{
  "id": "unique id in OUR dataset",
  "source": {"dataset": "spider2-lite", "source_commit": "cafb8673...", "instance_id": "local003", "db": "E_commerce"},
  "question": "NL question (current official text)",
  "external_knowledge": {"file": "RFM.md or null", "included_in_prompt": null},
  "mongo_target": {
    "database": "Mongo db name", "collections": ["..."],
    "schema_ref": "pointer to our own collection/field schema",
    "mapping_ref": "pointer to how SQLite tables map to collections"
  },
  "mql": {"kind": "aggregate | find | ...", "collection": "...", "pipeline_or_filter": "..."},
  "query_type": "from the taxonomy, once defined",
  "reference": {
    "gold_result_files": ["exec_result/..."],
    "comparison": {"condition_cols": [], "ignore_order": true},
    "gold_sql_file": "optional"
  },
  "validation": {
    "executed": false, "execution_error": null, "result_path": null,
    "matches_gold": null, "comparator": "which comparison rule was applied", "reviewed_by": null
  },
  "provenance": {"created_by": "human | model | rule", "notes": ""}
}
```
Rationale from the evidence:
- Keep `instance_id` plus the pinned commit so every record traces to one official question.
- Keep `condition_cols`, `ignore_order` and the **list** of gold CSVs, because correctness is defined by them rather than by a single answer.
- Store the Mongo schema by reference, since the Mongo schema is a project decision not yet made.
- Keep validation separate from content so a record can exist before it has been executed.
- Leave gold SQL optional: only 24 exist and 8 of them contradict the gold CSVs.

### 9.7 Remaining open items
The mentor questions in section 7 are unchanged. New ones from this audit:
1. Whether to trust the gold CSV over the gold SQL where they disagree (the evidence favors the CSV, but it is the team's decision).
2. Whether to exclude the 8 questions whose gold SQL fails to reproduce the gold, or keep them with CSV-only validation.
3. Whether to install MongoDB locally, use Docker, or use Atlas (nothing usable is available now).
