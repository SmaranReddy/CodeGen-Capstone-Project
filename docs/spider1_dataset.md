# Spider 1.0 → MongoDB/MQL dataset (279 validated examples)

**File:** `data/processed/spider1_nl_mql.jsonl`. Its SHA-256 is recorded in
`data/interim/spider1/dataset_summary.json`.

This is the dataset for the mentor's requirement: *"convert the Spider dataset, at least
200–300 examples, execute the MQL in MongoDB Atlas, then compare a frontier LM against a small
Qwen-2.5-Coder-Instruct baseline."*

## 1. Why Spider 1.0 (and how this relates to the earlier 68 Spider 2.0-Lite examples)

The earlier work converted **Spider 2.0-Lite**. Only 135 of its 547 records are locally
executable (SQLite). The other 412 need BigQuery/Snowflake credentials. That capped the local
pool at 135, and 68 of those were validated (`data/processed/nl_mql_dataset.jsonl`, see
`docs/conversion_report.md`).

The original **Spider 1.0** (Yu et al., EMNLP 2018) is fully executable:
- 1,034 dev questions over 20 SQLite databases;
- official gold SQL;
- the official test-suite databases (Zhong et al., 2020), which provide alternative database
  contents per DB.

That makes 200–300 genuinely validated examples achievable.

The two datasets are kept **separate**:
- the 68 Spider 2.0-Lite examples form a small, very hard secondary set;
- the 279 Spider 1.0 examples are the primary benchmark.

## 2. Provenance

| Artifact | Source | Integrity |
|---|---|---|
| `data/raw/spider1/spider_data/` | official `spider_data.zip` (Google Drive id `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J`, linked from yale-lily.github.io/spider) | zip sha256 `00636695…85b121b`; dev.json sha256 `30d64a3f…4d758a4` |
| `data/raw/spider1/official_eval/` | `evaluation.py`, `process_sql.py` from github.com/taoyds/spider | commit in `SOURCE_COMMIT.txt` |
| `data/raw/spider1/testsuite/` | 3 variants per DB from the official test-suite `database.zip` (Drive id `1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w`, taoyds/test-suite-sql-eval) | zip sha256 `9ec24ea8…f41da96`; variant list in `data/interim/spider1/testsuite_variants.json` |

Raw files are treated as read-only.

## 3. Pipeline (all scripts in `scripts/spider1/`)

1. **`build_pool.py` (pre-registered pool and frozen sample).** All 1,034 dev questions are
   parsed with the official `process_sql`. Each gets an official hardness label, and its gold
   SQL is executed. The rules below were fixed **before any MQL was written**:

   | Rule | Excluded |
   |---|---:|
   | duplicate `(db_id, normalised gold SQL)` (Spider dev pairs paraphrases) | 483 |
   | empty or all-NULL gold result (weak evidence) | 27 |
   | `ORDER BY … LIMIT` cutoff inside a tie (gold not unique) | 24 |
   | > 1,000 result rows | 6 |
   | ORDER BY tie structure not verifiable | 1 |
   | **eligible pool** | **493** |

   The sample is a seed-42 draw of **300**, stratified proportionally by hardness
   (easy 75 / medium 133 / hard 48 / extra 44). It is frozen in
   `data/interim/spider1/sample.jsonl`, and failures are never replaced with easier items.
2. **`import_atlas.py` (Atlas import).** All 20 dev DBs go into the Atlas database
   `codegen_spider1`, one collection per table, named `<db_id>__<table>`.
   - Every column is kept, and values keep their SQLite storage class.
   - Many Spider columns hold numbers as TEXT, and MQL sees exactly that.
   - **`testsuite_setup.py`** imports 3 variants per DB (except wta_1, whose variants are
     ~50 MB each) into `codegen_spider1_ts1..3`, using identical collection names.
3. **Gold MQL authoring.** Gold MQL lives in `data/interim/spider1/gold_mql/<db>.json`. The
   protocol is in `docs/spider1_gold_mql_authoring.md`.
   - It is **LLM-authored** (Claude), not human-annotated. The pilot DB was written directly;
     the rest was written by 6 parallel Claude agents.
   - Each pipeline must reproduce the official gold SQL **as SQLite evaluates it**. That
     includes type affinity, integer division, NULL semantics, LIKE case-insensitivity,
     set-operation DISTINCT, and bare columns.
4. **Acceptance.** Every check is re-run from scratch by `build_dataset.py`:
   - **Dev execution match.** The gold MQL runs on Atlas, and its result must match the gold
     SQL result under `exec_match`.
   - **Test-suite cross-check.** The same MQL and gold SQL run on 3 alternative DB contents.
     No variant may FAIL. A variant whose LIMIT cutoff falls in a tie is INCONCLUSIVE.
     This rules out pipelines that only match the dev data by coincidence.
   - **Curation.** Agents flagged gold/question problems in `gold_mql/*.issues.json`. The
     curator decided each one in `adjudication.json`, with a written reason.
5. **Audits.**
   - `audit_literals.py` found no hard-coded answer values. All hits are type names, coercion
     regexes, or LIKE patterns.
   - An **independent reviewer agent** audited 42 items: all 12 dev-only items plus a seed-7
     random 30. Findings: 0 incorrect, 3 doubtful (`data/interim/spider1/independent_review.json`).
     Two were missing null-key guards before `$lookup`. Both were fixed and re-validated. The
     third is a known SQL-level ambiguity that is already flagged.

## 4. Comparator (`spider1_kit.exec_match`, unit-tested in `tests/test_spider1_kit.py`)

**Spider-1-style execution match.** Gold MQL and model outputs are scored by the same
function.
- **Rows and columns.** Each output document is one row. An ObjectId `_id`, or an `_id` that
  is null in every row (`$group: {_id: null}`), is ignored. Nested objects are flattened.
- **Column count.** It must equal the gold's. Column names and column order are ignored.
- **Values.** Numbers are compared after rounding to 4 decimals, so `3 == 3.0`. Strings must
  match exactly.
- **Order.** Rows are compared as a multiset, unless the gold SQL has a top-level ORDER BY.
  In that case the order must match, except among rows tied on the ORDER BY key.

## 5. Result

| | count |
|---|---:|
| sampled | 300 |
| gold MQL authored | 300 (0 `cannot_convert`) |
| passed dev execution match | 300 |
| failed test-suite cross-check | 0 |
| excluded by adjudication | 21 |
| **validated** | **279** (279 unique questions, 279 unique gold SQL) |

- **By hardness:** easy 71, medium 126, hard 44, extra 38. All 20 dev DBs are covered.
- **Evidence:**
  - 267 examples match on the dev DB **and** on test-suite variants;
  - 12 are dev-only: 11 wta_1, with no variants imported, plus `spider1_dev_0667`, where all
    3 variants were tie-inconclusive. The wta_1 author also ran swapped-literal checks against
    the SQL on the dev DB.
- **Flags:** 13 kept examples are flagged `ambiguous`. These are genuine NL ambiguity, or a
  SQLite quirk that coincides with the intended answer on this data. The evaluation reports
  accuracy with and without them.

The 21 exclusions, with reasons, are in `data/interim/spider1/adjudication.json`:
- **11 `annotation_mismatch`:** the gold SQL answers a different question than the one the
  NL states. Examples: no dog filter for "youngest dog", a cross join, a case-sensitive `!=`
  that excludes nothing.
- **6 data quirks** that make the gold contradict the question:
  - TEXT `'null'` placeholders that SQLite's `avg` counts as 0;
  - whitespace-padded airport codes that make counts 0.
- **1** average of a DATETIME string.
- **2** golds that depend on unspecified bare-column choice.
- **1** `SELECT *` (19 columns) for "which countries".

## 6. Known limitations

- **Gold MQL authorship.** The gold MQL is LLM-authored and accepted on execution evidence.
  It was not reviewed line by line by a human.
- **Latent null-join risk.** 43 `$lookup`s rely on the fact that no foreign key is NULL in
  any of the 4 DB contents. On data with NULL foreign keys they could join NULL to NULL,
  which a SQL inner join never does.
- **Natural order.** A few golds (e.g. `0816`) reproduce SQLite's rowid-order tie choice
  through MongoDB natural (insertion) order. MongoDB does not guarantee that order.
- **Contamination.** Spider dev is public and probably in model training data. The MQL
  rendering is new, but contamination cannot be ruled out.
- **wta_1 evidence.** wta_1 has no test-suite evidence.

## 7. Reproduce

Fetch the raw data and models once. They go into gitignored folders and should be verified
against the sha256 values in §2.

```powershell
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
python -m gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O data/raw/spider1_download/spider_data.zip
python -m gdown 1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w -O data/raw/spider1_testsuite_download/testsuite_database.zip
python -I -c "import zipfile; zipfile.ZipFile('data/raw/spider1_download/spider_data.zip').extractall('data/raw/spider1')"
# official evaluation code (commit in data/raw/spider1/official_eval/SOURCE_COMMIT.txt)
curl -L -o data/raw/spider1/official_eval/evaluation.py  https://raw.githubusercontent.com/taoyds/spider/b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c/evaluation.py
curl -L -o data/raw/spider1/official_eval/process_sql.py https://raw.githubusercontent.com/taoyds/spider/b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c/process_sql.py
# Qwen GGUF (pinned revisions; expected sha256 in configs/evaluation_spider1.yaml / docs)
curl -L -o models/qwen2.5-coder-3b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/f74adce6aa16316c625447af059dbebe4983757c/qwen2.5-coder-3b-instruct-q4_k_m.gguf
#   sha256 724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7
curl -L -o models/qwen2.5-coder-7b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/13fb94bfda8c8cf22497dc57b78f391a9acb426a/qwen2.5-coder-7b-instruct-q4_k_m.gguf
#   sha256 509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c
```

Then build. Every step is deterministic, except gold MQL authoring, which is stored in
`data/interim/spider1/gold_mql/`:

```powershell
python scripts/spider1/build_pool.py            # pool + frozen sample (deterministic)
python scripts/spider1/import_atlas.py          # dev DBs -> codegen_spider1
python scripts/spider1/testsuite_setup.py       # extracts 3 variants/DB, imports -> codegen_spider1_ts1..3
python scripts/spider1/prompt_spider1.py        # freeze prompt sample documents
python scripts/spider1/build_dataset.py         # re-validate everything, write dataset
python -m pytest tests -q
```
