# spider-sql-v1: Spider 1.0 NL→SQL dataset

This is the primary NL→SQL dataset, built 2026-10-27 from the official Spider 1.0 files.

> **Correction 2026-10-29:** NL→SQL is an auxiliary, SQL-only track; the project's primary task is NL→MQL (`CLAUDE.md`). This dataset's SQL is not an MQL label. Its cleaned train/val_internal splits may serve only as a source pool for an audited SQL→MQL conversion (`docs/nl2mql_next_steps.md` §3–§4).

| | |
|---|---|
| Builder | `scripts/sql_v1/build_spider_sql.py` |
| Outputs | `data/processed/spider_sql_v1/`. The input and output sha256 hashes, the funnel and per-split summaries are in `manifest.json` |
| Build log | `outputs/sql_v1/build.log` |
| Tests | `tests/test_spider_sql_v1.py` |
| Licence | CC BY-SA 4.0 (Spider; see `manifest.json`) |

`data/raw` was not modified. The MQL-era artifacts were not touched or used. Those are `spider1_nl_mql*.jsonl`, `data/splits*`, `data/rag` and `data/finetune/ft_v1`, and all of them are Spider-dev derived.

## Files

| File | Records | DBs | Use |
|---|---:|---:|---|
| `train.jsonl` | **7,773** | 130 | Supervised fine-tuning |
| `val_internal.jsonl` | **797** | 16 | Checkpoint selection and early stopping. It is DB-disjoint from train, dev and test |
| `dev_full.jsonl` | 1,034 | 20 | **Headline benchmark** (official evaluator). Do not tune on it repeatedly |
| `dev_clean.jsonl` | 1,034 | 20 | Dev without broken golds (execution failure or a T5 defect confirmed against the DB). Identical to full: every dev gold passed all gold-only checks |
| `test_full.jsonl` | 2,147 | 40 | **Final one-shot evaluation only** |
| `test_clean.jsonl` | 2,123 | 40 | Test without broken golds: 14 token-rejoined literals and 10 `= "null"` compares were dropped. Secondary report only |
| `rejected.jsonl` | 89 | — | Dropped train rows, each with its `reject_reason` |
| `schemas_ddl.json` | 206 DBs | — | `CREATE TABLE` statements for every DB used, for building prompts |

- Train, validation and evaluation sets share no databases.
- No dev or test question (normalised) appears in train or `val_internal`. The tests check both properties.

## Train funnel (8,659 → 8,570)

| Rule | Removed |
|---|---:|
| T1: gold SQL fails to execute (2 `ORDER BY` before `INTERSECT`, 1 missing table) | 3 |
| T2: exact duplicate (db, question, SQL) | 8 |
| T3: normalised question also in dev (7 records) or test (33 records; the inventory's figure of 25 counted distinct questions with a slightly different normalisation) | 40 |
| T4: same question with different SQL whose results differ | 0 |
| T5: gold-label defect | 38 |
| **Kept** | **8,570** (train 7,773 + val_internal 797) |

The T5 defects break down as:
- `= "null"` string compare, where no such string exists in the DB: 18
- anonymised placeholder value, such as `'actor_name0'` when the question says "Kevin Spacey": 13
- token-rejoined value matching nothing in the DB, such as `'Louisville ,  Kentucky'`: 6
- placeholder inside the question (`keyphrase0`): 1

The T5 rules came from a hand spot-check of every `literal_ungrounded` row in the first build. After T5, no kept training row still has an ungrounded literal.

**val_internal** contains 16 whole `train_spider` DBs (populated, not `music_2`). They were picked by a seeded shuffle of db_ids (seed 20261027) until the set reached at least 10% of train_spider records.

## Verification tiers in train (`tier` field)

| Tier | Count | Meaning |
|---|---:|---|
| `exec_ok` | 6,029 | Executed on the official, populated DB |
| `exec_ok_testsuite_variant` | 1,644 | The official DB is empty (`academic`, `geo`, `imdb`, `restaurants`, `scholar`, `yelp`). The query was executed on one populated test-suite variant instead; the variants are recorded in `manifest.json`. This mainly confirms the SQL runs against the schema. Variant values are synthetic, so 1,076 of these return empty results |
| `exec_ok_empty_db` | 100 | `music_2` has no populated variant. Execution on the empty DB checks table and column names only |

## Flags (informational, nothing dropped for them)

| Flag | Train | Dev | Test |
|---|---:|---:|---:|
| `empty_result` | 1,299 | 49 | 54 |
| `all_null_result` | 29 | 4 | 8 |
| `limit_tie` (cutoff inside a tie) | not computed | 44 | 77 |
| `question_in_train` | — | 7 | 27 |
| `tokenized_literal` / `null_string_compare` / `literal_ungrounded` | 0 | 0 | 14 / 10 / 22 |

- **The `*_full` sets are deliberately unfiltered**, so the headline scores stay comparable to published Spider results. The `*_clean` sets drop only golds shown to be broken: 0 in dev and 24 in test. Report them as a secondary number.
- `limit_tie`, `literal_ungrounded`, `empty_result` and `question_in_train` stay flags in every set. They are weaker signals, and the test golds were not inspected by hand, to preserve test blindness.

## Limits

- Execution shows that a query runs. It does not show that the SQL answers the question.
- Semantic label errors that fit the data, such as a wrong column with a plausible result, are not detected. Spider train has known annotation noise of this kind.
- 1,744 train rows (the variant and empty tiers) have weaker verification than the rest.
- 8,570 rows hold 4,690 distinct (db, SQL) queries (train 4,221, val_internal 469). Several questions per query are paraphrases and are kept on purpose.
