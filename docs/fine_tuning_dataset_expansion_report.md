# Fine-tuning dataset expansion report: ft-v1

Built 2026-10-24.

| | |
|---|---|
| Code | `scripts/finetune_v1/` |
| Dataset | `data/finetune/ft_v1/` |
| Tests | `tests/test_finetune_v1.py` |

The 455-example benchmark is not modified by any of this work:

- `data/processed/spider1_nl_mql_expanded.jsonl`
- `data/splits_expanded/` (318/72/65)
- every prediction, report and metric under `outputs/`

The tests pin the benchmark and split-manifest sha256 values.

## 0. Summary

| | Count |
|---|---|
| Raw source rows | **14,815** |
| Unique (question, SQL) pairs after exact dedup | **4,337** |
| Pairs passing all reference-validity gates (eligible for conversion) | **2,621** |
| MQL candidates generated (sql2mql-v1) and passing the read-only validator | **2,617** |
| **Fixture-equivalence verified** | **2,561** |
| …after post-conversion dedup (−13) | 2,548 |
| …after the near-duplicate cap (3 per cluster; 964 capped out) | 1,584 |
| New examples in `train.jsonl` / `dev_internal.jsonl` | **1,504** / 80 |
| Existing Spider-train examples merged (RAG corpus v1, Atlas-verified on the real DB) | **280** |
| **Total `train.jsonl`** | **1,784** |
| Expansion vs the 280-example verified corpus | **6.37×** |
| Expansion vs the 455-example benchmark | 3.92× |

The 3.92× figure only compares counts. The benchmark is evaluation data and none of it is in
the training set beyond the 280 train-partition examples.

**Two things to know before you use these numbers:**
1. **The verified tier is not source-database verification.** The Kaggle dataset ships no
   database and no schema. Each new example is verified by *result equivalence* between its
   reference SQL (SQLite) and a candidate MQL (MongoDB 8.0.34 on Atlas), run on synthetic
   discriminating fixtures (§4). The status is `fixture_equivalence_verified`; it is never
   labelled execution-verified. The 280 Spider examples keep `execution_verified_source_db`.
2. **The new pool is easier than the benchmark and more homogeneous.**
   - 1,000 of the 1,504 new examples are easy, 458 medium, 46 hard.
   - All are from one synthetic healthcare/retail domain.
   - 962 use `SELECT *`. Only 30 use joins.
   - See the data-mix guidance in `docs/fine_tuning_plan.md` §2.4.

## 1. Source audit

| Item | Finding |
|---|---|
| Location | `C:\Users\Asus\Downloads\archive.zip` → `archive\NL2SQL_Query_Dataset.csv`. Nothing else was found in Downloads, on the Desktop or in the project. |
| Immutable copy | `data/raw/nl2sql_kaggle_v1/`. The CSV and zip are copied byte-for-byte and marked read-only. sha256 CSV `23d32e89…5004`, zip `6e1fe943…eb`, pinned in `nk_common.RAW_SHA256` and checked on every extract run. |
| Kaggle metadata (API) | `sureshmuthusamy001p/nl2sql-query-dataset`, version 1 ("Initial release"), updated 2023-12-22, 2,726,725 bytes. |
| Licence | **CC BY-NC-SA 4.0** (Kaggle `licenseName`): attribution, non-commercial use only, and derivatives must be shared under the same licence. ft-v1 inherits this, and its manifest records it. The 280 Spider records are CC BY-SA 4.0. The two ShareAlike terms can't both govern one combined adaptation, so every record keeps its own licence field and the sources stay separable (plan §2.10). |
| Provenance | **None published.** The description says only "14,815 examples … SELECT statements … from a specific table." |
| Schema of the file | Two columns, `Prompt` and `Query`. There are **no db identifiers, no table schemas and no databases.** |
| Domain | 85 table names: synthetic healthcare-operations tables (`event`, `assets`, `staff_directory`, `pharmacies`, `hazards`, …) plus generic retail ones (`orders`, `customers`, `products`, `sales`). |
| Dialect | 4,189 unique pairs parse as MySQL, 2 only as Postgres, 146 in no dialect. MySQL time functions (`NOW()`, `CURDATE()`, `DATE_SUB`, `INTERVAL`), Postgres `INTERVAL '7 days'` / `ILIKE`, and some T-SQL `GETDATE`/`DATEADD` all occur. |
| Spider overlap | None found. The highest question trigram-Jaccard against all 8.6K Spider train/dev questions is 0.66 (0 at ≥0.8); against the 137 held-out benchmark questions it is 0.53. |

**Quality problems found, all handled explicitly:**

- **CSV corruption (1,035 rows repaired, deterministically):**
  - 500 prompts were split at an unquoted comma; the tail landed in `Query` as
    `…ascending order.,SELECT …`.
  - 475 rows carry a stray trailing `"`.
  - 60 rows use `\"` escapes.
- **Massive duplication:** 10,474 rows are exact duplicates. One prompt appears 70 times.
- **Placeholders and invented literals:** `your_user_id`, `<city>`, `[id]`; "a specific role"
  paired with `roles_id = 789`.
- **Label noise:**
  - The question says 'Pharmacist D' and the SQL says 'Pharmacist B'.
  - 103 questions have several different SQL labels.
- **Row-shifted labels:** runs of prompts are paired with a neighbouring row's SQL, e.g.
  "names and addresses of organizations" → `SELECT name, is_default FROM event_staff_roles`.
  Truncated SQL also occurs, e.g. `… WHERE asset`.
- **Non-SELECT rows:** UPDATE/DELETE statements and DDL-like fragments.

## 2. Processing stages and counts (from `stage1_counts.json` and `build_manifest.json`)

| Stage | In | Out | Removed / routed |
|---|---|---|---|
| Raw CSV rows | — | 14,815 | 4 empty after repair |
| Exact dedup on normalised (question, SQL) | 14,811 | **4,337** | 10,474 duplicate rows collapsed; every source row id is kept on its pair |
| Parse (mysql→postgres→tsql→sqlite) + statement type + placeholder SQL | 4,337 | 3,854 | **483 rejected**: 146 parse errors, 22 non-SELECT, 315 placeholders |
| Supported subset (converter and verifier) | 3,854 | 3,283 | **571 unsupported**, kept as unverified candidates: subqueries 216, unsupported operands/functions 131, `t.*` 81, other join forms 37, … |
| Canonical schema and type check | 3,283 | 3,083 | **200 ambiguous** (`type_conflict`, e.g. `roles_id = 'Role234'` vs numeric `roles_id = 789` elsewhere) |
| Question↔SQL grounding (literals and identifiers/structure) and SQL-in-question | 3,083 | 2,676 | **406 ambiguous + 1 rejected** (below) |
| Conflicting labels (same question, ≥2 grounded SQL variants) | 2,676 | 2,621 | **83 ambiguous**. 28 unsupported pairs moved to ambiguous for the same reason. |
| Held-out overlap (trigram Jaccard ≥ 0.8 to any of 137 held-out questions) | 2,621 | 2,621 | 0 excluded (max similarity 0.53) |
| **Eligible** | | **2,621** | |
| Converter (`sql2mql-v1`) | 2,621 | 2,617 | 4 converter-unsupported |
| Read-only validator (`eval_kit.validate_mql`) | 2,617 | 2,617 | 0 |
| Fixture adequacy (every SQL mutant witnessed, non-empty result, ≥3 variants) | 2,617 | 2,593 | 24 not discriminating (contradictory or type-mismatched col=col predicates, etc.) |
| MongoDB execution = SQLite result on every variant + verifier sanity mutant | 2,593 | **2,561** | **32 failed_execution** (§4.4) |
| Post-conversion dedup (same loose question, same target MQL) | 2,561 | 2,548 | 13 |
| Near-duplicate cap (≤3 per question-template × SQL-skeleton cluster; 1,406 clusters) | 2,548 | 1,584 | 964 → `verified_capped_near_dups.jsonl` (verified, not trained on) |
| Internal dev split (cluster-grouped 5%) | 1,584 | 1,504 train + 80 dev | |

Grounding failures, first reason per pair:

| Reason | Pairs |
|---|---|
| SQL literal not stated in the question | 265 |
| Identifier (column or table) not mentioned | 54 |
| Stated value missing from SQL | 29 |
| Aggregate or shape mismatch | 40 |
| Underspecified "specific X" with no value | 9 |
| Interval/limit not stated | 9 |
| Question contains SQL text | 1 |

**Five-way classification of the 4,337 unique pairs**
(`pairs.jsonl: source_category`; reason-level detail in `status_manifest.jsonl`):

| Category | Pairs | Meaning here |
|---|---|---|
| 1. Directly usable | 2,474 | MySQL, no repair, passed the stage-1 gates |
| 2. Usable after schema/dialect handling | 347 | Needed a CSV repair or a non-MySQL dialect, or has a type conflict against the canonical schema |
| 3. Needs a database or an extended converter | 543 | Valid-looking SELECT outside the supported subset |
| 4. Duplicate / near-duplicate | 0 at pair level | Exact duplicates were collapsed at row level (10,474 rows). Near-duplicates are clustered and capped at build time (964). |
| 5. Invalid / ambiguous / unsuitable | 973 | Rejected (484) or ambiguous (489 grounding/type/conflict) |

Strictly speaking, every pair is in category 3, because no source database exists. The
fixture verification in §4 is how categories 1–2 become usable anyway.

## 3. Schema reconstruction

The source has no schema. `schema.json` is reconstructed as follows:

- **Columns:** each table gets the union of every column referenced by any parsed SELECT.
- **Types:** decided by majority evidence from literals and functions (date literals, numeric
  literals, LIKE, SUM/AVG, YEAR(), 0/1/TRUE on boolean-looking names). Column-name
  heuristics act only as a tie-breaker.
- **Storage:** dates and datetimes are stored as `YYYY-MM-DD[ HH:MM:SS]` strings, the same
  convention as the Spider collections. Booleans are stored as 0/1.
- **Distinct tables:** near-identical table names (`event`/`events`,
  `formsettings_`/`formssettings_summarytools`) are kept as separate tables. Their column sets
  differ, and merging them would be a guess.

Training prompts list the target collection(s) plus 2 deterministic distractor collections
with unrelated names. Each listed collection shows its fields, types and 2 frozen generic
sample documents. The prompt template is identical to the benchmark's (`nl-mql-spider1-v1`).
FKs are "(none)".

## 4. SQL→MQL conversion and verification

### 4.1 Candidate generation
- `sql2mql.py` is a deterministic converter that turns the reference SQL AST into the
  benchmark output contract `{"collection", "pipeline"}`.
- **Why a converter instead of an LLM:** a converter is free, local and reproducible. Groq's
  free tier, at about 200K tokens/day, would have taken over 10 days for 2.6K items. CPU Qwen
  was too slow and unreliable under the memory reaper.
- The converter output is only a **candidate**. Acceptance depends solely on execution.
- **Semantics it implements:**
  - SQL three-valued logic: `<>`, `NOT IN` and `NOT LIKE` never match NULL, and `$expr`
    comparisons are guarded with `$ne: null`.
  - Inner-join keys that are NULL never match.
  - LIKE becomes a case-insensitive anchored regex.
  - Clock functions become `$$NOW`, `$dateTrunc`, `$dateSubtract`/`$dateAdd` and
    `$dateToString`.

### 4.2 Expected result
- The reference SQL runs on SQLite (`sqlsem.to_sqlite`) with MySQL semantics where the
  engines differ:
  - `NOW()`/`CURDATE()`/INTERVAL arithmetic and `YEAR`/`MONTH`/`DAY`/`DATE` are Python UDFs
    on a frozen UTC clock;
  - `/` is real division;
  - TRUE/FALSE become 1/0.
- **Semantic sanity before execution:** the gates in §2 (grounding, structure, conflicts) plus
  a manual review of a random sample (§4.5). Successful execution alone is not treated as
  proof that the SQL answers the question.

### 4.3 Fixtures and comparison
- **Fixtures (`fixtures.py`):**
  - Every referenced table gets all its canonical columns, with explicit NULLs.
  - Values come from pools seeded with the query's own literals and their neighbours.
  - Witness rows satisfy the WHERE clause, and near-miss rows each falsify exactly one atom.
    Both are confirmed by SQLite.
  - Join keys and GROUP BY columns use small shared domains. DISTINCT queries get duplicate
    rows. LIMIT queries get more candidate rows than the limit.
  - Clock-relative rows are spread around the `$$NOW` boundary and never placed on it.
- **Adequacy:** a fixture set is accepted only if every SQL mutant gives a different result
  on at least one variant, the result is non-empty, and there are ≥3 valid variants.
  - Mutants: drop or negate each WHERE/HAVING atom; swap `>`↔`>=` / `<`↔`<=`; drop each IN
    element; un-anchor each LIKE; flip each ORDER BY; drop LIMIT/OFFSET/DISTINCT.
  - On average 3.7 mutants are required per example.
  - 546 mutants are exempt by documented rule:
    - redundant `IS NOT NULL`;
    - `>`/`>=` against a clock, or a date-only literal compared with a datetime column,
      where equality cannot occur on well-formed data.
  - Variants whose ORDER BY…LIMIT boundary falls inside a tie are discarded.
- **Execution:**
  - The candidate MQL runs on Atlas as `codegen_eval_readonly` via a database-level aggregate
    whose first stage is `$documents` holding the fixture rows. `$lookup.from` is rewritten
    to an inline `$documents` sub-pipeline.
  - **Nothing is read from or written to any collection, and no import was needed or
    performed.** M0 (free) usage only; no paid API was used.
- **Comparison:** results are compared with the benchmark's `spider1_kit.docs_to_rows`,
  `_norm` and `exec_match`: multiset, ordered, or tie-aware ordered.
  - When column names line up exactly, a name-aligned fast path applies the same rules under
    the identity column mapping.
  - It is never looser than `exec_match`, and a test checks this. It avoids `exec_match`'s
    exponential permutation search on wide `SELECT *` results with many 0/1 columns.
- **Verifier sanity:** for every WHERE query (1,897), the candidate with its WHERE `$match`
  removed must fail. It did.
- **Replay of the shipped artifact** (`replay_targets.py`): every train/dev target is parsed
  from its assistant-message text and executed again.
  - Spider: through `run_eval.score_output` on `codegen_spider1` against the gold rows.
  - Kaggle: on regenerated fixtures, with a fresh clock for the clock-relative ones.
  - **Result: 280/280 and 1,584/1,584 pass** (`replay_report.json`; summary in
    `build_manifest.json`).
- **Key-order bug, found and fixed before release:**
  - The first build wrote MQL through `json.dumps(sort_keys=True)`. That alphabetised keys
    and therefore changed multi-key `$sort` priority in **7** Kaggle targets (content hashes
    still matched).
  - Targets are now taken from the deterministic converter's order-preserving output, for
    Kaggle, and from the original model output text, for Spider. Each is asserted
    content-identical to the verified candidate.
  - Nothing is written with `sort_keys` any more.
  - Tests now require `compact(target) == assistant text == fresh converter output`, and the
    replay above runs on the text.

### 4.4 Verifier self-test and known semantic gaps
- **Self-test** (`verifier_selftest.py`, seed 20261024, 400 verified pairs, non-clock): mechanical
  MQL corruptions were applied and re-run on the same fixtures. **444 of 454 corruptions were
  caught (97.8%).** The uncorrupted baseline was checked first and passed
  333/333 (non-clock sample items).
  - Caught 100% (sort flip, dropped projected field, `$in` element drop, `$nin` null-guard
    removal, regex un-anchoring, skip+1): 41/41, 71/71, 4/4, 1/1, 4/4, 2/2.
  - `drop_match`: 227/230.
  - `limit+1`: 29/31.
  - `$gt`↔`$gte` / `$lt`↔`$lte`: 65/70. The misses are the provably equivalent
    datetime-vs-date-literal cases.
  - Full results: `data/interim/finetune_v1/verify/selftest.json`.
- **32 failed executions are real SQL↔MQL divergences, correctly rejected:**
  - `AVG` over text-typed columns: SQLite coerces text to 0, Mongo ignores it.
  - `SUM` over an all-NULL group: SQL gives NULL, `$sum` gives 0.
- **Weak check: LIMIT without ORDER BY.** These items (part of the 57 LIMIT examples) are
  "verified" only in the sense that both engines return rows in fixture insertion order. The
  result of such a query is implementation-defined, so the fixture check proves little about
  them.
- **Accepted, documented gaps** (they also exist in the benchmark):
  - `SUM` over all-NULL groups and `COUNT`/aggregates over an empty input differ between SQL
    and the idiomatic MQL. Fixtures require non-empty matches, so verified examples were
    never tested on these edges.
  - String `=` is case-sensitive (SQLite and Mongo), not MySQL's case-insensitive collation.
  - Datetime-vs-date-literal comparisons use string semantics.

### 4.5 Manual semantic review
- 60 seeded random new training records were read; the reviewer is the assistant, not an
  independent annotator (`data/finetune/ft_v1/semantic_audit_sample.json`).
- **57 correct, 3 arguable, 0 wrong.** The 3 arguable: "after 6 PM" as `$gte`; "average order
  value" read as AVG(`order`); "disabled" read as `date_disabled IS NOT NULL`.
- 55 of the 60 are still in the final train file after re-clustering.
- This bounds obvious label errors. It is not an inter-annotator-grade estimate.

## 5. Leakage, provenance and deduplication

All checks are automated; the build aborts on any violation. Results are in `leakage_report.json`.

| Check | Result |
|---|---|
| `scripts/rag/leakage.check` run over all train/dev records (held-out ids, db names, collection names, questions, gold SQL) | 0 errors |
| Exact question match with any held-out question | 0 |
| Held-out collection name in any prompt or target | 0 |
| Reference SQL equal to a held-out gold SQL, or present in the model input | 0 |
| Question trigram Jaccard vs held-out (137) | max 0.527, 0 at ≥0.6 |
| Question trigram Jaccard vs all Spider train/dev | max 0.658, 0 at ≥0.8 |
| Spider-280 examples from the train partition only | True (enforced) |
| Table-name coincidence with a held-out DB | `likes` (network_1). Schemas are disjoint: `post_id, like_date` vs `student_id, liked_id`. The collection is namespaced `nl2sqlkaggle__likes`; kept and recorded. |
| Benchmark and split files | sha256 recorded in the report and pinned in tests; unchanged |

Deduplication runs at three levels:

- exact (question, SQL) rows → pairs;
- identical loose question + identical target MQL after conversion;
- near-duplicate clusters, keyed on the question with its values masked (quoted spans,
  numbers, dates, and any text equal to an SQL string literal) plus the SQL skeleton with its
  literals masked, capped at 3. Dev and train never share a cluster (tested).

## 6. Breakdown of the 1,504 new training examples

- **Complexity** (nk-complexity-v1, a Spider-hardness-like score over components):
  easy 1,000 · medium 458 · hard 46.
  For comparison, the Spider 280 are easy 81 · medium 128 · hard 39 · extra 32.
- **Features:**

  | Feature | Count | Feature | Count | Feature | Count |
  |---|---|---|---|---|---|
  | where | 1,017 | `SELECT *` | 962 | aggregate | 284 |
  | clock-relative | 258 | group by | 202 | order by | 163 |
  | like | 142 | null check | 121 | count | 119 |
  | not | 119 | between | 79 | max | 77 |
  | distinct | 66 | limit | 57 | in | 55 |
  | min | 53 | or | 46 | date part | 36 |
  | avg | 31 | join | 30 | sum | 21 |
  | offset | 12 | count distinct | 3 | | |

- **MQL operators** (top): `$match`, `$project`, `$expr`, `$group`, `$dateToString`,
  `$dateSubtract`, `$dateTrunc`, `$sort`, `$regex`, `$sum`, `$limit`, `$lookup` (30).
  Full counts are in `build_manifest.json`.
- **Collections:** `event` 421 is the largest, followed by `staff_directory` 131, `events` 118,
  `hazards` 113 and `organizations` 112; 42 collections in all.
- **By source:** `train.jsonl` = 1,504 `kaggle_nl2sql_query_dataset_v1` + 280
  `spider1_dev_expanded_benchmark_train` (265 gpt-oss-120b and 15 qwen2.5-coder-7b
  predictions, each re-executed on Atlas).

## 7. Files

| File | Content |
|---|---|
| `data/raw/nl2sql_kaggle_v1/` | Immutable source copy (read-only, sha-pinned) |
| `data/interim/finetune_v1/source_rows.jsonl` | Every raw row with its repaired question/SQL, repair tags and row sha |
| `data/interim/finetune_v1/pairs.jsonl` | 4,337 unique pairs: dialect, analysis, grounding, overlap, cluster, class, reasons |
| `data/interim/finetune_v1/schema.json`, `overlap.json`, `stage1_counts.json` | Reconstructed schema, overlap evidence, stage counts |
| `data/interim/finetune_v1/verify/progress.jsonl` | Resumable per-pair verification log (fixture seeds, mutant coverage, execution results, hashes) |
| `data/finetune/ft_v1/train.jsonl` | **1,784** SFT records: messages, target MQL, labels, reference/validation/provenance metadata, hashes |
| `data/finetune/ft_v1/dev_internal.jsonl` | 80 cluster-grouped records for loss monitoring |
| `data/finetune/ft_v1/sft_{train,dev}.prompt_completion.jsonl` | TRL conversational prompt-completion view |
| `data/finetune/ft_v1/verified_capped_near_dups.jsonl` | 964 verified near-duplicates beyond the cap |
| `data/finetune/ft_v1/candidates_unverified.jsonl` | 571 unverified candidates with blockers |
| `data/finetune/ft_v1/rejected.jsonl` | 1,218 rejected, ambiguous, failed or duplicate records, with reasons |
| `data/finetune/ft_v1/status_manifest.jsonl` | One final status per unique pair (all 14,811 non-empty rows are accounted for) |
| `data/finetune/ft_v1/leakage_report.json`, `replay_report.json`, `token_audit.json`, `semantic_audit_sample.json`, `build_manifest.json` | Checks, lengths, review, and the reproducible build manifest (input/code/output hashes, counts) |

## 8. Remaining blockers and unverified candidates (571)

| Blocker | Pairs | Path forward |
|---|---|---|
| Subqueries (scalar, `IN (SELECT …)`, correlated) | 216 | Converter support via `$lookup` sub-pipelines, or LLM candidates verified with the same fixtures |
| Unsupported operands/functions (REVERSE, CONCAT, CAST, DATEDIFF, TIME(), POW, …) | ~170 | Extend the converter and the UDFs one function at a time |
| `t.*` / `SELECT *` with joins / non-equi or other joins | ~130 | Join-output shape needs a defined field-naming contract first |
| Fixture not discriminating | 24 | Mostly nonsense or contradictory predicates; leave out |
| Converter unsupported (grouping by CAST) | 4 | Small extension |

Also not recovered:

- **Row-shifted labels.** Re-pairing a prompt with a neighbouring row's SQL would recover some
  pairs, but the pairing would be our inference, not the source's.
- **The 489 ambiguous pairs.** Their labels disagree with their questions.

## 9. Reproducibility and tests

```bash
python scripts/finetune_v1/extract.py           # deterministic; checks raw sha256
python scripts/finetune_v1/verify.py --workers 6 # resumable; skips pairs whose input+code hash is unchanged
python scripts/finetune_v1/build_ft.py           # deterministic; aborts on any leakage violation
python scripts/finetune_v1/verifier_selftest.py --sample 400
python scripts/finetune_v1/token_audit.py
python scripts/finetune_v1/replay_targets.py     # 1,864/1,864 targets pass from their text
python -m pytest -q                              # 203 passed (30 new in tests/test_finetune_v1.py)
```

- **Resumability was checked.** After a clustering-only change, `verify.py` reported "to run 0"
  and the rebuild reused every verification record.
- **Caveat:** clock-relative examples (376 verified) are re-generated against the current
  `$$NOW` whenever they are re-verified. Their verdicts are reproducible; their exact fixture
  rows are not.
- **Tests (`tests/test_finetune_v1.py`, 30) cover:**
  - raw pinning;
  - CSV repair;
  - dialect parsing;
  - grounding (changed values, invented literals, shifted labels, placeholders);
  - converter NULL semantics and validator compatibility;
  - fixture adequacy and determinism;
  - comparator strictness;
  - record integrity and hashes;
  - no reference SQL or results in the model input;
  - dedup and the cap;
  - schema handling;
  - leakage;
  - the benchmark staying unchanged;
  - validation-status consistency;
  - every target passing the read-only validator;
  - key-order preservation of every target (multi-key `$sort` priority);
  - an all-pass replay report for the current train/dev files;
  - build-manifest reproducibility.
