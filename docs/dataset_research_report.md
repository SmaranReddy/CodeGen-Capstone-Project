# Dataset research report: sources for NL→MQL fine-tuning

Researched 2026-10-25. Research phase only:

- No training data, benchmark file, split, score or model-pipeline file was modified.
- Every download went to a session scratch directory outside the repository.
- Nothing from the repository (credentials, `.env`, data) was sent to any external service.
- The 65-item test partition's gold answers were not opened. Its *questions* were used only for the leakage scan (§5.3), the same way `scripts/finetune_v1/extract.py` already does.

## 1. Executive recommendation

**The single best source is one we already have on disk: Spider 1.0 `train_spider` + `train_others`, converted with our own `sql2mql` converter and verified on the *real* Spider databases plus our mutant-adequate fixtures.**

- **Why it is safe:** it is database-disjoint from all 20 benchmark DBs (measured: 0 shared `db_id`s). Its NL↔SQL labels are human-annotated. Its schema style (one collection per table) is the benchmark's. It is the only large source where every example can reach **Level 1 + Level 2** in our own pipeline.
- **Measured converter coverage, before any execution:**
  - 6,361 of 8,659 rows (73%) already convert with the current `sql2mql-v1`.
  - These are 3,483 unique (db, SQL) pairs; most Spider SQL has 2 paraphrased questions.
- **Projected verified yield:** about 4,800–5,700 rows, labelled as a projection (§4.2).

**Second: MongoDB's AptMQL-Bench *train* split (1,762 rows, CC BY-SA 4.0).**

- This is the strongest *native* MQL resource found. It has gold MQL on 10 restorable BIRD-train databases, with result equivalence to BIRD's SQL claimed by the authors.
- It teaches what Spider cannot: nested documents, `$unwind`, access-pattern schemas and much deeper pipelines.
- Prefer the 1,382 rows whose questions also survive BIRD's own quality filter.
- Re-execute every row locally before use.

**Together with the existing ft-v1 set (1,784)**, a realistic verified corpus is **about 8,500–10,500 examples** (projection, §4.2): roughly 5–6× ft-v1 and 30–38× the 280-example verified Spider corpus.

- This is far more in-distribution and harder than ft-v1's 1,504 Kaggle rows. Those are synthetic, single-domain and 66% easy, with an unknown, non-commercial source.
- The large Apache-2.0 synthetic sets (Gretel 106K, SynSQL 2.5M) are **not** recommended as core data:
  - their NL↔SQL alignment is LLM-generated, and we measured that 22% of Gretel's SELECTs fail on their own schema (§3.3);
  - they are useful only as a later, separately gated scale ablation.
- **Overlap is negligible** against the benchmark, ft-v1 train and the RAG corpus: 3 items at ≥0.8, all excluded (§5.3).

**Do not use:**
- **DocSpider dev:** it *is* our benchmark's source split, with 288 exact question overlaps.
- **TEND, mongodb-eai/natural-language-to-mongosh and AptMQL dev:** these are better kept as external out-of-distribution evaluations, and TEND's 11 DBs are exactly AptMQL dev's 11 DBs.
- Several unlicensed or LLM-generated HF/GitHub sets (§2.2).

## 2. Ranked comparison of the strongest candidates

Count definitions:

- **Raw** = rows as published.
- **Candidates** = rows that pass our converter or format gates (measured).
- **Projected verified** = a projection, not a measurement.

Verification-level definitions are those of the task brief (L1 original-DB, L2 multi-fixture, L3 independent review, L4 weak).

| Rank | Source | Raw | Measured candidates | Projected quality-qualified | NL↔SQL label origin | Achievable verification (ours) | Licence | Effort |
|---|---|---|---|---|---|---|---|---|
| 1 | **Spider 1.0 train_spider + train_others** (local) | 8,659 rows / 4,727 unique (db,SQL) / 146 DBs | 6,361 rows / 3,483 unique SQL convert with `sql2mql-v1` | **4,800–5,700** rows; +700–1,500 with converter extensions | Human expert annotation (Spider) | **L1 + L2**; L3 by sampled review | CC BY-SA 4.0 | Medium |
| 2 | **AptMQL-Bench train** (MongoDB, HF) | 1,762 rows / 10 DBs | 1,762 have gold MQL; **1,382** also kept by BIRD's quality filter | **1,300–1,650** after local re-execution | BIRD crowdsourcing, known noisy | L1 *claimed by authors*; we can reproduce L1 locally; L2 not applicable (native schema) | CC BY-SA 4.0 (see §6 note) | Medium (2.5 GB restore, prompt adaptation) |
| 3 | **Spider 1.0 test** (local; 40 DBs) | 2,147 rows / 1,163 unique SQL | 1,481 rows / 806 unique SQL convert | **1,100–1,350** | Human expert | **L1 + L2** | CC BY-SA 4.0 | Low once #1 exists; **costs any future Spider-test claim** |
| 4 | **BIRD train (filtered)** `birdsql/bird23-train-filtered` | 6,601 rows / 69 DBs (of 9,428) | 4,992 pass our SQL analyser (no conversion run yet) | 2,000–3,500 (wide range: large DBs, evidence, ties) | Crowdsourcing, then BIRD's 2025 consistency filter | L1 + L2 | CC BY-SA 4.0 | **High** (~33 GB DBs, dialect quirks, overlaps AptMQL train) |
| 5 | **Existing ft-v1 Kaggle rows** (already built) | 14,815 rows → 1,504 train | 1,504 | 1,504 (already built) | Unknown, noisy | L2 (fixture-equivalence only; no source DB) | **CC BY-NC-SA 4.0** | Done |
| 6 | **Gretel synthetic_text_to_sql** | 105,851 (100,000 train) | 89,757 SELECT; 67,812 execute on their own `sql_context`; 53,684 non-empty | Unknown. Only usable behind grounding, L2 and review gates; plausibly 10–30K | LLM-generated | L2 achievable for SQL↔MQL; **NL↔SQL L4** | Apache-2.0 | Medium–High |
| 7 | **SynSQL-2.5M** (OmniSQL) | 2,544,390 / 16,583 DBs | not measured (9.4 GB JSON) | Unknown | LLM-generated | DBs near-empty (54 MB zip for 16,583 DBs), so L2 only | Apache-2.0 | High |
| 8 | **SParC / CoSQL train** (Spider-train DBs) | ~12K / ~10K annotated SQL (all splits, per project pages) | not downloaded | Few new SQL. Mostly Spider-train SQL re-phrased in dialogue; many turns are context-dependent | Human | L1 + L2 (same DBs as #1) | CC BY-SA 4.0 | Medium |

**Evaluation-only native sets** (kept out of training; §3.1): TEND (1,210), mongodb-eai/natural-language-to-mongosh (766), AptMQL dev (1,424).

## 3. Findings by category

### 3.1 A — Native MongoDB/MQL datasets

| Dataset | Verified facts | Verdict |
|---|---|---|
| **AptMQL-Bench** — [HF](https://huggingface.co/datasets/MongoDB/AptMQL-Bench), [arXiv 2610.02770](https://arxiv.org/abs/2610.02770) | See the field-by-field record below. | **Use train (rank 2). Keep dev for evaluation.** |
| **TEND** — [arXiv 2502.11201](https://arxiv.org/abs/2502.11201) (v3, Jun 2026; ICDE 2027), [GitHub](https://github.com/Jinwei-Lu/Text-to-NoSQL) | 1,210 tasks. 11 MongoDB-native DBs (BIRD-dev domains, the same 11 as AptMQL dev). 269,177 witness documents. Expert-designed nested, sparse, polymorphic and dynamic-key worlds. "Fresh exact MongoDB execution 1,210/1,210". `NLQ` + `NLQ_colloquial` paraphrase. No schema distributed: structure must be induced from documents. Data on Google Drive (not downloaded here). Code MIT; **the data licence is not stated separately** (unverified). v1 of the paper described a different, larger automated dataset; the current release replaces it. | **Evaluation only.** Too small to matter for training (110 per DB). Its value is as an untouched out-of-distribution test of nested-document reasoning, and EvoMQL already reports on it. |
| **mongodb-eai/natural-language-to-mongosh** — [HF](https://huggingface.co/datasets/mongodb-eai/natural-language-to-mongosh) | 766 cases on 8 Atlas sample DBs (305 `sample_guides`). Apache-2.0. Gold `expected.dbQuery` is **gpt-4o-generated** (`generationUuid = referenceAnswers_dataset_gpt-4o…`). It is filtered only for successful execution, non-empty output and "reasonable" output. 340 of 766 are plain `find`. Output is mongosh code, not `{collection, pipeline}`. | **Evaluation only, L4.** Small, LLM-labelled gold, and it is the "EAI in-distribution" benchmark used by EvoMQL ([arXiv 2604.13045](https://arxiv.org/abs/2604.13045)). |
| **DocSpider** — [GitHub](https://github.com/arifgorkemozer/docspider), [NLP journal, DOI 10.1017/nlp.2024.63](https://doi.org/10.1017/nlp.2024.63) | Train 4,043 (139 Spider-train DBs) and dev 620. **Dev = Spider dev: all 20 benchmark DBs and 288 exact question matches with our 455** (measured). MQL is GPT-4-generated mongosh and is accepted down to the "unordered_extra_fields" level: wrong ORDER BY order and extra projected fields still count as ground truth (README). The article is CC BY 4.0 open access, but **the repository has no licence**. | **Reject dev (contamination). Do not copy train targets** (licence unclear, weak acceptance bar). We get the same Spider-train coverage at a higher bar by converting it ourselves (rank 1). Usable only as a comparison point. |
| **M2Q2** — [GitHub](https://github.com/AichaAGGOUNE/M2Q2) | 88,100 rows claimed "hand-annotated". Single `sample_mflix.movies` collection. Sampled rows are templated `db.movies.find({"title": …}, {field: 1})`. No licence. | **Reject:** single collection, template diversity, no licence, `find`-only. |
| **Spider-Mongo** — [GitHub](https://github.com/Cyc-striving/spider_mongo) | Partial preview (≈50 KB train/dev, 5 DBs). Full release pending publication. No licence. Its dev uses Spider dev DBs. | **Reject now.** Revisit if released, and exclude its dev. |
| **channudambal/text-to-mongodb-queries-llm** — [HF](https://huggingface.co/datasets/channudambal/text-to-mongodb-queries-llm) | ~10K LLM-generated rows, no licence. The first row answers "last 12 months" with a hard-coded `ISODate("2024-01-01")`. | **Reject:** no licence, visible semantic errors. |
| Other HF hits — `sublime-security/mql-benchmark` (Sublime's *Message* Query Language), `*mql5*` repos (MetaTrader code), `impacte/mongodb-8.0-eval` (100 developer Q&A, not NL→query), `mongodb-eai/natural-language-to-atlas-search` (Atlas Search `$search`, MIT, data partly internal), `ByteMaster01/synthetic_data_mongoDB` (empty repo) | | **Reject:** wrong task or language. |
| Oza 2024 (RIT) — [PDF](https://cs.rit.edu/~dataunitylab/uploads/publications/Oza2024.pdf) | 56,050 WikiSQL-derived NL→MongoDB pairs (single-table WHERE + aggregates). No public data link. | **Reject:** not released. WikiSQL single-table queries are trivial. |
| EvoMQL / DRO — [arXiv 2604.13045](https://arxiv.org/abs/2604.13045) | A method paper. Code on anonymous.4open.science. No training dataset release found. | Not a data source. |

**AptMQL-Bench, field by field:**

| Field | Finding |
|---|---|
| Contents | 3,186 rows: train 1,762 on 10 BIRD-train DBs, dev 1,424 on 11 BIRD-dev DBs. Fields: `question`, `evidence`, `mongodb_evidence`, `SQL`, `MQL.{collection, aggregation_pipeline}`. Databases ship as `mongodump` BSON (535 files, 2.55 GB). |
| Schema design | Schemas are designed from access patterns (embedded documents), not one table per collection. |
| Construction | Agent pipeline with human-in-the-loop verification. The authors claim result equivalence to the BIRD SQL and lossless migration. |
| Query shape (measured on all 3,186) | `$lookup` 724, `$unwind` 1,203, `$group` 1,094, `$facet` 62, `$setWindowFields` 6. |
| Licence | Card and `ATTRIBUTION.md`: CC BY-SA 4.0, inherited from BIRD. **`LICENSE.md` opens with the CC BY 4.0 (not BY-SA) legal text**, an inconsistency in the release; we treat it as CC BY-SA 4.0, the stricter reading. MongoDB states the artifacts "are not used in any of MongoDB's commercial products". |

**Native-MQL conclusion:** no public native MQL corpus is both large (>5K) and properly licensed and verified. The largest genuine one is AptMQL-Bench (3,186), and only its train half should be trained on. Every other large "NL→MongoDB" set is either LLM-generated without verification, unlicensed, single-domain, or contaminated.

### 3.2 B — SQL datasets convertible with verification

| Dataset | Verified facts | Convertibility (measured) | Verdict |
|---|---|---|---|
| **Spider 1.0** — [project page](https://yale-lily.github.io/spider) (CC BY-SA 4.0 stated on page), [HF card](https://huggingface.co/datasets/xlangai/spider) (cc-by-sa-4.0), [paper](https://arxiv.org/abs/1809.08887) | Local copy in `data/raw/spider1/spider_data/`. train_spider 7,000 (140 DBs), train_others 1,659 (6 DBs: GeoQuery, Scholar, IMDB, Yelp, …), test 2,147 (40 DBs). SQLite files for all 206 DBs (840 MB). Gold SQL is human-written. **0 DB overlap with the benchmark** for all three splits. | See the table below. | **Rank 1 (train + others); rank 3 (test).** |
| **BIRD** — [bird-bench.github.io](https://bird-bench.github.io/) (licence changed to CC BY-SA 4.0 on 2024-04-27), [filtered train](https://huggingface.co/datasets/birdsql/bird23-train-filtered) | Train 9,428 / dev 1,534; 95 DBs, 33.4 GB total. BIRD's 2025 filtered train keeps 6,601 rows on 69 DBs "consistent with schema and faithfully answer the question". It matches the full set's SFT quality on a Qwen2.5-3B (card). Questions carry `evidence` hints. | 4,992 / 6,601 pass our analyser. Rejections are dominated by `IIF`/`CASE` (590/493), subqueries (493) and arithmetic WHERE operands (200). The conversion itself was not run: no DBs downloaded. | **Rank 4: later phase.** Large, heavy, LIMIT-1 tie-prone, and its 10 AptMQL DBs would duplicate rank 2. |
| **SParC / CoSQL** — [sparc](https://yale-lily.github.io/sparc), [cosql](https://yale-lily.github.io/cosql) (both "distributed under the CC BY-SA 4.0 license") | Conversational, on Spider DBs. **Their dev sets use Spider dev DBs, so they are excluded.** Train turns are on Spider-train DBs. | Not downloaded. Most final-turn SQL repeats Spider-train SQL; earlier turns need dialogue context. | **Optional.** Self-contained turns only, as extra paraphrases of rank-1 SQL. Low marginal value. |
| **Spider 2.0-Lite** — [GitHub](https://github.com/xlang-ai/Spider2/tree/main/spider2-lite) (MIT); inspected in `docs/spider2_lite_inspection.md` | Our local file is the first 260-record version; the current official file has 547. Most items target BigQuery/Snowflake cloud DBs. | `docs/conversion_report.md`: the `local*` SQLite subset was converted **by hand**. 68 of these are validated on Atlas against the official gold CSVs (L1) and stored in `data/processed/nl_mql_dataset.jsonl`. At most 135 local items are feasible; cloud items cannot be converted or verified locally. | **Small existing source (68, L1).** It is the mentor's secondary evaluation set, so keep it out of training unless the user decides otherwise. Cloud items are rejected. |
| **KaggleDBQA**, **WikiSQL**, **b-mc2/sql-create-context** (CC BY 4.0; HF card) | KaggleDBQA has 272 examples (evaluation-scale). WikiSQL is single-table, and its HF licence field says "unknown". sql-create-context is built from WikiSQL + **Spider including dev**, with no data rows. | — | **Reject:** too small, too trivial, or contaminated. |

**Spider 1.0 convertibility, measured with `sql2mql-v1` (no execution):**

| Split | Raw rows | Unique (db, SQL) | Analyser-supported rows | **Converted rows / unique SQL** | Main blockers |
|---|---|---|---|---|---|
| train_spider | 7,000 | 3,977 | 5,666 | **5,130 / 2,921** | Set operations 510 and subqueries 509 rows (unsupported). 554 conversion failures, mostly SQLite's "GROUP BY key, SELECT non-key column" (functionally dependent columns). |
| train_others | 1,659 | 750 | 1,246 | **1,231 / 562** | Subqueries 331 |
| test | 2,147 | 1,163 | 1,704 | **1,481 / 806** | Same pattern as train_spider |

### 3.3 C — Synthetic and other sources

| Dataset | Verified facts | Measured | Verdict |
|---|---|---|---|
| **Gretel synthetic_text_to_sql** — [HF](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) (Apache-2.0) | 105,851 rows. 100 domains. `sql_context` holds `CREATE TABLE` + `INSERT` rows. Labels are fully LLM-generated. | On the 100,000 train rows: 89,757 are SELECT/WITH; 71,287 of those have inline rows; **67,812 execute in SQLite on their own context and 19,415 raise errors (21.6%)**; 53,684 return non-empty results. The first sampled row already mismatches NL and SQL ("sorted by salesperson", but the SQL orders by total volume and has an ambiguous GROUP BY column). Complexity: basic 39K, aggregation 22K, single join 15K, subqueries 6K. | **Optional scale ablation, after the core.** SQL↔MQL can reach L2 on our fixtures, but NL↔SQL stays L4. It needs the ft-v1 grounding gates plus a review sample. It is commercially clean. |
| **SynSQL-2.5M** — [HF](https://huggingface.co/datasets/seeklhy/SynSQL-2.5M) (Apache-2.0), [paper](https://arxiv.org/abs/2503.02240) | 2,544,390 quads (DB, question, SQL, CoT). 16,583 synthetic SQLite DBs. `data.json` is 9.4 GB. | The `databases.zip` is 54 MB for 16,583 DBs, so the DBs hold very few rows: L1 against them is nearly vacuous and L2 fixtures would be required. Rows not downloaded. | **Not now.** Only a streamed, filtered sample is worth a later ablation. |
| Programmatic template generation over Spider-train schemas | Our own converter can generate SQL+MQL pairs from templates, but the NL would have to be templated or LLM-written. | — | Not recommended: it adds MQL-syntax coverage, not NL understanding, and it risks template overfitting. |
| Query logs | None found with a usable licence or authorization. | — | Not pursued. |

## 4. Usable, quality-qualified counts

### 4.1 What is measured vs projected

| Source | Raw | Measured gate output | Status today |
|---|---|---|---|
| Spider train_spider + train_others | 8,659 | 6,361 converter candidates (3,483 unique SQL) | **Candidates, not verified** |
| Spider test | 2,147 | 1,481 candidates (806 unique SQL) | Candidates |
| AptMQL train | 1,762 | 1,762 with gold MQL; 1,382 in BIRD-filtered | Authors' L1 claim; **not re-executed by us** |
| BIRD filtered train | 6,601 | 4,992 analyser-supported | Not converted |
| Gretel | 105,851 | 67,812 executable SELECT; 53,684 non-empty | Not converted |
| ft-v1 (existing) | — | 1,504 Kaggle + 280 Spider | Verified at its stated tiers |

### 4.2 Projection for the recommended core (explicitly a projection)

**Spider train + others projection:**

- ft-v1 verified about 98% of eligible items, but that rate does **not** transfer. ft-v1 used single fixtures it designed itself, while Spider adds joins, real dirty data and type mismatches (text-typed numbers, `''` vs NULL, case-sensitive `=`), and LIMIT ties.
- Our own Spider-dev history is the best prior. When the benchmark was expanded, 38 of 493 items (7.7%) were excluded by parity and audit checks.
- Assume 75–90% of candidates verify: **6,361 → 4,800–5,700 rows**.
- Then apply the dedup and near-duplicate cap per SQL skeleton. Spider's two paraphrases per SQL are *kept*, since they are distinct NL, but capped.

**Converter extensions** (each proposed, not implemented):

- `$first` for functionally dependent GROUP BY columns. This is valid only when grouping by a primary key; otherwise SQLite's choice is arbitrary and the item must be rejected.
- `INTERSECT`/`EXCEPT`/`UNION` via `$unionWith`/`$setIntersection`.
- Uncorrelated `IN (SELECT …)` / scalar subqueries via `$lookup` pipelines.
- Together these cover most of the 2,298 unconverted rows. **Projection: +700–1,500.**

**Other sources:**

- AptMQL train: 1,300–1,650 after local re-execution and a BIRD-filter-informed review. The 380 rows BIRD's filter dropped are "needs review", not "accepted".
- Spider test (optional): 1,100–1,350.

**Total recommended (without Spider test):**

- Spider train + others: 5,500–7,200 (4,800–5,700 + 700–1,500 from extensions).
- AptMQL train: 1,300–1,650.
- **New verified rows: about 6,800–8,850.**
- Plus the existing 1,784 ft-v1 rows: **about 8,500–10,500**.
- With Spider test (+1,100–1,350): about 9,700–12,000.

The **first next action** is a 300-item pilot on Spider train, which replaces these ranges with a measured rate (§8 step 0).

## 5. Correctness evidence and remaining uncertainty

### 5.1 Two separate correctness questions

Each training example needs **(a) NL↔SQL label correctness** and **(b) SQL↔MQL translation correctness**. The sources differ sharply on (a):

| Source | (a) NL↔SQL | (b) SQL↔MQL in our pipeline |
|---|---|---|
| Spider | Human expert annotation; the Spider paper describes multiple review passes. Known residual gold errors exist, as in every benchmark. We would *sample-review* (L3) and not claim L3 globally. | **L1:** gold SQL on the original SQLite vs MQL on a parity-checked MongoDB import of the same DB (`parity_check.py` already does per-table multiset parity). **L2:** ft-v1's mutant-adequate fixtures generated from the real Spider schema. |
| AptMQL | BIRD crowdsourcing, which is known to be noisy. BIRD's own filter removed 30% of train, and 380 of AptMQL's 1,762 train questions are among those removed. | L1 claimed by the authors (human-in-the-loop). We can reproduce it on the restored dumps. L2 doesn't apply because there is no relational counterpart for nested schemas, so verification relies on L1 plus review. |
| ft-v1 Kaggle | Unknown provenance; row-shift label noise found and gated. 57/60 manual review correct. | L2 only (no source DB). |
| Gretel / SynSQL | LLM-generated: **L4**. 21.6% of Gretel SELECTs don't even run on their own schema. | L2 achievable |
| DocSpider / mongosh-EAI / channudambal | — | LLM-generated MQL. Weak acceptance (unordered + extra fields, or non-empty-only): **L4** |

### 5.2 Why Level 1 alone is insufficient on Spider train

Many Spider train DBs have only a handful of rows, and some tables are empty. A wrong MQL frequently returns the same (often empty) result as the gold SQL. The verified tier must therefore require all of:

1. L1 equality on the real DB.
2. **Non-empty** gold result, or an explicit empty-result rule plus L2 coverage.
3. L2 equality on ≥3 mutant-adequate fixture variants (every SQL mutant distinguished).
4. Order-sensitive comparison only when the SQL has ORDER BY.
5. Rejection when ORDER BY…LIMIT cuts through a tie (the 0934/0667 lessons in `outputs/rag_week3_clean/…/stability_audit/`).

`data/raw/spider1/testsuite/` covers only the 19 dev DBs, so the official test-suite databases cannot supply L2 for train or test DBs. Our own fixture generator must.

### 5.3 Contamination and leakage

The scan used trigram Jaccard against all 455 benchmark questions (`extract.NearIndex`), on questions only:

| Source | Unique questions | ≥0.8 | Max | Exact matches |
|---|---|---|---|---|
| Spider train_spider | 6,960 | 3 | 1.00 | 2 |
| Spider train_others | 1,626 | 0 | 0.63 | 0 |
| Spider test | 2,142 | 0 | 0.77 | 0 |
| AptMQL (all) | 3,182 | 0 | 0.58 | 0 |
| DocSpider dev | — | — | — | **288 (same 20 DBs: contaminated)** |

- The 2 exact matches are generic questions on *different* DBs. "How many flights do we have?" is `flight_1` vs benchmark-train `flight_2`, and "How many employees are there?" is `driving_school` vs benchmark-test `employee_hire_evaluation`. Their schemas and answers differ, but the rule should be conservative: **exclude the 3 items at ≥0.8**.
**Against the current training corpus** (ft-v1 `train.jsonl`, 1,784 rows; RAG `corpus_v1`, 280 questions):

| Source | vs ft-v1 train: exact / ≥0.8 / max | vs RAG corpus: exact / ≥0.8 / max |
|---|---|---|
| Spider train_spider | 1 / 2 / 1.00 (the same "How many flights do we have?") | 1 / 2 / 1.00 (same) |
| Spider train_others | 0 / 0 / 0.63 | 0 / 0 / 0.63 |
| Spider test | 0 / 0 / 0.71 | 0 / 0 / 0.71 |
| AptMQL train | 0 / 0 / 0.47 | 0 / 0 / 0.47 |

These are covered by the same ≥0.8 exclusion rule.

- No Spider train/others/test DB is a benchmark DB.
- SParC/CoSQL dev, DocSpider dev and Spider-Mongo dev all sit on Spider dev DBs and are excluded.
- **Cross-benchmark collision:** TEND's 11 DBs = AptMQL dev's 11 DBs = BIRD dev. Training on AptMQL dev would forfeit TEND and AptMQL dev as out-of-distribution evaluations.
- Training on Spider test forfeits any future Spider-test claim. It does not affect our Spider-dev benchmark.

### 5.4 Remaining uncertainties

- The real L1+L2 pass rate on Spider (the pilot resolves this).
- How many Spider gold SQL are wrong w.r.t. their question (sampled review).
- AptMQL token length. Its nested schemas may exceed the ft plan's `max_seq 4096` once rendered with sample documents, which needs a `token_audit.py` run.
- The AptMQL dumps' fidelity to the authors' claims (re-execution).
- TEND's data licence.

## 6. Licences and commercial use

| Source | Licence (verified at source) | Attribution | ShareAlike | Commercial model use |
|---|---|---|---|---|
| Spider, SParC, CoSQL | CC BY-SA 4.0 ([spider](https://yale-lily.github.io/spider), [sparc](https://yale-lily.github.io/sparc), [cosql](https://yale-lily.github.io/cosql)) | Yes | Derived *datasets* must stay CC BY-SA | Data licence permits commercial use. Whether model weights are an "adapted material" is legally unsettled; we make no claim either way. |
| BIRD (incl. filtered train) | CC BY-SA 4.0 ([bird-bench](https://bird-bench.github.io/), HF cards) | Yes | Yes | Same as above |
| AptMQL-Bench | CC BY-SA 4.0 (card, `ATTRIBUTION.md`, arXiv page). `LICENSE.md` text is CC BY 4.0, an inconsistency; we treat it as BY-SA. | "derived from BIRD (© BIRD authors)" | Yes | Same as above |
| ft-v1 Kaggle rows | **CC BY-NC-SA 4.0** | Yes | Yes | **Non-commercial only.** A model trained on it should be treated as non-commercial. |
| Gretel, SynSQL, mongosh-EAI | Apache-2.0 | NOTICE / attribution | No | The dataset licence is permissive. The data was LLM-generated (gpt-4o for mongosh-EAI), and **the generators' terms of use were not assessed**. |
| TEND | Code MIT; data licence **unverified** | — | — | Evaluation only, regardless |
| DocSpider, M2Q2, Spider-Mongo, channudambal | **No licence** in repo | — | — | Do not redistribute or train on |

**Rules for the combined corpus:**

- Every record keeps `source`, `source_split`, `source_id`, `licence` and `verification_tier`.
- Sources are stored as separable shards (`by_licence/cc-by-sa-4.0/…`, `by_licence/cc-by-nc-sa-4.0/…`). There is **no single "dataset licence"**: CC BY-SA and CC BY-NC-SA cannot jointly govern one adaptation, as already noted in `fine_tuning_dataset_expansion_report.md`.
- If a commercially usable model is ever needed, train an ablation **without** the Kaggle shard.

## 7. Recommended combination

| Shard | Rows (projected) | Tier | Role |
|---|---|---|---|
| Spider train_spider + train_others via `sql2mql` (v1 + extensions) | 5,500–7,200 | `execution_verified_source_db` + `fixture_equivalence_verified` (L1+L2) | Core, in-distribution |
| AptMQL train (re-executed) | 1,300–1,650 | `execution_verified_source_db` (L1, native) | Nested / native MQL skills |
| Existing ft-v1 (Kaggle 1,504 + Spider-dev-train 280) | 1,784 | as built | Keep. Kaggle can be down-weighted, since it is easy and single-domain. |
| *(optional)* Spider test | 1,100–1,350 | L1+L2 | Only if forfeiting Spider-test claims is acceptable |
| *(later ablation)* Gretel filtered | ≤10K sampled | L2 + L4 alignment | Scale experiment only, kept in its own shard |

**Expected corpus: about 8,500–10,500 verified examples without Spider test (projection).** If the pilot's verification rate is at the low end, expect the lower bound. It is dominated by human-labelled, DB-verified data and covers 146 + 10 new databases instead of ft-v1's single synthetic domain.

**Held-out evaluation, never trained on:**
- the 455 benchmark validation/test partitions;
- an internal DB-grouped dev slice from the Spider-train shard;
- TEND, mongosh-EAI and AptMQL dev as external out-of-distribution sets.

## 8. Ingestion, dedup, conversion, verification and splitting pipeline

Proposed as `scripts/finetune_v2/`. It reuses `scripts/finetune_v1/` modules **by import and does not edit them**, because ft-v1's resumable records hash their code.

0. **Pilot (first action):**
   - 300 random Spider-train items, stratified by Spider difficulty.
   - Measure the L1, L2 and L1∧L2 pass rates and the failure taxonomy, and inspect 50 accepted items by hand.
   - Report the result before scaling.
1. **Infrastructure:**
   - A local `mongo:8.0` container (Docker Desktop is installed, but its daemon was not running during this research).
   - Data lives under a scratch volume, never Atlas: M0 is 512 MB against 840 MB of Spider DBs plus 2.55 GB of AptMQL dumps, and our Atlas user is read-only by design.
   - Pin the image digest and record it in the manifest.
2. **Ingest (immutable):**
   - Copy each source into `data/raw/<source>_v1/` read-only with sha256 pins, as ft-v1 did.
   - Record the licence URL and fetch date.
3. **Import + parity:**
   - Load each Spider SQLite DB into Mongo one collection per table, using the benchmark's exact type and date conventions (the `seed_*` scripts).
   - Run `parity_check.py`-style per-table multiset parity; any table that fails parity is excluded with all its items.
   - For AptMQL, `mongorestore` and count-check against the dump metadata.
4. **Normalise + reference gates:**
   - Lowercase-insensitive identifier resolution against `tables.json`.
   - Drop the 3 items at ≥0.8 benchmark similarity.
   - Exact-dedup on (db, normalised SQL, normalised question).
   - Cluster near-duplicates by (question template × SQL skeleton) and cap them; keep both Spider paraphrases unless they collapse to one loose form.
5. **Convert:**
   - `sql2mql-v1` first, then a versioned `sql2mql-v2` with the §4.2 extensions.
   - Output in the benchmark contract `{"collection", "pipeline"}`, written order-preserving (never `sort_keys`).
   - AptMQL: rename `aggregation_pipeline` → `pipeline` and decode EJSON. Date literals (EJSON `$date`) stay native because the shard's DBs store real dates; keep this shard's prompt schema rendering native.
6. **Verify (accept only if all hold):**
   - L1 equality on the real DB.
   - L2 equality on ≥3 mutant-adequate fixtures.
   - Non-empty gold result (or the documented exception).
   - No LIMIT/tie ambiguity.
   - Validator pass (`eval_kit.validate_mql`).
   - Deterministic re-run equality.
7. **Review:**
   - Stratified manual sample (≥100 Spider, ≥60 AptMQL, with the 380 BIRD-filter-dropped AptMQL items reviewed separately) for NL↔SQL faithfulness.
   - Record per-item reasons. Report the error rate with a confidence interval.
8. **Split:**
   - DB-grouped internal dev (5–10% of Spider-train DBs) for early stopping. The benchmark validation (72) remains the model-selection set and the test (65) stays sealed.
   - Write `leakage_report.json` against the 455 and against TEND, EAI and AptMQL dev.
9. **Emit:**
   - Per-licence shards and the status manifest (accepted / rejected / unverified + reason for every input row).
   - `replay_targets.py`-style replay of every target from its text.
   - Token audit at the training prompt template.
   - Pytest pins on the benchmark and split sha256 values (already present) plus the new manifests.

## 9. Quality, quantity, effort and GPU suitability

| | Quality | Quantity (verified, projected) | Engineering effort | GPU fine-tuning suitability |
|---|---|---|---|---|
| Spider train + others | High: human labels, L1+L2 | 4.8–5.7K (v1 converter); 5.5–7.2K with extensions | Medium: Docker Mongo, import/parity at scale, converter extensions; a CPU-only job, hours not days | Excellent. Same prompt contract and schema style as the benchmark, short prompts; fits 3B/7B QLoRA at `max_seq` 4096. |
| AptMQL train | Medium–high: native gold, BIRD-noisy NL | 1.3–1.65K | Medium: 2.55 GB restore, schema rendering for nested documents, token audit | Good, but longer prompts (avg 7.1 collections/DB, deep pipelines). Check the token budget and possibly cap sample documents. |
| Spider test | High | 1.1–1.35K | Low after Spider train | Excellent |
| BIRD filtered | Medium | 2–3.5K | High (33 GB) | Long prompts with `evidence`. Fine on a Kaggle T4/P100 with care. |
| Gretel (filtered) | Low–medium (L4 NL) | 10K+ | Medium–high plus review | Fine (short), but distribution-shifted. Ablation only. |
| SynSQL | Low–medium (L4) | huge | High (9.4 GB) | Ablation only |

## 10. Official links and citations

- Spider 1.0: https://yale-lily.github.io/spider · https://arxiv.org/abs/1809.08887 · https://huggingface.co/datasets/xlangai/spider
- SParC: https://yale-lily.github.io/sparc · CoSQL: https://yale-lily.github.io/cosql
- BIRD: https://bird-bench.github.io/ · https://arxiv.org/abs/2305.03111 · filtered train: https://huggingface.co/datasets/birdsql/bird23-train-filtered
- AptMQL-Bench: https://huggingface.co/datasets/MongoDB/AptMQL-Bench · https://arxiv.org/abs/2610.02770
- TEND: https://arxiv.org/abs/2502.11201 · https://github.com/Jinwei-Lu/Text-to-NoSQL
- mongodb-eai NL→mongosh: https://huggingface.co/datasets/mongodb-eai/natural-language-to-mongosh · NL→Atlas Search: https://huggingface.co/datasets/mongodb-eai/natural-language-to-atlas-search
- DocSpider: https://github.com/arifgorkemozer/docspider · https://doi.org/10.1017/nlp.2024.63
- EvoMQL (DRO): https://arxiv.org/abs/2604.13045
- M2Q2: https://github.com/AichaAGGOUNE/M2Q2 · Spider-Mongo: https://github.com/Cyc-striving/spider_mongo
- Gretel synthetic_text_to_sql: https://huggingface.co/datasets/gretelai/synthetic_text_to_sql
- SynSQL-2.5M / OmniSQL: https://huggingface.co/datasets/seeklhy/SynSQL-2.5M · https://arxiv.org/abs/2503.02240
- b-mc2/sql-create-context: https://huggingface.co/datasets/b-mc2/sql-create-context
- channudambal/text-to-mongodb-queries-llm: https://huggingface.co/datasets/channudambal/text-to-mongodb-queries-llm
- Oza 2024: https://cs.rit.edu/~dataunitylab/uploads/publications/Oza2024.pdf
- Kaggle: the CLI/API was not usable in this environment (`kaggle` package broken) and `site:kaggle.com` searches returned no NL→MongoDB datasets. Kaggle coverage is therefore limited to the already-ingested NL2SQL set.

**Not verified in this research:**
- TEND data files and their licence (Google Drive).
- SParC/CoSQL exact train counts (not downloaded).
- The SynSQL rows.
- The BIRD train DB contents.

## 11. Shortlist

1. **Spider 1.0 train_spider + train_others, converted and verified by us (L1 + L2).** Largest human-labelled, DB-disjoint, in-distribution source. 6,361 converter candidates measured. Projected verified: 4.8–5.7K with the v1 converter, 5.5–7.2K with extensions. Low risk to the benchmark, CC BY-SA 4.0, fits the existing pipeline and prompt contract.
2. **AptMQL-Bench train.** The only sizeable, properly licensed *native* MQL training data, with real restorable DBs and harder, nested pipelines. Use 1,382 BIRD-filter-consistent rows first and review the other 380.
3. **Spider 1.0 test (same pipeline).** It adds about 1.1–1.35K high-quality rows on 40 more DBs, at the cost of never reporting Spider-test results. Choose it over Gretel/SynSQL: those offer scale and Apache-2.0 licensing, but only L4 NL alignment.

**Choice for the capstone: option 1, with option 2 added once the pilot passes.**

- Option 1 is the only path that makes the corpus *much* larger while every example stays checked against an original database and adversarial fixtures.
- It reuses code already tested in this repository (`sql2mql`, `fixtures`, `parity_check`), and it needs no API spending.
- It matches the benchmark's task format, so gains are attributable to data rather than format adaptation.
- AptMQL train then adds the nested-document skills the Spider shard cannot teach.
- The pilot (§8 step 0) is the gate. It turns the projected 75–90% verification rate into a measured one before any large conversion or fine-tuning run.
