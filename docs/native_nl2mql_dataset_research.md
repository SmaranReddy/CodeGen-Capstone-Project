# Native NL→MQL datasets: research report

Researched 2026-10-25. Research only.

- No benchmark file, split, dataset, converter or model-pipeline file was changed. This report is the only file added.
- Every download went to a session scratch directory outside the repository. Analysis scripts lived in a separate directory and ran with `python -I`.
- Nothing from the repository (data, credentials, `.env`) was sent to any external service.
- The 65-item test partition's gold answers were not opened. Leakage checks used only the `question` and `db_id` fields of all 455 benchmark items, as in `docs/dataset_research_report.md` §5.3.

This report builds on `docs/dataset_research_report.md` (the "prior report") and does not repeat its findings. Where the prior report already verified a fact (AptMQL-Bench, mongosh-EAI, DocSpider, M2Q2, Spider-Mongo, channudambal, Oza 2024), it is cited here, not re-derived. Everything else is new in this report. Two of the new findings change the picture: **TEND-v1** (an executable mirror of the original 17,020-pair TEND release) and **SM3-Text-to-Query** (native MQL written by hand, with a MongoDB loader).

Label conventions: **measured** = computed here from the downloaded files; **stated** = the authors' claim; **projected** = an estimate.

---

## 1. Executive summary

**Is there a usable native NL→MQL dataset we can download and train on?** Yes, but no single one is large, clean and clearly licensed at the same time. Three genuine native sources can be downloaded, and each has real database contents:

| | Native gold MQL | Usable size (measured, before re-execution) | Licence | Main limitation |
|---|---|---|---|---|
| **AptMQL-Bench train** (MongoDB) | Agent pipeline plus human-in-the-loop; result equivalence to BIRD SQL is stated | 1,762 rows on 10 DBs | CC BY-SA 4.0 | Small. BIRD-crowdsourced NL is noisy. |
| **TEND-v1** (Lu et al., 2025; mirrored in `Moon0256/text-to-nosql`) | LLM-generated (GPT-4o / GPT-4o-mini), kept only if its execution result matched the Spider SQL | **3,001 records / 15,005 NL on 135 DBs**, after removing all of our benchmark DBs | **Not stated for the data** (see §5) | Licence unclear. 19 of our 20 benchmark DBs are inside the full release and must be dropped. The SQL is 100% Spider-train SQL. |
| **SM3-Text-to-Query** (Sivasubramaniam et al., NeurIPS 2024 D&B) | Written by hand per template and checked by a second expert (stated) | 5,760 rows; 4,476 unique; **388 templates / ~234 MQL skeletons** | Apache-2.0 (repo and Synthea) | One medical domain, 4 collections, templated. No `$sort`/`$limit`/`$count` at all. |

**Bottom line:**

- Native data can add roughly **3,200–5,250 quality-qualified examples with clean licences** (AptMQL + SM3), or **5,700–8,100 if the TEND-v1 licence is cleared** (projection, §6.3).
- That is useful but not "substantially larger" than the Spider-train conversion route. That route remains the only path to 5K+ in-distribution examples.
- The most important new finding is **TEND-v1**:
  - It holds a native, nested-schema MQL target for about **3,000 Spider-train SQL queries** on 135 non-benchmark DBs.
  - Its main value is as **native training data**, but only if its unstated data licence can be cleared.
  - It **cannot** serve as an independent execution oracle for our converter. TEND's gold was kept only when its result matched the Spider SQL result on the same data, so comparing against it reduces to the L1 check we already plan (§9).
  - It is useful for **static triage** of `sql2mql` output and as a **reference for `sql2mql-v2`** subquery and set-operation rules.
  - **Warning:** the full release contains gold for 19 of our 20 benchmark DBs (§6.1). It must never enter training, RAG retrieval or few-shot pools without the DB exclusion.

**No substantial native dataset exists that is at once:** larger than 5K unique queries, multi-domain, clearly licensed, and executable with verified gold. Every larger "NL→MongoDB" set we found has at least one of these problems: no licence, LLM-generated without any execution check, a single domain or template, or contamination with our benchmark (§3).

---

## 2. Search coverage

| Channel | What was searched | Result |
|---|---|---|
| Web/Scholar/arXiv | "text-to-MQL", "text-to-NoSQL", "natural language to MongoDB", "NL2Mongo", "MongoDB aggregation generation", "multilingual NoSQL benchmark", SM3 | AptMQL-Bench, TEND (v1 & v3), MultiTEND, SM3, DocSpider, EvoMQL/DRO, Oza 2024, "Text-to-NoSQL via SQL" (Moon0256) |
| arXiv full text (read) | 2502.11201 **v1** and **v3** (TEND), 2502.11022 (MultiTEND), 2411.05521 (SM3) | §4 facts. AptMQL 2610.02770 was read in the prior report. |
| Hugging Face API | `/api/datasets?search=` with `mongodb`, `mongo`, `mql`, `nosql`, `mongosh`, `text2mongo`, `nl2mongo`, `text-to-mongo`, `nl2mql`, `mongo query` (≈170 hits; Mongolian-language and MQL5 noise excluded) | Native NL→query sets: AptMQL-Bench, mongodb-eai/natural-language-to-mongosh, natural-language-to-atlas-search, channudambal (10,000), skshmjn/mongo_prompt_query (411), kylemesh19/cleaned-mongosh-instructions (218), infinite-dataset-hub/NosqlQuerySolutions (100), Xelta/response_mongo_text (Spanish chatbot), jmorenas (empty). The rest are embeddings/RAG corpora. |
| GitHub (`gh search repos`, tree listings) | text-to-mongodb, nl2mongo, text-to-nosql, text2mql, nl-to-mql, "spider mongodb", MultiTEND | `Jinwei-Lu/Text-to-NoSQL` (TEND v3), **`Moon0256/text-to-nosql` (TEND-v1 mirror)**, `Moon0256/Text-to-NoSQLwithSMART` (TEND sample), `jf87/SM3-Text-to-Query`, `diwanshuydv/NL2Mongo` (3,889 rows via Git-LFS), `paulc-1212/nl2mongo-t5` (1,138 `find` rows), `Other-Now/nl2mongo-eval` (small eval set). About 30 others are apps or demos without data. |
| Kaggle | Web search, including one restricted to `kaggle.com` (the `kaggle` CLI in this environment is still broken; prior report) | No NL→MongoDB dataset found. The only Kaggle hit was a translation corpus exported *from* MongoDB. |
| Papers With Code | `paperswithcode.com` search/dataset URLs | Every URL now returns HTTP 302 to Hugging Face (`/papers/trending`, `/datasets/xlangai/spider`). PwC no longer serves listings, so it was covered through the HF sweep. |
| Other papers surfaced | arXiv 2602.12574 (Stage-MCTS, HKUST): **CoNoSQL**, a conversational Text-to-NoSQL set ("2,000+ dialogues across 150 databases") | No code or data link on arXiv; no GitHub/HF repository found. **Not released.** |
| MongoDB developer resources | mongodb.com research blog, docs "natural language to MongoDB", HF `MongoDB/*` and `mongodb-eai/*` orgs | Only AptMQL-Bench and the mongosh/Atlas-Search evaluation sets are NL→query. `mongodb-eai/code-examples` (Apache-2.0, 100K+) contains docs code snippets, not NL questions paired with queries. |

**Not reachable or not released:**
- MultiTEND data ("will be released"; no link).
- TEND-v1's original OneDrive "partial dataset" link (not fetched; the GitHub mirror was used instead).
- Spider-Mongo full release (pending, prior report).
- Oza 2024 data (no link).

---

## 3. Ranked list of credible native candidates

### Tier A: native question→MQL pairs with schemas and database contents

| Rank | Dataset | Records (measured) | Unique queries | DBs / domain | Gold origin | Executable locally | Licence |
|---|---|---|---|---|---|---|---|
| A1 | **AptMQL-Bench train** — [HF](https://huggingface.co/datasets/MongoDB/AptMQL-Bench), [arXiv 2610.02770](https://arxiv.org/abs/2610.02770) | 1,762 (train) | ≈1,762 | 10 BIRD-train DBs, access-pattern (embedded) schemas | Agent pipeline + human-in-the-loop; result equivalence to BIRD SQL stated | Yes, `mongodump` 2.55 GB | CC BY-SA 4.0 (inconsistency in `LICENSE.md` noted in prior report §3.1) |
| A2 | **TEND-v1** — paper [arXiv 2502.11201v1](https://arxiv.org/abs/2502.11201v1); data mirror [Moon0256/text-to-nosql](https://github.com/Moon0256/text-to-nosql) `data/benchmark/tend/` | 2,849 train + 555 test records × 5 NL = 17,020 NL (matches the paper's Table 1 exactly). **After dropping our benchmark DBs: 2,458 + 543 = 3,001 records, 15,005 NL.** | 2,457 + 542 unique (db, SQL); unique MQL ≈ same | 154 DBs in total (135 kept), all Spider; nested FK-embedding schemas | LLM-generated from SQL *execution results* and kept only if the MongoDB result equalled the SQLite result (stated). Paraphrases from 3 LLMs. | Yes: `mongodb_data/*.json` (189 MB), `mongodb_schema/`, `load_data.py`, docker-compose | **Data licence not stated** (§5) |
| A3 | **SM3-Text-to-Query** — [GitHub jf87/SM3-Text-to-Query](https://github.com/jf87/SM3-Text-to-Query), [arXiv 2411.05521](https://arxiv.org/abs/2411.05521) | train 6,000 + dev 2,000 (test 2,000 withheld) | train 3,978 unique MQL strings; 234 MQL skeletons; 405 templates | 1 synthetic medical DB (Synthea, 110 patients); 4 top-level collections with deep embedding | Hand-written per template and checked by a second expert (stated). Values filled automatically from the DB. | Yes: Synthea CSVs + `src/docker_configuration/standard/mongodb/setup-mongodb.py` | Apache-2.0 (repo `LICENSE`; Synthea `LICENSE` is also Apache-2.0) |

**Kept as evaluation-only** (prior report, re-confirmed here):
- **TEND v3** (1,210 tasks, 11 BIRD mini-dev DBs). The README now states the dataset licence is CC BY-SA 4.0, which settles the prior report's "unverified".
- **mongodb-eai/natural-language-to-mongosh** (766; gpt-4o gold).
- **AptMQL dev** (1,424). It shares its 11 DBs with TEND v3.

### Tier B: native pairs without databases or executable ground truth

| Dataset | Size (measured) | Notes | Verdict |
|---|---|---|---|
| `diwanshuydv/NL2Mongo` (`data/data_v6.csv`, Git-LFS) | 3,889 rows; 3,220 unique queries; 22 schemas | All 22 DBs are CCTV/surveillance (facial recognition, person tracking). JSON-Schema `bsonType` schemas, **no data**. 2,736 `find`, 1,113 `aggregate` (944 `$group`, 61 `$lookup`). Provenance undocumented; the uniform phrasing suggests LLM generation. MIT. | **Not core.** Executable only on synthetic fixtures we would have to build, and the NL↔MQL alignment is unverifiable (L4). Its single sensitive domain adds little. At most a later ablation. |
| `channudambal/text-to-mongodb-queries-llm` | 10,000 | No licence; visible semantic errors (prior report) | Reject |
| DocSpider train | 4,043 | GPT-4 mongosh, weak acceptance bar, **no licence**; its dev split = our benchmark source (prior report) | Reject |
| M2Q2 | 88,100 | One collection, templated `find`, no licence (prior report) | Reject |
| `skshmjn/mongo_prompt_query` / `kylemesh19/cleaned-mongosh-instructions` | 411 / 218 | Schema + prompt + query, no data; Apache-2.0 / no licence | Too small; reject |
| `paulc-1212/nl2mongo-t5` | 1,138 | `find` only, one DB, linearized | Reject |
| `infinite-dataset-hub/NosqlQuerySolutions` | 100 | LLM-generated | Reject |

### Tier C: native NoSQL pairs for other databases

- **SM3's Cypher/SPARQL** splits and the **Neo4j Text2Cypher** dataset are native, but only for graph languages. Converting them to MQL would bring back exactly the translation-correctness problem we are trying to avoid, and the SM3 MQL split already exists. **Nothing in Tier C is worth adapting.**
- **MultiTEND** (ACL Findings 2025) is TEND-v1 translated into 5 more languages (101,789 pairs stated, but its own size figures are inconsistent). It is not released and is not English, so it is not relevant for us.

### Tier D: fallback (SQL and synthetic)

This is unchanged from the prior report: Spider 1.0 train + others through `sql2mql`, verified with L1 + L2. That means 6,361 converter candidates measured and 4.8–7.2K verified rows projected. Gretel and SynSQL remain ablation-only. §9 explains how TEND-v1 changes the way this fallback should be run.

### 3.1 Consolidated evidence table

All counts are measured here, except AptMQL (measured in the prior report) and the "Usable" column, which is projected (§6.3).

| | AptMQL-Bench train | TEND-v1 (benchmark DBs removed) | SM3 train (+dev) | diwanshuydv/NL2Mongo |
|---|---|---|---|---|
| Records / unique questions / unique queries | 1,762 / ≈1,762 / ≈1,762 | 3,001 / 15,005 NL (5 per record) / ≈2,999 | 6,000 / 4,600 / 3,978 (+2,000 / 1,789 / 1,648 dev) | 3,889 / 3,884 / 3,220 |
| Schemas + DB contents | Yes (dumps, 2.55 GB) | Yes (JSON docs + per-DB schema files) | Yes (Synthea CSV + loader) | Schema only, no data |
| Gold origin | Agent + human-in-the-loop | LLM, execution-filtered against the SQL | Hand-written templates, expert-checked | Undocumented (likely LLM) |
| Executable locally | Yes | Yes | Yes (MQL results never stored by the authors) | No |
| Query shapes | `$lookup` / `$unwind` / `$group` heavy; `$facet`; `$setWindowFields` | aggregate 84%, find 16%; `$sort`/`$limit`/`$count`/`$size`/`$expr`/`$elemMatch` | `$unwind` + `$match` + `$group` (DISTINCT); `$lookup` 5%; no `$sort`/`$limit` | find 70%, `$group` 24%, `.sort`/`.limit`, dates |
| Domain diversity | 10 DBs (BIRD) | 135 DBs (Spider) | 1 (medical) | 1 (CCTV/surveillance) |
| Measured noise | BIRD filter dropped 380/1,762 questions | 30-record review: 1–2 wrong (CI 2–21%); paraphrase drift seen | 39 rows with a template bug; 30-row review: 0 wrong, 2 arguable | Not assessable (no DB) |
| Duplicates / templating | Low | 5 NL per query (LLM paraphrases) | 405 templates, ~234 MQL skeletons; dev reuses 230/244 skeletons | 22 schemas, about 175 rows each |
| Leakage vs our benchmark | 0 ≥0.8 | **403 records on 19 benchmark DBs (removed)**; 5 NL ≥0.8 remain | 0 | Not checked (disjoint domain) |
| Licence / training + redistribution | CC BY-SA 4.0: yes, with attribution + SA | **Not stated: no until clarified** | Apache-2.0: yes | MIT, but provenance unknown |
| Preprocessing effort | Medium (restore, EJSON, token audit) | Medium (mongosh parser, find→pipeline, re-exec) | Medium (mongosh parser, trailing commas, long nested schema) | High (fixtures needed) |
| Usable (projected) | 1,300–1,650 | 2,500–2,850 records | 1,900–3,600 (cap 5–10) | 0 in the core |

---

## 4. Example records (verbatim from the downloaded files)

**AptMQL-Bench (train, `works_cycles`):**
```
Q:   What is the average standard cost of product number CA-1098?
MQL: {"collection": "products", "aggregation_pipeline": [
       {"$match": {"productNumber": "CA-1098"}},
       {"$lookup": {"from": "productCostHistory", "localField": "_id", "foreignField": "productId", "as": "costHistory"}},
       {"$unwind": "$costHistory"},
       {"$group": {"_id": null, "avg_standard_cost": {"$avg": "$costHistory.standardCost"}}},
       {"$project": {"_id": 0, "avg_standard_cost": 1}}]}
```

**TEND-v1 (train, record 3999, `soccer_2`; `Tryout` is embedded in `College`):**
```
NL[0]: What is the total number of enrollment of schools that do not have any goalie player?
NL[1]: What is the total number of students enrolled in schools without any goalies?   (+3 more paraphrases)
SQL:   SELECT sum(enr) FROM college WHERE cName NOT IN (SELECT cName FROM tryout WHERE pPos = "goalie")
MQL:   db.College.aggregate([{ $match: { "Tryout.pPos": { $ne: "goalie" } } },
                             { $group: { _id: null, sum_enr: { $sum: "$enr" } } },
                             { $project: { _id: 0, sum_enr: 1 } }]);
```
Array semantics make `$ne` on an embedded array mean "no element equals". That is a correct NOT IN. This is the kind of document reasoning our flat Spider shard cannot teach.

**TEND-v1, a defect found in review (record 3435, `store_product`):**
```
NL[0]: Find the list of page size which have more than 3 product listed
NL[3]: Identify the highest page size among products that each have over three listings.   <- paraphrase drifted to fit the wrong MQL
SQL:   SELECT max_page_size FROM product GROUP BY max_page_size HAVING count(*) > 3
MQL:   ... {$match: {count: {$gt: 3}}}, {$group: {_id: null, max_page_size: {$max: "$_id"}}} ...   <- returns one value, not the list
```

**SM3 (train, template "What kind of allergy has the code $allergy_code?"):**
```
Q:   What kind of allergy has the code 264287008?
MQL: db.patients.aggregate([{ $match: {"ENCOUNTERS.ALLERGIES.CODE": 264287008} }, { $unwind: "$ENCOUNTERS" },
       { $unwind: "$ENCOUNTERS.ALLERGIES" }, { $match: {"ENCOUNTERS.ALLERGIES.CODE": 264287008} },
       { $group: { _id: "$ENCOUNTERS.ALLERGIES.TYPE" } }, { $project: { _id: 0, type: "$_id" } }])
SQL: SELECT DISTINCT type FROM allergies WHERE code='264287008';
```

**SM3, a systematic gold bug (39 train rows, 2 templates):**
```
Q:   Please provide me the provider associated with the claim of ID e9dc5b0c-c73d-e1be-0e11-7215c1a85c1d.
MQL: ... {$unwind: "$CLAIMS"}, ... {$unwind: "$CLAIMS"}, ...
     {$match: {"CLAIMS.CLAIM_TRANSACTIONS.CLAIM_TRANSACTION_ID": "1d29df54-f9ec-b8dd-eedb-82c59e299ecc"}}   <- hard-coded ID that is not in the question
```

---

## 5. Download and licensing status

| Source | Downloaded here | Licence (verified at source) | Train + redistribute a model? |
|---|---|---|---|
| AptMQL-Bench | Card/rows via HF API; dumps not re-downloaded (prior report) | CC BY-SA 4.0 (card, `ATTRIBUTION.md`); `LICENSE.md` text is CC BY 4.0, so treat it as BY-SA | Attribution + ShareAlike for derived *datasets*. Whether model weights are "adapted material" is legally unsettled; we make no claim. |
| TEND-v1 (Moon0256 mirror) | Yes: `git clone` (818 MB with predictions) | **Unresolved.** The v1 paper states no data licence; its only licence text is the VLDB boilerplate (CC BY-NC-ND 4.0) on the *paper*. The mirror's BSD-3 licence explicitly covers only the repo's own code ("Except for separately identified third-party materials"), so it does **not** license the TEND data. The underlying Spider data is CC BY-SA 4.0. Model involvement, per the v1 paper: the gold MQL came from GPT-4o (CoT exemplars) plus a second-tier generator (the text says GPT-4o-mini, the Figure 2 caption says GPT-3.5); the paraphrases came from gpt-3.5-turbo-16k, gpt-4o and claude-3-sonnet. We have not assessed those providers' output terms. TEND v3's CC BY-SA 4.0 covers only the v3 release. | **Not until clarified.** Ask the TEND authors whether the v1 data is released under CC BY-SA 4.0 like v3. Until then, the only low-risk use is *reading* it as a static reference while writing converter rules, which copies no labels. Training on it and redistributing a model is not low-risk. |
| SM3 | Yes: `git clone` (569 MB) | Apache-2.0 (repo `LICENSE`); Synthea `LICENSE` Apache-2.0; synthetic patients, no privacy implications (stated) | Yes, with attribution/NOTICE. This is the cleanest licence of all native sources. |
| diwanshuydv/NL2Mongo | Yes (one 8.7 MB LFS CSV) | MIT (repo); provenance of rows undocumented | Licence permits; the provenance gap makes it L4 |
| TEND v3, mongosh-EAI, AptMQL dev | Not re-downloaded | CC BY-SA 4.0 / Apache-2.0 / CC BY-SA 4.0 | Evaluation only, by our own rule |

---

## 6. Quality-qualified example estimates

### 6.1 Measured gates

**TEND-v1:**

- **Contamination, measured:**
  - The train split contains **18 of our 20 benchmark DBs (391 records)**. These are Spider dev DBs: `flight_2`, `world_1`, `concert_singer`, `car_1`, `pets_1`, `dog_kennels`, … The test split contains a 19th (`voter_1`, 12 records).
  - 353 NL strings on those DBs score ≥0.8 trigram similarity to a benchmark question, many of them exact.
  - Because our 318/72/65 split is database-disjoint, **every record on those DBs must go**, not just the near-duplicates: they expose the schemas and data of the sealed test DBs.
- **After exclusion:**
  - 3,001 records on 135 DBs, with 3 + 2 NL still ≥0.8 against the benchmark (generic questions on other DBs; drop them).
  - 1 record is ≥0.8 against ft-v1 train.
- **Provenance, measured:** all 2,458 kept train and 543 kept test records have SQL that exactly matches a Spider *train* (db, SQL) pair. The removed 391 + 12 are exactly Spider *dev* SQL.
  - That contradicts the v1 paper's description of its source as "the training set of Spider" (as read by the fetch agent). Anyone who evaluates on Spider dev after training on TEND-v1 is contaminated.
  - TEND-v1 adds **no new SQL semantics**. What it adds is a native MQL target and four LLM paraphrases for SQL we already hold.
- **Operator coverage (2,849 train):**
  - Stages: `aggregate` 2,387, `find` 462; `$project` 2,234, `$unwind` 1,482, `$group` 1,319, `$match` 1,274, `$sort` 561, `$limit` 406, `$lookup` 368, `$count` 275.
  - Operators: `$size` 263, `$ne` 130, `$expr` 91, `$regex` 63, `$elemMatch` 33.
  - 378 of the 3,001 kept records (310 in train) come from set-operation or subquery SQL, a class `sql2mql-v1` cannot convert.

**SM3:**

- **Gates, measured:**
  - 201 unanswerable rows are dropped (useful later as refusal training, but not MQL).
  - 39 rows from the two buggy claim→provider templates are dropped.
  - That leaves 5,760 rows; exact (question, MQL) dedup leaves **4,476** across 388 templates.
- **Template caps:** 3 per template → 1,164; 5 → 1,939; 10 → 3,595.
- **Operator coverage (train):**
  - Stages: `aggregate` 4,454, `find` 1,345; `$unwind` 4,232, `$group` 2,462 (mostly for DISTINCT), `$lookup` 308, `$sum` 74, `$avg` 33.
  - **Zero `$sort`, `$limit`, `$count`.**
- **Leakage:** the dev split reuses 230 of its 244 MQL skeletons from train, so SM3 dev is *not* an independent test. SM3 has 0 questions ≥0.8 against the benchmark (first 2,000 unique checked; the domain is disjoint).
- **No stored MQL execution:** `processed_train.csv` stores SQL, Cypher and SPARQL results, but `mql_results` is `nan` for all 5,799 answerable rows. The authors never recorded gold MQL execution.

### 6.2 Label-noise evidence

| Evidence | TEND-v1 | SM3 |
|---|---|---|
| Manual review, 30 random records (this session) | 26 correct. **1 wrong** (record 3435: `$max` instead of a list). **1 likely wrong** (wedding: counts matching `people` documents, not weddings). 2 data-dependent (`$size: 0` vs a missing array). The wrong-MQL rate is about 1–2/30, i.e. 3–7% (Wilson 95% CI for 2/30: 1.8–21%). One paraphrase drifted to match the wrong MQL. | 30 template-distinct rows: 28 correct, 2 arguable (set vs bag, i.e. DISTINCT vs row count), 0 wrong (95% upper bound 11%). Separately, one sampled **SQL** is wrong (template "reason for procedure" selects `stop`), but its MQL is right. |
| Independent cross-check already on disk | Moon0256's deterministic UnityJDBC SQL→MQL translator, run on gold SQL against TEND data, **agrees with TEND gold MQL on 538/555 test records (96.9%)** under order-aware multiset comparison (`results/evaluation/enhanced/mongotranslator_gold_sql_metrics.log`). I inspected all 17 disagreements: most are translator failures (empty output, wrong source collection, broken `$lookup`). A few are bag-vs-set ambiguity. **This is weak evidence of semantic correctness.** TEND's generation filter already forced each gold result to equal the SQL result on this same data, so the agreement mostly shows that the gold re-executes reproducibly. Wrong queries that return the same rows on tiny DBs pass both (record 3435 did). **The 30-record review is the real noise estimate.** | None. The second-expert check is stated, not evidenced. |

### 6.3 Projection of usable examples

| Source | Measured candidates | Projected usable (counted by unique query) | Reasoning |
|---|---|---|---|
| AptMQL train | 1,762 | 1,300–1,650 | Prior report §4.2 |
| TEND-v1 (excluding benchmark DBs) | 3,001 records / ~2,999 unique (db, SQL) / 15,005 NL | **2,500–2,850 records** after local re-execution against the Spider SQLite gold and mongosh→JSON parsing loss. Train on the original Spider question plus at most one checked paraphrase, giving **≈5,000–5,700 NL rows**. | Allows for 3–7% wrong gold, the ≥0.8 drops and about 2% parse loss. Spider DBs are small, so re-execution will not catch every error. Only if the licence is cleared. |
| SM3 | 4,476 unique rows / 388 templates | **1,900–3,600** (template cap 5–10) | The cap prevents a 4-collection medical schema from dominating. The effective diversity is ~234 query skeletons. |
| diwanshuydv | 3,889 | 0 in the core | L4, no DB |

**Native total, counted by record (one NL per unique query; TEND-v1 paraphrases are not counted):**
- **≈ 5,700–8,100** = 1,300–1,650 (AptMQL) + 2,500–2,850 (TEND-v1) + 1,900–3,600 (SM3). TEND-v1 needs a licence answer first.
- Without TEND-v1: **≈ 3,200–5,250**.

---

## 7. Accuracy and execution-verification limitations

1. **No source has been re-executed by us.** AptMQL's equivalence, TEND-v1's execution filter and SM3's expert check are all *stated*. The UnityJDBC agreement (96.9%) is third-party and covers TEND-v1 *test* only. Docker was not running and no local `mongod` exists. All three ship loaders, so re-execution is reproducible, not blocked.
2. **Execution success ≠ semantic correctness.**
   - TEND-v1's generation filter compared MongoDB output to SQLite output on Spider's tiny DBs. Wrong queries can pass that filter, and record 3435 shows one did.
   - SM3 queries are mostly single-entity lookups by ID. They return non-empty results by construction but test very little logic.
3. **NL quality.**
   - TEND-v1's four extra NL per record are LLM paraphrases, and at least one drifted toward its (wrong) MQL. Paraphrases must be checked against the SQL, not the MQL.
   - AptMQL's NL is BIRD-crowdsourced: 380 of its train questions were dropped by BIRD's own filter (prior report).
   - SM3 NL is templated (405 templates).
4. **Schema-representation mismatch.** Our benchmark is one collection per table. AptMQL, TEND-v1 and SM3 all use embedded documents. That is valuable for skill transfer, but gains on the Spider-dev benchmark may be smaller than the row counts suggest, and prompt rendering must state the nesting explicitly.
5. **Ordering and ties.** TEND-v1 has 406 `$limit` records. The tie-cutting lesson from Week 3 (0934/0667) applies: reject any record whose ORDER BY…LIMIT cuts through a tie.

---

## 8. Compatibility with the Qwen fine-tuning pipeline

| Issue | AptMQL | TEND-v1 | SM3 |
|---|---|---|---|
| Target format vs our `{"collection", "pipeline"}` | Rename `aggregation_pipeline` → `pipeline`; decode EJSON | **mongosh strings** with unquoted keys, trailing `;`, `find(filter, proj).sort().limit()` chains (462 `find`). They need a mongosh→JSON parser and a find→pipeline rewrite, done order-preserving (never `sort_keys`; see the ft-v1 fix). | Same, plus trailing commas in arrays/objects (1,345 `find`). |
| Schema for the prompt | Rendered from dumps | `mongodb_schema/<db>.json` provided per DB | Must be induced from the loader/documents. The `patients` document nests ~15 entity types, so **run `token_audit.py`** against the 4,096 `max_seq` in `fine_tuning_plan.md`. |
| Collection naming | Native | Case differs from Spider (`College`, `EMPLOYEE`), which is native to its own DB | Native |
| Fit to `train_qlora.py` | Good (prior report) | Good; prompts are short | Risk of long prompts |

Every shard needs a `schema_style: embedded` tag. Our benchmark items are `flat`. That allows an ablation (train with and without the embedded shards) to show whether the native data helps the Spider-dev benchmark or only general MQL skill.

---

## 9. Recommended combination and deduplication

**Dedup keys:**

1. **Spider-derived shards** (TEND-v1, our planned `sql2mql` Spider-train shard, SParC/CoSQL if ever used): key on `(db_id, normalised SQL)`. A TEND-v1 record and a `sql2mql` record for the same key are **one supervision unit**, not two. Count them once in every reported total.
   - If both are kept, they are two *views* (embedded vs flat schema) of one question. Place them in the same internal-dev partition, grouped by `db_id`, so a view never leaks across the split.
2. **Within each shard:** exact-dedup on (normalised question, normalised MQL). Near-duplicates at trigram ≥0.8 are capped per SQL/MQL skeleton (TEND-v1: ≤2 NL per record; SM3: ≤5–10 per template).
3. **Against the benchmark:** drop all records on the 20 benchmark DBs, then drop any NL ≥0.8 to a benchmark question. Already measured: 3 + 2 TEND-v1 NL and 0 SM3.
4. **Cross-source:** AptMQL (BIRD), TEND-v1 (Spider) and SM3 (Synthea) share no DBs, so no cross-key is needed.

**Combination (projected):**

| Shard | Rows | Role | Condition |
|---|---|---|---|
| Spider-train via `sql2mql` + L1/L2 (prior report rank 1) | 4,800–7,200 | Core, flat schema, in-distribution | Pilot passes, after the audit's §9 fixes. TEND-v1 serves as a triage flag only (below). |
| AptMQL train | 1,300–1,650 | Native embedded, harder pipelines | Local re-execution |
| SM3 (capped) | 1,900–3,600 | Native embedded `$unwind`/`$lookup` over deep nesting; Apache-clean | Parser + token audit; downweight or cap |
| TEND-v1 (native view) | 2,500–2,850 records | Native embedded view of Spider SQL | **Only after a licence answer**; otherwise a static rule/triage reference only |
| ft-v1 (existing) | 1,784 | As built | Unchanged |

**What TEND-v1 can and cannot do for converter verification.** For a shared (db, SQL) key, a three-way check would compare:

- (a) the Spider SQL on SQLite;
- (b) our `sql2mql` MQL on the flat import;
- (c) TEND-v1's MQL on TEND's nested data.

TEND kept (c) only if (c) = (a) on that same data, so (a) = (b) = (c) **reduces to (a) = (b), which is L1**. That adds no discriminating power and cannot catch a wrong translation that happens to return the right rows on tiny DBs. TEND-v1 is therefore **not** an independent oracle. Its honest uses:

1. **Static triage.** Where the stage structure of (b) and (c) differs on a semantic point (DISTINCT handling, grouping key, join direction, empty-group behaviour), flag the item for manual review. This targets exactly the audit's B1/B2 and DISTINCT+GROUP BY classes, without claiming verification.
2. **A reference for `sql2mql-v2`.** 378 kept records have subquery or set-operation SQL. Their MQL targets a *nested* schema, so it is not a flat-schema label, but it shows working patterns (`$size`, `$ne` on arrays, `$lookup` pipelines) for the rules the converter lacks.
3. **The costly option that would add real power:** migrate our L2 mutant fixtures into TEND's nesting layout, run (c) on them, and compare. Different data makes (c) independent evidence. That needs a fixture-to-nested migrator; not recommended before the pilot.

The fixture verifier's known weakness (it accepted 10/20 adversarial wrong translations; `sql2mql_independent_audit.md`) is therefore **not** fixed by TEND-v1. It still needs the audit's §9 remedies.

**Consequence for external evaluation:** TEND-v1 *test* sits on Spider-train DBs. Once Spider train is trained on, it is no longer a clean out-of-distribution test. Keep TEND v3, AptMQL dev and mongosh-EAI for that role.

---

## 10. Fallback strategy if native data stays scarce

This applies if the TEND-v1 licence is refused, or if re-execution rejects much more than projected:

1. **Primary:** the prior report's Spider-train `sql2mql` route. Fix the audit's §9 prerequisites first (B1, B2, DISTINCT+GROUP BY, identifier case), and verify with L1 on the real DBs plus L2 fixtures. TEND-v1 then serves only as a static triage signal and rule reference (§9), which copies no labels.
2. **Native skill shards with clean licences:** AptMQL train and capped SM3. Together they give 3,200–5,250 rows of genuine embedded-document MQL that no SQL conversion produces.
3. **Avoid** scaling ft-v1-style conversion of NL2SQL sets that have no source DB (Kaggle, Gretel), and LLM-generated NL→MQL sets without a DB (diwanshuydv, channudambal). Keep them for separately gated ablations at most.

---

## Conclusions

- **Best existing native dataset: AptMQL-Bench (train split, 1,762 rows).** It is the only native set that combines a clear licence (CC BY-SA 4.0), real multi-domain databases, deep realistic pipelines and verified-equivalence claims.
  - TEND-v1 is larger (3,001 records after exclusion), and in our sample about 3–7% of its MQL is wrong (CI 2–21%). But it has no data licence and its SQL is entirely Spider train.
  - SM3 has the cleanest licence and hand-written gold, but it is one templated domain.
- **Best realistic source for expanding the training set substantially:** still **Spider train converted by `sql2mql` and verified on the real DBs** (4.8–7.2K projected), once the audit's §9 fixes are in.
  - TEND-v1 helps it only as a static-triage signal and a rule reference (§9), not as an execution oracle.
  - Native sources add roughly 3,200–5,250 rows with clean licences (AptMQL + capped SM3), or up to 8,100 if TEND-v1 is licensed. That is meaningful, and it teaches embedded-document skills that conversion cannot. On their own, they do not double the corpus with in-distribution data.
- **Recommended next action for our capstone:**
  1. Start Docker and load TEND-v1 (135 non-benchmark DBs), SM3 and AptMQL train into a local `mongo:8` container.
  2. Parse every gold MQL into `{collection, pipeline}` and re-execute it.
  3. Check TEND-v1 against the Spider SQLite gold and SM3 against its stored `sql_results`. That yields measured pass rates in place of every projection above.
  4. In parallel, email the TEND authors to ask whether the v1 data (incl. the Moon0256 mirror) is released under CC BY-SA 4.0, as v3 is.
  5. Then run the already-planned 300-item Spider-train pilot (prior report §8 step 0) after the audit's §9 converter fixes. Add a TEND-v1 stage-structure diff to it as a triage flag for manual review, not as verification.

---

## Sources

- AptMQL-Bench: https://huggingface.co/datasets/MongoDB/AptMQL-Bench · https://arxiv.org/abs/2610.02770 · https://www.mongodb.com/company/research/introducing-aptmql-bench-text-to-mql-benchmark
- TEND v1: https://arxiv.org/abs/2502.11201v1 · TEND v3: https://arxiv.org/abs/2502.11201 · https://github.com/Jinwei-Lu/Text-to-NoSQL
- TEND-v1 mirror and translator results: https://github.com/Moon0256/text-to-nosql (commit cloned 2026-10-25) · https://github.com/Moon0256/Text-to-NoSQLwithSMART
- MultiTEND: https://arxiv.org/abs/2502.11022 · https://aclanthology.org/2025.findings-acl.1265.pdf
- SM3-Text-to-Query: https://github.com/jf87/SM3-Text-to-Query (commit `dd15ba5d`, 2025-10-28) · https://arxiv.org/abs/2411.05521 · Synthea licence: https://github.com/synthetichealth/synthea/blob/master/LICENSE
- DocSpider: https://doi.org/10.1017/nlp.2024.63 · https://github.com/arifgorkemozer/docspider
- mongodb-eai: https://huggingface.co/datasets/mongodb-eai/natural-language-to-mongosh
- Tier B: https://github.com/diwanshuydv/NL2Mongo · https://huggingface.co/datasets/channudambal/text-to-mongodb-queries-llm · https://huggingface.co/datasets/skshmjn/mongo_prompt_query · https://huggingface.co/datasets/kylemesh19/cleaned-mongosh-instructions · https://github.com/paulc-1212/nl2mongo-t5 · https://github.com/AichaAGGOUNE/M2Q2
- EvoMQL/DRO: https://arxiv.org/abs/2604.13045 · Oza 2024: https://cs.rit.edu/~dataunitylab/uploads/publications/Oza2024.pdf
