# Target-language scope review: NL→MQL vs NL→SQL

Review date: 2026-10-29. Review only.

- No model was trained or evaluated.
- No dataset, split, benchmark, converter or experiment output was changed.
- The held-out test answers were not opened: neither spider-sql-v1 `test_full` nor the 65-item MQL test partition.
- Besides this file, there are two edits, both appending a dated note and changing no existing text:
  - a scope-correction note in `docs/experiment_accuracy_ledger.md`;
  - a scope-correction note in `docs/presentation_notes.md`.

Labels used below: **finding** = read from a repo file or measured in this review; **recommendation** = proposed, not done.

---

## 1. Summary

1. **The project's intended target language is MongoDB MQL** (NL→MQL). The repo's founding documents say so, and so does the mentor deliverable. Everything run up to 2026-10-26 generated MQL.
2. **On 2026-10-27 the documentation was re-framed to "primary task: NL→SQL; NL→MQL secondary".** Every artifact produced since then follows that framing: spider-sql-v1, the SQL-0001 baseline, the parser audit, the test-suite evaluation, the ledger and the presentation notes. The repo records *that* the direction changed. It does not record who decided it, or why.
3. **SQL-0001 is an NL→SQL result.** It reports official EX 446/1,034 (43.1%), EM 378/1,034 (36.6%) and test-suite EX 630/1,034 (60.9%). None of these says anything about MQL generation. Do not put it next to any MQL number as a comparison (§3.3).
4. **There is no usable native NL→MQL training data in hand.**
   - The 2026-10-25 research found three downloadable native sources: AptMQL-Bench train, TEND-v1 and SM3.
   - None of them has been re-executed. None is present on disk today. TEND-v1's data licence is unresolved.
   - Every "usable" count in that report is a projection (§4).
5. **The sql2mql converter is not yet reliable enough to produce training labels at scale** (§5). The current ft-v1 training file still contains the 88 labels the audit found wrong.
6. **The existing MQL fine-tuning plan (`docs/fine_tuning_plan.md`, ft-v1) should not be run as written** (§6).
7. **Required next step before any MQL training:** a measured data-readiness gate (§7.3). It has five parts:
   - a local MongoDB;
   - `sql2mql-v2` with the audit's fixes;
   - a 300-item Spider-train conversion pilot with measured pass rates;
   - re-execution of the native sources;
   - the TEND licence question.

---

## 2. Evidence for the target language

### 2.1 Documents stating NL→MQL (finding)

| Source | What it says |
|---|---|
| `README.md`, line 3 | "with an initial focus on translating natural language into MongoDB queries (MQL)" |
| `README.md`, "Mentor deliverable" | "Spider → MQL dataset + model comparison (current)". The section describes the deliverable: Spider examples converted to MQL and executed on MongoDB Atlas, with a model comparison. |
| `README.md`, "Current Status: Week 1" | "define an **NL → MongoDB/MQL** task" |
| `docs/dataset_task_definition.md` §1 | "Given a natural-language question and a MongoDB schema, generate an **MQL read query**" |
| `docs/README.md` | Lists "formal NL → MongoDB/MQL task definition" |
| `docs/fine_tuning_plan.md` title | "Fine-tuning plan: Qwen2.5-Coder NL→MQL (ft-v1)" |
| Weeks 1–3 artifacts | Every model run up to 2026-10-26 generated MQL and scored it by execution: Spider2-Lite 68, v1 279, expanded 455, Week 3 RAG. |

### 2.2 Documents stating NL→SQL as primary (finding)

| Source (date) | What it says |
|---|---|
| `docs/current_dataset_inventory.md` (2026-10-27), line 3 | "written for the project's new direction: primary task: NL→SQL … secondary task, if time permits: NL→MQL" |
| `docs/spider_sql_v1_dataset.md` (2026-10-27) | "This is the primary NL→SQL dataset" |
| `docs/presentation_notes.md` (2026-10-27) | "Primary task: **NL→SQL** on Spider 1.0. NL→MQL is secondary." |
| `docs/experiment_accuracy_ledger.md` (2026-10-28) | Section headings "A. NL→SQL (primary task)" and "B. NL→MQL (secondary task …)" |

### 2.3 What is silent

- `CLAUDE.md` contains only the ledger rule. It does not state a target language.
- No document gives a reason for the 2026-10-27 re-framing or names who decided it.
- Neither `README.md` nor `docs/dataset_task_definition.md` was updated to say NL→SQL. The repo's top-level description has therefore said "MQL" throughout.

### 2.4 Conclusion

The intended behaviour, as the founding documents and the mentor deliverable define it, is **NL→MQL**. The NL→SQL framing is confined to documents written on 2026-10-27/28. Those documents should now be read as describing a **SQL-only side track**, not the project's target. This review does not delete or rewrite them. Dated correction notes were added to the ledger and the presentation notes (§8).

---

## 3. Artifact classification

### 3.1 MQL-specific (the target task)

| Artifact | State | Role under the NL→MQL target |
|---|---|---|
| `data/processed/spider1_nl_mql.jsonl` (279, authored and validated `gold_mql`) | Verified on Atlas, 279/279 gold replay | Evaluation set. Spider-dev derived. |
| `data/processed/spider1_nl_mql_expanded.jsonl` (455; no gold MQL; scored as MQL result vs gold-SQL result) | 3 models × 455 complete | **Headline MQL benchmark.** The 318/72/65 DB-disjoint split lives in `data/splits_expanded/`, and the 65-item test partition is unused. |
| Week 2 / expanded results: Qwen-3B 217/455 (47.7%), Qwen-7B 287/455 (63.1%), gpt-oss-120b 385/455 (84.6%) | Reported in `docs/spider1_expanded_benchmark.md`; **not yet back-filled into the ledger** | The existing MQL zero-shot baselines. |
| Week 3 RAG: clean Kaggle control, 318 train-partition items, zs 160 / random 155 / hybrid 172 | `docs/week3_rag_report.md`; not in the ledger | MQL RAG evidence. Hybrid vs random: p = 0.021, Holm 0.064, **not significant**. |
| `data/rag/corpus_v1/` (280 examples, 18 guide cards, 80 schema docs) | Built and re-executed | MQL retrieval resource. The 280 examples are **model-predicted MQL** (265 gpt-oss, 15 Qwen-7B), not gold. |
| `data/processed/nl_mql_dataset.jsonl` (Spider2-Lite, 68 validated) | Validated | Small, hard evaluation set. 51 of the 68 rest on reference SQL that we wrote. |
| `data/finetune/ft_v1/` (train 1,784 / dev 80) | Built 2026-10-24; **unchanged since the 2026-10-25 audit** (file date 2026-10-24) | MQL SFT candidate. **Not training-ready** (§6). |
| `scripts/finetune_v1/sql2mql.py` + verifier | Audited | Candidate-label generator. Needs v2 fixes (§5). |
| `scripts/spider1/`, `eval_kit.validate_mql`, `spider1_kit.exec_match`, `scripts/kaggle_week3/` | Tested; pytest passed at last run | MQL evaluation and safety harness. Reuse it unchanged. |

### 3.2 SQL-specific (a side track; not MQL evidence)

| Artifact | State | Role under the NL→MQL target |
|---|---|---|
| `data/processed/spider_sql_v1/` (train 7,773 / val_internal 797 / dev_full 1,034 / test_full 2,147) | Built 2026-10-27 | NL→SQL dataset. **Its SQL is not an MQL label.** |
| SQL-0001: zero-shot Qwen2.5-Coder-3B on `dev_full` | Verified 2026-10-28 | SQL-only baseline. Keep it as it is. |
| `docs/sql_v1_zero_shot_parser_audit.md`, `docs/sql_v1_test_suite_eval.md` | Done | SQL-only evaluation-protocol diagnostics. |
| `scripts/sql_v1/` (build, zero-shot prompt, Kaggle bundle, test-suite eval) | Done | SQL-only tooling. **No SQL training script or SQL SFT plan exists** (finding: none in `scripts/sql_v1/`). |

### 3.3 The SQL result must not be read as MQL performance

The two Qwen-3B zero-shot numbers below come from the same Spider dev question pool. The 455 MQL items are a subset of the 1,034. They are still **not comparable**:

| | SQL-0001 | Expanded MQL benchmark |
|---|---|---|
| Output language | SQL | MQL |
| Items | 1,034 | 455 (execution-eligible subset) |
| Metric | Official Spider EX (string-parser gated) / test-suite EX | MQL result on Atlas vs gold-SQL result on SQLite (`exec_match`) |
| Runtime | Kaggle T4, fp16, HF transformers | CPU llama.cpp GGUF (Week 2) |
| Score | 446/1,034 (43.1%) | 217/455 (47.7%) |

- These two figures must never appear side by side as "SQL vs MQL", or as "the 3B baseline".
- The only MQL zero-shot baselines are the 279 and 455 results, plus the Week 3 RAG controls.

### 3.4 Reusable infrastructure (finding: exists; reuse is a recommendation)

| From the SQL side track | Reuse for NL→MQL |
|---|---|
| spider-sql-v1 cleaning of Spider train | **Source pool for an audited conversion**, not labels: 40 dev/test question leaks removed, 38 gold defects removed, deduplicated, and a DB-disjoint `val_internal`. Its `rejected.jsonl` doubles as a ready exclusion list. |
| `schemas_ddl.json` (206 DBs) | Input for building MongoDB schema descriptions and identifier-case canonicalisation. |
| Test-suite variant DBs (`~/codegen_data/spider1_testsuite_e97acc5`) | Populated variants exist for academic, imdb, restaurants, scholar and yelp, all of which are **empty** in Spider train. Without them, real-DB L1 checks cannot work on those DBs. |
| Kaggle T4 batch workflow (`scripts/sql_v1/prepare_kaggle_dev_bundle.py`, `make_kaggle_zs_notebook.py`) | Pattern for packaging an MQL training run and a paired evaluation. `scripts/kaggle_week3` already does this for MQL. |
| Parser-sensitivity lesson (`sql_v1_zero_shot_parser_audit.md`) | Style-sensitive metrics reward learned surface form. MQL scoring is already execution-based, which is the right choice. Keep it. |

---

## 4. Native NL→MQL dataset research: what it does and does not establish

Source: `docs/native_nl2mql_dataset_research.md` (2026-10-25). It builds on `docs/dataset_research_report.md`.

### 4.1 What it establishes (measured in that session)

| Source | Downloaded then? | Measured candidates | Licence |
|---|---|---|---|
| AptMQL-Bench train | In the 10-25 session, the card and rows via the HF API. The prior report (`dataset_research_report.md` §3) describes the `mongodump` files (535 files, 2.55 GB). Neither report records a restore or a re-execution. | 1,762 rows, 10 BIRD DBs | CC BY-SA 4.0 (with a noted `LICENSE.md` inconsistency) |
| TEND-v1 (Moon0256 mirror) | Cloned to a session scratch directory | 3,001 records / 15,005 NL after dropping 403 records on 19 of our 20 benchmark DBs | **Not stated for the data.** The mirror's BSD-3 licence excludes it. |
| SM3-Text-to-Query | Cloned to a session scratch directory | 5,760 rows after gates; 4,476 unique; ~234 MQL skeletons; 1 medical domain | Apache-2.0 |

It also measured the following:
- TEND-v1 contamination: 19 of our benchmark DBs; all of its kept SQL is Spider-train SQL.
- TEND-v1 noise: 1–2 of 30 reviewed records wrong (Wilson CI 2–21%).
- SM3 template bugs: 39 rows.
- SM3 never stored gold-MQL execution results.

### 4.2 What it does not establish

1. **Usability.** "No source has been re-executed by us" (research §7.1). AptMQL's equivalence claim, TEND-v1's execution filter and SM3's expert check are all *stated* by their authors.
2. **Usable counts.** These are **projections**, not measurements:
   - 3,200–5,250 rows with clean licences (AptMQL + capped SM3);
   - 5,700–8,100 rows if TEND-v1's licence clears.
3. **Availability today (checked 2026-10-29):**
   - No TEND, SM3, AptMQL or text-to-nosql file is present under `data/` or `~/codegen_data`. The earlier clones were in a previous session's scratch directory.
   - The Docker daemon is **not running**: `docker info` cannot reach `dockerDesktopLinuxEngine`. Re-execution is therefore still blocked locally.
4. **Legal clearance for TEND-v1.** It is unresolved. The research recommends asking the authors, and no record shows that this has been done.
5. **Fit to our benchmark.** All three native sources use embedded document schemas, while our benchmark is flat (one collection per table). Gains on the 455 benchmark may be smaller than the row counts suggest.

**Answer to "does the research establish usable native datasets?" No.**
- It establishes that **three native sources exist and could be downloaded**. Two were cloned once. For AptMQL, the rows were read, and no restore of the dumps is recorded.
- It gives measured candidate counts and contamination facts.
- **Native NL→MQL data collected and verified in this project: 0.**
- The research's own executive summary answers "Yes, but no single one is large, clean and clearly licensed at the same time". Quote that sentence together with the projection label, never as a bare "yes".

---

## 5. Why sql2mql output cannot yet be assumed reliable for training

Source: `docs/sql2mql_independent_audit.md` (2026-10-25). Figures recomputed in `outputs/audit_sql2mql_v1/summary.json`.

1. **Known-wrong labels are still in the training file.**
   - 88 of 1,584 ft-v1 converter labels (5.6%, CI 4.5–6.8%) are wrong. All come from two bugs:
     - B1: an ungrouped aggregate over zero rows returns no document instead of `0`/NULL (84 labels);
     - B2: `SUM` over all-NULL input returns `0` instead of NULL (20 labels).
   - `data/finetune/ft_v1/train.jsonl` is dated 2026-10-24, before the audit. The labels have not been removed or repaired.
2. **The verifier certifies wrong MQL.**
   - It accepted 10 of 20 adversarial wrong translations.
   - It cannot distinguish a plausible wrong MQL from the label for 21 of 200 sampled labels (10.5%).
   - The status `fixture_equivalence_verified` therefore overstates the evidence.
3. **On real Spider-train DBs, 7.2% of clean converted queries are wrong** (CI 6.4–8.1%):
   - B1 5.5%, much of it on seven all-empty DBs;
   - missing implicit type coercion 1.7%.
   - On type-heterogeneous data, 58% are wrong.
   - Two new bugs: DISTINCT is ignored when GROUP BY is present, and date-string subtraction raises an error.
4. **Identifier case.** 55% of Spider-train queries need canonicalisation, because MongoDB field names are case-sensitive. Run raw, the converter would emit wrong field names for more than half of Spider.
5. **Coverage.** 26% of unique Spider-train queries are rejected as unsupported (subqueries, set operations, etc.). A converted set would therefore under-represent exactly the harder patterns.
6. **NULL contract.** Labels assume explicit `null`. If NULL is stored as an absent field, 62 labels (3.9%) fail.
7. **Labels can be wrong even when the SQL→MQL step is faithful.** 5 of 120 reviewed Kaggle SQL labels (4.2%) do not answer their question, and 51 labels carry the `datetime = date` pattern. No execution check can catch these.
8. **The 1,759 Spider-train records on empty DBs** (all of `train_others` plus 100 on `music_2`) cannot be validated by real-DB execution as shipped. They must be excluded, or checked through the populated test-suite variants where those exist.

**Audit verdict, kept here:**
- Safe for generating *candidate* MQL.
- Safe for training labels only after the fixes and with the NULL contract stated.
- **Not safe** as high-confidence labels without independent review.

The 7,773 spider-sql-v1 train records must therefore **not** be bulk-converted into MQL labels with `sql2mql-v1`.

---

## 6. Consequences for the training plan

### 6.1 The existing MQL plan (`docs/fine_tuning_plan.md`, ft-v1)

Finding: the plan says "No training has been run". `train_qlora.py` is written but has never been executed. The plan should **not** be run as written:

| Issue | Evidence |
|---|---|
| 88 known-wrong converter labels | Audit §0 and §9. Not yet removed. |
| 51 labels with SQL-level `datetime = date` errors | `outputs/audit_sql2mql_v1/datetime_date_equality.jsonl` |
| The 280 "Spider" rows are model-predicted MQL on Spider-**dev** questions | `current_dataset_inventory.md` §0.4. They are DB-disjoint from the 72/65 validation/test partitions, so they are legitimate for the 455-benchmark protocol. They are, however, contaminated for any other Spider-dev claim. |
| Mixed licences | Kaggle rows are CC BY-NC-SA, Spider rows CC BY-SA. Plan §2.10 already flags this. |
| Domain mismatch | 1,504 of the 1,784 rows are synthetic single-collection healthcare queries with no source DB. |
| Small, flat in-distribution core | The only in-distribution rows are those 280. |

What the plan still gets right, and what should carry over:
- QLoRA 3B pilot → 7B;
- completion-only loss;
- model selection on the 72-item validation partition, with the 65-item test partition run once;
- paired fresh generation on Kaggle;
- McNemar/Holm with a frozen analysis plan;
- the `validate_mql` safety gate.

### 6.2 The SQL side track

- No SQL fine-tune exists or is planned in code. Recommendation: do not start one as a substitute for the MQL target.
- If the SQL side track continues, its results stay in ledger section A, labelled SQL-only.

---

## 7. Recommended NL→MQL dataset strategy (recommendations)

Every row count below is a **projection** until measured by step 7.3. Each tier must be kept as a separately tagged shard (`source`, `label_origin`, `schema_style`, `licence`), so that it can be ablated and reported on its own.

### 7.1 Tiers

| Tier | Source | Label origin | Status today | Gate to enter training |
|---|---|---|---|---|
| **1. Native (unverified; enters only through the gate)** | AptMQL train (CC BY-SA); SM3 train, template-capped (Apache-2.0); TEND-v1 non-benchmark DBs **only if** the licence clears | Authors' native MQL | Not on disk; not re-executed | Local re-execution on `mongo:8`. AptMQL against its stated BIRD result; SM3 against its stored `sql_results`; TEND-v1 against Spider SQLite. Drop all 20 benchmark DBs and NL at ≥0.8 trigram similarity. Parse mongosh to `{collection, pipeline}` in key order (never `sort_keys`). |
| **2. Audited conversion** | spider-sql-v1 `train` (7,773) + `val_internal` (797), Spider-train DBs only, via `sql2mql-v2` | Deterministic converter; never the SQL itself | Converter v1 audited; v2 not written | Audit §9 fixes (B1, B2, DISTINCT+GROUP BY, date subtraction, identifier case, type-aware literals/keys or rejection). **L1 on the real DB** plus the strengthened L2 fixture families. Exclude or variant-check the 1,759 empty-DB records. Measure the pass rate on a 300-item pilot before scaling. Keep `val_internal` DBs as the internal validation partition. |
| **3. Controlled synthetic** | ft-v1 Kaggle-derived rows (after dropping the 88 B1/B2 and 51 date-equality labels); any future LLM-paraphrased NL | Converter on synthetic fixtures | Built; contains known errors | Ablation only. Never mixed into a headline run without a with/without comparison. NC licence noted. |
| **4. Retrieval resources** (not SFT labels) | RAG corpus v1 (280 model-predicted examples + 18 guide cards + 80 schema docs); after the licence check, native examples as retrieval-only items | Mixed | Built | Same benchmark-DB exclusion. The retrieval pool must never contain validation or test DBs. |

### 7.2 Evaluation (unchanged)

- Headline: the 455 expanded benchmark. Select on validation (72); run the test partition (65) once.
- Secondary external evaluation, kept eval-only: TEND v3, AptMQL dev and mongosh-EAI.
- Every new MQL result gets an `MQL-xxxx` ledger entry in section B.

### 7.3 Required next step before training an MQL model

The **data-readiness gate**, in this order:

1. **Start Docker and run a local `mongo:8`.** Both tiers 1 and 2 depend on it. Atlas M0 is read-only for the eval user and limited to 512 MB.
2. **Write `sql2mql-v2`** under a new versioned path; never edit `scripts/finetune_v1/`, since its code hash gates resume. Include the audit §9 fixes, and add the audit's fixture families to the verifier.
3. **Run the 300-item Spider-train pilot** (`dataset_research_report.md` §8 step 0). Use L1 on the real DBs and the strengthened L2. Report measured pass and wrong rates, plus a manual semantic review sample with a stated error bound.
4. **Re-execute AptMQL train and SM3** locally, to replace the projected counts with measured ones.
5. **In parallel:** ask the TEND authors whether the v1 data is CC BY-SA 4.0. Do not use TEND-v1 labels until they answer.
6. **Only then** build a versioned MQL SFT set (e.g. `ft-v2`) from the shards that pass, and revise `fine_tuning_plan.md` against it.

---

## 8. Changes made in this review

| File | Change |
|---|---|
| `docs/target_language_scope_review.md` | New (this file) |
| `docs/experiment_accuracy_ledger.md` | Appended a dated 2026-10-29 scope note below "Current best results". SQL-0001, the section headings and every existing entry are unchanged. |
| `docs/presentation_notes.md` | Appended a dated 2026-10-29 correction under "Project framing (as of 2026-10-27)". The existing table is unchanged. |

**Not done, and recommended as follow-ups:**
- Back-fill the 279/455 and Week 3 MQL results into ledger section B as `MQL-xxxx` entries. Each may be marked *verified* only after its artifacts are re-checked locally, per the ledger rules.
- Add a scope rule to `CLAUDE.md`: the target is NL→MQL, and SQL results are labelled SQL-only.
- Update `README.md` to mention the SQL side track and its status.
