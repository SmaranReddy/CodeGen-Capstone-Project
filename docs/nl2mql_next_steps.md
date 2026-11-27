# NL→MQL: prioritised next steps (scope, data gates and training roadmap)

Written 2026-10-29. Planning document only.

- No model was trained or run, no dataset was downloaded, no dataset artifact was modified, and no Atlas query was executed for this document.
- The MQL test partition (65 items) was not used for any new purpose.
- Sources: `docs/target_language_scope_review.md`, `docs/native_nl2mql_dataset_research.md`, `docs/sql2mql_independent_audit.md`, `docs/fine_tuning_plan.md`, `docs/week3_rag_report.md`, `docs/experiment_accuracy_ledger.md` (entries MQL-0001 to MQL-0005).

**Labels used throughout:**
- **measured**: computed from files in this repo; the source is named.
- **stated**: a dataset author's claim, not checked by us.
- **projected**: an estimate that a later step must replace with a measurement.
- **proposed**: a recommendation, including any threshold; not yet agreed or done.

---

## 0. Scope (fixed)

- **Primary task: NL→MQL**, executed on **MongoDB Atlas**. This is now a persistent rule in `CLAUDE.md`.
- The plan requires six components:
  1. a zero-shot Qwen 3B baseline;
  2. a RAG baseline;
  3. fine-tuning Qwen on MQL;
  4. a frontier-LLM comparison;
  5. error analysis and ablations;
  6. a final integrated demonstration.
- The NL→SQL track (spider-sql-v1, SQL-0001) is auxiliary and SQL-only. It supplies a **cleaned source pool** of Spider-train questions for conversion, never labels and never comparison numbers.

## 1. Where we stand (measured)

| Component | State | Evidence |
|---|---|---|
| Zero-shot Qwen-3B | **Done.** 132/279 (47.3%) on v1; 217/455 (47.7%) on expanded. CPU llama.cpp GGUF q4_k_m | MQL-0001, MQL-0002 |
| Large-model comparison | **Done with gpt-oss-120b** (open-weight, Groq): 232/279 (83.2%); 385/455 (84.6%). Qwen-7B: 181/279; 287/455 | MQL-0001, MQL-0002 |
| RAG baseline | **Done, pilot.** Clean Kaggle control on 318 train items: zero-shot 160, random_k3 155, hybrid_k3 172. Hybrid vs random +5.3 points, Holm p = 0.064, **not significant**. Validation (72, CPU): hybrid_k3 43 vs zs_rerun 33, Holm 0.16 | MQL-0003, MQL-0005 |
| Fine-tuning | **Not started.** ft-v1 exists but is not training-ready | §3, §4 |
| Error analysis | Partial: per-category buckets and hardness/pattern slices for zero-shot and RAG | `docs/week2_report.md`, `docs/week3_rag_report.md` |
| Final demo | Not started | — |
| MQL test partition (65) | Unused for any experiment. The 455 aggregates predate the split and include these items | MQL-0002 |

The SQL-0001 numbers (446/1,034 official EX, etc.) are not in this table on purpose. They are SQL-only.

---

## 2. Datasets: on disk vs research-only

### 2.1 Verified on local disk (measured 2026-10-29: line counts and file presence)

| Asset | Path | Size | Usable as MQL training labels? |
|---|---|---|---|
| v1 benchmark (authored gold MQL) | `data/processed/spider1_nl_mql.jsonl` | 279 | **No: evaluation only** (Spider dev) |
| Expanded benchmark | `data/processed/spider1_nl_mql_expanded.jsonl` + `data/splits_expanded/` | 455 (318 / 72 / 65) | No, except the 318 train partition as RAG corpus and in-distribution examples. Validation 72 = selection; test 65 = final only |
| Spider 2.0-Lite MQL | `data/processed/nl_mql_dataset.jsonl` | 68 | No: small, hard, eval only; 51 rest on reference SQL we wrote |
| RAG corpus v1 | `data/rag/corpus_v1/` | 280 examples + 18 guides + 80 schema docs | Retrieval only. The 280 examples are **model-predicted MQL** (not gold) on Spider-dev questions in the train partition |
| ft-v1 | `data/finetune/ft_v1/` | train 1,784 (1,504 Kaggle-derived + 280 Spider train-partition) / dev_internal 80 | **Not as is**: contains the 88 known-wrong B1/B2 labels (86 train, 2 dev_internal) and 51 date-equality labels (45 train, 6 dev_internal) (§3; measured by joining the audit's id lists, 2026-10-29). File date 2026-10-24, before the audit |
| spider-sql-v1 | `data/processed/spider_sql_v1/` | train 7,773 / val_internal 797 / dev_full 1,034 / test_full 2,147 / rejected 89 | **SQL source pool only.** Its SQL is not an MQL label |
| Spider 1.0 raw + DBs | `data/raw/spider1/spider_data/` | 166 DB directories; train_spider + train_others 8,659 records | Source DBs for conversion. 1,759 records sit on all-empty DBs |
| Spider test-suite variants | `~/codegen_data/spider1_testsuite_e97acc5/` | Dev-DB variants extracted. The zip also holds populated variants for 5 DBs that are empty in Spider train | Needed to L1-check those 5 DBs |
| Audit outputs | `outputs/audit_sql2mql_v1/` | `kaggle_failure_classes.jsonl`, `datetime_date_equality.jsonl`, `spider_failure_classes.jsonl`, `summary.json`, … | Ready-made exclusion lists (§3.3) |

**Native NL→MQL data on disk: 0 records** (measured: no TEND, SM3, AptMQL or text-to-nosql file under `data/` or `~/codegen_data`).

### 2.2 Candidates reported in research only (not on disk, not re-executed)

| Source | What research measured (2026-10-25, previous session's scratch clone) | Licence | Blockers |
|---|---|---|---|
| **AptMQL-Bench train** | 1,762 rows, 10 BIRD DBs, embedded schemas. Rows read through the HF API; dumps (2.55 GB) never restored | CC BY-SA 4.0 (`LICENSE.md` text says CC BY 4.0; treat as BY-SA) | (1) Not downloaded. (2) Never re-executed: equivalence to BIRD SQL is **stated**. (3) Needs a local MongoDB (2.55 GB exceeds the Atlas M0 512 MB). (4) ShareAlike obligations on derived datasets |
| **SM3-Text-to-Query** | 6,000 train + 2,000 dev; 5,760 after gates; 4,476 unique; ~234 MQL skeletons; 1 medical domain; 39 rows from 2 buggy templates; no `$sort`/`$limit`/`$count` | Apache-2.0 (repo and Synthea) | (1) Not downloaded. (2) `mql_results` were never stored by the authors, so gold MQL was never executed by anyone we can cite. (3) mongosh→JSON parser and find→pipeline rewrite needed. (4) Deeply nested `patients` document: prompt length must be checked against `max_seq` 4,096 |
| **TEND-v1** (Moon0256 mirror) | 3,001 records / 15,005 NL on 135 DBs after dropping 403 records on 19 of our 20 benchmark DBs; 30-record review: 1–2 wrong (Wilson CI 2–21%) | **Unresolved.** The mirror's BSD-3 covers code only; the v1 paper states no data licence; gold and paraphrases come from OpenAI/Anthropic models whose output terms are unassessed | (1) **Licence: blocks training use.** (2) Contamination: must drop all 20 benchmark DBs, not just near-duplicates. (3) Not an independent oracle (its gold was filtered on equality with the SQL result on the same data). (4) Not downloaded; not re-executed |
| TEND v3, AptMQL dev, mongosh-EAI | Not re-downloaded | CC BY-SA 4.0 / CC BY-SA 4.0 / Apache-2.0 | Evaluation only by our rule. Running them on **Atlas** may exceed the M0 512 MB quota (assumption: quota is unread) |
| MultiTEND, CoNoSQL, Spider-Mongo | — | — | Not released |

**All native "usable" counts are projections:** 3,200–5,250 rows with clean licences (AptMQL + capped SM3), 5,700–8,100 if TEND-v1 clears.

---

## 3. SQL→MQL converter: fixes and validations required before any converted label is accepted

Source: `docs/sql2mql_independent_audit.md` §0, §2, §5, §9. All rates are **measured** there.

### 3.1 Required code fixes (`sql2mql-v2`, new versioned path)

| # | Defect | Measured impact | Required fix |
|---|---|---|---|
| F1 | **B1**: an ungrouped COUNT/SUM/AVG/MIN/MAX over zero rows returns no document (SQL returns one row) | 84 Kaggle labels; 5.5% of clean Spider-train conversions | Emit a default row (`$group` with `_id: null`, then a `$unionWith`/`$facet` fallback or `$ifNull` defaults): COUNT → 0, others → null |
| F2 | **B2**: SUM over only NULLs returns 0 | 20 labels (16 also B1) | Return null when no non-null input exists (count non-nulls alongside the sum) |
| F3 | DISTINCT ignored when GROUP BY is present | Spider `train_spider:3451` | Read `distinct` in the grouped branch |
| F4 | Arithmetic on date/text columns (`$subtract` on strings raises an error) | Spider `train_spider:6101` | Reject (unsupported), or convert with explicit date parsing |
| F5 | Identifier case copied verbatim (MongoDB field names are case-sensitive) | **55% of Spider-train queries** need canonicalisation | Canonicalise every table and column name against the imported schema |
| F6 | Implicit type coercion missing (text `'40'` vs `50`; numbers stored as text) | 1.7% of clean Spider conversions; 58% wrong on type-heterogeneous data | Type-aware literals from the real column types, or reject |
| F7 | Join keys of mixed type (text `'1'` vs int `1` never match) | Spider census, J8 | Type-aware `$lookup` (convert keys) or reject |
| F8 | Division: by zero raises a MongoDB error (SQL gives NULL); real division emitted, while **Spider/SQLite uses integer division** | N1, N3; 0 Kaggle labels | Guard zero divisors; emit integer division (`$trunc`) when both operands are integers in SQLite |
| F9 | `ORDER BY <ordinal>` sorts on the output name before `$project` | O4, O4c, O5, D4 | Resolve ordinals to source expressions |
| F10 | `NOT <bare boolean column>` includes NULL rows | W14 | Match only stored `0` (v1 emits `$in: [0, null]`); NULL rows must be excluded |
| F11 | LEFT JOIN + COUNT/SUM(right.col): an unmatched (absent) field passes the `$ne: null` guard | J4 (count 1), J4b (SUM 0) | Missing-aware guard (`$ifNull` / `$type`) |
| F12 | `$lookup` `as` = bare table name overwrites a same-named base column | J6 | Use a reserved, collision-free `as` name |
| F13 | LIKE: `%`/`_` do not match a newline | W10d, W10e | Use the `s` (dotall) regex option |
| F14 | Rounding: MongoDB `$round` is half-even; SQLite `round()` is half away from zero (`docs/dataset_task_definition.md` §8) | Not measured by the audit (ROUND is currently unsupported) | If ROUND support is added, implement half-away-from-zero explicitly |
| F15 | NULL representation contract unstated | 62 labels (3.9%) fail if NULL is an absent field | State the contract (explicit `null`, all fields present) in the dataset card **and** match it in our Mongo import; or emit missing-safe MQL and verify under both |

Still unsupported by design (rejected, no label): subqueries, set operations, CASE/CAST/COALESCE, non-LEFT outer joins, CTEs, windows. These are about 26% of unique Spider-train queries. The converted corpus will therefore **under-represent the harder patterns**; report its pattern distribution next to the benchmark's.

### 3.2 Required validations (each must pass before labels are accepted)

| # | Validation | Pass criterion (proposed) |
|---|---|---|
| V1 | Rerun the 2026-10-25 adversarial suite (`scripts/audit_sql2mql_v1/adversarial.py`) against v2 | Every wrong translation rejected (v1: 10/20 wrongly accepted) |
| V2 | Rerun the mutant-kill study (`mql_mutants.py --n 200`) | Report the unkilled-but-non-equivalent rate (v1: 21/200 labels, 10.5%); target well below 5% |
| V3 | Add the audit's fixture families to the verifier: empty tables, zero-match WHERE, all-NULL aggregate columns, duplicate rows, case variants for LIKE, distinct values per GROUP BY key, zero divisors, text-typed numerics, newline strings | All present in the v2 verifier; covered by unit tests |
| V4 | **L1 on the real database**: converted MQL on the local Mongo import vs the gold SQL on the original SQLite, with `exec_match` | Required for every Spider-derived label. Records on all-empty DBs need a populated test-suite variant or are excluded |
| V5 | Tie check: reject any item whose ORDER BY … LIMIT cuts through a tie (Week 3 lesson, 0934/0667) | 0 accepted tie-cut items |
| V6 | Determinism: re-execute every accepted label twice | Identical results |
| V7 | Human semantic review of a random sample (question ↔ SQL ↔ MQL) | Report the wrong-label rate with a Wilson 95% bound; proposed sample ≥ 100 |
| V8 | Safety: every label passes `eval_kit.validate_mql` unchanged | 100% |
| V9 | Leakage: no record on any of the 20 benchmark DBs; no question ≥ 0.8 trigram similarity to a benchmark question | 0 |

`fixture_equivalence_verified` alone never makes a label high-confidence. Status names must say what was checked (for example `l1_real_db_verified`).

### 3.3 Non-overwrite rules

- `scripts/finetune_v1/` is not edited: its code hash gates resume. `sql2mql-v2` lives under a new path (proposed `scripts/sql2mql_v2/`).
- `data/finetune/ft_v1/` is never repaired or overwritten. The 88 B1/B2 labels (`outputs/audit_sql2mql_v1/kaggle_failure_classes.jsonl`) and 51 date-equality labels (`datetime_date_equality.jsonl`) become **exclusion lists** that a new build reads.
- Every new corpus goes to a new versioned directory with a manifest of input hashes.

---

## 4. Prioritised plan

Priority order follows the dependency chain. Nothing in P5–P9 may start until its gate passes.

### P1. Local MongoDB for label and dataset validation (blocks P2–P5)

- **Fact:** Docker is installed (client 29.3.1), but the daemon was not running on 2026-10-29. No local `mongod` exists. Atlas is read-only for `codegen_eval_readonly` and limited to 512 MB on M0.
- **Proposed:**
  - start Docker Desktop and run a container pinned to `mongo:8.0.x`, matching Atlas 8.0.34;
  - keep data volumes outside OneDrive (`~/codegen_data/mongo/`);
  - use a separate read-write local user.
- **Scope rule:** local Mongo is only for importing Spider-train DBs, running L1, and re-executing native gold. **Every reported metric stays on Atlas.**
- **Exit check:** import one Spider-train DB and pass the same per-table SQLite↔Mongo parity check as `scripts/spider1/parity_check.py`.

### P2. Native dataset acquisition and licensing (in parallel with P3–P4)

1. **Licensing, now, no compute:** ask the TEND authors whether the v1 data (and the Moon0256 mirror) is released under CC BY-SA 4.0, as v3 is. Until they answer, TEND-v1 is a static reference for converter rules only: no labels, no RAG, no few-shot.
2. **Downloads (need the user's go-ahead; none were done here):** AptMQL train (dumps 2.55 GB) and SM3, into `~/codegen_data/native/<source>@<commit>/`, each in its own empty directory, scripts run with `python -I`.
3. **Re-execution on local Mongo:**
   - AptMQL against its stated BIRD result;
   - SM3 against its stored `sql_results`;
   - TEND-v1 against the Spider SQLite gold, only if the licence clears.
   - Parse mongosh to `{collection, pipeline}` in key order (never `sort_keys`).
4. **Output:** a measured pass rate per source, replacing the projected 3,200–5,250 (8,100) figures. Each source is tagged `schema_style: embedded`, `licence`, `label_origin`.

### P3. Database and schema availability

- Spider train: 166 DB directories on disk. 1,759 records sit on all-empty DBs (all of `train_others` plus 100 on `music_2`). They are excluded unless a populated test-suite variant exists (5 DBs).
- Import with identifier-case canonicalisation (F5) and the stated NULL contract (F15). Record the column types, because F6/F7 depend on them.
- Build MongoDB schema descriptions from `schemas_ddl.json` (206 DBs) in the same format as the benchmark prompt (`nl-mql-spider1-v1`).
- Native sets need nested-schema rendering. Run `scripts/finetune_v1/token_audit.py`-style length checks against `max_seq` 4,096, especially for SM3.

### P4. Repair and validate the converter (`sql2mql-v2`)

> **Status 2026-10-30 (measured, local SQLite + MongoDB 8.0.34; see `docs/sql2mql_v2_quality_assessment.md`).**
> - **P1 done.** Docker `mongo:8.0.34` runs on 127.0.0.1:27018, data in `~/codegen_data/mongo_v2`. Import parity passed under explicit and absent-field NULL.
> - **P4 implemented** in `scripts/finetune_v2/`: converter, independent differential verifier, adversarial suite, mutant study.
> - **Gates passed:** G1, the coverage matrix (F1-F15 + every audit defect, 26/26); G2/V1, 0 false accepts in 139 adversarial cases; G3, 33/33 v1 wrong MQL rejected; G5/V3; G6/V8.
> - **G4/V2 FAILED.** Iteration 1: 27/200 labels (13.5%). Iteration 2, pre-registered held-out: 7/161 (4.35%), Wilson upper bound 8.7% > 5%.
> - **P4 is therefore NOT passed, and P5 (pilot) was not started.** The pilot ids are frozen in `data/interim/finetune_v2/pilot_freeze.json` and have not been used.
> - Dev sample: 361/500 accepted (378 converted). 0 L1/fixture mismatches.
> - Next: user decision between a fresh-holdout iteration 3 and a scope restriction (quality assessment §7).
>
> **Status 2026-10-31 (local; `docs/sql2mql_v2_round3_report.md`).**
> - Iteration 3, on a fresh pre-registered stratified sample (740 records outside pilot and dev) with the new verifier-3, **passed G4**: weighted Wilson upper bound 4.12% ≤ 5%. G2 and G3 still pass with verifier-3.
> - **P4 is passed** (G4 on the user-specified, pre-registered round-3 criterion; the unweighted form also passes at 4.52%). Multi-table joins (≥ 3 tables) remain weak: 7/104, upper bound 13.3%.
> - **P5 (pilot) not started.** It needs your go-ahead, a verifier-3 pilot runner and a named independent reviewer (`docs/sql2mql_v2_human_review.md`; the dry-run package exists).


Implement F1–F15 (§3.1) and pass V1–V3, V8 (§3.2) on fixtures before touching Spider train at scale.

### P5. Measured conversion pilot (300 items)

- **Source:** spider-sql-v1 `train` (already cleaned: leaks, defects and duplicates removed). Draw a seeded random sample of 300 from populated, non-benchmark DBs, stratified by hardness.
- **Measure and report:**
  - coverage (converted / attempted);
  - L1 pass rate (V4);
  - L2 pass rate on the strengthened fixtures;
  - tie rejections (V5);
  - manual review wrong-label rate with a Wilson bound (V7, proposed ≥ 100 reviewed);
  - per-pattern pass rates.
- **Proposed gate:** scale to the full 7,773 + 797 only if the upper 95% bound on the wrong-label rate among accepted labels is ≤ 5%, and the L1 pass rate is reported per pattern. Otherwise iterate on P4.
- Record the pilot as a dataset-audit entry in ledger section C.

### P6. Versioned NL→MQL corpus (`ft-v2`)

| Partition | Content | Rule |
|---|---|---|
| train | Shards: (a) Spider-train via `sql2mql-v2`, L1-verified, flat; (b) up to 318 benchmark-train items (280 had a verified label in ft-v1: gold where authored, else verified model predictions); (c) native embedded shards that passed P2; (d) ft-v1 Kaggle rows minus the exclusion lists, **ablation only** | Each record carries `source`, `label_origin`, `schema_style`, `licence`, verification status |
| internal validation | Converted items on the spider-sql-v1 `val_internal` DBs (DB-disjoint) | Loss and overfitting monitoring only |
| model selection | Benchmark validation, 72 items | Only selection signal |
| final evaluation | Benchmark test, 65 items | Run once, at the end, on explicit request |
| external evaluation (optional) | TEND v3 / AptMQL dev / mongosh-EAI | Eval only. Atlas quota must be checked first |

- **Dedup:** key Spider-derived shards on `(db_id, normalised SQL)`; exact dedup per shard; near-duplicate caps (TEND ≤ 2 NL per record, SM3 ≤ 5–10 per template).
- **Leakage:** V9 applied to every shard.
- **Licences:** ShareAlike (CC BY-SA) and NonCommercial (CC BY-NC-SA, Kaggle-derived) rows stay separable. An adapter trained on Kaggle rows inherits the NC restriction.
- Write a build manifest with input hashes and a dataset card that states the NULL contract.
- Revise `docs/fine_tuning_plan.md` against ft-v2 before training. ft-v1 stays as is.

### P7. One Atlas evaluation protocol for every remaining comparison

- **Harness:** `validate_mql` → Atlas (`codegen_eval_readonly`) → `exec_match` (order enforced only with ORDER BY, tie blocks allowed), unchanged from MQL-0001 to MQL-0005.
- **Prompt:** `nl-mql-spider1-v1` (zero-shot/FT) and `nl-mql-spider1-rag-v1` (RAG). Any change gets a new version and hash.
- **Same runtime for every compared arm.** The existing CPU baselines (MQL-0002) are not reused as FT controls. Every comparison generates all arms fresh in one Kaggle session with the KV cache reset per item (the MQL-0005 design). On the same 318 items, the stored CPU zero-shot scored 148 and a fresh Kaggle zero-shot scored 160 (MQL-0004, MQL-0005), so a reused baseline can shift a comparison by about 4 points.
- **Decoding:** greedy, seed fixed, `max_new_tokens` 1024.
- **Metrics:**
  - strict execution accuracy `correct/total`;
  - Wilson CI;
  - per-hardness and per-pattern slices;
  - outcome buckets (malformed, rejected, execution failure, mismatch);
  - an ordering-sensitivity count (tie-cut items, as for 0934/0667).
- **Statistics:** analysis plan frozen in the package before scoring; exact McNemar per planned contrast; Holm across the family; paired bootstrap CI.
- **Recording:** each run is a new `MQL-xxxx` ledger entry plus a `docs/presentation_notes.md` line.

### P8. Zero-shot Qwen-3B vs frontier model

- Existing evidence is MQL-0001 and MQL-0002. Those Qwen runs were CPU GGUF; gpt-oss-120b ran through the Groq API.
- **Open question for the mentor:** does gpt-oss-120b (open-weight, hosted) satisfy "frontier LLM"? Earlier closed-model attempts (Gemini) stopped at 9–17 items on free-tier quotas. A closed model needs a budget decision.
- **Proposed:**
  - run the chosen frontier model on the same 72 (selection phase) and the same 65 (final) with the P7 harness and prompt;
  - report it next to zero-shot, RAG, FT and FT+RAG Qwen on identical ids.
- **Distillation caveat:** 265 of the 280 RAG examples and ft-v1 Spider labels are gpt-oss-120b outputs. Fine-tuned Qwen vs gpt-oss is therefore partly a teacher–student comparison. The `label_origin` tag lets an ablation train without the gpt-oss-derived shard; report that ablation next to the frontier comparison.

### P9. RAG and fine-tuning experiments, then final evaluation

1. **Arms on validation (72), one Kaggle session:**
   - zero-shot base;
   - hybrid_k3 RAG;
   - FT;
   - FT + hybrid_k3.
   - Run Qwen-3B QLoRA first (pilot), then 7B (`fine_tuning_plan.md` §2.1–2.3 hyper-parameters carry over).
2. **Planned ablations:**
   - data mix: flat converted only / + native embedded / + Kaggle synthetic;
   - RAG k and guide cards;
   - 3B vs 7B;
   - optionally grammar-constrained decoding as a separate condition.
3. **Error analysis:**
   - outcome buckets;
   - hardness and pattern slices;
   - helped/hurt item lists per contrast;
   - a manual taxonomy on a fixed sample.
4. **Final:**
   - select configurations on validation only;
   - run the 65-item test **once** for the selected configurations and the frontier model, when explicitly requested;
   - record ledger entries.
5. **Integrated demo:** NL question → schema + retrieval → selected Qwen model → `validate_mql` → Atlas execution → result table.
6. **Report:**
   - the MQL results are the headline;
   - the SQL track appears in a separate, labelled appendix.

### Proposed week mapping (projection)

| Week | Items | Gate to move on |
|---|---|---|
| 4 | P1, P2 (licence email, downloads), P3, P4 | Local parity check passes; V1–V3 pass |
| 5 | P5 pilot; P2 re-execution; P6 build | Pilot gate met; ft-v2 manifest written |
| 6 | P7 package; P9 3B pilot FT, then 7B; FT+RAG | Training runs; validation scored |
| 7 | P8 frontier on validation; ablations; error analysis | Frozen analysis results |
| 8 | Final 65-item test (once), demo, report | — |

This schedule is an assumption. It depends on Kaggle GPU quota (about 30 GPU-h/week, a second-hand figure) and on the P5 pilot result.

---

## 5. Risks and unknowns

| Item | Type | Consequence |
|---|---|---|
| P5 wrong-label rate is above the gate | Unknown until measured | Fewer converted rows; native shards and the 318 in-distribution items carry more weight |
| TEND-v1 licence refused or unanswered | Unknown | Drop it from training (projected −2,500 to −2,850 records) |
| Native data uses embedded schemas; benchmark is flat | Measured mismatch | Gains on the 455 benchmark may be smaller than the row counts suggest; measured by the data-mix ablation |
| Atlas M0 512 MB quota | Fact (limit); usage unread | External evaluation sets may not fit on Atlas |
| Spider-dev contamination of pre-trained models | Assumption (literature) | Affects absolute numbers, not paired deltas |
| Small selection set (72) | Fact | Only large effects reach significance; the Week 3 RAG gain did not |

## 6. Single most important next action

**Start Docker Desktop and bring up a pinned local `mongo:8.0.x` (P1).** This is a user action.

- Converter repair (P4), the pilot (P5) and native re-execution (P2) all need a writable MongoDB to measure anything.
- Atlas cannot serve that role: the eval user is read-only, and M0 is capped at 512 MB.
- Until that environment exists, no label in any planned training corpus can be verified, so no fine-tuning should start.

In parallel, and at no compute cost: send the TEND-v1 licence question, and get the mentor's answer on the frontier-model choice.
