# Independent architecture review: why does the NL→MQL RAG not use embeddings?

Date: 2026-11-21. **Read-only investigation.** No code, config, corpus, registration or artifact was changed. Nothing was installed, no model was run, and Atlas and Kaggle were not used. The only computation was a local count of eligible Spider-train items (§5.3).

**Labels** (same convention as `docs/mql0009_post_experiment_analysis.md`, "PEA"):
- **[F] fact:** read from code, config or a verified artifact, with the source cited.
- **[E] conclusion:** inferred from facts, with the reasoning shown.
- **[H] hypothesis:** plausible but untested in this project.

---

## 0. Short answer

1. **[F] No embeddings are used anywhere** in the retrieval stack, from MQL-0005 (Week 3) to MQL-0010. One file, `scripts/rag/retrieval.py`, is copied byte-identically into every package. It ranks examples by **BM25 over schema-masked questions plus a Jaccard bonus from regex structure cues**. MQL-0010 adds structure-signature diversity on top.
2. **[F] The exclusion was intentional and is documented, but it was never tested.**
   - It was written down twice: Week 3 for a **280-document** corpus, and MQL-0010 §2.
   - No experiment ever ran a dense-retrieval arm.
   - The Week 3 justification ("a 280-document corpus needs no vector store") was carried over unchanged when the corpus grew to 3,589 and then 3,445 documents.
   - It also mixes up two claims: "no vector **database**" (true at any size here) and "no **embeddings**".
3. **[E] Given the evidence the project had, the decision was reasonable but incomplete.**
   - Measured over three experiments, the whole relevance effect of retrieval is small: hybrid − random is +3.8 pp in MQL-0009, and D0 − A0 is +2.2 pp (not significant) in MQL-0010.
   - The largest failure classes are things no retriever changes: schema-grounding errors, and constructs missing from the corpus.
   - So an embedding retriever can win at most a **few pp**, and probably **≤ 1–2 pp** of execution accuracy.
   - However, the method the project cites for its masking (DAIL-SQL) **does** use sentence embeddings over masked questions. The project kept the masking and swapped the similarity function for BM25 without testing that swap.
4. **[E] Recommendation.** Embeddings are a **legitimate untested gap, not a demonstrated missing capability**.
   - Test them cheaply, offline first. The candidate is **dense masked-question similarity fused with the existing BM25 + structure score, followed by the existing signature diversity**.
   - Escalate to a generation experiment only if the offline gate passes.
   - Do not build multi-vector or MQL-embedding retrieval without that evidence.

---

## 1. The current retrieval architecture

### 1.1 One retriever, unchanged since Week 3 [F]

`scripts/rag/retrieval.py` was compared with `diff` against the package copies. It is identical to:
- `kaggle/nl2mql_mql0007_rag_v1/src/retrieval.py`
- `kaggle/nl2mql_mql0008_approved_corpus_v2_1/src/retrieval.py`
- `kaggle/nl2mql_mql0010_holdout/src/retrieval.py`

MQL-0009 imports the same module (`kaggle/nl2mql_mql0009_retrieval_control/src/retrieval.py`). Its module docstring opens:

> "Local, deterministic retrievers over the RAG corpus (no external services, no embeddings)."

| Stage | Implementation | Source |
|---|---|---|
| Document construction | Corpus row → `{question, masked_question, mql, patterns}`. `patterns` = **SQL-derived** structure labels (`PT.sql_patterns(r["sql"])`), plus `text_number` | `mql0008_core.build_docs` (`kaggle/nl2mql_mql0009_retrieval_control/src/mql0008_core.py:173`) |
| Query representation | Question with schema words → `sch`, numbers → `num`, quoted or capitalised literals → `val` (DAIL-SQL-style masking) | `patterns.mask_question` (`scripts/rag/patterns.py`) |
| Candidate generation | **All** examples except those from the target DB; the index is rebuilt per target DB (leave-one-database-out IDF and normaliser) | `Retriever._index_for` |
| Lexical signal | Okapi BM25, k1 = 1.5, b = 0.75, over tokenised masked questions | `retrieval.BM25` |
| "Semantic" signal | **None.** TF-IDF cosine exists (`tfidf_*`) but is sparse; it was not selected after Week 3 | `retrieval.TFIDF` |
| Structural signal | `question_cues(q, schema)`: 12 regex families over the question, plus schema rules ("≥ 2 collections mentioned → join"; text-typed numeric fields → `text_number`), compared with the document's SQL labels | `patterns.question_cues`, `retrieval.jaccard` |
| Score fusion | `score = bm25 / max_bm25 + 1.0 · Jaccard(cues \ {select, where}, labels \ {select, where})` | `Retriever.rank_examples`, `hybrid_masked` |
| Tie-break / dedup | score desc, then `doc_id`; collapse on same id, MQL or normalised question | same |
| Selection (MQL-0005..0009) | Top-k = 3, greedy under a 1,500-token cap | `rag_prompt.py`, `mql0008_core` |
| Selection (MQL-0010, arm E / FULL) | Unchanged score; best document of each **new structure signature** in the top 50, k = 3 | `scripts/nl2mql_mql0010/src/retriever_v2.py` (`select_diverse`) |
| Evaluated but not adopted | Naive-Bayes structure predictor as the structural score (`structure_scoring="nb"`) | `retriever_v2.StructureAwareRetriever`; `outputs/nl2mql_mql0010/dev_evidence/retrieval_dev_proxy.json` |

**[F] No dependency for embeddings exists.**
- `requirements.txt` lists no sentence-encoder, `torch` encoder or ANN library (no `sentence-transformers`, `faiss`, `hnswlib` or vector DB client).
- `src/rag/` contains only `__init__.py`.
- The configs pin only BM25 parameters:
  - `kaggle/nl2mql_mql0008_approved_corpus_v2_1/configs/experiment.json:43-44`;
  - `kaggle/nl2mql_mql0009_retrieval_control/configs/experiment.json:236-237`;
  - `scripts/nl2mql_mql0010/configs/mql0010_experiment.json:129`: `"retrieval.Retriever(method='hybrid_masked', hybrid_weight=1.0), unchanged (BM25 k1 1.5, b 0.75)"`.

**Grep hygiene [F].** A repository-wide search for "embedding" also matches unrelated senses:
- `docs/schema_notes.md` (document **embedding** = denormalisation);
- `docs/native_nl2mql_dataset_research.md` (AptMQL's **embedded** schemas);
- `scripts/finetune_v1/train_qlora.py` (a `cosine` LR scheduler).

None of these is a retrieval embedding.

### 1.2 Experiment history of the retriever [F]

| Exp. | Corpus | Retriever arms | Result | Source |
|---|---|---|---|---|
| Week 3 offline | 280 docs (corpus_v1) | random, tfidf_raw/masked, bm25_raw/masked, hybrid w ∈ {0.25, 0.5, 1.0} | Selected hybrid w = 1.0: train-LODO mean Jaccard@3 0.561 vs bm25_masked 0.389, random 0.119. **No dense arm** | `docs/week3_rag_report.md` §5 |
| MQL-0005 | 280 | zero-shot / random_k3 / hybrid_k3 | 160 / 155 / 172 of 318; hybrid − random Holm p = 0.064 | ledger, `week3_rag_report.md` §7b |
| MQL-0007 | provisional v1, 3,589 | A zero-shot / B random / C hybrid | 421 / 433 / 437 of 892 (not in ledger) | `kaggle/nl2mql_mql0007_rag_v1/configs/experiment.json` |
| MQL-0008 | approved v2.1, 3,445 | zero-shot / hybrid | 415 / 457 of 892, p = 0.0018 | ledger |
| MQL-0009 | v2.1 | zero-shot / random ×5 / fixed / hybrid | 319 / 332.8 / 334 / 357 of 639; H1 hybrid − random +3.79 pp, Holm p = 0.0079 | ledger, PEA |
| MQL-0010 | v2.1 holdout view (2,912) | A0, D0, E, … (9 arms) | A0 397, D0 413, E 424, D0V 430, FULL 434 of 716; D0 − A0 +2.23 pp n.s.; E − D0 +1.54 pp, inconclusive | `docs/mql0010_final_results_analysis.md` |

### 1.3 Was the exclusion intentional? Documented or assumed? [F]

**Documented, three times:**
1. `docs/week3_rag_report.md:85`, decision table:

   > "local sparse retrieval (BM25 / TF-IDF), no embeddings | BEIR: BM25 is a robust zero-shot baseline. No paid APIs, CPU-only, deterministic. A 280-document corpus needs no vector store."

2. `kaggle/nl2mql_mql0008_approved_corpus_v2_1/README.md:30`:

   > "No embeddings, no vector database, no reranker, no guide cards."

   This line restates the frozen configuration; it is not a new argument.

3. `docs/mql0010_rag_architecture_design.md:63`:

   > "Not used, on purpose: embeddings, a vector database, an LLM reranker or an LLM-written query plan. Nothing in the evidence points at the semantic channel as the bottleneck. The lexical channel's randomised effect (+2 pp) is no larger than the structural one, and the dominant failure is not a retrieval failure at all."

**Not experimentally tested.**
- The Week 3 reasons are operational: free, CPU-only, deterministic, small corpus.
- The MQL-0010 reasons are indirect: error analysis shows the bottleneck lies elsewhere.
- Neither compared a dense retriever with BM25 on this task, offline or online.

**[E] Three weaknesses in the documented rationale:**
- **Scale drift.** "A 280-document corpus" was the Week 3 premise. From MQL-0007 on, the corpus held about 3,400–3,600 documents and the decision was not revisited.
- **Conflation.** "Needs no vector store" does not imply "needs no embeddings". Brute-force cosine over 3,445 × 384 floats (≈ 5 MB) needs no vector database and runs in milliseconds.
- **Citation gap.** Week 3 cites DAIL-SQL (Gao et al., arXiv 2308.15363) as the source for masking. DAIL-SQL §3.3 (checked 2026-11-21 on the arXiv HTML v4) says DAIL Selection "masks domain-specific words in both target question q and example questions q_i … [and] ranks the candidate examples based on the Euclidean distance between the **embeddings** of masked q and q_i". It then prefers candidates whose skeleton similarity to a pre-predicted SQL exceeds a threshold.
  - The appendix reportedly names `all-mpnet-base-v2` as the encoder. That detail comes from a summary of the appendix, not verbatim text, so treat it as unverified.
  - So the project adopted DAIL-SQL's masking and structural idea but replaced its **dense** question similarity with BM25, without an ablation.
  - In the paper's own Table 2, the dense masked-question baseline (MQS_S) is close to DAIL_S: 79.1 vs 80.2 at 1-shot and 82.0 vs 82.4 at 5-shot, GPT-4, single runs. Structure filtering added little there over dense masked similarity.

---

## 2. Why were embeddings excluded? Facts vs hypotheses

| Candidate reason | Status | Evidence |
|---|---|---|
| Small corpus (280 at the time) | **[F] documented** (Week 3) | `week3_rag_report.md:85`. Not re-examined at 3,445 |
| Lexical/hybrid considered sufficient | **[F] documented, partly evidenced** | Hybrid beat BM25 and TF-IDF on the structural proxy (Week 3 §5) and beat random on accuracy (MQL-0009 H1). Never compared with dense |
| No paid APIs, CPU-only, deterministic | **[F] documented** | Week 3 row. Local encoders satisfy all three (§4) |
| Semantic channel is not the bottleneck | **[F] documented** (MQL-0010 §2) | PEA §5.3, §6.1 (see §3.0 below) |
| Embeddings don't capture MongoDB structure | **[H] not documented** | Plausible (§3), never argued in the docs |
| Embeddings retrieve schema-incompatible examples | **[H] not documented** | The retriever never shows target-DB examples, so "schema-compatible" is not defined for examples here; see §3 B |
| Latency, memory, infrastructure | **[H] implied** by "CPU-only" | Real cost is negligible for a small encoder (§4) |
| No suitable model or evaluation evidence | **[E] partly true** | No embedding evaluation exists; models do exist (§5.2) |
| Avoiding another model or dependency | **[H] implied** by the determinism and reproducibility culture (byte-identical rebuilds, frozen packages) | Real cost: pinned weights, float nondeterminism across hardware |

---

## 3. Would embeddings help? By representation and by task facet

### 3.0 The ceiling: how much can *any* retriever change? [F → E]

The question needs this framing first, because it bounds every answer below.

| Quantity | Value | Source |
|---|---|---|
| Whole retrieval relevance effect, hybrid − random | +3.79 pp [1.28, 6.32] | MQL-0009 H1 |
| Retrieval vs zero-shot on the MQL-0010 holdout | +2.23 pp, not significant | `docs/mql0010_final_results_analysis.md` finding 6 (D0 − A0) |
| Schema-feedback retry, which is not retrieval | +2.37 pp, p = 1.5e-5 | MQL-0010 D0V − D0 |
| Empty-result failures (wrong field or collection paths) | ≈ 100 per run, **identical across zero-shot, random, fixed and hybrid** | PEA §6.1 |
| Questions needing constructs absent from the corpus (subqueries, set ops, anti-joins, `let`/`pipeline` lookups, conversions) | ≈ 14%, ≈ 11% accuracy in every run | MQL-0010 design §0, PEA §6.2 |
| Randomised within-question effects in random runs | exact structure shown +2.36 pp (p 0.19); lexical above median +2.01 pp (p 0.041, unadjusted) | PEA §5.3 |
| Dev proxy (779 dev items, 2,912-record pool): gold signature exists in pool | 679 / 779 | `retrieval_dev_proxy.json` |
| Baseline hit@3 on attainable items → with diversity (E) | 65.5% → 75.6% | same |

**[E] What follows from these numbers:**
- Retrieval quality moves accuracy by a few pp at most on this model (Qwen2.5-Coder-3B) and corpus.
- With diversity, three in four attainable dev items already see their exact gold structure in the top 3. The structural headroom is about 24% of 87% of items.
- Even if a better retriever closed half of that gap, the randomised effect of seeing the gold structure (≈ +2.4 pp per item) implies ≈ 0.12 × 2.4 pp ≈ **0.3 pp** through the structural channel.
- Gains through other channels are unknown: output-shape priming, column naming, paraphrase-matched idioms. PEA §6.4 says about half of the hybrid's gain is output-shape priming that random examples also deliver.
- So **any embedding gain lives in a window of roughly 0–2 pp**, which a 716-item paired test cannot resolve: E vs D0 was +1.54 pp with power 0.24–0.43.

### 3.1 Facet by facet

**A. Natural-language semantic retrieval.**
- [E] This is the one place embeddings can plausibly beat BM25: paraphrase ("how many" vs "number of", "oldest" vs "maximum age", "never" vs "not … any").
- BM25 over masked text matches only surface tokens. The regex cues partly compensate for a fixed vocabulary, with exact agreement on only 33% of questions (PEA §5.5).
- [H] A dense encoder will capture intent words ("each", "at least", "both", "more than average") more robustly than the regexes, especially the low-recall cues: `nested` recall 0.20, `group_by` 0.46, `join` 0.45, `count_distinct` 0.
- **Risk:** on **raw** questions, embeddings cluster by **topic** ("students", "flights"). Every candidate comes from another DB, so topic similarity is mostly noise. That is exactly why masking exists.

**B. Schema-aware retrieval.**
- [E] Low value as a retrieval signal.
  - The target schema is already in the prompt verbatim (Week 3 decision row; MQL-0010 §3).
  - Examples are always from other databases, so no candidate can be "schema-compatible" in the literal sense.
- What could transfer is schema **shape**, e.g. "query touches 2 collections linked by a FK" or "a text-typed numeric column". That is better expressed as discrete features (collection count, FK path length, type classes) than as an embedding of schema text.
- [F] The dominant schema failure (≈ 100 empty results) is not retrieval-addressable. The MQL-0010 validator + retry targets it and is the measured win.

**C. MongoDB operator and aggregation-structure retrieval.**
- [E] Embedding **MQL** pipelines is the wrong tool on the query side: at inference time there is no MQL for the question, only the question.
- MQL-side embeddings help only if:
  - (i) a draft pipeline is generated first (DAIL-style pre-prediction), costing an extra generation; or
  - (ii) they are used as a document-side feature.
- Exact structure is better captured by the existing discrete representations: SQL labels, structure signature, stage skeleton (44 distinct skeletons, 65 signatures; PEA §6.2). Jaccard or exact match on these is precise, interpretable and free.
- [H] General text encoders handle JSON pipelines poorly: field names dominate the token mass, and `$facet` count wrappers (23% of the corpus, PEA §5.4) would dominate similarity.

**D. Joins, grouping, subqueries, set operations.**
- [F] Subqueries, set operations and anti-joins have **0 corpus examples**, so no retriever can show them.
- [E] For joins and grouping, the question-side cue is the bottleneck: join recall 0.45, group_by 0.46. A dense question encoder may recall these better than regexes. That is testable offline.

**E. Different wording, same pipeline.**
- [E] This is the best-motivated case for embeddings, and the one the current offline proxy measures worst.
- The proxy (`scripts/nl2mql_mql0010/analysis/retrieval_dev_proxy.py`) scores **structure-signature** hit@3, which is the same signal the hybrid's Jaccard term optimises. A retriever that finds same-shape examples through paraphrase would score fairly. A retriever that finds better *semantic* matches with the same coarse signature would get no credit.
- §5 adds a finer proxy (stage skeleton plus output shape).

### 3.2 One embedding per whole example? [E]

Not appropriate. A concatenated "question + SQL + MQL" vector mixes three things:
- a query-side-available signal (the question);
- two document-only signals (SQL, MQL);
- a domain signal (schema nouns) that is irrelevant across databases.

The query vector can only be built from the question, so document vectors should be built from the **same representation** (masked question). Any structure-side similarity should stay in the existing discrete channel. Multi-vector designs only make sense with a query-side structure estimate (a draft SQL or MQL, or the structure predictor), and that adds a generation step or a weak signal (NB predictor: exact signature 36.8%).

---

## 4. Architecture alternatives

| # | Architecture | Expected relevance | Structural compatibility | Latency / memory | Complexity | Risk of misleading examples | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Current hybrid** (BM25-masked + cue Jaccard) + E diversity | Baseline. Only retriever with a registered effect (MQL-0009 H1) | Explicit but coarse (regex cues; score plateaus at 2.0 / 1.0, ties at ranks 3/4 on 12%) | ~4 ms/query CPU, no model | Done, frozen | Concentration (52% same-signature triples), fixed by E | Keep as control |
| 2 | Dense only (masked question) | [H] better paraphrase recall; loses exact-token precision on rare intent words | None explicit; likely **lower** hit@3 than the hybrid (BM25-masked alone 0.389 vs hybrid 0.561 mJ@3 in Week 3) | Encoder ~33–110 M params; 3,445 × 384–768 floats ≈ 5–10 MB; encode ≈ 1–3 min CPU once, ≈ 10–30 ms/query | Low | Topic leakage if masking is imperfect | Offline ablation only |
| 3 | **Lexical + dense hybrid + cue Jaccard + E** (replace or fuse the BM25 term with cosine, e.g. RRF or normalised sum) | [H] Small, positive: keeps structure, adds paraphrase robustness | Same as #1 | As #2, plus BM25 | Low (one score term) | Low; structure term and diversity still guard | **Test first** |
| 4 | Dense candidates + AST/skeleton reranking (DAIL-like, needs a draft SQL/MQL from a first pass) | [H] Better structural precision when the draft is right | High if the draft is right; misleading when the draft is wrong | **+1 generation per question** (≈ doubles the 3B cost) | Medium | Wrong draft → three confidently wrong demos | Only if #3 passes and the structure channel remains the bottleneck |
| 5 | Multi-vector (question, schema, MQL vectors with learned weights) | Unknown | Depends on query-side structure estimate | 3 indexes, weight tuning | High | Overfitting weights to ~800 dev items | **Not recommended**: no evidence, too many knobs for the data |

---

## 5. Proposed controlled experiment (smallest useful)

### 5.1 Constraints honoured
- Approved corpus v2.1 (`outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl`, sha256 96552f06…) is only read. Retrieval pools are **views**, as in MQL-0010.
- `scripts/rag/retrieval.py`, `retriever_v2.py` and all package copies stay unchanged. A new retriever goes in a **new** module that wraps them.
- Protected sets are never used for tuning or selection:
  - all 20 Spider-dev DBs: MQL-0008's 892 items (17 DBs) plus the 3 test DBs holding the 65-item test partition that CLAUDE.md reserves;
  - MQL-0009's 639 items (16 DBs);
  - MQL-0010's 716 holdout items (18 DBs).
- MQL-0010 registration A4 is untouched. This would be a new ID (e.g. MQL-0011), registered separately.

### 5.2 Embedding models (local, open weights, no API)

| Candidate | Size | Why |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 33 M, 384-d | Strong small general encoder, CPU-fast; the default |
| `sentence-transformers/all-mpnet-base-v2` | 110 M, 768-d | Reportedly the DAIL-SQL encoder (unverified, §1.3); replicates the cited method |
| `intfloat/e5-base-v2` or `Qwen/Qwen3-Embedding-0.6B` | 110 M / 0.6 B | One larger check, to see whether encoder capacity matters before concluding "dense doesn't help" |

Code-specific encoders are not first priority, because only the question is embedded (§3.2). Pin every model revision, as is done for Qwen.

### 5.3 Data (database-disjoint) [F for counts]

| Role | Data | Size |
|---|---|---|
| **Selection (offline only)** | The registered MQL-0010 **dev** split: 15 DBs, 779 items (`outputs/nl2mql_mql0010/preflight/split_registration.json`). Pool = v2.1 minus 18 holdout DBs (2,912), leave-own-DB-out. **507 of the 779** have a verified v2.1 MQL record, so the skeleton/shape proxies (§5.5) cover 507 items; the signature proxy covers all 779 | 779 |
| **Confirmation (generation)** | A new seeded, pre-registered DB-level draw from the **90 build DBs**. Same item filter as `make_split.py` + amendment A1: spider-sql-v1 train, `tier == exec_ok`, no degenerate gold, spider-pool-v3 membership. **3,295 eligible items** (2,405 are themselves corpus records; LODO removes their own DB from retrieval). Draw DBs until ≈ 1,400–1,600 items | ≈ 1,500 |

The confirmation DBs are disjoint from the dev DBs, the holdout, MQL-0009 and Spider dev.

**Retrieval view.** Use **per-query leave-own-DB-out** over v2.1 minus the 18 MQL-0010 holdout DBs, as on dev. This keeps the pool near 2,912. A holdout-style view, removing every confirmation DB at once, would strip a large share of the corpus, because the build DBs supply most of it. The pool would then no longer represent the deployed retriever. Exclude the dev DBs from the pool too, if the registration wants dev and confirmation to see the same pool.

**Leakage audit.** Re-run the `make_split.py` L1–L6 audit on the draw: DB and id overlap STOP; exact, normalised and near-duplicate (token Jaccard ≥ 0.8) question exclusion pairs, identical for every arm; identical gold SQL reported. **Prerequisite and cost:** these DBs are not known to be provisioned on Atlas, and the corpus sign-off keeps corpus data off Atlas except as eval collections. Provisioning build-DB collections as eval collections needs its own approval.

### 5.4 What to embed, indexing, ranking
- **Text:** `masked_question` (existing `PT.mask_question`), plus an arm on the raw question, because encoders may handle literal `sch`/`val` tokens poorly. [H] A variant replacing them with natural placeholders ("the table", "the value") may suit encoders better; include it only as a dev-side ablation.
- **Index:** brute-force cosine on L2-normalised vectors (numpy). No vector DB, no ANN. Leave-target-DB-out by masking, exactly as `_index_for`.
- **Fusion:**
  - (a) replace `bm25/max_bm25` by cosine (min-max normalised per query); or
  - (b) reciprocal-rank fusion of BM25 and dense, k = 60;
  - then add the unchanged `1.0 · Jaccard(cues, labels)` and apply the unchanged `select_diverse` over the top 50.
- **Determinism:** store the doc and query vectors and their hashes in the package, computed once on CPU in fp32, so the GPU run does no embedding.
- **K = 3**, as in every registered arm, keeping the token budget comparable. K = 5 only as an offline sensitivity check.

### 5.5 Stage 1: offline gate (no model generation, dev only)

| Arm | Description |
|---|---|
| R0 | `bm25_masked` |
| R1 | current hybrid (D0 ranking) |
| R2 | R1 + diversity (**E**, the control to beat) |
| R3 | dense masked question only |
| R4 | dense raw question only |
| R5 | E with BM25 term → cosine (fusion a) |
| R6 | E with RRF(BM25, dense) (fusion b) |
| R7 | best of R5/R6 with the second and third encoder |

**Metrics:**
1. Existing structure-signature hit@3 / hit@3-attainable / mJ@3.
2. **New, finer proxies** to avoid the bias in §3.1 E:
   - **stage-skeleton hit@3** against the dev item's verified corpus MQL (available for dev items that are corpus records);
   - **operator Jaccard@3** on `mql_operators`;
   - **output-shape match** (number of projected fields, `_id` handling).
3. Diversity: share of all-same-signature triples.
4. Latency per query.
5. **Triple-change rate:** the share of dev items whose selected triple differs from E. It feeds the stage-2 sample size (§5.6).
6. **Oracle-structure ceiling (dev only):** an arm that picks examples by the dev item's **gold** signature (allowed on dev, which is not protected). It bounds what any better retriever could add through the structural channel. Optionally, generate this arm on dev with the model to turn the bound into accuracy.

**Pre-registered gate:** the best dense arm must beat E on hit@3-attainable by **≥ 3 points** *and* not lose on skeleton hit@3. Paired bootstrap over items; the bar is fixed before anything is computed.
- Fails → stop and report "dense retrieval adds no **structural** relevance". That is a valid, citable null. It does **not** exclude a small accuracy effect through non-structural channels (paraphrase, output-shape priming), because every stage-1 proxy is structural.
- Cost: ≈ minutes on a laptop CPU.

### 5.6 Stage 2: accuracy test (only if the gate passes)
- **Arms (two, paired):** E (control, first pass, retry off) vs E-dense (the single arm selected in stage 1). Optional third arm: A0 zero-shot for context; it is not a hypothesis test.
- **Model and decoding:** unchanged (Qwen2.5-Coder-3B-Instruct @ 488639f1…, bf16, greedy, max_new_tokens 1024, T4, batch 8 + batch-1 fallback). Unchanged evaluator (`dev1034_core`), cross-engine, read-only Atlas.
- **Primary hypothesis:** E-dense vs E, exact McNemar **two-sided**, α = 0.05, as in earlier registrations. Report the paired bootstrap 95% CI.
- **Power:** E vs D0 had 49/716 ≈ 6.8% discordant pairs. At that rate, detecting +2 pp at 80% power two-sided needs n ≈ (1.96·√0.068 + 0.84·√0.068)² / 0.02² ≈ **1,330**. That rate is optimistic: E vs D0 changed only the selection, while a dense ranking changes more triples. At ≈ 15% discordance, n ≈ **2,900**, close to the 3,295-item cap. **Recompute n from the stage-1 triple-change rate before registering stage 2.** If the required n exceeds the pool, register a larger minimum detectable effect rather than an underpowered +2 pp test.
  - Effects below +2 pp, the likely range (§3.0), will not be detectable at this size. Say so in the registration.
- **Compute:** 2 × 1,330–2,900 ≈ 2,700–5,800 generations ≈ 4–8 MQL-0010-sized passes, i.e. one to three Kaggle T4 sessions. Embedding cost is negligible and done offline. Atlas scoring is as for MQL-0010, after provisioning.
- **Secondary (descriptive):** accuracy by gold family (single table / join / group / unsupported); triple-signature diversity; idiom copy rates (`__jN`, `$lookup`); the overlap of fixes with the D0V retry fixes.

---

## 6. Final answers

**1. Why did we originally avoid embeddings?**
- [F] Week 3 chose local sparse retrieval for operational reasons: no paid APIs, CPU-only, deterministic, and "a 280-document corpus needs no vector store". BEIR's finding that BM25 is a strong zero-shot baseline was cited.
- MQL-0010 re-affirmed the choice because its error analysis located the bottleneck outside retrieval.
- No experiment compared dense with sparse retrieval.

**2. Was that decision technically justified?**
- [E] **Partly.** For the MQL-0010 priority order, yes: the measured win came from schema-feedback retry (+2.37 pp, p = 1.5e-5), and empty-result errors are untouched by any retriever.
- As a claim that "embeddings would not help", **no**:
  - the premise (280 docs) expired;
  - "no vector store" was conflated with "no embeddings";
  - the cited method (DAIL-SQL) uses dense masked-question similarity, which the project replaced without an ablation.

**3. What limitations does the current architecture have because of that decision?**
- [E] Mainly **paraphrase brittleness on the question side**. Intent is read only through surface tokens (BM25) and 12 regex families (plus a schema-mention join rule) with weak recall:
  - exact agreement 33%;
  - join recall 0.45, group_by 0.46, nested 0.20, count_distinct never predicted.
- Indirectly, this produces the score plateaus and ties (12% tie at ranks 3/4) that the `doc_id` tie-break resolves arbitrarily.
- It does **not** cause:
  - the coverage gap (≈ 14% of questions, which needs corpus work);
  - schema-grounding errors (validator + retry).

**4. Would embeddings likely improve our NL→MQL retrieval?**
- [E] **Retrieval relevance: plausibly a little.**
- **Execution accuracy: probably ≤ 1–2 pp, possibly 0.** The whole relevance effect of retrieval measured so far is +2–4 pp, and with diversity three in four attainable items already see their gold structure.
- [H] The most likely gain is on join and group-by questions phrased without the regex trigger words.

**5. Which embedding architecture should be tested first?**
- **Alternative 3:** the existing hybrid and diversity pipeline with the BM25 term replaced by, or RRF-fused with, dense cosine over **masked questions** from a small pinned local encoder (`bge-small-en-v1.5`; `all-mpnet-base-v2` as the DAIL-SQL replication).
- It changes one score term, keeps the structural guard and E diversity, and costs milliseconds.

**6. What experiment would provide convincing evidence?**
- The two-stage design in §5:
  - an offline, pre-registered gate on the 779-item MQL-0010 dev split, with a finer skeleton- and shape-based proxy;
  - then, only if it passes, a paired E vs E-dense generation test on a fresh draw of ≈ 1,300–2,900 items from the 90 build DBs (n set from the stage-1 triple-change rate). Two-sided McNemar.
- A stage-1 null is strong, nearly free evidence that dense retrieval adds no **structural** relevance. It leaves open only a small accuracy effect through other channels, which the §3.0 ceiling already bounds at a few pp.

### Ranked recommendations

| Rank | Recommendation | Expected benefit | Cost | Complexity |
|---|---|---|---|---|
| 1 | Stage-1 offline gate (R0–R7) on the dev split, with the added skeleton/shape proxy and the oracle-structure ceiling | Answers the structural half of the mentor's question with data; a null rules out structural gains, not every accuracy effect | Minutes CPU; one new module; pinned encoder weights | Low |
| 2 | If the gate passes: stage-2 paired E vs E-dense on ≈ 1,300–2,900 build-DB items | Detects an accuracy effect of ≥ 2 pp, if one exists | ≈ 2,700–5,800 generations (1–3 T4 sessions) + Atlas provisioning approval | Medium |
| 3 | Independent of embeddings: corpus coverage for subqueries and set operations (MQL-0010 design §8) | Targets ≈ 14% of questions that no retriever can help | Converter + verifier work, a new corpus version | Medium–high |
| 4 | DAIL-style draft-then-rerank (alternative 4) | Only if structure remains the bottleneck after #1–#2 | +1 generation per question | Medium |
| 5 | Multi-vector or MQL-pipeline embeddings (alternative 5) | Unknown; high overfitting risk | High | High: not recommended without prior evidence |

**Bottom line [E].** Embeddings are an **untested design choice, not a demonstrated missing capability**. The current evidence says the large remaining errors are schema grounding and corpus coverage, not semantic retrieval. The mentor's question is still fair: the exclusion was never ablated, and the method the project cites uses embeddings. The proportionate response is the cheap offline gate, not a new architecture.
