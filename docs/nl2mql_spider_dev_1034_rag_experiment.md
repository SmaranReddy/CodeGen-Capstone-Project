# RAG + Qwen2.5-Coder-3B on the 892-item NL→MQL evaluation set (planned MQL-0007)

**Status: BLOCKED — NOT PREPARED (2026-10-30).** No bundle, notebook or scoring code was built, and nothing was run. The training corpus fails the leakage and label-verification rules. MQL-0007 is the next unused NL→MQL ledger ID. It has **not** been entered in the ledger, because no experiment has been prepared.

Audit: `scripts/nl2mql/audit_rag_corpus_dev1034.py` writes:
- `outputs/nl2mql_spider_dev_1034_rag/corpus_audit_v1/candidates.jsonl`, with one row per candidate giving source, record ID, licence, verification status, decision and reasons;
- `summary.json`, with counts and input hashes.

Tests are in `tests/test_nl2mql_rag_audit.py`.

## 1. Research question (unchanged, for when it is unblocked)

Does retrieving verified NL→MQL examples and MongoDB documentation improve Qwen2.5-Coder-3B-Instruct's cross-engine execution accuracy over the zero-shot MQL-0006 baseline? The test would use the same 892 items, prompt base, model revision and decoding.

Baseline MQL-0006 (verified 2026-10-30, `outputs/nl2mql_spider_dev_1034_model_comparison/v1/`): **421/892 (47.2%)**, Wilson 95% CI 43.9–50.5%. These figures were checked on 2026-10-30 against `scoring/models/qwen2.5-coder-3b-instruct/baseline_summary.json`:
- status complete;
- revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`;
- format validity 889/892 (99.66%);
- Atlas execution success 873/892 (97.87%);
- not_comparable 39.

## 2. Why it is blocked

The evaluation bundle `kaggle/nl2mql_spider_dev_1034_bundle_v2.zip` has SHA-256 `e9986dbe…0760`, which the audit verified. It contains 892 records on 17 databases.

| Candidate source | Rows | Accepted | Exclusion reason |
|---|---:|---:|---|
| `data/rag/corpus_v1` example docs (the MQL-0003..0005 retrieval corpus) | 280 | 0 | **Every one is an evaluation item.** Matched by ID (dev_index → `spider_dev_NNNN`) and by normalised question; all 12 source databases are evaluation databases. |
| `data/finetune/ft_v1` Spider rows | 280 | 0 | The same 280 evaluation items (identical pairs). |
| `data/finetune/ft_v1` Kaggle NL2SQL rows (train 1,504 + dev_internal 80) | 1,584 | 0 | Labels come from `sql2mql-v1`. `sql2mql-v2` does not exist (roadmap P4, V1–V3 unmet). They are only `fixture_equivalence_verified`, with no source DB, and the audit verifier accepted 10/20 wrong translations. 88 are known-wrong (B1/B2) and 51 have date-equality SQL errors. No evaluation leakage was found in these rows. |
| Native NL→MQL data on disk (AptMQL, SM3, TEND-v1) | 0 | 0 | Never downloaded or re-executed. |
| **Total supervised** | 2,144 rows (1,864 unique pairs) | **0** | |

Why the earlier RAG was legitimate but does not carry over: MQL-0003..0005 evaluated on the expanded 318/72 *partitions of Spider dev*, retrieving leave-one-database-out from other Spider-dev databases. The 892 set **is** Spider dev (minus the 3 test DBs). Every Spider-dev example is therefore an evaluation item, and the brief forbids evaluation items in the retrieval corpus.

**Documentation (RAG-only, not supervised):**
- 18 guide cards (`data/rag/sources/mongodb_guides.yaml`).
- 4 are excluded because their example runs on `concert_singer`, an evaluation database: `guide_join_lookup`, `guide_join_type_mismatch`, `guide_anti_join`, `guide_count_per_parent`.
- 14 have generic inline `$documents` examples. Their status is *pending licence check*: they paraphrase the official MongoDB manual (URLs per card), but the source file records no licence.

**Schema docs:** the 80 schema documents are not retrieval documents. The target schema is already in the prompt.

**Fine-tuning consequence:**
- ft-v1 cannot be the fine-tuning set for an experiment scored on these 892 items: 280 of its 1,784 train rows are evaluation items.
- `docs/fine_tuning_plan.md` has not been edited. It needs revising before any fine-tuning run is scored on this set.
- The "canonical approved training collection" this experiment was meant to create would be empty today.

## 3. Recovered MQL-0005 retrieval configuration (frozen, for reuse once a clean corpus exists)

The sources are `configs/rag_week3.yaml` (arm `hybrid_k3`), `scripts/rag/retrieval.py` and `scripts/rag/rag_prompt.py`.

**Retriever: `hybrid_masked`**
- Score = Okapi BM25 score, normalised by the top score, + 1.0 × Jaccard(question cues, document SQL-pattern labels).
- BM25 parameters: k1 = 1.5, b = 0.75.
- `select`/`where` are ignored in the Jaccard term.

**Query preprocessing (DAIL-SQL-style masking):**
- schema words → `sch`
- numbers → `num`
- quoted or capitalised literals → `val`
- Tokens are lowercase alphanumeric.

**Index and ranking:**
- The index is built per target database and excludes target-db documents.
- Ties are broken deterministically: score descending, then `doc_id`.
- Deduplication drops repeats of the same doc_id, the same MQL or the same normalised question; the higher-ranked copy is kept.

**Context:**
- k = 3 examples, k_guides = 0 (guides were not in the primary arm; `hybrid_k3_g3` scored 41 vs 43 on validation).
- Reference block budget: `max_context_tokens` 1500, `margin` 64, n_ctx 8192, `max_new_tokens` 1024.
- Over-budget documents are skipped and recorded in `dropped_budget`, never truncated.
- Prompt version `nl-mql-spider1-rag-v1`: the zero-shot `nl-mql-spider1-v1` prompt with one reference block inserted before the final `Question:` line.

**Known differences for a future run:**
- MQL-0005 ran GGUF q4_k_m and budgeted with the llama.cpp tokenizer. MQL-0006 ran HF bf16 at `488639f1ff808d1d3d0ba301aef8c11461451ec5`. The budget must use the HF tokenizer, which is a documented, unavoidable difference.
- The pattern labels for corpus examples come from gold SQL (`sql_patterns`). A corpus without SQL (for example native MQL) needs an MQL-derived equivalent, which would be a deviation from MQL-0005.

## 4. Options to unblock (user decision)

1. **`sql2mql-v2` + Spider-train conversion** (roadmap P4/P5). Spider *train* databases are disjoint from dev. This needs Docker and a local `mongo:8.0` for label validation, and takes several days. It gives the largest clean corpus, and the same pairs would be reusable for fine-tuning.
2. **Native sets: AptMQL train (CC BY-SA) / SM3 (Apache-2.0)**, fetched and re-executed locally (P2). This also needs Docker. TEND-v1 has no stated licence and contains 19 of our 20 benchmark DBs. Any new source must be added to `audit_rag_corpus_dev1034.py` and pass it before use.
3. **Documentation-only RAG** over the 14 generic guide cards, after their licence is verified. This would be a **new configuration**, not MQL-0005, and it answers a different question: does documentation help, rather than do examples help.

## 5. Disclosure to carry into any eventual result

"The metric measures cross-engine execution equivalence between generated MQL on MongoDB Atlas and Spider gold SQL on SQLite. It is not native MQL exact-match accuracy. Matching on a database instance does not prove general semantic correctness, and source gold SQL has not been independently human-verified for every question."

The old 318-item RAG results (MQL-0003..0005) are not comparable to the 892-item set.

## 6. Reproduce the audit

```
python scripts/nl2mql/audit_rag_corpus_dev1034.py
python -m pytest tests/test_nl2mql_rag_audit.py -q
```
