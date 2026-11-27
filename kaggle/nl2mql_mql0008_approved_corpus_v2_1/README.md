# MQL-0008 — Approved Corpus v2.1 Hybrid RAG (Kaggle package, generate only)

**Status: PACKAGE BUILT — NOT YET EXECUTED.** Nothing in this package has produced a model prediction. No MQL-0008 accuracy exists.

Question: does hybrid retrieval of examples from the **approved corpus v2.1** improve `Qwen/Qwen2.5-Coder-3B-Instruct`
on the 892 MQL-0006 questions, compared with zero-shot?

| arm | context | predictions |
|---|---|---|
| `arm_a_zero_shot` | none (frozen `nl-mql-spider1-v1` prompt) | 892, generated in this run |
| `arm_b_hybrid_rag` | 3 examples from the approved corpus v2.1, `hybrid_masked` retrieval | 892, generated in this run |

Exactly two arms, 1,784 predictions. There is no random-retrieval arm and no supplementary arm.

## Retrieval corpus

- Source: `outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl`, shipped byte for byte as `data/corpus/approved_corpus_v2_1.jsonl`.
- 3,445 records, SHA-256 `96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf`.
- The builder aborts and the notebook stops if the hash or the count differs. No other corpus is in the package.
- Review status: AI-reviewed (G-AIR, AI-only), never human-reviewed. Labels were validated on a local MongoDB 8.0.34 with
  verifier 3.1 and were never executed on Atlas.

## Frozen configuration (equal to MQL-0007, checked by the builder and the tests)

- Model `Qwen/Qwen2.5-Coder-3B-Instruct`, revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`, bfloat16, greedy
  (`do_sample=False`), `max_new_tokens` 1024, batch 8, seed 42, one GPU (Kaggle T4). A batch that hits CUDA OOM is retried
  one prompt at a time with the same settings.
- Retrieval: BM25 k1 1.5, b 0.75, plus 1.0 × pattern overlap (`hybrid_masked`), top K 3, context cap 1,500 tokens, over-budget
  examples skipped (never truncated), target database excluded, ties by score then `doc_id`, dedup on id / MQL / normalised
  question. No embeddings, no vector database, no reranker, no guide cards.
- Prompt: the RAG prompt is the zero-shot prompt with one reference block inserted before the final `Question:` line.
  System prompt, schema text and question are identical in both arms.
- `src/patterns.py`, `src/retrieval.py`, `src/rag_block.py` are byte-identical to the MQL-0007 copies.

Differences from MQL-0007 are listed in `configs/experiment.json` (`stated_differences_from_MQL-0007`). The intended one is
the corpus. The others: two arms instead of four; arm A is always generated here (MQL-0006 predictions are not bundled);
the corpus v2.1 record layout has no `schema.touched`, so the `text_number` pattern flag is empty (it was also empty for all
of corpus v1); extra identity fields in plan rows and prediction records; output folder `predictions/<arm>/`.

## Leakage audit (fresh, against corpus v2.1)

Report: `data/leakage_report.json`. Retrieval audit: `data/retrieval_audit.json`.

- Exact question string: 0. Normalised question: 0. Database overlap with the 17 evaluation databases: none.
- Corpus SQL equal to an evaluation gold SQL: 0.
- Near-duplicates (token Jaccard ≥ 0.8): 2 pairs, the same as in MQL-0007: `spider_train_00650` with `spider_dev_0383` and
  `spider_dev_0824` (same question template, one noun differs, different databases).
- Rule re-applied unchanged: the corpus example is excluded from retrieval **for those two evaluation items only**, after
  ranking. The corpus file is untouched and all 892 evaluation items stay in the set.
- Protected sets (Spider dev incl. the 65 held-out items, Spider test) were **not opened**. Every approved record has the same
  id, question, SQL and database as its corpus v1 record, and corpus v1 had zero membership hits in the MQL-0007 audit, so the
  result carries over to this subset. The three held-out database names are checked directly.

## What is in the package

In: the approved corpus, its derived retrieval features, the 892 evaluation inputs (`id`, `db_id`, `question` only), the frozen
zero-shot prompts, the schema context of the 17 evaluation databases, a Spider `tables.json` subset for masking, the frozen
retrieval plan of arm B, config, code, tests, notebook, audits, `package_manifest.json`, `SHA256SUMS.txt`.

Never in: gold SQL, gold rows, SQLite files, the held-out partition, credentials, corpus v1, provisional corpus v2,
spider-pool-v3, earlier predictions or scores. The notebook needs no database connection of any kind.

## Kaggle execution (exact steps)

1. **Dataset:** Kaggle → *Datasets → New Dataset* → upload `kaggle/nl2mql_mql0008_approved_corpus_v2_1_bundle.zip`. Keep it
   **Private**. Kaggle extracts it under `/kaggle/input/<dataset>/nl2mql_mql0008_approved_corpus_v2_1/`.
2. **Notebook:** *New Notebook → File → Import Notebook* → `nl2mql_mql0008_approved_corpus_v2_1.ipynb`. *Add data* → your dataset.
3. **Settings:** Accelerator **GPU T4 x2** (one card is used), Internet **On**, Persistence *Files only*. Attach no secrets.
4. After the first run starts, set *Settings → Environment Preferences → Pin to original environment* so resumes use the same libraries.
5. **Run:** *Save Version → Save & Run All (Commit)*. Defaults run both arms.
6. **Download** `/kaggle/working/mql0008_predictions.zip`. It holds `mql0008_outputs/` with:
   - `predictions/arm_a_zero_shot/predictions.jsonl`, `predictions/arm_b_hybrid_rag/predictions.jsonl`
   - per arm `run_config.json`, `retrieval_log.jsonl`, `checkpoints/`
   - `MQL0008_STATUS.json` (must say `"status": "complete"`, `"predictions_total": 1784`), `preflight/preflight.json`, `SHA256SUMS.txt`

**Time.** In MQL-0007 on a T4 the zero-shot arm took 2.3 h and the hybrid arm 3.4 h. Both arms should fit one 12-hour
session. To split, run `RUN_ARMS = ["arm_a_zero_shot"]` in one version and `["arm_b_hybrid_rag"]` in another, with the same
pinned environment, and keep both zips.

**Resume.** If a version stops, start a new version, *Add data → Notebook Output* of the stopped version, set
`RESUME_FROM = "auto"` and keep `RUN_ARMS`. The earlier output is hash-verified, checkpoints are copied, and only unfinished
ids are generated. Completed records are never regenerated, and a different configuration is refused.

## What the notebook checks (and stops on)

GPU present, ≥ 14 GiB, bfloat16 supported, enough disk and RAM · no credential file attached · every package file's
SHA-256, no extra file · corpus SHA-256 and 3,445 records · exactly two arms · 892 unique evaluation ids with the recorded
hash · leakage re-check · pinned model revision · corpus features re-derived equal to the frozen ones · **retrieval plan
re-derived with the model's tokenizer equal to the frozen plan** · nothing retrieved from the target database · context ≤
1,500 tokens · arm A prompts byte-identical to the frozen zero-shot prompts · no duplicate, missing or out-of-order id ·
every prediction from its planned prompt with its model, arm and retrieval identity.

## Each prediction record holds

`id`, `db_id`, `arm`, `raw_text` (the generated MQL), `status`, `model`, `model_id`, `model_revision`, `generation_config`,
`prompt_sha256`, `fingerprint`, token counts, OOM-fallback flags, and for arm B a `retrieval` object: retrieved ids, ranks,
scores, source databases, target-database exclusion flag, rule exclusions, over-budget skips, context tokens and cap, corpus
SHA-256 and the retrieval configuration fingerprint.

## Scoring (separate, local, not part of this package)

Kaggle only generates. Scoring runs on Windows against Atlas with the read-only account and the established evaluator
(the one used for MQL-0006 and MQL-0007). The existing adapter `scripts/nl2mql/mql0007_scoring.py` is bound to the MQL-0007
package and its three arms; a two-arm MQL-0008 adapter with a pre-registered analysis plan still has to be written before
scoring. The metric definitions must not change.

Historical results, for reference only. **They are not MQL-0008 results:**

- MQL-0006 zero-shot: 421/892 (47.20%).
- MQL-0007 hybrid RAG on corpus v1: 437/892 (48.99%).

## Verify the package locally

```
python -m pytest kaggle/nl2mql_mql0008_approved_corpus_v2_1/tests -q
# notebook end-to-end tests (fake generator, no GPU) also need `transformers` and the pinned tokenizer files:
MQL0008_TOKENIZER_DIR=<dir> python -m pytest kaggle/nl2mql_mql0008_approved_corpus_v2_1/tests -q
```

Rebuild (needs the repo): `python scripts/nl2mql/make_mql0008_notebook.py`, then
`python scripts/nl2mql/build_mql0008_package.py --tokenizer-dir <dir>`. After editing only README or tests:
`python scripts/nl2mql/build_mql0008_package.py --finalize`.
