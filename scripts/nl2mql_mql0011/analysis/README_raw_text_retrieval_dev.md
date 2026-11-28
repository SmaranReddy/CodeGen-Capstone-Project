# Raw-question embeddings for R3 retrieval

Schema masking may obscure which entity is excluded or which side of a set
difference a question asks for. This experiment tests whether embedding the
original wording improves retrieval of examples with matching SQL structure
for anti-joins and set operations.

## Paired comparison

Both arms use the registered 779-question development split and approved
2,912-example retrieval view. The control reads the existing masked BGE
vectors. The experimental arm embeds the original corpus and development
questions with the same pinned BGE-small-en-v1.5 revision. Both arms use the
same masked BM25, RRF, structural Jaccard scoring, database exclusions,
deduplication, and top-three selection through the existing R3 code. Both use
the exact NumPy store with canonical cosine scores.

The output includes per-question top-three IDs, exact signature match counts
in the eligible pool before ranking deduplication, and paired hit@3 gains and
losses. The overall, anti-join, and set-operation summaries also include hit@1,
hit@3, hit@3 among questions with an eligible signature match, skeleton hit@3
where a reference MQL skeleton exists, and mean best Jaccard@3. The slices can
overlap. Development gold labels are used only for slicing and scoring.

## Run

Use the project's RAG environment and BGE-small-en-v1.5 at revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The encoder uses NumPy, PyTorch,
Transformers, Hugging Face Hub, and safetensors. The runner uses an in-memory
store and needs no Qdrant service, Qwen weights, or Atlas connection.

Restore these original inputs at their repository-relative paths:

- `outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl`
- `outputs/nl2mql_mql0010/preflight/{split_registration.json,dev_items.jsonl}`
- `data/raw/spider1/spider_data/tables.json`
- `data/processed/spider_sql_v1/train.jsonl`
- Under `outputs/nl2mql_mql0011/embeddings/mql0011__v2_1_view2912__bge_small_en_v1_5__5c38ec7c__masked_q/`:
  `manifest.json`, `doc_index.jsonl`, `doc_vectors.npy`, and
  `query_cache_dev/{manifest.json,index.jsonl,vectors.npy}`.

From the repository root:

```sh
python -m pytest -q tests/test_raw_text_retrieval_dev.py
python -B scripts/nl2mql_mql0011/analysis/raw_text_retrieval_dev.py
```

Data preflight reports missing registered files before encoding. A run writes
raw vectors and reports to a fresh `outputs/mql0011_raw_question_dev/run-*/`
directory. Registered artifacts are read-only inputs.

## Verification status

Unit tests cover slice labels and summary metrics. The paired BGE evaluation
is blocked by missing registered inputs. Retrieval improvement, numerical
parity with the original Qdrant-backed screen on the 779 questions, and
downstream Qwen execution accuracy remain unverified. The final 1,000-question
test set is outside this experiment.
