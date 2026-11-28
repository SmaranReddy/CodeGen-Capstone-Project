"""Compare masked and raw BGE embeddings on the registered MQL-0011 dev split.

Run with the project's RAG environment:
    python -B scripts/nl2mql_mql0011/analysis/raw_text_retrieval_dev.py
"""
import json
import re
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import retrieval_screen as screen

C = screen.C



def slices(sql, patterns):
    """Use the MQL-0015 report flags on development SQL."""
    normalized = " ".join(sql.upper().split())
    result = ["overall"]
    if re.search(r"\bNOT IN\b|\bNOT EXISTS\b", normalized):
        result.append("anti_join")
    if patterns & {"union", "intersect"} or re.search(r"\bEXCEPT\b", normalized):
        result.append("set_operation")
    return result


def summarize(rows, pct):
    attainable = [row for row in rows if row["eligible_matching_examples"]]
    skeleton_rows = [row for row in rows if row["has_gold_skeleton"]]
    n = len(rows)
    summary = {
        "questions": n,
        "questions_with_eligible_signature_match": len(attainable),
        "eligible_matching_examples_total_across_questions": sum(
            row["eligible_matching_examples"] for row in rows
        ),
        "questions_with_gold_skeleton": len(skeleton_rows),
    }
    for arm in ("masked", "raw"):
        summary[arm] = {
            "hit@1": pct(sum(row[arm]["hit1"] for row in rows), n),
            "hit@3": pct(sum(row[arm]["hit3"] for row in rows), n),
            "hit@3_attainable": pct(
                sum(row[arm]["hit3"] for row in attainable), len(attainable)
            ),
            "skeleton_hit@3": pct(
                sum(row[arm]["skeleton_hit3"] for row in skeleton_rows), len(skeleton_rows)
            ),
            "mJ@3": round(sum(row[arm]["max_jaccard"] for row in rows) / n, 4) if n else None,
        }
    summary["hit@3_gains"] = sum(row["raw"]["hit3"] and not row["masked"]["hit3"] for row in rows)
    summary["hit@3_losses"] = sum(row["masked"]["hit3"] and not row["raw"]["hit3"] for row in rows)
    summary["hit@3_delta_pp"] = pct(summary["hit@3_gains"] - summary["hit@3_losses"], n)
    return summary


def run():
    artifact = C.ARTIFACT_DIR
    required = (
        C.CORPUS, C.REGISTRATION, C.DEV_ITEMS, C.TABLES, C.SPIDER_TRAIN,
        artifact / "manifest.json", artifact / "doc_index.jsonl", artifact / "doc_vectors.npy",
        artifact / "query_cache_dev/manifest.json", artifact / "query_cache_dev/index.jsonl",
        artifact / "query_cache_dev/vectors.npy",
    )
    missing = [path.relative_to(C.ROOT) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing registered development inputs:\n" + "\n".join(map(str, missing)))

    import numpy as np

    import embedder as EM
    import mql0008_core as M8
    import patterns as PT
    import retrieval as RT
    import retriever_dense as RD
    import structure as S
    import vector_store as VS
    from build_embedding_artifact import SPEC, load_artifact

    class RawQuestionRetriever(RD.DenseHybridRetriever):
        def rank_examples(self, question, info, qid=""):
            self.raw_question = question
            try:
                return super().rank_examples(question, info, qid)
            finally:
                del self.raw_question

        def dense_scores(self, masked_q, target, universe):
            return super().dense_scores(self.raw_question, target, universe)

    config = json.loads(C.CONFIG.read_text(encoding="utf-8"))
    r3 = next(arm for arm in config["arms"] if arm["name"] == "R3")
    k = config["retrieval"]["k"]
    if r3["diversify"] != "none" or k != 3:
        raise ValueError("Expected registered R3 top-three selection")

    dev = C.read_jsonl(C.DEV_ITEMS)
    if len(dev) != C.DEV_N or C.ids_sha256([r["id"] for r in dev]) != C.DEV_IDS_SHA256:
        raise ValueError("dev items differ from the registration")
    train = {r["id"]: r for r in C.read_jsonl(C.SPIDER_TRAIN)}
    items = [train[r["id"]] for r in dev]
    if any((d["question"], d["db_id"]) != (t["question"], t["db_id"]) for d, t in zip(dev, items)):
        raise ValueError("Spider train questions or databases differ from the registered dev items")

    rows, forbidden = C.load_view_rows()
    tables = C.load_tables()
    docs = M8.build_docs(rows, tables)
    infos = C.schema_infos(tables, sorted({r["db_id"] for r in dev}))
    spec, ids, dbs, hashes, masked_vectors = load_artifact()
    if list(zip(ids, dbs, hashes)) != [
        (d["doc_id"], d["db_id"], C.sha256_text(d["masked_question"])) for d in docs
    ] or spec.model_id != SPEC.model_id or spec.text_field != "masked_question":
        raise ValueError("masked embedding artifact differs from the approved documents or model")

    def make_retriever(cls, index_spec, vectors, encoder):
        return cls(
            docs, r3["lexical"], store=VS.NumpyExactStore(index_spec, ids, dbs, vectors),
            embedder=encoder, forbidden_db_ids=forbidden,
            rrf_k=config["retrieval"]["rrf_k"], scorer=VS.CanonicalScorer(ids, vectors, spec.dim),
        )

    masked = make_retriever(
        RD.DenseHybridRetriever, spec, masked_vectors,
        EM.CachedEmbedder.from_dir(SPEC, artifact / "query_cache_dev"),
    )
    doc_texts = [d["question"] for d in docs]
    query_texts = [r["question"] for r in items]
    encoder = EM.HFEmbedder(SPEC, threads=config["embedding"]["threads"])
    raw_vectors = encoder.encode(doc_texts)
    raw_queries = encoder.encode(query_texts)
    raw_spec = replace(
        spec, index_name=spec.index_name + "__raw_question", text_field="question", mask_source="",
        vectors_sha256=C.sha256_bytes(raw_vectors.tobytes()),
    )
    raw = make_retriever(
        RawQuestionRetriever, raw_spec, raw_vectors, EM.CachedEmbedder(SPEC, query_texts, raw_queries),
    )

    output_root = C.ROOT / "outputs/mql0011_raw_question_dev"
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=output_root))
    np.savez(
        output / "raw_embeddings.npz", doc_ids=ids, doc_vectors=raw_vectors,
        query_ids=[r["id"] for r in items], query_vectors=raw_queries,
    )
    C.write_json(output / "embedding_spec.json", SPEC.to_dict())

    corpus = {r["id"]: r for r in C.read_jsonl(C.CORPUS)}
    signatures = {d["doc_id"]: S.signature(d["patterns"]) for d in docs}
    skeletons = {d["doc_id"]: screen.skeleton(d["mql"]) for d in docs}
    signature_counts = Counter(signatures.values())
    db_signature_counts = Counter((d["db_id"], signatures[d["doc_id"]]) for d in docs)
    per_item = []
    for item in items:
        qid, question, db = item["id"], item["question"], item["db_id"]
        labels = PT.sql_patterns(item["sql"])
        gold = S.signature(labels)
        gold_skeleton = screen.skeleton(corpus[qid]["mql"]) if qid in corpus else None
        row = {
            "id": qid, "db_id": db, "slices": slices(item["sql"], labels),
            "eligible_matching_examples": signature_counts[gold] - db_signature_counts[db, gold],
            "has_gold_skeleton": gold_skeleton is not None,
        }
        for name, retriever in (("masked", masked), ("raw", raw)):
            top = retriever.rank_examples(question, infos[db], qid=qid)[:k]
            top_ids = [doc["doc_id"] for doc, *_ in top]
            top_signatures = [signatures[doc_id] for doc_id in top_ids]
            row[name] = {
                "top3": top_ids,
                "hit1": top_signatures[:1] == [gold],
                "hit3": gold in top_signatures,
                "skeleton_hit3": gold_skeleton in [skeletons[doc_id] for doc_id in top_ids]
                if gold_skeleton is not None else None,
                "max_jaccard": max((RT.jaccard(gold, sig) for sig in top_signatures), default=0.0),
            }
        per_item.append(row)

    results = {
        name: summarize([row for row in per_item if name in row["slices"]], screen.pct)
        for name in ("overall", "anti_join", "set_operation")
    }
    C.write_jsonl(output / "per_item.jsonl", per_item)
    C.write_json(output / "results.json", results)
    print(json.dumps(results, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    run()
