"""MQL-0011 vector-store ingestion and backend parity.

ingest : verified artifact -> embedded Qdrant collection INDEX_NAME at QDRANT_PATH (default ~/codegen_data/qdrant_mql0011,
         outside OneDrive; rebuildable from the artifact at any time). Idempotent (UUIDv5 ids); refuses a collection
         holding a different spec; verifies count, ids hash (artifact row order) and spec after the upsert.
parity : PRECONDITION of the retrieval screen, all 779 dev questions (cached query vectors):
         (a) store level: exact Qdrant search over the full eligible pool (target DB excluded) returns exactly the NumPy
             reference id set, with |score diff| <= 1e-6. Raw-score order differences are REPORTED only: backend cosines
             differ in the last float32 bits, so rounding/tie decisions can flip (the reason for CanonicalScorer);
         (b) retriever level (the binding check): DenseHybridRetriever rankings for dense_only and rrf_fused are
             identical with the Qdrant store and with the NumPy store (both re-scored by the CanonicalScorer).
Writes outputs/nl2mql_mql0011/preflight/{ingest_report.json, backend_parity.json}.
Run (venv_rag): python scripts/nl2mql_mql0011/ingest_vector_store.py ingest | parity
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import mql0011_common as C  # noqa: E402
import embedder as EM                                 # noqa: E402
import qdrant_store as QS                             # noqa: E402
import vector_store as VS                             # noqa: E402
from build_embedding_artifact import SPEC, load_artifact   # noqa: E402


def ingest():
    spec, ids, dbs, shas, vec = load_artifact()
    rows, hdb = C.load_view_rows()
    if ids != [r["id"] for r in rows] or set(dbs) & hdb:
        raise SystemExit("artifact ids differ from the registered view")
    C.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    client = QS.QdrantStore.connect(path=C.QDRANT_PATH)
    try:
        store = QS.QdrantStore.create(client, spec, ids, dbs, shas, vec)
        stored_ids = store.ids()
        rep = {"collection": spec.index_name, "path": str(C.QDRANT_PATH), "n_points": store.count(),
               "ids_sha256": C.ids_sha256(stored_ids), "ids_equal_artifact": stored_ids == ids,
               "holdout_dbs_present": sorted(set(store.known.values()) & hdb), "n_holdout_dbs_checked": len(hdb),
               "seconds": round(time.perf_counter() - t0, 2),
               "qdrant_client": __import__("importlib.metadata").metadata.version("qdrant-client"),
               "mode": "embedded-local (exact brute force)", "index_spec": spec.to_dict()}
    finally:
        client.close()
    rep["ok"] = (rep["n_points"] == C.VIEW_N and rep["ids_sha256"] == C.VIEW_IDS_SHA256 and rep["ids_equal_artifact"]
                 and not rep["holdout_dbs_present"])
    C.write_json(C.OUT / "preflight/ingest_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "index_spec"}))
    if not rep["ok"]:
        sys.exit(1)


def rerank(hits):
    return [h.doc_id for h in sorted(hits, key=lambda h: (-round(h.score, 6), h.doc_id))]


def retriever_parity(spec, ids, vec, ref, qs, cache):
    import mql0008_core as M8
    import retriever_dense as RD
    rows, hdb = C.load_view_rows()
    tabs = C.load_tables()
    docs = M8.build_docs(rows, tabs)
    dev = C.read_jsonl(C.DEV_ITEMS)
    infos = C.schema_infos(tabs, sorted({r["db_id"] for r in dev}))
    scorer = VS.CanonicalScorer(ids, vec, spec.dim)
    out = {}
    for mode in RD.DENSE_MODES:
        a = RD.DenseHybridRetriever(docs, mode, store=qs, embedder=cache, forbidden_db_ids=hdb, scorer=scorer)
        b = RD.DenseHybridRetriever(docs, mode, store=ref, embedder=cache, forbidden_db_ids=hdb, scorer=scorer)
        out[mode] = sum(
            [(d["doc_id"], s) for d, s, _ in a.rank_examples(e["question"], infos[e["db_id"]], e["id"])]
            != [(d["doc_id"], s) for d, s, _ in b.rank_examples(e["question"], infos[e["db_id"]], e["id"])]
            for e in dev)
    return out


def parity():
    spec, ids, dbs, shas, vec = load_artifact()
    ref = VS.NumpyExactStore(spec, ids, dbs, vec)
    cache = EM.CachedEmbedder.from_dir(SPEC, C.ARTIFACT_DIR / "query_cache_dev")
    qrows = C.read_jsonl(C.ARTIFACT_DIR / "query_cache_dev/index.jsonl")
    dev = {r["id"]: r for r in C.read_jsonl(C.DEV_ITEMS)}
    n = order_diff = set_diff = 0
    max_diff, t_q, t_n = 0.0, 0.0, 0.0
    examples = []
    with QS.QdrantStore.open(spec, path=C.QDRANT_PATH) as qs:
        for r in qrows:
            for item_id in r["ids"]:
                flt = VS.EligibilityFilter(exclude_db_ids=frozenset({dev[item_id]["db_id"]}))
                q = cache.encode([r["text"]])[0]
                t0 = time.perf_counter()
                a = qs.search(q, None, flt)
                t_q += time.perf_counter() - t0
                t0 = time.perf_counter()
                b = ref.search(q, None, flt)
                t_n += time.perf_counter() - t0
                n += 1
                sa, sb = {h.doc_id: h.score for h in a}, {h.doc_id: h.score for h in b}
                if set(sa) != set(sb) or len(a) != ref.count(flt):
                    set_diff += 1
                    continue
                max_diff = max(max_diff, max(abs(sa[k] - sb[k]) for k in sa))
                if rerank(a) != rerank(b):
                    order_diff += 1
                    examples.append(item_id)
        retr_diff = retriever_parity(spec, ids, vec, ref, qs, cache)
    rep = {"n_queries": n, "store_set_mismatches": set_diff, "store_max_abs_score_diff": max_diff,
           "store_raw_order_mismatches_after_round6_rerank_INFORMATIONAL": order_diff,
           "store_raw_order_mismatch_examples": examples[:10],
           "retriever_ranking_mismatches": retr_diff,
           "mean_ms_search_qdrant_local": round(1000 * t_q / max(n, 1), 3),
           "mean_ms_search_numpy": round(1000 * t_n / max(n, 1), 3),
           "note": "embedded-local latency; not evidence of production-scale performance"}
    rep["ok"] = (n == C.DEV_N and set_diff == 0 and max_diff <= 1e-6
                 and all(v == 0 for v in retr_diff.values()))
    C.write_json(C.OUT / "preflight/backend_parity.json", rep)
    print(json.dumps(rep))
    if not rep["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "parity"])
    {"ingest": ingest, "parity": parity}[ap.parse_args().cmd]()
