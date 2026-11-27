"""MQL-0011 embedding artifact (source of truth for every vector store).

build   : the registered 2,912-record retrieval view (holdout_view.load_view: corpus sha, view ids sha, no holdout DB)
          -> mql0008_core.build_docs -> `masked_question` (exactly the text BM25 indexes) -> HFEmbedder (bge-small-en-v1.5
          @ pinned commit, CLS, L2, fp32, CPU, 1 thread, batch 1) ->
          outputs/nl2mql_mql0011/embeddings/<INDEX_NAME>/{doc_vectors.npy, doc_index.jsonl, manifest.json}
          plus the dev query cache query_cache_dev/{vectors.npy, index.jsonl, manifest.json}: the 779 registered dev
          questions masked with the tables.json schema (questions only; no gold SQL/MQL is read).
verify  : re-encode and compare with the stored vectors (byte equality reported; allclose 1e-5 required).
Refuses to overwrite an existing artifact.

Run (venv_rag): python scripts/nl2mql_mql0011/build_embedding_artifact.py build
                python scripts/nl2mql_mql0011/build_embedding_artifact.py verify [--threads 2]
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import mql0011_common as C  # noqa: E402
import embedder as EM         # noqa: E402
from vector_store import IndexSpec, check_vectors   # noqa: E402

SPEC = EM.EmbedderSpec(model_id=C.MODEL_ID, model_revision=C.MODEL_REVISION)


def doc_texts():
    import mql0008_core as M8
    import patterns as PT
    rows, hdb = C.load_view_rows()
    docs = M8.build_docs(rows, C.load_tables())
    if [d["doc_id"] for d in docs] != [r["id"] for r in rows] or {d["db_id"] for d in docs} & hdb:
        raise ValueError("documents differ from the view")
    return docs, C.sha256_file(PT.__file__)


def dev_texts():
    import patterns as PT
    dev = C.read_jsonl(C.DEV_ITEMS)
    if len(dev) != C.DEV_N or C.ids_sha256([r["id"] for r in dev]) != C.DEV_IDS_SHA256:
        raise ValueError("dev items differ from the registration")
    infos = C.schema_infos(C.load_tables(), sorted({r["db_id"] for r in dev}))
    by_text = {}
    for r in dev:
        by_text.setdefault(PT.mask_question(r["question"], infos[r["db_id"]]), []).append(r["id"])
    return by_text


def env_versions():
    import torch
    import transformers
    import tokenizers
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}


def build(threads):
    d = C.ARTIFACT_DIR
    if d.exists():
        raise SystemExit(f"{d} exists; artifacts are immutable (delete deliberately to rebuild)")
    docs, patterns_sha = doc_texts()
    texts = [x["masked_question"] for x in docs]
    emb = EM.HFEmbedder(SPEC, threads=threads)
    t0 = time.perf_counter()
    vec = emb.encode(texts)
    t_docs = time.perf_counter() - t0
    by_text = dev_texts()
    q_texts = sorted(by_text)
    t0 = time.perf_counter()
    qvec = emb.encode(q_texts)
    t_q = time.perf_counter() - t0
    check_vectors(vec, SPEC.dim)
    check_vectors(qvec, SPEC.dim)
    spec = IndexSpec(index_name=C.INDEX_NAME, corpus_sha256=C.CORPUS_SHA256, view_ids_sha256=C.VIEW_IDS_SHA256,
                     n_records=len(docs), model_id=SPEC.model_id, model_revision=SPEC.model_revision,
                     pooling=SPEC.pooling, dim=SPEC.dim, normalized=True, dtype="float32",
                     text_field="masked_question", mask_source="tables.json", patterns_sha256=patterns_sha,
                     vectors_sha256=C.sha256_bytes(np.ascontiguousarray(vec).tobytes()))
    d.mkdir(parents=True)
    np.save(d / "doc_vectors.npy", vec)
    C.write_jsonl(d / "doc_index.jsonl", [{"row": i, "doc_id": x["doc_id"], "db_id": x["db_id"], "text": t,
                                           "text_sha256": C.sha256_text(t)} for i, (x, t) in enumerate(zip(docs, texts))])
    qd = d / "query_cache_dev"
    qd.mkdir()
    np.save(qd / "vectors.npy", qvec)
    qrows = [{"text": t, "text_sha256": C.sha256_text(t), "ids": by_text[t]} for t in q_texts]
    C.write_jsonl(qd / "index.jsonl", qrows)
    C.write_json(qd / "manifest.json", {
        "embedder": SPEC.to_dict(), "dev_ids_sha256": C.DEV_IDS_SHA256, "n_items": C.DEV_N, "n_texts": len(q_texts),
        "mask_source": "tables.json", "text_shas_sha256": C.sha256_text(json.dumps([r["text_sha256"] for r in qrows])),
        "vectors_sha256": C.sha256_bytes(np.ascontiguousarray(qvec).tobytes()),
        "files_sha256": {"vectors.npy": C.sha256_file(qd / "vectors.npy"), "index.jsonl": C.sha256_file(qd / "index.jsonl")}})
    C.write_json(d / "manifest.json", {
        "index_spec": spec.to_dict(), "embedder": SPEC.to_dict(), "created_utc": datetime.now(timezone.utc).isoformat(),
        "build": {"threads": threads, "batch_size": 1, "device": "cpu", "deterministic_algorithms": True,
                  "max_tokens_seen": emb.max_tokens_seen, "seconds_docs": round(t_docs, 2),
                  "seconds_dev_queries": round(t_q, 2)},
        "env": env_versions(), "registration_sha256": C.sha256_file(C.REGISTRATION),
        "files_sha256": {"doc_vectors.npy": C.sha256_file(d / "doc_vectors.npy"),
                         "doc_index.jsonl": C.sha256_file(d / "doc_index.jsonl")}})
    print(json.dumps({"n_docs": len(docs), "n_dev_texts": len(q_texts), "seconds_docs": round(t_docs, 1),
                      "seconds_dev": round(t_q, 1), "max_tokens": emb.max_tokens_seen, "dim": int(vec.shape[1]),
                      "dtype": str(vec.dtype), "vectors_sha256": spec.vectors_sha256}))


def load_artifact(d=None):
    """-> (IndexSpec, doc_ids, db_ids, text_shas, vectors) after verifying every recorded hash."""
    d = Path(d or C.ARTIFACT_DIR)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    for f, h in m["files_sha256"].items():
        if C.sha256_file(d / f) != h:
            raise ValueError(f"{f} hash differs from the manifest")
    spec = IndexSpec.from_dict(m["index_spec"])
    vec = check_vectors(np.load(d / "doc_vectors.npy"), spec.dim)
    rows = C.read_jsonl(d / "doc_index.jsonl")
    ids = [r["doc_id"] for r in rows]
    if (C.sha256_bytes(np.ascontiguousarray(vec).tobytes()) != spec.vectors_sha256 or len(ids) != spec.n_records
            or C.ids_sha256(ids) != spec.view_ids_sha256 or [r["row"] for r in rows] != list(range(len(rows)))
            or spec.model_revision != C.MODEL_REVISION or spec.corpus_sha256 != C.CORPUS_SHA256):
        raise ValueError("artifact is internally inconsistent or not the MQL-0011 frozen version")
    if any(C.sha256_text(r["text"]) != r["text_sha256"] for r in rows):
        raise ValueError("doc text hash mismatch")
    return spec, ids, [r["db_id"] for r in rows], [r["text_sha256"] for r in rows], vec


def verify(threads):
    spec, ids, dbs, shas, vec = load_artifact()
    rows = C.read_jsonl(C.ARTIFACT_DIR / "doc_index.jsonl")
    docs, patterns_sha = doc_texts()
    if [x["masked_question"] for x in docs] != [r["text"] for r in rows] or patterns_sha != spec.patterns_sha256:
        raise SystemExit("rebuilt doc texts differ from the artifact")
    emb = EM.HFEmbedder(SPEC, threads=threads)
    again = emb.encode([r["text"] for r in rows])
    qd = C.ARTIFACT_DIR / "query_cache_dev"
    qrows = C.read_jsonl(qd / "index.jsonl")
    qagain = emb.encode([r["text"] for r in qrows])
    qvec = np.load(qd / "vectors.npy")
    if sorted(dev_texts()) != [r["text"] for r in qrows]:
        raise SystemExit("rebuilt dev query texts differ from the cache")
    res = {"threads": threads, "docs_byte_identical": bool(np.array_equal(again, vec)),
           "docs_max_abs_diff": float(np.abs(again - vec).max()),
           "dev_byte_identical": bool(np.array_equal(qagain, qvec)), "dev_max_abs_diff": float(np.abs(qagain - qvec).max())}
    res["ok"] = res["docs_max_abs_diff"] <= 1e-5 and res["dev_max_abs_diff"] <= 1e-5
    C.write_json(C.OUT / f"preflight/artifact_verify_threads{threads}.json", res)
    print(json.dumps(res))
    if not res["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    build(a.threads) if a.cmd == "build" else verify(a.threads)
