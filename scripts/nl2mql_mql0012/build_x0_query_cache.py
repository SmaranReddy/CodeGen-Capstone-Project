"""MQL-0012: dense query vectors for the X0 questions (inputs only; no retrieval, no outcome).

Same embedder as the MQL-0011 artifact (bge-small-en-v1.5 @ 5c38ec7c, CLS, L2, fp32, CPU, 1 thread, batch 1) and the
same text: patterns.mask_question(question, tables.json schema of the question's own database). Same on-disk format as
the MQL-0011 dev cache, so embedder.CachedEmbedder.from_dir reads it. Refuses to overwrite.
Run (venv_rag): python scripts/nl2mql_mql0012/build_x0_query_cache.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import mql0012_common as M   # noqa: E402

C = M.C11


def main():
    import embedder as EM
    import patterns as PT
    from build_embedding_artifact import SPEC
    d = M.QUERY_CACHE_X0
    if d.exists():
        raise SystemExit(f"{d} exists; caches are immutable")
    split = json.loads(M.SPLIT.read_text(encoding="utf-8"))
    items = C.read_jsonl(M.X0_ITEMS)
    if C.ids_sha256([r["id"] for r in items]) != split["ids_sha256"]:
        raise SystemExit("X0 items differ from the split manifest")
    infos = C.schema_infos(C.load_tables(), sorted({r["db_id"] for r in items}))
    by_text = {}
    for r in items:
        by_text.setdefault(PT.mask_question(r["question"], infos[r["db_id"]]), []).append(r["id"])
    texts = sorted(by_text)
    emb = EM.HFEmbedder(SPEC, threads=1)
    t0 = time.perf_counter()
    vec = emb.encode(texts)
    secs = time.perf_counter() - t0
    d.mkdir(parents=True)
    np.save(d / "vectors.npy", vec)
    rows = [{"text": t, "text_sha256": C.sha256_text(t), "ids": by_text[t]} for t in texts]
    C.write_jsonl(d / "index.jsonl", rows)
    C.write_json(d / "manifest.json", {
        "embedder": SPEC.to_dict(), "x0_ids_sha256": split["ids_sha256"], "n_items": len(items), "n_texts": len(texts),
        "mask_source": "tables.json", "text_shas_sha256": C.sha256_text(json.dumps([r["text_sha256"] for r in rows])),
        "vectors_sha256": C.sha256_bytes(np.ascontiguousarray(vec).tobytes()), "seconds": round(secs, 1),
        "max_tokens_seen": emb.max_tokens_seen,
        "files_sha256": {"vectors.npy": C.sha256_file(d / "vectors.npy"), "index.jsonl": C.sha256_file(d / "index.jsonl")}})
    print(json.dumps({"n_items": len(items), "n_texts": len(texts), "seconds": round(secs, 1),
                      "max_tokens": emb.max_tokens_seen, "dtype": str(vec.dtype), "shape": list(vec.shape)}))


if __name__ == "__main__":
    main()
