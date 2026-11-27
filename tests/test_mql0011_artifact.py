"""MQL-0011 embedding artifact and ingestion report (the real artifact; numpy only, no model, no Qdrant client).

Skips when the artifact has not been built. The live collection is checked by ingest_vector_store.py (it holds the
embedded-Qdrant lock); this test checks the written ingest report against the artifact.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0011/src"))
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0011"))
import mql0011_common as C  # noqa: E402

pytestmark = pytest.mark.skipif(not (C.ARTIFACT_DIR / "manifest.json").exists(), reason="artifact not built")


@pytest.fixture(scope="module")
def art():
    from build_embedding_artifact import load_artifact
    return load_artifact()


def test_artifact_is_the_registered_view(art):
    spec, ids, dbs, shas, vec = art
    rows, hdb = C.load_view_rows()
    assert spec.n_records == len(ids) == 2912 == C.VIEW_N
    assert ids == [r["id"] for r in rows] and len(set(ids)) == len(ids)
    assert C.ids_sha256(ids) == spec.view_ids_sha256 == C.VIEW_IDS_SHA256
    assert len(hdb) == 18 and not set(dbs) & hdb
    assert dbs == [r["db_id"] for r in rows]


def test_vectors_and_spec(art):
    spec, ids, dbs, shas, vec = art
    assert vec.dtype == np.float32 and vec.shape == (2912, 384)
    assert np.allclose(np.linalg.norm(vec, axis=1), 1.0, atol=1e-5)
    assert (spec.model_id, spec.model_revision) == (C.MODEL_ID, C.MODEL_REVISION)
    assert (spec.pooling, spec.dim, spec.normalized, spec.dtype) == ("cls", 384, True, "float32")
    assert spec.corpus_sha256 == C.CORPUS_SHA256 and spec.index_name == C.INDEX_NAME


def test_doc_texts_are_the_bm25_texts(art):
    import mql0008_core as M8
    rows, _ = C.load_view_rows()
    docs = M8.build_docs(rows, C.load_tables())
    idx = C.read_jsonl(C.ARTIFACT_DIR / "doc_index.jsonl")
    assert [r["text"] for r in idx] == [d["masked_question"] for d in docs]


def test_dev_query_cache():
    qd = C.ARTIFACT_DIR / "query_cache_dev"
    m = json.loads((qd / "manifest.json").read_text(encoding="utf-8"))
    rows = C.read_jsonl(qd / "index.jsonl")
    vec = np.load(qd / "vectors.npy")
    assert m["dev_ids_sha256"] == C.DEV_IDS_SHA256 and m["embedder"]["model_revision"] == C.MODEL_REVISION
    assert sorted(i for r in rows for i in r["ids"]) == sorted(r["id"] for r in C.read_jsonl(C.DEV_ITEMS))
    assert vec.shape == (len(rows), 384) and vec.dtype == np.float32
    assert all(C.sha256_file(qd / f) == h for f, h in m["files_sha256"].items())


def test_ingest_report_matches_artifact(art):
    p = C.OUT / "preflight/ingest_report.json"
    if not p.exists():
        pytest.skip("not ingested yet")
    rep = json.loads(p.read_text(encoding="utf-8"))
    spec, ids, *_ = art
    assert rep["ok"] and rep["n_points"] == 2912 and rep["ids_equal_artifact"]
    assert rep["ids_sha256"] == C.VIEW_IDS_SHA256 and rep["holdout_dbs_present"] == []
    assert rep["index_spec"] == spec.to_dict()
