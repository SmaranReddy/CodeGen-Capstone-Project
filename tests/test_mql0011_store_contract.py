"""MQL-0011 VectorStore contract: the same suite on the NumPy reference and on embedded Qdrant (synthetic data).

Run with ~/codegen_data/venv_rag (qdrant-client); under an interpreter without qdrant-client the Qdrant half skips.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0011/src"))
import mql0011_common as C  # noqa: E402
import vector_store as VS     # noqa: E402

N, DIM = 40, 8


def make_data(seed=7):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N, DIM)).astype(np.float32)
    v[5] = v[4]                                   # an exact duplicate vector -> tie broken by doc_id
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ids = [f"doc_{i:03d}" for i in range(N)]
    dbs = [f"db_{i % 5}" for i in range(N)]
    shas = [C.sha256_text(i) for i in ids]
    spec = VS.IndexSpec(index_name="mql0011_test_index", corpus_sha256="c" * 64, view_ids_sha256=C.ids_sha256(ids),
                        n_records=N, model_id="test", model_revision="r", pooling="cls", dim=DIM, normalized=True,
                        dtype="float32", text_field="masked_question", mask_source="tables.json",
                        patterns_sha256="p" * 64, vectors_sha256=C.sha256_bytes(v.tobytes()))
    return spec, ids, dbs, shas, v


def _numpy(tmp_path):
    spec, ids, dbs, shas, v = make_data()
    return VS.NumpyExactStore(spec, ids, dbs, v)


def _qdrant(tmp_path):
    pytest.importorskip("qdrant_client")
    import qdrant_store as QS
    spec, ids, dbs, shas, v = make_data()
    client = QS.QdrantStore.connect(path=tmp_path / "q")
    QS.QdrantStore.create(client, spec, ids, dbs, shas, v)
    client.close()
    return QS.QdrantStore.open(spec, path=tmp_path / "q")


@pytest.fixture(params=["numpy", "qdrant"])
def store(request, tmp_path):
    s = {"numpy": _numpy, "qdrant": _qdrant}[request.param](tmp_path)
    yield s
    s.close()


def rerank(hits):
    return [h.doc_id for h in sorted(hits, key=lambda h: (-round(h.score, 6), h.doc_id))]


def test_ids_count_spec(store):
    spec, ids, *_ = make_data()
    assert store.ids() == ids and store.count() == N and store.spec() == spec


def test_exact_cosine_and_full_pool(store):
    spec, ids, dbs, shas, v = make_data()
    q = v[11]
    hits = store.search(q, None, VS.EligibilityFilter())
    assert len(hits) == N                                     # n=None -> actual eligible count
    ref = {i: float(v[j] @ q) for j, i in enumerate(ids)}
    assert all(abs(h.score - ref[h.doc_id]) < 1e-5 for h in hits)
    assert rerank(hits)[0] == "doc_011"


def test_tie_order_is_doc_id(store):
    spec, ids, dbs, shas, v = make_data()
    order = rerank(store.search(v[4], None, VS.EligibilityFilter()))
    assert order[:2] == ["doc_004", "doc_005"]


def test_filters_exclude_db_and_doc(store):
    spec, ids, dbs, shas, v = make_data()
    flt = VS.EligibilityFilter(exclude_db_ids=frozenset({"db_1", "db_3"}), exclude_doc_ids=frozenset({"doc_000"}))
    hits = store.search(v[1], None, flt)
    assert store.count(flt) == len(hits) == N - 16 - 1
    assert not {h.db_id for h in hits} & {"db_1", "db_3"} and "doc_000" not in {h.doc_id for h in hits}


def test_top_n_and_bad_n(store):
    spec, ids, dbs, shas, v = make_data()
    assert len(store.search(v[2], 3, VS.EligibilityFilter())) == 3
    with pytest.raises(ValueError):
        store.search(v[2], 0, VS.EligibilityFilter())
    with pytest.raises(VS.RetrievalStoreError):
        store.search(np.zeros(DIM + 1, np.float32), None, VS.EligibilityFilter())


def test_point_uuid_deterministic():
    assert VS.point_uuid("spider_train_00000") == VS.point_uuid("spider_train_00000")
    assert VS.point_uuid("a") != VS.point_uuid("b")


def test_check_hits_rejects_ineligible():
    known = {"a": "db_x", "b": "db_y"}
    flt = VS.EligibilityFilter(exclude_db_ids=frozenset({"db_y"}))
    VS.check_hits([VS.Hit("a", "db_x", 0.5)], flt, known)
    for bad in ([VS.Hit("b", "db_y", 0.4)], [VS.Hit("z", "db_x", 0.4)], [VS.Hit("a", "db_q", 0.4)],
                [VS.Hit("a", "db_x", 0.4), VS.Hit("a", "db_x", 0.4)]):
        with pytest.raises(VS.EligibilityViolation):
            VS.check_hits(bad, flt, known)


def test_numpy_rejects_bad_vectors():
    spec, ids, dbs, shas, v = make_data()
    with pytest.raises(VS.RetrievalStoreError):
        VS.NumpyExactStore(spec, ids, dbs, v * 2)                    # not normalised
    with pytest.raises(VS.RetrievalStoreError):
        VS.NumpyExactStore(spec, ids, dbs, v.astype(np.float64))     # wrong dtype
    with pytest.raises(VS.RetrievalStoreError):
        VS.NumpyExactStore(spec, ids[:-1] + ids[:1], dbs, v)         # duplicate id


# ---------------------------------------------------------------- Qdrant-specific behaviour
def test_qdrant_open_verifies_and_locks(tmp_path):
    pytest.importorskip("qdrant_client")
    import qdrant_store as QS
    spec, ids, dbs, shas, v = make_data()
    client = QS.QdrantStore.connect(path=tmp_path / "q")
    QS.QdrantStore.create(client, spec, ids, dbs, shas, v)
    QS.QdrantStore.create(client, spec, ids, dbs, shas, v)            # idempotent re-ingestion
    assert client.count(spec.index_name, exact=True).count == N
    with pytest.raises(VS.RetrievalStoreError):                        # different spec -> refuse to overwrite
        QS.QdrantStore.create(client, VS.IndexSpec(**{**spec.to_dict(), "model_revision": "other"}),
                              ids, dbs, shas, v)
    with pytest.raises(VS.RetrievalStoreError):                        # second client on the same path
        QS.QdrantStore.connect(path=tmp_path / "q")
    client.close()
    with pytest.raises(VS.RetrievalStoreError):                        # spec mismatch on open
        QS.QdrantStore.open(VS.IndexSpec(**{**spec.to_dict(), "vectors_sha256": "0" * 64}), path=tmp_path / "q")
    with pytest.raises(VS.RetrievalStoreError):                        # missing collection
        QS.QdrantStore.open(VS.IndexSpec(**{**spec.to_dict(), "index_name": "absent"}), path=tmp_path / "q")
    with QS.QdrantStore.open(spec, path=tmp_path / "q") as s:          # lock released after close
        assert s.count() == N


def test_qdrant_create_refuses_wrong_ids(tmp_path):
    pytest.importorskip("qdrant_client")
    import qdrant_store as QS
    spec, ids, dbs, shas, v = make_data()
    client = QS.QdrantStore.connect(path=tmp_path / "q")
    try:
        with pytest.raises(VS.RetrievalStoreError):
            QS.QdrantStore.create(client, spec, list(reversed(ids)), dbs, shas, v)
    finally:
        client.close()
