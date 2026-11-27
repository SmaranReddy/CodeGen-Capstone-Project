"""MQL-0011 dense / fused retriever on the REAL registered view (2,912 docs) with deterministic FAKE vectors.

No model, no Qdrant, no torch: the embedder is a text-hash -> unit-vector function, the store is NumpyExactStore.
Checks the candidate universe, ranking arithmetic, determinism, and that no arm can ever return an ineligible document.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0011/src"))
import mql0011_common as C  # noqa: E402
import embedder as EM         # noqa: E402
import mql0008_core as M8     # noqa: E402
import patterns as PT         # noqa: E402
import planner_dense as PD    # noqa: E402
import retrieval as RT        # noqa: E402
import retriever_dense as RD  # noqa: E402
import retriever_v2 as R2     # noqa: E402
import vector_store as VS     # noqa: E402

DIM = 16
pytestmark = pytest.mark.skipif(not C.CORPUS.exists(), reason="approved corpus absent")


def fake_vec(text):
    rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:16], 16))
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    spec = EM.EmbedderSpec(model_id="fake", model_revision="fake-rev", dim=DIM)

    def encode(self, texts):
        return np.stack([fake_vec(t) for t in texts])


def spec_for(ids, vecs, name="fake_index"):
    return VS.IndexSpec(index_name=name, corpus_sha256=C.CORPUS_SHA256, view_ids_sha256=C.ids_sha256(ids),
                        n_records=len(ids), model_id="fake", model_revision="fake-rev", pooling="cls", dim=DIM,
                        normalized=True, dtype="float32", text_field="masked_question", mask_source="tables.json",
                        patterns_sha256="x", vectors_sha256=C.sha256_bytes(vecs.tobytes()))


def store_for(docs):
    ids = [d["doc_id"] for d in docs]
    vecs = np.stack([fake_vec(d["masked_question"]) for d in docs])
    return VS.NumpyExactStore(spec_for(ids, vecs), ids, [d["db_id"] for d in docs], vecs)


@pytest.fixture(scope="module")
def world():
    rows, hdb = C.load_view_rows()
    tabs = C.load_tables()
    docs = M8.build_docs(rows, tabs)
    dev = C.read_jsonl(C.DEV_ITEMS)
    infos = C.schema_infos(tabs, sorted({r["db_id"] for r in dev} | hdb))
    return {"docs": docs, "hdb": hdb, "dev": dev, "infos": infos, "store": store_for(docs), "tabs": tabs}


def retr(world, mode, docs=None, store=None):
    docs = docs or world["docs"]
    dense = mode in RD.DENSE_MODES
    return RD.DenseHybridRetriever(docs, mode, store=(store or world["store"]) if dense else None,
                                   embedder=FakeEmbedder() if dense else None, forbidden_db_ids=world["hdb"])


SAMPLE = slice(0, 779, 37)


def test_view_constants(world):
    assert len(world["docs"]) == C.VIEW_N == 2912
    assert C.ids_sha256([d["doc_id"] for d in world["docs"]]) == C.VIEW_IDS_SHA256
    assert not {d["db_id"] for d in world["docs"]} & world["hdb"] and len(world["hdb"]) == 18
    assert len(world["dev"]) == C.DEV_N and C.ids_sha256([r["id"] for r in world["dev"]]) == C.DEV_IDS_SHA256


def test_competition_ranks():
    assert RD.competition_ranks([0.5, 0.9, 0.5, 0.1, 0.9]) == [3, 1, 3, 5, 1]
    assert RD.competition_ranks([0.3000000001, 0.3]) == [1, 1]
    assert RD.competition_ranks([]) == []


def test_bm25_mode_is_baseline(world):
    base = R2.StructureAwareRetriever(world["docs"])
    r0 = retr(world, "bm25")
    for e in world["dev"][SAMPLE]:
        info = world["infos"][e["db_id"]]
        assert r0.rank_examples(e["question"], info, e["id"]) == base.rank_examples(e["question"], info, e["id"])


@pytest.mark.parametrize("mode", RD.LEXICAL_MODES)
def test_universe_and_determinism(world, mode):
    """Pre-dedup universe U = view docs outside the target DB (same object the BM25 index uses). The baseline
    collapse keeps the HIGHER-RANKED copy of each duplicate group, so which copy survives legitimately depends on the
    ranking; the invariant is: output ⊆ U, and every dropped doc shares a dedup key with a higher-ranked kept doc."""
    r = retr(world, mode)
    km = {d["doc_id"]: repr(d.get("mql")) for d in world["docs"]}
    kq = {d["doc_id"]: RT._norm_q(d.get("question", "")) for d in world["docs"]}
    for e in world["dev"][SAMPLE]:
        info = world["infos"][e["db_id"]]
        universe = r.base._index_for(e["db_id"])[0]
        assert {d["doc_id"] for d in universe} == {d["doc_id"] for d in world["docs"] if d["db_id"] != e["db_id"]}
        out = r.rank_examples(e["question"], info, e["id"])
        kept = {d["doc_id"] for d, *_ in out}
        assert kept <= {d["doc_id"] for d in universe}
        kept_m, kept_q = {km[i] for i in kept}, {kq[i] for i in kept}
        assert all(km[d["doc_id"]] in kept_m or kq[d["doc_id"]] in kept_q
                   for d in universe if d["doc_id"] not in kept)
        assert all(d["db_id"] != e["db_id"] for d, *_ in out)    # target DB never present
        assert out == r.rank_examples(e["question"], info, e["id"])


def test_rrf_arithmetic(world):
    e = world["dev"][3]
    info = world["infos"][e["db_id"]]
    r = retr(world, "rrf_fused")
    docs, index = r.base._index_for(e["db_id"])
    masked = r.base.query_text(e["question"], info)
    raw = index.scores(PT.tokenize(masked))
    q = fake_vec(masked)
    cos = [round(float(fake_vec(d["masked_question"]) @ q), 6) for d in docs]
    rb, rd = RD.competition_ranks(raw), RD.competition_ranks(cos)
    l_raw = [1 / (60 + a) + 1 / (60 + b) for a, b in zip(rb, rd)]
    cues = PT.question_cues(e["question"], info)
    exp = {d["doc_id"]: l / max(l_raw) + RT.jaccard(cues, d["patterns"]) for d, l in zip(docs, l_raw)}
    out = r.rank_examples(e["question"], info, e["id"])
    for d, s, tr in out[:20]:
        assert s == round(exp[d["doc_id"]], 6)
        assert tr["bm25_rank"] == rb[docs.index(d)] and tr["dense_rank"] == rd[docs.index(d)]


def test_rrf_bm25_needs_no_store(world):
    r = RD.DenseHybridRetriever(world["docs"], "rrf_bm25")
    e = world["dev"][0]
    assert r.rank_examples(e["question"], world["infos"][e["db_id"]], e["id"])


# ---------------------------------------------------------------- ineligible documents: no arm may return them
def planted(world):
    """A copy of a real doc relabelled to a holdout DB (forbidden)."""
    d = dict(world["docs"][0])
    d.update(doc_id="planted_holdout_doc", db_id=sorted(world["hdb"])[0])
    return d


@pytest.mark.parametrize("mode", RD.LEXICAL_MODES)
def test_forbidden_doc_in_docs_aborts_every_arm(world, mode):
    docs = world["docs"] + [planted(world)]
    with pytest.raises((ValueError, VS.RetrievalStoreError)):
        PD.build_plan({"name": "x", "lexical": mode}, world["dev"][:1], [], docs, world["infos"], {}, {}, {},
                      len, forbidden_db_ids=world["hdb"], store=world["store"], embedder=FakeEmbedder())
    with pytest.raises(VS.RetrievalStoreError):
        RD.DenseHybridRetriever(docs, mode, store=world["store"], embedder=FakeEmbedder(),
                                forbidden_db_ids=world["hdb"])


@pytest.mark.parametrize("mode", RD.DENSE_MODES)
def test_extra_point_in_store_is_refused(world, mode):
    docs = world["docs"] + [planted(world)]
    with pytest.raises(VS.RetrievalStoreError):
        RD.DenseHybridRetriever(world["docs"], mode, store=store_for(docs), embedder=FakeEmbedder(),
                                forbidden_db_ids=world["hdb"])


class LeakyStore:
    """Wraps a store and smuggles one ineligible hit (target-DB doc) into every result."""

    def __init__(self, inner, leak):
        self.inner, self.leak = inner, leak

    def spec(self):
        return self.inner.spec()

    def ids(self):
        return self.inner.ids()

    def count(self, flt=None):
        return self.inner.count(flt)

    def search(self, vec, n, flt):
        hits = self.inner.search(vec, n, flt)
        return VS.check_hits(hits[:-1] + [self.leak], flt, self.inner.known)


@pytest.mark.parametrize("mode", RD.DENSE_MODES)
def test_leaky_store_is_fatal(world, mode):
    e = world["dev"][0]
    tgt_doc = next(d for d in world["docs"] if d["db_id"] == e["db_id"])
    leaky = LeakyStore(world["store"], VS.Hit(tgt_doc["doc_id"], tgt_doc["db_id"], 0.99))
    r = RD.DenseHybridRetriever(world["docs"], mode, store=leaky, embedder=FakeEmbedder(),
                                forbidden_db_ids=world["hdb"])
    with pytest.raises(VS.EligibilityViolation):
        r.rank_examples(e["question"], world["infos"][e["db_id"]], e["id"])


@pytest.mark.parametrize("mode", RD.LEXICAL_MODES)
def test_target_db_doc_never_returned_and_pair_exclusion(world, mode):
    e = next(x for x in world["dev"] if any(d["db_id"] == x["db_id"] for d in world["docs"]))
    r = retr(world, mode)
    out = r.rank_examples(e["question"], world["infos"][e["db_id"]], e["id"])
    assert all(d["db_id"] != e["db_id"] for d, *_ in out)
    top = out[0][0]["doc_id"]
    prompt = {"id": e["id"], "system": "S", "messages": [{"content": f"Schema\n\nQuestion: {e['question']}"}]}
    ctx = {"max_context_tokens": 10 ** 6, "n_ctx": 10 ** 7, "max_new_tokens": 1, "margin_tokens": 0}
    rows = PD.build_plan({"name": "x", "lexical": mode}, [e], [prompt], world["docs"], world["infos"], {},
                         {e["db_id"]: world["tabs"][e["db_id"]]}, ctx, len,
                         exclusions=[{"eval_id": e["id"], "corpus_id": top}], forbidden_db_ids=world["hdb"],
                         store=world["store"], embedder=FakeEmbedder())
    assert top not in rows[0]["selected_ids"] and rows[0]["excluded_by_rule"][0]["corpus_id"] == top
    assert len(rows[0]["selected_ids"]) == 3


def test_dense_arm_requires_store_and_matching_spec(world):
    with pytest.raises(VS.RetrievalStoreError):
        RD.DenseHybridRetriever(world["docs"], "rrf_fused")
    class Other(FakeEmbedder):
        spec = EM.EmbedderSpec(model_id="fake", model_revision="other", dim=DIM)
    with pytest.raises(VS.RetrievalStoreError):
        RD.DenseHybridRetriever(world["docs"], "rrf_fused", store=world["store"], embedder=Other())


def test_cached_embedder_and_pooling_checks():
    spec = EM.EmbedderSpec(model_id="m", model_revision="r", dim=DIM)
    ce = EM.CachedEmbedder(spec, ["a b", "c"], np.stack([fake_vec("a b"), fake_vec("c")]))
    assert np.array_equal(ce.encode(["c"])[0], fake_vec("c"))
    with pytest.raises(VS.RetrievalStoreError):
        ce.encode(["unseen"])
    cls = {"word_embedding_dimension": 384, "pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False,
           "pooling_mode_max_tokens": False, "pooling_mode_mean_sqrt_len_tokens": False}
    mods = [{"type": "sentence_transformers.models.Transformer"}, {"type": "sentence_transformers.models.Pooling"},
            {"type": "sentence_transformers.models.Normalize"}]
    EM.check_pooling_config(cls, mods, 384)
    for bad_cfg, bad_mods in (({**cls, "pooling_mode_cls_token": False, "pooling_mode_mean_tokens": True}, mods),
                              ({**cls, "word_embedding_dimension": 768}, mods), (cls, mods[:2])):
        with pytest.raises(VS.RetrievalStoreError):
            EM.check_pooling_config(bad_cfg, bad_mods, 384)


class JitterStore(LeakyStore):
    """Same candidates, scores perturbed by +-3e-7 (the measured Qdrant-vs-NumPy float32 difference)."""

    def search(self, vec, n, flt):
        return [VS.Hit(h.doc_id, h.db_id, h.score + (3e-7 if i % 2 else -3e-7))
                for i, h in enumerate(self.inner.search(vec, n, flt))]


@pytest.mark.parametrize("mode", RD.DENSE_MODES)
def test_canonical_scorer_makes_rankings_backend_independent(world, mode):
    docs = world["docs"]
    vecs = np.stack([fake_vec(d["masked_question"]) for d in docs])
    scorer = VS.CanonicalScorer([d["doc_id"] for d in docs], vecs, DIM)
    a = RD.DenseHybridRetriever(docs, mode, store=world["store"], embedder=FakeEmbedder(),
                                forbidden_db_ids=world["hdb"], scorer=scorer)
    b = RD.DenseHybridRetriever(docs, mode, store=JitterStore(world["store"], None), embedder=FakeEmbedder(),
                                forbidden_db_ids=world["hdb"], scorer=scorer)
    for e in world["dev"][SAMPLE]:
        info = world["infos"][e["db_id"]]
        assert a.rank_examples(e["question"], info, e["id"]) == b.rank_examples(e["question"], info, e["id"])
    with pytest.raises(VS.RetrievalStoreError):
        RD.DenseHybridRetriever(docs, mode, store=world["store"], embedder=FakeEmbedder(),
                                forbidden_db_ids=world["hdb"],
                                scorer=VS.CanonicalScorer([d["doc_id"] for d in docs[:-1]], vecs[:-1], DIM))


def test_split_arm_rejects_unsupported():
    with pytest.raises(ValueError):
        PD.split_arm({"lexical": "bogus"})
    with pytest.raises(ValueError):
        PD.split_arm({"structure_scoring": "nb"})
