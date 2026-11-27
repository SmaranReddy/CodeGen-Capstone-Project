"""MQL-0012 focused tests (real 2,912-record view, deterministic FAKE vectors; no torch, no Qdrant).

Covers: FlexRetriever(bm25+dense, regex) == unchanged MQL-0011 R3; X2 never builds or calls BM25 and keeps the regex
Jaccard; X1 voting (neighbour selection, strict majority, ties, schema join rule, representation, Jaccard use, no gold
leakage); eligibility (forbidden / target / leaky store); determinism; the X0 split's disjointness.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0012/src"))
import mql0012_common as M        # noqa: E402
import mql0012_vote as V          # noqa: E402
import mql0012_retriever as FR    # noqa: E402
import embedder as EM             # noqa: E402
import mql0008_core as M8         # noqa: E402
import patterns as PT             # noqa: E402
import retrieval as RT            # noqa: E402
import retriever_dense as RD      # noqa: E402
import vector_store as VS         # noqa: E402

C = M.C11
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


@pytest.fixture(scope="module")
def world():
    rows, hdb = C.load_view_rows()
    tabs = C.load_tables()
    docs = M8.build_docs(rows, tabs)
    ids = [d["doc_id"] for d in docs]
    vecs = np.stack([fake_vec(d["masked_question"]) for d in docs])
    spec = VS.IndexSpec(index_name="fake", corpus_sha256=C.CORPUS_SHA256, view_ids_sha256=C.ids_sha256(ids),
                        n_records=len(ids), model_id="fake", model_revision="fake-rev", pooling="cls", dim=DIM,
                        normalized=True, dtype="float32", text_field="masked_question", mask_source="tables.json",
                        patterns_sha256="x", vectors_sha256=C.sha256_bytes(vecs.tobytes()))
    store = VS.NumpyExactStore(spec, ids, [d["db_id"] for d in docs], vecs)
    dev = C.read_jsonl(C.DEV_ITEMS)
    infos = C.schema_infos(tabs, sorted({r["db_id"] for r in dev}))
    return {"docs": docs, "hdb": hdb, "dev": dev, "infos": infos, "store": store,
            "scorer": VS.CanonicalScorer(ids, vecs, DIM), "vecs": vecs}


def flex(w, channels, cues, docs=None, store=None):
    return FR.FlexRetriever(docs or w["docs"], channels, cues, store or w["store"], FakeEmbedder(), w["scorer"],
                            forbidden_db_ids=w["hdb"])


SAMPLE = slice(0, 779, 41)


def test_flex_regex_fused_equals_mql0011_r3(world):
    r3 = RD.DenseHybridRetriever(world["docs"], "rrf_fused", store=world["store"], embedder=FakeEmbedder(),
                                 forbidden_db_ids=world["hdb"], scorer=world["scorer"])
    f = flex(world, ("bm25", "dense"), "regex")
    for e in world["dev"][SAMPLE]:
        info = world["infos"][e["db_id"]]
        assert f.rank_examples(e["question"], info, e["id"]) == r3.rank_examples(e["question"], info, e["id"])


def test_x2_never_builds_or_calls_bm25(world, monkeypatch):
    f = flex(world, ("dense",), "regex")

    def boom(*a, **k):
        raise AssertionError("BM25 used by X2")
    monkeypatch.setattr(RT.BM25, "__init__", boom)
    monkeypatch.setattr(RT.BM25, "scores", boom)
    monkeypatch.setattr(RT.Retriever, "_index_for", boom)
    for e in world["dev"][SAMPLE]:
        out = f.rank_examples(e["question"], world["infos"][e["db_id"]], e["id"])
        assert out and all(t["bm25_rank"] is None for _, _, t in out)


def test_x2_scoring_equation_and_regex_jaccard(world):
    f = flex(world, ("dense",), "regex")
    e = world["dev"][5]
    info = world["infos"][e["db_id"]]
    out = f.rank_examples(e["question"], info, e["id"])
    U = [d for d in world["docs"] if d["db_id"] != e["db_id"]]
    q = fake_vec(PT.mask_question(e["question"], info))
    cos = [round(float(fake_vec(d["masked_question"]) @ q), 6) for d in U]
    rd = RD.competition_ranks(cos)
    top = max(1 / (60 + r) for r in rd)
    cues = PT.question_cues(e["question"], info)
    exp = {d["doc_id"]: (1 / (60 + r)) / top + RT.jaccard(cues, d["patterns"]) for d, r in zip(U, rd)}
    for d, s, t in out[:25]:
        assert s == round(exp[d["doc_id"]], 6) and t["structure"] == round(RT.jaccard(cues, d["patterns"]), 6)


def test_universe_identical_across_arms(world):
    f2, f3 = flex(world, ("dense",), "regex"), flex(world, ("bm25", "dense"), "regex")
    for e in world["dev"][SAMPLE]:
        u2 = [d["doc_id"] for d in f2.universe(e["db_id"])]
        u_bm25 = [d["doc_id"] for d in f3.base._index_for(e["db_id"])[0]]
        assert u2 == u_bm25 and e["db_id"] not in {d["db_id"] for d in f2.universe(e["db_id"])}


# ---------------------------------------------------------------- X1 voting
def nb(labels_list):
    return [{"doc_id": f"d{i:02d}", "db_id": f"db{i}", "question": f"q{i}", "mql": {"i": i}, "patterns": set(l)}
            for i, l in enumerate(labels_list)]


def test_vote_strict_majority_and_tie():
    docs = nb([["count", "where"]] * 6 + [["join"]] * 4)
    pred, votes, n = V.vote(docs)
    assert n == 10 and pred == {"count"} and votes["count"] == 6 and votes["join"] == 4
    pred5, _, _ = V.vote(nb([["count"]] * 5 + [["join"]] * 5))
    assert pred5 == set()                                   # 5/5 tie -> excluded
    pred_ign, votes_ign, _ = V.vote(nb([["select", "where", "text_number"]] * 10))
    assert pred_ign == set() and "where" not in votes_ign and "text_number" not in votes_ign


def test_neighbour_selection_order_dedup_and_target_guard():
    docs = nb([["count"]] * 12)
    docs[3]["question"] = docs[1]["question"]               # duplicate question -> collapsed
    cos = [0.1 * (12 - i) for i in range(12)]
    cos[7] = cos[6]                                         # tie -> doc_id order
    sel = V.select_neighbours(docs, cos, "other_db")
    ids = [d["doc_id"] for d in sel]
    assert len(sel) == 10 and "d03" not in ids and ids.index("d06") < ids.index("d07") and ids[0] == "d00"
    with pytest.raises(ValueError):
        V.select_neighbours(docs, cos, "db0")               # target-DB doc in the pool is refused


def test_schema_join_rule_is_unioned(world):
    e = next(x for x in world["dev"] if len(PT.mentioned_collections(x["question"], world["infos"][x["db_id"]])) >= 2)
    info = world["infos"][e["db_id"]]
    U = [d for d in world["docs"] if d["db_id"] != e["db_id"]]
    no_join = [{**d, "patterns": [p for p in d["patterns"] if p != "join"]} for d in U]
    cues, tr = V.predicted_cues(e["question"], info, no_join, [0.0] * len(no_join))
    assert "join" in cues and "join" not in tr["predicted"] and "join" in tr["schema_rule"] and "select" in cues


def test_x1_neighbours_from_universe_only_and_no_gold(world):
    f = flex(world, ("bm25", "dense"), "knn_vote")
    for e in world["dev"][SAMPLE]:
        info = world["infos"][e["db_id"]]
        f.rank_examples(e["question"], info, e["id"])
        tr = f.last_vote
        by = {d["doc_id"]: d for d in world["docs"]}
        assert len(tr["neighbours"]) == 10 and e["id"] not in tr["neighbours"]
        assert all(by[i]["db_id"] != e["db_id"] for i in tr["neighbours"])
        # the predicted cues are a function of (question, schema, neighbours' corpus labels) only
        U = [d for d in world["docs"] if d["db_id"] != e["db_id"]]
        q = fake_vec(PT.mask_question(e["question"], info))
        cos = [float(fake_vec(d["masked_question"]) @ q) for d in U]
        cues, tr2 = V.predicted_cues(e["question"], info, U, cos)
        assert sorted(cues) == f.last_cues and tr2 == tr


def test_x1_uses_predicted_cues_in_jaccard(world):
    f = flex(world, ("bm25", "dense"), "knn_vote")
    e = world["dev"][9]
    out = f.rank_examples(e["question"], world["infos"][e["db_id"]], e["id"])
    cues = set(f.last_cues)
    assert all(t["structure"] == round(RT.jaccard(cues, d["patterns"]), 6) for d, _, t in out[:30])


@pytest.mark.parametrize("arm", [(("bm25", "dense"), "knn_vote"), (("dense",), "regex"), (("bm25", "dense"), "regex")])
def test_eligibility_and_determinism(world, arm):
    planted = {**world["docs"][0], "doc_id": "planted", "db_id": sorted(world["hdb"])[0]}
    with pytest.raises(VS.RetrievalStoreError):
        flex(world, *arm, docs=world["docs"] + [planted])
    f = flex(world, *arm)
    for e in world["dev"][:5]:
        info = world["infos"][e["db_id"]]
        a = f.rank_examples(e["question"], info, e["id"])
        assert a == f.rank_examples(e["question"], info, e["id"])
        assert all(d["db_id"] != e["db_id"] and d["db_id"] not in world["hdb"] for d, *_ in a) and len(a) >= 3


class Leaky:
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


@pytest.mark.parametrize("arm", [(("bm25", "dense"), "knn_vote"), (("dense",), "regex")])
def test_leaky_store_is_fatal(world, arm):
    e = world["dev"][0]
    t = next(d for d in world["docs"] if d["db_id"] == e["db_id"])
    f = flex(world, *arm, store=Leaky(world["store"], VS.Hit(t["doc_id"], t["db_id"], 0.99)))
    with pytest.raises(VS.EligibilityViolation):
        f.rank_examples(e["question"], world["infos"][e["db_id"]], e["id"])


def test_unsupported_configurations_refused(world):
    with pytest.raises(ValueError):
        FR.FlexRetriever(world["docs"], ("bm25",), "regex", world["store"], FakeEmbedder(), world["scorer"])
    with pytest.raises(ValueError):
        FR.FlexRetriever(world["docs"], ("dense",), "nb", world["store"], FakeEmbedder(), world["scorer"])
    with pytest.raises(VS.RetrievalStoreError):
        FR.FlexRetriever(world["docs"], ("dense",), "regex", world["store"], FakeEmbedder(), None)


# ---------------------------------------------------------------- X0 split
def test_x0_split_disjoint_and_hashed():
    if not M.SPLIT.exists():
        pytest.skip("split not built")
    s = json.loads(M.SPLIT.read_text(encoding="utf-8"))
    items = C.read_jsonl(M.X0_ITEMS)
    reg = C.load_registration()
    assert C.ids_sha256([r["id"] for r in items]) == s["ids_sha256"] and len(items) == s["n_items"] == 3295
    dbs = {r["db_id"] for r in items}
    assert dbs <= set(reg["build_dbs"]) and not dbs & set(reg["holdout"]["dbs"]) and not dbs & set(reg["dev"]["dbs"])
    assert not dbs & {r["db_id"] for r in C.read_jsonl(M.MQL0009_EVAL)}
    assert not dbs & {r["db_id"] for r in C.read_jsonl(M.SPIDER_DEV_FULL)}
    assert not dbs & {r["db_id"] for p in M.EXPANDED_SPLITS for r in C.read_jsonl(p)}
    dev_ids = {r["id"] for r in C.read_jsonl(C.DEV_ITEMS)}
    assert not {r["id"] for r in items} & dev_ids
    assert s["retrieval_view"]["ids_sha256"] == C.VIEW_IDS_SHA256


# ---------------------------------------------------------------- pre-registered analysis on synthetic records
def test_analyze_gate_logic(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0012"))
    import run_mql0012 as RUN
    monkeypatch.setattr(RUN.M, "RESULTS", tmp_path)
    cfg = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    per = []
    for k in range(400):
        base = k % 10 < 6                       # R3 hits 60%
        x1 = base or k % 10 == 6                # X1 hits 70% (+10 pp, never loses)
        x2 = base                               # X2 identical to R3
        g = ["join", "where"] if k % 2 else ["count"]
        rec = {"id": f"i{k}", "attainable": True, "has_skeleton": k % 4 != 0, "gold_signature": g,
               "subgroups": ["join"] if k % 2 else ["ungrouped_aggregate"], "paraphrase_twin": k % 5 == 0,
               "regex_cues": ["count"], "X1": {}}
        for a, h in (("R0", base), ("R3", base), ("X1", x1), ("X2", x2)):
            rec[a] = {**rec.get(a, {}), "hit3": h, "hit1": h, "skel3": h if rec["has_skeleton"] else None, "mJ": 1.0 if h else 0.5}
        rec["X1"]["vote_cues"] = ["count"]
        per.append(rec)
    RUN.analyze(per, cfg, {"R3": 1.0})
    res = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    c = res["confirmatory"]
    assert c["X1_vs_R3_structure"]["delta_pp"] == 10.0 and c["X1_vs_R3_structure"]["b_only"] == 0
    assert c["X1"]["PASS"] and c["X1_vs_R3_structure"]["ci95"][0] > 0
    assert c["X2_vs_R3_structure"]["delta_pp"] == 0.0 and c["X2_vs_R3_structure"]["ci95"] == [0.0, 0.0]
    assert c["X2"]["PASS"]                       # zero discordance -> CI [0, 0] > -1
    assert c["holm_reject"] == {"X1": True, "X2": True}
    assert res["arms"]["X1"]["structure_match"] == 70.0 and res["n_attainable"] == 400
    # a join decline > 2 pp must fail X1 even with a large overall gain
    for r in per:
        if "join" in r["subgroups"] and r["R3"]["hit3"] and int(r["id"][1:]) % 20 == 1:
            r["X1"]["hit3"] = False
    RUN.analyze(per, cfg, {})
    c2 = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["confirmatory"]
    assert not c2["X1"]["c3_join_decline_le_2"] and not c2["X1"]["PASS"]
