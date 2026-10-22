"""Week 3 RAG: expanded split, corpus provenance, leakage, retrieval, context budget, prompt
construction, config reproducibility, scoring/failure handling and deterministic rescoring.
No model calls and no network (MongoDB is mocked)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/rag"))
import common as C  # noqa: E402
import leakage  # noqa: E402
import make_splits_expanded as SE  # noqa: E402
import patterns as PT  # noqa: E402
import prompt_spider1 as P  # noqa: E402
import rag_prompt as RP  # noqa: E402
import rescore_rag  # noqa: E402
import retrieval as RT  # noqa: E402
import run_eval as R  # noqa: E402
import run_rag_eval as RR  # noqa: E402

K = C.K
HAVE_CORPUS = (C.CORPUS_DIR / "corpus.jsonl").exists()
needs_corpus = pytest.mark.skipif(not HAVE_CORPUS, reason="corpus not built")
EXEC = {"timeout_ms": 30000, "max_result_rows": 100000}


def words(text):  # deterministic stand-in tokenizer for budget tests
    return len(text.split())


@pytest.fixture(scope="module")
def data():
    return K.read_jsonl(C.DATASET)


@pytest.fixture(scope="module")
def split():
    return C.load_split()


@pytest.fixture(scope="module")
def corpus():
    return C.load_corpus()


@pytest.fixture(scope="module")
def collections():
    return C.load_collections()


# ------------------------------------------------------------------ split
def test_expanded_split_reproduces_and_keeps_v1_partitions():
    assert SE.verify() == []


def test_expanded_split_is_database_disjoint_and_complete(data, split):
    db_part, id_part = split
    assert set(id_part) == {r["id"] for r in data} and len(id_part) == 455
    for r in data:
        assert id_part[r["id"]] == db_part[r["db_id"]]
    owners = {}
    for p in SE.PARTITIONS:
        for ln in (C.SPLIT_DIR / f"{p}.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                assert owners.setdefault(r["db_id"], p) == p
    m = json.loads((C.SPLIT_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    assert m["partition_counts"] == {"train": 318, "validation": 72, "test": 65}
    assert m["source"]["sha256"] == C.sha256_file(C.DATASET)


def test_v1_split_files_untouched():
    m = json.loads((ROOT / "data/splits/split_manifest.json").read_text(encoding="utf-8"))
    for p, info in m["partitions"].items():
        assert C.sha256_file(ROOT / info["file"]) == info["sha256"]


# ------------------------------------------------------------------ corpus provenance
@needs_corpus
def test_corpus_manifest_hashes_match_inputs(corpus):
    m = json.loads((C.CORPUS_DIR / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert m["corpus_sha256"] == C.sha256_file(C.CORPUS_DIR / "corpus.jsonl")
    for inp in m["inputs"].values():
        assert C.sha256_file(ROOT / inp["path"]) == inp["sha256"], inp["path"]
    assert m["counts"] == {k: sum(d["kind"] == k for d in corpus) for k in ("example", "guide", "schema")}


@needs_corpus
def test_examples_are_train_only_verified_saved_predictions(corpus, data, split):
    db_part, id_part = split
    rows = {r["id"]: r for r in data}
    files = {}
    for d in (d for d in corpus if d["kind"] == "example"):
        prov = d["provenance"]
        src = rows[prov["dataset_id"]]
        assert id_part[src["id"]] == "train" and db_part[d["db_id"]] == "train"
        assert d["question"] == src["question"] and "ambiguous" not in src["flags"]
        assert prov["expected_rows_sha256"] == src["expected_rows_sha256"]
        if prov["source_file"] not in files:
            lines = (ROOT / prov["source_file"]).read_bytes().splitlines()
            files[prov["source_file"]] = {hashlib.sha256(ln).hexdigest(): json.loads(ln) for ln in lines if ln.strip()}
        rec = files[prov["source_file"]][prov["source_record_sha256"]]   # exact saved line
        assert rec["id"] == src["id"] and rec["result"]["correct"] is True
        parsed, _ = R.parse_output(rec["raw_text"])
        assert parsed["collection"] == d["mql"]["collection"]
        assert parsed["pipeline"] == d["mql"]["pipeline"] or parsed["pipeline"] == []
        assert d["mql"]["collection"].startswith(d["db_id"] + "__")


@needs_corpus
def test_guides_cite_official_docs_and_use_train_dbs_only(corpus, split):
    db_part, _ = split
    guides = [d for d in corpus if d["kind"] == "guide"]
    assert len(guides) == len(yaml.safe_load(C.GUIDES_SRC.read_text(encoding="utf-8"))["cards"])
    for g in guides:
        assert g["provenance"]["urls"] and all(u.startswith("https://www.mongodb.com/docs/") for u in g["provenance"]["urls"])
        assert set(g["patterns"]) <= set(PT.PATTERNS)
    for card in yaml.safe_load(C.GUIDES_SRC.read_text(encoding="utf-8"))["cards"]:
        coll = card["example"].get("collection")
        if coll:
            assert db_part[coll.split("__")[0]] == "train"


# ------------------------------------------------------------------ leakage
@needs_corpus
def test_corpus_passes_leakage_check(corpus, data, split, collections):
    assert leakage.check(corpus, data, split[0], collections) == []


@needs_corpus
@pytest.mark.parametrize("mutate", ["heldout_question", "heldout_collection", "heldout_db_example",
                                    "heldout_source", "duplicate_id", "heldout_sql"])
def test_leakage_check_catches_injected_leaks(corpus, data, split, collections, mutate):
    db_part = split[0]
    val = next(r for r in data if db_part[r["db_id"]] == "validation")
    bad = dict(next(d for d in corpus if d["kind"] == "example"))
    bad["provenance"] = dict(bad["provenance"])
    if mutate == "heldout_question":
        bad["content"] += "\nQuestion: " + val["question"]
    elif mutate == "heldout_collection":
        bad["content"] += f' {val["db_id"]}__visitor'
        bad["content"] += " " + next(c for c, i in collections.items() if i["db_id"] == val["db_id"])
    elif mutate == "heldout_db_example":
        bad["db_id"] = val["db_id"]
    elif mutate == "heldout_source":
        bad["provenance"]["dataset_id"] = val["id"]
    elif mutate == "heldout_sql":
        bad["content"] += " " + val["gold_sql"]
    if mutate != "duplicate_id":
        bad["doc_id"] = "injected"
    assert leakage.check(corpus + [bad], data, db_part, collections)


@needs_corpus
def test_no_validation_or_test_text_in_retrievable_docs(corpus, data, split):
    db_part = split[0]
    held = [r for r in data if db_part[r["db_id"]] != "train"]
    blob = " ".join(d["content"].lower() for d in corpus if d["kind"] in ("example", "guide"))
    for r in held:
        assert r["question"].lower() not in blob


# ------------------------------------------------------------------ retrieval
def toy_corpus():
    def ex(i, db, q, mql, pats):
        return {"doc_id": f"ex_{i}", "kind": "example", "db_id": db, "question": q,
                "masked_question": q.lower(), "mql": mql, "patterns": pats, "content": f"Question: {q}"}
    return [
        ex(1, "a", "How many singers are there?", {"c": 1}, ["select", "count"]),
        ex(2, "b", "How many cars are there?", {"c": 2}, ["select", "count"]),
        ex(3, "b", "How many cars are there?", {"c": 3}, ["select", "count"]),          # dup question
        ex(4, "c", "List names ordered by age", {"c": 4}, ["select", "order_by"]),
        ex(5, "c", "Show each country and the number of people", {"c": 2}, ["select", "group_by"]),  # dup MQL of 2
        ex(6, "t", "How many dogs are there?", {"c": 6}, ["select", "count"]),          # target db
        {"doc_id": "g1", "kind": "guide", "patterns": ["count"], "content": "count"},
        {"doc_id": "g2", "kind": "guide", "patterns": ["order_by", "limit"], "content": "sort"},
    ]


INFO_T = {"db_id": "t", "tables": {}, "schema_words": set(), "text_number_fields": []}


@pytest.mark.parametrize("method", RT.EXAMPLE_METHODS)
def test_retrievers_exclude_target_db_collapse_duplicates_and_are_deterministic(method):
    rt = RT.Retriever(toy_corpus(), method)
    a = rt.rank_examples("How many dogs are there?", INFO_T, "q1")
    b = rt.rank_examples("How many dogs are there?", INFO_T, "q1")
    ids = [d["doc_id"] for d, _ in a]
    assert ids == [d["doc_id"] for d, _ in b]
    assert "ex_6" not in ids                                 # same database as the query
    assert len(ids) == len(set(ids))
    assert not ({"ex_2", "ex_3"} <= set(ids))                # duplicate question collapsed
    assert not ({"ex_2", "ex_5"} <= set(ids))                # duplicate MQL collapsed


def test_bm25_ranks_lexically_relevant_example_first():
    rt = RT.Retriever(toy_corpus(), "bm25_raw")
    top = rt.rank_examples("List the names ordered by age", INFO_T, "q")[0][0]
    assert top["doc_id"] == "ex_4"


def test_hybrid_uses_pattern_cues():
    rt = RT.Retriever(toy_corpus(), "hybrid_masked", hybrid_weight=5.0)
    top = rt.rank_examples("Show the youngest player sorted by order", INFO_T, "q")[0][0]
    assert "order_by" in top["patterns"]


def test_random_control_is_seeded_and_query_dependent():
    rt = RT.Retriever(toy_corpus(), "random", seed=1)
    a = [d["doc_id"] for d, _ in rt.rank_examples("x", INFO_T, "q1")]
    assert a == [d["doc_id"] for d, _ in RT.Retriever(toy_corpus(), "random", seed=1).rank_examples("x", INFO_T, "q1")]
    others = {tuple(d["doc_id"] for d, _ in rt.rank_examples("x", INFO_T, f"q{i}")) for i in range(2, 12)}
    assert len(others) > 1


def test_empty_corpus_gives_empty_retrieval_and_zero_shot_prompt():
    rt = RT.Retriever([], "bm25_masked")
    assert rt.rank_examples("How many singers?", INFO_T, "q") == []
    assert rt.rank_guides("How many singers?", INFO_T, []) == []
    msgs, info = RP.build_messages("concert_singer", "How many singers?", [], [], 3, 3, words, 1500, 8192, 1024)
    assert msgs == P.messages("concert_singer", "How many singers?") and info["empty_context"]


def test_guides_whose_example_uses_the_target_db_are_skipped():
    g = {"doc_id": "gx", "kind": "guide", "patterns": ["count"],
         "content": 'Example pipeline: [{"$lookup":{"from":"t__other"}}]'}
    rt = RT.Retriever(toy_corpus() + [g], "bm25_masked")
    assert "gx" not in [d["doc_id"] for d, _ in rt.rank_guides("How many dogs?", INFO_T, [])]
    info_u = dict(INFO_T, db_id="u")
    assert "gx" in [d["doc_id"] for d, _ in rt.rank_guides("How many dogs?", info_u, [])]


def test_guide_ranking_follows_patterns():
    rt = RT.Retriever(toy_corpus(), "bm25_masked")
    assert rt.rank_guides("How many dogs?", INFO_T, [])[0][0]["doc_id"] == "g1"
    assert rt.rank_guides("Who is the oldest?", INFO_T, [])[0][0]["doc_id"] == "g2"


@needs_corpus
def test_offline_retrieval_eval_selected_hybrid_and_beats_random():
    r = json.loads((ROOT / "outputs/rag_week3/retrieval_eval.json").read_text(encoding="utf-8"))
    assert r["selected_on_train_lodo_by_k3_mean_jaccard"] == "hybrid_masked_w1.0"
    for part in ("train_lodo", "validation"):
        assert r[part]["hybrid_masked_w1.0"]["k3"]["mean_jaccard"] > 3 * r[part]["random"]["k3"]["mean_jaccard"]


# ------------------------------------------------------------------ patterns / masking
def test_sql_patterns():
    p = PT.sql_patterns("SELECT name FROM a WHERE id NOT IN (SELECT id FROM b) ORDER BY x LIMIT 1")
    assert {"anti_join", "order_by", "limit", "where"} <= p and "nested" not in p
    p = PT.sql_patterns("SELECT count(DISTINCT x) FROM t JOIN u ON t.a=u.a WHERE y > (SELECT avg(y) FROM t)")
    assert {"count", "count_distinct", "distinct", "join", "nested", "aggregate"} <= p
    assert "union" not in PT.sql_patterns("SELECT 'a union b' FROM t")      # literal ignored


def test_masking_replaces_schema_words_numbers_and_names(collections):
    info = PT.schema_info("concert_singer", collections)
    m = PT.mask_question("How many singers from France are older than 30?", info)
    assert "singer" not in m and "france" not in m and "30" not in m
    assert m.startswith("how many") and "num" in m and "val" in m


# ------------------------------------------------------------------ prompt construction and budget
def test_k0_prompt_is_byte_identical_to_zero_shot():
    msgs, info = RP.build_messages("concert_singer", "How many singers do we have?", [], [], 0, 0,
                                   words, 1500, 8192, 1024)
    assert msgs == P.messages("concert_singer", "How many singers do we have?")
    assert info["context_tokens"] == 0 and info["context"] == ""


def test_block_inserted_before_question_and_rest_unchanged():
    docs = toy_corpus()
    ranked = [(docs[0], 1.0), (docs[3], 0.5)]
    msgs, info = RP.build_messages("concert_singer", "How many singers?", ranked, [(docs[6], 1)], 2, 1,
                                   words, 1500, 8192, 1024)
    base = P.messages("concert_singer", "How many singers?")[0]["content"]
    u = msgs[0]["content"]
    head, _, tail = base.rpartition(RP.SPLIT_MARK)
    assert u.startswith(head) and u.endswith(RP.SPLIT_MARK + tail)
    assert u.index("Example 1 (database a)") < u.index("Example 2 (database c)") < u.index("Question: How many singers?\n\nReturn")
    assert info["selected_examples"] == ["ex_1", "ex_4"] and info["selected_guides"] == ["g1"]


def test_context_budget_enforced_lowest_ranked_dropped():
    docs = toy_corpus()
    big = dict(docs[3], doc_id="ex_big", content="word " * 400)
    ranked = [(docs[0], 3.0), (big, 2.0), (docs[1], 1.0)]
    msgs, info = RP.build_messages("concert_singer", "q?", ranked, [], 3, 0, words, 120, 8192, 1024)
    assert info["context_tokens"] <= 120
    assert "ex_big" in info["dropped_budget"] and info["selected_examples"][0] == "ex_1"


def test_n_ctx_overflow_never_exceeded():
    docs = toy_corpus()
    base_tokens = words(P.SYSTEM) + words(P.messages("concert_singer", "q?")[0]["content"])
    n_ctx = base_tokens + 1024 + 64 + 5          # room for almost nothing
    msgs, info = RP.build_messages("concert_singer", "q?", [(docs[0], 1)], [], 1, 0, words, 1500, n_ctx, 1024)
    assert info["selected_examples"] == [] and info["dropped_budget"] == ["ex_1"]
    assert msgs == P.messages("concert_singer", "q?")


def test_fk_type_notes_are_schema_derived(collections):
    tables = K.load_json(K.SPIDER / "tables.json")
    notes = RP.fk_type_notes("museum_visit", collections, tables)
    assert notes == ["- museum_visit__visit.visitor_ID (string) -> museum_visit__visitor.ID (int)"]
    assert RP.fk_type_notes("battle_death", collections, tables) == []


def test_notes_rendered_and_dropped_when_over_budget():
    note = ["- a__x.k (string) -> a__y.k (int)"]
    msgs, info = RP.build_messages("concert_singer", "q?", [], [], 0, 0, words, 1500, 8192, 1024, notes=note)
    assert info["schema_notes"] == note and "DIFFERENT types" in msgs[0]["content"]
    msgs, info = RP.build_messages("concert_singer", "q?", [], [], 0, 0, words, 5, 8192, 1024, notes=note)
    assert info["schema_notes"] == [] and "schema_notes" in info["dropped_budget"]
    assert msgs == P.messages("concert_singer", "q?")


def test_saved_validation_runs_match_current_experiment_hashes():
    cfg, base, _ = RR.load_configs()
    for name in ("zs_rerun", "random_k3", "hybrid_k3", "hybrid_k3_g3"):
        path = ROOT / "outputs/rag_week3" / name / "executions.jsonl"
        if not path.exists():
            continue
        rec = K.read_jsonl(path)[0]
        assert rec["experiment_sha256"] == RR.experiment_hash(name, cfg["experiments"][name], cfg, base,
                                                              rec["corpus_sha256"])


def test_insert_block_requires_question_marker():
    with pytest.raises(ValueError):
        RP.insert_block("no marker here", "block")


# ------------------------------------------------------------------ configuration
def test_rag_config_uses_unchanged_base_config():
    cfg, base, sha = RR.load_configs()
    assert cfg["base_config"] == "configs/evaluation_spider1_expanded.yaml"
    assert base["generation"] == {"temperature": 0, "top_p": 1.0, "seed": 42, "max_new_tokens": 1024}
    assert base["models"]["qwen"]["gguf_sha256"].startswith("724fb256")
    assert cfg["provider"] == "qwen" and cfg["partition"] == "validation"


def test_experiment_hash_is_stable_and_sensitive():
    cfg, base, _ = RR.load_configs()
    e = cfg["experiments"]["hybrid_k3"]
    h = RR.experiment_hash("hybrid_k3", e, cfg, base, "c")
    assert h == RR.experiment_hash("hybrid_k3", dict(e), cfg, base, "c")
    assert h != RR.experiment_hash("hybrid_k3", {**e, "k": 4}, cfg, base, "c")
    assert h != RR.experiment_hash("hybrid_k3", e, cfg, base, "other-corpus")


def test_runner_refuses_test_partition():
    p = subprocess.run([sys.executable, str(ROOT / "scripts/rag/run_rag_eval.py"), "--experiment", "hybrid_k3",
                        "--partition", "test", "--dry-run"], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode != 0 and "test partition" in (p.stderr + p.stdout)


# ------------------------------------------------------------------ scoring and failure handling
class FakeColl:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def aggregate(self, pipeline, **kw):
        self.db.calls.append(self.name)
        if isinstance(self.db.result, Exception):
            raise self.db.result
        return iter(self.db.result)


class FakeDB:
    def __init__(self, result):
        self.result, self.calls = result, []

    def __getitem__(self, name):
        return FakeColl(self, name)


EX = {"id": "x", "db_id": "shop", "hardness": "easy", "question": "q", "gold_rows": [[3]],
      "order_by": False, "tie_blocks": None}
ALLOWED = {"shop__item"}


@pytest.mark.parametrize("text,category", [
    ("not json at all", "format_error"),
    ('{"collection": "shop__item", "pipeline": [{"$count": "n"}', "format_error"),        # truncated
    ('{"collection": "car_1__cars", "pipeline": [{"$count": "n"}]}', "invalid_mql"),       # copied other-db name
    ('{"collection": "shop__item", "pipeline": [{"$lookup": {"from": "car_1__cars", "as": "j", "localField": "a", "foreignField": "b"}}]}', "invalid_mql"),
    ('{"collection": "shop__item", "pipeline": [{"$out": "x"}]}', "invalid_mql"),
    ('{"collection": "shop__item", "pipeline": [{"$match": {"$expr": {"$function": {}}}}]}', "invalid_mql"),
])
def test_bad_outputs_never_reach_atlas(text, category):
    db = FakeDB([{"n": 3}])
    res, _ = R.score_output(db, text, EX, EXEC, ALLOWED)
    assert res["category"] == category and not res["correct"] and db.calls == []


def test_execution_error_and_correct_paths():
    from pymongo.errors import OperationFailure
    ok, _ = R.score_output(FakeDB([{"n": 3}]), '{"collection": "shop__item", "pipeline": [{"$count": "n"}]}', EX, EXEC, ALLOWED)
    assert ok["correct"]
    bad, _ = R.score_output(FakeDB(OperationFailure("boom")), '{"collection": "shop__item", "pipeline": [{"$count": "n"}]}', EX, EXEC, ALLOWED)
    assert bad["category"] == "execution_error"


def test_rescoring_saved_records_is_deterministic():
    rec_ok = {"id": "x", "raw_text": '{"collection": "shop__item", "pipeline": [{"$count": "n"}]}',
              "result": {"correct": True, "category": "correct"}}
    rec_bad = {"id": "x", "raw_text": "garbage", "result": {"correct": False, "category": "format_error"}}
    rec_wrong_saved = dict(rec_ok, result={"correct": False, "category": "wrong_values"})
    db = FakeDB([{"n": 3}])
    agree, diff = rescore_rag.rescore_records(db, [rec_ok, rec_bad, rec_ok], {"x": EX}, ["shop__item"])
    assert agree == 3 and diff == []
    agree, diff = rescore_rag.rescore_records(db, [rec_wrong_saved], {"x": EX}, ["shop__item"])
    assert agree == 0 and diff[0]["now"] == "correct"


def test_saved_rag_records_are_complete_and_resumable():
    out = ROOT / "outputs/rag_week3"
    runs = [p for p in out.iterdir() if (p / "executions.jsonl").exists()] if out.exists() else []
    if not runs:
        pytest.skip("no RAG runs yet")
    for run in runs:
        recs = K.read_jsonl(run / "executions.jsonl")
        assert len({r["id"] for r in recs}) == len(recs)                # no duplicates after resume
        assert len({r["experiment_sha256"] for r in recs}) == 1         # one config per directory
        part = run.name.split("__")[1] if "__" in run.name else "validation"
        assert part != "test"
        for r in recs:
            assert r["partition"] == part
            ret = r["retrieval"]
            for key in ("candidates", "selected_examples", "selected_guides", "context", "context_tokens",
                        "retrieval_latency_s"):
                assert key in ret
            assert ret["context_tokens"] <= 1500
            assert ("generation_wall_s" in r) and ("total_latency_s" in r)
            if r["config"]["k"] == 0 and r["config"]["k_guides"] == 0:
                assert ret["context"] == "" and ret["selected_examples"] == []


# ------------------------------------------------------------------ review follow-ups
import build_corpus as BC  # noqa: E402
import rag_report as RPT  # noqa: E402


def test_index_and_normaliser_exclude_target_db():
    """True leave-one-database-out: target-db docs must not shape IDF or the hybrid normaliser."""
    docs = toy_corpus()
    rt = RT.Retriever(docs, "hybrid_masked", hybrid_weight=1.0)
    info_b = dict(INFO_T, db_id="b")
    ranked = rt.rank_examples("How many cars are there?", info_b, "q")
    without_b = RT.Retriever([d for d in docs if d.get("db_id") != "b"], "hybrid_masked", hybrid_weight=1.0)
    assert ranked == without_b.rank_examples("How many cars are there?", info_b, "q")
    cand, _ = rt._index_for("b")
    assert all(d["db_id"] != "b" for d in cand)


def test_wilson_matches_known_values():
    lo, hi = RPT.wilson(43, 72)
    assert (lo, hi) == (48.2, 70.3)
    assert RPT.wilson(0, 0) == (None, None)


def test_paired_counts_mcnemar_and_bootstrap():
    a = {i: i < 30 for i in range(72)}            # baseline correct on 0..29
    b = {i: 6 <= i < 46 for i in range(72)}       # rag loses 0..5, gains 30..45
    p = RPT.paired(a, b)
    assert (p["both"], p["only_baseline"], p["only_rag"], p["neither"]) == (24, 6, 16, 26)
    assert p["diff_points"] == round((40 - 30) / 72 * 100, 1)
    from scipy.stats import binomtest
    assert p["mcnemar_exact_p"] == round(binomtest(16, 22, 0.5).pvalue, 4)
    lo, hi = p["bootstrap_95ci_points"]
    assert lo < p["diff_points"] < hi
    assert RPT.paired(a, b) == p                   # seeded


def test_holm_step_down():
    assert RPT.holm({"a": 0.0525, "b": 0.1338, "c": 0.424}) == {"a": 0.1575, "b": 0.2676, "c": 0.424}
    assert RPT.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert "hybrid_k3_fk" not in RPT.CONFIRMATORY


def test_cross_db_copy_detector():
    recs = [{"id": "1", "db_id": "museum_visit", "result": {"category": "invalid_mql",
             "reason": "collection 'car_1__cars' not among the example's allowed collections"}},
            {"id": "2", "db_id": "museum_visit", "result": {"category": "invalid_mql", "reason": "missing 'collection'"}},
            {"id": "3", "db_id": "museum_visit", "result": {"category": "correct", "reason": None}}]
    assert RPT.cross_db_copy(recs) == ["1"]


@needs_corpus
def test_guide_pipelines_pass_read_only_validator(corpus):
    colls = list(C.load_collections())
    for card in yaml.safe_load(C.GUIDES_SRC.read_text(encoding="utf-8"))["cards"]:
        BC.check_guide_pipeline(card["example"], colls)          # raises on violation
    with pytest.raises(ValueError):
        BC.check_guide_pipeline({"collection": None, "pipeline": [{"$documents": []}, {"$out": "x"}]}, colls)
    with pytest.raises(ValueError):
        BC.check_guide_pipeline({"collection": None, "pipeline": [{"$match": {}}]}, colls)
    with pytest.raises(ValueError):
        BC.check_guide_pipeline({"collection": "concert_singer__concert",
                                 "pipeline": [{"$lookup": {"from": "car_1__cars", "as": "j", "localField": "a",
                                                           "foreignField": "b"}}]}, colls)


@needs_corpus
@pytest.mark.parametrize("field", ["question", "masked_question", "mql", "content"])
def test_leakage_check_scans_every_field_and_near_variants(corpus, data, split, collections, field):
    db_part = split[0]
    val = next(r for r in data if db_part[r["db_id"]] == "validation")
    bad = dict(next(d for d in corpus if d["kind"] == "example"), doc_id="injected")
    variant = val["question"].rstrip("?").upper() + " !!"        # punctuation / case variant
    if field == "mql":
        bad["mql"] = {"collection": next(c for c, i in collections.items() if i["db_id"] == val["db_id"]),
                      "pipeline": []}
    else:
        bad[field] = (bad.get(field) or "") + " " + variant
    assert leakage.check(corpus + [bad], data, db_part, collections)


@needs_corpus
def test_leakage_check_flags_guide_with_heldout_start_collection(corpus, data, split, collections):
    db_part = split[0]
    card = {"id": "g_bad", "example": {"collection": "museum_visit__visitor", "pipeline": [{"$count": "n"}]}}
    errs = leakage.check(corpus, data, db_part, collections, [card])
    assert any("g_bad" in e for e in errs)
    card = {"id": "g_bad2", "example": {"collection": "concert_singer__singer", "pipeline": [],
                                        "check_db": "pets_1"}}
    assert any("g_bad2" in e for e in leakage.check(corpus, data, db_part, collections, [card]))


def test_saved_records_contexts_reproduce_under_current_code(collections):
    """Retrieval + context of every saved RAG record is re-derived with the current code."""
    out = ROOT / "outputs/rag_week3"
    cfg, base, _ = RR.load_configs()
    corpus = C.load_corpus()
    checked = 0
    for name, exp in cfg["experiments"].items():
        dirs = [out / name] + sorted(out.glob(f"{name}__*"))
        for path in (d / "executions.jsonl" for d in dirs if (d / "executions.jsonl").exists()):
            recs = K.read_jsonl(path)[:8]
            if not recs or exp["retrieval"] == "none":
                continue
            rt = RT.Retriever(corpus, exp["retrieval"], hybrid_weight=exp.get("hybrid_weight", 0.5),
                              seed=exp.get("seed", 20261020))
            for r in recs:
                ex = {"id": r["id"], "db_id": r["db_id"], "question": r["question"]}
                _, ret = RR.retrieve_and_prompt(ex, exp, rt, PT.schema_info(r["db_id"], collections),
                                                words, cfg, base, collections)
                assert ret["selected_examples"] == r["retrieval"]["selected_examples"], (path, r["id"])
                assert ret["selected_guides"] == r["retrieval"]["selected_guides"], (path, r["id"])
                assert ret["context"] == r["retrieval"]["context"], (path, r["id"])
                checked += 1
    if not checked:
        pytest.skip("no RAG runs yet")


class _FakeProvider:
    model_id = "fake"

    def available(self):
        return True, "fake"

    def generate(self, *a, **k):  # must never be reached when resume is refused
        raise AssertionError("generation attempted")


def _resume_setup(tmp_path, monkeypatch, mutate):
    cfg = yaml.safe_load((ROOT / "configs/rag_week3.yaml").read_text(encoding="utf-8"))
    cfg["output_dir"] = str(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    saved = next(iter(map(json.loads, (ROOT / "outputs/rag_week3/hybrid_k3/executions.jsonl")
                          .read_text(encoding="utf-8").splitlines())))
    mutate(saved)
    (tmp_path / "hybrid_k3").mkdir()
    (tmp_path / "hybrid_k3" / "executions.jsonl").write_text(json.dumps(saved) + "\n", encoding="utf-8")
    monkeypatch.setattr(RR, "build_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(RR, "make_counter", lambda cfg: lambda text: len(text) // 4)
    monkeypatch.setattr(RR.K, "mongo_client", lambda: (_ for _ in ()).throw(AssertionError("Atlas reached")))
    monkeypatch.setattr(sys, "argv", ["run_rag_eval.py", "--experiment", "hybrid_k3", "--config", str(cfg_path),
                                      "--limit", "1"])


@pytest.mark.parametrize("field,value,msg", [
    ("experiment_sha256", "0" * 64, "different config"),
    ("partition", "train", "different config"),
    ("dataset_sha256", "0" * 64, "dataset_sha256"),
    ("code_sha256", "0" * 64, "code_sha256"),
])
def test_resume_refuses_mismatched_records(tmp_path, monkeypatch, field, value, msg):
    """A saved record from another config/partition/dataset/code version stops the run before any
    model call or Atlas connection (no silent mixing of records)."""
    _resume_setup(tmp_path, monkeypatch, lambda r: r.__setitem__(field, value))
    with pytest.raises(SystemExit) as e:
        RR.main()
    assert msg in str(e.value)
