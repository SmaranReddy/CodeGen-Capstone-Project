"""Tests for the MQL-0007 Kaggle package. Run from anywhere:  python -m pytest kaggle/nl2mql_mql0007_rag_v1/tests -q

Package-only tests need nothing but pytest. Tests that need the repo (scripts/rag originals, protected datasets, gold
SQL) skip when the repo is absent (e.g. on Kaggle). The end-to-end notebook tests need `transformers` (tokenizer only)
and MQL0007_TOKENIZER_DIR (the pinned revision's tokenizer files); they skip otherwise. No GPU, no network, no Atlas.
"""
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
sys.path.insert(0, str(PKG / "src"))
import mql0007_core as C  # noqa: E402

TOK_DIR = os.environ.get("MQL0007_TOKENIZER_DIR")
HAVE_REPO = (REPO / "scripts/rag/retrieval.py").is_file() and (REPO / "kaggle/nl2mql_spider_dev_1034_bundle_v2/data/eval_records.jsonl").is_file()
needs_repo = pytest.mark.skipif(not HAVE_REPO, reason="repo (scripts/rag, bundle_v2) not available")


def rj(p):
    return C.read_jsonl(PKG / p)


@pytest.fixture(scope="module")
def cfg():
    return json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus():
    return rj("data/corpus/provisional_rag_corpus.jsonl")


@pytest.fixture(scope="module")
def evals():
    return rj("data/eval/eval_inputs.jsonl")


@pytest.fixture(scope="module")
def prompts():
    return rj("data/eval/prompts.jsonl")


@pytest.fixture(scope="module")
def tables():
    return {t["db_id"]: t for t in json.loads((PKG / "data/context/tables_subset.json").read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def docs(corpus, tables):
    return C.build_docs(corpus, tables)


@pytest.fixture(scope="module")
def einfo(evals):
    atlas = json.loads((PKG / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
    return C.eval_schema_infos(atlas, sorted({e["db_id"] for e in evals}))


def words(s):
    return len(s.split())


# ------------------------------------------------------------------ package integrity
def test_manifest_hashes_and_no_extra_files():
    man = C.verify_package(PKG)
    assert man["package"] == "nl2mql_mql0007_rag_v1" and man["status"] == "PREPARED - NOT RUN"
    sums = {ln.partition("  ")[2]: ln.partition("  ")[0] for ln in (PKG / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()}
    assert sums == {n: m["sha256"] for n, m in man["files"].items()}


def test_exact_counts_and_hashes(cfg, corpus, evals, prompts):
    assert len(corpus) == 3589 == len({r["id"] for r in corpus}) == cfg["corpus"]["n_entries"]
    assert C.sha256_file(PKG / "data/corpus/provisional_rag_corpus.jsonl") == cfg["corpus"]["source_sha256"]
    ids = [e["id"] for e in evals]
    assert len(ids) == 892 == len(set(ids))
    assert C.sha256_text("\n".join(ids)) == cfg["evaluation"]["ids_sha256"]
    assert [p["id"] for p in prompts] == ids
    assert len({e["db_id"] for e in evals}) == 17
    assert all(set(e) == {"id", "db_id", "question"} for e in evals)


# ------------------------------------------------------------------ leakage / eligibility
def test_no_database_overlap_and_no_held_out_databases(corpus, evals):
    assert not {r["db_id"] for r in corpus} & {e["db_id"] for e in evals}
    assert not {r["db_id"] for r in corpus} & {"employee_hire_evaluation", "pets_1", "tvshow"}   # held-out MQL test DBs


def test_no_exact_or_unacknowledged_near_duplicates(cfg, corpus, evals):
    ack = cfg["leakage_gates"]["acknowledged_near_duplicates"]
    rep = C.leakage_report(corpus, evals, cfg["leakage_gates"]["near_duplicate_token_jaccard"])
    assert rep["exact"] == [] and rep["db_overlap"] == []
    assert {(e, c) for e, c, _ in rep["near"]} == {(e, ack["corpus_id"]) for e in ack["eval_ids"]}   # user-acknowledged pair only
    assert rep["max_jaccard"] == pytest.approx(0.8)


def test_leakage_gate_detects_planted_duplicates(cfg, corpus, evals):
    planted = [dict(corpus[0], question=evals[5]["question"], question_raw=evals[5]["question"], db_id="zz_other")]
    assert C.leakage_report(planted, evals[:10], 0.8)["exact"]
    near = evals[7]["question"] + " today"
    planted = [dict(corpus[0], question=near, question_raw=near)]
    assert C.leakage_report(planted, evals[:10], 0.8)["near"]
    planted = [dict(corpus[0], db_id=evals[0]["db_id"])]
    assert C.leakage_report(planted, evals[:2], 0.8)["db_overlap"] == [evals[0]["db_id"]]


def test_only_eligible_training_rows(corpus):
    excl = json.loads((PKG / "data/corpus/excluded_accepted_ids.json").read_text(encoding="utf-8"))
    ids = {r["id"] for r in corpus}
    assert not ids & (set(excl["ids"]) | set(excl["pre_excluded"]))
    assert all(r["source"]["split"] == "train" and r["verification"]["status"] == "accepted" and r["id"].startswith("spider_train_")
               for r in corpus)
    assert all(r["corpus_status"] == "PROVISIONAL" and r["human_review_status"] == "NOT_REVIEWED" for r in corpus)


def test_retrieval_excludes_target_db_and_returns_only_corpus(docs, evals, einfo, cfg):
    by_id = {d["doc_id"]: d for d in docs}
    for arm in ("B_random_k3", "C_hybrid_k3"):
        plan = rj(f"data/plans/retrieval_plan_{arm}.jsonl")
        assert [p["id"] for p in plan] == [e["id"] for e in evals]
        for p in plan:
            assert len(p["selected_ids"]) == 3 and len(set(p["selected_ids"])) == 3
            assert all(i in by_id for i in p["selected_ids"])
            assert all(by_id[i]["db_id"] != p["db_id"] for i in p["selected_ids"])
            assert p["selected_db_ids"] == [by_id[i]["db_id"] for i in p["selected_ids"]]


@pytest.mark.parametrize("method", ["random", "hybrid_masked"])
def test_retriever_filters_target_database_even_if_present(docs, einfo, evals, method):
    """A target-database document with a perfect textual match must never be returned (guard is applied twice)."""
    e = evals[0]
    trap = dict(docs[0], doc_id="TRAP", db_id=e["db_id"], question=e["question"],
                masked_question=C.PT.mask_question(e["question"], einfo[e["db_id"]]), mql={"collection": "t", "pipeline": []})
    r = C.RT.Retriever(docs + [trap], method=method, hybrid_weight=1.0, seed=20261020)
    got = [d["doc_id"] for d, _ in r.rank_examples(e["question"], einfo[e["db_id"]], qid=e["id"])]
    assert "TRAP" not in got and len(got) > 1000


def test_retrieval_is_deterministic_and_tie_broken(docs, evals, einfo, cfg, prompts):
    for arm in ("B_random_k3", "C_hybrid_k3"):
        a = C.build_plan(cfg, arm, evals[:25], prompts, docs, einfo, words)
        b = C.build_plan(cfg, arm, evals[:25], prompts, docs, einfo, words)
        assert a == b
    r = C.make_retriever(cfg, "C_hybrid_k3", docs)
    ranked = r.rank_examples(evals[3]["question"], einfo[evals[3]["db_id"]], qid="x")
    keys = [(-s, d["doc_id"]) for d, s in ranked]
    assert keys == sorted(keys)                                      # score desc, then doc_id


def test_over_budget_examples_are_skipped_never_truncated(docs, evals, einfo, cfg, prompts):
    r = C.make_retriever(cfg, "C_hybrid_k3", docs)
    e, fp = evals[0], prompts[0]
    ranked = r.rank_examples(e["question"], einfo[e["db_id"]], qid=e["id"])
    big = dict(ranked[0][0], content="Question: x\nMQL: " + "y" * 4000)
    ranked = [(big, 9.0)] + ranked
    ct = lambda s: len(s)    # char budget so that only the huge example breaks the 1500 cap   # noqa: E731
    chosen, dropped, block, bt, total = C.select_examples(e["db_id"], fp["system"], fp["messages"][0]["content"], ranked, 3, ct,
                                                          1500, 10 ** 9, 1024, 64)
    assert big["doc_id"] in dropped and big not in chosen
    assert all(d["content"] in block for d in chosen) and "yyyy" not in block      # whole examples only
    assert bt <= 1500
    # n_ctx guard: nothing fits -> empty block, byte-identical to zero-shot
    chosen, dropped, block, bt, total = C.select_examples(e["db_id"], fp["system"], fp["messages"][0]["content"], ranked, 3, ct,
                                                          1500, 10, 1024, 64)
    assert chosen == [] and block == "" and C.RB.insert_block(fp["messages"][0]["content"], block) == fp["messages"][0]["content"]


def test_exclusion_is_frozen_in_the_plans_and_supplementary_arm_is_exact(cfg, evals):
    assert {(x["eval_id"], x["corpus_id"]) for x in cfg["retrieval_exclusions"]} == {("spider_dev_0383", "spider_train_00650"),
                                                                                      ("spider_dev_0824", "spider_train_00650")}
    for arm in ("B_random_k3", "C_hybrid_k3"):
        plan = rj(f"data/plans/retrieval_plan_{arm}.jsonl")
        assert all("spider_train_00650" not in p["selected_ids"] for p in plan if p["id"] in ("spider_dev_0383", "spider_dev_0824"))
        assert sum(bool(p["excluded_by_rule"]) for p in plan) == 2
    d = rj(f"data/plans/retrieval_plan_{C.SUPP_ARM}.jsonl")
    c = {p["id"]: p for p in rj("data/plans/retrieval_plan_C_hybrid_k3.jsonl")}
    assert [p["id"] for p in d] == ["spider_dev_0383"] and "spider_train_00650" in d[0]["selected_ids"] and d[0]["excluded_by_rule"] == []
    assert d[0]["selected_ids"] != c["spider_dev_0383"]["selected_ids"] and len(d) == 1 < len(evals)


def test_config_values_match_the_specification(cfg):
    assert cfg["model"]["revision"] == "488639f1ff808d1d3d0ba301aef8c11461451ec5" and cfg["model"]["id"] == "Qwen/Qwen2.5-Coder-3B-Instruct"
    r, c = cfg["retrieval"], cfg["context"]
    assert (r["bm25_k1"], r["bm25_b"], r["hybrid_weight"], r["guides"]) == (1.5, 0.75, 1.0, 0)
    assert (c["k"], c["max_context_tokens"]) == (3, 1500)
    assert cfg["arms"]["C_hybrid_k3"]["k"] == cfg["arms"]["B_random_k3"]["k"] == 3 and cfg["arms"]["B_random_k3"]["seed"] == 20261020
    assert C.retriever_constants_ok(cfg)


def test_prompts_arm_a_identical_and_rag_prompt_structure(docs, evals, einfo, cfg, prompts):
    plan_a = C.build_plan(cfg, "A_zero_shot", evals[:30], prompts, docs, einfo, words)
    pa = C.prompts_for_arm(plan_a, prompts, {d["doc_id"]: d for d in docs})
    assert [p["prompt_sha256"] for p in pa] == [p["prompt_sha256"] for p in prompts[:30]]
    plan_c = C.build_plan(cfg, "C_hybrid_k3", evals[:30], prompts, docs, einfo, words)
    pc = C.prompts_for_arm(plan_c, prompts, {d["doc_id"]: d for d in docs})
    for z, p in zip(prompts, pc):
        base, user = z["messages"][0]["content"], p["messages"][0]["content"]
        assert p["system"] == z["system"] and user.endswith("\n\nQuestion: " + base.rpartition("\n\nQuestion: ")[2])
        assert user.count("Reference material (retrieved automatically") == 1 and user.count("Example ") >= 3
        assert user.replace(user[user.index("\n\nReference material"):user.rindex("\n\nQuestion: ")], "") == base


def test_no_gold_fields_credentials_or_env_in_package():
    bad_files = [p for p in PKG.rglob("*") if p.is_file() and (p.name.startswith(".env") or p.suffix in (".sqlite", ".gz", ".db"))]
    assert bad_files == []
    me = Path(__file__).resolve()
    pat = re.compile(r"mongo" r"db\+srv|mongo" r"db://|GR" r"OQ|MONGO" r"DB_URI|gs" r"k_|api[_-]?key\s*[=:]\s*['\"]\w{8}", re.I)
    for p in PKG.rglob("*"):
        if (p.is_file() and p.suffix in (".py", ".json", ".jsonl", ".md", ".ipynb", ".txt") and p != me
                and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts):
            txt = p.read_text(encoding="utf-8")
            if p.name == "provisional_rag_corpus.jsonl":
                txt = txt[:0] + "\n".join(l[:0] for l in txt.splitlines()[:1])   # corpus checked structurally below
            assert not pat.search(txt), f"credential-like text in {p.relative_to(PKG)}"
    ev = rj("data/eval/eval_inputs.jsonl")
    assert all("sql" not in e and "gold" not in json.dumps(e) for e in ev)
    for z in rj("data/eval/prompts.jsonl"):
        t = z["system"] + json.dumps(z["messages"])
        assert not any(m in t for m in ("ordered_sha256", "multiset_sha256", '"sql":'))
    assert not any((PKG / "data").rglob("reference*"))


# ------------------------------------------------------------------ checkpoints, resume, OOM
def _fake_batch(batch):
    return [(f"OUT:{p['prompt_sha256'][:8]}", {"latency_s": 0.1, "input_tokens": 5, "output_tokens": 3, "finish_reason": "stop"}, None)
            for p in batch]


def _prompts(n):
    return [{"id": f"q{i:03d}", "db_id": "d", "prompt_sha256": f"{i:064x}"} for i in range(n)]


def test_checkpoint_resume_no_duplicates_or_gaps(tmp_path):
    ps = _prompts(21)
    ids = [p["id"] for p in ps]
    full = C.Checkpoint(tmp_path / "a/ck.jsonl", ids, "fp", C.is_generation_retry)
    C.run_generation(ps, _fake_batch, full, "m", "arm", 4)
    part = C.Checkpoint(tmp_path / "b/ck.jsonl", ids, "fp", C.is_generation_retry)
    C.run_generation(ps, _fake_batch, part, "m", "arm", 4, max_batches=2)            # "killed" after 2 batches
    assert len(part.todo()) == 21 - 8
    with open(tmp_path / "b/ck.jsonl", "a", encoding="utf-8") as f:
        f.write('{"id": "q00')                                                         # torn last line from the crash
    resumed = C.Checkpoint(tmp_path / "b/ck.jsonl", ids, "fp", C.is_generation_retry)
    assert resumed.warnings and len(resumed.todo()) == 13
    C.run_generation(ps, _fake_batch, resumed, "m", "arm", 4)
    again = C.Checkpoint(tmp_path / "b/ck.jsonl", ids, "fp", C.is_generation_retry)
    assert again.todo() == [] and [r["id"] for r in again.ordered()] == ids
    lines = (tmp_path / "b/ck.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 21 == len({json.loads(l)["id"] for l in lines})
    assert [(r["id"], r["raw_text"]) for r in again.ordered()] == [(r["id"], r["raw_text"]) for r in full.ordered()]
    with pytest.raises(RuntimeError):                                                  # another run's fingerprint is refused
        C.Checkpoint(tmp_path / "b/ck.jsonl", ids, "other", C.is_generation_retry)
    with pytest.raises(RuntimeError):                                                  # a final record is never rewritten
        again.append([dict(again.records["q000"])])


def test_error_records_are_retried_once_not_duplicated(tmp_path):
    ps = _prompts(4)
    calls = {"n": 0}

    def flaky(batch):
        calls["n"] += 1
        return [(None, {}, "boom") if calls["n"] == 1 else (f"ok{p['id']}", {}, None) for p in batch]
    ck = C.Checkpoint(tmp_path / "ck.jsonl", [p["id"] for p in ps], "fp", C.is_generation_retry)
    C.run_generation(ps, flaky, ck, "m", "arm", 4)
    assert len(ck.todo()) == 4
    C.run_generation(ps, flaky, ck, "m", "arm", 4)
    assert ck.todo() == [] and all(r["status"] == "ok" for r in ck.ordered())


def test_oom_batch_falls_back_to_single_prompts_with_same_settings():
    seen = []

    def run_batch(batch):
        seen.append(len(batch))
        if len(batch) > 1:
            raise MemoryError("CUDA out of memory")
        return [("t", {"latency_s": 1}, None)]
    counters = {"batch": 0, "single_prompt": 0}
    g = C.oom_safe_generator(run_batch, lambda e: isinstance(e, MemoryError), lambda: None, 8, counters)
    out = g(_prompts(8))
    assert seen == [8] + [1] * 8 and counters == {"batch": 1, "single_prompt": 0}
    assert all(m["oom_recovery"] and m["effective_batch_size"] == 1 and m["configured_batch_size"] == 8 for _, m, _ in out)
    with pytest.raises(ValueError):                                                    # non-OOM errors propagate
        C.oom_safe_generator(lambda b: (_ for _ in ()).throw(ValueError("x")), lambda e: False, lambda: None, 8, counters)(_prompts(2))


def test_required_dtype_is_enforced_without_fallback(monkeypatch):
    class T:  # minimal torch stub: CUDA present but no bf16
        class cuda:
            is_available = staticmethod(lambda: True)
            is_bf16_supported = staticmethod(lambda: False)
    monkeypatch.setitem(sys.modules, "torch", T)
    tf = type(sys)("transformers")
    tf.AutoModelForCausalLM = tf.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", tf)
    with pytest.raises(C.PreflightError, match="bfloat16 is required"):
        C.make_qwen_generator("m", "r" * 40, 10, "bfloat16")


# ------------------------------------------------------------------ equivalence with the repo originals
@needs_repo
def test_vendored_retrieval_code_is_byte_identical_to_repo():
    for f in ("patterns.py", "retrieval.py"):
        assert C.sha256_file(PKG / "src" / f) == C.sha256_file(REPO / "scripts/rag" / f)


@needs_repo
def test_rag_block_and_budget_loop_match_scripts_rag_build_messages(docs, evals, einfo, cfg, prompts):
    sys.path.insert(0, str(REPO / "scripts/rag"))
    sys.path.insert(0, str(REPO / "scripts/spider1"))
    spec = importlib.util.spec_from_file_location("orig_rag_prompt", REPO / "scripts/rag/rag_prompt.py")
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    assert (orig.HEADER, orig.GUIDE_HEADER, orig.EXAMPLES_HEADER, orig.SPLIT_MARK, orig.RAG_PROMPT_VERSION) == \
           (C.RB.HEADER, C.RB.GUIDE_HEADER, C.RB.EXAMPLES_HEADER, C.RB.SPLIT_MARK, C.RB.RAG_PROMPT_VERSION)
    by_prompt = {p["id"]: p for p in prompts}
    for arm in ("B_random_k3", "C_hybrid_k3"):
        retr = C.make_retriever(cfg, arm, docs)
        for e in evals[::37]:
            fp = by_prompt[e["id"]]
            orig.P.messages = lambda db, q, _fp=fp: _fp["messages"]
            orig.P.SYSTEM = fp["system"]
            ranked = retr.rank_examples(e["question"], einfo[e["db_id"]], qid=e["id"])
            msgs, info = orig.build_messages(e["db_id"], e["question"], ranked, [], 3, 0, words, 1500, 8192, 1024, 64)
            chosen, dropped, block, bt, total = C.select_examples(e["db_id"], fp["system"], fp["messages"][0]["content"], ranked, 3,
                                                                  words, 1500, 8192, 1024, 64)
            assert msgs[0]["content"] == C.RB.insert_block(fp["messages"][0]["content"], block)
            assert info["selected_examples"] == [d["doc_id"] for d in chosen] and info["dropped_budget"] == dropped


@needs_repo
def test_gold_sql_and_gold_rows_are_not_in_the_package_and_no_protected_overlap(corpus, evals):
    gold = C.read_jsonl(REPO / "kaggle/nl2mql_spider_dev_1034_bundle_v2/data/eval_records.jsonl")
    assert [g["id"] for g in gold] == [e["id"] for e in evals]
    gold_sql = {C.norm_sql(g["sql"]) for g in gold}
    assert not gold_sql & {C.norm_sql(r["sql"]) for r in corpus}
    sq = re.compile(r"\s+")
    for p in (PKG / "data").rglob("*"):
        if p.is_file() and p.name not in ("provisional_rag_corpus.jsonl", "corpus_features.jsonl") and not p.name.startswith("retrieval_plan"):
            t = sq.sub(" ", p.read_text(encoding="utf-8")).lower()
            assert not any(len(g) > 12 and g in t for g in gold_sql), p.name
    corp_q = {C.norm_question(r["question"]) for r in corpus} | {C.norm_question(r["question_raw"]) for r in corpus}
    corp_s = {C.norm_sql(r["sql"]) for r in corpus}
    for name in ("dev_full", "test_full"):
        rows = C.read_jsonl(REPO / f"data/processed/spider_sql_v1/{name}.jsonl")
        assert not [r["id"] for r in rows if C.norm_question(r["question"]) in corp_q]
        assert not [r["id"] for r in rows if C.norm_sql(r["sql"]) in corp_s]
        assert not {r["db_id"] for r in rows} & {r["db_id"] for r in corpus}


@needs_repo
def test_prediction_records_are_a_superset_of_mql0006_records():
    old = C.read_jsonl(PKG / "data/zero_shot_mql0006/predictions.jsonl")[0]
    new = C.pred_record("i", "d", "fp", "m", "arm", "t", {"prompt_sha256": "x"})
    assert set(old) - {"latency_s", "batch_latency_s", "input_tokens", "output_tokens", "finish_reason", "oom_recovery",
                       "oom_retries", "configured_batch_size", "effective_batch_size"} <= set(new)


# ------------------------------------------------------------------ the notebook
def _nb():
    import nbformat
    return nbformat.read(PKG / "nl2mql_mql0007_rag_v1.ipynb", 4)


def test_notebook_syntax_tag_order_and_no_secrets():
    import ast
    nb = _nb()
    tags = [c.metadata["tags"][0] for c in nb.cells if c.cell_type == "code"]
    assert tags == ["config", "env", "package", "model", "plan", "preflight", "generate", "export"]
    src = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for c in nb.cells:
        if c.cell_type == "code":
            ast.parse(c.source)
    assert not re.search(r"Mongo" r"Client|MONGO" r"DB_URI|GR" r"OQ|load_dot" r"env|py" r"mongo|get" r"pass|user" r"data", src)
    # execution order: each tag's cell only uses names defined by earlier cells
    assert src.index("PKG = _find_pkg()") < src.index("DOCS = C.build_docs") < src.index("PRE = {") < src.index("ARM_RESULT, RUN_CONFIGS")
    assert "bfloat16" in src and "batch_size" in src
    assert all(not c.outputs for c in nb.cells if c.cell_type == "code")               # shipped without outputs


K_SUPP = C.SUPP_ARM


def fake_factory(oom_on=None):
    def factory(cfg):
        def run_batch(batch):
            if oom_on and len(batch) > 1 and any(p["id"] == oom_on for p in batch):
                raise MemoryError("CUDA out of memory")
            return [("```json\n" + json.dumps({"collection": "c", "pipeline": [], "p": p["prompt_sha256"][:8]}) + "\n```",
                     {"latency_s": 0.0, "input_tokens": 1, "output_tokens": 1, "finish_reason": "stop"}, None) for p in batch]
        info = {"dtype": "bfloat16", "gpu": "FAKE", "torch": "0", "transformers": "0", "resolved_revision": cfg["model"]["revision"]}
        return run_batch, info, (lambda e: isinstance(e, MemoryError)), (lambda: None)
    return factory


def run_notebook(tmp, **over):
    ns = {"__name__": "nb"}
    for c in _nb().cells:
        if c.cell_type != "code":
            continue
        exec(compile(c.source, f"<cell {c.metadata['tags'][0]}>", "exec"), ns)
        if c.metadata["tags"][0] == "config":
            ns.update({"PKG_DIR": str(PKG), "OUT_DIR": str(tmp / "out"), "OUTPUT_ZIP": str(tmp / "out.zip"),
                       "TOKENIZER_DIR": TOK_DIR, "GENERATOR_FACTORY": fake_factory(over.pop("oom_on", None)), **over})
    return ns


def m6_env():
    c = json.loads((PKG / "data/zero_shot_mql0006/run_config.json").read_text(encoding="utf-8"))
    return {"packages": dict(c["environment"]["packages"]), "gpu": c["gpu"]}


needs_tok = pytest.mark.skipif(not TOK_DIR or importlib.util.find_spec("transformers") is None,
                               reason="set MQL0007_TOKENIZER_DIR and install transformers (tokenizer only)")


@needs_tok
def test_notebook_end_to_end_with_fake_generator_all_arms(tmp_path):
    ns = run_notebook(tmp_path, RUN_ARMS=["A_zero_shot", "B_random_k3", "C_hybrid_k3"], oom_on="spider_dev_0003", COMPAT_ENV_OVERRIDE=m6_env())
    assert ns["ZS_RESOLVED"] == "reuse_mql0006"
    st = json.loads((tmp_path / "out/MQL0007_STATUS.json").read_text(encoding="utf-8"))
    assert st["status"] == "complete" and st["problems"] == []
    ids = [e["id"] for e in rj("data/eval/eval_inputs.jsonl")]
    for arm in C.ARMS:
        preds = C.read_jsonl(tmp_path / f"out/models/{arm}/predictions.jsonl")
        assert [p["id"] for p in preds] == ids and all(p["status"] == "ok" for p in preds)
        assert len(C.read_jsonl(tmp_path / f"out/models/{arm}/retrieval_log.jsonl")) == 892
    # arm A reuse is byte-identical to MQL-0006
    assert (tmp_path / "out/models/A_zero_shot/predictions.jsonl").read_bytes() == (PKG / "data/zero_shot_mql0006/predictions.jsonl").read_bytes()
    b = C.read_jsonl(tmp_path / "out/models/B_random_k3/predictions.jsonl")
    assert any(p.get("oom_recovery") for p in b) and ns["OOM_EVENTS"]["batch"] >= 1
    d = C.read_jsonl(tmp_path / "out/models/D_supp_allow_near_dup/predictions.jsonl")        # supplementary sensitivity arm
    assert [p["id"] for p in d] == ["spider_dev_0383"] and st["arm_expected_ids"][K_SUPP] == 1
    assert d[0]["prompt_sha256"] != next(p["prompt_sha256"] for p in C.read_jsonl(tmp_path / "out/models/C_hybrid_k3/predictions.jsonl") if p["id"] == "spider_dev_0383")
    cl = {r["id"]: r for r in C.read_jsonl(tmp_path / "out/models/C_hybrid_k3/retrieval_log.jsonl")}
    assert cl["spider_dev_0383"]["excluded_by_rule"][0]["corpus_id"] == "spider_train_00650" and "spider_train_00650" not in cl["spider_dev_0383"]["selected_ids"]
    assert (tmp_path / "out.zip").is_file() and (tmp_path / "out/SHA256SUMS.txt").is_file()
    # predictions come from the planned prompts, and RAG prompts differ from zero-shot
    assert {p["prompt_sha256"] for p in b}.isdisjoint({p["prompt_sha256"] for p in rj("data/eval/prompts.jsonl")})


@needs_tok
def test_zero_shot_auto_mode_generates_fresh_when_environment_differs_and_strict_mode_refuses(tmp_path):
    drift = m6_env()
    drift["packages"]["transformers"] = "9.9.9"
    ns = run_notebook(tmp_path / "auto", RUN_ARMS=["A_zero_shot"], COMPAT_ENV_OVERRIDE=drift)
    assert ns["ZS_RESOLVED"] == "fresh"
    cfg_a = json.loads((tmp_path / "auto/out/models/A_zero_shot/run_config.json").read_text(encoding="utf-8"))
    assert cfg_a["mode"] == "generated" and len(C.read_jsonl(tmp_path / "auto/out/models/A_zero_shot/predictions.jsonl")) == 892
    pre = json.loads((tmp_path / "auto/out/preflight/preflight.json").read_text(encoding="utf-8"))
    assert pre["checks"]["zero_shot_resolved_mode"] == "fresh" and not pre["checks"]["zero_shot_environment_compat"]["all_same"]
    with pytest.raises(C.PreflightError, match="differs from MQL-0006"):
        run_notebook(tmp_path / "strict", RUN_ARMS=["A_zero_shot"], ZERO_SHOT_MODE="reuse_mql0006", COMPAT_ENV_OVERRIDE=drift)


@needs_tok
def test_notebook_kill_and_resume_equals_uninterrupted(tmp_path):
    run_notebook(tmp_path / "full", RUN_ARMS=["C_hybrid_k3"])
    run_notebook(tmp_path / "kill", RUN_ARMS=["C_hybrid_k3"], STOP_AFTER_BATCHES=5)       # killed after 5 batches (40 items)
    part = json.loads((tmp_path / "kill/out/MQL0007_STATUS.json").read_text(encoding="utf-8"))
    assert part["status"] == "partial" and part["arms"]["C_hybrid_k3"]["n"] == 40 and "D_supp_allow_near_dup" not in part["arms"]
    ns = run_notebook(tmp_path / "kill", RUN_ARMS=["C_hybrid_k3"])                         # same OUT_DIR: resumes from the checkpoint
    assert json.loads((tmp_path / "kill/out/MQL0007_STATUS.json").read_text(encoding="utf-8"))["status"] == "complete"
    ck = (tmp_path / "kill/out/models/C_hybrid_k3/checkpoints/predictions.partial.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ck) == 892 == len({json.loads(l)["id"] for l in ck})                          # no duplicate, no missing
    pick = lambda d: [(p["id"], p["raw_text"], p["prompt_sha256"]) for p in C.read_jsonl(d / "out/models/C_hybrid_k3/predictions.jsonl")]  # noqa: E731
    assert pick(tmp_path / "kill") == pick(tmp_path / "full")
    # RESUME_FROM: a new OUT_DIR seeded from the earlier output (hash-verified)
    run_notebook(tmp_path / "k2", RUN_ARMS=["B_random_k3"], STOP_AFTER_BATCHES=3)
    prev = tmp_path / "k2/out"
    ns = run_notebook(tmp_path / "k3", RUN_ARMS=["B_random_k3"], RESUME_FROM=str(prev))
    assert ns["RESUMED"]["seeded"] and len(C.read_jsonl(tmp_path / "k3/out/models/B_random_k3/predictions.jsonl")) == 892
    # resuming into a directory whose run used another config is refused
    (prev / "models/B_random_k3/run_config.json").write_text("{}", encoding="utf-8")


@needs_tok
def test_frozen_plans_equal_rederived_plans_with_the_pinned_tokenizer(cfg, docs, evals, einfo, prompts):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    ct = lambda s: len(tok.encode(s, add_special_tokens=False))   # noqa: E731
    for arm in ("B_random_k3", "C_hybrid_k3"):
        assert C.build_plan(cfg, arm, evals, prompts, docs, einfo, ct) == rj(f"data/plans/retrieval_plan_{arm}.jsonl")


@needs_tok
def test_notebook_refuses_tampered_package(tmp_path):
    import shutil
    pkg = tmp_path / "pkg"
    shutil.copytree(PKG, pkg, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    with open(pkg / "data/eval/eval_inputs.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(C.PreflightError):
        run_notebook(tmp_path, PKG_DIR=str(pkg))
