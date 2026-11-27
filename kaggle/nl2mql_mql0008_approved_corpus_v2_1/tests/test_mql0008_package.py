"""Tests for the MQL-0008 Kaggle package. Run from anywhere:
    python -m pytest kaggle/nl2mql_mql0008_approved_corpus_v2_1/tests -q

Package-only tests need nothing but pytest. Tests that need the repo (frozen MQL-0007 package, approved corpus, gold
SQL of the 892 evaluation items) skip when the repo is absent (e.g. on Kaggle). The end-to-end notebook tests need
`transformers` (tokenizer only) and MQL0008_TOKENIZER_DIR (the pinned revision's tokenizer files); they skip otherwise.
No GPU, no network, no database. No test opens Spider dev/test or the 65-item held-out partition.
"""
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
sys.path.insert(0, str(PKG / "src"))
import mql0008_core as C  # noqa: E402

TOK_DIR = os.environ.get("MQL0008_TOKENIZER_DIR")
M7 = REPO / "kaggle/nl2mql_mql0007_rag_v1"
APPROVED = REPO / "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl"
GOLD = REPO / "kaggle/nl2mql_spider_dev_1034_bundle_v2/data/eval_records.jsonl"
HAVE_REPO = (M7 / "configs/experiment.json").is_file() and APPROVED.is_file() and GOLD.is_file()
needs_repo = pytest.mark.skipif(not HAVE_REPO, reason="repo (MQL-0007 package, approved corpus, bundle_v2) not available")
A, B = "arm_a_zero_shot", "arm_b_hybrid_rag"
SHA = "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf"
PLAN = f"data/plans/retrieval_plan_{B}.jsonl"
NB = PKG / "nl2mql_mql0008_approved_corpus_v2_1.ipynb"


def rj(p):
    return C.read_jsonl(PKG / p)


@pytest.fixture(scope="module")
def cfg():
    return json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus():
    return rj(C.CORPUS_FILE)


@pytest.fixture(scope="module")
def evals():
    return rj("data/eval/eval_inputs.jsonl")


@pytest.fixture(scope="module")
def prompts():
    return rj("data/eval/prompts.jsonl")


@pytest.fixture(scope="module")
def plan():
    return rj(PLAN)


@pytest.fixture(scope="module")
def docs(corpus):
    tables = {t["db_id"]: t for t in json.loads((PKG / "data/context/tables_subset.json").read_text(encoding="utf-8"))}
    return C.build_docs(corpus, tables)


@pytest.fixture(scope="module")
def einfo(evals):
    atlas = json.loads((PKG / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
    return C.eval_schema_infos(atlas, sorted({e["db_id"] for e in evals}))


def words(s):
    return len(s.split())


# ------------------------------------------------------------------ package integrity and identity
def test_manifest_hashes_and_no_extra_files():
    man = C.verify_package(PKG)
    assert man["package"] == "nl2mql_mql0008_approved_corpus_v2_1" and man["experiment"] == "MQL-0008"
    assert man["status"] == "PACKAGE BUILT - NOT YET EXECUTED"
    sums = {ln.partition("  ")[2]: ln.partition("  ")[0] for ln in (PKG / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()}
    assert sums == {n: m["sha256"] for n, m in man["files"].items()}
    assert man["retrieval_corpus"] == {"source_path": "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl",
                                       "package_path": C.CORPUS_FILE, "records": 3445, "sha256": SHA}


def test_verify_package_detects_tampering_and_extra_files(tmp_path):
    pkg = tmp_path / "pkg"
    shutil.copytree(PKG, pkg, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    C.verify_package(pkg)
    (pkg / "data/extra.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(C.PreflightError, match="package tree != manifest"):
        C.verify_package(pkg)
    (pkg / "data/extra.jsonl").unlink()
    with open(pkg / C.CORPUS_FILE, "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(C.PreflightError, match="hash mismatch"):
        C.verify_package(pkg)


def test_approved_corpus_identity_and_no_other_corpus(cfg, corpus):
    assert C.sha256_file(PKG / C.CORPUS_FILE) == SHA == C.EXPECTED_CORPUS_SHA256 == cfg["corpus"]["source_sha256"]
    assert len(corpus) == 3445 == len({r["id"] for r in corpus}) == cfg["corpus"]["n_entries"] == C.EXPECTED_CORPUS
    assert sorted(p.name for p in (PKG / "data/corpus").iterdir()) == ["approved_corpus_v2_1.jsonl", "corpus_features.jsonl"]
    assert {r["corpus_version"] for r in corpus} == {"provisional_corpus_v2"}        # v2.1 = v2 minus two quarantined ids
    assert {r["converter_version"] for r in corpus} == {"sql2mql-v2.1"}
    assert {r["verification"]["verifier"] for r in corpus} == {"sql2mql-v2-verifier-3.1"}
    assert not {r["id"] for r in corpus} & {"spider_train_00145", "spider_train_00146"}
    names = [p.relative_to(PKG).as_posix() for p in PKG.rglob("*") if p.is_file()]
    assert not [n for n in names if re.search(r"provisional_rag_corpus|spider_pool|corpus_v1|zero_shot_mql0006", n)]


def test_exactly_two_arms(cfg):
    assert C.ARMS == (A, B) and list(cfg["arms"]) == [A, B] and C.RAG_ARM == B
    assert cfg["arms"][A]["retrieval"] == "none" and cfg["arms"][B]["retrieval"] == "hybrid_masked"
    assert sorted(p.name for p in (PKG / "data/plans").iterdir()) == [f"retrieval_plan_{B}.jsonl"]
    assert "random" not in json.dumps(cfg["arms"]).lower()
    assert json.loads((PKG / "package_manifest.json").read_text(encoding="utf-8"))["arms"] == [A, B]
    with pytest.raises(C.PreflightError):
        C.make_retriever({"arms": {"x": {"retrieval": "random", "seed": 1}}, "retrieval": cfg["retrieval"]}, "x", [])


def test_evaluation_set_is_the_892(cfg, evals, prompts):
    ids = [e["id"] for e in evals]
    assert len(ids) == 892 == len(set(ids))
    assert C.sha256_text("\n".join(ids)) == cfg["evaluation"]["ids_sha256"]
    assert [p["id"] for p in prompts] == ids
    assert len({e["db_id"] for e in evals}) == 17
    assert all(set(e) == {"id", "db_id", "question"} for e in evals)


# ------------------------------------------------------------------ frozen configuration
def test_config_values_match_the_specification(cfg):
    m = cfg["model"]
    assert m["id"] == "Qwen/Qwen2.5-Coder-3B-Instruct" and m["revision"] == "488639f1ff808d1d3d0ba301aef8c11461451ec5"
    assert (m["dtype"], m["decoding"], m["max_new_tokens"], m["batch_size"], m["seed"]) == ("bfloat16", "greedy (do_sample=False)", 1024, 8, 42)
    r, c = cfg["retrieval"], cfg["context"]
    assert (r["bm25_k1"], r["bm25_b"], r["hybrid_weight"], r["guides"], r["method_relevant"]) == (1.5, 0.75, 1.0, 0, "hybrid_masked")
    assert (c["k"], c["max_context_tokens"], cfg["arms"][B]["k"]) == (3, 1500, 3)
    assert C.retriever_constants_ok(cfg)
    assert cfg["status"] == "PACKAGE BUILT - NOT YET EXECUTED"


@needs_repo
def test_configuration_equals_frozen_mql0007(cfg):
    m7 = json.loads((M7 / "configs/experiment.json").read_text(encoding="utf-8"))
    for k in ("prompt", "retrieval", "context", "retrieval_exclusions"):
        assert cfg[k] == m7[k], k
    drop = lambda d: {k: v for k, v in d.items() if k != "dtype_note"}   # noqa: E731
    assert drop(cfg["model"]) == drop(m7["model"])
    assert cfg["leakage_gates"]["near_duplicate_token_jaccard"] == m7["leakage_gates"]["near_duplicate_token_jaccard"] == 0.8
    assert (cfg["arms"][B]["retrieval"], cfg["arms"][B]["k"]) == (m7["arms"]["C_hybrid_k3"]["retrieval"], m7["arms"]["C_hybrid_k3"]["k"])
    for f in ("patterns.py", "retrieval.py", "rag_block.py"):                 # retrieval code byte-identical to MQL-0007
        assert C.sha256_file(PKG / "src" / f) == C.sha256_file(M7 / "src" / f)
    for f in ("data/eval/eval_inputs.jsonl", "data/eval/prompts.jsonl", "data/context/atlas_collections.json"):
        assert C.sha256_file(PKG / f) == C.sha256_file(M7 / f), f               # same questions, schemas and prompts


@needs_repo
def test_packaged_corpus_is_the_approved_file():
    assert (PKG / C.CORPUS_FILE).read_bytes() == APPROVED.read_bytes()
    man = json.loads((APPROVED.parent / "manifest.json").read_text(encoding="utf-8"))
    assert man["files"]["corpus.jsonl"] == {"sha256": SHA, "records": 3445}


# ------------------------------------------------------------------ leakage
def test_no_database_overlap_and_no_held_out_databases(corpus, evals):
    assert not {r["db_id"] for r in corpus} & {e["db_id"] for e in evals}
    assert not {r["db_id"] for r in corpus} & {"employee_hire_evaluation", "pets_1", "tvshow"}   # held-out MQL test DBs


def test_no_exact_or_undocumented_near_duplicates(cfg, corpus, evals):
    ack = cfg["leakage_gates"]["acknowledged_near_duplicates"]
    rep = C.leakage_report(corpus, evals, cfg["leakage_gates"]["near_duplicate_token_jaccard"])
    assert rep["exact"] == [] and rep["db_overlap"] == []
    raw = {r["question"] for r in corpus} | {r["question_raw"] for r in corpus}
    assert not [e["id"] for e in evals if e["question"] in raw]
    assert {(e, c) for e, c, _ in rep["near"]} == {(e, ack["corpus_id"]) for e in ack["eval_ids"]}
    assert rep["max_jaccard"] == pytest.approx(0.8)
    assert {(x["eval_id"], x["corpus_id"]) for x in cfg["retrieval_exclusions"]} == {("spider_dev_0383", "spider_train_00650"),
                                                                                      ("spider_dev_0824", "spider_train_00650")}


def test_leakage_gate_detects_planted_duplicates(corpus, evals):
    planted = [dict(corpus[0], question=evals[5]["question"], question_raw=evals[5]["question"], db_id="zz_other")]
    assert C.leakage_report(planted, evals[:10], 0.8)["exact"]
    near = evals[7]["question"] + " today"
    assert C.leakage_report([dict(corpus[0], question=near, question_raw=near)], evals[:10], 0.8)["near"]
    assert C.leakage_report([dict(corpus[0], db_id=evals[0]["db_id"])], evals[:2], 0.8)["db_overlap"] == [evals[0]["db_id"]]


def test_leakage_report_is_shipped_and_passes():
    rep = json.loads((PKG / "data/leakage_report.json").read_text(encoding="utf-8"))
    ch = rep["leakage_eval"]["checks"]
    assert rep["experiment"] == "MQL-0008" and rep["audited_corpus"]["sha256"] == SHA and rep["audited_corpus"]["records"] == 3445
    assert (ch["1_exact_question_string"], ch["2_normalised_question"], ch["4_database_overlap"],
            ch["5_corpus_sql_equal_to_an_evaluation_gold_sql"]) == (0, 0, [], 0)
    assert ch["3_near_duplicate_token_jaccard_ge_threshold"] == 2 and rep["leakage_eval"]["evaluation_items_removed"] == 0
    assert rep["protected_sets"]["files_opened"].startswith("none")
    assert rep["protected_sets"]["approved_records_identical_to_corpus_v1_record"].startswith("3445/3445")


def test_exclusion_is_frozen_in_the_plan_and_no_evaluation_item_is_removed(plan, evals):
    assert [p["id"] for p in plan] == [e["id"] for e in evals]                       # all 892 kept
    hit = {p["id"]: p for p in plan if p["excluded_by_rule"]}
    assert sorted(hit) == ["spider_dev_0383", "spider_dev_0824"]
    for p in hit.values():
        assert [x["corpus_id"] for x in p["excluded_by_rule"]] == ["spider_train_00650"]
        assert "spider_train_00650" not in p["selected_ids"] and len(p["selected_ids"]) == 3


def test_no_selected_example_is_a_near_duplicate_of_its_query(plan, evals, docs):
    assert C.selection_neardup_violations(plan, evals, {d["doc_id"]: d for d in docs}, 0.8) == []


# ------------------------------------------------------------------ retrieval
def test_retrieval_excludes_target_db_and_returns_only_approved_records(docs, plan):
    by_id = {d["doc_id"]: d for d in docs}
    for p in plan:
        assert len(p["selected_ids"]) == 3 == len(set(p["selected_ids"]))
        assert all(i in by_id for i in p["selected_ids"])
        assert all(by_id[i]["db_id"] != p["db_id"] for i in p["selected_ids"])
        assert p["selected_db_ids"] == [by_id[i]["db_id"] for i in p["selected_ids"]]
        assert len(p["selected_ranks"]) == 3 and p["selected_ranks"] == sorted(p["selected_ranks"]) and p["selected_ranks"][0] >= 1


def test_retriever_filters_target_database_even_if_present(docs, einfo, evals):
    """A target-database document with a perfect textual match must never be returned."""
    e = evals[0]
    trap = dict(docs[0], doc_id="TRAP", db_id=e["db_id"], question=e["question"],
                masked_question=C.PT.mask_question(e["question"], einfo[e["db_id"]]), mql={"collection": "t", "pipeline": []})
    r = C.RT.Retriever(docs + [trap], method="hybrid_masked", hybrid_weight=1.0)
    got = [d["doc_id"] for d, _ in r.rank_examples(e["question"], einfo[e["db_id"]], qid=e["id"])]
    assert "TRAP" not in got and len(got) > 1000


def test_retrieval_is_deterministic_and_tie_broken_on_fixed_fixtures(cfg, einfo, evals, prompts):
    """Fixed fixture corpus: two documents with identical text and patterns must be ordered by doc_id, every time."""
    e = evals[3]
    mk = lambda i, q: {"doc_id": i, "kind": "example", "db_id": "fx", "split": "train", "question": q,   # noqa: E731
                       "masked_question": q, "mql": {"collection": "fx__t", "pipeline": [{"$limit": int(i[-1]) + 1}]},
                       "content": f"Question: {q}\nMQL: {i}", "patterns": ["aggregate"]}
    fx = [mk("fx_3", e["question"]), mk("fx_1", e["question"]), mk("fx_2", "something unrelated about ships"),
          mk("fx_0", e["question"] + " please")]
    runs = [C.build_plan(cfg, B, [e], prompts, fx, einfo, words, exclusions=[]) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    ranked = C.make_retriever(cfg, B, fx).rank_examples(e["question"], einfo[e["db_id"]], qid="x")
    keys = [(-s, d["doc_id"]) for d, s in ranked]
    assert keys == sorted(keys)                                                        # score desc, then doc_id
    got = [d["doc_id"] for d, _ in ranked]
    assert got[0] == "fx_1" and "fx_3" not in got            # equal score: lower doc_id wins; the duplicate question is dropped
    assert C.build_plan(cfg, B, [e], prompts, list(reversed(fx)), einfo, words, exclusions=[]) == runs[0]   # input order irrelevant


def test_plan_on_real_corpus_is_deterministic(docs, evals, einfo, cfg, prompts):
    assert C.build_plan(cfg, B, evals[:25], prompts, docs, einfo, words) == C.build_plan(cfg, B, evals[:25], prompts, docs, einfo, words)


def test_frozen_plan_respects_the_token_cap(plan, cfg):
    c = cfg["context"]
    assert all(0 < p["context_tokens"] <= c["max_context_tokens"] for p in plan)
    assert all(p["prompt_tokens_est"] + c["max_new_tokens"] + c["margin_tokens"] <= c["n_ctx"] for p in plan)
    assert all(len(p["selected_ids"]) <= c["k"] for p in plan)


def test_over_budget_examples_are_skipped_never_truncated(docs, evals, einfo, cfg, prompts):
    r = C.make_retriever(cfg, B, docs)
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
    chosen, dropped, block, bt, total = C.select_examples(e["db_id"], fp["system"], fp["messages"][0]["content"], ranked, 3, ct,
                                                          1500, 10, 1024, 64)
    assert chosen == [] and block == "" and C.RB.insert_block(fp["messages"][0]["content"], block) == fp["messages"][0]["content"]


def test_arm_a_prompts_identical_and_rag_prompt_adds_only_the_reference_block(docs, evals, einfo, cfg, prompts, plan):
    by_id = {d["doc_id"]: d for d in docs}
    plan_a = C.build_plan(cfg, A, evals, prompts, docs, einfo, words)
    pa = C.prompts_for_arm(plan_a, prompts, by_id)
    assert [p["prompt_sha256"] for p in pa] == [p["prompt_sha256"] for p in prompts]
    assert all(p["empty_context"] and p["selected_ids"] == [] for p in plan_a)
    pb = C.prompts_for_arm(plan, prompts, by_id)                                       # the frozen plan, all 892
    for z, p in zip(prompts, pb):
        base, user = z["messages"][0]["content"], p["messages"][0]["content"]
        assert p["system"] == z["system"] and user.endswith("\n\nQuestion: " + base.rpartition("\n\nQuestion: ")[2])
        assert user.count("Reference material (retrieved automatically") == 1 and user.count("Example ") >= 3
        assert user.replace(user[user.index("\n\nReference material"):user.rindex("\n\nQuestion: ")], "") == base


def test_no_gold_fields_credentials_or_database_access_in_package():
    bad_files = [p for p in PKG.rglob("*") if p.is_file() and (p.name.startswith(".env") or p.suffix in (".sqlite", ".gz", ".db"))]
    assert bad_files == []
    me = Path(__file__).resolve()
    pat = re.compile(r"mongo" r"db\+srv|mongo" r"db://|GR" r"OQ|MONGO" r"DB_URI|gs" r"k_|api[_-]?key\s*[=:]\s*['\"]\w{8}", re.I)
    for p in PKG.rglob("*"):
        if (p.is_file() and p.suffix in (".py", ".json", ".jsonl", ".md", ".ipynb", ".txt") and p != me
                and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and p.name != Path(C.CORPUS_FILE).name):
            assert not pat.search(p.read_text(encoding="utf-8")), f"credential-like text in {p.relative_to(PKG)}"
    # no database driver, network client, credential read or write operation in any shipped code
    banned = {"pymongo", "motor", "dotenv", "getpass", "requests", "socket", "subprocess_run"}
    for p in list((PKG / "src").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
               {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not mods & banned, (p.name, mods & banned)
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not attrs & {"insert_one", "insert_many", "update_one", "update_many", "delete_one", "delete_many", "drop",
                            "environ", "getenv"}, p.name
    ev = rj("data/eval/eval_inputs.jsonl")
    assert all("sql" not in e and "gold" not in json.dumps(e) for e in ev)
    for z in rj("data/eval/prompts.jsonl"):
        t = z["system"] + json.dumps(z["messages"])
        assert not any(m in t for m in ("ordered_sha256", "multiset_sha256", '"sql":'))
    assert not any((PKG / "data").rglob("reference*"))


@needs_repo
def test_gold_sql_is_not_in_the_package_or_in_any_prompt(corpus, evals, prompts, plan, docs):
    gold = C.read_jsonl(GOLD)                       # the 892 evaluation items (not the held-out partition)
    assert [g["id"] for g in gold] == [e["id"] for e in evals]
    gold_sql = {C.norm_sql(g["sql"]) for g in gold}
    assert not gold_sql & {C.norm_sql(r["sql"]) for r in corpus}
    sq = re.compile(r"\s+")
    for p in (PKG / "data").rglob("*"):
        if p.is_file() and p.name not in (Path(C.CORPUS_FILE).name, "corpus_features.jsonl") and not p.name.startswith("retrieval_plan"):
            t = sq.sub(" ", p.read_text(encoding="utf-8")).lower()
            assert not any(len(g) > 12 and g in t for g in gold_sql), p.name
    by_gold = {g["id"]: sq.sub(" ", g["sql"]).strip().lower() for g in gold}
    for p in C.prompts_for_arm(plan, prompts, {d["doc_id"]: d for d in docs}):
        t = sq.sub(" ", p["system"] + " " + p["messages"][0]["content"]).lower()
        assert by_gold[p["id"]] not in t, p["id"]


# ------------------------------------------------------------------ checkpoints, resume, OOM, record identity
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
    with pytest.raises(RuntimeError):                                                  # an id outside the 892 is refused
        (tmp_path / "c").mkdir()
        (tmp_path / "c/ck.jsonl").write_text(json.dumps({"id": "zzz", "fingerprint": "fp", "status": "ok"}) + "\n", encoding="utf-8")
        C.Checkpoint(tmp_path / "c/ck.jsonl", ids, "fp", C.is_generation_retry)


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


def test_prediction_records_preserve_identity_and_retrieval_metadata(cfg, plan, prompts, docs, tmp_path):
    fp = C.retrieval_fingerprint(cfg)
    assert fp == json.loads((PKG / "package_manifest.json").read_text(encoding="utf-8"))["retrieval_config_fingerprint"]
    assert fp != C.retrieval_fingerprint({**cfg, "context": {**cfg["context"], "k": 4}})        # any change moves the fingerprint
    rows = plan[:10]
    ps = C.prompts_for_arm(rows, prompts, {d["doc_id"]: d for d in docs})
    ident = {p["id"]: C.record_identity(cfg, B, p, fp) for p in rows}
    ck = C.Checkpoint(tmp_path / "ck.jsonl", [p["id"] for p in rows], "fp", C.is_generation_retry)
    C.run_generation(ps, _fake_batch, ck, cfg["model"]["label"], B, 4, identity=ident)
    for rec, p in zip(ck.ordered(), rows):
        assert rec["id"] == p["id"] and rec["arm"] == B and rec["status"] == "ok" and rec["raw_text"].startswith("OUT:")
        assert rec["model_id"] == cfg["model"]["id"] and rec["model_revision"] == cfg["model"]["revision"]
        assert rec["generation_config"] == {"dtype": "bfloat16", "decoding": "greedy (do_sample=False)", "max_new_tokens": 1024,
                                            "batch_size": 8, "seed": 42}
        r = rec["retrieval"]
        assert r["retrieved_ids"] == p["selected_ids"] and r["ranks"] == p["selected_ranks"] and r["scores"] == p["selected_scores"]
        assert r["target_db_excluded"] is True and r["corpus_sha256"] == SHA and r["retrieval_config_fingerprint"] == fp
        assert r["context_tokens"] == p["context_tokens"] <= r["max_context_tokens"] == 1500 and r["dropped_over_budget"] == p["dropped_budget"]
        assert rec["prompt_sha256"] == p["prompt_sha256"]
    za = C.record_identity(cfg, A, {"id": "x"}, fp)
    assert za["retrieval"] is None and za["model_revision"] == cfg["model"]["revision"]
    # the scorer's prediction hash (id, raw_text, status) is unaffected by the extra identity fields
    rec = ck.ordered()[0]
    h = lambda r: hashlib.sha256(C.canon({k: r.get(k) for k in ("id", "raw_text", "status")}).encode()).hexdigest()   # noqa: E731
    assert h(rec) == h({k: rec[k] for k in ("id", "raw_text", "status")})


# ------------------------------------------------------------------ the notebook
def _nb():
    import nbformat
    return nbformat.read(NB, 4)


def test_notebook_syntax_tag_order_and_no_secrets_or_database():
    nb = _nb()
    tags = [c.metadata["tags"][0] for c in nb.cells if c.cell_type == "code"]
    assert tags == ["config", "env", "package", "model", "plan", "preflight", "generate", "export"]
    src = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for c in nb.cells:
        if c.cell_type == "code":
            ast.parse(c.source)
    assert not re.search(r"Mongo" r"Client|MONGO" r"DB_URI|GR" r"OQ|load_dot" r"env|py" r"mongo|get" r"pass|user" r"data|kaggle_sec" r"rets", src)
    assert src.index("PKG = _find_pkg()") < src.index("DOCS = C.build_docs") < src.index("PRE = {") < src.index("ARM_RESULT, RUN_CONFIGS")
    assert "bfloat16" in src and "batch_size" in src and "EXPECTED_CORPUS_SHA256" in src
    assert 'RUN_ARMS = ["arm_a_zero_shot", "arm_b_hybrid_rag"]' in src and "random" not in src.lower()
    assert all(not c.outputs for c in nb.cells if c.cell_type == "code")               # shipped without outputs


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


def run_notebook(tmp, upto=None, **over):
    ns = {"__name__": "nb"}
    for c in _nb().cells:
        if c.cell_type != "code":
            continue
        tag = c.metadata["tags"][0]
        exec(compile(c.source, f"<cell {tag}>", "exec"), ns)
        if tag == "config":
            ns.update({"PKG_DIR": str(PKG), "OUT_DIR": str(tmp / "out"), "OUTPUT_ZIP": str(tmp / "out.zip"),
                       "INPUT_ROOT": str(tmp / "no_input"), "TOKENIZER_DIR": TOK_DIR,
                       "GENERATOR_FACTORY": fake_factory(over.pop("oom_on", None)), **over})
        if tag == upto:
            break
    return ns


def test_notebook_refuses_tampered_package_and_wrong_corpus(tmp_path):
    """Mocked preflight (no tokenizer needed): the first three cells stop on a modified package or a swapped corpus."""
    ns = run_notebook(tmp_path / "ok", upto="package")
    assert ns["CORPUS_SHA"] == SHA and len(ns["CORPUS_ROWS"]) == 3445 and len(ns["EVAL"]) == 892
    pkg = tmp_path / "pkg"
    shutil.copytree(PKG, pkg, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    with open(pkg / "data/eval/eval_inputs.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(C.PreflightError, match="hash mismatch"):
        run_notebook(tmp_path / "t1", upto="package", PKG_DIR=str(pkg))
    # a different corpus whose manifest was re-written to look consistent is still refused (pinned sha256 in the code)
    pkg2 = tmp_path / "pkg2"
    shutil.copytree(PKG, pkg2, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    lines = (pkg2 / C.CORPUS_FILE).read_text(encoding="utf-8").splitlines()
    (pkg2 / C.CORPUS_FILE).write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8", newline="\n")
    man = json.loads((pkg2 / "package_manifest.json").read_text(encoding="utf-8"))
    man["files"][C.CORPUS_FILE]["sha256"] = C.sha256_file(pkg2 / C.CORPUS_FILE)
    (pkg2 / "package_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(C.PreflightError, match="not the approved corpus"):
        run_notebook(tmp_path / "t2", upto="package", PKG_DIR=str(pkg2))


def test_notebook_refuses_attached_credential_files(tmp_path):
    inp = tmp_path / "inp/ds"
    inp.mkdir(parents=True)
    (inp / ".env").write_text("X=1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="credential files attached"):
        run_notebook(tmp_path, upto="env", INPUT_ROOT=str(tmp_path / "inp"))


needs_tok = pytest.mark.skipif(not TOK_DIR or importlib.util.find_spec("transformers") is None,
                               reason="set MQL0008_TOKENIZER_DIR and install transformers (tokenizer only)")


@needs_tok
def test_frozen_plan_equals_rederived_plan_with_the_pinned_tokenizer(cfg, docs, evals, einfo, prompts, plan):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    ct = lambda s: len(tok.encode(s, add_special_tokens=False))   # noqa: E731
    assert C.build_plan(cfg, B, evals, prompts, docs, einfo, ct) == plan


@needs_tok
def test_notebook_end_to_end_with_fake_generator_both_arms(tmp_path):
    ns = run_notebook(tmp_path, oom_on="spider_dev_0003")
    st = json.loads((tmp_path / "out/MQL0008_STATUS.json").read_text(encoding="utf-8"))
    assert st["status"] == "complete" and st["problems"] == [] and st["all_arms_in_this_output"]
    assert st["predictions_total"] == 1784 and st["run_arms"] == [A, B]
    assert st["retrieval_corpus"]["sha256"] == SHA and st["retrieval_corpus"]["records"] == 3445
    assert st["model"]["revision"] == "488639f1ff808d1d3d0ba301aef8c11461451ec5"
    ids = [e["id"] for e in rj("data/eval/eval_inputs.jsonl")]
    assert sorted(p.name for p in (tmp_path / "out/predictions").iterdir()) == [A, B]            # exactly two arms
    preds = {}
    for arm in (A, B):
        preds[arm] = C.read_jsonl(tmp_path / f"out/predictions/{arm}/predictions.jsonl")
        assert [p["id"] for p in preds[arm]] == ids and all(p["status"] == "ok" and p["arm"] == arm for p in preds[arm])
        assert len(C.read_jsonl(tmp_path / f"out/predictions/{arm}/retrieval_log.jsonl")) == 892
        rc = json.loads((tmp_path / f"out/predictions/{arm}/run_config.json").read_text(encoding="utf-8"))
        assert rc["model_revision"] == st["model"]["revision"] and rc["generated_this_session"] == 892
        assert rc["corpus_sha256"] == (SHA if arm == B else None)
    assert all(p["retrieval"] is None for p in preds[A])
    assert [p["prompt_sha256"] for p in preds[A]] == [p["prompt_sha256"] for p in rj("data/eval/prompts.jsonl")]
    assert {p["prompt_sha256"] for p in preds[B]}.isdisjoint({p["prompt_sha256"] for p in preds[A]})
    assert all(p["retrieval"]["target_db_excluded"] and len(p["retrieval"]["retrieved_ids"]) == 3 for p in preds[B])
    assert any(p.get("oom_recovery") for p in preds[A]) and ns["OOM_EVENTS"]["batch"] >= 1
    log = {r["id"]: r for r in C.read_jsonl(tmp_path / f"out/predictions/{B}/retrieval_log.jsonl")}
    assert log["spider_dev_0383"]["excluded_by_rule"][0]["corpus_id"] == "spider_train_00650"
    assert "spider_train_00650" not in log["spider_dev_0383"]["selected_ids"]
    assert (tmp_path / "out.zip").is_file()
    sums = dict(reversed(ln.partition("  ")[::2]) for ln in (tmp_path / "out/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())
    assert all(C.sha256_file(tmp_path / "out" / n) == h for n, h in sums.items()) and f"predictions/{B}/predictions.jsonl" in sums


@needs_tok
def test_notebook_kill_and_resume_equals_uninterrupted(tmp_path):
    run_notebook(tmp_path / "full", RUN_ARMS=[B])
    run_notebook(tmp_path / "kill", RUN_ARMS=[B], STOP_AFTER_BATCHES=5)                   # killed after 5 batches (40 items)
    part = json.loads((tmp_path / "kill/out/MQL0008_STATUS.json").read_text(encoding="utf-8"))
    assert part["status"] == "partial" and part["arms"][B]["n"] == 40
    run_notebook(tmp_path / "kill", RUN_ARMS=[B])                                         # same OUT_DIR: resumes from the checkpoint
    assert json.loads((tmp_path / "kill/out/MQL0008_STATUS.json").read_text(encoding="utf-8"))["status"] == "complete"
    ck = (tmp_path / f"kill/out/predictions/{B}/checkpoints/predictions.partial.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ck) == 892 == len({json.loads(l)["id"] for l in ck})                          # no duplicate, no missing
    pick = lambda d: [(p["id"], p["raw_text"], p["prompt_sha256"], p["retrieval"]) for p in C.read_jsonl(d / f"out/predictions/{B}/predictions.jsonl")]  # noqa: E731
    assert pick(tmp_path / "kill") == pick(tmp_path / "full")
    # RESUME_FROM: a new OUT_DIR seeded from the earlier output (hash-verified)
    run_notebook(tmp_path / "k2", RUN_ARMS=[A], STOP_AFTER_BATCHES=3)
    prev = tmp_path / "k2/out"
    ns = run_notebook(tmp_path / "k3", RUN_ARMS=[A], RESUME_FROM=str(prev))
    assert ns["RESUMED"]["seeded"] and len(C.read_jsonl(tmp_path / f"k3/out/predictions/{A}/predictions.jsonl")) == 892
    # a resume source whose files were altered after export is refused
    with open(prev / f"predictions/{A}/checkpoints/predictions.partial.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(C.PreflightError, match="hash check failed"):
        run_notebook(tmp_path / "k4", RUN_ARMS=[A], RESUME_FROM=str(prev))
    # resuming into a directory whose run used another configuration is refused
    rc = tmp_path / f"full/out/predictions/{B}/run_config.json"
    d = json.loads(rc.read_text(encoding="utf-8"))
    d["max_new_tokens"] = 512
    rc.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(C.PreflightError, match="refusing to mix"):
        run_notebook(tmp_path / "full", RUN_ARMS=[B])
