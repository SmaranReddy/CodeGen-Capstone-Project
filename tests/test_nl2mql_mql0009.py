"""MQL-0009 retrieval-control experiment: builder, harness, Kaggle notebook (mocked dry run), export/import and the
Windows scoring adapter (fake read-only Atlas). No model, no GPU, no MongoDB, no gold-SQL execution.

The end-to-end dry run needs the pinned Qwen tokenizer (the frozen plans were built with it). It is borrowed from a
separate Python that has `transformers` through a line-protocol subprocess; set MQL0009_TOKENIZER_PY /
MQL0009_TOKENIZER_DIR (defaults: ~/codegen_data/venv_tok, ~/codegen_data/qwen3b_tok_488639f1). Those tests are skipped
when either is missing.
"""
import copy
import hashlib
import inspect
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "kaggle/nl2mql_mql0009_retrieval_control"
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0009"))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
sys.path.insert(0, str(ROOT / "tests"))
import mql0009_core as C            # noqa: E402
import mql0008_core as M8           # noqa: E402
import dev1034_core as D            # noqa: E402
import build_mql0009_package as B   # noqa: E402
import mql0009_scoring as SC        # noqa: E402
import mql0009_stats as ST          # noqa: E402

TOK_PY = Path(os.environ.get("MQL0009_TOKENIZER_PY", Path.home() / "codegen_data/venv_tok/Scripts/python.exe"))
TOK_DIR = Path(os.environ.get("MQL0009_TOKENIZER_DIR", Path.home() / "codegen_data/qwen3b_tok_488639f1"))
HAVE_TOK = TOK_PY.exists() and TOK_DIR.exists()
stub_tokens = lambda s: len(s) // 4 + 1          # noqa: E731


# ------------------------------------------------------------------ synthetic fixtures
def _row(i, db, q, sql="SELECT a FROM t", mql=None):
    return {"id": f"spider_train_{i:05d}", "db_id": db, "question": q, "question_raw": q, "sql": sql,
            "mql": mql or {"collection": f"{db}__t", "pipeline": [{"$project": {"_id": 0, "a": 1, "n": i}}]},
            "verification": {"outcome": "ACCEPTED", "verifier": "sql2mql-v2-verifier-3.1"}, "converter_version": "sql2mql-v2.1",
            "source": {"file": "x", "origin": {"file": "train_spider.json", "index": i}}}


def _tables(db):
    return {"db_id": db, "table_names_original": ["t"], "column_names_original": [[-1, "*"], [0, "a"], [0, "b"]]}


def _mini():
    """6 corpus DBs x 5 rows, 1 eval DB with 4 questions, frozen-prompt stand-ins."""
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]
    corpus = [_row(10 * d + j, f"db{d}", f"How many {words[d]} items with value {j} exist in group {d}?",
                   sql=f"SELECT count(*) FROM t WHERE b = {j}") for d in range(6) for j in range(5)]
    evals = [{"id": f"spider_train_9{j:04d}", "db_id": "evaldb", "question": q} for j, q in enumerate(
        ["How many alpha items with value 1 exist in group 0?", "List all names sorted by age.",
         "What is the average price of each kind?", "Show the oldest person."])]
    prompts = [{"id": e["id"], "db_id": e["db_id"], "prompt_version": "nl-mql-spider1-v1", "template_sha256": "x",
                "system": "SYS", "messages": [{"role": "user", "content": f"Database: evaldb\n\nSchema...\n\nQuestion: {e['question']}\n\nReturn the JSON object."}]}
               for e in evals]
    for p in prompts:
        p["prompt_sha256"] = M8.prompt_sha(p["system"], p["messages"])
    tables = {f"db{d}": _tables(f"db{d}") for d in range(6)}
    atlas = {"evaldb__t": {"db_id": "evaldb", "table": "t", "fields": {"a": ["int"], "b": ["string"]}, "declared": {"a": "INT", "b": "TEXT"}}}
    return corpus, evals, prompts, tables, atlas


def _cfg(fixed_ids, exclusions=(), seeds=(1, 2, 3, 4, 5), max_ctx=1500):
    runs = {"A_zero_shot": {"family": "zero_shot", "k": 0}}
    for i, s in enumerate(seeds, 1):
        runs[f"B_random_s{i}"] = {"family": "random", "k": 3, "seed": s}
    runs["C_fixed"] = {"family": "fixed", "k": 3}
    runs["D_hybrid"] = {"family": "hybrid", "k": 3}
    return {"arm_runs": runs, "random_retrieval": {"seeds": list(seeds)}, "fixed_retrieval": {"ids": list(fixed_ids)},
            "retrieval": {"bm25_k1": 1.5, "bm25_b": 0.75, "hybrid_weight": 1.0},
            "context": {"k": 3, "max_context_tokens": max_ctx, "margin_tokens": 64, "n_ctx": 8192, "max_new_tokens": 1024},
            "retrieval_exclusions": [dict(x) for x in exclusions], "leakage_gates": {"near_duplicate_token_jaccard": 0.8},
            "corpus": {"source_sha256": "c", "eligible_view": {"ids_sha256": "v"}}, "model": {"id": C.MODEL_ID, "revision": C.MODEL_REVISION}}


def _plans(cfg, corpus, evals, prompts, tables, atlas, count=stub_tokens):
    docs = M8.build_docs(corpus, tables)
    infos = M8.eval_schema_infos(atlas, ["evaldb"])
    return {r["name"]: C.build_plan(cfg, r, evals, prompts, docs, infos, count) for r in C.arm_runs(cfg)}, {d["doc_id"]: d for d in docs}


# ------------------------------------------------------------------ split + leakage
def test_eval_split_is_database_disjoint_and_registered():
    corpus = B.load_corpus()
    recs, funnel = B.build_eval_set()
    assert len(recs) == 639 and funnel["spider_pool_v3_val_internal"] == 639
    assert len({r["db_id"] for r in recs}) == 16
    assert not {r["db_id"] for r in recs} & {r["db_id"] for r in corpus}
    prot = B.protected_checks(recs, corpus)
    assert not prot["eval_db_overlap_with_mql0008_eval"] and not prot["eval_db_overlap_with_spider_dev"]
    reg = json.loads((ROOT / "outputs/nl2mql_mql0009_retrieval_control/preflight/eval_split_registration.json").read_text(encoding="utf-8"))
    assert reg["ids"] == [r["id"] for r in recs] and reg["ids_sha256"] == C.sha256_text("\n".join(reg["ids"]))


def test_leakage_gates_stop_exclude_and_report_only():
    corpus, evals, _, _, _ = _mini()
    recs = [dict(e, sql="SELECT count(*) FROM t WHERE b = 1", source={"file": "train_spider.json", "index": 90000 + k})
            for k, e in enumerate(evals)]
    local, excl, pkg = B.leakage_audit(corpus, recs)
    assert {(x["eval_id"], x["corpus_id"]) for x in excl} >= {("spider_train_90000", "spider_train_00001")}   # exact question
    assert local["L6_normalised_sql_equal"]["items"] == 4 and local["L6_normalised_sql_equal"]["action"] == "report only"
    assert all(x["corpus_id"] == "spider_train_00001" or "L5" in " ".join(x["reason"]) for x in excl)   # SQL never excludes
    assert "L6_normalised_sql_equal" not in pkg and "sql" not in json.dumps(pkg["exclusions"])
    with pytest.raises(C.PreflightError):                                     # database overlap -> STOP
        B.leakage_audit(corpus + [_row(99, "evaldb", "anything")], recs)
    with pytest.raises(C.PreflightError):                                     # id overlap -> STOP
        B.leakage_audit(corpus + [dict(_row(1, "db9", "x"), id=recs[0]["id"])], recs)


def test_real_leakage_audit_results():
    lk = json.loads((ROOT / "outputs/nl2mql_mql0009_retrieval_control/preflight/leakage_audit.json").read_text(encoding="utf-8"))
    assert not lk["L1_db_overlap"] and not lk["L2_id_overlap"] and not lk["L2_source_index_overlap"]
    cfg = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    assert {(x["eval_id"], x["corpus_id"]) for x in lk["exclusions"]} == {(x["eval_id"], x["corpus_id"]) for x in cfg["retrieval_exclusions"]}


def test_sql_fingerprint_masks_schema_and_literals():
    assert B.sql_fingerprint("SELECT name FROM singer WHERE age > 30") == B.sql_fingerprint("select title from book where pages > 7")
    assert B.sql_fingerprint("SELECT a FROM t WHERE b = 'x'") != B.sql_fingerprint("SELECT a FROM t WHERE b > 'x'")


# ------------------------------------------------------------------ corpus view, selection, K, budget, prompts
def test_eligible_view_is_stable_and_never_mutates():
    corpus, *_ = _mini()
    before = copy.deepcopy(corpus)
    v1, d1 = C.eligible_view(corpus, ["db2"])
    v2, d2 = C.eligible_view(corpus, ["db2"])
    assert d1 == d2 and v1 == v2 and corpus == before
    assert all(r["db_id"] != "db2" for r in v1) and d1["removed"] == 5


def test_random_retrieval_reproducible_distinct_and_seed_dependent():
    corpus, evals, prompts, tables, atlas = _mini()
    cfg = _cfg(["spider_train_00000", "spider_train_00010", "spider_train_00020"])
    p1, _ = _plans(cfg, corpus, evals, prompts, tables, atlas)
    p2, _ = _plans(cfg, corpus, evals, prompts, tables, atlas)
    for i in range(1, 6):
        assert p1[f"B_random_s{i}"] == p2[f"B_random_s{i}"]
        assert all(len(set(r["selected_ids"])) == 3 for r in p1[f"B_random_s{i}"])
    sel = {i: [tuple(r["selected_ids"]) for r in p1[f"B_random_s{i}"]] for i in range(1, 6)}
    assert len({tuple(v) for v in sel.values()}) == 5


def test_random_selection_is_roughly_uniform():
    corpus, evals, prompts, tables, atlas = _mini()
    docs = M8.build_docs(corpus, tables)
    infos = M8.eval_schema_infos(atlas, ["evaldb"])
    counts = {}
    for s in range(400):
        run = {"name": "r", "family": "random", "seed": s, "k": 3}
        for r in C.build_plan(_cfg(["spider_train_00000"] * 3), run, evals[:1], prompts[:1], docs, infos, stub_tokens):
            for i in r["selected_ids"]:
                counts[i] = counts.get(i, 0) + 1
    assert len(counts) == 30 and max(counts.values()) < 3 * min(counts.values())


def test_fixed_arm_same_ids_every_question_and_selection_rule():
    corpus, evals, prompts, tables, atlas = _mini()
    recs = [dict(e, sql="", source={"file": "train_spider.json", "index": 1}) for e in evals]
    fixed, info = B.select_fixed(corpus, recs, seed=7)
    assert fixed == B.select_fixed(corpus, recs, seed=7)[0]
    assert len({next(r["db_id"] for r in corpus if r["id"] == i) for i in fixed}) == 3
    assert "spider_train_00001" not in fixed                     # its question equals an evaluation question (gate)
    cfg = _cfg(fixed)
    plans, by = _plans(cfg, corpus, evals, prompts, tables, atlas)
    assert all(r["selected_ids"] == fixed for r in plans["C_fixed"])
    C.plan_checks(cfg, C.arm_runs(cfg)[6], plans["C_fixed"], evals, by)


def test_hybrid_plan_equals_mql0008_build_plan_and_excludes_target_db():
    corpus, evals, prompts, tables, atlas = _mini()
    excl = [{"eval_id": evals[0]["id"], "corpus_id": "spider_train_00001"}]
    cfg = _cfg(["spider_train_00000", "spider_train_00010", "spider_train_00020"], excl)
    plans, _ = _plans(cfg, corpus, evals, prompts, tables, atlas)
    m8cfg = {"arms": {"arm_b_hybrid_rag": {"retrieval": "hybrid_masked", "k": 3}}, "retrieval": cfg["retrieval"],
             "context": cfg["context"], "retrieval_exclusions": excl}
    docs = M8.build_docs(corpus, tables)
    m8 = M8.build_plan(m8cfg, "arm_b_hybrid_rag", evals, prompts, docs, M8.eval_schema_infos(atlas, ["evaldb"]), stub_tokens, arm_label="D_hybrid")
    assert [{k: v for k, v in r.items() if k not in ("arm_family", "seed")} for r in plans["D_hybrid"]] == m8
    assert "spider_train_00001" not in plans["D_hybrid"][0]["selected_ids"]
    assert plans["D_hybrid"][0]["excluded_by_rule"][0]["corpus_id"] == "spider_train_00001"
    # a target-db document injected into the corpus is never selected (retriever filter), and the view removes it
    leak = _row(77, "evaldb", evals[1]["question"])
    p2, _ = _plans(cfg, corpus + [leak], evals, prompts, {**tables, "evaldb": _tables("evaldb")}, atlas)
    assert all(leak["id"] not in r["selected_ids"] for n in p2 for r in p2[n] if n != "C_fixed")
    assert leak not in C.eligible_view(corpus + [leak], ["evaldb"])[0]


def test_k3_enforced_and_budget_shortfall_stops():
    corpus, evals, prompts, tables, atlas = _mini()
    cfg = _cfg(["spider_train_00000", "spider_train_00010", "spider_train_00020"])
    bad = copy.deepcopy(cfg)
    bad["arm_runs"]["D_hybrid"]["k"] = 2
    with pytest.raises(C.PreflightError):
        C.check_arm_design(bad)
    bad = copy.deepcopy(cfg)
    bad["arm_runs"]["B_random_s2"]["seed"] = 1
    with pytest.raises(C.PreflightError):
        C.check_arm_design(bad)
    tight = _cfg(cfg["fixed_retrieval"]["ids"], max_ctx=60)          # budget allows < 3 examples
    plans, by = _plans(tight, corpus, evals, prompts, tables, atlas)
    assert any(len(r["selected_ids"]) < 3 for r in plans["D_hybrid"])
    assert any(r["dropped_budget"] for r in plans["D_hybrid"])
    with pytest.raises(C.PreflightError, match="K must be 3"):
        C.plan_checks(tight, C.arm_runs(tight)[7], plans["D_hybrid"], evals, by)


def test_prompt_equivalence_across_arms():
    corpus, evals, prompts, tables, atlas = _mini()
    cfg = _cfg(["spider_train_00000", "spider_train_00010", "spider_train_00020"])
    plans, by = _plans(cfg, corpus, evals, prompts, tables, atlas)
    built = {n: C.prompts_for_run(p, prompts, by) for n, p in plans.items()}
    C.check_prompt_equivalence(built, prompts)
    assert [p["prompt_sha256"] for p in built["A_zero_shot"]] == [p["prompt_sha256"] for p in prompts]
    broken = copy.deepcopy(built)
    broken["C_fixed"][0]["system"] = "OTHER"
    with pytest.raises(C.PreflightError):
        C.check_prompt_equivalence(broken, prompts)
    broken = copy.deepcopy(built)
    broken["D_hybrid"][1]["messages"][0]["content"] = broken["D_hybrid"][1]["messages"][0]["content"].replace("Schema", "Schemata")
    with pytest.raises(C.PreflightError):
        C.check_prompt_equivalence(broken, prompts)


def test_real_package_plans_k3_and_prompt_equivalence():
    P = C.load_package(PKG)
    docs = M8.build_docs(P["view_rows"], P["tables"])
    by = {d["doc_id"]: d for d in docs}
    built = {}
    for r in P["runs"]:
        plan = C.read_jsonl(PKG / f"data/plans/plan_{r['name']}.jsonl")
        C.plan_checks(P["cfg"], r, plan, P["evals"], by)
        built[r["name"]] = C.prompts_for_run(plan, P["prompts"], by)
    C.check_prompt_equivalence(built, P["prompts"])
    assert P["cfg"]["analysis_plan_sha256"] == ST.plan_sha()


def test_real_hybrid_plan_equals_mql0008_on_package_inputs():
    P = C.load_package(PKG)
    docs = M8.build_docs(P["view_rows"], P["tables"])
    infos = M8.eval_schema_infos(P["atlas"], P["eval_dbs"])
    run = next(r for r in P["runs"] if r["family"] == "hybrid")
    mine = C.build_plan(P["cfg"], run, P["evals"][:80], P["prompts"][:80], docs, infos, stub_tokens)
    m8cfg = {"arms": {"a": {"retrieval": "hybrid_masked", "k": 3}}, "retrieval": P["cfg"]["retrieval"], "context": P["cfg"]["context"],
             "retrieval_exclusions": P["cfg"]["retrieval_exclusions"]}
    m8 = M8.build_plan(m8cfg, "a", P["evals"][:80], P["prompts"][:80], docs, infos, stub_tokens, arm_label="D_hybrid")
    assert [{k: v for k, v in r.items() if k not in ("arm_family", "seed")} for r in mine] == m8


def test_shared_sources_byte_identical_and_parser_verbatim():
    for f in B.SHARED_SRC:
        assert C.sha256_file(PKG / "src" / f) == C.sha256_file(B.M8_PKG / "src" / f)
    assert C.sha256_file(PKG / "src/mql0009_core.py") == C.sha256_file(ROOT / "scripts/nl2mql_mql0009/pkg_src/mql0009_core.py")
    def body(f):                                                   # code lines without the def line and the docstring
        src = inspect.getsource(f).splitlines()[1:]
        if src and src[0].strip().startswith('"""'):
            end = next(i for i, ln in enumerate(src) if ln.rstrip().endswith('"""') and (i > 0 or ln.count('"""') == 2))
            src = src[end + 1:]
        return "\n".join(src)
    assert body(C.parse_output) == body(D.parse_output) and len(body(D.parse_output)) > 500


# ------------------------------------------------------------------ parsing, persistence, records
@pytest.mark.parametrize("text,status,want", [
    ('{"collection": "x__t", "pipeline": []}', "ok", "parsed"),
    ('```json\n{"collection": "x__t", "pipeline": [{"$match": {}}]}\n```', "ok", "parsed"),
    ('Here: {"collection": "x__t", "pipeline": [{"$limit": 1}]} done', "ok", "parsed"),
    ('{"pipeline": 3}', "ok", "parsed_bad_shape"),
    ("no json here", "ok", "unparseable"),
    (None, "error", "generation_failure")])
def test_normalize_prediction(text, status, want):
    out = C.normalize_prediction(text, status)
    assert out["parse_status"] == want
    if text and text.startswith('{"collection"'):
        assert out["normalized_mql"]["pipeline"] == [{"$match": {}}] and out["normalized_mql"]["method"] == "aggregate"


def test_safe_write_refuses_to_overwrite_completed_predictions(tmp_path):
    f = tmp_path / "p.jsonl"
    C.safe_write_predictions(f, [{"id": "a", "status": "ok", "raw_text": "1"}, {"id": "b", "status": "error", "raw_text": None}])
    C.safe_write_predictions(f, [{"id": "a", "status": "ok", "raw_text": "1"}, {"id": "b", "status": "ok", "raw_text": "2"}])
    with pytest.raises(C.PreflightError):
        C.safe_write_predictions(f, [{"id": "a", "status": "ok", "raw_text": "CHANGED"}, {"id": "b", "status": "ok", "raw_text": "2"}])
    with pytest.raises(C.PreflightError):
        C.safe_write_predictions(f, [{"id": "b", "status": "ok", "raw_text": "2"}])


def test_checkpoint_refuses_duplicates_and_foreign_fingerprints(tmp_path):
    ck = M8.Checkpoint(tmp_path / "c.jsonl", ["a", "b"], "fp1")
    ck.append([{"id": "a", "fingerprint": "fp1", "status": "ok"}])
    with pytest.raises(RuntimeError):
        ck.append([{"id": "a", "fingerprint": "fp1", "status": "ok"}])
    with pytest.raises(RuntimeError):
        M8.Checkpoint(tmp_path / "c.jsonl", ["a", "b"], "fp2")
    with open(tmp_path / "c.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"id": "b", "fingerp')                     # torn last line after a crash
    assert M8.Checkpoint(tmp_path / "c.jsonl", ["a", "b"], "fp1").todo() == ["b"]


def test_prediction_record_validation_forbids_gold():
    r = {k: None for k in C.REQUIRED_FIELDS}
    r.update(experiment="MQL-0009", arm="D_hybrid", id="spider_train_00001", prediction_id="MQL-0009:D_hybrid:spider_train_00001",
             model_id=C.MODEL_ID, model_revision=C.MODEL_REVISION, parse_status="parsed")
    assert C.validate_prediction_record(r, "D_hybrid") == []
    assert any("gold" in p for p in C.validate_prediction_record({**r, "sql": "SELECT 1"}, "D_hybrid"))
    assert C.validate_prediction_record({**r, "prediction_id": "x"}, "D_hybrid")


def test_precision_and_revision_checks(tmp_path):
    cfg = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    t4 = {"device0": "Tesla T4", "total_gib": 14.56, "bf16_supported_incl_emulation": True, "bf16_native": False, "compute_capability": "7.5"}
    assert C.check_precision(cfg, t4)["bf16_mode"].startswith("emulated")
    for bad in ({**t4, "device0": "Tesla P100-PCIE-16GB"}, {**t4, "bf16_supported_incl_emulation": False}, {**t4, "total_gib": 8}):
        with pytest.raises(C.PreflightError):
            C.check_precision(cfg, bad)
    md = tmp_path / ".cache/huggingface/download"
    md.mkdir(parents=True)
    (md / "model.safetensors.metadata").write_text(C.MODEL_REVISION + "\netag\n1\n", encoding="utf-8")
    assert C.check_snapshot_revision(tmp_path, C.MODEL_REVISION)["files_checked"] == 1
    (md / "config.json.metadata").write_text("deadbeef\n", encoding="utf-8")
    with pytest.raises(C.PreflightError):
        C.check_snapshot_revision(tmp_path, C.MODEL_REVISION)


def test_package_manifest_detects_tampering(tmp_path):
    cp = tmp_path / PKG.name
    shutil.copytree(PKG, cp, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    C.load_package(cp)
    f = cp / "data/plans/plan_C_fixed.jsonl"
    f.write_text(f.read_text(encoding="utf-8").replace("spider_train_02033", "spider_train_00000", 1), encoding="utf-8")
    with pytest.raises(C.PreflightError):
        C.load_package(cp)


def test_package_and_zip_hold_no_gold_and_no_secrets():
    recs, _ = B.build_eval_set()
    gold = {D._squash(r["sql"]) for r in recs}
    import re
    secret = re.compile(r"mongodb(\+srv)?://|MONGODB_URI\s*=|BEGIN (RSA|OPENSSH) PRIVATE KEY|hf_[A-Za-z0-9]{30,}")
    with zipfile.ZipFile(B.ZIP_OUT) as z:
        names = z.namelist()
        assert all(n.startswith(f"{C.PACKAGE}/") for n in names)
        assert not any(Path(n).name.startswith(".env") or n.endswith((".sqlite", ".pem", ".key")) for n in names)
        for n in names:
            txt = z.read(n).decode("utf-8")
            assert not secret.search(txt), n
            if n.endswith(("eval_inputs.jsonl", "prompts.jsonl", "leakage_report.json")) or "/plans/" in n:
                low = D._squash(txt)
                assert not any(g in low for g in gold if len(g) > 25), n
                assert '"sql"' not in txt and '"expected_output"' not in txt, n
    man = json.loads((PKG / "package_manifest.json").read_text(encoding="utf-8"))
    assert man["contains_gold_labels"] is False and man["contains_credentials"] is False
    sums = dict(reversed(l.split("  ", 1)) for l in (PKG / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())
    assert all(C.sha256_file(PKG / n) == h for n, h in sums.items())


# ------------------------------------------------------------------ stats
def test_stats_unit_is_question_and_seed_mean():
    ids = [f"q{i}" for i in range(40)]
    rnd = random.Random(3)
    ev = {}
    for r in ST.ARM_RUNS:
        p = 0.8 if r == ST.HYBRID else 0.5
        ev[r] = [{"id": i, "outcome": "correct" if rnd.random() < p else "incorrect"} for i in ids]
    meta = {i: {"hardness": "easy", "db_id": "d", "reference_status": "ok"} for i in ids}
    a = ST.analyze(ev, ids, meta, {i: {"join"} for i in ids})
    h1 = a["main"]["primary"]["H1_D_hybrid_vs_B_random_mean"]
    assert h1["n"] == 40                                          # not 5 x 40
    assert set(a["random_seed_variability"]["per_seed_percent"]) == set(ST.RANDOM)
    assert "holm_adjusted_p" in a["main"]["primary"]["H2_D_hybrid_vs_C_fixed"]
    with pytest.raises(ValueError):
        ST.analyze({k: v for k, v in ev.items() if k != "C_fixed"}, ids, meta, {i: set() for i in ids})


def test_analysis_plan_registered_and_frozen_in_config():
    doc = ST.load_registered_plan()
    cfg = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    assert doc["analysis_plan_sha256"] == cfg["analysis_plan_sha256"] == ST.plan_sha()


def test_scoring_readiness_reports_provisioning_blocker():
    r = SC.readiness(write=False)
    assert r["checks"]["package_pinned_and_verified"]["ok"] and r["checks"]["provisioning_contract"]["ok"]
    assert r["status"] == "BLOCKED" and any("DATA PROVISIONING" in b for b in r["blockers"])


# ------------------------------------------------------------------ mocked Kaggle dry run -> export -> Windows import -> fake scoring
class TokServer:
    """Pinned-tokenizer counts from a Python that has transformers (line protocol, memoised)."""
    CODE = ("import sys,json\nfrom transformers import AutoTokenizer\nt=AutoTokenizer.from_pretrained(sys.argv[1])\n"
            "for l in sys.stdin:\n    s=json.loads(l)\n    sys.stdout.write(str(len(t.encode(s,add_special_tokens=False)))+'\\n')\n"
            "    sys.stdout.flush()\n")

    def __init__(self):
        env = {**os.environ, "TRANSFORMERS_VERBOSITY": "error", "PYTHONIOENCODING": "utf-8"}
        self.p = subprocess.Popen([str(TOK_PY), "-c", self.CODE, str(TOK_DIR)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, encoding="utf-8", env=env)
        self.cache = {}

    def __call__(self, s):
        if s not in self.cache:
            self.p.stdin.write(json.dumps(s) + "\n")
            self.p.stdin.flush()
            self.cache[s] = int(self.p.stdout.readline())
        return self.cache[s]

    def close(self):
        self.p.kill()


class FakeOOM(Exception):
    pass


def fake_factory(calls):
    atlas = json.loads((PKG / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
    coll_of = {}
    for name, m in sorted(atlas.items()):
        coll_of.setdefault(m["db_id"], name)
    state = {"oom": True}

    def factory(cfg):
        def run_batch(batch):
            if state["oom"] and len(batch) > 1:
                state["oom"] = False
                raise FakeOOM("CUDA out of memory (fake)")
            out = []
            for p in batch:
                calls.append(p["prompt_sha256"])
                h = int(hashlib.sha256(p["prompt_sha256"].encode()).hexdigest(), 16)
                kind = "gold" if h % 2 == 0 else ("wrong" if h % 5 else "error")
                text = json.dumps({"collection": coll_of[p["db_id"]], "pipeline": [{"$match": {"__fx": f"{kind}:{p['id']}"}}]})
                if h % 97 == 0:
                    text = "not json"
                out.append((text, {"latency_s": 0.0, "input_tokens": len(p["messages"][0]["content"]) // 4, "output_tokens": 20,
                                   "finish_reason": "stop"}, None))
            return out
        info = {"dtype": "bfloat16", "gpu": "Tesla T4", "torch": "2.11.0+fake", "transformers": "5.20.2", "resolved_revision": None}
        return run_batch, info, (lambda e: isinstance(e, FakeOOM)), (lambda: None)
    return factory


FAKE_GPU = {"checked": True, "device0": "Tesla T4", "compute_capability": "7.5", "total_gib": 14.56,
            "bf16_supported_incl_emulation": True, "bf16_native": False, "cuda": "13.0", "torch": "2.11.0+fake"}


def run_notebook(overrides):
    nb = json.loads((PKG / f"{C.PACKAGE}.ipynb").read_text(encoding="utf-8"))
    ns = {"__name__": "__nb__"}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), f"<cell {cell['metadata']['tags'][0]}>", "exec"), ns)
        if cell["metadata"]["tags"][0] == "config":
            ns.update(overrides)
    return ns


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    if not HAVE_TOK:
        pytest.skip("pinned tokenizer / tokenizer Python not available")
    tok = TokServer()
    base = tmp_path_factory.mktemp("kaggle")
    (base / "input").mkdir()
    calls1, calls2 = [], []
    common = {"PKG_DIR": str(PKG), "INPUT_ROOT": str(base / "input"), "FAKE_GPU": FAKE_GPU, "COUNT_TOKENS": tok}
    try:
        ns1 = run_notebook({**common, "OUT_DIR": str(base / "s1/mql0009_outputs"), "OUTPUT_ZIP": str(base / "s1.zip"),
                            "RUN_ARMS": ["A_zero_shot", "D_hybrid"], "STOP_AFTER_BATCHES": 3,
                            "GENERATOR_FACTORY": fake_factory(calls1)})
        ns2 = run_notebook({**common, "OUT_DIR": str(base / "s2/mql0009_outputs"), "OUTPUT_ZIP": str(base / "s2.zip"),
                            "RUN_ARMS": [r["name"] for r in C.arm_runs(json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8")))],
                            "RESUME_FROM": str(base / "s1/mql0009_outputs"), "GENERATOR_FACTORY": fake_factory(calls2)})
    finally:
        tok.close()
    return {"base": base, "ns1": ns1, "ns2": ns2, "calls1": calls1, "calls2": calls2}


def test_dry_run_interrupt_and_resume_without_regeneration(dry_run):
    ns1, ns2 = dry_run["ns1"], dry_run["ns2"]
    assert ns1["ST"]["status"] == "partial" and ns1["STOPPED"]
    assert ns2["ST"]["status"] == "complete" and ns2["MAN"]["completed_predictions"] == ns2["MAN"]["expected_predictions"] == 639 * 8
    s1 = C.read_jsonl(dry_run["base"] / "s1/mql0009_outputs/predictions/A_zero_shot/predictions.jsonl")
    s2 = C.read_jsonl(dry_run["base"] / "s2/mql0009_outputs/predictions/A_zero_shot/predictions.jsonl")
    assert 0 < len(s1) < 639 and s2[:len(s1)] == s1                         # carried forward unchanged
    assert len(dry_run["calls1"]) == len(s1) and len(dry_run["calls2"]) == 639 * 8 - len(s1)    # nothing regenerated
    assert len(set(dry_run["calls1"]) & set(dry_run["calls2"])) == 0
    assert any(p.get("oom_recovery") for p in s1)                            # OOM batch-1 fallback recorded


def test_dry_run_records_provenance(dry_run):
    out = dry_run["base"] / "s2/mql0009_outputs"
    for n in ("B_random_s3", "C_fixed", "D_hybrid", "A_zero_shot"):
        recs = C.read_jsonl(out / "predictions" / n / "predictions.jsonl")
        plan = {p["id"]: p for p in C.read_jsonl(PKG / f"data/plans/plan_{n}.jsonl")}
        assert len({r["prediction_id"] for r in recs}) == 639
        for r in recs[:50]:
            assert C.validate_prediction_record(r, n) == []
            assert r["retrieval"]["retrieved_ids"] == plan[r["id"]]["selected_ids"]
            assert r["precision"] == {"dtype": "bfloat16", "bf16_native": False, "bf16_mode": "emulated (compute capability < 8.0)",
                                      "gpu": "Tesla T4", "compute_capability": "7.5"}
            assert r["generation_config"]["do_sample"] is False and r["generation_config"]["max_new_tokens"] == 1024
        rc = json.loads((out / "predictions" / n / "run_config.json").read_text(encoding="utf-8"))
        assert rc["plan_sha256"] == json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))["plans_sha256"][n]
    assert {p["parse_status"] for p in C.read_jsonl(out / "predictions/D_hybrid/predictions.jsonl")} <= {"parsed", "unparseable"}


def test_notebook_refuses_unregistered_gpu(dry_run, tmp_path):
    with pytest.raises(C.PreflightError):
        run_notebook({"PKG_DIR": str(PKG), "INPUT_ROOT": str(tmp_path), "FAKE_GPU": {**FAKE_GPU, "device0": "NVIDIA L4"},
                      "COUNT_TOKENS": stub_tokens, "OUT_DIR": str(tmp_path / "o"), "OUTPUT_ZIP": str(tmp_path / "o.zip"),
                      "GENERATOR_FACTORY": fake_factory([])})


def test_resume_in_other_library_environment_is_refused(dry_run):
    ns = dry_run["ns2"]
    run = next(r for r in ns["P"]["runs"] if r["name"] == "C_fixed")
    info = {**ns["INFO"], "torch": "2.99.0+other"}
    with pytest.raises(C.PreflightError, match="earlier run_config differs"):
        C.generate_run(ns["P"], run, ns["PLANS"]["C_fixed"], ns["ARM_PROMPTS"]["C_fixed"], ns["GEN"], info, ns["PRECISION"],
                       ns["GPU"], ns["OUT_DIR"], {}, log=lambda *_: None)


def test_export_import_verify_and_tamper_detection(dry_run, tmp_path):
    rep, _ = SC.cmd_verify([str(dry_run["base"] / "s2.zip")], dest_root=tmp_path / "dest")
    assert rep["ok"], rep["problems"][:5]
    assert set(rep["predictions_sha256"]) == set(ST.ARM_RUNS)
    rep1, _ = SC.cmd_verify([str(dry_run["base"] / "s1.zip")], dest_root=tmp_path / "dest")
    assert not rep1["ok"] and any("missing" in p for p in rep1["problems"])
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(dry_run["base"] / "s2.zip") as zin, zipfile.ZipFile(bad, "w") as zout:
        for it in zin.infolist():
            data = zin.read(it.filename)
            if it.filename.endswith("C_fixed/predictions.jsonl"):
                data = data.replace(b'"spider_train_', b'"spider_trainX', 1)
            zout.writestr(it, data)
    assert not SC.cmd_verify([str(bad)], dest_root=tmp_path / "dest")[0]["ok"]


class ParityColl:
    def __init__(self, db, name, m):
        self.db, self.name, self.m = db, name, m
        self._count = m["documents"]

    def count_documents(self, q):
        return self._count

    def find(self, q, proj, batch_size=None):
        path = SC.DB_DIR / self.m["db_id"] / f"{self.m['db_id']}.sqlite"
        con = D._ro_connect(path, 60)
        try:
            cols = self.m["fields"]
            rows = con.execute(f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} FROM "{self.m["table"]}"').fetchall()
        finally:
            con.close()
        import test_nl2mql_mql0007_scoring as T7
        return T7.FakeCursor([dict(zip(cols, r)) for r in rows])

    def aggregate(self, pipeline, **kw):
        assert kw.get("maxTimeMS", 0) > 0
        return self.db.behave(self.name, pipeline)


def test_fake_atlas_scoring_end_to_end(dry_run, tmp_path):
    import test_nl2mql_mql0007_scoring as T7

    def refs(recs, K, db_dir):
        return [{"id": r["id"], "db_id": r["db_id"], "hardness": r["hardness"], "gold_columns": ["c"], "gold_rows": [[1]], "n_rows": 1,
                 "order_by": False, "tie_blocks": None, "tie_status": "no_order", "limit": None, "reference_status": "ok", "reason": None}
                for r in recs]

    def factory():
        mapping = SC.load_contract("codegen_spider1_heldout_test")
        db = T7.FakeDB(mapping, refs(SC.gold_records(C.load_package(PKG)["ids"]), None, None))
        db.colls = {n: ParityColl(db, n, m) for n, m in mapping["collections"].items()}
        return T7.FakeClient(db), (lambda s: s)

    st, analysis, run_dir = SC.cmd_score([str(dry_run["base"] / "s2.zip")], "t1", "codegen_spider1_heldout_test", dest_root=tmp_path / "dest",
                                         client_factory=factory, log=lambda *_: None, compute_reference=refs)
    assert st["status"] == "complete" and st["atlas"]["full_parity"] and st["atlas"]["read_only_verified"]
    assert set(st["arms"]) == set(ST.ARM_RUNS) and all(sum(a.values()) == 639 for a in st["arms"].values())
    pa = analysis["main"]["primary"]
    assert pa["H1_D_hybrid_vs_B_random_mean"]["n"] == 639 and "holm_adjusted_p" in pa["H2_D_hybrid_vs_C_fixed"]
    paired = C.read_jsonl(run_dir / "scoring/paired_per_question.jsonl")
    assert len(paired) == 639 and set(ST.ARM_RUNS) <= set(paired[0])
    acct = st["arms"]["A_zero_shot"]
    assert acct["correct"] > 0 and acct["invalid"] > 0 and acct["incorrect"] > 0
    assert (run_dir / "scoring/results_report.md").exists()
    again = SC.cmd_analyze(run_dir)                                    # offline recompute equals the saved analysis
    assert again["main"] == analysis["main"]


def test_fake_atlas_parity_mismatch_fails_closed(dry_run, tmp_path):
    import test_nl2mql_mql0007_scoring as T7
    mapping = SC.load_contract("x_db")
    mapping = copy.deepcopy(mapping)
    first = sorted(mapping["collections"])[0]
    mapping["collections"][first]["row_multiset_sha256"] = "0" * 64
    db = T7.FakeDB(mapping, [])
    db.colls = {n: ParityColl(db, n, m) for n, m in mapping["collections"].items()}
    with pytest.raises(D.PreflightError, match="digest mismatch"):
        D.atlas_preflight(T7.FakeClient(db), mapping, full_parity=True, log=None)


def test_offline_context_equals_the_atlas_derived_context_on_dev_dbs():
    """Ground truth for derive_context: data/interim/spider1/{atlas_collections,prompt_samples}.json were produced from LIVE
    Atlas by import_atlas.py / prompt_spider1.build_samples for the Spider-dev DBs. The offline derivation (SQLite only, no
    questions, no SQL) must reproduce them exactly. wta_1 (2 known invalid-UTF-8 strings) and the 3 held-out test DBs are
    left out."""
    man = json.loads((ROOT / "data/interim/spider1/atlas_collections.json").read_text(encoding="utf-8"))
    smp = json.loads((ROOT / "data/interim/spider1/prompt_samples.json").read_text(encoding="utf-8"))
    dbs = sorted({v["db_id"] for v in man.values()} - {"wta_1", "employee_hire_evaluation", "pets_1", "tvshow"})
    atlas, samples, _ = B.derive_context(dbs)
    rt = lambda o: json.loads(json.dumps(o, default=str))          # noqa: E731
    exp = {k: v for k, v in man.items() if v["db_id"] in dbs}
    assert set(atlas) == set(exp) and len(exp) == 67
    for k, e in exp.items():
        for f in ("db_id", "table", "documents", "fields", "declared"):
            assert rt(atlas[k][f]) == rt(e[f]), (k, f)
        assert list(atlas[k]["fields"]) == list(e["fields"])
        assert rt(samples[k]) == rt(smp[k]) and [list(d) for d in samples[k]] == [list(d) for d in smp[k]], k
