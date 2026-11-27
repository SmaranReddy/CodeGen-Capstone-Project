"""MQL-0008 scoring adapter and pre-registered analysis: input validation (accept / reject), two-arm scoring with a FAKE
read-only Atlas on synthetic predictions, resume, offline re-analysis, statistics against hand-computed values.
No model, no GPU, no network, no real Atlas, no credentials. Nothing here is an MQL-0008 result."""
import json
import random
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0008"))
import dev1034_core as C  # noqa: E402
import mql0008_scoring as SC  # noqa: E402
import mql0008_stats as ST  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_nl2mql_mql0007_scoring as T7  # noqa: E402  (reused fake Atlas: FakeDB, FakeClient, zip_dir)

PKG, BUNDLE = SC.PKG, SC.BUNDLE
A, B = ST.ARM_A, ST.ARM_B
pytestmark = pytest.mark.skipif(not PKG.exists() or not BUNDLE.exists(), reason="MQL-0008 package / evaluation bundle not available")
sys.path.insert(0, str(PKG / "src"))
import mql0008_core as K  # noqa: E402

PKG_MANIFEST_SHA = "fb20c496fa20fbc5eecc60b1407ac3858a6370cd607df257709b3cea4e26fed8"
ENV = {"torch": "2.11.0+cu130", "transformers": "5.20.2", "accelerate": "1.14.0", "tokenizers": "0.23.3", "gpu": "Tesla T4"}


# ------------------------------------------------------------------ frozen artifacts are untouched
def test_kaggle_package_is_unchanged_and_protected():
    man = K.verify_package(PKG)                                    # all 19 files equal their recorded sha256
    assert len(man["files"]) == 19 and C.sha256_file(PKG / "package_manifest.json") == PKG_MANIFEST_SHA == SC.EXPECTED_PACKAGE_MANIFEST_SHA256
    assert C.sha256_file(PKG / K.CORPUS_FILE) == SC.EXPECTED_CORPUS_SHA256 == "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf"
    for bad in (PKG / "x", ROOT / "outputs/nl2mql_mql0007_rag_v1/v9", ROOT / "outputs/nl2mql_spider_dev_1034_model_comparison/v9"):
        with pytest.raises(SC.ScoringError):
            SC.guard_dest(bad)


# ------------------------------------------------------------------ analysis plan and statistics
def test_analysis_plan_is_registered_and_has_one_primary_comparison():
    doc = ST.load_registered_plan()
    assert doc["sha256"] == ST.plan_sha() and doc["plan"]["primary_comparison"] == [B, A]
    p = doc["plan"]
    assert "no multiple-comparison correction" in p["multiplicity"] and p["bootstrap"]["resamples"] >= 10000
    assert p["bootstrap"]["seed"] == C.BOOTSTRAP_SEED and "892" in p["metric"] and "EXPLORATORY" in p["secondary_note"]
    assert set(p["outcome_categories"]) == {"correct", "incorrect", "invalid", "execution_error", "timeout", "generation_failure",
                                             "infrastructure_failure", "not_comparable"}


def test_registered_plan_is_never_overwritten(tmp_path, monkeypatch):
    f = ST.write_plan(tmp_path / "plan.json")
    first = f.read_bytes()
    assert ST.write_plan(f).read_bytes() == first                      # idempotent, timestamp kept
    monkeypatch.setitem(ST.ANALYSIS_PLAN, "primary_test", "something else")
    with pytest.raises(RuntimeError, match="never overwritten"):
        ST.write_plan(f)
    with pytest.raises(RuntimeError, match="differs"):
        ST.load_registered_plan(f)


def _outs(a, b):
    ids = [f"q{i:03d}" for i in range(len(a))]
    return ids, {A: {i: "correct" if x else "incorrect" for i, x in zip(ids, a)}, B: {i: "correct" if x else "incorrect" for i, x in zip(ids, b)}}


def test_mcnemar_and_paired_table_against_hand_values():
    a = [1] * 50 + [0] * 50
    b = list(a)
    for i in range(50, 60):
        b[i] = 1                       # RAG fixes 10 baseline errors
    b[0] = b[1] = 0                    # RAG introduces 2 errors
    ids, outs = _outs(a, b)
    r = ST.primary(outs, ids)
    assert r["paired_table"] == {"both_correct": 48, "rag_fixes_baseline_error": 10, "rag_introduces_error": 2, "both_wrong": 40}
    assert r["mcnemar_exact_p"] == pytest.approx(158 / 4096)            # 2 * (C(12,0) + C(12,1) + C(12,2)) / 2^12
    assert r["arms"][A]["correct"] == 50 and r["arms"][B]["correct"] == 58 and r["arms"][A]["wilson95"] == C.wilson(50, 100)
    assert r["difference_b_minus_a"] == {"correct": 8, "percentage_points": 8.0} and r["significant_at_0.05"] is True
    assert "better" in r["conclusion"] and "holm" not in json.dumps(r).lower()
    assert C.mcnemar_exact(0, 0) == 1.0 and C.mcnemar_exact(5, 5) == 1.0 and C.mcnemar_exact(2, 10) == C.mcnemar_exact(10, 2)


def test_higher_raw_accuracy_alone_is_not_called_better():
    a = [1] * 40 + [0] * 60
    b = list(a)
    for i in range(40, 46):
        b[i] = 1                       # 6 fixes
    for i in range(0, 4):
        b[i] = 0                       # 4 new errors: B is 2 items ahead, p = 0.754
    ids, outs = _outs(a, b)
    r = ST.primary(outs, ids)
    assert r["difference_b_minus_a"]["correct"] == 2 and r["mcnemar_exact_p"] == pytest.approx(772 / 1024)
    assert r["significant_at_0.05"] is False and r["conclusion"].startswith("no statistically significant difference")
    assert "better than" not in r["conclusion"]


def test_paired_bootstrap_is_deterministic_and_matches_an_independent_recomputation():
    rnd = random.Random(7)
    a = [int(rnd.random() < 0.45) for _ in range(200)]
    b = [int(x or rnd.random() < 0.1) if rnd.random() > 0.05 else 0 for x in a]
    ids, outs = _outs(a, b)
    r1, r2 = ST.primary(outs, ids), ST.primary(outs, ids)
    ci = r1["bootstrap95_difference_b_minus_a"]
    assert ci == r2["bootstrap95_difference_b_minus_a"] and r1["bootstrap"] == {"resamples": 10000, "seed": 20261029}
    d = [y - x for x, y in zip(a, b)]                                   # independent recomputation: resample items, pairs kept together
    rng = random.Random(20261029)
    stats = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(10000))
    assert ci == [round(stats[250], 4), round(stats[9749], 4)]
    point = (sum(b) - sum(a)) / len(a)
    assert ci[0] <= point <= ci[1] and ci[0] < ci[1]
    assert ST.primary(*reversed(_outs(a, a)))["bootstrap95_difference_b_minus_a"] == [0.0, 0.0]   # identical arms: no difference


def test_outcome_aggregation_transitions_and_strata():
    mk = lambda o, fs=None, d=None: {"outcome": o, "failure_source": fs, "detail": d}   # noqa: E731
    kinds = [mk("correct"), mk("incorrect", "model"), mk("invalid", "model"), mk("timeout", "model"), mk("failed", "model", "execution_error"),
             mk("failed", "infrastructure", "generation_error"), mk("failed", "infrastructure", "atlas_unavailable"), mk("not_comparable", "reference")]
    ids = [f"i{n}" for n in range(8)]
    ev = {A: [{"id": i, **k} for i, k in zip(ids, kinds)], B: [{"id": i, **k} for i, k in zip(ids, kinds[1:] + kinds[:1])]}
    meta = {i: {"hardness": ["easy", "hard"][n % 2], "db_id": "d", "reference_status": "ok" if n else "tie_at_limit"} for n, i in enumerate(ids)}
    plan = [{"id": i, "selected_ids": ["t1", "t2", "t3"], "selected_ranks": [1, 2, 3], "selected_scores": [3.0, 2.0, 1.0], "selected_db_ids": ["x"] * 3,
             "context_tokens": 100 + n, "excluded_by_rule": [], "dropped_budget": [], "empty_context": False} for n, i in enumerate(ids)]
    an, rows = ST.analyze(ev, ids, meta, {i: {"join"} for i in ids}, ["i3"], plan)
    sec = an["secondary_exploratory"]
    for arm in (A, B):
        assert sum(sec["outcome_categories"][arm].values()) == 8 and all(v == 1 for v in sec["outcome_categories"][arm].values())
    t = sec["category_transitions"]["rows_arm_a_columns_arm_b"]
    assert sum(sum(r.values()) for r in t.values()) == 8 and t["correct"]["incorrect"] == 1 and t["not_comparable"]["correct"] == 1
    assert sec["category_transitions"]["arm_a_error_categories_resolved_by_rag"] == {"not_comparable": 1}
    assert sec["category_transitions"]["error_categories_introduced_by_rag"] == {"incorrect": 1}
    assert an["primary_analysis"]["n"] == 8 == an["denominator"]         # not_comparable and failures stay in the denominator
    assert an["primary_analysis"]["paired_table"] == {"both_correct": 0, "rag_fixes_baseline_error": 1, "rag_introduces_error": 1, "both_wrong": 6}
    assert sum(v["n"] for v in sec["breakdowns"]["hardness"].values()) == 8 and sec["sensitivity"]["S2_without_near_duplicate_items"]["n"] == 7
    assert sec["sensitivity"]["S3_reference_status_ok"]["n"] == 7 and len(sec["changed_items"]) == 2 and len(rows) == 8
    assert {r["pair_class"] for r in rows} == {"rag_fixes_baseline_error", "rag_introduces_error", "both_wrong"}
    assert sec["changed_items"][0]["retrieved_scores"] == [3.0, 2.0, 1.0] and sec["label"].startswith("EXPLORATORY")
    with pytest.raises(ValueError):
        ST.analyze({A: ev[A][:-1], B: ev[B]}, ids, meta, {i: set() for i in ids}, [], plan)       # a missing record is never dropped silently


# ------------------------------------------------------------------ synthetic Kaggle export (as the notebook writes it)
@pytest.fixture(scope="module")
def exp():
    return SC.load_expected()


@pytest.fixture(scope="module")
def coll_of():
    out = {}
    for coll, m in sorted(C.load_mapping(BUNDLE)["collections"].items()):
        out.setdefault(m["db_id"], coll)
    return out


def kinds_fn(arm, iid):
    n = int(iid.rsplit("_", 1)[1])
    special = {"spider_dev_0010": "timeout", "spider_dev_0011": "error", "spider_dev_0012": "wrong", "spider_dev_0013": "out", "spider_dev_0014": "garbage"}
    if iid in special:
        return special[iid]
    return "gold" if n % (4 if arm == A else 3) else "wrong"


def _raw(kind, iid, coll):
    if kind == "out":
        return json.dumps({"collection": coll, "pipeline": [{"$match": {"__fx": f"gold:{iid}"}}, {"$out": "pwned"}]})
    if kind == "garbage":
        return "I cannot answer that."
    return json.dumps({"collection": coll, "pipeline": [{"$match": {"__fx": f"{kind}:{iid}"}}]})


def write_sums(root):
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (root / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(root / n)}  {n}\n" for n in names), encoding="utf-8")


def write_status(root, exp, arms):
    cfgx, M = exp["cfg"], exp["cfg"]["model"]
    st_arms = {a: {"status": "complete", "problems": [], "n": 892, "n_ok": 892,
                   "predictions_sha256": C.sha256_file(root / "predictions" / a / "predictions.jsonl")} for a in arms}
    C.write_json(root / SC.STATUS_FILE, {
        "experiment": "MQL-0008", "stage": "predictions", "status": "complete", "problems": [], "arms": st_arms, "run_arms": list(arms),
        "package_manifest_sha256": exp["package_manifest_sha256"], "config_sha256": exp["config_sha256"],
        "retrieval_corpus": {"name": "approved_corpus_v2_1", "records": 3445, "sha256": exp["corpus_sha256"]},
        "retrieval_config_fingerprint": exp["retrieval_fp"], "model": {"id": M["id"], "revision": M["revision"], "dtype": "bfloat16"},
        "evaluation_ids_sha256": cfgx["evaluation"]["ids_sha256"]})


def make_export(tmp, exp, coll_of, name="mql0008_outputs", arms=(A, B), kinds=kinds_fn, env=None, pre=None, post=None, cfg_edit=None):
    """A synthetic MQL-0008 export. `pre(root)` runs before the status and checksums are written (a consistent but wrong
    export); `post(root)` runs after them (tampering)."""
    root = Path(tmp) / name
    cfgx, M, env = exp["cfg"], exp["cfg"]["model"], env or ENV
    plan = {p["id"]: p for p in exp["plan"]}
    nd = {x["eval_id"] for x in cfgx["retrieval_exclusions"]}
    for arm in arms:
        d = root / "predictions" / arm
        d.mkdir(parents=True)
        rag = arm == B
        cfg = {"experiment": "MQL-0008", "arm": arm, "mode": "generated", "arm_config": cfgx["arms"][arm],
               "retrieval_exclusions": cfgx["retrieval_exclusions"] if rag else [], "model_label": M["label"], "model_id": M["id"],
               "model_revision": M["revision"], "runtime": M["runtime"], "decoding": M["decoding"], "max_new_tokens": M["max_new_tokens"],
               "batch_size": M["batch_size"], "seed": M["seed"], "corpus_sha256": exp["corpus_sha256"] if rag else None,
               "corpus_records": 3445 if rag else None, "retrieval_config_fingerprint": exp["retrieval_fp"] if rag else None,
               "evaluation_ids_sha256": cfgx["evaluation"]["ids_sha256"], "config_sha256": exp["config_sha256"],
               "package_manifest_sha256": exp["package_manifest_sha256"], "dtype": "bfloat16", "gpu": env["gpu"], "torch": env["torch"],
               "transformers": env["transformers"]}
        if cfg_edit:
            cfg_edit(arm, cfg)
        cfg["fingerprint"] = C.fingerprint({k: v for k, v in cfg.items() if k not in SC.POST_FP})
        cfg.update(generation_wall_s=1.0, resolved_revision=None, generated_this_session=892,
                   environment={"packages": {k: env[k] for k in ("torch", "transformers", "accelerate", "tokenizers")}})
        recs = [{"id": i, "db_id": exp["db_of"][i], "fingerprint": cfg["fingerprint"], "model": M["label"], "arm": arm, "status": "ok",
                 "raw_text": _raw(kinds(arm, i), i, coll_of[exp["db_of"][i]]), "error": None, "created_at": "2026-11-17T00:00:00+00:00",
                 "latency_s": 0.1, "input_tokens": 5, "output_tokens": 3, "finish_reason": "stop",
                 "prompt_sha256": exp["prompt_hashes"][arm][i], **exp["identity"](arm, plan[i])} for i in exp["ids"]]
        C.write_jsonl(d / "predictions.jsonl", recs)
        C.write_jsonl(d / "checkpoints/predictions.partial.jsonl", recs)
        C.write_json(d / "run_config.json", cfg)
        if rag:
            log = [dict(p, near_duplicate_template_item=p["id"] in nd) for p in exp["plan"]]
        else:
            log = [{"id": i, "db_id": exp["db_of"][i], "arm": arm, "selected_ids": [], "selected_scores": [], "selected_ranks": [],
                    "selected_db_ids": [], "ranked_top10": [], "excluded_by_rule": [], "dropped_budget": [], "context_tokens": 0,
                    "prompt_tokens_est": None, "empty_context": True, "prompt_sha256": exp["prompt_hashes"][A][i],
                    "near_duplicate_template_item": i in nd} for i in exp["ids"]]
        C.write_jsonl(d / "retrieval_log.jsonl", log)
    C.write_json(root / "preflight/preflight.json", {"ok": True})
    if pre:
        pre(root)
    write_status(root, exp, arms)
    write_sums(root)
    if post:
        post(root)
    return root


def edit_preds(arm, fn):
    def go(root):
        pf = root / "predictions" / arm / "predictions.jsonl"
        recs = C.read_jsonl(pf)
        out = fn(recs)
        C.write_jsonl(pf, recs if out is None else out)
    return go


def edit_status(fn):
    def go(root):
        f = root / SC.STATUS_FILE
        st = json.loads(f.read_text(encoding="utf-8"))
        fn(st)
        C.write_json(f, st)
        write_sums(root)
    return go


def verify(exp, *dirs, **kw):
    return SC.verify_predictions(list(dirs), exp, **kw)


# ------------------------------------------------------------------ validation: accept
def test_valid_two_arm_export_verifies(tmp_path, exp, coll_of):
    rep = verify(exp, make_export(tmp_path, exp, coll_of))
    assert rep["ok"], rep["problems"]
    assert set(rep["arm_source"]) == {A, B} and not rep["environment_drift"]
    assert rep["frozen"]["corpus_sha256"] == SC.EXPECTED_CORPUS_SHA256 and rep["frozen"]["corpus_records"] == 3445
    names = {c["check"] for c in rep["checks"]}
    for want in ("identical_evaluation_ids_in_both_arms", f"{A}:no_retrieval_context", f"{B}:retrieval_log_equals_frozen_hybrid_plan",
                 f"{B}:target_database_excluded_and_only_approved_corpus", f"{A}:no_protected_holdout_database",
                 "mql0008_outputs:prediction_hashes_match_export_manifest", "mql0008_outputs:approved_corpus_hash",
                 "mql0008_outputs:only_approved_files", f"{B}:model_identity"):
        assert want in names, want


def test_one_zip_per_arm_verifies_and_identical_duplicates_are_tolerated(tmp_path, exp, coll_of):
    a = make_export(tmp_path / "a", exp, coll_of, arms=(A,))
    b = make_export(tmp_path / "b", exp, coll_of, arms=(B,))
    assert verify(exp, a, b)["ok"]
    both = make_export(tmp_path / "c", exp, coll_of)
    assert verify(exp, a, both)["ok"]                                   # arm A twice, byte-identical
    other = make_export(tmp_path / "d", exp, coll_of, arms=(A,), kinds=lambda arm, i: "wrong")
    rep = verify(exp, other, both)
    assert not rep["ok"] and any("DIFFERENT predictions" in p for p in rep["problems"])


# ------------------------------------------------------------------ validation: reject
def _swap(recs):
    recs[3], recs[4] = recs[4], recs[3]


def _unknown_id(recs):
    recs[5]["id"] = "spider_dev_9999"


def _holdout_db(recs):
    recs[6]["db_id"] = "pets_1"


def _bad_retrieved(recs):
    recs[7]["retrieval"]["retrieved_ids"] = list(reversed(recs[7]["retrieval"]["retrieved_ids"]))


def _target_db(recs):
    recs[8]["retrieval"]["target_db_excluded"] = False


def _wrong_rev(recs):
    for r in recs:
        r["model_revision"] = "0" * 40


def _wrong_prompt(recs):
    recs[9]["prompt_sha256"] = "0" * 64


def _a_with_context(recs):
    recs[10]["retrieval"] = {"retrieved_ids": ["spider_train_00001"]}


def _error_record(recs):
    recs[11].update(status="error", raw_text=None, error="boom")


REJECT = [
    ("missing prediction", edit_preds(B, lambda r: r[:-1]), "missing ids"),
    ("duplicate id", edit_preds(A, lambda r: r + [r[0]]), "duplicate ids"),
    ("ids out of order", edit_preds(B, _swap), "ids out of order"),
    ("different evaluation set", edit_preds(B, _unknown_id), "unexpected ids"),
    ("protected holdout database", edit_preds(A, _holdout_db), "no_protected_holdout_database"),
    ("retrieval metadata", edit_preds(B, _bad_retrieved), "record_retrieval_metadata_equals_frozen_plan"),
    ("target database flag", edit_preds(B, _target_db), "record_retrieval_metadata_equals_frozen_plan"),
    ("model revision in records", edit_preds(A, _wrong_rev), "record_model_experiment_generation_identity"),
    ("other prompt", edit_preds(B, _wrong_prompt), "different prompt"),
    ("zero-shot with retrieval context", edit_preds(A, _a_with_context), "no_retrieval_context"),
    ("generation error record", edit_preds(B, _error_record), "generation error record"),
]


@pytest.mark.parametrize("name,pre,needle", REJECT, ids=[r[0] for r in REJECT])
def test_bad_predictions_are_rejected(tmp_path, exp, coll_of, name, pre, needle):
    rep = verify(exp, make_export(tmp_path, exp, coll_of, pre=pre))      # status and checksums are consistent with the bad file
    assert not rep["ok"] and any(needle in p for p in rep["problems"]), rep["problems"]


def test_tampered_prediction_file_is_rejected(tmp_path, exp, coll_of):
    def tamper(root):
        pf = root / "predictions" / B / "predictions.jsonl"
        pf.write_text(pf.read_text(encoding="utf-8").replace("wrong:", "gold:", 1), encoding="utf-8", newline="\n")
    rep = verify(exp, make_export(tmp_path, exp, coll_of, post=tamper))
    assert not rep["ok"] and any("prediction_hashes_match_export_manifest" in p for p in rep["problems"])
    assert any("status_hash_equals_file" in p for p in rep["problems"])
    # re-writing the checksum file does not help: the export status still records the original hash
    root = make_export(tmp_path / "t2", exp, coll_of, post=lambda r: (tamper(r), write_sums(r)))
    rep = verify(exp, root)
    assert not rep["ok"] and any("status_hash_equals_file" in p for p in rep["problems"])


def test_modified_corpus_hash_is_rejected(tmp_path, exp, coll_of):
    rep = verify(exp, make_export(tmp_path / "s", exp, coll_of, post=edit_status(lambda st: st["retrieval_corpus"].update(sha256="0" * 64))))
    assert not rep["ok"] and any("approved_corpus_hash" in p for p in rep["problems"])
    rep = verify(exp, make_export(tmp_path / "n", exp, coll_of, post=edit_status(lambda st: st["retrieval_corpus"].update(records=3589))))
    assert not rep["ok"] and any("approved_corpus_hash" in p for p in rep["problems"])
    v1 = "61553d03fe4244d7d14a82c434482ef73f5e74b59f95e0cf983dd07806c7fc3d"       # corpus v1 must never pass as the approved corpus

    def cfg_edit(arm, cfg):
        if arm == B:
            cfg["corpus_sha256"] = v1
    rep = verify(exp, make_export(tmp_path / "c", exp, coll_of, cfg_edit=cfg_edit))
    assert not rep["ok"] and any(f"{B}:corpus_identity" in p for p in rep["problems"])
    # a package whose corpus (and manifest) were swapped is refused before anything else
    pkg = tmp_path / "pkg"
    shutil.copytree(PKG, pkg, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    lines = (pkg / K.CORPUS_FILE).read_text(encoding="utf-8").splitlines()
    (pkg / K.CORPUS_FILE).write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8", newline="\n")
    man = json.loads((pkg / "package_manifest.json").read_text(encoding="utf-8"))
    man["files"][K.CORPUS_FILE]["sha256"] = C.sha256_file(pkg / K.CORPUS_FILE)
    (pkg / "package_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(SC.ScoringError, match="frozen"):
        SC.load_expected(pkg)


def test_incorrect_model_revision_and_settings_are_rejected(tmp_path, exp, coll_of):
    rep = verify(exp, make_export(tmp_path / "r", exp, coll_of, cfg_edit=lambda arm, c: c.update(model_revision="f" * 40)))
    assert not rep["ok"] and any("model_identity" in p for p in rep["problems"])
    rep = verify(exp, make_export(tmp_path / "s", exp, coll_of, post=edit_status(lambda st: st["model"].update(revision="main"))))
    assert not rep["ok"] and any("model_and_revision" in p for p in rep["problems"])
    rep = verify(exp, make_export(tmp_path / "d", exp, coll_of, cfg_edit=lambda arm, c: c.update(max_new_tokens=512) if arm == B else None))
    assert not rep["ok"] and any(f"{B}:decoding" in p for p in rep["problems"])

    def resolved(root):
        f = root / "predictions" / A / "run_config.json"
        c = json.loads(f.read_text(encoding="utf-8"))
        c["resolved_revision"] = "a" * 40
        C.write_json(f, c)
    rep = verify(exp, make_export(tmp_path / "v", exp, coll_of, pre=resolved))
    assert not rep["ok"] and any("resolved_revision" in p for p in rep["problems"])


def test_wrong_experiment_missing_arm_unknown_arm_and_extra_files_are_rejected(tmp_path, exp, coll_of):
    rep = verify(exp, make_export(tmp_path / "e", exp, coll_of, post=edit_status(lambda st: st.update(experiment="MQL-0007"))))
    assert not rep["ok"] and any("experiment_id" in p for p in rep["problems"])
    rep = verify(exp, make_export(tmp_path / "m", exp, coll_of, arms=(A,)))
    assert not rep["ok"] and any(f"{B}:present" in p for p in rep["problems"])
    rep = verify(exp, make_export(tmp_path / "p", exp, coll_of, post=edit_status(lambda st: st.update(status="partial"))))
    assert not rep["ok"] and any("export_complete" in p for p in rep["problems"])

    def third_arm(root):
        shutil.copytree(root / "predictions" / B, root / "predictions/arm_c_random")
    rep = verify(exp, make_export(tmp_path / "x", exp, coll_of, pre=third_arm))
    assert not rep["ok"] and any("known_arm" in p for p in rep["problems"]) and any("only_approved_files" in p for p in rep["problems"])
    rep = verify(exp, make_export(tmp_path / "f", exp, coll_of, pre=lambda r: (r / "gold.jsonl").write_text("{}\n", encoding="utf-8")))
    assert not rep["ok"] and any("only_approved_files" in p for p in rep["problems"])


def test_retrieval_log_and_environment_are_enforced(tmp_path, exp, coll_of):
    def bad_log(root):
        f = root / "predictions" / B / "retrieval_log.jsonl"
        rows = C.read_jsonl(f)
        rows[0]["selected_scores"][0] += 0.5
        C.write_jsonl(f, rows)
    rep = verify(exp, make_export(tmp_path / "l", exp, coll_of, pre=bad_log))
    assert not rep["ok"] and any("retrieval_log_equals_frozen_hybrid_plan" in p for p in rep["problems"])
    a = make_export(tmp_path / "a", exp, coll_of, arms=(A,))
    b = make_export(tmp_path / "b", exp, coll_of, arms=(B,), env={**ENV, "transformers": "9.9.9"})
    rep = verify(exp, a, b)
    assert not rep["ok"] and rep["environment_drift"] and any("environment_identical" in p for p in rep["problems"])
    assert verify(exp, a, b, allow_env_drift=True)["ok"]                 # only with the explicit, recorded flag


# ------------------------------------------------------------------ scoring with a fake read-only Atlas
def tree(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()} if root.exists() else {}


def score(zpath, dest, db, run="t1"):
    return SC.cmd_score([str(zpath)], run, BUNDLE, PKG, dest, lambda: (T7.FakeClient(db), lambda s: s), log=lambda *_: None)


@pytest.fixture(scope="module")
def scored(tmp_path_factory, exp, coll_of):
    tmp = tmp_path_factory.mktemp("score8")
    zpath = T7.zip_dir(make_export(tmp, exp, coll_of), tmp / "export.zip")
    refs, mapping = C.load_reference(BUNDLE), C.load_mapping(BUNDLE)
    db = T7.FakeDB(mapping, refs)
    before = {"m7": tree(ROOT / "outputs/nl2mql_mql0007_rag_v1"), "pkg": tree(PKG)}
    st, an, run_dir = score(zpath, tmp / "dest", db)
    return {"st": st, "an": an, "run": run_dir, "db": db, "tmp": tmp, "zip": zpath, "before": before}


def test_two_arm_scoring_exact_892_accounting_and_outputs(scored, exp):
    st, run = scored["st"], scored["run"]
    assert st["experiment"] == "MQL-0008" and st["status"] == "complete" and st["n_expected"] == 892 and set(st["arms"]) == {A, B}
    for arm, acct in st["arms"].items():
        assert sum(acct.values()) == 892 and set(acct) == set(ST.CATEGORIES), arm
        ev = C.read_jsonl(run / "scoring/arms" / arm / "evaluation_results.jsonl")
        assert [e["id"] for e in ev] == exp["ids"]
        assert acct["timeout"] >= 1 and acct["execution_error"] >= 1 and acct["invalid"] >= 2 and acct["correct"] > 100
        by = {e["id"]: e for e in ev}
        assert by["spider_dev_0010"]["outcome"] == "timeout" and by["spider_dev_0011"]["detail"] == "execution_error"
        assert by["spider_dev_0013"]["outcome"] == "invalid" and by["spider_dev_0013"]["executed"] is False     # $out never reaches the server
    pa = scored["an"]["primary_analysis"]
    assert pa["n"] == 892 and pa["arms"][A]["total"] == pa["arms"][B]["total"] == 892
    assert sum(pa["paired_table"].values()) == 892 and pa["arms"][A]["correct"] == st["correct"][A] == st["arms"][A]["correct"]
    for f in ("input_validation_report.json", "run_log.jsonl", "scoring/RUN_MANIFEST.json", "scoring/preflight/atlas_check.json",
              "scoring/analysis.json", "scoring/paired_comparison.jsonl", "scoring/results_report.md", "scoring/SHA256SUMS.txt",
              f"scoring/arms/{A}/summary.json", f"scoring/arms/{B}/evaluation_results.jsonl", f"predictions/{B}/retrieval_log.jsonl"):
        assert (run / f).is_file(), f
    sums = dict(reversed(ln.partition("  ")[::2]) for ln in (run / "scoring/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())
    assert all(C.sha256_file(run / "scoring" / n) == h for n, h in sums.items()) and "analysis.json" in sums
    assert st["atlas"]["server_version"] == "8.0.fake" and st["atlas"]["read_only_verified"] is True
    assert st["atlas"]["authenticated_users"] and st["limits"] == {"exec_timeout_ms": C.EXEC_TIMEOUT_MS, "max_result_rows": C.MAX_RESULT_ROWS}
    assert st["analysis_plan_sha256"] == ST.plan_sha() and st["inputs"]["corpus_sha256"] == SC.EXPECTED_CORPUS_SHA256
    rows = C.read_jsonl(run / "scoring/paired_comparison.jsonl")
    assert [r["id"] for r in rows] == exp["ids"] and all(len(r["retrieved_ids"]) == 3 and len(r["retrieved_scores"]) == 3 for r in rows)
    rep = (run / "scoring/results_report.md").read_text(encoding="utf-8")
    assert rep.startswith("# MQL-0008 results") and "Historical context (not MQL-0008 results)" in rep and "EXPLORATORY" in rep
    assert "Corpus v1, 3,589 records" in rep and "McNemar" in rep and "Conclusion" in rep


def test_scoring_is_read_only_bounded_and_identical_for_both_arms(scored):
    db = scored["db"]
    assert {c[0] for c in db.calls} <= {"aggregate", "count_documents"}
    agg = [c for c in db.calls if c[0] == "aggregate"]
    assert agg and all(c[3] == C.EXEC_TIMEOUT_MS for c in agg)
    for banned in ("$out", "$merge", "$function", "$where", "$accumulator"):
        assert not any(banned in c[2] for c in agg), banned
    pat = re.compile(r"mongo" r"db\+srv|mongo" r"db://|MONGO" r"DB_URI|password\s*[=:]", re.I)
    for p in scored["run"].rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".md", ".txt"):
            assert not pat.search(p.read_text(encoding="utf-8", errors="ignore")), p.name


def test_scoring_touches_no_earlier_artifact(scored):
    assert tree(ROOT / "outputs/nl2mql_mql0007_rag_v1") == scored["before"]["m7"] and tree(PKG) == scored["before"]["pkg"]
    assert C.sha256_file(PKG / "package_manifest.json") == PKG_MANIFEST_SHA


def test_analysis_is_reproducible_offline_from_saved_results(scored):
    run = scored["run"]
    before = {n: (run / "scoring" / n).read_text(encoding="utf-8") for n in ("analysis.json", "paired_comparison.jsonl", "results_report.md")}
    calls = len(scored["db"].calls)
    st = SC.cmd_analyze(run)                                             # no client is created or used
    assert len(scored["db"].calls) == calls and st["correct"] == scored["st"]["correct"]
    after = {n: (run / "scoring" / n).read_text(encoding="utf-8") for n in before}
    assert after == before
    # a changed per-record result is detected through the saved prediction binding
    f = run / "scoring/arms" / A / "evaluation_results.jsonl"
    keep = f.read_bytes()
    rows = C.read_jsonl(f)
    rows[0]["prediction_sha256"] = "0" * 64
    C.write_jsonl(f, rows)
    with pytest.raises(SC.ScoringError, match="not bound"):
        SC.cmd_analyze(run)
    f.write_bytes(keep)


def test_resume_is_consistent_and_outages_are_reported_separately(tmp_path, exp, coll_of):
    target = exp["ids"][20]
    kinds = lambda arm, i: "infra" if (arm == B and i == target) else kinds_fn(arm, i)   # noqa: E731
    zpath = T7.zip_dir(make_export(tmp_path, exp, coll_of, kinds=kinds), tmp_path / "e.zip")
    db = T7.FakeDB(C.load_mapping(BUNDLE), C.load_reference(BUNDLE))
    st, _, run = score(zpath, tmp_path / "dest", db, "r1")
    assert st["status"] == "partial" and st["arms"][B]["infrastructure_failure"] == 1 and sum(st["arms"][B].values()) == 892
    assert "Incomplete" in (run / "scoring/results_report.md").read_text(encoding="utf-8")
    n1 = len([c for c in db.calls if c[0] == "aggregate"])
    st2, _, run2 = score(zpath, tmp_path / "dest", db, "r1")             # same run name: resumes
    assert run2 == run and st2["status"] == "complete" and st2["arms"][B]["infrastructure_failure"] == 0
    assert len([c for c in db.calls if c[0] == "aggregate"]) - n1 == 1    # only the unresolved item was executed again
    for arm in (A, B):
        ev = C.read_jsonl(run / "scoring/arms" / arm / "evaluation_results.jsonl")
        assert [e["id"] for e in ev] == exp["ids"]                        # no duplicate, no gap
    assert st2["arms"][A] == st["arms"][A]                                # finished records were not re-scored or changed
    log = [json.loads(x) for x in (run / "run_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in log].count("scoring_session_started") == 2 and log[-1] == {**log[-1], "event": "scoring_session_finished", "status": "complete"}
    second = [e for e in log if e["event"] == "arm_scored"][2:]
    assert [(e["arm"], e["evaluated_this_session"], e["final_before_this_session"]) for e in second] == [(A, 0, 892), (B, 1, 891)]
    # different predictions can never be mixed into an existing run
    other = T7.zip_dir(make_export(tmp_path / "o", exp, coll_of, kinds=lambda arm, i: "wrong"), tmp_path / "o.zip")
    with pytest.raises(SC.ScoringError, match="DIFFERENT predictions"):
        score(other, tmp_path / "dest", db, "r1")


def test_write_privileges_or_bad_inputs_stop_before_any_execution(tmp_path, exp, coll_of):
    zpath = T7.zip_dir(make_export(tmp_path, exp, coll_of), tmp_path / "e.zip")
    db = T7.FakeDB(C.load_mapping(BUNDLE), C.load_reference(BUNDLE), actions=("find", "listCollections", "killCursors", "insert", "update"))
    with pytest.raises(C.PreflightError, match="NOT read-only"):
        score(zpath, tmp_path / "dest", db, "w1")
    assert not [c for c in db.calls if c[0] == "aggregate"]
    bad = T7.zip_dir(make_export(tmp_path / "b", exp, coll_of, pre=edit_preds(B, lambda r: r[:-1])), tmp_path / "bad.zip")
    db2 = T7.FakeDB(C.load_mapping(BUNDLE), C.load_reference(BUNDLE))
    with pytest.raises(SC.ScoringError, match="failed validation"):
        score(bad, tmp_path / "dest", db2, "b1")
    assert db2.calls == []                                                # Atlas was never contacted
    rep = json.loads((tmp_path / "dest/scoring_runs/b1/input_validation_report.json").read_text(encoding="utf-8"))
    assert rep["ok"] is False and not (tmp_path / "dest/scoring_runs/b1/scoring").exists()


def test_default_run_name_is_timestamped():
    assert re.fullmatch(r"run_\d{8}T\d{6}Z", SC.default_run_name())
