"""MQL-0007 scoring infrastructure: statistics (hand-computed values), exact 892-id accounting, three-arm scoring with a fake
read-only Atlas on synthetic predictions, rejection of missing/duplicate/unexpected/malformed predictions, environment
consistency across arms, arm A compatibility, near-duplicate exclusion, retrieval determinism, read-only safeguards, and
isolation from the MQL-0006 results. No model, no GPU, no network, no real Atlas, no credentials."""
import copy
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
import dev1034_core as C  # noqa: E402
import mql0007_preflight as PF  # noqa: E402
import mql0007_scoring as SC  # noqa: E402
import mql0007_stats as ST  # noqa: E402

PKG = ROOT / "kaggle/nl2mql_mql0007_rag_v1"
BUNDLE = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2"
MQL6 = ROOT / "outputs/nl2mql_spider_dev_1034_model_comparison"
needs_bundle = pytest.mark.skipif(not BUNDLE.exists() or not PKG.exists(), reason="bundle/package not built")
sys.path.insert(0, str(PKG / "src"))
import mql0007_core as K  # noqa: E402


# ------------------------------------------------------------------ statistics (hand-computed)
def test_mcnemar_wilson_holm_against_hand_values():
    assert ST.C.mcnemar_exact(10, 2) == pytest.approx(158 / 4096)            # 2 * (1 + 12 + 66) / 4096 = 0.03857
    assert ST.C.mcnemar_exact(2, 10) == ST.C.mcnemar_exact(10, 2) and ST.C.mcnemar_exact(0, 0) == 1.0
    assert ST.C.mcnemar_exact(5, 5) == 1.0
    assert ST.holm({"a": 0.01, "b": 0.04}) == {"a": pytest.approx(0.02), "b": pytest.approx(0.04)}
    h = ST.holm({"a": 0.04, "b": 0.03})                                        # sorted: b(0.03)*2=0.06, a: max(0.06, 0.04*1)
    assert h == {"b": pytest.approx(0.06), "a": pytest.approx(0.06)}
    assert ST.holm({"a": 0.9, "b": 0.8})["b"] == 1.0 and all(0 <= v <= 1 for v in ST.holm({"a": 0.0, "b": 1.0}).values())
    for p in ({"x": 0.001, "y": 0.2}, {"x": 0.03, "y": 0.03}):                  # adjusted >= raw; monotone in raw order
        adj = ST.holm(p)
        assert all(adj[k] >= p[k] for k in p)
        o = sorted(p, key=lambda k: p[k])
        assert adj[o[0]] <= adj[o[1]]
    assert ST.C.wilson(0, 10)[0] == 0.0 and ST.C.wilson(10, 10)[1] == 1.0 and ST.C.wilson(5, 10) == [0.2366, 0.7634]
    assert ST.C.wilson(0, 0) is None
    assert PF.statistics_validation()["all_ok"]


def _outcomes(n, pa, pb, pc):
    ids = [f"q{i:03d}" for i in range(n)]
    return ids, {"A_zero_shot": {i: ("correct" if x else "incorrect") for i, x in zip(ids, pa)},
                 "B_random_k3": {i: ("correct" if x else "incorrect") for i, x in zip(ids, pb)},
                 "C_hybrid_k3": {i: ("correct" if x else "incorrect") for i, x in zip(ids, pc)}}


def test_paired_comparison_three_arms_and_holm():
    # C beats A on 10 items, loses on 2; C vs B: 4 vs 4 (n=100)
    a = [1] * 50 + [0] * 50
    c = list(a)
    for i in range(50, 60):
        c[i] = 1                      # 10 items only C correct
    for i in range(0, 2):
        c[i] = 0                      # 2 items only A correct
    b = list(c)
    for i in range(60, 64):
        b[i] = 1                      # 4 items only B correct
    for i in range(70, 74):
        c[i] = 1                      # 4 items only C correct vs B
    ids, out = _outcomes(100, a, b, c)
    r = ST.compare_arms(out, ids)
    ca = r["primary"]["C_hybrid_k3_vs_A_zero_shot"]
    assert (ca["a_only"], ca["b_only"]) == (14, 2) and ca["mcnemar_exact_p"] == pytest.approx(ST.C.mcnemar_exact(14, 2))
    cb = r["primary"]["C_hybrid_k3_vs_B_random_k3"]
    assert cb["a_only"] + cb["b_only"] == cb["both_correct"] * 0 + cb["a_only"] + cb["b_only"]
    adj = ST.holm({k: v["mcnemar_exact_p"] for k, v in r["primary"].items()})
    assert all(r["primary"][k]["holm_adjusted_p"] == pytest.approx(adj[k]) for k in adj)
    assert "B_random_k3_vs_A_zero_shot" in r["secondary"] and "holm_adjusted_p" not in r["secondary"]["B_random_k3_vs_A_zero_shot"]
    assert r["arms"]["A_zero_shot"]["correct"] == 50 and r["arms"]["A_zero_shot"]["wilson95"] == ST.C.wilson(50, 100)
    assert len(r["primary"]) == 2                                                # exactly the two pre-registered comparisons


def test_analysis_breakdowns_and_sensitivity_are_consistent():
    ids = [f"q{i:03d}" for i in range(20)]
    outs = {a: {i: ("correct" if (k + n) % 3 else "incorrect") for n, i in enumerate(ids)} for k, a in enumerate(ST.ARMS)}
    meta = {i: {"hardness": ["easy", "hard"][n % 2], "db_id": ["d1", "d2"][n % 2], "reference_status": "ok" if n % 5 else "tie_at_limit"}
            for n, i in enumerate(ids)}
    pat = {i: ({"join"} if n % 2 else set()) for n, i in enumerate(ids)}
    nd = ["q003", "q007"]
    supp = {"q003": "correct" if outs["C_hybrid_k3"]["q003"] == "incorrect" else "incorrect"}
    an = ST.analyze(outs, ids, meta, pat, nd, supp)
    pa = an["primary_analysis"]
    assert pa["n"] == 20 and sum(v["n"] for v in an["breakdowns"]["hardness"].values()) == 20
    assert sum(v["n"] for v in an["breakdowns"]["database"].values()) == 20
    assert set(an["breakdowns"]["query_pattern"]) == {"join", "none"}
    s1 = an["sensitivity"]["S1_swap_supplementary_arm_into_C"]
    assert s1["swapped_ids"] == ["q003"] and s1["result"]["n"] == 20 and "q003" in s1["changed_outcome"]
    assert an["sensitivity"]["S2_without_near_duplicate_items"]["n"] == 18          # both near-dup items dropped from EVERY arm
    assert all(v["total"] == 18 for v in an["sensitivity"]["S2_without_near_duplicate_items"]["arms"].values())
    assert an["sensitivity"]["S3_reference_status_ok"]["n"] == sum(m["reference_status"] == "ok" for m in meta.values())
    assert set(an["near_duplicate_items"]) == set(nd) and an["analysis_plan_sha256"] == ST.plan_sha()
    # the swap changes only C; A and B are untouched
    assert s1["result"]["arms"]["A_zero_shot"] == pa["arms"]["A_zero_shot"] and s1["result"]["arms"]["B_random_k3"] == pa["arms"]["B_random_k3"]


def test_category_mapping_and_exact_accounting():
    mk = lambda o, fs=None, d=None: {"outcome": o, "failure_source": fs, "detail": d}   # noqa: E731
    cases = [("correct", mk("correct")), ("incorrect", mk("incorrect", "model")), ("invalid", mk("invalid", "model")),
             ("timeout", mk("timeout", "model")), ("execution_error", mk("failed", "model", "execution_error")),
             ("generation_failure", mk("failed", "infrastructure", "generation_error")),
             ("infrastructure_failure", mk("failed", "infrastructure", "atlas_unavailable")), ("not_comparable", mk("not_comparable", "reference"))]
    for want, e in cases:
        assert ST.category(e) == want
    evals = [{"id": f"i{n}", **e} for n, (_, e) in enumerate(cases)]
    acct = ST.account(evals, [e["id"] for e in evals])
    assert sum(acct.values()) == 8 and set(acct) == set(ST.CATEGORIES) and all(v == 1 for v in acct.values())
    with pytest.raises(ValueError):
        ST.account(evals[:-1], [e["id"] for e in evals])                            # missing id
    with pytest.raises(ValueError):
        ST.account(evals + [evals[0]], [e["id"] for e in evals])                    # duplicate id
    with pytest.raises(ValueError):
        ST.category({"outcome": "weird"})


# ------------------------------------------------------------------ synthetic Kaggle outputs
@pytest.fixture(scope="module")
def exp():
    if not PKG.exists():
        pytest.skip("package not built")
    return SC.load_expected(PKG)


def make_output(tmp, exp, arms, *, env=None, mode_a="reuse_mql0006", name="out", kinds=None, mutate=None):
    """Write a synthetic MQL-0007 output directory (as the notebook would) for `arms`. kinds(arm, id) -> fixture kind."""
    root = Path(tmp) / name
    cfgx, M = exp["cfg"], exp["cfg"]["model"]
    m6 = exp["mql6_cfg"]
    env = env or {"torch": m6["torch"], "transformers": m6["transformers"], "gpu": m6["gpu"], "accelerate": m6["environment"]["packages"]["accelerate"],
                  "tokenizers": m6["environment"]["packages"]["tokenizers"]}
    ids_all = exp["ids"]
    for arm in arms:
        d = root / "models" / arm
        d.mkdir(parents=True)
        ids = exp["arm_ids"][arm]
        if arm == "A_zero_shot" and mode_a == "reuse_mql0006":
            shutil.copyfile(PKG / "data/zero_shot_mql0006/predictions.jsonl", d / "predictions.jsonl")
            cfg = {"experiment": "MQL-0007", "arm": arm, "mode": "reuse_mql0006", "fingerprint": m6["fingerprint"], "model_id": M["id"],
                   "model_revision": M["revision"], "dtype": m6["dtype"], "gpu": m6["gpu"], "torch": m6["torch"], "transformers": m6["transformers"],
                   "decoding": M["decoding"], "max_new_tokens": M["max_new_tokens"], "batch_size": M["batch_size"]}
        else:
            cfg = {"experiment": "MQL-0007", "arm": arm, "mode": "generated", "arm_config": cfgx["arms"][arm],
                   "retrieval_exclusions": cfgx["retrieval_exclusions"] if arm in ("B_random_k3", "C_hybrid_k3") else [],
                   "model_label": M["label"], "model_id": M["id"], "model_revision": M["revision"], "runtime": M["runtime"],
                   "decoding": M["decoding"], "max_new_tokens": M["max_new_tokens"], "batch_size": M["batch_size"], "seed": 42,
                   "dtype": "bfloat16", "gpu": env["gpu"], "torch": env["torch"], "transformers": env["transformers"]}
            cfg["fingerprint"] = C.fingerprint({k: v for k, v in cfg.items() if k not in SC.POST_FP})
            cfg.update(generation_wall_s=1.0, environment={"packages": {k: env[k] for k in ("torch", "transformers", "accelerate", "tokenizers")}})
            coll = {}
            recs = []
            for i in ids:
                kind = (kinds(arm, i) if kinds else "gold")
                recs.append({"id": i, "db_id": i.split(":")[0] if False else None, "fingerprint": cfg["fingerprint"], "model": M["label"], "arm": arm,
                             "status": "ok", "raw_text": json.dumps({"collection": "COLL", "pipeline": [{"$match": {"__fx": f"{kind}:{i}"}}]}),
                             "error": None, "created_at": "2026-11-06T00:00:00+00:00", "prompt_sha256": exp["prompt_hashes"][arm][i]})
            C.write_jsonl(d / "predictions.jsonl", recs)
        C.write_json(d / "run_config.json", cfg)
        if arm != "A_zero_shot":
            C.write_jsonl(d / "retrieval_log.jsonl", [dict(p, near_duplicate_template_item=False) for p in exp["plans"][arm]])
    C.write_json(root / "MQL0007_STATUS.json", {"experiment": "MQL-0007", "stage": "predictions", "status": "complete",
                 "package_manifest_sha256": exp["package_manifest_sha256"], "config_sha256": exp["config_sha256"],
                 "evaluation_ids_sha256": cfgx["evaluation"]["ids_sha256"]})
    if mutate:
        mutate(root)
    sums = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    (root / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(root / n)}  {n}\n" for n in sums), encoding="utf-8")
    return root


def fix_db_ids(root, ids_db):
    """Fill the db_id of generated records (the fixture above leaves it for the caller to set from the evaluation set)."""
    for pf in root.rglob("predictions.jsonl"):
        recs = C.read_jsonl(pf)
        if recs and recs[0].get("db_id") is None:
            for r in recs:
                r["db_id"] = ids_db[r["id"]]
                r["raw_text"] = r["raw_text"].replace("COLL", ids_db["__coll__" + ids_db[r["id"]]])
            C.write_jsonl(pf, recs)


@pytest.fixture(scope="module")
def dbmap(exp):
    ev = C.read_jsonl(PKG / "data/eval/eval_inputs.jsonl")
    m = {e["id"]: e["db_id"] for e in ev}
    if BUNDLE.exists():
        mapping = C.load_mapping(BUNDLE)
        for coll, mm in sorted(mapping["collections"].items()):
            m.setdefault("__coll__" + mm["db_id"], coll)
    else:
        for e in ev:
            m.setdefault("__coll__" + e["db_id"], e["db_id"] + "__t")
    return m


def build(tmp, exp, dbmap, arms=("A_zero_shot", "B_random_k3", "C_hybrid_k3"), **kw):
    mut = kw.pop("mutate", None)
    root = make_output(tmp, exp, arms, mutate=None, **kw)
    fix_db_ids(root, dbmap)
    if mut:
        mut(root)
    sums = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (root / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(root / n)}  {n}\n" for n in sums), encoding="utf-8")
    return root


def verify(exp, *dirs, **kw):
    return SC.verify_predictions(list(dirs), exp, **kw)


# ------------------------------------------------------------------ verification: accept / reject
def test_complete_three_arm_output_verifies(tmp_path, exp, dbmap):
    rep = verify(exp, build(tmp_path, exp, dbmap))
    assert rep["ok"], rep["problems"]
    assert rep["zero_shot_mode"] == "reuse_mql0006" and not rep["environment_drift"]


def test_fresh_arm_a_in_the_same_environment_verifies(tmp_path, exp, dbmap):
    rep = verify(exp, build(tmp_path, exp, dbmap, mode_a="fresh"))
    assert rep["ok"], rep["problems"]


def _edit_preds(arm, fn):
    def mut(root):
        pf = root / "models" / arm / "predictions.jsonl"
        lines = pf.read_text(encoding="utf-8").splitlines()
        pf.write_text("\n".join(fn(lines)) + "\n", encoding="utf-8")
    return mut


@pytest.mark.parametrize("name,fn,needle", [
    ("missing_id", lambda L: L[:-1], "missing"),
    ("duplicate_id", lambda L: L + [L[0]], "duplicate"),
    ("unexpected_id", lambda L: L + [L[0].replace('"spider_dev_0000"', '"spider_dev_9999"')], "unexpected"),
    ("malformed_json", lambda L: L[:5] + ["{not json"] + L[6:], "not JSON"),
    ("not_an_object", lambda L: L[:5] + ["[1,2]"] + L[6:], "not an object"),
    ("missing_field", lambda L: [L[0].replace('"prompt_sha256"', '"x"')] + L[1:], "missing/non-string"),
    ("wrong_prompt", lambda L: [L[0].replace(json.loads(L[0])["prompt_sha256"], "0" * 64)] + L[1:], "different prompt"),
    ("generation_error", lambda L: [json.dumps({**json.loads(L[0]), "status": "error", "raw_text": None, "error": "boom"})] + L[1:], "generation error"),
    ("out_of_order", lambda L: [L[1], L[0]] + L[2:], "out of order"),
])
def test_bad_predictions_are_rejected(tmp_path, exp, dbmap, name, fn, needle):
    rep = verify(exp, build(tmp_path, exp, dbmap, mutate=_edit_preds("C_hybrid_k3", fn)))
    assert not rep["ok"] and any(needle in p for p in rep["problems"]), rep["problems"][:3]


def test_missing_arm_wrong_experiment_and_tampered_files_are_rejected(tmp_path, exp, dbmap):
    assert not verify(exp, build(tmp_path, exp, dbmap, arms=("A_zero_shot", "B_random_k3"), name="o1"))["ok"]
    def other_exp(root):
        st = json.loads((root / "MQL0007_STATUS.json").read_text(encoding="utf-8"))
        st["experiment"] = "MQL-0006"
        C.write_json(root / "MQL0007_STATUS.json", st)
    rep = verify(exp, build(tmp_path, exp, dbmap, name="o2", mutate=other_exp))
    assert any("experiment_id" in p for p in rep["problems"])
    def other_pkg(root):
        st = json.loads((root / "MQL0007_STATUS.json").read_text(encoding="utf-8"))
        st["package_manifest_sha256"] = "0" * 64
        C.write_json(root / "MQL0007_STATUS.json", st)
    assert any("package_identity" in p for p in verify(exp, build(tmp_path, exp, dbmap, name="o3", mutate=other_pkg))["problems"])
    root = build(tmp_path, exp, dbmap, name="o4")
    with open(root / "models/B_random_k3/predictions.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")                                                                  # changed after SHA256SUMS was written
    assert any("checksums_match" in p for p in verify(exp, root)["problems"])


def test_model_config_fingerprint_and_exclusion_are_enforced(tmp_path, exp, dbmap):
    def bad_cfg(root):
        p = root / "models/C_hybrid_k3/run_config.json"
        c = json.loads(p.read_text(encoding="utf-8"))
        c["batch_size"] = 4
        C.write_json(p, c)
    rep = verify(exp, build(tmp_path, exp, dbmap, name="o1", mutate=bad_cfg))
    assert any("decoding" in p for p in rep["problems"]) and any("fingerprint_recomputes" in p for p in rep["problems"])
    def no_excl(root):
        p = root / "models/C_hybrid_k3/retrieval_log.jsonl"
        C.write_jsonl(p, [dict(r, excluded_by_rule=[]) for r in C.read_jsonl(p)])
    assert any("retrieval_log_equals_frozen_plan" in p for p in verify(exp, build(tmp_path, exp, dbmap, name="o2", mutate=no_excl))["problems"])
    def wrong_dtype(root):
        p = root / "models/B_random_k3/run_config.json"
        c = json.loads(p.read_text(encoding="utf-8"))
        c["dtype"] = "float16"
        C.write_json(p, c)
    assert not verify(exp, build(tmp_path, exp, dbmap, name="o3", mutate=wrong_dtype))["ok"]


def test_tampered_reused_arm_a_is_rejected(tmp_path, exp, dbmap):
    def tamper(root):
        pf = root / "models/A_zero_shot/predictions.jsonl"
        pf.write_text(pf.read_text(encoding="utf-8").replace("count", "COUNT", 1), encoding="utf-8")
    rep = verify(exp, build(tmp_path, exp, dbmap, mutate=tamper))
    assert any("reuse_is_byte_identical" in p for p in rep["problems"])


def test_several_zips_one_arm_each_and_duplicate_arm_rules(tmp_path, exp, dbmap):
    z1 = build(tmp_path, exp, dbmap, arms=("A_zero_shot", "B_random_k3"), name="v1")
    z2 = build(tmp_path, exp, dbmap, arms=("C_hybrid_k3", K.SUPP_ARM), name="v2")
    rep = verify(exp, z1, z2)
    assert rep["ok"], rep["problems"]
    assert set(rep["arm_source"]) == {"A_zero_shot", "B_random_k3", "C_hybrid_k3", K.SUPP_ARM}
    z3 = build(tmp_path, exp, dbmap, arms=("B_random_k3",), name="v3")                   # byte-identical copy of B: accepted
    assert verify(exp, z1, z2, z3)["ok"]
    z4 = build(tmp_path, exp, dbmap, arms=("B_random_k3",), name="v4", kinds=lambda a, i: "wrong")   # different B predictions: refused
    rep = verify(exp, z1, z2, z4)
    assert not rep["ok"] and any("duplicate_source_identical" in p for p in rep["problems"])


def test_environment_must_be_identical_across_arms(tmp_path, exp, dbmap):
    m6 = exp["mql6_cfg"]
    drift = {"torch": "2.12.0", "transformers": m6["transformers"], "gpu": m6["gpu"], "accelerate": m6["environment"]["packages"]["accelerate"],
             "tokenizers": m6["environment"]["packages"]["tokenizers"]}
    z1 = build(tmp_path, exp, dbmap, arms=("A_zero_shot",), name="v1")                      # reused A = MQL-0006 environment
    z2 = build(tmp_path, exp, dbmap, arms=("B_random_k3", "C_hybrid_k3"), name="v2", env=drift)
    rep = verify(exp, z1, z2)
    assert not rep["ok"] and rep["environment_drift"] and any("environment_identical" in p for p in rep["problems"])
    assert verify(exp, z1, z2, allow_env_drift=True)["ok"]                                   # explicit, recorded override
    za = build(tmp_path, exp, dbmap, arms=("A_zero_shot",), name="v3", mode_a="fresh", env=drift)   # fresh A in the drifted env: consistent
    assert verify(exp, za, z2)["ok"]
    gpu = dict(drift, gpu="Tesla P100")
    zg = build(tmp_path, exp, dbmap, arms=("B_random_k3", "C_hybrid_k3"), name="v4", env=gpu)
    assert not verify(exp, za, zg)["ok"]


# ------------------------------------------------------------------ near-duplicate exclusion, determinism, compatibility
def test_near_duplicate_exclusion_is_deterministic_and_logged():
    cfg = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    plans = {a: C.read_jsonl(PKG / f"data/plans/retrieval_plan_{a}.jsonl") for a in ("B_random_k3", "C_hybrid_k3", K.SUPP_ARM)}
    ex = {(x["eval_id"], x["corpus_id"]) for x in cfg["retrieval_exclusions"]}
    assert ex == {("spider_dev_0383", "spider_train_00650"), ("spider_dev_0824", "spider_train_00650")}
    for arm in ("B_random_k3", "C_hybrid_k3"):
        by = {p["id"]: p for p in plans[arm]}
        for e, c in ex:
            assert c not in by[e]["selected_ids"] and by[e]["excluded_by_rule"] and by[e]["excluded_by_rule"][0]["corpus_id"] == c
            assert by[e]["excluded_by_rule"][0]["original_rank"] >= 1
        assert sum(bool(p["excluded_by_rule"]) for p in plans[arm]) == 2                     # logged only where the rule fired
    assert [p["id"] for p in plans[K.SUPP_ARM]] == ["spider_dev_0383"] and "spider_train_00650" in plans[K.SUPP_ARM][0]["selected_ids"]
    assert plans[K.SUPP_ARM][0]["excluded_by_rule"] == [] and plans[K.SUPP_ARM][0]["arm"] == K.SUPP_ARM
    cby = {p["id"]: p for p in plans["C_hybrid_k3"]}
    assert cby["spider_dev_0383"]["selected_ids"] != plans[K.SUPP_ARM][0]["selected_ids"]
    assert cby["spider_dev_0824"]["selected_ids"][0] == "spider_train_04386"                 # unchanged: the example was rank 10 there


def test_exclusion_changes_only_the_pair_and_rest_of_ranking_is_untouched():
    cfg = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    corpus = C.read_jsonl(PKG / "data/corpus/provisional_rag_corpus.jsonl")
    tables = {t["db_id"]: t for t in json.loads((PKG / "data/context/tables_subset.json").read_text(encoding="utf-8"))}
    docs = K.build_docs(corpus, tables)
    atlas = json.loads((PKG / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
    prompts = C.read_jsonl(PKG / "data/eval/prompts.jsonl")
    ev = C.read_jsonl(PKG / "data/eval/eval_inputs.jsonl")
    pick = [e for e in ev if e["id"] in ("spider_dev_0383", "spider_dev_0001", "spider_dev_0500")]
    infos = K.eval_schema_infos(atlas, sorted({e["db_id"] for e in pick}))
    ct = lambda s: len(s.split())   # noqa: E731
    with_x = K.build_plan(cfg, "C_hybrid_k3", pick, prompts, docs, infos, ct)
    again = K.build_plan(cfg, "C_hybrid_k3", pick, prompts, docs, infos, ct)
    no_x = K.build_plan(cfg, "C_hybrid_k3", pick, prompts, docs, infos, ct, exclusions=[])
    assert with_x == again                                                                  # deterministic
    assert K.plan_diff_ids(with_x, no_x) == ["spider_dev_0383"]                             # only the excluded pair's item changes
    a = [x for x in with_x if x["id"] == "spider_dev_0383"][0]
    b = [x for x in no_x if x["id"] == "spider_dev_0383"][0]
    assert b["selected_ids"][0] == "spider_train_00650" and a["selected_ids"][:2] == b["selected_ids"][1:3]   # the next-ranked examples take its place
    assert K.selection_neardup_violations(no_x, pick, {d["doc_id"]: d for d in docs}, 0.8) == [("spider_dev_0383", "spider_train_00650", 0.8)]
    assert K.selection_neardup_violations(with_x, pick, {d["doc_id"]: d for d in docs}, 0.8) == []


def test_arm_a_compatibility_is_conditional_on_the_runtime_environment():
    rep = PF.arm_a_compatibility(PKG, BUNDLE)
    assert rep["static_ok"], {k: v for k, v in rep["static_checks"].items() if not v["ok"]}
    assert rep["mql0006_environment"]["transformers"] == "5.16.1" and rep["mql0006_environment"]["gpu"] == "Tesla T4"
    assert "CONDITIONAL" in rep["decision"] and "auto" in rep["decision"]


def test_near_duplicate_investigation_reports_ids_and_scores_only():
    rep = PF.near_duplicate_investigation(PKG, ROOT)
    assert rep["evaluation_pair"]["plan_changed_by_exclusion"] == {"B_random_k3": [], "C_hybrid_k3": ["spider_dev_0383"]}
    t = rep["protected_spider_test"]
    if "skipped" in t:
        pytest.skip("spider_sql_v1 test not available")
    assert t["pairs_at_or_above_threshold"] == 16 and t["exact_question_matches"] == 0 and t["sql_equal_pairs"] == 0 and t["all_different_database"]
    text = json.dumps(rep)
    test_q = [r["question"] for r in C.read_jsonl(ROOT / "data/processed/spider_sql_v1/test_full.jsonl")[:200]]
    assert not any(q in text for q in test_q) and "SELECT" not in text            # no test question or SQL in the report


# ------------------------------------------------------------------ Atlas scoring with a fake read-only Atlas
class FakeCursor:
    def __init__(self, docs):
        self.docs = docs

    def __iter__(self):
        return iter([dict(d) for d in self.docs])

    def close(self):
        pass


class Timeout(Exception):
    code = 50


class ExecutionTimeout(Exception):
    pass


class AutoReconnect(Exception):
    pass


class FakeColl:
    def __init__(self, db, name, count):
        self.db, self.name, self._count = db, name, count

    def count_documents(self, q):
        self.db.calls.append(("count_documents", self.name))
        return self._count

    def aggregate(self, pipeline, **kw):
        assert kw.get("maxTimeMS", 0) > 0 and kw.get("batchSize") == 1000          # always bounded
        self.db.calls.append(("aggregate", self.name, json.dumps(pipeline), kw["maxTimeMS"]))
        return self.db.behave(self.name, pipeline)


class FakeDB:
    def __init__(self, mapping, refs, actions=("find", "listCollections", "killCursors")):
        self.mapping, self.refs, self.actions, self.calls = mapping, {r["id"]: r for r in refs}, list(actions), []
        self.colls = {n: FakeColl(self, n, m["documents"]) for n, m in mapping["collections"].items()}
        self.infra_once = set()

    def __getitem__(self, n):
        return self.colls[n]

    def list_collection_names(self):
        return list(self.colls)

    def command(self, cmd):
        if isinstance(cmd, dict) and "connectionStatus" in cmd:
            return {"authInfo": {"authenticatedUsers": [{"user": "codegen_eval_readonly", "db": "admin"}],
                                 "authenticatedUserPrivileges": [{"resource": {"db": self.mapping["atlas_db"], "collection": ""}, "actions": self.actions}]}}
        raise PermissionError("not authorized")

    def behave(self, coll, pipeline):
        kind, _, iid = pipeline[0]["$match"]["__fx"].partition(":")
        ref = self.refs[iid]
        if kind == "gold":
            cols = ref["gold_columns"]
            return FakeCursor([dict(zip(cols, row)) for row in ref["gold_rows"]])
        if kind == "wrong":
            return FakeCursor([{"zz": 12345}])
        if kind == "timeout":
            raise Timeout("operation exceeded time limit")
        if kind == "error":
            raise RuntimeError("Unrecognized pipeline stage name")
        if kind == "infra" and iid not in self.infra_once:
            self.infra_once.add(iid)
            raise AutoReconnect("connection lost")
        return FakeCursor([{"zz": 1}])


class FakeAdmin:
    def command(self, c):
        return {"ok": 1} if c == "ping" else {"version": "8.0.fake"}


class FakeClient:
    def __init__(self, db):
        self.db, self.admin, self.closed = db, FakeAdmin(), False

    def __getitem__(self, name):
        return self.db

    def close(self):
        self.closed = True


def kinds_fn(arm, iid):
    n = int(iid.rsplit("_", 1)[1])
    base = {"A_zero_shot": 4, "B_random_k3": 3, "C_hybrid_k3": 2}.get(arm, 2)     # different hit rates per arm
    special = {"spider_dev_0010": "timeout", "spider_dev_0011": "error", "spider_dev_0012": "wrong", "spider_dev_0013": "out", "spider_dev_0014": "garbage"}
    if iid in special:
        return special[iid]
    return "gold" if n % base else "wrong"


def zip_dir(src, zpath):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(src).rglob("*")):
            if p.is_file():
                z.write(p, f"{Path(src).name}/{p.relative_to(src).as_posix()}")
    return zpath


def build_scoring_zip(tmp, exp, dbmap, name="o", **kw):
    def mut(root):
        for arm in ("B_random_k3", "C_hybrid_k3", "A_zero_shot", K.SUPP_ARM):
            pf = root / "models" / arm / "predictions.jsonl"
            if not pf.exists() or arm == "A_zero_shot" and (root / "models/A_zero_shot/run_config.json").exists() and \
                    json.loads((root / "models/A_zero_shot/run_config.json").read_text(encoding="utf-8")).get("mode") == "reuse_mql0006":
                continue
            recs = C.read_jsonl(pf)
            for r in recs:
                kind = kinds_fn(arm, r["id"])
                coll = dbmap["__coll__" + r["db_id"]]
                if kind == "out":
                    r["raw_text"] = json.dumps({"collection": coll, "pipeline": [{"$match": {"__fx": f"gold:{r['id']}"}}, {"$out": "pwned"}]})
                elif kind == "garbage":
                    r["raw_text"] = "I cannot answer that."
                else:
                    r["raw_text"] = json.dumps({"collection": coll, "pipeline": [{"$match": {"__fx": f"{kind}:{r['id']}"}}]})
            C.write_jsonl(pf, recs)
    root = build(tmp, exp, dbmap, name=name, mode_a="fresh", mutate=mut, **kw)
    return zip_dir(root, Path(tmp) / f"{name}.zip")


@pytest.fixture(scope="module")
def scored(tmp_path_factory, exp, dbmap):
    if not BUNDLE.exists():
        pytest.skip("bundle not available")
    tmp = tmp_path_factory.mktemp("score")
    zpath = build_scoring_zip(tmp, exp, dbmap)
    refs = C.load_reference(BUNDLE)
    mapping = C.load_mapping(BUNDLE)
    db = FakeDB(mapping, refs)
    before = SC_tree(MQL6)
    st, an, run_dir = SC.cmd_score([str(zpath)], "t1", BUNDLE, PKG, tmp / "dest", lambda: (FakeClient(db), lambda s: s), log=lambda *_: None)
    return {"st": st, "an": an, "run": run_dir, "db": db, "tmp": tmp, "before": before, "refs": refs, "mapping": mapping}


def SC_tree(root):
    root = Path(root)
    if not root.exists():
        return {}
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


@needs_bundle
def test_three_arm_scoring_exact_892_accounting(scored, exp):
    st = scored["st"]
    assert st["n_expected"] == 892 and set(st["arms"]) == {"A_zero_shot", "B_random_k3", "C_hybrid_k3"}   # D not part of this run (not supplied)
    for arm, acct in st["arms"].items():
        assert sum(acct.values()) == 892 and set(acct) == set(ST.CATEGORIES), arm
        ev = C.read_jsonl(scored["run"] / "scoring/models" / arm / "evaluation_results.jsonl")
        assert [e["id"] for e in ev] == exp["ids"]                                           # the exact same 892 ids, in order
        assert acct["timeout"] >= 1 and acct["execution_error"] >= 1 and acct["invalid"] >= 2 and acct["correct"] > 100
        assert acct["generation_failure"] == 0 and acct["infrastructure_failure"] == 0
    assert st["status"] == "complete" and scored["an"]["primary_analysis"]["n"] == 892
    for arm in ("A_zero_shot", "B_random_k3", "C_hybrid_k3"):
        ev = {e["id"]: e for e in C.read_jsonl(scored["run"] / "scoring/models" / arm / "evaluation_results.jsonl")}
        assert ev["spider_dev_0010"]["outcome"] == "timeout" and ev["spider_dev_0011"]["detail"] == "execution_error"
        assert ev["spider_dev_0013"]["outcome"] == "invalid" and ev["spider_dev_0014"]["outcome"] == "invalid"
        assert ev["spider_dev_0013"]["executed"] is False                                    # $out is rejected before execution
    pa = scored["an"]["primary_analysis"]
    assert set(pa["primary"]) == {"C_hybrid_k3_vs_A_zero_shot", "C_hybrid_k3_vs_B_random_k3"}
    assert (scored["run"] / "scoring/results_report.md").read_text(encoding="utf-8").startswith("# MQL-0007 results")


@needs_bundle
def test_scoring_is_read_only_and_bounded(scored):
    db = scored["db"]
    kinds = {c[0] for c in db.calls}
    assert kinds <= {"aggregate", "count_documents"}, kinds                                  # no find/insert/update/delete/index/command writes
    agg = [c for c in db.calls if c[0] == "aggregate"]
    assert agg and all(c[3] == C.EXEC_TIMEOUT_MS for c in agg)                               # maxTimeMS on every execution
    for banned in ("$out", "$merge", "$function", "$where", "$accumulator"):
        assert not any(banned in c[2] for c in agg), banned
    assert scored["db"].calls and not any(c[1].startswith("system.") for c in agg)


@needs_bundle
def test_no_credentials_or_uri_in_outputs(scored):
    import re
    pat = re.compile(r"mongo" r"db\+srv|mongo" r"db://|MONGO" r"DB_URI|password\s*[=:]", re.I)
    for p in scored["run"].rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".md", ".txt"):
            assert not pat.search(p.read_text(encoding="utf-8", errors="ignore")), p.name


@needs_bundle
def test_mql0006_results_untouched_and_paths_isolated(scored):
    assert SC_tree(MQL6) == scored["before"]                                                 # not one file added, removed or modified
    assert MQL6.resolve() not in SC.DEST_ROOT.resolve().parents and SC.DEST_ROOT.resolve() != MQL6.resolve()
    with pytest.raises(SC.ScoringError):
        SC.guard_dest(MQL6 / "v2")
    with pytest.raises(SC.ScoringError):
        SC.cmd_score([], "x", dest_root=MQL6 / "v1")


@needs_bundle
def test_write_privileges_stop_scoring_before_any_execution(tmp_path, exp, dbmap):
    zpath = build_scoring_zip(tmp_path, exp, dbmap)
    db = FakeDB(C.load_mapping(BUNDLE), C.load_reference(BUNDLE), actions=("find", "listCollections", "killCursors", "insert", "update"))
    with pytest.raises(C.PreflightError, match="NOT read-only"):
        SC.cmd_score([str(zpath)], "w1", BUNDLE, PKG, tmp_path / "dest", lambda: (FakeClient(db), lambda s: s), log=lambda *_: None)
    assert not [c for c in db.calls if c[0] == "aggregate"]                                   # nothing was executed


@needs_bundle
def test_unsafe_operators_are_rejected_without_execution():
    mapping = C.load_mapping(BUNDLE)
    refs = C.load_reference(BUNDLE)
    K_, _ = C.bundle_modules(BUNDLE)
    db = FakeDB(mapping, refs)
    allowed = C.allowed_collections(mapping)
    ref = refs[0]
    coll = sorted(allowed[ref["db_id"]])[0]
    for op in ({"$out": "x"}, {"$merge": {"into": "x"}}, {"$function": {"body": "x", "args": [], "lang": "js"}}, {"$match": {"$where": "1"}},
               {"$group": {"_id": None, "a": {"$accumulator": {"init": "x"}}}}):
        raw = json.dumps({"collection": coll, "pipeline": [op]})
        rec, rows = C.evaluate_prediction(db, K_, {"id": ref["id"], "status": "ok", "raw_text": raw}, ref, allowed[ref["db_id"]])
        assert rec["outcome"] == "invalid" and not rec["executed"] and rows is None, op
    assert db.calls == []


@needs_bundle
def test_atlas_outage_is_retriable_and_reported_separately(tmp_path, exp, dbmap):
    zpath = build_scoring_zip(tmp_path, exp, dbmap, kinds=None)
    # make one item of C an Atlas outage on its first execution
    z = tmp_path / "o"
    for pf in z.rglob("C_hybrid_k3/predictions.jsonl"):
        recs = C.read_jsonl(pf)
        r = recs[20]
        r["raw_text"] = json.dumps({"collection": dbmap["__coll__" + r["db_id"]], "pipeline": [{"$match": {"__fx": f"infra:{r['id']}"}}]})
        C.write_jsonl(pf, recs)
    sums = sorted(p.relative_to(z).as_posix() for p in z.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (z / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(z / n)}  {n}\n" for n in sums), encoding="utf-8")
    zpath = zip_dir(z, tmp_path / "o_infra.zip")
    db = FakeDB(C.load_mapping(BUNDLE), C.load_reference(BUNDLE))
    st, an, run_dir = SC.cmd_score([str(zpath)], "i1", BUNDLE, PKG, tmp_path / "dest", lambda: (FakeClient(db), lambda s: s), log=lambda *_: None)
    assert st["arms"]["C_hybrid_k3"]["infrastructure_failure"] == 1 and st["status"] == "partial"      # never silently counted as a model error
    assert sum(st["arms"]["C_hybrid_k3"].values()) == 892
    st2, _, _ = SC.cmd_score([str(zpath)], "i1", BUNDLE, PKG, tmp_path / "dest", lambda: (FakeClient(db), lambda s: s), log=lambda *_: None)
    assert st2["arms"]["C_hybrid_k3"]["infrastructure_failure"] == 0 and st2["status"] == "complete"   # the retry resolves it from the checkpoint


@needs_bundle
def test_scoring_refuses_unverified_predictions_before_touching_atlas(tmp_path, exp, dbmap):
    root = build(tmp_path, exp, dbmap, arms=("A_zero_shot", "B_random_k3"), name="bad")
    zpath = zip_dir(root, tmp_path / "bad.zip")
    db = FakeDB(C.load_mapping(BUNDLE), C.load_reference(BUNDLE))
    with pytest.raises(SC.ScoringError, match="failed verification"):
        SC.cmd_score([str(zpath)], "b1", BUNDLE, PKG, tmp_path / "dest", lambda: (FakeClient(db), lambda s: s), log=lambda *_: None)
    assert db.calls == []
