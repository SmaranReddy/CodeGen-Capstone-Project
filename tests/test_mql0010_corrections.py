"""MQL-0010 audit corrections R2 (matched delta), R5 (generic-retry control) and the scoring adapter.
Synthetic data and fake models only; no model, no database, no network."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0010/src"))
sys.path.insert(1, str(ROOT / "kaggle/nl2mql_mql0009_retrieval_control/src"))
import validator as V        # noqa: E402
import recovery as RC        # noqa: E402
import driver as DR          # noqa: E402
import scoring as SC         # noqa: E402

CTX = {
    "shop__emp": {"db_id": "shop", "table": "emp", "fields": {"emp_id": ["int"], "name": ["string"], "dept_id": ["int"]}},
    "shop__dept": {"db_id": "shop", "table": "dept", "fields": {"dept_id": ["int"], "city": ["string"]}},
}
GOOD = json.dumps({"collection": "shop__emp", "pipeline": [{"$project": {"_id": 0, "name": 1}}]})
BAD = json.dumps({"collection": "shop__emp", "pipeline": [{"$match": {"city": "x"}}, {"$project": {"_id": 0, "name": 1}}]})


@pytest.fixture
def val():
    return V.SchemaValidator("shop", CTX)


def _plan_rows(ids, sha="h"):
    return [{"id": i, "db_id": "shop", "prompt_sha256": f"{sha}{i}"} for i in ids]


# ------------------------------------------------------------------ R5 generic-retry control
def test_generic_feedback_has_no_schema_content(val):
    seen = []
    out, tr = RC.recover("SYS", "p", BAD, val, lambda sy, m: seen.append((sy, m)) or GOOD, feedback_mode="generic")
    assert tr["retried"] and tr["feedback"] == RC.GENERIC_FEEDBACK and tr["feedback_mode"] == "generic"
    txt = seen[0][1][2]["content"]
    assert "city" not in txt and "shop__" not in txt and "stage" not in txt
    assert seen[0][0] == "SYS" and out == GOOD          # same system prompt, same acceptance rule


def test_generic_and_schema_share_trigger_and_format_message(val):
    for mode in ("schema", "generic"):
        _, tr = RC.recover("S", "p", GOOD, val, lambda sy, m: BAD, feedback_mode=mode)
        assert not tr["retried"]
        _, tr2 = RC.recover("S", "p", "nonsense", val, lambda sy, m: GOOD, feedback_mode=mode)
        assert tr2["feedback"] == RC.FORMAT_FEEDBACK
    with pytest.raises(ValueError):
        RC.recover("S", "p", BAD, val, lambda sy, m: GOOD, feedback_mode="hint")


def test_retry_arms_schema_vs_generic_retry_the_same_items():
    prompts = {"a": ("SYS", "Qa"), "b": ("SYS", "Qb")}
    base = DR.first_pass("D0", _plan_rows(["a", "b"]), prompts, lambda s, m: GOOD if m[0]["content"] == "Qa" else BAD)
    v = DR.retry_arm("D0V", base, prompts, CTX, lambda s, m: GOOD)
    g = DR.retry_arm("D0G", base, prompts, CTX, lambda s, m: BAD, feedback_mode="generic")
    assert [r["recovery"]["retried"] for r in v] == [r["recovery"]["retried"] for r in g] == [False, True]
    assert g[1]["raw_text"] == BAD and g[1]["recovery"]["kept"] == "first"


# ------------------------------------------------------------------ R2 matched delta
def test_matched_delta_generates_control_and_new_only_for_changed_items():
    base_plan = _plan_rows(["a", "b", "c"])
    new_plan = [dict(base_plan[0]), dict(base_plan[1], prompt_sha256="NEW-b"), dict(base_plan[2])]
    base = DR.first_pass("E", base_plan, {i: ("S", f"E-{i}") for i in "abc"}, lambda s, m: "out:" + m[0]["content"])
    calls = []

    def gen(s, m):
        calls.append(m[0]["content"])
        assert len(m) == 1
        return "out:" + m[0]["content"]
    new, ctl = DR.matched_delta("EF", "EP", new_plan, base_plan, base, {i: ("S", f"F-{i}") for i in "abc"},
                                {i: ("S", f"E-{i}") for i in "abc"}, gen)
    assert calls == ["E-b", "F-b"]
    assert [r["source"] for r in new] == ["reused:E", "generated:matched_delta", "reused:E"]
    assert ctl[1]["raw_text"] == "out:E-b" and new[1]["raw_text"] == "out:F-b"
    assert ctl[0] == {**base[0], "arm": "EP", "source": "reused:E"}


# ------------------------------------------------------------------ scoring adapter
def _synthetic_exports(n=40, flag_every=4):
    ids = [f"q{i:03d}" for i in range(n)]

    def plan(tag):
        return [{"id": i, "db_id": "shop", "prompt_sha256": f"{tag}-{i}"} for i in ids]
    plans = {"A0": plan("A"), "D0": plan("D"), "E": plan("E"),
             "EF": [{"id": i, "db_id": "shop", "prompt_sha256": ("F-" if k < 3 else "E-") + i} for k, i in enumerate(ids)]}
    prompts = {i: ("S", "Q" + i) for i in ids}

    def first(s, m):
        return BAD if int(m[0]["content"][2:]) % flag_every == 0 else GOOD
    recs = {a: DR.first_pass(a, plans[a], prompts, first) for a in ("A0", "D0", "E")}
    recs["EF"], recs["EP"] = DR.matched_delta("EF", "EP", plans["EF"], plans["E"], recs["E"], prompts, prompts, first)
    recs["A0V"] = DR.retry_arm("A0V", recs["A0"], prompts, CTX, lambda s, m: GOOD)
    recs["D0V"] = DR.retry_arm("D0V", recs["D0"], prompts, CTX, lambda s, m: GOOD)
    recs["D0G"] = DR.retry_arm("D0G", recs["D0"], prompts, CTX, lambda s, m: BAD, feedback_mode="generic")
    recs["FULL"] = DR.retry_arm("FULL", recs["EF"], prompts, CTX, lambda s, m: GOOD)
    return ids, plans, recs


def test_verify_exports_accepts_contract_and_rejects_violations():
    ids, plans, recs = _synthetic_exports()
    assert SC.verify_exports(recs, plans, ids)["delta_items"] == 3

    def copy():
        return {a: [dict(r) for r in rows] for a, rows in recs.items()}
    bad = copy()
    bad["D0G"][1]["recovery"] = dict(bad["D0G"][1]["recovery"], retried=True)      # different retried set
    with pytest.raises(SC.ExportError):
        SC.verify_exports(bad, plans, ids)
    bad2 = copy()
    bad2["EF"][10]["first_raw_text"] = "regenerated"                                # unchanged prompt not reused
    with pytest.raises(SC.ExportError):
        SC.verify_exports(bad2, plans, ids)
    bad3 = copy()
    bad3["D0V"][5]["first_raw_text"] = "other first answer"                         # retry arm not on D0's answers
    with pytest.raises(SC.ExportError):
        SC.verify_exports(bad3, plans, ids)
    with pytest.raises(SC.ExportError):
        SC.verify_exports({k: v for k, v in recs.items() if k != "D0G"}, plans, ids)


def test_analysis_gating_and_practical_criterion():
    ids, plans, recs = _synthetic_exports(n=400, flag_every=4)
    out = SC.score_all(recs, lambda r: "incorrect" if "city" in r["raw_text"] else "correct")
    a = SC.analyze(out, ids, recs, reps=500)
    assert a["step1"]["reject"] and a["step1"]["practically_meaningful"] and a["step1"]["ci95_pp"][0] >= 1.0
    assert {(r["a"], r["b"]) for r in a["step2"]} == set(SC.STEP2)
    assert a["recovery"]["D0V"]["retried"] == a["recovery"]["D0G"]["retried"] == 100
    assert a["recovery"]["D0V"]["fixed_by_retry"] == 100 and a["recovery"]["D0G"]["fixed_by_retry"] == 0
    flat = {k: {i: "correct" for i in ids} for k in SC.ARMS}
    b = SC.analyze(flat, ids, reps=200)
    assert not b["step1"]["reject"] and b["step2"].startswith("not tested") and b["step3"].startswith("not tested")


def test_practical_criterion_can_fail_while_test_rejects():
    ids = [f"q{i:04d}" for i in range(2000)]
    oc = {k: {i: "incorrect" for i in ids} for k in SC.ARMS}
    for i in ids[:12]:                         # 12 fixes, 0 breaks on n = 2000: p < 0.001 but +0.6 pp
        oc["D0V"][i] = "correct"
    a = SC.analyze(oc, ids, reps=500)
    assert a["step1"]["reject"] and not a["step1"]["practically_meaningful"]


def test_holm_and_mcnemar():
    assert SC.mcnemar_exact(0, 0) == 1.0 and SC.mcnemar_exact(6, 0) == pytest.approx(0.03125)
    fam = [{"p": 0.01}, {"p": 0.04}]
    assert SC.holm(fam) is True and fam[0]["p_holm"] == 0.02 and fam[1]["p_holm"] == 0.04
    fam2 = [{"p": 0.03}, {"p": 0.04}]                   # 0.06 > 0.05: first fails, so the family does not pass
    assert SC.holm(fam2) is False and not fam2[0]["reject"] and not fam2[1]["reject"]
