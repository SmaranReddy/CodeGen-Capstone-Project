"""Corpus v2 sign-off helpers: nonconformant-field usage classification and the Atlas audit's offline gates.

No network: nothing here connects to Atlas or to the local MongoDB.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/corpus_v2_signoff"))

import adjudicate_nonconformant as AN  # noqa: E402
import atlas_exec_audit as AX  # noqa: E402


def kinds(pipeline, field):
    return [(u["op"], u["kind"]) for u in AN.usage(pipeline, field)]


def test_bare_projection_is_type_neutral():
    assert kinds([{"$project": {"_id": 0, "p": "$price"}}], "price") == [("$project", "projection_only")]
    assert kinds([{"$match": {"other": 1}}, {"$project": {"p": "$price"}}], "price") == [("$project", "projection_only")]


def test_filter_group_sort_and_expressions_are_type_dependent():
    assert kinds([{"$match": {"price": {"$gt": 3}}}], "price") == [("$match", "type_dependent")]
    assert kinds([{"$group": {"_id": {"g": {"$ifNull": ["$price", None]}}}}], "price") == [("$group", "type_dependent")]
    assert kinds([{"$sort": {"price": 1}}], "price") == [("$sort", "type_dependent")]
    assert kinds([{"$project": {"p": {"$add": ["$price", 1]}}}], "price") == [("$project", "type_dependent")]
    assert kinds([{"$group": {"_id": None, "a": {"$avg": "$price"}}}], "price") == [("$group", "type_dependent")]


def test_other_fields_are_not_confused_with_the_field():
    assert kinds([{"$match": {"price_range": 1}}, {"$sort": {"prices": 1}}], "price") == []


def test_safety_gate_rejects_writes_and_javascript():
    ok = {"collection": "c", "pipeline": [{"$match": {"a": 1}}]}
    assert AX.safety(ok)["ok"]
    for bad in ([{"$out": "x"}], [{"$merge": {"into": "x"}}], [{"$match": {"$where": "1"}}],
                [{"$project": {"a": {"$function": {"body": "f", "args": [], "lang": "js"}}}}]):
        assert not AX.safety({"collection": "c", "pipeline": bad})["ok"], bad


def test_collections_read_includes_lookup_and_union_targets():
    mql = {"collection": "a", "pipeline": [{"$lookup": {"from": "b", "localField": "x", "foreignField": "y", "as": "j"}},
                                           {"$unionWith": {"coll": "c", "pipeline": []}}]}
    assert AX.collections_read(mql) == ["a", "b", "c"]


def test_adjudication_log_never_rewrites_and_is_labelled_ai():
    p = ROOT / "data/interim/corpus_v2_adjudication_v1.jsonl"
    rows = [json.loads(x) for x in p.open(encoding="utf-8")]
    assert len(rows) == 6 and {r["decision"] for r in rows} <= {"INCLUDE", "QUARANTINE"}
    assert all(r["sql_rewritten"] is False and r["mql_rewritten"] is False for r in rows)
    assert all("not human-reviewed" in r["review"] for r in rows)
    for r in rows:   # the rule: any type-dependent use means quarantine
        dep = any(u["kind"] == "type_dependent" for f in r["nonconformant_fields"] for u in f["usage_in_mql"])
        assert (r["decision"] == "QUARANTINE") == dep, r["id"]


# ---------------------------------------------------------------- approval: a missing or failed validation never passes
import approve_corpus as AP  # noqa: E402


def _pair():
    mql = {"collection": "d__t", "pipeline": [{"$match": {"a": 1}}]}
    c = {"id": "x", "db_id": "d", "sql": "SELECT 1", "mql": mql, "type_contract": {"manifest_sha256": "T"},
         "v2_1_code_freeze_sha256": "F"}
    checks = {k: "ok" for k in AP.CHECK_KEYS}
    checks.update(l1_explicit_null="multiset", l1_missing="multiset", v3_reorder_reversed="multiset",
                  v3_reorder_shuffled_missing="ordered match (tie-aware)", v6_determinism="identical documents",
                  fixtures={"empty": "multiset", "dup": "skipped: degenerate: no row changed"})
    v = {"outcome": "accepted", "conversion": "converted", "safety": {"ok": True}, "mql": json.dumps(mql),
         "sql": "SELECT 1", "db_id": "d", "type_contract": {"manifest_sha256": "T"}, "v2_1_code_freeze_sha256": "F",
         "verification": {"verifier": AP.VERIFIER, "status": "accepted", "checks": checks}}
    return c, v


def test_complete_local_validation_is_accepted():
    c, v = _pair()
    assert AP.record_problems(c, v, "T", "F") == []


def test_missing_or_incomplete_validation_is_never_success():
    c, v = _pair()
    assert AP.record_problems(c, None, "T", "F") == ["no verification record"]
    for mutate in (lambda v: v.update(outcome="conversion_rejected"),
                   lambda v: v["safety"].update(ok=False),
                   lambda v: v["verification"].update(verifier="sql2mql-v2-verifier-3"),
                   lambda v: v["verification"].update(status="verifier_error"),
                   lambda v: v["verification"]["checks"].pop("v31_fold_traps"),
                   lambda v: v["verification"]["checks"].update(l1_missing="MISMATCH"),
                   lambda v: v["verification"]["checks"].update(v6_determinism="differs"),
                   lambda v: v["verification"]["checks"]["fixtures"].update(empty="MISMATCH"),
                   lambda v: v["verification"]["checks"].update(fixtures={"dup": "skipped: degenerate"}),
                   lambda v: v.update(mql=json.dumps({"collection": "d__t", "pipeline": [{"$match": {"a": 2}}]})),
                   lambda v: v.update(sql="SELECT 2"),
                   lambda v: v.update(v2_1_code_freeze_sha256="other"),
                   lambda v: v["type_contract"].update(manifest_sha256="other")):
        c, v = _pair()
        mutate(v)
        assert AP.record_problems(c, v, "T", "F"), "a defective validation record was accepted"


def test_approval_refuses_to_overwrite():
    src = (ROOT / "scripts/corpus_v2_signoff/approve_corpus.py").read_text(encoding="utf-8")
    assert "refusing to overwrite an approved artifact" in src and "provisional_corpus_v2" in src
    assert AP.OUT.name == "approved_corpus_v2_1" and AP.OUT != AP.CAND and AP.OUT != AP.SRC_DIR
