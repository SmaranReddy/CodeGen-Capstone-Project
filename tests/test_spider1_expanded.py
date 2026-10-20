"""Integrity tests for the expanded Spider 1.0 NL -> MQL benchmark (no gold MQL; offline)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/spider1"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
import build_expanded as B  # noqa: E402
import report_expanded as REP  # noqa: E402
import spider1_kit as K  # noqa: E402
from run_eval import forbidden_anywhere  # noqa: E402

EXPANDED = ROOT / "data/processed/spider1_nl_mql_expanded.jsonl"
V1 = ROOT / "data/processed/spider1_nl_mql.jsonl"
needs_data = pytest.mark.skipif(not EXPANDED.exists(), reason="expanded dataset not built")


@needs_data
def test_every_pool_item_has_exactly_one_status():
    pool = {r["id"] for r in K.read_jsonl(K.INTERIM / "pool.jsonl")}
    status = K.read_jsonl(K.INTERIM / "expanded/status.jsonl")
    assert [s["id"] for s in status] == sorted(pool, key=lambda i: int(i[-4:]))
    for s in status:
        assert s["status"] in ("included", "excluded")
        if s["status"] == "excluded":
            assert s["stage"] and s["reason"], s["id"]


@needs_data
def test_expanded_contains_v1_unchanged_and_no_gold_mql():
    d = {r["id"]: r for r in K.read_jsonl(EXPANDED)}
    v1 = K.read_jsonl(V1)
    assert len(d) == len(K.read_jsonl(EXPANDED))
    for r in v1:
        e = d[r["id"]]
        for k in ("question", "gold_sql", "gold_rows", "gold_columns", "order_by", "limit", "tie_blocks", "hardness"):
            assert e[k] == r[k], (r["id"], k)
        assert e["in_v1_benchmark"]
    assert all("gold_mql" not in e for e in d.values())
    assert len({(e["db_id"], e["question"]) for e in d.values()}) == len(d)
    status = {s["id"]: s for s in K.read_jsonl(K.INTERIM / "expanded/status.jsonl")}
    assert {i for i, s in status.items() if s["status"] == "included"} == set(d)


@needs_data
def test_new_items_have_audit_decisions():
    audit = K.load_json(K.INTERIM / "expanded/audit_new_items.json")["items"]
    d = {r["id"]: r for r in K.read_jsonl(EXPANDED)}
    for iid, a in audit.items():
        assert (iid in d) == (a["decision"] == "include"), iid
        if a["kind"] == "ambiguous":
            assert d[iid]["flags"] == ["ambiguous"]


def test_forbidden_operators_found_at_any_depth():
    assert forbidden_anywhere([{"$match": {"a": 1}}]) is None
    assert forbidden_anywhere([{"$addFields": {"x": {"$function": {"body": "f", "args": [], "lang": "js"}}}}]) == "$function"
    assert forbidden_anywhere([{"$lookup": {"from": "c", "pipeline": [{"$out": "t"}], "as": "y"}}]) == "$out"
    assert forbidden_anywhere([{"$match": {"$where": "1"}}]) == "$where"


def test_utf8_only_mismatch_rule():
    ok_tab = {"match": True}
    utf8 = {"match": False, "sqlite_rows": 5, "atlas_docs": 5, "rows_only_in_sqlite": 0, "rows_only_in_atlas": 0,
            "extra_fields": {}, "missing_fields": {}, "utf8_replaced_strings": 2}
    assert B.utf8_only_mismatch({"tables": {"a": ok_tab, "b": utf8}, "atlas_collections_without_table": []})
    lost = dict(utf8, atlas_docs=4)
    assert not B.utf8_only_mismatch({"tables": {"b": lost}, "atlas_collections_without_table": []})
    assert not B.utf8_only_mismatch({"tables": {"a": ok_tab}, "atlas_collections_without_table": []})


def test_wilson_and_mcnemar():
    lo, hi = REP.wilson(50, 100)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-9
    assert REP.wilson(0, 10)[0] == 0.0
    assert REP.mcnemar_exact(0, 0) == 1.0
    assert REP.mcnemar_exact(10, 10) == 1.0
    assert REP.mcnemar_exact(0, 10) < 0.01


def test_outcome_buckets():
    rec = {"result": {"category": "format_error"}, "stop_reason": "length", "output_tokens": 10}
    assert REP.outcome(rec, 1024) == "malformed_output_truncated"
    assert REP.outcome({"result": {"category": "timeout"}}, 1024) == "timeout"
    assert REP.outcome({"result": {"category": "wrong_values"}}, 1024) == "result_mismatch"
    assert REP.outcome({"result": {"category": "invalid_mql"}}, 1024) == "validation_rejected"
    json.dumps(REP.wilson(3, 7))
