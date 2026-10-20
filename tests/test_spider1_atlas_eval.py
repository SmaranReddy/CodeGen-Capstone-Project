"""Atlas execution + result validation path (run_eval.score_output / run_one) and the
saved-prediction verifier (rescore.py), with a mocked MongoDB -- no network, no model calls."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/spider1"))
import rescore  # noqa: E402
import run_eval  # noqa: E402

EXEC = {"timeout_ms": 30000, "max_result_rows": 100000}
ALLOWED = {"shop__item", "shop__order"}


class FakeCollection:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def aggregate(self, pipeline, **kw):
        self.db.calls.append((self.name, pipeline, kw))
        if isinstance(self.db.result, Exception):
            raise self.db.result
        return iter(self.db.result)


class FakeDB:
    def __init__(self, result):
        self.result, self.calls = result, []

    def __getitem__(self, name):
        return FakeCollection(self, name)


def ex(gold_rows, order_by=False, tie_blocks=None):
    return {"id": "x1", "db_id": "shop", "hardness": "easy", "question": "q",
            "gold_rows": gold_rows, "order_by": order_by, "tie_blocks": tie_blocks}


def mql(pipeline, coll="shop__item"):
    return json.dumps({"collection": coll, "pipeline": pipeline})


def test_correct_result_executes_on_the_named_collection_with_timeout():
    db = FakeDB([{"_id": None, "n": 3}])
    res, extra = run_eval.score_output(db, mql([{"$count": "n"}]), ex([(3,)]), EXEC, ALLOWED)
    assert res["correct"] and res["category"] == "correct" and res["executed"]
    assert db.calls[0][0] == "shop__item" and db.calls[0][2]["maxTimeMS"] == 30000
    assert extra["mql"]["pipeline"] == [{"$count": "n"}]


def test_different_query_text_with_equivalent_result_is_correct():
    gold = ex([("a", 2), ("b", 1)])
    a = run_eval.score_output(FakeDB([{"name": "a", "c": 2}, {"name": "b", "c": 1}]),
                              mql([{"$group": {"_id": "$name", "c": {"$sum": 1}}}]), gold, EXEC, ALLOWED)
    b = run_eval.score_output(FakeDB([{"cnt": 1, "nm": "b"}, {"cnt": 2, "nm": "a"}]),
                              mql([{"$sortByCount": "$name"}, {"$project": {"nm": "$_id", "cnt": "$count"}}]),
                              gold, EXEC, ALLOWED)
    assert a[0]["correct"] and b[0]["correct"]


def test_malformed_output_is_not_executed():
    db = FakeDB([])
    res, _ = run_eval.score_output(db, '{"collection":"shop__item","pipeline":[{"$count":"n"}]]}',
                                   ex([(1,)]), EXEC, ALLOWED)
    assert res["category"] == "format_error" and not res["executed"] and not db.calls
    assert run_eval.score_output(db, None, ex([(1,)]), EXEC, ALLOWED)[0]["category"] == "format_error"


@pytest.mark.parametrize("text", [
    mql([{"$out": "evil"}]),
    mql([{"$count": "n"}], coll="other__table"),
    mql([{"$lookup": {"from": "other__t", "localField": "a", "foreignField": "b", "as": "j"}}]),
    mql([{"$unionWith": "other__t"}]),
    mql([{"$unionWith": ""}]),
    mql([{"$unionWith": {"coll": "other__t", "pipeline": []}}]),
])
def test_invalid_mql_is_rejected_before_execution(text):
    db = FakeDB([])
    res, _ = run_eval.score_output(db, text, ex([(1,)]), EXEC, ALLOWED)
    assert res["category"] == "invalid_mql" and not res["executed"] and not db.calls


def test_server_error_is_an_execution_failure():
    from pymongo.errors import OperationFailure
    res, _ = run_eval.score_output(FakeDB(OperationFailure("bad $group")), mql([{"$group": {}}]),
                                   ex([(1,)]), EXEC, ALLOWED)
    assert res["category"] == "execution_error" and not res["executed"] and not res["correct"]


def test_oversized_result_is_an_execution_failure_and_maps_to_that_bucket():
    res, _ = run_eval.score_output(FakeDB([{"v": i} for i in range(5)]), mql([{"$match": {}}]),
                                   ex([(1,)]), {**EXEC, "max_result_rows": 3}, ALLOWED)
    assert res["category"] == "execution_error" and res["executed"]
    assert rescore.bucket({"id": "x", "result": res}, 1024) == "execution_failure"


@pytest.mark.parametrize("docs, gold, order_by, cat", [
    ([{"v": 1}, {"v": 2}], [(1,)], False, "wrong_row_count"),
    ([{"v": 1, "w": 2}], [(1,)], False, "wrong_column_count"),
    ([{"v": 9}], [(1,)], False, "wrong_values"),
    ([{"v": 2}, {"v": 1}], [(1,), (2,)], True, "order_mismatch"),
])
def test_result_mismatches_are_categorised(docs, gold, order_by, cat):
    res, extra = run_eval.score_output(FakeDB(docs), mql([{"$match": {}}]), ex(gold, order_by), EXEC, ALLOWED)
    assert res["executed"] and not res["correct"] and res["category"] == cat
    assert "pred_preview" in extra
    assert rescore.bucket({"id": "x", "result": res}, 1024) == "result_mismatch"


def test_run_one_provider_error_is_never_executed(monkeypatch):
    class Down:
        def generate(self, *a):
            return {"text": None, "error": "HTTP 503"}
    monkeypatch.setattr(run_eval.P, "messages", lambda db, q: [])
    db = FakeDB([])
    rec = run_eval.run_one(db, Down(), ex([(1,)]), {}, EXEC, ALLOWED)
    assert rec["result"]["category"] == "provider_error" and not db.calls
    assert rescore.bucket(rec, 1024) == "missing_prediction"


def test_run_one_records_mql_and_verdict(monkeypatch):
    class Model:
        def generate(self, *a):
            return {"text": mql([{"$count": "n"}]), "error": None, "model_id": "m", "latency_s": 0.1}
    monkeypatch.setattr(run_eval.P, "messages", lambda db, q: [])
    rec = run_eval.run_one(FakeDB([{"n": 3}]), Model(), ex([(3,)]), {}, EXEC, ALLOWED)
    assert rec["result"]["correct"] and rec["mql"]["collection"] == "shop__item"


def test_bucket_splits_truncated_from_malformed_json():
    fe = {"correct": False, "category": "format_error", "executed": False}
    assert rescore.bucket({"id": "a", "result": fe, "output_tokens": 1024}, 1024) == "malformed_truncated"
    assert rescore.bucket({"id": "a", "result": fe, "stop_reason": "length", "output_tokens": 5}, 1024) == "malformed_truncated"
    assert rescore.bucket({"id": "a", "result": fe, "stop_reason": "stop", "output_tokens": 200}, 1024) == "malformed_json"
    assert rescore.bucket(None, 1024) == "missing_prediction"


def _saved(i, cat, correct=False, **kw):
    r = {"id": i, "model_id": "m", "prompt_sha256": "h", "output_tokens": 10, "raw_text": "{}",
         "result": {"correct": correct, "category": cat, "executed": cat not in ("format_error",)}}
    if correct:
        r["mql"] = {"collection": "c", "pipeline": []}
    r.update(kw)
    return r


def test_audit_strict_denominators_and_paired_subset(tmp_path, monkeypatch):
    data = {i: {"id": i, "gold_rows": [(1,)], "order_by": False, "tie_blocks": None} for i in "abcd"}
    full = [_saved("a", "correct", True), _saved("b", "correct", True), _saved("c", "wrong_values"),
            _saved("d", "execution_error"), _saved("zz", "correct", True)]  # zz: outside dataset
    part = [_saved("a", "correct", True), _saved("b", "wrong_row_count"), _saved("c", "provider_error")]
    for lab, rows in (("full", full), ("part", part)):
        (tmp_path / lab).mkdir()
        (tmp_path / lab / "executions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(rescore, "RUNS", tmp_path)
    rep, _ = rescore.audit(["full", "part"], data, {"dataset": "d", "generation": {"max_new_tokens": 1024}})
    f, p = rep["runs"]["full"], rep["runs"]["part"]
    assert (f["strict_full"]["correct"], f["strict_full"]["n"]) == (2, 4)
    assert f["integrity"]["outside_dataset"] == 1
    assert (p["strict_full"]["correct"], p["strict_full"]["n"]) == (1, 4)   # unanswered = fail
    assert (p["strict_answered"]["correct"], p["strict_answered"]["n"]) == (1, 2)
    assert p["buckets"]["missing_prediction"] == 2 and p["buckets"]["result_mismatch"] == 1
    assert rep["paired_ids"] == ["a", "b"] and rep["paired"]["runs"]["full"]["correct"] == 2


def test_offline_recheck_flags_a_saved_mismatch_that_actually_matches():
    data = {"a": {"gold_rows": [(1,)], "order_by": False, "tie_blocks": None}}
    rec = _saved("a", "wrong_values", pred_preview=[[1]])
    rec["result"]["n_rows"] = 1
    out = rescore.offline_recheck({"a": rec}, data)
    assert out["mismatches_rescored"] == 1 and out["disagreements"][0]["now"] == "correct"
