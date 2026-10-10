"""Unit tests for the Spider 1.0 comparator, MQL row conversion and ORDER BY analysis."""
import sys
from pathlib import Path

import pytest
from bson import ObjectId

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/spider1"))
import spider1_kit as K  # noqa: E402


def test_multiset_match_ignores_row_and_column_order():
    gold = [("a", 1), ("b", 2)]
    assert K.exec_match([(2, "b"), (1, "a")], gold)[0]


def test_column_count_must_match():
    ok, why = K.exec_match([("a", 1, 9)], [("a", 1)])
    assert not ok and why.startswith("column count")


def test_row_count_and_duplicates_matter():
    assert not K.exec_match([("a",)], [("a",), ("a",)])[0]
    assert not K.exec_match([("a",), ("b",)], [("a",), ("a",)])[0]


def test_numeric_normalisation():
    assert K.exec_match([(3.0,)], [(3,)])[0]
    assert K.exec_match([(10621.66666667,)], [(10621.666666666666,)])[0]
    assert not K.exec_match([(3.5,)], [(3,)])[0]


def test_strings_are_case_sensitive_and_typed():
    assert not K.exec_match([("france",)], [("France",)])[0]
    assert not K.exec_match([("1",)], [(1,)])[0]


def test_ordered_comparison_and_ties():
    gold = [("x", 3), ("y", 2), ("z", 2), ("w", 1)]
    blocks = [0, 1, 1, 2]
    assert K.exec_match(gold, gold, blocks, ordered=True)[0]
    # swapping tied rows is allowed
    assert K.exec_match([("x", 3), ("z", 2), ("y", 2), ("w", 1)], gold, blocks, ordered=True)[0]
    # swapping non-tied rows is not
    assert not K.exec_match([("y", 2), ("x", 3), ("z", 2), ("w", 1)], gold, blocks, ordered=True)[0]
    # without tie info the order must be exact
    assert not K.exec_match([("x", 3), ("z", 2), ("y", 2), ("w", 1)], gold, None, ordered=True)[0]


def test_column_permutation_with_identical_columns():
    gold = [(1, 1), (2, 3)]
    assert K.exec_match([(1, 1), (3, 2)], gold)[0]


def test_docs_to_rows_drops_identity_and_null_group_ids():
    cols, rows = K.docs_to_rows([{"_id": ObjectId(), "a": 1}])
    assert cols == ["a"] and rows == [(1,)]
    cols, rows = K.docs_to_rows([{"_id": None, "n": 5}])
    assert cols == ["n"]
    cols, rows = K.docs_to_rows([{"_id": "France", "n": 2}, {"_id": "UK", "n": 1}])
    assert cols == ["_id", "n"]
    cols, rows = K.docs_to_rows([{"_id": {"a": 1, "b": "x"}, "n": 1}])
    assert cols == ["_id.a", "_id.b", "n"]


def test_docs_to_rows_missing_field_is_null():
    cols, rows = K.docs_to_rows([{"a": 1, "b": 2}, {"a": 3}])
    assert rows == [(1, 2), (3, None)]


@pytest.mark.parametrize("sql,ob,limit,expr", [
    ("SELECT name FROM singer ORDER BY age DESC LIMIT 3", True, 3, "age"),
    ("SELECT count(*) FROM singer", False, None, None),
    ("SELECT a FROM t WHERE x IN (SELECT y FROM u ORDER BY z LIMIT 1)", False, None, None),
    ("SELECT T1.name, count(*) FROM a AS T1 GROUP BY T1.id ORDER BY count(*) DESC LIMIT 1",
     True, 1, "count(*)"),
])
def test_order_info(sql, ob, limit, expr):
    info = K.order_info(sql)
    assert info["order_by"] == ob and info["limit"] == limit and info["order_expr"] == expr


def test_tie_analysis_on_real_db():
    sql = "SELECT name, country, age FROM singer ORDER BY age DESC"
    _, rows = K.run_sql("concert_singer", sql)
    t = K.tie_analysis("concert_singer", sql, rows)
    assert t["status"] == "ok" and len(t["blocks"]) == len(rows)


@pytest.mark.parametrize("msg,secs", [
    ("Please try again in 432ms.", 0.432),
    ("try again in 4m35.616s.", 275.616),
    ("try again in 1h2m3s", 3723.0),
    ("try again in 7.2s", 7.2),
    ("no hint", None),
])
def test_parse_retry_in(msg, secs):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/evaluation"))
    from providers import parse_retry_in
    got = parse_retry_in(msg)
    assert got == pytest.approx(secs) if secs is not None else got is None


# ---- partial-coverage handling (compare.py / error_analysis.py) ----
def _rec(i, correct, cat=None):
    return {"id": i, "result": {"correct": correct, "category": cat or ("correct" if correct else "wrong_values")}}


def test_paired_ids_is_ordered_intersection():
    import compare
    all_ids = ["a", "b", "c", "d"]
    runs = {"full": {i: _rec(i, True) for i in all_ids}, "partial": {"b": _rec("b", False), "d": _rec("d", True)}}
    assert compare.paired_ids(all_ids, runs) == ["b", "d"]


def test_own_coverage_excludes_unevaluated_ids():
    import compare
    cov = compare.own_coverage(["a", "b", "c", "d"], {"a": _rec("a", True), "b": _rec("b", False)})
    assert cov["n_answered"] == 2 and cov["n_not_evaluated"] == 2
    assert cov["correct"] == 1 and cov["accuracy"] == 0.5


def test_mcnemar_exact_matches_binomial():
    import compare
    assert compare.mcnemar_exact(0, 0) == 1.0
    assert compare.mcnemar_exact(7, 43) == pytest.approx(2.0987e-07, rel=1e-3)
    assert compare.mcnemar_exact(5, 5) == pytest.approx(1.0)


def test_error_analysis_skips_provider_errors_and_duplicates():
    import error_analysis
    data = {"a": {}, "b": {}, "c": {}}
    recs = [_rec("a", False), _rec("a", True), _rec("b", False, "provider_error"), _rec("zz", False)]
    out = error_analysis.evaluated_records(recs, data)
    assert [r["id"] for r in out] == ["a"] and out[0]["result"]["correct"] is False


def test_needs_conversion_detection():
    import slice_type_conversion as S
    assert S.needs_conversion({"gold_mql": {"pipeline": [{"$match": {"$expr": {"$eq": [{"$toString": "$x"}, "1"]}}}]}})
    assert not S.needs_conversion({"gold_mql": {"pipeline": [{"$count": "n"}]}})
