"""MQL-0010 component unit tests (synthetic data only; no model, no database, no network)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0010/src"))
sys.path.insert(1, str(ROOT / "kaggle/nl2mql_mql0009_retrieval_control/src"))
import patterns as PT        # noqa: E402
import retrieval as RT       # noqa: E402
import structure as S        # noqa: E402
import retriever_v2 as R2    # noqa: E402
import validator as V        # noqa: E402
import prompt_v2 as PV       # noqa: E402
import recovery as RC        # noqa: E402

# ------------------------------------------------------------------ synthetic schema
CTX = {
    "shop__emp": {"db_id": "shop", "table": "emp", "fields": {"emp_id": ["int"], "name": ["string"], "dept_id": ["int"],
                                                              "salary": ["double"]}},
    "shop__dept": {"db_id": "shop", "table": "dept", "fields": {"dept_id": ["double"], "dname": ["string"], "city": ["string"]}},
    "shop__sale": {"db_id": "shop", "table": "sale", "fields": {"emp_id": ["string"], "amount": ["int", "null"]}},
    "other__x": {"db_id": "other", "table": "x", "fields": {"a": ["int"]}},
}
TABLES = {"db_id": "shop", "table_names_original": ["emp", "dept", "sale"],
          "column_names_original": [[-1, "*"], [0, "emp_id"], [0, "name"], [0, "dept_id"], [0, "salary"],
                                    [1, "dept_id"], [1, "dname"], [1, "city"], [2, "emp_id"], [2, "amount"]],
          "foreign_keys": [[3, 5], [8, 1]]}


@pytest.fixture
def val():
    return V.SchemaValidator("shop", CTX)


def codes(issues):
    return sorted(i.code for i in issues)


# ------------------------------------------------------------------ validator (component V)
def test_valid_single_table(val):
    q = {"collection": "shop__emp", "pipeline": [{"$match": {"salary": {"$gt": 10}}}, {"$sort": {"name": 1}},
                                                 {"$project": {"_id": 0, "name": 1}}]}
    assert val.validate(q) == []


def test_unknown_field_suggests_other_collection(val):
    q = {"collection": "shop__emp", "pipeline": [{"$match": {"city": "London"}}, {"$project": {"_id": 0, "name": 1}}]}
    iss = val.validate(q)
    assert codes(iss) == ["V2_unknown_field"] and "shop__dept.city" in iss[0].suggestion


def test_join_alias_paths(val):
    ok = {"collection": "shop__emp", "pipeline": [
        {"$lookup": {"from": "shop__dept", "localField": "dept_id", "foreignField": "dept_id", "as": "d"}},
        {"$unwind": "$d"}, {"$match": {"d.city": "London"}}, {"$project": {"_id": 0, "name": 1, "city": "$d.city"}}]}
    assert val.validate(ok) == []
    bad = json.loads(json.dumps(ok))
    bad["pipeline"][2] = {"$match": {"city": "London"}}       # forgot the alias prefix
    iss = val.validate(bad)
    assert codes(iss) == ["V2_unknown_field"] and "d.city" in iss[0].suggestion


def test_group_and_project_shapes(val):
    ok = {"collection": "shop__emp", "pipeline": [{"$group": {"_id": "$dept_id", "n": {"$sum": 1}}},
                                                  {"$match": {"n": {"$gte": 2}}}, {"$sort": {"n": -1}},
                                                  {"$project": {"_id": 0, "dept": "$_id", "n": 1}}]}
    assert val.validate(ok) == []
    bad = {"collection": "shop__emp", "pipeline": [{"$group": {"_id": "$dept_id", "n": {"$sum": 1}}},
                                                   {"$match": {"salary": {"$gt": 5}}}]}
    assert codes(val.validate(bad)) == ["V2_unknown_field"]
    after_proj = {"collection": "shop__emp", "pipeline": [{"$project": {"_id": 0, "name": 1}}, {"$sort": {"salary": 1}}]}
    assert codes(val.validate(after_proj)) == ["V2_unknown_field"]


def test_expr_operator_inside_query(val):
    q = {"collection": "shop__emp", "pipeline": [{"$match": {"salary": {"$gt": {"$avg": "$salary"}}}}]}
    assert "V3_expr_in_query" in codes(val.validate(q))
    ok = {"collection": "shop__emp", "pipeline": [{"$match": {"$expr": {"$gt": ["$salary", 10]}}}]}
    assert val.validate(ok) == []


def test_fieldpath_string_literal_in_query(val):
    q = {"collection": "shop__emp", "pipeline": [{"$match": {"emp_id": "$dept_id"}}]}
    assert codes(val.validate(q)) == ["V4_fieldpath_literal"]


def test_unknown_collections(val):
    assert codes(val.validate({"collection": "shop__employees", "pipeline": []})) == ["V1_unknown_collection"]
    q = {"collection": "shop__emp", "pipeline": [{"$lookup": {"from": "other__x", "localField": "emp_id",
                                                              "foreignField": "a", "as": "x"}}]}
    assert "V1_unknown_collection" in codes(val.validate(q))
    q2 = {"collection": "shop__emp", "pipeline": [{"$lookup": {"from": "shop__dept", "localField": "dept_id",
                                                               "foreignField": "deptid", "as": "d"}}]}
    assert codes(val.validate(q2)) == ["V2_unknown_field"]


def test_opaque_stages_never_false_alarm(val):
    q = {"collection": "shop__emp", "pipeline": [{"$facet": {"r": [{"$count": "n"}]}}, {"$project": {"anything": "$whatever"}},
                                                 {"$match": {"zzz": 1}}]}
    assert val.validate(q) == []
    q2 = {"collection": "shop__emp", "pipeline": [
        {"$lookup": {"from": "shop__sale", "let": {"e": "$emp_id"}, "pipeline": [{"$match": {"$expr": {"$eq": ["$emp_id", {"$toString": "$$e"}]}}}],
                     "as": "s"}}, {"$match": {"s.amount": {"$gt": 1}}}]}
    assert val.validate(q2) == []


def test_corpus_style_count_wrapper_passes(val):
    q = {"collection": "shop__emp", "pipeline": [{"$match": {"salary": {"$gt": 1}}},
                                                 {"$facet": {"rows": [{"$group": {"_id": None, "a0": {"$sum": 1}}}]}},
                                                 {"$project": {"row": {"$ifNull": [{"$arrayElemAt": ["$rows", 0]}, {"a0": 0}]}}},
                                                 {"$replaceRoot": {"newRoot": "$row"}}, {"$project": {"_id": 0, "count": "$a0"}}]}
    assert val.validate(q) == []


def test_feedback_text(val):
    iss = val.validate({"collection": "shop__emp", "pipeline": [{"$match": {"city": "x"}}]})
    t = V.format_feedback(iss)
    assert "city" in t and t.startswith("Your previous query")


# ------------------------------------------------------------------ FK notes (component F)
def test_fk_notes_numeric_unified_vs_raw():
    raw = PV.fk_type_notes("shop", CTX, TABLES, "raw")
    uni = PV.fk_type_notes("shop", CTX, TABLES, "numeric_unified")
    assert len(raw) == 2                      # int vs double AND string vs int (raw sets differ)
    assert len(uni) == 1 and "shop__sale.emp_id (string)" in uni[0]   # only the genuinely incompatible pair


def test_type_class_ignores_null():
    assert PV.type_class(["int", "null"]) == PV.type_class(["double"]) == frozenset({"number"})


# ------------------------------------------------------------------ diversified selection (component E)
def _doc(i, pats, db="d1"):
    return {"doc_id": f"x{i:03d}", "kind": "example", "db_id": db, "question": f"q {i}", "masked_question": f"q {i}",
            "patterns": pats, "mql": {"i": i}, "content": f"Question: q {i}\nMQL: {{}}"}


def test_select_diverse_distinct_then_fill():
    ranked = [(_doc(1, ["count"]), 3.0, {}), (_doc(2, ["count"]), 2.9, {}), (_doc(3, ["join"]), 2.0, {}),
              (_doc(4, ["count"]), 1.9, {}), (_doc(5, ["order_by"]), 1.0, {})]
    sel = R2.select_diverse(ranked, 3)
    assert [d["doc_id"] for d, _, _ in sel] == ["x001", "x003", "x005"]
    few = ranked[:2]
    assert [d["doc_id"] for d, _, _ in R2.select_diverse(few, 3)] == ["x001", "x002"]
    two_sigs = ranked[:4]
    assert [d["doc_id"] for d, _, _ in R2.select_diverse(two_sigs, 3)] == ["x001", "x002", "x003"]


def test_select_diverse_ignores_where_select_only_if_same_signature():
    a, b = _doc(1, ["select", "where", "count"]), _doc(2, ["select", "count", "where", "text_number"])
    assert S.signature(a["patterns"]) == S.signature(b["patterns"])


# ------------------------------------------------------------------ retriever wrapper (component R)
def _corpus():
    qs = [("how many employees are there", ["count"]), ("count the number of singers", ["count"]),
          ("list names ordered by age", ["order_by"]), ("show names of people older than 30", ["where"]),
          ("which city has the most stores", ["group_by", "order_by", "limit", "count"]),
          ("how many stores per city", ["count", "group_by"])]
    return [{**_doc(i, ["select", *p], db=f"db{i % 3}"), "question": q, "masked_question": q} for i, (q, p) in enumerate(qs)]


INFO = {"db_id": "target", "tables": {}, "schema_words": set(), "text_number_fields": []}


def test_baseline_mode_is_the_unchanged_retriever():
    docs = _corpus()
    base = RT.Retriever(docs, method="hybrid_masked", hybrid_weight=1.0).rank_examples("how many shops are there", INFO, "q1")
    new = R2.StructureAwareRetriever(docs).rank_examples("how many shops are there", INFO, "q1")
    assert [(d["doc_id"], s) for d, s in base] == [(d["doc_id"], s) for d, s, _ in new]


def test_lexical_tiebreak_and_nb_mode_deterministic():
    docs = _corpus()
    tb = R2.StructureAwareRetriever(docs, tie_break="lexical").rank_examples("how many shops are there", INFO, "q1")
    assert len(tb) == len(docs)
    feats = [(S.features(d["question"], None), S.signature(d["patterns"])) for d in docs] * 3
    pred = S.StructurePredictor().fit(feats)
    r = R2.StructureAwareRetriever(docs, predictor=pred, structure_scoring="nb", tie_break="lexical")
    a = [d["doc_id"] for d, _, _ in r.rank_examples("how many shops are there", INFO, "q1")]
    b = [d["doc_id"] for d, _, _ in r.rank_examples("how many shops are there", INFO, "q1")]
    assert a == b and set(a) == {d["doc_id"] for d in docs}
    with pytest.raises(ValueError):
        R2.StructureAwareRetriever(docs, structure_scoring="nb")


def test_target_db_documents_never_returned():
    docs = _corpus()
    info = dict(INFO, db_id="db0")
    for mode in ({}, {"tie_break": "lexical"}):
        out = R2.StructureAwareRetriever(docs, **mode).rank_examples("how many", info, "q")
        assert all(d["db_id"] != "db0" for d, _, _ in out)


# ------------------------------------------------------------------ structure predictor (component S)
def test_structure_predictor_learns_and_roundtrips():
    ex = [(S.features("how many items are there", None), frozenset({"count"})),
          (S.features("count the items", None), frozenset({"count"})),
          (S.features("list the items ordered by price", None), frozenset({"order_by"})),
          (S.features("show items sorted by name", None), frozenset({"order_by"}))] * 5
    m = S.StructurePredictor().fit(ex)
    p = m.predict_proba(S.features("how many shops are there", None))
    assert p["count"] > 0.5 > p["order_by"]
    m2 = S.StructurePredictor.from_json(m.to_json())
    assert m2.predict_proba(S.features("how many shops are there", None)) == p
    assert 0.0 <= S.unsupported_probability(p) <= 1.0


# ------------------------------------------------------------------ recovery (component V, generation side)
GOOD = json.dumps({"collection": "shop__emp", "pipeline": [{"$project": {"_id": 0, "name": 1}}]})
BAD = json.dumps({"collection": "shop__emp", "pipeline": [{"$match": {"city": "x"}}, {"$project": {"_id": 0, "name": 1}}]})


def test_no_retry_when_clean(val):
    calls = []
    out, tr = RC.recover("SYS", "prompt", GOOD, val, lambda sy, m: calls.append(m) or GOOD)
    assert out == GOOD and not tr["retried"] and calls == []


def test_retry_kept_only_if_better(val):
    out, tr = RC.recover("SYS", "prompt", BAD, val, lambda sy, m: GOOD)
    assert out == GOOD and tr["kept"] == "retry" and tr["retried"]
    out2, tr2 = RC.recover("SYS", "prompt", BAD, val, lambda sy, m: BAD)
    assert out2 == BAD and tr2["kept"] == "first"
    out3, tr3 = RC.recover("SYS", "prompt", BAD, val, lambda sy, m: "not json")
    assert out3 == BAD and tr3["kept"] == "first"


def test_retry_messages_and_single_call(val):
    seen = []
    RC.recover("THE SYSTEM", "the prompt", BAD, val, lambda sy, m: seen.append((sy, m)) or GOOD)
    assert len(seen) == 1
    sy, m = seen[0]
    assert sy == "THE SYSTEM"
    assert [x["role"] for x in m] == ["user", "assistant", "user"] and m[0]["content"] == "the prompt" and m[1]["content"] == BAD
    assert "city" in m[2]["content"] and m[2]["content"].endswith(PV.RETRY_SUFFIX)


def test_unparseable_first_answer_gets_format_retry(val):
    out, tr = RC.recover("S", "p", "sorry, I cannot", val, lambda sy, m: GOOD)
    assert out == GOOD and tr["feedback"] == RC.FORMAT_FEEDBACK


def test_disabled_recovery_is_identity(val):
    out, tr = RC.recover("S", "p", BAD, val, lambda sy, m: GOOD, enabled=False)
    assert out == BAD and not tr["retried"]


def test_parser_matches_registered_evaluator():
    sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
    import dev1034_core as D
    for t in (GOOD, "```json\n" + GOOD + "\n```", "Here: " + BAD + " done", "{bad", ""):
        a = RC.parse_output(t)
        b, _ = D.parse_output(t)
        if a is not None:
            assert {k: v for k, v in a.items() if k != "method"} == b


# ------------------------------------------------------------------ generation driver with a fake model
import driver as DR  # noqa: E402


def _plan_rows(ids, sha="h"):
    return [{"id": i, "db_id": "shop", "prompt_sha256": f"{sha}{i}"} for i in ids]


def test_driver_first_pass_and_retry_reuse():
    prompts = {"a": ("SYS", "Qa"), "b": ("SYS", "Qb")}
    outputs = {"Qa": GOOD, "Qb": BAD}
    calls = []

    def gen(system, msgs):
        calls.append((system, len(msgs)))
        return outputs[msgs[0]["content"]] if len(msgs) == 1 else GOOD

    base = DR.first_pass("D0", _plan_rows(["a", "b"]), prompts, gen)
    assert [r["raw_text"] for r in base] == [GOOD, BAD] and len(calls) == 2
    v = DR.retry_arm("D0+V", base, prompts, CTX, gen)
    assert len(calls) == 3 and calls[-1] == ("SYS", 3)          # exactly one retry, only for the flagged item, same system
    assert v[0]["raw_text"] == GOOD and v[0]["recovery"]["retried"] is False and v[0]["source"] == "reused:D0"
    assert v[1]["raw_text"] == GOOD and v[1]["first_raw_text"] == BAD and v[1]["recovery"]["kept"] == "retry"
    assert DR.summarize_recovery(v) == {"n": 2, "retried": 1, "kept_retry": 1, "retry_clean": 1}


def test_driver_prompt_delta_generates_only_changed_prompts():
    prompts = {"a": ("SYS", "Qa"), "b": ("SYS", "Qb-with-note")}
    calls = []
    base_plan = _plan_rows(["a", "b"])
    base = [{"id": "a", "db_id": "shop", "arm": "E", "status": "ok", "raw_text": "A", "first_raw_text": "A", "source": "generated",
             "prompt_sha256": "ha", "recovery": None, "experiment": "MQL-0010"},
            {"id": "b", "db_id": "shop", "arm": "E", "status": "ok", "raw_text": "B", "first_raw_text": "B", "source": "generated",
             "prompt_sha256": "hb", "recovery": None, "experiment": "MQL-0010"}]
    new_plan = [dict(base_plan[0]), dict(base_plan[1], prompt_sha256="CHANGED")]
    out = DR.prompt_delta("E+F", new_plan, base_plan, base, prompts, lambda s, m: calls.append(m) or "NEW")
    assert len(calls) == 1 and out[0]["raw_text"] == "A" and out[0]["source"] == "reused:E" and out[1]["raw_text"] == "NEW"
