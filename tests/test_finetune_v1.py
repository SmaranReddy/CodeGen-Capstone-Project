"""Tests for the ft-v1 fine-tuning dataset expansion (scripts/finetune_v1, data/finetune/ft_v1).

Offline: no Atlas access is needed. Unit tests cover CSV repair, parsing, grounding, the
converter and fixture adequacy; artifact tests check the built dataset (skipped if absent).
"""
import collections
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/finetune_v1"))

import nk_common as C  # noqa: E402
import extract as X  # noqa: E402
import fixtures as F  # noqa: E402
import grounding as G  # noqa: E402
import sql2mql as M  # noqa: E402
import sqlsem as S  # noqa: E402

OUT = C.OUT_DIR
HAVE_BUILD = (OUT / "train.jsonl").exists()
needs_build = pytest.mark.skipif(not HAVE_BUILD, reason="ft-v1 not built")

# pinned hashes of the fixed benchmark (from data/rag/corpus_v1/corpus_manifest.json)
BENCH_SHA = "59b6c1f3b537974cc2b400ed27a1e7926401866aae7df47f18081d5719d18134"
SPLIT_MANIFEST_SHA = "a6bd1ca9efb2e4009049d4aaaccd80e6c3853c8edc02359e08c3af891352fc67"


def load(name):
    return C.read_jsonl(OUT / name)


# ---------------------------------------------------------------- raw data / repair
def test_raw_files_pinned_and_unchanged():
    for name, sha in C.RAW_SHA256.items():
        assert C.sha256_file(C.RAW_DIR / name) == sha


def test_repair_unquoted_comma_prompt():
    q, s, rep = X.repair(["Find users with a specific role",
                          "active status, and created after a date.,SELECT * FROM users WHERE a = 1;"])
    assert q == "Find users with a specific role, active status, and created after a date."
    assert s == "SELECT * FROM users WHERE a = 1"
    assert rep == ["unquoted_comma_in_prompt"]


def test_repair_stray_quote_and_escapes():
    q, s, rep = X.repair(["Q", 'SELECT role_id, \\"order\\" FROM t;"'])
    assert s == 'SELECT role_id, "order" FROM t'
    assert set(rep) == {"backslash_escaped_quotes", "stray_trailing_quote"}


def test_dml_is_not_repaired_into_select():
    _, s, rep = X.repair(["Delete a staff member", "DELETE FROM staff WHERE id = 3;"])
    assert s.startswith("DELETE") and "unquoted_comma_in_prompt" not in rep


# ---------------------------------------------------------------- parsing / analysis
def test_parse_records_dialect_and_rejects_dml():
    ast, d, err = S.parse_sql("SELECT * FROM t WHERE created_at >= NOW() - INTERVAL '7 days'")
    assert ast is not None and d == "mysql" and err is None
    ast, d, _ = S.parse_sql("DELETE FROM t WHERE id = 1")
    assert S.analyze(ast).unsupported == ["statement:delete"]


def test_analysis_flags_unsupported_and_truncated():
    a = S.analyze(S.parse_sql("SELECT * FROM p WHERE price > (SELECT AVG(price) FROM p)")[0])
    assert "subquery" in a.unsupported
    a = S.analyze(S.parse_sql("SELECT name FROM assets WHERE asset")[0])
    assert any("bare_non_boolean_column" in u for u in a.unsupported)


def test_sqlite_render_uses_mysql_semantics():
    sql = S.to_sqlite(S.parse_sql("SELECT a / b FROM t WHERE d >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) AND f = TRUE")[0])
    assert "_DTADD(_CURDATE(), -30, 'day')" in sql and "CAST(" in sql and "= 1" in sql


# ---------------------------------------------------------------- grounding
def test_grounding_rejects_changed_value():
    fails = G.check("Get pharmacies activated by 'Pharmacist D'",
                    "SELECT * FROM pharmacies WHERE activated_by = 'Pharmacist B'",
                    [{"kind": "str", "value": "Pharmacist B", "op": "eq", "column": "activated_by"}], [])
    assert any("Pharmacist B" in f for f in fails) and any("Pharmacist D" in f for f in fails)


def test_grounding_rejects_invented_literal():
    fails = G.check("Find users with a specific role", "SELECT * FROM users WHERE roles_id = 789",
                    [{"kind": "int", "value": 789, "op": "eq", "column": "roles_id"}], [])
    assert fails


def test_grounding_accepts_date_formats_and_periods():
    q = "Show me the orders placed on 27-06-2023 in the last week"
    assert G.date_grounded("2023-06-27", q)
    assert G.number_grounded(7, q, {"role": "interval", "unit": "day"})


def test_structure_check_catches_shifted_label():
    sql = "SELECT name, is_default FROM event_staff_roles"
    ast = S.parse_sql(sql)[0]
    a = S.analyze(ast)
    fails = G.structure_check("Get the names and addresses of organizations located in a country",
                              ast, a, {"event_staff_roles": {"name": "text", "is_default": "bool"}})
    assert any(f.startswith("table_not_in_question") for f in fails)


def test_placeholders_detected():
    assert G.placeholders("SELECT * FROM t WHERE a = '<city>' AND b = 'your_user_id'")


# ---------------------------------------------------------------- converter
def conv(sql, schema):
    ast = S.parse_sql(sql)[0]
    return M.convert(ast, S.analyze(ast), schema)


def test_converter_null_semantics_and_output_contract():
    m = conv("SELECT name FROM t WHERE status <> 'x' AND city NOT IN ('a', 'b')",
             {"t": {"name": "text", "status": "text", "city": "text"}})
    assert set(m) == {"collection", "pipeline"} and m["collection"] == "nl2sqlkaggle__t"
    match = m["pipeline"][0]["$match"]["$and"]
    assert {"status": {"$nin": ["x", None]}} in match
    assert {"city": {"$nin": ["a", "b", None]}} in match
    assert m["pipeline"][-1] == {"$project": {"_id": 0, "name": "$name"}}


def test_converter_like_regex_and_clock():
    assert M.like_regex("Safety%") == "^Safety"
    assert M.like_regex("%a.b%") == "a\\.b"
    m = conv("SELECT * FROM t WHERE created_at >= NOW() - INTERVAL 7 DAY", {"t": {"created_at": "datetime"}})
    s = json.dumps(m)
    assert "$$NOW" in s and "$dateSubtract" in s and '"$ne": ["$created_at", null]' in s


def test_converter_output_passes_benchmark_validator():
    import eval_kit as E
    m = conv("SELECT a.x, COUNT(*) FROM a JOIN b ON a.id = b.aid GROUP BY a.x ORDER BY COUNT(*) DESC LIMIT 3",
             {"a": {"id": "int", "x": "text"}, "b": {"aid": "int"}})
    ok, why, _ = E.validate_mql(m, {"nl2sqlkaggle__a", "nl2sqlkaggle__b"})
    assert ok, why
    bad = {"collection": "nl2sqlkaggle__a", "pipeline": [{"$out": "x"}]}
    assert not E.validate_mql(bad, {"nl2sqlkaggle__a"})[0]


# ---------------------------------------------------------------- fixtures
def test_fixture_set_is_adequate_and_deterministic():
    sql = "SELECT * FROM t WHERE price > 10 AND city = 'Paris' ORDER BY price DESC LIMIT 3"
    ast = S.parse_sql(sql)[0]
    an = S.analyze(ast).to_json()
    sch = {"t": {"price": "float", "city": "text", "name": "text"}}
    a = F.adequate_set(ast, an, sch, 7, F.BASE_CLOCK)
    b = F.adequate_set(ast, an, sch, 7, F.BASE_CLOCK)
    assert not a["uncovered"] and len(a["kept"]) >= F.MIN_VARIANTS
    assert [v["rows"] for v in a["kept"]] == [v["rows"] for v in b["kept"]]
    assert any(n.startswith("where:opswap") for n in a["need"])  # > vs >= must be witnessed


def test_duplicate_atom_is_an_equivalent_mutant():
    ast = S.parse_sql("SELECT * FROM t WHERE a = 1 AND a = 1")[0]
    assert not [m for m, _ in S.mutants(ast) if m.startswith("where:drop")]


def test_compare_fast_path_never_looser_than_exec_match():
    import spider1_kit as K
    gold = [(1, "a"), (2, "b")]
    pred_swapped_values = [(1, "b"), (2, "a")]
    assert not F.compare(["x", "y"], pred_swapped_values, ["x", "y"], gold, False, None)[0]
    assert not K.exec_match(pred_swapped_values, gold)[0]
    assert F.compare(["y", "x"], [("b", 2), ("a", 1)], ["x", "y"], gold, False, None)[0]
    assert not F.compare(["x", "y"], [(2, "b"), (1, "a")], ["x", "y"], gold, True, None)[0]


# ---------------------------------------------------------------- built dataset
@needs_build
def test_record_integrity():
    train, dev = load("train.jsonl"), load("dev_internal.jsonl")
    ids = [r["id"] for r in train + dev]
    assert len(ids) == len(set(ids))
    for r in train + dev:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert json.loads(r["messages"][-1]["content"]) == r["target_mql"]
        assert set(r["target_mql"]) == {"collection", "pipeline"}
        assert r["question"] in r["messages"][1]["content"]
        h = r["hashes"]
        assert h["messages_sha256"] == C.sha_obj(r["messages"])
        assert h["record_sha256"] == C.sha_obj({k: v for k, v in r.items() if k != "hashes"})


@needs_build
def test_reference_sql_and_results_not_in_model_input():
    for r in load("train.jsonl") + load("dev_internal.jsonl"):
        prompt = r["messages"][0]["content"] + r["messages"][1]["content"]
        if r["reference"].get("sql"):
            assert r["reference"]["sql"] not in prompt
        assert not G.SQL_IN_TEXT.search(r["messages"][1]["content"])
        for v in r["validation"].get("fixtures", {}).get("variants", []):
            assert v["expected_rows_sha256"] not in prompt


@needs_build
def test_dedup_and_near_dup_cap():
    train, dev = load("train.jsonl"), load("dev_internal.jsonl")
    new = [r for r in train + dev if r["source_dataset"] == C.SOURCE_ID]
    keys = [(C.loose(r["question"]), C.canon(r["target_mql"])) for r in new]
    assert len(keys) == len(set(keys))
    cl = collections.Counter(r["reference"]["near_dup_cluster"] for r in new)
    assert max(cl.values()) <= 3
    assert not ({r["reference"]["near_dup_cluster"] for r in train if r["source_dataset"] == C.SOURCE_ID}
                & {r["reference"]["near_dup_cluster"] for r in dev})
    # exact duplicate questions with conflicting targets are not both trained on
    by_q = collections.defaultdict(set)
    for r in new:
        by_q[C.loose(r["question"])].add(C.canon(r["target_mql"]))
    assert all(len(v) == 1 for v in by_q.values())


@needs_build
def test_schema_handling():
    schema = json.loads((C.INTERIM / "schema.json").read_text(encoding="utf-8"))["tables"]
    assert all(t in S.SQLITE_TYPES for cols in schema.values() for t in cols.values())
    for r in load("train.jsonl"):
        if r["source_dataset"] != C.SOURCE_ID:
            continue
        colls = set(r["context"]["collections"]) | set(r["context"]["distractor_collections"])
        assert r["target_mql"]["collection"] in r["context"]["collections"]
        for c in colls:
            assert f"- {c}\n" in r["messages"][1]["content"]
        assert not set(r["context"]["collections"]) & set(r["context"]["distractor_collections"])


@needs_build
def test_leakage_prevention():
    import common as RC
    rep = json.loads((OUT / "leakage_report.json").read_text(encoding="utf-8"))
    assert rep["rag_leakage_check_errors"] == [] and rep["extra_check_errors"] == []
    db_part, _ = RC.load_split()
    bench = C.read_jsonl(C.BENCHMARK)
    held_q = {C.loose(b["question"]) for b in bench if db_part[b["db_id"]] != "train"}
    held_dbs = {d for d, p in db_part.items() if p != "train"}
    for r in load("train.jsonl") + load("dev_internal.jsonl"):
        assert C.loose(r["question"]) not in held_q
        assert r["db_id"] not in held_dbs
        assert not r["target_mql"]["collection"].split("__")[0] in held_dbs


@needs_build
def test_benchmark_artifacts_unchanged():
    assert C.sha256_file(C.BENCHMARK) == BENCH_SHA
    assert C.sha256_file(C.SPLIT_DIR / "split_manifest.json") == SPLIT_MANIFEST_SHA
    counts = {p: len(C.read_jsonl(C.SPLIT_DIR / f"{p}.jsonl")) for p in ("train", "validation", "test")}
    assert counts == {"train": 318, "validation": 72, "test": 65}


@needs_build
def test_validation_status_consistency():
    for r in load("train.jsonl") + load("dev_internal.jsonl"):
        v = r["validation"]
        if r["source_dataset"] == C.SOURCE_ID:
            assert v["status"] == "fixture_equivalence_verified"
            assert v["fixtures"]["uncovered"] == [] and len(v["execution"]) >= F.MIN_VARIANTS
            assert all(e["match"] for e in v["execution"]) and v["validator"]["ok"]
        else:
            assert v["status"] == "execution_verified_source_db"
    for r in load("candidates_unverified.jsonl"):
        assert r["status"] != "fixture_equivalence_verified" and r["reasons"]
    for r in load("rejected.jsonl"):
        assert r["reasons"]
    st = load("status_manifest.jsonl")
    pairs = C.read_jsonl(C.INTERIM / "pairs.jsonl")
    assert len(st) == len(pairs) == len({s["pair_id"] for s in st})
    rows = sum(len(s["source_record_ids"]) for s in st)
    s1 = json.loads((C.INTERIM / "stage1_counts.json").read_text(encoding="utf-8"))
    assert rows == s1["raw_rows"] - s1["rows_empty"]


@needs_build
def test_build_manifest_reproducible():
    man = json.loads((OUT / "build_manifest.json").read_text(encoding="utf-8"))
    for f, sha in man["outputs_sha256"].items():
        assert C.sha256_file(OUT / f) == sha, f
    assert man["inputs"]["pairs.jsonl"] == C.sha256_file(C.INTERIM / "pairs.jsonl")
    assert man["inputs"]["raw"] == C.RAW_SHA256
    c = man["counts"]
    assert c["train_total"] == len(load("train.jsonl"))
    assert c["new_train"] + c["spider_train_existing"] == c["train_total"]


@needs_build
def test_every_training_target_passes_readonly_validator():
    import eval_kit as E
    for r in load("train.jsonl") + load("dev_internal.jsonl"):
        m = r["target_mql"]
        allowed = set(r["context"]["collections"] or []) or {m["collection"]} | E.referenced_collections(m["pipeline"])
        ok, why, _ = E.validate_mql({"method": "aggregate", **m}, allowed)
        assert ok, (r["id"], why)
        assert not any(name in E.FORBIDDEN_STAGES for name, _ in E.iter_stages(m["pipeline"]))


@needs_build
def test_target_text_preserves_key_order():
    """The assistant string IS the target; key order matters ($sort priority)."""
    schema = json.loads((C.INTERIM / "schema.json").read_text(encoding="utf-8"))["tables"]
    for r in load("train.jsonl") + load("dev_internal.jsonl"):
        content = r["messages"][-1]["content"]
        assert C.compact(json.loads(content)) == content
        assert C.compact(r["target_mql"]) == content
        if r["source_dataset"] == C.SOURCE_ID:
            ast = S.parse_sql(r["reference"]["sql"])[0]
            an = S.analyze(ast)
            regen = M.convert(ast, an, {t: schema[t] for t in an.to_json()["columns"]})
            assert C.compact(regen) == content, r["id"]


def test_multikey_sort_follows_order_by():
    m = conv("SELECT zeta, alpha FROM t ORDER BY zeta DESC, alpha ASC",
             {"t": {"zeta": "int", "alpha": "int"}})
    sort = next(s["$sort"] for s in m["pipeline"] if "$sort" in s)
    assert list(sort) == ["zeta", "alpha"]
    assert list(json.loads(C.compact(m))["pipeline"][0]["$sort"]) == ["zeta", "alpha"]


@needs_build
def test_replay_of_every_target_passed():
    path = OUT / "replay_report.json"
    assert path.exists(), "run scripts/finetune_v1/replay_targets.py"
    rep = json.loads(path.read_text(encoding="utf-8"))
    assert rep["train_sha256"] == C.sha256_file(OUT / "train.jsonl")
    assert rep["dev_sha256"] == C.sha256_file(OUT / "dev_internal.jsonl")
    assert rep["failures"] == []
    assert sum(s["n"] for s in rep["summary"].values()) == len(load("train.jsonl")) + len(load("dev_internal.jsonl"))
