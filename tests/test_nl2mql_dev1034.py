"""MQL-0006 (zero-shot Qwen2.5-Coder-3B-Instruct only, NL->MQL, Spider dev minus held-out DBs): bundle
integrity, input isolation, safety gate, comparator, reference handling, timeouts, checkpoints/resume, Groq client,
mocked Atlas preflight, seed workflow, the Qwen-only, generate-only Kaggle notebook run end-to-end with a fake
generator, then the Windows-side predictions verifier, local Atlas scoring (fake Atlas) and scoring verifier.
No model, no GPU, no network, no Atlas."""
import ast
import contextlib
import io
import json
import re
import shutil
import sqlite3
import sys
import types
import urllib.error
import zipfile
from collections import Counter
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
import dev1034_core as C  # noqa: E402
import prepare_spider_dev_1034_bundle as B  # noqa: E402
import seed_atlas_spider_dev as S  # noqa: E402
import verify_kaggle_spider_dev_1034 as V  # noqa: E402
import local_atlas_scoring as L  # noqa: E402

BUNDLE = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2"
ZIP = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2.zip"
N_EVAL = 892          # 1,034 dev minus the 142 records on the 3 held-out test databases
HELD_OUT_DBS = ["employee_hire_evaluation", "pets_1", "tvshow"]
NB = ROOT / "notebooks/kaggle/nl2mql_spider_dev_1034_model_comparison.ipynb"
needs_bundle = pytest.mark.skipif(not BUNDLE.exists(), reason="bundle not built")


@pytest.fixture(scope="module")
def KP():
    return C.bundle_modules(BUNDLE)


# ---------------------------------------------------------------- vendored code == repo code
def _func_ast(path, name):
    """AST of a top-level function (docstring removed) or assignment."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body[1:] if isinstance(node.body[0], ast.Expr) and isinstance(
                getattr(node.body[0], "value", None), ast.Constant) else node.body
            return ast.dump(ast.Module(body, [])) + ast.dump(node.args)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.dump(node)
    raise KeyError(name)


@pytest.mark.parametrize("src,names", [
    ("scripts/evaluation/eval_kit.py", ["WRITE_STAGES", "JS_STAGES", "SERVER_ADMIN_STAGES", "FORBIDDEN_STAGES",
                                        "iter_stages", "referenced_collections", "UNION_WITH_KEYS",
                                        "union_with_targets", "validate_mql"]),
    ("scripts/spider1/run_eval.py", ["forbidden_anywhere"]),
    ("scripts/evaluation/run_nl_mql_eval.py", ["parse_output"]),
    ("scripts/evaluation/providers.py", ["parse_retry_in"]),
])
def test_vendored_functions_identical_to_repo(src, names):
    core = ROOT / "scripts/nl2mql/dev1034_core.py"
    for n in names:
        assert _func_ast(ROOT / src, n) == _func_ast(core, n), f"{n} differs from {src}"


@needs_bundle
def test_bundled_repo_modules_are_byte_copies():
    man = json.loads((BUNDLE / "bundle_manifest.json").read_text(encoding="utf-8"))
    for rel, src in B.CODE_SOURCES.items():
        assert man["files"][rel]["sha256"] == C.sha256_file(src), f"{rel} is stale: rebuild the bundle"


# ---------------------------------------------------------------- input isolation / prompts
def test_prompt_builder_accepts_only_prompt_input(KP):
    K, P = KP
    rec = {"id": "spider_dev_0000", "db_id": "concert_singer", "question": "How many singers do we have?",
           "sql": "SELECT count(*) FROM singer", "execution": {"ordered_sha256": "x"}}
    with pytest.raises(TypeError):
        C.build_prompt(rec, P)
    with pytest.raises(TypeError):
        C.build_prompt(("spider_dev_0000", "concert_singer", "q"), P)
    pi = C.prompt_input(rec)
    assert tuple(pi) == ("spider_dev_0000", "concert_singer", "How many singers do we have?")
    with pytest.raises(TypeError):
        C.PromptInput("a", "b", "")
    system, msgs = C.build_prompt(pi, P)
    blob = system + msgs[0]["content"]
    assert "SELECT count(*)" not in blob and "ordered_sha256" not in blob


def test_gold_leak_detector():
    rec = {"id": "a", "db_id": "d", "question": "q", "sql": "SELECT x FROM t"}
    ok = [{"id": "a", "system": "s", "messages": [{"role": "user", "content": "q"}]}]
    assert C.check_no_gold_leak(ok, [rec])
    bad = [{"id": "a", "system": "s", "messages": [{"role": "user", "content": "q  select  X from T"}]}]
    with pytest.raises(C.PreflightError):
        C.check_no_gold_leak(bad, [rec])


@needs_bundle
def test_bundle_prompts_byte_identical_to_repo_prompt_module():
    sys.path.insert(0, str(ROOT / "scripts/spider1"))
    import prompt_spider1 as RP  # repo copy
    frozen = C.read_jsonl(BUNDLE / "data/prompts.jsonl")
    recs = C.load_dev(BUNDLE)
    assert len(frozen) == N_EVAL and [p["id"] for p in frozen] == [r["id"] for r in recs]
    assert RP.template_hash() == B.PROMPT_TEMPLATE_SHA256
    for p, r in zip(frozen, recs):
        assert p["system"] == RP.SYSTEM and p["messages"] == RP.messages(r["db_id"], r["question"])


# ---------------------------------------------------------------- safety gate
ALLOWED = {"concert_singer__singer", "concert_singer__concert"}


@pytest.mark.parametrize("mql,detail", [
    ({"collection": "concert_singer__singer", "pipeline": [{"$out": "x"}]}, "unsafe_mql"),
    ({"collection": "concert_singer__singer", "pipeline": [{"$merge": {"into": "x"}}]}, "unsafe_mql"),
    ({"collection": "concert_singer__singer", "pipeline": [{"$lookup": {"from": "concert_singer__concert", "as": "c",
      "pipeline": [{"$out": "y"}]}}]}, "unsafe_mql"),
    ({"collection": "concert_singer__singer", "pipeline": [{"$match": {"$expr": {"$function": {"body": "f",
      "args": [], "lang": "js"}}}}]}, "unsafe_mql"),
    ({"collection": "concert_singer__singer", "pipeline": [{"$group": {"_id": None, "x": {"$accumulator": {}}}}]},
     "unsafe_mql"),
    ({"method": "find", "collection": "concert_singer__singer", "filter": {"$where": "true"}}, "unsafe_mql"),
    ({"collection": "concert_singer__singer", "pipeline": [{"$unionWith": "world_1__city"}]}, "unsafe_mql"),
    ({"collection": "concert_singer__singer", "pipeline": [{"$unionWith": {"coll": "concert_singer__concert",
      "db": "admin"}}]}, "unsafe_mql"),
    ({"collection": "concert_singer__singer", "pipeline": [{"$lookup": {"from": {"db": "admin", "coll":
      "concert_singer__concert"}, "as": "c", "localField": "a", "foreignField": "b"}}]}, "unsafe_mql"),
    ({"collection": "concert_singer__singer", "pipeline": [{"$lookup": {"from": "world_1__city", "as": "c",
      "localField": "a", "foreignField": "b"}}]}, "unsafe_mql"),
    ({"collection": "world_1__city", "pipeline": [{"$match": {}}]}, "unsafe_mql"),
    ({"collection": "concert_singer__singer", "pipeline": [{"$changeStream": {}}]}, "unsafe_mql"),
    ({"collection": "concert_singer__singer", "pipeline": [{"$currentOp": {}}]}, "unsafe_mql"),
    ({"collection": "concert_singer__singer", "pipeline": [{"$match": {"x": "a" * 60000}}]}, "unsafe_mql"),
    ({"collection": "concert_singer__singer", "method": "mapReduce"}, "invalid_mql"),
    ({"pipeline": [{"$match": {}}]}, "invalid_mql"),
])
def test_unsafe_or_invalid_mql_rejected(mql, detail):
    stage, norm, det, why = C.check_mql(json.dumps(mql), ALLOWED)
    assert norm is None and det == detail, why


def test_safe_mql_accepted_and_normalised():
    s, n, _, _ = C.check_mql('```json\n{"collection":"concert_singer__singer","pipeline":[]}\n```', ALLOWED)
    assert s == "ok" and n["pipeline"] == [{"$match": {}}]
    s, n, _, _ = C.check_mql('Here: {"collection":"concert_singer__singer","pipeline":[{"$lookup":{"from":'
                             '"concert_singer__concert","localField":"a","foreignField":"b","as":"c"}},{"$count":"n"}]}',
                             ALLOWED)
    assert s == "ok" and n["collection"] == "concert_singer__singer"
    s, n, _, _ = C.check_mql('{"method":"find","collection":"concert_singer__singer","filter":{"Age":{"$gt":30}},'
                             '"projection":{"_id":0,"Name":1}}', ALLOWED)
    assert s == "ok" and n["pipeline"] == [{"$match": {"Age": {"$gt": 30}}}, {"$project": {"_id": 0, "Name": 1}}]
    assert C.check_mql("no json here", ALLOWED)[2] == "format_error"
    assert C.check_mql(None, ALLOWED)[2] == "format_error"


# ---------------------------------------------------------------- comparator + reference policy
def _ref(rows, order_by=False, blocks=None, status="ok"):
    return {"id": "x", "db_id": "d", "hardness": "easy", "gold_rows": rows, "order_by": order_by,
            "tie_blocks": blocks, "reference_status": status, "n_rows": len(rows)}


def test_comparator_semantics(KP):
    K, _ = KP
    from bson import ObjectId
    oid = ObjectId()
    cases = [
        ([{"_id": oid, "n": 3.0}], [[3]], True),                         # ObjectId dropped; 3.0 == 3
        ([{"_id": None, "n": 6}], [[6]], True),                          # $group null key dropped
        ([{"b": "x", "a": 1}, {"b": "y", "a": 2}], [[1, "x"], [2, "y"]], True),  # column order/aliases ignored
        ([{"a": 1}, {"a": 1}], [[1]], False),                            # duplicates matter (multiset)
        ([{"a": 1}, {"a": 1}], [[1], [1]], True),
        ([{"a": "3"}], [[3]], False),                                    # text '3' != number 3
        ([{"a": 1.23456}], [[1.23455999]], True),                        # 4-decimal rounding
        ([{"a": 1}], [[None]], False),
        ([{"b": 2}, {"a": 1, "b": 3}], [[None, 2], [1, 3]], True),      # missing field == NULL
        ([], [], True),                                                  # both empty
        ([{"a": 1}], [], False),
        ([{"_id": {"x": 1, "y": "k"}, "n": 2}], [[1, "k", 2]], True),     # nested values flattened
        ([{"a": 1, "b": 2}], [[1]], False),                              # extra column
    ]
    for docs, gold, want in cases:
        assert C.decide(K, K.docs_to_rows(docs)[1], _ref(gold))[0] == ("correct" if want else "incorrect"), docs
    # ordering: strict unless tied
    rows = K.docs_to_rows([{"a": 2}, {"a": 1}])[1]
    assert C.decide(K, rows, _ref([[1], [2]], order_by=True, blocks=[0, 1]))[:2] == ("incorrect", "order_mismatch")
    assert C.decide(K, rows, _ref([[1], [2]], order_by=True, blocks=[0, 0]))[0] == "correct"
    assert C.decide(K, rows, _ref([[1], [2]]))[0] == "correct"


def test_reference_status_policy(KP):
    K, _ = KP
    good = K.docs_to_rows([{"a": 1}])[1]
    wrong = K.docs_to_rows([{"a": 9}])[1]
    assert C.decide(K, good, _ref([[1]], True, [0], "tie_at_limit"))[0] == "correct"
    assert C.decide(K, wrong, _ref([[1]], True, [0], "tie_at_limit"))[0] == "not_comparable"
    assert C.decide(K, good, _ref([[1]], status="utf8_dependent"))[0] == "not_comparable"
    assert C.decide(K, good, _ref([[1]], status="reference_error"))[0] == "not_comparable"
    swapped = K.docs_to_rows([{"a": 2}, {"a": 1}])[1]
    assert C.decide(K, swapped, _ref([[1], [2]], True, None, "order_ties_unverified"))[0] == "not_comparable"
    assert C.decide(K, wrong, _ref([[1]], True, None, "order_ties_unverified"))[0] == "incorrect"


def test_bson_prenormalize_is_secondary_only(KP):
    K, _ = KP
    from datetime import datetime
    from bson import Decimal128, ObjectId
    docs = [{"_id": ObjectId(), "v": Decimal128("1.5"), "d": datetime(2018, 3, 17, 6, 4, 34), "o": ObjectId(
        "65f000000000000000000001")}]
    nd, notes = C.bson_prenormalize(docs)
    assert notes == Counter(decimal128=1, datetime=1, objectid=1)
    assert isinstance(nd[0]["_id"], ObjectId) and nd[0]["v"] == 1.5 and nd[0]["d"] == "2018-03-17 06:04:34"
    assert C.decide(K, K.docs_to_rows(nd)[1], _ref([[1.5, "2018-03-17 06:04:34", "65f000000000000000000001"]]))[0] \
        == "correct"


def test_storage_cells_reproduce_comparator_values(KP):
    K, _ = KP
    from bson import Decimal128, ObjectId
    vals = [None, True, 3, 3.5, "x", b"\xffab", Decimal128("2.5"), [1, ObjectId()], float("nan")]
    for v in vals:
        a, b = K._norm(v), K._norm(C.storage_cell(v))
        assert a == b or (a is None and b is None), v


@needs_bundle
def test_reference_census_and_utf8_policy(tmp_path, monkeypatch):
    exp = json.loads((BUNDLE / "data/reference_expected.json").read_text(encoding="utf-8"))
    assert exp["status_counts"] == {"ok": 846, "order_ties_unverified": 2, "tie_at_limit": 44}
    assert len(exp["items"]) == N_EVAL and not [i for i in exp["items"] if i["reference_status"] == "reference_error"]
    # a tiny DB with an invalid-UTF-8 string: a query that depends on it is flagged utf8_dependent
    db = tmp_path / "tiny" / "tiny.sqlite"
    db.parent.mkdir()
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE p (id INT, name TEXT)")
    con.execute("INSERT INTO p VALUES (1, CAST(X'41FF42' AS TEXT)), (2, 'ok')")
    con.commit()
    con.close()
    K, _ = C.bundle_modules(BUNDLE)
    monkeypatch.setattr(K, "DB_DIR", tmp_path)
    monkeypatch.setattr(C, "EXPECTED_UTF8_REPLACED_DBS", {"tiny": 1})

    def rec(i, sql):
        rows = C.run_sql_ro(db, sql)[1]
        o, m = C.result_hashes(rows)
        return {"id": f"t{i}", "db_id": "tiny", "hardness": "easy", "sql": sql,
                "execution": {"ordered_sha256": o, "multiset_sha256": m}}
    refs = C.compute_reference([rec(0, "SELECT count(*) FROM p WHERE name = 'ok'"),
                                rec(1, "SELECT count(*) FROM p WHERE length(CAST(name AS BLOB)) = 3"),
                                {**rec(2, "SELECT 1"), "execution": {"ordered_sha256": "bad", "multiset_sha256": "bad"}}],
                               K, tmp_path)
    assert [r["reference_status"] for r in refs] == ["ok", "utf8_dependent", "reference_error"]


@needs_bundle
def test_reference_recheck_tolerates_sqlite_version_noise(monkeypatch):
    K, _ = C.bundle_modules(BUNDLE)
    recs = C.load_dev(BUNDLE)
    refs = C.load_reference(BUNDLE)
    by = {r["id"]: r for r in recs}
    unordered = next(r["id"] for r in recs if not r["execution"]["order_sensitive"] and r["execution"]["n_rows"] > 2)
    with_float = next(x["id"] for x in refs if x["reference_status"] == "ok" and not by[x["id"]]["execution"]
                      ["order_sensitive"] and any(isinstance(v, float) for row in x["gold_rows"] for v in row))
    tie = next(x["id"] for x in refs if x["reference_status"] == "tie_at_limit")
    real = C.run_sql_ro
    pick = [unordered, with_float, tie]
    sub_recs, sub_refs = [by[i] for i in pick], [next(x for x in refs if x["id"] == i) for i in pick]

    def noisy(path, sql, timeout_s=C.SQLITE_TIMEOUT_S):
        cols, rows = real(path, sql, timeout_s)
        if sql == by[unordered]["sql"]:
            rows = rows[::-1]                                             # unscored row order
        if sql == by[with_float]["sql"]:
            rows = [tuple(v + 1e-12 if isinstance(v, float) else v for v in r) for r in rows]   # last bits
        if sql == by[tie]["sql"]:
            rows = rows[:-1] + [tuple("other tied row" for _ in rows[-1])]  # another valid tied row
        return cols, rows
    monkeypatch.setattr(C, "run_sql_ro", noisy)
    rc = C.recheck_reference(sub_recs, sub_refs, K, BUNDLE / "database")
    assert rc["problems"] == [] and len(rc["warnings"]) >= 2 and rc["sqlite_version"]

    def broken(path, sql, timeout_s=C.SQLITE_TIMEOUT_S):
        cols, rows = real(path, sql, timeout_s)
        return cols, rows[:-1]                                            # a real change
    monkeypatch.setattr(C, "run_sql_ro", broken)
    rc = C.recheck_reference(sub_recs[:1], sub_refs[:1], K, BUNDLE / "database")
    assert rc["problems"] and "multiset differs" in rc["problems"][0]


# ---------------------------------------------------------------- execution with a fake Atlas
class FakeCursor:
    def __init__(self, docs):
        self.docs, self.closed, self.served = docs, False, 0

    def __iter__(self):
        for d in self.docs:
            self.served += 1
            yield dict(d)

    def close(self):
        self.closed = True


class FakeColl:
    def __init__(self, docs=None, count=None, agg=None):
        self.docs, self._count, self.agg, self.inserted = list(docs or []), count, agg, 0

    def count_documents(self, q):
        return self._count if self._count is not None else len(self.docs)

    def find(self, q, proj=None, batch_size=None):
        return FakeCursor([{k: v for k, v in d.items() if k != "_id"} for d in self.docs])

    def aggregate(self, pipeline, **kw):
        assert kw["maxTimeMS"] > 0
        if self.agg:
            return self.agg(pipeline)
        return FakeCursor(self.docs)

    def insert_many(self, docs, ordered=True):
        self.docs.extend(dict(d) for d in docs)
        self.inserted += len(docs)


class FakeDB:
    def __init__(self, colls, actions=("find", "listCollections", "killCursors", "collStats", "dbStats"), name="db"):
        self.colls, self.actions, self.name = colls, list(actions), name

    def __getitem__(self, n):
        return self.colls.setdefault(n, FakeColl())

    def list_collection_names(self):
        return [n for n, c in self.colls.items() if c.docs or c._count is not None]

    def command(self, cmd):
        if isinstance(cmd, dict) and "connectionStatus" in cmd:
            return {"authInfo": {"authenticatedUsers": [{"user": "codegen_eval_readonly", "db": "admin"}],
                                 "authenticatedUserPrivileges": [{"resource": {"db": self.name, "collection": ""},
                                                                  "actions": self.actions}]}}
        if cmd == "dbStats":
            raise PermissionError("not authorized")
        raise KeyError(cmd)


class FakeAdmin:
    def command(self, c):
        return {"ok": 1} if c == "ping" else {"version": "8.0.34"}


class FakeClient:
    def __init__(self, db):
        self.db, self.admin = db, FakeAdmin()

    def __getitem__(self, name):
        return self.db

    def close(self):
        pass


def test_pipeline_execution_is_bounded():
    cur = FakeCursor([{"a": i} for i in range(100)])
    coll = FakeColl(agg=lambda p: cur)
    docs, trunc = C.run_pipeline({"c": coll}, "c", [{"$match": {}}], 1000, max_rows=10)
    assert trunc and len(docs) == 11 and cur.closed and cur.served == 11


def test_evaluate_prediction_outcomes(KP):
    K, _ = KP
    from pymongo.errors import AutoReconnect, ExecutionTimeout, OperationFailure
    ref = _ref([[6]])
    allowed = {"concert_singer__singer"}
    ok_text = '{"collection":"concert_singer__singer","pipeline":[{"$count":"n"}]}'

    def ev(agg, text=ok_text, status="ok", max_rows=C.MAX_RESULT_ROWS):
        mdb = {"concert_singer__singer": FakeColl(agg=agg)}
        return C.evaluate_prediction(mdb, K, {"id": "x", "status": status, "raw_text": text, "error": "boom"},
                                     ref, allowed, max_rows=max_rows)[0]

    def raise_(e):
        def f(p):
            raise e
        return f
    assert ev(lambda p: FakeCursor([{"n": 6}]))["outcome"] == "correct"
    r = ev(lambda p: FakeCursor([{"n": 7}]))
    assert (r["outcome"], r["detail"], r["failure_source"]) == ("incorrect", "wrong_values", "model")
    r = ev(raise_(ExecutionTimeout("operation exceeded time limit", 50)))
    assert (r["outcome"], r["failure_source"]) == ("timeout", "model")
    r = ev(raise_(AutoReconnect("connection reset")))
    assert (r["outcome"], r["failure_source"], C.is_eval_retry(r)) == ("failed", "infrastructure", True)
    r = ev(raise_(OperationFailure("Unrecognized expression '$foo'", 168)))
    assert (r["outcome"], r["failure_source"], r["detail"]) == ("failed", "model", "execution_error")
    r = ev(raise_(OperationFailure("not allowed in this atlas tier", 8000)))
    assert r["detail"] == "atlas_tier_restriction"
    r = ev(lambda p: FakeCursor([{"n": i} for i in range(50)]), max_rows=5)
    assert (r["outcome"], r["detail"]) == ("incorrect", "result_too_large")
    r = ev(None, text='{"collection":"concert_singer__singer","pipeline":[{"$out":"z"}]}')
    assert (r["outcome"], r["readonly_valid"], r["structure_valid"], r["executed"]) == ("invalid", False, True, False)
    r = ev(None, status="error")
    assert (r["outcome"], r["failure_source"], r["detail"]) == ("failed", "infrastructure", "generation_error")


# ---------------------------------------------------------------- checkpoints, generation loop, Groq client
def test_checkpoint_resume_rules(tmp_path):
    p = tmp_path / "ck.jsonl"
    ck = C.Checkpoint(p, ["a", "b", "c"], "fp", C.is_generation_retry)
    ck.append([{"id": "a", "fingerprint": "fp", "status": "error"}])
    ck.append([{"id": "a", "fingerprint": "fp", "status": "ok"}])     # retry after an error is allowed
    with pytest.raises(RuntimeError):
        ck.append([{"id": "a", "fingerprint": "fp", "status": "ok"}])  # a final record is never duplicated
    with pytest.raises(RuntimeError):
        ck.append([{"id": "b", "fingerprint": "other", "status": "ok"}])
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"id": "b", "fingerp')                                 # crash mid-write
    again = C.Checkpoint(p, ["a", "b", "c"], "fp", C.is_generation_retry)
    assert again.warnings and again.todo() == ["b", "c"] and again.records["a"]["status"] == "ok"
    with pytest.raises(RuntimeError):
        C.Checkpoint(p, ["a", "b", "c"], "different-fp")
    with pytest.raises(RuntimeError):
        C.Checkpoint(p, ["b", "c"], "fp")                               # unexpected id


def test_run_generation_stops_cleanly_and_resumes(tmp_path):
    prompts = [{"id": f"i{i}", "db_id": "d", "prompt_sha256": "h"} for i in range(6)]
    ids = [p["id"] for p in prompts]
    ck = C.Checkpoint(tmp_path / "p.jsonl", ids, "fp", C.is_generation_retry)
    calls = []

    def gen(batch):
        calls.append([p["id"] for p in batch])
        if batch[0]["id"] == "i3":
            raise C.QuotaStop("DAILY_QUOTA_EXHAUSTED: try again in 2h")
        return [("{}", {"input_tokens": 10}, None if p["id"] != "i1" else "HTTP 503") for p in batch]
    r = C.run_generation(prompts, gen, ck, "m", 1)
    assert r["stopped"].startswith("DAILY_QUOTA") and ck.todo() == ["i1", "i3", "i4", "i5"]
    ck2 = C.Checkpoint(tmp_path / "p.jsonl", ids, "fp", C.is_generation_retry)
    r = C.run_generation(prompts, lambda b: [("{}", {}, None) for _ in b], ck2, "m", 2,
                         stop=lambda meta: None)
    assert r["stopped"] is None and ck2.todo() == [] and len(ck2.ordered()) == 6
    ck3 = C.Checkpoint(tmp_path / "q.jsonl", ids, "fp")
    r = C.run_generation(prompts, lambda b: [("{}", {}, None) for _ in b], ck3, "m", 1,
                         stop=lambda meta: "token guard")
    assert r == {"stopped": "token guard", "generated": 1}


class FakeResp:
    def __init__(self, body, headers=None):
        self.body, self.headers = json.dumps(body).encode(), types.SimpleNamespace(
            items=lambda: list((headers or {}).items()))

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, body, headers=None):
    return urllib.error.HTTPError("u", code, "err", headers or {}, io.BytesIO(json.dumps(body).encode()))


def test_groq_client_retries_quota_and_redaction():
    seq = [_http_error(503, {"error": "busy"}), FakeResp(
        {"id": "req1", "model": "openai/gpt-oss-120b", "choices": [{"message": {"content": "{}"},
                                                                    "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 1000, "completion_tokens": 300, "total_time": 0.4,
                   "completion_tokens_details": {"reasoning_tokens": 250}}},
        {"x-ratelimit-remaining-tokens": "7000", "x-request-id": "r"})]
    sent = []

    def opener(req, timeout):
        sent.append(json.loads(req.data.decode()))
        x = seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x
    g = C.GroqClient("gsk_secretkey1234567890abcd", "openai/gpt-oss-120b", opener=opener, sleep=lambda s: None)
    text, meta, err = g.chat("sys", [{"role": "user", "content": "q"}])
    assert err is None and text == "{}" and meta["retries"] == 1 and meta["reasoning_tokens"] == 250
    assert meta["rate_limit"] == {"x-ratelimit-remaining-tokens": "7000"}
    assert sent[0]["temperature"] == 0.0 and sent[0]["seed"] == 42 and sent[0]["messages"][0]["role"] == "system"
    g2 = C.GroqClient("gsk_secretkey1234567890abcd", "m", opener=lambda r, t: (_ for _ in ()).throw(
        _http_error(429, {"error": "Rate limit reached ... tokens per day (TPD) ... try again in 3h"})),
        sleep=lambda s: None)
    with pytest.raises(C.QuotaStop):
        g2.chat("s", [])
    g3 = C.GroqClient("gsk_secretkey1234567890abcd", "m", retries=1, opener=lambda r, t: (_ for _ in ()).throw(
        _http_error(401, {"error": "bad key gsk_secretkey1234567890abcd"})), sleep=lambda s: None)
    text, meta, err = g3.chat("s", [])
    assert text is None and "gsk_secret" not in err and "<API_KEY>" in err


# ---------------------------------------------------------------- statistics
def test_statistics():
    assert C.wilson(0, 10)[0] == 0.0 and C.wilson(5, 10) == [0.2366, 0.7634]
    assert C.mcnemar_exact(0, 0) == 1.0
    assert abs(C.mcnemar_exact(2, 10) - 0.0386) < 1e-3
    a, b = [1] * 60 + [0] * 40, [1] * 50 + [0] * 50
    assert C.paired_bootstrap_ci(a, b, 2000, 1) == C.paired_bootstrap_ci(a, b, 2000, 1)
    lo, hi = C.paired_bootstrap_ci(a, b, 2000, 1)
    assert lo <= 0.1 <= hi
    st = C.latency_stats([1, 2, 3, 4, None])
    assert st["median"] == 2 and st["max"] == 4 and st["n"] == 4


# ---------------------------------------------------------------- Atlas preflight (mocked)
def _small_mapping(dbs=("concert_singer", "singer")):
    m = C.load_mapping(BUNDLE)
    m = json.loads(json.dumps(m))
    m["collections"] = {k: v for k, v in m["collections"].items() if v["db_id"] in dbs}
    m["db_ids"] = list(dbs)
    return m


def _fake_atlas(mapping, actions=None):
    colls = {}
    for name, mm in mapping["collections"].items():
        colls[name] = FakeColl(S.sqlite_docs(BUNDLE, mm))
    return FakeClient(FakeDB(colls, actions or ("find", "listCollections", "killCursors"), mapping["atlas_db"]))


@needs_bundle
def test_atlas_preflight_mocked():
    m = _small_mapping()
    rep = C.atlas_preflight(_fake_atlas(m), m, full_parity=True, log=None)
    assert rep["ok"] and rep["server_version"] == "8.0.34" and all(v["digest_match"] for v in rep["parity"].values())
    assert rep["db_stats"].startswith("not available")
    cl = _fake_atlas(m, ("find", "insert", "listCollections"))
    with pytest.raises(C.PreflightError, match="NOT read-only"):
        C.atlas_preflight(cl, m, log=None)
    cl = _fake_atlas(m)
    cl.db.colls.pop("concert_singer__stadium")
    with pytest.raises(C.PreflightError, match="missing"):
        C.atlas_preflight(cl, m, log=None)
    cl = _fake_atlas(m)
    cl.db.colls["concert_singer__stadium"].docs[0]["Capacity"] = str(cl.db.colls["concert_singer__stadium"].docs[0]["Capacity"])
    with pytest.raises(C.PreflightError, match="digest mismatch"):
        C.atlas_preflight(cl, m, log=None)
    cl = _fake_atlas(m)
    cl.db.colls["singer__extra"] = FakeColl([{"a": 1}])
    with pytest.raises(C.PreflightError, match="unexpected"):
        C.atlas_preflight(cl, m, log=None)
    cl = _fake_atlas(m)
    cl.db.colls["concert_singer__stadium"].docs.pop()
    with pytest.raises(C.PreflightError, match="documents on Atlas"):
        C.atlas_preflight(cl, m, full_parity=False, log=None)


def test_privileges_report():
    cs = {"authInfo": {"authenticatedUsers": [{"user": "u"}], "authenticatedUserPrivileges": [
        {"resource": {"db": "codegen_spider1", "collection": ""}, "actions": ["find", "killCursors", "listIndexes"]},
        {"resource": {"cluster": True}, "actions": ["dropDatabase"]}]}}
    users, per, bad = C.privileges_report(cs)
    assert users == ["u"] and bad == ["cluster:dropDatabase"] and "find" in per["codegen_spider1."]


# ---------------------------------------------------------------- bundle integrity
@needs_bundle
def test_bundle_verifies_and_excludes_test_data():
    man = B.verify_dir(BUNDLE)
    assert man["counts"]["records"] == N_EVAL and len(man["db_ids"]) == 17 and man["counts"]["collections"] == 70
    ev = man["evaluation_set"]
    assert ev["excluded_db_ids"] == HELD_OUT_DBS and ev["excluded_records"] == 142
    assert ev["overlap_with_heldout_questions"] == 65 and not set(man["db_ids"]) & set(HELD_OUT_DBS)
    names = list(man["files"])
    assert not [n for n in names if re.search(r"test|train|split|\.env|__pycache__|gguf|checkpoint", n, re.I)]
    assert sum(n.startswith("database/") for n in names) == 17
    for n in names:
        if n.endswith((".json", ".jsonl", ".py", ".md")):
            t = (BUNDLE / n).read_text(encoding="utf-8")
            assert not any(re.search(p, t) for p in B.SECRET_PATTERNS), n
    prior = json.loads((BUNDLE / "data/prior_audit.json").read_text(encoding="utf-8"))
    assert Counter(v["status"] for v in prior.values()) == {"included": 390, "excluded": 34}
    assert "partition" not in json.dumps(prior)
    with zipfile.ZipFile(ZIP) as z:
        assert sorted(z.namelist()) == sorted(f"{BUNDLE.name}/{n}" for n in names + ["bundle_manifest.json"])


@needs_bundle
def test_bundle_tamper_is_detected(tmp_path):
    t = tmp_path / BUNDLE.name
    shutil.copytree(BUNDLE, t)
    lines = (t / "data/eval_records.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    (t / "data/eval_records.jsonl").write_text("".join([lines[1], lines[0]] + lines[2:]), encoding="utf-8")
    with pytest.raises(C.PreflightError, match="hash mismatch"):
        C.verify_bundle(t)
    shutil.rmtree(t)
    shutil.copytree(BUNDLE, t)
    (t / "data/extra_test.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(C.PreflightError, match="extra"):
        C.verify_bundle(t)


@needs_bundle
def test_heldout_exclusion_never_opens_the_partition_file(monkeypatch):
    """The exclusion is derived from ID-only metadata; any attempt to read a split's test.jsonl fails the test."""
    import builtins
    real_open, real_read_text, real_read_bytes = builtins.open, Path.read_text, Path.read_bytes

    def guard(p):
        if "test.jsonl" in str(p).replace("\\", "/").split("/")[-1] and "split" in str(p):
            raise AssertionError(f"opened the held-out partition file {p}")
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (guard(f), real_open(f, *a, **k))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (guard(self), real_read_text(self, *a, **k))[1])
    monkeypatch.setattr(Path, "read_bytes", lambda self, *a, **k: (guard(self), real_read_bytes(self))[1])
    files, facts = B.plan()
    ex = facts["exclusion"]
    assert ex["excluded_db_ids"] == HELD_OUT_DBS and ex["excluded_records"] == 142
    assert ex["overlap_with_heldout_questions"] == 65 and facts["records"] == N_EVAL
    recs = [json.loads(x) for x in files["data/eval_records.jsonl"].decode("utf-8").splitlines()]
    assert len(recs) == N_EVAL and not [r for r in recs if r["db_id"] in HELD_OUT_DBS]
    idx = [r["source"]["index"] for r in recs]
    assert idx == sorted(idx) and len(set(idx)) == N_EVAL
    with real_open(ROOT / "data/processed/spider_sql_v1/dev_full.jsonl", "rb") as f:
        src = {json.loads(ln)["id"]: ln for ln in f if ln.strip()}
    assert all(src[r["id"]].rstrip(b"\r\n") == ln.encode("utf-8").rstrip(b"\r\n")
               for r, ln in zip(recs, files["data/eval_records.jsonl"].decode("utf-8").splitlines()))


@needs_bundle
def test_bundle_rejects_heldout_database_content(tmp_path):
    t = tmp_path / BUNDLE.name
    shutil.copytree(BUNDLE, t)
    p = t / "context/prompt_samples.json"
    p.write_text(p.read_text(encoding="utf-8").replace("{", '{"pets_1__Pets": [],', 1), encoding="utf-8")
    man = json.loads((t / "bundle_manifest.json").read_text(encoding="utf-8"))
    man["files"]["context/prompt_samples.json"]["sha256"] = C.sha256_file(p)   # even with a matching manifest
    (t / "bundle_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(B.BundleError, match="held-out database"):
        B.verify_dir(t)


@pytest.mark.parametrize("name,denied", [("data/test_full.jsonl", True), ("data/splits_expanded/test.jsonl", True),
                                         ("database/x/train.sqlite", True), (".env", True), ("code/a.pyc", True),
                                         ("code/__pycache__/x", True), ("data/dev_full.jsonl", False),
                                         ("database/wta_1/wta_1.sqlite", False), ("context/tables.json", False)])
def test_bundle_deny_list(name, denied):
    assert bool(B.DENY.search(name)) == denied


# ---------------------------------------------------------------- seed workflow
@needs_bundle
def test_seed_dry_run_and_safety(monkeypatch, capsys):
    assert S.main(["--only", "concert_singer__stadium"]) == 0
    assert "dry run only" in capsys.readouterr().out
    monkeypatch.setenv("MONGODB_URI", "mongodb://x")
    assert S.main(["--execute", "--uri-env", "MONGODB_URI", "--only", "concert_singer__stadium"]) == 2
    monkeypatch.delenv("ATLAS_SEED_URI", raising=False)
    assert S.main(["--execute", "--only", "concert_singer__stadium"]) == 2
    mapping, colls = S.plan(BUNDLE, ["concert_singer__stadium", "concert_singer__singer"])
    db = FakeDB({"concert_singer__stadium": FakeColl([{"a": 1}])})
    with pytest.raises(S.SeedError, match="already exist"):
        S.seed(FakeClient(db), "codegen_spider1", colls, BUNDLE, log=None)
    assert db.colls["concert_singer__stadium"].docs == [{"a": 1}]          # untouched
    db = FakeDB({})
    assert S.seed(FakeClient(db), "codegen_spider1", colls, BUNDLE, log=None) == sorted(colls)
    assert db.colls["concert_singer__singer"].count_documents({}) == colls["concert_singer__singer"]["documents"]


# ---------------------------------------------------------------- notebook structure (Qwen-only, generate-only)
QWEN = "qwen2.5-coder-3b-instruct"


def _cells():
    nb = nbformat.read(NB, as_version=4)
    return [c for c in nb.cells if c.cell_type == "code"]


def test_notebook_is_qwen_only_and_generate_only(tmp_path):
    cells = _cells()
    tags = [c.metadata["tags"][0] for c in cells]
    assert tags == ["config", "env", "bundle", "preflight", "qwen", "package"]
    for c in cells[4:]:
        assert c.source.lstrip().split("\n")[1].startswith("assert PREFLIGHT_OK")
    src = "\n".join(c.source for c in cells)
    full = "\n".join(c.source for c in nbformat.read(NB, 4).cells)
    # no frontier model, no external API, no secret, no Atlas
    for banned in ("groq", "GROQ", "GroqClient", "FRONTIER", "gpt-oss", "api_key", "UserSecretsClient",
                   "kaggle_secrets", "get_secret", "urllib", "requests", "chat(", "pymongo", "MongoClient", "MONGODB",
                   "atlas_preflight", "run_evaluation", "evaluate_prediction", "mongo_conn"):
        assert banned not in src, banned
    # the gold reference is read only to verify a Kaggle-decompressed bundle file, never for generation
    ref_uses = [ln for ln in src.splitlines() if "load_reference" in ln]
    assert ref_uses == ["    refs = C.load_reference(bundle)"]
    assert "gpt-oss" not in full and "Groq" not in full.replace("deferred", "")  # markdown: no Groq requirement
    assert not re.search(r"gsk_|mongodb(\+srv)?://|hf_[A-Za-z0-9]{10}", full)
    assert 'QWEN_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"' in src
    assert 'QWEN_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"' in src
    assert "do_sample=False" in src and "QWEN_MAX_NEW_TOKENS = 1024" in src
    import subprocess
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, str(ROOT / "scripts/nl2mql/make_dev1034_notebook.py"), str(out)], check=True,
                   capture_output=True)
    assert [c.source for c in nbformat.read(out, 4).cells] == [c.source for c in nbformat.read(NB, 4).cells]


def test_verifier_and_scorer_expect_qwen_only():
    assert list(V.EXPECTED_MODELS) == [QWEN] and L.LABELS == (QWEN,) and V.EXPECTED_N == 892
    want = V.EXPECTED_MODELS[QWEN]
    assert want["model_revision"] == "488639f1ff808d1d3d0ba301aef8c11461451ec5"
    assert want["template_sha256"] == B.PROMPT_TEMPLATE_SHA256


# ---------------------------------------------------------------- notebook end to end (fake Qwen) + local scoring
def _oracle_atlas(mapping, refs, broken=(), actions=None):
    """Fake Atlas whose aggregate answers {"$match": {"__oracle": id}} with that id's gold rows."""
    ref_by = {r["id"]: r for r in refs}
    from pymongo.errors import AutoReconnect, ExecutionTimeout
    broken = set(broken)

    def agg(pipeline):
        key = pipeline[0].get("$match", {}).get("__oracle")
        if key == "TIMEOUT":
            raise ExecutionTimeout("operation exceeded time limit", 50)
        if key in broken:
            broken.discard(key)  # transient infrastructure failure: succeeds on retry
            raise AutoReconnect("reset")
        if key == "WRONG":
            return FakeCursor([{"zz": -1}])
        r = ref_by[key]
        return FakeCursor([{f"c{i}": v for i, v in enumerate(row)} for row in r["gold_rows"]])
    colls = {n: FakeColl(count=m["documents"], agg=agg) for n, m in mapping["collections"].items()}
    return FakeClient(FakeDB(colls, actions or ("find", "listCollections", "killCursors"), mapping["atlas_db"]))


def _factory(client):
    return lambda: (client, lambda s: s)


def _answer(pid, first_coll):
    """Fake Qwen answer: right for most items, plus format errors, unsafe MQL, timeouts and wrong results."""
    n = int(pid[-4:])
    if n == 5:
        return "I cannot answer"
    if n == 9:
        return json.dumps({"collection": first_coll, "pipeline": [{"$out": "x"}]})
    key = "TIMEOUT" if n == 13 else ("WRONG" if n % 7 == 0 else pid)
    return json.dumps({"collection": first_coll, "pipeline": [{"$match": {"__oracle": key}}]})


class FakeOOM(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError."""


def _run_notebook(out_dir, out_zip, input_root, fail_after=None, crash_after=None, oom_if=None, raise_if=None,
                  calls=None, info=None, **overrides):
    """fail_after: batches after this many items return generation errors; crash_after: the generator raises.
    oom_if(batch) -> True raises FakeOOM; raise_if(batch) -> an exception to raise; calls collects each batch's ids."""
    cells = {c.metadata["tags"][0]: c.source for c in _cells()}
    ns = {"__name__": "__nb__", "_cleanups": [0]}
    exec(cells["config"], ns)
    mapping = C.load_mapping(BUNDLE)
    first = {}
    for n, m in sorted(mapping["collections"].items()):
        first.setdefault(m["db_id"], n)
    golds = {r["id"]: r["sql"] for r in C.load_dev(BUNDLE)}
    seen = {"n": 0}

    def qwen_factory():
        def gen(batch):
            for p in batch:
                assert golds[p["id"]] not in json.dumps(p)   # Qwen never sees gold SQL
                assert set(p) >= {"system", "messages"} and "sql" not in p and "gold_rows" not in p
            if calls is not None:
                calls.append([p["id"] for p in batch])
            if oom_if and oom_if(batch):
                raise FakeOOM("CUDA out of memory. Tried to allocate 800.00 MiB (simulated)")
            exc = raise_if(batch) if raise_if else None
            if exc is not None:
                raise exc
            seen["n"] += len(batch)
            if crash_after is not None and seen["n"] > crash_after:
                raise RuntimeError("CUDA error: simulated crash")
            if fail_after is not None and seen["n"] > fail_after:
                return [(None, {"latency_s": None}, "OutOfMemoryError: simulated") for _ in batch]
            return [(_answer(p["id"], first[p["db_id"]]),
                     {"latency_s": 0.1, "input_tokens": 1100, "output_tokens": 90, "finish_reason": "stop"}, None)
                    for p in batch]

        def cleanup():
            ns["_cleanups"][0] += 1
        return (gen, info or {"dtype": "float16", "gpu": "FakeGPU", "torch": "x", "transformers": "y"},
                (lambda e: isinstance(e, FakeOOM)), cleanup)

    ns.update(INPUT_ROOT=str(input_root), BUNDLE_DIR=str(BUNDLE), OUT_DIR=str(out_dir), OUTPUT_ZIP=str(out_zip),
              QWEN_GENERATOR_FACTORY=qwen_factory)
    ns.update(overrides)
    for tag in ("env", "bundle", "preflight", "qwen", "package"):
        exec(cells[tag], ns)
    return ns


@pytest.fixture(scope="module")
def predictions_runs(tmp_path_factory):
    """Session 1: generation errors after 300 items (partial). Session 2 (new version, RESUME_FROM): crash after 200
    more (partial, checkpoint kept). Session 3: resumes and completes. Finished ids are never regenerated."""
    tmp = tmp_path_factory.mktemp("nb")
    inp = tmp / "input"
    inp.mkdir()
    runs = {"tmp": tmp}
    prev = None
    for i, kw in enumerate(({"fail_after": 300}, {"crash_after": 200}, {}), 1):
        out, z = tmp / f"w{i}/nl2mql_spider_dev_predictions_v2", tmp / f"w{i}/preds.zip"
        ns = _run_notebook(out, z, inp, RESUME_FROM=str(prev) if prev else None, **kw)
        runs[i] = {"out": out, "zip": z, "ns": ns}
        prev = out
    return runs


@needs_bundle
def test_predictions_partial_crash_resume_complete(predictions_runs):
    r = predictions_runs
    st1 = json.loads((r[1]["out"] / "PREDICTIONS_STATUS.json").read_text(encoding="utf-8"))
    p1 = C.read_jsonl(r[1]["out"] / f"models/{QWEN}/predictions.jsonl")
    assert st1["status"] == "partial" and sum(p["status"] == "ok" for p in p1) == 296   # 37 full batches of 8 before the failing batch
    assert any("generation errors" in p for p in st1["problems"])
    assert st1["atlas"].startswith("not contacted") and st1["external_model_apis"] == "none"
    st2 = json.loads((r[2]["out"] / "PREDICTIONS_STATUS.json").read_text(encoding="utf-8"))
    assert st2["status"] == "partial" and r[2]["ns"]["MODEL_ERROR"].startswith("RuntimeError")
    ok2 = {p["id"] for p in C.read_jsonl(r[2]["out"] / f"models/{QWEN}/predictions.jsonl") if p["status"] == "ok"}
    assert {p["id"] for p in p1 if p["status"] == "ok"} < ok2                            # kept + extended
    st3 = json.loads((r[3]["out"] / "PREDICTIONS_STATUS.json").read_text(encoding="utf-8"))
    assert st3["status"] == "complete", st3["problems"]
    preds = C.read_jsonl(r[3]["out"] / f"models/{QWEN}/predictions.jsonl")
    assert [p["id"] for p in preds] == [x["id"] for x in C.load_dev(BUNDLE)] and len(preds) == N_EVAL
    assert all(p["status"] == "ok" and {"id", "db_id", "raw_text", "model", "fingerprint", "prompt_sha256"} <= set(p)
               for p in preds)
    cfg = json.loads((r[3]["out"] / f"models/{QWEN}/run_config.json").read_text(encoding="utf-8"))
    assert cfg["generated_this_session"] == N_EVAL - len(ok2) and cfg["model_revision"].startswith("488639f1")
    with zipfile.ZipFile(r[3]["zip"]) as z:
        names = z.namelist()
    assert sorted({n.split("/")[2] for n in names if "/models/" in n}) == [QWEN]
    assert not [n for n in names if re.search(r"reference|gold|\.env|atlas_mapping|groq", n, re.I)]
    rep = V.verify_predictions(V.safe_extract(r[3]["zip"], r["tmp"] / "vp3", "PREDICTIONS_STATUS.json"), BUNDLE)
    assert rep["all_checks_passed"] and rep["complete"], rep["problems"]
    rep1 = V.verify_predictions(V.safe_extract(r[1]["zip"], r["tmp"] / "vp1", "PREDICTIONS_STATUS.json"), BUNDLE)
    assert rep1["all_checks_passed"] and not rep1["complete"]


@needs_bundle
def test_predictions_tamper_and_extra_model_detected(predictions_runs, tmp_path):
    d = V.safe_extract(predictions_runs[3]["zip"], tmp_path / "x", "PREDICTIONS_STATUS.json")
    f = d / f"models/{QWEN}/predictions.jsonl"
    lines = f.read_text(encoding="utf-8").splitlines()
    r0 = json.loads(lines[0])
    r0["raw_text"] = '{"collection": "concert_singer__singer", "pipeline": [{"$count": "n"}]}'
    f.write_text("\n".join([json.dumps(r0)] + lines[1:]) + "\n", encoding="utf-8")
    rep = V.verify_predictions(d, BUNDLE)
    assert not rep["all_checks_passed"] and any("checksums_match" in p for p in rep["problems"])
    d2 = V.safe_extract(predictions_runs[3]["zip"], tmp_path / "y", "PREDICTIONS_STATUS.json")
    shutil.copytree(d2 / f"models/{QWEN}", d2 / "models/some-other-model")
    rep = V.verify_predictions(d2, BUNDLE)
    assert not rep["all_checks_passed"] and any("only_expected_models" in p for p in rep["problems"])
    d3 = V.safe_extract(predictions_runs[3]["zip"], tmp_path / "z", "PREDICTIONS_STATUS.json")
    f3 = d3 / f"models/{QWEN}/predictions.jsonl"
    keep = f3.read_text(encoding="utf-8").splitlines()
    f3.write_text("\n".join(keep[:-1] + [keep[0]]) + "\n", encoding="utf-8")     # duplicate id, one missing
    assert not V.verify_predictions(d3, BUNDLE)["all_checks_passed"]


@needs_bundle
def test_local_scoring_end_to_end_verify_and_no_overwrite(predictions_runs, tmp_path):
    refs = C.load_reference(BUNDLE)
    mapping = C.load_mapping(BUNDLE)
    dest = tmp_path / "outputs"
    zp = predictions_runs[3]["zip"]
    client = _oracle_atlas(mapping, refs, broken={"spider_dev_0002"})
    st, ver, run_dir = L.cmd_score(zp, BUNDLE, dest, client_factory=_factory(client), log=None)
    # one transient Atlas failure -> infrastructure, partial; re-running the same command retries only that item
    assert st["status"] == "partial" and any("infrastructure" in p for p in st["problems"])
    assert ver["all_checks_passed"], ver["problems"]
    preds_before = V.tree_sums(run_dir / "predictions")
    st, ver, run_dir = L.cmd_score(zp, BUNDLE, dest, client_factory=_factory(_oracle_atlas(mapping, refs)), log=None)
    assert st["status"] == "complete", st["problems"]
    assert ver["all_checks_passed"] and ver["complete"], ver["problems"]
    assert V.tree_sums(run_dir / "predictions") == preds_before          # predictions never regenerated or changed
    assert run_dir == dest / "v1"
    sd = run_dir / "scoring"
    s = json.loads((sd / f"models/{QWEN}/baseline_summary.json").read_text(encoding="utf-8"))
    assert sum(s["outcomes"].values()) == N_EVAL and s["status"] == "complete"
    assert s["outcomes"]["timeout"] >= 1 and s["outcomes"]["invalid"] >= 2 and s["outcomes"]["incorrect"] >= 1
    assert s["failure_source"].get("infrastructure") is None and s["failure_source"].get("model")
    assert s["detail_categories"].get("unsafe_mql") == 1 and s["detail_categories"].get("format_error") == 1
    assert "correct / 892" in s["primary_metric"] and "exact match" not in s["primary_metric"].lower()
    assert s["accuracy_wilson95"] and s["latency_s"]["n"] == N_EVAL
    assert sum(h["total"] for h in s["by_hardness"].values()) == N_EVAL
    assert not (sd / "comparison_summary.json").exists() and sorted(p.name for p in (sd / "models").iterdir()) == [QWEN]
    report = (sd / "results_report.md").read_text(encoding="utf-8")
    assert "not native MQL exact-match accuracy" in report and "local scoring" in report
    assert "zero-shot Qwen2.5-Coder-3B-Instruct" in report and "gpt-oss" not in report and "deferred" in report
    st_json = json.loads((sd / "RUN_STATUS.json").read_text(encoding="utf-8"))
    assert st_json["scope"].startswith("single model")
    # a completed scoring is never re-scored over
    with pytest.raises(L.ScoringError, match="already completely scored"):
        L.cmd_score(zp, BUNDLE, dest, client_factory=_factory(_oracle_atlas(mapping, refs)), log=None)
    # different predictions are never imported over the completed ones
    with pytest.raises(L.ScoringError, match="DIFFERENT predictions"):
        L.import_predictions(V.safe_extract(predictions_runs[1]["zip"], tmp_path / "other", "PREDICTIONS_STATUS.json"),
                             run_dir / "predictions")
    # tamper with stored outcomes -> the offline scoring verifier fails
    ev = sd / f"models/{QWEN}/evaluation_results.jsonl"
    lines = ev.read_text(encoding="utf-8").splitlines()
    r0 = json.loads(lines[0])
    r0["outcome"] = "incorrect" if r0["outcome"] == "correct" else "correct"
    ev.write_text("\n".join([json.dumps(r0)] + lines[1:]) + "\n", encoding="utf-8")
    bad = V.verify_scoring(run_dir, BUNDLE)
    assert not bad["all_checks_passed"] and any("checksums_match" in p for p in bad["problems"])


@needs_bundle
def test_local_scoring_refuses_partial_and_unverified(predictions_runs, tmp_path):
    refs, mapping = C.load_reference(BUNDLE), C.load_mapping(BUNDLE)
    dest = tmp_path / "outputs"
    with pytest.raises(L.ScoringError, match="PARTIAL"):
        L.cmd_score(predictions_runs[1]["zip"], BUNDLE, dest, client_factory=_factory(_oracle_atlas(mapping, refs)),
                    log=None)
    assert not (dest / "v1").exists()
    st, ver, run_dir = L.cmd_score(predictions_runs[1]["zip"], BUNDLE, dest, allow_partial=True,
                                   client_factory=_factory(_oracle_atlas(mapping, refs)), log=None)
    assert st["status"] == "partial" and "partial" in run_dir.parts and ver["all_checks_passed"]
    s = json.loads((run_dir / f"scoring/models/{QWEN}/baseline_summary.json").read_text(encoding="utf-8"))
    assert s["outcomes"]["failed"] >= 1 and s["failure_source"]["infrastructure"] >= 1   # generation errors
    # a tampered predictions zip is rejected before Atlas is touched
    d = V.safe_extract(predictions_runs[3]["zip"], tmp_path / "t", "PREDICTIONS_STATUS.json")
    (d / f"models/{QWEN}/predictions.jsonl").write_text("{}\n", encoding="utf-8")
    zp = tmp_path / "tampered.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for p in d.rglob("*"):
            z.write(p, f"{d.name}/{p.relative_to(d).as_posix()}")

    def must_not_connect():
        raise AssertionError("Atlas must not be contacted for an unverified predictions bundle")
    with pytest.raises(L.ScoringError, match="failed verification"):
        L.cmd_score(zp, BUNDLE, tmp_path / "o2", client_factory=must_not_connect, log=None)


@needs_bundle
def test_local_atlas_preflight_mocked(tmp_path):
    mapping, refs = C.load_mapping(BUNDLE), C.load_reference(BUNDLE)
    rep, out = L.cmd_preflight(BUNDLE, full_parity=False, dest_root=tmp_path,
                               client_factory=_factory(_oracle_atlas(mapping, refs)), log=None)
    assert rep["ok"] and rep["atlas"]["collections_found"] == 70 and out.is_file()
    assert rep["reference_recheck"]["problems"] == [] and rep["reference_recheck"]["checked"] == N_EVAL
    with pytest.raises(C.PreflightError, match="NOT read-only"):
        L.cmd_preflight(BUNDLE, False, tmp_path, _factory(_oracle_atlas(mapping, refs, actions=("find", "update"))),
                        log=None)
    cl = _oracle_atlas(mapping, refs)
    cl.db.colls.pop("world_1__city")
    with pytest.raises(C.PreflightError, match="missing"):
        L.cmd_preflight(BUNDLE, False, tmp_path, _factory(cl), log=None)
    reports = sorted(tmp_path.glob("preflight/atlas_preflight_*.json"))
    assert reports and not json.loads(reports[-1].read_text(encoding="utf-8"))["ok"]


@needs_bundle
def test_local_preflight_never_prints_the_uri(monkeypatch, capsys, tmp_path):
    """Default client: URI from the environment, redacted in every error; nothing reaches Atlas (localhost:1)."""
    secret_uri = "mongodb://evaluser:S3cretPassw0rd@127.0.0.1:1/?directConnection=true"
    monkeypatch.setenv("MONGODB_URI", secret_uri)
    monkeypatch.setattr(L, "SERVER_SELECTION_TIMEOUT_MS", 300)
    code = L.main(["preflight", "--quick", "--bundle", str(BUNDLE), "--dest-root", str(tmp_path)])
    out = capsys.readouterr()
    assert code == 2 and "S3cretPassw0rd" not in out.out + out.err and "evaluser" not in out.out + out.err
    for f in tmp_path.rglob("*.json"):
        assert "S3cretPassw0rd" not in f.read_text(encoding="utf-8")


@needs_bundle
def test_kaggle_preflight_failure_stops_before_model_load(tmp_path):
    used = {"qwen": 0}

    def qwen_factory():
        used["qwen"] += 1
        raise AssertionError("must not be called")
    with pytest.raises(Exception, match="QWEN_REVISION") as ei:
        _run_notebook(tmp_path / "p", tmp_path / "p.zip", tmp_path, QWEN_REVISION="main",
                      QWEN_GENERATOR_FACTORY=qwen_factory)
    assert type(ei.value).__name__ == "PreflightError" and used["qwen"] == 0
    st = json.loads((tmp_path / "p/PREDICTIONS_STATUS.json").read_text(encoding="utf-8"))
    assert st["status"] == "preflight_failed"
    with pytest.raises(Exception, match="Qwen2.5-Coder-3B-Instruct only"):
        _run_notebook(tmp_path / "q", tmp_path / "q.zip", tmp_path, QWEN_MODEL_ID="Qwen/Qwen2.5-Coder-7B-Instruct",
                      QWEN_GENERATOR_FACTORY=qwen_factory)
    assert used["qwen"] == 0


# ---------------------------------------------------------------- Kaggle's upload decompresses nested .gz files
def _kaggle_style_input(tmp_path, tamper=False):
    """The bundle as Kaggle presents it: data/reference.jsonl.gz arrives decompressed as data/reference.jsonl."""
    inp = tmp_path / "input/datasets/someone/nl2mql-spider-dev-1034-bundle"
    dst = inp / BUNDLE.name
    shutil.copytree(BUNDLE, dst)
    gz = dst / "data/reference.jsonl.gz"
    raw = __import__("gzip").decompress(gz.read_bytes())
    if tamper:
        lines = raw.decode("utf-8").splitlines()
        r0 = json.loads(lines[0])
        r0["gold_rows"] = [[999]]          # stored digest fields left as they were
        raw = ("\n".join([json.dumps(r0, ensure_ascii=False)] + lines[1:]) + "\n").encode("utf-8")
    (dst / "data/reference.jsonl").write_bytes(raw)
    gz.unlink()
    return tmp_path / "input", dst


@needs_bundle
@pytest.mark.parametrize("level,expect", [(9, "strict"), (1, "content was verified")])
def test_notebook_restores_kaggle_decompressed_reference(tmp_path, level, expect):
    root, dst = _kaggle_style_input(tmp_path)
    before = V.tree_sums(dst)
    out, zp = tmp_path / "w/nl2mql_spider_dev_predictions_v2", tmp_path / "w/preds.zip"
    ns = _run_notebook(out, zp, root, BUNDLE_DIR=None, BUNDLE_RESTORE_ROOT=str(tmp_path / "restore"), _GZIP_LEVEL=level)
    st = json.loads((out / "PREDICTIONS_STATUS.json").read_text(encoding="utf-8"))
    assert st["status"] == "complete" and st["bundle_restored_files"] == ["data/reference.jsonl.gz"]
    assert expect in st["bundle_check"] and st["bundle_manifest_sha256"] == C.sha256_file(BUNDLE / "bundle_manifest.json")
    assert V.tree_sums(dst) == before                         # the (read-only) input itself is never modified
    assert Path(ns["BUNDLE"]).is_relative_to(tmp_path / "restore")
    rep = V.verify_predictions(V.safe_extract(zp, tmp_path / "vp", "PREDICTIONS_STATUS.json"), BUNDLE)
    assert rep["all_checks_passed"] and rep["complete"], rep["problems"]


@needs_bundle
@pytest.mark.parametrize("level", [9, 1])
def test_notebook_rejects_tampered_decompressed_reference(tmp_path, level):
    root, _ = _kaggle_style_input(tmp_path, tamper=True)
    with pytest.raises(Exception, match="reference") as ei:
        _run_notebook(tmp_path / "w/o", tmp_path / "w/o.zip", root, BUNDLE_DIR=None,
                      BUNDLE_RESTORE_ROOT=str(tmp_path / "restore"), _GZIP_LEVEL=level)
    assert type(ei.value).__name__ == "PreflightError"


@needs_bundle
def test_notebook_untouched_bundle_needs_no_restoration(tmp_path):
    out, zp = tmp_path / "w/o", tmp_path / "w/o.zip"
    _run_notebook(out, zp, tmp_path, BUNDLE_RESTORE_ROOT=str(tmp_path / "restore"))
    st = json.loads((out / "PREDICTIONS_STATUS.json").read_text(encoding="utf-8"))
    assert st["bundle_restored_files"] == [] and st["bundle_check"].startswith("strict") and not (tmp_path / "restore").exists()


# ---------------------------------------------------------------- CUDA OOM: batch-size-1 fallback + verified resume
def _attach(src_dir, input_root, zip_path=None):
    """An earlier version's output as Kaggle attaches it: its folder, plus (optionally) its extracted zip."""
    base = input_root / "notebooks/someone/mql-0006-qwen"
    shutil.copytree(src_dir, base / src_dir.name)
    if zip_path is not None:
        V.safe_extract(zip_path, base / "nl2mql_spider_dev_predictions", "PREDICTIONS_STATUS.json")
    return base / src_dir.name


def _lines(out):
    return [ln for ln in (out / f"models/{QWEN}/predictions.jsonl").read_text(encoding="utf-8").splitlines() if ln]


@pytest.fixture(scope="module")
def oom_runs(tmp_path_factory):
    """Session 1: 424 completed, the rest generation errors (as the real Kaggle run: 716 + 176).
    Session 2: RESUME_FROM="auto"; the three batches holding a "long" prompt OOM at batch size 8, and one of those
    prompts OOMs even alone. Session 3: resumes from session 2 and completes."""
    tmp = tmp_path_factory.mktemp("oom")
    out1, z1 = tmp / "w1/nl2mql_spider_dev_predictions_v2", tmp / "w1/preds.zip"
    (tmp / "in1").mkdir()
    _run_notebook(out1, z1, tmp / "in1", fail_after=424)
    ids = [r["id"] for r in C.load_dev(BUNDLE)]
    todo = ids[424:]
    long_ids, single_oom = set(todo[16:40]), todo[20]      # queue batches 3-5: 3 full batches, 24 prompts
    in2 = tmp / "in2"
    prev1 = _attach(out1, in2, z1)                          # attached twice: folder + extracted zip
    inputs_before = V.tree_sums(in2)
    calls2, calls3 = [], []
    out2 = tmp / "w2/nl2mql_spider_dev_predictions_v2"
    ns2 = _run_notebook(out2, tmp / "w2/preds.zip", in2, RESUME_FROM="auto", RESUME_EXPECTED_OK=424, calls=calls2,
                        oom_if=lambda b: any(p["id"] in long_ids for p in b) and (len(b) > 1 or b[0]["id"] == single_oom))
    in3 = tmp / "in3"
    _attach(out2, in3)
    out3, z3 = tmp / "w3/nl2mql_spider_dev_predictions_v2", tmp / "w3/preds.zip"
    ns3 = _run_notebook(out3, z3, in3, RESUME_FROM="auto", calls=calls3)
    return {"tmp": tmp, "ids": ids, "todo": todo, "long_ids": long_ids, "single_oom": single_oom, "out1": out1,
            "prev1": prev1, "inputs_before": inputs_before, "in2": in2, "calls2": calls2, "ns2": ns2, "out2": out2,
            "calls3": calls3, "ns3": ns3, "out3": out3, "z3": z3}


@needs_bundle
def test_oom_batch_falls_back_to_batch_size_one(oom_runs):
    r = oom_runs
    preds = {p["id"]: p for p in C.read_jsonl(r["out2"] / f"models/{QWEN}/predictions.jsonl")}
    ok1 = {p["id"] for p in C.read_jsonl(r["out1"] / f"models/{QWEN}/predictions.jsonl") if p["status"] == "ok"}
    assert len(ok1) == 424
    # only the unfinished ids were generated; each OOM batch's prompts were retried alone exactly once
    flat = Counter(i for c in r["calls2"] for i in c)
    assert set(flat) == set(r["todo"]) and not set(flat) & ok1
    oom_batches = [c for c in r["calls2"] if len(c) > 1 and set(c) & r["long_ids"]]
    in_oom = {i for c in oom_batches for i in c}
    assert len(oom_batches) == 3 and len(in_oom) == 24
    assert all(flat[i] == 2 for i in in_oom) and all(flat[i] == 1 for i in set(flat) - in_oom)
    assert r["ns2"]["OOM_EVENTS"] == {"batch": 3, "single_prompt": 1} and r["ns2"]["_cleanups"][0] == 4
    # per-prediction metadata
    for i in r["todo"]:
        p = preds[i]
        assert p["configured_batch_size"] == 8
        if i in in_oom:
            assert p["oom_recovery"] is True and p["effective_batch_size"] == 1 and p["oom_retries"] == 1
        else:
            assert p["oom_recovery"] is False and p["oom_retries"] == 0 and p["status"] == "ok"
            assert p["effective_batch_size"] == len(next(c for c in r["calls2"] if i in c))
    lone = preds[r["single_oom"]]          # OOM even alone: recorded explicitly, not retried again this session
    assert lone["status"] == "error" and "CUDA out of memory" in lone["error"] and "at batch size 1" in lone["error"]
    assert all(preds[i]["status"] == "ok" for i in in_oom - {r["single_oom"]})
    # the 424 earlier predictions are carried over byte for byte; a run with a failure is never complete
    before = {json.loads(ln)["id"]: ln for ln in _lines(r["out1"])}
    after = {json.loads(ln)["id"]: ln for ln in _lines(r["out2"])}
    assert all(after[i] == before[i] for i in ok1)
    st = json.loads((r["out2"] / "PREDICTIONS_STATUS.json").read_text(encoding="utf-8"))
    g = st["models"][QWEN]["generation"]
    assert st["status"] == "partial" and g["remaining_failures"] == 1
    assert g["completed_batch_size_1_fallback"] == 23 and g["completed_normal_batch"] == N_EVAL - 24
    assert g["oom_events_this_session"] == {"batch": 3, "single_prompt": 1}
    assert g["oom_error_records_earlier_sessions"] == N_EVAL - 424 and g["earlier_completed_kept_byte_identical"] == 424
    assert "minor numerical differences" in g["note"]
    assert g["resumed_from"] in (str(r["prev1"]),    # either attached copy: its folder or its identical extracted zip
                                 str(r["prev1"].parent / "nl2mql_spider_dev_predictions" / r["prev1"].name))
    assert g["generation_wall_s_total"] >= g["generation_wall_s_this_session"]
    # same fingerprint as the resumed run; the attached inputs were never modified
    fp = lambda d: json.loads((d / f"models/{QWEN}/run_config.json").read_text(encoding="utf-8"))["fingerprint"]
    assert fp(r["out2"]) == fp(r["out1"]) and {p["fingerprint"] for p in preds.values()} == {fp(r["out1"])}
    assert V.tree_sums(r["in2"]) == r["inputs_before"]
    rep = V.verify_predictions(V.safe_extract(r["tmp"] / "w2/preds.zip", r["tmp"] / "vp2", "PREDICTIONS_STATUS.json"),
                               BUNDLE)
    assert rep["all_checks_passed"] and not rep["complete"]


@needs_bundle
def test_oom_resume_completes_without_regenerating(oom_runs):
    r = oom_runs
    assert r["calls3"] == [[r["single_oom"]]]                 # only the one remaining failure is generated
    lines2 = {json.loads(ln)["id"]: ln for ln in _lines(r["out2"])}
    lines3 = _lines(r["out3"])
    recs3 = [json.loads(ln) for ln in lines3]
    assert [p["id"] for p in recs3] == r["ids"] and len({p["id"] for p in recs3}) == N_EVAL
    assert all(ln == lines2[p["id"]] for ln, p in zip(lines3, recs3) if p["id"] != r["single_oom"])
    st = json.loads((r["out3"] / "PREDICTIONS_STATUS.json").read_text(encoding="utf-8"))
    g = st["models"][QWEN]["generation"]
    assert st["status"] == "complete", st["problems"]
    assert g["remaining_failures"] == 0 and g["earlier_completed_kept_byte_identical"] == N_EVAL - 1
    assert g["completed_batch_size_1_fallback"] == 23                 # session 2's fallback predictions are kept
    assert g["oom_events_this_session"] == {"batch": 0, "single_prompt": 0}
    rep = V.verify_predictions(V.safe_extract(r["z3"], r["tmp"] / "vp3", "PREDICTIONS_STATUS.json"), BUNDLE)
    assert rep["all_checks_passed"] and rep["complete"], rep["problems"]


@needs_bundle
def test_non_oom_exception_is_not_treated_as_oom(tmp_path):
    calls = []
    ns = _run_notebook(tmp_path / "o", tmp_path / "o.zip", tmp_path, calls=calls,
                       raise_if=lambda b: RuntimeError("CUDA error: device-side assert triggered")
                       if any(p["id"] == "spider_dev_0020" for p in b) else None)
    assert ns["OOM_EVENTS"] == {"batch": 0, "single_prompt": 0} and ns["_cleanups"][0] == 0
    assert ns["MODEL_ERROR"].startswith("RuntimeError: CUDA error") and not [c for c in calls if len(c) == 1]
    st = json.loads((tmp_path / "o/PREDICTIONS_STATUS.json").read_text(encoding="utf-8"))
    assert st["status"] == "partial"
    assert not [p for p in C.read_jsonl(tmp_path / f"o/models/{QWEN}/predictions.jsonl") if p.get("oom_recovery")]
    # the real generator re-raises only torch.cuda.OutOfMemoryError; every other generate() error stays a record
    src = next(c.source for c in _cells() if c.metadata["tags"][0] == "qwen")
    assert re.search(r"except torch\.cuda\.OutOfMemoryError:\n\s+raise\b.*\n\s+except Exception as e:", src)
    assert "isinstance(e, torch.cuda.OutOfMemoryError)" in src
    assert src.count("max_new_tokens=QWEN_MAX_NEW_TOKENS") == 1 and "do_sample=False" in src


@contextlib.contextmanager
def _nb_refused(match):
    """A PreflightError from the notebook's own dev1034_core (loaded from the bundle, so another class object)."""
    with pytest.raises(Exception, match=match) as ei:
        yield
    assert type(ei.value).__name__ == "PreflightError"


@needs_bundle
def test_resume_source_is_verified_before_use(oom_runs, tmp_path):
    r = oom_runs
    out = lambda n: tmp_path / f"{n}/nl2mql_spider_dev_predictions_v2"
    # wrong expected count
    with _nb_refused("RESUME_EXPECTED_OK"):
        _run_notebook(out("a"), tmp_path / "a.zip", tmp_path, RESUME_FROM=str(r["prev1"]), RESUME_EXPECTED_OK=716)
    # a changed earlier prediction
    t = tmp_path / "t/nl2mql_spider_dev_predictions_v2"
    shutil.copytree(r["prev1"], t)
    f = t / f"models/{QWEN}/predictions.jsonl"
    f.write_text(f.read_text(encoding="utf-8").replace('"status": "ok"', '"status":  "ok"', 1), encoding="utf-8")
    with _nb_refused("hash mismatch"):
        _run_notebook(out("b"), tmp_path / "b.zip", tmp_path, RESUME_FROM=str(t))
    # an id completed twice in the checkpoint (checksums re-signed, so only the conflict check can catch it)
    c = tmp_path / "c/nl2mql_spider_dev_predictions_v2"
    shutil.copytree(r["prev1"], c)
    rel = f"models/{QWEN}/checkpoints/predictions.partial.jsonl"
    ckf = c / rel
    first_ok = next(ln for ln in ckf.read_text(encoding="utf-8").splitlines() if '"status": "ok"' in ln)
    ckf.write_text(ckf.read_text(encoding="utf-8") + first_ok + "\n", encoding="utf-8")
    sums = c / "SHA256SUMS.txt"
    sums.write_text("".join(f"{C.sha256_file(ckf)}  {rel}\n" if ln.endswith("  " + rel) else ln + "\n"
                            for ln in sums.read_text(encoding="utf-8").splitlines()), encoding="utf-8")
    with _nb_refused("completed twice"):
        _run_notebook(out("c"), tmp_path / "c.zip", tmp_path, RESUME_FROM=str(c))
    # a different dtype (or GPU / library) in this session: refused before anything is generated
    calls = []
    ns = _run_notebook(out("d"), tmp_path / "d.zip", tmp_path, RESUME_FROM=str(r["prev1"]), calls=calls,
                       info={"dtype": "bfloat16", "gpu": "FakeGPU", "torch": "x", "transformers": "y"})
    assert calls == [] and "dtype" in ns["MODEL_ERROR"] and "refusing to mix" in ns["MODEL_ERROR"]
    st = json.loads((out("d") / "PREDICTIONS_STATUS.json").read_text(encoding="utf-8"))
    assert st["status"] == "partial"
    # two DIFFERENT attached earlier outputs: "auto" refuses to guess
    two = tmp_path / "two"
    _attach(r["prev1"], two)
    shutil.copytree(r["out2"], two / "notebooks/other/x/nl2mql_spider_dev_predictions_v2")
    with _nb_refused("different earlier outputs"):
        _run_notebook(out("e"), tmp_path / "e.zip", two, RESUME_FROM="auto")
