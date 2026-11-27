"""Atlas TC-1 / LIKE checks with a FAKE client (no network). Plan §E.3 item 3."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
import atlas_type_contract_check as A  # noqa: E402

CONTRACT = {"atlas_db": "db", "contract": "TC-1", "body_sha256": "x", "collections": {
    "d__t": {"db_id": "d", "table": "t", "rows": 2, "fields": {"a": {"K": "int", "nonnull": 2},
                                                              "b": {"K": "text", "nonnull": 1}}},
    "d__u": {"db_id": "d", "table": "u", "rows": 1, "fields": {"c": {"K": "num", "nonnull": 1}}}}}
GOOD = {"d__t": [{"a": 1, "b": "x"}, {"a": 2, "b": None}], "d__u": [{"c": 1.5}]}
RO = {"authInfo": {"authenticatedUsers": [{"user": "codegen_eval_readonly"}],
                   "authenticatedUserPrivileges": [{"resource": {"db": "db", "collection": ""}, "actions": ["find"]}]}}
RW = {"authInfo": {"authenticatedUsers": [{"user": "admin"}],
                   "authenticatedUserPrivileges": [{"resource": {"db": "db", "collection": ""}, "actions": ["find", "insert"]}]}}
SECRET = "mongodb+srv://user:hunter2@example.invalid/"


def bson_type(v):
    return {bool: "bool", int: "int", float: "double", str: "string", type(None): "null", list: "array",
            dict: "object"}[type(v)]


class Coll:
    def __init__(self, docs):
        self.docs = docs

    def count_documents(self, q):
        return len(self.docs)

    def aggregate(self, pipe, **kw):
        assert pipe == A.TC.TYPE_CENSUS_PIPELINE
        out = {}
        for d in self.docs:
            for k, v in dict(d, _id="oid").items():
                t = "objectId" if k == "_id" else bson_type(v)
                out[(k, t)] = out.get((k, t), 0) + 1
        return [{"_id": {"f": k, "t": t}, "n": n} for (k, t), n in out.items()]


class DB:
    def __init__(self, data, status):
        self.data, self.status = data, status

    def command(self, c):
        return self.status

    def list_collection_names(self):
        return list(self.data)

    def __getitem__(self, k):
        return Coll(self.data[k])


class Client:
    def __init__(self, data, status=RO):
        self.db = DB(data, status)
        self.admin = self

    def command(self, c):
        return {"version": "8.0.99"} if c == "buildInfo" else {"ok": 1}

    def __getitem__(self, k):
        return self.db


def test_pass():
    rep = A.check_contract(Client(GOOD), CONTRACT)
    assert rep["ok"] and rep["collections_checked"] == 2 and not rep["violations"]


@pytest.mark.parametrize("mut,needle", [
    (lambda d: d["d__t"][0].update(b=5), "disallowed_type"),
    (lambda d: d["d__u"][0].update(c="1.5"), "disallowed_type"),
    (lambda d: d["d__t"][0].update(a=True), "forbidden_type"),
    (lambda d: d["d__t"][0].update(zz=1), "extra_field"),
    (lambda d: d["d__t"].append({"a": 3, "b": "y"}), "count"),
    (lambda d: d["d__u"][0].update(c=[1]), "forbidden_type"),
])
def test_fails_closed(mut, needle):
    data = copy.deepcopy(GOOD)
    mut(data)
    rep = A.check_contract(Client(data), CONTRACT)
    assert not rep["ok"]
    assert any(needle in x for v in rep["violations"].values() for x in v), rep["violations"]


def test_missing_collection():
    data = copy.deepcopy(GOOD)
    del data["d__u"]
    rep = A.check_contract(Client(data), CONTRACT)
    assert not rep["ok"] and rep["missing_collections"] == ["d__u"]


def test_write_capable_user_refused():
    with pytest.raises(A.CheckError):
        A.check_contract(Client(GOOD, RW), CONTRACT)


def test_main_never_prints_uri_and_marks_pending(tmp_path, capsys, monkeypatch):
    def factory():
        raise RuntimeError("cannot connect " + SECRET.replace(SECRET, "<MONGODB_URI>"))
    code = A.main(["contract", "--experiment", "_test_tmp_atlas_check"], client_factory=factory)
    out = capsys.readouterr().out
    assert code == 2 and "pending_unreachable" in out and "hunter2" not in out
    d = ROOT / "outputs/_test_tmp_atlas_check/preflight"
    for p in d.glob("*.json"):
        assert "hunter2" not in p.read_text(encoding="utf-8")
        assert json.loads(p.read_text(encoding="utf-8"))["status"] == "pending_unreachable"
        p.unlink()
    d.rmdir()
    d.parent.rmdir()


def test_main_fail_on_violation(capsys):
    data = copy.deepcopy(GOOD)
    data["d__t"][0]["b"] = 5
    real = A.TC.load_manifest
    A.TC.load_manifest = lambda p: CONTRACT
    try:
        code = A.main(["contract", "--experiment", "_test_tmp_atlas_check2"],
                      client_factory=lambda: (Client(data), lambda s: s))
    finally:
        A.TC.load_manifest = real
    assert code == 1
    d = ROOT / "outputs/_test_tmp_atlas_check2/preflight"
    for p in d.glob("*.json"):
        assert json.loads(p.read_text(encoding="utf-8"))["status"] == "fail"
        p.unlink()
    d.rmdir()
    d.parent.rmdir()


def test_like_cases_sqlite_truth():
    t = {p: A.sqlite_like(v, p) for p, v in A.LIKE_CASES}
    assert t["k"] == [0, 1] and t["S"] == [0, 1] and t["Ⓐ"] == [0] and t["a_b"] == [0, 1]
