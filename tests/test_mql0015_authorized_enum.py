"""MQL-0015 find-only evaluator: collection enumeration via listCollections(nameOnly, authorizedCollections).

The fake server below is STRICT about privileges, like MongoDB:
- plain list_collection_names() (listCollections without authorizedCollections) needs the `listCollections` action;
- listCollections with BOTH nameOnly and authorizedCollections returns only the collections the user holds a privilege
  on, without needing `listCollections`.
The pinned preflight / readiness code is exercised unchanged through the wrapper's client proxy.
"""
import copy
import sys
from pathlib import Path

import pytest
from pymongo.errors import OperationFailure

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(1, str(ROOT / "scripts/nl2mql_mql0015"))
from test_mql0015 import FakeDB, FakeMongo, PR, SCO, T, synth   # noqa: E402,F401  (synth is a pytest fixture)
import mql0015_scoring_authorized as AUTH                         # noqa: E402

FIND_ONLY = [{"resource": {"db": T, "collection": ""}, "actions": ["find"]}]
# the real codegen_eval_readonly also keeps its earlier read-only roles on other databases
FIND_ONLY_PLUS_EXISTING = FIND_ONLY + [
    {"resource": {"db": "codegen_spider1", "collection": ""}, "actions": ["find", "listCollections", "killCursors"]},
    {"resource": {"db": "codegen_spider1_heldout", "collection": ""},
     "actions": ["find", "listCollections", "listIndexes", "dbStats", "collStats", "killCursors", "changeStream"]}]


def _unauthorized(what):
    return OperationFailure(f"not authorized on {T} to execute command {what}", code=13)


def _actions_on(privs, db):
    return {a for p in privs if p["resource"].get("db") in (db, "") for a in p["actions"]}


class StrictDB(FakeDB):
    BATCH = 1   # one name per batch: every enumeration in these tests goes through getMore

    def list_collection_names(self):
        if "listCollections" not in _actions_on(self.srv.privileges, self.name):
            raise _unauthorized("{ listCollections: 1, nameOnly: true }")
        return super().list_collection_names()

    def command(self, cmd):
        if cmd == "dbStats":
            raise _unauthorized("dbStats")
        if isinstance(cmd, dict) and next(iter(cmd)) == "listCollections":
            self.srv.list_cmds.append(dict(cmd))
            authorized = cmd.get("nameOnly") is True and cmd.get("authorizedCollections") is True
            if not authorized and "listCollections" not in _actions_on(self.srv.privileges, self.name):
                raise _unauthorized(str(cmd))
            names = super().list_collection_names() if _actions_on(self.srv.privileges, self.name) else []
            self.srv.pending = names[self.BATCH:]
            return {"cursor": {"id": 42 if self.srv.pending else 0, "ns": f"{self.name}.$cmd.listCollections",
                               "firstBatch": [{"name": n, "type": "collection"} for n in names[:self.BATCH]]}, "ok": 1}
        if isinstance(cmd, dict) and next(iter(cmd)) == "getMore":
            assert cmd["getMore"] == 42 and cmd["collection"] == "$cmd.listCollections"
            nxt, self.srv.pending = self.srv.pending[:self.BATCH], self.srv.pending[self.BATCH:]
            return {"cursor": {"id": 42 if self.srv.pending else 0, "nextBatch": [{"name": n} for n in nxt]}, "ok": 1}
        return {"authInfo": {"authenticatedUsers": [{"user": "codegen_eval_readonly", "db": "admin"}],
                             "authenticatedUserPrivileges": self.srv.privileges}}


class StrictMongo(FakeMongo):
    def __init__(self, privileges=None):
        super().__init__(privileges)
        self.list_cmds, self.pending = [], []

    def __getitem__(self, n):
        self.ops.append(("db", n))
        return StrictDB(self, n)


@pytest.fixture()
def provisioned(synth):
    srv = StrictMongo()                                        # provisioning role (has listCollections) for seeding
    PR.provision(srv, synth["contract"], synth["bundle"], log=None)
    srv.privileges = list(FIND_ONLY_PLUS_EXISTING)             # then switch to the find-only evaluator
    srv.ops.clear()
    return srv, {**synth["contract"], "atlas_db": T}


def _quiet(*a):
    return None


def test_problem_is_real_unwrapped_find_only_user_cannot_enumerate(provisioned):
    srv, mapping = provisioned
    with pytest.raises(OperationFailure, match="not authorized"):
        SCO.readiness(srv, mapping, log=_quiet)                # pinned path, unwrapped client


def test_wrapped_find_only_user_passes_readiness(provisioned):
    srv, mapping = provisioned
    rep = SCO.readiness(AUTH.AuthorizedClient(srv), mapping, log=_quiet)
    assert rep["ok"] and rep["atlas_db"] == T
    assert rep["collections_found"] == len(mapping["collections"]) and not rep["missing_collections"]
    assert not rep["offending_actions"] and not rep["unexpected_dev_collections"]
    assert all(v["digest_match"] for v in rep["parity"].values()) and len(rep["parity"]) == len(mapping["collections"])
    assert rep["counts"] == {c: m["documents"] for c, m in mapping["collections"].items()}
    assert "listCollections" not in rep["privileges"][f"{T}."]           # really find-only on the target
    assert srv.list_cmds == [AUTH.LIST_COLLECTIONS_CMD]                    # exact command, once
    assert not [o for o in srv.ops if o[0].startswith(("DESTRUCTIVE", "insert"))]   # read-only


def test_wrapped_still_refuses_write_capable_account(provisioned):
    import dev1034_core as D
    srv, mapping = provisioned
    srv.privileges = [{"resource": {"db": T, "collection": ""}, "actions": ["find", "insert"]}]
    with pytest.raises(D.PreflightError, match="NOT read-only"):
        SCO.readiness(AUTH.AuthorizedClient(srv), mapping, log=_quiet)


def test_wrapped_still_requires_find_on_target(provisioned):
    import dev1034_core as D
    srv, mapping = provisioned
    srv.privileges = [{"resource": {"db": "codegen_spider1_test_mql0013", "collection": ""}, "actions": ["find"]}]
    with pytest.raises(D.PreflightError, match="no 'find'"):
        SCO.readiness(AUTH.AuthorizedClient(srv), mapping, log=_quiet)


@pytest.mark.parametrize("damage", ["missing", "digest", "unexpected", "count"])
def test_wrapped_keeps_integrity_checks(provisioned, damage):
    import dev1034_core as D
    srv, mapping = provisioned
    data = srv.data[T]
    name = sorted(mapping["collections"])[0]
    if damage == "missing":
        data[name] = []
        match = "missing"
    elif damage == "digest":
        doc = data[name][0]
        k = next(k for k in doc if k != "_id")
        doc[k] = "tampered" if not isinstance(doc[k], str) else doc[k] + "x"
        match = "differ|digest"
    elif damage == "unexpected":
        data[f"{mapping['db_ids'][0]}__intruder"] = [{"x": 1}]
        match = "unexpected"
    else:
        data[name].append(copy.deepcopy(data[name][0]))
        match = "documents on Atlas"
    with pytest.raises(D.PreflightError, match=match):
        SCO.readiness(AUTH.AuthorizedClient(srv), mapping, log=_quiet)


@pytest.mark.parametrize("drop", ["nameOnly", "authorizedCollections"])
def test_fake_is_strict_without_both_flags(provisioned, drop):
    srv, _ = provisioned
    cmd = {k: v for k, v in AUTH.LIST_COLLECTIONS_CMD.items() if k != drop}
    with pytest.raises(OperationFailure, match="not authorized"):
        srv[T].command(cmd)


def test_enumeration_follows_getmore_and_parses_names(provisioned):
    srv, mapping = provisioned
    db = srv[T]
    calls = []
    real = db.command
    db.command = lambda c: (calls.append(next(iter(c))), real(c))[1]
    names = AUTH.authorized_collection_names(db)
    assert len(mapping["collections"]) > StrictDB.BATCH                      # several batches were needed
    assert calls == ["listCollections"] + ["getMore"] * (len(mapping["collections"]) - 1)
    assert sorted(names) == sorted(mapping["collections"])


def test_enumeration_fails_closed_on_bad_reply():
    class Bad:
        def command(self, cmd):
            return {"ok": 1}
    with pytest.raises(AUTH.EnumerationError):
        AUTH.authorized_collection_names(Bad())

    class Dup:
        def command(self, cmd):
            return {"cursor": {"id": 0, "firstBatch": [{"name": "a"}, {"name": "a"}]}}
    with pytest.raises(AUTH.EnumerationError, match="duplicate"):
        AUTH.authorized_collection_names(Dup())


def test_proxy_changes_only_target_enumeration(provisioned):
    srv, _ = provisioned
    cl = AUTH.AuthorizedClient(srv)
    assert isinstance(cl[T], AUTH._AuthorizedDB)
    assert type(cl["codegen_spider1"]) is StrictDB                           # other databases untouched
    assert type(cl[T][sorted(srv.data[T])[0]]).__name__ == "FakeColl"        # collections are the real objects
    assert cl.admin is srv.admin and cl[T].name == T
    with pytest.raises(AUTH.EnumerationError):
        cl[T].list_collection_names(filter={"name": "x"})


def test_wrap_factory_keeps_redaction():
    srv = StrictMongo(FIND_ONLY)
    redact = lambda s: s.replace("secret", "<credentials>")                  # noqa: E731
    client, r = AUTH.wrap_factory(lambda: (srv, redact))()
    assert isinstance(client, AUTH.AuthorizedClient) and r is redact


def test_pinned_files_unchanged():
    P = SCO._core(SCO.CM.PKG_OUT).load_package(SCO.CM.PKG_OUT)
    for rel, h in P["cfg"]["scoring_code_sha256"].items():
        assert SCO.CM.sha256_file(ROOT / rel) == h, rel
