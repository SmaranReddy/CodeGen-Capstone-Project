"""MQL-0009 held-out Atlas provisioning wrapper: mocked clients + synthetic SQLite only. No MongoDB, no network."""
import copy
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0009"))
sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
sys.path.insert(0, str(ROOT / "scripts/spider1"))
import provision_heldout_atlas as PV   # noqa: E402
import seed_atlas_spider_dev as SD     # noqa: E402
import dev1034_core as D               # noqa: E402

T = PV.TARGET_DB


# ------------------------------------------------------------------ fake MongoDB
class Cursor(list):
    def close(self):
        pass


class FakeColl:
    def __init__(self, srv, db, name):
        self.srv, self.db, self.name = srv, db, name

    def _docs(self):
        return self.srv.data.setdefault(self.db, {}).setdefault(self.name, [])

    def insert_many(self, docs, ordered=True):
        self.srv.ops.append(("insert_many", self.db, self.name, len(docs)))
        if self.srv.fail_insert_on == self.name:
            raise ConnectionError("network blip (fake)")
        docs = [dict(d) for d in docs]
        if self.srv.corrupt_on == self.name:
            docs[0][next(iter(docs[0]))] = "CORRUPTED"
        if self.srv.lose_one_on == self.name:
            docs = docs[:-1]
        self._docs().extend(docs)

    def find(self, q=None, proj=None, batch_size=None):
        self.srv.ops.append(("find", self.db, self.name))
        return Cursor({k: v for k, v in d.items() if k != "_id"} for d in self._docs())

    def count_documents(self, q):
        return len(self._docs())

    def __getattr__(self, a):                       # drop, delete_*, update_*, replace_*, rename, aggregate, ...
        def destructive(*args, **kw):
            self.srv.ops.append(("DESTRUCTIVE:" + a, self.db, self.name))
        return destructive


class FakeDB:
    def __init__(self, srv, name):
        self.srv, self.name = srv, name

    def list_collection_names(self):
        self.srv.ops.append(("list_collection_names", self.name))
        return [n for n, v in self.srv.data.get(self.name, {}).items() if v or n in self.srv.created_empty.get(self.name, set())]

    def __getitem__(self, n):
        return FakeColl(self.srv, self.name, n)

    def command(self, cmd):
        assert "connectionStatus" in cmd
        return {"authInfo": {"authenticatedUsers": [{"user": "codegen_heldout_provision", "db": "admin"}],
                             "authenticatedUserPrivileges": self.srv.privileges}}

    def __getattr__(self, a):
        def destructive(*args, **kw):
            self.srv.ops.append(("DESTRUCTIVE:" + a, self.name))
        return destructive


class Admin:
    def command(self, c):
        return {"ok": 1} if c == "ping" else {"version": "8.0.fake"}


class FakeMongo:
    def __init__(self, privileges=None):
        self.data, self.ops, self.created_empty = {}, [], {}
        self.privileges = privileges if privileges is not None else minimal_role()
        self.fail_insert_on = self.corrupt_on = self.lose_one_on = None
        self.admin = Admin()

    def __getitem__(self, n):
        self.ops.append(("db", n))
        return FakeDB(self, n)

    def close(self):
        pass

    def __getattr__(self, a):
        def destructive(*args, **kw):
            self.ops.append(("DESTRUCTIVE:" + a,))
        return destructive

    def destructive_ops(self):
        return [o for o in self.ops if str(o[0]).startswith("DESTRUCTIVE")]


def minimal_role(db=T):
    return [{"resource": {"db": db, "collection": ""}, "actions": ["find", "insert", "listCollections", "createCollection"]}]


# ------------------------------------------------------------------ synthetic SQLite + contract (via the real builder derivation)
SCHEMA = {
    "alpha_db": {"owner": ("CREATE TABLE owner (id INTEGER PRIMARY KEY, name TEXT, score REAL)",
                           [(1, "Ann", 3.5), (2, "Bo", None), (3, "Cy", 7.0)]),
                 "pet": ("CREATE TABLE pet (pid INTEGER, owner_id INTEGER, kind TEXT)", [(10, 1, "cat"), (11, 1, "dog"), (12, 3, "fish")])},
    "beta_db": {"item": ("CREATE TABLE item (code TEXT, qty INTEGER)", [("A1", 5), ("B2", 0)])},
    "gamma": {"log": ("CREATE TABLE log (n INTEGER, v TEXT)", [(i, f"v{i}") for i in range(25)])},
}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    base = tmp_path_factory.mktemp("spider")
    for db, tabs in SCHEMA.items():
        (base / "database" / db).mkdir(parents=True)
        con = sqlite3.connect(base / "database" / db / f"{db}.sqlite")
        for ddl, rows in tabs.values():
            con.execute(ddl)
            tname = ddl.split()[2]
            con.executemany(f"INSERT INTO {tname} VALUES ({','.join('?' * len(rows[0]))})", rows)
        con.commit()
        con.close()
    import spider1_kit as K
    import build_mql0009_package as B
    saved = K.DB_DIR
    try:
        _, _, contract = B.derive_context(list(SCHEMA), db_dir=base / "database")
    finally:
        K.DB_DIR = saved
    tables = {"alpha_db": {"table_names_original": ["owner", "pet"],
                           "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [0, "score"], [1, "pid"], [1, "owner_id"], [1, "kind"]],
                           "foreign_keys": [[5, 1]]},
              "beta_db": {"table_names_original": ["item"], "column_names_original": [[-1, "*"], [0, "code"], [0, "qty"]], "foreign_keys": []},
              "gamma": {"table_names_original": ["log"], "column_names_original": [[-1, "*"], [0, "n"], [0, "v"]], "foreign_keys": []}}
    return {"bundle": base, "contract": contract, "tables": tables}


def run(srv, synth, contract=None):
    return PV.provision(srv, contract or synth["contract"], bundle=synth["bundle"], log=None)


# ------------------------------------------------------------------ destination enforcement
def test_destination_database_enforced():
    PV.check_target(T)
    for bad in ("codegen_spider1", "admin", "local", "config", "codegen_spider1_heldoutX", "", "CODEGEN_SPIDER1_HELDOUT"):
        with pytest.raises(PV.ProvisionError):
            PV.check_target(bad)
    with pytest.raises(PV.ProvisionError):
        PV.check_target(T, confirm=None, execute=True)
    PV.check_target(T, confirm=T, execute=True)
    with pytest.raises(PV.ProvisionError):
        PV.provision(FakeMongo(), {"db_ids": [], "collections": {}}, target_db="codegen_spider1")
    assert PV.main(["--target-db", "codegen_spider1"]) == 2


def test_guard_refuses_other_databases_and_destructive_operations():
    srv = FakeMongo()
    g = PV.GuardedClient(srv, T, insertable={"a__t"})
    for f in (lambda: g["codegen_spider1"], lambda: g["admin"], lambda: g.drop_database(T),
              lambda: g[T].drop_collection("a__t"), lambda: g[T].command({"dropDatabase": 1}),
              lambda: g[T]["a__t"].drop(), lambda: g[T]["a__t"].delete_many({}), lambda: g[T]["a__t"].update_one({}, {}),
              lambda: g[T]["a__t"].replace_one({}, {}), lambda: g[T]["a__t"].aggregate([{"$out": "x"}]),
              lambda: g[T]["a__t"].rename("b"), lambda: g[T]["a__t"].create_index("x"), lambda: g[T]["a__t"].bulk_write([]),
              lambda: g[T]["other__t"].insert_many([{"x": 1}]), lambda: g[T]["a__t"].insert_many([{"x": 1}], bypass_document_validation=True)):
        with pytest.raises(PV.DestructiveOperationRefused):
            f()
    g[T]["a__t"].insert_many([{"x": 1}])
    assert srv.destructive_ops() == []
    assert {o[0] for o in srv.ops} <= {"db", "insert_many"}


# ------------------------------------------------------------------ mapping
def test_collision_and_mapping_checks(synth):
    c = synth["contract"]
    mp = PV.build_mapping(c, synth["tables"])
    assert sorted(mp["by_db"]) == ["alpha_db", "beta_db", "gamma"] and mp["lookups"]["alpha_db"] == [["alpha_db__pet", "alpha_db__owner"]]
    bad = copy.deepcopy(c)                                           # case-insensitive collision
    bad["collections"]["alpha_db__OWNER"] = dict(bad["collections"]["alpha_db__owner"], table="OWNER")
    with pytest.raises(PV.ProvisionError, match="case-insensitive"):
        PV.build_mapping(bad)
    bad = copy.deepcopy(c)                                           # ambiguous prefix: db "gamma" vs db "gamma__x"
    bad["db_ids"] = bad["db_ids"] + ["gamma__x"]
    bad["collections"]["gamma__x__log"] = dict(bad["collections"]["gamma__log"], db_id="gamma__x", table="log")
    with pytest.raises(PV.ProvisionError, match="ambiguous"):
        PV.build_mapping(bad)
    bad = copy.deepcopy(c)                                           # name not <db>__<table>
    bad["collections"]["renamed"] = bad["collections"].pop("beta_db__item")
    with pytest.raises(PV.ProvisionError):
        PV.build_mapping(bad)
    t = copy.deepcopy(synth["tables"])                               # FK to a table outside the mapping
    t["alpha_db"]["table_names_original"].append("ghost")
    t["alpha_db"]["column_names_original"].append([2, "gid"])
    t["alpha_db"]["foreign_keys"].append([7, 1])
    with pytest.raises(PV.ProvisionError, match="outside the mapping"):
        PV.build_mapping(c, t)
    with pytest.raises(PV.ProvisionError, match="Spider-dev"):
        PV.build_mapping(c, dev_names={"beta_db__item"})


def test_real_contract_maps_all_16_databases_offline(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymongo", None)               # any client creation would fail: proves no network
    contract = PV.load_contract()
    tables, dev = PV._context()
    rep = PV.dry_run(contract, tables=tables, dev_names=dev)
    assert (rep["databases"], rep["collections"], rep["documents"]) == (16, 69, 16885)
    assert rep["target_db"] == T and rep["sqlite_sources_verified"] == 69
    assert len(rep["collections_by_logical_db"]) == 16 and sum(len(v) for v in rep["collections_by_logical_db"].values()) == 69
    assert all(r["physical_db"] == T and r["collection"] == f"{r['logical_db']}__{r['table']}" for r in rep["mapping"])
    assert not set(r["collection"] for r in rep["mapping"]) & dev
    ctx = json.loads((PV.PKG_CONTEXT / "atlas_collections.json").read_text(encoding="utf-8"))
    assert {r["collection"] for r in rep["mapping"]} == set(ctx)        # same names the model saw in its prompts
    assert rep["lookup_pairs_checked"] == sum(len(tables[d]["foreign_keys"]) for d in contract["db_ids"])


def test_contract_pin(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(PV.CONTRACT.read_text(encoding="utf-8").replace('"documents": 1', '"documents": 2', 1), encoding="utf-8")
    with pytest.raises(PV.ProvisionError, match="changed"):
        PV.load_contract(f)


def test_dry_run_on_synthetic_sources_and_tampered_sqlite(synth, tmp_path):
    rep = PV.dry_run(synth["contract"], bundle=synth["bundle"], tables=synth["tables"], expected=None)
    assert rep["collections"] == 4 and rep["documents"] == 3 + 3 + 2 + 25
    bad = copy.deepcopy(synth["contract"])
    bad["collections"]["gamma__log"]["row_multiset_sha256"] = "0" * 64
    with pytest.raises(PV.ProvisionError, match="differs from the contract"):
        PV.dry_run(bad, bundle=synth["bundle"], expected=None)


# ------------------------------------------------------------------ provisioning behaviour (mocked)
def test_fresh_provisioning_writes_only_target_and_verifies(synth):
    srv = FakeMongo()
    rep = run(srv, synth)
    assert rep["status"] == "complete" and sorted(rep["written"]) == sorted(synth["contract"]["collections"])
    assert rep["verification"] == {"collections": 4, "documents": 33}
    assert set(srv.data) == {T} and srv.destructive_ops() == []
    assert {o[1] for o in srv.ops if o[0] in ("insert_many", "find", "list_collection_names", "db")} == {T}
    assert set(rep["operations"]) <= {"ping", "connectionStatus", "list_collection_names", "find", "insert_many"}
    assert rep["privileges"]["destructive_on_target"] == []


def test_idempotent_rerun_and_partial_resume(synth):
    srv = FakeMongo()
    run(srv, synth)
    n_inserts = sum(o[0] == "insert_many" for o in srv.ops)
    rep2 = run(srv, synth)
    assert rep2["written"] == [] and rep2["preflight_states"]["verified_existing"] == 4
    assert sum(o[0] == "insert_many" for o in srv.ops) == n_inserts               # nothing re-inserted
    srv2 = FakeMongo()                                                            # one collection already correct
    srv2.data[T] = {"beta_db__item": [{"code": "A1", "qty": 5}, {"code": "B2", "qty": 0}]}
    rep3 = run(srv2, synth)
    assert "beta_db__item" in rep3["skipped_verified_existing"] and "beta_db__item" not in rep3["written"]
    assert srv2.destructive_ops() == []


def test_conflicting_existing_collection_aborts_before_any_write(synth):
    for docs in ([{"code": "A1", "qty": 5}, {"code": "B2", "qty": 999}],          # wrong value
                 [{"code": "A1", "qty": 5}],                                       # partial
                 [{"code": "A1", "qty": 5}, {"code": "B2", "qty": 0}, {"code": "C3", "qty": 1}],   # extra doc
                 [{"code": "A1", "qty": 5, "x": 1}, {"code": "B2", "qty": 0}]):   # extra field
        srv = FakeMongo()
        srv.data[T] = {"beta_db__item": copy.deepcopy(docs)}
        with pytest.raises(PV.ProvisionError, match="conflict"):
            run(srv, synth)
        assert not any(o[0] == "insert_many" for o in srv.ops) and srv.destructive_ops() == []
        assert srv.data[T] == {"beta_db__item": docs}                              # untouched


def test_unexpected_collection_in_target_aborts(synth):
    srv = FakeMongo()
    srv.data[T] = {"wta_1__players": [{"a": 1}]}
    with pytest.raises(PV.ProvisionError, match="outside the contract"):
        run(srv, synth)
    assert not any(o[0] == "insert_many" for o in srv.ops)


def test_document_count_and_digest_verification(synth):
    srv = FakeMongo()
    srv.lose_one_on = "gamma__log"
    with pytest.raises(SD.SeedError, match="post-insert integrity"):
        run(srv, synth)
    srv = FakeMongo()
    srv.corrupt_on = "alpha_db__pet"
    with pytest.raises(SD.SeedError, match="digest match False"):
        run(srv, synth)
    assert srv.destructive_ops() == []
    with pytest.raises(PV.ProvisionError, match="conflict"):                       # the bad collection is never overwritten
        run(srv, synth)


def test_failure_mid_run_then_safe_resume(synth):
    srv = FakeMongo()
    srv.fail_insert_on = "beta_db__item"
    with pytest.raises(ConnectionError):
        run(srv, synth)
    done_before = {n for n, v in srv.data.get(T, {}).items() if v}
    assert done_before and "beta_db__item" not in done_before
    srv.fail_insert_on = None
    rep = run(srv, synth)                                                          # resumes: verifies done ones, inserts rest
    assert set(rep["skipped_verified_existing"]) == done_before and "beta_db__item" in rep["written"]
    assert rep["verification"]["documents"] == 33 and srv.destructive_ops() == []


@pytest.mark.parametrize("privs,msg", [
    ([{"resource": {"db": T, "collection": ""}, "actions": ["find", "insert", "listCollections", "dropCollection", "remove", "update"]}],
     "destructive"),                                                                # built-in readWrite-like role
    (minimal_role() + [{"resource": {"db": "codegen_spider1", "collection": ""}, "actions": ["insert"]}], "outside"),
    ([{"resource": {"db": "", "collection": ""}, "actions": ["find", "insert", "listCollections"]}], "outside"),   # any database
    ([{"resource": {"cluster": True}, "actions": ["dropDatabase"]}] + minimal_role(), "outside"),
    ([{"resource": {"db": T, "collection": ""}, "actions": ["find", "listCollections"]}], "lacks"),
])
def test_privileges_fail_closed(synth, privs, msg):
    srv = FakeMongo(privileges=privs)
    with pytest.raises(PV.ProvisionError, match=msg):
        run(srv, synth)
    assert not any(o[0] == "insert_many" for o in srv.ops)


def test_credentials_kept_separate(monkeypatch):
    for env in ("MONGODB_URI", "MONGODB_ATLAS_URI", "ATLAS_SEED_URI"):
        with pytest.raises(PV.ProvisionError):
            PV.make_client(env)
    monkeypatch.delenv("MQL0009_PROVISION_URI", raising=False)
    with pytest.raises(PV.ProvisionError, match="needs the provisioning URI"):
        PV.make_client("MQL0009_PROVISION_URI")
    monkeypatch.setenv("MQL0009_PROVISION_URI", "mongodb://placeholder.invalid/x")
    monkeypatch.setenv("MONGODB_URI", "mongodb://placeholder.invalid/x")
    with pytest.raises(PV.ProvisionError, match="equals MONGODB_URI"):
        PV.make_client("MQL0009_PROVISION_URI")


def test_existing_importer_and_seeder_unchanged():
    assert D.sha256_file(ROOT / "scripts/nl2mql/seed_atlas_spider_dev.py") == "01b0b07d77e9a4afcc3a72dc3523745451966fe31ff2333ef41aec75361c7fdb"
    assert D.sha256_file(ROOT / "scripts/spider1/import_atlas.py") == "58ad4f3829065e3559fb68a83d9bf925d61653b1da9055b0acaaa55628f06ab6"
    assert D.sha256_file(ROOT / "kaggle/nl2mql_mql0009_retrieval_control/package_manifest.json") == \
        "2a4f22e0eb526540e70f62f561d0314b1b2faa5c12917196a953d9f98aecface"


def test_readiness_accepts_a_complete_provisioning_report(tmp_path, monkeypatch):
    import mql0009_scoring as SC
    monkeypatch.setattr(SC, "DEST_ROOT", tmp_path)
    assert SC.readiness(write=False)["status"] == "BLOCKED"
    (tmp_path / "provisioning").mkdir()
    (tmp_path / "provisioning/provision_bad.json").write_text(json.dumps(
        {"status": "complete", "target_db": T, "verification": {"collections": 68, "documents": 16885}}), encoding="utf-8")
    assert SC.readiness(write=False)["status"] == "BLOCKED"
    (tmp_path / "provisioning/provision_ok.json").write_text(json.dumps(
        {"status": "complete", "target_db": T, "verification": {"collections": 69, "documents": 16885}}), encoding="utf-8")
    r = SC.readiness(write=False)
    assert r["status"] == "READY" and r["checks"]["atlas_provisioning_report"]["ok"]
