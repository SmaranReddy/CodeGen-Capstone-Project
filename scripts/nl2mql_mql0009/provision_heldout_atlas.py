"""MQL-0009 held-out Atlas provisioning wrapper: insert-only, single-database, fail-closed, idempotent where safe.

Writes the 16 held-out Spider-train databases (69 collections, 16,885 documents) of the MQL-0009 provisioning contract
into exactly ONE Atlas database, `codegen_spider1_heldout`, and nowhere else.

Starting point and re-use (both UNCHANGED, imported, not copied):
  scripts/nl2mql/seed_atlas_spider_dev.py   sqlite_docs()  - rows in rowid order, import_atlas.py value mapping
                                            seed()         - clash check, insert_many(ordered=True), per-collection
                                                             post-insert count + type-tagged digest verification
  scripts/nl2mql/dev1034_core.py            atlas_collection_digest(), sqlite_table_digest(), privileges_report()
`seed()` is only ever handed (a) a guarded client that exposes nothing but the target database and (b) the subset of
collections that are ABSENT on the target, so its clash check can never fire on data it should not touch.

Safety model
  * Target database is a constant. `--target-db` must equal it AND `--confirm-target` must repeat it; protected names
    (codegen_spider1, admin, local, config) are refused regardless.
  * GuardedClient / GuardedDB / GuardedCollection expose only: ping/buildInfo/connectionStatus, list_collection_names,
    find, count_documents, insert_many. Any other attribute (drop_database, drop_collection, drop, delete_*, update_*,
    replace_*, rename, create_index, bulk_write, aggregate with $out/$merge, command, ...) raises DestructiveOperationRefused.
    insert_many is allowed only on a collection that was ABSENT at plan time and is in the contract, once.
  * Preflight classifies every contract collection BEFORE any write:
        absent             -> will be inserted
        verified_existing  -> count, digest and field set already equal the contract: skipped (idempotent re-run)
        conflict           -> anything else (partial, extra docs, wrong digest, extra fields): ABORT, nothing written
    Any collection in the target database that is NOT in the contract also aborts.
  * Credentials: the URI comes from a dedicated variable (default MQL0009_PROVISION_URI). The read-only scoring variables
    (MONGODB_URI, MONGODB_ATLAS_URI) and the dev seeder's ATLAS_SEED_URI are refused by name, and the provisioning URI
    must differ from MONGODB_URI. The URI is never printed. After connecting, the user's privileges must be insert-capable
    on the target and hold NO write/admin action on any other resource and NO destructive action on the target
    (custom minimal role; see docs). Otherwise abort before writing.
  * After writing: the target database must hold exactly the 69 contract collections, each with the contract count,
    digest and no extra field. The report is written under outputs/nl2mql_mql0009_retrieval_control/provisioning/.
    data/interim/spider1/atlas_collections.json (the repo inventory) is never touched.

Usage (repo root):
  python -B scripts/nl2mql_mql0009/provision_heldout_atlas.py                      # DRY RUN (default): no network
  python -B scripts/nl2mql_mql0009/provision_heldout_atlas.py --execute --target-db codegen_spider1_heldout \
      --confirm-target codegen_spider1_heldout                                     # needs $MQL0009_PROVISION_URI
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
sys.path.insert(1, str(ROOT / "scripts"))
import dev1034_core as C                  # noqa: E402  (unchanged)
import seed_atlas_spider_dev as SD        # noqa: E402  (unchanged production seeder: sqlite_docs, seed)

TARGET_DB = "codegen_spider1_heldout"
PROTECTED_DBS = {"codegen_spider1", "admin", "local", "config"}
CONTRACT = ROOT / "outputs/nl2mql_mql0009_retrieval_control/scoring_contract/atlas_mapping.json"
CONTRACT_SHA256 = "f85008040a37f1f7dcf0a5346c2359736e903f2b486d30cac220e3fe68ba4134"   # as built with package 2a4f22e0...
PKG_CONTEXT = ROOT / "kaggle/nl2mql_mql0009_retrieval_control/data/context"
SPIDER_DATA = ROOT / "data/raw/spider1/spider_data"         # SD.sqlite_docs layout: <bundle>/database/<db>/<db>.sqlite
DEV_ATLAS_MANIFEST = ROOT / "data/interim/spider1/atlas_collections.json"
REPORT_DIR = ROOT / "outputs/nl2mql_mql0009_retrieval_control/provisioning"
DEFAULT_URI_ENV = "MQL0009_PROVISION_URI"
REFUSED_URI_ENVS = {"MONGODB_URI", "MONGODB_ATLAS_URI", "ATLAS_SEED_URI"}
EXPECTED = {"databases": 16, "collections": 69, "documents": 16885}
ALLOWED_TARGET_ACTIONS = {"find", "insert", "listCollections", "createCollection", "killCursors", "listIndexes",
                          "collStats", "dbStats"}
DESTRUCTIVE_TARGET_ACTIONS = {"remove", "update", "dropCollection", "dropDatabase", "renameCollectionSameDB", "collMod",
                              "convertToCapped", "compact", "reIndex", "dropIndex", "createIndex", "applyOps",
                              "bypassDocumentValidation", "enableSharding", "moveChunk", "splitChunk"}


class ProvisionError(RuntimeError):
    pass


class DestructiveOperationRefused(ProvisionError):
    pass


def need(cond, msg, exc=ProvisionError):
    if not cond:
        raise exc(msg)


# ------------------------------------------------------------------ target + contract + mapping
def check_target(target_db, confirm=None, execute=False):
    need(target_db == TARGET_DB, f"target database must be {TARGET_DB!r}, got {target_db!r}")
    need(target_db not in PROTECTED_DBS, f"{target_db} is protected")
    if execute:
        need(confirm == TARGET_DB, f"--confirm-target must repeat {TARGET_DB!r} for --execute")
    return target_db


def load_contract(path=CONTRACT, expected_sha=CONTRACT_SHA256):
    need(C.sha256_file(path) == expected_sha, f"provisioning contract {path} changed (sha256 != pinned {expected_sha})")
    return json.loads(Path(path).read_text(encoding="utf-8"))


_BAD_NAME = re.compile(r"[\x00$]")


def build_mapping(contract, tables=None, dev_names=(), target_db=TARGET_DB):
    """Explicit logical (db_id, table) -> physical collection mapping, with collision and $lookup checks.
    -> {"rows": [...], "by_db": {...}, "lookups": {...}}."""
    db_ids = contract["db_ids"]
    rows, seen, seen_ci = [], {}, {}
    for name, m in sorted(contract["collections"].items()):
        phys = f"{m['db_id']}__{m['table']}"
        need(phys == name, f"{name}: physical name != <db_id>__<table> ({phys})")
        need(m["db_id"] in db_ids, f"{name}: db_id {m['db_id']} not in the contract's databases")
        need(name not in seen, f"collection-name collision: {name} from {seen.get(name)} and {(m['db_id'], m['table'])}")
        need(name.lower() not in seen_ci, f"case-insensitive collision: {name} vs {seen_ci.get(name.lower())}")
        need(not _BAD_NAME.search(name) and not name.startswith("system.") and name,
             f"{name}: invalid MongoDB collection name")
        need(len((target_db + "." + name).encode("utf-8")) <= 235, f"{name}: namespace too long")
        owners = [d for d in db_ids if name.startswith(d + "__")]
        need(owners == [m["db_id"]], f"{name}: ambiguous database prefix {owners}")
        need(name not in dev_names, f"{name}: also an existing Spider-dev collection name")
        seen[name], seen_ci[name.lower()] = (m["db_id"], m["table"]), name
        rows.append({"logical_db": m["db_id"], "table": m["table"], "physical_db": target_db, "collection": name,
                     "documents": m["documents"], "fields": len(m["fields"]), "row_multiset_sha256": m["row_multiset_sha256"]})
    by_db = {}
    for r in rows:
        by_db.setdefault(r["logical_db"], []).append(r["collection"])
    need(sorted(by_db) == sorted(db_ids), f"databases without collections: {sorted(set(db_ids) - set(by_db))}")
    lookups = {}
    if tables is not None:
        for db in db_ids:
            t = tables[db]
            tabs, cols = t["table_names_original"], t["column_names_original"]
            pairs = []
            for x, y in t["foreign_keys"]:
                a, b = f"{db}__{tabs[cols[x][0]]}", f"{db}__{tabs[cols[y][0]]}"
                need(a in seen and b in seen, f"{db}: foreign key {a} -> {b} references a collection outside the mapping")
                need(seen[a][0] == seen[b][0] == db, f"{db}: foreign key crosses logical databases")
                pairs.append([a, b])
            lookups[db] = pairs
    return {"rows": rows, "by_db": by_db, "lookups": lookups}


def verify_sources(contract, bundle=SPIDER_DATA):
    """Offline: SQLite files equal the contract hashes, and every table re-reads to the contract count and digest."""
    out = {}
    for db, h in contract["sqlite_sha256"].items():
        p = Path(bundle) / "database" / db / f"{db}.sqlite"
        need(p.is_file() and C.sha256_file(p) == h, f"{db}: SQLite file missing or changed")
    for name, m in sorted(contract["collections"].items()):
        p = Path(bundle) / "database" / m["db_id"] / f"{m['db_id']}.sqlite"
        n, dig, _ = C.sqlite_table_digest(p, m["table"], m["fields"])
        need(n == m["documents"] and dig == m["row_multiset_sha256"], f"{name}: SQLite table differs from the contract")
        docs = SD.sqlite_docs(bundle, m)
        need(len(docs) == m["documents"], f"{name}: sqlite_docs returned {len(docs)} rows")
        out[name] = n
    return out


# ------------------------------------------------------------------ guarded client (structural write firewall)
class GuardedCollection:
    def __init__(self, gdb, name):
        self._gdb, self._name = gdb, name
        self._coll = gdb._db[name]

    def find(self, *a, **kw):
        self._gdb._log("find", self._name)
        return self._coll.find(*a, **kw)

    def count_documents(self, *a, **kw):
        self._gdb._log("count_documents", self._name)
        return self._coll.count_documents(*a, **kw)

    def insert_many(self, docs, ordered=True, **kw):
        need(not kw, f"insert_many options {sorted(kw)} not allowed", DestructiveOperationRefused)
        need(self._name in self._gdb._insertable, f"insert into {self._name} refused: not an absent contract collection",
             DestructiveOperationRefused)
        self._gdb._log("insert_many", self._name, len(docs))
        return self._coll.insert_many(docs, ordered=True)

    def __getattr__(self, attr):
        raise DestructiveOperationRefused(f"collection operation {attr!r} on {self._name} is not allowed by the guard")


class GuardedDB:
    def __init__(self, client, db_name, insertable, oplog):
        self._db, self._name, self._insertable, self._oplog = client[db_name], db_name, set(insertable), oplog
        self._done = set()

    def _log(self, op, coll=None, n=None):
        self._oplog.append({"op": op, "db": self._name, "collection": coll, "n": n})

    def list_collection_names(self):
        self._log("list_collection_names")
        return self._db.list_collection_names()

    def __getitem__(self, name):
        need(isinstance(name, str) and name, "bad collection name", DestructiveOperationRefused)
        return GuardedCollection(self, name)

    def __getattr__(self, attr):
        raise DestructiveOperationRefused(f"database operation {attr!r} on {self._name} is not allowed by the guard")


class GuardedClient:
    """Exposes exactly one database. Everything else, including other databases and drop_database, is refused."""

    def __init__(self, client, target_db, insertable=(), oplog=None):
        check_target(target_db)
        self._client, self._target, self.oplog = client, target_db, oplog if oplog is not None else []
        self._gdb = GuardedDB(client, target_db, insertable, self.oplog)

    def __getitem__(self, name):
        need(name == self._target, f"access to database {name!r} refused (only {self._target!r})", DestructiveOperationRefused)
        return self._gdb

    def ping(self):
        self.oplog.append({"op": "ping"})
        return self._client.admin.command("ping")

    def build_info(self):
        return self._client.admin.command("buildInfo")

    def connection_status(self):
        self.oplog.append({"op": "connectionStatus"})
        return self._client[self._target].command({"connectionStatus": 1, "showPrivileges": True})

    def close(self):
        self._client.close()

    def __getattr__(self, attr):
        raise DestructiveOperationRefused(f"client operation {attr!r} is not allowed by the guard")


# ------------------------------------------------------------------ privileges, classification, verification
def check_privileges(conn_status, target_db=TARGET_DB):
    """Provisioning user: insert + find + listCollections on the target; no write/admin action anywhere else; no
    destructive action even on the target (so a bug cannot drop or modify data). -> report; raises on violation."""
    users, per, _bad = C.privileges_report(conn_status)
    need(users, "not authenticated")
    tgt = set(per.get(f"{target_db}.", [])) | set(per.get(".", []))
    need({"insert", "find", "listCollections"} <= tgt, f"provisioning user lacks insert/find/listCollections on {target_db}: {sorted(tgt)}")
    outside = {}
    for res, acts in per.items():
        if res in (f"{target_db}.",) or res.startswith(f"{target_db}."):
            continue
        w = sorted(a for a in acts if a in C.WRITE_OR_ADMIN_ACTIONS or a.startswith(C._WRITE_PREFIXES) or a in DESTRUCTIVE_TARGET_ACTIONS)
        w = [a for a in w if a not in C.READ_ACTIONS_OK]
        if w:
            outside[res or "<cluster/any>"] = w
    need(not outside, f"provisioning user holds write/admin actions outside {target_db}: {outside}. Use a role scoped "
                      f"to {target_db} only.")
    destructive = sorted(a for res, acts in per.items() if res.startswith(f"{target_db}.") for a in acts if a in DESTRUCTIVE_TARGET_ACTIONS)
    need(not destructive, f"provisioning user holds destructive actions on {target_db}: {destructive}. Use the minimal custom "
                          f"role (find, insert, listCollections, createCollection) instead of readWrite.")
    return {"users": users, "target_actions": sorted(tgt), "outside_write_actions": {}, "destructive_on_target": []}


def classify(gdb, contract):
    """-> {collection: state} for every contract collection, plus unexpected collections. No writes."""
    names = set(gdb.list_collection_names())
    expected = set(contract["collections"])
    states, details = {}, {}
    for name in sorted(expected):
        m = contract["collections"][name]
        if name not in names:
            states[name] = "absent"
            continue
        n, dig, extra = C.atlas_collection_digest(gdb, name, m["fields"])
        ok = n == m["documents"] and dig == m["row_multiset_sha256"] and not extra
        states[name] = "verified_existing" if ok else "conflict"
        details[name] = {"documents": n, "expected_documents": m["documents"], "digest_match": dig == m["row_multiset_sha256"],
                         "extra_fields": extra}
    unexpected = sorted(names - expected)
    return states, details, unexpected


def verify_target(gdb, contract):
    names = set(gdb.list_collection_names())
    expected = set(contract["collections"])
    need(names == expected, f"target holds {len(names)} collections; missing {sorted(expected - names)[:5]}, "
                            f"unexpected {sorted(names - expected)[:5]}")
    res = {}
    for name in sorted(expected):
        m = contract["collections"][name]
        n, dig, extra = C.atlas_collection_digest(gdb, name, m["fields"])
        need(n == m["documents"] and dig == m["row_multiset_sha256"] and not extra,
             f"{name}: post-provisioning verification failed (docs {n}/{m['documents']}, digest match "
             f"{dig == m['row_multiset_sha256']}, extra {extra})")
        res[name] = n
    need(sum(res.values()) == sum(m["documents"] for m in contract["collections"].values()), "total document count")
    return {"collections": len(res), "documents": sum(res.values())}


# ------------------------------------------------------------------ dry run + provision
def dry_run(contract, bundle=SPIDER_DATA, tables=None, dev_names=(), expected=EXPECTED, batch=20000):
    check_target(TARGET_DB)
    mp = build_mapping(contract, tables, dev_names)
    src = verify_sources(contract, bundle)
    docs = sum(r["documents"] for r in mp["rows"])
    if expected:
        need(len(contract["db_ids"]) == expected["databases"] and len(mp["rows"]) == expected["collections"]
             and docs == expected["documents"], f"contract size differs from the expected {expected}")
    ops = []
    for r in mp["rows"]:
        nb = max(1, -(-r["documents"] // batch))
        ops.append({"collection": r["collection"], "if_absent": f"insert_many(ordered=True) x{nb} ({r['documents']} docs), then "
                    "count + type-tagged digest verification (seed_atlas_spider_dev.seed)",
                    "if_present_and_identical": "verify only (no write)", "if_present_and_different": "ABORT before any write"})
    return {"mode": "dry run (no network, no database connection)", "generated_at": C.now(), "target_db": TARGET_DB,
            "contract": {"path": str(CONTRACT.relative_to(ROOT)) if Path(CONTRACT).is_relative_to(ROOT) else str(CONTRACT),
                         "sha256": CONTRACT_SHA256},
            "databases": len(contract["db_ids"]), "collections": len(mp["rows"]), "documents": docs,
            "sqlite_sources_verified": len(src), "mapping": mp["rows"], "collections_by_logical_db": mp["by_db"],
            "lookup_pairs_checked": sum(len(v) for v in mp["lookups"].values()), "lookups": mp["lookups"],
            "planned_operations": ops,
            "operations_never_used": ["dropDatabase", "dropCollection", "drop", "delete_*", "update_*", "replace_*",
                                      "rename", "createIndex", "$out/$merge", "bulk_write"],
            "files_written": ["outputs/nl2mql_mql0009_retrieval_control/provisioning/<report>.json (only)"]}


def provision(raw_client, contract, bundle=SPIDER_DATA, target_db=TARGET_DB, log=print, oplog=None):
    """Execute against an already-created client (real or mock). Fail-closed at every step; returns the report."""
    check_target(target_db)
    build_mapping(contract)
    rep = {"mode": "execute", "target_db": target_db, "started_at": C.now()}
    probe = GuardedClient(raw_client, target_db, (), oplog)
    probe.ping()
    rep["server_version"] = probe.build_info().get("version")
    rep["privileges"] = check_privileges(probe.connection_status(), target_db)
    states, details, unexpected = classify(probe[target_db], contract)
    rep.update(preflight_states={s: sum(v == s for v in states.values()) for s in ("absent", "verified_existing", "conflict")},
               conflicts=details and {k: v for k, v in details.items() if states[k] == "conflict"}, unexpected=unexpected)
    need(not unexpected, f"target database holds collections outside the contract: {unexpected[:5]}; refusing")
    conflicts = [k for k, v in states.items() if v == "conflict"]
    need(not conflicts, f"{len(conflicts)} existing collections conflict with the contract (e.g. {conflicts[:3]}); nothing was "
                        "written. Inspect and remove them manually (this tool never drops or overwrites).")
    absent = {k: contract["collections"][k] for k, v in states.items() if v == "absent"}
    guarded = GuardedClient(raw_client, target_db, absent, probe.oplog)
    written = SD.seed(guarded, target_db, absent, bundle, log=log) if absent else []
    rep["written"] = written
    rep["skipped_verified_existing"] = sorted(k for k, v in states.items() if v == "verified_existing")
    rep["verification"] = verify_target(guarded[target_db], contract)
    rep["operations"] = sorted({o["op"] for o in probe.oplog})
    need(set(rep["operations"]) <= {"ping", "connectionStatus", "list_collection_names", "find", "count_documents", "insert_many"},
         f"unexpected operations used: {rep['operations']}")
    rep["finished_at"] = C.now()
    rep["status"] = "complete"
    return rep


def make_client(uri_env):
    need(uri_env not in REFUSED_URI_ENVS, f"{uri_env} is a read-only/evaluation or dev-seed variable; use {DEFAULT_URI_ENV}")
    uri = os.environ.get(uri_env, "").strip()
    need(uri, f"--execute needs the provisioning URI in ${uri_env} (never the read-only scoring URI)")
    ro = os.environ.get("MONGODB_URI", "").strip()
    need(not ro or ro != uri, "the provisioning URI equals MONGODB_URI (the scoring account); use separate credentials")
    from mongo_conn import _redact, tls_kwargs
    from pymongo import MongoClient
    try:
        return MongoClient(uri, serverSelectionTimeoutMS=20000, appname="codegen-mql0009-provision", **tls_kwargs(uri))
    except Exception as e:  # noqa: BLE001
        raise ProvisionError(f"cannot create client: {_redact(str(e), uri)[:200]}") from None


def _context():
    tables = {t["db_id"]: t for t in json.loads((PKG_CONTEXT / "tables_subset.json").read_text(encoding="utf-8"))}
    dev_names = set(json.loads(DEV_ATLAS_MANIFEST.read_text(encoding="utf-8")))
    return tables, dev_names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--target-db", default=TARGET_DB)
    ap.add_argument("--confirm-target")
    ap.add_argument("--uri-env", default=DEFAULT_URI_ENV)
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    try:
        check_target(a.target_db, a.confirm_target, a.execute)
        contract = load_contract()
        tables, dev_names = _context()
        report = dry_run(contract, tables=tables, dev_names=dev_names)
        if a.execute:
            client = make_client(a.uri_env)
            try:
                report = {**report, **provision(client, contract)}
            finally:
                client.close()
        out = Path(a.report) if a.report else REPORT_DIR / (("provision_" if a.execute else "dry_run_") +
                                                            time.strftime("%Y%m%dT%H%M%S") + ".json")
        C.write_json(out, report)
        print(json.dumps({k: report[k] for k in ("mode", "target_db", "databases", "collections", "documents",
                                                 "sqlite_sources_verified", "lookup_pairs_checked") if k in report}, indent=1))
        print("report:", out)
        return 0
    except ProvisionError as e:
        print(f"PROVISION ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
