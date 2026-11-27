"""MQL-0015 Atlas provisioning: the 40 Spider test_full databases into ONE new database, `codegen_spider1_test_mql0015`
(insert-only, fail-closed). User decision D2 (2026-11-23).

Safety logic reused from the audited MQL-0009 wrapper (scripts/nl2mql_mql0009/provision_heldout_atlas.py, UNCHANGED):
GuardedClient write firewall (exactly one database; drop/delete/update/rename/index/$out refused), privilege check
(insert/find/listCollections on the target; no write/admin action anywhere else; no destructive action on the target),
absent / verified_existing / conflict preflight (conflict -> abort before any write), post-insert count + digest
verification, seed_atlas_spider_dev.seed.

Fix of the MQL-0013 mislabel (see preparation report): MQL-0009 functions read the module global TARGET_DB at CALL
time but `build_mapping(..., target_db=TARGET_DB)` froze its DEFAULT at DEFINITION time ("codegen_spider1_heldout"),
and the MQL-0009 `dry_run()` / `provision()` call build_mapping without target_db. Here every MQL-0009 function gets
target_db explicitly, its globals are bound only inside `_bound()` (restored afterwards, nothing leaks into other
importers), and every mapping row / report / operation is asserted to name codegen_spider1_test_mql0015.

Usage (repo root):
  python -B scripts/nl2mql_mql0015/provision_mql0015_atlas.py                       # DRY RUN: no network
  python -B scripts/nl2mql_mql0015/provision_mql0015_atlas.py --execute --target-db codegen_spider1_test_mql0015 \
      --confirm-target codegen_spider1_test_mql0015                               # needs $MQL0015_PROVISION_URI
"""
import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(1, str(ROOT / "scripts/nl2mql_mql0009"))
import mql0015_common as CM               # noqa: E402
import provision_heldout_atlas as PV9     # noqa: E402  (unchanged, audited MQL-0009 wrapper)

TARGET_DB = CM.ATLAS_DB
PROTECTED_DBS = set(CM.PROTECTED_DBS)
BUNDLE = Path.home() / "codegen_data/mql0015_test_bundle"
REPORT_DIR = CM.PROVISION_DIR
DEFAULT_URI_ENV = "MQL0015_PROVISION_URI"
REFUSED_URI_ENVS = {"MONGODB_URI", "MONGODB_ATLAS_URI", "ATLAS_SEED_URI", "MQL0009_PROVISION_URI", "MQL0010_PROVISION_URI",
                    "MQL0013_PROVISION_URI"}
ALLOWED_OPS = {"ping", "connectionStatus", "list_collection_names", "find", "count_documents", "insert_many"}
ProvisionError = PV9.ProvisionError
need = PV9.need

assert TARGET_DB not in PROTECTED_DBS and TARGET_DB == "codegen_spider1_test_mql0015"


@contextlib.contextmanager
def _bound():
    """Bind the MQL-0009 module globals to MQL-0015 for the duration of one call; always restored."""
    names = ("TARGET_DB", "PROTECTED_DBS", "DEFAULT_URI_ENV", "REFUSED_URI_ENVS")
    saved = {n: getattr(PV9, n) for n in names}
    PV9.TARGET_DB, PV9.PROTECTED_DBS = TARGET_DB, set(PROTECTED_DBS)
    PV9.DEFAULT_URI_ENV, PV9.REFUSED_URI_ENVS = DEFAULT_URI_ENV, set(REFUSED_URI_ENVS)
    try:
        yield
    finally:
        for n, v in saved.items():
            setattr(PV9, n, v)


def check_target(target_db, confirm=None, execute=False):
    need(target_db == TARGET_DB, f"target database must be {TARGET_DB!r}, got {target_db!r}")
    need(target_db not in PROTECTED_DBS, f"{target_db} is protected")
    if execute:
        need(confirm == TARGET_DB, f"--confirm-target must repeat {TARGET_DB!r} for --execute")
    with _bound():
        PV9.check_target(target_db, confirm, execute)
    return target_db


def package_config():
    return json.loads((CM.PKG_OUT / "configs/experiment.json").read_text(encoding="utf-8"))


def load_contract(contract_path=None, expected_sha=None):
    cfg = package_config() if expected_sha is None else None
    path = Path(contract_path or CM.CONTRACT)
    sha = expected_sha or cfg["scoring_contract"]["sha256"]
    with _bound():
        contract = PV9.load_contract(path, sha)
    need(contract["atlas_db"] == TARGET_DB, f"contract target {contract['atlas_db']} != {TARGET_DB}")
    src = json.loads(CM.MQL13_CONTRACT.read_text(encoding="utf-8"))
    diff = sorted(k for k in set(src) | set(contract) if src.get(k) != contract.get(k))
    need(diff == ["atlas_db", "status"], f"contract differs from the MQL-0013 content beyond atlas_db/status: {diff}")
    return contract


def build_mapping(contract, tables=None, dev_names=()):
    with _bound():
        mp = PV9.build_mapping(contract, tables, dev_names, target_db=TARGET_DB)
    bad = [r for r in mp["rows"] if r["physical_db"] != TARGET_DB]
    need(not bad, f"mapping rows with a physical database other than {TARGET_DB}: {bad[:3]}")
    need(sorted(mp["by_db"]) == sorted(contract["db_ids"]), "mapping does not cover every source database")
    return mp


def stage_bundle(contract, bundle=BUNDLE, src_dir=CM.TEST_DB_DIR):
    """Copy each test_database SQLite file into the seeder layout; verify every copy against the contract hash."""
    for db, h in contract["sqlite_sha256"].items():
        src = Path(src_dir) / db / f"{db}.sqlite"
        dst = Path(bundle) / "database" / db / f"{db}.sqlite"
        need(CM.sha256_file(src) == h, f"{db}: test_database SQLite differs from the contract")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        need(CM.sha256_file(dst) == h, f"{db}: staged copy differs from the contract")
    return Path(bundle)


def dry_run(contract, bundle, tables, dev_names, expected):
    check_target(TARGET_DB)
    mp = build_mapping(contract, tables, dev_names)
    with _bound():
        src = PV9.verify_sources(contract, bundle)
    docs = sum(r["documents"] for r in mp["rows"])
    need(len(contract["db_ids"]) == expected["databases"] and len(mp["rows"]) == expected["collections"]
         and docs == expected["documents"], f"contract size differs from the expected {expected}")
    ops = [{"collection": r["collection"], "target_namespace": f"{TARGET_DB}.{r['collection']}",
            "if_absent": f"insert_many(ordered=True) ({r['documents']} docs) then count + digest verification",
            "if_present_and_identical": "verify only (no write)", "if_present_and_different": "ABORT before any write"}
           for r in mp["rows"]]
    rep = {"mode": "dry run (no network, no database connection)", "experiment": CM.EXPERIMENT, "generated_at": PV9.C.now(),
           "target_db": TARGET_DB, "protected_dbs": sorted(PROTECTED_DBS),
           "contract": {"path": "outputs/nl2mql_final_1000/scoring_contract/atlas_mapping.json", "sha256": CM.sha256_file(CM.CONTRACT)},
           "databases": len(contract["db_ids"]), "collections": len(mp["rows"]), "documents": docs,
           "sqlite_sources_verified": len(src), "mapping": mp["rows"], "collections_by_source_db": mp["by_db"],
           "lookup_pairs_checked": sum(len(v) for v in mp["lookups"].values()), "planned_operations": ops,
           "operations_never_used": ["dropDatabase", "dropCollection", "drop", "delete_*", "update_*", "replace_*", "rename",
                                     "createIndex", "$out/$merge", "bulk_write"],
           "files_written": ["outputs/nl2mql_final_1000/provisioning/<report>.json", "~/codegen_data/mql0015_test_bundle (copies)"]}
    assert_report_targets(rep)
    return rep


def assert_report_targets(rep):
    """Every database name in a report must be the MQL-0015 target."""
    need(rep["target_db"] == TARGET_DB, "report target")
    need(all(r["physical_db"] == TARGET_DB for r in rep.get("mapping", [])), "report mapping rows")
    need(all(o["target_namespace"].startswith(TARGET_DB + ".") for o in rep.get("planned_operations", [])), "planned namespaces")
    blob = json.dumps(rep)
    for p in PROTECTED_DBS - {"admin", "local", "config"}:
        need(f'"{p}"' not in blob.replace(json.dumps(sorted(PROTECTED_DBS)), ""), f"report mentions protected database {p}")


def provision(raw_client, contract, bundle, log=print, oplog=None):
    """MQL-0009 provision() with the target passed explicitly to every step (real or mock client)."""
    check_target(TARGET_DB)
    build_mapping(contract)
    rep = {"mode": "execute", "experiment": CM.EXPERIMENT, "target_db": TARGET_DB, "started_at": PV9.C.now()}
    with _bound():
        probe = PV9.GuardedClient(raw_client, TARGET_DB, (), oplog)
        probe.ping()
        rep["server_version"] = probe.build_info().get("version")
        rep["privileges"] = PV9.check_privileges(probe.connection_status(), TARGET_DB)
        states, details, unexpected = PV9.classify(probe[TARGET_DB], contract)
        rep.update(preflight_states={s: sum(v == s for v in states.values()) for s in ("absent", "verified_existing", "conflict")},
                   conflicts={k: v for k, v in details.items() if states[k] == "conflict"}, unexpected=unexpected)
        need(not unexpected, f"target database holds collections outside the contract: {unexpected[:5]}; refusing")
        conflicts = [k for k, v in states.items() if v == "conflict"]
        need(not conflicts, f"{len(conflicts)} existing collections conflict with the contract (e.g. {conflicts[:3]}); nothing "
                            "was written. Inspect them manually (this tool never drops or overwrites).")
        absent = {k: contract["collections"][k] for k, v in states.items() if v == "absent"}
        guarded = PV9.GuardedClient(raw_client, TARGET_DB, absent, probe.oplog)
        rep["written"] = PV9.SD.seed(guarded, TARGET_DB, absent, bundle, log=log) if absent else []
        rep["skipped_verified_existing"] = sorted(k for k, v in states.items() if v == "verified_existing")
        rep["verification"] = PV9.verify_target(guarded[TARGET_DB], contract)
        rep["operations"] = sorted({o["op"] for o in probe.oplog})
    need(set(rep["operations"]) <= ALLOWED_OPS, f"unexpected operations used: {rep['operations']}")
    rep["finished_at"] = PV9.C.now()
    rep["status"] = "complete"
    return rep


def make_client(uri_env):
    need(uri_env == DEFAULT_URI_ENV, f"--uri-env must be {DEFAULT_URI_ENV} (MQL-0015-specific credentials)")
    need(uri_env not in REFUSED_URI_ENVS, f"{uri_env} is refused")
    uri = os.environ.get(uri_env, "").strip()
    need(uri, f"--execute needs the MQL-0015 provisioning URI in ${uri_env} (never the read-only scoring URI)")
    for other in REFUSED_URI_ENVS:
        v = os.environ.get(other, "").strip()
        need(not v or v != uri, f"the provisioning URI equals ${other}; use separate MQL-0015 credentials")
    sys.path.insert(0, str(ROOT / "scripts"))
    from mongo_conn import _redact, tls_kwargs
    from pymongo import MongoClient
    try:
        return MongoClient(uri, serverSelectionTimeoutMS=20000, appname="codegen-mql0015-provision", **tls_kwargs(uri))
    except Exception as e:  # noqa: BLE001
        raise ProvisionError(f"cannot create client: {_redact(str(e), uri)[:200]}") from None


def expected_sizes(contract):
    return {"databases": len(contract["db_ids"]), "collections": len(contract["collections"]),
            "documents": sum(m["documents"] for m in contract["collections"].values())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--target-db", default=TARGET_DB)
    ap.add_argument("--confirm-target")
    ap.add_argument("--uri-env", default=DEFAULT_URI_ENV)
    a = ap.parse_args(argv)
    try:
        check_target(a.target_db, a.confirm_target, a.execute)
        contract = load_contract()
        bundle = stage_bundle(contract)
        tables = {t["db_id"]: t for t in json.loads(CM.TEST_TABLES.read_text(encoding="utf-8"))}
        dev_names = set(json.loads(PV9.DEV_ATLAS_MANIFEST.read_text(encoding="utf-8")))
        report = dry_run(contract, bundle, tables, dev_names, expected_sizes(contract))
        if a.execute:
            client = make_client(a.uri_env)
            try:
                report = {**report, **provision(client, contract, bundle)}
            finally:
                client.close()
        out = REPORT_DIR / (("provision_" if a.execute else "dry_run_") + time.strftime("%Y%m%dT%H%M%S") + ".json")
        CM.write_json(out, report)
        print(json.dumps({k: report[k] for k in ("mode", "target_db", "databases", "collections", "documents",
                                                 "sqlite_sources_verified", "lookup_pairs_checked", "status") if k in report}, indent=1))
        print("report:", out)
        return 0
    except ProvisionError as e:
        print(f"PROVISION ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
