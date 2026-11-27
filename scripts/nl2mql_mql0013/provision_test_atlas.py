"""MQL-0013 Atlas provisioning: the 40 Spider test_full databases into ONE new database, `codegen_spider1_test_mql0013`
(insert-only, fail-closed). Authorised by the user on 2026-11-22.

All safety logic is the UNCHANGED, audited MQL-0009 wrapper (scripts/nl2mql_mql0009/provision_heldout_atlas.py:
GuardedClient write firewall, absent/verified/conflict preflight, privilege check, post-insert count + digest
verification, seed_atlas_spider_dev.seed). This module only binds it to the MQL-0013 target, contract, URI variable,
report folder, and a SQLite bundle in the seeder's layout (<bundle>/database/<db>/<db>.sqlite) built by COPYING the
40 test_database files (hash-checked against the contract) to ~/codegen_data/mql0013_test_bundle. No index is created
(same as every earlier evaluation database).

Usage (repo root):
  python -B scripts/nl2mql_mql0013/provision_test_atlas.py                      # DRY RUN: no network
  python -B scripts/nl2mql_mql0013/provision_test_atlas.py --execute --target-db codegen_spider1_test_mql0013 \
      --confirm-target codegen_spider1_test_mql0013                             # needs $MQL0013_PROVISION_URI
The provisioning user must hold ONLY find/insert/listCollections/createCollection on codegen_spider1_test_mql0013 (no
write action on any other database, no destructive action anywhere), else the wrapper refuses before writing.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(1, str(ROOT / "scripts/nl2mql_mql0009"))
import mql0013_common as CM               # noqa: E402
import provision_heldout_atlas as PV9     # noqa: E402  (unchanged, audited MQL-0009 wrapper)

TARGET_DB = CM.ATLAS_DB
PROTECTED_DBS = {"codegen_spider1", "codegen_spider1_heldout", "codegen_spider1_holdout_mql0010", "admin", "local", "config"}
BUNDLE = Path.home() / "codegen_data/mql0013_test_bundle"
REPORT_DIR = CM.OUT / "provisioning"
DEFAULT_URI_ENV = "MQL0013_PROVISION_URI"
REFUSED_URI_ENVS = {"MONGODB_URI", "MONGODB_ATLAS_URI", "ATLAS_SEED_URI", "MQL0009_PROVISION_URI", "MQL0010_PROVISION_URI"}


def _bind():
    cfg = json.loads((CM.PKG_OUT / "configs/experiment.json").read_text(encoding="utf-8"))
    PV9.TARGET_DB = TARGET_DB
    PV9.PROTECTED_DBS = set(PROTECTED_DBS) - {TARGET_DB}
    PV9.CONTRACT = CM.CONTRACT
    PV9.CONTRACT_SHA256 = cfg["scoring_contract"]["mapping_sha256"]
    PV9.PKG_CONTEXT = CM.PKG_OUT / "data/context"
    PV9.REPORT_DIR = REPORT_DIR
    PV9.DEFAULT_URI_ENV = DEFAULT_URI_ENV
    PV9.REFUSED_URI_ENVS = set(REFUSED_URI_ENVS)
    PV9.EXPECTED = {"databases": len(cfg["evaluation"]["db_ids"]), "collections": cfg["scoring_contract"]["collections"],
                    "documents": cfg["scoring_contract"]["documents"]}
    return cfg


def load_contract():
    cfg = _bind()
    contract = PV9.load_contract(CM.CONTRACT, cfg["scoring_contract"]["mapping_sha256"])
    PV9.need(contract["atlas_db"] == TARGET_DB, f"contract target {contract['atlas_db']} != {TARGET_DB}")
    PV9.need(sorted(contract["db_ids"]) == sorted(cfg["evaluation"]["db_ids"]), "contract databases != the 40 test DBs")
    return cfg, contract


def stage_bundle(contract):
    """Copy each test_database SQLite file into the seeder layout; verify every copy against the contract hash."""
    for db, h in contract["sqlite_sha256"].items():
        src = CM.TEST_DB_DIR / db / f"{db}.sqlite"
        dst = BUNDLE / "database" / db / f"{db}.sqlite"
        PV9.need(CM.sha256_file(src) == h, f"{db}: test_database SQLite differs from the contract")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        PV9.need(CM.sha256_file(dst) == h, f"{db}: staged copy differs from the contract")
    return BUNDLE


def tables_and_dev_names():
    tabs = {t["db_id"]: t for t in json.loads(CM.TEST_TABLES.read_text(encoding="utf-8"))}
    dev_names = set(json.loads(PV9.DEV_ATLAS_MANIFEST.read_text(encoding="utf-8")))
    return tabs, dev_names


def dry_run():
    cfg, contract = load_contract()
    bundle = stage_bundle(contract)
    tables, dev_names = tables_and_dev_names()
    rep = PV9.dry_run(contract, bundle=bundle, tables=tables, dev_names=dev_names, expected=PV9.EXPECTED)
    rep["experiment"] = CM.EXPERIMENT
    rep["contract"] = {"path": "outputs/nl2mql_mql0013/scoring_contract/atlas_mapping.json", "sha256": PV9.CONTRACT_SHA256}
    rep["files_written"] = ["outputs/nl2mql_mql0013/provisioning/<report>.json", "~/codegen_data/mql0013_test_bundle (copies)"]
    return rep


def provision(raw_client, log=print, oplog=None):
    _, contract = load_contract()
    bundle = stage_bundle(contract)
    return PV9.provision(raw_client, contract, bundle=bundle, target_db=TARGET_DB, log=log, oplog=oplog)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--target-db", default=TARGET_DB)
    ap.add_argument("--confirm-target")
    ap.add_argument("--uri-env", default=DEFAULT_URI_ENV)
    a = ap.parse_args(argv)
    try:
        _bind()
        PV9.check_target(a.target_db, a.confirm_target, a.execute)
        report = dry_run()
        if a.execute:
            client = PV9.make_client(a.uri_env)
            try:
                report = {**report, **provision(client)}
            finally:
                client.close()
        out = REPORT_DIR / (("provision_" if a.execute else "dry_run_") + time.strftime("%Y%m%dT%H%M%S") + ".json")
        CM.write_json(out, report)
        print(json.dumps({k: report[k] for k in ("mode", "target_db", "databases", "collections", "documents",
                                                 "sqlite_sources_verified", "lookup_pairs_checked", "status") if k in report}, indent=1))
        print("report:", out)
        return 0
    except PV9.ProvisionError as e:
        print(f"PROVISION ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
