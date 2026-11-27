"""MQL-0010 holdout Atlas provisioning (insert-only, single database, fail-closed). PREPARED, NOT EXECUTED.

Loads the 18 MQL-0010 holdout databases (contract: outputs/nl2mql_mql0010/scoring_contract/atlas_mapping.json,
99 collections, 1,280 documents) into exactly ONE new Atlas database, `codegen_spider1_holdout_mql0010`.

All safety logic is the UNCHANGED, previously audited MQL-0009 wrapper
(scripts/nl2mql_mql0009/provision_heldout_atlas.py: GuardedClient write firewall, absent/verified/conflict
preflight classification, privilege check, post-insert count + digest verification, seed_atlas_spider_dev.seed). This
module only binds that wrapper to the MQL-0010 target, contract, URI variable and report folder, and every call passes
those values explicitly. The MQL-0009 file is imported, never edited.

Differences from MQL-0009 (all stricter or target-specific):
  * target database `codegen_spider1_holdout_mql0010`; `codegen_spider1` AND `codegen_spider1_heldout` (MQL-0009) are
    added to the protected names;
  * the contract hash is read from the frozen MQL-0010 package config (configs/experiment.json -> scoring_contract),
    which the package manifest pins;
  * credentials: $MQL0010_PROVISION_URI (MONGODB_URI, MONGODB_ATLAS_URI, ATLAS_SEED_URI and MQL0009_PROVISION_URI refused);
  * reports go to outputs/nl2mql_mql0010/provisioning/.

Usage (repo root):
  python -B scripts/nl2mql_mql0010/provision_holdout_atlas.py          # DRY RUN (default): no network, no database
  python -B scripts/nl2mql_mql0010/provision_holdout_atlas.py --execute --target-db codegen_spider1_holdout_mql0010 \
      --confirm-target codegen_spider1_holdout_mql0010                 # needs $MQL0010_PROVISION_URI; separate approval
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0009"))
import provision_heldout_atlas as PV9     # noqa: E402  (unchanged, audited MQL-0009 wrapper)

C = PV9.C
TARGET_DB = "codegen_spider1_holdout_mql0010"
PROTECTED_DBS = {"codegen_spider1", "codegen_spider1_heldout", "admin", "local", "config"}
CONTRACT = ROOT / "outputs/nl2mql_mql0010/scoring_contract/atlas_mapping.json"
PKG = ROOT / "kaggle/nl2mql_mql0010_holdout"
REPORT_DIR = ROOT / "outputs/nl2mql_mql0010/provisioning"
DEFAULT_URI_ENV = "MQL0010_PROVISION_URI"
REFUSED_URI_ENVS = {"MONGODB_URI", "MONGODB_ATLAS_URI", "ATLAS_SEED_URI", "MQL0009_PROVISION_URI"}


def _bind():
    """Point the reused wrapper's module-level settings at the MQL-0010 target (process-local; file unchanged)."""
    cfg = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    PV9.TARGET_DB = TARGET_DB
    PV9.PROTECTED_DBS = set(PROTECTED_DBS) - {TARGET_DB}
    PV9.CONTRACT = CONTRACT
    PV9.CONTRACT_SHA256 = cfg["scoring_contract"]["mapping_sha256"]
    PV9.PKG_CONTEXT = PKG / "data/context"
    PV9.REPORT_DIR = REPORT_DIR
    PV9.DEFAULT_URI_ENV = DEFAULT_URI_ENV
    PV9.REFUSED_URI_ENVS = set(REFUSED_URI_ENVS)
    PV9.EXPECTED = {"databases": len(cfg["evaluation"]["db_ids"]), "collections": cfg["scoring_contract"]["collections"],
                    "documents": cfg["scoring_contract"]["documents"]}
    return cfg


def load_contract():
    cfg = _bind()
    contract = PV9.load_contract(CONTRACT, cfg["scoring_contract"]["mapping_sha256"])
    PV9.need(contract["atlas_db"] == TARGET_DB, f"contract target {contract['atlas_db']} != {TARGET_DB}")
    PV9.need(sorted(contract["db_ids"]) == sorted(cfg["evaluation"]["db_ids"]), "contract databases != the 18 holdout DBs")
    return cfg, contract


def dry_run():
    cfg, contract = load_contract()
    tables, dev_names = PV9._context()
    rep = PV9.dry_run(contract, tables=tables, dev_names=dev_names, expected=PV9.EXPECTED)
    rep["experiment"] = "MQL-0010"
    rep["contract"] = {"path": str(CONTRACT.relative_to(ROOT)).replace("\\", "/"), "sha256": PV9.CONTRACT_SHA256}
    rep["files_written"] = ["outputs/nl2mql_mql0010/provisioning/<report>.json (only)"]
    return rep


def provision(raw_client, log=print, oplog=None):
    _, contract = load_contract()
    return PV9.provision(raw_client, contract, target_db=TARGET_DB, log=log, oplog=oplog)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--target-db", default=TARGET_DB)
    ap.add_argument("--confirm-target")
    ap.add_argument("--uri-env", default=DEFAULT_URI_ENV)
    ap.add_argument("--report", default=None)
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
        out = Path(a.report) if a.report else REPORT_DIR / (("provision_" if a.execute else "dry_run_") +
                                                            time.strftime("%Y%m%dT%H%M%S") + ".json")
        C.write_json(out, report)
        print(json.dumps({k: report[k] for k in ("mode", "target_db", "databases", "collections", "documents",
                                                 "sqlite_sources_verified", "lookup_pairs_checked") if k in report}, indent=1))
        print("report:", out)
        return 0
    except PV9.ProvisionError as e:
        print(f"PROVISION ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
