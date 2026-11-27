"""MQL-0015 Atlas capacity check (read-only; never writes, never prints credentials).

1. Offline: exact BSON size of the 245,574 documents the provisioner would insert, computed with the production
   seeder's own row->document mapping (seed_atlas_spider_dev.sqlite_docs) on the hash-verified staged bundle.
2. Online (optional, --online): dbStats of every database the read-only evaluation account (MONGODB_URI in .env)
   may see. That account cannot read the cluster tier, hostInfo, serverStatus or databases outside its grants,
   so the observed usage is a LOWER bound on cluster usage.

Writes outputs/nl2mql_final_1000/atlas/capacity_check_<ts>.json (new file each run; never overwrites).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/nl2mql"), str(ROOT / "scripts")]
import seed_atlas_spider_dev as SD  # noqa: E402  (unchanged production seeder)
import bson  # noqa: E402

CONTRACT = ROOT / "outputs/nl2mql_final_1000/scoring_contract/atlas_mapping.json"
CONTRACT_SHA = "6095bf442f29a2409241165cdc2a24ed5b9ace505a14c09462a2fee9ab89e45e"
BUNDLE = Path.home() / "codegen_data/mql0015_test_bundle"
OUT = ROOT / "outputs/nl2mql_final_1000/atlas"
VISIBLE_DBS = ["codegen_spider1", "codegen_spider1_heldout", "codegen_spider1_holdout_mql0010",
               "codegen_spider1_test_mql0013", "codegen_spider1_test_mql0015", "codegen_capstone_pilot"]
# Index model calibrated on the cluster's own measured databases (see report): _id index bytes per document and
# per-collection fixed overhead (index file + collection file) for tiny collections.
ID_INDEX_BYTES_PER_DOC = 40      # codegen_spider1 measures ~34 B/doc; rounded up
PER_COLLECTION_OVERHEAD = 72 * 1024  # holdout_mql0010 measures ~69 KB (storage+index) per near-empty collection


def offline_estimate():
    import hashlib
    need = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    if need != CONTRACT_SHA:
        raise SystemExit("contract hash differs from the MQL-0015 pin")
    con = json.loads(CONTRACT.read_text(encoding="utf-8"))
    per_coll, per_db = [], {}
    for name, m in sorted(con["collections"].items()):
        docs = SD.sqlite_docs(BUNDLE, m)
        if len(docs) != m["documents"]:
            raise SystemExit(f"{name}: {len(docs)} rows vs contract {m['documents']}")
        size = sum(len(bson.encode({"_id": bson.ObjectId(), **d})) for d in docs)
        per_coll.append({"collection": name, "documents": len(docs), "bson_bytes": size})
        per_db[m["db_id"]] = per_db.get(m["db_id"], 0) + size
    n_docs = sum(c["documents"] for c in per_coll)
    data = sum(c["bson_bytes"] for c in per_coll)
    idx = n_docs * ID_INDEX_BYTES_PER_DOC + len(per_coll) * PER_COLLECTION_OVERHEAD // 2
    overhead = len(per_coll) * PER_COLLECTION_OVERHEAD // 2
    return {"atlas_db": con["atlas_db"], "databases": len(con["db_ids"]), "collections": len(per_coll),
            "documents": n_docs, "bson_data_bytes": data, "estimated_index_bytes": idx,
            "estimated_collection_overhead_bytes": overhead,
            "estimated_total_bytes": data + idx + overhead,
            "aan_1_bson_bytes": per_db.get("aan_1", 0),
            "largest_collections": sorted(per_coll, key=lambda c: -c["bson_bytes"])[:6]}


def online_usage():
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"')
    from pymongo import MongoClient
    c = MongoClient(env["MONGODB_URI"], serverSelectionTimeoutMS=15000)
    try:
        roles = c.admin.command("connectionStatus")["authInfo"]["authenticatedUserRoles"]
        ld = c.admin.command("listDatabases")
        out = {"account": "read-only evaluation account (MONGODB_URI); credentials not recorded",
               "account_roles": roles, "server_version": c.server_info().get("version"),
               "listDatabases_visible": [{"name": d["name"], "sizeOnDisk": d["sizeOnDisk"]} for d in ld["databases"]],
               "listDatabases_totalSize": ld.get("totalSize"), "dbStats": {}, "not_visible": {}}
        for d in VISIBLE_DBS:
            try:
                s = c[d].command("dbStats")
                out["dbStats"][d] = {k: s.get(k) for k in ("collections", "objects", "dataSize", "storageSize",
                                                           "indexSize", "indexes")}
            except Exception as e:  # noqa: BLE001
                out["not_visible"][d] = type(e).__name__ + ": " + str(e).split(",")[0][:120]
        for cmd in ("hostInfo", "serverStatus", "getCmdLineOpts"):
            try:
                c.admin.command(cmd)
                out.setdefault("admin_commands_allowed", []).append(cmd)
            except Exception:  # noqa: BLE001
                out.setdefault("admin_commands_refused", []).append(cmd)
        return out
    finally:
        c.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true")
    a = ap.parse_args(argv)
    rep = {"experiment": "MQL-0015", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "model": {"id_index_bytes_per_doc": ID_INDEX_BYTES_PER_DOC,
                     "per_collection_overhead_bytes": PER_COLLECTION_OVERHEAD},
           "estimate": offline_estimate()}
    if a.online:
        rep["observed"] = online_usage()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"capacity_check_{time.strftime('%Y%m%dT%H%M%S')}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; refusing to overwrite")
    p.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep["estimate"].items() if k != "largest_collections"}, indent=1))
    if a.online:
        print(json.dumps({k: rep["observed"][k] for k in ("dbStats", "not_visible", "admin_commands_refused")}, indent=1))
    print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
