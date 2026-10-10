"""Audit the frozen 68-example NL -> MQL dataset before any evaluation.

Verifies each record has a question, schema context, reference MQL, validation evidence
and an executable gold; distinguishes *dataset completeness gaps* (reconstructable from
the conversion artifacts) from real *execution blockers*. Records the snapshot hash and
writes outputs/evaluation_68/audit.json.

Read-only. Never modifies the dataset or any conversion artifact.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_kit import (DATASET_PATH, ROOT, allowed_collections, imports_for, load_artifact,
                      load_dataset, start_collection, validate_mql)

OUT_DIR = ROOT / "outputs/evaluation_68"

REQUIRED_ROW_FIELDS = ["instance_id", "db", "question", "schema", "query",
                       "expected_result_references", "validation", "validation_passed"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit():
    rows = load_dataset()
    ids = [r["instance_id"] for r in rows]
    from conversion_kit import manifest, mongo_client
    man = manifest()
    client = mongo_client()
    try:
        existing = set(client["codegen_capstone_pilot"].list_collection_names())
    finally:
        client.close()

    out = {
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": sha256(DATASET_PATH),
        "record_count": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": [k for k, v in Counter(ids).items() if v > 1],
        "records": [],
    }
    evaluable = 0
    for r in rows:
        iid = r["instance_id"]
        e = {"instance_id": iid, "db": r.get("db"), "dataset_gaps": [], "blockers": []}
        for f in REQUIRED_ROW_FIELDS:
            if r.get(f) in (None, "", [], {}):
                e["dataset_gaps"].append(f"dataset field '{f}' empty/absent")
        rec = man.get(iid)
        if not rec:
            e["blockers"].append("no manifest record")
        else:
            e["sqlite"] = rec["database_path"]
            e["sqlite_exists"] = (ROOT / rec["database_path"]).exists()
            if not e["sqlite_exists"]:
                e["blockers"].append("source sqlite missing")
            e["gold_files"] = [Path(p).name for p in rec["gold_result_paths"]]
            e["gold_exist"] = all((ROOT / p).exists() for p in rec["gold_result_paths"])
            if not e["gold_exist"]:
                e["blockers"].append("gold csv missing")
            cfg = rec.get("gold_eval_config") or {}
            e["ignore_order"] = cfg.get("ignore_order")
            e["condition_cols"] = cfg.get("condition_cols")
            if cfg.get("ignore_order") is None:
                e["blockers"].append("gold_eval_config missing ignore_order")
        art = load_artifact(iid)
        if not art:
            e["blockers"].append("no conversion/pilot artifact")
        else:
            e["artifact"] = "conversions" if (ROOT / "data/interim/conversions" / f"{iid}.json").exists() else "pilot"
            e["start_collection"] = start_collection(art)
            imps = imports_for(iid, art)
            owned = allowed_collections(iid, art)
            e["imports"] = [i["collection"] for i in imps]
            e["owned_collections"] = sorted(owned)
            e["collections_existing_in_atlas"] = sorted(owned & existing)
            if not imps and not (owned & existing):
                e["blockers"].append("no resolvable imports and no owned collection in Atlas")
            if not e["start_collection"]:
                e["blockers"].append("no start collection")
            mql = {"method": (art.get("mql") or {}).get("method", "aggregate"),
                   "collection": e["start_collection"], "pipeline": (art.get("mql") or {}).get("pipeline")}
            ok, reason, _ = validate_mql(mql, owned)
            e["reference_structurally_valid"] = ok
            if not ok:
                e["blockers"].append(f"reference MQL rejected: {reason}")
        e["evaluable"] = not e["blockers"]
        evaluable += int(e["evaluable"])
        out["records"].append(e)
    out["evaluable_count"] = evaluable
    out["blocked_count"] = len(rows) - evaluable
    out["evaluable_with_dataset_gaps"] = sum(1 for e in out["records"] if e["evaluable"] and e["dataset_gaps"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audit.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = audit()
    print(f"dataset: {res['dataset_path']}")
    print(f"sha256 : {res['dataset_sha256']}")
    print(f"records: {res['record_count']}  unique: {res['unique_ids']}  duplicates: {res['duplicate_ids']}")
    print(f"evaluable: {res['evaluable_count']}  blocked: {res['blocked_count']} "
          f"(of evaluable, {res['evaluable_with_dataset_gaps']} have reconstructable dataset gaps)")
    for r in res["records"]:
        if not r["evaluable"]:
            print(f"  BLOCKED {r['instance_id']}: {r['blockers']}")
        elif r["dataset_gaps"]:
            print(f"  OK(gap) {r['instance_id']}: {r['dataset_gaps']}")
    print(f"wrote {OUT_DIR / 'audit.json'}")
