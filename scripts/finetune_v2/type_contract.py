"""TC-1 type-conformant domain contract (plan docs/pilot_v3_followup_implementation_plan.md Part A).

Converter-independent: imports no sql2mql_v2*, verifier*, fixtures*, mongo_import or schema_profile module.

    python -I -B scripts/finetune_v2/type_contract.py build          -> data/interim/finetune_v2/type_contract_v1.json
    python -I -B scripts/finetune_v2/type_contract.py build-eval     -> data/interim/finetune_v2/type_contract_eval_v1.json
    python -I -B scripts/finetune_v2/type_contract.py census         -> prints the M1 / M2 census summaries
    python -I -B scripts/finetune_v2/type_contract.py check-local    -> outputs/pilot_v3_followup/type_contract_check_local.json

Contract class K(C, f), from the OBSERVED non-NULL SQLite storage classes (declared type kept as metadata only):
    int  INTEGER only | num  INTEGER and/or REAL | text  TEXT only | null  no non-NULL value
    nonconformant  anything else (several classes incl. TEXT, or BLOB)
Allowed BSON types for non-null values: int -> int,long ; num -> int,long,double ; text -> string ; null -> none.
Forbidden anywhere: bool, date, timestamp, decimal, objectId (except _id), binData, array, object.
A field may be explicit null or (missing representation) absent, only where the SQLite value is NULL.
"""
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPIDER_DB = ROOT / "data/raw/spider1/spider_data/database"
POOL = ROOT / "data/processed/spider_pool_v2"
INTERIM = ROOT / "data/interim/finetune_v2"
MANIFEST = INTERIM / "type_contract_v1.json"
EVAL_MANIFEST = INTERIM / "type_contract_eval_v1.json"
BUNDLE = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2"
CONTRACT_VERSION = "TC-1"
ALLOWED = {"int": {"int", "long"}, "num": {"int", "long", "double"}, "text": {"string"}, "null": set(),
           "nonconformant": None}   # None: any non-forbidden scalar type; position rules live in the converter
FORBIDDEN = {"bool", "date", "timestamp", "decimal", "objectId", "binData", "array", "object", "regex",
             "javascript", "minKey", "maxKey", "undefined", "dbPointer", "symbol"}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def affinity(decl):
    """SQLite 3 affinity rules (https://sqlite.org/datatype3.html section 3.1)."""
    d = (decl or "").upper()
    if "INT" in d:
        return "INTEGER"
    if any(k in d for k in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if "BLOB" in d or not d.strip():
        return "BLOB"
    if any(k in d for k in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return "NUMERIC"


def contract_class(classes):
    cs = set(classes)
    if not cs:
        return "null"
    if cs == {"integer"}:
        return "int"
    if cs <= {"integer", "real"}:
        return "num"
    if cs == {"text"}:
        return "text"
    return "nonconformant"


def declared_conformant(aff, classes):
    """Does the observed class set match the declared affinity? (metadata; NOT the contract)"""
    cs = set(classes)
    if aff == "BLOB":
        return False   # BLOB affinity (incl. no declared type): the declared type gives no expectation
    if not cs:
        return True
    if aff in ("INTEGER", "REAL", "NUMERIC"):
        return cs <= {"integer", "real"}
    if aff == "TEXT":
        return cs == {"text"}
    return False


def sqlite_ro(path):
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="surrogateescape")
    return con


def has_bad_utf8(v):
    return isinstance(v, str) and any(0xDC80 <= ord(ch) <= 0xDCFF for ch in v)


def profile_db(path, db_id):
    con = sqlite_ro(path)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        out = {}
        for t in tables:
            info = con.execute(f'PRAGMA table_info("{t}")').fetchall()
            rows = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
            fields = {}
            for r in info:
                c, decl = r[1], r[2]
                cnt = dict(con.execute(f'SELECT typeof("{c}"), count(*) FROM "{t}" GROUP BY 1').fetchall())
                classes = sorted(k for k in cnt if k != "null")
                bad = 0
                if "text" in classes:
                    bad = sum(1 for (v,) in con.execute(f'SELECT "{c}" FROM "{t}" WHERE typeof("{c}")=\'text\'')
                              if has_bad_utf8(v))
                aff = affinity(decl)
                fields[c] = {"declared": decl, "affinity": aff, "observed_classes": classes,
                             "class_counts": {k: cnt[k] for k in sorted(cnt)}, "nonnull": rows - cnt.get("null", 0),
                             "K": contract_class(classes),
                             "declared_mismatch": not declared_conformant(aff, classes),
                             "utf8_invalid_values": bad}
            out[f"{db_id}__{t}"] = {"db_id": db_id, "table": t, "rows": rows, "column_order": [r[1] for r in info],
                                    "fields": fields}
        return out
    finally:
        con.close()


def pool_db_ids():
    ids = set()
    for f in ("train.jsonl", "val_internal.jsonl"):
        for line in open(POOL / f, encoding="utf-8"):
            ids.add(json.loads(line)["db_id"])
    return sorted(ids)


def census(collections):
    """M1-style census summary over a {coll: entry} map."""
    n = Counter()
    text_in_numeric = Counter()
    mixed_by_db = Counter()
    for e in collections.values():
        for c, f in e["fields"].items():
            n["columns"] += 1
            cls, aff = set(f["observed_classes"]), f["affinity"]
            if not f["declared_mismatch"]:
                n["conform_declared"] += 1
            elif cls == {"text"} and aff in ("INTEGER", "REAL", "NUMERIC"):
                n["text_under_numeric_affinity"] += 1
                text_in_numeric[(f["declared"] or "").upper().split("(")[0].strip()] += 1
            elif f["K"] == "nonconformant":
                n["mixed_class"] += 1
                mixed_by_db[e["db_id"]] += 1
            elif aff == "BLOB":
                n["blob_affinity_single_class"] += 1
            else:
                n["other_mismatch"] += 1
            n[f"K_{f['K']}"] += 1
    return {"counts": dict(n), "text_under_numeric_by_declared": dict(text_in_numeric),
            "mixed_by_db": dict(mixed_by_db)}


def write_manifest(path, body):
    if path.exists():
        sys.exit(f"refusing to overwrite {path}")
    blob = json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False)
    body_sha = sha_bytes(blob.encode("utf-8"))
    doc = dict(body, body_sha256=body_sha)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)} body_sha256={body_sha} file_sha256={sha_file(path)}")


def build():
    dbs = pool_db_ids()
    colls, db_sha = {}, {}
    for d in dbs:
        p = SPIDER_DB / d / f"{d}.sqlite"
        db_sha[d] = sha_file(p)
        colls.update(profile_db(p, d))
    body = {"contract": CONTRACT_VERSION, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script": "scripts/finetune_v2/type_contract.py", "script_sha256": sha_file(__file__),
            "scope": "databases referenced by data/processed/spider_pool_v2/{train,val_internal}.jsonl",
            "pool_files_sha256": {f: sha_file(POOL / f) for f in ("train.jsonl", "val_internal.jsonl")},
            "allowed_bson": {k: (sorted(v) if v is not None else "any non-forbidden scalar") for k, v in ALLOWED.items()},
            "forbidden_bson": sorted(FORBIDDEN), "db_ids": dbs, "sqlite_sha256": db_sha,
            "census": census(colls), "collections": colls}
    write_manifest(MANIFEST, body)


def build_eval():
    m = json.loads((BUNDLE / "data/atlas_mapping.json").read_text(encoding="utf-8"))
    colls, db_sha = {}, {}
    for d in sorted(m["db_ids"]):
        p = BUNDLE / "database" / d / f"{d}.sqlite"
        db_sha[d] = sha_file(p)
        prof = profile_db(p, d)
        for k, v in prof.items():
            if k in m["collections"]:
                colls[k] = v
    missing = sorted(set(m["collections"]) - set(colls))
    if missing:
        sys.exit(f"mapped collections without a bundle table: {missing[:5]}")
    body = {"contract": CONTRACT_VERSION, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script": "scripts/finetune_v2/type_contract.py", "script_sha256": sha_file(__file__),
            "scope": "the 70 collections of kaggle/nl2mql_spider_dev_1034_bundle_v2/data/atlas_mapping.json, "
                     "classes from the bundled SQLite files",
            "atlas_db": m["atlas_db"], "atlas_mapping_sha256": sha_file(BUNDLE / "data/atlas_mapping.json"),
            "allowed_bson": {k: (sorted(v) if v is not None else "any non-forbidden scalar") for k, v in ALLOWED.items()},
            "forbidden_bson": sorted(FORBIDDEN), "db_ids": sorted(m["db_ids"]), "sqlite_sha256": db_sha,
            "census": census(colls), "collections": colls}
    write_manifest(EVAL_MANIFEST, body)


def load_manifest(path=MANIFEST):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in doc.items() if k != "body_sha256"}
    blob = json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False)
    if sha_bytes(blob.encode("utf-8")) != doc["body_sha256"]:
        raise RuntimeError(f"{path}: body_sha256 does not match its content")
    return doc


# ------------------------------------------------------------------ census comparison (shared with Atlas check)
TYPE_CENSUS_PIPELINE = [
    {"$project": {"_id": 0, "kv": {"$objectToArray": "$$ROOT"}}},
    {"$unwind": "$kv"},
    {"$group": {"_id": {"f": "$kv.k", "t": {"$type": "$kv.v"}}, "n": {"$sum": 1}}},
]


def mongo_type_census(coll):
    """-> (doc_count, {field: {bson_type: n}}) via a read-only aggregation (_id excluded)."""
    n = coll.count_documents({})
    out = {}
    for r in coll.aggregate(TYPE_CENSUS_PIPELINE, allowDiskUse=False):
        out.setdefault(r["_id"]["f"], {})[r["_id"]["t"]] = r["n"]
    return n, out


def compare_census(entry, n_docs, types, representation):
    """Violations of TC-1 for one collection. representation: 'explicit_null' | 'missing' | 'any'."""
    v = []
    fields = entry["fields"]
    if n_docs != entry["rows"]:
        v.append(f"count: {n_docs} documents, contract {entry['rows']}")
    for f in types:
        if f == "_id":
            if set(types[f]) - {"objectId"}:
                v.append(f"_id_type: {sorted(types[f])}")
            continue
        if f not in fields:
            v.append(f"extra_field: {f}")
    for f, spec in fields.items():
        got = types.get(f, {})
        nonnull = {t: k for t, k in got.items() if t != "null"}
        for t in nonnull:
            if t in FORBIDDEN:
                v.append(f"forbidden_type: {f} has {t}")
            elif ALLOWED[spec["K"]] is not None and t not in ALLOWED[spec["K"]]:
                v.append(f"disallowed_type: {f} (K={spec['K']}) has {t} x{nonnull[t]}")
        nn = sum(nonnull.values())
        if nn != spec["nonnull"]:
            v.append(f"nonnull_count: {f} has {nn} non-null values, contract {spec['nonnull']}")
        present = sum(got.values())
        if representation == "explicit_null" and present != entry["rows"]:
            v.append(f"absent_field: {f} present in {present}/{entry['rows']} documents (explicit-null representation)")
        if representation == "missing" and got.get("null"):
            v.append(f"explicit_null_in_missing_repr: {f} x{got['null']}")
    return v


def check_local():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pymongo import MongoClient
    doc = load_manifest()
    cl = MongoClient("127.0.0.1", 27018, directConnection=True, serverSelectionTimeoutMS=5000)
    ver = cl.server_info()["version"]
    rep = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "where": "LOCAL MongoDB 127.0.0.1:27018 (not Atlas)",
           "server_version": ver, "manifest_sha256": sha_file(MANIFEST), "manifest_body_sha256": doc["body_sha256"],
           "databases": {}}
    ok = True
    for dbname, repr_ in (("spider_train_v2", "explicit_null"), ("spider_train_v2_missing", "missing")):
        mdb = cl[dbname]
        present = set(mdb.list_collection_names())
        meta = {m["_id"]: m for m in cl["spider_train_v2"]["_import_meta"].find({})}
        res = {"checked": 0, "violations": {}, "not_present": [], "stale_import": []}
        for coll, entry in doc["collections"].items():
            d = entry["db_id"]
            if coll not in present:
                res["not_present"].append(coll)
                continue
            if meta.get(d, {}).get("sha256") != doc["sqlite_sha256"][d]:
                res["stale_import"].append(coll)
            n, types = mongo_type_census(mdb[coll])
            viol = compare_census(entry, n, types, repr_)
            res["checked"] += 1
            if viol:
                res["violations"][coll] = viol
        ok &= not res["violations"] and not res["not_present"] and not res["stale_import"]
        rep["databases"][dbname] = res
    rep["ok"] = ok
    out = ROOT / "outputs/pilot_v3_followup/type_contract_check_local.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if k != "databases" else {d: {"checked": r["checked"], "violating_collections": len(r["violations"]),
                                                           "not_present": len(r["not_present"]), "stale": len(r["stale_import"])}
                                                       for d, r in v.items()}) for k, v in rep.items()}, indent=1))
    return ok


def main():
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "build-eval":
        build_eval()
    elif cmd == "census":
        for p in (MANIFEST, EVAL_MANIFEST):
            if p.exists():
                print(p.name, json.dumps(load_manifest(p)["census"], indent=1))
    elif cmd == "check-local":
        sys.exit(0 if check_local() else 1)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
