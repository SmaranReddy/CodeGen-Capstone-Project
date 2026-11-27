"""Source-reconstructed, Atlas-verifiable backup of codegen_spider1_ts1..3 (no Atlas access in this script).

The three Atlas databases were created by scripts/spider1/testsuite_setup.py, which imported official Spider
test-suite variant SQLite files (choice recorded in data/interim/spider1/testsuite_variants.json) with
scripts/spider1/import_atlas.import_db: one collection per table, rowid order, SQLite storage class kept,
text decoded UTF-8 with U+FFFD replacement. This script repeats exactly that row->document mapping
(using the same spider1_kit helpers) WITHOUT connecting to MongoDB and writes:

  <out>/dump/<db>/<coll>.bson + <coll>.metadata.json   (mongorestore layout; documents without _id)
  <out>/sources/<db_id>/<variant>.sqlite               (copies, each verified against the official zip member)
  <out>/sources/testsuite_variants.json
  <out>/expected_fingerprints.json                      (per collection: docs, fields, multiset hash)
  <out>/atlas_pipelines/<db>.json                       (read-only aggregation computing the same fingerprint)
  <out>/SHA256SUMS

Fingerprint (per collection): n documents and H = sum over documents of mod(toHashedIndexKey(s), 2^31-1), where
s concatenates, in stored field order and excluding _id, `name 0x1F bsontype 0x1F value-as-string 0x1E`.
Python reproduces MongoDB's $toHashedIndexKey for strings (md5, seed 0) and $toString for int/long/double
(both validated against Atlas 8.0.39 on literals). Order-independent, type- and field-order-sensitive.

Usage: python -B build_ts_backup.py --out <dir outside OneDrive>
"""
import argparse
import hashlib
import json
import shutil
import struct
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/spider1"))
import spider1_kit as K  # noqa: E402  (unchanged helpers; no network)
import bson  # noqa: E402

VARIANTS = ROOT / "data/interim/spider1/testsuite_variants.json"
ZIP = ROOT / "data/raw/spider1_testsuite_download/testsuite_database.zip"
M = 2147483647


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hashed_index_key_str(s):
    b = s.encode("utf-8")
    d = hashlib.md5(struct.pack("<i", 0) + struct.pack("<i", 15) + struct.pack("<i", len(b) + 1) + b + b"\0").digest()
    return struct.unpack("<q", d[:8])[0]


def cmod(x):
    r = abs(x) % M
    return -r if x < 0 else r


def bson_type(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int" if -2**31 <= v < 2**31 else "long"
    if isinstance(v, float):
        return "double"
    if isinstance(v, str):
        return "string"
    raise TypeError(f"unsupported value type {type(v)}")


def mongo_str(v):
    if v is None:
        return ""
    if isinstance(v, float):
        r = repr(v)
        return r[:-2] if r.endswith(".0") else r
    return str(v)


def doc_string(d):
    return "".join(f"{k}\x1f{bson_type(v)}\x1f{mongo_str(v)}\x1e" for k, v in d.items())


def table_docs(db_id, variant, table):
    cols = [c for c, _ in K.table_schema(db_id, table, variant)]
    con = K.sqlite_ro(db_id, variant)
    try:
        q = ", ".join(f'"{c}"' for c in cols)
        try:
            con.execute(f'SELECT rowid FROM "{table}" LIMIT 1')
            rows = con.execute(f'SELECT {q} FROM "{table}" ORDER BY rowid').fetchall()
        except Exception:  # noqa: BLE001  (same fallback as import_atlas._has_rowid)
            rows = con.execute(f'SELECT {q} FROM "{table}"').fetchall()
    finally:
        con.close()
    return cols, [dict(zip(cols, r)) for r in rows]


TAG = "__fp_coll"
FP_TAIL = [
    {"$project": {"_id": 0, TAG: 1, "s": {"$reduce": {
        "input": {"$filter": {"input": {"$objectToArray": "$$ROOT"},
                              "cond": {"$not": [{"$in": ["$$this.k", ["_id", TAG]]}]}}},
        "initialValue": "",
        "in": {"$concat": ["$$value", "$$this.k", "", {"$type": "$$this.v"}, "",
                           {"$ifNull": [{"$toString": "$$this.v"}, ""]}, ""]}}}}},
    {"$group": {"_id": "$" + TAG, "n": {"$sum": 1}, "h": {"$sum": {"$mod": [{"$toHashedIndexKey": "$s"}, M]}}}},
    {"$sort": {"_id": 1}},
]


def atlas_pipeline(names):
    pipe = [{"$set": {TAG: {"$literal": names[0]}}}]
    pipe += [{"$unionWith": {"coll": c, "pipeline": [{"$set": {TAG: {"$literal": c}}}]}} for c in names[1:]]
    return pipe + FP_TAIL


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if "onedrive" in str(out.resolve()).lower():
        raise SystemExit("backup must be outside OneDrive")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; refusing to overwrite")
    variants = json.loads(VARIANTS.read_text(encoding="utf-8"))
    z = zipfile.ZipFile(ZIP)
    expected, src_manifest = {}, {}
    for k in range(3):
        mdb = f"{K.ATLAS_DB}_ts{k + 1}"
        expected[mdb] = {}
        for db_id, picks in sorted(variants.items()):
            if k >= len(picks):
                continue
            v = picks[k]
            src = K.TS_DIR / db_id / f"{v}.sqlite"
            member = hashlib.sha256(z.read(f"database/{db_id}/{v}.sqlite")).hexdigest()
            local = sha256_file(src)
            if local != member:
                raise SystemExit(f"{src}: differs from the official zip member")
            dst = out / "sources" / db_id / f"{v}.sqlite"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            src_manifest[f"{db_id}/{v}.sqlite"] = local
            for table in K.tables(db_id, v):
                coll = K.collection_name(db_id, table)
                cols, docs = table_docs(db_id, v, table)
                H = sum(cmod(hashed_index_key_str(doc_string(d))) for d in docs)
                types = sorted({bson_type(x) for d in docs for x in d.values()})
                expected[mdb][coll] = {"db_id": db_id, "variant": v, "table": table, "fields": cols,
                                       "documents": len(docs), "hash_sum": H, "value_types": types}
                ddir = out / "dump" / mdb
                ddir.mkdir(parents=True, exist_ok=True)
                with open(ddir / f"{coll}.bson", "wb") as f:
                    for d in docs:
                        f.write(bson.encode(d))
                (ddir / f"{coll}.metadata.json").write_text(json.dumps(
                    {"options": {}, "indexes": [{"v": 2, "key": {"_id": 1}, "name": "_id_"}],
                     "collectionName": coll, "type": "collection"}), encoding="utf-8")
        names = sorted(expected[mdb])
        pipe = atlas_pipeline(names)
        pdir = out / "atlas_pipelines"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / f"{mdb}.json").write_text(json.dumps({"database": mdb, "collection": names[0], "pipeline": pipe}),
                                          encoding="utf-8")
    shutil.copyfile(VARIANTS, out / "sources" / "testsuite_variants.json")
    summary = {mdb: {"collections": len(c), "documents": sum(x["documents"] for x in c.values())}
               for mdb, c in expected.items()}
    (out / "expected_fingerprints.json").write_text(json.dumps(
        {"method": __doc__.split("Fingerprint (per collection): ")[1].split("\n\nUsage")[0],
         "zip_sha256": sha256_file(ZIP), "variants_json_sha256": sha256_file(VARIANTS),
         "source_sqlite_sha256": src_manifest, "summary": summary, "collections": expected}, indent=1),
        encoding="utf-8")
    lines = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{sha256_file(p)}  {p.relative_to(out).as_posix()}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1), "\nfiles:", len(lines))


if __name__ == "__main__":
    main()
