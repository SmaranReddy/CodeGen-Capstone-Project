"""Import the Spider 1.0 dev databases into the dedicated Atlas database `codegen_spider1`.

One collection per table, named `<db_id>__<table>`; every column is kept; values keep
their SQLite storage class (INTEGER->int, REAL->double, TEXT->string, NULL->null), so
MongoDB sees the same data SQLite does. Rows are inserted in rowid order (the default
ObjectId therefore follows SQLite scan order). Only collections inside `codegen_spider1`
are ever dropped/recreated, and only those named after a Spider table.

Writes data/interim/spider1/atlas_collections.json (counts + observed BSON types).

Usage: python scripts/spider1/import_atlas.py [db_id ...]   (default: every db in sample.jsonl)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402

MANIFEST = K.INTERIM / "atlas_collections.json"


def bson_type(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int" if -2**31 <= v < 2**31 else "long"
    if isinstance(v, float):
        return "double"
    return "string"


def import_db(mdb, db_id, variant=None, quiet=False):
    out = {}
    con = K.sqlite_ro(db_id, variant)
    for table in K.tables(db_id, variant):
        cols = [c for c, _ in K.table_schema(db_id, table, variant)]
        name = K.collection_name(db_id, table)
        qcols = ", ".join(f'"{c}"' for c in cols)
        rows = con.execute(f'SELECT {qcols} FROM "{table}" ORDER BY rowid').fetchall() \
            if _has_rowid(con, table) else con.execute(f'SELECT {qcols} FROM "{table}"').fetchall()
        docs = [dict(zip(cols, r)) for r in rows]
        mdb.drop_collection(name)
        for i in range(0, len(docs), 20000):
            mdb[name].insert_many(docs[i:i + 20000], ordered=True)
        n = mdb[name].count_documents({})
        assert n == len(docs), (name, n, len(docs))
        types = {}
        for d in docs:
            for c in cols:
                types.setdefault(c, set()).add(bson_type(d[c]))
        out[name] = {"db_id": db_id, "table": table, "documents": n,
                     "fields": {c: sorted(types.get(c, {"null"})) for c in cols},
                     "declared": dict(K.table_schema(db_id, table, variant))}
        if not quiet:
            print(f"  {name:55s} {n:>8d}")
    con.close()
    return out


def _has_rowid(con, table):
    try:
        con.execute(f'SELECT rowid FROM "{table}" LIMIT 1')
        return True
    except Exception:  # noqa: BLE001
        return False


def main():
    dbs = sys.argv[1:] or sorted({r["db_id"] for r in K.read_jsonl(K.INTERIM / "sample.jsonl")})
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    client = K.mongo_client()
    try:
        mdb = client[K.ATLAS_DB]
        for db_id in dbs:
            print(f"== {db_id}")
            manifest.update(import_db(mdb, db_id))
            MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        st = mdb.command("dbStats")
        print(f"{K.ATLAS_DB}: collections={st['collections']} data={st['dataSize']/1e6:.1f}MB "
              f"index={st['indexSize']/1e6:.1f}MB")
    finally:
        client.close()


if __name__ == "__main__":
    main()
