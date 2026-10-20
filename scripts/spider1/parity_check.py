"""Verify that every Spider 1.0 dev table in SQLite has an equivalent collection on Atlas.

Read-only on both sides (SQLite opened with mode=ro; Atlas only `find`). For each table of
each db_id that has pool items, compares the official SQLite table with the Atlas
collection `codegen_spider1.<db_id>__<table>`:
  * document count == row count
  * field names == column names (plus the ObjectId `_id`)
  * multiset of type-tagged rows is identical: a value's storage class is part of the key,
    so TEXT '3' != INTEGER 3 != REAL 3.0, and NULL != missing field
  * natural (insertion) order equals SQLite rowid order (reported, not required)
  * strings that needed UTF-8 replacement on read are counted (the Atlas copy holds the
    replaced text, so such tables are flagged)

Writes data/interim/spider1/expanded/parity.json. A db_id is `parity: true` only if every
one of its tables matches; the expanded dataset admits only items from such db_ids.

Usage: python scripts/spider1/parity_check.py [db_id ...]
"""
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402

OUT = K.INTERIM / "expanded/parity.json"


def tag(v):
    if v is None:
        return ("null", None)
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, float):
        return ("real", v)
    if isinstance(v, (bytes, bytearray)):
        return ("blob", bytes(v))
    return ("text", v)


def sqlite_rows(db_id, table):
    con = K.sqlite_ro(db_id)
    try:
        cols = [c for c, _ in K.table_schema(db_id, table)]
        q = ", ".join(f'"{c}"' for c in cols)
        try:
            rows = con.execute(f'SELECT {q} FROM "{table}" ORDER BY rowid').fetchall()
        except Exception:  # noqa: BLE001  (WITHOUT ROWID table)
            rows = con.execute(f'SELECT {q} FROM "{table}"').fetchall()
        return cols, rows
    finally:
        con.close()


def check_table(mdb, db_id, table):
    cols, rows = sqlite_rows(db_id, table)
    name = K.collection_name(db_id, table)
    docs = list(mdb[name].find({}))
    res = {"collection": name, "sqlite_rows": len(rows), "atlas_docs": len(docs)}
    extra_fields = Counter(k for d in docs for k in d if k != "_id" and k not in cols)
    missing_fields = Counter(c for d in docs for c in cols if c not in d)
    s_keys = [tuple(tag(v) for v in r) for r in rows]
    a_keys = [tuple(tag(d[c]) if c in d else ("missing", None) for c in cols) for d in docs]
    sc, ac = Counter(s_keys), Counter(a_keys)
    only_s, only_a = sc - ac, ac - sc
    res.update(
        columns=cols,
        extra_fields=dict(extra_fields), missing_fields=dict(missing_fields),
        rows_only_in_sqlite=sum(only_s.values()), rows_only_in_atlas=sum(only_a.values()),
        examples_only_in_sqlite=[[list(x) for x in k] for k in list(only_s)[:3]],
        examples_only_in_atlas=[[list(x) for x in k] for k in list(only_a)[:3]],
        natural_order_matches_rowid=s_keys == a_keys,
        utf8_replaced_strings=sum(isinstance(v, str) and "�" in v for r in rows for v in r),
        type_classes={c: sorted({tag(r[i])[0] for r in rows}) for i, c in enumerate(cols)},
    )
    res["match"] = (len(rows) == len(docs) and not extra_fields and not missing_fields
                    and not only_s and not only_a and res["utf8_replaced_strings"] == 0)
    return res


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    pool = K.read_jsonl(K.INTERIM / "pool.jsonl")
    dbs = sys.argv[1:] or sorted({r["db_id"] for r in pool})
    report = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() and sys.argv[1:] else {}
    dbr = report.get("databases", {})
    client = K.mongo_client()
    try:
        mdb = client[K.ATLAS_DB]
        existing = set(mdb.list_collection_names())
        for db_id in dbs:
            t0 = time.time()
            tabs = K.tables(db_id)
            expected = {K.collection_name(db_id, t) for t in tabs}
            on_atlas = {c for c in existing if c.startswith(db_id + "__")}
            res = {"tables": {}, "atlas_collections_without_table": sorted(on_atlas - expected)}
            for t in tabs:
                if K.collection_name(db_id, t) not in existing:
                    res["tables"][t] = {"match": False, "error": "collection missing on Atlas"}
                    continue
                res["tables"][t] = check_table(mdb, db_id, t)
            res["parity"] = all(v["match"] for v in res["tables"].values()) \
                and not res["atlas_collections_without_table"]
            res["natural_order_matches_rowid"] = all(
                v.get("natural_order_matches_rowid") for v in res["tables"].values())
            dbr[db_id] = res
            print(f"{'OK ' if res['parity'] else 'BAD'} {db_id:30s} tables={len(tabs)} "
                  f"rows={sum(v.get('sqlite_rows', 0) for v in res['tables'].values())} "
                  f"order={res['natural_order_matches_rowid']} {time.time() - t0:.1f}s", flush=True)
            for t, v in res["tables"].items():
                if not v["match"]:
                    print("    mismatch", t, {k: v.get(k) for k in (
                        "error", "sqlite_rows", "atlas_docs", "rows_only_in_sqlite", "rows_only_in_atlas",
                        "extra_fields", "missing_fields", "utf8_replaced_strings")})
    finally:
        client.close()
    report = {
        "atlas_db": K.ATLAS_DB,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "type-tagged row multiset equality per table (SQLite mode=ro vs Atlas find)",
        "prompt_context_sha256": {p: sha(K.INTERIM / p) for p in ("atlas_collections.json", "prompt_samples.json")},
        "databases": dict(sorted(dbr.items())),
        "all_parity": all(v["parity"] for v in dbr.values()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("all_parity:", report["all_parity"])


if __name__ == "__main__":
    main()
