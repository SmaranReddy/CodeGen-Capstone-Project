"""Result-invariant speed-up: single-field indexes on primary-key / foreign-key / *id* columns of
large (> 2000 documents) collections of the local real-data import (spider_train_v2 and
spider_train_v2_missing). An index never changes an aggregation's result; it only makes
$lookup on that field fast (fewer verifier timeouts). Local MongoDB only.
    python -B scripts/finetune_v2/index_real.py
"""
import json
import sqlite3

import v2_common as C


def key_columns(db_id, table):
    con = sqlite3.connect(f"file:{C.sqlite_path(db_id).as_posix()}?mode=ro", uri=True)
    try:
        info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
        cols = {r[1] for r in info if r[5]}
        cols |= {r[3] for r in con.execute(f'PRAGMA foreign_key_list("{table}")')}
        for db2 in [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]:
            for fk in con.execute(f'PRAGMA foreign_key_list("{db2}")'):
                if fk[2].lower() == table.lower() and fk[4]:
                    cols.add(fk[4])
        cols |= {r[1] for r in info if "id" in r[1].lower()}
        real = {r[1].lower(): r[1] for r in info}
        return sorted({real[c.lower()] for c in cols if c and c.lower() in real})
    finally:
        con.close()


def main():
    client = C.local_client()
    C.check_server(client)
    made = {}
    for dbname in (C.REAL_DB, C.REAL_DB + "_missing"):
        mdb = client[dbname]
        for name in mdb.list_collection_names():
            if "__" not in name or mdb[name].estimated_document_count() <= 2000:
                continue
            db_id, table = name.split("__", 1)
            for c in key_columns(db_id, table):
                mdb[name].create_index([(c, 1)])
                made.setdefault(f"{dbname}.{name}", []).append(c)
    print(json.dumps(made, indent=0))


if __name__ == "__main__":
    main()
