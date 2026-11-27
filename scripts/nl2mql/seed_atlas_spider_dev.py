"""MANUAL, contingency-only seed of the evaluation databases' collections into Atlas (MQL-0006).

Use it only if the Kaggle preflight reports missing collections. Project evidence (2026-10-20 parity check) says
all 80 dev collections already exist in `codegen_spider1`, so normally this script is NOT needed.

Safety rules (unlike scripts/spider1/import_atlas.py, this script never drops anything):
  * dry run by default: prints the plan from the bundle's data/atlas_mapping.json and the bundled SQLite files;
    no network access at all
  * --execute requires an explicit writable URI in the environment variable named by --uri-env (default
    ATLAS_SEED_URI); the read-only evaluation URI (MONGODB_URI) is refused by name
  * aborts before writing if ANY target collection already exists (it never overwrites, drops or appends)
  * deterministic: rows inserted in rowid order with the same value mapping as the original import
    (INTEGER->int, REAL->double, TEXT->string with U+FFFD for invalid UTF-8, NULL->explicit null)
  * after inserting, every collection is re-read and must match the bundle's count and type-tagged row digest;
    on a mismatch it stops and reports (it does not delete what it wrote: inspect and clean up by hand)

Usage:
  python scripts/nl2mql/seed_atlas_spider_dev.py                       # dry run, every mapped collection
  python scripts/nl2mql/seed_atlas_spider_dev.py --only wta_1__players # dry run, one collection
  set ATLAS_SEED_URI=...  (PowerShell: $env:ATLAS_SEED_URI = "...")
  python scripts/nl2mql/seed_atlas_spider_dev.py --execute --target-db codegen_spider1
The evaluation itself must then run with the read-only account, never with the seed URI.
"""
import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
import dev1034_core as C  # noqa: E402

DEFAULT_BUNDLE = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2"
REFUSED_URI_ENVS = {"MONGODB_URI", "MONGODB_ATLAS_URI"}


class SeedError(RuntimeError):
    pass


def sqlite_docs(bundle, m):
    path = Path(bundle) / "database" / m["db_id"] / f"{m['db_id']}.sqlite"
    con = C._ro_connect(path, 600)
    try:
        q = ", ".join(f'"{c}"' for c in m["fields"])
        try:
            rows = con.execute(f'SELECT {q} FROM "{m["table"]}" ORDER BY rowid').fetchall()
        except Exception:  # noqa: BLE001  (WITHOUT ROWID table)
            rows = con.execute(f'SELECT {q} FROM "{m["table"]}"').fetchall()
    finally:
        con.close()
    return [dict(zip(m["fields"], r)) for r in rows]


def plan(bundle, only=None):
    mapping = C.load_mapping(bundle)
    colls = {k: v for k, v in mapping["collections"].items() if not only or k in only}
    if only and set(only) - set(colls):
        raise SeedError(f"unknown collections: {sorted(set(only) - set(colls))}")
    return mapping, colls


def seed(client, target_db, colls, bundle, log=print, batch=20000):
    db = client[target_db]
    existing = set(db.list_collection_names())
    clash = sorted(set(colls) & existing)
    if clash:
        raise SeedError(f"{len(clash)} target collections already exist (e.g. {clash[:5]}); nothing was written. "
                        "This script never overwrites or drops.")
    written = []
    for name, m in sorted(colls.items()):
        docs = sqlite_docs(bundle, m)
        if len(docs) != m["documents"]:
            raise SeedError(f"{name}: {len(docs)} SQLite rows, mapping says {m['documents']}")
        for i in range(0, len(docs), batch):
            db[name].insert_many(docs[i:i + batch], ordered=True)
        written.append(name)
        n, dig, extra = C.atlas_collection_digest(db, name, m["fields"])
        if n != m["documents"] or dig != m["row_multiset_sha256"] or extra:
            raise SeedError(f"{name}: post-insert integrity check failed (docs {n}, digest match "
                            f"{dig == m['row_multiset_sha256']}, extra fields {extra}); written so far: {written}")
        if log:
            log(f"  seeded + verified {name} ({n} docs)")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--uri-env", default="ATLAS_SEED_URI")
    ap.add_argument("--target-db", default=C.ATLAS_DB)
    a = ap.parse_args(argv)
    try:
        C.verify_bundle(a.bundle)
        mapping, colls = plan(a.bundle, a.only)
        total = sum(m["documents"] for m in colls.values())
        print(f"plan: {len(colls)} collections, {total} documents -> {a.target_db} (mode: "
              f"{'EXECUTE' if a.execute else 'dry run'})")
        for name, m in sorted(colls.items()):
            print(f"  {name:55s} {m['documents']:>8d} docs  fields={len(m['fields'])}")
        if not a.execute:
            print("dry run only: nothing was contacted or written")
            return 0
        if a.uri_env in REFUSED_URI_ENVS:
            raise SeedError(f"{a.uri_env} is the read-only evaluation URI; use a separate writable URI variable")
        uri = os.environ.get(a.uri_env, "").strip()
        if not uri:
            raise SeedError(f"--execute needs a writable URI in ${a.uri_env}")
        from mongo_conn import tls_kwargs
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=20000, **tls_kwargs(uri))
        try:
            seed(client, a.target_db, colls, a.bundle)
        finally:
            client.close()
        print("done: re-run the Kaggle preflight with the READ-ONLY account")
        return 0
    except (SeedError, C.PreflightError) as e:
        print(f"SEED ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
