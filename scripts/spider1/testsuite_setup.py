"""Extract and import Spider test-suite database variants (coincidence-proof evidence).

The official test suite (Zhong et al., 2020; taoyds/test-suite-sql-eval, Google Drive id
1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w) ships many alternative contents per Spider database.
For every sampled db_id except wta_1 (variants are ~50 MB each), pick V variants with a
fixed seed, extract them to data/raw/spider1/testsuite/<db>/ and import each into its own
Atlas database `codegen_spider1_ts<k>` with the SAME collection names as the dev import,
so gold MQL runs unchanged.

Usage: python scripts/spider1/testsuite_setup.py [--variants 3] [--seed 42]
"""
import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402
from import_atlas import import_db  # noqa: E402

ZIP = K.ROOT / "data/raw/spider1_testsuite_download/testsuite_database.zip"
SKIP = {"wta_1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dbs = sorted({r["db_id"] for r in K.read_jsonl(K.INTERIM / "sample.jsonl")} - SKIP)
    z = zipfile.ZipFile(ZIP)
    names = [n for n in z.namelist() if n.startswith("database/") and n.endswith(".sqlite")]
    rng = random.Random(args.seed)
    chosen = {}
    for db in dbs:
        cands = sorted(Path(n).stem for n in names
                       if n.split("/")[1] == db and Path(n).stem != db)
        pick = rng.sample(cands, min(args.variants, len(cands)))
        chosen[db] = pick
        (K.TS_DIR / db).mkdir(parents=True, exist_ok=True)
        for v in pick:
            dest = K.TS_DIR / db / f"{v}.sqlite"
            if not dest.exists():
                dest.write_bytes(z.read(f"database/{db}/{v}.sqlite"))
    client = K.mongo_client()
    try:
        for k in range(args.variants):
            mdb = client[f"{K.ATLAS_DB}_ts{k + 1}"]
            for db, pick in chosen.items():
                if k < len(pick):
                    import_db(mdb, db, pick[k], quiet=True)
            st = mdb.command("dbStats")
            print(f"{mdb.name}: collections={st['collections']} data={st['dataSize']/1e6:.1f}MB")
    finally:
        client.close()
    (K.INTERIM / "testsuite_variants.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    main()
