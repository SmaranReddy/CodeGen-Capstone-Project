"""Report live Atlas storage: cluster-wide databases and the pilot database usage.

Read-only. Never prints the Mongo URI. Usage: python scripts/atlas_status.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conversion_kit import PILOT_DB, mongo_client


def main():
    client = mongo_client()
    try:
        r = client.admin.command("listDatabases")
        total = 0.0
        for d in sorted(r["databases"], key=lambda x: -x["sizeOnDisk"]):
            mb = d["sizeOnDisk"] / 1e6
            total += d["sizeOnDisk"]
            print(f"{d['name']:35s} {mb:10.2f} MB")
        print(f"{'TOTAL (sizeOnDisk)':35s} {total / 1e6:10.2f} MB  (limit 512 MB)")
        db = client[PILOT_DB]
        print(f"\npilot db '{PILOT_DB}':")
        stats = db.command("dbStats")
        print(f"  collections={stats['collections']} objects={stats['objects']} "
              f"data={stats['dataSize'] / 1e6:.2f} MB storage={stats['storageSize'] / 1e6:.2f} MB "
              f"indexes={stats['indexSize'] / 1e6:.2f} MB")
        for name in sorted(db.list_collection_names()):
            c = db[name]
            print(f"    {name:45s} docs={c.count_documents({}):>8d}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
