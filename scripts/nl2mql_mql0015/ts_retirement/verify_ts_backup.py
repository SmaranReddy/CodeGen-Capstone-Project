"""Verify an extracted ts-retirement backup independently of the SQLite sources.

1. Every file listed in SHA256SUMS matches.
2. Every dump/<db>/<coll>.bson decodes; its documents' fingerprint (same method as build_ts_backup.py) and count
   equal the Atlas-observed values saved from the read-only MCP aggregation (observed/<db>_chunk*.json).
3. Every metadata file declares only the default _id index.

Usage: python -B verify_ts_backup.py <extracted_backup_dir> <observed_dir>
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_ts_backup as T  # noqa: E402
import bson  # noqa: E402


def main(root, observed):
    root = Path(root)
    bad_sums = []
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for ln in lines:
        h, rel = ln.split("  ", 1)
        if T.sha256_file(root / rel) != h:
            bad_sums.append(rel)
    res = {"sha256sums_checked": len(lines), "sha256sums_failed": bad_sums, "databases": {}}
    ok = not bad_sums
    for dbdir in sorted((root / "dump").iterdir()):
        obs = {}
        for f in glob.glob(str(Path(observed) / f"{dbdir.name}_chunk*.json")):
            for d in json.loads(Path(f).read_text(encoding="utf-8"))["documents"]:
                obs[d["_id"]] = (d["n"], int(d["h"]["$numberLong"]))
        mism, idx_bad, n_docs, n_coll = [], [], 0, 0
        for bf in sorted(dbdir.glob("*.bson")):
            coll = bf.stem
            docs = bson.decode_all(bf.read_bytes())
            fp = (len(docs), sum(T.cmod(T.hashed_index_key_str(T.doc_string(d))) for d in docs))
            if obs.get(coll) != fp:
                mism.append(coll)
            meta = json.loads((dbdir / f"{coll}.metadata.json").read_text(encoding="utf-8"))
            if [i["name"] for i in meta["indexes"]] != ["_id_"]:
                idx_bad.append(coll)
            n_docs += len(docs)
            n_coll += 1
        missing = sorted(set(obs) - {p.stem for p in dbdir.glob("*.bson")})
        r = {"collections": n_coll, "documents": n_docs, "fingerprint_mismatch_vs_atlas": mism,
             "missing_vs_atlas": missing, "non_default_index_metadata": idx_bad}
        r["ok"] = not (mism or missing or idx_bad) and n_coll == len(obs)
        ok &= r["ok"]
        res["databases"][dbdir.name] = r
    res["all_ok"] = ok
    print(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
