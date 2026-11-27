"""Compare Atlas-observed fingerprints (saved MCP aggregation outputs) with the backup's expected fingerprints.

Usage: python -B compare_ts_fingerprints.py <backup_dir> <observed_dir>
Writes <observed_dir>/comparison.json (refuses to overwrite) and exits 1 on any difference.
"""
import glob
import json
import sys
from pathlib import Path


def main(backup, observed):
    E = json.loads((Path(backup) / "expected_fingerprints.json").read_text(encoding="utf-8"))["collections"]
    res, ok = {}, True
    for db in sorted(E):
        obs = {}
        for f in sorted(glob.glob(str(Path(observed) / f"{db}_chunk*.json"))):
            for d in json.loads(Path(f).read_text(encoding="utf-8"))["documents"]:
                if d["_id"] in obs:
                    raise SystemExit(f"{db}: {d['_id']} observed twice")
                obs[d["_id"]] = (d["n"], int(d["h"]["$numberLong"]))
        exp = {c: (v["documents"], v["hash_sum"]) for c, v in E[db].items()}
        r = {"expected_collections": len(exp), "observed_collections": len(obs),
             "missing_on_atlas": sorted(set(exp) - set(obs)), "extra_on_atlas": sorted(set(obs) - set(exp)),
             "mismatched": sorted(c for c in exp if c in obs and obs[c] != exp[c]),
             "documents_expected": sum(v[0] for v in exp.values()),
             "documents_observed": sum(v[0] for v in obs.values())}
        r["exact_match"] = not (r["missing_on_atlas"] or r["extra_on_atlas"] or r["mismatched"]) and \
            r["documents_expected"] == r["documents_observed"]
        ok &= r["exact_match"]
        res[db] = r
    out = Path(observed) / "comparison.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.write_text(json.dumps({"all_exact_match": ok, "databases": res}, indent=1), encoding="utf-8")
    print(json.dumps({"all_exact_match": ok, "databases": res}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
