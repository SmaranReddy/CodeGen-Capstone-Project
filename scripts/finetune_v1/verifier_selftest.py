"""Verifier self-test: does fixture-equivalence verification reject WRONG MQL?

    python scripts/finetune_v1/verifier_selftest.py [--sample 300] [--seed 20261024]

For a seeded sample of verified pairs, applies mechanical MQL corruptions (strict<->non-strict
comparison, flipped sort direction, limit+1, skip+1, removed projection field, `$nin` null
guard removed, `$in` element dropped, regex anchor removed, `$match` dropped) and re-runs the
SAME comparison on the SAME regenerated fixtures. A corruption that still matches everywhere
is "not caught". Some corruptions are semantically equivalent on all well-formed data (e.g.
`$gte` vs `$gt` against a date-only literal on a datetime column); those are reported by
operator so the catch rate can be read per class. Writes
data/interim/finetune_v1/verify/selftest.json.
"""
import argparse
import collections
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nk_common as C  # noqa: E402
import fixtures as F  # noqa: E402
import sqlsem as S  # noqa: E402
import verify as V  # noqa: E402


def walk_replace(obj, fn):
    if isinstance(obj, dict):
        return fn({k: walk_replace(v, fn) for k, v in obj.items()})
    if isinstance(obj, list):
        return [walk_replace(x, fn) for x in obj]
    return obj


def corruptions(mql):
    out = []
    s = json.dumps(mql)
    for a, b in (("$gt", "$gte"), ("$gte", "$gt"), ("$lt", "$lte"), ("$lte", "$lt")):
        if f'"{a}"' in s:
            def f(d, a=a, b=b):
                return {(b if k == a else k): v for k, v in d.items()}
            out.append((f"op_{a}->{b}", walk_replace(copy.deepcopy(mql), f)))
    for i, st in enumerate(mql["pipeline"]):
        if "$sort" in st:
            m = copy.deepcopy(mql)
            m["pipeline"][i]["$sort"] = {k: -v for k, v in st["$sort"].items()}
            out.append(("sort_flip", m))
        if "$limit" in st:
            m = copy.deepcopy(mql)
            m["pipeline"][i]["$limit"] = st["$limit"] + 1
            out.append(("limit_plus1", m))
        if "$skip" in st:
            m = copy.deepcopy(mql)
            m["pipeline"][i]["$skip"] = st["$skip"] + 1
            out.append(("skip_plus1", m))
        if "$match" in st:
            m = copy.deepcopy(mql)
            del m["pipeline"][i]
            out.append(("drop_match", m))
        if "$project" in st and len([k for k in st["$project"] if k != "_id"]) > 1:
            m = copy.deepcopy(mql)
            k = [k for k in st["$project"] if k != "_id"][-1]
            del m["pipeline"][i]["$project"][k]
            out.append(("drop_projected_field", m))
    if '"$nin"' in s and "null" in s:
        def f(d):
            if "$nin" in d and isinstance(d["$nin"], list) and None in d["$nin"]:
                d = dict(d)
                d["$nin"] = [x for x in d["$nin"] if x is not None]
            return d
        out.append(("nin_null_guard_removed", walk_replace(copy.deepcopy(mql), f)))
    if '"$in"' in s:
        def f(d):
            if "$in" in d and isinstance(d["$in"], list) and len(d["$in"]) > 1 and not isinstance(d["$in"][0], str) or (
                    "$in" in d and isinstance(d["$in"], list) and len(d["$in"]) > 1 and isinstance(d["$in"][0], str)
                    and not d["$in"][0].startswith("$")):
                d = dict(d)
                d["$in"] = d["$in"][:-1]
            return d
        out.append(("in_drop_last", walk_replace(copy.deepcopy(mql), f)))
    if '"$regex"' in s:
        def f(d):
            if "$regex" in d and isinstance(d["$regex"], str):
                d = dict(d)
                d["$regex"] = d["$regex"].lstrip("^").rstrip("$")
            return d
        m = walk_replace(copy.deepcopy(mql), f)
        if m != mql:
            out.append(("regex_unanchored", m))
    return [(n, m) for n, m in out if m != mql]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()
    import spider1_kit as K
    latest = {}
    for r in C.read_jsonl(V.PROGRESS):
        latest[r["pair_id"]] = r
    ver = sorted(pid for pid, r in latest.items() if r["status"] == V.VERIFIED)
    rng = random.Random(args.seed)
    sample = rng.sample(ver, min(args.sample, len(ver)))
    pairs = {p["pair_id"]: p for p in C.read_jsonl(C.INTERIM / "pairs.jsonl")}
    schema = json.loads((C.INTERIM / "schema.json").read_text(encoding="utf-8"))["tables"]
    per = collections.defaultdict(lambda: [0, 0])
    missed = []
    for pid in sample:
        p, r = pairs[pid], latest[pid]
        ast, _, _ = S.parse_sql(p["sql"])
        an = S.analyze(ast).to_json()
        if an["time_relative"]:
            continue  # clock moved since verification; regenerated fixtures would differ
        sch = {t: schema[t] for t in an["columns"]}
        fx = F.adequate_set(ast, an, sch, int(pid[3:11], 16) % 100000, F.BASE_CLOCK)
        base_mql = json.loads(r["mql_text"]) if r.get("mql_text") else r["mql"]
        # baseline: the uncorrupted candidate must pass on these fixtures, or the test is void
        base_ok = all(F.compare((lambda d: K.docs_to_rows(d)[0] if d else None)(docs := V.run_mql(base_mql, v["tables"])),
                                K.docs_to_rows(docs)[1], v["columns"], v["rows"], v["ordered"],
                                v["blocks"] if v["ordered"] else None)[0] for v in fx["kept"])
        per["_baseline_uncorrupted_passes"][0] += 1
        per["_baseline_uncorrupted_passes"][1] += base_ok
        if not base_ok:
            missed.append({"pair_id": pid, "sql": p["sql"], "corruption": "BASELINE_FAILED"})
            continue
        for name, bad in corruptions(base_mql):
            try:
                ok = True
                for v in fx["kept"]:
                    docs = V.run_mql(bad, v["tables"])
                    pcols, rows = K.docs_to_rows(docs)
                    if not F.compare(pcols if docs else None, rows, v["columns"], v["rows"], v["ordered"],
                                     v["blocks"] if v["ordered"] else None)[0]:
                        ok = False
                        break
            except Exception:  # noqa: BLE001 - a corruption that errors is caught
                ok = False
            per[name][0] += 1
            per[name][1] += not ok
            if ok:
                missed.append({"pair_id": pid, "sql": p["sql"], "corruption": name})
    report = {"sample": len(sample), "seed": args.seed,
              "by_corruption": {k: {"tested": v[0], "caught": v[1]} for k, v in sorted(per.items())},
              "total": {"tested": sum(v[0] for k, v in per.items() if not k.startswith("_")),
                        "caught": sum(v[1] for k, v in per.items() if not k.startswith("_"))},
              "not_caught": missed}
    (V.PROGRESS.parent / "selftest.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "not_caught"}, indent=1))
    for m in missed[:25]:
        print("NOT CAUGHT", m["corruption"], "|", m["sql"])


if __name__ == "__main__":
    main()
