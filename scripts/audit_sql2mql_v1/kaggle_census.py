"""Independent multi-fixture census of every Kaggle-derived ft-v1 label (train + dev_internal).

    python -B scripts/audit_sql2mql_v1/kaggle_census.py [--limit N] [--workers 6]

The label under test is the stored `target_mql` (not a fresh conversion). Gold = reference
SQL on SQLite (sqlsem.to_sqlite). Families from audit_fixtures (independent of fixtures.py).
Evidence label: `audit_multi_fixture` -- synthetic data on the INFERRED schema; this is not a
source-database check (none exists for these examples). Resumable:
outputs/audit_sql2mql_v1/kaggle_census.jsonl.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json

import audit_fixtures as AF
import common as K

OUT = K.OUT / "kaggle_census.jsonl"
FT = K.ROOT / "data/finetune/ft_v1"
MISSING_FAMS = ("rand0", "rand1", "rand2", "rand3")


def labels():
    out = []
    for f in ("train.jsonl", "dev_internal.jsonl"):
        for r in K.read_jsonl(FT / f):
            if r["source_dataset"].startswith("kaggle"):
                out.append(r)
    return out


def verdict(vs, min_pass=3):
    """fail if any family fails or errors on MongoDB; pass if none fails and >= min_pass pass."""
    if any(v in ("fail", "mongo_error") for v in vs):
        return "fail"
    return "pass" if sum(v == "pass" for v in vs) >= min_pass else "inconclusive"


def audit_one(r, schema):
    sql, mql = r["reference"]["sql"], r["target_mql"]
    ast, _, _ = K.S.parse_sql(sql)
    an = K.S.analyze(ast)
    sch = {t: schema[t] for t in an.columns}
    rec = {"id": r["id"], "split": r["split"], "sql": sql, "features": sorted(an.features),
           "time_relative": an.time_relative}
    try:
        again = K.M.convert(ast, an, sch)
        rec["reconvert_identical"] = json.dumps(again) == json.dumps(mql)
    except K.M.Unsupported as e:
        rec["reconvert_identical"] = f"unsupported:{e}"
    clock = K.server_clock()  # per label: $$NOW must stay within the +/-90 s boundary jitter
    rec["clock"] = clock.isoformat(sep=" ")
    seed = int(hashlib.sha256(r["id"].encode()).hexdigest()[:8], 16)
    fam = AF.families(ast, an, sch, clock, seed)
    res = {}
    for name, tables in fam.items():
        res[name] = AF.compare_family(ast, an, mql, tables, sch, clock)
    for name in MISSING_FAMS:
        res[name + "@missing"] = AF.compare_family(ast, an, mql, fam[name], sch, clock, missing=True)
    rec["families"] = {k: v[0] for k, v in res.items()}
    rec["details"] = {k: v[1] for k, v in res.items() if v[0] != "pass"}
    rec["n_rows"] = {k: sum(len(x) for x in v.values()) for k, v in fam.items()}
    expl = [v for k, v in rec["families"].items() if "@" not in k]
    miss = [v for k, v in rec["families"].items() if "@" in k]
    rec["verdict_explicit_null"] = verdict(expl)
    rec["verdict_missing_field"] = verdict(miss, min_pass=2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    schema = json.loads((K.ROOT / "data/interim/finetune_v1/schema.json").read_text(encoding="utf-8"))["tables"]
    done = {x["id"] for x in K.read_jsonl(OUT)}
    todo = [r for r in labels() if r["id"] not in done][: a.limit]
    print("labels", len(labels()), "todo", len(todo))

    def work(r):
        try:
            return audit_one(r, schema)
        except Exception as e:  # noqa: BLE001
            return {"id": r["id"], "verdict_explicit_null": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}

    n = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(work, todo):
            K.append_jsonl(OUT, rec)
            n += 1
            if n % 100 == 0:
                print(n, flush=True)
    recs = K.read_jsonl(OUT)
    import collections
    print(collections.Counter(x["verdict_explicit_null"] for x in recs))
    print(collections.Counter(x.get("verdict_missing_field") for x in recs))


if __name__ == "__main__":
    main()
