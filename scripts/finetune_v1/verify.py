"""Stage 2: generate MQL candidates and verify them by execution (resumable).

    python scripts/finetune_v1/verify.py [--limit N] [--workers 6] [--ids nk_a,nk_b]

For every stage-1 `eligible` pair:
  1. build an adequate fixture set (fixtures.adequate_set) and the expected result of the
     reference SQL on SQLite for each variant;
  2. generate a candidate MQL with the deterministic converter (sql2mql);
  3. validate it with the benchmark's read-only validator (eval_kit.validate_mql);
  4. execute it on MongoDB (Atlas, read-only user) against the same fixture rows supplied
     inline with `$documents` (no collection is read or written; `$lookup.from` is rewritten
     to an inline `$documents` sub-pipeline);
  5. compare with the benchmark comparator (spider1_kit.docs_to_rows + exec_match; ordered
     and tie-aware when the SQL has ORDER BY);
  6. sanity-check the verifier: the candidate with its first `$match` removed must FAIL on
     at least one variant whenever the SQL has a WHERE clause.
Status `fixture_equivalence_verified` requires 1-6. It certifies SQL<->MQL result equivalence
on discriminating synthetic data, NOT correctness against a source database (none exists).

Progress: data/interim/finetune_v1/verify/progress.jsonl, one record per pair, appended as
each finishes. A pair is skipped on re-run if a record with the same input hash exists.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nk_common as C  # noqa: E402
import fixtures as F  # noqa: E402
import sql2mql as M  # noqa: E402
import sqlsem as S  # noqa: E402

PROGRESS = C.INTERIM / "verify" / "progress.jsonl"
CODE_FILES = ("sqlsem.py", "sql2mql.py", "fixtures.py", "verify.py")
VERIFIED = "fixture_equivalence_verified"


def code_hash():
    return C.sha256_bytes(b"".join((C.HERE / f).read_bytes() for f in CODE_FILES))


def input_hash(pair, schema_sha, chash):
    return C.sha_obj({"q": pair["question"], "sql": pair["sql"], "schema": schema_sha, "code": chash})


# ---------------------------------------------------------------- Mongo execution
def inline_pipeline(mql, docs_by_coll):
    """Prefix `$documents` for the start collection and inline every `$lookup.from` /
    `$unionWith` target, so the pipeline runs as a database-level aggregate."""
    def rw_stage(st):
        st = json.loads(json.dumps(st))
        if "$lookup" in st:
            spec = st["$lookup"]
            frm = spec.pop("from", None)
            if frm is not None:
                sub = rw_pipe(spec.get("pipeline", []))
                spec["pipeline"] = [{"$documents": docs_by_coll[frm]}] + sub
            elif "pipeline" in spec:
                spec["pipeline"] = rw_pipe(spec["pipeline"])
        if "$unionWith" in st:
            u = st["$unionWith"]
            u = {"coll": u} if isinstance(u, str) else u
            st["$unionWith"] = {"pipeline": [{"$documents": docs_by_coll[u["coll"]]}] + rw_pipe(u.get("pipeline", []))}
        if "$facet" in st:
            st["$facet"] = {k: rw_pipe(v) for k, v in st["$facet"].items()}
        return st

    def rw_pipe(p):
        return [rw_stage(s) for s in p]

    return [{"$documents": docs_by_coll[mql["collection"]]}] + rw_pipe(mql["pipeline"])


_local = threading.local()


def mongo_db():
    if not hasattr(_local, "db"):
        import spider1_kit as K
        _local.client = K.mongo_client()
        _local.db = _local.client[K.ATLAS_DB]
    return _local.db


def run_mql(mql, tables):
    docs = {C.coll(t): rows for t, rows in tables.items()}
    pipe = inline_pipeline(mql, docs)
    return list(mongo_db().aggregate(pipe, maxTimeMS=30000))


def server_clock():
    d = list(mongo_db().aggregate([{"$documents": [{}]}, {"$project": {"n": "$$NOW"}}]))[0]["n"]
    return d.replace(tzinfo=None, microsecond=0)


# ---------------------------------------------------------------- one pair
def verify_pair(pair, schema):
    import eval_kit as E
    import spider1_kit as K

    t0 = time.time()
    rec = {"pair_id": pair["pair_id"], "converter": M.CONVERTER_VERSION,
           "fixture_version": F.FIXTURE_VERSION, "status": None, "reasons": []}
    ast, dialect, err = S.parse_sql(pair["sql"])
    an_obj = S.analyze(ast)
    an = an_obj.to_json()
    sch = {t: schema[t] for t in an["columns"]}
    try:
        mql = M.convert(ast, an_obj, sch)
    except M.Unsupported as e:
        rec.update(status="unverified", reasons=[f"converter_unsupported:{e}"])
        return rec
    if mql["pipeline"] and "$limit" in mql["pipeline"][-1] and mql["pipeline"][-1]["$limit"] <= 0:
        rec.update(status="unverified", reasons=["limit_zero"])
        return rec
    rec["mql"] = mql
    rec["mql_text"] = C.compact(mql)
    allowed = {C.coll(t) for t in an["columns"]}
    ok, why, _ = E.validate_mql(mql, allowed)
    rec["validator"] = {"ok": ok, "reason": why}
    if not ok:
        rec.update(status="rejected", reasons=[f"validator:{why}"])
        return rec
    seed_base = int(pair["pair_id"][3:11], 16) % 100000
    attempts = 2 if an["time_relative"] else 1
    for attempt in range(attempts):
        clock = server_clock() if an["time_relative"] else F.BASE_CLOCK
        fx = F.adequate_set(ast, an, sch, seed_base + attempt, clock)
        rec["fixtures"] = {"clock": clock.isoformat(sep=" "), "seeds_tried": fx["seeds_tried"],
                           "need": fx["need"], "covered": fx["covered"], "uncovered": fx["uncovered"],
                           "exempt_mutants": fx["exempt_mutants"], "invalid_variants": fx["invalid"],
                           "variants": [{"seed": v["seed"], "p_hit": v["p_hit"], "scale": v["scale"],
                                         "rows_per_table": {t: len(r) for t, r in v["tables"].items()},
                                         "expected_n_rows": len(v["rows"]),
                                         "matching_rows": v["matching_rows"],
                                         "expected_rows_sha256": C.sha_obj(v["rows"]),
                                         "ordered": v["ordered"]} for v in fx["kept"]]}
        if fx["uncovered"]:
            rec.update(status="unverified", reasons=[f"fixture_not_discriminating:{u}" for u in fx["uncovered"]][:6])
            return rec
        results = []
        for v in fx["kept"]:
            docs = run_mql(mql, v["tables"])
            pcols, rows = K.docs_to_rows(docs)
            match, reason = F.compare(pcols if docs else None, rows, v["columns"], v["rows"],
                                      v["ordered"], v["blocks"] if v["ordered"] else None)
            results.append({"seed": v["seed"], "match": match, "reason": reason, "n_pred": len(rows)})
        rec["execution"] = results
        if all(r["match"] for r in results):
            break
    if not all(r["match"] for r in results):
        rec.update(status="failed_execution",
                   reasons=[f"mismatch_on_seed_{r['seed']}:{r['reason']}" for r in results if not r["match"]][:3])
        rec["expected_preview"] = [list(map(str, r)) for r in fx["kept"][0]["rows"][:5]]
        return rec
    # verifier sanity: dropping the first $match must be caught
    last_join = max((i for i, s in enumerate(mql["pipeline"]) if "$lookup" in s or "$unwind" in s), default=-1)
    idx = next((i for i, s in enumerate(mql["pipeline"]) if "$match" in s and i > last_join), None)
    if ast.args.get("where") is not None and idx is not None:
        mut = {"collection": mql["collection"], "pipeline": mql["pipeline"][:idx] + mql["pipeline"][idx + 1:]}
        caught = False
        for v in fx["kept"]:
            mdocs = run_mql(mut, v["tables"])
            pcols, rows = K.docs_to_rows(mdocs)
            if not F.compare(pcols if mdocs else None, rows, v["columns"], v["rows"], v["ordered"],
                             v["blocks"] if v["ordered"] else None)[0]:
                caught = True
                break
        rec["mql_mutant_drop_match_caught"] = caught
        if not caught:
            rec.update(status="unverified", reasons=["verifier_sanity:drop_match_not_caught"])
            return rec
    rec["expected_rows_sha256"] = [v["expected_rows_sha256"] for v in rec["fixtures"]["variants"]]
    rec.update(status=VERIFIED, reasons=[], elapsed_s=round(time.time() - t0, 2))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ids")
    args = ap.parse_args()
    pairs = [p for p in C.read_jsonl(C.INTERIM / "pairs.jsonl") if p["stage1_class"] == "eligible"]
    if args.ids:
        want = set(args.ids.split(","))
        pairs = [p for p in pairs if p["pair_id"] in want]
    schema_doc = json.loads((C.INTERIM / "schema.json").read_text(encoding="utf-8"))
    schema, schema_sha = schema_doc["tables"], C.sha_obj(schema_doc)
    chash = code_hash()
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if PROGRESS.exists():
        for r in C.read_jsonl(PROGRESS):
            done[r["pair_id"]] = r
    todo = [p for p in pairs if done.get(p["pair_id"], {}).get("input_sha256") != input_hash(p, schema_sha, chash)]
    if args.limit:
        todo = todo[:args.limit]
    print(f"eligible {len(pairs)}, already done {len(pairs) - len(todo) if not args.limit else 'n/a'}, to run {len(todo)}")
    stats = {}
    t0 = time.time()

    def work(p):
        try:
            rec = verify_pair(p, schema)
        except Exception as e:  # noqa: BLE001 - recorded per pair, the run continues
            rec = {"pair_id": p["pair_id"], "status": "error", "reasons": [f"{type(e).__name__}: {str(e)[:300]}"],
                   "traceback": traceback.format_exc()[-1500:]}
        rec["input_sha256"] = input_hash(p, schema_sha, chash)
        rec["code_sha256"] = chash
        rec["verified_at"] = dt.datetime.now().isoformat(timespec="seconds")
        return rec

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, rec in enumerate(ex.map(work, todo), 1):
            C.append_jsonl(PROGRESS, rec)
            stats[rec["status"]] = stats.get(rec["status"], 0) + 1
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} {stats} {time.time() - t0:.0f}s", flush=True)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
