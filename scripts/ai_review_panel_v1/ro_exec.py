"""Read-only, logged query executor for the AI review panel v1 (plan §E.7).

DERIVED from outputs/sql2mql_v2/pilot_v3/run01/manual_human_review/ai_multiagent_review/tools/ro_exec.py
(same safety rules and output format). It is a copy rather than the unchanged file because the original
hard-codes run01 paths and writes its logs inside the frozen run01 directory. Changes: --work names the
agent's own log directory (must be inside outputs/ai_review_panel_v1/), --package names the batch file,
items are looked up by panel_id, and log ids are <TAG>-####.

Every call is appended to <work>/exec_log.jsonl. Cite the printed ids in `execution_evidence`.

Usage:
  python ro_exec.py --work DIR --tag T --package FILE item <panel_id>
  python ro_exec.py --work DIR --tag T sql <db_id> <file_or_->
  python ro_exec.py --work DIR --tag T mql <collection> <file_or_->     (file may hold {"collection","pipeline"})
  python ro_exec.py --work DIR --tag T sql_cx <db_id> <setup_file> <query_file>   (in-memory copy; disk never written)
  python ro_exec.py --work DIR --tag T mql_cx <file>                    ({"docs": [...], "pipeline": [...]} via $documents)
Options: --all prints all rows (default first 60).
Safety: SQLite opened mode=ro; MongoDB (local 127.0.0.1:27018, spider_train_v2) only via aggregate(); $out/$merge refused.
"""
import argparse, datetime, hashlib, json, os, sqlite3, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                         # codegen-capstone/
PANEL = REPO / "outputs" / "ai_review_panel_v1"
SPIDER_DB = REPO / "data" / "raw" / "spider1" / "spider_data" / "database"
MONGO_DB = "spider_train_v2"
WORK = None
TAG = None


def read_src(arg):
    if arg == "-":
        return sys.stdin.read()
    return Path(arg).read_text(encoding="utf-8")


def forbid_writes(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("$out", "$merge"):
                raise SystemExit("REFUSED: $out/$merge are not allowed")
            forbid_writes(v)
    elif isinstance(obj, list):
        for v in obj:
            forbid_writes(v)


def jsonable(v):
    try:
        from bson import ObjectId, Decimal128
        if isinstance(v, ObjectId):
            return {"$oid": str(v)}
        if isinstance(v, Decimal128):
            return {"$numberDecimal": str(v)}
    except Exception:
        pass
    if isinstance(v, bytes):
        return {"$bytes": v.hex()}
    if isinstance(v, (datetime.datetime, datetime.date)):
        return {"$date": v.isoformat()}
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def log(agent, kind, target, query, result, error=None):
    d = WORK
    d.mkdir(parents=True, exist_ok=True)
    p = d / "exec_log.jsonl"
    n = sum(1 for _ in open(p, encoding="utf-8")) + 1 if p.exists() else 1
    eid = f"{TAG}-{n:04d}"
    blob = json.dumps(result, sort_keys=True, ensure_ascii=False, default=str)
    rec = {"id": eid, "ts": datetime.datetime.now().isoformat(timespec="seconds"), "kind": kind,
           "target": target, "query": query, "error": error,
           "row_count": None if result is None else len(result.get("rows", [])),
           "columns": None if result is None else result.get("columns"),
           "result_sha256": hashlib.sha256(blob.encode()).hexdigest(),
           "rows_head": None if result is None else result.get("rows", [])[:20]}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return eid


def sql_rows(con, q):
    cur = con.execute(q)
    cols = [c[0] for c in cur.description] if cur.description else []
    rows = [list(r) for r in cur.fetchall()]
    return {"columns": cols, "rows": jsonable(rows)}


def mongo():
    import pymongo
    return pymongo.MongoClient("127.0.0.1", 27018, serverSelectionTimeoutMS=8000)[MONGO_DB]


def emit(eid, res, show_all, err=None):
    print(f"[{eid}]")
    if err:
        print("ERROR:", err)
        return
    rows = res["rows"]
    print("columns:", json.dumps(res.get("columns"), ensure_ascii=False))
    print("row_count:", len(rows))
    for r in (rows if show_all else rows[:60]):
        print(json.dumps(r, ensure_ascii=False, default=str))
    if not show_all and len(rows) > 60:
        print(f"... ({len(rows) - 60} more rows; use --all)")


def main():
    ap = argparse.ArgumentParser()
    global WORK, TAG
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--package")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("cmd", choices=["item", "sql", "mql", "sql_cx", "mql_cx"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    WORK = Path(a.work).resolve()
    if PANEL.resolve() not in WORK.parents:
        raise SystemExit(f"--work must be inside {PANEL}")
    TAG = a.tag
    a.agent = a.tag

    if a.cmd == "item":
        rid = a.args[0]
        for line in open(a.package, encoding="utf-8"):
            it = json.loads(line)
            if it["panel_id"] == rid:
                print(json.dumps(it, indent=1, ensure_ascii=False))
                return
        raise SystemExit(f"no item {rid}")

    res, err = None, None
    if a.cmd == "sql":
        db, q = a.args[0], read_src(a.args[1])
        path = SPIDER_DB / db / f"{db}.sqlite"
        try:
            con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            con.text_factory = lambda b: b.decode("utf-8", errors="replace")
            res = sql_rows(con, q)
        except Exception as e:
            err = repr(e)
        emit(log(a.agent, "sql", db, q, res, err), res, a.all, err)
    elif a.cmd == "sql_cx":
        db, setup, q = a.args[0], read_src(a.args[1]), read_src(a.args[2])
        path = SPIDER_DB / db / f"{db}.sqlite"
        try:
            src = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            mem = sqlite3.connect(":memory:")
            src.backup(mem)
            src.close()
            mem.text_factory = lambda b: b.decode("utf-8", errors="replace")
            mem.executescript(setup)
            res = sql_rows(mem, q)
        except Exception as e:
            err = repr(e)
        emit(log(a.agent, "sql_cx", db, {"setup": setup, "query": q}, res, err), res, a.all, err)
    elif a.cmd == "mql":
        coll, raw = a.args[0], json.loads(read_src(a.args[1]))
        if isinstance(raw, dict):
            coll, pipe = raw["collection"], raw["pipeline"]
        else:
            pipe = raw
        forbid_writes(pipe)
        try:
            docs = list(mongo()[coll].aggregate(pipe))
            cols = []
            for d in docs:
                for k in d:
                    if k not in cols:
                        cols.append(k)
            res = {"columns": cols, "rows": jsonable(docs)}
        except Exception as e:
            err = repr(e)
        emit(log(a.agent, "mql", coll, pipe, res, err), res, a.all, err)
    elif a.cmd == "mql_cx":
        spec = json.loads(read_src(a.args[0]))
        pipe = [{"$documents": spec["docs"]}] + spec["pipeline"]
        forbid_writes(pipe)
        try:
            docs = list(mongo().aggregate(pipe))
            res = {"columns": sorted({k for d in docs for k in d}), "rows": jsonable(docs)}
        except Exception as e:
            err = repr(e)
        emit(log(a.agent, "mql_cx", "$documents", pipe, res, err), res, a.all, err)


if __name__ == "__main__":
    main()
