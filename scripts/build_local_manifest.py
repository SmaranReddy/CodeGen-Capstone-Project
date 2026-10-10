"""Build data/interim/spider2_local_manifest.jsonl for the local* Spider 2.0-Lite examples.

Read-only over data/raw/. No conversion. Usage: python -I scripts/build_local_manifest.py
"""
import json, re, sqlite3, hashlib
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw" / "spider2"
OUT = PROJ / "data" / "interim" / "spider2_local_manifest.jsonl"
DB_DIR_ALIAS = {"sqlite-sakila": "SQLITE_SAKILA", "Db-IMDB": "DB_IMDB"}  # resource folder names

def rel(p): return p.relative_to(PROJ).as_posix()

recs = [json.loads(l) for l in (RAW / "spider2-lite-current.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
local = [r for r in recs if r["instance_id"].startswith("local")]
localmap = {}
for l in (RAW / "resource/databases/spider2-localdb/local-map.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip(): localmap.update(json.loads(l))
gold_eval = {r["instance_id"]: r for r in map(json.loads, (RAW / "evaluation_suite/gold/spider2lite_eval.jsonl").read_text(encoding="utf-8").splitlines()) if r}
docs_dir = RAW / "resource/documents"

db_cache = {}
def db_info(dbfile):
    if dbfile not in db_cache:
        if not dbfile.exists(): db_cache[dbfile] = None
        else:
            c = sqlite3.connect(f"file:{dbfile.as_posix()}?mode=ro", uri=True)
            tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
            db_cache[dbfile] = tabs; c.close()
    return db_cache[dbfile]

rows = []
for r in local:
    i, db = r["instance_id"], r["db"]
    dbfile = RAW / "resource/databases/spider2-localdb" / f"{db}.sqlite"
    schema_dir = RAW / "resource/databases/sqlite" / DB_DIR_ALIAS.get(db, db)
    ddl = schema_dir / "DDL.csv"
    gold_results = sorted((RAW / "evaluation_suite/gold/exec_result").glob(f"{i}.csv")) + sorted((RAW / "evaluation_suite/gold/exec_result").glob(f"{i}_*.csv"))
    gsql = RAW / "evaluation_suite/gold/sql" / f"{i}.sql"
    ek = r["external_knowledge"]
    ekp = docs_dir / ek if ek else None
    tabs = db_info(dbfile)
    problems = []
    if tabs is None: problems.append("sqlite_missing")
    elif not tabs: problems.append("sqlite_no_tables")
    if not ddl.exists(): problems.append("ddl_missing")
    if not gold_results: problems.append("gold_result_missing")
    if ekp and not ekp.exists(): problems.append("external_knowledge_doc_missing")
    if localmap.get(i) != db: problems.append(f"local_map_mismatch({localmap.get(i)})")
    if i not in gold_eval: problems.append("gold_eval_config_missing")
    rows.append({
        "instance_id": i, "db": db, "question": r["question"], "external_knowledge": ek,
        "external_knowledge_path": rel(ekp) if ekp and ekp.exists() else None,
        "database_path": rel(dbfile) if tabs is not None else None,
        "schema_path": rel(ddl) if ddl.exists() else None,
        "schema_dir": rel(schema_dir) if schema_dir.exists() else None,
        "gold_result_paths": [rel(p) for p in gold_results],
        "gold_sql_path": rel(gsql) if gsql.exists() else None,
        "gold_eval_config": gold_eval.get(i),
        "n_tables_in_sqlite": len(tabs) if tabs else 0,
        "executable_locally": not problems,
        "problems": problems,
    })
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
print("wrote", rel(OUT), len(rows))
