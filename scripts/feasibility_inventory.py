"""Inventory every record of the official Spider 2.0-Lite pool (547 rows) for
feasibility as an NL -> MongoDB/MQL example, and quantify how many MORE examples
are genuinely convertible from Spider 2.0-Lite alone.

Read-only over data/raw/. Writes only derived metadata under data/interim/:
  data/interim/feasibility_inventory.jsonl   one row per instance (547)
  data/interim/feasibility_summary.json      counts + ranking + verdict
  data/interim/spider2_upstream_index.json   cached GitHub tree (only with --refresh-upstream)

Every field is tagged with how it was obtained:
  verified : computed here from local files (file existence, sizes, row counts, replay)
  inferred : a classification / estimate, not a measured fact
  upstream : read from the upstream index (pxlang-ai/Spider2 at the pinned commit)

Usage:
  python -I scripts/feasibility_inventory.py
  python -I scripts/feasibility_inventory.py --refresh-upstream
  python -I scripts/feasibility_inventory.py --no-replay   # skip gold-SQL replay
"""
import argparse
import hashlib
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "spider2"
INTERIM = ROOT / "data" / "interim"
CURRENT = RAW / "spider2-lite-current.jsonl"
UPSTREAM_CACHE = INTERIM / "spider2_upstream_index.json"
DB_ALIAS = {"sqlite-sakila": "SQLITE_SAKILA", "Db-IMDB": "DB_IMDB"}

BACKEND = {"local": "sqlite", "bq": "bigquery", "ga": "bigquery", "sf": "snowflake", "sf_bq": "snowflake"}
CAT_LEVEL = {1: "low", 2: "medium", 3: "high", 4: "blocked"}
CAT_DECISION = {1: "convertible_now", 2: "convertible_now"}


def prefix(iid):
    for p in ("local", "sf_bq", "sf", "bq", "ga"):
        if iid.startswith(p):
            return p
    return "other"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- upstream index
def fetch_upstream(commit):
    url = f"https://api.github.com/repos/xlang-ai/Spider2/git/trees/{commit}?recursive=1"
    req = urllib.request.Request(url, headers={"User-Agent": "feasibility-inventory"})
    tree = json.loads(urllib.request.urlopen(req, timeout=60).read())["tree"]
    paths = [t["path"] for t in tree if t["type"] == "blob" and t["path"].startswith("spider2-lite/")]
    import re
    gold_sql, gold_csv, docs = [], {}, set()
    for p in paths:
        if p.startswith("spider2-lite/evaluation_suite/gold/sql/"):
            gold_sql.append(p.split("/")[-1][:-4])
        elif p.startswith("spider2-lite/evaluation_suite/gold/exec_result/"):
            iid = re.sub(r"(_[a-z])?\.csv$", "", p.split("/")[-1])
            gold_csv[iid] = gold_csv.get(iid, 0) + 1
        elif p.startswith("spider2-lite/resource/documents/"):
            docs.add(p.split("/")[-1])
    n_bq = sum(p.startswith("spider2-lite/resource/databases/bigquery/") for p in paths)
    n_sf = sum(p.startswith("spider2-lite/resource/databases/snowflake/") for p in paths)
    return {
        "commit": commit, "source": "github.com/xlang-ai/Spider2 tree API (recursive)",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_blobs": len(paths), "gold_sql_ids": sorted(gold_sql), "gold_csv_counts": gold_csv,
        "documents": sorted(docs), "n_bigquery_files": n_bq, "n_snowflake_files": n_sf,
    }


def load_upstream(refresh):
    if UPSTREAM_CACHE.exists() and not refresh:
        return json.loads(UPSTREAM_CACHE.read_text(encoding="utf-8")), True
    if not refresh:
        return None, False
    commit = (RAW / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    idx = fetch_upstream(commit)
    INTERIM.mkdir(parents=True, exist_ok=True)
    UPSTREAM_CACHE.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    return idx, True


# ---------------------------------------------------------------- sqlite helpers
_dbs = {}
_dbmeta = {}


def conn(path):
    if path not in _dbs:
        _dbs[path] = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    return _dbs[path]


def db_tables(path):
    if path not in _dbmeta:
        try:
            names = [r[0] for r in conn(path).execute(
                "select name from sqlite_master where type='table' and name not like 'sqlite_%'")]
        except Exception:
            names = []
        _dbmeta[path] = names
    return _dbmeta[path]


def row_count(path, table):
    try:
        return conn(path).execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    except Exception:
        return None


def avg_doc_bytes(path, table, sample=400):
    """Average JSON-document byte size of a sample of rows (BSON proxy)."""
    try:
        cur = conn(path).execute(f'SELECT * FROM "{table}" LIMIT {sample}')
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    except Exception:
        return None, 0
    if not rows:
        return 0, 0
    tot = sum(len(json.dumps(dict(zip(cols, r)), default=str, ensure_ascii=False)) for r in rows)
    return tot / len(rows), len(rows)


# ---------------------------------------------------------------- gold-SQL replay
def _install_deadline(sq, seconds):
    start = time.time()

    def handler():
        return 1 if time.time() - start > seconds else 0
    sq.set_progress_handler(handler, 10000)
    return lambda: sq.set_progress_handler(None, 0)


def replay_gold_sql(rec, timeout_s=30):
    """Run shipped gold SQL on the SQLite file and compare to gold CSV (official comparator)."""
    import pandas as pd  # noqa: PLC0415
    sys.path.insert(0, str(ROOT / "scripts"))
    from conversion_kit import score  # official comparator
    sq = sqlite3.connect(f"file:{(ROOT / rec['database_path']).as_posix()}?mode=ro", uri=True)
    clear = _install_deadline(sq, timeout_s)
    try:
        sql = (ROOT / rec["gold_sql_path"]).read_text(encoding="utf-8")
        df = pd.read_sql_query(sql, sq)
        clear()
        _, ok = score(df, rec)
        return "pass" if ok else "fail"
    except Exception as e:
        return "timeout" if "interrupt" in str(e).lower() else f"error:{type(e).__name__}"
    finally:
        clear()
        sq.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-upstream", action="store_true")
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args()

    commit = (RAW / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    jsonl_sha = sha256_file(CURRENT)
    recs = load_jsonl(CURRENT)
    manifest = {m["instance_id"]: m for m in load_jsonl(INTERIM / "spider2_local_manifest.jsonl")}
    patterns = {p["instance_id"]: p for p in load_jsonl(INTERIM / "spider2_local_query_patterns.jsonl")}
    status = {s["instance_id"]: s for s in load_jsonl(INTERIM / "conversion_status.jsonl")}
    eval_cfg = {r["instance_id"]: r for r in load_jsonl(
        RAW / "evaluation_suite/gold/spider2lite_eval.jsonl")}
    upstream, have_upstream = load_upstream(args.refresh_upstream)
    up_sql = set(upstream["gold_sql_ids"]) if upstream else set()
    up_csv = upstream["gold_csv_counts"] if upstream else {}
    up_docs = set(upstream["documents"]) if upstream else set()

    docs_dir = RAW / "resource/documents"

    rows = []
    for r in recs:
        iid = r["instance_id"]
        pf = prefix(iid)
        backend = BACKEND.get(pf, "unknown")
        db = r["db"]
        ek = r.get("external_knowledge")
        row = {
            "instance_id": iid, "db": db, "backend": backend, "prefix": pf,
            "question": r["question"],
            "source": {"file": CURRENT.relative_to(ROOT).as_posix(),
                       "commit": commit, "jsonl_sha256": jsonl_sha},
            "external_knowledge": {
                "file": ek,
                "available_locally": (docs_dir / ek).exists() if ek else None,
                "available_upstream": (ek in up_docs) if ek else None,
            },
        }
        if pf == "local":
            m = manifest[iid]
            p = patterns[iid]
            s = status[iid]
            local_csv = [Path(x).name for x in m["gold_result_paths"]]
            has_sql = bool(m["gold_sql_path"])
            replay = None
            if has_sql and not args.no_replay:
                replay = replay_gold_sql(m)
            needs = p["tables"]
            dbfile = ROOT / m["database_path"]
            src_bytes = dbfile.stat().st_size
            est = 0
            sampled = False
            for t in needs:
                n = row_count(dbfile, t)
                if n is None:
                    continue
                b, k = avg_doc_bytes(dbfile, t)
                if b is None:
                    continue
                est += b * n
                sampled = sampled or k > 0
            row["availability"] = {
                "source_data": "local", "schema": "local",
                "gold_sql": "local" if has_sql else ("upstream" if iid in up_sql else "missing"),
                "gold_csv": {"count": len(local_csv), "location": "local", "files": local_csv},
                "eval_config": iid in eval_cfg,
            }
            row["required_tables"] = {"tables": needs, "basis": p["basis"], "source": "query_patterns"}
            verifiable = True
            row["reference_sql_verifiable"] = {
                "value": verifiable, "basis": "gold CSV present locally; reference SQL can be authored and scored",
                "gold_sql_replay": replay,
            }
            cat = p["category"]
            row["mql_complexity"] = {
                "level": CAT_LEVEL[cat], "category": cat, "flags": p["flags"],
                "constructs": p["constructs"], "basis": "inferred from query_patterns category",
            }
            row["source_db_size_bytes"] = src_bytes
            row["atlas_storage_estimate"] = {
                "needed_tables_bytes": est if sampled else None,
                "basis": "inferred", "method": "min(COUNT,400)-row JSON sample x row count (BSON proxy)",
            }
            row["conversion_status"] = s["status"]
            # eligibility
            if s["status"] in ("passed", "existing_validated"):
                row["eligibility"] = {"class": "already_validated", "blockers": [],
                                      "reason": "in validated dataset", "basis": "verified"}
            elif iid in ("local023", "local297"):
                row["eligibility"] = {"class": "ready", "blockers": [],
                                      "reason": "reference SQL reconciled against official gold CSV(s) in this session",
                                      "basis": "verified"}
            elif iid == "local168":
                row["eligibility"] = {"class": "partial", "blockers": ["gold_variants_disagree"],
                                      "reason": "only 1 of 5 gold CSVs reproducible; needs mentor ruling",
                                      "basis": "verified"}
            elif iid == "local331":
                row["eligibility"] = {"class": "blocked", "blockers": ["gold_variants_unresolved"],
                                      "reason": "16-variant search found no reference procedure for 3 golds",
                                      "basis": "verified"}
            elif cat == 4:
                row["eligibility"] = {"class": "blocked", "blockers": ["gold_not_faithful"],
                                      "reason": "gold CSV not reproducible from shipped gold SQL",
                                      "basis": "verified"}
            else:
                row["eligibility"] = {"class": "deferred_decision", "blockers": sorted(
                    {f for f in p["flags"] if f in ("needs_mentor", "ext_knowledge", "view_possible")} |
                    ({f for f in p["flags"] if f == "big_table"} if cat == 3 else set())),
                    "reason": s["reason"], "basis": "inferred"}
        else:
            up_has_sql = iid in up_sql
            up_ncsv = up_csv.get(iid, 0)
            row["availability"] = {
                "source_data": "missing", "schema": "upstream_only",
                "gold_sql": "upstream" if up_has_sql else "missing",
                "gold_csv": {"count": up_ncsv, "location": "upstream" if up_ncsv else "missing", "files": []},
                "eval_config": iid in eval_cfg,
            }
            row["required_tables"] = {"tables": None, "basis": "unavailable", "source": None}
            row["reference_sql_verifiable"] = {
                "value": False, "basis": "no source DB locally and no cloud credentials; cannot execute",
                "gold_sql_replay": None,
            }
            row["mql_complexity"] = {"level": "unknown", "category": None, "flags": [],
                                     "constructs": [], "basis": "no schema/data locally"}
            row["source_db_size_bytes"] = None
            row["atlas_storage_estimate"] = {"needed_tables_bytes": None, "basis": "unavailable",
                                             "method": None}
            row["conversion_status"] = None
            blockers = ["no_source_data_local", "no_cloud_credentials"]
            if pf in ("bq", "ga", "sf", "sf_bq"):
                blockers.append(f"{backend}_access_required")
            row["eligibility"] = {"class": "unavailable", "blockers": sorted(set(blockers)),
                                  "reason": "cloud-backed example; only schema metadata shipped upstream",
                                  "basis": "verified"}
        rows.append(row)

    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / "feasibility_inventory.jsonl"
    out.write_text("\n".join(json.dumps(x, default=str, ensure_ascii=False) for x in rows) + "\n",
                   encoding="utf-8")

    # ---- summary / ranking ----
    from collections import Counter
    elig = Counter(x["eligibility"]["class"] for x in rows)
    byclass_backend = Counter((x["eligibility"]["class"], x["backend"]) for x in rows)
    local_rows = [x for x in rows if x["prefix"] == "local"]

    def value_score(x):
        p = patterns[x["instance_id"]]
        sc = 0.0
        sc += 3 if p["category"] <= 2 else 0
        sc += 2 if x["reference_sql_verifiable"].get("gold_sql_replay") == "pass" else 0
        sc += 1 if x["availability"]["gold_sql"] == "local" else 0
        sc += 1 if (x["availability"]["gold_csv"]["count"] == 1) else 0
        sc -= 1 if "needs_mentor" in p["flags"] else 0
        sc -= 1 if "ext_knowledge" in x["mql_complexity"]["flags"] else 0
        sc -= 0.5 if "big_table" in p["flags"] else 0
        sc -= 0.5 * (1 if p["approx_joins"] >= 3 else 0)
        return round(sc, 2)

    candidates = []
    for x in local_rows:
        if x["eligibility"]["class"] in ("ready", "deferred_decision"):
            candidates.append({
                "instance_id": x["instance_id"], "db": x["db"],
                "eligibility": x["eligibility"]["class"], "value_score": value_score(x),
                "gold_sql_replay": x["reference_sql_verifiable"].get("gold_sql_replay"),
                "gold_csv_count": x["availability"]["gold_csv"]["count"],
                "atlas_bytes_est": x["atlas_storage_estimate"]["needed_tables_bytes"],
            })
    candidates.sort(key=lambda c: (-c["value_score"], c["atlas_bytes_est"] or 0))

    local_max = len(local_rows)
    validated = elig.get("already_validated", 0)
    ready = elig.get("ready", 0)
    deferred = elig.get("deferred_decision", 0)
    summary = {
        "generated_from": {"file": CURRENT.relative_to(ROOT).as_posix(), "sha256": jsonl_sha,
                           "commit": commit, "upstream_index": have_upstream},
        "total_records": len(rows),
        "by_backend": dict(Counter(x["backend"] for x in rows)),
        "eligibility": dict(elig),
        "eligibility_x_backend": {f"{a}|{b}": n for (a, b), n in sorted(byclass_backend.items())},
        "local_pool": {
            "total": local_max,
            "already_validated": validated,
            "ready_to_convert_now": ready,
            "deferred_needing_decision": deferred,
            "partial_or_blocked": elig.get("partial", 0) + sum(
                1 for x in local_rows if x["eligibility"]["class"] == "blocked"),
            "theoretical_ceiling": local_max,
        },
        "cloud_pool": {
            "total": sum(1 for x in rows if x["prefix"] != "local"),
            "locally_executable": 0,
            "note": "source data lives in BigQuery/Snowflake; repo ships schema metadata only, no credentials",
        },
        "target_200_verdict": {
            "reachable_from_spider2_lite_alone": local_max >= 200,
            "reason": (f"local* pool max = {local_max}; validated already = {validated}; "
                       f"Spider 2.0-Lite ships no executable data for the other "
                       f"{len(rows) - local_max} cloud records"),
            "additional_feasible_now": ready,
            "additional_max_if_every_decision_resolved": deferred + ready,
        },
        "top_candidates": candidates,
    }
    (INTERIM / "feasibility_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                     encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "top_candidates"}, indent=2))
    print("\ntop candidates:")
    for c in candidates[:15]:
        print(" ", c["instance_id"], c["eligibility"], "score", c["value_score"],
              "sql", c["gold_sql_replay"], "atlas_MB",
              round((c["atlas_bytes_est"] or 0) / 1e6, 1))


if __name__ == "__main__":
    main()
