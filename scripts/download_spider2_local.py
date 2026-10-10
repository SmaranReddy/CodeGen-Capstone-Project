"""Download ONLY the Spider 2.0-Lite resources needed for the local* (SQLite) examples.

Source: github.com/xlang-ai/Spider2 pinned to a commit. Does not touch existing files in
data/raw/spider2 other than creating new ones; refuses to overwrite.
Usage: python -I scripts/download_spider2_local.py <commit_sha>
"""
import json, re, sys, urllib.request
from pathlib import Path

SHA = sys.argv[1]
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "spider2"
BASE = f"https://raw.githubusercontent.com/xlang-ai/Spider2/{SHA}/spider2-lite/"
API = f"https://api.github.com/repos/xlang-ai/Spider2/git/trees/{SHA}?recursive=1"

def get(url):
    with urllib.request.urlopen(url) as r:
        return r.read()

def save(rel_remote, rel_local=None):
    dest = ROOT / (rel_local or rel_remote)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(get(BASE + rel_remote))
    return dest

tree = json.loads(get(API))["tree"]
paths = [t["path"][len("spider2-lite/"):] for t in tree if t["type"] == "blob" and t["path"].startswith("spider2-lite/")]

cur = save("spider2-lite.jsonl", "spider2-lite-current.jsonl")
recs = [json.loads(l) for l in cur.read_text(encoding="utf-8").splitlines() if l.strip()]
local = [r for r in recs if r["instance_id"].startswith("local")]
ids = {r["instance_id"] for r in local}
print("local examples:", len(local))

# Official resource folder names differ from `db` for two dbs.
DB_DIR_ALIAS = {"sqlite-sakila": "SQLITE_SAKILA", "Db-IMDB": "DB_IMDB"}
dbs = {DB_DIR_ALIAS.get(r["db"], r["db"]) for r in local}
want = []
for p in paths:
    parts = p.split("/")
    if p.startswith("resource/databases/sqlite/") and parts[3] in dbs:
        want.append(p)
    elif p.startswith("evaluation_suite/gold/sql/") and parts[-1][:-4] in ids:
        want.append(p)
    elif p.startswith("evaluation_suite/gold/exec_result/") and re.sub(r"(_[a-z])?\.csv$", "", parts[-1]) in ids:
        want.append(p)
docs = {r["external_knowledge"] for r in local if r["external_knowledge"]}
want += [p for p in paths if p.startswith("resource/documents/") and p.split("/")[-1] in docs]
want += ["evaluation_suite/gold/spider2lite_eval.jsonl", "evaluation_suite/evaluate.py", "evaluation_suite/evaluate_utils.py"]
for i, p in enumerate(want):
    save(p)
print("files fetched/present:", len(want))
