"""Shared plumbing for sql2mql-v2 (scripts/finetune_v2). NO SQL or MQL semantics live here.

Both the converter (sql2mql_v2.py, schema_profile.py) and the verifier (verifier.py,
comparator.py, fixtures.py, mongo_import.py) may import this module; they may not import each
other (enforced by tests/test_finetune_v2.py).

Local MongoDB only: `local_db()` connects to 127.0.0.1:27018 (the pinned mongo:8.0.34 Docker
container) and refuses any other host. It never reads MONGODB_URI or .env, so Atlas cannot be
reached through this package.
"""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPIDER = ROOT / "data/raw/spider1/spider_data"
TRAIN = ROOT / "data/processed/spider_sql_v1/train.jsonl"
INTERIM = ROOT / "data/interim/finetune_v2"
OUT = ROOT / "outputs/sql2mql_v2"
FREEZE = INTERIM / "pilot_freeze.json"

LOCAL_HOST, LOCAL_PORT = "127.0.0.1", 27018
MONGO_IMAGE = "mongo:8.0.34"
MONGO_DIGEST = "sha256:b9924c330a8bbc5800fe9c69f04a84e221eca7b28fd049a0354147d9908fa142"
REAL_DB = "spider_train_v2"      # local import of Spider-train databases (collections <db>__<table>)
FIXTURE_DB = "sql2mql_v2_fixtures"
TIMEOUT_MS = 30000


def local_client(host=LOCAL_HOST, port=LOCAL_PORT):
    if (host, port) != (LOCAL_HOST, LOCAL_PORT):
        raise RuntimeError(f"refusing MongoDB connection to {host}:{port}: only "
                           f"{LOCAL_HOST}:{LOCAL_PORT} (local {MONGO_IMAGE}) is allowed")
    from pymongo import MongoClient
    return MongoClient(host=host, port=port, directConnection=True,
                       serverSelectionTimeoutMS=5000, uuidRepresentation="standard")


def check_server(client):
    v = client.server_info()["version"]
    if v != MONGO_IMAGE.split(":")[1]:
        raise RuntimeError(f"local MongoDB is {v}, expected {MONGO_IMAGE}")
    return v


def coll(db_id, table):
    return f"{db_id}__{table}"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


def canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def compact(obj):
    """Key order kept (MQL key order is meaningful: never sort_keys an MQL object)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(p, recs):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in recs),
                 encoding="utf-8")


def sqlite_path(db_id):
    return SPIDER / "database" / db_id / f"{db_id}.sqlite"


def sqlite_ro(path):
    import sqlite3
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    # invalid UTF-8 survives as lone surrogates, which `has_bad_text` detects (never silently
    # replaced); BLOBs still arrive as bytes
    con.text_factory = lambda b: b.decode("utf-8", errors="surrogateescape")
    return con


def has_bad_text(v):
    return isinstance(v, str) and any(0xD800 <= ord(ch) <= 0xDFFF for ch in v)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, (c - h) / d), 4), round(min(1.0, (c + h) / d), 4))
