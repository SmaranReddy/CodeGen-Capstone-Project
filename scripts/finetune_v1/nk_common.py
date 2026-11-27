"""Shared paths, ids and helpers for the fine-tuning dataset expansion (scripts/finetune_v1).

Source: Kaggle "NL2SQL_Query_Dataset" (sureshmuthusamy001p/nl2sql-query-dataset, v1,
CC BY-NC-SA 4.0). Only (Prompt, Query) pairs: no schema, no database, no db identifiers.
Everything produced here is versioned under `ft-v1` and never touches the 455-example
benchmark, its splits, predictions or reports.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "scripts/spider1", ROOT / "scripts/evaluation", ROOT / "scripts/rag"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

BUILD_VERSION = "ft-v1"
SOURCE_ID = "kaggle_nl2sql_query_dataset_v1"
SOURCE_INFO = {
    "dataset_id": SOURCE_ID,
    "title": "NL2SQL_Query_Dataset",
    "kaggle_ref": "sureshmuthusamy001p/nl2sql-query-dataset",
    "url": "https://www.kaggle.com/datasets/sureshmuthusamy001p/nl2sql-query-dataset",
    "kaggle_version": 1,
    "last_updated": "2023-12-22T16:10:30.71Z",
    "license": "CC BY-NC-SA 4.0",
    "license_source": "Kaggle API /datasets/view licenseName field, fetched 2026-10-24",
    "provenance_statement": "none published (description: 14,815 prompt/SQL pairs, SELECT-focused)",
}
# pseudo database id of the synthetic fixtures; collections are `<DB_ID>__<table>` so they can
# never collide with a Spider collection name (`<spider_db>__<table>`)
DB_ID = "nl2sqlkaggle"

RAW_DIR = ROOT / "data/raw/nl2sql_kaggle_v1"
RAW_CSV = RAW_DIR / "NL2SQL_Query_Dataset.csv"
RAW_ZIP = RAW_DIR / "archive.zip"
RAW_SHA256 = {
    "NL2SQL_Query_Dataset.csv": "23d32e8962fd2a6f3678b4adce8a23c76616bc926513f49cdec45b279bf75004",
    "archive.zip": "6e1fe943f43bfd6a013c3f1cca95c557aeed70cd2670df33828b9571d6c3e2eb",
}
INTERIM = ROOT / "data/interim/finetune_v1"
OUT_DIR = ROOT / "data/finetune/ft_v1"

BENCHMARK = ROOT / "data/processed/spider1_nl_mql_expanded.jsonl"
SPLIT_DIR = ROOT / "data/splits_expanded"
CORPUS = ROOT / "data/rag/corpus_v1/corpus.jsonl"
SPIDER_DIR = ROOT / "data/raw/spider1/spider_data"


def coll(table):
    return f"{DB_ID}__{table}"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def sha_obj(obj):
    return sha256_bytes(canon(obj).encode("utf-8"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# NOTE: JSON is written WITHOUT sort_keys. MQL key order is semantic ($sort priority, output
# column order); verification records written before 2026-10-24 17:00 have sorted keys, so
# build_ft.py never trusts a stored MQL dict's key order (see target_for_* there).
def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()


def norm_ws(s):
    return re.sub(r"\s+", " ", s.strip())


def loose(s):
    """lowercase alphanumeric tokens only (same normalisation as scripts/rag/leakage.py)."""
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))
