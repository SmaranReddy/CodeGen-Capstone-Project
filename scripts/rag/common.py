"""Shared paths and loaders for the Week 3 RAG pipeline (scripts/rag)."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/spider1"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

import spider1_kit as K  # noqa: E402,F401

DATASET = ROOT / "data/processed/spider1_nl_mql_expanded.jsonl"
SPLIT_DIR = ROOT / "data/splits_expanded"
CORPUS_DIR = ROOT / "data/rag/corpus_v1"
GUIDES_SRC = ROOT / "data/rag/sources/mongodb_guides.yaml"
COLLECTIONS = K.INTERIM / "atlas_collections.json"
BASELINE_DIR = ROOT / "outputs/evaluation_spider1_expanded"
# preference order for picking ONE verified prediction per train item (strongest model first)
EXAMPLE_SOURCES = ("gpt-oss-120b", "qwen2.5-coder-7b", "qwen2.5-coder-3b")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def canon(obj):
    """Canonical JSON (sorted keys, compact) used for content hashes."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def compact(obj):
    """Compact JSON, key order kept (how MQL is shown in prompts)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def load_collections():
    return json.loads(COLLECTIONS.read_text(encoding="utf-8"))


def load_split(split_dir=SPLIT_DIR):
    m = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    db_part = {db: v["partition"] for db, v in m["database_assignments"].items()}
    id_part = {e["id"]: e["partition"] for e in m["examples"]}
    return db_part, id_part


def load_corpus(corpus_dir=CORPUS_DIR):
    return K.read_jsonl(corpus_dir / "corpus.jsonl")
