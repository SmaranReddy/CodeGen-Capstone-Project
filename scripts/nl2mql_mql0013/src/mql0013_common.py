"""MQL-0013 shared constants and paths: one controlled execution comparison, R0 (BM25 hybrid) vs R3 (RRF(BM25, dense)
+ Jaccard), each through the MQL-0010 D0V pipeline (first pass + validator + one schema retry), on Spider test_full
(2,147 questions, 40 databases disjoint from every corpus / evaluation database used so far).

Nothing here touches a database, the corpus, or any frozen artifact.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]

EXPERIMENT = "MQL-0013"
PACKAGE = "nl2mql_mql0013_test_full"
OUT = ROOT / "outputs/nl2mql_mql0013"
WORK = OUT / "work"
REG_DIR = OUT / "registration"
REGISTRATION = REG_DIR / "mql0013_registration.json"
CONTRACT_DIR = OUT / "scoring_contract"
CONTRACT = CONTRACT_DIR / "atlas_mapping.json"
PKG_OUT = ROOT / "kaggle" / PACKAGE
ZIP_OUT = ROOT / "kaggle" / f"{PACKAGE}_bundle.zip"

TEST_FULL = ROOT / "data/processed/spider_sql_v1/test_full.jsonl"
SPIDER_DATA = ROOT / "data/raw/spider1/spider_data"
TEST_DB_DIR = SPIDER_DATA / "test_database"
TEST_TABLES = SPIDER_DATA / "test_tables.json"
TABLES = SPIDER_DATA / "tables.json"
CORPUS = ROOT / "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl"
CORPUS_SHA256 = "96552f06"            # prefix check only; the full hash is pinned by holdout_view.CORPUS_SHA256
ATLAS_DB = "codegen_spider1_test_mql0013"

# Every evaluation / corpus source the benchmark must be disjoint from (asserted by database and by id).
PROTECTED_SPLITS = {
    "mql0010_split_registration": ROOT / "outputs/nl2mql_mql0010/preflight/split_registration.json",
    "mql0010_holdout_items": ROOT / "outputs/nl2mql_mql0010/preflight/holdout_items.jsonl",
    "mql0010_dev_items": ROOT / "outputs/nl2mql_mql0010/preflight/dev_items.jsonl",
    "mql0012_x0_items": ROOT / "outputs/nl2mql_mql0012/split/x0_items.jsonl",
    "spider_sql_v1_train": ROOT / "data/processed/spider_sql_v1/train.jsonl",
    "spider_sql_v1_val_internal": ROOT / "data/processed/spider_sql_v1/val_internal.jsonl",
    "spider_sql_v1_dev_full": ROOT / "data/processed/spider_sql_v1/dev_full.jsonl",
    "splits_expanded_manifest": ROOT / "data/splits_expanded/split_manifest.json",
}

# Arms. Retrieval flags = the registered MQL-0010 D0 flags (hybrid, K=3, diversify none, no notes); lexical mode is the
# only difference. R0V / R3V = the D0V recipe (validator + one schema-feedback retry).
PLAN_ARMS = ("R0", "R3")
LEXICAL = {"R0": "bm25", "R3": "rrf_fused"}
ARMS = ("R0", "R0V", "R3", "R3V")
RETRY_ARMS = {"R0V": "R0", "R3V": "R3"}
PRIMARY = ("R3V", "R0V")
CTX = {"k": 3, "max_context_tokens": 1500, "n_ctx": 8192, "max_new_tokens": 1024, "margin_tokens": 64}
TARGET_ACCURACY = 0.60


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(s):
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_sha256(ids):
    return sha256_text("\n".join(ids))


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class MQL0013Error(RuntimeError):
    pass


def need(c, msg):
    if not c:
        raise MQL0013Error(msg)
