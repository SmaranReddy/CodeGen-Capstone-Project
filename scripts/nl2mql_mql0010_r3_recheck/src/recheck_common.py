"""MQL-0014 (MQL-0010 R3 execution recheck) shared constants and paths.

SECONDARY evaluation / re-analysis of the existing MQL-0010 holdout (716 questions, 18 databases): R0V (BM25 hybrid =
MQL-0010 D0V) vs R3V (MQL-0011 R3 retrieval + the same validator / one schema retry). Not a fresh independent test.
Everything new lives under outputs/nl2mql_mql0010_r3_execution_recheck/ and kaggle/nl2mql_mql0014_r3_recheck/;
MQL-0010 artifacts are only read.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]

EXPERIMENT = "MQL-0014"
PACKAGE = "nl2mql_mql0014_r3_recheck"
OUT = ROOT / "outputs/nl2mql_mql0010_r3_execution_recheck"
WORK = OUT / "work"
REG_DIR = OUT / "registration"
REGISTRATION = REG_DIR / "mql0014_registration.json"
PKG_OUT = ROOT / "kaggle" / PACKAGE
ZIP_OUT = ROOT / "kaggle" / f"{PACKAGE}_bundle.zip"

M10_PKG = ROOT / "kaggle/nl2mql_mql0010_holdout"
M10_CONTRACT = ROOT / "outputs/nl2mql_mql0010/scoring_contract/atlas_mapping.json"
M10_RUN = ROOT / "outputs/nl2mql_mql0010/scoring_runs/run_final"
M10_REFERENCE_SHA = "1ff619f2f11bbe943c474f71ba2bd63b128cbc33fdf712672a81a53df3c473fc"
M10_D0V_SHA = "efc13d50301556922320ac46ad078cea8b55667156543f1b18bc011b5a4aa8ef"
TRAIN = ROOT / "data/processed/spider_sql_v1/train.jsonl"
DB_DIR = ROOT / "data/raw/spider1/spider_data/database"
ATLAS_DB = "codegen_spider1_holdout_mql0010"

PLAN_ARMS = ("R0", "R3")
LEXICAL = {"R0": "bm25", "R3": "rrf_fused"}
ARMS = ("R0", "R0V", "R3", "R3V")


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
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class RecheckError(RuntimeError):
    pass


def need(c, msg):
    if not c:
        raise RecheckError(msg)
