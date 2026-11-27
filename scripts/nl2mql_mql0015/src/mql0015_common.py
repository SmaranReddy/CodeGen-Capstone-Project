"""MQL-0015 shared constants and paths (final three-arm evaluation: A0V, R3V, FT-R3V on 1,000 Spider test_full items).

Nothing here touches a database, the corpus, R3 or any frozen artifact. MQL-0013 files are READ-ONLY inputs.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]

EXPERIMENT = "MQL-0015"
PACKAGE = "nl2mql_mql0015_final_1000"
OUT = ROOT / "outputs/nl2mql_final_1000"
SAMPLING = OUT / "sampling"
REG_DIR = OUT / "registration"
REGISTRATION = REG_DIR / "mql0015_registration.json"
AMENDMENT_A1 = REG_DIR / "amendment_A1_2026-11-23.json"
PROMPT_PINS = REG_DIR / "prompt_pins.jsonl"
CONTRACT_DIR = OUT / "scoring_contract"
CONTRACT = CONTRACT_DIR / "atlas_mapping.json"
PROVISION_DIR = OUT / "provisioning"
PREP = OUT / "preparation"
PKG_OUT = ROOT / "kaggle" / PACKAGE
ZIP_OUT = ROOT / "kaggle" / f"{PACKAGE}_bundle.zip"

# read-only inputs
SAMPLE_IDS = SAMPLING / "sample_ids.txt"
SAMPLE_MANIFEST = SAMPLING / "sample_manifest.json"
UNIVERSE_AUDIT = SAMPLING / "universe_overlap_audit.json"
TEST_FULL = ROOT / "data/processed/spider_sql_v1/test_full.jsonl"
SPIDER_DATA = ROOT / "data/raw/spider1/spider_data"
TEST_DB_DIR = SPIDER_DATA / "test_database"
TEST_TABLES = SPIDER_DATA / "test_tables.json"
MQL13_PKG = ROOT / "kaggle/nl2mql_mql0013_test_full"
MQL13_CONTRACT = ROOT / "outputs/nl2mql_mql0013/scoring_contract/atlas_mapping.json"
MQL13_PROMPT_CONTEXT = ROOT / "outputs/nl2mql_mql0013/work/prompt_context"
HANDOFF = ROOT / "outputs/finetuning_reuse/handoff_extract/ft_handoff_q3b_r16_q10"
ADAPTER_DIR = HANDOFF / "checkpoint/q3b_r16_q10"
EMB_MANIFEST = ROOT / "outputs/nl2mql_mql0011/embeddings/mql0011__v2_1_view2912__bge_small_en_v1_5__5c38ec7c__masked_q/manifest.json"
CORPUS = ROOT / "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl"

# pins (verified in the MQL-0015 preflight; re-checked by the builder)
REGISTRATION_SHA256 = "fb5d07d0c282b4eeb5ffd5da59c2bc3ab09068eb420a59cbf559920e6d85ab69"
SAMPLE_IDS_SHA256 = "5439b8b45c493b07dc7b4fa2ccae980e338cda8190ccb21c75119c216eb6fceb"
SAMPLE_MANIFEST_SHA256 = "640482af7855b78b46394316329da1810f2520ca033dd2438b7658c6cc3aeddf"
PROMPT_PINS_SHA256 = "f251746419101174154b7ebac76f0f5b5047df7e1ed65952afc2d7c3162487c4"
TEST_FULL_SHA256 = "c6601ecdec5a9461aafb9015d334233ff92d00d69d659314a08784e34d6552af"
MQL13_BUNDLE_SHA256 = "5cd0034064950325474e028824a577ba7506caeccf6e005d6bc7292d6244f89a"
MQL13_MANIFEST_SHA256 = "b43c56255c624a2ab44a560397bf72fc7304ab082bf325e6cfb03c7a45c253a3"
MQL13_CONTRACT_SHA256 = "32cbbc4dd82a18d12f1288b6491e14931481543b09c09f1df711f745e9cf4f42"
ADAPTER_SHA256 = "7096760ac2ed7e4a6c2eefdd7c0c249d4b3dba723f069982a499b876d434e09d"
ADAPTER_CONFIG_SHA256 = "3302913371469f750b4f121b833f0a8e93081f70ae9647a250bef3beec0926a5"
CORPUS_SHA256 = "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf"
VIEW_IDS_SHA256 = "cfde5e9c2df334bbfd08f159671b819e84ca9850c3fbb35d59c02ffb7a8d08de"
R3_INDEX = "mql0011__v2_1_view2912__bge_small_en_v1_5__5c38ec7c__masked_q"
EMBEDDER_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"

ATLAS_DB = "codegen_spider1_test_mql0015"
# every database that must never be written by MQL-0015 tooling
PROTECTED_DBS = frozenset({"codegen_spider1", "codegen_spider1_heldout", "codegen_spider1_holdout_mql0010",
                           "codegen_spider1_test_mql0013", "admin", "local", "config"})

ARMS = ("A0", "A0V", "R3", "R3V", "FT-R3", "FT-R3V")
REGISTERED_ARMS = ("A0V", "R3V", "FT-R3V")
PRIMARY = ("FT-R3V", "R3V")
SECONDARY = (("R3V", "A0V"), ("FT-R3V", "A0V"))


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


def sample_ids():
    t = SAMPLE_IDS.read_text(encoding="utf-8")
    need(sha256_text(t) == SAMPLE_IDS_SHA256, "sample_ids.txt differs from the registered sample")
    ids = t.split()
    need(len(ids) == 1000 == len(set(ids)), "sample size")
    return ids


class MQL0015Error(RuntimeError):
    pass


def need(c, msg):
    if not c:
        raise MQL0015Error(msg)
