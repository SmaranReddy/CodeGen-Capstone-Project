"""MQL-0011 shared constants and import paths (dense retrieval; offline research phase).

Frozen inputs are only READ. The MQL-0010 view loader (`holdout_view.load_view`) is reused unchanged: it checks the
approved-corpus hash, the registered view ids/count and the absence of the 18 holdout databases, failing closed.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
for p in (HERE / "src", ROOT / "scripts/nl2mql_mql0010/src", ROOT / "kaggle/nl2mql_mql0009_retrieval_control/src"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

CORPUS = ROOT / "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl"
CORPUS_SHA256 = "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf"
REGISTRATION = ROOT / "outputs/nl2mql_mql0010/preflight/split_registration.json"
DEV_ITEMS = ROOT / "outputs/nl2mql_mql0010/preflight/dev_items.jsonl"
TABLES = ROOT / "data/raw/spider1/spider_data/tables.json"
SPIDER_TRAIN = ROOT / "data/processed/spider_sql_v1/train.jsonl"
VIEW_IDS_SHA256 = "cfde5e9c2df334bbfd08f159671b819e84ca9850c3fbb35d59c02ffb7a8d08de"
VIEW_N = 2912
DEV_IDS_SHA256 = "b65ef62cfd0d667ef89f6eeac23fd1f9a4f0366aa982a5eedab45c265e354bcb"
DEV_N = 779

MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
INDEX_NAME = "mql0011__v2_1_view2912__bge_small_en_v1_5__5c38ec7c__masked_q"
OUT = ROOT / "outputs/nl2mql_mql0011"
ARTIFACT_DIR = OUT / "embeddings" / INDEX_NAME
CONFIG = HERE / "configs/mql0011_screen.json"
QDRANT_PATH = Path(os.environ.get("MQL0011_QDRANT_PATH", str(Path.home() / "codegen_data/qdrant_mql0011")))


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(s):
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


def ids_sha256(ids):
    """Registered id-list hash scheme (holdout_view.load_view): sha256 of the ids joined by newlines."""
    return sha256_text("\n".join(ids))


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8",
                       newline="\n")


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def load_registration():
    return json.loads(REGISTRATION.read_text(encoding="utf-8"))


def load_view_rows():
    """-> (view rows in corpus order, holdout db set). All checks of the registered MQL-0010 view apply."""
    import holdout_view as HV
    reg = load_registration()
    rows = HV.load_view(CORPUS, reg)
    if len(rows) != VIEW_N or ids_sha256([r["id"] for r in rows]) != VIEW_IDS_SHA256:
        raise ValueError("retrieval view differs from the MQL-0011 frozen constants")
    return rows, HV.holdout_dbs(reg)


def load_tables():
    return {t["db_id"]: t for t in json.loads(TABLES.read_text(encoding="utf-8"))}


def schema_infos(tabs, db_ids):
    import mql0008_core as M8
    import patterns as PT
    return {db: PT.schema_info(db, M8.schema_manifest_from_tables(tabs[db])) for db in db_ids}
