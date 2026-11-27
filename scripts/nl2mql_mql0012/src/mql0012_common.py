"""MQL-0012 shared constants and paths. Reuses MQL-0011 modules READ-ONLY (imported, never modified)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
for p in (HERE / "src", ROOT / "scripts/nl2mql_mql0011/src", ROOT / "scripts/nl2mql_mql0011"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import mql0011_common as C11   # noqa: E402  (paths, hashing, view loader, MQL-0011 constants)

OUT = ROOT / "outputs/nl2mql_mql0012"
CONFIG = HERE / "configs/mql0012_config.json"
SPLIT = OUT / "split/x0_split.json"
X0_ITEMS = OUT / "split/x0_items.jsonl"
QUERY_CACHE_X0 = OUT / "query_cache_x0"
REGISTRATION = OUT / "registration/mql0012_registration.json"
RESULTS = OUT / "results"
POOL_V3_TRAIN = ROOT / "data/processed/spider_pool_v3/train.jsonl"
MQL0009_EVAL = ROOT / "kaggle/nl2mql_mql0009_retrieval_control/data/eval/eval_inputs.jsonl"
SPIDER_DEV_FULL = ROOT / "data/processed/spider_sql_v1/dev_full.jsonl"
EXPANDED_SPLITS = [ROOT / f"data/splits_expanded/{n}.jsonl" for n in ("train", "validation", "test")]
MQL0011_PER_ITEM = ROOT / "outputs/nl2mql_mql0011/screen/per_item.jsonl"
MQL0011_RESULTS = ROOT / "outputs/nl2mql_mql0011/screen/screen_results.json"


def eligible_item(r):
    """Identical to scripts/nl2mql_mql0010/preflight/make_split.py::eligible_item (registered MQL-0010 rule)."""
    return r["tier"] == "exec_ok" and not ({"empty_result", "all_null_result"} & set(r["flags"]))
