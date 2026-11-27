"""MQL-0011: planner_dense(lexical="bm25") reproduces the registered planner.build_plan on all 779 dev questions
(arms D0 and E, real Qwen tokenizer budget). Every row field is identical; the only addition is "retrieval_mql0011".
Needs transformers + the pinned tokenizer (venv_rag or venv_tok); skips otherwise. Writes nothing.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0011/analysis"))
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0011/src"))
import planner_equivalence as PE   # noqa: E402


def test_planner_equivalence_779_dev():
    transformers = pytest.importorskip("transformers")
    if not (PE.TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("pinned Qwen tokenizer absent")
    tok = transformers.AutoTokenizer.from_pretrained(str(PE.TOKENIZER_DIR))
    rep = PE.compare(lambda s: len(tok.encode(s, add_special_tokens=False)))
    for arm, r in rep["arms"].items():
        assert r["n_orig"] == r["n_new"] == 779, arm
        assert r["rows_differing"] == 0, (arm, r["differing_ids"])
        assert r["plan_sha256_orig"] == r["plan_sha256_new_minus_extra_key"], arm
    assert rep["equivalent"]
