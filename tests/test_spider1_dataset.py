"""Integrity tests for the validated Spider 1.0 NL -> MQL dataset and scoring helpers."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/spider1"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
import spider1_kit as K  # noqa: E402

DATASET = ROOT / "data/processed/spider1_nl_mql.jsonl"
pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="dataset not built")


def rows():
    return K.read_jsonl(DATASET)


def test_size_and_uniqueness():
    d = rows()
    assert 200 <= len(d) <= 300
    assert len({r["id"] for r in d}) == len(d)
    assert len({(r["db_id"], r["question"]) for r in d}) == len(d)
    assert len({(r["db_id"], r["gold_sql"]) for r in d}) == len(d)


def test_every_row_traces_to_frozen_sample_and_official_dev():
    sample = {r["id"]: r for r in K.read_jsonl(K.INTERIM / "sample.jsonl")}
    dev = K.load_json(K.SPIDER / "dev.json")
    for r in rows():
        s = sample[r["id"]]
        assert r["question"] == s["question"] == dev[r["dev_index"]]["question"]
        assert r["gold_sql"] == dev[r["dev_index"]]["query"]
        assert r["gold_rows"] == s["gold_rows"]


def test_gold_mql_is_read_only_and_scoped():
    from validate_gold import FORBIDDEN, refs, walk_keys
    for r in rows():
        p = r["gold_mql"]["pipeline"]
        assert not (set(walk_keys(p)) & FORBIDDEN)
        for c in {r["gold_mql"]["collection"]} | refs(p):
            assert c.startswith(r["db_id"] + "__")


def test_evidence_and_exclusions_recorded():
    adj = json.loads((K.INTERIM / "adjudication.json").read_text(encoding="utf-8"))
    excluded = {k for k, v in adj.items() if v["decision"] == "exclude"}
    ids = {r["id"] for r in rows()}
    assert not ids & excluded
    assert all(r["evidence"] in ("dev+testsuite", "dev_only") for r in rows())


def test_model_output_validator_rejects_writes():
    from eval_kit import validate_mql
    allowed = {"x__t"}
    ok, _, _ = validate_mql({"method": "aggregate", "collection": "x__t",
                             "pipeline": [{"$match": {}}, {"$out": "x__t"}]}, allowed)
    assert not ok
    ok, _, _ = validate_mql({"method": "aggregate", "collection": "x__t",
                             "pipeline": [{"$lookup": {"from": "other__t", "localField": "a",
                                                       "foreignField": "b", "as": "c"}}]}, allowed)
    assert not ok
