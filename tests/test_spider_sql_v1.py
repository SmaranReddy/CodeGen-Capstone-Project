"""Integrity of the Spider NL->SQL dataset (data/processed/spider_sql_v1/)."""
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/spider_sql_v1"
SPIDER = ROOT / "data/raw/spider1/spider_data"
MANIFEST = OUT / "manifest.json"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="spider_sql_v1 not built")


def norm_q(q):
    return re.sub(r"\s+", " ", q.strip().lower()).rstrip("?.! ")


def rows(name):
    return [json.loads(ln) for ln in (OUT / name).read_text(encoding="utf-8").splitlines() if ln.strip()]


def raw(name):
    return json.loads((SPIDER / name).read_text(encoding="utf-8"))


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_files_match_manifest_hashes():
    for name, sha in manifest()["files_sha256"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == sha, name


def test_raw_inputs_unchanged():
    for name, sha in manifest()["inputs_sha256"].items():
        assert hashlib.sha256((SPIDER / name).read_bytes()).hexdigest() == sha, name


def test_train_accounting():
    n_raw = len(raw("train_spider.json")) + len(raw("train_others.json"))
    tr, va, rej = rows("train.jsonl"), rows("val_internal.jsonl"), rows("rejected.jsonl")
    assert len(tr) + len(va) + len(rej) == n_raw
    ids = [r["id"] for r in tr + va + rej]
    assert len(ids) == len(set(ids))
    assert all(r.get("reject_reason") for r in rej)


def test_kept_train_rows_execute():
    for r in rows("train.jsonl") + rows("val_internal.jsonl"):
        assert r["execution"]["status"] == "ok" and r["tier"], r["id"]


def test_database_disjointness():
    tr = {r["db_id"] for r in rows("train.jsonl")}
    va = {r["db_id"] for r in rows("val_internal.jsonl")}
    dev = {e["db_id"] for e in raw("dev.json")}
    test = {e["db_id"] for e in raw("test.json")}
    assert not tr & va
    assert not (tr | va) & dev
    assert not (tr | va) & test


def test_no_eval_question_in_training_files():
    evalq = {norm_q(e["question"]) for f in ("dev.json", "test.json") for e in raw(f)}
    for r in rows("train.jsonl") + rows("val_internal.jsonl"):
        assert norm_q(r["question"]) not in evalq, r["id"]


def test_no_exact_duplicates_in_training_files():
    keys = [(r["db_id"], r["question"], r["sql"]) for r in rows("train.jsonl") + rows("val_internal.jsonl")]
    assert len(keys) == len(set(keys))


def test_eval_full_sets_are_complete_and_unaltered():
    for split in ("dev", "test"):
        full, src = rows(f"{split}_full.jsonl"), raw(f"{split}.json")
        assert len(full) == len(src)
        for r, e in zip(full, src):
            assert (r["db_id"], r["question"], r["sql"]) == (e["db_id"], e["question"], e["query"])


def test_eval_clean_drops_only_broken_golds():
    for split in ("dev", "test"):
        full, clean = rows(f"{split}_full.jsonl"), rows(f"{split}_clean.jsonl")
        assert [r["id"] for r in clean] == [r["id"] for r in full
                                            if r["execution"]["status"] == "ok" and not r.get("label_defects")]


def test_no_label_defects_in_training_files():
    for r in rows("train.jsonl") + rows("val_internal.jsonl"):
        assert not r.get("label_defects"), r["id"]


def test_val_internal_uses_only_populated_official_dbs():
    weak = set(manifest()["empty_official_train_dbs"])
    for r in rows("val_internal.jsonl"):
        assert r["db_id"] not in weak and r["tier"] == "exec_ok", r["id"]
