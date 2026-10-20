"""Integrity of the database-disjoint train/validation/test split (data/splits/)."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/spider1"))
import make_splits as S  # noqa: E402

SPLITS = ROOT / "data/splits"
MANIFEST = SPLITS / "split_manifest.json"
pytestmark = pytest.mark.skipif(not (MANIFEST.exists() and S.SOURCE.exists()),
                                reason="split or dataset not built")


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def part_rows(p):
    return [json.loads(ln) for ln in (SPLITS / f"{p}.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def source_rows():
    return [r for _, r in S.load_source()[1]]


def test_source_hash_matches_manifest():
    m = manifest()
    assert hashlib.sha256(S.SOURCE.read_bytes()).hexdigest() == m["source"]["sha256"]
    assert m["source"]["n_examples"] == len(source_rows()) == 279


def test_every_example_in_exactly_one_partition():
    ids = [r["id"] for p in S.PARTITIONS for r in part_rows(p)]
    assert len(ids) == len(set(ids))                       # no overlap
    assert set(ids) == {r["id"] for r in source_rows()}    # nothing lost or added


def test_database_disjoint():
    owner = {}
    for p in S.PARTITIONS:
        for r in part_rows(p):
            assert owner.setdefault(r["db_id"], p) == p, r["db_id"]
    m = manifest()
    assert m["database_disjoint"] is True
    assert {db: v["partition"] for db, v in m["database_assignments"].items()} == owner
    assert all(owner[e["db_id"]] == e["partition"] for e in m["examples"])
    assert [e["id"] for e in m["examples"]] == [r["id"] for r in source_rows()]


def test_counts_ids_and_hashes_match_manifest():
    m = manifest()
    for p in S.PARTITIONS:
        rows = part_rows(p)
        assert m["partition_counts"][p] == m["partitions"][p]["n_examples"] == len(rows)
        assert m["partitions"][p]["ids"] == [r["id"] for r in rows]
        assert hashlib.sha256((SPLITS / f"{p}.jsonl").read_bytes()).hexdigest() == \
            m["partitions"][p]["sha256"]
    assert sum(m["partition_counts"].values()) == 279


def test_roughly_70_15_15():
    c = manifest()["partition_counts"]
    assert abs(c["train"] / 279 - 0.70) < 0.03
    assert abs(c["validation"] / 279 - 0.15) < 0.03 and abs(c["test"] / 279 - 0.15) < 0.03


def test_records_are_unchanged_copies_of_the_source():
    src = {r["id"]: r for r in source_rows()}
    for p in S.PARTITIONS:
        for r in part_rows(p):
            assert r == src[r["id"]]


def test_reproducible_from_recorded_seed_and_method():
    m = manifest()
    rebuilt, files = S.build(m["seed"], m["methodology"]["draws"])
    assert rebuilt == m
    for p in S.PARTITIONS:
        assert (SPLITS / f"{p}.jsonl").read_bytes() == files[p]
    assert S.verify() == []


def test_assignment_ignores_input_order():
    rows = source_rows()
    a, _ = S.assign(rows, seed=7, draws=200)
    b, _ = S.assign(list(reversed(rows)), seed=7, draws=200)
    assert a == b


def test_retrospective_limitation_is_recorded():
    note = manifest()["retrospective_evaluation_note"].lower()
    assert "retrospective" in note and "not an untouched held-out" in note
