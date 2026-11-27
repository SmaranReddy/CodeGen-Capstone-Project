"""RAG corpus eligibility audit for the 892-item NL->MQL evaluation set
(scripts/nl2mql/audit_rag_corpus_dev1034.py). Reads repo data only; writes to tmp_path.
No model, no network, no Atlas."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
import audit_rag_corpus_dev1034 as A  # noqa: E402


@pytest.fixture(scope="module")
def audit(tmp_path_factory):
    out = tmp_path_factory.mktemp("audit")
    argv = sys.argv
    sys.argv = ["audit", "--out", str(out)]
    try:
        A.main()
    finally:
        sys.argv = argv
    rows = [json.loads(ln) for ln in (out / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    return rows, json.loads((out / "summary.json").read_text(encoding="utf-8"))


def test_eval_bundle_pinned_and_unchanged():
    assert A.sha256_file(A.BUNDLE_ZIP) == A.BUNDLE_SHA256
    rows, _ = A.load_eval()
    assert len(rows) == 892 and len({r["db_id"] for r in rows}) == 17


def test_every_spider_corpus_example_is_an_eval_item_by_id(audit):
    rows, _ = audit
    sp = [r for r in rows if r["candidate_kind"] == "supervised_pair"
          and r["source"] in ("rag_corpus_v1_example", "ft_v1:spider1_dev_expanded_benchmark_train")]
    assert len(sp) == 560
    assert all(r["leakage"]["eval_id_match"] and r["leakage"]["eval_question_match"] for r in sp)
    assert all(r["decision"] == "exclude" for r in sp)


def test_kaggle_rows_excluded_for_label_quality_not_leakage(audit):
    rows, s = audit
    kg = [r for r in rows if r["source"] == "ft_v1:kaggle_nl2sql_query_dataset_v1"]
    assert len(kg) == 1584 and all(r["decision"] == "exclude" for r in kg)
    assert not any(r["leakage"]["eval_question_match"] or r["leakage"]["eval_database_match"] for r in kg)
    known_wrong = sum(any(x.startswith("known-wrong") for x in r["reasons"]) for r in kg)
    assert known_wrong == 88   # docs/sql2mql_independent_audit.md B1/B2


def test_no_supervised_pair_accepted_and_verdict_blocked(audit):
    rows, s = audit
    assert s["supervised_pairs"]["accepted"] == 0
    assert s["verdict"].startswith("BLOCKED")
    assert all(r.get("reasons") for r in rows if r["decision"] != "accept")


def test_guides_on_eval_databases_excluded(audit):
    rows, s = audit
    g = {r["record_id"]: r for r in rows if r["candidate_kind"] == "documentation"}
    assert len(g) == 18
    excluded = {k for k, r in g.items() if r["decision"] == "exclude"}
    assert excluded == {"guide_join_lookup", "guide_join_type_mismatch", "guide_anti_join",
                        "guide_count_per_parent"}
    assert s["documentation"]["pending_licence_check"] == 14


def test_leakage_detector_flags_normalised_question_match():
    ev = {"ids": {"spider_dev_0001"}, "questions": {A.loose("How many singers do we have?")},
          "dbs": {"concert_singer"}}
    r = A.supervised_row("x", "r1", "how many SINGERS do we have", {}, "other_db", "l", "s", {}, ev, [])
    assert r["decision"] == "exclude" and r["leakage"]["eval_question_match"]
    ok = A.supervised_row("x", "r2", "List all ships.", {}, "other_db", "l", "s", {}, ev, [])
    assert ok["decision"] == "accept"


def test_audit_does_not_write_into_previous_experiment_outputs():
    assert A.OUT.relative_to(A.ROOT).as_posix() == "outputs/nl2mql_spider_dev_1034_rag/corpus_audit_v1"
