"""Tests for the provisional RAG corpus v1 (scripts/finetune_v2/run_corpus_v1.py).

Eligibility tests need only the pool/audit files. Output tests read outputs/sql2mql_v2/provisional_corpus_v1/ and
are skipped until `finalize` has run. Nothing here executes MongoDB or modifies any artifact.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts/finetune_v2"))
import run_corpus_v1 as K  # noqa: E402

OUT = ROOT / K.OUT_REL
FINAL = (OUT / "manifest.json").exists()
needs_final = pytest.mark.skipif(not FINAL, reason="corpus not finalized")


def rd(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.fixture(scope="module")
def elig():
    return K.derive_eligibility()


def test_eligibility_counts_and_disjointness(elig):
    assert len(elig["eligible"]) == 5045 == len(set(elig["eligible"]))
    assert all(elig["audit"][i]["status"] == "NO_ISSUE_DETECTED" for i in elig["eligible"])
    for name, ids in elig["other_ids"].items():
        assert not set(ids) & set(elig["eligible"]), name


def test_pilot_flagged_ids_are_excluded_from_corpus(elig):
    assert len(elig["pre_excluded"]) == 9
    for i, why in elig["pre_excluded"].items():
        assert why.startswith("pilot_cross_review_")


def test_status_mapping_is_total_and_exclusive():
    base = {"outcome": "x", "verification": {}}
    for oc, want in (("accepted", "ACCEPTED"), ("conversion_rejected", "UNSUPPORTED"), ("safety_rejected", "INVALID"),
                     ("timeout", "TIMEOUT"), ("verifier_error", "VERIFICATION_ERROR"), ("conversion_error", "EXECUTION_ERROR")):
        assert K.map_status({**base, "outcome": oc}) == want
    assert K.map_status({"outcome": "verification_failed", "verification": {"status": "mongo_error"}}) == "EXECUTION_ERROR"
    assert K.map_status({"outcome": "verification_failed", "verification": {"status": "fixture_mismatch"}}) == "REJECTED"


def test_exclude_database_helper():
    es = [{"db_id": "a"}, {"db_id": "b"}, {"db_id": "a"}]
    assert K.exclude_database(es, "a") == [{"db_id": "b"}]


def test_disposition_rules():
    rec = {"id": "x", "status": "ACCEPTED", "multi_table": {"multi_table_3plus": False}}
    ev = {"dbs": set(), "q": set(), "sql": set()}
    row = {"db_id": "d", "question": "q", "sql": "s"}
    assert K.corpus_disposition(rec, {}, ev, row) == ("INCLUDED", [])
    assert K.corpus_disposition({**rec, "multi_table": {"multi_table_3plus": True}}, {}, ev, row)[1] == ["multi_table_3plus"]
    assert K.corpus_disposition(rec, {"x": "pilot_cross_review_INCORRECT"}, ev, row)[0] == "EXCLUDED"
    assert K.corpus_disposition({**rec, "status": "REJECTED"}, {}, ev, row)[0] == "EXCLUDED"
    assert "leakage_db_overlap" in K.corpus_disposition(rec, {}, {**ev, "dbs": {"d"}}, row)[1]


def test_no_corpus_database_is_an_evaluation_database(elig):
    ev = K.load_eval_hashes()
    assert not {r["db_id"] for r in elig["rows"].values()} & ev["dbs"]


# ------------------------------------------------------------------ finalized outputs
@needs_final
def test_status_accounting_reconciles(elig):
    st = rd(OUT / "conversion_status.jsonl")
    assert [x["id"] for x in st] == sorted(elig["eligible"])
    assert len({x["id"] for x in st}) == 5045
    assert all(x["status"] in K.STATUSES for x in st)
    summ = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert sum(summ["statuses"].values()) == 5045
    assert dict(Counter(x["status"] for x in st)) == {k: v for k, v in summ["statuses"].items() if v}
    assert len(rd(OUT / "accepted_conversions.jsonl")) == summ["statuses"]["ACCEPTED"]
    assert len(rd(OUT / "rejected_conversions.jsonl")) == 5045 - summ["statuses"]["ACCEPTED"]


@needs_final
def test_corpus_eligibility_and_fields(elig):
    corpus = rd(OUT / "provisional_rag_corpus.jsonl")
    st = {x["id"]: x for x in rd(OUT / "conversion_status.jsonl")}
    assert len(corpus) == sum(x["corpus_disposition"] == "INCLUDED" for x in st.values()) > 0
    ev = K.load_eval_hashes()
    banned = set()
    for n in ("val_internal.jsonl", "quarantine.jsonl", "exclusions.jsonl"):
        banned |= set(elig["other_ids"][n])
    for e in corpus:
        s = st[e["id"]]
        assert s["status"] == "ACCEPTED" and not s["multi_table_3plus"]
        assert e["id"] not in elig["pre_excluded"] and e["id"] not in banned
        assert elig["audit"][e["id"]]["status"] == "NO_ISSUE_DETECTED"
        assert e["corpus_status"] == "PROVISIONAL" and e["human_review_status"] == "NOT_REVIEWED"
        assert e["verifier_version"] == "sql2mql-v2-verifier-3" and e["converter_version"] == "sql2mql-v2"
        assert not K.leakage_reasons(elig["rows"][e["id"]], ev)
        row = elig["rows"][e["id"]]       # originals preserved
        assert (e["question"], e["sql"], e["db_id"]) == (row["question"], row["sql"], row["db_id"])
        assert e["mql"] and e["schema"]["touched"] and e["source_label_audit"]["status"] == "NO_ISSUE_DETECTED"
    for t in ("db_id", "question", "mql", "schema", "id", "verification"):
        assert all(t in e for e in corpus)


@needs_final
def test_target_database_exclusion_on_real_corpus():
    corpus = rd(OUT / "provisional_rag_corpus.jsonl")
    for db in sorted({e["db_id"] for e in corpus})[:5]:
        assert all(e["db_id"] != db for e in K.exclude_database(corpus, db))


@needs_final
def test_no_forbidden_split_in_any_output():
    for e in rd(OUT / "provisional_rag_corpus.jsonl"):
        assert e["source"]["split"] == "train"


@needs_final
def test_reproducible_rebuild_is_byte_identical(elig):
    """Rebuild every derived file from records.jsonl: bytes must equal what finalize wrote."""
    fz = json.loads((ROOT / K.FREEZE_REL).read_text(encoding="utf-8"))
    recs = K.load_records(OUT / "records.jsonl", set(elig["eligible"]))
    ev = K.load_eval_hashes()
    man = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    summ = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    ctx = {"eligible": elig["eligible"], "rows": elig["rows"], "audit": elig["audit"], "pre_excluded": elig["pre_excluded"],
           "pre_excluded_in_eligible": elig["pre_excluded_in_eligible"], "cross_review": elig["cross_review"], "ev": ev,
           "freeze": fz, "slices": K.overlap_slices(elig["eligible"]), "leakage_summary": summ["leakage"]}
    files, _ = K.build_outputs(recs, ctx)
    for name, b in files.items():
        assert K.R.sha_b(b) == man["outputs"][name]["sha256"], name
        assert (OUT / name).read_bytes() == b, name


@needs_final
def test_manifest_hashes_match_files_and_code_freeze():
    man = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for name, m in man["outputs"].items():
        assert K.R.sha_f(OUT / name) == m["sha256"], name
    assert man["records_jsonl_sha256"] == K.R.sha_f(OUT / "records.jsonl")
    assert len(man["frozen_code_19"]) == 19
    assert all(K.R.check_frozen_code()[k]["actual"] == v for k, v in man["frozen_code_19"].items())
    assert man["verifier"] == "sql2mql-v2-verifier-3"


@needs_final
def test_pilot_overlap_substantively_reproduces():
    rep = json.loads((OUT / "pilot_overlap_reproducibility.json").read_text(encoding="utf-8"))
    assert rep["compared"] > 0
    assert rep["substantive_identical(outcome,detail,mql,verifier_status)"] == rep["compared"], rep["substantive_differing_ids"]
