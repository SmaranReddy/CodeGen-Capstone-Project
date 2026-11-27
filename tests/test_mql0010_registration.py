"""MQL-0010 registration integrity (R1) and holdout retrieval-view enforcement (R3). Offline; no model, no database."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0010/src"))
sys.path.insert(1, str(ROOT / "kaggle/nl2mql_mql0009_retrieval_control/src"))
import holdout_view as HV     # noqa: E402
import planner as PL          # noqa: E402
import recovery as RC         # noqa: E402
import scoring as SC          # noqa: E402

PRE = ROOT / "outputs/nl2mql_mql0010/preflight"
CORPUS = ROOT / "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def reg():
    return HV.load_registration(PRE / "split_registration.json")


@pytest.fixture(scope="module")
def index():
    p = PRE / "registration_index.json"
    if not p.exists():
        pytest.skip("registration index not written yet")
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ R1
def test_index_pins_every_registration_document(index):
    roles = {d["role"] for d in index["documents"]}
    assert roles == {"config", "analysis_plan", "split_registration", "amendment_A2", "amendment_A3", "amendment_A4"}
    for d in index["documents"]:
        assert sha(ROOT / d["path"]) == d["sha256"], d["path"]


def test_index_pins_source_and_corpus(index):
    src = ROOT / "scripts/nl2mql_mql0010/src"
    assert set(index["source_sha256"]) == {f.name for f in src.glob("*.py")}
    for name, h in index["source_sha256"].items():
        assert sha(src / name) == h, name
    assert sha(CORPUS) == index["corpus_sha256"] == HV.CORPUS_SHA256


def test_amendment_A4_matches_the_scoring_implementation(index):
    a4 = json.loads((ROOT / next(d["path"] for d in index["documents"] if d["role"] == "amendment_A4")).read_text(encoding="utf-8"))
    assert [a["name"] for a in a4["arms"]] == list(SC.ARMS) == index["effective_arms"]
    for a in a4["arms"]:
        PL.arm_settings(a["flags"])
    retry = {a["name"]: (a["source"], a["feedback_mode"]) for a in a4["arms"] if a["kind"] == "retry"}
    assert retry == SC.RETRY_SOURCES
    seq = a4["effective_test_sequence"]
    assert "D0V vs D0" in seq["step1"] and "FULL vs D0" in seq["step2"] and "D0V vs D0G" in seq["step2"]
    assert "E vs D0" in seq["step3"] and "EF vs EP" in seq["step3"]
    assert SC.STEP1 == ("D0V", "D0") and SC.STEP2 == (("FULL", "D0"), ("D0V", "D0G")) and SC.STEP3 == (("E", "D0"), ("EF", "EP"))
    assert SC.PRACTICAL_MIN_PP == 1.0 and "+1.0 percentage point" in a4["R4_power_correction_and_practical_criterion"]["criterion"]
    assert a4["R5_generic_retry_control"]["generic_feedback_text"] == RC.GENERIC_FEEDBACK
    assert {a["name"]: a["batch_size"] for a in a4["arms"]} == {"A0": 8, "A0V": 1, "D0": 8, "D0V": 1, "D0G": 1, "E": 8,
                                                                 "EP": 1, "EF": 1, "FULL": 1}


def test_original_registration_intact_and_superseded_not_rewritten(index):
    plan = next(d for d in index["documents"] if d["role"] == "analysis_plan")
    cfg = json.loads((ROOT / "scripts/nl2mql_mql0010/configs/mql0010_experiment.json").read_text(encoding="utf-8"))
    assert plan["sha256"] == cfg["analysis_plan_sha256"]


# ------------------------------------------------------------------ R3
def test_view_matches_registration_and_excludes_all_holdout_dbs(reg):
    view = HV.load_view(CORPUS, reg)
    assert len(view) == reg["holdout_retrieval_view"]["n_records"] == 2912
    assert not {r["db_id"] for r in view} & HV.holdout_dbs(reg)
    assert len(HV.holdout_dbs(reg)) == 18
    with open(PRE / "holdout_items.jsonl", encoding="utf-8") as f:
        hold_ids = {json.loads(x)["id"] for x in f if x.strip()}
    assert not {r["id"] for r in view} & hold_ids


def test_view_fails_closed_on_tampering(reg, tmp_path):
    bad = json.loads(json.dumps(reg))
    bad["holdout_retrieval_view"]["ids_sha256"] = "0" * 64
    with pytest.raises(HV.ViewError):
        HV.load_view(CORPUS, bad)
    bad2 = json.loads(json.dumps(reg))
    bad2["holdout"]["dbs"] = bad2["holdout"]["dbs"][:-1]          # one holdout DB "forgotten"
    with pytest.raises(HV.ViewError):
        HV.load_view(CORPUS, bad2)
    copy = tmp_path / "corpus.jsonl"
    copy.write_bytes(CORPUS.read_bytes() + b"\n")
    with pytest.raises(HV.ViewError):
        HV.load_view(copy, reg)


def test_docs_and_planner_refuse_holdout_documents(reg):
    hdb = sorted(HV.holdout_dbs(reg))
    docs = [{"doc_id": "x", "kind": "example", "db_id": hdb[0], "question": "q", "masked_question": "q",
             "patterns": ["select"], "mql": {}, "content": "c"}]
    with pytest.raises(HV.ViewError):
        HV.assert_docs_exclude(docs, set(hdb))
    with pytest.raises(ValueError):
        PL.build_plan({"family": "hybrid"}, [], [], docs, {}, {}, {}, {}, len, forbidden_db_ids=hdb)
    assert PL.build_plan({"family": "hybrid"}, [], [], [], {}, {}, {}, {}, len, forbidden_db_ids=hdb) == []
