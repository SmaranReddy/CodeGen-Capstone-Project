"""MQL-0013 unit tests (pure Python; no model, no database). The end-to-end mocked run is preflight/dry_run.py."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0013/src"))
sys.path.insert(1, str(ROOT / "scripts/nl2mql_mql0013"))
import mql0013_analysis as AN   # noqa: E402
import mql0013_common as CM     # noqa: E402


def _ev(o, **kw):
    return {"outcome": o, "reference_status": "ok", **kw}


def test_mcnemar_exact_known_values():
    assert AN.mcnemar_exact(0, 0) == 1.0
    assert AN.mcnemar_exact(5, 5) == 1.0
    assert abs(AN.mcnemar_exact(0, 6) - 2 / 64) < 1e-12


def test_taxonomy_maps_every_failure_kind():
    assert AN.taxonomy(_ev("failed", detail="generation_error", failure_source="infrastructure")) == "generation_failure"
    assert AN.taxonomy(_ev("failed", failure_source="model")) == "execution_failure"
    assert AN.taxonomy(_ev("failed", failure_source="infrastructure", detail="atlas_unavailable")) == "infrastructure"
    for o in ("correct", "incorrect", "invalid", "timeout", "not_comparable"):
        assert AN.taxonomy(_ev(o)) == o


def test_analyze_strict_denominator_and_target():
    ids = [f"q{i}" for i in range(10)]
    outs = {"R0": ["correct"] * 5 + ["invalid"] * 5, "R0V": ["correct"] * 6 + ["timeout"] * 4,
            "R3": ["correct"] * 6 + ["not_comparable"] * 4, "R3V": ["correct"] * 7 + ["incorrect"] * 3}
    evals = {a: {i: _ev(o) for i, o in zip(ids, v)} for a, v in outs.items()}
    a = AN.analyze(evals, ids, {i: "easy" for i in ids}, {i: ["simple"] for i in ids}, reps=200)
    assert a["accuracy"]["R3V"]["correct"] == 7 and a["accuracy"]["R3V"]["total"] == 10
    assert a["accuracy"]["R3"]["correct"] == 6          # not_comparable counts as wrong
    assert a["primary"]["a"] == "R3V" and a["primary"]["b"] == "R0V" and a["primary"]["a_only"] == 1
    assert a["target_60pct"]["threshold_correct"] == 6 and a["target_60pct"]["met"] is True
    assert a["taxonomy"]["R0"]["invalid"] == 5 and a["taxonomy"]["R0V"]["timeout"] == 4
    assert a["by_hardness"]["easy"]["n"] == 10


def test_target_threshold_for_full_benchmark():
    import math
    assert math.ceil(AN.TARGET * 2147) == 1289


def test_unique_slots_share_identical_text():
    import mql0013_scoring as S
    ids = ["a", "b"]
    mk = lambda arm, i, t: {"id": i, "raw_text": t, "status": "ok", "arm": arm}   # noqa: E731
    recs = {"R0": [mk("R0", "a", "x"), mk("R0", "b", "y")], "R0V": [mk("R0V", "a", "x"), mk("R0V", "b", "y2")],
            "R3": [mk("R3", "a", "x"), mk("R3", "b", "z")], "R3V": [mk("R3V", "a", "x"), mk("R3V", "b", "y2")]}
    slots, slot_of = S.unique_slots(recs, ids)
    assert len(slots) == 1 + 3
    assert len({slot_of[(arm, "a")] for arm in AN.ARMS}) == 1
    assert slot_of[("R0V", "b")] == slot_of[("R3V", "b")] != slot_of[("R0", "b")]
    assert all(s["item_id"] in ids for s in slots)


def test_benchmark_is_disjoint_from_protected_sources():
    import build_mql0013_package as B
    recs = B.load_benchmark()
    rep = B.disjointness(recs)
    assert rep["overlaps"] == 0 and rep["n_test_dbs"] == 40


def test_verify_exports_rejects_unexplained_change():
    sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0013/pkg_src"))
    sys.path.insert(1, str(ROOT / "kaggle/nl2mql_mql0010_holdout/src"))
    import mql0013_core as K
    ids = ["a"]
    plans = {"R0": [{"id": "a", "prompt_sha256": "p0"}], "R3": [{"id": "a", "prompt_sha256": "p3"}]}
    tr = {"retried": False, "kept": "first", "feedback_mode": "schema"}
    base = lambda arm, p, t, f: {"id": "a", "arm": arm, "status": "ok", "prompt_sha256": p, "raw_text": t,   # noqa: E731
                                 "first_raw_text": f, "recovery": tr if arm.endswith("V") else None}
    recs = {"R0": [base("R0", "p0", "x", "x")], "R3": [base("R3", "p3", "y", "y")],
            "R0V": [base("R0V", "p0", "x", "x")], "R3V": [base("R3V", "p3", "y", "y")]}
    assert K.verify_exports(recs, plans, ids, [])["ok"]
    recs["R3V"] = [base("R3V", "p3", "changed", "y")]
    with pytest.raises(Exception):
        K.verify_exports(recs, plans, ids, [])


def test_registration_pins_sources_if_present():
    if not CM.REGISTRATION.exists():
        pytest.skip("not registered yet")
    reg = json.loads(CM.REGISTRATION.read_text(encoding="utf-8"))
    for rel, h in reg["source_sha256"].items():
        assert CM.sha256_file(ROOT / rel) == h, rel
    assert reg["evaluation"]["n_items"] == 2147 and reg["arms"] == list(AN.ARMS)
