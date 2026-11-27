"""Tests for the verifier-3 pilot runner (scripts/finetune_v2/run_pilot_v3.py).

Pure tests need no MongoDB. `live` tests use the local mongo:8.0.34 container (127.0.0.1:27018) and are
skipped without it. Live tests use DEV ids only (the 500 development ids of pilot_freeze.json); no pilot id is
converted or verified here, and no round-3 id is used. Frozen files are never edited: corruption tests work on
copies in tmp_path.
"""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "scripts/finetune_v2"
sys.dont_write_bytecode = True
sys.path.insert(0, str(FT))
import run_pilot_v3 as R  # noqa: E402

HASHES = R.check_frozen_code()
MODS = R.load_frozen_modules()
C = MODS["v2_common"]
FREEZE = json.loads((ROOT / R.PILOT_FREEZE_REL).read_text(encoding="utf-8"))
PILOT = set(FREEZE["pilot"]["ids"])
DEV = set(FREEZE["dev"]["ids"])
R3_IDS = {i for v in json.loads((ROOT / R.PREREG_REL).read_text(encoding="utf-8"))["sampling"]["ids"].values() for i in v}


def _copy_frozen(tmp):
    ft = tmp / "scripts/finetune_v2"
    ft.mkdir(parents=True)
    for n in R.FROZEN_FILES:
        if "/" not in n:
            shutil.copy2(FT / n, ft / n)
    (tmp / "scripts/evaluation").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/evaluation/eval_kit.py", tmp / "scripts/evaluation/eval_kit.py")
    (tmp / "data/interim/finetune_v2").mkdir(parents=True)
    shutil.copy2(ROOT / R.CODE_FREEZE_REL, tmp / R.CODE_FREEZE_REL)
    return ft


# ------------------------------------------------------------------ 19 frozen hashes
def test_all_19_frozen_hashes_match():
    assert len(HASHES) == 19 and set(HASHES) == set(R.FROZEN_FILES)
    assert all(v["match"] and v["actual"] == v["expected"] for v in HASHES.values())
    fz = json.loads((ROOT / R.CODE_FREEZE_REL).read_text(encoding="utf-8"))
    assert {k: v["expected"] for k, v in HASHES.items()} == fz["code_sha256"]
    assert R.sha_f(ROOT / R.CODE_FREEZE_REL) == R.CODE_FREEZE_SHA


def test_missing_verifier_is_rejected(tmp_path):
    ft = _copy_frozen(tmp_path)
    R.check_frozen_code(ft, tmp_path)            # the copy itself passes
    (ft / "verifier3.py").unlink()
    with pytest.raises(R.IntegrityError, match="verifier3.py \\(missing\\)"):
        R.check_frozen_code(ft, tmp_path)


@pytest.mark.parametrize("name", ["verifier3.py", "fixtures3.py", "verifier.py", "sql2mql_v2.py",
                                  "scripts/evaluation/eval_kit.py"])
def test_modified_frozen_file_is_rejected(tmp_path, name):
    ft = _copy_frozen(tmp_path)
    p = (tmp_path / name) if "/" in name else ft / name
    p.write_bytes(p.read_bytes() + b"\n# one extra byte\n")
    with pytest.raises(R.IntegrityError, match="differs"):
        R.check_frozen_code(ft, tmp_path)


def test_incomplete_or_altered_manifest_is_rejected(tmp_path):
    ft = _copy_frozen(tmp_path)
    fz = json.loads((ROOT / R.CODE_FREEZE_REL).read_text(encoding="utf-8"))
    for mutate in (lambda c: c.pop("verifier3.py"), lambda c: c.update({"extra.py": "0" * 64})):
        bad = json.loads(json.dumps(fz))
        mutate(bad["code_sha256"])
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(R.IntegrityError, match="incomplete or altered"):
            R.check_frozen_code(ft, tmp_path, p, freeze_sha=None)
        with pytest.raises(R.IntegrityError, match="manifest itself changed"):
            R.check_frozen_code(ft, tmp_path, p)
    with pytest.raises(R.IntegrityError, match="missing"):
        R.check_frozen_code(ft, tmp_path, tmp_path / "nope.json")
    bad = json.loads(json.dumps(fz))
    bad["verifier"] = "sql2mql-v2-verifier-2"
    p = tmp_path / "v2.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(R.IntegrityError, match="not sql2mql-v2-verifier-3"):
        R.check_frozen_code(ft, tmp_path, p, freeze_sha=None)


def test_cli_fails_closed_without_verifier3(tmp_path):
    ft = _copy_frozen(tmp_path)
    shutil.copy2(FT / "run_pilot_v3.py", ft / "run_pilot_v3.py")
    (ft / "verifier3.py").unlink()
    p = subprocess.run([sys.executable, "-B", str(ft / "run_pilot_v3.py"), "check"], capture_output=True, text=True,
                       timeout=120)
    assert p.returncode != 0
    assert "INTEGRITY FAILURE" in p.stderr and "verifier3.py (missing)" in p.stderr
    assert "frozen_code_19" not in p.stdout          # nothing ran


# ------------------------------------------------------------------ no verifier-2 fallback
def test_runner_never_imports_verifier2_or_old_runner():
    tree = ast.parse((FT / "run_pilot_v3.py").read_text(encoding="utf-8"))
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert not imported & {"verifier", "run_v2", "fixtures", "common"}
    assert "verifier" not in [m for m in R.FROZEN_MODULES if m == "run_v2"]
    assert MODS["run_v3"].V3 is MODS["verifier3"] and MODS["verifier3"].VERIFIER_VERSION == R.EXPECTED_VERIFIER


def test_import_failure_of_verifier3_fails_closed(monkeypatch):
    real = R.importlib.import_module

    def fake(name, *a, **k):
        if name == "verifier3":
            raise ImportError("simulated missing verifier3")
        return real(name, *a, **k)
    monkeypatch.setattr(R.importlib, "import_module", fake)
    with pytest.raises(R.IntegrityError, match="import failed"):
        R.load_frozen_modules()


def test_wrong_verifier_version_fails_closed(monkeypatch):
    monkeypatch.setattr(MODS["verifier3"], "VERIFIER_VERSION", "sql2mql-v2-verifier-2")
    with pytest.raises(R.IntegrityError, match="VERIFIER_VERSION"):
        R.load_frozen_modules()


def test_verifier2_record_stops_resume(tmp_path):
    ids = sorted(DEV)[:2]
    p = tmp_path / "records.jsonl"
    p.write_text(json.dumps({"id": ids[0], "runner": R.RUNNER_VERSION, "outcome": "accepted",
                             "verification": {"verifier": "sql2mql-v2-verifier-2"}}) + "\n", encoding="utf-8")
    with pytest.raises(R.IntegrityError, match="verifier-2"):
        R.load_records(p, set(ids))


# ------------------------------------------------------------------ sample integrity
def test_pilot_sample_is_the_frozen_300_and_disjoint():
    ids, rows = R.load_pilot()
    assert ids == FREEZE["pilot"]["ids"] and len(set(ids)) == 300
    assert not set(ids) & DEV and not set(ids) & R3_IDS and len(R3_IDS) == 740
    src = {}
    for x in (ROOT / R.TRAIN_REL).read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if r["id"] in PILOT:
            src[r["id"]] = r
    assert all(rows[i] == src[i] for i in ids)   # unchanged rows


def test_pilot_freeze_tamper_is_rejected(tmp_path):
    (tmp_path / "data/interim/finetune_v2").mkdir(parents=True)
    shutil.copy2(ROOT / R.PREREG_REL, tmp_path / R.PREREG_REL)
    fr = json.loads((ROOT / R.PILOT_FREEZE_REL).read_text(encoding="utf-8"))
    fr["pilot"]["ids"][0] = sorted(DEV)[0]
    (tmp_path / R.PILOT_FREEZE_REL).write_text(json.dumps(fr, indent=1), encoding="utf-8")
    with pytest.raises(R.IntegrityError, match="pilot freeze changed"):
        R.check_inputs(tmp_path)


def test_membership_counts_are_recomputed():
    m, _, _ = R.load_membership()
    from collections import Counter
    got = Counter(R.membership_of(i, m) for i in PILOT)
    assert dict(got) == {"train": 271, "quarantined": 1, "excluded": 4, "not_in_pool": 24}
    assert R.membership_of("spider_train_04586", m) == "quarantined"
    assert {i for i in PILOT if R.membership_of(i, m) == "excluded"} == {
        "spider_train_01259", "spider_train_02225", "spider_train_06102", "spider_train_06104"}


# ------------------------------------------------------------------ classification
def test_outcome_classification():
    cv = R.classify_verification
    assert cv({"status": "accepted"}) == ("accepted", None)
    assert cv({"status": "mongo_timeout"}) == ("timeout", "mongo_timeout")
    assert cv({"status": "sqlite_error", "detail": "interrupted"}) == ("timeout", "sqlite_interrupted")
    assert cv({"status": "sqlite_error", "detail": "no such column"}) == ("verification_failed", "untestable_precondition")
    assert cv({"status": "fixture_mismatch"}) == ("verification_failed", "semantic_mismatch")
    assert cv({"status": "v3_lookup_keys"}) == ("verification_failed", "structure_or_safety")
    assert cv({"status": "tie_cut"}) == ("verification_failed", "untestable_precondition")
    assert cv({"status": "something_new"}) == ("verification_failed", "other")


def test_three_or_more_table_rule():
    j3 = "SELECT a.x FROM a JOIN b ON a.id = b.aid JOIN c ON b.id = c.bid"
    union3 = "SELECT a.x FROM a JOIN b ON a.id = b.aid UNION SELECT c.x FROM c"
    sub3 = "SELECT x FROM a WHERE id IN (SELECT aid FROM b JOIN c ON b.id = c.bid)"
    two = "SELECT a.x FROM a JOIN b ON a.id = b.aid"
    S = MODS["round3_stats"]
    f = lambda sql, mql=None: R.multi_table_flags(S.stratum(sql), R.sql_profile(sql), mql)  # noqa: E731
    assert f(j3)["multi_table_3plus"] and f(j3)["j2_top_level_stratum"]
    assert S.stratum(union3) == "G" and f(union3)["multi_table_3plus"] and f(union3)["sql_distinct_tables_ge3"]
    assert f(sub3)["multi_table_3plus"] and not f(sub3)["j2_top_level_stratum"]
    assert not f(two)["multi_table_3plus"]
    mql = {"collection": "d__a", "pipeline": [{"$lookup": {}}, {"$unwind": "$x"}, {"$lookup": {}}]}
    assert f(two, mql)["mql_lookups_ge2"] and f(two, mql)["multi_table_3plus"]


def test_corpus_status_keeps_pilot_and_corpus_apart():
    no, yes = {"multi_table_3plus": False}, {"multi_table_3plus": True}
    assert R.corpus_status("accepted", "train", no) == ("candidate_pending_P2_review", [])
    assert R.corpus_status("accepted", "train", yes) == ("quarantined_multi_table_3plus", ["multi_table_3plus_quarantine"])
    for m in ("quarantined", "excluded", "not_in_pool", "val_internal"):
        assert R.corpus_status("accepted", m, no)[0] == "not_candidate"
    assert R.corpus_status("timeout", "train", no) == ("not_candidate", ["outcome:timeout"])


def test_read_only_safety_scan():
    EK = MODS["eval_kit"]
    allowed = {"d__a", "d__b"}
    ok = {"collection": "d__a", "pipeline": [{"$match": {"x": 1}}, {"$project": {"_id": 0, "x": 1}}]}
    assert R.runner_safety(ok, allowed, EK)["ok"]
    for bad in ({"collection": "d__a", "pipeline": [{"$match": {}}, {"$out": "z"}]},
                {"collection": "d__a", "pipeline": [{"$merge": {"into": "d__b"}}]},
                {"collection": "d__a", "pipeline": [{"$match": {"$where": "1"}}]},
                {"collection": "d__a", "pipeline": [{"$project": {"y": {"$function": {"body": "f", "args": [], "lang": "js"}}}}]},
                {"collection": "d__a", "pipeline": [{"$lookup": {"from": "other__t", "as": "o"}}]},
                {"collection": "zzz", "pipeline": [{"$match": {}}]}):
        assert not R.runner_safety(bad, allowed, EK)["ok"], bad


# ------------------------------------------------------------------ process_one with stubs (no MongoDB)
def _row(i="spider_train_99999"):
    return {"id": i, "db_id": "allergy_1", "hardness": "easy", "question": "q?", "sql": "SELECT count(*) FROM Student"}


def _stub_convert(monkeypatch, mql):
    def conv(r, profs):
        return {"id": r["id"], "db_id": r["db_id"], "hardness": r["hardness"], "question": r["question"], "sql": r["sql"],
                "stratum": "G", "converter": "x", "conversion": "converted", "features": [], "mql": mql}
    monkeypatch.setattr(MODS["run_v3"], "convert", conv)


GOOD = {"collection": "allergy_1__Student", "pipeline": [{"$count": "n"}, {"$project": {"_id": 0, "n": 1}}]}


def test_unsafe_mql_is_never_verified(monkeypatch):
    _stub_convert(monkeypatch, {"collection": "allergy_1__Student", "pipeline": [{"$out": "x"}]})
    called = []
    monkeypatch.setattr(MODS["verifier3"], "verify", lambda *a, **k: called.append(1))
    rec = R.process_one(_row(), MODS, None, {}, "train")
    assert rec["outcome"] == "safety_rejected" and not called and rec["corpus_status"] == "not_candidate"


def test_timeouts_and_verifier_errors_are_separate(monkeypatch):
    _stub_convert(monkeypatch, GOOD)
    V = R.EXPECTED_VERIFIER
    monkeypatch.setattr(MODS["verifier3"], "verify", lambda *a, **k: {"verifier": V, "status": "mongo_timeout", "checks": {}})
    assert R.process_one(_row(), MODS, None, {}, "train")["outcome"] == "timeout"
    monkeypatch.setattr(MODS["verifier3"], "verify",
                        lambda *a, **k: {"verifier": V, "status": "sqlite_error", "detail": "interrupted", "checks": {}})
    assert R.process_one(_row(), MODS, None, {}, "train")["outcome_detail"] == "sqlite_interrupted"

    def boom(*a, **k):
        raise ValueError("unexpected")
    monkeypatch.setattr(MODS["verifier3"], "verify", boom)
    rec = R.process_one(_row(), MODS, None, {}, "train")
    assert rec["outcome"] == "verifier_error" and rec["verification"]["status"] == "verifier_error"
    monkeypatch.setattr(MODS["verifier3"], "verify", lambda *a, **k: {"verifier": "sql2mql-v2-verifier-2", "status": "accepted"})
    with pytest.raises(R.IntegrityError):
        R.process_one(_row(), MODS, None, {}, "train")


def test_lost_mongodb_aborts_without_a_record(monkeypatch, tmp_path):
    from pymongo.errors import AutoReconnect
    _stub_convert(monkeypatch, GOOD)

    def lost(*a, **k):
        raise AutoReconnect("connection refused")
    monkeypatch.setattr(MODS["verifier3"], "verify", lost)
    ids = sorted(DEV)[:3]
    rows = {i: _row(i) for i in ids}
    with pytest.raises(R.InfrastructureAbort):
        R.execute(tmp_path, ids, rows, {}, MODS)
    assert R.load_records(tmp_path / "records.jsonl", set(ids)) == []


# ------------------------------------------------------------------ interruption / resume / duplicates
def _fake_process(r, mods, real, profs, membership):
    return {"id": r["id"], "db_id": r["db_id"], "runner": R.RUNNER_VERSION, "outcome": "conversion_rejected",
            "membership": membership}


def test_interrupted_run_resumes_without_duplicates_or_gaps(tmp_path):
    ids = sorted(DEV)[:6]
    rows = {i: _row(i) for i in ids}
    n = {"k": 0}

    def flaky(r, *a):
        n["k"] += 1
        if n["k"] == 4:
            raise KeyboardInterrupt
        return _fake_process(r, *a)
    with pytest.raises(KeyboardInterrupt):
        R.execute(tmp_path, ids, rows, {}, MODS, process=flaky)
    assert len(R.load_records(tmp_path / "records.jsonl", set(ids))) == 3
    with (tmp_path / "records.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"id": "spider_train_')                     # a cut append
    recs, done = R.execute(tmp_path, ids, rows, {}, MODS, process=_fake_process)
    assert done and sorted(r["id"] for r in recs) == ids
    log = [json.loads(x) for x in (tmp_path / "resume_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(x["cut_trailing_line_dropped"] == 1 for x in log)
    recs2, _ = R.execute(tmp_path, ids, rows, {}, MODS, process=lambda *a: pytest.fail("re-processed"))
    assert len(recs2) == 6


def test_budget_stops_cleanly(tmp_path):
    ids = sorted(DEV)[:4]
    recs, done = R.execute(tmp_path, ids, {i: _row(i) for i in ids}, {}, MODS, process=_fake_process, budget=-1)
    assert recs == [] and not done


def test_duplicates_and_foreign_ids_fail_closed(tmp_path):
    ids = sorted(DEV)[:2]
    p = tmp_path / "records.jsonl"
    line = json.dumps({"id": ids[0], "runner": R.RUNNER_VERSION, "outcome": "accepted"}) + "\n"
    p.write_text(line + line, encoding="utf-8")
    with pytest.raises(R.IntegrityError, match="duplicate"):
        R.load_records(p, set(ids))
    p.write_text(json.dumps({"id": "spider_dev_0001", "runner": R.RUNNER_VERSION, "outcome": "accepted"}) + "\n",
                 encoding="utf-8")
    with pytest.raises(R.IntegrityError, match="not a pilot id"):
        R.load_records(p, set(ids))
    p.write_text('{"broken\n' + line, encoding="utf-8")
    with pytest.raises(R.IntegrityError, match="not the last line"):
        R.load_records(p, set(ids))


# ------------------------------------------------------------------ outputs: accounting, manifest, reproducibility
def _synthetic(n=300):
    recs = []
    outs = ["accepted", "conversion_rejected", "verification_failed", "timeout", "accepted", "conversion_error"]
    mems = ["train"] * 9 + ["excluded"]
    for k in range(n):
        o = outs[k % len(outs)]
        mt = {"j2_top_level_stratum": k % 7 == 0, "sql_distinct_tables_ge3": k % 7 == 0, "mql_lookups_ge2": False}
        mt["multi_table_3plus"] = any(mt.values())
        r = {"id": f"x{k:03d}", "db_id": "d", "hardness": ["easy", "medium", "hard", "extra"][k % 4],
             "stratum": ["J2", "OL", "J1", "G"][k % 4], "conversion": "converted" if o != "conversion_rejected" else "unsupported",
             "code": "construct" if o == "conversion_rejected" else ("KeyError" if o == "conversion_error" else None),
             "outcome": o, "outcome_detail": {"timeout": "mongo_timeout", "verification_failed": "untestable_precondition",
                                              "accepted": None}.get(o, "c"),
             "membership": mems[k % 10], "multi_table": mt, "sql_profile": {"pattern": ["join"] if k % 2 else ["simple_select"]},
             "features": ["f"], "safety": {"ok": True}, "runner": R.RUNNER_VERSION}
        if o in ("accepted", "verification_failed", "timeout"):
            r["verification"] = {"verifier": R.EXPECTED_VERIFIER, "status": {"accepted": "accepted", "timeout": "mongo_timeout"}.get(o, "tie_cut"),
                                 "seconds": k / 10, "checks": {"l1_explicit_null": "multiset", "l1_missing": "multiset",
                                                               "v6_determinism": "x", "v8_validate_mql": "ok",
                                                               "v3_reorder_reversed": "multiset", "v3_reorder_shuffled_missing": "multiset",
                                                               "v3_lookup_keys": "ok", "fixtures": {"empty": "multiset"}}}
        r["corpus_status"], r["corpus_reasons"] = R.corpus_status(o, r["membership"], mt)
        recs.append(r)
    return recs


def test_outputs_account_for_every_example_and_are_reproducible(tmp_path):
    recs = _synthetic()
    ids = [r["id"] for r in recs]
    hashes = {k: {"actual": v["actual"]} for k, v in HASHES.items()}
    mans = []
    for d in ("a", "b"):
        out = tmp_path / d
        out.mkdir()
        C.write_jsonl(out / "records.jsonl", list(reversed(recs)) if d == "b" else recs)
        summ, man = R.write_outputs(out, recs if d == "a" else list(reversed(recs)), ids, {"x": 1}, hashes, {"e": 1}, C.wilson)
        mans.append(man)
        for name, meta in man["outputs"].items():
            assert R.sha_f(out / name) == meta["sha256"]
        assert summ["all"]["examples"] == 300 and sum(summ["all"]["outcomes"].values()) == 300
        assert sum(b["examples"] for b in summ["by_membership"].values()) == 300
        assert sum(b["examples"] for b in summ["by_hardness"].values()) == 300
        assert summ["multi_table_3plus"]["flagged"]["examples"] + summ["multi_table_3plus"]["not_flagged"]["examples"] == 300
        cls = C.read_jsonl(out / "classifications.jsonl")
        assert len(cls) == 300 and all(c["pilot_evaluation_included"] for c in cls)
        assert summ["gates"]["pilot_gate"].startswith("NOT PASSED")
        assert summ["gates"]["P1_equivalence_checks_on_accepted"]["status"] == "pass"
    assert mans[0]["outputs"] == mans[1]["outputs"]           # identical regardless of record order
    a = {r["id"]: R.outcome_key(r) for r in recs}
    rec2 = json.loads(json.dumps(recs))
    for r in rec2:
        if "verification" in r:
            r["verification"]["seconds"] = 999
        r["runner_seconds"] = 5
    assert a == {r["id"]: R.outcome_key(r) for r in rec2}
    rec2[0]["outcome"] = "timeout"
    assert a != {r["id"]: R.outcome_key(r) for r in rec2}


def test_summary_refuses_incomplete_accounting():
    recs = _synthetic()
    with pytest.raises(AssertionError):
        R.summarize(recs[:-1], [r["id"] for r in recs], C.wilson)


# ------------------------------------------------------------------ live: real conversion + verifier-3 on DEV items
@pytest.fixture(scope="module")
def live():
    try:
        client = C.local_client()
        C.check_server(client)
    except Exception:  # noqa: BLE001
        pytest.skip("local mongo:8.0.34 on 127.0.0.1:27018 not available")
    return MODS["verifier3"].RealDbs(client)


TRAIN = {}


def _train_row(i):
    if not TRAIN:
        for x in (ROOT / R.TRAIN_REL).read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            TRAIN[r["id"]] = r
    assert i in DEV and i not in PILOT and i not in R3_IDS
    return TRAIN[i]


@pytest.mark.parametrize("i,outcome,detail", [
    ("spider_train_00470", "accepted", None),                         # DEV, verifier-3 accepted in dev3
    ("spider_train_06799", "conversion_rejected", "not_a_single_select"),
    ("spider_train_06737", "verification_failed", "untestable_precondition"),   # tie_cut in dev3
])
def test_live_conversion_and_verifier3(live, i, outcome, detail):
    rec = R.process_one(_train_row(i), MODS, live, {}, "train")
    assert rec["outcome"] == outcome and rec["outcome_detail"] == detail
    assert rec["question"] == TRAIN[i]["question"] and rec["sql"] == TRAIN[i]["sql"] and rec["db_id"] == TRAIN[i]["db_id"]
    if "verification" in rec:
        assert rec["verification"]["verifier"] == R.EXPECTED_VERIFIER and rec["safety"]["ok"]
    if outcome == "accepted":
        assert rec["corpus_status"] == "candidate_pending_P2_review"


def test_live_three_table_dev_item_is_quarantined_not_dropped(live):
    rec = R.process_one(_train_row("spider_train_02889"), MODS, live, {}, "train")
    assert rec["stratum"] == "J2" and rec["multi_table"]["multi_table_3plus"]
    assert rec["outcome"] == "accepted" and rec["corpus_status"] == "quarantined_multi_table_3plus"


def test_live_rerun_is_identical(live):
    a = R.process_one(_train_row("spider_train_00470"), MODS, live, {}, "train")
    b = R.process_one(_train_row("spider_train_00470"), MODS, live, {}, "train")
    assert R.outcome_key(a) == R.outcome_key(b)
