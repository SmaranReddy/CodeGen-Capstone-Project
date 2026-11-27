"""MQL-0010 Kaggle package: integrity, registration preservation, view enforcement, mocked notebook execution
(kill / resume / two sessions) and mocked scoring + provisioning. No model, no GPU, no network, no database."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "kaggle/nl2mql_mql0010_holdout"
ZIP = ROOT / "kaggle/nl2mql_mql0010_holdout_bundle.zip"
NB = PKG / "nl2mql_mql0010_holdout.ipynb"
PRE = ROOT / "outputs/nl2mql_mql0010/preflight"
VENV_PY = Path.home() / "codegen_data/venv_tok/Scripts/python.exe"
TOK = Path.home() / "codegen_data/qwen3b_tok_488639f1"

pytestmark = pytest.mark.skipif(not (PKG / "package_manifest.json").exists(), reason="MQL-0010 package not built")
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(1, str(ROOT / "scripts/nl2mql_mql0010"))
import mql0010_core as C      # noqa: E402
import scoring as SC          # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def P():
    return C.load_package(PKG)


# ------------------------------------------------------------------ package integrity
def test_package_loads_and_every_file_is_hashed(P):
    assert len(P["manifest"]["files"]) >= 30 and P["manifest"]["contains_gold_labels"] is False
    sums = dict(reversed(l.split("  ", 1)) for l in (PKG / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())
    for name, h in sums.items():
        assert sha(PKG / name) == h, name
    assert set(sums) == set(P["manifest"]["files"]) | {"package_manifest.json"}


def test_zip_is_byte_identical_to_the_package_tree():
    with zipfile.ZipFile(ZIP) as z:
        names = {n.split("/", 1)[1]: n for n in z.namelist()}
        listed = set(json.loads((PKG / "package_manifest.json").read_text(encoding="utf-8"))["files"])
        assert set(names) == listed | {"package_manifest.json", "SHA256SUMS.txt"}
        for rel, n in names.items():
            assert z.read(n) == (PKG / rel).read_bytes(), rel


def test_registration_A4_preserved_in_package(P):
    idx = json.loads((PRE / "registration_index.json").read_text(encoding="utf-8"))
    assert P["cfg"]["registration_index_sha256"] == sha(PRE / "registration_index.json")
    for d in idx["documents"]:
        assert sha(PKG / "registration" / Path(d["path"]).name) == d["sha256"] == sha(ROOT / d["path"])
    for name, h in idx["source_sha256"].items():
        assert sha(PKG / "src" / name) == h == sha(ROOT / "scripts/nl2mql_mql0010/src" / name)
    for name in ("mql0008_core.py", "mql0009_core.py", "patterns.py", "retrieval.py", "rag_block.py"):
        assert (PKG / "src" / name).read_bytes() == (ROOT / "kaggle/nl2mql_mql0009_retrieval_control/src" / name).read_bytes()
    assert (PKG / "src/mql0010_core.py").read_bytes() == (ROOT / "scripts/nl2mql_mql0010/pkg_src/mql0010_core.py").read_bytes()


def test_config_matches_A4(P):
    cfg, a4 = P["cfg"], P["a4"]
    assert cfg["arms"] == a4["arms"] and [a["name"] for a in cfg["arms"]] == list(SC.ARMS)
    assert cfg["test_sequence"] == a4["effective_test_sequence"]
    m = cfg["model"]
    assert (m["id"], m["revision"], m["dtype"], m["max_new_tokens"], m["seed"]) == (C.MODEL_ID, C.MODEL_REVISION, "bfloat16", 1024, 42)
    assert (m["batch_size_first_pass"], m["batch_size_delta"], m["batch_size_retry"]) == (8, 1, 1)
    assert set(cfg["session_plan"]["1"]["passes"]) | set(cfg["session_plan"]["2"]["passes"]) == set(C.PASSES)
    assert len(cfg["delta_ids"]) == 52 and cfg["budget_drops"] == {"A0": 0, "D0": 0, "E": 0, "EF": 0}


def test_view_and_holdout_exclusion_enforced(P):
    hdb = set(P["holdout_dbs"])
    assert len(hdb) == 18 and len(P["view_rows"]) == 2912 and not {r["db_id"] for r in P["view_rows"]} & hdb
    for a in C.PLAN_ARMS:
        plan = C.read_jsonl(PKG / f"data/plans/plan_{a}.jsonl")
        excl = {(x["eval_id"], x["corpus_id"]) for x in P["cfg"]["retrieval_exclusions"]}
        for r in plan:
            assert not set(r["selected_db_ids"]) & hdb
            assert not {(r["id"], s) for s in r["selected_ids"]} & excl


def test_no_gold_in_package_inputs(P):
    for e in P["evals"]:
        assert set(e) == {"id", "db_id", "question"}
    text = (PKG / "data/eval/prompts.jsonl").read_text(encoding="utf-8").lower()
    assert "select " not in text.replace("selected", "")       # no SQL in prompts


def _tampered_copy(tmp_path, rel, mutate, rehash=False):
    dst = tmp_path / "pkg"
    shutil.copytree(PKG, dst)
    p = dst / rel
    p.write_bytes(mutate(p.read_bytes()))
    if rehash:
        man = json.loads((dst / "package_manifest.json").read_text(encoding="utf-8"))
        man["files"][rel]["sha256"] = sha(p)
        (dst / "package_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return dst


def test_tampering_is_refused(tmp_path):
    with pytest.raises(Exception):
        C.load_package(_tampered_copy(tmp_path / "a", "data/eval/prompts.jsonl", lambda b: b + b"\n"))
    # second layer: even with a consistent manifest, a changed registration copy or registered source is refused
    with pytest.raises(Exception, match="registration copy|registered source"):
        C.load_package(_tampered_copy(tmp_path / "b", "registration/amendment_A4_2026-11-19.json", lambda b: b + b" ", rehash=True))
    with pytest.raises(Exception, match="registered source"):
        C.load_package(_tampered_copy(tmp_path / "c", "src/recovery.py", lambda b: b + b"\n# x\n", rehash=True))
    with pytest.raises(Exception):     # corpus bytes changed -> view check fails
        C.load_package(_tampered_copy(tmp_path / "d", C.CORPUS_FILE, lambda b: b + b"\n", rehash=True))


def test_notebook_structure():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    tags = [c["metadata"]["tags"][0] for c in nb["cells"] if c["cell_type"] == "code"]
    assert tags == ["config", "package", "env", "model", "plan", "preflight", "generate", "export"]
    src = "".join("".join(c["source"]) for c in nb["cells"])
    for forbidden in ("pymongo", "MongoClient", "MONGODB_URI", "api_key", "Atlas("):
        assert forbidden not in src


# ------------------------------------------------------------------ mocked notebook execution + scoring
def _harness(out, session, resume=None, stop=None):
    cmd = [str(VENV_PY), "-I", "-B", str(ROOT / "tests/mql0010_nb_harness.py"), "--pkg", str(PKG), "--nb", str(NB),
           "--tok", str(TOK), "--out", str(out), "--session", str(session)]
    if resume:
        cmd += ["--resume", str(resume)]
    if stop is not None:
        cmd += ["--stop-after", str(stop)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)
    assert r.returncode == 0, r.stderr[-3000:]
    line = next(l for l in r.stdout.splitlines() if l.startswith("HARNESS_RESULT "))
    return json.loads(line[len("HARNESS_RESULT "):])


@pytest.mark.skipif(not VENV_PY.exists() or not TOK.exists(), reason="pinned tokenizer venv not available")
def test_mocked_notebook_kill_resume_two_sessions_then_mocked_scoring(tmp_path):
    r1 = _harness(tmp_path / "s1a", 1, stop=1)                      # killed after one batch
    assert r1["status"] == "partial" and r1["passes"]["first:A0"]["status"] == "partial"
    r2 = _harness(tmp_path / "s1b", 1, resume=tmp_path / "s1a")     # session 1 resumed to completion
    assert all(r2["passes"][p]["status"] == "complete" for p in ("first:A0", "first:D0", "first:E", "delta"))
    assert r2["status"] == "partial" and r2["passes"]["delta"]["n_expected"] == 104
    r3 = _harness(tmp_path / "s2", 2, resume=tmp_path / "s1b")      # session 2: retries
    assert r3["status"] == "complete" and r3["verify_exports"]["ok"]
    assert r3["verify_exports"]["retried"]["D0V"] == r3["verify_exports"]["retried"]["D0G"] > 0

    import mql0010_scoring as S

    class FakeClient:
        def __getitem__(self, name):
            return {"db": name}

        def close(self):
            pass

    def fake_ref(recs, K, db_dir):        # never computes or reads gold results
        return [{"id": r["id"], "db_id": r["db_id"], "reference_status": "ok"} for r in recs]

    def fake_eval(arm, preds, refs, mapping, mdb, K, out_dir, fp):
        assert mdb == {"db": "codegen_spider1_holdout_mql0010"}
        return {p["id"]: ("incorrect" if "__missing_field__" in p["raw_text"] else
                          "correct" if int(hashlib.sha256((p["id"] + p["raw_text"]).encode()).hexdigest()[:4], 16) % 2 else "incorrect")
                for p in preds}
    ana, man, run_dir = S.cmd_score([r3["zip"]], "mock_run", client_factory=lambda: (FakeClient(), None),
                                    preflight=lambda c, m, full_parity, log: {"ok": True, "mock": True},
                                    evaluate_arm=fake_eval, compute_reference=fake_ref, dest_root=tmp_path / "scoring_root",
                                    log=lambda *a: None)
    assert set(ana["accuracy"]) == set(SC.ARMS) and ana["step1"]["n"] == 716
    assert (run_dir / "scoring/analysis.json").exists() and (run_dir / "scoring/results_report.md").exists()
    assert man["registration_index_sha256"] == sha(PRE / "registration_index.json")
    # a second zip with different content for an arm is refused at verification (tamper inside the export)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(r3["zip"]) as zin, zipfile.ZipFile(bad, "w") as zout:
        for n in zin.namelist():
            data = zin.read(n)
            if n.endswith("predictions/D0G/predictions.jsonl"):
                data = data.replace(b'"kept": "first"', b'"kept": "retry"', 1)
            zout.writestr(n, data)
    with pytest.raises(Exception):
        S.cmd_verify([bad], dest_root=tmp_path / "scoring_root2")


# ------------------------------------------------------------------ provisioning (mocked; no network)
def test_provisioning_dry_run_and_target_guards():
    import provision_holdout_atlas as PH
    rep = PH.dry_run()
    assert (rep["target_db"], rep["databases"], rep["collections"], rep["documents"]) == ("codegen_spider1_holdout_mql0010", 18, 99, 1280)
    assert rep["sqlite_sources_verified"] == 99
    for bad in ("codegen_spider1", "codegen_spider1_heldout", "admin"):
        with pytest.raises(PH.PV9.ProvisionError):
            PH.PV9.check_target(bad)
    with pytest.raises(PH.PV9.ProvisionError):
        PH.PV9.check_target(PH.TARGET_DB, confirm=None, execute=True)
    for env in ("MONGODB_URI", "MQL0009_PROVISION_URI"):
        with pytest.raises(PH.PV9.ProvisionError):
            PH.PV9.make_client(env)
    with pytest.raises(PH.PV9.DestructiveOperationRefused):
        PH.PV9.GuardedClient(_Raw(), PH.TARGET_DB)["codegen_spider1"]
    g = PH.PV9.GuardedClient(_Raw(), PH.TARGET_DB)[PH.TARGET_DB]
    with pytest.raises(PH.PV9.DestructiveOperationRefused):
        g.drop_collection("x")
    with pytest.raises(PH.PV9.DestructiveOperationRefused):
        g["chinook_1__Album"].insert_many([{}])               # not an absent contract collection


class _RawDB:
    def __getitem__(self, name):
        return object()


class _Raw:
    def __getitem__(self, name):
        return _RawDB()

    def close(self):
        pass
