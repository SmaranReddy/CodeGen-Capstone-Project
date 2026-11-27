"""MQL-0015 preparation tests: package, runtime core, scorer, analysis, provisioning guards.
CPU only; fake model, fake merge, fake Atlas, fake MongoDB; writes only to pytest tmp dirs."""
import copy
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0015/src"))
sys.path.insert(1, str(ROOT / "scripts/nl2mql_mql0015"))
sys.path.insert(2, str(ROOT / "scripts/nl2mql_mql0015/preflight"))
import mql0015_common as CM          # noqa: E402
import mql0015_analysis as AN        # noqa: E402
import mql0015_scoring as SCO        # noqa: E402
import provision_mql0015_atlas as PR  # noqa: E402

PV9 = PR.PV9
T = CM.ATLAS_DB


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ frozen inputs untouched
def test_frozen_inputs_unchanged():
    assert _sha(CM.REGISTRATION) == CM.REGISTRATION_SHA256
    assert _sha(CM.SAMPLE_MANIFEST) == CM.SAMPLE_MANIFEST_SHA256
    assert _sha(CM.SAMPLE_IDS) == CM.SAMPLE_IDS_SHA256
    assert _sha(CM.PROMPT_PINS) == CM.PROMPT_PINS_SHA256
    assert _sha(CM.TEST_FULL) == CM.TEST_FULL_SHA256
    assert _sha(CM.MQL13_CONTRACT) == CM.MQL13_CONTRACT_SHA256
    assert _sha(ROOT / "kaggle/nl2mql_mql0013_test_full_bundle.zip") == CM.MQL13_BUNDLE_SHA256
    assert _sha(CM.CORPUS) == CM.CORPUS_SHA256


# ------------------------------------------------------------------ analysis
def test_mcnemar_and_holm():
    assert AN.mcnemar_exact(0, 0) == 1.0
    assert math.isclose(AN.mcnemar_exact(40, 22), 0.0302, abs_tol=5e-4)          # MQL-0014 primary
    h = AN.holm({"a": 0.01, "b": 0.04})
    assert h["a"]["reject"] and h["b"]["reject"] and math.isclose(h["a"]["p_holm"], 0.02)
    h = AN.holm({"a": 0.03, "b": 0.04})
    assert not h["a"]["reject"] and not h["b"]["reject"] and math.isclose(h["a"]["p_holm"], 0.06)
    h = AN.holm({"a": 0.001, "b": 0.6, "c": 0.02})
    assert h["a"]["reject"] and h["c"]["reject"] and not h["b"]["reject"] and h["b"]["p_holm"] >= h["c"]["p_holm"]


def test_compare_bootstrap_deterministic_and_clustered():
    ids = [f"i{k}" for k in range(200)]
    corr = {"x": {i: (k % 3 != 0) for k, i in enumerate(ids)}, "y": {i: (k % 2 == 0) for k, i in enumerate(ids)}}
    cl = {i: f"c{k // 2}" for k, i in enumerate(ids)}
    a = AN.compare(corr, ids, "x", "y", reps=500, clusters=cl)
    b = AN.compare(corr, ids, "x", "y", reps=500, clusters=cl)
    assert a == b and a["n_clusters"] == 100 and a["ci95_pp"][0] <= a["diff_pp"] <= a["ci95_pp"][1]


def test_taxonomy_maps_every_category():
    cases = {("correct", None, None): "correct", ("incorrect", "model", None): "incorrect",
             ("invalid", "model", "unsafe_mql"): "invalid", ("timeout", "model", "timeout"): "timeout",
             ("failed", "model", "execution_error"): "execution_failure",
             ("failed", "infrastructure", "atlas_unavailable"): "infrastructure",
             ("failed", "infrastructure", "generation_error"): "generation_failure",
             ("not_comparable", None, None): "not_comparable", ("weird", None, None): "not_comparable"}
    for (o, s, d), want in cases.items():
        assert AN.taxonomy({"outcome": o, "failure_source": s, "detail": d}) == want


def test_registered_subsets_on_real_metadata():
    meta, near_dup, toks, clusters, pop = SCO.metadata()
    ids = CM.sample_ids()
    sub = AN.subsets(meta, ids, near_dup, toks)
    assert {k: len(v) for k, v in sub.items()} == {"degenerate_gold": 26, "ft_l5_partner": 5, "near_duplicate_11": 11,
                                                   "question_in_train": 7, "r3_prompt_over_2048": 144}
    assert sub["ft_l5_partner"] <= sub["near_duplicate_11"]
    assert math.isclose(sum(pop.values()), 1.0) and len(set(clusters.values())) == 845
    corr = {a: {i: True for i in ids} for a in AN.REGISTERED}
    ps = AN.post_stratified(corr, ids, meta, pop)
    assert all(v == 100.0 for v in ps["accuracy_pct"].values())
    assert math.isclose(sum(c["sample_share"] * c["weight"] for c in ps["cells"].values()), 1.0, abs_tol=1e-3)


# ------------------------------------------------------------------ package
def test_real_package_loads_and_pins():
    C = SCO._core(CM.PKG_OUT)
    P = C.load_package(CM.PKG_OUT)
    assert P["ids"] == CM.sample_ids()
    cfg = P["cfg"]
    assert cfg["registration_sha256"] == CM.REGISTRATION_SHA256 and cfg["evaluation"]["ids_sha256"] == CM.SAMPLE_IDS_SHA256
    assert cfg["model"]["revision"] == CM.MODEL_REVISION and cfg["ft"]["adapter_sha256"] == CM.ADAPTER_SHA256
    assert cfg["retrieval"]["index"] == CM.R3_INDEX and cfg["retrieval"]["corpus_sha256"] == CM.CORPUS_SHA256
    assert cfg["retrieval"]["view_ids_sha256"] == CM.VIEW_IDS_SHA256 and cfg["scoring_contract"]["atlas_db"] == T
    assert cfg["analysis_sha256"]["scripts/nl2mql_mql0015/src/mql0015_analysis.py"] == _sha(ROOT / "scripts/nl2mql_mql0015/src/mql0015_analysis.py")
    adapters = list(CM.PKG_OUT.rglob("adapter_model.safetensors"))
    assert len(adapters) == 1 and _sha(adapters[0]) == CM.ADAPTER_SHA256
    for line in (CM.PKG_OUT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, n = line.split("  ", 1)
        assert _sha(CM.PKG_OUT / n) == h, n


@pytest.fixture()
def pkg_copy(tmp_path):
    dst = tmp_path / CM.PACKAGE
    shutil.copytree(CM.PKG_OUT, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def _refresh_manifest(pkg):
    man = json.loads((pkg / "package_manifest.json").read_text(encoding="utf-8"))
    for n in man["files"]:
        man["files"][n]["sha256"] = _sha(pkg / n)
    (pkg / "package_manifest.json").write_text(json.dumps(man), encoding="utf-8")


@pytest.mark.parametrize("tamper", ["prompt", "adapter", "ids", "gold_field", "extra_file"])
def test_package_tampering_fails_closed(pkg_copy, tamper):
    C = SCO._core(pkg_copy)
    if tamper == "prompt":
        f = pkg_copy / "data/prompts/prompts_R3.jsonl"
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
        rows[5]["messages"][0]["content"] += " "
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        _refresh_manifest(pkg_copy)
    elif tamper == "adapter":
        f = pkg_copy / "model/ft_adapter/adapter_model.safetensors"
        b = bytearray(f.read_bytes()[:4096])
        f.write_bytes(bytes(b) + b"x")
        _refresh_manifest(pkg_copy)
    elif tamper == "ids":
        f = pkg_copy / "data/eval/eval_inputs.jsonl"
        lines = f.read_text(encoding="utf-8").splitlines()
        f.write_text("\n".join(lines[1:] + lines[:1]) + "\n", encoding="utf-8")
        _refresh_manifest(pkg_copy)
    elif tamper == "gold_field":
        f = pkg_copy / "data/eval/eval_inputs.jsonl"
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
        rows[0]["sql"] = "SELECT 1"
        f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        _refresh_manifest(pkg_copy)
    else:
        (pkg_copy / "model/ft_adapter/other_adapter.safetensors").write_bytes(b"x")
    with pytest.raises(Exception):
        C.load_package(pkg_copy)


# ------------------------------------------------------------------ full mocked pipeline (fake model / merge / Atlas)
@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    import dry_run as DR
    d = tmp_path_factory.mktemp("mql0015_dry")
    DR.DRY = d
    DR.CALLS.clear()
    DR.INFRA_ONCE.clear()
    DR.INFRA_ALWAYS.clear()
    rep = DR.main()
    return DR, d, rep


def test_pipeline_resume_retries_failures_sharing(pipeline):
    DR, d, rep = pipeline
    assert rep["session1_killed"]["status"] == "partial" and rep["session1_resumed"]["status"] == "partial"
    assert rep["session2_ft"]["status"] == "complete" and rep["session1_resumed"]["seeded"] > 0
    ver = rep["session2_ft"]["verify_exports"]
    assert ver["ok"] and ver["n"] == 1000
    assert all(ver["generation_failures"][a] == ver["generation_failures"][a + "V"] for a in ("A0", "R3", "FT-R3"))
    assert ver["retry_generation_failed"]["FT-R3V"] > 0 and ver["retry_generation_failed"]["R3V"] == 0
    assert rep["session1_resumed"]["oom_events"]["batch"] > 0
    assert rep["scoring"]["shared_rows"] > 0 and rep["scoring"]["unique_slots"] < 6000
    assert rep["package"]["gold_hits"] == 0
    tax = rep["taxonomy"]
    for cat in ("correct", "incorrect", "invalid", "execution_failure", "timeout", "generation_failure", "infrastructure",
                "not_comparable"):
        assert sum(tax[a][cat] for a in tax) > 0, cat
    assert rep["subset_sizes"]["near_duplicate_11"] == 11 and rep["sensitivity_n"]["exclude_near_duplicate_11"] == 989
    inf = rep["scoring"]["infrastructure"]
    assert inf["rounds"][0]["slots"] > inf["rounds"][0]["still_infra"] and len(inf["residual_items"]) <= 10


def test_pipeline_package_is_deterministic(pipeline):
    _, _, rep = pipeline
    assert rep["package"]["zip_sha256"] == _sha(CM.ZIP_OUT)


def test_exports_match_registered_ids_and_weights(pipeline):
    _, d, _ = pipeline
    out = d / "kaggle_out_s2" / "mql0015_outputs"
    ids = CM.sample_ids()
    for a in CM.ARMS:
        recs = CM.read_jsonl(out / "predictions" / a / "predictions.jsonl")
        assert [r["id"] for r in recs] == ids
        if a.startswith("FT"):
            assert all(r["weights"]["weights"] == "ft" and r["weights"]["adapter_sha256"] == CM.ADAPTER_SHA256 for r in recs)
        else:
            assert all(r["weights"]["weights"] == "base" for r in recs)
    r3 = {r["id"]: r["prompt_sha256"] for r in CM.read_jsonl(out / "predictions/R3/predictions.jsonl")}
    ft = {r["id"]: r["prompt_sha256"] for r in CM.read_jsonl(out / "predictions/FT-R3/predictions.jsonl")}
    assert r3 == ft


def test_scorer_stops_on_residual_infrastructure(pipeline):
    DR, d, _ = pipeline
    zipf = str(d / "kaggle_out_s2" / "mql0015_outputs") + ".zip"

    def always_infra(slots, refs, mapping, mdb, K, out_dir, fp):
        return {s["id"]: {"id": s["id"], "outcome": "failed", "failure_source": "infrastructure", "detail": "atlas_unavailable"}
                for s in slots}
    with pytest.raises(CM.MQL0015Error, match="remain infrastructure"):
        SCO.cmd_score([zipf], "infra_stop", pkg=d / "kaggle" / CM.PACKAGE, client_factory=lambda: (DR.FakeClient(), str),
                      preflight=lambda c, m, **k: {"ok": True, "atlas_db": m["atlas_db"]}, evaluate=always_infra, dest_root=d,
                      contract_path=d / "scoring_contract/atlas_mapping.json", reps=50, log=lambda *a: None)
    assert not (d / "scoring_runs/infra_stop/scoring/analysis.json").exists()


def test_scorer_refuses_other_database_contract(pipeline, tmp_path):
    DR, d, _ = pipeline
    bad = json.loads((d / "scoring_contract/atlas_mapping.json").read_text(encoding="utf-8"))
    bad["atlas_db"] = "codegen_spider1_test_mql0013"
    p = tmp_path / "c.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    C = SCO._core(d / "kaggle" / CM.PACKAGE)
    P = C.load_package(d / "kaggle" / CM.PACKAGE)
    with pytest.raises(CM.MQL0015Error):
        SCO.load_contract(P, p)


def test_resume_source_tamper_refused(pipeline, tmp_path):
    _, d, _ = pipeline
    src = d / "kaggle_out_s1b" / "mql0015_outputs"
    cp = tmp_path / "prev"
    shutil.copytree(src, cp)
    f = next((cp / "passes").rglob("checkpoint.jsonl"))
    f.write_text(f.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    C = SCO._core(d / "kaggle" / CM.PACKAGE)
    P = C.load_package(d / "kaggle" / CM.PACKAGE)
    with pytest.raises(Exception, match="hash check failed"):
        C.verify_resume_source(cp, P["manifest_sha256"])


def test_pass_config_refuses_mixed_weights(tmp_path):
    C = SCO._core(CM.PKG_OUT)
    C._write_pass_config(tmp_path, {"pass": "first:FT-R3", "weights_info": {"merged_weights_sha256": "a"}, "sessions": []})
    with pytest.raises(Exception, match="refusing to mix runs"):
        C._write_pass_config(tmp_path, {"pass": "first:FT-R3", "weights_info": {"merged_weights_sha256": "b"}, "sessions": []})


def test_attempt_policy():
    C = SCO._core(CM.PKG_OUT)
    assert C.is_retryable({"status": "error", "attempt": 1})
    assert not C.is_retryable({"status": "error", "attempt": 2})
    assert not C.is_retryable({"status": "ok", "attempt": 1})


# ------------------------------------------------------------------ provisioning guards (fake MongoDB, synthetic SQLite)
class _Cursor(list):
    def close(self):
        pass


class FakeColl:
    def __init__(self, srv, db, name):
        self.srv, self.db, self.name = srv, db, name

    def _docs(self):
        return self.srv.data.setdefault(self.db, {}).setdefault(self.name, [])

    def insert_many(self, docs, ordered=True):
        self.srv.ops.append(("insert_many", self.db, self.name, len(docs)))
        self._docs().extend(dict(x) for x in docs)

    def find(self, q=None, proj=None, batch_size=None):
        self.srv.ops.append(("find", self.db, self.name))
        return _Cursor({k: v for k, v in x.items() if k != "_id"} for x in self._docs())

    def count_documents(self, q):
        return len(self._docs())

    def __getattr__(self, a):
        def destructive(*args, **kw):
            self.srv.ops.append(("DESTRUCTIVE:" + a, self.db, self.name))
        return destructive


class FakeDB:
    def __init__(self, srv, name):
        self.srv, self.name = srv, name

    def list_collection_names(self):
        return [n for n, v in self.srv.data.get(self.name, {}).items() if v]

    def __getitem__(self, n):
        return FakeColl(self.srv, self.name, n)

    def command(self, cmd):
        return {"authInfo": {"authenticatedUsers": [{"user": "mql0015_provisioner", "db": "admin"}],
                             "authenticatedUserPrivileges": self.srv.privileges}}


class _Admin:
    def command(self, c):
        return {"ok": 1} if c == "ping" else {"version": "8.0.fake"}


class FakeMongo:
    def __init__(self, privileges=None):
        self.data, self.ops, self.admin = {}, [], _Admin()
        self.privileges = privileges if privileges is not None else minimal_role()

    def __getitem__(self, n):
        self.ops.append(("db", n))
        return FakeDB(self, n)

    def close(self):
        pass

    def databases_touched(self):
        return {o[1] for o in self.ops if o[0] in ("db", "insert_many", "find")}


def minimal_role(db=T):
    return [{"resource": {"db": db, "collection": ""}, "actions": ["find", "insert", "listCollections", "createCollection"]}]


SCHEMA = {"alpha_db": {"owner": ("CREATE TABLE owner (id INTEGER PRIMARY KEY, name TEXT)", [(1, "Ann"), (2, "Bo")])},
          "beta_db": {"item": ("CREATE TABLE item (code TEXT, qty INTEGER)", [("A1", 5), ("B2", 0)])}}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    base = tmp_path_factory.mktemp("spider15")
    for db, tabs in SCHEMA.items():
        (base / "database" / db).mkdir(parents=True)
        con = sqlite3.connect(base / "database" / db / f"{db}.sqlite")
        for ddl, rows in tabs.values():
            con.execute(ddl)
            con.executemany(f"INSERT INTO {ddl.split()[2]} VALUES ({','.join('?' * len(rows[0]))})", rows)
        con.commit()
        con.close()
    sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0009"))
    sys.path.insert(0, str(ROOT / "scripts/spider1"))
    import spider1_kit as K
    import build_mql0009_package as B
    saved = K.DB_DIR
    try:
        _, _, contract = B.derive_context(list(SCHEMA), db_dir=base / "database")
    finally:
        K.DB_DIR = saved
    return {"bundle": base, "contract": contract}


def test_target_name_enforced_everywhere():
    PR.check_target(T)
    for bad in sorted(CM.PROTECTED_DBS) + ["codegen_spider1_test_mql0015x", "", "CODEGEN_SPIDER1_TEST_MQL0015"]:
        with pytest.raises(PR.ProvisionError):
            PR.check_target(bad)
        assert PR.main(["--target-db", bad]) == 2
    with pytest.raises(PR.ProvisionError):
        PR.check_target(T, confirm=None, execute=True)
    with pytest.raises(PR.ProvisionError):
        PR.check_target(T, confirm="codegen_spider1_test_mql0013", execute=True)
    assert PV9.TARGET_DB == "codegen_spider1_heldout", "PR must not leak its binding into the MQL-0009 module"


def test_mislabel_root_cause_reproduced_and_fixed():
    contract = PR.load_contract()
    saved = PV9.TARGET_DB
    try:
        PV9.TARGET_DB = "codegen_spider1_test_mql0013"           # what the MQL-0013 wrapper did
        rows = PV9.build_mapping(contract)["rows"]               # what PV9.dry_run() then called (no target_db)
        assert {r["physical_db"] for r in rows} == {"codegen_spider1_heldout"}
    finally:
        PV9.TARGET_DB = saved
    mp = PR.build_mapping(contract)
    assert {r["physical_db"] for r in mp["rows"]} == {T} and len(mp["rows"]) == 180


def test_real_dry_run_report_targets_only_mql0015():
    reps = sorted(CM.PROVISION_DIR.glob("dry_run_*.json"))
    assert reps, "run the provisioning dry run first"
    rep = json.loads(reps[-1].read_text(encoding="utf-8"))
    PR.assert_report_targets(rep)
    assert rep["collections"] == 180 and rep["documents"] == 245574 and rep["sqlite_sources_verified"] == 180
    assert {r["physical_db"] for r in rep["mapping"]} == {T}


def test_contract_copy_differs_only_in_target_fields():
    c15 = json.loads(CM.CONTRACT.read_text(encoding="utf-8"))
    c13 = json.loads(CM.MQL13_CONTRACT.read_text(encoding="utf-8"))
    assert sorted(k for k in set(c13) | set(c15) if c13.get(k) != c15.get(k)) == ["atlas_db", "status"]
    assert c15["atlas_db"] == T and c13["atlas_db"] == "codegen_spider1_test_mql0013"


def test_guard_refuses_other_databases():
    srv = FakeMongo()
    with PR._bound():
        g = PV9.GuardedClient(srv, T, insertable={"a__t"})
        for other in ("codegen_spider1_test_mql0013", "codegen_spider1_holdout_mql0010", "codegen_spider1_heldout",
                      "codegen_spider1", "admin"):
            with pytest.raises(PV9.DestructiveOperationRefused):
                g[other]
        for f in (lambda: g.drop_database(T), lambda: g[T]["a__t"].drop(), lambda: g[T]["a__t"].delete_many({}),
                  lambda: g[T]["a__t"].update_one({}, {}), lambda: g[T]["a__t"].aggregate([{"$out": "x"}])):
            with pytest.raises(PV9.DestructiveOperationRefused):
                f()
    assert PV9.TARGET_DB == "codegen_spider1_heldout"


def test_provision_writes_only_target_and_refuses_bad_privileges(synth):
    srv = FakeMongo()
    rep = PR.provision(srv, synth["contract"], synth["bundle"], log=None)
    assert rep["status"] == "complete" and rep["target_db"] == T
    assert srv.databases_touched() == {T}
    assert not [o for o in srv.ops if str(o[0]).startswith("DESTRUCTIVE")]
    rep2 = PR.provision(srv, synth["contract"], synth["bundle"], log=None)        # idempotent: verify only
    assert rep2["written"] == [] and rep2["preflight_states"]["verified_existing"] == len(synth["contract"]["collections"])
    for privs in (minimal_role() + [{"resource": {"db": "codegen_spider1_test_mql0013", "collection": ""}, "actions": ["insert"]}],
                  minimal_role() + [{"resource": {"db": "codegen_spider1_holdout_mql0010", "collection": ""}, "actions": ["remove"]}],
                  [{"resource": {"db": T, "collection": ""}, "actions": ["find", "insert", "listCollections", "dropCollection"]}],
                  [{"resource": {"db": T, "collection": ""}, "actions": ["find"]}]):
        s = FakeMongo(privs)
        with pytest.raises(PR.ProvisionError):
            PR.provision(s, synth["contract"], synth["bundle"], log=None)
        assert not [o for o in s.ops if o[0] == "insert_many"]


def test_provision_conflict_aborts_before_write(synth):
    srv = FakeMongo()
    name = sorted(synth["contract"]["collections"])[0]
    srv.data[T] = {name: [{"bogus": 1}]}
    with pytest.raises(PR.ProvisionError, match="conflict"):
        PR.provision(srv, synth["contract"], synth["bundle"], log=None)
    assert not [o for o in srv.ops if o[0] == "insert_many"]


def test_readiness_refuses_provisioning_user_and_accepts_find_only(synth):
    import dev1034_core as D
    srv = FakeMongo()
    PR.provision(srv, synth["contract"], synth["bundle"], log=None)               # provisioned fake target
    mapping = {**synth["contract"], "atlas_db": T}
    with pytest.raises(D.PreflightError, match="NOT read-only"):                   # provisioning role holds insert
        SCO.readiness(srv, mapping, log=lambda *a: None)
    srv.privileges = [{"resource": {"db": T, "collection": ""}, "actions": ["find", "listCollections", "killCursors"]}]
    rep = SCO.readiness(srv, mapping, log=lambda *a: None)
    assert rep["atlas_db"] == T and rep["collections_found"] == len(mapping["collections"]) and not rep["offending_actions"]
    srv.privileges = [{"resource": {"db": "codegen_spider1_test_mql0013", "collection": ""}, "actions": ["find"]}]
    with pytest.raises(D.PreflightError, match="no 'find'"):
        SCO.readiness(srv, mapping, log=lambda *a: None)
    with pytest.raises(CM.MQL0015Error):
        SCO.readiness(srv, {**mapping, "atlas_db": "codegen_spider1_test_mql0013"}, log=lambda *a: None)


def test_scoring_code_is_pinned_in_package():
    C = SCO._core(CM.PKG_OUT)
    P = C.load_package(CM.PKG_OUT)
    pins = P["cfg"]["scoring_code_sha256"]
    assert set(pins) == {"scripts/nl2mql_mql0015/mql0015_scoring.py", "scripts/nl2mql_mql0015/src/mql0015_common.py",
                         "scripts/nl2mql/dev1034_core.py"}
    SCO.load_expected(CM.PKG_OUT)                                                  # all pins match the files on disk


def test_merge_smoke_if_peft_available(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    import merge_smoke_test as MS
    ok = MS.main(tmp_path)
    assert ok["deterministic_merge"] and ok["refuses_wrong_adapter"] and ok["changed_tensors"] == "14/14"


def test_make_client_requires_mql0015_credentials(monkeypatch):
    for v in ("MQL0015_PROVISION_URI", "MONGODB_URI", "MQL0013_PROVISION_URI"):
        monkeypatch.delenv(v, raising=False)
    for bad in ("MONGODB_URI", "MQL0013_PROVISION_URI", "MQL0010_PROVISION_URI", "MQL0009_PROVISION_URI"):
        with pytest.raises(PR.ProvisionError):
            PR.make_client(bad)
    with pytest.raises(PR.ProvisionError, match="needs the MQL-0015"):
        PR.make_client("MQL0015_PROVISION_URI")
    monkeypatch.setenv("MQL0015_PROVISION_URI", "mongodb+srv://u:p@x.example.net/")
    monkeypatch.setenv("MONGODB_URI", "mongodb+srv://u:p@x.example.net/")
    with pytest.raises(PR.ProvisionError, match="separate MQL-0015 credentials"):
        PR.make_client("MQL0015_PROVISION_URI")
