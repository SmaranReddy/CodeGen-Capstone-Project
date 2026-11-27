"""MQL-0015 Kaggle orchestrator tests: a mocked Kaggle API, no network, no real Kaggle run, no credentials.

The fake Kaggle mimics the behaviour the launcher depends on: per-version status sequences (incl. a stale "complete"
of the previous version right after a push), output listing of the latest finished version, private datasets whose
uploaded zip is (or, in a failure test, is not) extracted, push responses with dropped sources. Its notebook runner
reads SESSION / RESUME_FROM from the pushed notebook and builds the session-2 output from the files of the attached
handoff zip (as the real notebook's seed_from_previous would)."""
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0015"))
import launch_mql0015_sessions as L     # noqa: E402

FAKE_KEY = "FAKEKEY0123456789abcdef"     # must never appear in any state/log/staging file
ADAPTER = "7096760ac2ed7e4a6c2eefdd7c0c249d4b3dba723f069982a499b876d434e09d"


# ------------------------------------------------------------------ fake export builder
def _h(b):
    return hashlib.sha256(b).hexdigest()


def build_output(session, passes_done, seeded=None, resumed=None, gpu="Tesla T4", problems=(), env_ok=True,
                 ft_weights=None, final=False, ran=None, tamper=False, extra_pkg_manifest=False):
    """-> zip bytes of a mql0015_outputs/ export."""
    files = dict(seeded or {})
    for p in passes_done:
        d = f"passes/{L.PASS_DIR[p]}/"
        files.setdefault(d + "checkpoint.jsonl", f"{p} records\n".encode())
        files.setdefault(d + "pass_config.json", json.dumps({"pass": p}).encode())
    pst = {p: {"status": "complete" if p in passes_done else ("partial" if p in (ran or []) else "absent")} for p in L.ALL_PASSES}
    man = {"experiment": "MQL-0015", "status": "complete" if final else "partial", "passes": pst,
           "package_manifest_sha256": L.PACKAGE_MANIFEST_SHA256,
           "per_arm": {a: {"n": 1000, "predictions_sha256": "x"} for a in L.ARMS} if final else {},
           "verify_exports": {"ok": True, "n": 1000} if final else None}
    st = {"experiment": "MQL-0015", "package": "nl2mql_mql0015_final_1000", "status": "complete" if final else "partial",
          "problems": list(problems), "passes": pst, "environment_consistent": env_ok, "weights_consistent": True,
          "package_manifest_sha256": L.PACKAGE_MANIFEST_SHA256, "session": session,
          "passes_this_session": list(ran if ran is not None else [p for p in L.SESSION_PLAN[session] if p in passes_done]),
          "resumed": resumed, "stopped": None if set(L.SESSION_PLAN[session]) <= set(passes_done) else "time budget",
          "environment": {"gpu": {"device0": gpu, "total_gib": 14.74}},
          "weights_this_session": ({"ft": ft_weights} if ft_weights else {"base": {"weights": "original"}})}
    files[L.STATUS_FILE] = json.dumps(st).encode()
    files[L.MANIFEST_FILE] = json.dumps(man).encode()
    if extra_pkg_manifest:
        files["package_manifest.json"] = b"{}"
    sums = "".join(f"{_h(b)}  {n}\n" for n, b in sorted(files.items()))
    if tamper:
        files[L.STATUS_FILE] = files[L.STATUS_FILE] + b" "
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, b in sorted(files.items()):
            z.writestr(L.OUT_PREFIX + n, b)
        z.writestr(L.OUT_PREFIX + "SHA256SUMS.txt", sums)
    return buf.getvalue()


def session1_files(zb):
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        return {n[len(L.OUT_PREFIX):]: z.read(n) for n in z.namelist() if n.startswith(L.OUT_PREFIX + "passes/")}


FT_OK = {"adapter_sha256": ADAPTER, "changed_tensors": "252/252", "decoding_files_sha256_equal_to_base": {"g": "x"}}


# ------------------------------------------------------------------ fake Kaggle
class FakeKaggle:
    username = "tester"

    def __init__(self, scenario=None, extract=True, prefix=""):
        self.kernels, self.datasets, self.blobs, self.pushes, self.creates = {}, {}, {}, [], []
        self.staged_codes, self.fail_create_once = [], set()
        self.scenario = scenario or Scenario()
        self.extract, self.prefix = extract, prefix

    # kernels
    def kernel_status(self, ref):
        k = self.kernels.get(ref)
        if not k:
            raise L.NotFound(ref)
        v = k[-1]
        s = v["seq"][min(v["ptr"], len(v["seq"]) - 1)]
        v["ptr"] += 1
        return s, ("boom" if s == "error" else None)

    def push_kernel(self, folder):
        folder = Path(folder)
        meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
        raw = (folder / meta["code_file"]).read_bytes()
        assert all(b < 128 for b in raw), "push payload must be pure ASCII (kaggle reads it with the Windows default)"
        code = raw.decode("cp1252")                       # exactly what kaggle 1.7.4.5 open() would read on Windows
        self.staged_codes.append(code)
        for f in folder.rglob("*"):
            assert FAKE_KEY not in f.read_text(encoding="utf-8", errors="ignore")
        ref = meta["id"]
        for ds in meta["dataset_sources"]:
            assert ds in self.datasets, f"push references unknown dataset {ds}"
        vers = self.kernels.setdefault(ref, [])
        resp = {"error": None, "version": len(vers) + 1, "url": f"https://kaggle/{ref}/v{len(vers) + 1}", "ref": ref,
                "invalid_dataset_sources": [], "invalid_kernel_sources": [], "invalid_model_sources": [],
                "invalid_competition_sources": []}
        seq, outputs, override = self.scenario.on_push(self, ref, meta, code, len(vers))
        resp.update(override or {})
        if vers and self.scenario.stale_first:
            seq = ["complete"] + seq                  # status still reports the previous version for one poll
        vers.append({"meta": meta, "code": code, "seq": seq, "ptr": 0, "outputs": outputs})
        self.pushes.append((ref, meta, code))
        return resp

    def list_output(self, ref):
        vers = self.kernels[ref]
        cur = vers[-1]
        finished = cur["ptr"] >= len(cur["seq"])          # its terminal status has been reported
        src = cur if finished else (vers[-2] if len(vers) > 1 else None)
        if src is None:
            return [], None
        files = []
        for n, b in src["outputs"].items():
            url = f"fake://{ref}/{id(src)}/{n}"
            self.blobs[url] = b
            files.append((n, url))
        return files, "log text"

    def download(self, url, dest):
        b = self.blobs[url]
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b)
        return _h(b)

    # datasets
    def dataset_status(self, ref):
        d = self.datasets.get(ref)
        if not d:
            raise L.NotFound(ref)
        s = d["seq"][min(d["ptr"], len(d["seq"]) - 1)]
        d["ptr"] += 1
        return s

    def create_dataset(self, folder):
        folder = Path(folder)
        meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
        if meta["id"] in self.fail_create_once:
            self.fail_create_once.discard(meta["id"])
            raise RuntimeError("transient upload failure")
        zips = [p for p in folder.iterdir() if p.suffix == ".zip"]
        assert len(zips) == 1
        b = zips[0].read_bytes()
        self.add_dataset(meta["id"], b, zips[0].name)
        self.creates.append(meta["id"])
        return {"error": None, "ref": meta["id"], "url": f"https://kaggle/{meta['id']}"}

    def add_dataset(self, ref, zip_bytes, zip_name):
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            ents = [(self.prefix + i.filename, i.file_size) for i in z.infolist() if not i.filename.endswith("/")]
        files = ents if self.extract else [(zip_name, len(zip_bytes))]
        self.datasets[ref] = {"seq": ["pending", "ready"], "ptr": 0, "files": files, "zip": zip_bytes}

    def dataset_files(self, ref):
        return list(self.datasets[ref]["files"])


class Scenario:
    """Default: every run succeeds. Override fields to inject failures."""

    def __init__(self, probe_devices=("Tesla T4", "Tesla T4"), peft="0.13.0", s1_partial=False, s1_status="complete",
                 s2_status="complete", stale_first=False, invalid_sources=None, s2_wrong_session=False, s2_corrupt_seed=False,
                 s2_bad_adapter=False, s2_env_mismatch=False, tamper=None, missing_zip=None, s1_partial_then_ok=False,
                 hf_ok=True):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def on_push(self, fk, ref, meta, code, n_prev):
        if ref.endswith(L.PROBE_SLUG):
            nonce = re.search(r'"nonce": "([0-9a-f]+)"', code).group(1)
            pkg_dir = [d for d in meta["dataset_sources"] if d.endswith(L.BUNDLE_DATASET_SLUG)]
            out = {"probe": "MQL-0015", "nonce": nonce, "devices": [{"name": d, "total_gib": 14.74} for d in self.probe_devices],
                   "packages": {"torch": "2.5", "transformers": "4.46", "peft": self.peft},
                   "package_manifests": [{"dir": "/kaggle/input/" + pkg_dir[0].split("/")[1] + "/" + "nl2mql_mql0015_final_1000",
                                          "package": "nl2mql_mql0015_final_1000"}],
                   "status_files": [], "input_files": 28}
            rev = re.search(r'revision="([0-9a-f]+)"', code).group(1)
            assert meta["enable_internet"] is True
            out.update({"hf_model_info_sha": rev} if self.hf_ok else {"hf_error": "ConnectionError"})
            return ["queued", "running", "complete"], {L.PROBE_FILE: json.dumps(out).encode()}, None
        nb = json.loads(code)
        cfg = "".join(nb["cells"][1]["source"])
        session = int(re.search(r"^SESSION = (\d+)", cfg, re.M).group(1))
        resume = bool(re.search(r'^RESUME_FROM = "auto"', cfg, re.M))
        override = {"invalid_dataset_sources": self.invalid_sources} if self.invalid_sources else None
        if session == 1 and not resume:
            if self.s1_status != "complete":
                return ["queued", "running", self.s1_status], {}, override
            done = ["first:A0", "first:R3"] if (self.s1_partial or self.s1_partial_then_ok) else L.SESSION_PLAN[1]
            ran = L.SESSION_PLAN[1] if done != L.SESSION_PLAN[1] else None
            z = build_output(1, done, ran=ran, tamper=self.tamper == 1)
        else:
            hand = [d for d in meta["dataset_sources"] if "handoff" in d]
            assert len(hand) == 1, "resume run must attach exactly one handoff dataset"
            seeded = session1_files(fk.datasets[hand[0]]["zip"])
            res = {"from": "/kaggle/input/" + hand[0].split("/")[1] + "/mql0015_outputs", "seeded": sorted(seeded)}
            if session == 1:                                          # continuation of session 1
                z = build_output(1, L.SESSION_PLAN[1], seeded=seeded, resumed=res, ran=["retry:base"])
            else:
                if self.s2_status != "complete":
                    return ["queued", "running", self.s2_status], {}, override
                if self.s2_corrupt_seed:
                    k = sorted(seeded)[0]
                    seeded[k] = seeded[k] + b"x"
                z = build_output(1 if self.s2_wrong_session else 2, L.ALL_PASSES, seeded=seeded, resumed=res,
                                 ft_weights={**FT_OK, "adapter_sha256": "0" * 64} if self.s2_bad_adapter else FT_OK,
                                 final=True, ran=L.SESSION_PLAN[2], tamper=self.tamper == 2,
                                 problems=["passes ran in different environments: [...]"] if self.s2_env_mismatch else (),
                                 env_ok=not self.s2_env_mismatch)
        outs = {} if self.missing_zip == session else {L.OUTPUT_ZIP_NAME: z, "hf_model/model.safetensors": b"big"}
        return ["queued", "running", "complete"], outs, override


# ------------------------------------------------------------------ harness
@pytest.fixture(scope="module")
def real_local(tmp_path_factory):
    """Run the REAL local checks once (bundle sha, notebook sha, package == bundle, notebook defaults)."""
    o = L.Orchestrator(FakeKaggle(), root=tmp_path_factory.mktemp("lc"), sleep=lambda s: None, log=lambda *a: None,
                       stage_root=tmp_path_factory.mktemp("st"))
    info = o.local_checks()
    return info, o.hardware, o.adapter_sha, o.notebook, o.model


@pytest.fixture()
def make(tmp_path, real_local, monkeypatch):
    info, hw, ad, nb, model = real_local

    def fake_local_checks(self):
        self.hardware, self.adapter_sha, self.notebook, self.model = hw, ad, nb, model
        return info
    monkeypatch.setattr(L.Orchestrator, "local_checks", fake_local_checks)
    verified = []

    def build(scenario=None, fk=None, **kw):
        fk = fk or FakeKaggle(scenario)
        o = L.Orchestrator(fk, root=tmp_path / "orch", sleep=lambda s: None, log=lambda *a: None,
                           local_verify=lambda z: verified.append(z) or {"ok": True}, stage_root=tmp_path / "stage", **kw)
        return o, fk
    build.verified = verified
    return build


def _kernel_pushes(fk):
    return [p for p in fk.pushes if p[0].endswith(L.KERNEL_SLUG)]


def _scan_secret(root):
    for f in Path(root).rglob("*"):
        if f.is_file():
            assert FAKE_KEY not in f.read_text(encoding="utf-8", errors="ignore"), f


# ------------------------------------------------------------------ tests
def test_real_local_checks_pass(real_local):
    info, hw, ad, nb, model = real_local
    assert model == {"id": "Qwen/Qwen2.5-Coder-3B-Instruct", "revision": "488639f1ff808d1d3d0ba301aef8c11461451ec5"}
    assert info["bundle_sha256"] == L.BUNDLE_SHA256 and info["notebook_sha256"] == L.NOTEBOOK_SHA256
    assert info["package_manifest_sha256"] == L.PACKAGE_MANIFEST_SHA256 and ad == ADAPTER and "Tesla T4" in hw["allowed_gpus"]


def test_happy_path_probe_s1_handoff_s2(make, tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", FAKE_KEY)
    o, fk = make()
    final = o.run_all()
    ks = _kernel_pushes(fk)
    assert len(ks) == 2                                                     # exactly session 1 + session 2
    m1, m2 = ks[0][1], ks[1][1]
    assert m1["dataset_sources"] == [o.bundle_ref] and m1["is_private"] and m1["enable_gpu"] and m1["enable_internet"]
    assert m1["docker_image_pinning_type"] == "original" == m2["docker_image_pinning_type"]
    assert m2["dataset_sources"] == [o.bundle_ref, "tester/mql0015-handoff-r0"]
    assert [p[0] for p in fk.pushes][0].endswith(L.PROBE_SLUG)              # probe first, on its own slug
    assert fk.creates == [o.bundle_ref, "tester/mql0015-handoff-r0"]       # handoff = byte-identical verified s1 zip
    s1zip = Path(o.state["runs"][0]["zip_path"]).read_bytes()
    assert fk.datasets["tester/mql0015-handoff-r0"]["zip"] == s1zip
    assert o.state["runs"][1]["notebook_changes"] and len(o.state["runs"][1]["notebook_changes"]) == 2
    assert o.state["runs"][0]["notebook_changes"] == []
    assert final["zip_sha256"] == o.state["runs"][1]["zip_sha256"] and o.state["done"] and make.verified
    assert not any("hf_model" in str(p) for p in (tmp_path / "orch").rglob("*"))   # model cache never downloaded
    _scan_secret(tmp_path / "orch")


def test_session1_notebook_is_frozen_source_and_s2_differs_in_two_lines(make):
    o, fk = make()
    o.run_all()
    ks = _kernel_pushes(fk)
    frozen = json.loads(L.NOTEBOOK.read_text(encoding="utf-8"))
    assert L.notebook_diff(frozen, json.loads(ks[0][2])) == []
    d = L.notebook_diff(frozen, json.loads(ks[1][2]))
    assert [(a, b) for _, a, b in d] == [("SESSION = 1", "SESSION = 2"),
                                         (L.CFG_LINES["RESUME_FROM"], L.CFG_LINES["RESUME_FROM"].replace("RESUME_FROM = None  ", 'RESUME_FROM = "auto"', 1))]


def test_generated_notebook_variants():
    nb = json.loads(L.NOTEBOOK.read_text(encoding="utf-8"))
    assert L.make_session_notebook(nb, 1, False)[1] == []
    assert len(L.make_session_notebook(nb, 1, True)[1]) == 1
    _, ch = L.make_session_notebook(nb, 2, True, install_missing=True)
    assert [b for _, _, b in ch][2] == "INSTALL_MISSING = True" and len(ch) == 3
    bad = json.loads(json.dumps(nb))
    bad["cells"][1]["source"] = "".join(bad["cells"][1]["source"]).replace("SESSION = 1", "SESSION = 3")
    with pytest.raises(L.LaunchError):
        L.make_session_notebook(bad, 2, True)


def test_restart_after_completion_pushes_nothing(make):
    o, fk = make()
    o.run_all()
    n = len(fk.pushes)
    o2, _ = make(fk=fk)
    o2.run_all()
    assert len(fk.pushes) == n


def test_restart_mid_session1_does_not_push_again(make):
    sc = Scenario()
    fk = FakeKaggle(sc)
    o, _ = make(fk=fk)
    o.local_checks(); o.ensure_bundle_dataset(); o.run_probe()
    real_drive = L.Orchestrator._wait_terminal
    calls = {"n": 0}

    def crash(self, ref, rec, first_run, accept_complete=None):
        if ref.endswith(L.KERNEL_SLUG) and calls["n"] == 0:
            calls["n"] += 1
            raise KeyboardInterrupt
        return real_drive(self, ref, rec, first_run, accept_complete)
    L.Orchestrator._wait_terminal = crash
    try:
        with pytest.raises(KeyboardInterrupt):
            o.run_sessions()
    finally:
        L.Orchestrator._wait_terminal = real_drive
    assert len(_kernel_pushes(fk)) == 1
    o2, _ = make(fk=fk)
    o2.run_all()
    assert len(_kernel_pushes(fk)) == 2                                      # s1 adopted from state, then s2 only


def test_unconfirmed_push_refuses_to_push_again(make):
    o, fk = make()
    o.local_checks(); o.ensure_bundle_dataset(); o.run_probe()
    o.state["runs"].append({"index": 0, "session": 1, "resume": False, "datasets": [o.bundle_ref], "handoff_from": None,
                            "push_requested": L.now()})
    o.save()
    o2, _ = make(fk=fk)
    with pytest.raises(L.LaunchError, match="no version was recorded"):
        o2.run_all()
    assert not _kernel_pushes(fk)


@pytest.mark.parametrize("devices,match", [(("Tesla P100-PCIE-16GB",), "not the registered"),
                                           (("Tesla T4",), "1 GPUs")])
def test_probe_wrong_gpu_aborts_before_session1(make, devices, match):
    o, fk = make(Scenario(probe_devices=devices))
    with pytest.raises(L.LaunchError, match=match):
        o.run_all()
    assert not _kernel_pushes(fk)


def test_probe_without_peft_needs_explicit_flag(make):
    o, fk = make(Scenario(peft=None))
    with pytest.raises(L.LaunchError, match="no peft"):
        o.run_all()
    assert not _kernel_pushes(fk)


@pytest.mark.parametrize("status", ["error", "cancel_acknowledged"])
def test_session1_failure_stops(make, status):
    o, fk = make(Scenario(s1_status=status))
    with pytest.raises(L.LaunchError, match=f"ended {status}"):
        o.run_all()
    assert len(_kernel_pushes(fk)) == 1 and not any("handoff" in c for c in fk.creates)


def test_partial_session1_stops_by_default(make):
    o, fk = make(Scenario(s1_partial=True))
    with pytest.raises(L.LaunchError, match="PARTIAL"):
        o.run_all()
    assert len(_kernel_pushes(fk)) == 1


def test_partial_session1_continues_only_with_flag(make):
    o, fk = make(Scenario(s1_partial_then_ok=True), continue_partial=1)
    o.run_all()
    sessions = [re.search(r"SESSION = (\d)", "".join(json.loads(c)["cells"][1]["source"])).group(1) for _, _, c in _kernel_pushes(fk)]
    assert sessions == ["1", "1", "2"]
    assert o.state["runs"][2]["datasets"][1] == "tester/mql0015-handoff-r1"   # session 2 resumes from the completed s1


def test_stale_complete_after_push_is_not_taken_for_session2(make):
    o, fk = make(Scenario(stale_first=True))
    o.run_all()
    assert o.state["runs"][1]["zip_sha256"] != o.state["runs"][0]["zip_sha256"]
    assert ["complete"] == [s for _, s in o.state["runs"][1]["status_history"][:1]]   # the stale status was observed


def test_handoff_zip_not_extracted_stops_before_session2(make):
    sc = Scenario()
    fk = FakeKaggle(sc)
    o, _ = make(fk=fk)
    orig = fk.add_dataset

    def add(ref, b, name):
        fk.extract = "handoff" not in ref
        orig(ref, b, name)
    fk.add_dataset = add
    with pytest.raises(L.LaunchError, match="unextracted zip"):
        o.run_all()
    assert len(_kernel_pushes(fk)) == 1


def test_bundle_not_extracted_stops_before_probe(make):
    o, fk = make(fk=FakeKaggle(extract=False))
    with pytest.raises(L.LaunchError, match="unextracted zip"):
        o.run_all()
    assert not fk.pushes


def test_existing_bundle_dataset_is_only_verified(make, real_local):
    fk = FakeKaggle()
    fk.add_dataset("tester/" + L.BUNDLE_DATASET_SLUG, L.BUNDLE_ZIP.read_bytes(), L.BUNDLE_ZIP.name)
    o, _ = make(fk=fk)
    o.run_all()
    assert "tester/" + L.BUNDLE_DATASET_SLUG not in fk.creates


def test_existing_kernel_without_state_is_refused(make):
    fk = FakeKaggle()
    fk.kernels["tester/" + L.KERNEL_SLUG] = [{"meta": {}, "code": "", "seq": ["complete"], "ptr": 0, "outputs": {}}]
    o, _ = make(fk=fk)
    with pytest.raises(L.LaunchError, match="already exists"):
        o.run_all()


def test_dropped_dataset_source_is_fatal(make):
    o, fk = make(Scenario(invalid_sources=["tester/" + L.BUNDLE_DATASET_SLUG]))
    with pytest.raises(L.LaunchError, match="dropped data sources"):
        o.run_all()


@pytest.mark.parametrize("session", [1, 2])
def test_tampered_output_zip_fails(make, session):
    o, fk = make(Scenario(tamper=session))
    with pytest.raises(L.LaunchError):
        o.run_all()
    assert len(_kernel_pushes(fk)) == session


@pytest.mark.parametrize("session", [1, 2])
def test_missing_output_zip_fails(make, session):
    o, fk = make(Scenario(missing_zip=session))
    with pytest.raises(L.LaunchError):
        o.run_all()
    assert not o.state.get("done")


def test_session2_output_with_wrong_session_fails(make):
    o, fk = make(Scenario(s2_wrong_session=True))
    with pytest.raises(L.LaunchError):
        o.run_all()
    assert not o.state.get("done")


def test_session2_changed_session1_files_fail_handoff_integrity(make):
    o, fk = make(Scenario(s2_corrupt_seed=True))
    with pytest.raises(L.LaunchError, match="handoff integrity"):
        o.run_all()


def test_session2_wrong_adapter_fails(make):
    o, fk = make(Scenario(s2_bad_adapter=True))
    with pytest.raises(L.LaunchError, match="another adapter"):
        o.run_all()


def test_environment_mismatch_is_hard_failure(make):
    o, fk = make(Scenario(s2_env_mismatch=True))
    with pytest.raises(L.LaunchError, match="different environments"):
        o.run_all()


def test_handoff_guards_status_and_manifest_files():
    good = build_output(1, L.SESSION_PLAN[1])
    assert sum(Path(n).name == L.STATUS_FILE for n in zipfile.ZipFile(io.BytesIO(good)).namelist()) == 1
    bad = build_output(1, L.SESSION_PLAN[1], extra_pkg_manifest=True)
    p = Path(__file__).parent / "_tmp_bad.zip"
    try:
        p.write_bytes(bad)
        with pytest.raises(L.LaunchError, match="package_manifest"):
            L.verify_output_zip(p)
    finally:
        p.unlink()


def test_lock_prevents_concurrent_launchers(tmp_path):
    with L._Lock(tmp_path / "launcher.lock"):
        with pytest.raises(L.LaunchError, match="another launcher"):
            with L._Lock(tmp_path / "launcher.lock"):
                pass
    with L._Lock(tmp_path / "launcher.lock"):
        pass


def test_main_reports_missing_credentials(tmp_path, capsys):
    def no_auth():
        raise L.LaunchError("Kaggle API not available/authenticated (OSError).")
    rc = L.main(["check", "--state-dir", str(tmp_path)], backend_factory=no_auth)
    assert rc == 2 and "not available/authenticated" in capsys.readouterr().err


def test_pushed_notebook_survives_windows_default_encoding(make):
    """kaggle 1.7.4.5 reads code_file with open() (cp1252 on Windows); the frozen notebook has a non-ASCII char."""
    frozen_text = L.NOTEBOOK.read_text(encoding="utf-8")
    assert any(ord(c) > 127 for c in frozen_text)
    o, fk = make()
    o.run_all()
    s1_code = [c for c in fk.staged_codes if c.lstrip().startswith("{")][0]
    assert json.loads(s1_code) == json.loads(frozen_text)                    # byte-for-byte same characters


def test_probe_without_internet_or_revision_aborts(make):
    o, fk = make(Scenario(hf_ok=False))
    with pytest.raises(L.LaunchError, match="Hugging Face"):
        o.run_all()
    assert not _kernel_pushes(fk)


def test_dataset_create_can_be_retried_after_transient_failure(make):
    fk = FakeKaggle()
    fk.fail_create_once.add("tester/" + L.BUNDLE_DATASET_SLUG)
    o, _ = make(fk=fk)
    with pytest.raises(RuntimeError, match="transient"):
        o.run_all()
    assert o.state["bundle_dataset"]["create_requested"] and not fk.pushes
    o2, _ = make(fk=fk)
    o2.run_all()
    assert fk.creates.count("tester/" + L.BUNDLE_DATASET_SLUG) == 1 and o2.state["done"]
