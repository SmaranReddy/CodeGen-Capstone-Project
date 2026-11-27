"""MQL-0015 Kaggle orchestrator: GPU/input probe -> session 1 -> verified handoff -> session 2, from ONE local command.

Runs on the owner's machine with the official `kaggle` Python API (credentials from ~/.kaggle/kaggle.json or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables; never read, printed, logged or uploaded by this script).
Nothing goes to Kaggle except: the frozen bundle zip (private dataset), the frozen notebook (session 1, byte-identical
source), a generated session-2 copy that differs from it in exactly two config lines, a tiny GPU probe script, and the
verified session-1 prediction zip (private handoff dataset).

Flow (each step recorded in a persistent state file; a restart resumes and never submits a duplicate):
  0. local checks: bundle sha 91ab0e9f..., notebook sha 839a8602..., package loads, every bundle entry == the package
     file, notebook config cell == the registered session-1 defaults, no MQL0015_STATUS.json in the bundle.
  1. auth + bundle dataset (private; created from the exact zip if absent, NEVER re-versioned if present) + listing
     check: every bundle entry present, extracted, with its byte size.
  2. probe kernel (separate slug, never the experiment notebook): GPU name/memory vs the frozen config hardware,
     2 visible devices (T4 x2), exactly one MQL-0015 package under /kaggle/input, image library versions incl. peft.
  3. session 1 = version 1 of the experiment notebook (frozen source), docker pinning "original".
  4. poll -> download ONLY mql0015_predictions.zip (paged output listing; the 6 GB model cache is never fetched) ->
     verify every SHA256SUMS entry, package manifest sha, session identity, all session-1 passes complete, consistent
     environment/weights, no problems, registered GPU.  A partial export STOPS (continuation only with an explicit flag).
  5. handoff: the verified zip, byte-identical, becomes a new private dataset; its listing must show the extracted
     mql0015_outputs/... files with matching sizes (else stop before launching).
  6. session 2 = next version of the SAME notebook (environment pinned to the original) with SESSION = 2,
     RESUME_FROM = "auto", attached: bundle + handoff dataset only.  The real FT merge gate inside the notebook
     (adapter hash, trained LoRA, changed tensors, byte-identical decoding files) runs before any FT generation.
  7. verify session 2: session identity, resumed from /kaggle/input, only FT passes ran, session-1 pass files
     byte-identical to the verified session-1 output, merged adapter sha == pinned, final status complete with
     consistent environment, then the local `mql0015_scoring_authorized.py verify` on the final zip.

Usage (repo root):
  python -B scripts/nl2mql_mql0015/launch_mql0015_sessions.py check      # local + auth + dataset listing, no upload/push
  python -B scripts/nl2mql_mql0015/launch_mql0015_sessions.py run        # the whole sequence (resumable)
  python -B scripts/nl2mql_mql0015/launch_mql0015_sessions.py status     # print the recorded state (no API call)
"""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(1, str(HERE))
import mql0015_common as CM       # noqa: E402

BUNDLE_ZIP = ROOT / "kaggle/nl2mql_mql0015_final_1000_bundle.zip"
BUNDLE_SHA256 = "91ab0e9f9080a45b22279be3436ee64ccfc231827e4d0a72663b578ccd087e3f"
NOTEBOOK = CM.PKG_OUT / "nl2mql_mql0015_final_1000.ipynb"
NOTEBOOK_SHA256 = "839a8602e760c34121fc2790c166689c1be040d3848038e176a612a868a063c1"
PACKAGE_MANIFEST_SHA256 = "74e724f1ff54e8a20777b8acd35fe605d69c3c33df58e86557e2b21128934766"
ORCH_ROOT = CM.OUT / "kaggle_orchestration"
STAGE_ROOT = Path.home() / "codegen_data" / "mql0015_kaggle_staging"     # outside OneDrive (same volume: hard links)

KERNEL_SLUG = "nl2mql-mql0015-final-1000"
KERNEL_TITLE = "nl2mql mql0015 final 1000"
PROBE_SLUG = "nl2mql-mql0015-gpu-probe"
PROBE_TITLE = "nl2mql mql0015 gpu probe"
BUNDLE_DATASET_SLUG = "nl2mql-mql0015-final-1000-bundle"
BUNDLE_DATASET_TITLE = "nl2mql mql0015 final 1000 bundle"
OUTPUT_ZIP_NAME = "mql0015_predictions.zip"
PROBE_FILE = "mql0015_probe.json"
OUT_PREFIX = "mql0015_outputs/"
STATUS_FILE = "MQL0015_STATUS.json"
MANIFEST_FILE = "MQL0015_PREDICTIONS_MANIFEST.json"
SESSION_PLAN = {1: ["first:A0", "first:R3", "retry:base"], 2: ["first:FT-R3", "retry:FT"]}
ALL_PASSES = ["first:A0", "first:R3", "retry:base", "first:FT-R3", "retry:FT"]
PASS_DIR = {p: p.replace(":", "_") for p in ALL_PASSES}
ARMS = ("A0", "A0V", "R3", "R3V", "FT-R3", "FT-R3V")

ACTIVE = {"queued", "running", "new_script"}
CANCELLED = {"cancel_requested", "cancel_acknowledged"}

# the registered session-1 configuration lines of the frozen notebook (cell 1), matched exactly
CFG_LINES = {
    "SESSION": "SESSION = 1",
    "RUN_PASSES": 'RUN_PASSES = None                 # None = the passes of SESSION_PLAN[SESSION] that are not complete yet',
    "SESSION_PLAN": 'SESSION_PLAN = {1: ["first:A0", "first:R3", "retry:base"], 2: ["first:FT-R3", "retry:FT"]}',
    "RESUME_FROM": 'RESUME_FROM = None                # None (session 1) | "auto" (attach the previous version\'s output) | a path',
    "INSTALL_MISSING": "INSTALL_MISSING = False",
    "GENERATOR_FACTORY": "GENERATOR_FACTORY = None          # callable(cfg, model_key, model_path) -> (run_batch, info, is_oom, cleanup)",
    "FT_MERGER": "FT_MERGER = None                  # callable(base_path, adapter_dir, merged_dir) -> weights_info",
    "FAKE_GPU": "FAKE_GPU = None",
    "STOP_AFTER_BATCHES": "STOP_AFTER_BATCHES = None",
}

PROBE_SCRIPT = r'''# MQL-0015 GPU/input probe (no experiment code, no data written except the probe JSON)
import json, os, subprocess
from importlib import metadata
from pathlib import Path
out = {"probe": "MQL-0015", "nonce": "__NONCE__", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
try:
    import torch
    out["cuda_available"] = torch.cuda.is_available()
    out["devices"] = [{"name": torch.cuda.get_device_properties(i).name,
                       "total_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
                      for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
except Exception as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"
def _v(p):
    try:
        return metadata.version(p)
    except metadata.PackageNotFoundError:
        return None
out["packages"] = {p: _v(p) for p in ("torch", "transformers", "accelerate", "tokenizers", "huggingface_hub",
                                      "safetensors", "peft")}
try:
    out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
except Exception as e:
    out["nvidia_smi"] = f"{type(e).__name__}"
pk = []
for m in Path("/kaggle/input").rglob("package_manifest.json"):
    try:
        pk.append({"dir": str(m.parent), "package": json.loads(m.read_text(encoding="utf-8")).get("package")})
    except Exception:
        pk.append({"dir": str(m.parent), "package": None})
out["package_manifests"] = pk
out["status_files"] = [str(p) for p in Path("/kaggle/input").rglob("MQL0015_STATUS.json")]
try:
    from huggingface_hub import HfApi
    out["hf_model_info_sha"] = HfApi().model_info("__MODEL_ID__", revision="__MODEL_REV__").sha
except Exception as e:
    out["hf_error"] = f"{type(e).__name__}"
out["input_files"] = sum(1 for p in Path("/kaggle/input").rglob("*") if p.is_file())
Path("/kaggle/working/mql0015_probe.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
'''


class LaunchError(RuntimeError):
    pass


class NotFound(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise LaunchError(msg)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ Kaggle backend (real one = official kaggle API)
class RealKaggle:
    """Thin adapter over the official `kaggle` package (v1.7.x). Importing `kaggle` authenticates from
    ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY; credentials never pass through this class."""

    def __init__(self):
        try:
            import kaggle                    # noqa: F401  (authenticates on import)
            from kaggle import api
        except BaseException as e:          # noqa: BLE001  (the package calls exit/raises on missing credentials)
            if isinstance(e, KeyboardInterrupt):
                raise
            raise LaunchError(f"Kaggle API not available/authenticated ({type(e).__name__}). Install `kaggle` and put "
                              "the token in ~/.kaggle/kaggle.json (see the runbook).") from None
        self.api = api
        self.username = api.get_config_value("username")
        need(self.username, "Kaggle username not configured")

    @staticmethod
    def _is_404(e):
        r = getattr(e, "response", None)
        return getattr(r, "status_code", None) == 404 or "404" in str(getattr(e, "status", "")) or "Not Found" in str(e)

    @staticmethod
    def _enum_name(v):
        return (getattr(v, "name", None) or str(v)).lower()

    def kernel_status(self, ref):
        try:
            r = self.api.kernels_status(ref)
        except Exception as e:  # noqa: BLE001
            if self._is_404(e):
                raise NotFound(ref) from None
            raise
        return self._enum_name(r.status), (r.failure_message or None)

    def push_kernel(self, folder):
        r = self.api.kernels_push(str(folder))
        return {"error": r.error or None, "version": r.version_number, "url": r.url, "ref": r.ref,
                "invalid_dataset_sources": list(r.invalid_dataset_sources or []),
                "invalid_kernel_sources": list(r.invalid_kernel_sources or []),
                "invalid_model_sources": list(r.invalid_model_sources or []),
                "invalid_competition_sources": list(r.invalid_competition_sources or [])}

    def list_output(self, ref):
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        owner, slug = ref.split("/")
        files, log, token = [], None, None
        for _ in range(1000):
            with self.api.build_kaggle_client() as k:
                req = ApiListKernelSessionOutputRequest()
                req.user_name, req.kernel_slug = owner, slug
                if token:
                    req.page_token = token
                resp = k.kernels.kernels_api_client.list_kernel_session_output(req)
            files += [(f.file_name, f.url) for f in resp.files]
            log = log if log is not None else resp.log
            token = resp.next_page_token
            if not token:
                return files, log
        raise LaunchError("kernel output listing did not terminate")

    def download(self, url, dest):
        import requests
        h = hashlib.sha256()
        tmp = Path(str(dest) + ".part")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
        os.replace(tmp, dest)
        return h.hexdigest()

    def dataset_status(self, ref):
        try:
            r = self.api.dataset_status(ref)
        except Exception as e:  # noqa: BLE001
            if self._is_404(e):
                raise NotFound(ref) from None
            raise
        return self._enum_name(getattr(r, "status", r))

    def create_dataset(self, folder):
        r = self.api.dataset_create_new(str(folder), public=False, quiet=True, convert_to_csv=False, dir_mode="skip")
        return {"error": getattr(r, "error", None) or None, "ref": getattr(r, "ref", None), "url": getattr(r, "url", None)}

    def dataset_files(self, ref):
        out, token = [], None
        for _ in range(1000):
            r = self.api.dataset_list_files(ref, page_token=token, page_size=200)
            need(not getattr(r, "error_message", None), f"dataset listing error for {ref}: {r.error_message}")
            items = getattr(r, "dataset_files", None)
            items = items if items is not None else getattr(r, "files", [])
            out += [(f.name, int(f.total_bytes)) for f in items]
            token = r.next_page_token
            if not token:
                return out
        raise LaunchError("dataset listing did not terminate")


# ------------------------------------------------------------------ notebooks
def _cfg_cell(nb):
    idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
           and "".join(c["source"]).lstrip().startswith("# ---- configuration (edit here only)")]
    need(len(idx) == 1, "notebook: expected exactly one configuration cell")
    return idx[0]


def check_session1_notebook(nb):
    lines = "".join(nb["cells"][_cfg_cell(nb)]["source"]).splitlines()
    for key, line in CFG_LINES.items():
        need(lines.count(line) == 1, f"notebook config: expected exactly one line {line!r} ({key})")
    return True


def make_session_notebook(nb, session, resume, install_missing=False):
    """Copy of the frozen notebook; only config-cell lines change: SESSION, RESUME_FROM (and INSTALL_MISSING if asked)."""
    check_session1_notebook(nb)
    out = copy.deepcopy(nb)
    i = _cfg_cell(out)
    src = "".join(out["cells"][i]["source"])
    repl = {}
    if session != 1:
        repl[CFG_LINES["SESSION"]] = f"SESSION = {int(session)}"
    if resume:
        repl[CFG_LINES["RESUME_FROM"]] = CFG_LINES["RESUME_FROM"].replace("RESUME_FROM = None  ", 'RESUME_FROM = "auto"', 1)
    if install_missing:
        repl[CFG_LINES["INSTALL_MISSING"]] = "INSTALL_MISSING = True"
    lines = src.split("\n")
    for old, new in repl.items():
        need(lines.count(old) == 1, f"notebook config line not found exactly once: {old!r}")
        lines[lines.index(old)] = new
    new_src = "\n".join(lines)
    out["cells"][i]["source"] = new_src.splitlines(keepends=True)
    changed = notebook_diff(nb, out)
    need(len(changed) == len(repl), f"generated notebook changed {changed}, expected only {list(repl)}")
    return out, changed


def notebook_diff(a, b):
    """-> [(cell_index, old_line, new_line)]; raises if cells/structure differ beyond line substitutions."""
    need(len(a["cells"]) == len(b["cells"]), "cell count differs")
    need({k: v for k, v in a.items() if k != "cells"} == {k: v for k, v in b.items() if k != "cells"}, "notebook metadata differs")
    out = []
    for i, (ca, cb) in enumerate(zip(a["cells"], b["cells"])):
        need({k: v for k, v in ca.items() if k != "source"} == {k: v for k, v in cb.items() if k != "source"},
             f"cell {i}: non-source fields differ")
        la, lb = "".join(ca["source"]).split("\n"), "".join(cb["source"]).split("\n")
        need(len(la) == len(lb), f"cell {i}: line count differs")
        out += [(i, x, y) for x, y in zip(la, lb) if x != y]
    return out


# ------------------------------------------------------------------ output zips
def zip_entries(path):
    with zipfile.ZipFile(path) as z:
        return {i.filename: i.file_size for i in z.infolist() if not i.filename.endswith("/")}


def verify_output_zip(path):
    """Full integrity check of an exported mql0015_predictions.zip -> summary dict (raises LaunchError)."""
    need(Path(path).is_file(), f"output zip missing: {path}")
    with zipfile.ZipFile(path) as z:
        names = [i.filename for i in z.infolist() if not i.filename.endswith("/")]
        need(names and all(n.startswith(OUT_PREFIX) for n in names), "output zip: entries outside mql0015_outputs/")
        need(OUT_PREFIX + "SHA256SUMS.txt" in names, "output zip: no SHA256SUMS.txt")
        sums = {}
        for ln in z.read(OUT_PREFIX + "SHA256SUMS.txt").decode("utf-8").splitlines():
            h, _, n = ln.partition("  ")
            sums[n] = h
        files = {n[len(OUT_PREFIX):] for n in names} - {"SHA256SUMS.txt"}
        need(files == set(sums), f"output zip: SHA256SUMS lists {len(sums)} files, zip holds {len(files)}")
        for n, h in sums.items():
            need(hashlib.sha256(z.read(OUT_PREFIX + n)).hexdigest() == h, f"output zip: hash mismatch for {n}")
        need(STATUS_FILE in sums and MANIFEST_FILE in sums, "output zip: status or manifest file missing")
        need("package_manifest.json" not in {Path(n).name for n in files}, "output zip holds a package_manifest.json")
        st = json.loads(z.read(OUT_PREFIX + STATUS_FILE))
        man = json.loads(z.read(OUT_PREFIX + MANIFEST_FILE))
    need(st.get("experiment") == CM.EXPERIMENT and st.get("package") == CM.PACKAGE, "output: another experiment/package")
    need(st.get("package_manifest_sha256") == PACKAGE_MANIFEST_SHA256, "output: built from another package manifest")
    need(man.get("package_manifest_sha256") == PACKAGE_MANIFEST_SHA256, "output manifest: another package manifest")
    return {"zip_sha256": sha256_file(path), "sums": sums, "status": st, "manifest": man}


def _gpu_ok(env, hardware):
    gpu = (env or {}).get("gpu") or {}
    return gpu.get("device0") in hardware["allowed_gpus"] and (gpu.get("total_gib") or 0) >= hardware["min_gpu_gib"]


def check_session_output(v, session, hardware, prev=None, resume=False, adapter_sha=None):
    """Identity + completeness of one session's export. -> (complete: bool, problems/notes list). Raises on hard failure."""
    st = v["status"]
    need(st.get("session") == session, f"output belongs to session {st.get('session')}, expected {session}")
    need(bool(st.get("resumed")) == bool(resume), f"output resumed={bool(st.get('resumed'))}, expected {bool(resume)}")
    ran = st.get("passes_this_session") or []
    need(set(ran) <= set(SESSION_PLAN[session]), f"session {session} ran passes outside its plan: {ran}")
    need(_gpu_ok(st.get("environment"), hardware), f"output GPU {((st.get('environment') or {}).get('gpu') or {}).get('device0')} "
                                                   f"is not the registered hardware {hardware['allowed_gpus']}")
    passes = st.get("passes") or {}
    plan_done = all((passes.get(p) or {}).get("status") == "complete" for p in SESSION_PLAN[session])
    if session == 1:
        need(all((passes.get(p) or {}).get("status") == "absent" for p in SESSION_PLAN[2]), "session 1 produced FT passes")
    if prev is not None:
        # handoff integrity: every earlier-session pass file is byte-identical in this output
        earlier = [p for s in SESSION_PLAN if s < session for p in SESSION_PLAN[s]]
        for p in earlier:
            pre = f"passes/{PASS_DIR[p]}/"
            a = {n: h for n, h in prev["sums"].items() if n.startswith(pre)}
            b = {n: h for n, h in v["sums"].items() if n.startswith(pre)}
            need(a and a == b, f"handoff integrity: files of {p} differ from the verified earlier output")
        res = st.get("resumed") or {}
        need(str(res.get("from", "")).startswith("/kaggle/input/"), f"resume source not under /kaggle/input: {res.get('from')}")
        need(not (set(ran) & set(earlier)), f"session {session} re-ran earlier passes: {ran}")
    if any(p in SESSION_PLAN[2] for p in ran):
        w = (st.get("weights_this_session") or {}).get("ft")
        need(w, "FT passes ran but no merged-weights record (merge gate) is present")
        need(w.get("adapter_sha256") == adapter_sha, "merged FT weights come from another adapter")
        k, _, n = str(w.get("changed_tensors", "0/0")).partition("/")
        need(k.isdigit() and int(k) > 0, f"FT merge changed no tensor ({w.get('changed_tensors')})")
        need(w.get("decoding_files_sha256_equal_to_base"), "FT merge: decoding files not recorded")
    problems = list(st.get("problems") or [])
    if any("different environments" in p for p in problems):
        raise LaunchError("export reports passes in different environments (Kaggle image changed between sessions "
                          "despite pinning). Stopped; see the runbook (environment risk).")
    need(st.get("environment_consistent") is True and st.get("weights_consistent") is True,
         f"environment/weights not consistent: {problems}")
    need(not problems, f"export problems: {problems}")
    return plan_done, ran


def is_own_output(v, run, prev_zip_sha):
    st = v["status"]
    if st.get("session") != run["session"] or bool(st.get("resumed")) != bool(run["resume"]):
        return False
    return prev_zip_sha is None or v["zip_sha256"] != prev_zip_sha


# ------------------------------------------------------------------ orchestrator
class Orchestrator:
    def __init__(self, backend, root=ORCH_ROOT, poll_seconds=120, sleep=time.sleep, local_verify=None,
                 continue_partial=0, install_missing_s2=False, stale_limit_s=3600, run_timeout_s=14 * 3600, log=print,
                 stage_root=STAGE_ROOT, dataset_license="other"):
        self.k = backend
        self.root = Path(root)
        self.poll, self.sleep, self.log_print = poll_seconds, sleep, log
        self.local_verify = local_verify or _default_local_verify
        self.continue_partial, self.install_missing_s2 = continue_partial, install_missing_s2
        self.stale_limit_s, self.run_timeout_s = stale_limit_s, run_timeout_s
        self.stage_root, self.dataset_license = Path(stage_root), dataset_license
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else \
            {"experiment": CM.EXPERIMENT, "created_at": now(), "runs": [], "probe": None, "bundle_dataset": None,
             "done": False}
        u = getattr(backend, "username", None)
        need(u, "Kaggle username unknown")
        if self.state.get("owner"):
            need(self.state["owner"] == u, f"state belongs to Kaggle user {self.state['owner']}, authenticated as {u}")
        self.state["owner"] = u
        self.kernel_ref = f"{u}/{KERNEL_SLUG}"
        self.probe_ref = f"{u}/{PROBE_SLUG}"
        self.bundle_ref = f"{u}/{BUNDLE_DATASET_SLUG}"

    # -- persistence
    def save(self):
        write_json_atomic(self.state_path, self.state)

    def event(self, kind, **kw):
        rec = {"ts": now(), "event": kind, **kw}
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        self.log_print(f"[{rec['ts']}] {kind} " + json.dumps(kw, sort_keys=True)[:400])

    # -- local checks
    def local_checks(self):
        need(sha256_file(BUNDLE_ZIP) == BUNDLE_SHA256, "bundle zip hash differs from the frozen pin")
        need(sha256_file(NOTEBOOK) == NOTEBOOK_SHA256, "notebook hash differs from the frozen pin")
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        check_session1_notebook(nb)
        C = _load_core()
        P = C.load_package(CM.PKG_OUT)
        need(P["manifest_sha256"] == PACKAGE_MANIFEST_SHA256, "local package manifest sha differs from the pin")
        with zipfile.ZipFile(BUNDLE_ZIP) as z:
            ents = [i for i in z.infolist() if not i.filename.endswith("/")]
            pre = CM.PACKAGE + "/"
            need(all(i.filename.startswith(pre) for i in ents), "bundle entries outside the package folder")
            rel = {i.filename[len(pre):] for i in ents}
            local = {p.relative_to(CM.PKG_OUT).as_posix() for p in CM.PKG_OUT.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts}
            need(rel == local, f"bundle entries != package files: {sorted(rel ^ local)[:5]}")
            for i in ents:
                need(hashlib.sha256(z.read(i)).hexdigest() == sha256_file(CM.PKG_OUT / i.filename[len(pre):]),
                     f"bundle entry differs from the package: {i.filename}")
            need(not any(Path(i.filename).name == STATUS_FILE for i in ents), "bundle contains a status file")
            need(sum(Path(i.filename).name == "package_manifest.json" for i in ents) == 1, "bundle: package_manifest count")
        self.hardware = P["cfg"]["hardware"]
        self.model = {"id": P["cfg"]["model"]["id"], "revision": P["cfg"]["model"]["revision"]}
        self.adapter_sha = C.ADAPTER_SHA256
        self.notebook = nb
        return {"bundle_sha256": BUNDLE_SHA256, "notebook_sha256": NOTEBOOK_SHA256, "package_manifest_sha256": P["manifest_sha256"],
                "allowed_gpus": self.hardware["allowed_gpus"], "min_gpu_gib": self.hardware["min_gpu_gib"]}

    # -- datasets
    def _listing_matches(self, ref, entries):
        """entries {relpath: size}; the dataset listing must contain each entry (same size) under one common prefix."""
        listing = dict(self.k.dataset_files(ref))
        need(listing, f"dataset {ref} lists no files")
        need(not any(n.lower().endswith(".zip") for n in listing),
             f"dataset {ref} holds an unextracted zip ({[n for n in listing if n.lower().endswith('.zip')][:3]}); "
             "the notebook needs extracted files. Stopped before launching.")
        anchor = next(iter(sorted(entries)))
        cands = [n[:-len(anchor)] for n in listing if n.endswith(anchor) and (n == anchor or n[:-len(anchor)].endswith("/"))]
        need(len(cands) == 1, f"dataset {ref}: cannot locate {anchor} in the listing")
        prefix = cands[0]
        for rel, size in entries.items():
            need(listing.get(prefix + rel) == size, f"dataset {ref}: {rel} missing or wrong size")
        extra = set(listing) - {prefix + r for r in entries}
        need(not extra, f"dataset {ref}: unexpected files {sorted(extra)[:5]}")
        return {"files": len(entries), "prefix": prefix}

    def _wait_dataset_ready(self, ref, timeout_s=3600):
        t0 = time.monotonic()
        while True:
            st = self.k.dataset_status(ref)
            if st == "ready":
                return
            need(st not in ("error", "failed", "deleted"), f"dataset {ref} processing failed: {st}")
            need(time.monotonic() - t0 < timeout_s, f"dataset {ref} not ready after {timeout_s}s ({st})")
            self.sleep(min(self.poll, 60))

    def _create_dataset_from_zip(self, slug, title, zip_path, expect_sha):
        stage = self.stage_root / slug
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        dst = stage / Path(zip_path).name
        try:
            os.link(zip_path, dst)                       # hard link: no extra disk space
        except OSError:
            shutil.copyfile(zip_path, dst)
        need(sha256_file(dst) == expect_sha, f"staged copy of {zip_path} differs")
        write_json_atomic(stage / "dataset-metadata.json",
                          {"title": title, "id": f"{self.state['owner']}/{slug}", "licenses": [{"name": self.dataset_license}]})
        _assert_clean_stage(stage, {dst.name, "dataset-metadata.json"})
        try:
            r = self.k.create_dataset(stage)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        need(not r.get("error"), f"dataset create failed for {slug}: {r.get('error')}")
        return r

    def ensure_bundle_dataset(self):
        bd = self.state.get("bundle_dataset") or {}
        if bd.get("verified"):
            return bd
        ents = {n[len(CM.PACKAGE) + 1:]: s for n, s in zip_entries(BUNDLE_ZIP).items()}
        try:
            st = self.k.dataset_status(self.bundle_ref)
            created = bool(bd.get("created_by_launcher"))
            self.event("bundle_dataset_exists", ref=self.bundle_ref, status=st, created_by_launcher=created)
        except NotFound:
            if bd.get("create_requested"):
                self.event("bundle_dataset_create_retry", previous_request=bd.get("create_requested"))
            self.state["bundle_dataset"] = {"ref": self.bundle_ref, "create_requested": now()}
            self.save()
            r = self._create_dataset_from_zip(BUNDLE_DATASET_SLUG, BUNDLE_DATASET_TITLE, BUNDLE_ZIP, BUNDLE_SHA256)
            self.state["bundle_dataset"].update(created_by_launcher=True, url=r.get("url"), uploaded_sha256=BUNDLE_SHA256)
            self.save()
            self.event("bundle_dataset_created", ref=self.bundle_ref, url=r.get("url"), zip_sha256=BUNDLE_SHA256)
        self._wait_dataset_ready(self.bundle_ref)
        m = self._listing_matches(self.bundle_ref, ents)
        self.state["bundle_dataset"] = {**(self.state.get("bundle_dataset") or {}), "ref": self.bundle_ref, "verified": now(),
                                        "listing": m}
        self.save()
        self.event("bundle_dataset_verified", ref=self.bundle_ref, **m)
        return self.state["bundle_dataset"]

    # -- kernels
    def _push(self, slug, title, code_name, code_text, kernel_type, datasets, enable_internet, pinning):
        stage = self.stage_root / f"push_{slug}_{len(self.state['runs'])}_{int(time.time())}"
        stage.mkdir(parents=True)
        # kaggle 1.7.4.5 kernels_push reads code_file with open() and NO encoding (cp1252 on Windows): ship pure ASCII
        # (JSON unicode escapes) so the server receives exactly the frozen characters
        need(code_text.isascii(), "push payload must be ASCII (json-escaped)")
        (stage / code_name).write_text(code_text, encoding="ascii")
        meta = {"id": f"{self.state['owner']}/{slug}", "title": title, "code_file": code_name, "language": "python",
                "kernel_type": kernel_type, "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": enable_internet, "dataset_sources": list(datasets), "kernel_sources": [],
                "competition_sources": [], "model_sources": []}
        if pinning:
            meta["docker_image_pinning_type"] = pinning
        write_json_atomic(stage / "kernel-metadata.json", meta)
        _assert_clean_stage(stage, {code_name, "kernel-metadata.json"})
        for ds in datasets:
            need(self.k.dataset_status(ds) == "ready", f"attached dataset {ds} is not ready")
        r = self.k.push_kernel(stage)
        bad = {k: v for k, v in r.items() if k.startswith("invalid_") and v}
        need(not r.get("error"), f"kernel push error: {r.get('error')}")
        need(not bad, f"Kaggle dropped data sources from the push: {bad}")
        need(r.get("version"), "kernel push returned no version number")
        return r, meta

    def run_probe(self):
        pr = self.state.get("probe") or {}
        if pr.get("passed"):
            return pr
        need(not (pr.get("push_requested") and not pr.get("version")),
             "probe push was requested but not confirmed; check Kaggle, then rerun with --reset-probe")
        if not pr.get("version"):
            nonce = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
            self.state["probe"] = {"push_requested": now(), "nonce": nonce}
            self.save()
            script = PROBE_SCRIPT.replace("__NONCE__", nonce).replace("__MODEL_ID__", self.model["id"])                 .replace("__MODEL_REV__", self.model["revision"])
            r, meta = self._push(PROBE_SLUG, PROBE_TITLE, "probe.py", script, "script", [self.bundle_ref], True, None)
            self.state["probe"].update(version=r["version"], url=r["url"], pushed_at=now(), metadata=meta)
            self.save()
            self.event("probe_pushed", ref=self.probe_ref, version=r["version"], url=r["url"])
        dest = self.root / "probe" / PROBE_FILE

        def fetch():
            files, _log = self.k.list_output(self.probe_ref)
            hit = [u for n, u in files if n == PROBE_FILE]
            if len(hit) != 1:
                return None
            self.k.download(hit[0], dest)
            return json.loads(dest.read_text(encoding="utf-8"))

        def accept():
            q = fetch()
            return bool(q) and q.get("nonce") == self.state["probe"]["nonce"]

        status = self._wait_terminal(self.probe_ref, self.state["probe"], first_run=False, accept_complete=accept)
        need(status == "complete", f"probe kernel ended {status}")
        p = fetch()
        need(p and p.get("nonce") == self.state["probe"]["nonce"], "probe output missing or from another probe run")
        devs = p.get("devices") or []
        need(devs and all(d.get("name") in self.hardware["allowed_gpus"] for d in devs),
             f"probe GPU {[d.get('name') for d in devs]} is not the registered {self.hardware['allowed_gpus']}: the Kaggle "
             "API assigned another accelerator. Stopped before session 1 (the notebook would refuse this GPU).")
        need(len(devs) == 2, f"probe sees {len(devs)} GPUs; GPU T4 x2 expected")
        need(devs[0].get("total_gib", 0) >= self.hardware["min_gpu_gib"], "probe GPU memory below the registered minimum")
        pk = [x for x in p.get("package_manifests") or [] if x.get("package") == CM.PACKAGE]
        need(len(pk) == 1 and str(pk[0]["dir"]).startswith("/kaggle/input/"), f"probe found {len(pk)} MQL-0015 packages under /kaggle/input")
        need(not p.get("status_files"), "probe found MQL0015_STATUS.json in the inputs")
        need(p.get("hf_model_info_sha") == self.model["revision"],
             f"probe could not resolve {self.model['id']}@{self.model['revision']} on Hugging Face "
             f"({p.get('hf_error') or p.get('hf_model_info_sha')}): internet or the pinned revision is unavailable")
        if not (p.get("packages") or {}).get("peft"):
            need(self.install_missing_s2, "the Kaggle image has no peft: session 2 would stop at its environment check. "
                                          "Rerun with --session2-install-missing (sets the notebook's INSTALL_MISSING = True "
                                          "for session 2 only) or stop here.")
        self.state["probe"].update(passed=now(), result={"devices": devs, "packages": p.get("packages"),
                                                         "package_dir": pk[0]["dir"], "input_files": p.get("input_files")})
        self.save()
        self.event("probe_passed", devices=devs, package_dir=pk[0]["dir"], packages=p.get("packages"))
        return self.state["probe"]

    def _wait_terminal(self, ref, rec, first_run, accept_complete=None):
        """Poll until a terminal status that belongs to this push. accept_complete(): True if the COMPLETE output is ours."""
        t0, stale_since = time.monotonic(), None
        while True:
            status, msg = self.k.kernel_status(ref)
            if status != rec.get("last_status"):
                rec["last_status"] = status
                rec.setdefault("status_history", []).append([now(), status])
                self.save()
                self.event("kernel_status", ref=ref, version=rec.get("version"), status=status, message=msg)
            if status in ACTIVE:
                rec["seen_active"] = True
                stale_since = None
            elif status == "error" or status in CANCELLED:
                return status
            elif status == "complete":
                if first_run or rec.get("seen_active"):
                    if accept_complete is None or accept_complete():
                        return status
                    need(not rec.get("seen_active"), "kernel completed after this push but its output is not this run's")
                elif accept_complete is not None and accept_complete():
                    return status
                stale_since = stale_since or time.monotonic()
                need(time.monotonic() - stale_since < self.stale_limit_s, "kernel still reports the previous version as complete")
            else:
                raise LaunchError(f"unexpected kernel status {status!r}")
            need(time.monotonic() - t0 < self.run_timeout_s, f"{ref} did not finish within {self.run_timeout_s}s")
            self.sleep(self.poll)

    def _fetch_output(self, run):
        files, log = self.k.list_output(self.kernel_ref)
        hit = [u for n, u in files if n == OUTPUT_ZIP_NAME]
        need(len(hit) == 1, f"kernel output has {len(hit)} {OUTPUT_ZIP_NAME} files")
        d = self.root / "runs" / f"run{run['index']}"
        d.mkdir(parents=True, exist_ok=True)
        if log:
            (d / "kernel.log").write_text(log if isinstance(log, str) else json.dumps(log), encoding="utf-8")
        dest = d / OUTPUT_ZIP_NAME
        if dest.exists():
            dest.unlink()
        got = self.k.download(hit[0], dest)
        need(got == sha256_file(dest), "download hash mismatch")
        return dest

    def _prev_verified(self):
        done = [r for r in self.state["runs"] if r.get("verified")]
        return done[-1] if done else None

    def _plan_next_run(self):
        """-> (session, resume, handoff_from_run_index) or None when finished."""
        last = self._prev_verified()
        if last is None:
            return 1, False, None
        if last["session"] == 2 and last["plan_complete"]:
            return None
        if last["plan_complete"]:
            return 2, True, last["index"]
        n_cont = sum(1 for r in self.state["runs"] if r["session"] == last["session"]) - 1
        need(n_cont < self.continue_partial,
             f"session {last['session']} export is PARTIAL (stopped={last.get('stopped')!r}; passes {last.get('passes')}). "
             "Stopped: nothing further is launched on an incomplete run. --continue-partial N allows up to N continuation "
             "versions of the same session (resume from the verified partial output).")
        return last["session"], True, last["index"]

    def _handoff(self, src_run):
        h = src_run.get("handoff") or {}
        if h.get("verified"):
            return h["ref"]
        slug = f"mql0015-handoff-r{src_run['index']}"
        ref = f"{self.state['owner']}/{slug}"
        zp = Path(src_run["zip_path"])
        need(sha256_file(zp) == src_run["zip_sha256"], "verified run zip changed on disk")
        ents = zip_entries(zp)
        need(sum(Path(n).name == STATUS_FILE for n in ents) == 1, "handoff zip must hold exactly one status file")
        need(not any(Path(n).name == "package_manifest.json" for n in ents), "handoff zip holds a package_manifest.json")
        try:
            self.k.dataset_status(ref)
            need(h.get("create_requested"), f"handoff dataset {ref} already exists but was not created by this launcher")
        except NotFound:
            if h.get("create_requested"):
                self.event("handoff_dataset_create_retry", ref=ref, previous_request=h.get("create_requested"))
            src_run["handoff"] = {"ref": ref, "create_requested": now()}
            self.save()
            r = self._create_dataset_from_zip(slug, f"mql0015 handoff run {src_run['index']}", zp, src_run["zip_sha256"])
            src_run["handoff"].update(url=r.get("url"), uploaded_sha256=src_run["zip_sha256"])
            self.save()
            self.event("handoff_dataset_created", ref=ref, zip_sha256=src_run["zip_sha256"])
        self._wait_dataset_ready(ref)
        m = self._listing_matches(ref, ents)
        src_run["handoff"].update(verified=now(), listing=m)
        self.save()
        self.event("handoff_verified", ref=ref, **m)
        return ref

    def check_kernel_slug_free(self):
        if self.state["runs"]:
            return
        try:
            st, _ = self.k.kernel_status(self.kernel_ref)
        except NotFound:
            return
        raise LaunchError(f"kernel {self.kernel_ref} already exists ({st}) but this launcher has no record of it; "
                          "refusing to push into it (rename/delete it on Kaggle first)")

    def run_sessions(self):
        self.check_kernel_slug_free()
        while True:
            pending = [r for r in self.state["runs"] if not r.get("verified")]
            if pending:
                run = pending[0]
            else:
                nxt = self._plan_next_run()
                if nxt is None:
                    return self._finish()
                session, resume, src_idx = nxt
                datasets = [self.bundle_ref]
                if resume:
                    datasets.append(self._handoff(self.state["runs"][src_idx]))
                run = {"index": len(self.state["runs"]), "session": session, "resume": resume, "datasets": datasets,
                       "handoff_from": src_idx}
                self.state["runs"].append(run)
                self.save()
            self._drive(run)

    def _drive(self, run):
        if not run.get("version"):
            need(not run.get("push_requested"),
                 f"run {run['index']}: a push was requested at {run.get('push_requested')} but no version was recorded. "
                 "Check the notebook's versions on Kaggle, then rerun with --adopt-pushed-version <N> (if it exists) "
                 "or --forget-unconfirmed-push (if it does not).")
            nb, changed = make_session_notebook(self.notebook, run["session"], run["resume"],
                                                install_missing=(run["session"] == 2 and self.install_missing_s2))
            run.update(push_requested=now(), notebook_changes=[[i, a, b] for i, a, b in changed])
            self.save()
            r, meta = self._push(KERNEL_SLUG, KERNEL_TITLE, NOTEBOOK.name, json.dumps(nb, ensure_ascii=True, indent=1),
                                 "notebook", run["datasets"], True, "original")
            run.update(version=r["version"], url=r["url"], pushed_at=now(), metadata=meta)
            self.save()
            self.event("session_pushed", run=run["index"], session=run["session"], resume=run["resume"],
                       version=r["version"], url=r["url"], datasets=run["datasets"], notebook_changes=run["notebook_changes"])
        prev = self._prev_verified()
        prev_sha = prev["zip_sha256"] if prev else None
        cache = {}

        def accept():
            try:
                z = self._fetch_output(run)
                v = verify_output_zip(z)
            except LaunchError:
                return False
            cache["v"], cache["z"] = v, z
            return is_own_output(v, run, prev_sha)

        status = self._wait_terminal(self.kernel_ref, run, first_run=(run["index"] == 0), accept_complete=accept)
        if status != "complete":
            run["final_status"] = status
            self.save()
            raise LaunchError(f"run {run['index']} (session {run['session']}, version {run.get('version')}) ended "
                              f"{status}; nothing further is launched")
        if "v" not in cache:
            z = self._fetch_output(run)
            cache["v"], cache["z"] = verify_output_zip(z), z
        v, z = cache["v"], cache["z"]
        need(is_own_output(v, run, prev_sha), "completed output does not belong to this run")
        src = self.state["runs"][run["handoff_from"]] if run.get("handoff_from") is not None else None
        prev_v = verify_output_zip(src["zip_path"]) if src else None
        plan_complete, ran = check_session_output(v, run["session"], self.hardware, prev=prev_v, resume=run["resume"],
                                                  adapter_sha=self.adapter_sha)
        run.update(verified=now(), zip_path=str(z), zip_sha256=v["zip_sha256"], plan_complete=plan_complete,
                   export_status=v["status"].get("status"), passes_this_session=ran,
                   passes={p: (v["status"]["passes"].get(p) or {}).get("status") for p in ALL_PASSES},
                   gpu=((v["status"].get("environment") or {}).get("gpu") or {}).get("device0"),
                   stopped=v["status"].get("stopped"))
        self.save()
        self.event("run_verified", run=run["index"], session=run["session"], version=run.get("version"),
                   zip_sha256=v["zip_sha256"], plan_complete=plan_complete, passes=run["passes"], gpu=run["gpu"],
                   stopped=run["stopped"])

    def _finish(self):
        last = self._prev_verified()
        v = verify_output_zip(last["zip_path"])
        st, man = v["status"], v["manifest"]
        need(st.get("status") == "complete" and man.get("status") == "complete", "final export is not complete")
        need((man.get("verify_exports") or {}).get("ok") is True, "final export: verify_exports not ok")
        need(sorted(man.get("per_arm") or {}) == sorted(ARMS), f"final export arms {sorted(man.get('per_arm') or {})}")
        need(all(a["n"] == 1000 for a in man["per_arm"].values()), "final export: an arm does not hold 1,000 records")
        lv = self.local_verify(last["zip_path"])
        self.state.update(done=now(), final={"zip_path": last["zip_path"], "zip_sha256": v["zip_sha256"], "local_verify": lv})
        self.save()
        self.event("experiment_generation_complete", zip=last["zip_path"], zip_sha256=v["zip_sha256"], local_verify=lv)
        return self.state["final"]

    def run_all(self):
        info = self.local_checks()
        self.event("local_checks_passed", **info)
        if self.state.get("done"):
            self.event("already_complete", final=self.state.get("final"))
            return self.state.get("final")
        self.check_kernel_slug_free()          # before any upload or GPU use
        self.ensure_bundle_dataset()
        self.run_probe()
        return self.run_sessions()


def _assert_clean_stage(stage, allowed):
    names = {p.relative_to(stage).as_posix() for p in Path(stage).rglob("*") if p.is_file()}
    need(names == set(allowed), f"staging folder holds unexpected files: {sorted(names - set(allowed))}")
    for n in names:
        need(n.lower() not in ("kaggle.json", ".env") and not n.lower().endswith((".pem", ".key")), "credential-like file staged")


def _load_core():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mql0015_core", CM.PKG_OUT / "src/mql0015_core.py")
    sys.path.insert(0, str(CM.PKG_OUT / "src"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _default_local_verify(zip_path):
    import mql0015_scoring as SCO
    ver, _, _, _ = SCO.cmd_verify([str(zip_path)])     # raises on any failure (registered local verify, no database)
    need(ver.get("ok") is True and (ver.get("verify_exports") or {}).get("ok") is True, "local verify failed")
    return {"ok": True, "n": ver["verify_exports"].get("n"), "predictions_sha256": ver.get("predictions_sha256")}


class _Lock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise LaunchError(f"another launcher holds {self.path}; if none is running, delete that file and rerun") from None
        os.write(fd, json.dumps({"pid": os.getpid(), "at": now()}).encode())
        os.close(fd)
        return self

    def __exit__(self, *a):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def main(argv=None, backend_factory=RealKaggle):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "run", "status"])
    ap.add_argument("--state-dir", default=str(ORCH_ROOT))
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--continue-partial", type=int, default=0,
                    help="allowed continuation versions of a session whose export is partial (default 0 = stop)")
    ap.add_argument("--session2-install-missing", action="store_true",
                    help="set INSTALL_MISSING = True in the session-2 notebook (only if the probe finds no peft)")
    ap.add_argument("--adopt-pushed-version", type=int, help="record this Kaggle version for an unconfirmed push")
    ap.add_argument("--forget-unconfirmed-push", action="store_true", help="drop an unconfirmed push (verify on Kaggle first)")
    ap.add_argument("--reset-probe", action="store_true", help="forget an unfinished/failed probe so it is pushed again")
    ap.add_argument("--dataset-license", default="other", help="license name for the PRIVATE datasets (default: other)")
    a = ap.parse_args(argv)
    root = Path(a.state_dir)
    if a.cmd == "status":
        p = root / "state.json"
        print(p.read_text(encoding="utf-8") if p.exists() else "no state yet")
        return 0
    try:
        with _Lock(root / "launcher.lock"):
            o = Orchestrator(backend_factory(), root=root, poll_seconds=a.poll_seconds, continue_partial=a.continue_partial,
                             install_missing_s2=a.session2_install_missing, dataset_license=a.dataset_license)
            if a.reset_probe:
                need(not (o.state.get("probe") or {}).get("passed"), "the probe already passed; nothing to reset")
                o.state["probe"] = None
                o.save()
                o.event("probe_reset")
            if a.adopt_pushed_version or a.forget_unconfirmed_push:
                un = [r for r in o.state["runs"] if r.get("push_requested") and not r.get("version")]
                need(len(un) == 1, "no unconfirmed push to resolve")
                if a.adopt_pushed_version:
                    un[0].update(version=a.adopt_pushed_version, adopted_at=now())
                else:
                    un[0].pop("push_requested")
                o.save()
                o.event("unconfirmed_push_resolved", run=un[0]["index"], adopted=a.adopt_pushed_version)
            if a.cmd == "check":
                info = o.local_checks()
                o.event("local_checks_passed", **info)
                ents = {n[len(CM.PACKAGE) + 1:]: s for n, s in zip_entries(BUNDLE_ZIP).items()}
                try:
                    o.k.dataset_status(o.bundle_ref)
                    print(json.dumps({"bundle_dataset": o._listing_matches(o.bundle_ref, ents)}, indent=1))
                except NotFound:
                    print(f"bundle dataset {o.bundle_ref} not created yet (the run command creates it, private)")
                try:
                    st, _ = o.k.kernel_status(o.kernel_ref)
                    print(f"WARNING: kernel {o.kernel_ref} exists ({st}); run will refuse unless this state folder created it")
                except NotFound:
                    print(f"kernel {o.kernel_ref}: not created yet (ok)")
                print("check passed: authenticated as", o.state["owner"])
                return 0
            final = o.run_all()
            print("MQL-0015 generation complete:", json.dumps(final, indent=1))
            return 0
    except LaunchError as e:
        print(f"LAUNCH STOPPED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
