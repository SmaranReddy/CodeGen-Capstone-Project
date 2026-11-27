"""Write the MQL-0014 Kaggle notebook (plain JSON, nbformat 4).

Logic lives in src/mql0014_core.py; cells only wire configuration, checks and the passes. Each code cell carries a tag
so tests can execute the cells in order with test hooks injected after `config` (same pattern as MQL-0010).
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "notebook" / "nl2mql_mql0014_r3_recheck.ipynb"
cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src, tag):
    cells.append({"cell_type": "code", "metadata": {"tags": [tag]}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# MQL-0014: R0 (BM25 hybrid) vs R3 (RRF BM25 + dense) on the MQL-0010 holdout (generate only)

**Secondary re-analysis** of the existing MQL-0010 holdout: 716 questions over 18 databases, all excluded from
retrieval (view of 2,912 corpus records). Both arms use the MQL-0010 D0V recipe: first answer, static schema validator,
at most one schema-feedback retry. R0 is byte-identical to MQL-0010 D0. Prompts are frozen in the package and re-hashed.

| pass | generates | batch |
|---|---|---|
| `first:shared` | items whose R0 and R3 prompts are identical (one answer for both arms) | 8 |
| `first:R0only`, `first:R3only` | the other items, R0 / R3 prompt | 8 |
| `retry` | every distinct retry prompt needed by R0V or R3V | 1 |

- **Generate only.** No MongoDB, no Atlas, no secrets, no external model API, no gold labels in the package.
- **Settings:** Accelerator *GPU T4 x2* (one card used), Internet **on**, Persistence *Files only*. Attach the package
  dataset and **no** secrets. Expected ~6.5 T4-hours: one session. If the time budget stops it, attach this version's
  output, set `RESUME_FROM = "auto"`, `SESSION = 2`, keep the same environment.
- bfloat16 on a **Tesla T4** (emulated, as MQL-0006..0010). Any other GPU stops the run.
""")

code(r"""
# ---- configuration (edit here only) --------------------------------------------------------------
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SESSION = 1
RUN_PASSES = None                 # None = all remaining passes in order; or e.g. ["first:shared"]
RESUME_FROM = None                # None (session 1) | "auto" (attach the previous version's output) | a path
INPUT_ROOT = "/kaggle/input"
PKG_DIR = None
OUT_DIR = "/kaggle/working/mql0014_outputs"
OUTPUT_ZIP = "/kaggle/working/mql0014_predictions.zip"
MODEL_CACHE = "/kaggle/working/hf_model"
TIME_BUDGET_HOURS = 11.0
MIN_FREE_DISK_GIB = 10.0
MIN_RAM_GIB = 8.0
INSTALL_MISSING = False

# ---- test hooks (leave the defaults on Kaggle) ----
GENERATOR_FACTORY = None
FAKE_GPU = None
STOP_AFTER_BATCHES = None
""", "config")

code(r"""
# ---- package integrity ---------------------------------------------------------------------------
import json, platform, shutil, sys, time
from importlib import metadata
from pathlib import Path

def _find_pkg():
    if PKG_DIR:
        return Path(PKG_DIR)
    hits = []
    for m in Path(INPUT_ROOT).rglob("package_manifest.json"):
        try:
            if json.loads(m.read_text(encoding="utf-8")).get("package") == "nl2mql_mql0014_r3_recheck":
                hits.append(m.parent)
        except (OSError, ValueError):
            pass
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one nl2mql_mql0014_r3_recheck package under {INPUT_ROOT}, found {hits}")
    return hits[0]

PKG = _find_pkg()
sys.path.insert(0, str(PKG / "src"))
import mql0014_core as C

P = C.load_package(PKG)
CFG, M = P["cfg"], P["cfg"]["model"]
if RUN_PASSES is None:
    RUN_PASSES = list(C.PASSES)
C.need(set(RUN_PASSES) <= set(C.PASSES), f"RUN_PASSES must be a subset of {C.PASSES}")
RUN_PASSES = [p for p in C.PASSES if p in RUN_PASSES]
print(f"package OK: {PKG} | {len(P['manifest']['files'])} files | {len(P['ids'])} items | shared {len(P['shared'])} | passes {RUN_PASSES}")
""", "package")

code(r"""
# ---- environment ---------------------------------------------------------------------------------
def _ver(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None

NEEDED = ["torch", "transformers", "accelerate", "tokenizers", "huggingface_hub"]
missing = [p for p in NEEDED if _ver(p) is None]
if missing and GENERATOR_FACTORY is None:
    if not INSTALL_MISSING:
        raise RuntimeError(f"missing packages {missing}: set INSTALL_MISSING = True (installs only the missing ones)")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
attached = [str(p) for p in Path(INPUT_ROOT).rglob("*") if p.name.startswith(".env") or p.suffix in (".pem", ".key")] \
    if Path(INPUT_ROOT).exists() else []
C.need(not attached, f"refusing to run: credential-like files attached as input: {attached[:5]}")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
ENV = {"python": platform.python_version(), "platform": platform.platform(), "packages": {p: _ver(p) for p in NEEDED}}
try:
    _ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
except (AttributeError, ValueError, OSError):
    _ram = None
HOST = {"ram_gib": _ram, "free_disk_gib": round(shutil.disk_usage(OUT_DIR).free / 2**30, 2)}
GPU = FAKE_GPU if FAKE_GPU is not None else C.K9.gpu_runtime_info()
PRECISION = C.K9.check_precision(CFG, GPU)
C.need(HOST["free_disk_gib"] >= MIN_FREE_DISK_GIB or GENERATOR_FACTORY is not None, f"free disk {HOST['free_disk_gib']} GiB")
C.need(HOST["ram_gib"] is None or HOST["ram_gib"] >= MIN_RAM_GIB, f"RAM {HOST['ram_gib']} GiB < {MIN_RAM_GIB}")
ENV["gpu"], ENV["host"], ENV["precision"] = GPU, HOST, PRECISION
print(json.dumps(ENV, indent=1))
""", "env")

code(r"""
# ---- model files at the pinned revision + revision check ------------------------------------------
REVISION_CHECK = None
MODEL_PATH = None
if GENERATOR_FACTORY is None:
    from huggingface_hub import snapshot_download, HfApi
    MODEL_PATH = snapshot_download(M["id"], revision=M["revision"], local_dir=MODEL_CACHE)
    REVISION_CHECK = C.K9.check_snapshot_revision(MODEL_PATH, M["revision"])
    C.need(HfApi().model_info(M["id"], revision=M["revision"]).sha == M["revision"], "Hugging Face did not resolve the pinned revision")
    REVISION_CHECK["hub_model_info_sha"] = M["revision"]
C.need((M["id"], M["revision"]) == (C.MODEL_ID, C.MODEL_REVISION), "model identity")
C.need((M["dtype"], M["batch_size_first_pass"], M["batch_size_retry"], M["max_new_tokens"], M["decoding"])
       == ("bfloat16", 8, 1, 1024, "greedy (do_sample=False)"), "generation settings differ from the registration")
PRE = {"ok": True, "at": C.now(), "session": SESSION, "passes": RUN_PASSES, "package_manifest_sha256": P["manifest_sha256"],
       "config_sha256": P["cfg_sha256"], "registration_sha256": CFG["registration_sha256"], "precision": PRECISION,
       "revision_check": REVISION_CHECK, "env": ENV}
C.write_json(Path(OUT_DIR) / f"preflight/preflight_session{SESSION}.json", PRE)
print("PREFLIGHT OK", MODEL_PATH, REVISION_CHECK)
""", "preflight")

code(r"""
# ---- resume, then the passes (fingerprinted append-only checkpoints) --------------------------------
RESUMED = None
if RESUME_FROM:
    hits = sorted({p.parent for p in Path(INPUT_ROOT).rglob(C.STATUS_FILE)}) if RESUME_FROM == "auto" else [Path(RESUME_FROM)]
    hits = [h for h in hits if h.resolve() != Path(OUT_DIR).resolve()]
    C.need(len(hits) == 1, f"RESUME_FROM: expected exactly one earlier output, found {hits}")
    _st = C.verify_resume_source(hits[0], P["manifest_sha256"])
    RESUMED = {"from": str(hits[0]), "seeded": C.seed_from_previous(hits[0], OUT_DIR), "previous_status": _st["status"]}
    print("resume source verified:", RESUMED["from"], len(RESUMED["seeded"]), "files seeded")

class SessionTimeBudget(Exception):
    pass

T_START = time.time()
OOM_EVENTS = {"batch": 0, "single_prompt": 0}
if GENERATOR_FACTORY is not None:
    RUN_BATCH, INFO, IS_OOM, CLEANUP = GENERATOR_FACTORY(CFG)
else:
    RUN_BATCH, INFO, IS_OOM, CLEANUP, _tok = C.M8.make_qwen_generator(M["id"], M["revision"], M["max_new_tokens"], M["dtype"],
                                                                        local_dir=MODEL_PATH)
C.need(INFO.get("dtype") == M["dtype"], f"model dtype {INFO.get('dtype')} != required {M['dtype']}")
RUNTIME = {"gpu": INFO.get("gpu"), "compute_capability": GPU.get("compute_capability"), "bf16_native": GPU.get("bf16_native"),
           "torch": INFO.get("torch"), "transformers": INFO.get("transformers"), "cuda": GPU.get("cuda")}

def _budgeted(gen):
    def g(batch):
        if time.time() - T_START > TIME_BUDGET_HOURS * 3600:
            raise SessionTimeBudget()
        return gen(batch)
    return g

GEN8 = _budgeted(C.M8.oom_safe_generator(RUN_BATCH, IS_OOM, CLEANUP, M["batch_size_first_pass"], OOM_EVENTS))
GEN1 = _budgeted(C.M8.oom_safe_generator(RUN_BATCH, IS_OOM, CLEANUP, 1, OOM_EVENTS))
STOPPED = None
for pid in RUN_PASSES:
    try:
        if pid != "retry":
            C.run_first_pass(P, pid, GEN8, PRECISION, RUNTIME, OUT_DIR, STOP_AFTER_BATCHES)
        else:
            C.check_dependencies(P, OUT_DIR, pid)
            C.run_retry_pass(P, GEN1, PRECISION, RUNTIME, OUT_DIR, STOP_AFTER_BATCHES)
    except SessionTimeBudget:
        STOPPED = f"time budget of {TIME_BUDGET_HOURS} h reached during {pid}"
        print("STOPPED:", STOPPED)
        break
    if STOP_AFTER_BATCHES is not None:
        STOPPED = "STOP_AFTER_BATCHES (test hook)"
        break
""", "generate")

code(r"""
# ---- export ----------------------------------------------------------------------------------------
ST, MAN = C.export(P, OUT_DIR, OUTPUT_ZIP,
                   extra_status={"session": SESSION, "passes_this_session": RUN_PASSES, "resumed": RESUMED, "stopped": STOPPED,
                                 "oom_events_this_session": dict(OOM_EVENTS), "revision_check": REVISION_CHECK, "environment": ENV})
print("MQL-0014 STATUS:", ST["status"])
print(json.dumps(MAN["passes"], indent=1))
print("\n".join(ST["problems"]) or "no problems")
if ST["status"] != "complete":
    print("NOT COMPLETE: download", OUTPUT_ZIP, "; next version: attach this version's output, RESUME_FROM = 'auto', SESSION += 1.")
""", "export")


def build(out=DEFAULT_OUT):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    print("wrote", build(ap.parse_args().out))
