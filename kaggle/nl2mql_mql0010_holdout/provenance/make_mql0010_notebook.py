"""Write the MQL-0010 Kaggle notebook (plain JSON, nbformat 4; no nbformat dependency).

Usage: python scripts/nl2mql_mql0010/make_mql0010_notebook.py [--out PATH]
Logic lives in src/mql0010_core.py and the registered modules; cells only wire configuration, checks and the passes.
Each code cell carries a `tags` entry so tests can execute the cells in order with test hooks injected after `config`.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "notebook" / "nl2mql_mql0010_holdout.ipynb"
cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src, tag):
    cells.append({"cell_type": "code", "metadata": {"tags": [tag]}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# MQL-0010: controlled recovery, diversified selection and join-key notes (generate only)

Registration **A4** (pinned by `registration/registration_index.json`), 716 held-out Spider-train questions over 18
databases that are **excluded from retrieval** (view of 2,912 corpus records, hash-checked here).

| pass | generates | batch |
|---|---|---|
| `first:A0`, `first:D0`, `first:E` | zero-shot / MQL-0008 hybrid / diversified-selection prompts, every item | 8 |
| `delta` | EP (E prompt) and EF (E + join-key notes) for the items whose prompt changes | 1 |
| `retry:A0V`, `retry:D0V`, `retry:D0G`, `retry:FULL` | one retry for validator-flagged answers (D0G = generic message) | 1 |

The 9 arms (A0, A0V, D0, D0V, D0G, E, EP, EF, FULL) are assembled from these passes at export and checked by
`scoring.verify_exports`.

- **Generate only.** No MongoDB, no Atlas, no secrets, no external model API, no gold labels in the package.
- **Settings:** Accelerator *GPU T4 x2* (one card used), Internet **on** (public model download at the pinned revision),
  Persistence *Files only*. Attach the package dataset. Attach **no** secrets.
- **Sessions:** session 1 = first passes + delta (~7.6 T4-h), session 2 = retries (~2.4 T4-h). From session 2 on, attach
  the previous version's output and set `RESUME_FROM = "auto"`. Pin the environment after session 1.
- bfloat16 on a **Tesla T4** (emulated, as MQL-0006..0009). Any other GPU stops the run.
- Every plan and prompt is **re-derived here** with the pinned tokenizer and compared with the frozen copy.
""")

code(r"""
# ---- configuration (edit here only) --------------------------------------------------------------
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SESSION = 1                       # 1 = first passes + delta; 2 = retries (configs/experiment.json -> session_plan)
RUN_PASSES = None                 # None = session plan; or an explicit list, e.g. ["first:A0"]
RESUME_FROM = None                # None (session 1) | "auto" (attach the previous version's output) | a path
INPUT_ROOT = "/kaggle/input"
PKG_DIR = None                    # None = search INPUT_ROOT for the nl2mql_mql0010_holdout package
OUT_DIR = "/kaggle/working/mql0010_outputs"
OUTPUT_ZIP = "/kaggle/working/mql0010_predictions.zip"
MODEL_CACHE = "/kaggle/working/hf_model"
TIME_BUDGET_HOURS = 11.0          # stop starting new batches after this; checkpoints are kept and exported
MIN_FREE_DISK_GIB = 10.0
MIN_RAM_GIB = 8.0
INSTALL_MISSING = False           # pip-install ONLY missing packages (never upgrades preinstalled ones)

# ---- test hooks (leave the defaults on Kaggle) ----
GENERATOR_FACTORY = None          # callable(cfg) -> (run_batch, info, is_oom, cleanup); None = real Qwen on the GPU
FAKE_GPU = None                   # dict replacing the GPU probe (tests only)
COUNT_TOKENS = None               # callable(str) -> int replacing the tokenizer (tests only)
STOP_AFTER_BATCHES = None         # kill-and-resume tests only
""", "config")

code(r"""
# ---- package integrity: manifest hashes, registration A4 pins, corpus + retrieval view, evaluation inputs ----------
import json, platform, shutil, sys, time
from importlib import metadata
from pathlib import Path

def _find_pkg():
    if PKG_DIR:
        return Path(PKG_DIR)
    hits = []
    for m in Path(INPUT_ROOT).rglob("package_manifest.json"):
        try:
            if json.loads(m.read_text(encoding="utf-8")).get("package") == "nl2mql_mql0010_holdout":
                hits.append(m.parent)
        except (OSError, ValueError):
            pass
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one nl2mql_mql0010_holdout package under {INPUT_ROOT}, found {hits}")
    return hits[0]

PKG = _find_pkg()
sys.path.insert(0, str(PKG / "src"))
import mql0010_core as C

P = C.load_package(PKG)
CFG, M = P["cfg"], P["cfg"]["model"]
if RUN_PASSES is None:
    RUN_PASSES = CFG["session_plan"][str(SESSION)]["passes"]
C.need(RUN_PASSES and set(RUN_PASSES) <= set(C.PASSES) and len(set(RUN_PASSES)) == len(RUN_PASSES), f"RUN_PASSES must be a subset of {C.PASSES}")
RUN_PASSES = [p for p in C.PASSES if p in RUN_PASSES]           # canonical dependency order
print(f"package OK: {PKG}\n  {len(P['manifest']['files'])} files hash-verified; registration index {CFG['registration_index_sha256'][:12]}; "
      f"retrieval view {len(P['view_rows'])} records (18 holdout DBs excluded); evaluation {len(P['ids'])} items; passes: {RUN_PASSES}")
""", "package")

code(r"""
# ---- environment: dependencies, credentials, GPU + precision, disk, RAM --------------------------------------------
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
PRECISION = C.K9.check_precision(CFG, GPU)       # bf16 on the registered T4 (emulated), or STOP
C.need(HOST["free_disk_gib"] >= MIN_FREE_DISK_GIB or GENERATOR_FACTORY is not None, f"free disk {HOST['free_disk_gib']} GiB")
C.need(HOST["ram_gib"] is None or HOST["ram_gib"] >= MIN_RAM_GIB, f"RAM {HOST['ram_gib']} GiB < {MIN_RAM_GIB}")
ENV["gpu"], ENV["host"], ENV["precision"] = GPU, HOST, PRECISION
print(json.dumps(ENV, indent=1))
""", "env")

code(r"""
# ---- model files at the pinned revision (public weights; no token) + independent revision check -------------------
REVISION_CHECK = None
MODEL_PATH = None
if GENERATOR_FACTORY is None:
    from huggingface_hub import snapshot_download, HfApi
    MODEL_PATH = snapshot_download(M["id"], revision=M["revision"], local_dir=MODEL_CACHE)
    REVISION_CHECK = C.K9.check_snapshot_revision(MODEL_PATH, M["revision"])
    C.need(HfApi().model_info(M["id"], revision=M["revision"]).sha == M["revision"], "Hugging Face did not resolve the pinned revision")
    REVISION_CHECK["hub_model_info_sha"] = M["revision"]
if COUNT_TOKENS is None:
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(MODEL_PATH)
    count_tokens = lambda s: len(TOK.encode(s, add_special_tokens=False))
else:
    count_tokens = COUNT_TOKENS
print("model files:", MODEL_PATH, "| revision check:", REVISION_CHECK)
""", "model")

code(r"""
# ---- deterministic plans + prompts: re-derive A0 / D0 / E / EF and compare with the frozen copies ------------------
DOCS, PLANS, PROMPTS = C.derive_plans(P, count_tokens)
DELTA_IDS = C.delta_ids(PLANS)
C.need(DELTA_IDS == CFG["delta_ids"], "delta items differ from the frozen config")
print({a: {"budget_drops": sum(bool(r["dropped_budget"]) for r in PLANS[a]), "notes": sum(bool(r["schema_notes"]) for r in PLANS[a]),
           "max_block_tokens": max(r["context_tokens"] for r in PLANS[a])} for a in C.PLAN_ARMS}, "| delta items:", len(DELTA_IDS))
""", "plan")

code(r"""
# ---- preflight record -----------------------------------------------------------------------------------------------
C.need((M["id"], M["revision"]) == (C.MODEL_ID, C.MODEL_REVISION), "model identity")
C.need((M["dtype"], M["batch_size_first_pass"], M["batch_size_delta"], M["batch_size_retry"], M["max_new_tokens"], M["decoding"])
       == ("bfloat16", 8, 1, 1, 1024, "greedy (do_sample=False)"), "generation settings differ from registration A4")
PRE = {"ok": True, "at": C.now(), "session": SESSION, "passes": RUN_PASSES, "package_manifest_sha256": P["manifest_sha256"],
       "config_sha256": P["cfg_sha256"], "registration_index_sha256": CFG["registration_index_sha256"],
       "plans_equal_frozen": True, "precision": PRECISION, "revision_check": REVISION_CHECK, "env": ENV}
C.write_json(Path(OUT_DIR) / f"preflight/preflight_session{SESSION}.json", PRE)
print("PREFLIGHT OK")
""", "preflight")

code(r"""
# ---- resume (hash-verified earlier output), then the passes with fingerprinted append-only checkpoints --------------
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
        C.check_dependencies(OUT_DIR, P, PLANS, PROMPTS, pid)
        if pid in C.FIRST_PASSES:
            C.run_first_pass(P, pid.split(":")[1], PLANS, PROMPTS, GEN8, PRECISION, RUNTIME, OUT_DIR, STOP_AFTER_BATCHES)
        elif pid == "delta":
            C.run_delta_pass(P, PLANS, PROMPTS, GEN1, PRECISION, RUNTIME, OUT_DIR, STOP_AFTER_BATCHES)
        else:
            src = C.RETRY_PASSES[pid][1]
            C.run_retry_pass(P, pid, C.retry_sources(P, PLANS, OUT_DIR, src), PROMPTS, GEN1, PRECISION, RUNTIME, OUT_DIR,
                             STOP_AFTER_BATCHES)
    except SessionTimeBudget:
        STOPPED = f"time budget of {TIME_BUDGET_HOURS} h reached during {pid}"
        print("STOPPED:", STOPPED)
        break
    if STOP_AFTER_BATCHES is not None:
        STOPPED = "STOP_AFTER_BATCHES (test hook)"
        break
""", "generate")

code(r"""
# ---- export: pass completeness, environment consistency, arm assembly + verify_exports, manifest, zip -----------------
ST, MAN = C.export(P, PLANS, PROMPTS, OUT_DIR, OUTPUT_ZIP,
                   extra_status={"session": SESSION, "passes_this_session": RUN_PASSES, "resumed": RESUMED, "stopped": STOPPED,
                                 "oom_events_this_session": dict(OOM_EVENTS), "revision_check": REVISION_CHECK, "environment": ENV})
print("MQL-0010 STATUS:", ST["status"])
print(json.dumps(MAN["passes"], indent=1))
print("\n".join(ST["problems"]) or "no problems")
if ST["status"] != "complete":
    print("NOT COMPLETE: download", OUTPUT_ZIP, "; next version: attach this version's output, RESUME_FROM = 'auto', next SESSION.")
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
