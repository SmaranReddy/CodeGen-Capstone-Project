"""Write the MQL-0015 Kaggle notebook (plain JSON, nbformat 4).

Logic lives in src/mql0015_core.py; cells only wire configuration, checks and the passes. Each code cell carries a tag
so tests can execute the cells in order with test hooks injected after `config` (same pattern as MQL-0010/0013).
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "notebook" / "nl2mql_mql0015_final_1000.ipynb"
cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src, tag):
    cells.append({"cell_type": "code", "metadata": {"tags": [tag]}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# MQL-0015: A0V vs R3V vs FT-R3V on 1,000 Spider test_full questions (generate only)

Registered final evaluation (registration fb5d07d0…, amendment A1). Same 1,000 ids, same schema context, same
output contract, same decoding, same validator + one schema-feedback retry in every arm.

| pass | model | generates | batch |
|---|---|---|---|
| `first:A0` | base | zero-shot prompts | 8 |
| `first:R3` | base | frozen R3 prompts (hybrid BM25 + bge-small dense, RRF, Jaccard, top 3) | 8 |
| `retry:base` | base | distinct retry prompts of A0V and R3V | 1 |
| `first:FT-R3` | base + verified LoRA (merged) | the same R3 prompts | 8 |
| `retry:FT` | base + verified LoRA (merged) | distinct retry prompts of FT-R3V | 1 |

- **Generate only.** No MongoDB, no Atlas, no secrets, no external model API, no gold labels in the package.
- **Settings:** Accelerator *GPU T4 x2* (one card used), Internet **on** (base model download at the pinned revision),
  Persistence *Files only*. Attach ONLY the MQL-0015 package dataset. The fine-tuned adapter is inside the package and is
  hash-checked; no other adapter is ever loaded.
- **Sessions:** session 1 runs the base passes, session 2 the FT passes (`SESSION_PLAN`). Each session stops starting
  new batches after `TIME_BUDGET_HOURS` and exports a partial zip. Next version: attach the previous version's output,
  `RESUME_FROM = "auto"`, `SESSION += 1`. Pin the environment after session 1 (export refuses mixed environments).
- bfloat16 on a **Tesla T4** (emulated). Any other GPU stops the run.
""")

code(r"""
# ---- configuration (edit here only) --------------------------------------------------------------
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SESSION = 1
RUN_PASSES = None                 # None = the passes of SESSION_PLAN[SESSION] that are not complete yet
SESSION_PLAN = {1: ["first:A0", "first:R3", "retry:base"], 2: ["first:FT-R3", "retry:FT"]}
RESUME_FROM = None                # None (session 1) | "auto" (attach the previous version's output) | a path
INPUT_ROOT = "/kaggle/input"
PKG_DIR = None
OUT_DIR = "/kaggle/working/mql0015_outputs"
OUTPUT_ZIP = "/kaggle/working/mql0015_predictions.zip"
MODEL_CACHE = "/kaggle/working/hf_model"
MERGED_DIR = "/tmp/mql0015_ft_merged"     # ~6 GB, not saved as notebook output
TIME_BUDGET_HOURS = 11.0
MIN_FREE_DISK_GIB = 16.0
MIN_RAM_GIB = 12.0
INSTALL_MISSING = False

# ---- test hooks (leave the defaults on Kaggle) ----
GENERATOR_FACTORY = None          # callable(cfg, model_key, model_path) -> (run_batch, info, is_oom, cleanup)
FT_MERGER = None                  # callable(base_path, adapter_dir, merged_dir) -> weights_info
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
            if json.loads(m.read_text(encoding="utf-8")).get("package") == "nl2mql_mql0015_final_1000":
                hits.append(m.parent)
        except (OSError, ValueError):
            pass
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one nl2mql_mql0015_final_1000 package under {INPUT_ROOT}, found {hits}")
    return hits[0]

PKG = _find_pkg()
sys.path.insert(0, str(PKG / "src"))
import mql0015_core as C

P = C.load_package(PKG)
CFG, M = P["cfg"], P["cfg"]["model"]
others = [str(p.parent) for p in Path(INPUT_ROOT).rglob("adapter_model.safetensors")
          if Path(INPUT_ROOT).exists() and PKG.resolve() not in p.resolve().parents]
C.need(not others, f"refusing to run: other adapters are attached ({others[:3]}); attach only the MQL-0015 package")
print(f"package OK: {PKG} | {len(P['manifest']['files'])} files | {len(P['ids'])} items | adapter {C.ADAPTER_SHA256[:12]}")
""", "package")

code(r"""
# ---- environment ---------------------------------------------------------------------------------
def _ver(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None

def _wanted_passes():
    if RUN_PASSES is not None:
        C.need(set(RUN_PASSES) <= set(C.PASSES), f"RUN_PASSES must be a subset of {C.PASSES}")
        return [p for p in C.PASSES if p in RUN_PASSES]
    st = C.pass_status(P, OUT_DIR) if Path(OUT_DIR).exists() else {}
    return [p for p in SESSION_PLAN[SESSION] if (st.get(p) or {}).get("status") != "complete"]

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
NEEDED = ["torch", "transformers", "accelerate", "tokenizers", "huggingface_hub", "safetensors"]
if any(C.PASS_MODEL[p] == "ft" for p in SESSION_PLAN.get(SESSION, [])) or (RUN_PASSES and any(C.PASS_MODEL[p] == "ft" for p in RUN_PASSES)):
    NEEDED.append("peft")
missing = [p for p in NEEDED if _ver(p) is None]
if missing and GENERATOR_FACTORY is None:
    if not INSTALL_MISSING:
        raise RuntimeError(f"missing packages {missing}: set INSTALL_MISSING = True (installs only the missing ones, pinned to the image's torch/transformers)")
    import subprocess
    _pins = [f"{p}=={_ver(p)}" for p in ("torch", "transformers", "accelerate") if _ver(p)]
    Path("/tmp/mql0015_constraints.txt").write_text("\n".join(_pins))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing, "-c", "/tmp/mql0015_constraints.txt"])
attached = [str(p) for p in Path(INPUT_ROOT).rglob("*") if p.name.startswith(".env") or p.suffix in (".pem", ".key")] \
    if Path(INPUT_ROOT).exists() else []
C.need(not attached, f"refusing to run: credential-like files attached as input: {attached[:5]}")
ENV = {"python": platform.python_version(), "platform": platform.platform(), "packages": {p: _ver(p) for p in NEEDED}}
try:
    _ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
except (AttributeError, ValueError, OSError):
    _ram = None
HOST = {"ram_gib": _ram, "free_disk_gib": round(shutil.disk_usage(OUT_DIR).free / 2**30, 2)}
GPU = FAKE_GPU if FAKE_GPU is not None else C.K9.gpu_runtime_info()
PRECISION = C.K9.check_precision(CFG, GPU)
C.need(HOST["free_disk_gib"] >= MIN_FREE_DISK_GIB or GENERATOR_FACTORY is not None, f"free disk {HOST['free_disk_gib']} GiB")
C.need(HOST["ram_gib"] is None or HOST["ram_gib"] >= MIN_RAM_GIB or GENERATOR_FACTORY is not None, f"RAM {HOST['ram_gib']} GiB < {MIN_RAM_GIB}")
ENV["gpu"], ENV["host"], ENV["precision"] = GPU, HOST, PRECISION
WANTED = _wanted_passes()
print(json.dumps(ENV, indent=1)); print("passes this session:", WANTED)
""", "env")

code(r"""
# ---- base model files at the pinned revision + revision check -------------------------------------
REVISION_CHECK = None
MODEL_PATH = None
if GENERATOR_FACTORY is None and WANTED:
    from huggingface_hub import snapshot_download, HfApi
    MODEL_PATH = snapshot_download(M["id"], revision=M["revision"], local_dir=MODEL_CACHE)
    REVISION_CHECK = C.K9.check_snapshot_revision(MODEL_PATH, M["revision"])
    C.need(HfApi().model_info(M["id"], revision=M["revision"]).sha == M["revision"], "Hugging Face did not resolve the pinned revision")
    REVISION_CHECK["hub_model_info_sha"] = M["revision"]
C.need((M["id"], M["revision"]) == (C.MODEL_ID, C.MODEL_REVISION), "model identity")
C.need((M["dtype"], M["batch_size_first_pass"], M["batch_size_retry"], M["max_new_tokens"], M["decoding"], M["seed"])
       == ("bfloat16", 8, 1, 1024, "greedy (do_sample=False)", 42), "generation settings differ from the registration")
PRE = {"ok": True, "at": C.now(), "session": SESSION, "passes": WANTED, "package_manifest_sha256": P["manifest_sha256"],
       "config_sha256": P["cfg_sha256"], "registration_sha256": CFG["registration_sha256"], "precision": PRECISION,
       "revision_check": REVISION_CHECK, "adapter_sha256": C.sha256_file(P["adapter_dir"] / "adapter_model.safetensors"), "env": ENV}
C.write_json(Path(OUT_DIR) / f"preflight/preflight_session{SESSION}.json", PRE)
print("PREFLIGHT OK", MODEL_PATH, REVISION_CHECK)
""", "preflight")

code(r"""
# ---- resume (hash-verified earlier output) ---------------------------------------------------------
RESUMED = None
if RESUME_FROM:
    hits = sorted({p.parent for p in Path(INPUT_ROOT).rglob(C.STATUS_FILE)}) if RESUME_FROM == "auto" else [Path(RESUME_FROM)]
    hits = [h for h in hits if h.resolve() != Path(OUT_DIR).resolve()]
    C.need(len(hits) == 1, f"RESUME_FROM: expected exactly one earlier output, found {hits}")
    _st = C.verify_resume_source(hits[0], P["manifest_sha256"])
    RESUMED = {"from": str(hits[0]), "seeded": C.seed_from_previous(hits[0], OUT_DIR), "previous_status": _st["status"]}
    print("resume source verified:", RESUMED["from"], len(RESUMED["seeded"]), "files seeded")
    WANTED = _wanted_passes()
    print("passes this session after resume:", WANTED)
""", "resume")

code(r"""
# ---- passes, one model at a time (fingerprinted append-only checkpoints) -----------------------------
class SessionTimeBudget(Exception):
    pass

T_START = time.time()
OOM_EVENTS = {"batch": 0, "single_prompt": 0}
WEIGHTS = {}
STOPPED = None

def _budgeted(gen):
    def g(batch):
        if time.time() - T_START > TIME_BUDGET_HOURS * 3600:
            raise SessionTimeBudget()
        return gen(batch)
    return g

def _load(model_key):
    if model_key == "ft":
        merger = FT_MERGER or (lambda b, a, m: C.merge_adapter(b, a, m, M["dtype"]))
        WEIGHTS["ft"] = merger(MODEL_PATH, P["adapter_dir"], MERGED_DIR)
        C.need(WEIGHTS["ft"]["adapter_sha256"] == C.ADAPTER_SHA256, "merged weights from another adapter")
        path = MERGED_DIR
    else:
        WEIGHTS["base"] = {"base_revision": M["revision"], "weights": "original"}
        path = MODEL_PATH
    if GENERATOR_FACTORY is not None:
        return GENERATOR_FACTORY(CFG, model_key, path)
    rb, info, is_oom, cleanup, _tok = C.M8.make_qwen_generator(M["id"], M["revision"], M["max_new_tokens"], M["dtype"], local_dir=path)
    return rb, info, is_oom, cleanup

groups = []
for pid in WANTED:
    if not groups or groups[-1][0] != C.PASS_MODEL[pid]:
        groups.append((C.PASS_MODEL[pid], []))
    groups[-1][1].append(pid)
RUNTIME = None
for model_key, pids in groups:
    if STOPPED:
        break
    RUN_BATCH, INFO, IS_OOM, CLEANUP = _load(model_key)
    C.need(INFO.get("dtype") == M["dtype"], f"model dtype {INFO.get('dtype')} != required {M['dtype']}")
    RUNTIME = {"gpu": INFO.get("gpu"), "compute_capability": GPU.get("compute_capability"), "bf16_native": GPU.get("bf16_native"),
               "torch": INFO.get("torch"), "transformers": INFO.get("transformers"), "cuda": GPU.get("cuda")}
    GEN8 = _budgeted(C.M8.oom_safe_generator(RUN_BATCH, IS_OOM, CLEANUP, M["batch_size_first_pass"], OOM_EVENTS))
    GEN1 = _budgeted(C.M8.oom_safe_generator(RUN_BATCH, IS_OOM, CLEANUP, 1, OOM_EVENTS))
    for pid in pids:
        try:
            if pid in C.RETRY_PASSES:
                C.check_dependencies(P, OUT_DIR, pid)
                C.run_retry_pass(P, pid, GEN1, PRECISION, RUNTIME, OUT_DIR, WEIGHTS[model_key], STOP_AFTER_BATCHES)
            else:
                C.run_first_pass(P, pid, GEN8, GEN1, PRECISION, RUNTIME, OUT_DIR, WEIGHTS[model_key], STOP_AFTER_BATCHES)
        except SessionTimeBudget:
            STOPPED = f"time budget of {TIME_BUDGET_HOURS} h reached during {pid}"
            print("STOPPED:", STOPPED)
            break
        if STOP_AFTER_BATCHES is not None:
            STOPPED = "STOP_AFTER_BATCHES (test hook)"
            break
    CLEANUP()
    del RUN_BATCH
""", "generate")

code(r"""
# ---- export ----------------------------------------------------------------------------------------
ST, MAN = C.export(P, OUT_DIR, OUTPUT_ZIP,
                   extra_status={"session": SESSION, "passes_this_session": WANTED, "resumed": RESUMED, "stopped": STOPPED,
                                 "oom_events_this_session": dict(OOM_EVENTS), "revision_check": REVISION_CHECK,
                                 "weights_this_session": WEIGHTS, "environment": ENV})
print("MQL-0015 STATUS:", ST["status"])
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
