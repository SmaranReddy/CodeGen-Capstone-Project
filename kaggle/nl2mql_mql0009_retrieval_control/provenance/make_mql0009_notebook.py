"""Write the MQL-0009 Kaggle notebook (plain JSON, nbformat 4; no nbformat dependency).

Usage: python scripts/nl2mql_mql0009/make_mql0009_notebook.py [--out PATH]
The notebook keeps logic in src/mql0009_core.py; cells only wire configuration, checks and the generation loop.
Each code cell carries a `tags` entry so tests can execute the cells in order with test hooks injected after `config`.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "notebook" / "nl2mql_mql0009_retrieval_control.ipynb"
cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src, tag):
    cells.append({"cell_type": "code", "metadata": {"tags": [tag]}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# MQL-0009: NL→MQL retrieval-control experiment (generate only)

Does the MQL-0008 hybrid-RAG gain come from **relevant** retrieval, or just from showing Qwen three examples?
Four arms on the same held-out Spider-train questions (16 databases, none in the retrieval corpus):

| run | demonstrations (K=3 except A) |
|---|---|
| `A_zero_shot` | none |
| `B_random_s1` … `B_random_s5` | 3 random corpus examples per question, five registered seeds |
| `C_fixed` | the same 3 frozen corpus examples for every question |
| `D_hybrid` | the unchanged MQL-0008 hybrid retriever |

- **Generate only.** No MongoDB, no Atlas, no secrets, no external model API. No gold labels are in the package.
- **Settings:** Accelerator *GPU T4 x2* (one card is used), Internet **on** (Hugging Face download of the public model at
  the pinned revision), Persistence *Files only*. Attach the package dataset. Attach **no** secrets.
- **Sessions.** All 8 runs take about 18 GPU-hours on a T4; that does not fit one 12-hour session. Set `SESSION` (1, 2, 3)
  as in the README. From session 2 on, attach the previous version's output and set `RESUME_FROM = "auto"`. After the
  first session, pin the environment (*Settings → Environment → Pin to original environment*): a resumed run with
  another torch / transformers / GPU is refused.
- Precision is **bfloat16 on a Tesla T4** (emulated bf16, compute capability 7.5), exactly as MQL-0006/0007/0008.
  Any other GPU, or no bf16 support, **stops the run**; nothing is silently changed.
- Every plan is **re-derived here** with the pinned tokenizer and compared with the frozen plan; any difference stops.
""")

code(r"""
# ---- configuration (edit here only) --------------------------------------------------------------
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SESSION = 1                       # 1, 2, 3: picks RUN_ARMS from configs/experiment.json -> session_plan; or set RUN_ARMS below
RUN_ARMS = None                   # None = session plan; or an explicit list, e.g. ["D_hybrid"]
RESUME_FROM = None                # None (session 1) | "auto" (attach the previous version's output) | a path
INPUT_ROOT = "/kaggle/input"
PKG_DIR = None                    # None = search INPUT_ROOT for the nl2mql_mql0009_retrieval_control package
OUT_DIR = "/kaggle/working/mql0009_outputs"
OUTPUT_ZIP = "/kaggle/working/mql0009_predictions.zip"
MODEL_CACHE = "/kaggle/working/hf_model"
TIME_BUDGET_HOURS = 11.0          # stop starting new batches after this; checkpoints are kept and exported
MIN_FREE_DISK_GIB = 10.0
MIN_RAM_GIB = 8.0
INSTALL_MISSING = False           # pip-install ONLY packages that are missing (never upgrades preinstalled ones)

# ---- test hooks (leave the defaults on Kaggle) ----
GENERATOR_FACTORY = None          # callable(cfg) -> (run_batch, info, is_oom, cleanup); None = real Qwen on the GPU
FAKE_GPU = None                   # dict replacing the GPU probe (tests only)
COUNT_TOKENS = None               # callable(str) -> int replacing the tokenizer (tests only)
STOP_AFTER_BATCHES = None         # kill-and-resume tests only
""", "config")

code(r"""
# ---- package integrity: file hashes, corpus identity, evaluation inputs, frozen config, leakage re-check ----------
import json, platform, shutil, sys, time
from importlib import metadata
from pathlib import Path

def _find_pkg():
    if PKG_DIR:
        return Path(PKG_DIR)
    hits = []
    for m in Path(INPUT_ROOT).rglob("package_manifest.json"):
        try:
            if json.loads(m.read_text(encoding="utf-8")).get("package") == "nl2mql_mql0009_retrieval_control":
                hits.append(m.parent)
        except (OSError, ValueError):
            pass
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one nl2mql_mql0009_retrieval_control package under {INPUT_ROOT}, found {hits}")
    return hits[0]

PKG = _find_pkg()
sys.path.insert(0, str(PKG / "src"))
import mql0009_core as C

P = C.load_package(PKG)
CFG, M = P["cfg"], P["cfg"]["model"]
RUN_NAMES = [r["name"] for r in P["runs"]]
if RUN_ARMS is None:
    RUN_ARMS = CFG["session_plan"][str(SESSION)]["runs"]
C.need(RUN_ARMS and set(RUN_ARMS) <= set(RUN_NAMES) and len(set(RUN_ARMS)) == len(RUN_ARMS), f"RUN_ARMS must be a subset of {RUN_NAMES}")
print(f"package OK: {PKG}\n  {len(P['manifest']['files'])} files hash-verified; corpus {len(P['corpus'])} records "
      f"(eligible view {len(P['view_rows'])}); evaluation {len(P['ids'])} items / {len(P['eval_dbs'])} DBs; runs this session: {RUN_ARMS}")
""", "package")

code(r"""
# ---- environment: dependencies, credentials, GPU + precision compatibility, disk, RAM ----------------------------
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
ENV = {"python": platform.python_version(), "platform": platform.platform(),
       "packages": {p: _ver(p) for p in ("torch", "transformers", "accelerate", "tokenizers", "huggingface_hub")}}
try:
    _ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
except (AttributeError, ValueError, OSError):
    _ram = None
HOST = {"ram_gib": _ram, "free_disk_gib": round(shutil.disk_usage(OUT_DIR).free / 2**30, 2)}
GPU = FAKE_GPU if FAKE_GPU is not None else C.gpu_runtime_info()
PRECISION = C.check_precision(CFG, GPU)          # bf16 on the registered T4 (emulated), or STOP
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
    REVISION_CHECK = C.check_snapshot_revision(MODEL_PATH, M["revision"])
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
# ---- deterministic selection: re-derive features, all 8 plans and prompts; compare with the frozen copies ----------
DOCS, DOCS_BY_ID, PLANS, ARM_PROMPTS = C.derive_plans(P, count_tokens)
print({n: {"k_min": min(len(p["selected_ids"]) for p in PLANS[n]), "k_max": max(len(p["selected_ids"]) for p in PLANS[n]),
           "skipped_over_budget": sum(bool(p["dropped_budget"]) for p in PLANS[n]),
           "max_block_tokens": max(p["context_tokens"] for p in PLANS[n])} for n in RUN_NAMES})
""", "plan")

code(r"""
# ---- preflight record ---------------------------------------------------------------------------------------------
C.need((M["id"], M["revision"]) == (C.MODEL_ID, C.MODEL_REVISION), "model identity")
C.need((M["dtype"], M["batch_size"], M["max_new_tokens"], M["decoding"]) == ("bfloat16", 8, 1024, "greedy (do_sample=False)"), "decoding settings")
PRE = {"ok": True, "at": C.now(), "session": SESSION, "run_arms": RUN_ARMS, "package_manifest_sha256": P["manifest_sha256"],
       "config_sha256": P["cfg_sha256"], "plans_equal_frozen": True, "precision": PRECISION, "revision_check": REVISION_CHECK,
       "expected_predictions_all_runs": len(P["ids"]) * len(RUN_NAMES), "env": ENV}
C.write_json(Path(OUT_DIR) / f"preflight/preflight_session{SESSION}.json", PRE)
print("PREFLIGHT OK")
""", "preflight")

code(r"""
# ---- resume (hash-verified earlier output), then generation with fingerprinted append-only checkpoints -------------
RESUMED = None
if RESUME_FROM:
    hits = sorted({p.parent for p in Path(INPUT_ROOT).rglob(C.STATUS_FILE)}) if RESUME_FROM == "auto" else [Path(RESUME_FROM)]
    hits = [h for h in hits if h.resolve() != Path(OUT_DIR).resolve()]
    C.need(len(hits) == 1, f"RESUME_FROM: expected exactly one earlier output, found {hits}")
    _st = C.verify_resume_source(hits[0], P["manifest_sha256"])
    RESUMED = {"from": str(hits[0]), "seeded": C.seed_from_previous(hits[0], OUT_DIR, RUN_NAMES), "previous_status": _st["status"]}
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
_GEN = C.M8.oom_safe_generator(RUN_BATCH, IS_OOM, CLEANUP, M["batch_size"], OOM_EVENTS)

def GEN(batch):
    if time.time() - T_START > TIME_BUDGET_HOURS * 3600:
        raise SessionTimeBudget()
    return _GEN(batch)

STOPPED = None
RUN_CONFIGS = {}
for run in [r for r in P["runs"] if r["name"] in RUN_ARMS]:
    try:
        RUN_CONFIGS[run["name"]], _ = C.generate_run(P, run, PLANS[run["name"]], ARM_PROMPTS[run["name"]], GEN, INFO, PRECISION, GPU,
                                                     OUT_DIR, {**ENV, "revision_check": REVISION_CHECK}, STOP_AFTER_BATCHES)
    except SessionTimeBudget:
        STOPPED = f"time budget of {TIME_BUDGET_HOURS} h reached during {run['name']}"
        print("STOPPED:", STOPPED)
        break
    if STOP_AFTER_BATCHES is not None:
        STOPPED = "STOP_AFTER_BATCHES (test hook)"
        break
""", "generate")

code(r"""
# ---- export: completeness per run, environment consistency, manifest with checksums, zip -------------------------
ST, MAN = C.export(P, PLANS, ARM_PROMPTS, OUT_DIR, OUTPUT_ZIP,
                   extra_status={"session": SESSION, "run_arms_this_session": RUN_ARMS, "resumed": RESUMED, "stopped": STOPPED,
                                 "oom_events_this_session": dict(OOM_EVENTS), "revision_check": REVISION_CHECK, "environment": ENV})
print("MQL-0009 STATUS:", ST["status"], "|", MAN["completed_predictions"], "/", MAN["expected_predictions"], "predictions")
print(json.dumps(MAN["per_arm"], indent=1))
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
