"""Generate kaggle/nl2mql_mql0007_rag_v1/nl2mql_mql0007_rag_v1.ipynb (experiment MQL-0007, GENERATE-ONLY).

Kaggle only generates. The notebook never contacts MongoDB Atlas, has no credential, no API key and no external model
API. Scoring runs later on Windows. Cells are tagged so tests/test_mql0007_package.py can execute every cell with a fake
generator (injectable GENERATOR_FACTORY) and a kill-and-resume run.

Usage: python scripts/nl2mql/make_mql0007_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kaggle/nl2mql_mql0007_rag_v1/nl2mql_mql0007_rag_v1.ipynb"
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src, tag):
    c = nbf.v4.new_code_cell(src.strip("\n"))
    c.metadata["tags"] = [tag]
    cells.append(c)


md(r"""
# MQL-0007 (generation only): zero-shot vs random vs relevant RAG, Qwen2.5-Coder-3B-Instruct, 892 NL→MQL questions

Three arms, one model, one prompt base, one decoding setup:

| arm | context |
|---|---|
| **A_zero_shot** | none (the frozen `nl-mql-spider1-v1` prompt). `ZERO_SHOT_MODE = "auto"` (default): MQL-0006's verified predictions are reused **only if** model revision, prompt, decoding, dtype, batch size, torch, transformers, accelerate, tokenizers **and the GPU name** equal MQL-0006's; otherwise arm A is generated fresh in this environment (about +2 h). `"reuse_mql0006"` = reuse or fail; `"fresh"` = always regenerate. |
| **B_random_k3** | 3 random training examples (frozen seed 20261020) from `provisional_corpus_v1` |
| **D_supp_allow_near_dup** (supplementary) | sensitivity arm: C *without* the near-duplicate exclusion, only on the item(s) the exclusion changes (1 item); never part of the 892-item checks |
| **C_hybrid_k3** | MQL-0005 `hybrid_masked` retrieval (BM25 k1=1.5 b=0.75 + 1.0 × pattern overlap), k=3, ≤1,500-token block, no guide cards, over-budget examples skipped, never truncated |

- **No Atlas, no secrets, no external model API.** Scoring happens on Windows afterwards.
- **Settings:** Accelerator *GPU T4 x2* (one card is used), Internet **on** (Hugging Face model download), Persistence: files.
- **Input:** the private dataset made from `nl2mql_mql0007_rag_v1_bundle.zip`. **Run:** Save Version → Save & Run All (Commit).
- Fixed MQL-0006 inference settings: bfloat16, greedy, batch 8, `max_new_tokens` 1024, one GPU. A required setting that
  cannot be met **stops the run**; nothing is silently changed. A batch that hits CUDA OOM is retried one prompt at a time
  (same prompt/dtype/decoding) and the record says so.
- Retrieval is **re-derived here** with the pinned model's tokenizer and compared with the frozen plans in the package;
  any difference stops the run.
- Checkpoints are per arm and fingerprinted; `RESUME_FROM = "auto"` continues an earlier version's output.
""")

code(r"""
# ---- configuration (edit here only) --------------------------------------------------------------
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

INPUT_ROOT = "/kaggle/input"
PKG_DIR = None                    # None = search INPUT_ROOT for package_manifest.json of nl2mql_mql0007_rag_v1
OUT_DIR = "/kaggle/working/mql0007_outputs"
OUTPUT_ZIP = "/kaggle/working/mql0007_predictions.zip"
RUN_ARMS = ["A_zero_shot", "B_random_k3", "C_hybrid_k3"]   # e.g. ["B_random_k3"] for one arm per Kaggle version
ZERO_SHOT_MODE = "auto"          # "auto" | "reuse_mql0006" (reuse or fail) | "fresh"
RUN_SUPPLEMENTARY = True          # also run the 1-item sensitivity arm D whenever C is run
RESUME_FROM = None                # None | "auto" (an earlier version's output attached under INPUT_ROOT) | its path
MODEL_CACHE = "/kaggle/working/hf_model"
MIN_GPU_GIB = 14.0                # MQL-0006 ran on a 15 GiB Tesla T4
INSTALL_MISSING = False           # pip-install ONLY missing packages (never upgrades preinstalled ones)

# ---- test hooks (leave the defaults on Kaggle) ----
GENERATOR_FACTORY = None          # callable(cfg) -> (run_batch, info, is_oom, cleanup) ; None = real Qwen on the GPU
TOKENIZER_DIR = None              # local tokenizer dir at the pinned revision; None = download from Hugging Face
STOP_AFTER_BATCHES = None         # kill-and-resume tests only
COMPAT_ENV_OVERRIDE = None        # tests only: {"packages": {...}, "gpu": "..."} replaces the observed environment for the reuse check
""", "config")

code(r"""
# ---- environment: dependencies, GPU, no credentials -----------------------------------------------
import json, platform, sys, time, zipfile
from importlib import metadata
from pathlib import Path

def _ver(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None

NEEDED = ["torch", "transformers", "accelerate", "tokenizers", "huggingface_hub"]
missing = [p for p in NEEDED if _ver(p) is None]
if missing and GENERATOR_FACTORY is None:
    if not INSTALL_MISSING:
        raise RuntimeError(f"missing packages {missing}: set INSTALL_MISSING = True (pip install of the missing ones only)")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

attached_env = [str(p) for p in Path(INPUT_ROOT).rglob(".env*")] if Path(INPUT_ROOT).exists() else []
if attached_env:
    raise RuntimeError(f"refusing to run: credential files attached as input: {attached_env}")

ENV = {"python": platform.python_version(), "platform": platform.platform(),
       "packages": {p: _ver(p) for p in ("torch", "transformers", "accelerate", "tokenizers")}}
GPU = {"checked": False}
if GENERATOR_FACTORY is None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU is visible (Settings -> Accelerator -> GPU T4 x2)")
    props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    GPU = {"checked": True, "visible_devices": len(props), "device0": props[0].name,
           "total_gib": round(props[0].total_memory / 2**30, 2), "bf16_supported": bool(torch.cuda.is_bf16_supported()),
           "free_gib": round(torch.cuda.mem_get_info(0)[0] / 2**30, 2)}
    if GPU["total_gib"] < MIN_GPU_GIB:
        raise RuntimeError(f"GPU memory {GPU['total_gib']} GiB < {MIN_GPU_GIB} GiB required (MQL-0006 used a 15 GiB T4); "
                           "refusing to run with a different memory budget")
    if not GPU["bf16_supported"]:
        raise RuntimeError("bfloat16 is required (MQL-0006 ran bfloat16) but is not supported; not falling back")
ENV["gpu"] = GPU
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
print(json.dumps(ENV, indent=1))
""", "env")

code(r"""
# ---- package integrity: every file hash, counts, config, leakage re-check on the shipped data -------
def _find_pkg():
    if PKG_DIR:
        return Path(PKG_DIR)
    hits = [m.parent for m in Path(INPUT_ROOT).rglob("package_manifest.json")
            if json.loads(m.read_text(encoding="utf-8")).get("package") == "nl2mql_mql0007_rag_v1"]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one nl2mql_mql0007_rag_v1 package under {INPUT_ROOT}, found {hits}")
    return hits[0]

PKG = _find_pkg()
sys.path.insert(0, str(PKG / "src"))
import mql0007_core as C

PKG_MANIFEST = C.verify_package(PKG)                # sha256 of every file, no extra files
CFG = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
CFG_SHA = C.sha256_file(PKG / "configs/experiment.json")
CORPUS_ROWS = C.read_jsonl(PKG / "data/corpus/provisional_rag_corpus.jsonl")
EVAL = C.read_jsonl(PKG / "data/eval/eval_inputs.jsonl")
PROMPTS = C.read_jsonl(PKG / "data/eval/prompts.jsonl")
ATLAS = json.loads((PKG / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
TABLES = {t["db_id"]: t for t in json.loads((PKG / "data/context/tables_subset.json").read_text(encoding="utf-8"))}

C.need(C.sha256_file(PKG / "data/corpus/provisional_rag_corpus.jsonl") == CFG["corpus"]["source_sha256"], "corpus hash != config")
C.need(len(CORPUS_ROWS) == CFG["corpus"]["n_entries"] == C.EXPECTED_CORPUS and
       len({r["id"] for r in CORPUS_ROWS}) == C.EXPECTED_CORPUS, "corpus is not exactly 3,589 unique entries")
C.need(len(EVAL) == CFG["evaluation"]["n_items"] == C.EXPECTED_EVAL and
       C.sha256_text("\n".join(e["id"] for e in EVAL)) == CFG["evaluation"]["ids_sha256"], "evaluation ids != 892 / ids hash")
C.need(set(EVAL[0]) == {"id", "db_id", "question"}, "evaluation inputs must hold only id, db_id, question")
C.need([p["id"] for p in PROMPTS] == [e["id"] for e in EVAL], "prompt ids/order != evaluation ids")
C.need({p["template_sha256"] for p in PROMPTS} == {CFG["prompt"]["zero_shot_template_sha256"]}, "zero-shot template hash")
_ph = C.sha256_text("".join(p["prompt_sha256"] + "\n" for p in PROMPTS))
EXPECTED_IDS = [e["id"] for e in EVAL]
EVAL_DBS = sorted({e["db_id"] for e in EVAL})
C.need(len(EVAL_DBS) == CFG["evaluation"]["n_databases"], "evaluation database count")
C.need(not {r["db_id"] for r in CORPUS_ROWS} & set(EVAL_DBS), "corpus and evaluation share a database")
C.need(all(r["source"]["split"] == "train" and r["verification"]["status"] == "accepted" for r in CORPUS_ROWS), "ineligible corpus row")
_ex = json.loads((PKG / "data/corpus/excluded_accepted_ids.json").read_text(encoding="utf-8"))
C.need(not ({r["id"] for r in CORPUS_ROWS} & (set(_ex["ids"]) | set(_ex["pre_excluded"]))), "corpus holds an excluded id")
_ack = CFG["leakage_gates"]["acknowledged_near_duplicates"]
_lk = C.leakage_report(CORPUS_ROWS, EVAL, CFG["leakage_gates"]["near_duplicate_token_jaccard"])
C.need(not _lk["exact"] and not _lk["db_overlap"] and
       {(e, c) for e, c, _ in _lk["near"]} == {(e, _ack["corpus_id"]) for e in _ack["eval_ids"]},
       f"leakage re-check failed beyond the acknowledged near-duplicates: {_lk['exact'][:3]} {_lk['near'][:5]}")
NEAR_DUP_ITEMS = sorted(_ack["eval_ids"])
C.retriever_constants_ok(CFG)
print(f"package OK: {PKG}\n  {len(PKG_MANIFEST['files'])} files hash-verified; corpus {len(CORPUS_ROWS)}; evaluation {len(EVAL)} items / "
      f"{len(EVAL_DBS)} DBs; 0 exact / 0 DB overlap; acknowledged template near-duplicates {NEAR_DUP_ITEMS} (Jaccard 0.80)")
""", "package")

code(r"""
# ---- model files at the pinned revision (public weights; no token needed) ----------------------------
M = CFG["model"]
if TOKENIZER_DIR is None and GENERATOR_FACTORY is None:
    from huggingface_hub import snapshot_download
    MODEL_PATH = snapshot_download(M["id"], revision=M["revision"], local_dir=MODEL_CACHE)
    from huggingface_hub import HfApi
    C.need(HfApi().model_info(M["id"], revision=M["revision"]).sha == M["revision"], "Hugging Face did not resolve the pinned revision")
    TOKENIZER_PATH = MODEL_PATH
else:
    MODEL_PATH, TOKENIZER_PATH = None, TOKENIZER_DIR
from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
count_tokens = lambda s: len(TOK.encode(s, add_special_tokens=False))
print("tokenizer:", TOKENIZER_PATH, "| model files:", MODEL_PATH)
""", "model")

code(r"""
# ---- deterministic retrieval: re-derive corpus features and every arm's plan, compare with the frozen ones ----
DOCS = C.build_docs(CORPUS_ROWS, TABLES)
DOCS_BY_ID = {d["doc_id"]: d for d in DOCS}
_frozen = C.read_jsonl(PKG / "data/corpus/corpus_features.jsonl")
C.need([{k: d[k] for k in ("doc_id", "db_id", "question", "masked_question", "patterns", "content")} for d in DOCS] == _frozen,
       "re-derived corpus features differ from the frozen corpus_features.jsonl")
EVAL_INFOS = C.eval_schema_infos(ATLAS, EVAL_DBS)
PLANS, ARM_PROMPTS = {}, {}
for arm in C.ARMS:
    plan = C.build_plan(CFG, arm, EVAL, PROMPTS, DOCS, EVAL_INFOS, count_tokens)
    if arm != "A_zero_shot":
        frozen = C.read_jsonl(PKG / f"data/plans/retrieval_plan_{arm}.jsonl")
        bad = [a["id"] for a, b in zip(plan, frozen) if a != b]
        C.need(len(plan) == len(frozen) and not bad, f"{arm}: re-derived retrieval plan differs from the frozen plan on "
               f"{len(bad)} items, e.g. {bad[:3]} (tokenizer/library difference?)")
    else:
        C.need(all(p["empty_context"] for p in plan), "arm A must have an empty context")
    for p in plan:
        p["near_duplicate_template_item"] = p["id"] in NEAR_DUP_ITEMS
        C.need(all(d != p["db_id"] for d in p["selected_db_ids"]), f"{p['id']}: retrieved from the target database")
        C.need(all(i in DOCS_BY_ID for i in p["selected_ids"]), f"{p['id']}: retrieved a non-corpus id")
    PLANS[arm] = plan
    ARM_PROMPTS[arm] = C.prompts_for_arm(plan, PROMPTS, DOCS_BY_ID)
    if arm != "A_zero_shot":   # selection-level near-duplicate gate + the evaluation-specific exclusions held
        viol = C.selection_neardup_violations(plan, EVAL, DOCS_BY_ID, CFG["leakage_gates"]["near_duplicate_token_jaccard"])
        C.need(not viol, f"{arm}: a selected example is a near-duplicate of its query: {viol[:3]}")
        for x in CFG["retrieval_exclusions"]:
            C.need(x["corpus_id"] not in next(p for p in plan if p["id"] == x["eval_id"])["selected_ids"], f"{arm}: exclusion {x} violated")
# supplementary sensitivity arm D: frozen plan (exclusion OFF) re-derived on exactly its frozen ids
_d_frozen = C.read_jsonl(PKG / f"data/plans/retrieval_plan_{C.SUPP_ARM}.jsonl")
_d_rows = [e for e in EVAL if e["id"] in {p["id"] for p in _d_frozen}]
_d_plan = C.build_plan(CFG, "C_hybrid_k3", _d_rows, PROMPTS, DOCS, EVAL_INFOS, count_tokens, exclusions=[], arm_label=C.SUPP_ARM)
C.need(_d_plan == _d_frozen, "supplementary plan differs from the frozen plan")
_c_by = {p["id"]: p for p in PLANS["C_hybrid_k3"]}
_changed = sorted(i for i, p in _c_by.items() if p["excluded_by_rule"] and p["excluded_by_rule"][0]["original_rank"] <= CFG["arms"]["C_hybrid_k3"]["k"])
C.need(sorted(p["id"] for p in _d_plan) == _changed and all(_c_by[p["id"]]["selected_ids"] != p["selected_ids"] for p in _d_plan),
       "supplementary ids must be exactly the items whose selection the exclusion changes")
PLANS[C.SUPP_ARM] = _d_plan
ARM_PROMPTS[C.SUPP_ARM] = C.prompts_for_arm(_d_plan, PROMPTS, DOCS_BY_ID)
C.need([p["prompt_sha256"] for p in ARM_PROMPTS["A_zero_shot"]] == [p["prompt_sha256"] for p in PROMPTS],
       "arm A prompts are not byte-identical to the frozen zero-shot prompts")
print({a: {"empty": sum(p["empty_context"] for p in PLANS[a]), "skipped_over_budget": sum(bool(p["dropped_budget"]) for p in PLANS[a]),
           "max_block_tokens": max(p["context_tokens"] for p in PLANS[a])} for a in PLANS})
""", "plan")

code(r"""
# ---- preflight: fixed settings, library differences, eligibility of the MQL-0006 zero-shot reuse ------------
PRE = {"started_at": C.now(), "checks": {}}
C.need(M["id"] == "Qwen/Qwen2.5-Coder-3B-Instruct" and M["revision"] == "488639f1ff808d1d3d0ba301aef8c11461451ec5", "model identity")
C.need((M["dtype"], M["batch_size"], M["max_new_tokens"], M["decoding"]) == ("bfloat16", 8, 1024, "greedy (do_sample=False)"), "decoding settings")
C.need(set(RUN_ARMS) <= set(C.ARMS) and len(set(RUN_ARMS)) == len(RUN_ARMS), f"RUN_ARMS must be a subset of {C.ARMS}")
M6_DIR = PKG / "data/zero_shot_mql0006"
M6_CFG = json.loads((M6_DIR / "run_config.json").read_text(encoding="utf-8"))
_lib = {p: (M6_CFG["environment"]["packages"].get(p), ENV["packages"].get(p)) for p in ("torch", "transformers", "accelerate", "tokenizers")}
PRE["checks"]["library_versions_mql0006_vs_now"] = {p: {"mql0006": a, "now": b, "same": a == b} for p, (a, b) in _lib.items()}
for p, (a, b) in _lib.items():
    if a != b:
        print(f"NOTE library differs from MQL-0006: {p} {a} -> {b}   (recorded; greedy outputs can differ across versions)")
C.need(ZERO_SHOT_MODE in ("auto", "reuse_mql0006", "fresh"), "ZERO_SHOT_MODE must be auto | reuse_mql0006 | fresh")
OBS = COMPAT_ENV_OVERRIDE or {"packages": ENV["packages"], "gpu": GPU.get("device0")}
COMPAT = {"libraries": {p: {"mql0006": M6_CFG["environment"]["packages"].get(p), "now": OBS["packages"].get(p),
                           "same": M6_CFG["environment"]["packages"].get(p) == OBS["packages"].get(p)}
                        for p in ("torch", "transformers", "accelerate", "tokenizers")},
          "gpu": {"mql0006": M6_CFG["gpu"], "now": OBS.get("gpu"), "same": M6_CFG["gpu"] == OBS.get("gpu")}}
COMPAT["all_same"] = all(v["same"] for v in COMPAT["libraries"].values()) and COMPAT["gpu"]["same"]
PRE["checks"]["zero_shot_environment_compat"] = COMPAT
ZS_RESOLVED = ZERO_SHOT_MODE
if ZERO_SHOT_MODE == "auto":
    ZS_RESOLVED = "reuse_mql0006" if COMPAT["all_same"] else "fresh"
elif ZERO_SHOT_MODE == "reuse_mql0006":
    C.need(COMPAT["all_same"], f"ZERO_SHOT_MODE='reuse_mql0006' but the environment differs from MQL-0006: {COMPAT}")
PRE["checks"]["zero_shot_resolved_mode"] = ZS_RESOLVED
print("arm A:", ZS_RESOLVED, "(requested " + ZERO_SHOT_MODE + ");", "environment identical to MQL-0006" if COMPAT["all_same"] else "environment differs from MQL-0006")
ZS_REUSE = None
if "A_zero_shot" in RUN_ARMS and ZS_RESOLVED == "reuse_mql0006":
    same = {"model_id": M["id"], "model_revision": M["revision"], "decoding": M["decoding"], "max_new_tokens": M["max_new_tokens"],
            "batch_size": M["batch_size"], "dtype": M["dtype"], "prompt_version": CFG["prompt"]["zero_shot_version"],
            "template_sha256": CFG["prompt"]["zero_shot_template_sha256"], "prompts_sha256": CFG["prompt"]["zero_shot_prompts_sha256"]}
    diff = {k: (M6_CFG.get(k), v) for k, v in same.items() if M6_CFG.get(k) != v}
    C.need(not diff, f"MQL-0006 predictions are NOT reusable (earlier, now): {diff}; set ZERO_SHOT_MODE = 'fresh'")
    sums = dict(reversed(ln.partition("  ")[::2]) for ln in (M6_DIR / "MQL-0006_SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())
    C.need(C.sha256_file(M6_DIR / "predictions.jsonl") == sums["models/qwen2.5-coder-3b-instruct/predictions.jsonl"] and
           C.sha256_file(M6_DIR / "run_config.json") == sums["models/qwen2.5-coder-3b-instruct/run_config.json"],
           "MQL-0006 files differ from their recorded hashes")
    zp = C.read_jsonl(M6_DIR / "predictions.jsonl")
    C.need([r["id"] for r in zp] == EXPECTED_IDS and all(r["status"] == "ok" for r in zp), "MQL-0006 predictions: ids/order/status")
    C.need([r["prompt_sha256"] for r in zp] == [p["prompt_sha256"] for p in PROMPTS], "MQL-0006 prompts differ from the frozen prompts")
    ZS_REUSE = {"n": len(zp), "fingerprint": M6_CFG["fingerprint"], "oom_batch1_items": sum(bool(r.get("oom_recovery")) for r in zp)}
    PRE["checks"]["zero_shot_reuse"] = ZS_REUSE
PRE["finished_at"] = C.now()
C.write_json(Path(OUT_DIR) / "preflight/preflight.json", {**PRE, "zero_shot_requested": ZERO_SHOT_MODE, "ok": True, "env": ENV, "package_manifest_sha256": C.sha256_file(PKG / "package_manifest.json")})
print("PREFLIGHT OK", json.dumps(PRE["checks"].get("zero_shot_reuse", "no reuse")))
""", "preflight")

code(r"""
# ---- generation: per-arm checkpoints, fingerprints, safe resume ------------------------------------------
import shutil
OOM_EVENTS = {"batch": 0, "single_prompt": 0}
_POST_FP = {"experiment", "fingerprint", "generation_wall_s", "resolved_revision", "generated_this_session", "environment", "reused_from"}

def arm_dir(arm):
    return Path(OUT_DIR) / "models" / arm

def _sums_ok(prev):
    sums = {}
    for ln in (prev / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, _, n = ln.partition("  ")
        sums[n] = h
    files = {p.relative_to(prev).as_posix() for p in prev.rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
    C.need(files == set(sums) and all(C.sha256_file(prev / n) == h for n, h in sums.items()), f"resume source {prev}: hash check failed")

GEN_ARMS = list(RUN_ARMS) + ([C.SUPP_ARM] if RUN_SUPPLEMENTARY and "C_hybrid_k3" in RUN_ARMS else [])
ARM_IDS = {a: [p["id"] for p in PLANS[a]] for a in C.ARMS + (C.SUPP_ARM,)}
RESUMED = None
if RESUME_FROM:
    hits = sorted({p.parent for p in Path(INPUT_ROOT).rglob("MQL0007_STATUS.json")}) if RESUME_FROM == "auto" else [Path(RESUME_FROM)]
    hits = [h for h in hits if h.resolve() != Path(OUT_DIR).resolve()]
    C.need(len(hits) == 1, f"RESUME_FROM: expected exactly one earlier output, found {hits}")
    prev = hits[0]
    _sums_ok(prev)
    st = json.loads((prev / "MQL0007_STATUS.json").read_text(encoding="utf-8"))
    C.need(st.get("experiment") == "MQL-0007" and st.get("package_manifest_sha256") == C.sha256_file(PKG / "package_manifest.json"),
           "resume source: another experiment or another package")
    seeded = []
    for arm in GEN_ARMS:
        for rel in (f"models/{arm}/checkpoints/predictions.partial.jsonl", f"models/{arm}/run_config.json"):
            if (prev / rel).is_file() and not (Path(OUT_DIR) / rel).exists():
                (Path(OUT_DIR) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(prev / rel, Path(OUT_DIR) / rel)
                seeded.append(rel)
    RESUMED = {"from": str(prev), "seeded": seeded}
    print("resume source verified:", RESUMED)

NEED_MODEL = any(a != "A_zero_shot" or ZS_REUSE is None for a in GEN_ARMS)
RUN_BATCH = INFO = None
if NEED_MODEL:
    if GENERATOR_FACTORY is not None:
        RUN_BATCH, INFO, IS_OOM, CLEANUP = GENERATOR_FACTORY(CFG)
    else:
        RUN_BATCH, INFO, IS_OOM, CLEANUP, _tok = C.make_qwen_generator(M["id"], M["revision"], M["max_new_tokens"], M["dtype"],
                                                                         local_dir=MODEL_PATH)
    GEN = C.oom_safe_generator(RUN_BATCH, IS_OOM, CLEANUP, M["batch_size"], OOM_EVENTS)
    C.need(INFO.get("dtype") == M["dtype"], f"model dtype {INFO.get('dtype')} != required {M['dtype']}")

ARM_RESULT, RUN_CONFIGS = {}, {}
for arm in GEN_ARMS:
    d, t0 = arm_dir(arm), time.time()
    prompts = ARM_PROMPTS[arm]
    C.write_jsonl(d / "retrieval_log.jsonl", PLANS[arm])
    if arm == "A_zero_shot" and ZS_REUSE is not None:
        (d).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(M6_DIR / "predictions.jsonl", d / "predictions.jsonl")          # byte-identical to MQL-0006
        cfg_a = {"experiment": "MQL-0007", "arm": arm, "mode": "reuse_mql0006", "reused_from": "MQL-0006 (verified 2026-10-30)",
                 "fingerprint": M6_CFG["fingerprint"], "model_id": M["id"], "model_revision": M["revision"], "dtype": M6_CFG["dtype"],
                 "gpu": M6_CFG["gpu"], "torch": M6_CFG["torch"], "transformers": M6_CFG["transformers"], "decoding": M["decoding"],
                 "max_new_tokens": M["max_new_tokens"], "batch_size": M["batch_size"], "prompts_sha256": CFG["prompt"]["zero_shot_prompts_sha256"],
                 "mql0006_oom_batch1_items": ZS_REUSE["oom_batch1_items"], "generated_this_session": 0}
        C.write_json(d / "run_config.json", cfg_a)
        RUN_CONFIGS[arm] = cfg_a
        ARM_RESULT[arm] = {"generated": 0}
        print(f"{arm}: reused {ZS_REUSE['n']} MQL-0006 predictions byte for byte")
        continue
    qcfg = {"experiment": "MQL-0007", "arm": arm, "mode": "generated", "arm_config": CFG["arms"][arm], "retrieval_exclusions": CFG["retrieval_exclusions"] if arm in ("B_random_k3", "C_hybrid_k3") else [], "model_label": M["label"],
            "model_id": M["id"], "model_revision": M["revision"], "runtime": M["runtime"], "decoding": M["decoding"],
            "max_new_tokens": M["max_new_tokens"], "batch_size": M["batch_size"], "seed": M["seed"],
            "prompt_version": CFG["prompt"]["rag_version"] if arm != "A_zero_shot" else CFG["prompt"]["zero_shot_version"],
            "arm_prompts_sha256": C.sha256_text("".join(p["prompt_sha256"] + "\n" for p in prompts)),
            "plan_sha256": C.sha256_text(C.canon([{k: v for k, v in p.items() if k != "near_duplicate_template_item"} for p in PLANS[arm]])),
            "corpus_sha256": CFG["corpus"]["source_sha256"], "config_sha256": CFG_SHA,
            **{k: INFO.get(k) for k in ("dtype", "gpu", "torch", "transformers")}}
    qcfg["fingerprint"] = C.fingerprint({k: v for k, v in qcfg.items() if k not in _POST_FP})
    old = d / "run_config.json"
    if old.exists():
        prev_cfg = json.loads(old.read_text(encoding="utf-8"))
        diff = {k: (prev_cfg.get(k), v) for k, v in qcfg.items() if k not in _POST_FP and prev_cfg.get(k) != v}
        C.need(not diff, f"{arm}: earlier run_config differs (earlier, now): {diff}; refusing to mix")
    ck = C.Checkpoint(d / "checkpoints/predictions.partial.jsonl", ARM_IDS[arm], qcfg["fingerprint"], C.is_generation_retry)
    print(f"{arm}: {len(ck.todo())} of {len(ARM_IDS[arm])} to generate")
    try:
        r = C.run_generation(prompts, GEN, ck, M["label"], arm, M["batch_size"], max_batches=STOP_AFTER_BATCHES,
                             progress=lambda n, tot: print(f"  {arm} {n}/{tot}", flush=True) if n % 80 < M["batch_size"] else None)
    finally:
        qcfg.update(generation_wall_s=round(time.time() - t0, 1), resolved_revision=INFO.get("resolved_revision"),
                    generated_this_session=None, environment=ENV)
        C.write_jsonl(d / "predictions.jsonl", ck.ordered())
        C.write_json(d / "run_config.json", qcfg)
    qcfg["generated_this_session"] = r["generated"]
    C.write_json(d / "run_config.json", qcfg)
    RUN_CONFIGS[arm], ARM_RESULT[arm] = qcfg, r
    if STOP_AFTER_BATCHES is not None:
        break
""", "generate")

code(r"""
# ---- export: completeness (no duplicate / missing ids), status, checksums, zip ------------------------------
problems, arms = [], {}
for arm in GEN_ARMS:
    EXP = ARM_IDS[arm]
    pf = arm_dir(arm) / "predictions.jsonl"
    if not pf.exists():
        problems.append(f"{arm}: no predictions"); continue
    preds = C.read_jsonl(pf)
    ids = [p["id"] for p in preds]
    n_ok = sum(p["status"] == "ok" for p in preds)
    mp = []
    if len(set(ids)) != len(ids):
        mp.append("duplicate ids")
    if ids != [i for i in EXP if i in set(ids)] or set(ids) - set(EXP):
        mp.append("ids out of order or unknown")
    if len(ids) != len(EXP):
        mp.append(f"{len(EXP) - len(ids)} ids missing")
    if n_ok != len(ids):
        mp.append(f"{len(ids) - n_ok} generation errors (retried on resume)")
    want_sha = {p["id"]: p["prompt_sha256"] for p in ARM_PROMPTS[arm]}
    if any(p.get("prompt_sha256") != want_sha[p["id"]] for p in preds):
        mp.append("a prediction was generated from a different prompt")
    if arm != "A_zero_shot":
        rl = C.read_jsonl(arm_dir(arm) / "retrieval_log.jsonl")
        if [r["id"] for r in rl] != EXP:
            mp.append("retrieval log ids != evaluation ids")
    arms[arm] = {"status": "complete" if not mp else "partial", "problems": mp, "n": len(ids), "n_ok": n_ok,
                 "predictions_sha256": C.sha256_file(pf), "fingerprint": RUN_CONFIGS.get(arm, {}).get("fingerprint"),
                 "mode": RUN_CONFIGS.get(arm, {}).get("mode"),
                 "oom_batch1_fallback_items": sum(bool(p.get("oom_recovery")) for p in preds),
                 "tokens": {"input": sum(p.get("input_tokens") or 0 for p in preds), "output": sum(p.get("output_tokens") or 0 for p in preds)}}
    problems += [f"{arm}: {x}" for x in mp]
status = "complete" if arms and not problems and set(arms) == set(GEN_ARMS) else "partial"
C.write_json(Path(OUT_DIR) / "MQL0007_STATUS.json", {
    "experiment": "MQL-0007", "stage": "predictions", "status": status, "written_at": C.now(), "problems": problems, "arms": arms,
    "run_arms": GEN_ARMS, "zero_shot_requested": ZERO_SHOT_MODE, "zero_shot_resolved": ZS_RESOLVED, "zero_shot_env_compat": COMPAT,
    "arm_expected_ids": {a: len(ARM_IDS[a]) for a in GEN_ARMS}, "resumed": RESUMED, "oom_events_this_session": dict(OOM_EVENTS),
    "package_manifest_sha256": C.sha256_file(PKG / "package_manifest.json"), "config_sha256": CFG_SHA,
    "evaluation_ids_sha256": CFG["evaluation"]["ids_sha256"], "near_duplicate_template_items": NEAR_DUP_ITEMS,
    "atlas": "not contacted (scoring runs locally on Windows)", "external_model_apis": "none", "environment": ENV})
names = sorted(p.relative_to(OUT_DIR).as_posix() for p in Path(OUT_DIR).rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
(Path(OUT_DIR) / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(Path(OUT_DIR) / n)}  {n}\n" for n in names), encoding="utf-8")
with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(Path(OUT_DIR).rglob("*")):
        if p.is_file():
            z.write(p, f"{Path(OUT_DIR).name}/{p.relative_to(OUT_DIR).as_posix()}")
print("MQL-0007 STATUS:", status); print("\n".join(problems) or "no problems"); print(json.dumps(arms, indent=1))
if status != "complete":
    print("NOT COMPLETE: attach this version's output as input, set RESUME_FROM = 'auto', and run again.")
""", "export")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT)
