"""Generate kaggle/nl2mql_mql0008_approved_corpus_v2_1/nl2mql_mql0008_approved_corpus_v2_1.ipynb (MQL-0008, GENERATE-ONLY).

Kaggle only generates. The notebook never contacts MongoDB or Atlas, has no credential, no API key and no external
model API. Scoring runs later on Windows. Cells are tagged so the package tests can execute every cell with a fake
generator (injectable GENERATOR_FACTORY) and a kill-and-resume run. Derived from make_mql0007_notebook.py.

Usage: python scripts/nl2mql/make_mql0008_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kaggle/nl2mql_mql0008_approved_corpus_v2_1/nl2mql_mql0008_approved_corpus_v2_1.ipynb"
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src, tag):
    c = nbf.v4.new_code_cell(src.strip("\n"))
    c.metadata["tags"] = [tag]
    cells.append(c)


md(r"""
# MQL-0008 (generation only): zero-shot vs hybrid RAG over the approved corpus v2.1

Qwen2.5-Coder-3B-Instruct, 892 NL→MQL questions, **exactly two arms**, one model, one prompt base, one decoding setup:

| arm | context | predictions |
|---|---|---|
| **arm_a_zero_shot** | none (the frozen `nl-mql-spider1-v1` prompt) | 892, generated in this run |
| **arm_b_hybrid_rag** | 3 examples from the **approved corpus v2.1** (3,445 records), `hybrid_masked` retrieval: BM25 k1=1.5 b=0.75 + 1.0 × pattern overlap, block ≤ 1,500 tokens, target database excluded, no guide cards, over-budget examples skipped, never truncated | 892, generated in this run |

Total: 1,784 predictions. The retrieved examples are the only difference between the arms.

- **No MongoDB, no Atlas, no secrets, no external model API.** Scoring happens on Windows afterwards.
- **Settings:** Accelerator *GPU T4 x2* (one card is used), Internet **on** (Hugging Face model download), Persistence: files.
- **Input:** the private dataset made from `nl2mql_mql0008_approved_corpus_v2_1_bundle.zip`. **Run:** Save Version → Save & Run All (Commit).
- Fixed inference settings (as MQL-0006 / MQL-0007): bfloat16, greedy, batch 8, `max_new_tokens` 1024, one GPU. A required
  setting that cannot be met **stops the run**; nothing is silently changed. A batch that hits CUDA OOM is retried one prompt
  at a time (same prompt/dtype/decoding) and the record says so.
- Retrieval is **re-derived here** with the pinned model's tokenizer and compared with the frozen plan in the package;
  any difference stops the run.
- Checkpoints are per arm and fingerprinted; `RESUME_FROM = "auto"` continues an earlier version's output.
""")

code(r"""
# ---- configuration (edit here only) --------------------------------------------------------------
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

INPUT_ROOT = "/kaggle/input"
PKG_DIR = None                    # None = search INPUT_ROOT for package_manifest.json of nl2mql_mql0008_approved_corpus_v2_1
OUT_DIR = "/kaggle/working/mql0008_outputs"
OUTPUT_ZIP = "/kaggle/working/mql0008_predictions.zip"
RUN_ARMS = ["arm_a_zero_shot", "arm_b_hybrid_rag"]   # one arm per Kaggle version is also fine, e.g. ["arm_b_hybrid_rag"]
RESUME_FROM = None                # None | "auto" (an earlier version's output attached under INPUT_ROOT) | its path
MODEL_CACHE = "/kaggle/working/hf_model"
MIN_GPU_GIB = 14.0                # MQL-0006 / MQL-0007 ran on a 15 GiB Tesla T4
MIN_FREE_DISK_GIB = 10.0          # model files (~6.2 GiB) + outputs
MIN_RAM_GIB = 8.0
INSTALL_MISSING = False           # pip-install ONLY missing packages (never upgrades preinstalled ones)

# ---- test hooks (leave the defaults on Kaggle) ----
GENERATOR_FACTORY = None          # callable(cfg) -> (run_batch, info, is_oom, cleanup) ; None = real Qwen on the GPU
TOKENIZER_DIR = None              # local tokenizer dir at the pinned revision; None = download from Hugging Face
STOP_AFTER_BATCHES = None         # kill-and-resume tests only
""", "config")

code(r"""
# ---- environment: dependencies, GPU and memory preflight, no credentials ---------------------------
import json, platform, shutil, sys, time, zipfile
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

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
ENV = {"python": platform.python_version(), "platform": platform.platform(),
       "packages": {p: _ver(p) for p in ("torch", "transformers", "accelerate", "tokenizers")}}
try:
    _ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
except (AttributeError, ValueError, OSError):
    _ram = None                                           # not available on this platform (local tests on Windows)
HOST = {"ram_gib": _ram, "free_disk_gib": round(shutil.disk_usage(OUT_DIR).free / 2**30, 2)}
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
        raise RuntimeError(f"GPU memory {GPU['total_gib']} GiB < {MIN_GPU_GIB} GiB required (earlier runs used a 15 GiB T4); "
                           "refusing to run with a different memory budget")
    if not GPU["bf16_supported"]:
        raise RuntimeError("bfloat16 is required (earlier runs used bfloat16) but is not supported; not falling back")
    if HOST["free_disk_gib"] < MIN_FREE_DISK_GIB:
        raise RuntimeError(f"free disk {HOST['free_disk_gib']} GiB < {MIN_FREE_DISK_GIB} GiB needed for the model files and outputs")
    if HOST["ram_gib"] is not None and HOST["ram_gib"] < MIN_RAM_GIB:
        raise RuntimeError(f"RAM {HOST['ram_gib']} GiB < {MIN_RAM_GIB} GiB")
ENV["gpu"], ENV["host"] = GPU, HOST
print(json.dumps(ENV, indent=1))
""", "env")

code(r"""
# ---- package integrity: every file hash, corpus identity, counts, config, leakage re-check on the shipped data ----
def _find_pkg():
    if PKG_DIR:
        return Path(PKG_DIR)
    hits = [m.parent for m in Path(INPUT_ROOT).rglob("package_manifest.json")
            if json.loads(m.read_text(encoding="utf-8")).get("package") == "nl2mql_mql0008_approved_corpus_v2_1"]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one nl2mql_mql0008_approved_corpus_v2_1 package under {INPUT_ROOT}, found {hits}")
    return hits[0]

PKG = _find_pkg()
sys.path.insert(0, str(PKG / "src"))
import mql0008_core as C

PKG_MANIFEST = C.verify_package(PKG)                # sha256 of every file, no extra files
CFG = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
CFG_SHA = C.sha256_file(PKG / "configs/experiment.json")
CORPUS_SHA = C.sha256_file(PKG / C.CORPUS_FILE)
C.need(CORPUS_SHA == C.EXPECTED_CORPUS_SHA256 == CFG["corpus"]["source_sha256"] == PKG_MANIFEST["retrieval_corpus"]["sha256"],
       f"retrieval corpus is not the approved corpus v2.1 (sha256 {CORPUS_SHA})")
CORPUS_ROWS = C.read_jsonl(PKG / C.CORPUS_FILE)
EVAL = C.read_jsonl(PKG / "data/eval/eval_inputs.jsonl")
PROMPTS = C.read_jsonl(PKG / "data/eval/prompts.jsonl")
ATLAS = json.loads((PKG / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
TABLES = {t["db_id"]: t for t in json.loads((PKG / "data/context/tables_subset.json").read_text(encoding="utf-8"))}

C.need(len(CORPUS_ROWS) == CFG["corpus"]["n_entries"] == C.EXPECTED_CORPUS and
       len({r["id"] for r in CORPUS_ROWS}) == C.EXPECTED_CORPUS, "corpus is not exactly 3,445 unique entries")
C.need(not [p.name for p in (PKG / "data/corpus").iterdir() if p.name not in ("approved_corpus_v2_1.jsonl", "corpus_features.jsonl")],
       "an unexpected corpus file is in the package")
C.need(tuple(CFG["arms"]) == C.ARMS and len(C.ARMS) == 2, "the experiment must have exactly two arms")
C.need(len(EVAL) == CFG["evaluation"]["n_items"] == C.EXPECTED_EVAL and len({e["id"] for e in EVAL}) == C.EXPECTED_EVAL and
       C.sha256_text("\n".join(e["id"] for e in EVAL)) == CFG["evaluation"]["ids_sha256"], "evaluation ids != 892 / ids hash")
C.need(all(set(e) == {"id", "db_id", "question"} for e in EVAL), "evaluation inputs must hold only id, db_id, question")
C.need([p["id"] for p in PROMPTS] == [e["id"] for e in EVAL], "prompt ids/order != evaluation ids")
C.need({p["template_sha256"] for p in PROMPTS} == {CFG["prompt"]["zero_shot_template_sha256"]}, "zero-shot template hash")
EXPECTED_IDS = [e["id"] for e in EVAL]
EVAL_DBS = sorted({e["db_id"] for e in EVAL})
C.need(len(EVAL_DBS) == CFG["evaluation"]["n_databases"], "evaluation database count")
C.need(not {r["db_id"] for r in CORPUS_ROWS} & set(EVAL_DBS), "corpus and evaluation share a database")
C.need(all(r["id"].startswith("spider_train_") and r["verification"]["outcome"] == "ACCEPTED" and
           r["converter_version"] == "sql2mql-v2.1" for r in CORPUS_ROWS), "ineligible corpus row")
_ack = CFG["leakage_gates"]["acknowledged_near_duplicates"]
_lk = C.leakage_report(CORPUS_ROWS, EVAL, CFG["leakage_gates"]["near_duplicate_token_jaccard"])
C.need(not _lk["exact"] and not _lk["db_overlap"] and
       {(e, c) for e, c, _ in _lk["near"]} == {(e, _ack["corpus_id"]) for e in _ack["eval_ids"]},
       f"leakage re-check failed beyond the documented near-duplicates: {_lk['exact'][:3]} {_lk['near'][:5]}")
C.need({(x["eval_id"], x["corpus_id"]) for x in CFG["retrieval_exclusions"]} == {(e, _ack["corpus_id"]) for e in _ack["eval_ids"]},
       "retrieval exclusions != the documented near-duplicate pairs")
NEAR_DUP_ITEMS = sorted(_ack["eval_ids"])
C.retriever_constants_ok(CFG)
RETRIEVAL_FP = C.retrieval_fingerprint(CFG)
C.need(RETRIEVAL_FP == PKG_MANIFEST["retrieval_config_fingerprint"], "retrieval configuration fingerprint != manifest")
print(f"package OK: {PKG}\n  {len(PKG_MANIFEST['files'])} files hash-verified; approved corpus v2.1: {len(CORPUS_ROWS)} records, sha256 {CORPUS_SHA}\n"
      f"  evaluation {len(EVAL)} items / {len(EVAL_DBS)} DBs; 0 exact / 0 DB overlap; documented template near-duplicates "
      f"{NEAR_DUP_ITEMS} (Jaccard 0.80, excluded from retrieval for those items)")
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
# ---- deterministic retrieval: re-derive corpus features and the plan, compare with the frozen ones ----
DOCS = C.build_docs(CORPUS_ROWS, TABLES)
DOCS_BY_ID = {d["doc_id"]: d for d in DOCS}
_frozen = C.read_jsonl(PKG / "data/corpus/corpus_features.jsonl")
C.need([{k: d[k] for k in ("doc_id", "db_id", "question", "masked_question", "patterns", "content")} for d in DOCS] == _frozen,
       "re-derived corpus features differ from the frozen corpus_features.jsonl")
EVAL_INFOS = C.eval_schema_infos(ATLAS, EVAL_DBS)
PLANS, ARM_PROMPTS = {}, {}
for arm in C.ARMS:
    plan = C.build_plan(CFG, arm, EVAL, PROMPTS, DOCS, EVAL_INFOS, count_tokens)
    if arm == C.RAG_ARM:
        frozen = C.read_jsonl(PKG / f"data/plans/retrieval_plan_{arm}.jsonl")
        bad = [a["id"] for a, b in zip(plan, frozen) if a != b]
        C.need(len(plan) == len(frozen) and not bad, f"{arm}: re-derived retrieval plan differs from the frozen plan on "
               f"{len(bad)} items, e.g. {bad[:3]} (tokenizer/library difference?)")
        viol = C.selection_neardup_violations(plan, EVAL, DOCS_BY_ID, CFG["leakage_gates"]["near_duplicate_token_jaccard"])
        C.need(not viol, f"{arm}: a selected example is a near-duplicate of its query: {viol[:3]}")
        for x in CFG["retrieval_exclusions"]:
            C.need(x["corpus_id"] not in next(p for p in plan if p["id"] == x["eval_id"])["selected_ids"], f"{arm}: exclusion {x} violated")
        C.need(all(p["context_tokens"] <= CFG["context"]["max_context_tokens"] for p in plan), "context token cap exceeded")
    else:
        C.need(all(p["empty_context"] and not p["selected_ids"] for p in plan), "arm A must have an empty context")
    for p in plan:
        p["near_duplicate_template_item"] = p["id"] in NEAR_DUP_ITEMS
        C.need(all(d != p["db_id"] for d in p["selected_db_ids"]), f"{p['id']}: retrieved from the target database")
        C.need(all(i in DOCS_BY_ID for i in p["selected_ids"]), f"{p['id']}: retrieved a non-corpus id")
    PLANS[arm] = plan
    ARM_PROMPTS[arm] = C.prompts_for_arm(plan, PROMPTS, DOCS_BY_ID)
C.need([p["prompt_sha256"] for p in ARM_PROMPTS["arm_a_zero_shot"]] == [p["prompt_sha256"] for p in PROMPTS],
       "arm A prompts are not byte-identical to the frozen zero-shot prompts")
C.need(all(a["system"] == b["system"] for a, b in zip(ARM_PROMPTS["arm_a_zero_shot"], ARM_PROMPTS[C.RAG_ARM])),
       "the system prompt differs between the arms")
print({a: {"empty": sum(p["empty_context"] for p in PLANS[a]), "skipped_over_budget": sum(bool(p["dropped_budget"]) for p in PLANS[a]),
           "max_block_tokens": max(p["context_tokens"] for p in PLANS[a])} for a in PLANS})
""", "plan")

code(r"""
# ---- preflight: fixed settings ----------------------------------------------------------------------------
PRE = {"started_at": C.now(), "checks": {}}
C.need(M["id"] == "Qwen/Qwen2.5-Coder-3B-Instruct" and M["revision"] == "488639f1ff808d1d3d0ba301aef8c11461451ec5", "model identity")
C.need((M["dtype"], M["batch_size"], M["max_new_tokens"], M["decoding"]) == ("bfloat16", 8, 1024, "greedy (do_sample=False)"), "decoding settings")
C.need(set(RUN_ARMS) <= set(C.ARMS) and len(set(RUN_ARMS)) == len(RUN_ARMS) and RUN_ARMS, f"RUN_ARMS must be a non-empty subset of {C.ARMS}")
PRE["checks"] = {"model": {"id": M["id"], "revision": M["revision"]}, "corpus_sha256": CORPUS_SHA, "corpus_records": len(CORPUS_ROWS),
                 "evaluation_items": len(EVAL), "arms": list(C.ARMS), "run_arms": list(RUN_ARMS),
                 "retrieval_config_fingerprint": RETRIEVAL_FP, "plan_equals_frozen_plan": True,
                 "expected_predictions_when_all_arms_run": len(EVAL) * len(C.ARMS)}
PRE["finished_at"] = C.now()
C.write_json(Path(OUT_DIR) / "preflight/preflight.json", {**PRE, "ok": True, "env": ENV, "package_manifest_sha256": C.sha256_file(PKG / "package_manifest.json")})
print("PREFLIGHT OK", json.dumps(PRE["checks"]))
""", "preflight")

code(r"""
# ---- generation: per-arm checkpoints, fingerprints, safe resume ------------------------------------------
OOM_EVENTS = {"batch": 0, "single_prompt": 0}
_POST_FP = {"experiment", "fingerprint", "generation_wall_s", "resolved_revision", "generated_this_session", "environment"}

def arm_dir(arm):
    return Path(OUT_DIR) / "predictions" / arm

def _sums_ok(prev):
    sums = {}
    for ln in (prev / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, _, n = ln.partition("  ")
        sums[n] = h
    files = {p.relative_to(prev).as_posix() for p in prev.rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
    C.need(files == set(sums) and all(C.sha256_file(prev / n) == h for n, h in sums.items()), f"resume source {prev}: hash check failed")

GEN_ARMS = [a for a in C.ARMS if a in RUN_ARMS]
ARM_IDS = {a: [p["id"] for p in PLANS[a]] for a in C.ARMS}
RESUMED = None
if RESUME_FROM:
    hits = sorted({p.parent for p in Path(INPUT_ROOT).rglob("MQL0008_STATUS.json")}) if RESUME_FROM == "auto" else [Path(RESUME_FROM)]
    hits = [h for h in hits if h.resolve() != Path(OUT_DIR).resolve()]
    C.need(len(hits) == 1, f"RESUME_FROM: expected exactly one earlier output, found {hits}")
    prev = hits[0]
    _sums_ok(prev)
    st = json.loads((prev / "MQL0008_STATUS.json").read_text(encoding="utf-8"))
    C.need(st.get("experiment") == "MQL-0008" and st.get("package_manifest_sha256") == C.sha256_file(PKG / "package_manifest.json"),
           "resume source: another experiment or another package")
    seeded = []
    for arm in GEN_ARMS:
        for rel in (f"predictions/{arm}/checkpoints/predictions.partial.jsonl", f"predictions/{arm}/run_config.json"):
            if (prev / rel).is_file() and not (Path(OUT_DIR) / rel).exists():
                (Path(OUT_DIR) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(prev / rel, Path(OUT_DIR) / rel)
                seeded.append(rel)
    RESUMED = {"from": str(prev), "seeded": seeded}
    print("resume source verified:", RESUMED)

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
    is_rag = arm == C.RAG_ARM
    qcfg = {"experiment": "MQL-0008", "arm": arm, "mode": "generated", "arm_config": CFG["arms"][arm],
            "retrieval_exclusions": CFG["retrieval_exclusions"] if is_rag else [], "model_label": M["label"],
            "model_id": M["id"], "model_revision": M["revision"], "runtime": M["runtime"], "decoding": M["decoding"],
            "max_new_tokens": M["max_new_tokens"], "batch_size": M["batch_size"], "seed": M["seed"],
            "prompt_version": CFG["prompt"]["rag_version"] if is_rag else CFG["prompt"]["zero_shot_version"],
            "arm_prompts_sha256": C.sha256_text("".join(p["prompt_sha256"] + "\n" for p in prompts)),
            "plan_sha256": C.sha256_text(C.canon([{k: v for k, v in p.items() if k != "near_duplicate_template_item"} for p in PLANS[arm]])),
            "corpus_sha256": CORPUS_SHA if is_rag else None, "corpus_records": len(CORPUS_ROWS) if is_rag else None,
            "retrieval_config_fingerprint": RETRIEVAL_FP if is_rag else None,
            "evaluation_ids_sha256": CFG["evaluation"]["ids_sha256"], "config_sha256": CFG_SHA,
            "package_manifest_sha256": C.sha256_file(PKG / "package_manifest.json"),
            **{k: INFO.get(k) for k in ("dtype", "gpu", "torch", "transformers")}}
    qcfg["fingerprint"] = C.fingerprint({k: v for k, v in qcfg.items() if k not in _POST_FP})
    old = d / "run_config.json"
    if old.exists():
        prev_cfg = json.loads(old.read_text(encoding="utf-8"))
        diff = {k: (prev_cfg.get(k), v) for k, v in qcfg.items() if k not in _POST_FP and prev_cfg.get(k) != v}
        C.need(not diff, f"{arm}: earlier run_config differs (earlier, now): {diff}; refusing to mix")
    ck = C.Checkpoint(d / "checkpoints/predictions.partial.jsonl", ARM_IDS[arm], qcfg["fingerprint"], C.is_generation_retry)
    identity = {p["id"]: C.record_identity(CFG, arm, p, RETRIEVAL_FP) for p in PLANS[arm]}
    print(f"{arm}: {len(ck.todo())} of {len(ARM_IDS[arm])} to generate")
    try:
        r = C.run_generation(prompts, GEN, ck, M["label"], arm, M["batch_size"], max_batches=STOP_AFTER_BATCHES, identity=identity,
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
# ---- export: coverage (no duplicate / missing ids), identity, status, checksums, zip -------------------------
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
    if any(p.get("arm") != arm or p.get("model_id") != M["id"] or p.get("model_revision") != M["revision"] for p in preds):
        mp.append("a prediction carries another arm or model identity")
    plan_by = {p["id"]: p for p in PLANS[arm]}
    if arm == C.RAG_ARM:
        if any(not p.get("retrieval") or p["retrieval"]["retrieved_ids"] != plan_by[p["id"]]["selected_ids"]
               or not p["retrieval"]["target_db_excluded"] or p["retrieval"]["corpus_sha256"] != CORPUS_SHA for p in preds):
            mp.append("retrieval metadata of a prediction differs from the plan")
    elif any(p.get("retrieval") is not None for p in preds):
        mp.append("a zero-shot prediction carries retrieval metadata")
    if [r["id"] for r in C.read_jsonl(arm_dir(arm) / "retrieval_log.jsonl")] != EXP:
        mp.append("retrieval log ids != evaluation ids")
    arms[arm] = {"status": "complete" if not mp else "partial", "problems": mp, "n": len(ids), "n_ok": n_ok,
                 "predictions_sha256": C.sha256_file(pf), "fingerprint": RUN_CONFIGS.get(arm, {}).get("fingerprint"),
                 "oom_batch1_fallback_items": sum(bool(p.get("oom_recovery")) for p in preds),
                 "finish_length_items": sum(p.get("finish_reason") == "length" for p in preds),
                 "tokens": {"input": sum(p.get("input_tokens") or 0 for p in preds), "output": sum(p.get("output_tokens") or 0 for p in preds)}}
    problems += [f"{arm}: {x}" for x in mp]
status = "complete" if arms and not problems and set(arms) == set(GEN_ARMS) else "partial"
C.write_json(Path(OUT_DIR) / "MQL0008_STATUS.json", {
    "experiment": "MQL-0008", "stage": "predictions", "status": status, "written_at": C.now(), "problems": problems, "arms": arms,
    "run_arms": GEN_ARMS, "all_arms_in_this_output": set(GEN_ARMS) == set(C.ARMS),
    "predictions_total": sum(a["n"] for a in arms.values()), "predictions_expected_for_run_arms": sum(len(ARM_IDS[a]) for a in GEN_ARMS),
    "arm_expected_ids": {a: len(ARM_IDS[a]) for a in GEN_ARMS}, "resumed": RESUMED, "oom_events_this_session": dict(OOM_EVENTS),
    "package": PKG_MANIFEST["package"], "package_manifest_sha256": C.sha256_file(PKG / "package_manifest.json"), "config_sha256": CFG_SHA,
    "retrieval_corpus": {"name": CFG["corpus"]["name"], "source_path": CFG["corpus"]["source_path"], "records": len(CORPUS_ROWS), "sha256": CORPUS_SHA},
    "retrieval_config_fingerprint": RETRIEVAL_FP,
    "model": {"id": M["id"], "revision": M["revision"], "resolved_revision": INFO.get("resolved_revision"), "dtype": INFO.get("dtype")},
    "evaluation_ids_sha256": CFG["evaluation"]["ids_sha256"], "near_duplicate_template_items": NEAR_DUP_ITEMS,
    "database": "not contacted (scoring runs locally on Windows against Atlas)", "external_model_apis": "none", "environment": ENV})
names = sorted(p.relative_to(OUT_DIR).as_posix() for p in Path(OUT_DIR).rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
(Path(OUT_DIR) / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(Path(OUT_DIR) / n)}  {n}\n" for n in names), encoding="utf-8")
with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(Path(OUT_DIR).rglob("*")):
        if p.is_file():
            z.write(p, f"{Path(OUT_DIR).name}/{p.relative_to(OUT_DIR).as_posix()}")
print("MQL-0008 STATUS:", status); print("\n".join(problems) or "no problems"); print(json.dumps(arms, indent=1))
if status != "complete":
    print("NOT COMPLETE: attach this version's output as input, set RESUME_FROM = 'auto', and run again.")
""", "export")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT)
