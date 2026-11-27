"""Generate notebooks/kaggle/spider_sql_v1_zero_shot_3b.ipynb (cells are tagged so tests/test_kaggle_sql_v1_baseline.py can run the non-GPU cells).

Usage: python scripts/sql_v1/make_kaggle_zs_notebook.py notebooks/kaggle/spider_sql_v1_zero_shot_3b.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src, tags=()):
    c = nbf.v4.new_code_cell(src.strip("\n"))
    c.metadata["tags"] = list(tags)
    cells.append(c)


md(r"""
# Zero-shot SQL baseline: Qwen2.5-Coder-3B-Instruct on Spider dev (spider-sql-v1 `dev_full`, 1,034 examples)

This notebook evaluates the base model with no fine-tuning, on the full Spider dev set. It needs one attached private dataset,
`spider_sql_v1_dev_bundle_v1`, built by `scripts/sql_v1/prepare_kaggle_dev_bundle.py`.

**Settings:** Accelerator **GPU T4 x2** (one card is used), Internet **on** to download the model, Persistence **Files only**.
Run it with **Save Version → Save & Run All (Commit)**.

The run goes in this order:
1. checks the environment, the GPU and the bundle hashes;
2. builds prompts from `{id, db_id, question}` and the schema DDL only, then checks that no prompt contains gold SQL;
3. checks the scoring stack (evaluator import, nltk data, read-only SQLite) so it cannot fail after generation; then loads the model at a pinned revision and runs a 5-example smoke test;
4. generates greedily, appending each batch to `checkpoints/predictions.partial.jsonl` (the run resumes from that file);
5. scores the predictions with the official Spider evaluator, writes the summary and the report, and writes `RUN_STATUS.json`.

Every output goes to `OUT_DIR` under `/kaggle/working/`. Before scoring, gold SQL is used only to assert that it is
absent from every prompt, and is then deleted. It is reloaded in step 5.
""")

code(r"""
# ---- configuration (edit here only) ------------------------------------------------------------
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # whole 3B model on one card
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"   # HF commit, checked 2026-10-27
LOCAL_MODEL_DIR = None        # optional: path of an attached copy of the same model; skips the download

BUNDLE_DIR = None             # None = search /kaggle/input for the bundle's bundle_manifest.json
INPUT_ROOT = "/kaggle/input"
OUT_DIR = "/kaggle/working/spider_sql_v1_zs_qwen25coder3b_v1"
RESUME_FROM = None            # optional: output dir of an earlier version, attached as input, to continue from

BATCH_SIZE = 8
MAX_NEW_TOKENS = 256          # gold SQL is at most 422 characters; 256 tokens is ample
MAX_TIME_PER_BATCH_S = 240    # generation timeout per batch (transformers max_time)
EXEC_TIMEOUT_S = 30.0         # SQLite time limit per query during scoring
SEED = 20261027
SMOKE_N = 5
RETRY_ERRORS = True           # on resume, retry examples whose generation raised an error
""", tags=["config"])

code(r"""
# ---- environment, GPU and dependency versions --------------------------------------------------
import importlib, json, platform, subprocess, sys, sqlite3, time
from importlib import metadata

def _ver(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None

# Kaggle images ship these. Install pinned versions only if one is missing or too old.
from packaging.version import Version
NEED = {"transformers": "4.43.0", "accelerate": "0.30.0", "nltk": "3.8"}
missing = [p for p, v in NEED.items() if _ver(p) is None or Version(_ver(p)) < Version(v)]
if missing:
    print("installing:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "transformers==4.46.3", "accelerate==1.1.1", "nltk==3.9.1"])
    importlib.invalidate_caches()

import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU. Settings -> Accelerator -> GPU T4 x2, then run again.")
GPU_INFO = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    GPU_INFO.append({"index": i, "name": p.name, "vram_gb": round(p.total_memory / 1024**3, 2),
                     "capability": f"{p.major}.{p.minor}"})
major = torch.cuda.get_device_properties(0).major
# T4 (7.5) and P100 (6.0) have no native bf16; torch.cuda.is_bf16_supported() can report emulation, so use capability
DTYPE = torch.bfloat16 if major >= 8 else torch.float16
DTYPE_NAME = str(DTYPE).replace("torch.", "")
torch.ones(2, 2, device="cuda", dtype=DTYPE) @ torch.ones(2, 2, device="cuda", dtype=DTYPE)  # kernel smoke check

VERSIONS = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": _ver("transformers"), "accelerate": _ver("accelerate"), "tokenizers": _ver("tokenizers"),
            "huggingface_hub": _ver("huggingface_hub"), "safetensors": _ver("safetensors"), "nltk": _ver("nltk"),
            "sqlite": sqlite3.sqlite_version}
print(json.dumps({"gpus": GPU_INFO, "dtype": DTYPE_NAME, "versions": VERSIONS}, indent=1))
""", tags=["gpu"])

code(r"""
# ---- locate and verify the bundle; import the bundled code -------------------------------------
import sys, json, time, uuid, shutil
from pathlib import Path
sys.dont_write_bytecode = True        # never write __pycache__ next to the bundle

def _find_bundle():
    if BUNDLE_DIR:
        return Path(BUNDLE_DIR)
    hits = [m.parent for m in Path(INPUT_ROOT).rglob("bundle_manifest.json")
            if json.loads(m.read_text(encoding="utf-8")).get("bundle", "").startswith("spider_sql_v1_dev_bundle")]
    if len(hits) != 1:
        raise SystemExit(f"expected one spider_sql_v1_dev_bundle under {INPUT_ROOT}, found {hits}. Add Input -> your private dataset.")
    return hits[0]

BUNDLE = _find_bundle()
sys.path.insert(0, str(BUNDLE / "code"))
import spider_zs_sql as Z
assert Path(Z.__file__).resolve().parent == (BUNDLE / "code").resolve(), Z.__file__
MANIFEST = Z.verify_bundle(BUNDLE)
MANIFEST_SHA = Z.sha256_file(BUNDLE / "bundle_manifest.json")
for bad in ("test_full", "test_clean", "test_database", "train.jsonl", "val_internal", "dev_clean"):
    assert not any(bad in n for n in MANIFEST["files"]), bad
print(f"bundle {MANIFEST['bundle']} at {BUNDLE}: {MANIFEST['counts']['records']} records, "
      f"{MANIFEST['counts']['databases']} databases, all {len(MANIFEST['files'])} file hashes OK")
""", tags=["bundle"])

code(r"""
# ---- prompts (gold-free view) and run identity -------------------------------------------------
OUT = Path(OUT_DIR)
(OUT / "checkpoints").mkdir(parents=True, exist_ok=True)

_records = Z.load_dev(BUNDLE)                 # full records; gold SQL is used only for the leak check here
ITEMS = Z.prompt_view(_records)               # {id, db_id, question}: the only fields generation sees
DDL = Z.load_ddl(BUNDLE)
PROMPTS = Z.build_prompts(ITEMS, DDL)
Z.check_no_gold_leak(PROMPTS, _records)
assert len(ITEMS) == Z.N_EXPECTED == len(PROMPTS)
del _records                                  # reloaded for scoring after generation
print(f"{len(PROMPTS)} prompts, no gold SQL in any prompt; template sha {Z.template_sha()[:12]}")
print("---- example prompt ----")
for m in PROMPTS[ITEMS[0]["id"]]:
    print(f"[{m['role']}]\n{m['content']}\n")

DECODING = {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS, "repetition_penalty": 1.0,
            "max_time_per_batch_s": MAX_TIME_PER_BATCH_S, "eos": "<|im_end|> / <|endoftext|>",
            "padding_side": "left", "chat_template": "tokenizer default, add_generation_prompt=True"}
RUN_IDENTITY = {"model_id": MODEL_ID, "model_source": LOCAL_MODEL_DIR or f"hf:{MODEL_REVISION}",
                "dtype": DTYPE_NAME, "prompt_version": Z.PROMPT_VERSION, "template_sha256": Z.template_sha(),
                "decoding": DECODING, "bundle": MANIFEST["bundle"], "bundle_manifest_sha256": MANIFEST_SHA}
FINGERPRINT = Z.run_fingerprint(RUN_IDENTITY)

# resume across sessions: copy an earlier version's log in (only if this OUT_DIR has none yet)
LOG_PATH = OUT / "checkpoints/predictions.partial.jsonl"
if RESUME_FROM and not LOG_PATH.exists():
    src = Path(RESUME_FROM)
    shutil.copyfile(src / "checkpoints/predictions.partial.jsonl", LOG_PATH)
    shutil.copyfile(src / "run_config.json", OUT / "run_config.json")
    print("resuming from", src)
cfg_path = OUT / "run_config.json"
if cfg_path.exists():
    old = json.loads(cfg_path.read_text(encoding="utf-8"))
    if old["fingerprint"] != FINGERPRINT:
        raise SystemExit(f"{cfg_path} belongs to another configuration ({old['fingerprint']} != {FINGERPRINT}). "
                         "Use a new OUT_DIR, or restore the old settings.")
    RUN_STARTED = old["run_started_at"]
else:
    RUN_STARTED = Z.now()
    Z.write_json(cfg_path, {"fingerprint": FINGERPRINT, "identity": RUN_IDENTITY, "run_started_at": RUN_STARTED,
                            "system_prompt": Z.SYSTEM, "user_template": Z.USER_TEMPLATE, "seed": SEED})
SESSION_ID = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
LOG = Z.PredictionLog(LOG_PATH, [it["id"] for it in ITEMS], FINGERPRINT)
for w in LOG.warnings:
    print("WARNING:", w)
print(f"fingerprint {FINGERPRINT}; session {SESSION_ID}; {len(LOG.records)} records already in the log, "
      f"{len(LOG.todo(RETRY_ERRORS))} to generate")
""", tags=["prompts"])

code(r"""
# ---- scoring preflight (no gold): evaluator import, nltk punkt, read-only SQLite on the input mount ----
_oe = Z.OfficialEval(BUNDLE, EXEC_TIMEOUT_S)                 # ensure_nltk + bundle-path import check + FK maps
for _d in MANIFEST["db_ids"]:
    _oe.schema(_d)                                            # process_sql.get_schema through the read-only shim
_con = Z.ro_connect(_oe.db_path("wta_1"), EXEC_TIMEOUT_S)
_n = len(_con.execute("SELECT last_name FROM players").fetchall())   # non-UTF-8 bytes -> lenient decoder
_con.close()
_toks = _oe.process_sql.tokenize("SELECT count(*) FROM singer")
print(f"scoring preflight OK: nltk {_oe.nltk_version}, {len(MANIFEST['db_ids'])} schemas, wta_1 players {_n} rows, "
      f"tokenize -> {_toks}")
del _oe, _con
""", tags=["preflight"])

code(r"""
# ---- load the model ----------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
set_seed(SEED)
if LOCAL_MODEL_DIR:
    MODEL_PATH = LOCAL_MODEL_DIR
else:
    from huggingface_hub import snapshot_download
    MODEL_PATH = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                   allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab.json"])
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=DTYPE, device_map={"": 0},
                                             attn_implementation="sdpa").eval()
print(f"loaded {MODEL_ID} from {MODEL_PATH} in {time.time() - t0:.0f}s, dtype {model.dtype}, "
      f"{torch.cuda.memory_allocated(0) / 1024**3:.1f} GB allocated")
print("shipped generation_config (replaced below):", model.generation_config.to_diff_dict())

_eos = model.generation_config.eos_token_id
EOS_IDS = sorted(set(_eos if isinstance(_eos, list) else [_eos]) | {tok.convert_tokens_to_ids("<|im_end|>")})
# Replace the model's generation config outright: the shipped one has sampling defaults and repetition_penalty
# 1.05, and newer transformers fill unset values from it. Greedy, no penalty.
model.generation_config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
                                           max_time=MAX_TIME_PER_BATCH_S, repetition_penalty=1.0,
                                           eos_token_id=EOS_IDS, pad_token_id=tok.pad_token_id)
EFFECTIVE_GEN_CONFIG = json.loads(model.generation_config.to_json_string(use_diff=False))   # JSON-safe
print("effective generation_config:", model.generation_config.to_diff_dict())
""", tags=["gpu", "model"])

code(r"""
# ---- generate function + smoke test ------------------------------------------------------------
import gc
EOS_SET = set(EOS_IDS)

def hf_generate(batch_messages):
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc)
    new = out[:, enc["input_ids"].shape[1]:].tolist()
    plen = enc["attention_mask"].sum(1).tolist()
    res = []
    for ids, pl in zip(new, plen):
        stop = next((k for k, t in enumerate(ids) if t in EOS_SET), None)
        if stop is not None:
            ids, finish = ids[:stop], "eos"
        else:
            finish = "length" if len(ids) >= MAX_NEW_TOKENS else "time"
        res.append({"text": tok.decode(ids, skip_special_tokens=True), "finish": finish,
                    "n_new_tokens": len(ids), "n_prompt_tokens": int(pl)})
    return res

def on_gen_error(e):
    print(f"  generation error ({type(e).__name__}: {str(e)[:200]}); retrying one by one", flush=True)
    gc.collect(); torch.cuda.empty_cache()

smoke_ids = [it["id"] for it in ITEMS[:SMOKE_N]]
t0 = time.time()
run1 = hf_generate([PROMPTS[i] for i in smoke_ids])
run2 = hf_generate([PROMPTS[i] for i in smoke_ids])
smoke = [{"id": i, "text": a["text"], "pred_sql": Z.extract_sql(a["text"])[0], "finish": a["finish"],
          "repeat_identical": a["text"] == b["text"]} for i, a, b in zip(smoke_ids, run1, run2)]
Z.write_jsonl(OUT / "smoke/smoke_outputs.jsonl", smoke)
for s in smoke:
    print(s["id"], s["finish"], "|", s["pred_sql"])
print(f"smoke: {time.time() - t0:.1f}s for 2 x {SMOKE_N}; deterministic: {all(s['repeat_identical'] for s in smoke)}")
_n_select = sum(s["pred_sql"].lower().startswith(("select", "with")) for s in smoke)
if _n_select < SMOKE_N - 1:
    raise SystemExit(f"smoke test: only {_n_select}/{SMOKE_N} outputs are SELECT statements - check dtype / model "
                     "files before the full run")
""", tags=["gpu", "smoke"])

code(r"""
# ---- full generation (resumable) ---------------------------------------------------------------
GEN_STARTED = Z.now()
_t_start, _n_start = time.time(), len(LOG.records)

def on_batch(done, total, recs):
    new = done - _n_start
    rate = (time.time() - _t_start) / max(new, 1)
    bad = [r["id"] + ":" + r["status"] for r in recs if r["status"] != "ok"]
    print(f"{done}/{total}  {rate:.2f}s/example  eta {rate * (total - done) / 60:.1f} min"
          + (f"  non-ok: {bad}" if bad else ""), flush=True)
    Z.write_json(OUT / "checkpoints/progress.json", {"done": done, "total": total, "session_id": SESSION_ID,
                                                     "updated_at": Z.now(), "fingerprint": FINGERPRINT})

Z.run_generation(ITEMS, PROMPTS, hf_generate, LOG, BATCH_SIZE, SESSION_ID, retry_errors=RETRY_ERRORS,
                 on_batch=on_batch, on_error=on_gen_error)
GEN_FINISHED = Z.now()
print("generation finished; status counts:", dict(Z.Counter(r["status"] for r in LOG.records.values())))
""", tags=["generate"])

code(r"""
# ---- predictions.jsonl (all 1,034, dev order) --------------------------------------------------
missing = LOG.missing()
if missing:
    Z.write_json(OUT / "RUN_STATUS.json", {"status": "incomplete", "missing_ids": missing, "updated_at": Z.now()})
    raise SystemExit(f"INCOMPLETE: {len(missing)} examples have no prediction. Run again to resume.")
PREDICTIONS = Z.finalize_predictions(LOG, OUT / "predictions.jsonl")
print(f"predictions.jsonl: {len(PREDICTIONS)} records")
""", tags=["finalize"])

code(r"""
# ---- free the GPU, then score with the official Spider evaluator -------------------------------
if "model" in globals():
    del model
    gc.collect(); torch.cuda.empty_cache()
RECORDS = Z.load_dev(BUNDLE)   # gold SQL is read from here on
t0 = time.time()
RESULTS, CLI, EVAL_VERSIONS = Z.score_all(BUNDLE, RECORDS, PREDICTIONS, OUT, EXEC_TIMEOUT_S,
                                          progress=lambda k, n: print(f"scored {k}/{n}", flush=True))
EVAL_SECONDS = round(time.time() - t0, 1)
print(f"scored {len(RESULTS)} in {EVAL_SECONDS}s; official CLI (match) -> {CLI}")
""", tags=["evaluate"])

code(r"""
# ---- summary, report, run status ---------------------------------------------------------------
CONFIG = {**RUN_IDENTITY, "fingerprint": FINGERPRINT, "model_revision": MODEL_REVISION, "model_path": str(MODEL_PATH),
          "effective_generation_config": {k: v for k, v in EFFECTIVE_GEN_CONFIG.items() if not k.startswith("_")},
          "batch_size": BATCH_SIZE, "exec_timeout_s": EXEC_TIMEOUT_S, "seed": SEED, "gpus": GPU_INFO,
          "versions": {**VERSIONS, **EVAL_VERSIONS}, "system_prompt": Z.SYSTEM, "user_template": Z.USER_TEMPLATE,
          "official_eval": MANIFEST["provenance"]["official_eval"],
          "timestamps": {"run_started": RUN_STARTED, "last_generation_started": GEN_STARTED,
                         "last_generation_finished": GEN_FINISHED, "evaluation_finished": Z.now(),
                         "evaluation_seconds": EVAL_SECONDS},
          "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE")}
SUMMARY = Z.summarize(PREDICTIONS, RESULTS, CLI, CONFIG)
Z.write_json(OUT / "baseline_summary.json", SUMMARY)
(OUT / "baseline_report.md").write_text(Z.render_report(SUMMARY), encoding="utf-8")

problems = []
if not SUMMARY["complete"]:
    problems.append("summary not complete")
cross = SUMMARY["evaluation"]["official_cli_crosscheck"]
if not cross or not cross["agree_to_3dp"]:
    problems.append(f"official CLI exact match does not agree with the per-example score: {cross}")
names = ["predictions.jsonl", "evaluation_results.jsonl", "baseline_summary.json", "baseline_report.md", "run_config.json"]
Z.write_json(OUT / "RUN_STATUS.json", {"status": "complete" if not problems else "needs_review", "problems": problems,
                                       "n_expected": Z.N_EXPECTED, "n_predictions": len(PREDICTIONS),
                                       "n_evaluated": len(RESULTS), "fingerprint": FINGERPRINT,
                                       "sha256": Z.output_sums(OUT, names), "updated_at": Z.now()})
print(Z.render_report(SUMMARY))
print("RUN_STATUS:", "complete" if not problems else problems)
""", tags=["summary"])

md(r"""
## Outputs (`OUT_DIR`)

| File | Content |
|---|---|
| `predictions.jsonl` | one record per dev example in dev order: id, db_id, `pred_sql`, `raw_output`, status (`ok` / `empty` / `truncated` / `timeout` / `error`), tokens, latency, batch, session, fingerprint |
| `evaluation_results.jsonl` | per example: official EX and EM, official parse, SQLite prepare/execute status and errors, unofficial result-set match, evaluator errors |
| `baseline_summary.json` | aggregate metrics with numerators and denominators, generation statistics, full configuration |
| `baseline_report.md` | readable summary and limitations |
| `run_config.json`, `RUN_STATUS.json` | run identity, completion status, output hashes |
| `checkpoints/` | append-only generation log (`predictions.partial.jsonl`) and progress |
| `official_eval/` | the gold and prediction files given to the official `evaluate(..., "match")`, and its printout |
| `smoke/` | smoke-test outputs (not part of the evaluation) |

A run is finished only when `RUN_STATUS.json` says `"status": "complete"`.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nb.metadata["kaggle"] = {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}
nbf.validate(nb)
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print("wrote", OUT, len(cells), "cells")
