"""Generate notebooks/kaggle/nl2mql_spider_dev_1034_model_comparison.ipynb (experiment MQL-0006: Qwen-only, GENERATE-ONLY).

Scope (2026-10-30): MQL-0006 evaluates ONLY Qwen/Qwen2.5-Coder-3B-Instruct. The Groq-hosted gpt-oss-120b comparison is
deferred to a later experiment. The notebook has no frontier-model configuration, no API key and no API client.

Architecture: Kaggle only generates (Qwen on the GPU). The notebook never contacts MongoDB Atlas and has no Atlas
secret, URI or client. All SQLite-vs-Atlas scoring runs on Windows with scripts/nl2mql/local_atlas_scoring.py.

Cells are tagged (config, env, bundle, preflight, qwen, package), so tests/test_nl2mql_dev1034.py can run every cell
locally with a fake generator. The model code sits behind an injectable factory; the Kaggle default is the real one.

Usage: python scripts/nl2mql/make_dev1034_notebook.py [notebooks/kaggle/nl2mql_spider_dev_1034_model_comparison.ipynb]
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks/kaggle/nl2mql_spider_dev_1034_model_comparison.ipynb"
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src, tags):
    c = nbf.v4.new_code_cell(src.strip("\n"))
    c.metadata["tags"] = list(tags)
    cells.append(c)


md(r"""
# MQL-0006 (generation only): zero-shot Qwen2.5-Coder-3B-Instruct, NL→MQL on Spider dev (892 questions)

This notebook generates zero-shot NL→MQL predictions with **Qwen2.5-Coder-3B-Instruct only**, on the evaluation set
of the bundle. That set is Spider dev minus every record on the databases of the held-out MQL test partition:
**892 questions on 17 databases**. The ids and their order are fixed by the bundle manifest. The prompt is the
frozen `nl-mql-spider1-v1` (question + schema context + instructions + JSON MQL output format). Qwen never sees
gold SQL or gold results.

- This notebook does **not** contact MongoDB Atlas and calls **no** external model API.
- Scoring happens on Windows with `scripts/nl2mql/local_atlas_scoring.py` (run guide:
  `docs/kaggle_nl2mql_spider_dev_1034.md`).
- The frontier-model comparison is deferred to a later experiment.

Setup:
- **Input:** the private dataset `nl2mql_spider_dev_1034_bundle_v2`.
- **Secrets:** none.
- **Settings:** GPU T4 x2 (one card is used), Internet on (for the Hugging Face model download).
- **Run:** Save Version → Save & Run All (Commit).

Order:
1. environment;
2. bundle integrity;
3. preflight (prompts, model configuration, GPU);
4. Qwen generation, with checkpoints and resume;
5. predictions bundle `nl2mql_spider_dev_predictions.zip`.

If the preflight fails, nothing else runs and no model is loaded. A run with missing or failed generations is
labelled `partial`.

- **CUDA out of memory:** a batch of 8 that runs out of GPU memory is retried one prompt at a time, with the same
  prompt, `max_new_tokens`, dtype and greedy decoding. Each prediction records its configured and effective batch
  size and whether this fallback was used.
- **Resume:** set `RESUME_FROM = "auto"` with the earlier version's output attached. That output is hash-verified
  first; only its unfinished ids are generated, and its completed predictions are carried over byte for byte.
""")

code(r"""
# ---- configuration (edit here only) ------------------------------------------------------------
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EXPERIMENT_ID = "MQL-0006"
INPUT_ROOT = "/kaggle/input"
BUNDLE_DIR = None               # None = search INPUT_ROOT for the bundle's bundle_manifest.json
OUT_DIR = "/kaggle/working/nl2mql_spider_dev_predictions_v2"
OUTPUT_ZIP = "/kaggle/working/nl2mql_spider_dev_predictions.zip"
RESUME_FROM = None              # None | "auto" (find the earlier version's output attached under INPUT_ROOT) | its path
RESUME_EXPECTED_OK = None       # optional: completed predictions the earlier output must hold (e.g. 716), else refuse

QWEN_LABEL = "qwen2.5-coder-3b-instruct"
QWEN_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
QWEN_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"   # HF commit, as SQL-0001 (checked 2026-10-27)
QWEN_LOCAL_DIR = None           # optional attached copy of the same revision
QWEN_DTYPE = "auto"             # auto = bfloat16 if torch.cuda.is_bf16_supported(), else float16. On Kaggle's T4 that
                                # returns True (emulated bf16), so MQL-0006 ran in bfloat16 (run_config.json "dtype")
QWEN_BATCH_SIZE = 8             # a batch that hits CUDA OOM is retried one prompt at a time (recorded per prediction)
QWEN_MAX_NEW_TOKENS = 1024      # as MQL-0001..0005

RETRY_GENERATION_ERRORS = True
BUNDLE_RESTORE_ROOT = "/tmp/nl2mql_bundle_restored"   # working copy if Kaggle decompressed nested .gz files
# test hooks (keep the defaults on Kaggle)
QWEN_GENERATOR_FACTORY = None
_GZIP_LEVEL = 9                 # gzip.compress default, as used by the bundle builder
""", ["config"])

code(r"""
# ---- environment, dependency versions, no credentials attached ---------------------------------
import json, platform, re, sys, time
from importlib import metadata
from pathlib import Path

def _ver(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None

attached_env = [str(p) for p in Path(INPUT_ROOT).rglob(".env*")] if Path(INPUT_ROOT).exists() else []
if attached_env:
    raise RuntimeError(f"refusing to run: credential files attached as input: {attached_env}")

ENV = {"python": platform.python_version(), "platform": platform.platform(),
       "packages": {p: _ver(p) for p in ("torch", "transformers", "accelerate", "tokenizers")}}
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
print(json.dumps(ENV, indent=1))
""", ["env"])

code(r"""
# ---- bundle integrity --------------------------------------------------------------------------
import importlib.util

def _find_manifest():
    if BUNDLE_DIR:
        return Path(BUNDLE_DIR)
    hits = [m.parent for m in Path(INPUT_ROOT).rglob("bundle_manifest.json")
            if json.loads(m.read_text(encoding="utf-8")).get("bundle", "").startswith("nl2mql_spider_dev_1034_bundle")]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one nl2mql_spider_dev_1034_bundle under {INPUT_ROOT}, found {hits}")
    return hits[0]

BUNDLE = _find_manifest()
_spec = importlib.util.spec_from_file_location("dev1034_core", BUNDLE / "code/dev1034_core.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
sys.dont_write_bytecode = True

# Kaggle's dataset upload transparently decompresses nested .gz files (data/reference.jsonl.gz arrives as
# data/reference.jsonl). The read-only input is never changed: the bundle is copied to BUNDLE_RESTORE_ROOT and the
# file is re-compressed exactly as the builder did (gzip, mtime=0). The unchanged strict C.verify_bundle then checks
# every hash. If this zlib produces different gzip bytes, the restored file's CONTENT is verified instead against the
# per-item digests in data/reference_expected.json (itself hash-checked); every other file stays strictly hash-checked.
import gzip, shutil

def _restore_kaggle_decompression(bundle):
    man = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    todo = [n for n in man["files"] if n.endswith(".gz") and not (bundle / n).exists() and (bundle / n[:-3]).exists()]
    if not todo:
        return bundle, []
    work = Path(BUNDLE_RESTORE_ROOT) / bundle.name
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(bundle, work)
    for n in todo:
        raw = (work / n[:-3]).read_bytes()
        (work / n).write_bytes(gzip.compress(raw, compresslevel=_GZIP_LEVEL, mtime=0))
        (work / n[:-3]).unlink()
    return work, todo

def _verify_bundle_content_restored(bundle, restored):
    man = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    on_disk = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()} - {"bundle_manifest.json"}
    C.need(on_disk == set(man["files"]), "bundle tree != manifest after restoring Kaggle-decompressed files")
    for name, m in man["files"].items():
        if name not in restored:
            C.need(C.sha256_file(bundle / name) == m["sha256"], f"hash mismatch: {name}")
    C.need(restored == ["data/reference.jsonl.gz"], f"unexpected restored files {restored}")
    exp = json.loads((bundle / "data/reference_expected.json").read_text(encoding="utf-8"))   # hash-checked above
    refs = C.load_reference(bundle)
    C.need([C.reference_digest(r) for r in refs] == exp["items"] and
           all(C.result_hashes(r["gold_rows"]) == (r["ordered_sha256"], r["multiset_sha256"]) for r in refs),
           "restored data/reference.jsonl.gz content differs from data/reference_expected.json")
    recs = C.load_dev(bundle)                                                                  # hash-checked above
    ev = man["evaluation_set"]
    C.need(len(recs) == ev["records"] and C.sha256_text("\n".join(r["id"] for r in recs)) == ev["ids_sha256"],
           "evaluation ids or their order changed")
    C.need(not [r for r in recs if r["db_id"] in set(ev["excluded_db_ids"])], "record on a held-out database")
    C.need(sorted({r["db_id"] for r in recs}) == man["db_ids"] and
           all((bundle / "database" / d / f"{d}.sqlite").is_file() for d in man["db_ids"]), "database set changed")
    return man

BUNDLE, RESTORED = _restore_kaggle_decompression(BUNDLE)
BUNDLE_CHECK = "strict sha256 of every manifest file"
try:
    MANIFEST = C.verify_bundle(BUNDLE)      # every hash, ids + order (ids_sha256), no held-out database
except C.PreflightError as e:
    if not RESTORED or str(e) != "hash mismatch: data/reference.jsonl.gz":
        raise
    MANIFEST = _verify_bundle_content_restored(BUNDLE, RESTORED)
    BUNDLE_CHECK = ("strict sha256 of every manifest file except the Kaggle-decompressed data/reference.jsonl.gz, "
                    "whose content was verified against the 892 per-item digests in data/reference_expected.json")
if RESTORED:
    print(f"restored Kaggle-decompressed {RESTORED} in a working copy ({BUNDLE}); check: {BUNDLE_CHECK}")
MANIFEST_SHA256 = C.sha256_file(BUNDLE / "bundle_manifest.json")
RECORDS = C.load_dev(BUNDLE)               # gold SQL is used ONLY by the leak check below, never in a prompt
EXPECTED_IDS = [r["id"] for r in RECORDS]
PROMPTS = C.read_jsonl(BUNDLE / "data/prompts.jsonl")
K, P = C.bundle_modules(BUNDLE)
print(f"bundle OK: {BUNDLE} ({len(RECORDS)} evaluation records, {len(MANIFEST['db_ids'])} DBs, "
      f"{MANIFEST['evaluation_set']['excluded_records']} held-out-database records excluded, manifest {MANIFEST_SHA256[:12]})")

def model_dir(label):
    return Path(OUT_DIR) / "models" / label

def write_status(status, problems, extra=None):
    st = {"experiment": EXPERIMENT_ID, "stage": "predictions", "scope": "single model (Qwen only)",
          "status": status, "written_at": C.now(), "problems": problems, "bundle": MANIFEST["bundle"],
          "bundle_manifest_sha256": MANIFEST_SHA256, "evaluation_ids_sha256": MANIFEST["evaluation_set"]["ids_sha256"],
          "n_expected": len(EXPECTED_IDS), "bundle_check": BUNDLE_CHECK, "bundle_restored_files": RESTORED}
    st.update(extra or {})
    C.write_json(Path(OUT_DIR) / "PREDICTIONS_STATUS.json", st)
    return st

# ---- resume: the earlier version's output is verified before anything is copied from it ---------------
# run_config fields written after the generation fingerprint was taken (as verify_kaggle_spider_dev_1034.POST_FP)
_POST_FP = {"experiment", "fingerprint", "generation_wall_s", "resolved_revision", "stopped", "generated_this_session",
            "environment"}
RESUME = None

def _find_resume_dir():
    if RESUME_FROM != "auto":
        return Path(RESUME_FROM)
    hits = sorted(p.parent for p in Path(INPUT_ROOT).rglob("PREDICTIONS_STATUS.json")
                  if p.parent.name == Path(OUT_DIR).name and p.parent.resolve() != Path(OUT_DIR).resolve())
    C.need(hits, f"RESUME_FROM='auto': no earlier {Path(OUT_DIR).name}/ attached under {INPUT_ROOT}")
    sums = {(h / "SHA256SUMS.txt").read_bytes() if (h / "SHA256SUMS.txt").is_file() else None for h in hits}
    C.need(len(sums) == 1, f"RESUME_FROM='auto': different earlier outputs attached, set RESUME_FROM to one of {hits}")
    return hits[0]   # several hits = the same output twice (its folder and its extracted zip)

def _verify_resume_source(prev):
    for name in ("PREDICTIONS_STATUS.json", "SHA256SUMS.txt"):
        C.need((prev / name).is_file(), f"resume source {prev}: {name} missing")
    sums = {}
    for ln in (prev / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, _, n = ln.partition("  ")
        sums[n] = h
    files = {p.relative_to(prev).as_posix() for p in prev.rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
    C.need(files == set(sums), f"resume source: files differ from SHA256SUMS.txt {sorted(files ^ set(sums))[:5]}")
    bad = [n for n, h in sums.items() if C.sha256_file(prev / n) != h]
    C.need(not bad, f"resume source: hash mismatch {bad[:5]}")
    st = json.loads((prev / "PREDICTIONS_STATUS.json").read_text(encoding="utf-8"))
    C.need(st.get("experiment") == EXPERIMENT_ID and st.get("stage") == "predictions", "resume source: another experiment")
    C.need(st.get("bundle_manifest_sha256") == MANIFEST_SHA256 and
           st.get("evaluation_ids_sha256") == MANIFEST["evaluation_set"]["ids_sha256"], "resume source: another bundle")
    md = prev / "models" / QWEN_LABEL
    cfg = json.loads((md / "run_config.json").read_text(encoding="utf-8"))
    fp = cfg.get("fingerprint")
    C.need(C.fingerprint({k: v for k, v in cfg.items() if k not in _POST_FP}) == fp,
           "resume source: run_config fingerprint does not recompute")
    want = {"model_label": QWEN_LABEL, "model_id": QWEN_MODEL_ID, "model_revision": QWEN_REVISION,
            "decoding": "greedy (do_sample=False)", "max_new_tokens": QWEN_MAX_NEW_TOKENS, "batch_size": QWEN_BATCH_SIZE,
            "prompts_sha256": MANIFEST["prompt"]["prompts_sha256"], "bundle_manifest_sha256": MANIFEST_SHA256}
    diff = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    C.need(not diff, f"resume source: configuration differs (earlier, now): {diff}")
    ms = (st.get("models") or {}).get(QWEN_LABEL) or {}
    C.need(ms.get("fingerprint") == fp and ms.get("predictions_sha256") == C.sha256_file(md / "predictions.jsonl") and
           ms.get("run_config_sha256") == C.sha256_file(md / "run_config.json"), "resume source: status != its files")
    lines = [ln for ln in (md / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    preds = [json.loads(ln) for ln in lines]
    ids = [p["id"] for p in preds]
    C.need(len(set(ids)) == len(ids) and ids == [i for i in EXPECTED_IDS if i in set(ids)],
           "resume source: ids duplicated, unknown or out of order")
    C.need({p.get("fingerprint") for p in preds} <= {fp}, "resume source: records from another run")
    # the checkpoint history: an id is never completed twice, and its final records are exactly predictions.jsonl
    hist = C.read_jsonl(md / "checkpoints/predictions.partial.jsonl")
    C.need({r.get("fingerprint") for r in hist} <= {fp}, "resume source: checkpoint records from another run")
    last = {}
    for r in hist:
        C.need(r.get("id") in set(EXPECTED_IDS), f"resume source: unexpected id {r.get('id')}")
        C.need(last.get(r["id"], {}).get("status") != "ok", f"resume source: {r['id']} completed twice (conflict)")
        last[r["id"]] = r
    C.need([last[i] for i in EXPECTED_IDS if i in last] == preds, "resume source: checkpoint != predictions.jsonl")
    ok_lines = {p["id"]: ln for p, ln in zip(preds, lines) if p["status"] == "ok"}
    C.need(ms.get("n_ok") == len(ok_lines), "resume source: n_ok in the status differs from predictions.jsonl")
    C.need(RESUME_EXPECTED_OK is None or len(ok_lines) == RESUME_EXPECTED_OK,
           f"resume source holds {len(ok_lines)} completed predictions, RESUME_EXPECTED_OK = {RESUME_EXPECTED_OK}")
    prev_gen = ms.get("generation") or {}
    return {"from": str(prev), "fingerprint": fp, "config": cfg, "ok_lines": ok_lines,
            "todo": [i for i in EXPECTED_IDS if i not in ok_lines],
            "earlier_wall_s": prev_gen.get("generation_wall_s_total", cfg.get("generation_wall_s")),
            "earlier_oom_error_records": sum(r.get("status") != "ok" and "OutOfMemoryError" in (r.get("error") or "")
                                             for r in hist)}

if RESUME_FROM:
    RESUME = _verify_resume_source(_find_resume_dir())
    rels = [f"models/{QWEN_LABEL}/checkpoints/predictions.partial.jsonl", f"models/{QWEN_LABEL}/run_config.json"]
    RESUME["seeded"] = C.seed_from_previous(RESUME["from"], OUT_DIR, rels)
    print(f"resume source verified: {RESUME['from']} (fingerprint {RESUME['fingerprint']}, "
          f"{len(RESUME['ok_lines'])} completed predictions kept, {len(RESUME['todo'])} to generate); "
          f"copied: {RESUME['seeded']}")
""", ["bundle"])

code(r"""
# ---- preflight: nothing below runs unless every check passes (no Atlas, no external API) -------
PREFLIGHT_OK = False
PRE = {"started_at": C.now(), "checks": {}}
try:
    # 1. prompts re-derived from (db_id, question) with the bundled code == frozen prompts; order; no gold leak
    C.compare_prompts(PROMPTS, C.build_prompts(RECORDS, P))
    C.check_no_gold_leak(PROMPTS, RECORDS)
    C.need([p["id"] for p in PROMPTS] == EXPECTED_IDS, "prompt order differs from the evaluation order")
    C.need(len({p["template_sha256"] for p in PROMPTS}) == 1 and PROMPTS[0]["template_sha256"] ==
           MANIFEST["prompt"]["template_sha256"], "prompt template differs from the manifest")
    PRE["checks"]["prompts"] = f"{len(PROMPTS)} prompts re-derived identically, no gold leak"
    # 2. model configuration
    C.need(QWEN_MODEL_ID == "Qwen/Qwen2.5-Coder-3B-Instruct", "MQL-0006 evaluates Qwen2.5-Coder-3B-Instruct only")
    C.need(re.fullmatch(r"[0-9a-f]{40}", QWEN_REVISION or "") is not None, "QWEN_REVISION must be a 40-hex HF commit")
    C.need(QWEN_MAX_NEW_TOKENS > 0 and QWEN_BATCH_SIZE > 0, "invalid decoding limits")
    PRE["checks"]["model_config"] = {"qwen": f"{QWEN_MODEL_ID}@{QWEN_REVISION[:8]}", "decoding": "greedy",
                                     "max_new_tokens": QWEN_MAX_NEW_TOKENS, "batch_size": QWEN_BATCH_SIZE}
    # 3. GPU available (before any download)
    if QWEN_GENERATOR_FACTORY is None:
        import torch
        C.need(torch.cuda.is_available(), "no CUDA GPU is visible (Settings -> Accelerator -> GPU T4 x2)")
        PRE["checks"]["gpu"] = torch.cuda.get_device_name(0)
    PREFLIGHT_OK = True
except Exception as e:
    PRE["error"] = f"{type(e).__name__}: {e}"
    C.write_json(Path(OUT_DIR) / "preflight/kaggle_preflight.json", {**PRE, "ok": False})
    write_status("preflight_failed", [PRE["error"]])
    raise
PRE["finished_at"] = C.now()
C.write_json(Path(OUT_DIR) / "preflight/kaggle_preflight.json", {**PRE, "ok": True})
print("PREFLIGHT OK:", json.dumps({k: (v if isinstance(v, str) else "ok") for k, v in PRE["checks"].items()}))
""", ["preflight"])

code(r"""
# ---- Qwen2.5-Coder-3B-Instruct: GPU generation (checkpointed, resumable) -----------------------
assert PREFLIGHT_OK, "preflight did not pass"
MODEL_ERROR, RUN_CONFIG = None, None
OOM_EVENTS = {"batch": 0, "single_prompt": 0}   # this session only

def _gen_meta(n, recovered):
    return {"configured_batch_size": QWEN_BATCH_SIZE, "effective_batch_size": n, "oom_recovery": recovered,
            "oom_retries": 1 if recovered else 0}

def oom_safe_generator(run_batch, is_oom, cleanup):
    # A batch that raises CUDA OOM (is_oom; any other exception propagates unchanged) is retried one prompt at a time
    # with the same prompt, max_new_tokens, dtype and greedy decoding. A prompt that also OOMs alone is recorded as a
    # generation error (no further retry in this session). cleanup() runs after the except block is left, so the
    # exception's frames, which hold the failed batch's tensors, are already released.
    def generate_batch(batch):
        oom = None
        try:
            outs = run_batch(batch)
        except Exception as e:
            if not is_oom(e):
                raise
            oom = f"{type(e).__name__}: {e}"
        if oom is None:
            return [(t, {**(m or {}), **_gen_meta(len(batch), False)}, err) for t, m, err in outs]
        OOM_EVENTS["batch"] += 1
        cleanup()
        print(f"  CUDA OOM on a batch of {len(batch)} (from {batch[0]['id']}): retrying each prompt at batch size 1",
              flush=True)
        res = []
        for p in batch:
            single = None
            try:
                (t, m, err), = run_batch([p])
            except Exception as e:
                if not is_oom(e):
                    raise
                single = f"{type(e).__name__}: {e}"
            if single is None:
                res.append((t, {**(m or {}), **_gen_meta(1, True)}, err))
            else:
                OOM_EVENTS["single_prompt"] += 1
                cleanup()
                res.append((None, {"latency_s": None, **_gen_meta(1, True)}, f"{single} [at batch size 1, after a batch OOM]"))
        return res
    return generate_batch

def make_qwen_generator():
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(42)
    src = QWEN_LOCAL_DIR or QWEN_MODEL_ID
    rev = None if QWEN_LOCAL_DIR else QWEN_REVISION
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if QWEN_DTYPE == "auto" \
        else getattr(torch, QWEN_DTYPE)
    tok = AutoTokenizer.from_pretrained(src, revision=rev)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(src, revision=rev, torch_dtype=dtype).to("cuda").eval()
    info = {"dtype": str(dtype).replace("torch.", ""), "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "transformers": _ver("transformers"),
            "resolved_revision": getattr(model.config, "_commit_hash", None)}
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id, tok.pad_token_id}
    stop_ids.discard(None)

    def run_batch(batch):
        texts = [tok.apply_chat_template([{"role": "system", "content": p["system"]}] + p["messages"],
                                         tokenize=False, add_generation_prompt=True) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False, max_new_tokens=QWEN_MAX_NEW_TOKENS,
                                     pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        except torch.cuda.OutOfMemoryError:
            raise                         # -> oom_safe_generator: batch-size-1 fallback
        except Exception as e:            # any other error: recorded for the batch, retried on resume (as before)
            return [(None, {"latency_s": None}, f"{type(e).__name__}: {e}") for _ in batch]
        dt = time.time() - t0
        res = []
        for i in range(len(batch)):
            gen = out[i, enc["input_ids"].shape[1]:].tolist()
            hit = next((j for j, t in enumerate(gen) if t in stop_ids), None)
            n_out = hit + 1 if hit is not None else len(gen)
            res.append((tok.decode(gen[:n_out], skip_special_tokens=True),
                        {"latency_s": round(dt / len(batch), 4), "batch_latency_s": round(dt, 3),
                         "input_tokens": int(enc["attention_mask"][i].sum()), "output_tokens": n_out,
                         "finish_reason": "stop" if hit is not None else "length"}, None))
        del out, enc
        return res

    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()
    return run_batch, info, (lambda e: isinstance(e, torch.cuda.OutOfMemoryError)), cleanup

try:
    t0 = time.time()
    run_batch, qinfo, is_oom, cleanup = (QWEN_GENERATOR_FACTORY or make_qwen_generator)()
    gen_fn = oom_safe_generator(run_batch, is_oom, cleanup)
    qcfg = {"experiment": EXPERIMENT_ID, "model_label": QWEN_LABEL, "model_id": QWEN_MODEL_ID,
            "model_revision": QWEN_REVISION, "runtime": "transformers GPU", "decoding": "greedy (do_sample=False)",
            "max_new_tokens": QWEN_MAX_NEW_TOKENS, "batch_size": QWEN_BATCH_SIZE, "seed": 42,
            "prompt_version": PROMPTS[0]["prompt_version"], "template_sha256": PROMPTS[0]["template_sha256"],
            "prompts_sha256": MANIFEST["prompt"]["prompts_sha256"], "bundle_manifest_sha256": MANIFEST_SHA256,
            **{k: qinfo.get(k) for k in ("dtype", "gpu", "torch", "transformers")}}
    qcfg["fingerprint"] = C.fingerprint({k: v for k, v in qcfg.items() if k != "experiment"})
    if RESUME:   # same model, revision, prompt, decoding, dtype, GPU and library versions as the resumed run, or stop
        diff = {k: (RESUME["config"].get(k), v) for k, v in qcfg.items()
                if k not in _POST_FP and RESUME["config"].get(k) != v}
        C.need(qcfg["fingerprint"] == RESUME["fingerprint"] and not diff,
               f"this session's configuration differs from the resumed run (earlier, now): {diff}; refusing to mix")
    ck = C.Checkpoint(model_dir(QWEN_LABEL) / "checkpoints/predictions.partial.jsonl", EXPECTED_IDS,
                      qcfg["fingerprint"], C.is_generation_retry if RETRY_GENERATION_ERRORS else (lambda r: False))
    if RESUME and RESUME["seeded"]:
        C.need(ck.todo() == RESUME["todo"], "the resumed checkpoint does not match the verified resume source")
    print(f"Qwen: {len(ck.todo())} of {len(EXPECTED_IDS)} to generate ({qinfo})")
    try:
        r = C.run_generation(PROMPTS, gen_fn, ck, QWEN_LABEL, QWEN_BATCH_SIZE,
                             progress=lambda n, t: print(f"  qwen {n}/{t}", flush=True) if n % 80 < QWEN_BATCH_SIZE else None)
    finally:   # whatever happened, everything checkpointed so far is kept and packaged
        qcfg.update(generation_wall_s=round(time.time() - t0, 1), resolved_revision=qinfo.get("resolved_revision"),
                    stopped=None, generated_this_session=None, environment=ENV)
        C.write_jsonl(model_dir(QWEN_LABEL) / "predictions.jsonl", ck.ordered())
        C.write_json(model_dir(QWEN_LABEL) / "run_config.json", qcfg)
        RUN_CONFIG = qcfg
    qcfg["generated_this_session"] = r["generated"]
    C.write_json(model_dir(QWEN_LABEL) / "run_config.json", qcfg)
    del gen_fn
except Exception as e:
    MODEL_ERROR = f"{type(e).__name__}: {e}"
    print("Qwen run failed; checkpointed predictions are kept for RESUME_FROM:", MODEL_ERROR)
""", ["qwen"])

code(r"""
# ---- predictions bundle: status, checksums, zip (scored later on Windows) -----------------------
assert PREFLIGHT_OK, "preflight did not pass"
import zipfile
problems, models = [], {}
pf, cf = model_dir(QWEN_LABEL) / "predictions.jsonl", model_dir(QWEN_LABEL) / "run_config.json"
if MODEL_ERROR:
    problems.append(f"{QWEN_LABEL}: {MODEL_ERROR}")
if pf.exists() and cf.exists():
    preds, cfg = C.read_jsonl(pf), json.loads(cf.read_text(encoding="utf-8"))
    ids = [p["id"] for p in preds]
    n_ok = sum(p["status"] == "ok" for p in preds)
    mp = []
    if ids != [i for i in EXPECTED_IDS if i in set(ids)] or len(set(ids)) != len(ids):
        mp.append("ids out of order or duplicated")
    if len(ids) != len(EXPECTED_IDS):
        mp.append(f"{len(ids)}/{len(EXPECTED_IDS)} ids generated")
    if n_ok != len(ids):
        mp.append(f"{len(ids) - n_ok} generation errors (retried on resume)")
    if cfg.get("stopped"):
        mp.append(f"generation stopped: {cfg['stopped']}")
    if MODEL_ERROR:
        mp.append("model run raised an error")
    wall = cfg.get("generation_wall_s") if RUN_CONFIG else None
    gen = {"configured_batch_size": QWEN_BATCH_SIZE,
           "completed_normal_batch": sum(p["status"] == "ok" and not p.get("oom_recovery") for p in preds),
           "completed_batch_size_1_fallback": sum(p["status"] == "ok" and bool(p.get("oom_recovery")) for p in preds),
           "remaining_failures": len(EXPECTED_IDS) - n_ok,
           "oom_events_this_session": dict(OOM_EVENTS),
           "oom_error_records_earlier_sessions": RESUME["earlier_oom_error_records"] if RESUME else 0,
           "generation_wall_s_this_session": wall,
           "generation_wall_s_total": round((RESUME["earlier_wall_s"] or 0) + (wall or 0), 1) if RESUME else wall,
           "resumed_from": RESUME["from"] if RESUME else None}
    if RESUME:   # every prediction completed before the resume must be carried over byte for byte
        now_lines = {json.loads(ln)["id"]: ln for ln in pf.read_text(encoding="utf-8").splitlines() if ln.strip()}
        changed = sorted(i for i, ln in RESUME["ok_lines"].items() if now_lines.get(i) != ln)
        gen["earlier_completed_kept_byte_identical"] = len(RESUME["ok_lines"]) - len(changed)
        if changed:
            mp.append(f"{len(changed)} predictions completed before the resume changed, e.g. {changed[:3]}")
    if gen["completed_batch_size_1_fallback"]:
        gen["note"] = (f"{gen['completed_batch_size_1_fallback']} predictions were generated at batch size 1 after a "
                       f"CUDA OOM on their batch of {QWEN_BATCH_SIZE}; batch size and padding differences can cause "
                       "minor numerical differences even with greedy decoding")
    lat = [p.get("latency_s") for p in preds if p.get("latency_s") is not None]
    models[QWEN_LABEL] = {"status": "complete" if not mp else "partial", "problems": mp, "n": len(ids), "n_ok": n_ok,
                          "model_id": cfg.get("model_id"), "model_revision": cfg.get("model_revision"),
                          "fingerprint": cfg.get("fingerprint"), "predictions_sha256": C.sha256_file(pf),
                          "run_config_sha256": C.sha256_file(cf),
                          "tokens": {"input": sum(p.get("input_tokens") or 0 for p in preds),
                                     "output": sum(p.get("output_tokens") or 0 for p in preds)},
                          "mean_latency_s": round(sum(lat) / len(lat), 4) if lat else None, "generation": gen}
    problems += [f"{QWEN_LABEL}: {x}" for x in mp]
else:
    problems.append(f"{QWEN_LABEL}: no predictions were written")
status = "complete" if models and not problems else "partial"
RS = write_status(status, problems, {"models": models, "atlas": "not contacted (scoring runs locally on Windows)",
                                     "external_model_apis": "none"})
names = sorted(p.relative_to(OUT_DIR).as_posix() for p in Path(OUT_DIR).rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
(Path(OUT_DIR) / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(Path(OUT_DIR) / n)}  {n}\n" for n in names),
                                              encoding="utf-8")
with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(Path(OUT_DIR).rglob("*")):
        if p.is_file():
            z.write(p, f"{Path(OUT_DIR).name}/{p.relative_to(OUT_DIR).as_posix()}")
print("PREDICTIONS STATUS:", status); print("\n".join(problems) or "no problems")
print(json.dumps(models, indent=1))
if status != "complete":
    print("NOT COMPLETE: attach this version's output as input, set RESUME_FROM, and run again.")
""", ["package"])

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT)
