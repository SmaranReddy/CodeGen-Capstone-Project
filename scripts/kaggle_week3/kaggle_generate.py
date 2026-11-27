"""Generation-only runner for the clean Week 3 control (runs on Kaggle GPU; also on CPU for smoke tests).

Reads the frozen prompts of a package built by scripts/kaggle_week3/build_package.py and saves raw
model outputs. It does NOT parse, validate, execute or score anything, needs no database and no
credentials; scoring happens on Windows (scripts/kaggle_week3/score_kaggle_outputs.py).

Generation is the project's LlamaCppProvider.generate (providers.py, copied unchanged into the
package) with two run-level settings that apply identically to every arm:
  * n_gpu_layers (default -1 = all layers on the GPU);
  * the llama.cpp KV cache is reset before every item, so an output depends only on its prompt,
    not on the previous item, the arm order or where a resumed session restarted.

Arms run sequentially in the package order (zs_rerun, random_k3, hybrid_k3) on ONE model load.
Each completed item is appended to <out>/<arm>/generations.jsonl and fsynced at once; a resumed
run skips finished ids, drops a truncated trailing line, and refuses records from another
package, prompt or backend. Provider errors go to <arm>/provider_errors.jsonl and are retried.

Usage:
  python kaggle_generate.py --package PKG --out OUT --model GGUF [--arms ...] [--limit N]
                            [--n-gpu-layers -1] [--require-gpu] [--determinism-check N]
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

PKG_FILES = "package_manifest.json"


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_checkpoint(path, log):
    """-> records; a truncated (unparseable) LAST line is dropped and the file rewritten."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    recs = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                sys.exit(f"{path}: unparseable line {i + 1} that is not the last; refusing to continue")
            log(f"dropping truncated trailing line of {path}")
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    return recs


def append_record(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def verify_package(pkg):
    man_path = pkg / PKG_FILES
    man = json.loads(man_path.read_text(encoding="utf-8"))
    bad = [n for n, h in man["files"].items() if sha256_file(pkg / n) != h]
    if bad:
        sys.exit(f"package files differ from package_manifest.json: {bad}")
    return man, sha256_file(man_path)


def gpu_info():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        head = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        cuda = re.search(r"CUDA Version:\s*([\d.]+)", head)
        return {"gpus": [x.strip() for x in q.stdout.strip().splitlines()],
                "cuda_driver_api": cuda.group(1) if cuda else None}
    except Exception as e:  # noqa: BLE001
        return {"gpus": [], "error": f"{type(e).__name__}: {e}"}


def gpu_mem_used_mib():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return [int(x) for x in q.stdout.split()]
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="path to the GGUF (sha256 must match the manifest)")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--determinism-check", type=int, default=0,
                    help="after generation, regenerate the first N finished items of each arm in "
                         "reverse order and require byte-identical raw_text")
    a = ap.parse_args()
    pkg, out = Path(a.package).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "generate.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    man, man_sha = verify_package(pkg)
    log(f"package {man['package']} verified ({len(man['files'])} files), manifest sha256 {man_sha}")
    model_sha = sha256_file(a.model)
    if model_sha != man["model"]["gguf_sha256"]:
        sys.exit(f"model sha256 {model_sha} != manifest {man['model']['gguf_sha256']}; refusing to run")
    log(f"model verified: {Path(a.model).name} sha256 {model_sha}")

    import llama_cpp
    sys.path.insert(0, str(pkg))
    from providers import LlamaCppProvider

    want = man["inference"]["llama_cpp_python_version"]
    if llama_cpp.__version__ != want:
        sys.exit(f"llama-cpp-python {llama_cpp.__version__} != {want} (prompts were budgeted with {want})")
    offload = bool(llama_cpp.llama_supports_gpu_offload())
    if a.require_gpu and not (offload and a.n_gpu_layers != 0):
        sys.exit(f"--require-gpu: llama_supports_gpu_offload={offload}, n_gpu_layers={a.n_gpu_layers}; "
                 "this llama-cpp-python build cannot use the GPU")

    class KaggleLlamaCpp(LlamaCppProvider):
        """LlamaCppProvider with GPU offload and a KV reset per item; generate() is the parent's."""

        def __init__(self, *args, n_gpu_layers=-1, **kw):
            super().__init__(*args, **kw)
            self.n_gpu_layers = n_gpu_layers

        def _load(self, seed):
            if self._llm is None:
                from llama_cpp import Llama
                self._llm = Llama(model_path=self.gguf_path, n_ctx=self.n_ctx, seed=seed,
                                  n_threads=self.n_threads, n_gpu_layers=self.n_gpu_layers, verbose=True)
                self._llm.verbose = False  # keep the load log (layer offload), drop per-call timings

        def generate(self, messages, system, generation):
            if self._llm is not None:
                self._llm.reset()
            return super().generate(messages, system, generation)

    gen = dict(man["generation"])
    provider = KaggleLlamaCpp(man["model"]["id"], str(Path(a.model).resolve()), man["model"]["n_ctx"],
                              n_gpu_layers=a.n_gpu_layers)
    ok, msg = provider.available()
    if not ok:
        sys.exit(f"provider unavailable: {msg}")
    mem_before = gpu_mem_used_mib()
    t0 = time.time()
    sys.stderr.flush()
    provider._load(gen["seed"])
    load_s = round(time.time() - t0, 2)
    mem_after = gpu_mem_used_mib()
    llm = provider._llm
    backend = {
        "llama_cpp_python_version": llama_cpp.__version__, "gpu_offload_supported": offload,
        "n_gpu_layers": a.n_gpu_layers, "n_ctx": llm.n_ctx(), "n_batch": llm.n_batch,
        **{k: (int(v) if isinstance(v, (bool, int)) else str(v)) for k in
           ("n_ubatch", "flash_attn", "flash_attn_type", "type_k", "type_v", "offload_kqv")
           if (v := getattr(llm.context_params, k, None)) is not None},
        "chat_format": llm.chat_format, "kv_reset_per_item": True, "seed": gen["seed"],
        "system_info": llama_cpp.llama_print_system_info().decode("utf-8", "replace").strip(),
        "gpu": gpu_info(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version.split()[0], "platform": platform.platform(),
    }
    backend_sha = canon_sha(backend)
    session = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "model_load_s": load_s,
               "gpu_mem_used_mib_before_load": mem_before, "gpu_mem_used_mib_after_load": mem_after,
               "argv": sys.argv[1:]}
    log(f"backend: offload={offload} n_gpu_layers={a.n_gpu_layers} gpu={backend['gpu'].get('gpus')} "
        f"gpu_mem {mem_before} -> {mem_after} MiB; load {load_s}s")
    if a.require_gpu and (not mem_before or not mem_after or max(mem_after) - max(mem_before) < 1500):
        sys.exit("--require-gpu: GPU memory did not grow by >= 1.5 GB at model load; weights are not on the GPU")

    run_path = out / "run_manifest.json"
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["package_manifest_sha256"] != man_sha or run["model_sha256"] != model_sha:
            sys.exit(f"{run_path} belongs to another package or model; use a new --out")
        if run["backend_sha256"] != backend_sha:
            sys.exit(f"backend differs from the first session of {out} (would mix backends):\n"
                     f"then {json.dumps(run['backend'])}\nnow  {json.dumps(backend)}")
        run["sessions"].append(session)
    else:
        run = {"package": man["package"], "package_manifest_sha256": man_sha, "model_sha256": model_sha,
               "model_id": man["model"]["id"], "generation": gen, "backend": backend,
               "backend_sha256": backend_sha, "arm_order": man["arm_order"], "sessions": [session]}
    run_path.write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")

    system = man["system_prompt"]
    arms = a.arms or man["arm_order"]
    prompts_by_arm = {}
    for arm in arms:
        prompts = read_jsonl(pkg / man["arms"][arm]["prompt_file"])
        if [p["id"] for p in prompts] != man["item_ids"]:
            sys.exit(f"{arm}: prompt ids differ from the manifest item list")
        for p in prompts:
            if canon_sha({"system": system, "messages": p["messages"]}) != p["prompt_sha256"]:
                sys.exit(f"{arm}/{p['id']}: prompt hash mismatch")
        prompts_by_arm[arm] = prompts
        todo = [p for p in prompts if (not a.ids or p["id"] in a.ids)]
        if a.limit:
            todo = todo[:a.limit]
        arm_dir = out / arm
        arm_dir.mkdir(exist_ok=True)
        gpath = arm_dir / "generations.jsonl"
        by_id = {p["id"]: p for p in prompts}
        done = {}
        for r in load_checkpoint(gpath, log):
            if r["id"] in done:
                sys.exit(f"{gpath}: duplicate id {r['id']}")
            if (r["package_manifest_sha256"] != man_sha or r["backend_sha256"] != backend_sha
                    or r["id"] not in by_id or r["prompt_sha256"] != by_id[r["id"]]["prompt_sha256"]):
                sys.exit(f"{gpath}: record {r['id']} was produced by another package/backend/prompt")
            done[r["id"]] = r
        log(f"{arm}: {len(done)} saved, {sum(p['id'] not in done for p in todo)} to generate")
        consecutive_errors = 0
        for i, p in enumerate(todo):
            if p["id"] in done:
                continue
            t1 = time.perf_counter()
            res = provider.generate(p["messages"], system, dict(gen))
            wall = round(time.perf_counter() - t1, 3)
            rec = {"id": p["id"], "db_id": p["db_id"], "hardness": p["hardness"], "arm": arm,
                   "experiment": arm, "experiment_sha256": man["arms"][arm]["experiment_sha256"],
                   "prompt_sha256": p["prompt_sha256"], "selected_examples": p["retrieval"]["selected_examples"],
                   "package_manifest_sha256": man_sha, "backend_sha256": backend_sha, "model_sha256": model_sha,
                   "model_id": res.get("model_id"), "raw_text": res.get("text"), "provider_error": res.get("error"),
                   "input_tokens": res.get("input_tokens"), "output_tokens": res.get("output_tokens"),
                   "latency_s": round(res.get("latency_s") or 0.0, 3), "generation_wall_s": wall,
                   "session": len(run["sessions"]), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if res.get("error"):
                append_record(arm_dir / "provider_errors.jsonl", rec)
                consecutive_errors += 1
                log(f"[{arm} {i + 1}/{len(todo)}] {p['id']} PROVIDER ERROR {res['error'][:200]}")
                if consecutive_errors >= 5:
                    sys.exit("5 consecutive provider errors; stopping (saved records are kept)")
                continue
            consecutive_errors = 0
            append_record(gpath, rec)
            done[p["id"]] = rec
            log(f"[{arm} {i + 1}/{len(todo)}] {p['id']} out_tokens={rec['output_tokens']} gen={wall}s")
        n_done = sum(i in done for i in man["item_ids"])
        log(f"{arm}: {n_done}/{len(man['item_ids'])} items have a saved generation")

    if a.determinism_check:
        report = {"backend_sha256": backend_sha, "arms": {}}
        for arm in arms:
            saved = {r["id"]: r for r in load_checkpoint(out / arm / "generations.jsonl", log)}
            ids = [i for i in man["item_ids"] if i in saved][:a.determinism_check]
            by_id = {p["id"]: p for p in prompts_by_arm[arm]}
            same, diff = 0, []
            for i in reversed(ids):
                res = provider.generate(by_id[i]["messages"], system, dict(gen))
                if res.get("text") == saved[i]["raw_text"]:
                    same += 1
                else:
                    diff.append(i)
            report["arms"][arm] = {"checked": len(ids), "identical": same, "different": diff}
            log(f"determinism {arm}: {same}/{len(ids)} identical when regenerated in reverse order {diff}")
        (out / "determinism_check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        if any(v["different"] for v in report["arms"].values()):
            sys.exit("determinism check FAILED (see determinism_check.json)")


if __name__ == "__main__":
    main()
