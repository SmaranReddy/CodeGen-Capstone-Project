"""Import, verify and score the Kaggle outputs of the clean Week 3 control (Windows; read-only Atlas).

The downloaded ZIP is untrusted data: it is extracted into a new empty staging directory (no
absolute or parent paths) and only JSON is read from it. Nothing from it is executed.

Verification (all must pass before any scoring):
  * every file listed in the run's SHA256SUMS.txt matches;
  * run_manifest: same package manifest hash as the local package, same model sha256, one backend for
    the whole run (and a GPU backend with all layers requested unless --allow-cpu);
  * per arm: exactly the 318 package ids, no duplicates, no provider errors left, and each record's
    prompt / experiment / package / backend hash equals the frozen package;
  * the frozen prompts are rebuilt now from the canonical pipeline and must be identical (random
    selection reproducible, zero-shot prompt without examples, hybrid excludes the target database);
  * the Kaggle determinism check reported no differences.

Scoring: each raw output goes through run_eval.score_output (parse -> read-only validation ->
Atlas aggregate -> exec_match), exactly as in run_rag_eval.run_one, under the read-only account.
Records are appended per item (resumable). A second pass re-executes every saved verdict and
writes rescore_check.json. Results go to outputs/rag_week3_clean/<package>/, never to
outputs/rag_week3/.

Usage:
  python scripts/kaggle_week3/score_kaggle_outputs.py --zip PATH [--package kaggle/week3_control_v1]
  python scripts/kaggle_week3/score_kaggle_outputs.py --verify-only   # re-check staging + prompts, no Atlas
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rag"))

import common as C  # noqa: E402
from common import K  # noqa: E402
import build_package as BP  # noqa: E402
import run_rag_eval as RR  # noqa: E402
import run_eval as R  # noqa: E402

OUT_ROOT = C.ROOT / "outputs/rag_week3_clean"
RUN_DIR = "week3_control_outputs"


def safe_extract(zpath, dest):
    if dest.exists():
        sys.exit(f"{dest} exists; staging directories are never overwritten (use --staging-name)")
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or ":" in info.filename or "\\" in info.filename:
                sys.exit(f"unsafe path in zip: {info.filename!r}")
        z.extractall(dest)


def verify_sums(run_dir):
    sums = (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    bad = []
    for line in sums:
        h, name = line.split("  ", 1)
        if C.sha256_file(run_dir / name) != h:
            bad.append(name)
    listed = {line.split("  ", 1)[1] for line in sums}
    present = {p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
    return {"files_listed": len(listed), "mismatched": bad, "unlisted_files": sorted(present - listed)}


def verify(run_dir, pkg, allow_cpu=False):
    """-> (report, {arm: [generation records in item order]}); exits on any failure."""
    man = json.loads((pkg / "package_manifest.json").read_text(encoding="utf-8"))
    man_sha = C.sha256_file(pkg / "package_manifest.json")
    rep = {"package": man["package"], "package_manifest_sha256": man_sha, "checks": {}}
    fail = []
    sums = verify_sums(run_dir)
    rep["checks"]["sha256sums"] = sums
    if sums["mismatched"] or sums["unlisted_files"]:
        fail.append("SHA256SUMS")
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    be = run["backend"]
    rep["backend"] = be
    rep["sessions"] = run["sessions"]
    if run["package_manifest_sha256"] != man_sha:
        fail.append("run used a different package")
    if run["model_sha256"] != man["model"]["gguf_sha256"]:
        fail.append("model sha256")
    if run["generation"] != man["generation"]:
        fail.append("generation settings")
    if not allow_cpu and not (be["gpu_offload_supported"] and be["n_gpu_layers"] == -1 and be["gpu"].get("gpus")):
        fail.append("backend is not a full-GPU-offload build")
    det = run_dir / "determinism_check.json"
    rep["checks"]["determinism"] = json.loads(det.read_text(encoding="utf-8")) if det.exists() else None
    if not det.exists() or any(v["different"] for v in rep["checks"]["determinism"]["arms"].values()):
        fail.append("determinism check missing or failed")

    rebuilt = rebuild_prompts(man)
    gens = {}
    for arm in man["arm_order"]:
        frozen = {r["id"]: r for r in K.read_jsonl(pkg / man["arms"][arm]["prompt_file"])}
        recs = K.read_jsonl(run_dir / arm / "generations.jsonl")
        ids = [r["id"] for r in recs]
        errs = run_dir / arm / "provider_errors.jsonl"
        n_err = len(K.read_jsonl(errs)) if errs.exists() else 0
        problems = []
        if len(ids) != len(set(ids)):
            problems.append("duplicate ids")
        if set(ids) != set(man["item_ids"]):
            problems.append(f"ids differ from package: {len(set(man['item_ids']) - set(ids))} missing, "
                            f"{len(set(ids) - set(man['item_ids']))} unexpected")
        for r in recs:
            f = frozen.get(r["id"])
            if (f is None or r["prompt_sha256"] != f["prompt_sha256"] or r["arm"] != arm
                    or r["experiment_sha256"] != man["arms"][arm]["experiment_sha256"]
                    or r["package_manifest_sha256"] != man_sha or r["backend_sha256"] != run["backend_sha256"]
                    or r["model_sha256"] != man["model"]["gguf_sha256"] or r.get("provider_error")
                    or r["selected_examples"] != f["retrieval"]["selected_examples"]):
                problems.append(f"record {r['id']} does not match the frozen package")
        same = sum(rebuilt[arm][i] == f["prompt_sha256"] for i, f in frozen.items())
        if same != len(frozen):
            problems.append(f"frozen prompts differ from a rebuild with the canonical code ({same}/{len(frozen)})")
        rep["checks"][arm] = {"records": len(recs), "unique_ids": len(set(ids)), "expected": man["n_items"],
                              "provider_error_records_retried": n_err, "prompts_rebuilt_identical": same,
                              "problems": problems[:10]}
        if problems:
            fail.append(arm)
        order = {i: k for k, i in enumerate(man["item_ids"])}
        gens[arm] = sorted(recs, key=lambda r: order.get(r["id"], 1e9))
    rep["passed"] = not fail
    rep["failures"] = fail
    return rep, gens


def rebuild_prompts(man):
    cfg, base, _ = RR.load_configs()
    rows = BP.train_rows(base, cfg)
    assert [r["id"] for r in rows] == man["item_ids"]
    corpus = C.load_corpus(C.ROOT / cfg["corpus_dir"])
    collections = C.load_collections()
    counter = RR.make_counter(cfg)
    return {arm: {r["id"]: r["prompt_sha256"] for r in BP.build_arm(arm, rows, cfg, base, corpus, collections, counter)}
            for arm in man["arm_order"]}


def score(gens, pkg, dest):
    man = json.loads((pkg / "package_manifest.json").read_text(encoding="utf-8"))
    cfg, base, base_sha = RR.load_configs()
    exec_cfg = base["execution"]
    data = {r["id"]: r for r in K.read_jsonl(C.ROOT / base["dataset"])}
    meta_common = {"package": man["package"], "partition": "train", "atlas_db": base["atlas_db"],
                   "base_config_sha256": base_sha, **man["provenance"]}
    client = K.mongo_client()
    try:
        mdb = client[base["atlas_db"]]
        colls = mdb.list_collection_names()
        for arm, recs in gens.items():
            frozen = {r["id"]: r for r in K.read_jsonl(pkg / man["arms"][arm]["prompt_file"])}
            out = dest / arm / "executions.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            done = {r["id"]: r for r in K.read_jsonl(out)} if out.exists() else {}
            for r in done.values():
                if r["raw_text_sha256"] != C.sha256_bytes((r["raw_text"] or "").encode("utf-8")):
                    sys.exit(f"{out}: {r['id']} raw_text changed; refusing to mix")
            n_new = 0
            for g in recs:
                if g["id"] in done:
                    continue
                ex = data[g["id"]]
                allowed = {c for c in colls if c.startswith(ex["db_id"] + "__")}
                t0 = time.perf_counter()
                res, extra = R.score_output(mdb, g["raw_text"], ex, exec_cfg, allowed)
                ret = {k: v for k, v in frozen[g["id"]]["retrieval"].items()}
                rec = {**meta_common, "experiment": arm, "config": man["arms"][arm]["config"],
                       **{k: g[k] for k in ("id", "db_id", "hardness", "experiment_sha256", "prompt_sha256",
                                            "package_manifest_sha256", "backend_sha256", "model_sha256", "model_id",
                                            "raw_text", "input_tokens", "output_tokens", "latency_s",
                                            "generation_wall_s", "session", "timestamp")},
                       "question": ex["question"], "retrieval": ret,
                       "raw_text_sha256": C.sha256_bytes((g["raw_text"] or "").encode("utf-8")),
                       "expected_columns": ex.get("gold_columns"), "expected_rows": ex["gold_rows"],
                       "result": res, **extra, "scoring_latency_s": round(time.perf_counter() - t0, 3),
                       "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                done[g["id"]] = rec
                n_new += 1
            print(f"{arm}: scored {n_new} new, {len(done)} total", flush=True)
        check = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": {}}
        for arm in gens:
            recs = K.read_jsonl(dest / arm / "executions.jsonl")
            agree, diff = 0, []
            for rec in recs:
                ex = data[rec["id"]]
                allowed = {c for c in colls if c.startswith(ex["db_id"] + "__")}
                res, _ = R.score_output(mdb, rec["raw_text"], ex, exec_cfg, allowed)
                if (res["correct"], res["category"]) == (rec["result"]["correct"], rec["result"]["category"]):
                    agree += 1
                else:
                    diff.append({"id": rec["id"], "saved": rec["result"]["category"], "now": res["category"]})
            check["runs"][arm] = {"records": len(recs), "verdicts_reproduced": agree, "changed": diff}
            print(f"re-execution {arm}: {agree}/{len(recs)} verdicts reproduced; changed {diff}", flush=True)
        (dest / "rescore_check.json").write_text(json.dumps(check, indent=1) + "\n", encoding="utf-8")
    finally:
        client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip")
    ap.add_argument("--package", default="kaggle/week3_control_v1")
    ap.add_argument("--staging-name", default="import")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--allow-cpu", action="store_true", help="accept a CPU backend (local self-test only)")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    pkg = (C.ROOT / a.package).resolve()
    man = json.loads((pkg / "package_manifest.json").read_text(encoding="utf-8"))
    dest = Path(a.out_root) / man["package"]
    staging = dest / a.staging_name
    if a.zip:
        safe_extract(Path(a.zip), staging)
    run_dir = staging / RUN_DIR
    rep, gens = verify(run_dir, pkg, a.allow_cpu)
    rep["staging"] = str(staging.relative_to(C.ROOT)) if staging.is_relative_to(C.ROOT) else str(staging)
    rep["verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (dest / "verification.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("passed", "failures")} | {"checks": {
        k: v for k, v in rep["checks"].items() if k not in ("determinism",)}}, indent=1))
    if not rep["passed"]:
        sys.exit("verification FAILED; nothing scored (see verification.json)")
    if not a.verify_only:
        score(gens, pkg, dest)


if __name__ == "__main__":
    main()
