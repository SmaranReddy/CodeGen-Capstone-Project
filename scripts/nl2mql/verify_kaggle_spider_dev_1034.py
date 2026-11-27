"""Verify MQL-0006 artifacts offline: the Kaggle predictions bundle and the local Atlas scoring outputs.

No inference, no API call, no Atlas connection. Used by scripts/nl2mql/local_atlas_scoring.py, which refuses to
score a predictions bundle that fails, and runnable on its own.

Predictions bundle (from the Kaggle notebook, nl2mql_spider_dev_predictions.zip):
  1. safe extraction (no absolute paths, no '..', no links) into a fresh staging dir
  2. SHA256SUMS.txt covers every file exactly; every hash matches
  3. PREDICTIONS_STATUS.json names the local bundle (manifest sha256) and its evaluation ids (ids sha256)
  4. per model: run_config identity (model id, revision / provider, decoding, prompt + bundle hashes), generation
     fingerprint recomputed from the config, one fingerprint per record
  5. ids are the bundle's evaluation ids, in order, without duplicates (complete = all of them, all generated), the
     db_id of every record matches the bundle, prompt sha256 per id equals the frozen prompt
  6. the per-model status in PREDICTIONS_STATUS.json is recomputed and equal
  7. no held-out database name, no credential-like string, no test/train/split file

Scoring outputs (written by local_atlas_scoring.py score):
  8. SHA256SUMS.txt over the scoring dir; RUN_STATUS.json names the bundle and the verified predictions
     (PREDICTIONS_STATUS.json sha256 and every predictions.jsonl sha256)
  9. reference: the bundled gold rows match their digests; every gold SQL re-executed here (no real difference)
 10. per model: evaluation ids in order; one evaluation fingerprint; each record bound to its prediction
     (prediction sha256); the safety gate re-run on every raw output reproduces the validation stage and MQL; every
     pipeline starts from a collection of its own db_id; every executed verdict re-scored from stored rows
 11. baseline_summary.json recomputed and equal (single-model experiment: no comparison file may exist)
 12. no held-out database name, no credential-like string in any output

Usage:
  python scripts/nl2mql/verify_kaggle_spider_dev_1034.py predictions --zip <nl2mql_spider_dev_predictions.zip>
  python scripts/nl2mql/verify_kaggle_spider_dev_1034.py predictions --dir <extracted predictions dir>
  python scripts/nl2mql/verify_kaggle_spider_dev_1034.py scoring --run-dir outputs/nl2mql_spider_dev_1034_model_comparison/v1
"""
import argparse
import json
import re
import sys
import time
import uuid
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import dev1034_core as C  # noqa: E402

DEFAULT_BUNDLE = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2"
DEST_ROOT = ROOT / "outputs/nl2mql_spider_dev_1034_model_comparison"
# MQL-0006 is Qwen-only (2026-10-30): the Groq-hosted gpt-oss-120b comparison is deferred to a later experiment.
EXPECTED_MODELS = {
    "qwen2.5-coder-3b-instruct": {"model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
                                  "model_revision": "488639f1ff808d1d3d0ba301aef8c11461451ec5", "max_new_tokens": 1024,
                                  "decoding": "greedy (do_sample=False)", "prompt_version": "nl-mql-spider1-v1",
                                  "template_sha256": "7401bf140dd1b8a9dac057edd9f729a8624823604fa2bd6ae9cfaf1c1890c1be"},
}
EXPECTED_N = 892            # the leakage-safe evaluation set (17 DBs, 70 collections)
# run_config fields written after the generation fingerprint was taken (plus `experiment`, excluded from it)
POST_FP = {"experiment", "fingerprint", "generation_wall_s", "resolved_revision", "stopped", "generated_this_session",
           "environment", "pricing"}
DENIED_OUTPUT = re.compile(r"(^|/)(test[_.]|train|val_internal|splits|\.env)|test_full|test_clean", re.I)
SECRET = [r"mongodb(\+srv)?://[^\s\"'<]*:[^\s\"'<]*@", r"gsk_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9]{32,}",
          r"hf_[A-Za-z0-9]{30,}", r"Bearer [A-Za-z0-9_\-]{20,}"]


class VerifyError(RuntimeError):
    pass


class Report:
    def __init__(self):
        self.checks, self.problems = [], []

    def ok(self, name, cond, detail=""):
        self.checks.append({"check": name, "ok": bool(cond), "detail": "" if cond else str(detail)})
        if not cond:
            self.problems.append(f"{name}: {detail}")
        return bool(cond)


def safe_extract(zip_path, dest, marker):
    """Extract into a NEW dir; return the single directory holding `marker`."""
    dest = Path(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None:
            raise VerifyError("zip CRC check failed")
        for info in z.infolist():
            n = info.filename
            if n.startswith(("/", "\\")) or ".." in Path(n).parts or re.match(r"^[A-Za-z]:", n):
                raise VerifyError(f"unsafe path in zip: {n}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise VerifyError(f"symlink in zip: {n}")
        z.extractall(dest)
    hits = [p.parent for p in dest.rglob(marker)]
    if len(hits) != 1:
        raise VerifyError(f"expected one {marker} in the zip, found {len(hits)}")
    return hits[0]


def tree_sums(d, skip=()):
    return {p.relative_to(d).as_posix(): C.sha256_file(p) for p in Path(d).rglob("*")
            if p.is_file() and p.relative_to(d).as_posix() not in skip}


def _checksums(R, d, prefix):
    sums = {}
    f = Path(d) / "SHA256SUMS.txt"
    if not R.ok(f"{prefix}:checksum_file_present", f.is_file(), "SHA256SUMS.txt missing"):
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        h, _, n = line.partition("  ")
        sums[n] = h
    files = {p.relative_to(d).as_posix() for p in Path(d).rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
    R.ok(f"{prefix}:checksums_cover_all_files", files == set(sums),
         f"unlisted={sorted(files - set(sums))[:5]} missing={sorted(set(sums) - files)[:5]}")
    bad = [n for n in sums if n in files and C.sha256_file(Path(d) / n) != sums[n]]
    R.ok(f"{prefix}:checksums_match", not bad, f"{bad[:5]}")


def _scan_outputs(R, d, man, prefix):
    outs = [p.relative_to(d).as_posix() for p in Path(d).rglob("*") if p.is_file()]
    R.ok(f"{prefix}:no_test_or_train_files", not [n for n in outs if DENIED_OUTPUT.search(n)],
         [n for n in outs if DENIED_OUTPUT.search(n)][:5])
    held = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, man["evaluation_set"]["excluded_db_ids"]))
                      + r")(?![A-Za-z0-9])")
    leaks, held_hits = [], []
    for n in outs:
        p = Path(d) / n
        if n.endswith(".jsonl.gz"):
            text = "\n".join(json.dumps(r) for r in C.read_jsonl(p))
        elif p.suffix in (".json", ".jsonl", ".md", ".txt"):
            text = p.read_text(encoding="utf-8", errors="replace")
        else:
            continue
        if any(re.search(pat, text) for pat in SECRET):
            leaks.append(n)
        if held.search(n) or held.search(text):
            held_hits.append(n)
    R.ok(f"{prefix}:no_credentials", not leaks, leaks[:5])
    R.ok(f"{prefix}:no_heldout_database_content", not held_hits, held_hits[:5])


def model_status(preds, cfg, expected_ids):
    """Per-model predictions status, exactly as the notebook's package cell computes it."""
    ids = [p["id"] for p in preds]
    n_ok = sum(p["status"] == "ok" for p in preds)
    mp = []
    if ids != [i for i in expected_ids if i in set(ids)] or len(set(ids)) != len(ids):
        mp.append("ids out of order or duplicated")
    if len(ids) != len(expected_ids):
        mp.append(f"{len(ids)}/{len(expected_ids)} ids generated")
    if n_ok != len(ids):
        mp.append(f"{len(ids) - n_ok} generation errors (retried on resume)")
    if cfg.get("stopped"):
        mp.append(f"generation stopped: {cfg['stopped']}")
    return ("complete" if not mp else "partial"), mp


def verify_predictions(pred_dir, bundle=DEFAULT_BUNDLE):
    pred_dir, bundle = Path(pred_dir), Path(bundle)
    R = Report()
    man = C.verify_bundle(bundle)
    man_sha = C.sha256_file(bundle / "bundle_manifest.json")
    recs = C.load_dev(bundle)
    ids = [r["id"] for r in recs]
    db_of = {r["id"]: r["db_id"] for r in recs}
    prompts = {p["id"]: p for p in C.read_jsonl(bundle / "data/prompts.jsonl")}
    _checksums(R, pred_dir, "predictions")
    st = json.loads((pred_dir / "PREDICTIONS_STATUS.json").read_text(encoding="utf-8"))
    R.ok("predictions:experiment_and_stage", st.get("experiment") == C.EXPERIMENT_ID and st.get("stage") == "predictions",
         f"{st.get('experiment')}/{st.get('stage')}")
    R.ok("predictions:bundle_identity", st.get("bundle_manifest_sha256") == man_sha, "bundle manifest hash differs")
    R.ok("predictions:evaluation_ids", st.get("evaluation_ids_sha256") == man["evaluation_set"]["ids_sha256"]
         and st.get("n_expected") == len(ids) == EXPECTED_N, f"evaluation ids differ from the bundle ({len(ids)})")
    present = sorted(p.name for p in (pred_dir / "models").iterdir()) if (pred_dir / "models").is_dir() else []
    R.ok("predictions:only_expected_models", present == sorted(EXPECTED_MODELS) and
         sorted(st.get("models") or {}) == sorted(EXPECTED_MODELS), f"model dirs {present}")
    models = {}
    for label, want in EXPECTED_MODELS.items():
        d = pred_dir / "models" / label
        if not R.ok(f"{label}:predictions_present", (d / "predictions.jsonl").is_file() and (d / "run_config.json").is_file(),
                    "missing predictions.jsonl or run_config.json"):
            continue
        try:
            cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
            preds = C.read_jsonl(d / "predictions.jsonl")
            bad = [i for i, p in enumerate(preds) if not isinstance(p, dict) or not {"id", "status"} <= set(p)]
        except (ValueError, OSError) as e:
            R.ok(f"{label}:readable", False, f"{type(e).__name__}: {e}")
            continue
        if not R.ok(f"{label}:well_formed_records", not bad, f"{len(bad)} malformed records, e.g. line {bad[:3]}"):
            continue
        R.ok(f"{label}:model_identity", all(cfg.get(k) == v for k, v in want.items()) and cfg.get("model_label") == label,
             {k: cfg.get(k) for k in want})
        R.ok(f"{label}:prompt_and_bundle_hashes", cfg.get("prompts_sha256") == man["prompt"]["prompts_sha256"]
             and cfg.get("template_sha256") == man["prompt"]["template_sha256"]
             and cfg.get("bundle_manifest_sha256") == man_sha, "prompt or bundle hash differs")
        R.ok(f"{label}:fingerprint", C.fingerprint({k: v for k, v in cfg.items() if k not in POST_FP})
             == cfg.get("fingerprint"), "recomputed fingerprint differs")
        R.ok(f"{label}:record_fingerprints", {p.get("fingerprint") for p in preds} <= {cfg.get("fingerprint")},
             "records from another run")
        pi = [p["id"] for p in preds]
        R.ok(f"{label}:ids_ordered_subset", pi == [i for i in ids if i in set(pi)] and len(set(pi)) == len(pi),
             f"{len(pi)} ids, {len(set(pi))} unique")
        R.ok(f"{label}:db_ids", all(p.get("db_id") == db_of.get(p["id"]) for p in preds), "db_id differs from bundle")
        R.ok(f"{label}:prompt_sha_per_item", all(p.get("prompt_sha256") == prompts[p["id"]]["prompt_sha256"]
                                                 for p in preds if p["id"] in prompts), "prompt hash differs")
        R.ok(f"{label}:model_label_per_item", {p.get("model") for p in preds} <= {label}, "foreign model label")
        status, mp = model_status(preds, cfg, ids)
        recorded = (st.get("models") or {}).get(label, {})
        R.ok(f"{label}:status_recomputed", recorded.get("status") == status and
             recorded.get("predictions_sha256") == C.sha256_file(d / "predictions.jsonl"), f"{recorded.get('status')} vs {status}")
        models[label] = {"status": status, "problems": mp, "n": len(pi),
                         "n_ok": sum(p["status"] == "ok" for p in preds)}
    _scan_outputs(R, pred_dir, man, "predictions")
    all_ok = not R.problems
    complete = all_ok and st.get("status") == "complete" and len(models) == len(EXPECTED_MODELS) and \
        all(m["status"] == "complete" for m in models.values())
    return {"verified_at": C.now(), "kind": "predictions", "dir": str(pred_dir), "status": st.get("status"),
            "all_checks_passed": all_ok, "complete": complete, "problems": R.problems, "checks": R.checks,
            "models": models}


def merged_config(pred_dir, score_dir, label):
    run_cfg = json.loads((Path(pred_dir) / "models" / label / "run_config.json").read_text(encoding="utf-8"))
    sc = json.loads((Path(score_dir) / "models" / label / "scoring_config.json").read_text(encoding="utf-8"))
    return {**run_cfg, **sc}


def summarize_model(label, cfg, preds, evals, refs, ids):
    s = C.model_summary(label, cfg, preds, evals, refs, ids, cfg.get("pricing"))
    if cfg.get("stopped"):
        s["status"] = "partial"
        s["problems"].append(f"generation stopped: {cfg['stopped']}")
    return s


def verify_scoring(run_dir, bundle=DEFAULT_BUNDLE):
    """run_dir holds predictions/ (verified Kaggle bundle) and scoring/ (local outputs)."""
    run_dir, bundle = Path(run_dir), Path(bundle)
    pred_dir, score_dir = run_dir / "predictions", run_dir / "scoring"
    prep = verify_predictions(pred_dir, bundle)
    R = Report()
    R.ok("predictions_verified", prep["all_checks_passed"], prep["problems"][:5])
    man = C.verify_bundle(bundle)
    _checksums(R, score_dir, "scoring")
    st = json.loads((score_dir / "RUN_STATUS.json").read_text(encoding="utf-8"))
    R.ok("scoring:bundle_identity", st.get("bundle_manifest_sha256") == C.sha256_file(bundle / "bundle_manifest.json"))
    R.ok("scoring:bound_to_predictions",
         st.get("predictions_status_sha256") == C.sha256_file(pred_dir / "PREDICTIONS_STATUS.json") and
         all(st.get("predictions_sha256", {}).get(l) == C.sha256_file(pred_dir / "models" / l / "predictions.jsonl")
             for l in EXPECTED_MODELS if (pred_dir / "models" / l / "predictions.jsonl").is_file()),
         "scoring was not produced from these predictions")
    recs = C.load_dev(bundle)
    ids = [r["id"] for r in recs]
    mapping = C.load_mapping(bundle)
    prior = json.loads((bundle / "data/prior_audit.json").read_text(encoding="utf-8"))
    K, P = C.bundle_modules(bundle)
    refs = C.load_reference(bundle)
    exp = json.loads((bundle / "data/reference_expected.json").read_text(encoding="utf-8"))["items"]
    R.ok("reference:bundled_rows_match_digests", [C.reference_digest(r) for r in refs] == exp)
    rc = C.recheck_reference(recs, refs, K, bundle / "database")
    R.ok("reference:gold_sql_reexecuted", not rc["problems"], rc["problems"][:5])
    rr = score_dir / "preflight/reference_recheck.json"
    R.ok("reference:scoring_recheck_clean", rr.is_file() and not json.loads(rr.read_text(encoding="utf-8"))["problems"])
    ref_by = {r["id"]: r for r in refs}
    allowed = C.allowed_collections(mapping)
    summaries, evals_by = {}, {}
    for label in EXPECTED_MODELS:
        d = score_dir / "models" / label
        if not R.ok(f"{label}:scoring_present", (d / "evaluation_results.jsonl").is_file(), "no evaluation_results.jsonl"):
            continue
        cfg = merged_config(pred_dir, score_dir, label)
        preds = C.read_jsonl(pred_dir / "models" / label / "predictions.jsonl")
        evals = C.read_jsonl(d / "evaluation_results.jsonl")
        rows = {r["id"]: r for r in C.read_jsonl(d / "prediction_rows.jsonl.gz")} \
            if (d / "prediction_rows.jsonl.gz").is_file() else {}
        ei = [e["id"] for e in evals]
        R.ok(f"{label}:evaluation_ids", ei == [p["id"] for p in preds], f"{len(ei)} evaluated vs {len(preds)} predicted")
        R.ok(f"{label}:evaluation_fingerprint", {e["fingerprint"] for e in evals} <= {cfg.get("evaluation_fingerprint")})
        pred_by = {p["id"]: p for p in preds}
        mism, mapbad, rescore = [], [], []
        for e in evals:
            p, ref = pred_by.get(e["id"]), ref_by[e["id"]]
            if p is None:
                mism.append(e["id"])
                continue
            if e.get("prediction_sha256") != C.sha256_text(C.canon({k: p.get(k) for k in ("id", "raw_text", "status")})):
                mism.append(e["id"])
            if p.get("status") == "ok":
                stage, mql, _, _ = C.check_mql(p.get("raw_text"), allowed[ref["db_id"]])
                if (stage == "ok") != e["readonly_valid"] or (stage == "ok" and mql != e["mql"]):
                    mism.append(e["id"])
                if e["mql"] and (e["mql"]["collection"] not in allowed[ref["db_id"]] or
                                 mapping["collections"][e["mql"]["collection"]]["db_id"] != ref["db_id"]):
                    mapbad.append(e["id"])
            if e["executed"] and e.get("detail") != "result_too_large":
                r = rows.get(e["id"])
                if r is None or C.recompute_outcome(K, e, r, ref) != e["outcome"]:
                    rescore.append(e["id"])
        R.ok(f"{label}:validation_reproduced", not mism, f"{len(mism)} records differ, e.g. {mism[:5]}")
        R.ok(f"{label}:db_mapping_consistent", not mapbad, mapbad[:5])
        R.ok(f"{label}:verdicts_rescored_offline", not rescore, f"{len(rescore)} differ, e.g. {rescore[:5]}")
        s = summarize_model(label, cfg, preds, evals, refs, ids)
        saved = json.loads((d / "baseline_summary.json").read_text(encoding="utf-8"))
        R.ok(f"{label}:baseline_summary_recomputed", C.canon(s) == C.canon(saved), "recomputed summary differs")
        summaries[label], evals_by[label] = s, evals
    R.ok("scoring:single_model_no_comparison", not (score_dir / "comparison_summary.json").exists() and
         sorted(p.name for p in (score_dir / "models").iterdir()) == sorted(EXPECTED_MODELS),
         "unexpected comparison file or model directory")
    _scan_outputs(R, score_dir, man, "scoring")
    all_ok = not R.problems and prep["all_checks_passed"]
    complete = all_ok and prep["complete"] and st.get("status") == "complete" and \
        len(summaries) == len(EXPECTED_MODELS) and \
        all(s["status"] == "complete" for s in summaries.values())
    return {"verified_at": C.now(), "kind": "scoring", "run_dir": str(run_dir), "run_status": st.get("status"),
            "all_checks_passed": all_ok, "complete": complete, "problems": prep["problems"] + R.problems,
            "checks": prep["checks"] + R.checks,
            "summaries": {k: {"correct": v["correct"], "accuracy": v["accuracy"], "status": v["status"]}
                          for k, v in summaries.items()}}


def _print(rep):
    for c in rep["checks"]:
        print(("OK   " if c["ok"] else "FAIL ") + c["check"] + ("" if c["ok"] else f"  {c['detail']}"))
    print(json.dumps({k: rep.get(k) for k in ("status", "run_status", "all_checks_passed", "complete", "models",
                                               "summaries") if k in rep}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predictions")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip")
    g.add_argument("--dir")
    p.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    p.add_argument("--staging-root", default=str(DEST_ROOT / ".staging"))
    s = sub.add_parser("scoring")
    s.add_argument("--run-dir", default=str(DEST_ROOT / "v1"))
    s.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    a = ap.parse_args(argv)
    try:
        if a.cmd == "predictions":
            d = safe_extract(a.zip, Path(a.staging_root) / (time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]), "PREDICTIONS_STATUS.json") \
                if a.zip else Path(a.dir)
            rep = verify_predictions(d, a.bundle)
        else:
            rep = verify_scoring(a.run_dir, a.bundle)
        _print(rep)
        return 0 if rep["all_checks_passed"] else 1
    except (VerifyError, C.PreflightError, FileNotFoundError) as e:
        print(f"VERIFY ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
