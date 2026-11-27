"""MQL-0007 local multi-arm scoring (Windows -> Atlas, READ-ONLY). Kaggle only generates.

A separate adapter: it imports the MQL-0006 core (dev1034_core: safety gate, read-only Atlas execution, comparator,
checkpoints) and `local_atlas_scoring.default_client_factory` UNCHANGED, and never reads or writes anything under
outputs/nl2mql_spider_dev_1034_model_comparison (the MQL-0006 results). Output root: outputs/nl2mql_mql0007_rag_v1/.

Commands
  verify   --predictions-zip Z [--predictions-zip Z2 ...]      offline: validate the Kaggle outputs, no Atlas
  score    --predictions-zip Z [...] [--run-name v1]            verify, Atlas preflight (read-only privileges), score A, B, C
                                                                (+ supplementary D), analyse, report
  analyze  --run-dir D                                          recompute the statistics from stored scoring (offline)

Each of the arms A, B, C must come from exactly one complete output (several zips are fine: one per Kaggle version). An arm present
in two zips is accepted only if the two copies are byte-identical. Partial arms are refused. torch, transformers,
accelerate, tokenizers, GPU name and dtype must be identical across the arms (the "same environment" requirement).
"""
import argparse
import json
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
import dev1034_core as C  # noqa: E402
import mql0007_stats as S  # noqa: E402

PKG = ROOT / "kaggle/nl2mql_mql0007_rag_v1"
BUNDLE = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2"
DEST_ROOT = ROOT / "outputs/nl2mql_mql0007_rag_v1"
MQL6_ROOT = ROOT / "outputs/nl2mql_spider_dev_1034_model_comparison"
EXPERIMENT = "MQL-0007"
ARMS, SUPP = S.ARMS, "D_supp_allow_near_dup"
POST_FP = {"experiment", "fingerprint", "generation_wall_s", "resolved_revision", "generated_this_session", "environment", "reused_from"}
GEN_KEYS = ("id", "db_id", "status", "model", "arm", "fingerprint", "prompt_sha256")


class ScoringError(RuntimeError):
    pass


def need(cond, msg):
    if not cond:
        raise ScoringError(msg)


def guard_dest(dest):
    """The adapter may never write into (or read results from) the MQL-0006 output tree."""
    d = Path(dest).resolve()
    need(MQL6_ROOT.resolve() not in (d, *d.parents), f"refusing to use a path under the MQL-0006 results: {d}")
    return d


# ------------------------------------------------------------------ expected inputs, derived from the package only
def load_expected(pkg=PKG):
    pkg = Path(pkg)
    man = C.read_jsonl  # noqa: F841 (readability)
    cfg = json.loads((pkg / "configs/experiment.json").read_text(encoding="utf-8"))
    prompts = C.read_jsonl(pkg / "data/eval/prompts.jsonl")
    ids = [p["id"] for p in prompts]
    plans = {a: C.read_jsonl(pkg / f"data/plans/retrieval_plan_{a}.jsonl") for a in ("B_random_k3", "C_hybrid_k3", SUPP)}
    ph = {"A_zero_shot": {p["id"]: p["prompt_sha256"] for p in prompts}}
    ph.update({a: {p["id"]: p["prompt_sha256"] for p in pl} for a, pl in plans.items()})
    arm_ids = {"A_zero_shot": ids, "B_random_k3": [p["id"] for p in plans["B_random_k3"]],
               "C_hybrid_k3": [p["id"] for p in plans["C_hybrid_k3"]], SUPP: [p["id"] for p in plans[SUPP]]}
    need(arm_ids["B_random_k3"] == ids and arm_ids["C_hybrid_k3"] == ids and len(ids) == 892 == len(set(ids)),
         "package plans do not cover exactly the 892 evaluation ids")
    need(C.sha256_text("\n".join(ids)) == cfg["evaluation"]["ids_sha256"], "package ids hash != config")
    return {"cfg": cfg, "ids": ids, "arm_ids": arm_ids, "prompt_hashes": ph, "plans": plans,
            "package_manifest_sha256": C.sha256_file(pkg / "package_manifest.json"), "config_sha256": C.sha256_file(pkg / "configs/experiment.json"),
            "mql6_predictions_sha256": C.sha256_file(pkg / "data/zero_shot_mql0006/predictions.jsonl"),
            "mql6_cfg": json.loads((pkg / "data/zero_shot_mql0006/run_config.json").read_text(encoding="utf-8"))}


# ------------------------------------------------------------------ verification of Kaggle outputs (offline)
class Report:
    def __init__(self):
        self.problems, self.checks = [], []

    def ok(self, name, cond, detail=""):
        self.checks.append({"check": name, "ok": bool(cond), "detail": "" if cond else str(detail)[:400]})
        if not cond:
            self.problems.append(f"{name}: {str(detail)[:300]}")
        return bool(cond)


def _read_sums(d):
    f = Path(d) / "SHA256SUMS.txt"
    need(f.is_file(), f"{d}: SHA256SUMS.txt missing")
    sums = {}
    for ln in f.read_text(encoding="utf-8").splitlines():
        h, _, n = ln.partition("  ")
        sums[n] = h
    return sums


def parse_predictions(path, arm, exp_ids, exp_prompt_sha, reused):
    """-> (records | None, problems). Rejects malformed, duplicate, unexpected, missing, misordered, wrong-prompt records."""
    probs, recs = [], []
    try:
        lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError as e:
        return None, [f"unreadable: {e}"]
    for n, ln in enumerate(lines, 1):
        try:
            r = json.loads(ln)
        except ValueError:
            probs.append(f"line {n}: not JSON")
            continue
        if not isinstance(r, dict):
            probs.append(f"line {n}: not an object")
            continue
        keys = [k for k in GEN_KEYS if not (reused and k == "arm")]
        miss = [k for k in keys if not isinstance(r.get(k), str) or not r.get(k)]
        if miss:
            probs.append(f"line {n} ({r.get('id')}): missing/non-string {miss}")
            continue
        if r["status"] not in ("ok", "error"):
            probs.append(f"{r['id']}: status {r['status']!r}")
        elif r["status"] == "ok" and not isinstance(r.get("raw_text"), str):
            probs.append(f"{r['id']}: ok record without raw_text")
        elif r["status"] == "error":
            probs.append(f"{r['id']}: generation error record (arm is not complete)")
        if not reused and r.get("arm") != arm:
            probs.append(f"{r['id']}: record of arm {r.get('arm')!r} in arm {arm}")
        recs.append(r)
    ids = [r["id"] for r in recs if isinstance(r, dict) and "id" in r]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        probs.append(f"duplicate ids {dup[:5]}")
    unexpected = sorted(set(ids) - set(exp_ids))
    if unexpected:
        probs.append(f"unexpected ids {unexpected[:5]}")
    missing = [i for i in exp_ids if i not in set(ids)]
    if missing:
        probs.append(f"{len(missing)} missing ids, e.g. {missing[:3]}")
    if not dup and not unexpected and not missing and ids != list(exp_ids):
        probs.append("ids out of order")
    bad = [r["id"] for r in recs if r.get("prompt_sha256") != exp_prompt_sha.get(r.get("id"))]
    if bad:
        probs.append(f"{len(bad)} predictions generated from a different prompt than the frozen one, e.g. {bad[:3]}")
    return recs, probs


def env_signature(cfg):
    pk = (cfg.get("environment") or {}).get("packages") or {}
    return {"torch": cfg.get("torch"), "transformers": cfg.get("transformers"), "accelerate": pk.get("accelerate"),
            "tokenizers": pk.get("tokenizers"), "gpu": cfg.get("gpu"), "dtype": cfg.get("dtype")}


def verify_predictions(dirs, exp, allow_env_drift=False):
    """dirs: extracted MQL-0007 output directories. -> report with arm -> source mapping."""
    R = Report()
    cfgx = exp["cfg"]
    M = cfgx["model"]
    arm_src, arm_cfg, env = {}, {}, {}
    for d in map(Path, dirs):
        try:
            sums = _read_sums(d)
            files = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
            R.ok(f"{d.name}:checksums_cover_all_files", files == set(sums), sorted(files ^ set(sums))[:5])
            R.ok(f"{d.name}:checksums_match", all(C.sha256_file(d / n) == h for n, h in sums.items() if (d / n).is_file()), "hash mismatch")
            st = json.loads((d / "MQL0007_STATUS.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, ScoringError) as e:
            R.ok(f"{d.name}:readable", False, e)
            continue
        R.ok(f"{d.name}:experiment_id", st.get("experiment") == EXPERIMENT and st.get("stage") == "predictions", (st.get("experiment"), st.get("stage")))
        R.ok(f"{d.name}:package_identity", st.get("package_manifest_sha256") == exp["package_manifest_sha256"] and
             st.get("config_sha256") == exp["config_sha256"] and st.get("evaluation_ids_sha256") == cfgx["evaluation"]["ids_sha256"],
             "package manifest/config/ids hash differ from the local package")
        for arm_dir in sorted((d / "models").iterdir()) if (d / "models").is_dir() else []:
            arm = arm_dir.name
            if arm not in ARMS + (SUPP,):
                R.ok(f"{d.name}:{arm}:known_arm", False, "unknown arm directory")
                continue
            if arm in arm_src:
                same = (C.sha256_file(arm_dir / "predictions.jsonl") == C.sha256_file(arm_src[arm] / "predictions.jsonl"))
                R.ok(f"{arm}:duplicate_source_identical", same, "the arm appears in two outputs with DIFFERENT predictions")
                continue
            arm_src[arm] = arm_dir
    for arm in ARMS:
        R.ok(f"{arm}:present", arm in arm_src, "no output provides this arm")
    for arm, ad in arm_src.items():
        try:
            cfg = json.loads((ad / "run_config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            R.ok(f"{arm}:run_config", False, e)
            continue
        arm_cfg[arm] = cfg
        reused = cfg.get("mode") == "reuse_mql0006"
        R.ok(f"{arm}:model_identity", cfg.get("model_id") == M["id"] and cfg.get("model_revision") == M["revision"], (cfg.get("model_id"), cfg.get("model_revision")))
        R.ok(f"{arm}:decoding", cfg.get("decoding") == M["decoding"] and cfg.get("max_new_tokens") == M["max_new_tokens"] and
             cfg.get("batch_size") == M["batch_size"] and cfg.get("dtype") == M["dtype"], {k: cfg.get(k) for k in ("decoding", "max_new_tokens", "batch_size", "dtype")})
        if reused:
            R.ok(f"{arm}:reuse_is_byte_identical_to_mql0006", C.sha256_file(ad / "predictions.jsonl") == exp["mql6_predictions_sha256"], "reused predictions differ from MQL-0006")
            R.ok(f"{arm}:reuse_only_for_arm_A", arm == "A_zero_shot", "reuse mode on a retrieval arm")
            R.ok(f"{arm}:reuse_fingerprint", cfg.get("fingerprint") == exp["mql6_cfg"]["fingerprint"], "fingerprint != MQL-0006's")
            env[arm] = env_signature({**exp["mql6_cfg"]})
        else:
            R.ok(f"{arm}:fingerprint_recomputes", C.fingerprint({k: v for k, v in cfg.items() if k not in POST_FP}) == cfg.get("fingerprint"), "run_config fingerprint does not recompute")
            R.ok(f"{arm}:arm_label", cfg.get("arm") == arm and cfg.get("experiment") == EXPERIMENT, (cfg.get("arm"), cfg.get("experiment")))
            if arm != "A_zero_shot":
                R.ok(f"{arm}:exclusions_in_config", cfg.get("retrieval_exclusions") == (cfgx["retrieval_exclusions"] if arm != SUPP else []), "retrieval exclusions differ")
            env[arm] = env_signature(cfg)
        recs, probs = parse_predictions(ad / "predictions.jsonl", arm, exp["arm_ids"][arm], exp["prompt_hashes"][arm], reused)
        R.ok(f"{arm}:predictions_exact_ids_prompts_complete", not probs, "; ".join(probs[:6]))
        if recs and not reused:
            R.ok(f"{arm}:record_fingerprints", {r["fingerprint"] for r in recs} <= {cfg.get("fingerprint")}, "records from another run")
        if arm != "A_zero_shot":
            log = C.read_jsonl(ad / "retrieval_log.jsonl") if (ad / "retrieval_log.jsonl").is_file() else None
            ok = log is not None and [{k: v for k, v in r.items() if k != "near_duplicate_template_item"} for r in log] == exp["plans"][arm]
            R.ok(f"{arm}:retrieval_log_equals_frozen_plan", ok, "retrieval log differs from the frozen plan (exclusion/selection)")
    primary_env = {a: env[a] for a in ARMS if a in env}
    sigs = {json.dumps(v, sort_keys=True) for v in primary_env.values()}
    drift = len(sigs) > 1
    if drift and not allow_env_drift:
        R.ok("environment_identical_across_arms", False, {a: v for a, v in primary_env.items()})
    else:
        R.checks.append({"check": "environment_identical_across_arms", "ok": True,
                         "detail": "allowed drift (explicit flag)" if drift else "identical"})
    return {"verified_at": C.now(), "ok": not R.problems, "problems": R.problems, "checks": R.checks,
            "arm_source": {a: str(p) for a, p in arm_src.items()}, "environment": env, "environment_drift": drift,
            "zero_shot_mode": (arm_cfg.get("A_zero_shot") or {}).get("mode")}


# ------------------------------------------------------------------ Atlas scoring
def eval_fingerprint(bundle_man):
    return C.fingerprint({"core": bundle_man["code_sha256"], "mapping": bundle_man["files"]["data/atlas_mapping.json"]["sha256"],
                          "reference": bundle_man["files"]["data/reference.jsonl.gz"]["sha256"], "timeout_ms": C.EXEC_TIMEOUT_MS,
                          "max_rows": C.MAX_RESULT_ROWS, "adapter": "mql0007-v1"})


def score_arm(arm, preds, ids, refs, mapping, mdb, K, out_dir, fp, log=print):
    """Execute + compare one arm's predictions through the unchanged MQL-0006 evaluator."""
    d = Path(out_dir) / "models" / arm
    eck = C.Checkpoint(d / "checkpoints/evaluation.partial.jsonl", ids, fp, C.is_eval_retry)
    rck = C.Checkpoint(d / "checkpoints/rows.partial.jsonl", ids, fp)
    t0 = time.time()
    n = C.run_evaluation(preds, refs, mapping, mdb, K, eck, rck, C.EXEC_TIMEOUT_MS, progress=lambda n, t: log(f"  {arm}: {n}/{t}"))
    evals = [eck.records[i] for i in ids if i in eck.records]
    C.write_jsonl(d / "evaluation_results.jsonl", evals)
    acct = S.account(evals, ids)
    summ = {"experiment": EXPERIMENT, "arm": arm, "n_expected": len(ids), "accounting": acct, "correct": acct["correct"],
            "accuracy": S.acc([e["outcome"] == "correct" for e in evals]), "evaluated_this_session": n,
            "evaluation_wall_s": round(time.time() - t0, 1), "evaluation_fingerprint": fp,
            "unresolved_infrastructure_failures": acct["infrastructure_failure"] + acct["generation_failure"],
            "status": "complete" if not (acct["infrastructure_failure"] or acct["generation_failure"]) else "partial"}
    C.write_json(d / "summary.json", summ)
    return evals, summ


def query_patterns(bundle):
    """Gold-SQL pattern labels (local only; never shipped)."""
    sys.path.insert(0, str(PKG / "src"))
    import patterns as PT
    return {r["id"]: PT.sql_patterns(r["sql"]) - {"select"} for r in C.load_dev(bundle)}


def render_report(st, analysis, summaries):
    L = [f"# MQL-0007 results (cross-engine execution accuracy, Atlas vs SQLite gold)", "",
         f"Status: **{st['status']}**. Denominator: {st['n_expected']} items per arm. Zero-shot arm: `{st['zero_shot_mode']}`.", "",
         "Not native MQL exact match. Corpus is provisional (not human-reviewed). Analysis plan sha256 "
         f"`{analysis['analysis_plan_sha256'][:16]}…` (registered before any prediction existed).", "",
         "## Outcome accounting (each arm sums to 892)", "", "| arm | " + " | ".join(S.CATEGORIES) + " |", "|---|" + "---|" * len(S.CATEGORIES)]
    for a, s in summaries.items():
        L.append(f"| {a} | " + " | ".join(str(s["accounting"][c]) for c in S.CATEGORIES) + " |")
    pa = analysis["primary_analysis"]
    L += ["", "## Accuracy (Wilson 95%)", "", "| arm | correct / n | % | 95% CI |", "|---|---|---|---|"]
    for a, v in pa["arms"].items():
        L.append(f"| {a} | {v['correct']} / {v['total']} | {v['percent']} | {v['wilson95'][0]:.4f}–{v['wilson95'][1]:.4f} |")
    L += ["", "## Primary paired comparisons (exact McNemar, Holm over the two)", "",
          "| comparison | a only | b only | diff | bootstrap 95% | p | Holm p |", "|---|---|---|---|---|---|---|"]
    for k, v in pa["primary"].items():
        L.append(f"| {k} | {v['a_only']} | {v['b_only']} | {v['diff_a_minus_b']:+.4f} | {v['diff_bootstrap95']} | {v['mcnemar_exact_p']:.4g} | {v['holm_adjusted_p']:.4g} |")
    L += ["", "Secondary (unadjusted): " + "; ".join(f"{k}: a-only {v['a_only']}, b-only {v['b_only']}, p={v['mcnemar_exact_p']:.4g}" for k, v in pa["secondary"].items()), "",
          "Sensitivity analyses (S1 swap arm D into C, S2 drop the two near-duplicate items, S3 reference_status ok) are in `analysis.json`.", ""]
    return "\n".join(L)


def cmd_score(zips, run_name="v1", bundle=BUNDLE, pkg=PKG, dest_root=DEST_ROOT, client_factory=None, full_parity=False,
              allow_env_drift=False, log=print):
    dest_root = guard_dest(dest_root)
    exp = load_expected(pkg)
    bman = C.verify_bundle(bundle)
    ts = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    import verify_kaggle_spider_dev_1034 as V
    staged = [V.safe_extract(z, dest_root / ".staging" / ts / f"z{i}", "MQL0007_STATUS.json") for i, z in enumerate(zips)]
    ver = verify_predictions(staged, exp, allow_env_drift)
    run_dir = dest_root / run_name
    C.write_json(run_dir / "predictions.verification.json", ver)
    need(ver["ok"], f"predictions failed verification: {ver['problems'][:5]}")
    pred_dir = run_dir / "predictions"
    for arm, src in ver["arm_source"].items():
        dst = pred_dir / arm
        if dst.exists():
            need(C.sha256_file(dst / "predictions.jsonl") == C.sha256_file(Path(src) / "predictions.jsonl"), f"{dst} already holds DIFFERENT predictions")
        else:
            shutil.copytree(src, dst)
    # reference + read-only Atlas preflight BEFORE any scoring
    K, _ = C.bundle_modules(bundle)
    refs, recs = C.load_reference(bundle), C.load_dev(bundle)
    mapping = C.load_mapping(bundle)
    if client_factory is None:
        import local_atlas_scoring as L
        client_factory = L.default_client_factory
    client, _redact = client_factory()
    score_dir = run_dir / "scoring"
    try:
        atl = C.atlas_preflight(client, mapping, full_parity=full_parity, log=log)   # fails closed unless the user is read-only
        C.write_json(score_dir / "preflight/atlas_check.json", atl)
        mdb = client[mapping["atlas_db"]]
        fp = eval_fingerprint(bman)
        evals_by, summaries = {}, {}
        for arm in ARMS + ((SUPP,) if SUPP in ver["arm_source"] else ()):
            preds = C.read_jsonl(pred_dir / arm / "predictions.jsonl")
            ev, sm = score_arm(arm, preds, exp["arm_ids"][arm], refs, mapping, mdb, K, score_dir, fp, log)
            evals_by[arm], summaries[arm] = ev, sm
            log(f"{arm}: {sm['correct']}/{sm['n_expected']} correct; {sm['accounting']}")
    finally:
        client.close()
    outcomes = {a: {e["id"]: e["outcome"] for e in evals_by[a]} for a in ARMS}
    meta = {r["id"]: {"hardness": r["hardness"], "db_id": r["db_id"], "reference_status": r["reference_status"]} for r in refs}
    nd = sorted({x["eval_id"] for x in exp["cfg"]["retrieval_exclusions"]})
    supp = {e["id"]: e["outcome"] for e in evals_by.get(SUPP, [])}
    analysis = S.analyze(outcomes, exp["ids"], meta, query_patterns(bundle), nd, supp)
    C.write_json(score_dir / "analysis.json", analysis)
    complete = all(s["status"] == "complete" for s in summaries.values())
    st = {"experiment": EXPERIMENT, "stage": "scoring", "status": "complete" if complete else "partial", "written_at": C.now(),
          "n_expected": len(exp["ids"]), "zero_shot_mode": ver["zero_shot_mode"], "arms": {a: s["accounting"] for a, s in summaries.items()},
          "package_manifest_sha256": exp["package_manifest_sha256"], "analysis_plan_sha256": S.plan_sha(),
          "metric": "cross-engine execution accuracy (Atlas MQL result vs SQLite gold-SQL result); NOT native MQL exact match",
          "atlas_server_version": atl.get("server_version"), "scorer": "scripts/nl2mql/mql0007_scoring.py",
          "scorer_sha256": C.sha256_file(Path(__file__)), "stats_sha256": C.sha256_file(HERE / "mql0007_stats.py")}
    C.write_json(score_dir / "RUN_STATUS.json", st)
    (score_dir / "results_report.md").write_text(render_report(st, analysis, summaries), encoding="utf-8")
    names = sorted(p.relative_to(score_dir).as_posix() for p in score_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (score_dir / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(score_dir / n)}  {n}\n" for n in names), encoding="utf-8")
    return st, analysis, run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "score"):
        p = sub.add_parser(name)
        p.add_argument("--predictions-zip", action="append", required=True)
        p.add_argument("--run-name", default="v1")
        p.add_argument("--allow-env-drift", action="store_true", help="accept different library/GPU versions across arms (recorded)")
        if name == "score":
            p.add_argument("--full-parity", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        import verify_kaggle_spider_dev_1034 as V
        exp = load_expected()
        ts = time.strftime("%Y%m%dT%H%M%S")
        staged = [V.safe_extract(z, guard_dest(DEST_ROOT) / ".staging" / ts / f"z{i}", "MQL0007_STATUS.json") for i, z in enumerate(a.predictions_zip)]
        rep = verify_predictions(staged, exp, a.allow_env_drift)
        print(json.dumps({k: rep[k] for k in ("ok", "problems", "zero_shot_mode", "environment_drift")}, indent=1))
        return 0 if rep["ok"] else 1
    st, analysis, run_dir = cmd_score(a.predictions_zip, a.run_name, allow_env_drift=a.allow_env_drift, full_parity=a.full_parity)
    print(json.dumps({"status": st["status"], "arms": st["arms"]}, indent=1))
    print("report:", run_dir / "scoring/results_report.md")
    return 0 if st["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
