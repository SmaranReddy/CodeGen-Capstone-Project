"""MQL-0008 local two-arm scoring (Windows -> Atlas, READ-ONLY). Kaggle only generates.

A thin adapter. Execution, the safety gate, the result comparison and the outcome classification are the UNCHANGED
MQL-0006 evaluator (scripts/nl2mql/dev1034_core.py: evaluate_prediction, run_evaluation, atlas_preflight, Checkpoint)
and the read-only client of scripts/nl2mql/local_atlas_scoring.py. Nothing about result equivalence is re-implemented.
It never writes under the MQL-0006 or MQL-0007 result trees and never modifies the MQL-0008 Kaggle package.

Commands
  verify   --predictions-zip Z [--predictions-zip Z2]         offline: validate the Kaggle export(s); no Atlas
  score    --predictions-zip Z [...] [--run-name NAME]         verify, read-only Atlas preflight, score both arms, analyse, report
                                                               (re-run with the same --run-name to resume)
  analyze  --run-dir D                                         recompute analysis + report from saved per-record results; no Atlas

Each arm must come from exactly one complete export (one zip with both arms, or one zip per arm). An arm present in two
zips is accepted only if both copies are byte-identical. Incomplete, reordered or mismatched inputs are refused; nothing
is repaired or substituted.
"""
import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
sys.path.insert(0, str(ROOT / "scripts"))
import dev1034_core as C  # noqa: E402  (unchanged MQL-0006 evaluator core)
import mql0007_scoring as SC7  # noqa: E402  (reused: Report, parse_predictions, env_signature, _read_sums)
import mql0008_stats as S  # noqa: E402

PKG = ROOT / "kaggle/nl2mql_mql0008_approved_corpus_v2_1"
BUNDLE = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2"
DEST_ROOT = ROOT / "outputs/nl2mql_mql0008_approved_corpus_v2_1"
PROTECTED_ROOTS = (ROOT / "outputs/nl2mql_spider_dev_1034_model_comparison", ROOT / "outputs/nl2mql_mql0007_rag_v1", ROOT / "kaggle")
EXPERIMENT = "MQL-0008"
ARM_A, ARM_B, ARMS = S.ARM_A, S.ARM_B, S.ARMS
STATUS_FILE = "MQL0008_STATUS.json"
# frozen identities (the adapter refuses anything else)
EXPECTED_CORPUS_SHA256 = "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf"
EXPECTED_CORPUS_RECORDS = 3445
EXPECTED_PACKAGE_MANIFEST_SHA256 = "fb20c496fa20fbc5eecc60b1407ac3858a6370cd607df257709b3cea4e26fed8"
EXPECTED_MODEL = ("Qwen/Qwen2.5-Coder-3B-Instruct", "488639f1ff808d1d3d0ba301aef8c11461451ec5")
N_EVAL = 892
POST_FP = {"experiment", "fingerprint", "generation_wall_s", "resolved_revision", "generated_this_session", "environment"}
ARM_FILES = {"predictions.jsonl", "run_config.json", "retrieval_log.jsonl", "checkpoints/predictions.partial.jsonl"}
TOP_FILES = {STATUS_FILE, "preflight/preflight.json"}


class ScoringError(RuntimeError):
    pass


def need(cond, msg):
    if not cond:
        raise ScoringError(msg)


def guard_dest(dest):
    """The adapter writes only under its own output tree: never into earlier experiments' results or a Kaggle package."""
    d = Path(dest).resolve()
    for root in PROTECTED_ROOTS:
        need(root.resolve() not in (d, *d.parents), f"refusing to write under {root}: {d}")
    return d


# ------------------------------------------------------------------ expected inputs, derived from the frozen package
def load_expected(pkg=PKG, bundle=BUNDLE, expected_manifest_sha=EXPECTED_PACKAGE_MANIFEST_SHA256):
    pkg = Path(pkg)
    sys.path.insert(0, str(pkg / "src"))
    import mql0008_core as K
    K.verify_package(pkg)                                             # every package file equals its manifest hash
    man_sha = C.sha256_file(pkg / "package_manifest.json")
    need(man_sha == expected_manifest_sha, f"MQL-0008 package manifest {man_sha} != frozen {expected_manifest_sha}")
    cfg = json.loads((pkg / "configs/experiment.json").read_text(encoding="utf-8"))
    corpus_sha = C.sha256_file(pkg / K.CORPUS_FILE)
    need(corpus_sha == EXPECTED_CORPUS_SHA256 == cfg["corpus"]["source_sha256"], "package corpus is not the approved corpus v2.1")
    corpus_db = {r["id"]: r["db_id"] for r in C.read_jsonl(pkg / K.CORPUS_FILE)}
    need(len(corpus_db) == EXPECTED_CORPUS_RECORDS, "approved corpus record count")
    need((cfg["model"]["id"], cfg["model"]["revision"]) == EXPECTED_MODEL and tuple(cfg["arms"]) == ARMS, "package model / arms")
    prompts = C.read_jsonl(pkg / "data/eval/prompts.jsonl")
    evals = C.read_jsonl(pkg / "data/eval/eval_inputs.jsonl")
    ids = [e["id"] for e in evals]
    plan = C.read_jsonl(pkg / f"data/plans/retrieval_plan_{ARM_B}.jsonl")
    need(len(ids) == N_EVAL == len(set(ids)) and [p["id"] for p in prompts] == ids == [p["id"] for p in plan],
         "package does not cover exactly the 892 evaluation ids")
    need(C.sha256_text("\n".join(ids)) == cfg["evaluation"]["ids_sha256"], "package ids hash != config")
    bman = json.loads((Path(bundle) / "bundle_manifest.json").read_text(encoding="utf-8"))
    need(bman["evaluation_set"]["ids_sha256"] == cfg["evaluation"]["ids_sha256"], "package ids != evaluation bundle ids")
    return {"cfg": cfg, "ids": ids, "db_of": {e["id"]: e["db_id"] for e in evals}, "plan": plan, "corpus_db": corpus_db,
            "prompt_hashes": {ARM_A: {p["id"]: p["prompt_sha256"] for p in prompts},
                              ARM_B: {p["id"]: p["prompt_sha256"] for p in plan}},
            "held_out_dbs": sorted(bman["evaluation_set"]["excluded_db_ids"]),      # names only; the partition is never opened
            "retrieval_fp": K.retrieval_fingerprint(cfg), "identity": lambda arm, row: K.record_identity(cfg, arm, row, K.retrieval_fingerprint(cfg)),
            "package_manifest_sha256": man_sha, "config_sha256": C.sha256_file(pkg / "configs/experiment.json"),
            "corpus_sha256": corpus_sha}


# ------------------------------------------------------------------ verification of the Kaggle export (offline)
def _check_records(R, arm, recs, exp):
    """Per-record identity: model, arm, generation settings, no holdout, and (arm B) retrieval metadata == frozen plan."""
    cfg, M = exp["cfg"], exp["cfg"]["model"]
    plan = {p["id"]: p for p in exp["plan"]}
    held = set(exp["held_out_dbs"])
    bad = {k: [] for k in ("identity", "holdout", "db", "retrieval", "target_db", "context")}
    for r in recs:
        i = r.get("id")
        if i not in exp["db_of"]:
            continue                                                   # unexpected ids are reported by parse_predictions
        want = exp["identity"](arm, plan[i])
        if any(r.get(k) != want[k] for k in ("experiment", "model_id", "model_revision", "generation_config")):
            bad["identity"].append(i)
        if r.get("db_id") in held:
            bad["holdout"].append(i)
        if r.get("db_id") != exp["db_of"][i]:
            bad["db"].append(i)
        if arm == ARM_A:
            if r.get("retrieval") is not None:
                bad["retrieval"].append(i)
            continue
        got = r.get("retrieval")
        if not isinstance(got, dict) or got != want["retrieval"]:
            bad["retrieval"].append(i)
            continue
        dbs = [exp["corpus_db"].get(x) for x in got["retrieved_ids"]]
        if (None in dbs or exp["db_of"][i] in dbs or dbs != got["retrieved_db_ids"] or got["target_db_excluded"] is not True
                or set(dbs) & (held | set(exp["db_of"].values()))):
            bad["target_db"].append(i)
        if not (0 < got["context_tokens"] <= cfg["context"]["max_context_tokens"] and len(got["retrieved_ids"]) <= cfg["context"]["k"]):
            bad["context"].append(i)
    R.ok(f"{arm}:record_model_experiment_generation_identity", not bad["identity"], f"{len(bad['identity'])} records, e.g. {bad['identity'][:3]}; expected {M['id']}@{M['revision']}")
    R.ok(f"{arm}:no_protected_holdout_database", not bad["holdout"], f"records on held-out databases: {bad['holdout'][:3]}")
    R.ok(f"{arm}:record_db_equals_evaluation_db", not bad["db"], f"{bad['db'][:3]}")
    if arm == ARM_A:
        R.ok(f"{arm}:no_retrieval_context", not bad["retrieval"], f"zero-shot records carrying retrieval metadata: {bad['retrieval'][:3]}")
    else:
        R.ok(f"{arm}:record_retrieval_metadata_equals_frozen_plan", not bad["retrieval"], f"{len(bad['retrieval'])} records, e.g. {bad['retrieval'][:3]}")
        R.ok(f"{arm}:target_database_excluded_and_only_approved_corpus", not bad["target_db"], f"{bad['target_db'][:3]}")
        R.ok(f"{arm}:context_within_cap", not bad["context"], f"{bad['context'][:3]}")


def verify_predictions(dirs, exp, allow_env_drift=False):
    """dirs: extracted MQL-0008 output directories. -> input validation report (ok, problems, checks, arm sources)."""
    R = SC7.Report()
    cfgx, M = exp["cfg"], exp["cfg"]["model"]
    arm_src, env, pred_sha, arm_ids = {}, {}, {}, {}
    for d in map(Path, dirs):
        try:
            sums = SC7._read_sums(d)
            files = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
            R.ok(f"{d.name}:checksums_cover_all_files", files == set(sums), sorted(files ^ set(sums))[:5])
            R.ok(f"{d.name}:prediction_hashes_match_export_manifest",
                 all((d / n).is_file() and C.sha256_file(d / n) == h for n, h in sums.items()), "a file differs from SHA256SUMS.txt")
            allowed = TOP_FILES | {f"predictions/{a}/{f}" for a in ARMS for f in ARM_FILES}
            R.ok(f"{d.name}:only_approved_files", files <= allowed, f"unexpected files: {sorted(files - allowed)[:5]}")
            st = json.loads((d / STATUS_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError, SC7.ScoringError) as e:
            R.ok(f"{d.name}:readable", False, e)
            continue
        R.ok(f"{d.name}:experiment_id", st.get("experiment") == EXPERIMENT and st.get("stage") == "predictions", (st.get("experiment"), st.get("stage")))
        R.ok(f"{d.name}:export_complete", st.get("status") == "complete" and not st.get("problems"), (st.get("status"), st.get("problems")))
        R.ok(f"{d.name}:package_identity", st.get("package_manifest_sha256") == exp["package_manifest_sha256"] and
             st.get("config_sha256") == exp["config_sha256"] and st.get("evaluation_ids_sha256") == cfgx["evaluation"]["ids_sha256"],
             "package manifest / config / evaluation ids hash differ from the frozen package")
        rc = st.get("retrieval_corpus") or {}
        R.ok(f"{d.name}:approved_corpus_hash", rc.get("sha256") == EXPECTED_CORPUS_SHA256 and rc.get("records") == EXPECTED_CORPUS_RECORDS,
             f"export used corpus {rc.get('sha256')} ({rc.get('records')} records)")
        sm = st.get("model") or {}
        R.ok(f"{d.name}:model_and_revision", (sm.get("id"), sm.get("revision")) == EXPECTED_MODEL, (sm.get("id"), sm.get("revision")))
        R.ok(f"{d.name}:retrieval_config_fingerprint", st.get("retrieval_config_fingerprint") == exp["retrieval_fp"], st.get("retrieval_config_fingerprint"))
        pdir = d / "predictions"
        for arm_dir in sorted(pdir.iterdir()) if pdir.is_dir() else []:
            arm = arm_dir.name
            if arm not in ARMS:
                R.ok(f"{d.name}:{arm}:known_arm", False, "unknown arm directory (only the two MQL-0008 arms are accepted)")
                continue
            psha = C.sha256_file(arm_dir / "predictions.jsonl") if (arm_dir / "predictions.jsonl").is_file() else None
            R.ok(f"{d.name}:{arm}:status_hash_equals_file", psha is not None and (st.get("arms") or {}).get(arm, {}).get("predictions_sha256") == psha,
                 "predictions.jsonl differs from the hash recorded in the export status")
            if arm in arm_src:
                R.ok(f"{arm}:duplicate_source_identical", psha == pred_sha[arm], "the arm appears in two exports with DIFFERENT predictions")
                continue
            arm_src[arm], pred_sha[arm] = arm_dir, psha
    for arm in ARMS:
        R.ok(f"{arm}:present", arm in arm_src, "no export provides this arm")
    for arm, ad in arm_src.items():
        try:
            cfg = json.loads((ad / "run_config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            R.ok(f"{arm}:run_config", False, e)
            continue
        is_rag = arm == ARM_B
        R.ok(f"{arm}:model_identity", (cfg.get("model_id"), cfg.get("model_revision")) == EXPECTED_MODEL, (cfg.get("model_id"), cfg.get("model_revision")))
        rr = cfg.get("resolved_revision")
        R.ok(f"{arm}:resolved_revision", rr in (None, M["revision"]), f"model resolved to {rr}")
        R.ok(f"{arm}:decoding", cfg.get("decoding") == M["decoding"] and cfg.get("max_new_tokens") == M["max_new_tokens"] and
             cfg.get("batch_size") == M["batch_size"] and cfg.get("dtype") == M["dtype"] and cfg.get("seed") == M["seed"],
             {k: cfg.get(k) for k in ("decoding", "max_new_tokens", "batch_size", "dtype", "seed")})
        R.ok(f"{arm}:fingerprint_recomputes", C.fingerprint({k: v for k, v in cfg.items() if k not in POST_FP}) == cfg.get("fingerprint"),
             "run_config fingerprint does not recompute")
        R.ok(f"{arm}:arm_and_experiment", cfg.get("arm") == arm and cfg.get("experiment") == EXPERIMENT and cfg.get("mode") == "generated" and
             cfg.get("arm_config") == cfgx["arms"][arm], (cfg.get("arm"), cfg.get("experiment"), cfg.get("mode")))
        R.ok(f"{arm}:package_and_evaluation_identity", cfg.get("package_manifest_sha256") == exp["package_manifest_sha256"] and
             cfg.get("config_sha256") == exp["config_sha256"] and cfg.get("evaluation_ids_sha256") == cfgx["evaluation"]["ids_sha256"], "run_config identity")
        R.ok(f"{arm}:corpus_identity", (cfg.get("corpus_sha256"), cfg.get("corpus_records"), cfg.get("retrieval_config_fingerprint")) ==
             ((EXPECTED_CORPUS_SHA256, EXPECTED_CORPUS_RECORDS, exp["retrieval_fp"]) if is_rag else (None, None, None)),
             (cfg.get("corpus_sha256"), cfg.get("corpus_records"), cfg.get("retrieval_config_fingerprint")))
        R.ok(f"{arm}:exclusions_in_config", cfg.get("retrieval_exclusions") == (cfgx["retrieval_exclusions"] if is_rag else []), "retrieval exclusions differ")
        env[arm] = SC7.env_signature(cfg)
        recs, probs = SC7.parse_predictions(ad / "predictions.jsonl", arm, exp["ids"], exp["prompt_hashes"][arm], False)
        R.ok(f"{arm}:exactly_892_unique_ids_in_order_from_the_frozen_prompts", not probs, "; ".join(probs[:6]))
        recs = recs or []
        arm_ids[arm] = [r.get("id") for r in recs]
        R.ok(f"{arm}:record_fingerprints", {r.get("fingerprint") for r in recs} <= {cfg.get("fingerprint")}, "records from another run")
        _check_records(R, arm, recs, exp)
        try:
            log = C.read_jsonl(ad / "retrieval_log.jsonl")
        except (OSError, ValueError):
            log = None
        strip = lambda rows: [{k: v for k, v in r.items() if k != "near_duplicate_template_item"} for r in rows]   # noqa: E731
        if is_rag:
            R.ok(f"{arm}:retrieval_log_equals_frozen_hybrid_plan", log is not None and strip(log) == exp["plan"],
                 "retrieval log differs from the frozen plan (selection, scores, exclusion or budget)")
        else:
            R.ok(f"{arm}:retrieval_log_is_empty_context", log is not None and [r.get("id") for r in log] == exp["ids"] and
                 all(r.get("empty_context") is True and r.get("selected_ids") == [] for r in log), "zero-shot retrieval log is not empty-context")
    if set(arm_ids) == set(ARMS):
        R.ok("identical_evaluation_ids_in_both_arms", arm_ids[ARM_A] == arm_ids[ARM_B] == exp["ids"], "the two arms do not cover the same ids in the same order")
    sigs = {json.dumps(env[a], sort_keys=True) for a in ARMS if a in env}
    drift = len(sigs) > 1
    if drift and not allow_env_drift:
        R.ok("environment_identical_across_arms", False, env)
    else:
        R.checks.append({"check": "environment_identical_across_arms", "ok": True, "detail": "allowed drift (explicit flag)" if drift else "identical"})
    return {"experiment": EXPERIMENT, "verified_at": C.now(), "ok": not R.problems, "problems": R.problems, "checks": R.checks,
            "arm_source": {a: str(p) for a, p in arm_src.items()}, "predictions_sha256": pred_sha, "environment": env,
            "environment_drift": drift,
            "frozen": {"package_manifest_sha256": exp["package_manifest_sha256"], "corpus_sha256": EXPECTED_CORPUS_SHA256,
                       "corpus_records": EXPECTED_CORPUS_RECORDS, "model": list(EXPECTED_MODEL),
                       "retrieval_config_fingerprint": exp["retrieval_fp"], "evaluation_ids_sha256": cfgx["evaluation"]["ids_sha256"]}}


# ------------------------------------------------------------------ Atlas scoring (unchanged evaluator)
def eval_fingerprint(bundle_man, predictions_sha256):
    """Scoring identity of one arm: evaluator code, mapping, reference, limits and the exact prediction file."""
    return C.fingerprint({"core": bundle_man["code_sha256"], "mapping": bundle_man["files"]["data/atlas_mapping.json"]["sha256"],
                          "reference": bundle_man["files"]["data/reference.jsonl.gz"]["sha256"], "timeout_ms": C.EXEC_TIMEOUT_MS,
                          "max_rows": C.MAX_RESULT_ROWS, "adapter": "mql0008-v1", "predictions": predictions_sha256})


def score_arm(arm, preds, ids, refs, mapping, mdb, K, out_dir, fp, log=print):
    """Execute + compare one arm through the unchanged MQL-0006 evaluator. Resumable: finished ids are never re-scored."""
    d = Path(out_dir) / "arms" / arm
    eck = C.Checkpoint(d / "checkpoints/evaluation.partial.jsonl", ids, fp, C.is_eval_retry)
    rck = C.Checkpoint(d / "checkpoints/rows.partial.jsonl", ids, fp)
    already = len(ids) - len(eck.todo())
    t0 = time.time()
    n = C.run_evaluation(preds, refs, mapping, mdb, K, eck, rck, C.EXEC_TIMEOUT_MS, progress=lambda n, t: log(f"  {arm}: {n}/{t}"))
    evals = [eck.records[i] for i in ids if i in eck.records]
    C.write_jsonl(d / "evaluation_results.jsonl", evals)
    summ = summarize(arm, evals, ids, fp)
    summ.update(evaluated_this_session=n, final_before_this_session=already, evaluation_wall_s=round(time.time() - t0, 1))
    C.write_json(d / "summary.json", summ)
    return evals, summ


def summarize(arm, evals, ids, fp=None):
    acct = S.S7.account(evals, ids)
    unresolved = acct["infrastructure_failure"] + acct["generation_failure"]
    return {"experiment": EXPERIMENT, "arm": arm, "n_expected": len(ids), "accounting": acct, "correct": acct["correct"],
            "accuracy": S.S7.acc([e["outcome"] == "correct" for e in evals]), "evaluation_fingerprint": fp,
            "unresolved_infrastructure_failures": acct["infrastructure_failure"], "generation_failures": acct["generation_failure"],
            "status": "complete" if not unresolved else "partial"}


def query_patterns(bundle, pkg=PKG):
    """Gold-SQL pattern labels of the 892 evaluation items (local only; never shipped)."""
    sys.path.insert(0, str(Path(pkg) / "src"))
    import patterns as PT
    return {r["id"]: PT.sql_patterns(r["sql"]) - {"select"} for r in C.load_dev(bundle)}


def render_report(st, analysis, summaries):
    pa, sec = analysis["primary_analysis"], analysis["secondary_exploratory"]
    a, b = pa["arms"][ARM_A], pa["arms"][ARM_B]
    t, d = pa["paired_table"], pa["difference_b_minus_a"]
    L = ["# MQL-0008 results: zero-shot vs hybrid RAG on the approved corpus v2.1", "",
         f"Status: **{st['status']}**. Denominator: {st['n_expected']} items per arm, none excluded.", "",
         "Metric: cross-engine execution accuracy (MQL result on Atlas vs gold-SQL result on SQLite); not native MQL exact match. "
         f"Atlas server {st['atlas'].get('server_version')}, read-only user verified. Retrieval corpus: approved corpus v2.1, "
         f"{EXPECTED_CORPUS_RECORDS:,} records, sha256 `{EXPECTED_CORPUS_SHA256}` (AI-reviewed, not human-reviewed). "
         f"Analysis plan sha256 `{analysis['analysis_plan_sha256']}`, registered before any prediction existed.", "",
         "## Primary comparison: arm B (hybrid RAG) vs arm A (zero-shot)", "",
         "| arm | correct / n | % | Wilson 95% |", "|---|---|---|---|",
         f"| A zero-shot | {a['correct']} / {a['total']} | {a['percent']} | {a['wilson95'][0]:.4f}–{a['wilson95'][1]:.4f} |",
         f"| B hybrid RAG | {b['correct']} / {b['total']} | {b['percent']} | {b['wilson95'][0]:.4f}–{b['wilson95'][1]:.4f} |", "",
         f"- Difference B − A: {d['correct']:+d} items, {d['percentage_points']:+.2f} percentage points.",
         f"- Paired bootstrap 95% interval of the difference: {pa['bootstrap95_difference_b_minus_a']} "
         f"({pa['bootstrap']['resamples']:,} resamples, seed {pa['bootstrap']['seed']}).",
         f"- RAG fixes a baseline error on {t['rag_fixes_baseline_error']} items; RAG introduces an error on {t['rag_introduces_error']} items.",
         f"- Exact two-sided McNemar p = {pa['mcnemar_exact_p']:.4g}.",
         f"- **Conclusion:** {pa['conclusion']}.", "",
         "| | B correct | B not correct |", "|---|---|---|",
         f"| A correct | {t['both_correct']} | {t['rag_introduces_error']} |",
         f"| A not correct | {t['rag_fixes_baseline_error']} | {t['both_wrong']} |", "",
         "## Outcome accounting (each arm sums to 892)", "", "| arm | " + " | ".join(S.CATEGORIES) + " |", "|---|" + "---|" * len(S.CATEGORIES)]
    for arm, s in summaries.items():
        L.append(f"| {arm} | " + " | ".join(str(s["accounting"][c]) for c in S.CATEGORIES) + " |")
    r = sec["retrieval"]
    L += ["", "## Secondary diagnostics (EXPLORATORY, descriptive only)", "",
          f"- Retrieval coverage: examples per item {r['examples_per_item']}; empty-context items {r['empty_context_items']}; "
          f"context tokens {r['context_tokens']['min']}–{r['context_tokens']['max']}.",
          f"- Arm A error categories resolved by RAG: {sec['category_transitions']['arm_a_error_categories_resolved_by_rag']}.",
          f"- Error categories introduced by RAG: {sec['category_transitions']['error_categories_introduced_by_rag']}.",
          "- By hardness: " + "; ".join(f"{g}: A {v['arms'][ARM_A]['correct']}/{v['n']}, B {v['arms'][ARM_B]['correct']}/{v['n']}"
                                         for g, v in sec["breakdowns"]["hardness"].items()) + ".",
          "- Per-record differences with retrieved ids, ranks and scores: `paired_comparison.jsonl`; database and query-pattern "
          "breakdowns, the category transition table and sensitivity analyses S2 and S3: `analysis.json`.", "",
          "## Historical context (not MQL-0008 results)", "",
          "| Experiment | Corpus | Accuracy |", "|---|---|---:|",
          "| MQL-0006 zero-shot | No retrieval | 47.20% (421/892) |",
          "| MQL-0007 hybrid RAG | Corpus v1, 3,589 records | 48.99% (437/892) |",
          f"| MQL-0008 zero-shot | No retrieval | {a['percent']}% ({a['correct']}/{a['total']}) |",
          f"| MQL-0008 hybrid RAG | Approved corpus v2.1, 3,445 records | {b['percent']}% ({b['correct']}/{b['total']}) |", "",
          "MQL-0006 and MQL-0007 are earlier runs and are shown for context only. MQL-0007 used corpus v1, not the approved corpus.", ""]
    if st["status"] != "complete":
        L.insert(3, "**Incomplete: some items are unresolved infrastructure or generation failures. Figures are not final. "
                    "Re-run `score` with the same `--run-name` to retry Atlas outages.**")
        L.insert(4, "")
    return "\n".join(L)


def _write_outputs(score_dir, st, analysis, rows, summaries):
    C.write_json(score_dir / "analysis.json", analysis)
    C.write_jsonl(score_dir / "paired_comparison.jsonl", rows)
    C.write_json(score_dir / "RUN_MANIFEST.json", st)
    (score_dir / "results_report.md").write_text(render_report(st, analysis, summaries), encoding="utf-8", newline="\n")
    names = sorted(p.relative_to(score_dir).as_posix() for p in score_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (score_dir / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(score_dir / n)}  {n}\n" for n in names), encoding="utf-8", newline="\n")


def _append_log(run_dir, event):
    f = Path(run_dir) / "run_log.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a", encoding="utf-8", newline="\n") as h:
        h.write(json.dumps({"at": C.now(), **event}, ensure_ascii=False, default=str) + "\n")


def default_run_name():
    return "run_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def cmd_verify(zips, dest_root=DEST_ROOT, pkg=PKG, bundle=BUNDLE, allow_env_drift=False, expected_manifest_sha=EXPECTED_PACKAGE_MANIFEST_SHA256):
    import verify_kaggle_spider_dev_1034 as V
    dest_root = guard_dest(dest_root)
    exp = load_expected(pkg, bundle, expected_manifest_sha)
    ts = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    staged = [V.safe_extract(z, dest_root / ".staging" / ts / f"z{i}", STATUS_FILE) for i, z in enumerate(zips)]
    return verify_predictions(staged, exp, allow_env_drift), exp


def cmd_score(zips, run_name=None, bundle=BUNDLE, pkg=PKG, dest_root=DEST_ROOT, client_factory=None, full_parity=False,
              allow_env_drift=False, log=print, expected_manifest_sha=EXPECTED_PACKAGE_MANIFEST_SHA256):
    dest_root = guard_dest(dest_root)
    plan_doc = S.load_registered_plan()                    # the plan must be registered before anything is scored
    bman = C.verify_bundle(bundle)
    ver, exp = cmd_verify(zips, dest_root, pkg, bundle, allow_env_drift, expected_manifest_sha)
    run_dir = dest_root / "scoring_runs" / (run_name or default_run_name())
    C.write_json(run_dir / "input_validation_report.json", ver)
    if not ver["ok"]:
        _append_log(run_dir, {"event": "input_validation_failed", "problems": ver["problems"][:20]})
        raise ScoringError(f"predictions failed validation: {ver['problems'][:5]}")
    pred_dir = run_dir / "predictions"
    for arm, src in ver["arm_source"].items():
        dst = pred_dir / arm
        if dst.exists():
            need(C.sha256_file(dst / "predictions.jsonl") == ver["predictions_sha256"][arm],
                 f"{dst} already holds DIFFERENT predictions; use a new --run-name")
        else:
            shutil.copytree(src, dst)
    K, _ = C.bundle_modules(bundle)
    refs, mapping = C.load_reference(bundle), C.load_mapping(bundle)
    need([r["id"] for r in refs] == exp["ids"], "evaluation references are not the 892 ids in order")
    if client_factory is None:
        import local_atlas_scoring as L
        client_factory = L.default_client_factory
    client, _redact = client_factory()
    score_dir = run_dir / "scoring"
    resumed = (score_dir / "arms").exists()
    _append_log(run_dir, {"event": "scoring_session_started", "resumed_existing_run": resumed, "zips": [Path(z).name for z in zips]})
    try:
        atl = C.atlas_preflight(client, mapping, full_parity=full_parity, log=log)   # fails closed unless the user is read-only
        C.write_json(score_dir / "preflight/atlas_check.json", atl)
        mdb = client[mapping["atlas_db"]]
        evals_by, summaries = {}, {}
        for arm in ARMS:                                    # same evaluator, references, limits for both arms
            preds = C.read_jsonl(pred_dir / arm / "predictions.jsonl")
            fp = eval_fingerprint(bman, ver["predictions_sha256"][arm])
            ev, sm = score_arm(arm, preds, exp["ids"], refs, mapping, mdb, K, score_dir, fp, log)
            evals_by[arm], summaries[arm] = ev, sm
            _append_log(run_dir, {"event": "arm_scored", "arm": arm, "evaluated_this_session": sm["evaluated_this_session"],
                                  "final_before_this_session": sm["final_before_this_session"], "accounting": sm["accounting"]})
            log(f"{arm}: {sm['correct']}/{sm['n_expected']} correct; {sm['accounting']}")
    except Exception as e:  # noqa: BLE001
        _append_log(run_dir, {"event": "scoring_session_failed", "error": f"{type(e).__name__}: {str(e)[:300]}"})
        raise
    finally:
        client.close()
    st = _finish(run_dir, exp, bundle, pkg, evals_by, summaries, ver, atl, plan_doc)
    _append_log(run_dir, {"event": "scoring_session_finished", "status": st["status"]})
    return st, json.loads((score_dir / "analysis.json").read_text(encoding="utf-8")), run_dir


def _finish(run_dir, exp, bundle, pkg, evals_by, summaries, ver, atl, plan_doc):
    score_dir = Path(run_dir) / "scoring"
    refs = C.load_reference(bundle)
    meta = {r["id"]: {"hardness": r["hardness"], "db_id": r["db_id"], "reference_status": r["reference_status"]} for r in refs}
    nd = sorted({x["eval_id"] for x in exp["cfg"]["retrieval_exclusions"]})
    plan_rows = C.read_jsonl(Path(run_dir) / "predictions" / ARM_B / "retrieval_log.jsonl")
    analysis, rows = S.analyze(evals_by, exp["ids"], meta, query_patterns(bundle, pkg), nd, plan_rows)
    complete = all(s["status"] == "complete" for s in summaries.values())
    st = {"experiment": EXPERIMENT, "stage": "scoring", "status": "complete" if complete else "partial", "written_at": C.now(),
          "n_expected": len(exp["ids"]), "arms": {a: s["accounting"] for a, s in summaries.items()},
          "correct": {a: s["correct"] for a, s in summaries.items()},
          "metric": "cross-engine execution accuracy (Atlas MQL result vs SQLite gold-SQL result), correct / 892; NOT native MQL exact match",
          "limits": {"exec_timeout_ms": C.EXEC_TIMEOUT_MS, "max_result_rows": C.MAX_RESULT_ROWS},
          "atlas": {"database": atl.get("atlas_db"), "server_version": atl.get("server_version"), "checked_at": atl.get("checked_at"),
                    "authenticated_users": atl.get("authenticated_users"), "offending_write_or_admin_actions": atl.get("offending_actions"),
                    "read_only_verified": atl.get("offending_actions") == []},
          "inputs": {"predictions_sha256": ver["predictions_sha256"], **ver["frozen"]},
          "analysis_plan_sha256": S.plan_sha(), "analysis_plan_registered_at": plan_doc.get("written_at"),
          "evaluator": "scripts/nl2mql/dev1034_core.py (unchanged MQL-0006 evaluator)",
          "code_sha256": {"scripts/nl2mql/dev1034_core.py": C.sha256_file(ROOT / "scripts/nl2mql/dev1034_core.py"),
                          "scripts/nl2mql_mql0008/mql0008_scoring.py": C.sha256_file(Path(__file__)),
                          "scripts/nl2mql_mql0008/mql0008_stats.py": C.sha256_file(HERE / "mql0008_stats.py"),
                          "scripts/nl2mql/mql0007_stats.py": C.sha256_file(ROOT / "scripts/nl2mql/mql0007_stats.py")}}
    _write_outputs(score_dir, st, analysis, rows, summaries)
    return st


def cmd_analyze(run_dir, bundle=BUNDLE, pkg=PKG, expected_manifest_sha=EXPECTED_PACKAGE_MANIFEST_SHA256):
    """Recompute analysis.json, paired_comparison.jsonl and the report from the saved per-record results. No Atlas."""
    run_dir = guard_dest(run_dir)
    exp = load_expected(pkg, bundle, expected_manifest_sha)
    plan_doc = S.load_registered_plan()
    score_dir = run_dir / "scoring"
    old = json.loads((score_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    ver = json.loads((run_dir / "input_validation_report.json").read_text(encoding="utf-8"))
    evals_by, summaries = {}, {}
    for arm in ARMS:
        evals_by[arm] = C.read_jsonl(score_dir / "arms" / arm / "evaluation_results.jsonl")
        preds = {p["id"]: p for p in C.read_jsonl(run_dir / "predictions" / arm / "predictions.jsonl")}
        need(C.sha256_file(run_dir / "predictions" / arm / "predictions.jsonl") == ver["predictions_sha256"][arm], f"{arm}: saved predictions changed")
        bound = all(e["prediction_sha256"] == C.sha256_text(C.canon({k: preds[e["id"]].get(k) for k in ("id", "raw_text", "status")}))
                    for e in evals_by[arm])
        need(bound, f"{arm}: a scored record is not bound to its saved prediction")
        summaries[arm] = summarize(arm, evals_by[arm], exp["ids"])
    atl = {"atlas_db": old["atlas"]["database"], "server_version": old["atlas"]["server_version"], "checked_at": old["atlas"]["checked_at"],
           "authenticated_users": old["atlas"]["authenticated_users"], "offending_actions": old["atlas"]["offending_write_or_admin_actions"]}
    st = _finish(run_dir, exp, bundle, pkg, evals_by, summaries, ver, atl, plan_doc)
    _append_log(run_dir, {"event": "analysis_recomputed_offline", "status": st["status"]})
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "score"):
        p = sub.add_parser(name)
        p.add_argument("--predictions-zip", action="append", required=True)
        p.add_argument("--allow-env-drift", action="store_true", help="accept different library/GPU versions across the arms (recorded)")
        if name == "score":
            p.add_argument("--run-name", default=None, help="default: run_<UTC timestamp>; pass an existing name to resume")
            p.add_argument("--full-parity", action="store_true")
    p = sub.add_parser("analyze")
    p.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        rep, _ = cmd_verify(a.predictions_zip, allow_env_drift=a.allow_env_drift)
        print(json.dumps({k: rep[k] for k in ("ok", "problems", "environment_drift")}, indent=1))
        return 0 if rep["ok"] else 1
    if a.cmd == "analyze":
        st = cmd_analyze(a.run_dir)
        print(json.dumps({"status": st["status"], "correct": st["correct"]}, indent=1))
        return 0
    st, analysis, run_dir = cmd_score(a.predictions_zip, a.run_name, allow_env_drift=a.allow_env_drift, full_parity=a.full_parity)
    print(json.dumps({"status": st["status"], "arms": st["arms"], "conclusion": analysis["primary_analysis"]["conclusion"]}, indent=1))
    print("run directory:", run_dir)
    print("report:", run_dir / "scoring/results_report.md")
    return 0 if st["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
