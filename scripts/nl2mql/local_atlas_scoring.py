"""Local (Windows) Atlas preflight and scoring for MQL-0006. Kaggle only generates; all execution happens here.

Read-only throughout:
  * Atlas is reached with MONGODB_URI from the process environment or .env (the read-only `codegen_eval_readonly`
    account). The URI is never printed or written; .env is never modified. The preflight FAILS if the account holds
    any write or admin privilege. No Atlas data, index or permission is ever created, changed or dropped.
  * SQLite files are opened in mode=ro from the bundle.
  * The held-out test databases are not in the bundle's mapping, so they are never read or queried.

Commands
  preflight  run BEFORE any Kaggle GPU time.
             - bundle integrity;
             - every gold SQL re-executed against the bundled reference;
             - Atlas: ping, server version, read-only privileges, all mapped collections present, exact counts, and
               (default) the full type-tagged row digest of every collection against the bundled SQLite tables.
             Writes outputs/nl2mql_spider_dev_1034_model_comparison/preflight/atlas_preflight_<ts>.json.
  score      score a downloaded Kaggle predictions bundle:
             - verify it (verify_kaggle_spider_dev_1034.verify_predictions);
             - import it without overwriting (v1/predictions/);
             - re-check gold SQL and Atlas (privileges, coverage, counts; --full-parity for the digest too);
             - per model: safety gate -> Atlas execution (maxTimeMS, bounded cursor) -> comparator, with resumable
               checkpoints;
             - write evaluation_results.jsonl, prediction_rows.jsonl.gz, scoring_config.json and
               baseline_summary.json for the model, then RUN_STATUS.json, results_report.md and SHA256SUMS.txt in
               v1/scoring/ (MQL-0006 is single-model: Qwen only; no paired comparison);
             - finally run the offline scoring verifier.
             A partial predictions bundle is scored only with --allow-partial, under v1/partial/<ts>/.
  verify     offline verification of v1/ (predictions + scoring); no Atlas.

Usage (PowerShell, repo root):
  python scripts/nl2mql/local_atlas_scoring.py preflight
  python scripts/nl2mql/local_atlas_scoring.py score --predictions-zip "$HOME\\Downloads\\nl2mql_spider_dev_predictions.zip"
  python scripts/nl2mql/local_atlas_scoring.py verify
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
sys.path.insert(0, str(ROOT / "scripts"))
import dev1034_core as C  # noqa: E402
import verify_kaggle_spider_dev_1034 as V  # noqa: E402

DEFAULT_BUNDLE = V.DEFAULT_BUNDLE
DEST_ROOT = V.DEST_ROOT
LABELS = tuple(V.EXPECTED_MODELS)
SERVER_SELECTION_TIMEOUT_MS = 20000
SOCKET_TIMEOUT_MS = 120000


class ScoringError(RuntimeError):
    pass


def default_client_factory():
    """MongoClient for MONGODB_URI (environment, else .env) with the project TLS rule; the URI is never shown."""
    import mongo_conn
    from pymongo import MongoClient
    uri = mongo_conn.mongo_uri(ROOT / ".env")
    try:
        return MongoClient(uri, serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS, socketTimeoutMS=SOCKET_TIMEOUT_MS,
                           appname="codegen-mql0006-local-scoring", **mongo_conn.tls_kwargs(uri)), \
            (lambda s: mongo_conn._redact(s, uri))
    except Exception as e:  # noqa: BLE001
        raise ScoringError(f"cannot create the Atlas client ({type(e).__name__}): "
                           f"{mongo_conn._redact(str(e), uri)[:300]}") from None


def atlas_check(mapping, full_parity, client_factory, log=print):
    client, redact = client_factory()
    try:
        return C.atlas_preflight(client, mapping, full_parity=full_parity, log=log), client
    except C.PreflightError:
        client.close()
        raise
    except Exception as e:  # noqa: BLE001
        client.close()
        raise C.PreflightError(f"Atlas preflight failed ({type(e).__name__}): {redact(str(e))[:300]}") from None


def reference_recheck(bundle):
    K, _ = C.bundle_modules(bundle)
    recs, refs = C.load_dev(bundle), C.load_reference(bundle)
    exp = json.loads((Path(bundle) / "data/reference_expected.json").read_text(encoding="utf-8"))["items"]
    C.need([C.reference_digest(r) for r in refs] == exp, "bundled reference does not match its digests")
    rc = C.recheck_reference(recs, refs, K, Path(bundle) / "database")
    return rc


def cmd_preflight(bundle=DEFAULT_BUNDLE, full_parity=True, dest_root=DEST_ROOT, client_factory=None, log=print):
    bundle, log = Path(bundle), log or (lambda *_: None)
    man = C.verify_bundle(bundle)
    rep = {"experiment": C.EXPERIMENT_ID, "kind": "local_atlas_preflight", "started_at": C.now(),
           "bundle": man["bundle"], "bundle_manifest_sha256": C.sha256_file(bundle / "bundle_manifest.json"),
           "evaluation_set": {k: man["evaluation_set"][k] for k in ("records", "databases", "collections")}}
    out = Path(dest_root) / "preflight" / f"atlas_preflight_{time.strftime('%Y%m%dT%H%M%S')}.json"
    try:
        rc = reference_recheck(bundle)
        rep["reference_recheck"] = {k: rc[k] for k in ("checked", "sqlite_version")} | \
            {"problems": rc["problems"], "warnings": len(rc["warnings"])}
        C.need(not rc["problems"], f"gold SQL re-execution differs from the bundled reference: {rc['problems'][:3]}")
        atl, client = atlas_check(C.load_mapping(bundle), full_parity, client_factory or default_client_factory, log)
        client.close()
        rep["atlas"], rep["ok"] = atl, True
    except (C.PreflightError, ScoringError, RuntimeError) as e:
        rep["ok"], rep["error"] = False, str(e)
        C.write_json(out, rep)
        raise
    rep["finished_at"] = C.now()
    C.write_json(out, rep)
    return rep, out


def import_predictions(staged, dest):
    """Copy a verified predictions dir to `dest`; identical content is reused, different content is refused."""
    dest = Path(dest)
    src = V.tree_sums(staged)
    if dest.exists() and any(dest.iterdir()):
        if V.tree_sums(dest) == src:
            return "already imported (identical hashes)"
        raise ScoringError(f"{dest} already holds DIFFERENT predictions; not overwriting. Score the new bundle into a "
                           "new run directory with --run-name.")
    tmp = dest.with_name(dest.name + ".importing")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(staged, tmp)
    if V.tree_sums(tmp) != src:
        shutil.rmtree(tmp)
        raise ScoringError("staged predictions copy differs from the verified files")
    if dest.exists():
        dest.rmdir()
    tmp.rename(dest)
    return "imported"


def render_single_model_report(st, summaries):
    """The shared report renderer (dev1034_core, unchanged in the bundle) with single-model wording."""
    note = {"note": "MQL-0006 is a single-model experiment (Qwen2.5-Coder-3B-Instruct only); the frontier-model "
                    "comparison is deferred to a later experiment"}
    text = C.render_report(st, summaries, note)
    text = text.replace("Qwen2.5-Coder-3B vs Groq-hosted open-weight gpt-oss-120b, NL→MQL on Spider dev",
                        "zero-shot Qwen2.5-Coder-3B-Instruct, NL→MQL on Spider dev")
    text = text.replace("by the Kaggle notebook", "by local scoring (scripts/nl2mql/local_atlas_scoring.py)")
    return text.replace("## Paired comparison", "## Model comparison")


def score_dir_paths(run_dir):
    return Path(run_dir) / "predictions", Path(run_dir) / "scoring"


def cmd_score(predictions_zip, bundle=DEFAULT_BUNDLE, dest_root=DEST_ROOT, run_name="v1", allow_partial=False,
              full_parity=False, client_factory=None, log=print):
    bundle, dest_root, log = Path(bundle), Path(dest_root), log or (lambda *_: None)
    man = C.verify_bundle(bundle)
    man_sha = C.sha256_file(bundle / "bundle_manifest.json")
    # 1. stage + verify the downloaded predictions (no Atlas yet)
    ts = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]   # unique even within one second
    staged = V.safe_extract(predictions_zip, dest_root / ".staging" / ts, "PREDICTIONS_STATUS.json")
    prep = V.verify_predictions(staged, bundle)
    C.write_json(staged.parent / "predictions.verification.json", prep)
    if not prep["all_checks_passed"]:
        raise ScoringError(f"predictions bundle failed verification: {prep['problems'][:5]}")
    if not prep["complete"] and not allow_partial:
        raise ScoringError("predictions bundle is PARTIAL; resume it on Kaggle, or pass --allow-partial to score it "
                           "for the record under v1/partial/")
    run_dir = dest_root / run_name if prep["complete"] else dest_root / run_name / "partial" / ts
    pred_dir, score_dir = score_dir_paths(run_dir)
    how = import_predictions(staged, pred_dir)
    C.write_json(run_dir / "predictions.verification.json", prep)
    log(f"predictions {how}: {pred_dir}")
    rs_path = score_dir / "RUN_STATUS.json"
    if rs_path.exists() and json.loads(rs_path.read_text(encoding="utf-8")).get("status") == "complete":
        raise ScoringError(f"{score_dir} is already completely scored; not re-scoring over it")
    # 2. reference + Atlas checks
    rc = reference_recheck(bundle)
    C.write_json(score_dir / "preflight/reference_recheck.json", rc)
    C.need(not rc["problems"], f"gold SQL re-execution differs from the bundled reference: {rc['problems'][:3]}")
    mapping = C.load_mapping(bundle)
    atl, client = atlas_check(mapping, full_parity, client_factory or default_client_factory, log)
    C.write_json(score_dir / "preflight/atlas_check.json", atl)
    K, _ = C.bundle_modules(bundle)
    refs, recs = C.load_reference(bundle), C.load_dev(bundle)
    ids = [r["id"] for r in recs]
    prior = json.loads((bundle / "data/prior_audit.json").read_text(encoding="utf-8"))
    eval_fp = C.fingerprint({"core": man["code_sha256"], "mapping": man["files"]["data/atlas_mapping.json"]["sha256"],
                             "reference": man["files"]["data/reference.jsonl.gz"]["sha256"],
                             "timeout_ms": C.EXEC_TIMEOUT_MS, "max_rows": C.MAX_RESULT_ROWS})
    summaries, evals_by, problems = {}, {}, []
    try:
        mdb = client[mapping["atlas_db"]]
        for label in LABELS:
            md = pred_dir / "models" / label
            if not (md / "predictions.jsonl").is_file():
                problems.append(f"{label}: no predictions")
                continue
            t0 = time.time()
            preds = C.read_jsonl(md / "predictions.jsonl")
            d = score_dir / "models" / label
            eck = C.Checkpoint(d / "checkpoints/evaluation.partial.jsonl", ids, eval_fp, C.is_eval_retry)
            rck = C.Checkpoint(d / "checkpoints/rows.partial.jsonl", ids, eval_fp)
            n = C.run_evaluation(preds, refs, mapping, mdb, K, eck, rck, C.EXEC_TIMEOUT_MS,
                                 progress=lambda n, t: log(f"  {label}: {n}/{t}"))
            evals = [eck.records[p["id"]] for p in preds if p["id"] in eck.records]
            rows_ids = {e["id"] for e in evals}
            C.write_jsonl(d / "evaluation_results.jsonl", evals)
            C.write_jsonl(d / "prediction_rows.jsonl.gz", [r for r in rck.ordered() if r["id"] in rows_ids])
            sc = {"evaluation_fingerprint": eval_fp, "evaluation_wall_s": round(time.time() - t0, 1),
                  "evaluated_this_session": n, "scored_at": C.now(), "atlas_db": mapping["atlas_db"],
                  "atlas_server_version": atl.get("server_version"), "exec_timeout_ms": C.EXEC_TIMEOUT_MS,
                  "max_result_rows": C.MAX_RESULT_ROWS, "scorer": "scripts/nl2mql/local_atlas_scoring.py",
                  "scorer_sha256": C.sha256_file(Path(__file__)),
                  "evaluation": "local Windows -> Atlas (read-only); comparator spider1_kit.exec_match"}
            C.write_json(d / "scoring_config.json", sc)
            cfg = {**json.loads((md / "run_config.json").read_text(encoding="utf-8")), **sc}
            s = V.summarize_model(label, cfg, preds, evals, refs, ids)
            C.write_json(d / "baseline_summary.json", s)
            summaries[label], evals_by[label] = s, evals
            problems += [f"{label}: {p}" for p in s["problems"]]
            log(f"{label}: {s['correct']}/{s['n_expected']} correct; outcomes {s['outcomes']}")
    finally:
        client.close()
    if len(summaries) != len(LABELS):
        problems.append(f"{len(LABELS) - len(summaries)} expected model(s) not scored")
    status = "complete" if not problems and prep["complete"] and \
        all(s["status"] == "complete" for s in summaries.values()) else "partial"
    st = {"experiment": C.EXPERIMENT_ID, "stage": "scoring", "status": status, "written_at": C.now(),
          "problems": problems, "bundle": man["bundle"], "bundle_manifest_sha256": man_sha,
          "n_expected": len(ids), "predictions_status_sha256": C.sha256_file(pred_dir / "PREDICTIONS_STATUS.json"),
          "predictions_sha256": {l: C.sha256_file(pred_dir / "models" / l / "predictions.jsonl") for l in LABELS
                                 if (pred_dir / "models" / l / "predictions.jsonl").is_file()},
          "models": {l: {"status": s["status"], "correct": s["correct"], "outcomes": s["outcomes"]}
                     for l, s in summaries.items()},
          "scope": "single model (Qwen2.5-Coder-3B-Instruct); frontier-model comparison deferred",
          "reference_status_counts": man["counts"]["reference_status"],
          "metric": "cross-engine execution accuracy (Atlas MQL result vs SQLite gold-SQL result); "
                    "NOT native MQL exact match"}
    C.write_json(score_dir / "RUN_STATUS.json", st)
    report = render_single_model_report(st, summaries)
    (score_dir / "results_report.md").write_text(report, encoding="utf-8")
    names = sorted(p.relative_to(score_dir).as_posix() for p in score_dir.rglob("*")
                   if p.is_file() and p.name != "SHA256SUMS.txt")
    (score_dir / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(score_dir / n)}  {n}\n" for n in names),
                                              encoding="utf-8")
    ver = V.verify_scoring(run_dir, bundle)
    C.write_json(run_dir / "scoring.verification.json", ver)
    return st, ver, run_dir


def main(argv=None, client_factory=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    p.add_argument("--quick", action="store_true", help="skip the full row-digest parity (counts/privileges only)")
    s = sub.add_parser("score")
    s.add_argument("--predictions-zip", required=True)
    s.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    s.add_argument("--run-name", default="v1")
    s.add_argument("--allow-partial", action="store_true")
    s.add_argument("--full-parity", action="store_true", help="also re-run the full row-digest parity before scoring")
    v = sub.add_parser("verify")
    v.add_argument("--run-dir", default=str(DEST_ROOT / "v1"))
    v.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    for x in (p, s):
        x.add_argument("--dest-root", default=str(DEST_ROOT))
    a = ap.parse_args(argv)
    try:
        if a.cmd == "preflight":
            rep, out = cmd_preflight(a.bundle, not a.quick, a.dest_root, client_factory)
            at = rep["atlas"]
            print(f"PREFLIGHT OK: Atlas {at['server_version']}, user {at['authenticated_users']}, "
                  f"{at['collections_found']}/{rep['evaluation_set']['collections']} collections, counts exact, "
                  f"full parity {'checked' if at['full_parity'] else 'skipped (--quick)'}; "
                  f"gold SQL re-check {rep['reference_recheck']['checked']} items, 0 problems")
            print(f"report: {out}")
            return 0
        if a.cmd == "score":
            st, ver, run_dir = cmd_score(a.predictions_zip, a.bundle, a.dest_root, a.run_name, a.allow_partial,
                                         a.full_parity, client_factory)
            print(f"SCORING {st['status'].upper()}: {run_dir / 'scoring'}")
            for l, m in st["models"].items():
                print(f"  {l}: {m['correct']}/{st['n_expected']} ({m['status']})")
            print(f"verification: all checks passed = {ver['all_checks_passed']}, complete = {ver['complete']}")
            return 0 if ver["all_checks_passed"] else 1
        rep = V.verify_scoring(a.run_dir, a.bundle)
        V._print(rep)
        return 0 if rep["all_checks_passed"] else 1
    except (ScoringError, C.PreflightError, V.VerifyError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
