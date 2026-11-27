"""MQL-0009 Windows-side scoring adapter (Kaggle export -> Atlas, READ-ONLY). Kaggle only generates.

A thin adapter around the UNCHANGED MQL-0006 evaluator (scripts/nl2mql/dev1034_core.py: compute_reference,
evaluate_prediction via run_evaluation, atlas_preflight, Checkpoint) and the read-only Atlas client of
scripts/nl2mql/local_atlas_scoring.py. Result equivalence, the safety gate and the outcome classes (correct, incorrect,
invalid, timeout, failed/model = execution error, failed/infrastructure = generation or Atlas failure, not_comparable)
are not re-implemented.

Commands
  readiness                                   offline: is scoring possible? (package pin, contract, gold records, SQLite,
                                              Atlas provisioning status from the last recorded inventory). No network.
  verify   --predictions-zip Z [...]          offline: validate the Kaggle export(s) against the frozen package
  score    --predictions-zip Z [...] [--run-name N] [--atlas-db DB]
                                              verify -> gold reference from local SQLite -> read-only Atlas preflight with
                                              MANDATORY full parity against the provisioning contract -> score all 8 runs
                                              on the same items and snapshot -> paired analysis -> report (resumable)
  analyze  --run-dir D                        recompute analysis + report from saved per-record results; no Atlas

Gold SQL, gold rows and query-family labels exist only here (local), never in the Kaggle package.
"""
import argparse
import gzip
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PKG = ROOT / "kaggle/nl2mql_mql0009_retrieval_control"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(1, str(ROOT / "scripts/nl2mql"))
sys.path.insert(2, str(ROOT / "scripts"))
sys.path.insert(3, str(HERE))
import mql0009_core as K9        # noqa: E402  (the package core, loaded from the frozen package)
import dev1034_core as D         # noqa: E402  (unchanged MQL-0006 evaluator)
import mql0007_stats as S7       # noqa: E402
import mql0009_stats as S        # noqa: E402
import patterns as PT            # noqa: E402  (package copy, byte-identical to MQL-0008)

DEST_ROOT = ROOT / "outputs/nl2mql_mql0009_retrieval_control"
CONTRACT = DEST_ROOT / "scoring_contract/atlas_mapping.json"
LEAKAGE_LOCAL = DEST_ROOT / "preflight/leakage_audit.json"
S1_VAL = ROOT / "data/processed/spider_sql_v1/val_internal.jsonl"
V3_VAL = ROOT / "data/processed/spider_pool_v3/val_internal.jsonl"
DB_DIR = ROOT / "data/raw/spider1/spider_data/database"
SPIDER1_CODE = ROOT / "scripts/spider1"
DEV_ATLAS_MANIFEST = ROOT / "data/interim/spider1/atlas_collections.json"
PROTECTED_ROOTS = (ROOT / "outputs/nl2mql_spider_dev_1034_model_comparison", ROOT / "outputs/nl2mql_mql0007_rag_v1",
                   ROOT / "outputs/nl2mql_mql0008_approved_corpus_v2_1", ROOT / "kaggle")
# frozen package identity (set from the final build; the adapter refuses any other package)
EXPECTED_PACKAGE_MANIFEST_SHA256 = "2a4f22e0eb526540e70f62f561d0314b1b2faa5c12917196a953d9f98aecface"
ADAPTER_VERSION = "mql0009-scoring-v1"
STATUS = K9.STATUS_FILE


class ScoringError(RuntimeError):
    pass


def need(cond, msg):
    if not cond:
        raise ScoringError(msg)


def guard_dest(dest):
    d = Path(dest).resolve()
    for root in PROTECTED_ROOTS:
        need(root.resolve() not in (d, *d.parents), f"refusing to write under {root}: {d}")
    return d


# ------------------------------------------------------------------ frozen expectations
def load_expected(pkg=PKG, expected_manifest_sha=EXPECTED_PACKAGE_MANIFEST_SHA256):
    P = K9.load_package(pkg)
    need(P["manifest_sha256"] == expected_manifest_sha, f"package manifest {P['manifest_sha256']} != frozen {expected_manifest_sha}")
    cfg = P["cfg"]
    plans = {r["name"]: K9.read_jsonl(Path(pkg) / f"data/plans/plan_{r['name']}.jsonl") for r in P["runs"]}
    for n, pl in plans.items():
        need(K9.plan_sha(pl) == cfg["plans_sha256"][n] and [p["id"] for p in pl] == P["ids"], f"{n}: frozen plan check")
    return {"P": P, "cfg": cfg, "ids": P["ids"], "plans": plans, "runs": P["runs"],
            "prompt_hashes": {n: {p["id"]: p["prompt_sha256"] for p in pl} for n, pl in plans.items()}}


def gold_records(ids):
    """LOCAL gold: spider-sql-v1 val_internal records for the package ids, in package order (SQL equal to v3)."""
    s1 = {r["id"]: r for r in D.read_jsonl(S1_VAL)}
    v3 = {r["id"]: r for r in D.read_jsonl(V3_VAL)}
    out = []
    for i in ids:
        need(i in s1 and i in v3 and s1[i]["sql"] == v3[i]["sql"], f"{i}: gold record missing or inconsistent")
        out.append(s1[i])
    return out


def load_contract(atlas_db=None, pkg=PKG):
    """Provisioning contract (derived offline at build time) + consistency with the package prompt context and the
    SQLite files. `atlas_db` overrides the proposed target database name (recorded)."""
    m = json.loads(CONTRACT.read_text(encoding="utf-8"))
    ctx = json.loads((Path(pkg) / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
    need(set(m["collections"]) == set(ctx), "contract collections != package prompt context")
    for name, c in m["collections"].items():
        need(c["fields"] == list(ctx[name]["fields"]) and c["documents"] == ctx[name]["documents"]
             and c["observed_bson_types"] == ctx[name]["fields"], f"{name}: contract differs from prompt context")
    for db, h in m["sqlite_sha256"].items():
        need(D.sha256_file(DB_DIR / db / f"{db}.sqlite") == h, f"{db}: SQLite file changed since the package was built")
    if atlas_db:
        m = {**m, "atlas_db": atlas_db}
    return m


# ------------------------------------------------------------------ readiness (offline)
def readiness(pkg=PKG, expected_manifest_sha=EXPECTED_PACKAGE_MANIFEST_SHA256, write=True):
    rep = {"experiment": K9.EXPERIMENT, "checked_at": D.now(), "mode": "offline (no network, no database connection)",
           "checks": {}, "blockers": []}

    def chk(name, ok, detail):
        rep["checks"][name] = {"ok": bool(ok), "detail": detail}
        return ok

    try:
        exp = load_expected(pkg, expected_manifest_sha)
        chk("package_pinned_and_verified", True, exp["P"]["manifest_sha256"])
    except Exception as e:  # noqa: BLE001
        chk("package_pinned_and_verified", False, str(e)[:300])
        rep["blockers"].append("package does not verify against the frozen pin")
        exp = None
    if exp:
        try:
            recs = gold_records(exp["ids"])
            chk("gold_records_local", True, f"{len(recs)} records, all with spider-sql-v1 execution hashes")
        except Exception as e:  # noqa: BLE001
            chk("gold_records_local", False, str(e)[:300])
            rep["blockers"].append("local gold records unavailable")
    try:
        m = load_contract(pkg=pkg)
        docs = sum(c["documents"] for c in m["collections"].values())
        chk("provisioning_contract", True, f"{len(m['db_ids'])} databases, {len(m['collections'])} collections, {docs} documents; "
            f"SQLite hashes unchanged")
        size = sum((DB_DIR / d / f"{d}.sqlite").stat().st_size for d in m["db_ids"])
        rep["storage"] = {"sqlite_bytes_16_dbs": size, "documents": docs, "atlas_recorded_data_mb": 54.4, "atlas_tier_limit_mb": 512,
                          "note": "the held-out data is about 1.4 MB as SQLite; BSON size will be a few MB, well inside the tier"}
        dev = json.loads(DEV_ATLAS_MANIFEST.read_text(encoding="utf-8"))
        present = sorted(set(m["collections"]) & set(dev))
        on_atlas_dbs = sorted({v["db_id"] for v in dev.values()})
        prov = []
        for f in sorted((DEST_ROOT / "provisioning").glob("provision_*.json")):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                continue
            v = r.get("verification") or {}
            if (r.get("status") == "complete" and r.get("target_db") == m["atlas_db"]
                    and v.get("collections") == len(m["collections"]) and v.get("documents") == docs):
                prov.append(f.name)
        if prov:
            chk("atlas_provisioning_report", True, f"provision_heldout_atlas.py reported a complete, digest-verified provisioning "
                f"of {m['atlas_db']} ({prov[-1]}); `score` re-verifies live with full parity")
            present = list(m["collections"])
        provisioned = chk("atlas_collections_recorded", len(present) == len(m["collections"]),
                          f"last recorded Atlas inventory (data/interim/spider1/atlas_collections.json) lists {len(dev)} collections "
                          f"for {len(on_atlas_dbs)} Spider-dev DBs; {len(present)} of the {len(m['collections'])} required held-out "
                          "collections are recorded there")
        if not provisioned:
            rep["blockers"].append(
                f"DATA PROVISIONING: the {len(m['db_ids'])} held-out Spider-train databases ({len(m['collections'])} collections, "
                f"{docs} documents) are not on Atlas. They must be inserted (same value mapping as import_atlas.py, but NOT with "
                "import_atlas.py, which drops collections in codegen_spider1 and rewrites the repo inventory) by a non-destructive "
                "seeder such as seed_atlas_spider_dev.seed() behind a wrapper for this contract, with a writable account (never the "
                f"evaluation account), into a separate database (proposed '{m['atlas_db']}'); the read-only role must be granted "
                "`find` on it; `score` then requires full parity with the contract digests. Not done by this task.")
    except Exception as e:  # noqa: BLE001
        chk("provisioning_contract", False, str(e)[:300])
        rep["blockers"].append("provisioning contract missing or inconsistent")
    try:
        import pymongo  # noqa: F401
        chk("pymongo_available", True, "importable")
    except Exception:  # noqa: BLE001
        chk("pymongo_available", False, "pymongo not importable in this Python")
        rep["blockers"].append("pymongo not installed in the scoring environment")
    try:
        S.load_registered_plan()
        chk("analysis_plan_registered", True, S.plan_sha())
    except Exception as e:  # noqa: BLE001
        chk("analysis_plan_registered", False, str(e)[:200])
        rep["blockers"].append("analysis plan not registered")
    rep["live_atlas_check"] = "NOT RUN (forbidden in this task); `score` runs the fail-closed read-only preflight with full parity"
    rep["status"] = "READY" if not rep["blockers"] else "BLOCKED"
    if write:
        D.write_json(guard_dest(DEST_ROOT) / "scoring_readiness/scoring_readiness.json", rep)
    return rep


# ------------------------------------------------------------------ verification of the Kaggle export (offline)
def safe_extract(zip_path, dest):
    import verify_kaggle_spider_dev_1034 as V
    return V.safe_extract(zip_path, dest, STATUS)


def _sums_ok(d):
    sums = {}
    for ln in (Path(d) / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, _, n = ln.partition("  ")
        sums[n] = h
    files = {p.relative_to(d).as_posix() for p in Path(d).rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
    return files == set(sums) and all(D.sha256_file(Path(d) / n) == h for n, h in sums.items())


def verify_predictions(dirs, exp):
    """All 8 runs, each from complete identical sources; ids/prompts/plan/identity/environment equal to the package."""
    cfg, ids = exp["cfg"], exp["ids"]
    problems, src, pred_sha, envs, rcs = [], {}, {}, {}, {}
    for d in dirs:
        d = Path(d)
        if not _sums_ok(d):
            problems.append(f"{d}: SHA256SUMS mismatch")
            continue
        st = json.loads((d / STATUS).read_text(encoding="utf-8"))
        if st.get("package_manifest_sha256") != exp["P"]["manifest_sha256"] or st.get("config_sha256") != exp["P"]["cfg_sha256"]:
            problems.append(f"{d}: another package or config")
            continue
        for run in exp["runs"]:
            n = run["name"]
            pf = d / "predictions" / n / "predictions.jsonl"
            if not pf.exists():
                continue
            h = D.sha256_file(pf)
            if n in src:
                if pred_sha[n] != h:
                    problems.append(f"{n}: present in two exports with DIFFERENT predictions")
                continue
            src[n], pred_sha[n] = d / "predictions" / n, h
    for run in exp["runs"]:
        n = run["name"]
        if n not in src:
            problems.append(f"{n}: missing from the export(s)")
            continue
        recs = D.read_jsonl(src[n] / "predictions.jsonl")
        rc = json.loads((src[n] / "run_config.json").read_text(encoding="utf-8"))
        rcs[n] = rc
        plan = {p["id"]: p for p in exp["plans"][n]}
        if [r["id"] for r in recs] != ids:
            problems.append(f"{n}: ids != the {len(ids)} evaluation ids in order")
        if any(r["status"] != "ok" for r in recs):
            problems.append(f"{n}: {sum(r['status'] != 'ok' for r in recs)} generation failures (resume on Kaggle first)")
        if rc.get("plan_sha256") != cfg["plans_sha256"][n] or rc.get("config_sha256") != exp["P"]["cfg_sha256"]:
            problems.append(f"{n}: run_config plan/config hash differs")
        want = K9.sha256_text("".join(exp["prompt_hashes"][n][i] + "\n" for i in ids))
        if rc.get("arm_prompts_sha256") != want:
            problems.append(f"{n}: run_config prompt hash differs")
        fp = K9.fingerprint({k: v for k, v in rc.items() if k not in K9.POST_FP})
        if fp != rc.get("fingerprint"):
            problems.append(f"{n}: run_config fingerprint does not recompute")
        for r in recs:
            v = K9.validate_prediction_record(r, n)
            if r.get("id") in plan and r.get("prompt_sha256") != plan[r["id"]]["prompt_sha256"]:
                v.append("prompt differs from plan")
            if r.get("id") in plan and r.get("retrieval", {}).get("retrieved_ids") != plan[r["id"]]["selected_ids"]:
                v.append("retrieval differs from plan")
            if r.get("fingerprint") != rc.get("fingerprint"):
                v.append("fingerprint")
            if r.get("precision", {}).get("dtype") != "bfloat16":
                v.append("precision")
            if v:
                problems.append(f"{n} {r.get('id')}: {v[:4]}")
                break
        envs[n] = K9.env_key(rc)
    keys = {K9.canon(e) for e in envs.values()}
    if len(keys) > 1:
        problems.append(f"arm runs come from different environments (GPU/dtype/torch/transformers): {sorted(keys)}")
    gpus = {e.get("gpu") for e in envs.values()}
    if gpus - set(cfg["hardware"]["allowed_gpus"]):
        problems.append(f"unregistered GPU: {sorted(gpus)}")
    return {"ok": not problems, "problems": problems, "arm_source": {n: str(p) for n, p in src.items()},
            "predictions_sha256": pred_sha, "environment": next(iter(envs.values()), None),
            "frozen": {"package_manifest_sha256": exp["P"]["manifest_sha256"], "config_sha256": exp["P"]["cfg_sha256"],
                       "analysis_plan_sha256": S.plan_sha(), "evaluation_ids_sha256": cfg["evaluation"]["ids_sha256"]}}


def cmd_verify(zips, dest_root=DEST_ROOT, pkg=PKG, expected_manifest_sha=EXPECTED_PACKAGE_MANIFEST_SHA256):
    dest_root = guard_dest(dest_root)
    exp = load_expected(pkg, expected_manifest_sha)
    ts = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    staged = [safe_extract(z, dest_root / ".staging" / ts / f"z{i}") for i, z in enumerate(zips)]
    return verify_predictions(staged, exp), exp


# ------------------------------------------------------------------ reference + scoring
def context_modules(run_dir, pkg=PKG):
    """spider1_kit / prompt_spider1 (unchanged repo copies) pointed at the package context + local SQLite."""
    ctx = Path(run_dir) / "context"
    ctx.mkdir(parents=True, exist_ok=True)
    for f in ("atlas_collections.json", "prompt_samples.json"):
        shutil.copyfile(Path(pkg) / "data/context" / f, ctx / f)
    shutil.copyfile(Path(pkg) / "data/context/tables_subset.json", ctx / "tables.json")
    return D.load_repo_modules(SPIDER1_CODE, ctx, DB_DIR)


def reference(run_dir, recs, K, compute=None):
    """Gold reference (D.compute_reference, unchanged) computed once per run dir and then frozen by hash."""
    f = Path(run_dir) / "reference/reference.jsonl.gz"
    if f.exists():
        meta = json.loads((f.parent / "reference_meta.json").read_text(encoding="utf-8"))
        need(D.sha256_file(f) == meta["sha256"], "stored reference changed")
        with gzip.open(f, "rt", encoding="utf-8") as g:
            return [json.loads(x) for x in g if x.strip()], meta
    refs = (compute or D.compute_reference)(recs, K, DB_DIR)
    need([r["id"] for r in refs] == [r["id"] for r in recs], "reference ids/order")
    f.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(f, "wb", mtime=0) as g:
        g.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in refs).encode("utf-8"))
    meta = {"sha256": D.sha256_file(f), "n": len(refs), "computed_at": D.now(),
            "status_counts": {s: sum(r["reference_status"] == s for r in refs) for s in D.REFERENCE_STATUSES}}
    D.write_json(f.parent / "reference_meta.json", meta)
    return refs, meta


def eval_fingerprint(mapping, ref_sha, pred_sha):
    return D.fingerprint({"core": D.sha256_file(ROOT / "scripts/nl2mql/dev1034_core.py"), "mapping": D.sha256_text(D.canon(mapping)),
                          "reference": ref_sha, "timeout_ms": D.EXEC_TIMEOUT_MS, "max_rows": D.MAX_RESULT_ROWS,
                          "adapter": ADAPTER_VERSION, "predictions": pred_sha})


def score_run(name, preds, ids, refs, mapping, mdb, K, out_dir, fp, log=print):
    d = Path(out_dir) / "arms" / name
    eck = D.Checkpoint(d / "checkpoints/evaluation.partial.jsonl", ids, fp, D.is_eval_retry)
    rck = D.Checkpoint(d / "checkpoints/rows.partial.jsonl", ids, fp)
    before = len(ids) - len(eck.todo())
    n = D.run_evaluation(preds, refs, mapping, mdb, K, eck, rck, D.EXEC_TIMEOUT_MS, progress=lambda a, b: log(f"  {name}: {a}/{b}"))
    evals = [eck.records[i] for i in ids if i in eck.records]
    D.write_jsonl(d / "evaluation_results.jsonl", evals)
    acct = S7.account(evals, ids)
    summ = {"experiment": K9.EXPERIMENT, "arm": name, "n_expected": len(ids), "accounting": acct, "correct": acct["correct"],
            "accuracy": S7.acc([e["outcome"] == "correct" for e in evals]), "evaluation_fingerprint": fp,
            "evaluated_this_session": n, "final_before_this_session": before,
            "status": "complete" if not (acct["infrastructure_failure"] + acct["generation_failure"]) else "partial"}
    D.write_json(d / "summary.json", summ)
    return evals, summ


def cmd_score(zips, run_name=None, atlas_db=None, pkg=PKG, dest_root=DEST_ROOT, client_factory=None, log=print,
              expected_manifest_sha=EXPECTED_PACKAGE_MANIFEST_SHA256, compute_reference=None):
    dest_root = guard_dest(dest_root)
    S.load_registered_plan()
    ver, exp = cmd_verify(zips, dest_root, pkg, expected_manifest_sha)
    run_dir = dest_root / "scoring_runs" / (run_name or ("run_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
    D.write_json(run_dir / "input_validation_report.json", ver)
    need(ver["ok"], f"predictions failed validation: {ver['problems'][:5]}")
    for n, srcd in ver["arm_source"].items():
        dst = run_dir / "predictions" / n
        if dst.exists():
            need(D.sha256_file(dst / "predictions.jsonl") == ver["predictions_sha256"][n], f"{dst} holds DIFFERENT predictions; new --run-name")
        else:
            shutil.copytree(srcd, dst)
    mapping = load_contract(atlas_db, pkg)
    recs = gold_records(exp["ids"])
    K, _ = context_modules(run_dir, pkg)
    refs, rmeta = reference(run_dir, recs, K, compute_reference)
    if client_factory is None:
        import local_atlas_scoring as L
        client_factory = L.default_client_factory
    client, _redact = client_factory()
    evals_by, summaries = {}, {}
    try:
        atl = D.atlas_preflight(client, mapping, full_parity=True, log=log)   # MANDATORY full parity; fails closed
        D.write_json(run_dir / "scoring/preflight/atlas_check.json", atl)
        mdb = client[mapping["atlas_db"]]
        for run in exp["runs"]:                                               # same evaluator, references, snapshot
            n = run["name"]
            preds = D.read_jsonl(run_dir / "predictions" / n / "predictions.jsonl")
            fp = eval_fingerprint(mapping, rmeta["sha256"], ver["predictions_sha256"][n])
            evals_by[n], summaries[n] = score_run(n, preds, exp["ids"], refs, mapping, mdb, K, run_dir / "scoring", fp, log)
            log(f"{n}: {summaries[n]['correct']}/{len(exp['ids'])}")
    finally:
        client.close()
    return finish(run_dir, exp, refs, evals_by, summaries, ver, atl, rmeta, mapping)


def _analysis_inputs(run_dir, exp, refs):
    meta = {r["id"]: {"hardness": r["hardness"], "db_id": r["db_id"], "reference_status": r["reference_status"]} for r in refs}
    pats = {r["id"]: PT.sql_patterns(r["sql"]) - {"select"} for r in gold_records(exp["ids"])}
    excl = sorted({x["eval_id"] for x in exp["cfg"]["retrieval_exclusions"]})
    lk = json.loads(LEAKAGE_LOCAL.read_text(encoding="utf-8"))
    sql_eq = sorted({x["eval_id"] for x in lk["L6_normalised_sql_equal"]["pairs"]})
    preds = {n: D.read_jsonl(Path(run_dir) / "predictions" / n / "predictions.jsonl") for n in exp["plans"]}
    return meta, pats, excl, sql_eq, preds


def finish(run_dir, exp, refs, evals_by, summaries, ver, atl, rmeta, mapping):
    meta, pats, excl, sql_eq, preds = _analysis_inputs(run_dir, exp, refs)
    analysis = S.analyze(evals_by, exp["ids"], meta, pats, excl, preds, exp["plans"], sql_eq)
    paired = [{"id": i, "db_id": meta[i]["db_id"], "hardness": meta[i]["hardness"], "reference_status": meta[i]["reference_status"],
               "query_family": S.family_of(pats[i]), **{n: S7.category({e["id"]: e for e in evals_by[n]}[i]) for n in evals_by}}
              for i in exp["ids"]]
    sd = Path(run_dir) / "scoring"
    D.write_json(sd / "analysis.json", analysis)
    D.write_jsonl(sd / "paired_per_question.jsonl", paired)
    complete = all(s["status"] == "complete" for s in summaries.values()) and len(summaries) == 8
    st = {"experiment": K9.EXPERIMENT, "status": "complete" if complete else "partial", "written_at": D.now(),
          "n_expected": len(exp["ids"]), "arms": {n: s["accounting"] for n, s in summaries.items()},
          "metric": "cross-engine execution accuracy (Atlas MQL result vs SQLite gold-SQL result), correct / N",
          "atlas": {"database": mapping["atlas_db"], "server_version": (atl or {}).get("server_version"),
                    "full_parity": True, "read_only_verified": (atl or {}).get("offending_actions") == []},
          "reference": rmeta, "inputs": {"predictions_sha256": ver["predictions_sha256"], **ver["frozen"]},
          "code_sha256": {p: D.sha256_file(ROOT / p) for p in ("scripts/nl2mql/dev1034_core.py", "scripts/nl2mql_mql0009/mql0009_scoring.py",
                                                               "scripts/nl2mql_mql0009/mql0009_stats.py")}}
    D.write_json(sd / "RUN_MANIFEST.json", st)
    (sd / "results_report.md").write_text(render_report(st, analysis), encoding="utf-8", newline="\n")
    return st, analysis, run_dir


def cmd_analyze(run_dir, pkg=PKG, expected_manifest_sha=EXPECTED_PACKAGE_MANIFEST_SHA256):
    exp = load_expected(pkg, expected_manifest_sha)
    run_dir = Path(run_dir)
    with gzip.open(run_dir / "reference/reference.jsonl.gz", "rt", encoding="utf-8") as g:
        refs = [json.loads(x) for x in g if x.strip()]
    evals_by = {r["name"]: D.read_jsonl(run_dir / "scoring/arms" / r["name"] / "evaluation_results.jsonl") for r in exp["runs"]}
    meta, pats, excl, sql_eq, preds = _analysis_inputs(run_dir, exp, refs)
    return S.analyze(evals_by, exp["ids"], meta, pats, excl, preds, exp["plans"], sql_eq)


def render_report(st, a):
    m = a["main"]
    h1, h2 = m["primary"]["H1_D_hybrid_vs_B_random_mean"], m["primary"]["H2_D_hybrid_vs_C_fixed"]
    acc = m["accuracy"]
    lines = [f"# MQL-0009 results ({st['status']})", "",
             f"Cross-engine execution accuracy on {a['n']} held-out Spider-train questions (16 databases), Atlas `{st['atlas']['database']}`.", "",
             "| run | correct / N | % |", "|---|---:|---:|"]
    for n in S.ARM_RUNS:
        lines.append(f"| {n} | {acc[n]['correct']} / {acc[n]['total']} | {acc[n]['percent']} |")
    lines.append(f"| B_random_mean (per-question mean of 5 seeds) | {acc[S.RBAR]['mean_correct']} / {acc[S.RBAR]['total']} | {acc[S.RBAR]['percent']} |")
    lines += ["", "## Primary (Holm over H1, H2)", "",
              f"- H1 D_hybrid − B_random_mean: {h1['diff_a_minus_b']:+.4f}, bootstrap 95% {h1['diff_bootstrap95']}, sign-flip p {h1['signflip_p']}, Holm p {h1['holm_adjusted_p']}",
              f"- H2 D_hybrid − C_fixed: {h2['diff_a_minus_b']:+.4f}, bootstrap 95% {h2.get('diff_bootstrap95')}, McNemar p {h2['mcnemar_exact_p']} "
              f"(D-only {h2['a_only']}, C-only {h2['b_only']}), Holm p {h2['holm_adjusted_p']}",
              "", f"Random-seed variability: {a['random_seed_variability']['per_seed_percent']} (SD {a['random_seed_variability']['sd_percent']} points)",
              "", "Seeds are not pooled as independent items. Query-family breakdowns are exploratory.", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("readiness")
    v = sub.add_parser("verify")
    v.add_argument("--predictions-zip", action="append", required=True)
    s = sub.add_parser("score")
    s.add_argument("--predictions-zip", action="append", required=True)
    s.add_argument("--run-name")
    s.add_argument("--atlas-db")
    an = sub.add_parser("analyze")
    an.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "readiness":
        r = readiness()
        print(json.dumps(r, indent=1))
        return 0 if r["status"] == "READY" else 2
    if a.cmd == "verify":
        r, _ = cmd_verify(a.predictions_zip)
        print(json.dumps({k: r[k] for k in ("ok", "problems", "predictions_sha256", "environment")}, indent=1))
        return 0 if r["ok"] else 2
    if a.cmd == "score":
        r = readiness(write=False)
        need(r["checks"].get("package_pinned_and_verified", {}).get("ok"), "package pin check failed")
        st, _, run_dir = cmd_score(a.predictions_zip, a.run_name, a.atlas_db)
        print(json.dumps(st, indent=1), "\n", run_dir)
        return 0
    print(json.dumps(cmd_analyze(a.run_dir)["main"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
