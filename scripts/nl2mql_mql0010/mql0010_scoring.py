"""MQL-0010 local scoring adapter (Windows; read-only Atlas). NOT run in this preparation phase.

Pipeline (one session, one full-parity preflight, every arm scored by the same evaluator state):
  verify   extract the Kaggle export zip(s) (hash-checked), require status complete, the frozen package manifest and
           registration index, plans equal to the frozen plans, and scoring.verify_exports (reuse, matched delta,
           identical retried sets for D0V / D0G, prompt hashes). No database access.
  score    verify + gold reference (computed once from local SQLite, frozen by hash) + read-only full-parity Atlas
           preflight on `codegen_spider1_holdout_mql0010` + the UNCHANGED registered evaluator
           (dev1034_core.run_evaluation, resumable checkpoints) for all 9 arms + scoring.analyze in the A4 order.

Outputs: outputs/nl2mql_mql0010/scoring_runs/<run>/ (input_validation_report.json, predictions/, context/,
reference/, scoring/preflight/atlas_check.json, scoring/arms/<arm>/evaluation_results.jsonl, scoring/analysis.json,
scoring/RUN_MANIFEST.json, scoring/results_report.md).

Usage (repo root):
  python -B scripts/nl2mql_mql0010/mql0010_scoring.py verify --predictions-zip mql0010_predictions.zip
  python -B scripts/nl2mql_mql0010/mql0010_scoring.py score  --predictions-zip mql0010_predictions.zip --run-name run_final
"""
import argparse
import gzip
import json
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PKG = ROOT / "kaggle/nl2mql_mql0010_holdout"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(1, str(ROOT / "scripts/nl2mql"))
sys.path.insert(2, str(ROOT / "scripts"))
import mql0010_core as C         # noqa: E402  (frozen package core)
import scoring as SC             # noqa: E402  (registered adapter logic, package copy)
import dev1034_core as D         # noqa: E402  (unchanged MQL-0006..0009 evaluator)

DEST_ROOT = ROOT / "outputs/nl2mql_mql0010"
CONTRACT = DEST_ROOT / "scoring_contract/atlas_mapping.json"
TRAIN = ROOT / "data/processed/spider_sql_v1/train.jsonl"
DB_DIR = ROOT / "data/raw/spider1/spider_data/database"
SPIDER1_CODE = ROOT / "scripts/spider1"
ATLAS_DB = "codegen_spider1_holdout_mql0010"
ADAPTER_VERSION = "mql0010-scoring-v1"


class ScoringError(RuntimeError):
    pass


def need(c, msg):
    if not c:
        raise ScoringError(msg)


def guard_dest(dest, root=None):
    root = Path(root or DEST_ROOT).resolve()
    dest = Path(dest).resolve()
    need(dest == root or root in dest.parents, f"refusing to write outside {root}: {dest}")
    return dest


def expected_manifest_sha():
    """The frozen package identity = the hash of the built package's manifest (checked against the registration)."""
    return C.sha256_file(PKG / "package_manifest.json")


def load_expected(pkg=PKG, manifest_sha=None):
    P = C.load_package(pkg)
    need(P["manifest_sha256"] == (manifest_sha or expected_manifest_sha()), "package manifest differs from the frozen package")
    plans = {a: C.read_jsonl(Path(pkg) / f"data/plans/plan_{a}.jsonl") for a in C.PLAN_ARMS}
    for a, pl in plans.items():
        need(C.sha256_text(C.canon(pl)) == P["cfg"]["plans_sha256"][a] and [r["id"] for r in pl] == P["ids"], f"{a}: frozen plan")
    return {"P": P, "ids": P["ids"], "plans": plans}


def gold_records(ids):
    tr = {r["id"]: r for r in D.read_jsonl(TRAIN)}
    need(all(i in tr for i in ids), "gold record missing")
    return [tr[i] for i in ids]


def load_contract(pkg=PKG):
    cfg = json.loads((Path(pkg) / "configs/experiment.json").read_text(encoding="utf-8"))
    need(D.sha256_file(CONTRACT) == cfg["scoring_contract"]["mapping_sha256"], "scoring contract differs from the package pin")
    m = json.loads(CONTRACT.read_text(encoding="utf-8"))
    need(m["atlas_db"] == ATLAS_DB, f"contract target {m['atlas_db']} != {ATLAS_DB}")
    ctx = json.loads((Path(pkg) / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
    need(set(m["collections"]) == set(ctx), "contract collections != package prompt context")
    for name, c in m["collections"].items():
        need(c["fields"] == list(ctx[name]["fields"]) and c["documents"] == ctx[name]["documents"], f"{name}: contract != context")
    for db, h in m["sqlite_sha256"].items():
        need(D.sha256_file(DB_DIR / db / f"{db}.sqlite") == h, f"{db}: SQLite file changed since the package was built")
    return m


# ------------------------------------------------------------------ verification of exports
def safe_extract(zip_path, dest):
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            need(not n.startswith(("/", "\\")) and ".." not in Path(n).parts, f"unsafe path in zip: {n}")
        z.extractall(dest)


def _sums_ok(d):
    sums = {}
    for ln in (d / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, _, n = ln.partition("  ")
        sums[n] = h
    files = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
    return files == set(sums) and all(D.sha256_file(d / n) == h for n, h in sums.items())


def verify_predictions(zips, exp, staging):
    staging = Path(staging)
    done = []
    for k, z in enumerate(zips):
        dz = staging / f"z{k}"
        safe_extract(z, dz)
        roots = [p.parent for p in dz.rglob(C.STATUS_FILE)]
        need(len(roots) == 1, f"{z}: expected one export folder")
        r = roots[0]
        need(_sums_ok(r), f"{z}: SHA256SUMS check failed")
        st = json.loads((r / C.STATUS_FILE).read_text(encoding="utf-8"))
        need(st["package_manifest_sha256"] == exp["P"]["manifest_sha256"], f"{z}: exported from another package")
        if st["status"] == "complete":
            done.append(r)
    need(len(done) >= 1, "no complete export: all 8 passes must be finished (resume the notebook)")
    r = done[-1]
    man = json.loads((r / C.MANIFEST_FILE).read_text(encoding="utf-8"))
    need(man["registration_index_sha256"] == exp["P"]["cfg"]["registration_index_sha256"], "registration index differs")
    recs = {a: C.read_jsonl(r / "predictions" / a / "predictions.jsonl") for a in SC.ARMS}
    plans = {a: C.read_jsonl(r / "plans" / f"plan_{a}.jsonl") for a in C.PLAN_ARMS}
    need(plans == exp["plans"], "exported plans differ from the frozen plans")
    ver = SC.verify_exports(recs, plans, exp["ids"])
    for a in SC.ARMS:
        need(D.sha256_file(r / "predictions" / a / "predictions.jsonl") == man["per_arm"][a]["predictions_sha256"], f"{a}: hash")
    return {"ok": True, "export_root": str(r), "verify_exports": ver,
            "predictions_sha256": {a: man["per_arm"][a]["predictions_sha256"] for a in SC.ARMS}}, recs


def cmd_verify(zips, pkg=PKG, manifest_sha=None, dest_root=None):
    exp = load_expected(pkg, manifest_sha)
    root = Path(dest_root or DEST_ROOT)
    staging = guard_dest(root / ".staging" / (time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]), root)
    ver, recs = verify_predictions(zips, exp, staging)
    return ver, recs, exp


# ------------------------------------------------------------------ scoring
def context_modules(run_dir, pkg=PKG):
    ctx = Path(run_dir) / "context"
    ctx.mkdir(parents=True, exist_ok=True)
    for f in ("atlas_collections.json", "prompt_samples.json"):
        shutil.copyfile(Path(pkg) / "data/context" / f, ctx / f)
    shutil.copyfile(Path(pkg) / "data/context/tables_subset.json", ctx / "tables.json")
    return D.load_repo_modules(SPIDER1_CODE, ctx, DB_DIR)


def reference(run_dir, recs, K, compute=None):
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


def default_evaluate_arm(arm, preds, refs, mapping, mdb, K, out_dir, fp):
    d = Path(out_dir) / "arms" / arm
    ids = [p["id"] for p in preds]
    eck = D.Checkpoint(d / "checkpoints/evaluation.partial.jsonl", ids, fp, D.is_eval_retry)
    rck = D.Checkpoint(d / "checkpoints/rows.partial.jsonl", ids, fp)
    D.run_evaluation(preds, refs, mapping, mdb, K, eck, rck, D.EXEC_TIMEOUT_MS)
    evals = [eck.records[i] for i in ids]
    D.write_jsonl(d / "evaluation_results.jsonl", evals)
    return {e["id"]: e["outcome"] for e in evals}


def cmd_score(zips, run_name, pkg=PKG, client_factory=None, preflight=None, evaluate_arm=None, compute_reference=None,
              manifest_sha=None, dest_root=None, log=print):
    ver, recs, exp = cmd_verify(zips, pkg, manifest_sha, dest_root)
    root = Path(dest_root or DEST_ROOT)
    run_dir = guard_dest(root / "scoring_runs" / run_name, root)
    D.write_json(run_dir / "input_validation_report.json", ver)
    for a, rs in recs.items():
        dst = run_dir / "predictions" / a / "predictions.jsonl"
        if dst.exists():
            need(D.sha256_file(dst) == ver["predictions_sha256"][a], f"{dst} holds DIFFERENT predictions; use a new --run-name")
        else:
            D.write_jsonl(dst, rs)
            need(D.sha256_file(dst) == ver["predictions_sha256"][a], f"{a}: copied predictions hash")
    mapping = load_contract(pkg)
    K, _ = context_modules(run_dir, pkg)
    refs, rmeta = reference(run_dir, gold_records(exp["ids"]), K, compute_reference)
    if client_factory is None:
        import local_atlas_scoring as L
        client_factory = L.default_client_factory
    client, _redact = client_factory()
    outcomes = {}
    try:
        atl = (preflight or D.atlas_preflight)(client, mapping, full_parity=True, log=log)   # fail-closed, read-only
        need(atl.get("ok"), "Atlas preflight failed")
        D.write_json(run_dir / "scoring/preflight/atlas_check.json", atl)
        mdb = client[mapping["atlas_db"]]
        for a in SC.ARMS:
            fp = D.fingerprint({"adapter": ADAPTER_VERSION, "mapping": D.sha256_text(D.canon(mapping)), "reference": rmeta["sha256"],
                                "predictions": ver["predictions_sha256"][a], "core": D.sha256_file(ROOT / "scripts/nl2mql/dev1034_core.py")})
            outcomes[a] = (evaluate_arm or default_evaluate_arm)(a, recs[a], refs, mapping, mdb, K, run_dir / "scoring", fp)
            need(set(outcomes[a]) == set(exp["ids"]), f"{a}: outcomes incomplete")
            log(f"{a}: {sum(v == 'correct' for v in outcomes[a].values())}/{len(exp['ids'])}")
    finally:
        client.close()
    ana = SC.analyze(outcomes, exp["ids"], recs)
    sd = run_dir / "scoring"
    D.write_json(sd / "analysis.json", ana)
    manifest = {"experiment": C.EXPERIMENT, "adapter": ADAPTER_VERSION, "scored_at": D.now(), "atlas_db": mapping["atlas_db"],
                "atlas_check_sha256": D.sha256_file(sd / "preflight/atlas_check.json"), "reference": rmeta,
                "package_manifest_sha256": exp["P"]["manifest_sha256"],
                "registration_index_sha256": exp["P"]["cfg"]["registration_index_sha256"],
                "predictions_sha256": ver["predictions_sha256"],
                "code_sha256": {p: D.sha256_file(ROOT / p) for p in ("scripts/nl2mql/dev1034_core.py",
                                                                      "scripts/nl2mql_mql0010/mql0010_scoring.py")}}
    D.write_json(sd / "RUN_MANIFEST.json", manifest)
    (sd / "results_report.md").write_text(render_report(ana), encoding="utf-8", newline="\n")
    return ana, manifest, run_dir


def render_report(a):
    lines = ["# MQL-0010 results", "", "| arm | correct / N |", "|---|---:|"]
    n = a["step1"]["n"]
    lines += [f"| {k} | {v} / {n} |" for k, v in a["accuracy"].items()]
    s1 = a["step1"]
    lines += ["", "## Registered sequence (A4)", "",
              f"- Step 1 D0V vs D0: {s1['diff_pp']:+.2f} pp, 95% CI {s1['ci95_pp']}, McNemar p {s1['p']:.4g}, "
              f"reject {s1['reject']}, practically meaningful (CI lower >= +1.0 pp) {s1['practically_meaningful']}"]
    for step in ("step2", "step3"):
        v = a[step]
        if isinstance(v, list):
            for r in v:
                lines.append(f"- {step} {r['a']} vs {r['b']}: {r['diff_pp']:+.2f} pp, CI {r['ci95_pp']}, p {r['p']:.4g}, "
                             f"Holm p {r['p_holm']:.4g}, reject {r['reject']}")
        else:
            lines.append(f"- {step}: {v}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--predictions-zip", action="append", required=True)
    s = sub.add_parser("score")
    s.add_argument("--predictions-zip", action="append", required=True)
    s.add_argument("--run-name", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        ver, _, _ = cmd_verify(a.predictions_zip)
        print(json.dumps(ver, indent=1))
    else:
        ana, man, run_dir = cmd_score(a.predictions_zip, a.run_name)
        print(render_report(ana))
        print("run:", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
