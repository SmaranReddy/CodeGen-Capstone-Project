"""MQL-0013 local scoring adapter (Windows, system Python with pymongo; read-only Atlas).

  verify  extract the Kaggle export zip(s) (SHA256SUMS), require status complete + the frozen package manifest, and
          re-run mql0013_core.verify_exports against the frozen plans. No database access.
  score   verify + gold reference (computed once from the test_database SQLite files, frozen by hash) + read-only
          full-parity Atlas preflight on `codegen_spider1_test_mql0013` + the UNCHANGED registered evaluator
          (dev1034_core.run_evaluation / evaluate_prediction) + the registered analysis (src/mql0013_analysis.py).

Each DISTINCT (item, raw_text) is executed once and its outcome is shared by every arm holding that exact text
(registered; MQL-0010 showed byte-identical queries flipping outcome on re-execution from tie-order noise).

Outputs: outputs/nl2mql_mql0013/scoring_runs/<run>/ (input_validation_report.json, predictions/, context/, reference/,
scoring/preflight/atlas_check.json, scoring/evaluation_unique.jsonl, scoring/arms/<arm>/evaluation_results.jsonl,
scoring/analysis.json, scoring/RUN_MANIFEST.json, scoring/results_report.md).

Usage (repo root):
  python -B scripts/nl2mql_mql0013/mql0013_scoring.py verify --predictions-zip mql0013_predictions.zip
  python -B scripts/nl2mql_mql0013/mql0013_scoring.py score  --predictions-zip mql0013_predictions.zip --run-name run_final
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
PKG = ROOT / "kaggle/nl2mql_mql0013_test_full"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(1, str(HERE / "src"))
sys.path.insert(2, str(ROOT / "scripts/nl2mql"))
sys.path.insert(3, str(ROOT / "scripts"))
import mql0013_common as CM      # noqa: E402
import mql0013_analysis as AN    # noqa: E402

DEST_ROOT = CM.OUT
DB_DIR = CM.TEST_DB_DIR
SPIDER1_CODE = ROOT / "scripts/spider1"
ADAPTER_VERSION = "mql0013-scoring-v1"
need = CM.need


def _core(pkg=PKG):
    sys.path.insert(0, str(Path(pkg) / "src"))
    import mql0013_core as C
    return C


def guard_dest(dest, root=None):
    root = Path(root or DEST_ROOT).resolve()
    dest = Path(dest).resolve()
    need(dest == root or root in dest.parents, f"refusing to write outside {root}: {dest}")
    return dest


def load_expected(pkg=PKG, manifest_sha=None):
    C = _core(pkg)
    P = C.load_package(pkg)
    if manifest_sha is not None:
        need(P["manifest_sha256"] == manifest_sha, "package manifest differs from the frozen package")
    reg = json.loads(CM.REGISTRATION.read_text(encoding="utf-8"))
    need(CM.sha256_file(CM.REGISTRATION) == P["cfg"]["registration_sha256"], "registration differs from the package pin")
    need(CM.sha256_file(HERE / "src/mql0013_analysis.py") == reg["source_sha256"]["scripts/nl2mql_mql0013/src/mql0013_analysis.py"],
         "analysis module changed after registration")
    return C, P


def gold_records(ids):
    t = {r["id"]: r for r in CM.read_jsonl(CM.TEST_FULL)}
    need(all(i in t for i in ids), "gold record missing")
    return [t[i] for i in ids]


def load_contract(P):
    import dev1034_core as D
    need(D.sha256_file(CM.CONTRACT) == P["cfg"]["scoring_contract"]["mapping_sha256"], "scoring contract differs from the package pin")
    m = json.loads(CM.CONTRACT.read_text(encoding="utf-8"))
    need(m["atlas_db"] == CM.ATLAS_DB, f"contract target {m['atlas_db']} != {CM.ATLAS_DB}")
    need(set(m["collections"]) == set(P["atlas"]), "contract collections != package prompt context")
    for db, h in m["sqlite_sha256"].items():
        need(D.sha256_file(DB_DIR / db / f"{db}.sqlite") == h, f"{db}: SQLite file changed since the package was built")
    return m


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
    return files == set(sums) and all(CM.sha256_file(d / n) == h for n, h in sums.items())


def cmd_verify(zips, pkg=PKG, manifest_sha=None, dest_root=None):
    C, P = load_expected(pkg, manifest_sha)
    root = Path(dest_root or DEST_ROOT)
    staging = guard_dest(root / ".staging" / (time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]), root)
    done = []
    for k, z in enumerate(zips):
        dz = staging / f"z{k}"
        safe_extract(z, dz)
        roots = [p.parent for p in dz.rglob(C.STATUS_FILE)]
        need(len(roots) == 1, f"{z}: expected one export folder")
        r = roots[0]
        need(_sums_ok(r), f"{z}: SHA256SUMS check failed")
        st = json.loads((r / C.STATUS_FILE).read_text(encoding="utf-8"))
        need(st["package_manifest_sha256"] == P["manifest_sha256"], f"{z}: exported from another package")
        if st["status"] == "complete":
            done.append(r)
    need(done, "no complete export: every pass must be finished (resume the notebook)")
    r = done[-1]
    man = json.loads((r / C.MANIFEST_FILE).read_text(encoding="utf-8"))
    recs = {a: C.read_jsonl(r / "predictions" / a / "predictions.jsonl") for a in C.ARMS}
    ver = C.verify_exports(recs, P["plans"], P["ids"], P["shared"])
    for a in C.ARMS:
        need(CM.sha256_file(r / "predictions" / a / "predictions.jsonl") == man["per_arm"][a]["predictions_sha256"], f"{a}: hash")
    return ({"ok": True, "export_root": str(r), "verify_exports": ver,
             "predictions_sha256": {a: man["per_arm"][a]["predictions_sha256"] for a in C.ARMS}}, recs, C, P)


def context_modules(run_dir, pkg=PKG):
    import dev1034_core as D
    ctx = Path(run_dir) / "context"
    ctx.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(pkg) / "data/context/atlas_collections.json", ctx / "atlas_collections.json")
    shutil.copyfile(CM.WORK / "prompt_context/prompt_samples.json", ctx / "prompt_samples.json")
    shutil.copyfile(CM.WORK / "prompt_context/tables.json", ctx / "tables.json")
    return D.load_repo_modules(SPIDER1_CODE, ctx, DB_DIR)


def reference(run_dir, recs, K, compute=None):
    import dev1034_core as D
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


def unique_slots(recs, ids):
    """-> (slots: [prediction records with id 'item#k'], slot_of: {(arm, id): slot id}). One slot per distinct
    (item, raw_text, status); arm order R0, R0V, R3, R3V fixes the numbering."""
    slots, key_to_slot, slot_of = [], {}, {}
    by = {a: {r["id"]: r for r in recs[a]} for a in AN.ARMS}
    for i in ids:
        for a in AN.ARMS:
            r = by[a][i]
            key = (i, r["raw_text"], r["status"])
            if key not in key_to_slot:
                sid = f"{i}#{sum(k[0] == i for k in key_to_slot)}"
                key_to_slot[key] = sid
                slots.append({**r, "id": sid, "item_id": i})
            slot_of[(a, i)] = key_to_slot[key]
    return slots, slot_of


def default_evaluate(slots, refs, mapping, mdb, K, out_dir, fp):
    import dev1034_core as D
    ref_by = {r["id"]: r for r in refs}
    srefs = [{**ref_by[s["item_id"]], "id": s["id"]} for s in slots]
    ids = [s["id"] for s in slots]
    eck = D.Checkpoint(Path(out_dir) / "checkpoints/evaluation_unique.partial.jsonl", ids, fp, D.is_eval_retry)
    rck = D.Checkpoint(Path(out_dir) / "checkpoints/rows_unique.partial.jsonl", ids, fp)
    D.run_evaluation(slots, srefs, mapping, mdb, K, eck, rck, D.EXEC_TIMEOUT_MS,
                     progress=lambda n, t: print(f"  evaluated {n}/{t}", flush=True))
    return {i: eck.records[i] for i in ids}


def item_hardness_structure(gold):
    sys.path.insert(0, str(PKG / "src"))
    import patterns as PT
    import structure as S
    return ({r["id"]: r["hardness"] for r in gold},
            {r["id"]: AN.structure_labels(r["sql"], PT, S) for r in gold})


def cmd_score(zips, run_name, pkg=PKG, client_factory=None, preflight=None, evaluate=None, compute_reference=None,
              manifest_sha=None, dest_root=None, log=print):
    import dev1034_core as D
    ver, recs, C, P = cmd_verify(zips, pkg, manifest_sha, dest_root)
    ids = P["ids"]
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
    mapping = load_contract(P)
    K, _ = context_modules(run_dir, pkg)
    gold = gold_records(ids)
    refs, rmeta = reference(run_dir, gold, K, compute_reference)
    slots, slot_of = unique_slots(recs, ids)
    if client_factory is None:
        import local_atlas_scoring as L
        client_factory = L.default_client_factory
    client, _redact = client_factory()
    try:
        atl = (preflight or D.atlas_preflight)(client, mapping, full_parity=True, log=log)
        need(atl.get("ok"), "Atlas preflight failed")
        D.write_json(run_dir / "scoring/preflight/atlas_check.json", atl)
        mdb = client[mapping["atlas_db"]]
        fp = D.fingerprint({"adapter": ADAPTER_VERSION, "mapping": D.sha256_text(D.canon(mapping)), "reference": rmeta["sha256"],
                            "predictions": ver["predictions_sha256"], "core": D.sha256_file(ROOT / "scripts/nl2mql/dev1034_core.py")})
        ev_slots = (evaluate or default_evaluate)(slots, refs, mapping, mdb, K, run_dir / "scoring", fp)
    finally:
        client.close()
    need(set(ev_slots) == {s["id"] for s in slots}, "unique evaluations incomplete")
    D.write_jsonl(run_dir / "scoring/evaluation_unique.jsonl", [ev_slots[s["id"]] for s in slots])
    evals = {}
    for a in AN.ARMS:
        evals[a] = {i: {**ev_slots[slot_of[(a, i)]], "id": i, "slot": slot_of[(a, i)]} for i in ids}
        D.write_jsonl(run_dir / "scoring/arms" / a / "evaluation_results.jsonl", [evals[a][i] for i in ids])
        log(f"{a}: {sum(e['outcome'] == 'correct' for e in evals[a].values())}/{len(ids)}")
    hard, struct = item_hardness_structure(gold)
    ana = AN.analyze(evals, ids, hard, struct, recs)
    ana["unique_evaluations"] = len(slots)
    sd = run_dir / "scoring"
    D.write_json(sd / "analysis.json", ana)
    manifest = {"experiment": CM.EXPERIMENT, "adapter": ADAPTER_VERSION, "scored_at": D.now(), "atlas_db": mapping["atlas_db"],
                "atlas_check_sha256": D.sha256_file(sd / "preflight/atlas_check.json"), "reference": rmeta,
                "package_manifest_sha256": P["manifest_sha256"], "registration_sha256": P["cfg"]["registration_sha256"],
                "predictions_sha256": ver["predictions_sha256"], "unique_evaluations": len(slots),
                "code_sha256": {p: D.sha256_file(ROOT / p) for p in ("scripts/nl2mql/dev1034_core.py",
                                                                      "scripts/nl2mql_mql0013/mql0013_scoring.py",
                                                                      "scripts/nl2mql_mql0013/src/mql0013_analysis.py")}}
    D.write_json(sd / "RUN_MANIFEST.json", manifest)
    (sd / "results_report.md").write_text(AN.render(ana), encoding="utf-8", newline="\n")
    return ana, manifest, run_dir


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
        ver, _, _, _ = cmd_verify(a.predictions_zip)
        print(json.dumps(ver, indent=1))
    else:
        ana, _man, run_dir = cmd_score(a.predictions_zip, a.run_name)
        print(AN.render(ana))
        print("run:", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
