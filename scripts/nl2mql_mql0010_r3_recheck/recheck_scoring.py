"""MQL-0014 local scoring adapter (Windows, system Python with pymongo; read-only Atlas).

  verify  extract the Kaggle export zip(s) (SHA256SUMS), require status complete + the frozen MQL-0014 package manifest,
          re-run mql0014_core.verify_exports against the frozen plans. No database access.
  score   verify + gold reference (recomputed from local SQLite and REQUIRED to equal the frozen MQL-0010 reference
          hash) + read-only full-parity Atlas preflight on codegen_spider1_holdout_mql0010 (the MQL-0010 contract)
          + the UNCHANGED evaluator (dev1034_core.run_evaluation) + the registered analysis.

Arms scored: R0, R0V, R3, R3V, plus D0V_mql0010 = the original MQL-0010 D0V predictions (hash-checked, read only),
rescored in the same session as a descriptive reproducibility check. Each DISTINCT (item, raw_text) is executed once
and the outcome is shared by every arm holding that exact text.

Outputs: outputs/nl2mql_mql0010_r3_execution_recheck/scoring_runs/<run>/.
Usage (repo root):
  python -B scripts/nl2mql_mql0010_r3_recheck/recheck_scoring.py verify --predictions-zip mql0014_predictions.zip
  python -B scripts/nl2mql_mql0010_r3_recheck/recheck_scoring.py score  --predictions-zip mql0014_predictions.zip --run-name run_final
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
PKG = ROOT / "kaggle/nl2mql_mql0014_r3_recheck"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(1, str(ROOT / "scripts/nl2mql"))
sys.path.insert(2, str(ROOT / "scripts"))
import recheck_common as CM      # noqa: E402
import recheck_analysis as AN    # noqa: E402

SPIDER1_CODE = ROOT / "scripts/spider1"
ADAPTER_VERSION = "mql0014-scoring-v1"
REPRO = AN.REPRO_ARM
need = CM.need


def _core(pkg):
    p = str(Path(pkg) / "src")
    if p not in sys.path:
        sys.path.insert(0, p)
    import mql0014_core as K
    return K


def guard_dest(dest, root):
    root, dest = Path(root).resolve(), Path(dest).resolve()
    need(dest == root or root in dest.parents, f"refusing to write outside {root}: {dest}")
    return dest


def load_expected(pkg=PKG):
    K = _core(pkg)
    P = K.load_package(pkg)
    reg = json.loads(CM.REGISTRATION.read_text(encoding="utf-8"))
    need(CM.sha256_file(CM.REGISTRATION) == P["cfg"]["registration_sha256"], "registration differs from the package pin")
    need(CM.sha256_file(HERE / "src/recheck_analysis.py") ==
         reg["source_sha256"]["scripts/nl2mql_mql0010_r3_recheck/src/recheck_analysis.py"], "analysis changed after registration")
    return K, P


def load_contract(P):
    import dev1034_core as D
    need(D.sha256_file(CM.M10_CONTRACT) == P["cfg"]["scoring_contract"]["mapping_sha256"], "scoring contract differs from the pin")
    m = json.loads(CM.M10_CONTRACT.read_text(encoding="utf-8"))
    need(m["atlas_db"] == CM.ATLAS_DB, f"contract target {m['atlas_db']}")
    need(set(m["collections"]) == set(P["atlas"]), "contract collections != package context")
    for db, h in m["sqlite_sha256"].items():
        need(D.sha256_file(CM.DB_DIR / db / f"{db}.sqlite") == h, f"{db}: SQLite file changed")
    return m


def _sums_ok(d):
    sums = {}
    for ln in (d / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, _, n = ln.partition("  ")
        sums[n] = h
    files = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
    return files == set(sums) and all(CM.sha256_file(d / n) == h for n, h in sums.items())


def cmd_verify(zips, pkg=PKG, dest_root=None):
    K, P = load_expected(pkg)
    root = Path(dest_root or CM.OUT)
    staging = guard_dest(root / ".staging" / (time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]), root)
    done = []
    for k, z in enumerate(zips):
        dz = staging / f"z{k}"
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                need(not n.startswith(("/", "\\")) and ".." not in Path(n).parts, f"unsafe path in zip: {n}")
            zf.extractall(dz)
        roots = [p.parent for p in dz.rglob(K.STATUS_FILE)]
        need(len(roots) == 1, f"{z}: expected one export folder")
        r = roots[0]
        need(_sums_ok(r), f"{z}: SHA256SUMS check failed")
        st = json.loads((r / K.STATUS_FILE).read_text(encoding="utf-8"))
        need(st["package_manifest_sha256"] == P["manifest_sha256"], f"{z}: exported from another package")
        if st["status"] == "complete":
            done.append(r)
    need(done, "no complete export: every pass must be finished (resume the notebook)")
    r = done[-1]
    man = json.loads((r / K.MANIFEST_FILE).read_text(encoding="utf-8"))
    recs = {a: K.read_jsonl(r / "predictions" / a / "predictions.jsonl") for a in K.ARMS}
    ver = K.verify_exports(recs, P["plans"], P["ids"], P["shared"])
    for a in K.ARMS:
        need(CM.sha256_file(r / "predictions" / a / "predictions.jsonl") == man["per_arm"][a]["predictions_sha256"], f"{a}: hash")
    return ({"ok": True, "export_root": str(r), "verify_exports": ver,
             "predictions_sha256": {a: man["per_arm"][a]["predictions_sha256"] for a in K.ARMS}}, recs, K, P)


def load_repro(ids):
    f = CM.M10_RUN / "predictions/D0V/predictions.jsonl"
    need(CM.sha256_file(f) == CM.M10_D0V_SHA, "original MQL-0010 D0V predictions changed")
    rows = CM.read_jsonl(f)
    need([r["id"] for r in rows] == list(ids), "MQL-0010 D0V ids/order")
    return rows


def context_modules(run_dir):
    import dev1034_core as D
    ctx = Path(run_dir) / "context"
    ctx.mkdir(parents=True, exist_ok=True)
    for f in ("atlas_collections.json", "prompt_samples.json"):
        shutil.copyfile(CM.M10_PKG / "data/context" / f, ctx / f)
    shutil.copyfile(CM.M10_PKG / "data/context/tables_subset.json", ctx / "tables.json")
    return D.load_repo_modules(SPIDER1_CODE, ctx, CM.DB_DIR)


TIMING_KEYS = ("gold_ms",)


def reference(run_dir, gold, K, compute=None):
    """The gold reference IS the frozen MQL-0010 reference (copied, sha-checked). It is also recomputed from local SQLite
    and must equal the frozen one on every field except timings (gold_ms)."""
    import dev1034_core as D
    src = CM.M10_RUN / "reference/reference.jsonl.gz"
    need(D.sha256_file(src) == CM.M10_REFERENCE_SHA, "frozen MQL-0010 reference changed")
    f = Path(run_dir) / "reference/reference.jsonl.gz"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, f)
    need(D.sha256_file(f) == CM.M10_REFERENCE_SHA, "copied reference hash")
    with gzip.open(f, "rt", encoding="utf-8") as g:
        refs = [json.loads(x) for x in g if x.strip()]
    need([r["id"] for r in refs] == [r["id"] for r in gold], "reference ids/order")
    again = (compute or D.compute_reference)(gold, K, CM.DB_DIR)
    strip = lambda r: {k: v for k, v in r.items() if k not in TIMING_KEYS}   # noqa: E731
    mism = [a["id"] for a, b in zip(refs, again) if strip(a) != strip(b)]
    meta = {"sha256": CM.M10_REFERENCE_SHA, "source": "outputs/nl2mql_mql0010/scoring_runs/run_final/reference (copied)",
            "n": len(refs), "recompute_mismatches_excluding_timing": len(mism), "mismatch_ids": mism[:10],
            "status_counts": {s: sum(r["reference_status"] == s for r in refs) for s in D.REFERENCE_STATUSES}}
    meta["equals_mql0010_reference"] = not mism
    D.write_json(f.parent / "reference_meta.json", meta)
    return refs, meta


def unique_slots(recs, ids, arms):
    slots, key_to_slot, slot_of = [], {}, {}
    by = {a: {r["id"]: r for r in recs[a]} for a in arms}
    for i in ids:
        for a in arms:
            r = by[a][i]
            key = (i, r["raw_text"], r["status"])
            if key not in key_to_slot:
                key_to_slot[key] = f"{i}#{sum(k[0] == i for k in key_to_slot)}"
                slots.append({**r, "id": key_to_slot[key], "item_id": i})
            slot_of[(a, i)] = key_to_slot[key]
    return slots, slot_of


def default_evaluate(slots, refs, mapping, mdb, K, out_dir, fp):
    import dev1034_core as D
    ref_by = {r["id"]: r for r in refs}
    srefs = [{**ref_by[s["item_id"]], "id": s["id"]} for s in slots]
    ids = [s["id"] for s in slots]
    eck = D.Checkpoint(Path(out_dir) / "checkpoints/evaluation_unique.partial.jsonl", ids, fp, D.is_eval_retry)
    rck = D.Checkpoint(Path(out_dir) / "checkpoints/rows_unique.partial.jsonl", ids, fp)
    D.run_evaluation(slots, srefs, mapping, mdb, K, eck, rck, D.EXEC_TIMEOUT_MS)
    return {i: eck.records[i] for i in ids}


def cmd_score(zips, run_name, pkg=PKG, client_factory=None, preflight=None, evaluate=None, compute_reference=None,
              dest_root=None, log=print):
    import dev1034_core as D
    ver, recs, K, P = cmd_verify(zips, pkg, dest_root)
    ids = P["ids"]
    root = Path(dest_root or CM.OUT)
    run_dir = guard_dest(root / "scoring_runs" / run_name, root)
    D.write_json(run_dir / "input_validation_report.json", ver)
    for a, rs in recs.items():
        dst = run_dir / "predictions" / a / "predictions.jsonl"
        if dst.exists():
            need(D.sha256_file(dst) == ver["predictions_sha256"][a], f"{dst} holds DIFFERENT predictions; use a new --run-name")
        else:
            D.write_jsonl(dst, rs)
    recs = {**recs, REPRO: load_repro(ids)}
    arms = list(AN.ARMS) + [REPRO]
    mapping = load_contract(P)
    Kmod, _ = context_modules(run_dir)
    gold_by = {r["id"]: r for r in CM.read_jsonl(CM.TRAIN)}
    gold = [gold_by[i] for i in ids]
    refs, rmeta = reference(run_dir, gold, Kmod, compute_reference)
    need(rmeta["equals_mql0010_reference"], "recomputed reference differs from the frozen MQL-0010 reference")
    slots, slot_of = unique_slots(recs, ids, arms)
    if client_factory is None:
        import local_atlas_scoring as L
        client_factory = L.default_client_factory
    client, _ = client_factory()
    try:
        atl = (preflight or D.atlas_preflight)(client, mapping, full_parity=True, log=log)
        need(atl.get("ok"), "Atlas preflight failed")
        D.write_json(run_dir / "scoring/preflight/atlas_check.json", atl)
        fp = D.fingerprint({"adapter": ADAPTER_VERSION, "mapping": D.sha256_text(D.canon(mapping)), "reference": rmeta["sha256"],
                            "predictions": ver["predictions_sha256"], "repro": CM.M10_D0V_SHA,
                            "core": D.sha256_file(ROOT / "scripts/nl2mql/dev1034_core.py")})
        ev = (evaluate or default_evaluate)(slots, refs, mapping, client[mapping["atlas_db"]], Kmod, run_dir / "scoring", fp)
    finally:
        client.close()
    need(set(ev) == {s["id"] for s in slots}, "unique evaluations incomplete")
    sd = run_dir / "scoring"
    D.write_jsonl(sd / "evaluation_unique.jsonl", [ev[s["id"]] for s in slots])
    evals = {}
    for a in arms:
        evals[a] = {i: {**ev[slot_of[(a, i)]], "id": i, "slot": slot_of[(a, i)]} for i in ids}
        D.write_jsonl(sd / "arms" / a / "evaluation_results.jsonl", [evals[a][i] for i in ids])
        log(f"{a}: {sum(e['outcome'] == 'correct' for e in evals[a].values())}/{len(ids)}")
    sys.path.insert(0, str(Path(pkg) / "src"))
    import patterns as PT
    import structure as S
    hard = {r["id"]: r["hardness"] for r in gold}
    struct = {r["id"]: AN.structure_labels(r["sql"], PT, S) for r in gold}
    ana = AN.analyze(evals, ids, hard, struct, {a: recs[a] for a in AN.ARMS})
    ana["unique_evaluations"] = len(slots)
    ana["identical_final_text_R0V_vs_D0V_mql0010"] = sum(slot_of[("R0V", i)] == slot_of[(REPRO, i)] for i in ids)
    ana["nature"] = "SECONDARY re-analysis of the MQL-0010 holdout; not a fresh independent test"
    D.write_json(sd / "analysis.json", ana)
    manifest = {"experiment": CM.EXPERIMENT, "adapter": ADAPTER_VERSION, "scored_at": D.now(), "atlas_db": mapping["atlas_db"],
                "atlas_check_sha256": D.sha256_file(sd / "preflight/atlas_check.json"), "reference": rmeta,
                "package_manifest_sha256": P["manifest_sha256"], "registration_sha256": P["cfg"]["registration_sha256"],
                "predictions_sha256": {**ver["predictions_sha256"], REPRO: CM.M10_D0V_SHA}, "unique_evaluations": len(slots),
                "code_sha256": {p: D.sha256_file(ROOT / p) for p in ("scripts/nl2mql/dev1034_core.py",
                                                                      "scripts/nl2mql_mql0010_r3_recheck/recheck_scoring.py",
                                                                      "scripts/nl2mql_mql0010_r3_recheck/src/recheck_analysis.py")}}
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
        print(json.dumps(cmd_verify(a.predictions_zip)[0], indent=1))
    else:
        ana, _m, run_dir = cmd_score(a.predictions_zip, a.run_name)
        print(AN.render(ana))
        print("run:", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
