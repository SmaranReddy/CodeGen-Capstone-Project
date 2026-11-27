"""MQL-0015 local scoring adapter (Windows, system Python with pymongo; read-only Atlas).

  verify  extract the Kaggle export zip(s) (SHA256SUMS), require status complete + the frozen package manifest, and
          re-run mql0015_core.verify_exports (1,000 registered ids, frozen prompts, weights per arm). No database access.
  score   verify + gold reference (computed once from the test_database SQLite files, frozen by hash) + read-only
          full-parity Atlas preflight on `codegen_spider1_test_mql0015` (dev1034_core.atlas_preflight: refuses any
          account with a write/admin action) + the UNCHANGED evaluator (dev1034_core.run_evaluation) + infrastructure
          re-execution (<= 3 rounds; > 10 residual -> stop before analysis) + the registered analysis.

All three registered arms (and the three first-answer arms) are scored in ONE session; each DISTINCT (item, raw_text,
status) is executed once and its outcome is shared by every arm holding that exact text.

Usage (repo root):
  python -B scripts/nl2mql_mql0015/mql0015_scoring.py readiness        # read-only Atlas access + parity check only
  python -B scripts/nl2mql_mql0015/mql0015_scoring.py verify --predictions-zip mql0015_predictions.zip
  python -B scripts/nl2mql_mql0015/mql0015_scoring.py score  --predictions-zip mql0015_predictions.zip --run-name run_final
"""
import argparse
import gzip
import importlib.util
import json
import re
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(1, str(ROOT / "scripts/nl2mql"))
sys.path.insert(2, str(ROOT / "scripts"))
import mql0015_common as CM      # noqa: E402
import mql0015_analysis as AN    # noqa: E402

PKG = CM.PKG_OUT
DEST_ROOT = CM.OUT
DB_DIR = CM.TEST_DB_DIR
SPIDER1_CODE = ROOT / "scripts/spider1"
ADAPTER_VERSION = "mql0015-scoring-v1"
MAX_INFRA_ROUNDS = 3
MAX_RESIDUAL_INFRA = 10
need = CM.need


def _core(pkg=PKG):
    spec = importlib.util.spec_from_file_location("mql0015_core", Path(pkg) / "src/mql0015_core.py")
    sys.path.insert(0, str(Path(pkg) / "src"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    need(CM.sha256_file(CM.REGISTRATION) == P["cfg"]["registration_sha256"], "registration differs from the package pin")
    pin = P["cfg"]["analysis_sha256"]["scripts/nl2mql_mql0015/src/mql0015_analysis.py"]
    need(CM.sha256_file(HERE / "src/mql0015_analysis.py") == pin, "analysis module changed after the package was frozen")
    for rel, h in P["cfg"]["scoring_code_sha256"].items():
        need(CM.sha256_file(ROOT / rel) == h, f"{rel} changed after the package was frozen (registered scoring rules)")
    return C, P


def readiness(client, mapping, preflight=None, log=print):
    """Read-only Atlas readiness on the MQL-0015 database only: dev1034_core.atlas_preflight (refuses any account that
    holds a write/admin action anywhere; needs find on the target; all 180 collections, exact counts, full parity)."""
    import dev1034_core as D
    need(mapping["atlas_db"] == CM.ATLAS_DB, f"readiness target {mapping['atlas_db']} != {CM.ATLAS_DB}")
    rep = (preflight or D.atlas_preflight)(client, mapping, full_parity=True, log=log)
    need(rep.get("atlas_db") == CM.ATLAS_DB, "preflight checked another database")
    return rep


def cmd_readiness(pkg=PKG, client_factory=None, contract_path=None, out_dir=None, log=print):
    import dev1034_core as D
    C, P = load_expected(pkg)
    mapping = load_contract(P, contract_path)
    if client_factory is None:
        import local_atlas_scoring as L
        client_factory = L.default_client_factory
    client, _redact = client_factory()
    try:
        rep = readiness(client, mapping, log=log)
    finally:
        client.close()
    out = guard_dest(Path(out_dir or DEST_ROOT / "readiness") / f"atlas_readiness_{time.strftime('%Y%m%dT%H%M%S')}.json")
    D.write_json(out, rep)
    return rep, out


def load_contract(P, contract_path=None):
    import dev1034_core as D
    path = Path(contract_path or ROOT / P["cfg"]["scoring_contract"]["path"])
    need(D.sha256_file(path) == P["cfg"]["scoring_contract"]["sha256"], "scoring contract differs from the package pin")
    m = json.loads(path.read_text(encoding="utf-8"))
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
    ver = C.verify_exports(recs, P)
    for a in C.ARMS:
        need(CM.sha256_file(r / "predictions" / a / "predictions.jsonl") == man["per_arm"][a]["predictions_sha256"], f"{a}: hash")
    return ({"ok": True, "export_root": str(r), "verify_exports": ver,
             "predictions_sha256": {a: man["per_arm"][a]["predictions_sha256"] for a in C.ARMS}}, recs, C, P)


def context_modules(run_dir, P, pkg=PKG):
    import dev1034_core as D
    ctx = Path(run_dir) / "context"
    ctx.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(pkg) / "data/context/atlas_collections.json", ctx / "atlas_collections.json")
    for n, h in P["cfg"]["prompt_context_sha256"].items():
        need(D.sha256_file(CM.MQL13_PROMPT_CONTEXT / n) == h, f"prompt context {n} changed")
        shutil.copyfile(CM.MQL13_PROMPT_CONTEXT / n, ctx / n)
    return D.load_repo_modules(SPIDER1_CODE, ctx, DB_DIR)


def gold_records(ids):
    need(CM.sha256_file(CM.TEST_FULL) == CM.TEST_FULL_SHA256, "test_full changed")
    t = {r["id"]: r for r in CM.read_jsonl(CM.TEST_FULL)}
    need(all(i in t for i in ids), "gold record missing")
    return [t[i] for i in ids]


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
    """One slot per distinct (item, raw_text, status); arm order AN.ARMS fixes the numbering."""
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
    eck = D.Checkpoint(Path(out_dir) / f"checkpoints/evaluation_{fp}.partial.jsonl", ids, fp, D.is_eval_retry)
    rck = D.Checkpoint(Path(out_dir) / f"checkpoints/rows_{fp}.partial.jsonl", ids, fp)
    D.run_evaluation(slots, srefs, mapping, mdb, K, eck, rck, D.EXEC_TIMEOUT_MS,
                     progress=lambda n, t: print(f"  evaluated {n}/{t}", flush=True))
    return {i: eck.records[i] for i in ids}


def is_infra(ev):
    return AN.taxonomy(ev) == "infrastructure"


def evaluate_with_infra_reruns(slots, refs, mapping, mdb, K, out_dir, fp, evaluate, log=print):
    """Evaluate every slot; re-execute infrastructure outcomes up to MAX_INFRA_ROUNDS times (same session)."""
    ev = evaluate(slots, refs, mapping, mdb, K, out_dir, fp)
    rounds = []
    for k in range(1, MAX_INFRA_ROUNDS + 1):
        bad = [s for s in slots if is_infra(ev[s["id"]])]
        if not bad:
            break
        log(f"infrastructure re-execution round {k}: {len(bad)} slots")
        again = evaluate(bad, refs, mapping, mdb, K, out_dir, f"{fp}-infra{k}")
        for s in bad:
            ev[s["id"]] = {**again[s["id"]], "infra_reruns": k}
        rounds.append({"round": k, "slots": len(bad), "still_infra": sum(is_infra(ev[s["id"]]) for s in bad)})
    return ev, rounds


def metadata():
    """Gold-derived reporting metadata (outside the package): sample manifest items, D3 list, prompt tokens,
    (db, gold SQL) clusters and population hardness x HAVING shares over all 2,147 test_full items."""
    need(CM.sha256_file(CM.SAMPLE_MANIFEST) == CM.SAMPLE_MANIFEST_SHA256, "sample manifest changed")
    man = json.loads(CM.SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    meta = {it["id"]: it for it in man["items"]}
    a1 = json.loads(CM.AMENDMENT_A1.read_text(encoding="utf-8"))
    near_dup = a1["near_duplicate_items_D3"]
    toks = {p["id"]: p["r3_prompt_tokens_est"] for p in CM.read_jsonl(CM.PROMPT_PINS)}
    spec = importlib.util.spec_from_file_location("mql0015_sampler", ROOT / "scripts/nl2mql_final_1000/sample_final_1000.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    PT = S.load_patterns()
    full = CM.read_jsonl(CM.TEST_FULL)
    cells = {}
    for r in full:
        c = f"{r['hardness']}|{'having' if 'having' in S.labels(r['sql'], PT) else 'no_having'}"
        cells[c] = cells.get(c, 0) + 1
    pop = {c: v / len(full) for c, v in cells.items()}
    ws = re.compile(r"\s+")
    clusters = {r["id"]: r["db_id"] + "|" + ws.sub(" ", r["sql"].strip().lower()) for r in full if r["id"] in meta}
    return meta, near_dup, toks, clusters, pop


def cmd_score(zips, run_name, pkg=PKG, client_factory=None, preflight=None, evaluate=None, compute_reference=None,
              manifest_sha=None, dest_root=None, contract_path=None, reps=AN.BOOT_REPS, log=print):
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
    mapping = load_contract(P, contract_path)
    K, _ = context_modules(run_dir, P, pkg)
    gold = gold_records(ids)
    refs, rmeta = reference(run_dir, gold, K, compute_reference)
    slots, slot_of = unique_slots(recs, ids)
    if client_factory is None:
        import local_atlas_scoring as L
        client_factory = L.default_client_factory
    client, _redact = client_factory()
    try:
        atl = (preflight or D.atlas_preflight)(client, mapping, full_parity=True, log=log)
        need(atl.get("ok", True) and atl.get("atlas_db", mapping["atlas_db"]) == CM.ATLAS_DB, "Atlas preflight failed")
        D.write_json(run_dir / "scoring/preflight/atlas_check.json", atl)
        mdb = client[mapping["atlas_db"]]
        fp = D.fingerprint({"adapter": ADAPTER_VERSION, "mapping": D.sha256_text(D.canon(mapping)), "reference": rmeta["sha256"],
                            "predictions": ver["predictions_sha256"], "core": D.sha256_file(ROOT / "scripts/nl2mql/dev1034_core.py")})
        ev_slots, infra_rounds = evaluate_with_infra_reruns(slots, refs, mapping, mdb, K, run_dir / "scoring", fp,
                                                            evaluate or default_evaluate, log)
    finally:
        client.close()
    need(set(ev_slots) == {s["id"] for s in slots}, "unique evaluations incomplete")
    D.write_jsonl(run_dir / "scoring/evaluation_unique.jsonl", [ev_slots[s["id"]] for s in slots])
    residual = sorted({s["item_id"] for s in slots if is_infra(ev_slots[s["id"]])})
    D.write_json(run_dir / "scoring/infrastructure.json", {"rounds": infra_rounds, "residual_items": residual})
    need(len(residual) <= MAX_RESIDUAL_INFRA, f"{len(residual)} items remain infrastructure failures after {MAX_INFRA_ROUNDS} "
                                              "re-execution rounds (> 10): re-score in a new session before analysis "
                                              "(registered rule); no analysis written")
    evals = {}
    for a in AN.ARMS:
        evals[a] = {i: {**ev_slots[slot_of[(a, i)]], "id": i, "slot": slot_of[(a, i)]} for i in ids}
        D.write_jsonl(run_dir / "scoring/arms" / a / "evaluation_results.jsonl", [evals[a][i] for i in ids])
        log(f"{a}: {sum(e['outcome'] == 'correct' for e in evals[a].values())}/{len(ids)}")
    meta, near_dup, toks, clusters, pop = metadata()
    ana = AN.analyze(evals, ids, meta, near_dup, toks, clusters, pop, recs, reps=reps)
    ana["unique_evaluations"] = len(slots)
    ana["infrastructure_reruns"] = infra_rounds
    sd = run_dir / "scoring"
    D.write_json(sd / "analysis.json", ana)
    manifest = {"experiment": CM.EXPERIMENT, "adapter": ADAPTER_VERSION, "scored_at": D.now(), "atlas_db": mapping["atlas_db"],
                "atlas_check_sha256": D.sha256_file(sd / "preflight/atlas_check.json"), "reference": rmeta,
                "package_manifest_sha256": P["manifest_sha256"], "registration_sha256": P["cfg"]["registration_sha256"],
                "amendment_A1_sha256": P["cfg"]["amendment_A1_sha256"], "predictions_sha256": ver["predictions_sha256"],
                "unique_evaluations": len(slots), "bootstrap_reps": reps,
                "code_sha256": {p: D.sha256_file(ROOT / p) for p in ("scripts/nl2mql/dev1034_core.py",
                                                                      "scripts/nl2mql_mql0015/mql0015_scoring.py",
                                                                      "scripts/nl2mql_mql0015/src/mql0015_analysis.py")}}
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
    sub.add_parser("readiness", help="read-only Atlas check of codegen_spider1_test_mql0015 (after provisioning + grants)")
    a = ap.parse_args(argv)
    if a.cmd == "readiness":
        rep, out = cmd_readiness()
        print(json.dumps({k: rep.get(k) for k in ("atlas_db", "server_version", "authenticated_users", "offending_actions",
                                                  "collections_found", "missing_collections")}, indent=1))
        print("report:", out)
    elif a.cmd == "verify":
        ver, _, _, _ = cmd_verify(a.predictions_zip)
        print(json.dumps(ver, indent=1))
    else:
        ana, _man, run_dir = cmd_score(a.predictions_zip, a.run_name)
        print(AN.render(ana))
        print("run:", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
