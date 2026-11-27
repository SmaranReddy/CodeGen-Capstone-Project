"""S0 / E.6: hash baseline of every frozen artifact, and re-verification.

python -I -B scripts/pilot_v3_followup/hash_baseline.py baseline   -> writes outputs/pilot_v3_followup/hash_baseline.json (refuses to overwrite)
python -I -B scripts/pilot_v3_followup/hash_baseline.py verify [label]
        -> re-hashes everything, compares with (a) the recorded manifest hashes and (b) the baseline,
           writes outputs/pilot_v3_followup/hash_verify_<label>.json, exits 1 on any change.

Two kinds of check:
  * "recorded": a hash written in an existing freeze manifest (round3_code_freeze, pilot_v3_run01_runner_freeze,
    corpus_v1_runner_freeze, pilot_freeze, spider_pool_v2 manifest, manual_human_review/source_hashes_before.json).
    Mismatches here are reported as "recorded_mismatch"; a path that cannot be resolved is "unresolved".
  * "snapshot": every file under the frozen output directories (provisional_corpus_v1, pilot_v3/run01) and the
    freeze manifests themselves, hashed at baseline time and compared later.

The protected holdout data/splits_expanded/test.jsonl is never opened: only its size and mtime are recorded.
"""
import hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "pilot_v3_followup"
BASE = OUT / "hash_baseline.json"
FV2 = "scripts/finetune_v2/"
PROTECTED = {"data/splits_expanded/test.jsonl"}
MANIFESTS = {
    "round3_code_freeze": "data/interim/finetune_v2/round3_code_freeze.json",
    "pilot_v3_run01_runner_freeze": "data/interim/finetune_v2/pilot_v3_run01_runner_freeze.json",
    "corpus_v1_runner_freeze": "data/interim/finetune_v2/corpus_v1_runner_freeze.json",
    "pilot_freeze": "data/interim/finetune_v2/pilot_freeze.json",
    "spider_pool_v2_manifest": "data/processed/spider_pool_v2/manifest.json",
    "source_hashes_before": "outputs/sql2mql_v2/pilot_v3/run01/manual_human_review/source_hashes_before.json",
}
SNAPSHOT_DIRS = ["outputs/sql2mql_v2/provisional_corpus_v1", "outputs/sql2mql_v2/pilot_v3/run01",
                 "data/processed/spider_pool_v2", "data/interim/finetune_v2"]
SNAPSHOT_FILES = ["spider_pool.zip"]
# the snapshot of data/interim/finetune_v2 only covers files that existed at baseline; new versioned files are allowed there


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_path(k):
    return k if "/" in k else FV2 + k


def recorded_pairs():
    """(source, key, repo_relative_path_or_None, expected_sha)"""
    L = []
    j = lambda n: json.loads((ROOT / MANIFESTS[n]).read_text(encoding="utf-8"))
    r3 = j("round3_code_freeze")
    for k, h in r3["code_sha256"].items():
        L.append(("round3_code_freeze", k, code_path(k), h))
    pf = j("pilot_v3_run01_runner_freeze")
    L.append(("pilot_v3_run01_runner_freeze", "runner_sha256", FV2 + "run_pilot_v3.py", pf["runner_sha256"]))
    for k, h in pf["frozen_code_19"].items():
        L.append(("pilot_v3_run01_runner_freeze", k, code_path(k), h))
    L.append(("pilot_v3_run01_runner_freeze", "conversion_kit", "scripts/conversion_kit.py",
              pf["conversion_kit_sha256 (imported by eval_kit, not in the 19)"]))
    inp = pf["inputs"]
    L.append(("pilot_v3_run01_runner_freeze", "spider_sql_v1_train", "data/processed/spider_sql_v1/train.jsonl", inp["spider_sql_v1_train_sha256"]))
    L.append(("pilot_v3_run01_runner_freeze", "pilot_freeze_file", MANIFESTS["pilot_freeze"], inp["pilot_freeze_file_sha256"]))
    L.append(("pilot_v3_run01_runner_freeze", "spider_pool_v2_manifest", MANIFESTS["spider_pool_v2_manifest"], inp["spider_pool_v2_manifest_sha256"]))
    for k, h in inp["spider_pool_v2_files"].items():
        L.append(("pilot_v3_run01_runner_freeze", k, "data/processed/spider_pool_v2/" + k, h))
    cf = j("corpus_v1_runner_freeze")
    for k, h in cf["runner_hashes"].items():
        L.append(("corpus_v1_runner_freeze", k, code_path(k), h))
    for k, h in cf["frozen_code_19"].items():
        L.append(("corpus_v1_runner_freeze", k, code_path(k), h))
    el = cf["eligibility_inputs"]
    emap = {"pool_train.jsonl": "data/processed/spider_pool_v2/train.jsonl",
            "record_audit.jsonl": "outputs/spider_pool_semantic_audit_v1/record_audit.jsonl",
            "audit_summary.json": "outputs/spider_pool_semantic_audit_v1/audit_summary.json",
            "pilot_cross_review_csv": "outputs/sql2mql_v2/pilot_v3/run01/cross_review/cross_verified_review.csv",
            "pilot_run01_records.jsonl": "outputs/sql2mql_v2/pilot_v3/run01/records.jsonl",
            "pilot_run01_runner_freeze.json": MANIFESTS["pilot_v3_run01_runner_freeze"]}
    for k, p in emap.items():
        L.append(("corpus_v1_runner_freeze", "eligibility/" + k, p, el[k]))
    for p, v in cf["eval_sources"].items():
        h = v.get("sha256") if isinstance(v, dict) else None
        if h:
            L.append(("corpus_v1_runner_freeze", "eval_sources", p, h))
    pz = j("pilot_freeze")
    L.append(("pilot_freeze", "source", pz["source"]["path"], pz["source"]["sha256"]))
    m = j("spider_pool_v2_manifest")
    L.append(("spider_pool_v2_manifest", "builder", m["builder"]["path"], m["builder"]["sha256"]))
    L.append(("spider_pool_v2_manifest", "zip", m["source"]["zip"]["path"], m["source"]["zip"]["sha256"]))
    for k, v in m["files"].items():
        L.append(("spider_pool_v2_manifest", k, "data/processed/spider_pool_v2/" + k, v["sha256"]))
    sh = j("source_hashes_before")
    for p, h in sh.items():
        if isinstance(h, str):
            L.append(("source_hashes_before", p, p, h))
    return L


def snapshot():
    snap = {}
    for d in SNAPSHOT_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                snap[p.relative_to(ROOT).as_posix()] = sha(p)
    for f in SNAPSHOT_FILES:
        snap[f] = sha(ROOT / f)
    for f in MANIFESTS.values():
        snap[f] = sha(ROOT / f)
    return snap


def protected_stat():
    out = {}
    for p in PROTECTED:
        st = (ROOT / p).stat()
        out[p] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "note": "never opened; size+mtime only"}
    return out


def check_recorded():
    rows, cache = [], {}
    for src, key, p, h in recorded_pairs():
        if p in PROTECTED:
            rows.append({"source": src, "key": key, "path": p, "status": "protected_not_opened"})
            continue
        fp = ROOT / p if p else None
        if not fp or not fp.is_file():
            rows.append({"source": src, "key": key, "path": p, "status": "unresolved", "expected": h})
            continue
        if p not in cache:
            cache[p] = sha(fp)
        rows.append({"source": src, "key": key, "path": p, "status": "ok" if cache[p] == h else "recorded_mismatch",
                     "expected": h, "actual": cache[p]})
    return rows


def summarize(rows):
    c = {}
    for r in rows:
        c[r["status"]] = c.get(r["status"], 0) + 1
    return c


def main():
    mode = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    rec = check_recorded()
    snap = snapshot()
    prot = protected_stat()
    if mode == "baseline":
        if BASE.exists():
            sys.exit(f"refusing to overwrite {BASE}")
        doc = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "script": "scripts/pilot_v3_followup/hash_baseline.py",
               "script_sha256": sha(__file__), "recorded_summary": summarize(rec), "recorded": rec,
               "snapshot_n": len(snap), "snapshot": snap, "protected": prot}
        BASE.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(json.dumps({"recorded_summary": doc["recorded_summary"], "snapshot_n": len(snap)}, indent=1))
        bad = [r for r in rec if r["status"] in ("recorded_mismatch", "unresolved")]
        for r in bad:
            print("  ", r["status"], r["source"], r["key"], r["path"])
        return
    label = sys.argv[2] if len(sys.argv) > 2 else time.strftime("%Y%m%dT%H%M%S")
    base = json.loads(BASE.read_text(encoding="utf-8"))
    changed = sorted(k for k, v in base["snapshot"].items() if snap.get(k) != v)
    added_in_frozen = sorted(k for k in snap if k not in base["snapshot"]
                             and not k.startswith("data/interim/finetune_v2/"))
    prot_changed = {k: (base["protected"][k], prot[k]) for k in prot
                    if (base["protected"][k]["size"], base["protected"][k]["mtime_ns"]) != (prot[k]["size"], prot[k]["mtime_ns"])}
    rec_diff = [r for r in rec if r["status"] in ("recorded_mismatch", "unresolved")]
    base_rec_bad = {(r["source"], r["key"]) for r in base["recorded"] if r["status"] in ("recorded_mismatch", "unresolved")}
    new_rec_bad = [r for r in rec_diff if (r["source"], r["key"]) not in base_rec_bad]
    ok = not changed and not added_in_frozen and not prot_changed and not new_rec_bad
    doc = {"verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "ok": ok,
           "recorded_summary": summarize(rec), "recorded_problems": rec_diff, "new_recorded_problems_vs_baseline": new_rec_bad,
           "snapshot_changed": changed, "files_added_inside_frozen_dirs": added_in_frozen,
           "protected_changed": prot_changed, "snapshot_n": len(base["snapshot"])}
    (OUT / f"hash_verify_{label}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("ok", "recorded_summary", "snapshot_changed", "files_added_inside_frozen_dirs",
                                           "protected_changed")}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
