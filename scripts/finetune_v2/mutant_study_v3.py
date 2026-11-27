"""Mutant-kill study, iteration 3 runner (G4). Same metric, operators (mutants_v2.py) and judge
(oracle_v2.py) as mutant_study.py; differences are operational only:

  * --verifier picks the verifier module under test (`verifier` = verifier-2, `verifier3`)
  * per-label results are APPENDED to mutants.jsonl as they finish, and a re-run resumes
    (a memory-pressure kill no longer loses the work done so far); on resume a label whose
    row count differs from its expected mutant count, or a cut trailing line, is dropped and
    re-run, and the event is appended to resume_log.jsonl
  * the label set is exactly --only-labels (no internal re-sampling), so a pre-registered
    sample is used as frozen
  * refuses pilot ids, and refuses ids of the iteration-2 G4 holdout (never re-used)
  * --out round3_mutants (the pre-registered round 3) requires both freeze stages to match and
    the label set outputs/sql2mql_v2/round3/accepted_labels.json with --verifier verifier3

    python -B scripts/finetune_v2/mutant_study_v3.py --verifier verifier --records R \
        --only-labels L.json --out <dir under outputs/sql2mql_v2>
    python -B scripts/finetune_v2/mutant_study_v3.py ... --summarize   (summary.json only)
"""
import argparse
import importlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import v2_common as C
import mutants_v2 as MU
import oracle_v2 as O

MAX_PER_LABEL = 30          # as mutant_study.py
ROUND3_OUT = "round3_mutants"


def holdout2_label_ids():
    """Label ids of the iteration-2 G4 holdout, reconstructed WITHOUT opening its results:
    dev labels accepted by verifier-2 minus the 200 diagnosis labels (its pre-registered
    definition, data/interim/finetune_v2/g4_remeasure_prereg.json)."""
    acc = {r["id"] for r in C.read_jsonl(C.OUT / "dev_iter2" / "records.jsonl")
           if r.get("verification", {}).get("status") == "accepted"}
    diag = set(json.loads((C.OUT / "mutant_study" / "diagnosis_labels.json").read_text(encoding="utf-8"))["ids"])
    return acc - diag


def load_jsonl_tolerant(path):
    """-> (rows, n_dropped_trailing). A line that does not parse can only be the LAST line
    (an append cut by a kill); anything else is an error."""
    if not path.exists():
        return [], 0
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = []
    for i, x in enumerate(lines):
        try:
            rows.append(json.loads(x))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            return rows, 1
    return rows, 0


def resume_state(path, recs, out_dir):
    """Drop labels whose rows are incomplete (row count != expected mutant count) and a cut
    trailing line; rewrite mutants.jsonl without them and log the event."""
    rows, cut = load_jsonl_tolerant(path)
    by = {}
    for x in rows:
        by.setdefault(x["label"], []).append(x)
    bad = sorted(lid for lid, xs in by.items()
                 if lid not in recs or len(xs) != len(MU.mutants(recs[lid]["mql"])[:MAX_PER_LABEL]))
    if bad or cut:
        keep_rows = [x for x in rows if x["label"] not in set(bad)]
        path.write_text("".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in keep_rows),
                        encoding="utf-8")
        rows = keep_rows
    if rows or cut:
        with (out_dir / "resume_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "labels_kept": len({x["label"] for x in rows}),
                                "incomplete_labels_dropped": bad, "cut_trailing_line_dropped": cut}) + "\n")
    return rows


def summarize(rows, labels, verifier_name, records, out_dir):
    by_op = defaultdict(Counter)
    for x in rows:
        by_op[x["operator"]][x["verdict"]] += 1
    bad = sorted({x["label"] for x in rows if x["verdict"] == "non_equivalent"})
    lo, hi = C.wilson(len(bad), len(labels))
    V = importlib.import_module(verifier_name)
    fx = getattr(V, "FX").__name__
    summ = {
        "verifier": V.VERIFIER_VERSION, "labels": len(labels), "mutants": len(rows),
        "killed_by_verifier": sum(x["verdict"] == "killed" for x in rows),
        "survivors_non_equivalent (verifier false accepts)": sum(x["verdict"] == "non_equivalent" for x in rows),
        "survivors_unkilled_by_both": sum(x["verdict"] == "unkilled" for x in rows),
        "labels_with_false_accept": {"k": len(bad), "n": len(labels),
                                     "rate": round(len(bad) / len(labels), 4) if labels else None,
                                     "wilson95": [lo, hi], "ids": bad},
        "by_operator": {op: dict(c) for op, c in sorted(by_op.items())},
        "kill_reasons": dict(Counter(x["verifier"] for x in rows if x["verdict"] == "killed").most_common()),
        "code_sha256": {p: C.sha256_file(C.HERE / p) for p in
                        ("mutants_v2.py", "oracle_v2.py", f"{verifier_name}.py", f"{fx}.py",
                         "comparator.py", "mongo_import.py", "mutant_study_v3.py")},
        "records": str(Path(records).resolve().relative_to(C.ROOT)).replace("\\", "/"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", required=True, choices=("verifier", "verifier3"))
    ap.add_argument("--records", required=True)
    ap.add_argument("--only-labels", required=True, help='JSON {"ids": [...]}: exactly these labels')
    ap.add_argument("--out", required=True, help="directory under outputs/sql2mql_v2/")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    out_dir = C.OUT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "summary.json").exists() and not a.summarize:
        raise SystemExit(f"{out_dir} has a summary: refusing to overwrite a finished study")
    if a.out == ROUND3_OUT:
        import freeze_round3_code as FZ
        FZ.check_frozen()
        if a.verifier != "verifier3" or Path(a.only_labels).resolve() != (C.OUT / "round3" / "accepted_labels.json").resolve():
            raise SystemExit("round 3 runs verifier3 on exactly outputs/sql2mql_v2/round3/accepted_labels.json")
    V = importlib.import_module(a.verifier)
    keep = set(json.loads(Path(a.only_labels).read_text(encoding="utf-8"))["ids"])
    fr = json.loads(C.FREEZE.read_text(encoding="utf-8"))
    if keep & set(fr["pilot"]["ids"]):
        raise SystemExit("label set contains pilot ids")
    if keep & holdout2_label_ids():
        raise SystemExit("label set contains iteration-2 G4 holdout labels (never re-used)")
    recs = {r["id"]: r for r in C.read_jsonl(Path(a.records))
            if r["id"] in keep and r.get("verification", {}).get("status") == "accepted"}
    if set(recs) != keep:
        raise SystemExit(f"{len(keep - set(recs))} labels are not accepted in {a.records}")
    path = out_dir / "mutants.jsonl"
    done_rows = resume_state(path, recs, out_dir) if not a.summarize else C.read_jsonl(path)
    done = {x["label"] for x in done_rows}
    done |= {x["label"] for x in C.read_jsonl(out_dir / "no_mutants.jsonl")}
    if a.summarize:
        missing = keep - done
        if missing:
            raise SystemExit(f"{len(missing)} labels not finished")
        print(json.dumps(summarize(done_rows, sorted(keep), a.verifier, a.records, out_dir), indent=1))
        return
    client = C.local_client()
    C.check_server(client)
    real = V.RealDbs(client)
    todo = sorted(keep - done)
    t0 = time.time()
    for k, lid in enumerate(todo, 1):
        r = recs[lid]
        item = {"id": r["id"], "db_id": r["db_id"], "sql": r["sql"]}
        muts = MU.mutants(r["mql"])[:MAX_PER_LABEL]
        rows, oracle, base = [], None, None
        for op, m in muts:
            v = V.verify(item, m, real)
            row = {"label": lid, "db_id": r["db_id"], "operator": op, "verifier": v["status"],
                   "verifier_detail": v.get("detail"), "mutant": m}
            if v["status"] == "accepted":
                if base is None:
                    oracle, base = O.battery(item, r["mql"], real)
                _, res = O.battery(item, m, real, oracle)
                sep = [f for f, ok in res.items() if base.get(f) is True and ok is not True]
                row.update(oracle_separating=sep, verdict="non_equivalent" if sep else "unkilled",
                           original_oracle_failures=[f for f, ok in base.items() if ok is False])
            else:
                row["verdict"] = "killed"
            rows.append(row)
        if rows:
            with path.open("a", encoding="utf-8") as f:
                f.write("".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in rows))
        else:
            with (out_dir / "no_mutants.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"label": lid}) + "\n")
        print(f"{k}/{len(todo)} labels ({len(done)} earlier), {time.time() - t0:.0f}s", flush=True)
    rows = C.read_jsonl(path)
    print(json.dumps({k: v for k, v in summarize(rows, sorted(keep), a.verifier, a.records, out_dir).items()
                      if k != "by_operator"}, indent=1))


if __name__ == "__main__":
    main()
