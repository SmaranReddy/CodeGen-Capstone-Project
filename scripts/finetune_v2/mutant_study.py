"""V2 mutant-kill study for sql2mql-v2 (frozen criterion G4), on accepted DEV labels only.

    python -B scripts/finetune_v2/mutant_study.py --records outputs/sql2mql_v2/<dev run>/records.jsonl

For up to N_LABELS accepted dev labels (seeded sample, seed 20261030), every single-edit mutant
(mutants_v2.py, at most MAX_PER_LABEL per label) is run through the FULL verifier.
  killed      the verifier rejects the mutant
  survivor    the verifier accepts it -> judged by the stronger oracle battery (oracle_v2.py):
     non-equivalent  some oracle family where the ORIGINAL label passes gives a different
                     result (or an error) for the mutant  => a verifier FALSE ACCEPT
     unkilled        no oracle family separates them (equivalent, or undetected by both)
Primary metric (as audit §4): labels with >= 1 non-equivalent accepted mutant / labels, with a
Wilson 95% interval. Frozen gate: upper bound <= 0.05.
Writes outputs/sql2mql_v2/mutant_study/{mutants.jsonl, summary.json}.
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import v2_common as C
import mutants_v2 as MU
import oracle_v2 as O
import verifier as V

N_LABELS, MAX_PER_LABEL, SEED = 200, 30, 20261030
OUT = C.OUT / "mutant_study"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--labels", type=int, default=N_LABELS)
    ap.add_argument("--out", default="mutant_study", help="directory under outputs/sql2mql_v2/")
    ap.add_argument("--exclude-labels", default=None,
                    help="JSON {ids: [...]} of labels excluded (held-out re-measurement)")
    ap.add_argument("--only-labels", default=None, help="JSON {ids: [...]}: the ONLY labels used")
    a = ap.parse_args()
    out_dir = C.OUT / a.out
    if (out_dir / "summary.json").exists():
        raise SystemExit(f"{out_dir} exists: refusing to overwrite an earlier mutant study")
    fr = json.loads(C.FREEZE.read_text(encoding="utf-8"))
    pilot = set(fr["pilot"]["ids"])
    recs = [r for r in C.read_jsonl(Path(a.records)) if r.get("verification", {}).get("status") == "accepted"]
    if any(r["id"] in pilot for r in recs):
        raise SystemExit("records contain pilot ids: the mutant study runs on DEV labels only")
    def label_ids(path):
        """A JSON file {"ids": [...]} (exact label set, including labels without mutants)."""
        return set(json.loads(Path(path).read_text(encoding="utf-8"))["ids"])

    excluded = set()
    if a.only_labels:   # secondary in-sample re-run on exactly the labels of an earlier study
        keep = label_ids(a.only_labels)
        recs = [r for r in recs if r["id"] in keep]
        if len(recs) != len(keep):
            raise SystemExit(f"--only-labels: {len(keep) - len(recs)} labels not accepted in these records")
    if a.exclude_labels:
        excluded = label_ids(a.exclude_labels)
        recs = [r for r in recs if r["id"] not in excluded]
    recs.sort(key=lambda r: r["id"])
    if len(recs) > a.labels:
        recs = sorted(random.Random(SEED).sample(recs, a.labels), key=lambda r: r["id"])
    client = C.local_client()
    C.check_server(client)
    real = V.RealDbs(client)
    out, t0 = [], time.time()
    for k, r in enumerate(recs, 1):
        item = {"id": r["id"], "db_id": r["db_id"], "sql": r["sql"]}
        muts = MU.mutants(r["mql"])[:MAX_PER_LABEL]
        oracle, base = None, None
        for op, m in muts:
            v = V.verify(item, m, real)
            row = {"label": r["id"], "db_id": r["db_id"], "operator": op, "verifier": v["status"],
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
            out.append(row)
        print(f"{k}/{len(recs)} labels, {len(out)} mutants, {time.time() - t0:.0f}s", flush=True)
    C.write_jsonl(out_dir / "mutants.jsonl", out)
    by_op = defaultdict(Counter)
    for x in out:
        by_op[x["operator"]][x["verdict"]] += 1
    bad_labels = sorted({x["label"] for x in out if x["verdict"] == "non_equivalent"})
    lo, hi = C.wilson(len(bad_labels), len(recs))
    summ = {
        "labels": len(recs), "mutants": len(out),
        "killed_by_verifier": sum(x["verdict"] == "killed" for x in out),
        "survivors": sum(x["verdict"] != "killed" for x in out),
        "survivors_non_equivalent (verifier false accepts)": sum(x["verdict"] == "non_equivalent" for x in out),
        "survivors_unkilled_by_both": sum(x["verdict"] == "unkilled" for x in out),
        "labels_with_false_accept": {"k": len(bad_labels), "n": len(recs),
                                     "rate": round(len(bad_labels) / len(recs), 4) if recs else None,
                                     "wilson95": [lo, hi], "ids": bad_labels},
        "gate_G4_upper_bound_le_0.05": (hi is not None and hi <= 0.05),
        "by_operator": {op: dict(c) for op, c in sorted(by_op.items())},
        "kill_reasons": dict(Counter(x["verifier"] for x in out if x["verdict"] == "killed").most_common()),
        "seconds": round(time.time() - t0),
        "code_sha256": {p: C.sha256_file(C.HERE / p) for p in
                        ("mutants_v2.py", "oracle_v2.py", "verifier.py", "fixtures.py", "comparator.py")},
        "excluded_labels": len(excluded),
        "records": str(Path(a.records).resolve().relative_to(C.ROOT)).replace("\\", "/"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "by_operator"}, indent=1))


if __name__ == "__main__":
    main()
