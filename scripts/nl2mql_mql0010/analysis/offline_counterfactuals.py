"""DIAG offline counterfactuals on the scored MQL-0009 result rows (read-only; no Atlas, no model).

1. Sanity: re-derive every executed outcome from the stored rows with the registered evaluator
   (dev1034_core.recompute_outcome) and require equality with the scored outcome.
2. `_id` result-contract post-processor: a deterministic rewrite that appends {"$project": {"_id": 0}} under a
   static rule, simulated by dropping the `_id` column from the stored result rows (exactly what the extra stage
   does to the returned documents) and re-deciding with the same evaluator. Two rules:
     R1 "implicit": the final stage is an inclusion $project that never mentions _id (so _id is the source
        document id, never a computed answer value);
     R2 "not-group-key": _id is in the output and the final stage is not $group/$bucket/$sortByCount
        (i.e. _id is not a fresh group key).
   The rules are generic MongoDB semantics, fixed before this script was run; the 639 items only SIZE them.
   They are re-tested on the MQL-0010 development and holdout sets before any claim.
"""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/nl2mql_mql0009_retrieval_control/scoring_runs/run_final"
OUT = ROOT / "outputs/nl2mql_mql0010/mql0009_post_analysis"
sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
import dev1034_core as D  # noqa: E402  (unchanged registered evaluator)

ARMS = ["A_zero_shot", "B_random_s1", "B_random_s2", "B_random_s3", "B_random_s4", "B_random_s5", "C_fixed", "D_hybrid"]
GROUP_STAGES = {"$group", "$bucket", "$bucketAuto", "$sortByCount"}


def rl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def final_stage(mql):
    p = (mql or {}).get("pipeline") or []
    return p[-1] if p and isinstance(p[-1], dict) and len(p[-1]) == 1 else None


def rule_r1(mql):
    st = final_stage(mql)
    if not st or "$project" not in st or not isinstance(st["$project"], dict):
        return False
    spec = st["$project"]
    if "_id" in spec:
        return False
    vals = [v for v in spec.values()]
    exclusion = vals and all(v in (0, False) for v in vals)
    return not exclusion


def rule_r2(mql):
    st = final_stage(mql)
    if not st:
        return False
    k = next(iter(st))
    if k in GROUP_STAGES:
        return False
    if k == "$project" and isinstance(st[k], dict) and st[k].get("_id") not in (None, 0, False):
        return False   # _id explicitly computed/kept in the final projection
    return True


def main():
    K, _ = D.load_repo_modules(ROOT / "scripts/spider1", RUN / "context", ROOT / "data/raw/spider1/spider_data/database")
    with gzip.open(RUN / "reference/reference.jsonl.gz", "rt", encoding="utf-8") as g:
        refs = {r["id"]: r for r in (json.loads(x) for x in g if x.strip())}
    out = {}
    for arm in ARMS:
        ev = {e["id"]: e for e in rl(RUN / f"scoring/arms/{arm}/evaluation_results.jsonl")}
        rows = {r["id"]: r for r in rl(RUN / f"scoring/arms/{arm}/checkpoints/rows.partial.jsonl")}
        mism = sum(1 for i, e in ev.items() if e["executed"] and i in rows and
                   D.recompute_outcome(K, e, rows[i], refs[i]) != e["outcome"])
        res = {"sanity_mismatches": mism}
        for name, rule in (("R1_implicit", rule_r1), ("R2_not_group_key", rule_r2)):
            c = Counter()
            for i, e in ev.items():
                if not e["executed"] or i not in rows or not rule(e["mql"]):
                    continue
                c["rule_applies"] += 1
                cols = rows[i]["columns"]
                if "_id" not in cols:
                    continue
                c["applies_and__id_present"] += 1
                k = cols.index("_id")
                new = [tuple(v for j, v in enumerate(r) if j != k) for r in rows[i]["rows"]]
                if not [x for x in cols if x != "_id"]:
                    continue  # dropping would leave no column; the rewrite is skipped
                o = D.decide(K, new, refs[i])[0]
                c[f"{e['outcome']}->{o}"] += 1
            c["net"] = c["incorrect->correct"] + c["not_comparable->correct"] - c["correct->incorrect"] - c["correct->not_comparable"]
            res[name] = dict(c)
        res["base_correct"] = sum(e["outcome"] == "correct" for e in ev.values())
        out[arm] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "offline_counterfactuals.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
