"""Evidence for component V (offline; no database, no model).

(a) False-alarm rate on known-correct pipelines: the 3,445 verified gold MQL of approved corpus v2.1, validated
    against their own databases' schema context (derived from tables.json + observed types via the corpus type
    contract is not needed: only field NAMES are used). Train-side, legitimate.
(b) DIAG sizing on the scored MQL-0009 predictions: issue rate among correct vs incorrect predictions, per arm and
    per failure detail. Sizes the ceiling of one feedback retry; it does not tune the validator (its rules are
    generic and were written before this script ran) and it is NOT a measured repair rate.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0010/src"))
sys.path.insert(1, str(ROOT / "kaggle/nl2mql_mql0009_retrieval_control/src"))
import validator as V          # noqa: E402
import mql0008_core as M8      # noqa: E402

RUN = ROOT / "outputs/nl2mql_mql0009_retrieval_control/scoring_runs/run_final"
PKG = ROOT / "kaggle/nl2mql_mql0009_retrieval_control"
OUT = ROOT / "outputs/nl2mql_mql0010/dev_evidence"
ARMS = ["A_zero_shot", "B_random_s1", "B_random_s2", "B_random_s3", "B_random_s4", "B_random_s5", "C_fixed", "D_hybrid"]


def rl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def corpus_context():
    tabs = {t["db_id"]: t for t in json.loads((ROOT / "data/raw/spider1/spider_data/tables.json").read_text(encoding="utf-8"))}
    ctx = {}
    for db, t in tabs.items():
        man = M8.schema_manifest_from_tables(t)
        for name, info in man.items():
            ctx[name] = {"db_id": db, "fields": info["fields"]}
    return ctx


def main():
    res = {}
    # (a) corpus false alarms
    ctx = corpus_context()
    corpus = rl(ROOT / "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl")
    vals = {}
    fa, codes, examples = 0, Counter(), []
    for r in corpus:
        v = vals.setdefault(r["db_id"], V.SchemaValidator(r["db_id"], ctx))
        iss = v.validate(r["mql"])
        if iss:
            fa += 1
            codes.update(x.code for x in iss)
            if len(examples) < 8:
                examples.append({"id": r["id"], "issues": [x.text() for x in iss]})
    res["corpus_false_alarms"] = {"n": len(corpus), "flagged": fa, "codes": dict(codes), "examples": examples}
    # (b) DIAG on MQL-0009 predictions (package schema context = the prompt context)
    pctx = json.loads((PKG / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
    vals = {}
    per_arm = {}
    for arm in ARMS:
        c = defaultdict(Counter)
        code_by = Counter()
        for e in rl(RUN / f"scoring/arms/{arm}/evaluation_results.jsonl"):
            mql = e.get("mql")
            if mql is None:
                c["not_executable_or_invalid"][e["outcome"]] += 1
                continue
            v = vals.setdefault(e["db_id"], V.SchemaValidator(e["db_id"], pctx))
            iss = v.validate(mql)
            key = "flagged" if iss else "clean"
            c[key][e["outcome"] if e["outcome"] != "incorrect" else f"incorrect:{e['detail']}"] += 1
            if iss:
                code_by.update({x.code for x in iss})
        # how much of the pipeline population the validator can see at all, and the empty-result bucket
        rows = {r["id"]: r for r in rl(RUN / f"scoring/arms/{arm}/checkpoints/rows.partial.jsonl")}
        opaque_first, empty_total, empty_flagged = 0, 0, 0
        for e in rl(RUN / f"scoring/arms/{arm}/evaluation_results.jsonl"):
            mql = e.get("mql")
            if mql and mql.get("pipeline") and isinstance(mql["pipeline"][0], dict) and set(mql["pipeline"][0]) & V.OPAQUE_STAGES:
                opaque_first += 1
            if e["detail"] == "wrong_row_count" and e["id"] in rows and not rows[e["id"]]["rows"]:
                empty_total += 1
                empty_flagged += bool(vals[e["db_id"]].validate(mql))
        flagged = c["flagged"]
        n_fl = sum(flagged.values())
        per_arm[arm] = {"flagged": dict(flagged), "clean_outcomes": dict(c["clean"]),
                        "flagged_total": n_fl, "flagged_correct": flagged.get("correct", 0),
                        "flag_precision_for_wrong": round(1 - flagged.get("correct", 0) / max(1, n_fl), 3),
                        "codes": dict(code_by), "invalid_or_unparsed": dict(c["not_executable_or_invalid"]),
                        "opaque_from_first_stage": opaque_first, "empty_result_failures": empty_total,
                        "empty_result_failures_flagged": empty_flagged}
    res["mql0009_DIAG"] = per_arm
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validator_evidence.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(res["corpus_false_alarms"], indent=1))
    for a, v in per_arm.items():
        print(a, {k: v[k] for k in ("flagged_total", "flagged_correct", "flag_precision_for_wrong", "codes",
                                    "opaque_from_first_stage", "empty_result_failures", "empty_result_failures_flagged")})
        print("   flagged:", v["flagged"])


if __name__ == "__main__":
    main()
