"""G2 / G3 re-run with verifier-3 (outputs of verifier-2 are kept unchanged).

    python -B scripts/finetune_v2/adversarial_v3.py
G2  every converted case of the adversarial suite (outputs/sql2mql_v2/adversarial/suite.jsonl:
    MQL from the unchanged converter, correctness already judged against the hand-specified /
    documented-SQLite expectation under both NULL representations) is re-verified by
    verifier-3; a false accept is a wrong case verifier-3 accepts. The converter is re-run to
    confirm the MQL is byte-identical.
G3  the 33 v1 wrong MQL negative controls are re-verified by verifier-3; all must be rejected.
-> outputs/sql2mql_v2/adversarial_v3/{suite.jsonl, negative.jsonl, summary.json}
"""
import json

import v2_common as C
import adversarial_v2 as A
import suite_v2 as SV
import schema_profile as SP
import sql2mql_v2 as M
import verifier3 as V3

OUT = C.OUT / "adversarial_v3"


def main():
    client = C.local_client()
    C.check_server(client)
    real = V3.RealDbs(client, path_of=A.db_path)
    cases = {c["id"]: c for c in A.all_cases()}
    suite = C.read_jsonl(A.OUT / "suite.jsonl")
    out = []
    for r in suite:
        if r["converter"] != "converted":
            continue
        case = cases[r["id"]]
        SV.build_db(case)
        dbid = A.db_id(case)
        mql, _ = M.convert(case["sql"], SP.Profile(dbid, A.db_path(dbid)))
        v = V3.verify({"id": r["id"], "db_id": dbid, "sql": case["sql"]}, mql, real)
        wrong = (not r["expect_reject"]) and not (r["explicit_null"]["correct"] and r["missing"]["correct"])
        out.append({"id": r["id"], "same_mql": C.compact(mql) == C.compact(r["mql"]), "wrong": wrong,
                    "correct": (not r["expect_reject"]) and not wrong,
                    "verifier2": r["verifier"], "verifier3": v["status"], "detail": v.get("detail")})
    neg_in = C.read_jsonl(A.OUT / "negative_controls_v1_mql.jsonl")
    neg = []
    for r in neg_in:
        case = cases[r["id"]]
        SV.build_db(case)
        v = V3.verify({"id": r["id"], "db_id": A.db_id(case), "sql": case["sql"]}, r["v1_mql"], real)
        neg.append({"id": r["id"], "verifier2": r["v2_verifier"], "verifier3": v["status"], "detail": v.get("detail")})
    OUT.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(OUT / "suite.jsonl", out)
    C.write_jsonl(OUT / "negative.jsonl", neg)
    summ = {
        "verifier": V3.VERIFIER_VERSION,
        "G2": {"converted_cases": len(out), "mql_identical_to_verifier2_run": sum(x["same_mql"] for x in out),
               "wrong": sum(x["wrong"] for x in out),
               "FALSE_ACCEPTS": [x["id"] for x in out if x["wrong"] and x["verifier3"] == "accepted"],
               "correct": sum(x["correct"] for x in out),
               "correct_accepted_v3": sum(x["correct"] and x["verifier3"] == "accepted" for x in out),
               "correct_accepted_v2_rejected_v3": {x["id"]: x["verifier3"] for x in out if x["correct"]
                                                   and x["verifier2"] == "accepted" and x["verifier3"] != "accepted"},
               "superset_violations (v2 rejected, v3 accepted)": [x["id"] for x in out if x["verifier2"] != "accepted"
                                                                 and x["verifier3"] == "accepted"]},
        "G3": {"controls": len(neg), "rejected": sum(x["verifier3"] != "accepted" for x in neg),
               "FALSE_ACCEPTS": [x["id"] for x in neg if x["verifier3"] == "accepted"]},
    }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
