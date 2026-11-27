"""Round-3 G4 result, computed exactly as pre-registered (round3_stats.weighted_g4).

    python -B scripts/finetune_v2/round3_report.py
Reads outputs/sql2mql_v2/round3/records.jsonl and outputs/sql2mql_v2/round3_mutants/ and writes
outputs/sql2mql_v2/round3_report.json. Refuses to run if the frozen code changed.
Gate figures: G4 (weighted, as pre-registered). Reported, not gated: unweighted and per-stratum
rates, per-operator table, kills split into semantic / structural / non-semantic, truncation at
the 30-mutant cap per stratum, unkilled survivors per stratum x operator and a seeded sample of
20 of them for manual equivalence adjudication (the oracle-judged rate is a lower bound).
"""
import json
import random
from collections import Counter, defaultdict

import v2_common as C
import round3_stats as S
import freeze_round3_code as FZ
import mutants_v2 as MU

NON_SEMANTIC_KILLS = ("mongo_timeout", "too_few_fixtures", "sqlite_error", "mongo_error", "order_plan")
STRUCTURAL_KILLS = ("v3_lookup_keys", "structure", "unsafe_mql")
ADJUDICATION_N, ADJUDICATION_SEED = 20, 20261102


def main():
    pre = FZ.check_frozen()
    N, n = pre["strata"]["N_h"], pre["sampling"]["n_h_records"]
    recs = C.read_jsonl(C.OUT / "round3" / "records.jsonl")
    sample = {i for h in S.STRATA for i in pre["sampling"]["ids"][h]}
    assert {r["id"] for r in recs} == sample and len(recs) == len(sample)
    acc = {r["id"]: r for r in recs if r.get("verification", {}).get("status") == "accepted"}
    listed = set(json.loads((C.OUT / "round3" / "accepted_labels.json").read_text(encoding="utf-8"))["ids"])
    assert listed == set(acc)
    mdir = C.OUT / "round3_mutants"
    rows = C.read_jsonl(mdir / "mutants.jsonl")
    nomut = {x["label"] for x in C.read_jsonl(mdir / "no_mutants.jsonl")}
    done = {x["label"] for x in rows} | nomut
    assert done == set(acc), f"{len(set(acc) - done)} accepted labels without a mutant result"
    bad = {x["label"] for x in rows if x["verdict"] == "non_equivalent"}
    labels = [(acc[i]["stratum"], int(i in bad)) for i in sorted(acc)]
    res = S.weighted_g4(labels, N, n)
    by_op = defaultdict(Counter)
    for x in rows:
        by_op[x["operator"]][x["verdict"]] += 1
    esc = defaultdict(Counter)
    for x in rows:
        if x["verdict"] == "non_equivalent":
            esc[acc[x["label"]]["stratum"]][x["operator"]] += 1
    trunc = defaultdict(lambda: {"labels_over_cap": 0, "operators_cut": Counter()})
    for i, r in acc.items():
        allm = MU.mutants(r["mql"])
        if len(allm) > 30:
            trunc[r["stratum"]]["labels_over_cap"] += 1
            trunc[r["stratum"]]["operators_cut"].update(op for op, _ in allm[30:])
    kills = Counter(x["verifier"] for x in rows if x["verdict"] == "killed")
    unk = [x for x in rows if x["verdict"] == "unkilled"]
    unk_tab = defaultdict(Counter)
    for x in unk:
        unk_tab[acc[x["label"]]["stratum"]][x["operator"]] += 1
    adj = sorted(random.Random(ADJUDICATION_SEED).sample(range(len(unk)), min(ADJUDICATION_N, len(unk))))
    out = {
        "prereg_sha256": pre["sha256_of_body_without_this_field"],
        "records": len(recs),
        "records_by_stratum": {h: {"sampled": n[h], "population": N[h],
                                   "converted": sum(r["stratum"] == h and r["conversion"] == "converted" for r in recs),
                                   "accepted": sum(r["stratum"] == h for r in acc.values())} for h in S.STRATA},
        "labels_evaluated": len(labels), "labels_without_mutants": len(nomut),
        "mutants": len(rows),
        "killed_by_verifier": sum(x["verdict"] == "killed" for x in rows),
        "survivors_non_equivalent (false accepts)": sum(x["verdict"] == "non_equivalent" for x in rows),
        "survivors_unkilled_by_both": sum(x["verdict"] == "unkilled" for x in rows),
        "G4": res,
        "false_accept_labels": sorted(bad),
        "escapes_by_stratum_operator": {h: dict(c) for h, c in esc.items()},
        "by_operator": {op: dict(c) for op, c in sorted(by_op.items())},
        "kill_reasons": dict(kills.most_common()),
        "kills_split (reported, not gated)": {
            "semantic (result differs on real data / fixtures, nondeterminism)":
                sum(v for k, v in kills.items() if k not in NON_SEMANTIC_KILLS + STRUCTURAL_KILLS),
            "structural (v3_lookup_keys, structure, unsafe_mql)": sum(kills[k] for k in STRUCTURAL_KILLS),
            "non-semantic (timeouts, too few fixtures, engine errors)": sum(kills[k] for k in NON_SEMANTIC_KILLS)},
        "truncation_at_30_mutants_by_stratum": {h: {"labels_over_cap": v["labels_over_cap"],
                                                   "operators_cut": dict(v["operators_cut"])}
                                               for h, v in trunc.items()},
        "unkilled_survivors_by_stratum_operator (reported, not gated)": {h: dict(c) for h, c in unk_tab.items()},
        "adjudication_sample (seeded; for manual equivalence review, reported not gated)":
            [{"label": unk[i]["label"], "operator": unk[i]["operator"], "stratum": acc[unk[i]["label"]]["stratum"]}
             for i in adj],
    }
    (C.OUT / "round3_report.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("labels_evaluated", "mutants", "G4", "escapes_by_stratum_operator")},
                     indent=1))


if __name__ == "__main__":
    main()
