"""Round-3 G4 pre-registration, STAGE 1 (sample + criteria), written BEFORE verifier-3 exists.

    python -B scripts/finetune_v2/freeze_round3.py

Writes data/interim/finetune_v2/round3_prereg.json (refuses to overwrite):
  * population: the frozen pilot population (pilot_freeze.json, tier exec_ok, not leaked)
    minus the 300 PILOT and 500 DEV ids  -> 5,229 records
  * strata (round3_stats.stratum, from SQL text only) with population counts N_h
  * a stratified simple random sample WITHOUT replacement of n_h records per stratum
    (seed 20261031, ids sorted before sampling), inclusion probability pi_h = n_h / N_h
  * the frozen statistic, gate, stop rule and a power table
Stage 2 (freeze_round3_code.py) adds the code hashes once verifier-3 is final; run_v3.py
refuses to run round 3 unless both stages exist and every hash matches.
Reads only train.jsonl and pilot_freeze.json. No converter or verifier is imported.
"""
import json
import random
import sys
import time
from collections import Counter

import v2_common as C
import round3_stats as S

SEED = 20261031
ALLOC = {"J2": 130, "OL": 130, "J1": 130, "G": 350}     # records, not labels
PREREG = C.INTERIM / "round3_prereg.json"
PLANNING_ACCEPT = {"G": 0.767, "J1": 0.762, "OL": 0.543, "J2": 0.674}   # verifier-2 on DEV (planning only)


def main():
    if PREREG.exists():
        sys.exit(f"{PREREG} exists: round 3 is pre-registered; refusing to overwrite")
    fr = json.loads(C.FREEZE.read_text(encoding="utf-8"))
    src = C.TRAIN
    assert C.sha256_file(src) == fr["source"]["sha256"]
    used = set(fr["pilot"]["ids"]) | set(fr["dev"]["ids"])
    rows = C.read_jsonl(src)
    leaked = set(fr["v9_leakage_screen"].get("leaked_ids", []) or [])
    pop = [r for r in rows if r["tier"] == "exec_ok" and r["id"] not in leaked]
    assert len(pop) == fr["population"]["records"] == 6029, len(pop)
    pop = sorted((r for r in pop if r["id"] not in used), key=lambda r: r["id"])
    assert len(pop) == 5229, len(pop)
    strata = {h: [] for h in S.STRATA}
    for r in pop:
        strata[S.stratum(r["sql"])].append(r["id"])
    N = {h: len(v) for h, v in strata.items()}
    rng = random.Random(SEED)
    sample = {}
    for h in S.STRATA:
        sample[h] = sorted(rng.sample(strata[h], ALLOC[h]))
    ids = [i for h in S.STRATA for i in sample[h]]
    assert len(set(ids)) == len(ids) == sum(ALLOC.values())
    assert not set(ids) & used
    hard = Counter(r["hardness"] for r in pop if r["id"] in set(ids))
    exp_labels = [(h, 0) for h in S.STRATA for _ in range(round(ALLOC[h] * PLANNING_ACCEPT[h]))]
    plan = S.weighted_g4(exp_labels, N, ALLOC)
    n_eff = plan["weighted"]["n_eff"]
    doc = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": 1,
        "purpose": "Round-3 independent re-measurement of Phase 1 gate G4 (verifier-3), pre-registered "
                   "before verifier-3 code exists",
        "script": "scripts/finetune_v2/freeze_round3.py",
        "source": {"path": "data/processed/spider_sql_v1/train.jsonl", "sha256": fr["source"]["sha256"]},
        "pilot_freeze_sha256": fr["freeze_sha256"],
        "population": {"rule": "pilot_freeze population (tier exec_ok, not leaked) minus PILOT (300) and "
                               "DEV (500) ids", "records": len(pop)},
        "strata": {
            "definition": "round3_stats.stratum(sql): top-level SELECT parsed by sqlglot (sqlite); "
                          "precedence J2 > OL > J1 > G",
            "J2": ">= 2 JOIN clauses (multi-table join, >= 3 tables)",
            "OL": "ORDER BY with LIMIT (includes LIMIT 1), < 2 joins",
            "J1": "exactly 1 JOIN, no ORDER BY ... LIMIT",
            "G": "everything else (single table, nested queries, set operations, unparsable)",
            "N_h": N,
        },
        "sampling": {
            "design": "stratified simple random sampling without replacement; ids sorted within stratum; "
                      "random.Random(seed).sample per stratum in order " + " > ".join(S.STRATA),
            "seed": SEED, "n_h_records": ALLOC,
            "inclusion_probability": {h: round(ALLOC[h] / N[h], 5) for h in S.STRATA},
            "design_weight_w_h": {h: round(N[h] / ALLOC[h], 4) for h in S.STRATA},
            "oversampled": "J2, OL and J1 are oversampled relative to G; the raw (unweighted) false-accept "
                           "percentage of this sample is NOT the corpus-wide rate",
            "by_hardness": dict(sorted(hard.items())),
            "ids": sample,
        },
        "procedure": [
            "1. run_v3.py round3: convert every sampled record with sql2mql-v2 (unchanged) and verify with "
            "verifier-3 (local MongoDB 8.0.34 only); unsupported / rejected records are reported, not evaluated",
            "2. mutant_study_v3.py --verifier verifier3 on EVERY accepted label of the sample (no sub-sampling), "
            "operators mutants_v2.py (unchanged, max 30 per label), judge oracle_v2.py (unchanged)",
            "3. round3_report.py: estimator below; per-stratum results",
            "Each step runs ONCE. A run interrupted by the OS / memory reaper is resumed with identical "
            "frozen code (append-only, per-label); the interruption is logged and reported",
        ],
        "statistic": {
            "unit": "accepted label",
            "y": "1 if >= 1 verifier-3-accepted mutant is shown non-equivalent by the oracle battery",
            "estimator": "p_w = sum(w_h y_i)/sum(w_h) over accepted labels, w_h = N_h/n_h (ratio estimator; "
                         "estimates the share among the labels the population would yield)",
            "n_eff": "Kish (sum w)^2 / sum w^2",
            "interval": "Wilson 95% (z=1.96) on (p_w*n_eff, n_eff), the rounds 1-2 method "
                        "(v2_common.wilson) with Kish effective n; equal weights give exactly k/n Wilson",
            "also_reported": ["unweighted k/n with Wilson (NOT the corpus-wide rate)",
                              "per-stratum k/n with Wilson", "per-operator kill table",
                              "records converted / accepted per stratum"],
        },
        "gate": {
            "G4_round3": "PASS iff the weighted Wilson 95% upper bound (unrounded) <= 0.05",
            "threshold_fixed": True,
            "no_change_after_results": "threshold, strata, weights, operators, oracle, label set and "
                                       "estimator are fixed by this file and stage 2",
        },
        "stop_rule": "One run. If G4_round3 fails: Phase 1 is NOT passed; stop and report; no verifier "
                     "change is re-measured on this sample; a further round needs a new pre-registered "
                     "sample from records never used.",
        "power_planning": {
            "assumed_acceptance (verifier-2 on DEV)": PLANNING_ACCEPT,
            "expected_accepted_labels": len(exp_labels),
            "expected_n_eff": n_eff,
            "max_weighted_escapes_passing": round(S.max_k_passing(n_eff), 2),
            "pass_probability_if_true_rate": {str(p): S.pass_probability(n_eff, p)
                                              for p in (0.01, 0.015, 0.02, 0.025, 0.03, 0.0435)},
        },
        "never_used": {
            "pilot (300)": "excluded",
            "dev (500), incl. the iteration-2 G4 holdout (161 labels)": "excluded from the population; "
                "the holdout's mutants.jsonl was never opened after its run; its class-level summary and "
                "the failing-label id list printed in its log were seen (2026-10-30/31), not used",
            "MQL 892-item eval set and 65-item test partition": "not read",
        },
        "diagnosis_data_for_verifier3": "ONLY the 200 iteration-1 diagnosis labels (DEV, "
                                        "outputs/sql2mql_v2/mutant_study/diagnosis_labels.json) and "
                                        "hand-written synthetic cases",
    }
    body = json.dumps(doc, indent=1, ensure_ascii=False)
    doc["sha256_of_body_without_this_field"] = C.sha256_bytes(body.encode("utf-8"))
    PREREG.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("population", "power_planning")}, indent=1))
    print("N_h", N, "sha", doc["sha256_of_body_without_this_field"])


if __name__ == "__main__":
    main()
