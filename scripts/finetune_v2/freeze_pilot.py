"""Freeze the sql2mql-v2 pilot BEFORE any converter or verifier output exists.

Writes data/interim/finetune_v2/pilot_freeze.json (refuses to overwrite) holding:
  * the source file hash and the V9 leakage screen over ALL 7,773 train records:
      - database disjointness from the 20 Spider-dev databases (db ids from the ID-only split
        manifest + the MQL-0006 bundle manifest; no question text of the held-out test
        partition is read: data/splits_expanded/test.jsonl is never opened);
      - character-trigram Jaccard of every train question against the 892 evaluation
        questions (bundle v2 eval_records.jsonl); >= 0.8 is a leak;
  * the pilot population: tier `exec_ok` (gold executed on the official, populated SQLite
    database), not leaked. The 1,744 records on the seven all-empty databases (test-suite
    variant / empty-DB tiers) are outside the pilot population;
  * PILOT: 300 records, seeded, stratified by Spider hardness (proportional allocation);
  * DEV: 500 records from the rest of the population (same strata), the ONLY Spider items
    the converter may be developed against; the pilot ids are refused by dev tooling;
  * the frozen acceptance criteria (Phase 1 gate and pilot gate).

Not filtered by convertibility: unsupported queries stay in the pilot and count as rejections.

Usage: python scripts/finetune_v2/freeze_pilot.py
"""
import json
import random
import re
import sys
import time
from collections import Counter

import v2_common as C

SEED = 20261030
N_PILOT, N_DEV = 300, 500
LEAK_THRESHOLD = 0.8
BUNDLE = C.ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2"
SPLIT_MANIFEST = C.ROOT / "data/splits_expanded/split_manifest.json"

CRITERIA = {
    "phase1_gate (roadmap P4: F1-F15 + V1-V3 + V8 on fixtures/dev)": {
        "G1_coverage_matrix": "every F1-F15 item and every audit defect id (A1 A1b A2 A6 A6b W10d "
                              "W10e W14 W17 O4 O4c O5 D4 J4 J4b J6 J8 N1 N3 C4@missing, DISTINCT+GROUP "
                              "BY, date subtraction, identifier case, implicit cast, int-vs-text join) "
                              "is 'implemented' or 'rejected' AND has a passing regression test",
        "G2_V1_adversarial": "0 false accepts: no case where the verifier accepts an MQL whose result "
                             "differs from the hand-specified (or documented-SQLite) expectation",
        "G3_negative_controls": "verifier rejects 100% of the v1 converter's wrong MQL on the audit's "
                                "20 wrong cases (the subset still expressible) and 100% of the "
                                "hand-written wrong MQL controls",
        "G4_V2_mutants": "on up to 200 accepted DEV labels: labels with >=1 verifier-accepted mutant "
                         "that the stronger oracle battery shows non-equivalent; Wilson 95% upper "
                         "bound <= 0.05",
        "G5_V3_fixture_families": "all nine audit families present (empty tables, zero-match WHERE, "
                                  "all-NULL aggregate columns, duplicate rows, LIKE case variants, "
                                  "distinct values per GROUP BY key, zero divisors, text-typed "
                                  "numerics, newline strings) and unit-tested",
        "G6_V8_safety": "100% of accepted labels pass eval_kit.validate_mql unchanged",
    },
    "pilot_gate (Phase 2; measured on the 300 PILOT ids only)": {
        "P1_equivalence": "every ACCEPTED pilot label passes: L1 on the real database under explicit "
                          "null AND absent-field NULL (V4), every fixture family, determinism by "
                          "double execution (V6), no ORDER BY ... LIMIT tie cut (V5), V8, V9",
        "P2_manual_review": "stratified random sample of min(120, accepted) accepted labels reviewed "
                            "(question <-> SQL <-> MQL); wrong-label Wilson 95% upper bound <= 0.05",
        "P3_no_systematic_error": "no error pattern found >= 2 times in the review",
        "P4_no_open_critical_bug": "every verifier-rejected or failed pilot item has a classified "
                                   "cause; no accepted label is known wrong",
        "reported_not_gated": ["coverage (converted / attempted)", "rejection reasons",
                               "execution success", "semantic equivalence rate", "per-pattern rates"],
    },
    "never_counted_as_correct": "successful execution alone; fixture equivalence alone",
}


def trigrams(s):
    s = " " + " ".join(re.findall(r"[a-z0-9]+", s.lower())) + " "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def main():
    if C.FREEZE.exists():
        sys.exit(f"{C.FREEZE} exists: the pilot is frozen; refusing to overwrite")
    rows = C.read_jsonl(C.TRAIN)
    assert len(rows) == 7773, len(rows)
    dev_dbs = set(json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))["database_assignments"])
    bman = json.loads((BUNDLE / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert C.sha256_file(BUNDLE.with_suffix(".zip")) == \
        "e9986dbef13cab8e1e047465b04ff8593d7b54f700cca4bcd1ed28f80d3b0760"
    dev_dbs |= set(bman["db_ids"])
    assert len(dev_dbs) == 20, len(dev_dbs)
    ev_q = [json.loads(x)["question"] for x in
            (BUNDLE / "data/eval_records.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(ev_q) == 892
    ev_tri = [trigrams(q) for q in ev_q]
    by_tri = {}
    for i, t in enumerate(ev_tri):
        for g in t:
            by_tri.setdefault(g, []).append(i)

    leak, db_leak = {}, []
    for r in rows:
        if r["db_id"] in dev_dbs:
            db_leak.append(r["id"])
        t = trigrams(r["question"])
        hits = Counter(i for g in t for i in by_tri.get(g, ()))
        best = max((h / len(t | ev_tri[i]) for i, h in hits.items()), default=0.0)
        if best >= LEAK_THRESHOLD:
            leak[r["id"]] = round(best, 4)
    if db_leak:
        sys.exit(f"FAIL: {len(db_leak)} train records on Spider-dev databases")

    pop = [r for r in rows if r["tier"] == "exec_ok" and r["id"] not in leak]
    strata = {}
    for r in sorted(pop, key=lambda r: r["id"]):
        strata.setdefault(r["hardness"], []).append(r["id"])
    rng = random.Random(SEED)
    pilot, dev = [], []
    for h in sorted(strata):
        ids = strata[h][:]
        rng.shuffle(ids)
        k_p = round(N_PILOT * len(ids) / len(pop))
        k_d = round(N_DEV * len(ids) / len(pop))
        pilot += ids[:k_p]
        dev += ids[k_p:k_p + k_d]
    # rounding repair to exact sizes (deterministic: largest strata first)
    assert abs(len(pilot) - N_PILOT) <= 2 and abs(len(dev) - N_DEV) <= 2, (len(pilot), len(dev))
    pilot, dev = sorted(pilot), sorted(dev)
    assert not set(pilot) & set(dev)
    hard = {r["id"]: r["hardness"] for r in rows}
    dbs = {r["id"]: r["db_id"] for r in rows}
    body = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "scripts/finetune_v2/freeze_pilot.py",
        "seed": SEED,
        "source": {"path": "data/processed/spider_sql_v1/train.jsonl", "sha256": C.sha256_file(C.TRAIN),
                   "records": len(rows)},
        "v9_leakage_screen": {
            "dev_databases": sorted(dev_dbs), "train_records_on_dev_databases": 0,
            "trigram_threshold": LEAK_THRESHOLD, "compared_against": "892 MQL-0006 evaluation questions",
            "not_read": "data/splits_expanded/test.jsonl and every question on the 3 held-out test DBs",
            "leaked_ids": leak},
        "population": {"rule": "tier == exec_ok (official populated DB) and not leaked",
                       "records": len(pop), "by_hardness": {h: len(v) for h, v in sorted(strata.items())},
                       "outside_population": dict(Counter(r["tier"] for r in rows if r["tier"] != "exec_ok"))},
        "pilot": {"n": len(pilot), "by_hardness": dict(Counter(hard[i] for i in pilot)),
                  "databases": len({dbs[i] for i in pilot}), "ids": pilot},
        "dev": {"n": len(dev), "by_hardness": dict(Counter(hard[i] for i in dev)),
                "databases": len({dbs[i] for i in dev}), "ids": dev},
        "acceptance_criteria": CRITERIA,
    }
    body["freeze_sha256"] = C.sha256_bytes(C.canon(body).encode("utf-8"))
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    C.FREEZE.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: body[k] for k in ("population",)} | {
        "pilot_n": len(pilot), "dev_n": len(dev), "leaked": len(leak),
        "freeze_sha256": body["freeze_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
