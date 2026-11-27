"""Round-3 pre-registration, STAGE 2: freeze the code under test once verifier-3 is final.

    python -B scripts/finetune_v2/freeze_round3_code.py

Writes data/interim/finetune_v2/round3_code_freeze.json (refuses to overwrite) with the SHA-256
of every module that decides or judges round 3, the stage-1 file hash, the MongoDB image
digest and a timestamp. `check_frozen()` (used by run_v3.py round3, mutant_study_v3.py for the
round-3 output and round3_report.py) refuses to run if any hash differs.
"""
import json
import sys
import time

import v2_common as C

PREREG = C.INTERIM / "round3_prereg.json"
CODE_FREEZE = C.INTERIM / "round3_code_freeze.json"
FROZEN = (
    # converter (unchanged since Phase 1)
    "sql2mql_v2.py", "schema_profile.py",
    # verifier-3 and the verifier-2 code it runs
    "verifier3.py", "fixtures3.py", "verifier.py", "v2_fixtures.py", "comparator.py",
    "mongo_import.py", "index_real.py", "v2_common.py",
    # judge and negative controls (unchanged since iteration 1)
    "oracle_v2.py", "mutants_v2.py",
    # round-3 procedure
    "run_v3.py", "mutant_study_v3.py", "round3_stats.py", "round3_report.py", "freeze_round3.py",
    "freeze_round3_code.py",
)
EXTERNAL = ("scripts/evaluation/eval_kit.py",)


def code_hashes():
    h = {p: C.sha256_file(C.HERE / p) for p in FROZEN}
    h.update({p: C.sha256_file(C.ROOT / p) for p in EXTERNAL})
    return h


def check_frozen():
    """-> the stage-1 pre-registration dict, after verifying both stages."""
    if not PREREG.exists() or not CODE_FREEZE.exists():
        sys.exit("round 3 is not fully pre-registered (stage 1 and stage 2 are both required)")
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    body = {k: v for k, v in pre.items() if k != "sha256_of_body_without_this_field"}
    if C.sha256_bytes(json.dumps(body, indent=1, ensure_ascii=False).encode("utf-8")) != \
            pre["sha256_of_body_without_this_field"]:
        sys.exit("stage-1 pre-registration was modified")
    fz = json.loads(CODE_FREEZE.read_text(encoding="utf-8"))
    if fz["prereg_sha256"] != pre["sha256_of_body_without_this_field"]:
        sys.exit("stage-2 freeze does not belong to this stage-1 file")
    now = code_hashes()
    bad = [p for p, h in fz["code_sha256"].items() if now.get(p) != h]
    if bad:
        sys.exit(f"frozen code changed since stage 2: {bad}")
    return pre


def main():
    if CODE_FREEZE.exists():
        sys.exit(f"{CODE_FREEZE} exists: refusing to overwrite")
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    import verifier3 as V3
    now = code_hashes()
    # the pre-freeze evidence must have been produced by exactly these verifier-side bytes
    vside = ("verifier3.py", "fixtures3.py", "index_real.py", "verifier.py", "v2_fixtures.py",
             "comparator.py", "mongo_import.py", "oracle_v2.py", "mutants_v2.py")
    evidence = {}
    for name in ("dev3", "diag_v3_insample"):
        summ = json.loads((C.OUT / name / "summary.json").read_text(encoding="utf-8"))
        rec = summ["code_sha256"]
        bad = [p for p in vside if p in rec and rec[p] != now[p]]
        if bad:
            sys.exit(f"{name} was produced by different verifier code {bad}: re-run it before freezing")
        evidence[name] = {"summary_sha256": C.sha256_file(C.OUT / name / "summary.json"),
                          "verifier_side_hashes_match": sorted(p for p in vside if p in rec)}
    for name in ("adversarial_v3", "synthetic_v3"):
        evidence[name] = {"summary_sha256": C.sha256_file(C.OUT / name / "summary.json"),
                          "note": "summary records no code hashes; produced by the verifier-side files above "
                                  "(no verifier-side file changed after these runs: dev3 and diag_v3_insample, "
                                  "run later, match)"}
    doc = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": 2,
        "prereg_sha256": pre["sha256_of_body_without_this_field"],
        "verifier": V3.VERIFIER_VERSION,
        "mongo": {"image": C.MONGO_IMAGE, "digest": C.MONGO_DIGEST, "host": f"{C.LOCAL_HOST}:{C.LOCAL_PORT}"},
        "code_sha256": now,
        "pre_freeze_evidence": evidence,
        "go_no_go": "verifier-3 in-sample mutant study (200 diagnosis labels, NOT a gate measurement): "
                    "0 labels with a false accept; verifier-2 on the same labels: 3",
        "note": "Any change to a listed file after this timestamp makes run_v3.py round3, the round-3 "
                "mutant study and round3_report.py refuse to run.",
    }
    CODE_FREEZE.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
