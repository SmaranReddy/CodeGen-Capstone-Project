"""Approve the training corpus under the LOCAL VALIDATION POLICY (owner decision of 2026-11-16).

    python -I -B scripts/corpus_v2_signoff/approve_corpus.py check            # run every check, write nothing approved
    python -I -B scripts/corpus_v2_signoff/approve_corpus.py approve "<test-suite result line>"

`approve` writes outputs/sql2mql_v2_1/approved_corpus_v2_1/{corpus.jsonl, exclusions.jsonl, approval_checks.json,
manifest.json} only if EVERY check passes, and refuses to overwrite an existing approved directory. On any failure
it writes nothing and exits 1. Provisional corpus v2 and the candidate are read, never written.

Policy recorded in the manifest:
  training-label validation  local MongoDB 8.0.34 (verifier 3.1)
  Atlas                      evaluation environment and the only source of reported benchmark metrics
  corpus execution on Atlas  NOT PERFORMED (the 516 training collections are not on Atlas; accepted limitation)
AI-reviewed, never human-reviewed. The run01 human P2 gate is NOT passed.
"""
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "scripts/finetune_v2"
for p in (ROOT / "scripts", V2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SRC_DIR = ROOT / "outputs/sql2mql_v2_1/provisional_corpus_v2"
SRC = SRC_DIR / "provisional_rag_corpus.jsonl"
RECORDS = SRC_DIR / "records.jsonl"
CAND = ROOT / "outputs/sql2mql_v2_1/corpus_v2_1_candidate"
ADJ = ROOT / "data/interim/corpus_v2_adjudication_v1.jsonl"
POOL = ROOT / "data/processed/spider_pool_v3"
TC1 = ROOT / "data/interim/finetune_v2/type_contract_v1.json"
TC_EVAL = ROOT / "data/interim/finetune_v2/type_contract_eval_v1.json"
CODE_FREEZE = ROOT / "data/interim/finetune_v2/v2_1_code_freeze.json"
PANEL = ROOT / "outputs/ai_review_panel_v1"
ATLAS_AUDIT = ROOT / "outputs/corpus_v2_atlas_audit/run_20261116T183448"
PREFLIGHT = ROOT / "outputs/pilot_v3_followup/preflight"
RUN_LOG = ROOT / "outputs/sql2mql_v2_1/logs/s6_s11b_run.log"
HOLDOUT = ROOT / "data/splits_expanded/test.jsonl"
BASELINE = ROOT / "outputs/pilot_v3_followup/hash_baseline.json"
OUT = ROOT / "outputs/sql2mql_v2_1/approved_corpus_v2_1"

VERIFIER = "sql2mql-v2-verifier-3.1"
MONGO_VERSION = "8.0.34"
MATCH = {"multiset", "ordered match (tie-aware)"}
DETERMINISM = {"identical documents", "same result, different document order"}
CHECK_KEYS = {"v8_validate_mql", "v3_lookup_keys", "order", "real_rows", "l1_explicit_null", "v6_determinism",
              "l1_missing", "v3_reorder_reversed", "v3_reorder_shuffled_missing", "witness_rows", "fixtures",
              "v31_fold_traps"}
EXPECTED_ADJ = {"spider_train_00145": "QUARANTINE", "spider_train_00146": "QUARANTINE",
                "spider_train_05887": "INCLUDE", "spider_train_05888": "INCLUDE",
                "spider_train_05895": "INCLUDE", "spider_train_05896": "INCLUDE"}
SAMPLE_N, SAMPLE_SEED = 60, 20261116


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canon(o):
    o = json.loads(o) if isinstance(o, str) else o
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def jl(p):
    return [json.loads(x) for x in Path(p).open(encoding="utf-8") if x.strip()]


def record_problems(c, v, tc_sha, freeze_sha):
    """Problems with one corpus record `c` given its verification record `v` (None if absent). [] means validated."""
    if v is None:
        return ["no verification record"]
    out = []
    ver = v.get("verification")
    ver = json.loads(ver) if isinstance(ver, str) else (ver or {})
    ch = ver.get("checks") or {}
    if v.get("outcome") != "accepted" or v.get("conversion") != "converted":
        out.append(f"outcome {v.get('outcome')}/{v.get('conversion')}")
    if not (v.get("safety") or {}).get("ok"):
        out.append("safety gate not ok")
    if ver.get("verifier") != VERIFIER or ver.get("status") != "accepted":
        out.append(f"verifier {ver.get('verifier')} status {ver.get('status')}")
    if CHECK_KEYS - set(ch):
        out.append(f"missing checks {sorted(CHECK_KEYS - set(ch))}")
    for k in ("l1_explicit_null", "l1_missing", "v3_reorder_reversed", "v3_reorder_shuffled_missing"):
        if ch.get(k) not in MATCH:
            out.append(f"{k}={ch.get(k)!r}")
    if ch.get("v8_validate_mql") != "ok":
        out.append(f"v8_validate_mql={ch.get('v8_validate_mql')!r}")
    if ch.get("v6_determinism") not in DETERMINISM:
        out.append(f"v6_determinism={ch.get('v6_determinism')!r}")
    fx = ch.get("fixtures") or {}
    bad = {k: x for k, x in fx.items() if x not in MATCH and not str(x).startswith("skipped:")}
    if bad:
        out.append(f"fixture failures {sorted(bad)[:5]}")
    if not any(x in MATCH for x in fx.values()):
        out.append("no fixture family was compared")
    if canon(c["mql"]) != canon(v.get("mql")):
        out.append("corpus MQL differs from the verified MQL")
    if c["sql"] != v.get("sql") or c["db_id"] != v.get("db_id"):
        out.append("corpus SQL or db_id differs from the verified record")
    if (v.get("type_contract") or {}).get("manifest_sha256") != tc_sha or c["type_contract"]["manifest_sha256"] != tc_sha:
        out.append("type-contract hash differs")
    if v.get("v2_1_code_freeze_sha256") != freeze_sha or c.get("v2_1_code_freeze_sha256") != freeze_sha:
        out.append("code-freeze hash differs")
    return out


def holdout_stat():
    st = HOLDOUT.stat()
    rec = json.loads(BASELINE.read_text(encoding="utf-8"))["protected"]["data/splits_expanded/test.jsonl"]
    return {"opened": False, "matches_baseline": st.st_size == rec["size"] and st.st_mtime_ns == rec["mtime_ns"]}


def run_checks():
    checks, fail = {}, []

    def put(name, ok, **info):
        checks[name] = {"ok": bool(ok), **info}
        if not ok:
            fail.append(name)

    # 1. reproducibility: source corpus minus the documented quarantine == candidate, byte for byte
    src_man = json.loads((SRC_DIR / "manifest.json").read_text(encoding="utf-8"))
    cand_man = json.loads((CAND / "manifest.json").read_text(encoding="utf-8"))
    adj = jl(ADJ)
    decisions = {a["id"]: a["decision"] for a in adj}
    drop = {i for i, d in decisions.items() if d == "QUARANTINE"}
    kept_lines, dropped = [], []
    for line in SRC.open(encoding="utf-8", newline=""):
        (dropped if json.loads(line)["id"] in drop else kept_lines).append(line)
    body = "".join(kept_lines).encode("utf-8")
    body_sha = hashlib.sha256(body).hexdigest()
    put("source_corpus_unchanged", sha(SRC) == cand_man["source"]["corpus_sha256"]
        and sha(RECORDS) == src_man["inputs"]["records_sha256"],
        corpus_sha256=sha(SRC), records_sha256=sha(RECORDS))
    put("candidate_reproduced", body == (CAND / "corpus.jsonl").read_bytes()
        and body_sha == cand_man["files"]["corpus.jsonl"]["sha256"] and len(dropped) == len(drop) == 2,
        sha256=body_sha, records=len(kept_lines), quarantined=sorted(drop))

    # 2. adjudication is the documented one; no rewrite
    put("adjudication_as_documented", decisions == EXPECTED_ADJ
        and all(a["sql_rewritten"] is False and a["mql_rewritten"] is False for a in adj),
        decisions=decisions, sha256=sha(ADJ))
    corpus = [json.loads(x) for x in kept_lines]
    ids = [c["id"] for c in corpus]
    put("included_and_excluded_ids", len(set(ids)) == len(ids) and not (set(ids) & drop)
        and all(i in set(ids) for i, d in decisions.items() if d == "INCLUDE"), duplicates=len(ids) - len(set(ids)))

    # 3. every included record completed local validation with verifier 3.1
    tc_sha, freeze_sha = sha(TC1), sha(CODE_FREEZE)
    ver, dup = {}, 0
    for r in jl(RECORDS):
        dup += r["id"] in ver
        ver[r["id"]] = r
    problems = {}
    skipped_fx = compared_fx = 0
    for c in corpus:
        pr = record_problems(c, ver.get(c["id"]), tc_sha, freeze_sha)
        if pr:
            problems[c["id"]] = pr
        else:
            v = ver[c["id"]]["verification"]
            v = json.loads(v) if isinstance(v, str) else v
            fx = v["checks"]["fixtures"]
            skipped_fx += sum(str(x).startswith("skipped:") for x in fx.values())
            compared_fx += sum(x in MATCH for x in fx.values())
    put("local_validation_complete", not problems and not dup, validated=len(corpus) - len(problems),
        total=len(corpus), duplicate_verification_records=dup, problems=dict(list(problems.items())[:20]),
        fixture_comparisons=compared_fx, fixture_families_skipped_as_degenerate_or_drift=skipped_fx)

    # 4. code and contract freezes
    fz = json.loads(CODE_FREEZE.read_text(encoding="utf-8"))
    changed = [f for grp in ("new_modules", "frozen_v2_modules") for f, h in fz.get(grp, {}).items()
               if sha(ROOT / f) != h]
    r = subprocess.run([sys.executable, "-I", "-B", str(V2 / "freeze_v2_1_code.py"), "check"],
                       capture_output=True, text=True)
    put("code_freeze_intact", not changed and r.returncode == 0
        and freeze_sha == src_man["inputs"]["v2_1_code_freeze_sha256"]
        and tc_sha == src_man["inputs"]["type_contract_v1_sha256"],
        changed_modules=changed, v2_1_code_freeze_sha256=freeze_sha, type_contract_v1_sha256=tc_sha)

    # 5. provenance: every record comes from spider-pool-v3 train and is not excluded anywhere
    put("pool_v3_unchanged", sha(POOL / "train.jsonl") == src_man["inputs"]["spider_pool_v3_train_sha256"]
        and sha(POOL / "manifest.json") == src_man["inputs"]["spider_pool_v3_manifest_sha256"])
    pool = {r["id"]: r for r in jl(POOL / "train.jsonl")}
    excluded = set()
    for name in ("exclusions.jsonl", "exclusions_v3.jsonl", "quarantine.jsonl"):
        excluded |= {r.get("id") for r in jl(POOL / name)}
    prov_bad = [c["id"] for c in corpus if c["id"] not in pool or pool[c["id"]]["db_id"] != c["db_id"]
                or pool[c["id"]].get("sql", c["sql"]) != c["sql"] or not c.get("source", {}).get("origin")]
    in_excl = sorted(set(ids) & excluded)
    held = [r for r in jl(POOL / "exclusions_v3.jsonl")]
    put("provenance", not prov_bad and not in_excl, not_in_pool_or_mismatch=prov_bad[:10],
        in_an_exclusion_list=in_excl[:10], detector_or_panel_exclusions_listed=len(held))

    # 6. local MongoDB: exact version now, and enforced on every start of the conversion run
    try:
        import v2_common as C
        live = C.check_server(C.local_client())
    except Exception as e:  # noqa: BLE001
        live = f"unavailable: {type(e).__name__}"
    log = RUN_LOG.read_text(encoding="utf-8", errors="replace")
    put("local_mongodb_version", live == MONGO_VERSION and "8.0.34" in log, live_version=live,
        enforced_by="v2_common.check_server, called by run_pilot_v4.real_dbs before every (re)start of corpus-run",
        run_log_mentions_8_0_34=log.count("8.0.34"), finished_line="finished 5045" in log)

    # 7. live reproduction of a seeded sample with the frozen runner (local MongoDB)
    sample = sorted(random.Random(SAMPLE_SEED).sample(ids, SAMPLE_N)) + \
        sorted(i for i, d in decisions.items() if d == "INCLUDE")
    sample = list(dict.fromkeys(sample))
    mismatch = []
    try:
        import run_pilot_v4 as R4
        R4.freeze_check()
        mods = R4.load_modules()
        real, profs = R4.real_dbs(mods), {}
        for i in sample:
            rec = R4.process_v21(pool[i], mods, real, profs)
            if rec["outcome"] != "accepted" or canon(rec.get("mql")) != canon(ver[i]["mql"]):
                mismatch.append({"id": i, "outcome": rec["outcome"], "detail": rec.get("outcome_detail")})
        err = None
    except SystemExit as e:
        err = f"SystemExit: {e}"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:200]}"
    put("live_sample_reverification", err is None and not mismatch, sample=len(sample), seed=SAMPLE_SEED,
        reproduced=len(sample) - len(mismatch) if err is None else 0, mismatches=mismatch, error=err)

    # 8. frozen artifacts, holdout, Atlas evidence (read from earlier reports; Atlas is not contacted here)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/pilot_v3_followup/hash_baseline.py"), "verify", "approval"],
                       capture_output=True, text=True, cwd=ROOT)
    try:
        hb = json.loads(r.stdout[r.stdout.index("{"):])
    except ValueError:
        hb = {"ok": False, "raw": r.stdout[-300:]}
    put("frozen_artifacts", hb.get("ok") is True and not hb.get("snapshot_changed") and not hb.get("protected_changed"),
        recorded=hb.get("recorded_summary"))
    put("holdout_not_opened", holdout_stat()["matches_baseline"], **holdout_stat())
    audit = json.loads((ATLAS_AUDIT / "report.json").read_text(encoding="utf-8"))
    pre = sorted(PREFLIGHT.glob("atlas_type_contract_*.json"))
    pre_rep = json.loads(pre[-1].read_text(encoding="utf-8")) if pre else {}
    put("atlas_evidence_recorded_honestly", audit["status"] == "BLOCKED" and audit["attempted"] == 0
        and audit["outcome_counts"]["validated"] == 0 and audit["tc1_recheck"]["ok"] is True
        and audit["tc1_recheck"]["collections_checked"] == 70 and pre_rep.get("status") == "pass",
        audit_status=audit["status"], attempted=audit["attempted"], preflight=pre[-1].name if pre else None)
    panel = json.loads((PANEL / "panel_report.json").read_text(encoding="utf-8"))
    put("g_air_report_present", panel.get("panel_labels_usable") is True, sha256=sha(PANEL / "panel_report.json"))
    return checks, fail, {"body": body, "body_sha": body_sha, "corpus": corpus, "dropped": dropped, "adj": adj,
                          "src_man": src_man, "cand_man": cand_man, "audit": audit, "tc_sha": tc_sha,
                          "freeze_sha": freeze_sha, "held": held, "live": checks["local_mongodb_version"]["live_version"]}


def main(argv):
    cmd = argv[1]
    checks, fail, ctx = run_checks()
    print(json.dumps({k: v["ok"] for k, v in checks.items()}, indent=1))
    if fail:
        print("NOT APPROVED. Failed checks:", fail)
        print(json.dumps({k: checks[k] for k in fail}, indent=1, default=str)[:3000])
        return 1
    if cmd == "check":
        print("all checks pass; nothing written (use `approve`)")
        return 0
    tests = argv[2]
    if not tests.strip():
        raise SystemExit("approve needs the test-suite result line")
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite an approved artifact")
    OUT.mkdir(parents=True)
    (OUT / "corpus.jsonl").write_bytes(ctx["body"])
    drop = {a["id"]: a for a in ctx["adj"] if a["decision"] == "QUARANTINE"}
    excl = [{"id": json.loads(x)["id"], "db_id": json.loads(x)["db_id"], "provenance": json.loads(x)["source"],
             "reason": "nonconformant_field_type_dependent", "detail": drop[json.loads(x)["id"]]["reason"]}
            for x in ctx["dropped"]]
    (OUT / "exclusions.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in excl),
                                          encoding="utf-8")
    (OUT / "approval_checks.json").write_text(json.dumps(checks, indent=1, default=str), encoding="utf-8")
    audit, src_man = ctx["audit"], ctx["src_man"]
    man = {
        "corpus_version": "approved_corpus_v2_1",
        "status": "CONDITIONAL GO - APPROVED FOR TRAINING UNDER LOCAL VALIDATION POLICY",
        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "owner_decision": {
            "date": "2026-11-16",
            "decision": "Option 1: accept local MongoDB verification for training labels, under the existing project "
                        "policy that training labels may be validated locally while reported evaluation metrics "
                        "must come from MongoDB Atlas",
            "decided_by": "project owner (not an AI reviewer)"},
        "review_label": "AI-reviewed (G-AIR, AI-only protocol); NEVER human-reviewed. Human P2 of run01: NOT PASSED.",
        "validation_policy": {
            "training_label_validation": f"local MongoDB {MONGO_VERSION} (127.0.0.1:27018), verifier 3.1",
            "atlas_role": "evaluation environment and the only source of reported benchmark metrics",
            "atlas_preflight": "passed for the 70 evaluation collections (type contract; LIKE semantics)",
            "training_corpus_execution_on_atlas": "NOT PERFORMED",
            "accepted_limitation": "the 516 training collections read by the corpus are absent from Atlas and the "
                                   "account is read-only; this is an accepted infrastructure limitation under this "
                                   "policy, NOT a successful Atlas execution audit",
            "evaluation_metrics": "must continue to be generated against Atlas"},
        "builder": "scripts/corpus_v2_signoff/approve_corpus.py", "builder_sha256": sha(__file__),
        "counts": {"included": len(ctx["corpus"]), "source_records": len(ctx["corpus"]) + len(excl),
                   "quarantined_nonconformant_type_dependent": len(excl),
                   "detector_hit_records_held_out_of_the_pool": 667,
                   "databases": len({c["db_id"] for c in ctx["corpus"]})},
        "files": {"corpus.jsonl": {"sha256": ctx["body_sha"], "records": len(ctx["corpus"])},
                  "exclusions.jsonl": {"sha256": sha(OUT / "exclusions.jsonl"), "records": len(excl)},
                  "approval_checks.json": {"sha256": sha(OUT / "approval_checks.json")}},
        "identical_to_candidate": {"path": "outputs/sql2mql_v2_1/corpus_v2_1_candidate/corpus.jsonl",
                                   "sha256": ctx["cand_man"]["files"]["corpus.jsonl"]["sha256"]},
        "source": {"corpus": SRC.relative_to(ROOT).as_posix(), "corpus_sha256": sha(SRC),
                   "manifest_sha256": sha(SRC_DIR / "manifest.json"), "records_sha256": sha(RECORDS)},
        "reproduce": "drop the ids in exclusions.jsonl from the source corpus; keep every other line byte for byte",
        "adjudication": {"file": ADJ.relative_to(ROOT).as_posix(), "sha256": sha(ADJ),
                         "rule": "quarantine if a type-dependent operation reads the nonconformant field; include "
                                 "if it is only copied to the output",
                         "decisions": {a["id"]: a["decision"] for a in ctx["adj"]},
                         "sql_or_mql_rewritten": False},
        "converter": src_man["converter"], "verifier": src_man["verifier"],
        "hashes": {"v2_1_code_freeze_sha256": ctx["freeze_sha"],
                   "pilot_v4_runner_freeze_sha256": src_man["inputs"]["pilot_v4_runner_freeze_sha256"],
                   "type_contract_v1_sha256": ctx["tc_sha"], "type_contract_eval_v1_sha256": sha(TC_EVAL),
                   "spider_pool_v3_manifest_sha256": sha(POOL / "manifest.json"),
                   "spider_pool_v3_train_sha256": sha(POOL / "train.jsonl")},
        "local_validation": {
            "mongodb_version_enforced": MONGO_VERSION, "mongodb_version_at_approval": ctx["live"],
            "records_validated": checks["local_validation_complete"]["validated"],
            "records_total": checks["local_validation_complete"]["total"],
            "fixture_comparisons": checks["local_validation_complete"]["fixture_comparisons"],
            "fixture_families_skipped_as_degenerate_or_drift":
                checks["local_validation_complete"]["fixture_families_skipped_as_degenerate_or_drift"],
            "live_sample_reverification": {k: checks["live_sample_reverification"][k]
                                           for k in ("sample", "seed", "reproduced")},
            "establishes": "SQL-to-MQL result equivalence under TC-1 on the real database and fixtures, locally; "
                           "NOT that the SQL answers the question"},
        "atlas": {"audit_run": ATLAS_AUDIT.relative_to(ROOT).as_posix(), "audit_report_sha256": sha(ATLAS_AUDIT / "report.json"),
                  "audit_status": audit["status"], "corpus_records_executed_on_atlas": 0,
                  "evaluation_collections_rechecked": audit["tc1_recheck"]["collections_checked"],
                  "violations": audit["tc1_recheck"]["violations"]},
        "g_air": {"status": "passed under its AI-only procedure (translation 0 NO + 0 UNSURE of 200)",
                  "human_p2": "NOT PASSED (never run)",
                  "report": "outputs/ai_review_panel_v1/panel_report.json", "report_sha256": sha(PANEL / "panel_report.json"),
                  "protocol_note": "docs/g_air_protocol_decision_2026-11-16.md",
                  "source_label_observation": "7 of the 200 sampled labels were flagged (6 NO, 1 UNSURE) and "
                                              "quarantined; an observed result, not a corpus-wide error-rate estimate",
                  "limitations": [
                      "AI-reviewed, not human-reviewed",
                      "all reviewers are one model family; agreement is not independent evidence",
                      "reviewer isolation was partial",
                      "the stability run involved a model change",
                      "calibration (60 / 60 planted errors) does not prove that unknown defects are detected",
                      "the 200-item sample does not prove a population error rate below 5%",
                      "corpus records were not individually reviewed"]},
        "detector_quarantine": {"records_held": 667, "listed_in": "data/processed/spider_pool_v3/exclusions_v3.jsonl",
                                "note": "conservative exclusion; low detector precision; NOT confirmed defects; "
                                        "not reintroduced; provenance kept for a later recovery audit"},
        "test_suite": tests,
        "fine_tuning": {"authorized": True, "scope": "training on this corpus only; evaluation metrics must be "
                        "produced on Atlas; the protected 65-item holdout stays unused until explicitly requested",
                        "started_by_this_task": False},
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": man["status"], "path": (OUT / "corpus.jsonl").relative_to(ROOT).as_posix(),
                      "records": len(ctx["corpus"]), "sha256": ctx["body_sha"],
                      "manifest_sha256": sha(OUT / "manifest.json")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
