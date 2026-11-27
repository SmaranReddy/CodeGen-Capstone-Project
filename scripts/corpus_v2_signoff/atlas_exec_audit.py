"""Atlas execution audit of provisional corpus v2 (sign-off task, section 5). READ-ONLY on Atlas.

    python -I -B scripts/corpus_v2_signoff/atlas_exec_audit.py

Writes a new directory outputs/corpus_v2_atlas_audit/run_<timestamp>/ (never an existing one):
    report.json        run identity, input hashes, backend checks, TC-1 re-check, outcome counts, blocker
    per_record.jsonl   one row per corpus record: safety gate, referenced collections, outcome
    tc1_recheck.json   the live type-contract census of the Atlas evaluation collections

What "validated on Atlas" would need, and why this script can stop short of it:
  verifier 3.1 compares the label's MQL result with the SQLite reference on (a) the real database imported
  into MongoDB and (b) 47 generated fixture families. Both need the record's collections to exist in the
  MongoDB being tested, and (b) needs write access to create fixture collections. This script therefore
  first establishes, per record, whether every collection the label reads exists on Atlas. A record whose
  collections are absent is NOT executed and is reported as `not_executable_collection_absent_on_atlas`;
  it is never counted as validated. If any record were executable, the run stops with status INCOMPLETE,
  because running verifier 3.1 against Atlas is not implemented here and must not be approximated.

Commands sent to Atlas: ping, buildInfo, connectionStatus, listCollections, and the `$group`/`$type`
census aggregates of type_contract.mongo_type_census. No write command, no `$out`/`$merge`.
The connection string comes from MONGODB_URI through scripts/mongo_conn.py and is never printed or stored.
The protected holdout (data/splits_expanded/test.jsonl) is only stat()-ed, never opened.
"""
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "scripts", ROOT / "scripts/nl2mql", ROOT / "scripts/finetune_v2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import atlas_type_contract_check as A  # noqa: E402
import dev1034_core as D  # noqa: E402
import type_contract as TC  # noqa: E402
import v2_common as C  # noqa: E402

CORPUS_DIR = ROOT / "outputs/sql2mql_v2_1/provisional_corpus_v2"
CORPUS = CORPUS_DIR / "provisional_rag_corpus.jsonl"
ADJ = ROOT / "data/interim/corpus_v2_adjudication_v1.jsonl"
HOLDOUT = ROOT / "data/splits_expanded/test.jsonl"
BASELINE = ROOT / "outputs/pilot_v3_followup/hash_baseline.json"
OUT_ROOT = ROOT / "outputs/corpus_v2_atlas_audit"
WRITE_STAGES = {"$out", "$merge"}
JS_OPERATORS = {"$where", "$function", "$accumulator"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def operators(node, acc=None):
    acc = set() if acc is None else acc
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.startswith("$"):
                acc.add(k)
            operators(v, acc)
    elif isinstance(node, list):
        for x in node:
            operators(x, acc)
    return acc


def safety(mql):
    """Read-only gate, independent of the server: the project's validate_mql plus an explicit operator scan."""
    ok, reason, _ = D.validate_mql(mql, set())
    ops = operators(mql.get("pipeline", []))
    bad = sorted(ops & (WRITE_STAGES | JS_OPERATORS))
    return {"validate_mql": "ok" if ok else reason, "write_or_js_operators": bad, "ok": bool(ok) and not bad}


def collections_read(mql):
    pipe = mql.get("pipeline", [])
    union, _ = D.union_with_targets(pipe)
    return sorted({mql["collection"]} | set(D.referenced_collections(pipe)) | set(union))


def holdout_stat():
    st = HOLDOUT.stat()
    rec = json.loads(BASELINE.read_text(encoding="utf-8"))["protected"]["data/splits_expanded/test.jsonl"]
    return {"path": "data/splits_expanded/test.jsonl", "opened": False, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "matches_baseline": st.st_size == rec["size"] and st.st_mtime_ns == rec["mtime_ns"]}


def main():
    ts = time.strftime("%Y%m%dT%H%M%S")
    out = OUT_ROOT / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    man = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    eval_contract = TC.load_manifest(TC.EVAL_MANIFEST)
    rep = {
        "run_id": f"corpus_v2_atlas_audit_{ts}", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "scripts/corpus_v2_signoff/atlas_exec_audit.py", "script_sha256": sha(__file__),
        "review_label": "AI-run audit; the corpus is AI-reviewed, never human-reviewed",
        "inputs": {
            "corpus": str(CORPUS.relative_to(ROOT).as_posix()), "corpus_sha256": sha(CORPUS),
            "corpus_manifest_sha256": sha(CORPUS_DIR / "manifest.json"),
            "v2_1_code_freeze_sha256": man["inputs"]["v2_1_code_freeze_sha256"],
            "v2_1_code_freeze_file_sha256": sha(ROOT / "data/interim/finetune_v2/v2_1_code_freeze.json"),
            "pilot_v4_runner_freeze_sha256": man["inputs"]["pilot_v4_runner_freeze_sha256"],
            "verifier": man["verifier"], "converter": man["converter"],
            "type_contract_v1_sha256": sha(ROOT / "data/interim/finetune_v2/type_contract_v1.json"),
            "type_contract_eval_v1_sha256": sha(ROOT / "data/interim/finetune_v2/type_contract_eval_v1.json"),
            "type_contract_eval_body_sha256": eval_contract["body_sha256"],
            "nonconformant_adjudication_sha256": sha(ADJ) if ADJ.exists() else None,
        },
        "holdout_before": holdout_stat(),
    }
    recs = [json.loads(x) for x in CORPUS.open(encoding="utf-8")]
    adj = {}
    if ADJ.exists():
        adj = {json.loads(x)["id"]: json.loads(x)["decision"] for x in ADJ.open(encoding="utf-8")}
    rep["records_discovered"] = len(recs)

    # ---- backend: live Atlas, read-only, never the local MongoDB
    redact = lambda s: s  # noqa: E731
    try:
        client, redact = A.default_client()
        client.admin.command("ping")
        hosts = {h for h, _ in client.nodes} or ({client.address[0]} if client.address else set())
        is_local = any(h in ("127.0.0.1", "localhost", "::1", C.LOCAL_HOST) for h in hosts)
        users, _, bad = A.privileges(client, eval_contract["atlas_db"])
        names = set(client[eval_contract["atlas_db"]].list_collection_names())
        rep["backend"] = {
            "where": "MongoDB Atlas (live), reached with MONGODB_URI via scripts/mongo_conn.py",
            "server_version": client.admin.command("buildInfo").get("version"),
            "is_local_host": is_local, "topology_nodes": len(hosts),
            "local_mongo_guard_used": False, "local_port_in_use_by_this_run": None,
            "authenticated_users": len(users), "write_or_admin_actions_granted": len(bad),
            "read_only": bool(users) and not bad,
            "atlas_db": eval_contract["atlas_db"], "collections_on_atlas": len(names),
        }
        if is_local or not rep["backend"]["read_only"]:
            raise A.CheckError("backend is local or the account is not read-only; refusing to continue")
        tc = A.check_contract(client, eval_contract)
        (out / "tc1_recheck.json").write_text(json.dumps(tc, indent=1, default=str), encoding="utf-8")
        rep["tc1_recheck"] = {k: tc.get(k) for k in ("ok", "collections_checked", "missing_collections",
                                                      "unknown_nonconformant", "contract_body_sha256")}
        rep["tc1_recheck"]["violations"] = len(tc.get("violations", {}))
        rep["tc1_recheck"]["scope"] = ("the Atlas EVALUATION collections (type_contract_eval_v1); the training "
                                       "contract type_contract_v1 cannot be re-checked on Atlas: its collections "
                                       "are not there")
    except Exception as e:  # noqa: BLE001
        rep.update(status="BLOCKED", blocker=f"Atlas not usable: {type(e).__name__}: {redact(str(e))[:300]}",
                   attempted=0, completed=0, skipped=len(recs), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   holdout_after=holdout_stat())
        (out / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        print(json.dumps({k: rep[k] for k in ("run_id", "status", "blocker")}, indent=1))
        return 2

    # ---- per record: safety gate, then can it be executed at all?
    outcomes, rows, absent, dbs = Counter(), [], Counter(), set()
    for r in recs:
        mql = r["mql"] if isinstance(r["mql"], dict) else json.loads(r["mql"])
        s = safety(mql)
        colls = collections_read(mql)
        missing = [c for c in colls if c not in names]
        if not s["ok"]:
            outcome = "invalid_query_safety_rejected"
        elif missing:
            outcome = "not_executable_collection_absent_on_atlas"
        else:
            outcome = "executable_not_run"
        outcomes[outcome] += 1
        absent.update(missing)
        dbs.add(r["db_id"])
        rows.append({"id": r["id"], "db_id": r["db_id"], "collection": mql["collection"], "collections_read": colls,
                     "collections_absent_on_atlas": missing, "safety": s, "outcome": outcome,
                     "executed_on_atlas": False, "nonconformant_adjudication": adj.get(r["id"]),
                     "mql_sha256": hashlib.sha256(json.dumps(mql, sort_keys=True).encode()).hexdigest()})
    with (out / "per_record.jsonl").open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    cat = {k: 0 for k in ("validated", "incorrect_result", "invalid_query", "execution_error", "timeout",
                          "not_comparable", "infrastructure_failure")}
    cat["invalid_query"] = outcomes["invalid_query_safety_rejected"]
    cat["not_executable_collection_absent_on_atlas"] = outcomes["not_executable_collection_absent_on_atlas"]
    cat["executable_not_run"] = outcomes["executable_not_run"]
    sqlite_bytes = sum(Path(C.sqlite_path(d)).stat().st_size for d in dbs)
    rep.update(
        attempted=0, completed=0, skipped=len(recs), outcome_counts=cat,
        safety_gate={"passed": sum(x["safety"]["ok"] for x in rows), "rejected": cat["invalid_query"]},
        collections={"read_by_corpus": len({c for x in rows for c in x["collections_read"]}),
                     "present_on_atlas": len({c for x in rows for c in x["collections_read"]} & names),
                     "absent_on_atlas": len(absent), "corpus_databases": len(dbs),
                     "overlap_with_eval_contract": len({c for x in rows for c in x["collections_read"]}
                                                       & set(eval_contract["collections"]))},
        six_nonconformant_records=[{"id": x["id"], "adjudication": x["nonconformant_adjudication"],
                                    "outcome": x["outcome"], "executed_on_atlas": False}
                                   for x in rows if x["nonconformant_adjudication"]],
        infrastructure={"interruptions": 0, "retries": 0},
        atlas_modified=False,
        atlas_commands_issued=["ping", "buildInfo", "connectionStatus", "listCollections",
                               "aggregate ($type census, read-only)"],
        facts_for_the_decision={
            "source_sqlite_bytes_of_corpus_databases": sqlite_bytes,
            "single_collection_records": sum(len(x["collections_read"]) == 1 for x in rows),
            "multi_collection_records": sum(len(x["collections_read"]) > 1 for x in rows)},
        holdout_after=holdout_stat(), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    if cat["executable_not_run"]:
        rep["status"] = "INCOMPLETE"
        rep["blocker"] = (f"{cat['executable_not_run']} record(s) have all their collections on Atlas, which the "
                          "corpus design rules out; verifier 3.1 execution against Atlas is not implemented")
    elif cat["not_executable_collection_absent_on_atlas"]:
        rep["status"] = "BLOCKED"
        rep["blocker"] = (
            f"{cat['not_executable_collection_absent_on_atlas']} of {len(recs)} corpus records read collections "
            f"that do not exist on Atlas ({len(absent)} collections of {len(dbs)} Spider training databases). Atlas "
            f"holds only the {len(names)} evaluation collections, and the corpus is disjoint from them by design "
            "(leakage rule). The only credential is read-only, so neither the training databases nor verifier "
            "3.1's fixture collections can be created. No corpus label was executed on Atlas.")
    else:
        rep["status"] = "INCOMPLETE"
        rep["blocker"] = "no record was executable and none was blocked: unexpected state"
    rep["atlas_execution_validation_passed"] = False
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("run_id", "status", "records_discovered", "attempted", "completed",
                                          "skipped", "outcome_counts", "safety_gate", "collections", "backend",
                                          "tc1_recheck", "facts_for_the_decision", "holdout_after")}, indent=1))
    print("report:", (out / "report.json").relative_to(ROOT).as_posix())
    return 3


if __name__ == "__main__":
    sys.exit(main())
