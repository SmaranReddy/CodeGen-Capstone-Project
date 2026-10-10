"""Assemble the validated Spider 1.0 NL -> MQL dataset from the frozen sample.

Re-runs both checks from scratch (never trusts cached results):
  1. dev execution match of the gold MQL vs gold SQL result (validate_gold.validate)
  2. test-suite cross-check on the imported variants (testsuite_check.check_item)

Inclusion rule (an item enters data/processed/spider1_nl_mql.jsonl only if ALL hold):
  * gold MQL exists and passes the dev check
  * no test-suite variant FAILs (PASS or INCONCLUSIVE; wta_1 has no variants -> dev only)
  * the curator's adjudication (data/interim/spider1/adjudication.json) does not exclude it.
    Default policy for flagged items: `annotation_mismatch` (gold SQL answers a different
    question than the NL states) and `cannot_convert` are EXCLUDED; `ambiguous` is KEPT
    and flagged. Per-id overrides in adjudication.json carry a written reason.

Outputs:
  data/processed/spider1_nl_mql.jsonl         validated examples
  data/interim/spider1/status.jsonl          per-item status for all 300 sampled items
  data/interim/spider1/dataset_summary.json  counts / attrition / evidence levels

Usage: python scripts/spider1/build_dataset.py
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402
from testsuite_check import check_item  # noqa: E402
from validate_gold import GOLD_DIR, load_gold, validate  # noqa: E402

OUT = K.ROOT / "data/processed/spider1_nl_mql.jsonl"
ADJ = K.INTERIM / "adjudication.json"
EXCLUDE_KINDS = {"annotation_mismatch", "cannot_convert"}


def load_issues():
    out = {}
    for p in GOLD_DIR.glob("*.issues.json"):
        out.update(json.loads(p.read_text(encoding="utf-8")))
    return out


def main():
    sample = K.read_jsonl(K.INTERIM / "sample.jsonl")
    variants = json.loads((K.INTERIM / "testsuite_variants.json").read_text(encoding="utf-8"))
    issues = load_issues()
    adj = json.loads(ADJ.read_text(encoding="utf-8")) if ADJ.exists() else {}
    client = K.mongo_client()
    mdb = client[K.ATLAS_DB]
    existing = set(mdb.list_collection_names())
    status, dataset = [], []
    try:
        for item in sample:
            iid = item["id"]
            entry = load_gold(item["db_id"]).get(iid)
            st = {"id": iid, "db_id": item["db_id"], "hardness": item["hardness"],
                  "issue": issues.get(iid), "adjudication": adj.get(iid)}
            if entry is None:
                st.update(status="not_converted", reason=(issues.get(iid) or {}).get("detail"))
                status.append(st)
                continue
            dev = validate(mdb, item, entry, existing)
            st["dev"] = {"pass": dev["pass"], "reason": dev["reason"]}
            vs = variants.get(item["db_id"]) or []
            ts = check_item(client, item, entry, vs) if vs else []
            ts_st = Counter(c["status"] for c in ts)
            st["testsuite"] = dict(ts_st)
            if not dev["pass"]:
                st.update(status="failed_dev")
            elif ts_st.get("fail"):
                st.update(status="failed_testsuite",
                          reason="; ".join(c["reason"] for c in ts if c["status"] == "fail"))
            else:
                decision = adj.get(iid, {}).get("decision")
                kind = (issues.get(iid) or {}).get("kind")
                excluded = decision == "exclude" or (decision != "include" and kind in EXCLUDE_KINDS)
                if excluded:
                    st.update(status="excluded_annotation",
                              reason=adj.get(iid, {}).get("reason") or (issues.get(iid) or {}).get("detail"))
                else:
                    st["status"] = "validated"
            st["evidence"] = ("dev+testsuite" if ts_st.get("pass") else "dev_only") if dev["pass"] else None
            status.append(st)
            print(f"{st['status']:20s} {iid} dev={dev['pass']} ts={dict(ts_st)}")
            if st["status"] == "validated":
                dataset.append({
                    "id": iid, "source": "spider1_dev", "dev_index": item["dev_index"],
                    "db_id": item["db_id"], "question": item["question"], "hardness": item["hardness"],
                    "gold_sql": item["gold_sql"],
                    "gold_mql": {"collection": entry["collection"], "pipeline": entry["pipeline"]},
                    "gold_columns": item["gold_columns"], "gold_rows": item["gold_rows"],
                    "order_by": item["order_by"], "limit": item["limit"], "tie_blocks": item["tie_blocks"],
                    "evidence": st["evidence"], "testsuite": st["testsuite"],
                    "flags": [issues[iid]["kind"]] if iid in issues else [],
                    "notes": entry.get("notes"),
                })
    finally:
        client.close()
    K.write_jsonl(OUT, dataset)
    K.write_jsonl(K.INTERIM / "status.jsonl", status)
    uniq = {(d["db_id"], d["question"]) for d in dataset}
    summary = {
        "sample_size": len(sample),
        "status": dict(Counter(s["status"] for s in status)),
        "validated": len(dataset),
        "unique_questions": len(uniq),
        "unique_gold_sql": len({(d["db_id"], d["gold_sql"]) for d in dataset}),
        "by_hardness": dict(Counter(d["hardness"] for d in dataset)),
        "by_db": dict(sorted(Counter(d["db_id"] for d in dataset).items())),
        "evidence": dict(Counter(d["evidence"] for d in dataset)),
        "flagged_kept": dict(Counter(f for d in dataset for f in d["flags"])),
        "dataset_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    (K.INTERIM / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
