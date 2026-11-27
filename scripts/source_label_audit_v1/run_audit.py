"""Source-label audit v1 (plan Part D, steps S7-S8). Read-only over spider_pool_v2 and the SQLite databases.

    python -B scripts/source_label_audit_v1/run_audit.py prereg    freeze detectors + thresholds -> data/interim/source_label_audit_v1/prereg.json
    python -B scripts/source_label_audit_v1/run_audit.py recall    the 120 run01 review items (development data) -> recall_120.json
    python -B scripts/source_label_audit_v1/run_audit.py pool      spider_pool_v2 train + val_internal -> hits.jsonl, pool_summary.json
    python -B scripts/source_label_audit_v1/run_audit.py siblings  same-(db_id, normalised SQL) siblings of the 10 run01 findings

Outputs under outputs/source_label_audit_v1/. Every run checks the pre-registration hashes first. No record is
modified; no SQL is repaired; hits are quarantine candidates pending AI-panel review.
"""
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import detectors as D  # noqa: E402

PREREG = ROOT / "data/interim/source_label_audit_v1/prereg.json"
OUT = ROOT / "outputs/source_label_audit_v1"
POOL = ROOT / "data/processed/spider_pool_v2"
REVIEW = ROOT / "outputs/sql2mql_v2/pilot_v3/run01/review_package/review_items.jsonl"
TRAIN = ROOT / "data/processed/spider_sql_v1/train.jsonl"
MUST = {  # plan §D7 (R009 is 'weak', not required)
    "R084": {"D1": ["D1a"]}, "R097": {"D6": ["D6b", "D6c"]}, "R101": {"D1": ["D1a"]}, "R119": {"D2": ["D2"]},
    "R010": {"D3": ["D3a"]}, "R012": {"D5": ["D5"], "D6": ["D6a"]}, "R090": {"D3": ["D3b"]},
    "R111": {"D1": ["D1a", "D1b"]}, "R118": {"D4": ["D4_data"]},
}
MUST_NOT = {"R084": ["D2"]}
FINDINGS = {"R084": "spider_train_02406", "R097": "spider_train_04071", "R101": "spider_train_05405",
            "R119": "spider_train_03948", "R009": "spider_train_02184", "R010": "spider_train_06757",
            "R012": "spider_train_04685", "R090": "spider_train_00515", "R111": "spider_train_04709",
            "R118": "spider_train_04227"}
FIRM = {"R084", "R097", "R101", "R119"}
PRECISION = {"seed": 20261113, "per_detector": 40, "labelled_by": "AI review panel v1 (multi-agent), never human"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def norm_sql(s):
    return " ".join(re.sub(r"\s+", " ", s.strip().rstrip(";")).lower().replace('"', "'").split())


def rd(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def check_prereg():
    pr = json.loads(PREREG.read_text(encoding="utf-8"))
    if pr["detectors_sha256"] != sha(HERE / "detectors.py") or pr["params_sha256"] != sha_obj([D.PARAMS, D.LEXICON]):
        raise SystemExit("detectors.py / thresholds changed after pre-registration: stop")
    return pr


def review_records():
    src = {r["id"]: r for r in rd(TRAIN)}
    out = []
    for it in rd(REVIEW):
        out.append({"review_id": it["review_id"], "id": it["item_id"], "db_id": it["db_id"], "question": it["question"],
                    "sql": it["sql"], "sqlite_result": it["sqlite_result"],
                    "question_raw": src[it["item_id"]].get("question_raw")})
    return out


def recall():
    pr = check_prereg()
    rows, per = [], {}
    for r in review_records():
        hits = D.run_all(r)
        per[r["review_id"]] = hits
        rows.append({"review_id": r["review_id"], "id": r["id"],
                     "hits": [{"detector": h["detector"], "subclass": h["subclass"], "action": h["action"]} for h in hits]})
    miss, bad = [], []
    for rid, need in MUST.items():
        subs = {h["subclass"] for h in per[rid]}
        for det, allowed in need.items():
            if not subs & set(allowed):
                miss.append((rid, det, allowed))
    for rid, dets in MUST_NOT.items():
        if any(h["detector"] in dets and h["action"] == "quarantine" for h in per[rid]):
            bad.append(rid)
    accepted110 = [rid for rid in per if rid not in FINDINGS]
    fp = Counter(h["subclass"] for rid in accepted110 for h in per[rid] if h["action"] == "quarantine")
    fp_items = sorted({rid for rid in accepted110 for h in per[rid] if h["action"] == "quarantine"})
    rep = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "prereg_sha256": sha(PREREG),
           "note": "the 120 items are DEVELOPMENT data for the detectors (used to find these defects), not test data",
           "must_detect_missed": miss, "must_not_violations": bad, "recall_pass": not miss and not bad,
           "must_detect_items": len(MUST), "flags_among_110_audit_accepted": dict(fp),
           "items_flagged_among_110": len(fp_items), "flagged_items_110": fp_items,
           "R009_hits": per["R009"], "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recall_120.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("recall_pass", "must_detect_missed", "must_not_violations",
                                           "flags_among_110_audit_accepted", "items_flagged_among_110")}, indent=1))
    return rep


def pool():
    pr = check_prereg()
    OUT.mkdir(parents=True, exist_ok=True)
    hp = OUT / "hits.jsonl"
    if hp.exists():
        raise SystemExit(f"refusing to overwrite {hp}")
    n, per_det, per_rec, recs_q = 0, Counter(), defaultdict(set), Counter()
    with hp.open("w", encoding="utf-8") as f:
        for split in ("train.jsonl", "val_internal.jsonl"):
            for r in rd(POOL / split):
                n += 1
                for h in D.run_all(r):
                    row = {"id": r["id"], "db_id": r["db_id"], "split": split.split(".")[0], **h,
                           "detector_version": D.DETECTOR_VERSION}
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    per_det[(h["detector"], h["subclass"], h["action"])] += 1
                    if h["action"] == "quarantine":
                        per_rec[r["id"]].add(h["subclass"])
                        recs_q[split] += 1 if len(per_rec[r["id"]]) == 1 else 0
    summ = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "prereg_sha256": sha(PREREG), "records_scanned": n,
            "pool_files_sha256": {s: sha(POOL / s) for s in ("train.jsonl", "val_internal.jsonl")},
            "hits_by_detector": {f"{a}|{b}|{c}": k for (a, b, c), k in sorted(per_det.items())},
            "records_with_quarantine_hit": len(per_rec), "records_with_quarantine_hit_by_split": dict(recs_q),
            "records_by_hit_count": dict(Counter(len(v) for v in per_rec.values())),
            "hits_sha256": sha(hp), "where": "local SQLite (read-only); no MongoDB / Atlas involved"}
    (OUT / "pool_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


def siblings():
    check_prereg()
    by_key = defaultdict(list)
    files = {}
    for split in ("train.jsonl", "val_internal.jsonl", "quarantine.jsonl"):
        for r in rd(POOL / split):
            by_key[(r["db_id"], norm_sql(r["sql"]))].append(r["id"])
            files[r["id"]] = (split.split(".")[0], r)
    excl = {e["id"]: e for e in rd(POOL / "exclusions.jsonl")}
    src = {r["id"]: r for r in rd(TRAIN)}
    out = []
    for rid, iid in FINDINGS.items():
        r = src[iid]
        key = (r["db_id"], norm_sql(r["sql"]))
        sib = [i for i in by_key.get(key, []) if i != iid]
        out.append({"review_id": rid, "finding_id": iid, "confidence": "high" if rid in FIRM else "medium",
                    "finding_in_pool": files[iid][0] if iid in files else ("excluded:" + excl[iid]["disposition"]
                                                                            if iid in excl else "not_in_pool"),
                    "siblings": [{"id": i, "split": files[i][0], "question": files[i][1]["question"]} for i in sib]})
    (OUT / "finding_siblings.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for x in out:
        print(x["review_id"], x["finding_id"], x["finding_in_pool"], [(s["id"], s["split"]) for s in x["siblings"]])


def prereg():
    if PREREG.exists():
        raise SystemExit(f"refusing to overwrite {PREREG}")
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    doc = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "plan": "docs/pilot_v3_followup_implementation_plan.md Part D",
           "detector_version": D.DETECTOR_VERSION, "detectors_sha256": sha(HERE / "detectors.py"),
           "run_audit_sha256": sha(__file__), "params_sha256": sha_obj([D.PARAMS, D.LEXICON]),
           "params": D.PARAMS, "lexicon": D.LEXICON, "must_detect": MUST, "must_not_detect": MUST_NOT,
           "type_contract_manifest_sha256": sha(ROOT / "data/interim/finetune_v2/type_contract_v1.json"),
           "scope": "spider_pool_v2 train.jsonl + val_internal.jsonl (read-only); never Spider dev/test or "
                    "data/splits_expanded/test.jsonl",
           "actions": {"quarantine": "D1a, D1b (non-ISO), D2, D3a, D3b ('all X' policy = quarantine, user 2026-11-13), "
                                     "D4_data, D5, D6a, D6b, D6c -> C_semantic_review unless the AI panel clearly accepts",
                       "report_only": "D1b_iso, D4_schema_only, parse failures, detector errors"},
           "precision_sampling": PRECISION,
           "development_note": "detector rules were developed on the 120 run01 review items (development data); "
                               "thresholds frozen here before the first pool run",
           "sibling_rule": "same db_id and normalised SQL (whitespace collapsed, lower-case, double->single quotes); "
                           "each sibling's question is judged separately by the AI panel"}
    PREREG.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {PREREG.relative_to(ROOT)} sha256={sha(PREREG)}")


if __name__ == "__main__":
    {"prereg": prereg, "recall": recall, "pool": pool, "siblings": siblings}[sys.argv[1]]()
