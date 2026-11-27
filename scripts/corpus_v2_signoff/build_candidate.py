"""Build the Atlas-validation CANDIDATE corpus v2.1 from provisional corpus v2 (sign-off task, sections 4 and 8).

    python -I -B scripts/corpus_v2_signoff/build_candidate.py <atlas_audit_run_dir> "<test suite result>"

Output: outputs/sql2mql_v2_1/corpus_v2_1_candidate/{corpus.jsonl, exclusions.jsonl, manifest.json}.
corpus.jsonl holds the provisional-corpus-v2 lines byte for byte, minus the records quarantined in
data/interim/corpus_v2_adjudication_v1.jsonl; nothing is rewritten. Provisional corpus v2 is not touched.

The manifest status is APPROVED only if the Atlas audit report says every record was validated; otherwise
it is CANDIDATE_NOT_APPROVED. This script cannot be told to approve.
AI-reviewed, never human-reviewed.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "outputs/sql2mql_v2_1/provisional_corpus_v2"
SRC = SRC_DIR / "provisional_rag_corpus.jsonl"
ADJ = ROOT / "data/interim/corpus_v2_adjudication_v1.jsonl"
OUT = ROOT / "outputs/sql2mql_v2_1/corpus_v2_1_candidate"
POOL = ROOT / "data/processed/spider_pool_v3"
PANEL = ROOT / "outputs/ai_review_panel_v1"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(audit_dir, tests):
    audit_dir = (ROOT / audit_dir).resolve()
    audit = json.loads((audit_dir / "report.json").read_text(encoding="utf-8"))
    adj = [json.loads(x) for x in ADJ.open(encoding="utf-8")]
    drop = {a["id"]: a for a in adj if a["decision"] == "QUARANTINE"}
    kept, excl = [], []
    for line in SRC.open(encoding="utf-8", newline=""):
        r = json.loads(line)
        if r["id"] in drop:
            excl.append({"id": r["id"], "db_id": r["db_id"], "provenance": r["source"],
                         "reason": "nonconformant_field_type_dependent", "detail": drop[r["id"]]["reason"],
                         "adjudication": "data/interim/corpus_v2_adjudication_v1.jsonl"})
        else:
            kept.append(line)
    assert len(excl) == len(drop), "a quarantined id is not in the source corpus"
    OUT.mkdir(parents=True, exist_ok=True)
    body = "".join(kept).encode("utf-8")
    (OUT / "corpus.jsonl").write_bytes(body)
    (OUT / "exclusions.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in excl),
                                          encoding="utf-8")
    src_man = json.loads((SRC_DIR / "manifest.json").read_text(encoding="utf-8"))
    pool_man = json.loads((POOL / "manifest.json").read_text(encoding="utf-8"))
    panel = json.loads((PANEL / "panel_report.json").read_text(encoding="utf-8"))
    oc = audit.get("outcome_counts", {})
    approved = audit.get("status") == "PASSED" and oc.get("validated") == len(kept) + len(excl)
    man = {
        "corpus_version": "corpus_v2_1_candidate",
        "status": "APPROVED" if approved else "CANDIDATE_NOT_APPROVED",
        "status_reason": None if approved else
        f"Atlas execution validation is {audit.get('status')}: {audit.get('blocker')}",
        "review_label": "AI-reviewed (G-AIR, AI-only protocol); NEVER human-reviewed. The run01 human P2 gate is NOT passed.",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "builder": "scripts/corpus_v2_signoff/build_candidate.py", "builder_sha256": sha(__file__),
        "counts": {"source_records": len(kept) + len(excl), "included": len(kept),
                   "quarantined_from_source_corpus": len(excl)},
        "files": {"corpus.jsonl": {"sha256": hashlib.sha256(body).hexdigest(), "records": len(kept)},
                  "exclusions.jsonl": {"sha256": sha(OUT / "exclusions.jsonl"), "records": len(excl)}},
        "source": {"corpus": SRC.relative_to(ROOT).as_posix(), "corpus_sha256": sha(SRC),
                   "manifest_sha256": sha(SRC_DIR / "manifest.json")},
        "reproduce": "drop the ids listed in exclusions.jsonl from the source corpus; keep every other line byte for byte",
        "adjudication": {"file": ADJ.relative_to(ROOT).as_posix(), "sha256": sha(ADJ),
                         "decisions": {a["id"]: a["decision"] for a in adj}},
        "converter": src_man["converter"], "verifier": src_man["verifier"],
        "hashes": {
            "v2_1_code_freeze_sha256": src_man["inputs"]["v2_1_code_freeze_sha256"],
            "pilot_v4_runner_freeze_sha256": src_man["inputs"]["pilot_v4_runner_freeze_sha256"],
            "type_contract_v1_sha256": sha(ROOT / "data/interim/finetune_v2/type_contract_v1.json"),
            "type_contract_eval_v1_sha256": sha(ROOT / "data/interim/finetune_v2/type_contract_eval_v1.json"),
            "spider_pool_v3_manifest_sha256": sha(POOL / "manifest.json"),
        },
        "verification_backend": "local MongoDB 8.0.34 only (verifier 3.1). No record has been executed on Atlas.",
        "atlas_validation": {"run_dir": audit_dir.relative_to(ROOT).as_posix(), "run_id": audit.get("run_id"),
                             "report_sha256": sha(audit_dir / "report.json"), "status": audit.get("status"),
                             "attempted": audit.get("attempted"), "completed": audit.get("completed"),
                             "outcome_counts": oc},
        "g_air": {"report": "outputs/ai_review_panel_v1/panel_report.json", "report_sha256": sha(PANEL / "panel_report.json"),
                  "prereg": "outputs/ai_review_panel_v1/prereg.json",
                  "deviations": "outputs/ai_review_panel_v1/deviations.jsonl",
                  "panel_labels_usable": panel.get("panel_labels_usable"),
                  "limitations": [
                      "AI-only review; reviewers are one model family, so agreement is not independent evidence",
                      "reviewer isolation was partial (memory index summaries were visible)",
                      "the last stability roles ran on a different model",
                      "0 / 200 translation errors bounds the panel-detected rate, not the true population error rate",
                      "60 / 60 planted errors caught does not show that unknown real defects are caught",
                      "the individual corpus records were not reviewed; only a 200-item sample was"]},
        "detector_quarantine": {
            "records_held": 667, "pool_manifest": "data/processed/spider_pool_v3/manifest.json",
            "ids_and_reasons": "data/processed/spider_pool_v3/exclusions_v3.jsonl",
            "note": "conservative data-quality exclusion; detector precision was low, so these are NOT confirmed "
                    "defects; provenance kept for a later recovery audit", "tiers_in_manifest": pool_man.get("tiers")},
        "test_suite": tests,
        "fine_tuning_authorized": bool(approved),
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": man["status"], "counts": man["counts"], "corpus_sha256": man["files"]["corpus.jsonl"]["sha256"],
                      "source_sha256": man["source"]["corpus_sha256"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
