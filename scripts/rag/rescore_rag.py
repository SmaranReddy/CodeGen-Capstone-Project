"""Re-score saved RAG predictions without model calls (deterministic re-execution check).

Every saved raw_text goes back through run_eval.score_output (parse -> read-only validation ->
Atlas aggregate -> exec_match). Records are never modified; the agreement report goes to
outputs/rag_week3/rescore_check.json.

Usage: python scripts/rag/rescore_rag.py [experiment ...]   (default: every experiment dir)
"""
import json
import sys
import time

import yaml

import common as C
from common import K

import run_eval as R  # noqa: E402

OUT_DIR = C.ROOT / "outputs/rag_week3"
EXEC_CFG = yaml.safe_load((C.ROOT / "configs/evaluation_spider1_expanded.yaml")
                          .read_text(encoding="utf-8"))["execution"]


def rescore_records(mdb, records, data, colls, exec_cfg=EXEC_CFG):
    """-> (n_agree, [disagreements]). Provider errors are skipped (nothing to score)."""
    agree, diff = 0, []
    for rec in records:
        if rec["result"]["category"] == "provider_error":
            continue
        ex = data[rec["id"]]
        allowed = {c for c in colls if c.startswith(ex["db_id"] + "__")}
        res, _ = R.score_output(mdb, rec["raw_text"], ex, exec_cfg, allowed)
        if (res["correct"], res["category"]) == (rec["result"]["correct"], rec["result"]["category"]):
            agree += 1
        else:
            diff.append({"id": rec["id"], "saved": rec["result"]["category"], "now": res["category"]})
    return agree, diff


def main():
    names = sys.argv[1:] or sorted(p.name for p in OUT_DIR.iterdir() if (p / "executions.jsonl").exists())
    data = {r["id"]: r for r in K.read_jsonl(C.DATASET)}
    client = K.mongo_client()
    report = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": {}}
    try:
        mdb = client[K.ATLAS_DB]
        colls = mdb.list_collection_names()
        for name in names:
            recs = K.read_jsonl(OUT_DIR / name / "executions.jsonl")
            agree, diff = rescore_records(mdb, recs, data, colls)
            report["runs"][name] = {"records": len(recs), "verdicts_reproduced": agree, "changed": diff}
            print(f"{name}: reproduced {agree}/{len(recs)}; changed {diff}")
    finally:
        client.close()
    (OUT_DIR / "rescore_check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
