"""Re-execute every saved prediction of the expanded benchmark (no model calls).

Each record's raw_text goes back through run_eval.score_output (parse -> read-only
validation -> aggregate on Atlas `codegen_spider1` -> exec_match vs the SQLite result).
This (a) verifies that the recorded verdicts reproduce on live Atlas and (b) applies scorer
fixes to saved outputs without regenerating. A record whose verdict changes is updated in
place and keeps the old verdict in `previous_results` with the reason; every change is
listed in outputs/evaluation_spider1_expanded/rescore_log.json.

Usage: python scripts/spider1/rescore_expanded.py [--reason "..."] [--dry-run]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval as R  # noqa: E402
import spider1_kit as K  # noqa: E402

CFG = K.ROOT / "configs/evaluation_spider1_expanded.yaml"
LABELS = ["qwen2.5-coder-3b", "qwen2.5-coder-7b", "gpt-oss-120b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", default="re-execution check")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    data = {r["id"]: r for r in K.read_jsonl(K.ROOT / cfg["dataset"])}
    out_dir = K.ROOT / cfg["output_dir"]
    log_path = out_dir / "rescore_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": a.reason, "dry_run": a.dry_run, "runs": {}}
    client = K.mongo_client()
    mdb = client[cfg["atlas_db"]]
    colls = mdb.list_collection_names()
    try:
        for label in LABELS:
            path = out_dir / label / "executions.jsonl"
            recs = K.read_jsonl(path)
            changes, agree = [], 0
            for rec in recs:
                if rec["result"]["category"] == "provider_error":
                    continue
                ex = data[rec["id"]]
                allowed = {c for c in colls if c.startswith(ex["db_id"] + "__")}
                res, extra = R.score_output(mdb, rec["raw_text"], ex, cfg["execution"], allowed)
                old = rec["result"]
                if (res["correct"], res["category"]) == (old["correct"], old["category"]):
                    agree += 1
                    continue
                changes.append({"id": rec["id"], "from": old["category"], "to": res["category"],
                                "reason": res.get("reason")})
                if not a.dry_run:
                    rec.setdefault("previous_results", []).append({**old, "superseded_at": entry["at"],
                                                                   "superseded_because": a.reason})
                    for k in ("mql", "pred_columns", "pred_rows", "pred_rows_truncated", "pred_preview"):
                        rec.pop(k, None)
                    rec["result"] = res
                    rec.update(extra)
            if changes and not a.dry_run:
                K.write_jsonl(path, recs)
            entry["runs"][label] = {"records": len(recs), "verdicts_reproduced": agree, "changed": changes}
            print(f"{label}: reproduced {agree}/{len(recs)}; changed {len(changes)} {changes}", flush=True)
    finally:
        client.close()
    log.append(entry)
    log_path.write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
