"""Stage 4: replay every training/dev TARGET exactly as the model will see it.

    python scripts/finetune_v1/replay_targets.py [--workers 6]

The artifact, not the in-memory object: each target is parsed from the assistant message
text (`messages[-1].content`, json.loads keeps key order) and executed again.
  * Spider-train records: through the benchmark's own scorer, run_eval.score_output
    (parse -> read-only validator -> Atlas `codegen_spider1` -> exec_match vs gold_rows,
    ordered/tie-aware as in the benchmark).
  * Kaggle-derived records: on freshly regenerated fixtures (same seeds; a fresh server clock
    for clock-relative examples), compared with fixtures.compare.
Writes data/finetune/ft_v1/replay_report.json and records its summary + sha256 in
build_manifest.json. Every record must pass; the script exits non-zero otherwise.
"""
import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nk_common as C  # noqa: E402
import fixtures as F  # noqa: E402
import sqlsem as S  # noqa: E402
import verify as V  # noqa: E402


def replay_new(r, schema):
    import spider1_kit as K
    text = r["messages"][-1]["content"]
    mql = json.loads(text)
    ast = S.parse_sql(r["reference"]["sql"])[0]
    an = S.analyze(ast).to_json()
    sch = {t: schema[t] for t in an["columns"]}
    seed_base = int(r["source_pair_id"][3:11], 16) % 100000
    for attempt in range(2 if an["time_relative"] else 1):
        clock = V.server_clock() if an["time_relative"] else F.BASE_CLOCK
        fx = F.adequate_set(ast, an, sch, seed_base + attempt, clock)
        if fx["uncovered"]:
            continue
        ok = True
        for v in fx["kept"]:
            docs = V.run_mql(mql, v["tables"])
            cols, rows = K.docs_to_rows(docs)
            if not F.compare(cols if docs else None, rows, v["columns"], v["rows"], v["ordered"],
                             v["blocks"] if v["ordered"] else None)[0]:
                ok = False
                break
        if ok:
            return True, f"{len(fx['kept'])} variants match"
    return False, "mismatch or fixture not adequate on replay"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    import yaml
    import run_eval as R
    import spider1_kit as K
    cfg = yaml.safe_load((C.ROOT / "configs/evaluation_spider1_expanded.yaml").read_text(encoding="utf-8"))
    bench = {b["id"]: b for b in C.read_jsonl(C.BENCHMARK)}
    schema = json.loads((C.INTERIM / "schema.json").read_text(encoding="utf-8"))["tables"]
    recs = C.read_jsonl(C.OUT_DIR / "train.jsonl") + C.read_jsonl(C.OUT_DIR / "dev_internal.jsonl")
    client = K.mongo_client()
    mdb = client[cfg["atlas_db"]]
    colls = mdb.list_collection_names()

    def work(r):
        try:
            if r["source_dataset"] == C.SOURCE_ID:
                ok, why = replay_new(r, schema)
            else:
                ex = bench[r["source_pair_id"]]
                allowed = {c for c in colls if c.startswith(ex["db_id"] + "__")}
                res, _ = R.score_output(mdb, r["messages"][-1]["content"], ex, cfg["execution"], allowed)
                ok, why = res["correct"], f"{res['category']}: {res.get('reason')}"
        except Exception as e:  # noqa: BLE001
            ok, why = False, f"{type(e).__name__}: {str(e)[:200]}"
        return {"id": r["id"], "source_dataset": r["source_dataset"], "ok": ok, "reason": why,
                "target_sha256": C.sha256_bytes(r["messages"][-1]["content"].encode("utf-8"))}

    with cf.ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(work, recs))
    client.close()
    summary = {}
    for res in results:
        s = summary.setdefault(res["source_dataset"], {"n": 0, "passed": 0})
        s["n"] += 1
        s["passed"] += res["ok"]
    report = {"what": "every train/dev target replayed from its assistant-message text",
              "train_sha256": C.sha256_file(C.OUT_DIR / "train.jsonl"),
              "dev_sha256": C.sha256_file(C.OUT_DIR / "dev_internal.jsonl"),
              "summary": summary, "failures": [x for x in results if not x["ok"]], "results": results}
    path = C.OUT_DIR / "replay_report.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    man_path = C.OUT_DIR / "build_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["replay"] = {"summary": summary, "report_sha256": C.sha256_file(path),
                     "command": "python scripts/finetune_v1/replay_targets.py"}
    man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps(summary), "failures:", len(report["failures"]))
    for f in report["failures"][:20]:
        print(f)
    sys.exit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
