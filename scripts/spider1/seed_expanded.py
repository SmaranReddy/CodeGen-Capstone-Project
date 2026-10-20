"""Seed the expanded benchmark's executions.jsonl with REUSABLE v1 predictions (no model calls).

A v1 prediction is reused for an expanded-dataset id only if ALL hold:
  * the v1 run's model config block and generation block equal the expanded config's
    (configs/evaluation_spider1.yaml run summary vs configs/evaluation_spider1_expanded.yaml)
  * the record's model_id equals the configured model id, and its prompt_version and
    prompt_sha256 equal the current prompt template
  * the per-item prompt context files (atlas_collections.json, prompt_samples.json) have the
    sha256 recorded in expanded/parity.json and were last modified before the v1 run began
  * for GGUF models: the model file's sha256 equals the pinned value
  * the record is not a provider_error, and the question/db_id match the dataset item
Reused raw outputs are RE-EXECUTED on Atlas through run_eval.score_output (read-only), so
each record carries actual result rows; the v1 verdict is kept as `v1_result` and any
disagreement is reported. Ids already present in the target file are never overwritten.

Writes outputs/evaluation_spider1_expanded/<label>/executions.jsonl and reuse_manifest.json.

Usage: python scripts/spider1/seed_expanded.py [--labels qwen2.5-coder-3b ...]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation"))
import prompt_spider1 as P  # noqa: E402
import run_eval as R  # noqa: E402
import spider1_kit as K  # noqa: E402

V1_CFG = K.ROOT / "configs/evaluation_spider1.yaml"
EXP_CFG = K.ROOT / "configs/evaluation_spider1_expanded.yaml"
RUNS = {  # label -> provider key (same in both configs)
    "qwen2.5-coder-3b": "qwen",
    "qwen2.5-coder-7b": "qwen7b",
    "gpt-oss-120b": "frontier_groq",
}
PINNED_GGUF = {  # docs/spider1_dataset.md section 7
    "qwen": "724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7",
    "qwen7b": "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c",
}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="*", default=list(RUNS))
    a = ap.parse_args()
    v1cfg = yaml.safe_load(V1_CFG.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(EXP_CFG.read_text(encoding="utf-8"))
    rows = K.read_jsonl(K.ROOT / cfg["dataset"])
    by_id = {r["id"]: r for r in rows}
    parity = K.load_json(K.INTERIM / "expanded/parity.json")
    ctx = {}
    for name, want in parity["prompt_context_sha256"].items():
        p = K.INTERIM / name
        ctx[name] = {"sha256": sha_file(p), "expected": want,
                     "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(p)))}
    client = K.mongo_client()
    mdb = client[cfg["atlas_db"]]
    colls = mdb.list_collection_names()
    manifest = {}
    try:
        for label in a.labels:
            prov = RUNS[label]
            v1_dir = K.ROOT / v1cfg["output_dir"] / label
            summ = K.load_json(v1_dir / "summary.json")
            mc = cfg["models"][prov]
            checks = {
                "model_config_equal": summ.get("config") == mc == v1cfg["models"][prov],
                "generation_equal": summ.get("generation") == cfg["generation"] == v1cfg["generation"],
                "prompt_template_equal": summ.get("prompt_sha256") == P.template_hash()
                and summ.get("prompt_version") == P.PROMPT_VERSION,
            }
            v1_records = K.read_jsonl(v1_dir / "executions.jsonl")
            first_ts = min(r["timestamp"] for r in v1_records)
            checks["prompt_context_unchanged"] = all(
                c["sha256"] == c["expected"] and c["mtime"] < first_ts for c in ctx.values())
            if mc.get("gguf_path"):
                got = sha_file(K.ROOT / mc["gguf_path"])
                checks["gguf_sha256_pinned"] = got == PINNED_GGUF[prov]
            run_ok = all(checks.values())
            out_dir = K.ROOT / cfg["output_dir"] / label
            out_dir.mkdir(parents=True, exist_ok=True)
            ex_path = out_dir / "executions.jsonl"
            done = {r["id"]: r for r in K.read_jsonl(ex_path)} if ex_path.exists() else {}
            reused, rejected, changed = [], {}, []
            for old in v1_records if run_ok else []:
                iid = old["id"]
                ex = by_id.get(iid)
                why = None
                if ex is None:
                    why = "not in expanded dataset"
                elif old["result"]["category"] == "provider_error":
                    why = "provider_error"
                elif old.get("model_id") != mc["id"]:
                    why = f"model_id {old.get('model_id')!r}"
                elif old.get("prompt_sha256") != P.template_hash() or old.get("prompt_version") != P.PROMPT_VERSION:
                    why = "prompt mismatch"
                elif (old["db_id"], old["question"]) != (ex["db_id"], ex["question"]):
                    why = "question mismatch"
                elif iid in done:
                    why = "already present"
                if why:
                    rejected[iid] = why
                    continue
                allowed = {c for c in colls if c.startswith(ex["db_id"] + "__")}
                res, extra = R.score_output(mdb, old["raw_text"], ex, cfg["execution"], allowed)
                rec = {k: v for k, v in old.items() if k not in ("result", "mql", "pred_columns", "pred_preview")}
                rec.update(expected_columns=ex["gold_columns"], expected_rows=ex["gold_rows"],
                           result=res, v1_result=old["result"],
                           reused_from=str((v1_dir / "executions.jsonl").relative_to(K.ROOT)).replace("\\", "/"),
                           rescored_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                rec.update(extra)
                if res["correct"] != old["result"]["correct"] or res["category"] != old["result"]["category"]:
                    changed.append({"id": iid, "v1": old["result"]["category"], "now": res["category"],
                                    "reason": res.get("reason")})
                done[iid] = rec
                reused.append(iid)
            K.write_jsonl(ex_path, [done[r["id"]] for r in rows if r["id"] in done])
            manifest[label] = {
                "provider": prov, "v1_run": str(v1_dir.relative_to(K.ROOT)).replace("\\", "/"),
                "checks": checks, "reusable": run_ok, "reused": len(reused),
                "rejected": {w: sum(1 for x in rejected.values() if x == w) for w in sorted(set(rejected.values()))},
                "verdict_changes_on_reexecution": changed,
                "remaining_to_generate": sum(1 for r in rows if r["id"] not in done),
            }
            print(label, json.dumps({k: manifest[label][k] for k in ("checks", "reused", "rejected", "remaining_to_generate")}),
                  "verdict changes:", len(changed), flush=True)
    finally:
        client.close()
    path = K.ROOT / cfg["output_dir"] / "reuse_manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(manifest)
    old["_prompt_context"] = ctx
    path.write_text(json.dumps(old, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
