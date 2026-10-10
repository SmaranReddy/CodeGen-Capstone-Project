"""Run a model on the Spider 1.0 NL -> MQL benchmark (same prompt, data and scorer for all).

Per example: prompt -> raw output -> JSON parse -> structural read-only validation ->
execution on Atlas `codegen_spider1` -> exec_match vs the gold SQL result (the same
comparator that accepted the gold MQL). Wrong answers are never retried; only provider
transport errors are retried (by the SDK). Resumable: finished ids in the output
executions.jsonl are skipped unless --fresh (provider errors are always re-run).

Usage:
  python scripts/spider1/run_eval.py --provider reference            # harness self-check
  python scripts/spider1/run_eval.py --provider qwen [--limit 10] [--ids ...]
  python scripts/spider1/run_eval.py --provider frontier
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation"))
import prompt_spider1 as P  # noqa: E402
import spider1_kit as K  # noqa: E402
from eval_kit import validate_mql  # noqa: E402
from providers import build_provider  # noqa: E402
from run_nl_mql_eval import parse_output  # noqa: E402

CONFIG = K.ROOT / "configs/evaluation_spider1.yaml"


class ReferenceSpider1:
    """Replays the gold MQL through the exact model path (harness oracle, NOT a model)."""
    name, model_id = "reference", "reference-oracle (gold MQL replay)"

    def __init__(self, rows):
        self.by_q = {(r["db_id"], r["question"]): r["gold_mql"] for r in rows}
        self.current = None

    def available(self):
        return True, "replays gold MQL (harness validation only)"

    def generate(self, messages, system, generation):
        return {"text": json.dumps(self.by_q[self.current]), "input_tokens": 0, "output_tokens": 0,
                "latency_s": 0.0, "model_id": self.model_id, "error": None}


def classify_mismatch(rows, ex):
    ok, why = K.exec_match(rows, ex["gold_rows"])  # order-insensitive re-check
    if ex["order_by"] and ok:
        return "order_mismatch"
    if why.startswith("row count"):
        return "wrong_row_count"
    if why.startswith("column count"):
        return "wrong_column_count"
    return "wrong_values"


def run_one(mdb, provider, ex, gen, exec_cfg, allowed):
    msgs = P.messages(ex["db_id"], ex["question"])
    if isinstance(provider, ReferenceSpider1):
        provider.current = (ex["db_id"], ex["question"])
    out = provider.generate(msgs, P.SYSTEM, dict(gen))
    rec = {"id": ex["id"], "db_id": ex["db_id"], "hardness": ex["hardness"],
           "question": ex["question"], "model_id": out.get("model_id"),
           "prompt_version": P.PROMPT_VERSION, "prompt_sha256": P.template_hash(),
           "raw_text": out.get("text"), "provider_error": out.get("error"),
           "stop_reason": out.get("stop_reason"), "request_id": out.get("request_id"),
           "latency_s": round(out.get("latency_s") or 0.0, 3),
           "wall_s": out.get("wall_s"), "rate_limit_retries": out.get("rate_limit_retries"),
           "quota_wait_s": out.get("quota_wait_s"),
           "server_time_s": out.get("server_time_s"), "thought_tokens": out.get("thought_tokens"),
           "input_tokens": out.get("input_tokens"), "output_tokens": out.get("output_tokens"),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    res = {"correct": False, "category": None, "reason": None, "executed": False}
    rec["result"] = res
    if out.get("error"):
        res.update(category="provider_error", reason=out["error"])
        return rec
    parsed, perr = parse_output(out.get("text"))
    if parsed is None:
        res.update(category="format_error", reason=perr)
        return rec
    if isinstance(parsed, dict) and "method" not in parsed:
        parsed = {"method": "aggregate", **parsed}
    ok, why, norm = validate_mql(parsed, allowed)
    if not ok:
        res.update(category="invalid_mql", reason=why)
        return rec
    rec["mql"] = {"collection": norm["collection"], "pipeline": norm["pipeline"]}
    try:
        docs = list(mdb[norm["collection"]].aggregate(
            norm["pipeline"], maxTimeMS=exec_cfg["timeout_ms"], allowDiskUse=True))
    except Exception as e:  # noqa: BLE001
        res.update(category="execution_error", reason=f"{type(e).__name__}: {str(e)[:300]}")
        return rec
    res["executed"] = True
    res["n_rows"] = len(docs)
    if len(docs) > exec_cfg["max_result_rows"]:
        res.update(category="execution_error", reason="result too large")
        return rec
    cols, rows = K.docs_to_rows(docs)
    match, why = K.exec_match(rows, ex["gold_rows"], ex.get("tie_blocks"), ex["order_by"])
    res["reason"] = why
    if match:
        res.update(correct=True, category="correct")
    else:
        res["category"] = classify_mismatch(rows, ex)
        rec["pred_columns"] = cols
        rec["pred_preview"] = [list(r) for r in rows[:10]]
    return rec


def summarize(records, price, label, model_id):
    n = len(records)
    cat = Counter(r["result"]["category"] for r in records)
    correct = sum(r["result"]["correct"] for r in records)
    by_h = {}
    for h in ("easy", "medium", "hard", "extra"):
        hs = [r for r in records if r["hardness"] == h]
        if hs:
            c = sum(r["result"]["correct"] for r in hs)
            by_h[h] = {"n": len(hs), "correct": c, "accuracy": round(c / len(hs), 4)}
    lat = sorted(r["latency_s"] for r in records)
    itok = sum(r["input_tokens"] or 0 for r in records)
    otok = sum(r["output_tokens"] or 0 for r in records)
    cost = round(itok / 1000 * (price.get("input_per_1k") or 0)
                 + otok / 1000 * (price.get("output_per_1k") or 0), 4)
    return {"label": label, "model_id": model_id, "prompt_version": P.PROMPT_VERSION,
            "prompt_sha256": P.template_hash(), "n": n, "correct": correct,
            "execution_accuracy": round(correct / n, 4) if n else None,
            "valid_json": n - cat.get("format_error", 0) - cat.get("provider_error", 0),
            "executed": sum(r["result"]["executed"] for r in records),
            "by_hardness": by_h, "categories": dict(cat),
            "latency_s": {"mean": round(sum(lat) / n, 3) if n else None,
                          "median": lat[n // 2] if n else None,
                          "p90": lat[min(n - 1, int(n * 0.9))] if n else None,
                          "total": round(sum(lat), 1)},
            "input_tokens": itok, "output_tokens": otok, "cost_usd": cost}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["reference", "qwen", "qwen7b", "frontier", "frontier_groq", "frontier_g36", "frontier_claude"])
    ap.add_argument("--label")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--dataset")
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(K.ROOT / ".env")  # ANTHROPIC_API_KEY / MONGODB_URI; never printed
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    dataset = a.dataset or cfg["dataset"]
    rows = K.read_jsonl(K.ROOT / dataset)
    if a.ids:
        rows = [r for r in rows if r["id"] in a.ids]
    if a.limit:
        rows = rows[:a.limit]
    if a.provider == "reference":
        provider = ReferenceSpider1(rows)
    else:
        mc = dict(cfg["models"][a.provider])
        if mc.get("gguf_path"):
            mc["gguf_path"] = str(K.ROOT / mc["gguf_path"])
        provider = build_provider(mc, a.provider)
    ok, msg = provider.available()
    print(f"provider {a.provider}: {msg}")
    if not ok:
        raise SystemExit(1)
    label = a.label or a.provider
    out_dir = K.ROOT / cfg["output_dir"] / label
    out_dir.mkdir(parents=True, exist_ok=True)
    ex_path = out_dir / "executions.jsonl"
    done = {}
    if ex_path.exists() and not a.fresh:
        for r in K.read_jsonl(ex_path):
            if r["result"]["category"] != "provider_error":
                done[r["id"]] = r
    client = K.mongo_client()
    mdb = client[cfg["atlas_db"]]
    colls = mdb.list_collection_names()
    t0 = time.time()
    try:
        for i, ex in enumerate(rows):
            if ex["id"] in done:
                continue
            allowed = {c for c in colls if c.startswith(ex["db_id"] + "__")}
            rec = run_one(mdb, provider, ex, cfg["generation"], cfg["execution"], allowed)
            done[ex["id"]] = rec
            K.write_jsonl(ex_path, [done[r["id"]] for r in rows if r["id"] in done])
            print(f"[{i + 1}/{len(rows)}] {ex['id']} {rec['result']['category']} {rec['latency_s']}s",
                  flush=True)
            if "DAILY_QUOTA_EXHAUSTED" in (rec.get("provider_error") or ""):
                print("stopping: provider daily quota exhausted (resume later; finished ids are kept)")
                break
    finally:
        client.close()
    records = [done[r["id"]] for r in rows if r["id"] in done]
    summ = summarize(records, cfg["pricing"].get(a.provider, {}), label,
                     getattr(provider, "model_id", None))
    summ.update(provider=a.provider, config=cfg["models"].get(a.provider), generation=cfg["generation"],
                wall_clock_s=round(time.time() - t0, 1), dataset=dataset)
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("n", "correct", "execution_accuracy", "by_hardness",
                                           "categories", "latency_s", "cost_usd")}, indent=1))


if __name__ == "__main__":
    main()
