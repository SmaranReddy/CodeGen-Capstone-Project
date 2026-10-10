"""Reproducible runner for the 68-example NL -> MQL model comparison.

Pipeline per example (identical for both models):
    prompt (same schema + question) -> raw output -> parse JSON -> structural validation
    -> read-only execution on MongoDB -> official Spider comparator vs gold.

No reference SQL/MQL, gold CSV contents, expected results or prior answers are given to
a model. Destructive stages ($out/$merge) and server-side JavaScript are rejected before
execution. Transient transport errors may be retried; a semantically wrong answer is never
retried or corrected against the gold.

Usage:
    python scripts/evaluation/run_nl_mql_eval.py probe
    python scripts/evaluation/run_nl_mql_eval.py smoke --provider reference
    python scripts/evaluation/run_nl_mql_eval.py run --provider reference --restore group
    python scripts/evaluation/run_nl_mql_eval.py run --provider qwen --ids local041 local133
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompt_template as PT  # noqa: E402
from eval_kit import (ROOT, allowed_collections, execute_pipeline, get_client, imports_for,
                      load_artifact, load_dataset, sample_collection_schema, score_against_gold,
                      start_collection, to_dataframe, validate_mql)  # noqa: E402
from providers import ProviderError, build_provider  # noqa: E402

CONFIG_PATH = ROOT / "configs/evaluation_68.yaml"


def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_restore_module():
    spec = importlib.util.spec_from_file_location("restore_data",
                                                  Path(__file__).resolve().parent / "restore_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_output(text):
    if text is None:
        return None, "empty output"
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t), None
    except Exception:
        pass
    start = t.find("{")
    if start == -1:
        return None, "no JSON object found"
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1]), None
                except Exception as e:
                    return None, f"JSON parse error: {e}"
    return None, "unbalanced JSON braces"


def reference_pipelines():
    out = {}
    for r in load_dataset():
        iid = r["instance_id"]
        art = load_artifact(iid)
        mql = art.get("mql") or {}
        if mql:
            out[iid] = {"method": mql.get("method", "aggregate"),
                        "collection": mql.get("collection"), "pipeline": mql.get("pipeline")}
    return out


def example_schema(db, iid, artifact):
    from eval_kit import schema_for
    schema = schema_for(iid, artifact)
    for coll, fields in list(schema.items()):
        if not fields and coll in db.list_collection_names():
            schema[coll] = sample_collection_schema(db, coll)
    return schema


def process_example(db, cfg, provider, iid, artifact, rec, generation, exec_cfg):
    row = next(r for r in load_dataset() if r["instance_id"] == iid)
    allowed = allowed_collections(iid, artifact)
    schema = example_schema(db, iid, artifact)
    messages = PT.messages(row.get("db"), row.get("question"), schema)
    gen = dict(generation)
    gen["_iid"] = iid

    t0 = time.time()
    out = provider.generate(messages, PT.SYSTEM, gen)
    elapsed = time.time() - t0
    rec_out = {
        "instance_id": iid, "db": row.get("db"), "question": row.get("question"),
        "model_id": out.get("model_id"), "provider": provider.name,
        "prompt_version": PT.PROMPT_VERSION, "prompt_sha256": PT.template_hash(),
        "generation": {k: v for k, v in generation.items()},
        "schema": schema, "allowed_collections": sorted(allowed),
        "raw_text": out.get("text"), "provider_error": out.get("error"),
        "latency_s": round(out.get("latency_s") or elapsed, 3),
        "input_tokens": out.get("input_tokens"), "output_tokens": out.get("output_tokens"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    result = {"error_category": None, "failure_reason": None, "score": None,
              "matched_gold": [], "executed": False, "result_rows": None,
              "parsed": None, "structural_ok": False, "exec_status": None}

    if out.get("error"):
        result.update(error_category="provider_error", failure_reason=out["error"])
        rec_out["result"] = result
        return rec_out

    parsed, perr = parse_output(out.get("text"))
    if parsed is None:
        result.update(error_category="format_error", failure_reason=perr)
        rec_out["result"] = result
        return rec_out
    result["parsed"] = parsed

    ok, reason, norm = validate_mql(parsed, allowed)
    result["structural_ok"] = ok
    if not ok:
        result.update(error_category="invalid_mql", failure_reason=reason)
        rec_out["result"] = result
        return rec_out

    existing = _retry_transient(db.list_collection_names)
    missing = [c for c in {norm["collection"]} | _refs(norm["pipeline"]) if c not in existing]
    if missing:
        result.update(error_category="infrastructure_missing_data",
                      failure_reason=f"collections not in MongoDB: {missing}")
        rec_out["result"] = result
        return rec_out

    try:
        docs = _retry_transient(lambda: execute_pipeline(
            db, norm["collection"], norm["pipeline"],
            timeout_ms=exec_cfg["timeout_ms"],
            allow_disk_use=exec_cfg["allow_disk_use"]))
        result["executed"] = True
        result["result_rows"] = len(docs)
        result["exec_status"] = "ok"
        if len(docs) > exec_cfg.get("max_result_rows", 200000):
            result.update(error_category="execution_error", failure_reason="result too large")
            rec_out["result"] = result
            return rec_out
        pred = to_dataframe(docs)
        verdicts, passed, matched = score_against_gold(pred, iid)
        result["score"] = int(passed)
        result["matched_gold"] = matched
        result["verdicts"] = {k: v["score"] for k, v in verdicts.items()}
        if not passed:
            result.update(error_category="incorrect", failure_reason="result did not match any gold")
        rec_out["preview_rows"] = docs[:100]
    except Exception as e:
        result.update(error_category="execution_error", failure_reason=f"{type(e).__name__}: {e}")
    rec_out["result"] = result
    return rec_out


def _refs(pipeline):
    from eval_kit import referenced_collections
    return referenced_collections(pipeline)


_TRANSIENT = ("resolution lifetime expired", "ServerSelectionTimeout", "AutoReconnect",
              "Connection reset", "connection closed", "Temporary failure", "network",
              "timed out", "No address associated", "getaddrinfo failed", "WinError")


def _retry_transient(fn, tries=5, delay=8.0):
    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e)
            if attempt >= tries or not any(k in msg for k in _TRANSIENT):
                raise
            print(f"    [retry {attempt}/{tries - 1}] {msg.splitlines()[0][:120]}")
            time.sleep(delay * attempt)
    raise last


def cmd_probe(cfg):
    model_cfgs = cfg["models"]
    providers = [
        ("reference", {"backend": "reference"}, None),
        ("qwen", model_cfgs.get("qwen"), None),
        ("frontier", model_cfgs.get("frontier"), None),
    ]
    report = {}
    for name, mcfg, _ in providers:
        p = None
        try:
            if name == "reference":
                p = build_provider({"backend": "reference"}, "reference", reference_pipelines())
            else:
                p = build_provider(mcfg, name)
            ok, msg = p.available()
        except Exception as e:
            ok, msg = False, f"{type(e).__name__}: {e}"
        report[name] = {"model_id": getattr(p, "model_id", None) if p else None,
                        "available": ok, "detail": msg}
        print(f"{name:10s} available={ok}  {msg}")
    out = cfg["output_dir"] + "/provider_probe.json"
    (ROOT / out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return report


def _run(cfg, provider_name, ids, restore_mode, keep, limit, label):
    rows = load_dataset()
    if ids:
        rows = [r for r in rows if r["instance_id"] in ids]
    if limit:
        rows = rows[:limit]
    generation = cfg["generation"]
    exec_cfg = cfg["execution"]
    model_cfgs = cfg["models"]
    pipeline_by_iid = reference_pipelines()
    if provider_name == "reference":
        provider = build_provider({"backend": "reference"}, "reference", pipeline_by_iid)
    else:
        provider = build_provider(model_cfgs[provider_name], provider_name)
    ok, msg = provider.available()
    if not ok:
        raise SystemExit(f"provider '{provider_name}' unavailable: {msg}")

    out_dir = ROOT / cfg["output_dir"] / label
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    client = get_client()
    restore = load_restore_module()
    log = restore.load_restore_log()
    records = []
    started = time.time()
    execlog = open(out_dir / "executions.jsonl", "w", encoding="utf-8")
    try:
        db = client[cfg["pilot_db"]]
        groups = defaultdict(list)
        for r in rows:
            groups[r.get("db")].append(r["instance_id"])
        for grp, gids in groups.items():
            if restore_mode in ("group", "auto"):
                try:
                    print(f"== restore group {grp} ==")
                    restore.import_group(grp, log)
                except Exception as e:
                    print(f"!! group {grp} import failed: {type(e).__name__}: {e}")
            for iid in gids:
                try:
                    art = load_artifact(iid)
                    rec = _manifest_rec(iid)
                    rec_out = process_example(db, cfg, provider, iid, art, rec, generation, exec_cfg)
                except Exception as e:
                    rec_out = _infra_record(provider, iid, f"runner error: {type(e).__name__}: {e}")
                (out_dir / "raw" / f"{iid}.json").write_text(
                    json.dumps(rec_out, indent=2, default=str), encoding="utf-8")
                records.append(rec_out)
                execlog.write(json.dumps(rec_out, default=str, ensure_ascii=False) + "\n")
                execlog.flush()
                r = rec_out["result"]
                print(f"[{iid}] {r['error_category'] or 'correct'} "
                      f"score={r['score']} rows={r['result_rows']} {rec_out['latency_s']}s")
            if restore_mode == "group" and not keep:
                try:
                    print(f"== drop group {grp} ==")
                    restore.drop_group(grp, log)
                except Exception as e:
                    print(f"!! group {grp} drop failed: {e}")
    finally:
        execlog.close()
        client.close()

    summary = summarize(records, cfg, provider_name, label, time.time() - started, rows)
    (out_dir / "run_metadata.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    return records, summary


def _manifest_rec(iid):
    from conversion_kit import manifest
    return manifest()[iid]


def _infra_record(provider, iid, reason):
    return {"instance_id": iid, "model_id": getattr(provider, "model_id", None),
            "provider": provider.name, "prompt_version": PT.PROMPT_VERSION,
            "prompt_sha256": PT.template_hash(), "raw_text": None, "latency_s": 0.0,
            "input_tokens": None, "output_tokens": None,
            "result": {"error_category": "infrastructure_error", "failure_reason": reason,
                       "score": None, "matched_gold": [], "executed": False,
                       "result_rows": None, "parsed": None, "structural_ok": False,
                       "exec_status": None}}


def summarize(records, cfg, provider_name, label, elapsed, rows):
    cat = Counter(r["result"]["error_category"] or "correct" for r in records)
    scored = [r for r in records if r["result"]["score"] is not None]
    correct = sum(1 for r in scored if r["result"]["score"] == 1)
    exec_ok = sum(1 for r in records if r["result"]["executed"])
    parse_ok = sum(1 for r in records if r["result"]["parsed"] is not None)
    struct_ok = sum(1 for r in records if r["result"]["structural_ok"])
    infra = sum(1 for r in records if (r["result"]["error_category"] or "").startswith("infrastructure"))
    in_tok = sum(r["input_tokens"] or 0 for r in records)
    out_tok = sum(r["output_tokens"] or 0 for r in records)
    price = (cfg.get("pricing") or {}).get(provider_name) or {}
    cost = None
    if price.get("input_per_1k") is not None and price.get("output_per_1k") is not None:
        cost = round(in_tok / 1000 * price["input_per_1k"] + out_tok / 1000 * price["output_per_1k"], 4)
    metrics = {
        "attempted": len(records),
        "parsed_valid_json": parse_ok,
        "structurally_valid_mql": struct_ok,
        "executed": exec_ok,
        "scored": len(scored),
        "correct": correct,
        "execution_accuracy": round(correct / len(scored), 4) if scored else None,
        "infrastructure_failures": infra,
        "error_categories": dict(cat),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "estimated_cost_usd": cost,
        "total_runtime_s": round(elapsed, 2),
        "total_model_latency_s": round(sum(r["latency_s"] for r in records), 2),
    }
    return {"provider": provider_name, "label": label, "model_id": records[0]["model_id"] if records else None,
            "prompt_version": PT.PROMPT_VERSION, "prompt_sha256": PT.template_hash(),
            "generation": cfg["generation"], "metrics": metrics}


def cmd_run(args, cfg):
    label = args.label or f"{args.provider}"
    if args.provider == "qwen":
        label = args.label or "qwen"
    elif args.provider == "frontier":
        label = args.label or "frontier"
    _run(cfg, args.provider, args.ids, args.restore, args.keep, args.limit, label)


def cmd_smoke(args, cfg):
    ids = args.ids or ["local041", "local133", "local163"]
    print(f"smoke test on {ids} with provider '{args.provider}'")
    _run(cfg, args.provider, ids, args.restore or "group", args.keep, None, "smoke")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    s = sub.add_parser("smoke")
    s.add_argument("--provider", default="reference")
    s.add_argument("--ids", nargs="*")
    s.add_argument("--restore", default="group", choices=["none", "group", "auto"])
    s.add_argument("--keep", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--provider", required=True)
    r.add_argument("--ids", nargs="*")
    r.add_argument("--limit", type=int)
    r.add_argument("--restore", default="none", choices=["none", "group", "auto"])
    r.add_argument("--keep", action="store_true")
    r.add_argument("--label")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "probe":
        cmd_probe(cfg)
    elif args.cmd == "smoke":
        cmd_smoke(args, cfg)
    else:
        cmd_run(args, cfg)


if __name__ == "__main__":
    main()
