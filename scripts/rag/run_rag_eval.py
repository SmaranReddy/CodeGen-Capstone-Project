"""Run a RAG experiment: retrieve -> build prompt -> generate -> validate -> execute -> score.

Stages are separate functions/modules: retrieval.py (ranking), rag_prompt.py (context + prompt),
providers.py (generation, unchanged), run_eval.score_output (parse, read-only validation, Atlas
aggregate, exec_match; unchanged and shared with the zero-shot benchmark).

Each record of outputs/rag_week3/<experiment>/executions.jsonl stores the experiment config and
its hash, the retrieval query, the ranked candidates (doc id, score, rank), the selected and
budget-dropped document ids, the inserted context, the raw output, the validated MQL, the
execution result, the verdict, and retrieval / generation / scoring / total latency.

Resumable: a finished id is never regenerated unless --fresh; a record whose experiment hash
differs from the current config is refused (change the experiment name instead).
Provider errors are always re-run. The test partition is refused without --allow-test.

Usage:
  python scripts/rag/run_rag_eval.py --experiment hybrid_k3 [--limit 5] [--ids ...]
  python scripts/rag/run_rag_eval.py --experiment hybrid_k3 --dry-run   # retrieval + prompts only
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

import common as C
from common import K
import patterns as PT
import rag_prompt as RP
import retrieval as RT

import prompt_spider1 as P  # noqa: E402
import run_eval as R  # noqa: E402
from providers import build_provider  # noqa: E402

CONFIG = C.ROOT / "configs/rag_week3.yaml"
N_CANDIDATES_SAVED = 10


def load_configs(path=CONFIG):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    base_path = C.ROOT / cfg["base_config"]
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    return cfg, base, C.sha256_file(base_path)


def experiment_hash(name, exp, cfg, base, corpus_sha):
    blob = {"name": name, "exp": exp, "context": cfg["context"], "provider": cfg["provider"],
            "model": base["models"][cfg["provider"]], "generation": base["generation"],
            "execution": base["execution"], "corpus_sha256": corpus_sha,
            "rag_prompt": RP.template_hash(), "zero_shot_prompt": P.template_hash()}
    if exp.get("schema_notes"):
        blob["notes_header"] = RP.NOTES_HEADER
    return C.sha256_bytes(C.canon(blob).encode("utf-8"))


def run_dir_name(experiment, partition):
    """validation runs keep the bare experiment name; other partitions get a suffix."""
    return experiment if partition == "validation" else f"{experiment}__{partition}"


def make_counter(cfg):
    from llama_cpp import Llama
    tok = Llama(model_path=str(C.ROOT / cfg["tokenizer_gguf"]), vocab_only=True, verbose=False)
    return lambda text: len(tok.tokenize(text.encode("utf-8"), add_bos=False, special=False))


_TABLES = None


def schema_notes(ex, exp, collections):
    global _TABLES
    if exp.get("schema_notes") != "fk_type_mismatch":
        return []
    if _TABLES is None:
        _TABLES = K.load_json(K.SPIDER / "tables.json")
    return RP.fk_type_notes(ex["db_id"], collections, _TABLES)


def retrieve_and_prompt(ex, exp, retriever, info, counter, cfg, base, collections=None):
    t0 = time.perf_counter()
    if exp["retrieval"] == "none" or (exp["k"] == 0 and exp["k_guides"] == 0):
        ranked, guides, qtext = [], [], None
    else:
        qtext = retriever.query_text(ex["question"], info)
        ranked = retriever.rank_examples(ex["question"], info, ex["id"]) if exp["k"] else []
        guides = retriever.rank_guides(ex["question"], info, [d for d, _ in ranked[:exp["k"]]]) \
            if exp["k_guides"] else []
    t_ret = time.perf_counter() - t0
    gen = base["generation"]
    msgs, pinfo = RP.build_messages(
        ex["db_id"], ex["question"], ranked, guides, exp["k"], exp["k_guides"], counter,
        cfg["context"]["max_context_tokens"], base["models"][cfg["provider"]]["n_ctx"],
        gen["max_new_tokens"], cfg["context"]["margin_tokens"],
        notes=schema_notes(ex, exp, collections) if collections is not None else ())
    t_ctx = time.perf_counter() - t0 - t_ret
    retrieval = {
        "method": exp["retrieval"], "query_text": qtext,
        "query_cues": sorted(PT.question_cues(ex["question"], info)),
        "candidates": [{"doc_id": d["doc_id"], "rank": i + 1, "score": s, "db_id": d["db_id"]}
                       for i, (d, s) in enumerate(ranked[:N_CANDIDATES_SAVED])],
        "guide_candidates": [{"doc_id": g["doc_id"], "rank": i + 1, "score": s}
                             for i, (g, s) in enumerate(guides[:N_CANDIDATES_SAVED])],
        "n_candidates": len(ranked), "empty_retrieval": exp["k"] > 0 and not ranked,
        **pinfo, "retrieval_latency_s": round(t_ret, 4), "context_build_latency_s": round(t_ctx, 4)}
    return msgs, retrieval


def run_one(mdb, provider, ex, msgs, retrieval, base, allowed):
    t0 = time.perf_counter()
    out = provider.generate(msgs, P.SYSTEM, dict(base["generation"]))
    t_gen = time.perf_counter() - t0
    rec = {"id": ex["id"], "db_id": ex["db_id"], "hardness": ex["hardness"], "question": ex["question"],
           "model_id": out.get("model_id"), "prompt_version": RP.RAG_PROMPT_VERSION,
           "prompt_sha256": RP.template_hash(), "zero_shot_prompt_sha256": P.template_hash(),
           "retrieval": retrieval, "raw_text": out.get("text"), "provider_error": out.get("error"),
           "input_tokens": out.get("input_tokens"), "output_tokens": out.get("output_tokens"),
           "latency_s": round(out.get("latency_s") or 0.0, 3), "generation_wall_s": round(t_gen, 3),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "expected_columns": ex.get("gold_columns"), "expected_rows": ex["gold_rows"]}
    t1 = time.perf_counter()
    if out.get("error"):
        rec["result"] = {"correct": False, "category": "provider_error", "reason": out["error"], "executed": False}
    else:
        res, extra = R.score_output(mdb, out.get("text"), ex, base["execution"], allowed)
        rec["result"] = res
        rec.update(extra)
    rec["scoring_latency_s"] = round(time.perf_counter() - t1, 3)
    rec["total_latency_s"] = round(retrieval["retrieval_latency_s"] + retrieval["context_build_latency_s"]
                                   + t_gen + rec["scoring_latency_s"], 3)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--partition", choices=["train", "validation", "test"])
    ap.add_argument("--allow-test", action="store_true", help="required to touch the test partition")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="retrieval + prompts only; no model, no Atlas")
    a = ap.parse_args()
    cfg, base, base_sha = load_configs(a.config)
    exp = cfg["experiments"][a.experiment]
    partition = a.partition or cfg["partition"]
    if partition == "test" and not a.allow_test:
        sys.exit("refusing to run on the test partition without --allow-test")
    rows = K.read_jsonl(C.ROOT / base["dataset"])
    db_part, _ = C.load_split(C.ROOT / cfg["split_dir"])
    rows = [r for r in rows if db_part[r["db_id"]] == partition]
    if a.ids:
        rows = [r for r in rows if r["id"] in a.ids]
    if a.limit:
        rows = rows[:a.limit]
    corpus_dir = C.ROOT / cfg["corpus_dir"]
    corpus = C.load_corpus(corpus_dir)
    corpus_sha = C.sha256_file(corpus_dir / "corpus.jsonl")
    collections = C.load_collections()
    retriever = RT.Retriever(corpus, exp["retrieval"] if exp["retrieval"] != "none" else "bm25_masked",
                             hybrid_weight=exp.get("hybrid_weight", 0.5), seed=exp.get("seed", 20261020))
    counter = make_counter(cfg)
    ehash = experiment_hash(a.experiment, exp, cfg, base, corpus_sha)
    code_sha = C.sha256_bytes(b"".join((C.HERE / f).read_bytes() for f in
                                       ("retrieval.py", "patterns.py", "rag_prompt.py", "run_rag_eval.py")))
    dataset_sha = C.sha256_file(C.ROOT / base["dataset"])
    meta = {"experiment": a.experiment, "experiment_sha256": ehash, "code_sha256": code_sha,
            "dataset_sha256": dataset_sha, "split_dir": cfg["split_dir"], "atlas_db": base["atlas_db"],
            "tokenizer_gguf": cfg["tokenizer_gguf"],
            "config": exp, "partition": partition,
            "base_config": cfg["base_config"], "base_config_sha256": base_sha, "corpus_sha256": corpus_sha}
    out_dir = C.ROOT / cfg["output_dir"] / run_dir_name(a.experiment, partition)
    if a.dry_run:
        for ex in rows:
            info = PT.schema_info(ex["db_id"], collections)
            msgs, ret = retrieve_and_prompt(ex, exp, retriever, info, counter, cfg, base, collections)
            print(ex["id"], ret["selected_examples"], ret["selected_guides"], ret["context_tokens"],
                  ret["prompt_tokens_est"], ret["dropped_budget"])
        return
    mc = dict(base["models"][cfg["provider"]])
    mc["gguf_path"] = str(C.ROOT / mc["gguf_path"])
    provider = build_provider(mc, cfg["provider"])
    ok, msg = provider.available()
    print(f"provider {cfg['provider']}: {msg}")
    if not ok:
        raise SystemExit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    ex_path = out_dir / "executions.jsonl"
    done = {}
    if ex_path.exists() and not a.fresh:
        for r in K.read_jsonl(ex_path):
            if r["experiment_sha256"] != ehash or r["partition"] != partition:
                sys.exit(f"{ex_path} was produced by a different config ({r['id']}); use a new experiment name")
            for key, now in (("dataset_sha256", dataset_sha), ("code_sha256", code_sha)):
                if key in r and r[key] != now:
                    sys.exit(f"{ex_path}: {key} of {r['id']} differs from the current one; refusing to "
                             "mix records (use a new experiment name)")
            if r["result"]["category"] != "provider_error":
                done[r["id"]] = r
    order = [r["id"] for r in K.read_jsonl(C.ROOT / base["dataset"])]
    client = K.mongo_client()
    mdb = client[base["atlas_db"]]
    colls = mdb.list_collection_names()
    try:
        for i, ex in enumerate(rows):
            if ex["id"] in done:
                continue
            info = PT.schema_info(ex["db_id"], collections)
            msgs, ret = retrieve_and_prompt(ex, exp, retriever, info, counter, cfg, base, collections)
            allowed = {c for c in colls if c.startswith(ex["db_id"] + "__")}
            rec = {**meta, **run_one(mdb, provider, ex, msgs, ret, base, allowed)}
            done[ex["id"]] = rec
            K.write_jsonl(ex_path, [done[x] for x in order if x in done])
            print(f"[{i + 1}/{len(rows)}] {ex['id']} {rec['result']['category']} ctx={ret['context_tokens']} "
                  f"gen={rec['generation_wall_s']}s", flush=True)
    finally:
        client.close()
    recs = [done[x] for x in order if x in done]
    summ = R.summarize(recs, {}, a.experiment, getattr(provider, "model_id", None))
    summ.update(prompt_version=RP.RAG_PROMPT_VERSION, prompt_sha256=RP.template_hash(), **meta)
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("n", "correct", "execution_accuracy", "categories")}, indent=1))


if __name__ == "__main__":
    main()
