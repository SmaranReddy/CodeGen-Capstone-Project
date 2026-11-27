"""MQL-0009 retrieval-control experiment (zero-shot vs random vs fixed vs hybrid RAG, Qwen2.5-Coder-3B-Instruct,
held-out Spider-train databases): package core.

Shipped as src/mql0009_core.py of the Kaggle package and imported by the notebook, the package builder, the Windows
scoring adapter and the tests. No torch / transformers import at module level. No MongoDB, no credentials, no network.

Re-use, not re-implementation: the MQL-0008 package core (src/mql0008_core.py), retrieval (src/retrieval.py),
patterns (src/patterns.py) and reference-block rendering (src/rag_block.py) are shipped BYTE-IDENTICAL to the
MQL-0008 package copies (a test asserts it). From mql0008_core this module uses, unchanged:
  build_docs, eval_schema_infos, select_examples (budget loop), prompt_sha, prompts_for_arm, leakage_report,
  Checkpoint, run_generation, oom_safe_generator, make_qwen_generator, verify_package, the hashing and IO helpers.
The only intended difference between the four arms is how (up to) three demonstrations are chosen:
  A_zero_shot   none
  B_random_sN   retrieval.Retriever(method="random", seed=N)  (the unmodified MQL-0005/MQL-0007 random retriever)
  C_fixed       the three frozen corpus ids of configs/experiment.json, in their frozen order, for every question
  D_hybrid      retrieval.Retriever(method="hybrid_masked", hybrid_weight=1.0)  (the MQL-0008 arm B retriever)
Every demonstration arm then goes through the same budget loop, the same render_block / insert_block, the same
evaluation-specific exclusions, the same generator and the same decoding.
"""
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mql0008_core as M8      # noqa: E402  (byte-identical copy of the MQL-0008 package core)
import patterns as PT          # noqa: E402  (byte-identical copy)
import retrieval as RT         # noqa: E402  (byte-identical copy)
import rag_block as RB         # noqa: E402  (byte-identical copy)

EXPERIMENT = "MQL-0009"
PACKAGE = "nl2mql_mql0009_retrieval_control"
STATUS_FILE = "MQL0009_STATUS.json"
MANIFEST_FILE = "MQL0009_PREDICTIONS_MANIFEST.json"
CORPUS_FILE = "data/corpus/approved_corpus_v2_1.jsonl"
EXPECTED_CORPUS = 3445
EXPECTED_CORPUS_SHA256 = "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"
K_DEMOS = 3
FAMILIES = ("zero_shot", "random", "fixed", "hybrid")
# keys of run_config.json that are written after the fingerprint is computed
POST_FP = {"fingerprint", "generation_wall_s", "generated_this_session", "environment", "revision_check", "sessions"}
ARM_FILES = ("predictions.jsonl", "run_config.json", "retrieval_log.jsonl", "checkpoints/predictions.partial.jsonl")
GOLD_KEYS = ("sql", "gold", "gold_sql", "gold_mql", "expected_output", "result_sha256", "gold_rows", "gold_columns",
             "ordered_sha256", "multiset_sha256", "outcome", "correct", "query_class", "hardness", "tie_blocks")

PreflightError = M8.PreflightError
need = M8.need
sha256_file, sha256_text, canon, fingerprint, now = M8.sha256_file, M8.sha256_text, M8.canon, M8.fingerprint, M8.now
read_jsonl, write_jsonl, write_json = M8.read_jsonl, M8.write_jsonl, M8.write_json


# ------------------------------------------------------------------ arm runs
def arm_runs(cfg):
    """Ordered arm runs [{name, family, seed}] from the frozen config. 1 + 5 + 1 + 1 = 8 runs."""
    out = []
    for name, a in cfg["arm_runs"].items():
        need(a["family"] in FAMILIES, f"{name}: unknown family {a['family']}")
        out.append({"name": name, "family": a["family"], "seed": a.get("seed"), "k": a["k"]})
    return out


def check_arm_design(cfg):
    """The registered design: exactly one zero-shot, five random seeds, one fixed, one hybrid run; K=3 for all
    demonstration runs; distinct seeds equal to the registered list; three distinct fixed ids."""
    runs = arm_runs(cfg)
    fam = [r["family"] for r in runs]
    need(fam.count("zero_shot") == 1 and fam.count("random") == 5 and fam.count("fixed") == 1 and fam.count("hybrid") == 1
         and len(runs) == 8, f"arm design is not 1 zero-shot + 5 random + 1 fixed + 1 hybrid: {fam}")
    for r in runs:
        need(r["k"] == (0 if r["family"] == "zero_shot" else K_DEMOS), f"{r['name']}: k={r['k']} (must be 0 / {K_DEMOS})")
    seeds = [r["seed"] for r in runs if r["family"] == "random"]
    need(seeds == cfg["random_retrieval"]["seeds"] and len(set(seeds)) == 5 and all(isinstance(s, int) for s in seeds),
         f"random seeds {seeds} != registered {cfg['random_retrieval']['seeds']}")
    fx = cfg["fixed_retrieval"]["ids"]
    need(len(fx) == K_DEMOS and len(set(fx)) == K_DEMOS, f"fixed ids must be {K_DEMOS} distinct ids: {fx}")
    return runs


# ------------------------------------------------------------------ eligible corpus view (never modifies the corpus)
def eligible_view(corpus_rows, eval_db_ids, excluded_db_ids=()):
    """Deterministic filtered index view: corpus rows whose database is neither an evaluation database nor an
    explicitly excluded one, in corpus order. Returns (rows, view descriptor). The corpus file itself is untouched."""
    bad = set(eval_db_ids) | set(excluded_db_ids)
    rows = [r for r in corpus_rows if r["db_id"] not in bad]
    ids = [r["id"] for r in rows]
    need(len(ids) == len(set(ids)), "duplicate ids in the corpus view")
    return rows, {"n": len(rows), "n_source": len(corpus_rows), "ids_sha256": sha256_text("\n".join(ids)),
                  "removed": len(corpus_rows) - len(rows), "db_ids": sorted({r["db_id"] for r in rows}),
                  "rule": "corpus rows whose db_id is not an evaluation database (and not otherwise excluded), corpus order"}


def eligible_ok(row):
    return (row["id"].startswith("spider_train_") and row["verification"]["outcome"] == "ACCEPTED"
            and row["verification"]["verifier"] == "sql2mql-v2-verifier-3.1" and row["converter_version"] == "sql2mql-v2.1")


# ------------------------------------------------------------------ ranking per arm family
class FixedRanker:
    """Every question gets the same frozen ids, in the frozen order. Target-db and duplicate filtering exactly as in
    retrieval.Retriever.rank_examples (the frozen ids come from non-evaluation databases, so nothing is filtered)."""

    def __init__(self, docs, ids):
        by = {d["doc_id"]: d for d in docs}
        missing = [i for i in ids if i not in by]
        need(not missing, f"fixed ids not in the eligible corpus view: {missing}")
        self.docs = [by[i] for i in ids]

    def rank_examples(self, question, info, qid=""):
        return [(d, 0.0) for d in self.docs if d.get("db_id") != info["db_id"]]


def make_ranker(cfg, run, docs):
    fam = run["family"]
    if fam == "zero_shot":
        return None
    if fam == "random":
        return RT.Retriever(docs, method="random", seed=run["seed"])
    if fam == "fixed":
        return FixedRanker(docs, cfg["fixed_retrieval"]["ids"])
    return RT.Retriever(docs, method="hybrid_masked", hybrid_weight=cfg["retrieval"]["hybrid_weight"])


def retriever_constants_ok(cfg):
    import inspect
    sig = inspect.signature(RT.BM25.__init__).parameters
    need(sig["k1"].default == cfg["retrieval"]["bm25_k1"] and sig["b"].default == cfg["retrieval"]["bm25_b"],
         "retrieval.py BM25 defaults differ from configs/experiment.json")
    return True


# ------------------------------------------------------------------ plan (selection + budget), per arm run
def build_plan(cfg, run, eval_rows, frozen_prompts, docs, eval_infos, count_tokens):
    """Per-question plan for one arm run. Same row layout as the MQL-0008 plan plus `arm_family` / `seed`.
    For the hybrid run this reproduces mql0008_core.build_plan row for row (tested). Exclusions are applied AFTER
    ranking (index, IDF and dedup untouched); the next eligible example takes the place."""
    ctx = cfg["context"]
    exclusions = cfg.get("retrieval_exclusions", [])
    retr = make_ranker(cfg, run, docs)
    by_prompt = {p["id"]: p for p in frozen_prompts}
    rows = []
    for e in eval_rows:
        fp = by_prompt[e["id"]]
        system, base = fp["system"], fp["messages"][0]["content"]
        excl_ids = {x["corpus_id"] for x in exclusions if x["eval_id"] == e["id"]}
        excluded, rank_of = [], {}
        if retr is None:
            ranked, chosen, dropped, block, bt, total = [], [], [], "", 0, None
        else:
            ranked_full = retr.rank_examples(e["question"], eval_infos[e["db_id"]], qid=e["id"])
            rank_of = {d["doc_id"]: i + 1 for i, (d, _) in enumerate(ranked_full)}
            excluded = [{"corpus_id": d["doc_id"], "original_rank": i + 1}
                        for i, (d, _) in enumerate(ranked_full) if d["doc_id"] in excl_ids]
            ranked = [(d, s) for d, s in ranked_full if d["doc_id"] not in excl_ids]
            chosen, dropped, block, bt, total = M8.select_examples(
                e["db_id"], system, base, ranked, run["k"], count_tokens, ctx["max_context_tokens"], ctx["n_ctx"],
                ctx["max_new_tokens"], ctx["margin_tokens"])
        user = RB.insert_block(base, block)
        score = {d["doc_id"]: s for d, s in ranked[:200]}
        rows.append({
            "id": e["id"], "db_id": e["db_id"], "arm": run["name"],
            "selected_ids": [d["doc_id"] for d in chosen],
            "selected_scores": [score.get(d["doc_id"]) for d in chosen],
            "selected_ranks": [rank_of[d["doc_id"]] for d in chosen],
            "selected_db_ids": [d["db_id"] for d in chosen],
            "ranked_top10": [[d["doc_id"], s] for d, s in ranked[:10]],
            "excluded_by_rule": excluded,
            "dropped_budget": dropped, "context_tokens": bt, "prompt_tokens_est": total,
            "empty_context": not block,
            "user_prompt_sha256": sha256_text(user),
            "prompt_sha256": M8.prompt_sha(system, [{"role": "user", "content": user}]),
            "arm_family": run["family"], "seed": run["seed"],
        })
    return rows


def plan_checks(cfg, run, plan, eval_rows, docs_by_id):
    """Gates every plan must pass (builder AND notebook): K exactly 3 for demonstration runs (0 for zero-shot), no
    target-db example, no non-corpus id, context cap, exclusions honoured, no selected near-duplicate, fixed arm uses
    exactly the frozen ids in order."""
    k = run["k"]
    for p in plan:
        need(len(p["selected_ids"]) == k, f"{run['name']} {p['id']}: {len(p['selected_ids'])} demonstrations, K must be {k}")
        need(len(set(p["selected_ids"])) == len(p["selected_ids"]), f"{run['name']} {p['id']}: duplicate demonstrations")
        need(all(d != p["db_id"] for d in p["selected_db_ids"]), f"{run['name']} {p['id']}: example from the target db")
        need(all(i in docs_by_id for i in p["selected_ids"]), f"{run['name']} {p['id']}: id outside the eligible view")
        need(p["context_tokens"] <= cfg["context"]["max_context_tokens"], f"{run['name']} {p['id']}: context cap exceeded")
        if run["family"] == "fixed":
            need(p["selected_ids"] == cfg["fixed_retrieval"]["ids"], f"{p['id']}: fixed arm did not use the frozen ids")
        if run["family"] == "zero_shot":
            need(p["empty_context"], f"{p['id']}: zero-shot must have an empty context")
    for x in cfg.get("retrieval_exclusions", []):
        row = next((p for p in plan if p["id"] == x["eval_id"]), None)
        need(row is None or x["corpus_id"] not in row["selected_ids"], f"{run['name']}: exclusion {x} violated")
    viol = M8.selection_neardup_violations(plan, eval_rows, docs_by_id, cfg["leakage_gates"]["near_duplicate_token_jaccard"])
    need(not viol, f"{run['name']}: a selected example is a near-duplicate of its query: {viol[:3]}")
    return True


def prompts_for_run(plan, frozen_prompts, docs_by_id):
    return M8.prompts_for_arm(plan, frozen_prompts, docs_by_id)


def plan_sha(plan):
    return sha256_text(canon(plan))


# ------------------------------------------------------------------ output normalisation (informational)
def parse_output(text):
    """VERBATIM copy of scripts/nl2mql/dev1034_core.parse_output (the MQL-0006 evaluator's parser; a test asserts
    equality of the source). The Windows evaluator re-parses raw_text and is the only authority for scoring."""
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


def normalize_prediction(raw_text, status):
    """-> {parse_status, parse_error, normalized_mql}. Same normalisation steps as dev1034_core.check_mql before its
    safety gate (method defaulted to aggregate, empty pipeline -> [{"$match": {}}]); NO safety or schema validation
    here (that is the evaluator's job on Windows)."""
    if status != "ok":
        return {"parse_status": "generation_failure", "parse_error": None, "normalized_mql": None}
    parsed, err = parse_output(raw_text)
    if parsed is None:
        return {"parse_status": "unparseable", "parse_error": err, "normalized_mql": None}
    if isinstance(parsed, dict) and "method" not in parsed:
        parsed = {"method": "aggregate", **parsed}
    if isinstance(parsed, dict) and parsed.get("method") == "aggregate" and parsed.get("pipeline") == []:
        parsed = {**parsed, "pipeline": [{"$match": {}}]}
    shape_ok = (isinstance(parsed, dict) and isinstance(parsed.get("collection"), str)
                and isinstance(parsed.get("pipeline"), list))
    return {"parse_status": "parsed" if shape_ok else "parsed_bad_shape", "parse_error": None,
            "normalized_mql": parsed if isinstance(parsed, dict) else {"_non_object": parsed}}


# ------------------------------------------------------------------ hardware / precision / revision
def gpu_runtime_info():
    """GPU facts that decide precision compatibility. bf16 on compute capability < 8.0 (Tesla T4 = 7.5) has no native
    bf16 tensor cores: PyTorch runs it through emulation. MQL-0006/0007/0008 ran exactly that (bf16 on a T4)."""
    import torch
    need(torch.cuda.is_available(), "no CUDA GPU is visible (Settings -> Accelerator -> GPU T4 x2)")
    props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    cap = torch.cuda.get_device_capability(0)
    try:
        native = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:                                   # older torch without the keyword
        native = cap >= (8, 0)
    return {"checked": True, "visible_devices": len(props), "device0": props[0].name,
            "compute_capability": f"{cap[0]}.{cap[1]}", "total_gib": round(props[0].total_memory / 2**30, 2),
            "bf16_supported_incl_emulation": bool(torch.cuda.is_bf16_supported()), "bf16_native": native,
            "free_gib": round(torch.cuda.mem_get_info(0)[0] / 2**30, 2),
            "cuda": torch.version.cuda, "torch": torch.__version__}


def check_precision(cfg, gpu):
    """Registered: bfloat16 on a Tesla T4 (emulated bf16, as MQL-0006/0007/0008). Never substitutes fp16/fp32.
    -> precision descriptor stored in every run config and prediction record."""
    hw, m = cfg["hardware"], cfg["model"]
    need(m["dtype"] == "bfloat16", "registered dtype must be bfloat16")
    need(gpu.get("device0") in hw["allowed_gpus"], f"GPU {gpu.get('device0')} is not the registered hardware "
         f"{hw['allowed_gpus']}; refusing (greedy outputs depend on the GPU/kernels; all arms must share one GPU type)")
    need(gpu.get("total_gib", 0) >= hw["min_gpu_gib"], f"GPU memory {gpu.get('total_gib')} GiB < {hw['min_gpu_gib']}")
    need(gpu.get("bf16_supported_incl_emulation"), "bfloat16 is not supported on this GPU even with emulation; "
         "refusing to change precision")
    if not gpu.get("bf16_native"):
        need(hw["bf16_emulation_allowed"], "bf16 would be emulated on this GPU and the config does not allow it")
    return {"dtype": "bfloat16", "bf16_native": bool(gpu.get("bf16_native")),
            "bf16_mode": "native" if gpu.get("bf16_native") else "emulated (compute capability < 8.0)",
            "gpu": gpu.get("device0"), "compute_capability": gpu.get("compute_capability")}


def check_snapshot_revision(model_dir, revision):
    """Independent revision check of a huggingface_hub `snapshot_download(..., local_dir=...)` folder: every
    `.cache/huggingface/download/*.metadata` file records the commit it was downloaded from on its first line.
    -> {"resolved_revision", "files_checked"}; raises if any file is from another commit or none is found."""
    meta = sorted(Path(model_dir).glob(".cache/huggingface/download/**/*.metadata"))
    need(meta, f"{model_dir}: no huggingface download metadata; cannot verify the revision")
    commits = {}
    for p in meta:
        first = p.read_text(encoding="utf-8").splitlines()[0].strip()
        commits.setdefault(first, []).append(p.name)
    need(set(commits) == {revision}, f"model files come from commits {sorted(commits)} != pinned {revision}")
    return {"resolved_revision": revision, "files_checked": len(meta)}


# ------------------------------------------------------------------ per-record identity
def record_identity(cfg, run, plan_row, cfg_sha, precision, runtime):
    """Stored in EVERY prediction record. Deterministic within one arm-run fingerprint (no timestamps)."""
    m = cfg["model"]
    out = {"experiment": EXPERIMENT, "prediction_id": f"{EXPERIMENT}:{run['name']}:{plan_row['id']}",
           "arm_family": run["family"], "random_seed": run["seed"],
           "model_id": m["id"], "model_revision": m["revision"], "precision": precision,
           "generation_config": {"dtype": m["dtype"], "decoding": m["decoding"], "do_sample": False,
                                 "max_new_tokens": m["max_new_tokens"], "batch_size": m["batch_size"], "seed": m["seed"]},
           "runtime": runtime, "config_sha256": cfg_sha}
    if run["family"] == "zero_shot":
        out["retrieval"] = {"strategy": "none", "retrieved_ids": [], "context_tokens": 0, "empty_context": True,
                            "prompt_tokens_est": plan_row["prompt_tokens_est"]}
    else:
        out["retrieval"] = {
            "strategy": run["family"], "seed": run["seed"], "corpus_sha256": cfg["corpus"]["source_sha256"],
            "eligible_view_ids_sha256": cfg["corpus"]["eligible_view"]["ids_sha256"],
            "retrieval_config_fingerprint": retrieval_fingerprint(cfg, run),
            "retrieved_ids": plan_row["selected_ids"], "ranks": plan_row["selected_ranks"],
            "scores": plan_row["selected_scores"], "retrieved_db_ids": plan_row["selected_db_ids"],
            "target_db_excluded": all(d != plan_row["db_id"] for d in plan_row["selected_db_ids"]),
            "excluded_by_rule": plan_row["excluded_by_rule"], "dropped_over_budget": plan_row["dropped_budget"],
            "context_tokens": plan_row["context_tokens"], "max_context_tokens": cfg["context"]["max_context_tokens"],
            "prompt_tokens_est": plan_row["prompt_tokens_est"], "empty_context": plan_row["empty_context"]}
    return out


def retrieval_fingerprint(cfg, run):
    return fingerprint({"corpus_sha256": cfg["corpus"]["source_sha256"], "view": cfg["corpus"]["eligible_view"]["ids_sha256"],
                        "retrieval": cfg["retrieval"], "context": cfg["context"], "run": cfg["arm_runs"][run["name"]],
                        "fixed": cfg["fixed_retrieval"]["ids"] if run["family"] == "fixed" else None,
                        "retrieval_exclusions": cfg["retrieval_exclusions"]})


def finalize_records(recs):
    """Add parse status + normalised MQL to checkpoint records (pure; raw_text untouched)."""
    out = []
    for r in recs:
        r = dict(r)
        r.update(normalize_prediction(r.get("raw_text"), r.get("status")))
        r["generation_failure"] = r.get("status") != "ok"
        out.append(r)
    return out


# ------------------------------------------------------------------ prediction record schema / validation
REQUIRED_FIELDS = ("prediction_id", "experiment", "arm", "arm_family", "random_seed", "id", "db_id", "raw_text",
                   "normalized_mql", "parse_status", "status", "generation_failure", "retrieval", "prompt_sha256",
                   "model_id", "model_revision", "precision", "generation_config", "runtime", "config_sha256",
                   "fingerprint", "input_tokens", "output_tokens")


def validate_prediction_record(r, run_name=None):
    """-> list of problems (empty = valid). Also forbids any gold / label field."""
    probs = [f"missing {k}" for k in REQUIRED_FIELDS if k not in r]
    probs += [f"forbidden gold/label field {k}" for k in GOLD_KEYS if k in r]
    if run_name and r.get("arm") != run_name:
        probs.append(f"arm {r.get('arm')} != {run_name}")
    if r.get("experiment") != EXPERIMENT:
        probs.append("experiment")
    if (r.get("model_id"), r.get("model_revision")) != (MODEL_ID, MODEL_REVISION):
        probs.append("model identity")
    if r.get("prediction_id") != f"{EXPERIMENT}:{r.get('arm')}:{r.get('id')}":
        probs.append("prediction_id is not stable experiment:arm:id")
    if r.get("parse_status") not in ("parsed", "parsed_bad_shape", "unparseable", "generation_failure"):
        probs.append("parse_status")
    return probs


# ------------------------------------------------------------------ loading + preflight of the package
def load_package(pkg):
    """Verify every package file, the corpus identity, the eval inputs and the frozen config. -> dict P."""
    pkg = Path(pkg)
    man = M8.verify_package(pkg)
    cfg = json.loads((pkg / "configs/experiment.json").read_text(encoding="utf-8"))
    need(cfg["experiment_id"] == EXPERIMENT and man["package"] == PACKAGE, "not the MQL-0009 package")
    corpus_sha = sha256_file(pkg / CORPUS_FILE)
    need(corpus_sha == EXPECTED_CORPUS_SHA256 == cfg["corpus"]["source_sha256"], f"corpus sha256 {corpus_sha} is not the approved corpus v2.1")
    corpus = read_jsonl(pkg / CORPUS_FILE)
    need(len(corpus) == EXPECTED_CORPUS == len({r["id"] for r in corpus}), "corpus is not 3,445 unique records")
    need(all(eligible_ok(r) for r in corpus), "ineligible corpus row (verification / converter / id prefix)")
    evals = read_jsonl(pkg / "data/eval/eval_inputs.jsonl")
    need(all(set(e) == {"id", "db_id", "question"} for e in evals), "evaluation inputs must hold only id, db_id, question")
    ids = [e["id"] for e in evals]
    need(len(ids) == len(set(ids)) == cfg["evaluation"]["n_items"] and sha256_text("\n".join(ids)) == cfg["evaluation"]["ids_sha256"],
         "evaluation ids / count / hash != config")
    eval_dbs = sorted({e["db_id"] for e in evals})
    need(eval_dbs == cfg["evaluation"]["db_ids"], "evaluation databases != config")
    need(not set(eval_dbs) & set(cfg["evaluation"]["protected_db_ids"]), "a protected (Spider-dev) database is in the evaluation set")
    rows, view = eligible_view(corpus, eval_dbs)
    need(view["ids_sha256"] == cfg["corpus"]["eligible_view"]["ids_sha256"] and view["n"] == cfg["corpus"]["eligible_view"]["n"],
         "eligible corpus view differs from the frozen view")
    need(not {r["db_id"] for r in rows} & set(eval_dbs), "eligible view contains an evaluation database")
    prompts = read_jsonl(pkg / "data/eval/prompts.jsonl")
    need([p["id"] for p in prompts] == ids, "prompt ids/order != evaluation ids")
    need({p["template_sha256"] for p in prompts} == {cfg["prompt"]["zero_shot_template_sha256"]}, "zero-shot template hash")
    need(sha256_file(pkg / "data/eval/prompts.jsonl") == cfg["prompt"]["zero_shot_prompts_sha256"], "frozen prompts hash")
    runs = check_arm_design(cfg)
    retriever_constants_ok(cfg)
    lk = M8.leakage_report(rows, evals, cfg["leakage_gates"]["near_duplicate_token_jaccard"])
    exp_pairs = {(x["eval_id"], x["corpus_id"]) for x in cfg["retrieval_exclusions"]}
    got_pairs = {(e, c) for e, cs in lk["exact"] for c in cs} | {(e, c) for e, c, _ in lk["near"]}
    need(not lk["db_overlap"], "corpus view and evaluation share a database")
    need(got_pairs <= exp_pairs, f"leakage re-check found pairs not covered by the registered exclusions: {sorted(got_pairs - exp_pairs)[:5]}")
    return {"pkg": pkg, "manifest": man, "manifest_sha256": sha256_file(pkg / "package_manifest.json"), "cfg": cfg,
            "cfg_sha256": sha256_file(pkg / "configs/experiment.json"), "corpus": corpus, "view_rows": rows,
            "evals": evals, "ids": ids, "eval_dbs": eval_dbs, "prompts": prompts, "runs": runs,
            "atlas": json.loads((pkg / "data/context/atlas_collections.json").read_text(encoding="utf-8")),
            "tables": {t["db_id"]: t for t in json.loads((pkg / "data/context/tables_subset.json").read_text(encoding="utf-8"))}}


def derive_plans(P, count_tokens, compare_frozen=True):
    """Re-derive corpus features, every plan and every prompt; compare with the frozen package copies."""
    pkg, cfg = P["pkg"], P["cfg"]
    docs = M8.build_docs(P["view_rows"], P["tables"])
    docs_by_id = {d["doc_id"]: d for d in docs}
    if compare_frozen:
        frozen = read_jsonl(pkg / "data/corpus/corpus_features.jsonl")
        need([{k: d[k] for k in ("doc_id", "db_id", "question", "masked_question", "patterns", "content")} for d in docs] == frozen,
             "re-derived corpus features differ from the frozen corpus_features.jsonl")
    infos = M8.eval_schema_infos(P["atlas"], P["eval_dbs"])
    plans, prompts = {}, {}
    for run in P["runs"]:
        plan = build_plan(cfg, run, P["evals"], P["prompts"], docs, infos, count_tokens)
        plan_checks(cfg, run, plan, P["evals"], docs_by_id)
        if compare_frozen:
            frozen = read_jsonl(pkg / f"data/plans/plan_{run['name']}.jsonl")
            bad = [a["id"] for a, b in zip(plan, frozen) if a != b]
            need(len(plan) == len(frozen) and not bad, f"{run['name']}: re-derived plan differs from the frozen plan on "
                 f"{len(bad)} items, e.g. {bad[:3]} (tokenizer/library difference?)")
            need(plan_sha(plan) == cfg["plans_sha256"][run["name"]], f"{run['name']}: plan hash != config")
        plans[run["name"]] = plan
        prompts[run["name"]] = prompts_for_run(plan, P["prompts"], docs_by_id)
    zs = next(r["name"] for r in P["runs"] if r["family"] == "zero_shot")
    need([p["prompt_sha256"] for p in prompts[zs]] == [p["prompt_sha256"] for p in P["prompts"]],
         "zero-shot prompts are not byte-identical to the frozen prompts")
    check_prompt_equivalence(prompts, P["prompts"])
    return docs, docs_by_id, plans, prompts


def check_prompt_equivalence(prompts_by_run, frozen):
    """Across all runs: identical system prompt, and every user prompt with its reference block removed equals the
    frozen zero-shot user prompt (same schema, question and base instructions in every arm)."""
    base = {p["id"]: p for p in frozen}
    for run, ps in prompts_by_run.items():
        for p in ps:
            b = base[p["id"]]
            need(p["system"] == b["system"], f"{run} {p['id']}: system prompt differs from zero-shot")
            u, bu = p["messages"][0]["content"], b["messages"][0]["content"]
            if u != bu:
                head, mark, tail = bu.rpartition(RB.SPLIT_MARK)
                need(u.startswith(head + "\n\n" + RB.HEADER.format(db=p["db_id"])) and u.endswith(mark + tail),
                     f"{run} {p['id']}: prompt differs from zero-shot other than by one inserted reference block")
    return True


# ------------------------------------------------------------------ run config + generation + export
def run_config(P, run, plan, prompts, precision, gen_info):
    cfg, m = P["cfg"], P["cfg"]["model"]
    q = {"experiment": EXPERIMENT, "arm": run["name"], "arm_family": run["family"], "random_seed": run["seed"],
         "arm_config": cfg["arm_runs"][run["name"]], "model_id": m["id"], "model_revision": m["revision"],
         "model_label": m["label"], "runtime": m["runtime"], "decoding": m["decoding"],
         "max_new_tokens": m["max_new_tokens"], "batch_size": m["batch_size"], "seed": m["seed"], "precision": precision,
         "prompt_version": cfg["prompt"]["zero_shot_version"] if run["family"] == "zero_shot" else cfg["prompt"]["rag_version"],
         "arm_prompts_sha256": sha256_text("".join(p["prompt_sha256"] + "\n" for p in prompts)),
         "plan_sha256": plan_sha(plan), "corpus_sha256": cfg["corpus"]["source_sha256"],
         "eligible_view_ids_sha256": cfg["corpus"]["eligible_view"]["ids_sha256"],
         "retrieval_config_fingerprint": None if run["family"] == "zero_shot" else retrieval_fingerprint(cfg, run),
         "evaluation_ids_sha256": cfg["evaluation"]["ids_sha256"], "config_sha256": P["cfg_sha256"],
         "package_manifest_sha256": P["manifest_sha256"],
         **{k: gen_info.get(k) for k in ("dtype", "gpu", "torch", "transformers")}}
    q["fingerprint"] = fingerprint({k: v for k, v in q.items() if k not in POST_FP})
    return q


def runtime_brief(gen_info, gpu):
    return {"gpu": gen_info.get("gpu"), "compute_capability": gpu.get("compute_capability"), "bf16_native": gpu.get("bf16_native"),
            "torch": gen_info.get("torch"), "transformers": gen_info.get("transformers"), "cuda": gpu.get("cuda")}


def arm_dir(out_dir, name):
    return Path(out_dir) / "predictions" / name


def generate_run(P, run, plan, prompts, generate_batch, gen_info, precision, gpu, out_dir, env, stop_after_batches=None,
                 log=print):
    """Generate every not-yet-final question of one arm run (resumable, append-only, fingerprinted checkpoints)."""
    d, t0 = arm_dir(out_dir, run["name"]), time.time()
    qcfg = run_config(P, run, plan, prompts, precision, gen_info)
    old = d / "run_config.json"
    sessions = []
    if old.exists():
        prev = json.loads(old.read_text(encoding="utf-8"))
        diff = {k: (prev.get(k), v) for k, v in qcfg.items() if k not in POST_FP and prev.get(k) != v}
        need(not diff, f"{run['name']}: earlier run_config differs (earlier, now): {diff}; refusing to mix runs")
        sessions = prev.get("sessions") or []
    old_plan = d / "retrieval_log.jsonl"
    if old_plan.exists():
        need(read_jsonl(old_plan) == plan, f"{run['name']}: existing retrieval log differs from this plan")
    write_jsonl(d / "retrieval_log.jsonl", plan)
    ids = [p["id"] for p in plan]
    ck = M8.Checkpoint(d / "checkpoints/predictions.partial.jsonl", ids, qcfg["fingerprint"], M8.is_generation_retry)
    rt = runtime_brief(gen_info, gpu)
    identity = {p["id"]: record_identity(P["cfg"], run, p, P["cfg_sha256"], precision, rt) for p in plan}
    todo = len(ck.todo())
    log(f"{run['name']}: {todo} of {len(ids)} to generate")
    r = {"generated": 0}
    try:
        r = M8.run_generation(prompts, generate_batch, ck, P["cfg"]["model"]["label"], run["name"], P["cfg"]["model"]["batch_size"],
                              max_batches=stop_after_batches, identity=identity,
                              progress=lambda n, tot: log(f"  {run['name']} {n}/{tot}") if n % 80 < 8 else None)
    finally:
        safe_write_predictions(d / "predictions.jsonl", finalize_records(ck.ordered()))
        qcfg.update(generation_wall_s=round(time.time() - t0, 1), environment=env,
                    sessions=sessions + [{"at": now(), "todo_at_start": todo, "generated": r.get("generated")}])
        write_json(d / "run_config.json", qcfg)
    return qcfg, r


def safe_write_predictions(path, recs):
    """Atomic write that never silently changes a completed prediction: every record already in the file must be
    present, unchanged, in the new content (a generation_failure record may be replaced by its retry)."""
    path = Path(path)
    if path.exists():
        new = {r["id"]: r for r in recs}
        for old in read_jsonl(path):
            if old.get("status") == "ok":
                need(new.get(old["id"]) == old, f"{path}: completed prediction {old['id']} would be overwritten or dropped")
    write_jsonl(path, recs)


def verify_resume_source(prev, pkg_manifest_sha):
    sums = {}
    for ln in (prev / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, _, n = ln.partition("  ")
        sums[n] = h
    files = {p.relative_to(prev).as_posix() for p in prev.rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
    need(files == set(sums) and all(sha256_file(prev / n) == h for n, h in sums.items()), f"resume source {prev}: hash check failed")
    st = json.loads((prev / STATUS_FILE).read_text(encoding="utf-8"))
    need(st.get("experiment") == EXPERIMENT and st.get("package_manifest_sha256") == pkg_manifest_sha,
         "resume source: another experiment or another package")
    return st


def seed_from_previous(prev, out_dir, run_names):
    """Copy every arm run's files of an earlier, hash-verified output into OUT_DIR (never overwrites)."""
    seeded = []
    for name in run_names:
        for rel in ARM_FILES:
            src = Path(prev) / "predictions" / name / rel
            dst = arm_dir(out_dir, name) / rel
            if src.is_file():
                need(not dst.exists() or sha256_file(dst) == sha256_file(src), f"{dst} exists with different content")
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    seeded.append(f"{name}/{rel}")
    return seeded


def check_run_output(P, run, plan, prompts, out_dir):
    """Completeness + identity of one arm run on disk. -> summary dict (status complete|partial|absent)."""
    d = arm_dir(out_dir, run["name"])
    pf = d / "predictions.jsonl"
    if not pf.exists():
        return {"status": "absent", "problems": ["no predictions"], "n": 0, "n_expected": len(plan)}
    preds = read_jsonl(pf)
    exp_ids = [p["id"] for p in plan]
    ids = [p["id"] for p in preds]
    pr = []
    if len(set(ids)) != len(ids):
        pr.append("duplicate ids")
    if ids != [i for i in exp_ids if i in set(ids)] or set(ids) - set(exp_ids):
        pr.append("ids out of order or unknown")
    if len(ids) != len(exp_ids):
        pr.append(f"{len(exp_ids) - len(ids)} ids missing")
    n_ok = sum(p["status"] == "ok" for p in preds)
    if n_ok != len(ids):
        pr.append(f"{len(ids) - n_ok} generation failures (retried on resume)")
    want = {p["id"]: p["prompt_sha256"] for p in prompts}
    if any(p.get("prompt_sha256") != want[p["id"]] for p in preds):
        pr.append("a prediction was generated from a different prompt")
    plan_by = {p["id"]: p for p in plan}
    for p in preds:
        v = validate_prediction_record(p, run["name"])
        if v:
            pr.append(f"{p['id']}: {v[:3]}")
            break
        if p["retrieval"]["retrieved_ids"] != plan_by[p["id"]]["selected_ids"]:
            pr.append(f"{p['id']}: retrieval metadata differs from the plan")
            break
    rc = json.loads((d / "run_config.json").read_text(encoding="utf-8")) if (d / "run_config.json").exists() else {}
    if {p.get("fingerprint") for p in preds} - {rc.get("fingerprint")}:
        pr.append("prediction fingerprint != run_config fingerprint")
    return {"status": "complete" if not pr else "partial", "problems": pr, "n": len(ids), "n_expected": len(exp_ids),
            "n_ok": n_ok, "predictions_sha256": sha256_file(pf), "fingerprint": rc.get("fingerprint"),
            "environment_key": env_key(rc), "precision": rc.get("precision"),
            "oom_batch1_fallback_items": sum(bool(p.get("oom_recovery")) for p in preds),
            "finish_length_items": sum(p.get("finish_reason") == "length" for p in preds),
            "parse_status": {s: sum(p.get("parse_status") == s for p in preds) for s in
                             ("parsed", "parsed_bad_shape", "unparseable", "generation_failure")},
            "tokens": {"input": sum(p.get("input_tokens") or 0 for p in preds),
                       "output": sum(p.get("output_tokens") or 0 for p in preds),
                       "context_planned": sum(plan_by[p["id"]]["context_tokens"] for p in preds)}}


def env_key(rc):
    """What must be identical across all arm runs of the experiment (GPU type, dtype, library versions)."""
    pr = rc.get("precision") or {}
    return {"gpu": rc.get("gpu"), "dtype": rc.get("dtype"), "torch": rc.get("torch"), "transformers": rc.get("transformers"),
            "bf16_native": pr.get("bf16_native"), "compute_capability": pr.get("compute_capability")}


def export(P, plans, prompts, out_dir, output_zip, extra_status=None):
    """Status + predictions manifest (expected vs completed, checksums) + SHA256SUMS + zip of OUT_DIR."""
    out_dir = Path(out_dir)
    runs = {}
    for run in P["runs"]:
        runs[run["name"]] = check_run_output(P, run, plans[run["name"]], prompts[run["name"]], out_dir)
    present = {n: s for n, s in runs.items() if s["status"] != "absent"}
    keys = {canon(s["environment_key"]) for s in present.values()}
    env_consistent = len(keys) <= 1
    problems = [f"{n}: {p}" for n, s in present.items() for p in s["problems"]]
    if not env_consistent:
        problems.append(f"arm runs were generated in different environments: {sorted(keys)}")
    all_done = all(s["status"] == "complete" for s in runs.values())
    status = "complete" if all_done and env_consistent else "partial"
    n_exp = len(P["ids"]) * len(P["runs"])
    man = {"experiment": EXPERIMENT, "written_at": now(), "status": status,
           "expected_predictions": n_exp, "completed_predictions": sum(s.get("n_ok", 0) for s in runs.values()),
           "per_arm": {n: {"expected": s["n_expected"], "completed_ok": s.get("n_ok", 0), "status": s["status"],
                           "predictions_sha256": s.get("predictions_sha256")} for n, s in runs.items()},
           "package_manifest_sha256": P["manifest_sha256"], "config_sha256": P["cfg_sha256"],
           "evaluation_ids_sha256": P["cfg"]["evaluation"]["ids_sha256"]}
    write_json(out_dir / MANIFEST_FILE, man)
    st = {"experiment": EXPERIMENT, "stage": "predictions", "status": status, "written_at": now(), "problems": problems,
          "arm_runs": runs, "environment_consistent": env_consistent,
          "predictions_expected_all_runs": n_exp, "predictions_present": sum(s["n"] for s in runs.values()),
          "package": PACKAGE, "package_manifest_sha256": P["manifest_sha256"], "config_sha256": P["cfg_sha256"],
          "model": {"id": P["cfg"]["model"]["id"], "revision": P["cfg"]["model"]["revision"]},
          "database": "not contacted (scoring runs on Windows against Atlas)", "external_model_apis": "none",
          **(extra_status or {})}
    write_json(out_dir / STATUS_FILE, st)
    names = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"
                   and not p.name.endswith(".tmp"))
    (out_dir / "SHA256SUMS.txt").write_text("".join(f"{sha256_file(out_dir / n)}  {n}\n" for n in names), encoding="utf-8",
                                            newline="\n")
    tmp = Path(str(output_zip) + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names + ["SHA256SUMS.txt"]:
            z.write(out_dir / n, f"{out_dir.name}/{n}")
    os.replace(tmp, output_zip)
    return st, man
