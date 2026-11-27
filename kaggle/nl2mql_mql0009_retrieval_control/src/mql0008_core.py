"""MQL-0008 (zero-shot vs hybrid RAG over approved corpus v2.1, Qwen2.5-Coder-3B-Instruct, 892 NL->MQL items): package core.

Derived from the MQL-0007 package core (kaggle/nl2mql_mql0007_rag_v1/src/mql0007_core.py). Differences, all listed in
configs/experiment.json `stated_differences_from_MQL-0007`: two arms only (no random arm, no supplementary arm), the
approved corpus v2.1 record layout (no `schema.touched`), `selected_ranks` in plan rows, and per-record identity
metadata in prediction records. Retrieval, budget loop, prompt construction, checkpointing and generation are unchanged.

Everything the Kaggle notebook, the package builder and the tests share. No torch / transformers import at module
level (the GPU generator is built lazily by make_qwen_generator). No MongoDB, no credentials, no network.

Retrieval is the MQL-0005 `hybrid_masked` retriever (src/retrieval.py, src/patterns.py: unmodified copies of
scripts/rag/*) over the approved corpus v2.1 training examples, through the thin adapter below.
"""
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import patterns as PT          # noqa: E402  (unmodified copy of scripts/rag/patterns.py)
import retrieval as RT         # noqa: E402  (unmodified copy of scripts/rag/retrieval.py)
import rag_block as RB         # noqa: E402

EXPERIMENT = "MQL-0008"
PACKAGE = "nl2mql_mql0008_approved_corpus_v2_1"
ARMS = ("arm_a_zero_shot", "arm_b_hybrid_rag")       # exactly two arms
RAG_ARM = "arm_b_hybrid_rag"
CORPUS_FILE = "data/corpus/approved_corpus_v2_1.jsonl"
EXPECTED_CORPUS = 3445
EXPECTED_CORPUS_SHA256 = "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf"
EXPECTED_EVAL = 892


# ------------------------------------------------------------------ small utils
class PreflightError(RuntimeError):
    pass


def need(cond, msg):
    if not cond:
        raise PreflightError(msg)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(s):
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def compact(obj):
    """Compact JSON, key order kept (how MQL is shown in prompts; as scripts/rag/common.compact)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def fingerprint(obj):
    return sha256_text(canon(obj))[:16]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)
    return path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------ package manifest
def verify_package(pkg):
    """Every file listed in package_manifest.json exists with the recorded sha256, and no other file is present."""
    pkg = Path(pkg)
    man = json.loads((pkg / "package_manifest.json").read_text(encoding="utf-8"))
    on_disk = {p.relative_to(pkg).as_posix() for p in pkg.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
               and ".pytest_cache" not in p.parts}
    on_disk -= {"package_manifest.json", "SHA256SUMS.txt"}
    # run outputs and local scratch are not package files
    on_disk = {n for n in on_disk if not n.startswith(("outputs/", "work/"))}
    listed = set(man["files"])
    need(on_disk == listed, f"package tree != manifest: extra {sorted(on_disk - listed)[:5]}, "
                            f"missing {sorted(listed - on_disk)[:5]}")
    bad = [n for n, m in man["files"].items() if sha256_file(pkg / n) != m["sha256"]]
    need(not bad, f"hash mismatch: {bad[:5]}")
    return man


# ------------------------------------------------------------------ question normalisation (leakage gates)
def norm_question(q):
    return " ".join(re.findall(r"[a-z0-9]+", (q or "").lower()))


def token_set(q):
    return set(re.findall(r"[a-z0-9]+", (q or "").lower()))


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def norm_sql(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower().rstrip(";").strip()


# ------------------------------------------------------------------ corpus adapter
def schema_manifest_from_tables(tables_entry):
    """A scripts/rag-style collections manifest for one Spider database, from official tables.json.
    Only names matter for masking (PT.schema_info uses table and column NAME words)."""
    db = tables_entry["db_id"]
    tabs = tables_entry["table_names_original"]
    fields = {t: {} for t in tabs}
    for ti, cname in tables_entry["column_names_original"]:
        if ti >= 0:
            fields[tabs[ti]][cname] = ["x"]
    # SQLite's internal sqlite_sequence (listed by Spider's tables.json for world_1) is not a Spider table and is
    # not imported into Atlas, so it is ignored here; this makes the words equal the Atlas manifest's on all 17 eval DBs
    return {f"{db}__{t}": {"db_id": db, "table": t, "fields": f, "declared": {}}
            for t, f in fields.items() if t != "sqlite_sequence"}


def text_number_dbs(corpus_rows):
    """DB-level 'text_number' flag from the columns the corpus rows touched: observed text, numeric declared type.
    Approved corpus v2.1 records carry no `schema.touched` (the type contract TC-1 replaced it), so the flag is empty
    for them. It was also empty for every database of corpus v1 in MQL-0007 (0 of 123; checked by the builder), so the
    retrieval features are computed exactly as in MQL-0007. Recorded in configs/experiment.json."""
    out = set()
    for r in corpus_rows:
        for t in (r.get("schema") or {}).get("touched", []):
            for c in t["columns"]:
                if c["observed_kind"] == "text" and PT.NUMERIC_SQL.search(c.get("declared_type") or ""):
                    out.add(r["db_id"])
    return out


def build_docs(corpus_rows, tables_by_db):
    """Corpus rows -> retrieval documents (the scripts/rag document schema). Pure function of its inputs."""
    tn = text_number_dbs(corpus_rows)
    infos, docs = {}, []
    for r in corpus_rows:
        db = r["db_id"]
        if db not in infos:
            infos[db] = PT.schema_info(db, schema_manifest_from_tables(tables_by_db[db]))
        info = infos[db]
        q = r["question"]
        docs.append({
            "doc_id": r["id"], "kind": "example", "db_id": db, "split": "train",
            "question": q, "masked_question": PT.mask_question(q, info), "mql": r["mql"],
            "content": f"Question: {q}\nMQL: {compact(r['mql'])}",
            "patterns": sorted(PT.sql_patterns(r["sql"]) | ({"text_number"} if db in tn else set())),
        })
    return docs


def eval_schema_infos(atlas_collections, db_ids):
    return {db: PT.schema_info(db, atlas_collections) for db in db_ids}


# ------------------------------------------------------------------ plan (selection + budget), per arm
def arm_settings(cfg, arm):
    a = cfg["arms"][arm]
    return a


def make_retriever(cfg, arm, docs):
    a = cfg["arms"][arm]
    if a["retrieval"] == "none":
        return None
    r = cfg["retrieval"]
    need(a["retrieval"] == "hybrid_masked", f"unknown retrieval {a['retrieval']}")
    return RT.Retriever(docs, method="hybrid_masked", hybrid_weight=r["hybrid_weight"])


def retriever_constants_ok(cfg):
    """BM25 k1/b are hard-coded defaults of the unmodified retrieval.py; assert they are the configured ones."""
    import inspect
    sig = inspect.signature(RT.BM25.__init__).parameters
    need(sig["k1"].default == cfg["retrieval"]["bm25_k1"] and sig["b"].default == cfg["retrieval"]["bm25_b"],
         "retrieval.py BM25 defaults differ from configs/experiment.json")
    return True


def select_examples(db_id, system, base_user, ranked, k, count_tokens, max_ctx, n_ctx, max_new, margin):
    """The budget loop of scripts/rag/rag_prompt.build_messages for k_guides=0, notes=(): over-budget documents
    are skipped and recorded, never truncated. -> (chosen docs, dropped doc_ids, block, block_tokens, total)."""
    base_tokens = count_tokens(system) + count_tokens(base_user)
    chosen, dropped = [], []

    def fits(ex):
        block = RB.render_block(db_id, [], ex, [])
        bt = count_tokens(block) if block else 0
        total = base_tokens + (bt + 2 if block else 0)
        return bt <= max_ctx and total + max_new + margin <= n_ctx, bt, total

    for d, _ in ranked:
        if len(chosen) >= k:
            break
        ok, _, _ = fits(chosen + [d])
        (chosen.append(d) if ok else dropped.append(d["doc_id"]))
    _, bt, total = fits(chosen)
    return chosen, dropped, RB.render_block(db_id, [], chosen, []), bt, total


def prompt_sha(system, messages):
    return sha256_text(canon({"system": system, "messages": messages}))


def build_plan(cfg, arm, eval_rows, frozen_prompts, docs, eval_infos, count_tokens, exclusions=None, arm_label=None):
    """Per-question retrieval plan for one arm. Deterministic: same inputs -> byte-identical rows.

    `exclusions` = evaluation-specific retrieval exclusions [{"eval_id", "corpus_id"}] (default: the config's
    `retrieval_exclusions`). They are applied AFTER ranking (the BM25 index, its IDF and the dedup are untouched, so
    no other query's plan can change); the next-ranked eligible example takes the place, and each plan row records
    `excluded_by_rule` with the excluded document's original rank. `arm_label` renames the arm (supplementary arm)."""
    a = cfg["arms"][arm]
    ctx = cfg["context"]
    exclusions = cfg.get("retrieval_exclusions", []) if exclusions is None else exclusions
    retr = make_retriever(cfg, arm, docs)
    by_prompt = {p["id"]: p for p in frozen_prompts}
    rows = []
    for e in eval_rows:
        fp = by_prompt[e["id"]]
        system, base = fp["system"], fp["messages"][0]["content"]
        excl_ids = {x["corpus_id"] for x in exclusions if x["eval_id"] == e["id"]}
        excluded = []
        rank_of = {}
        if retr is None:
            ranked, chosen, dropped, block, bt, total = [], [], [], "", 0, None
        else:
            ranked_full = retr.rank_examples(e["question"], eval_infos[e["db_id"]], qid=e["id"])
            rank_of = {d["doc_id"]: i + 1 for i, (d, _) in enumerate(ranked_full)}
            excluded = [{"corpus_id": d["doc_id"], "original_rank": i + 1}
                        for i, (d, _) in enumerate(ranked_full) if d["doc_id"] in excl_ids]
            ranked = [(d, s) for d, s in ranked_full if d["doc_id"] not in excl_ids]
            chosen, dropped, block, bt, total = select_examples(
                e["db_id"], system, base, ranked, a["k"], count_tokens, ctx["max_context_tokens"], ctx["n_ctx"],
                ctx["max_new_tokens"], ctx["margin_tokens"])
        user = RB.insert_block(base, block)
        score = {d["doc_id"]: s for d, s in ranked[:200]}
        rows.append({
            "id": e["id"], "db_id": e["db_id"], "arm": arm_label or arm,
            "selected_ids": [d["doc_id"] for d in chosen],
            "selected_scores": [score.get(d["doc_id"]) for d in chosen],
            "selected_ranks": [rank_of[d["doc_id"]] for d in chosen],      # 1-based rank in the full ranking
            "selected_db_ids": [d["db_id"] for d in chosen],
            "ranked_top10": [[d["doc_id"], s] for d, s in ranked[:10]],
            "excluded_by_rule": excluded,
            "dropped_budget": dropped, "context_tokens": bt, "prompt_tokens_est": total,
            "empty_context": not block,
            "user_prompt_sha256": sha256_text(user),
            "prompt_sha256": prompt_sha(system, [{"role": "user", "content": user}]),
        })
    return rows


def plan_diff_ids(plan_a, plan_b):
    """Ids whose selected examples differ between two plans over the same ids."""
    return [a["id"] for a, b in zip(plan_a, plan_b) if a["selected_ids"] != b["selected_ids"] or a["id"] != b["id"]]


def selection_neardup_violations(plan, eval_rows, docs_by_id, threshold):
    """Selection-level gate: (eval id, selected corpus id, jaccard) for every SELECTED example whose question has token
    Jaccard >= threshold with the query. Must be empty for the primary plans."""
    q = {e["id"]: token_set(e["question"]) for e in eval_rows}
    out = []
    for p in plan:
        for i in p["selected_ids"]:
            d = docs_by_id[i]
            j = jaccard(q[p["id"]], token_set(d["question"]))
            if j >= threshold:
                out.append((p["id"], i, round(j, 4)))
    return out


def prompts_for_arm(plan_rows, frozen_prompts, docs_by_id):
    """Rebuild the exact prompts of an arm from its plan (+ corpus). RAG prompt == insert_block(frozen base, block)."""
    by_prompt = {p["id"]: p for p in frozen_prompts}
    out = []
    for pr in plan_rows:
        fp = by_prompt[pr["id"]]
        chosen = [docs_by_id[i] for i in pr["selected_ids"]]
        block = RB.render_block(pr["db_id"], [], chosen, [])
        user = RB.insert_block(fp["messages"][0]["content"], block)
        msgs = [{"role": "user", "content": user}]
        rec = {"id": pr["id"], "db_id": pr["db_id"], "prompt_version": fp["prompt_version"],
               "template_sha256": fp["template_sha256"], "system": fp["system"], "messages": msgs,
               "prompt_sha256": prompt_sha(fp["system"], msgs), "arm": pr["arm"]}
        need(rec["prompt_sha256"] == pr["prompt_sha256"], f"{pr['id']}: rebuilt prompt != planned prompt")
        if not block:
            need(rec["prompt_sha256"] == fp["prompt_sha256"], f"{pr['id']}: empty-context prompt != frozen zero-shot")
        out.append(rec)
    return out


# ------------------------------------------------------------------ leakage / eligibility gates
def leakage_report(corpus_rows, eval_rows, near_dup_jaccard):
    """Exact (normalised) and near-duplicate (token Jaccard >= threshold, fixed in the config BEFORE it was run)
    matches between every evaluation question and every corpus question. Also DB overlap."""
    corp_norm = {}
    for r in corpus_rows:
        for q in (r["question"], r.get("question_raw") or r["question"]):
            corp_norm.setdefault(norm_question(q), []).append(r["id"])
    corp_sets = [(r["id"], token_set(r["question"]) | token_set(r.get("question_raw") or "")) for r in corpus_rows]
    exact, near, best = [], [], 0.0
    for e in eval_rows:
        n = norm_question(e["question"])
        if n in corp_norm:
            exact.append((e["id"], corp_norm[n][:3]))
        ts = token_set(e["question"])
        for cid, cs in corp_sets:
            j = jaccard(ts, cs)
            best = max(best, j)
            if j >= near_dup_jaccard and norm_question(e["question"]) not in corp_norm:
                near.append((e["id"], cid, round(j, 4)))
    return {"exact": exact, "near": near, "max_jaccard": round(best, 4),
            "db_overlap": sorted({r["db_id"] for r in corpus_rows} & {e["db_id"] for e in eval_rows})}


# ------------------------------------------------------------------ checkpoints and generation (copied logic of
# scripts/nl2mql/dev1034_core.py Checkpoint / run_generation, so resume behaves exactly as in MQL-0006)
class Checkpoint:
    """Append-only JSONL keyed by id, fsynced per append. A record is final unless `retry(rec)` says so.
    Refuses records written under another fingerprint; tolerates a torn last line after a crash."""

    def __init__(self, path, expected_ids, fp, retry=lambda r: False):
        self.path, self.expected, self.fp, self.retry = Path(path), list(expected_ids), fp, retry
        self.records, self.warnings = {}, []
        exp = set(self.expected)
        if self.path.exists():
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for n, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    if n == len(lines) - 1:
                        self.warnings.append(f"ignored torn last line {n + 1}")
                        continue
                    raise RuntimeError(f"{self.path}: corrupt line {n + 1}")
                if r.get("fingerprint") != fp:
                    raise RuntimeError(f"{self.path}: record {r.get('id')} has a different run fingerprint; "
                                       "refusing to mix runs (start a fresh OUT_DIR)")
                if r.get("id") not in exp:
                    raise RuntimeError(f"{self.path}: unexpected id {r.get('id')}")
                self.records[r["id"]] = r
            if self.warnings:
                write_jsonl(self.path, list(self.records.values()))

    def done(self, iid):
        r = self.records.get(iid)
        return r is not None and not self.retry(r)

    def todo(self):
        return [i for i in self.expected if not self.done(i)]

    def append(self, recs):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            for r in recs:
                if r.get("fingerprint") != self.fp:
                    raise RuntimeError("record fingerprint differs from the checkpoint's")
                if self.done(r["id"]):
                    raise RuntimeError(f"duplicate final record for {r['id']}")
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                self.records[r["id"]] = r
            f.flush()
            os.fsync(f.fileno())

    def ordered(self):
        return [self.records[i] for i in self.expected if i in self.records]


def is_generation_retry(rec):
    return rec.get("status") != "ok"


def pred_record(item_id, db_id, fp, model_label, arm, text, meta):
    return {"id": item_id, "db_id": db_id, "fingerprint": fp, "model": model_label, "arm": arm, "status": "ok",
            "raw_text": text, "error": None, "created_at": now(), **meta}


def error_record(item_id, db_id, fp, model_label, arm, err, meta):
    return {"id": item_id, "db_id": db_id, "fingerprint": fp, "model": model_label, "arm": arm, "status": "error",
            "raw_text": None, "error": str(err)[:500], "created_at": now(), **meta}


def record_identity(cfg, arm, plan_row, retrieval_fp):
    """Identity metadata stored in EVERY prediction record (MQL-0008): model, revision, arm, generation settings and,
    for the RAG arm, what was retrieved. Deterministic (no timestamps), so a resumed run writes identical values."""
    m = cfg["model"]
    out = {"experiment": EXPERIMENT, "model_id": m["id"], "model_revision": m["revision"],
           "generation_config": {"dtype": m["dtype"], "decoding": m["decoding"], "max_new_tokens": m["max_new_tokens"],
                                 "batch_size": m["batch_size"], "seed": m["seed"]}}
    if arm == RAG_ARM:
        out["retrieval"] = {
            "corpus_sha256": cfg["corpus"]["source_sha256"], "retrieval_config_fingerprint": retrieval_fp,
            "retrieved_ids": plan_row["selected_ids"], "ranks": plan_row["selected_ranks"],
            "scores": plan_row["selected_scores"], "retrieved_db_ids": plan_row["selected_db_ids"],
            "target_db_excluded": all(d != plan_row["db_id"] for d in plan_row["selected_db_ids"]),
            "excluded_by_rule": plan_row["excluded_by_rule"], "dropped_over_budget": plan_row["dropped_budget"],
            "context_tokens": plan_row["context_tokens"], "max_context_tokens": cfg["context"]["max_context_tokens"],
            "prompt_tokens_est": plan_row["prompt_tokens_est"], "empty_context": plan_row["empty_context"]}
    else:
        out["retrieval"] = None
    return out


def retrieval_fingerprint(cfg):
    """Fingerprint of everything that defines retrieval: corpus hash, retriever constants, context budget, exclusions."""
    return fingerprint({"corpus_sha256": cfg["corpus"]["source_sha256"], "retrieval": cfg["retrieval"],
                        "context": cfg["context"], "arm": cfg["arms"][RAG_ARM],
                        "retrieval_exclusions": cfg["retrieval_exclusions"]})


def run_generation(prompts, generate_batch, ckpt, model_label, arm, batch_size=1, progress=None, max_batches=None,
                   identity=None):
    """Generate for every prompt not yet final in `ckpt` (in prompt order). generate_batch(list of prompt records)
    -> list of (text | None, meta, error | None). `max_batches` exists for kill-and-resume tests only.
    `identity` = {id: dict} merged into each record (see record_identity)."""
    todo = set(ckpt.todo())
    queue = [p for p in prompts if p["id"] in todo]
    n = 0
    for bi, i in enumerate(range(0, len(queue), batch_size)):
        if max_batches is not None and bi >= max_batches:
            break
        batch = queue[i:i + batch_size]
        outs = generate_batch(batch)
        if len(outs) != len(batch):
            raise RuntimeError("generate_batch returned a different number of outputs")
        recs = []
        for p, (text, meta, err) in zip(batch, outs):
            meta = dict(meta or {})
            meta["prompt_sha256"] = p["prompt_sha256"]
            meta.update((identity or {}).get(p["id"], {}))
            recs.append(pred_record(p["id"], p["db_id"], ckpt.fp, model_label, arm, text, meta) if err is None
                        else error_record(p["id"], p["db_id"], ckpt.fp, model_label, arm, err, meta))
        ckpt.append(recs)
        n += len(recs)
        if progress:
            progress(n, len(queue))
    return {"generated": n}


def oom_safe_generator(run_batch, is_oom, cleanup, configured_batch_size, counters):
    """A batch that raises CUDA OOM is retried one prompt at a time (same prompt, max_new_tokens, dtype, greedy
    decoding); other exceptions propagate. Identical behaviour to MQL-0006."""
    def meta(n, recovered):
        return {"configured_batch_size": configured_batch_size, "effective_batch_size": n,
                "oom_recovery": recovered, "oom_retries": 1 if recovered else 0}

    def generate_batch(batch):
        oom = None
        try:
            outs = run_batch(batch)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e):
                raise
            oom = f"{type(e).__name__}: {e}"
        if oom is None:
            return [(t, {**(m or {}), **meta(len(batch), False)}, err) for t, m, err in outs]
        counters["batch"] += 1
        cleanup()
        res = []
        for p in batch:
            single = None
            try:
                (t, m, err), = run_batch([p])
            except Exception as e:  # noqa: BLE001
                if not is_oom(e):
                    raise
                single = f"{type(e).__name__}: {e}"
            if single is None:
                res.append((t, {**(m or {}), **meta(1, True)}, err))
            else:
                counters["single_prompt"] += 1
                cleanup()
                res.append((None, {"latency_s": None, **meta(1, True)},
                            f"{single} [at batch size 1, after a batch OOM]"))
        return res
    return generate_batch


def make_qwen_generator(model_id, revision, max_new_tokens, required_dtype="bfloat16", local_dir=None):
    """The MQL-0006 generator: transformers, one GPU, greedy, left padding, bf16. Fails (never falls back) when the
    required dtype is unsupported."""
    import gc
    import torch
    from importlib import metadata
    from transformers import AutoModelForCausalLM, AutoTokenizer
    need(torch.cuda.is_available(), "no CUDA GPU is visible (Settings -> Accelerator -> GPU T4 x2)")
    if required_dtype == "bfloat16":
        need(torch.cuda.is_bf16_supported(), "bfloat16 is required (MQL-0006 ran bfloat16) but this GPU does not "
                                             "support it; refusing to change precision silently")
    dtype = getattr(torch, required_dtype)
    torch.manual_seed(42)
    src, rev = (local_dir, None) if local_dir else (model_id, revision)
    tok = AutoTokenizer.from_pretrained(src, revision=rev)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(src, revision=rev, torch_dtype=dtype).to("cuda").eval()
    resolved = getattr(model.config, "_commit_hash", None)
    need(local_dir or resolved == revision, f"resolved model revision {resolved} != required {revision}")
    info = {"dtype": str(dtype).replace("torch.", ""), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": metadata.version("transformers"), "resolved_revision": resolved}
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id, tok.pad_token_id}
    stop_ids.discard(None)

    def run_batch(batch):
        texts = [tok.apply_chat_template([{"role": "system", "content": p["system"]}] + p["messages"],
                                         tokenize=False, add_generation_prompt=True) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                     pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001
            return [(None, {"latency_s": None}, f"{type(e).__name__}: {e}") for _ in batch]
        dt = time.time() - t0
        res = []
        for i in range(len(batch)):
            gen = out[i, enc["input_ids"].shape[1]:].tolist()
            hit = next((j for j, t in enumerate(gen) if t in stop_ids), None)
            n_out = hit + 1 if hit is not None else len(gen)
            res.append((tok.decode(gen[:n_out], skip_special_tokens=True),
                        {"latency_s": round(dt / len(batch), 4), "batch_latency_s": round(dt, 3),
                         "input_tokens": int(enc["attention_mask"][i].sum()), "output_tokens": n_out,
                         "finish_reason": "stop" if hit is not None else "length"}, None))
        del out, enc
        return res

    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()
    return run_batch, info, (lambda e: isinstance(e, torch.cuda.OutOfMemoryError)), cleanup, tok
