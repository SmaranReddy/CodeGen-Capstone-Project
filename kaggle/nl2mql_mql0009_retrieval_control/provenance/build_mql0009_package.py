"""Build the MQL-0009 retrieval-control Kaggle package (generate-only) and its local preflight records.

Run with a Python that has `transformers` (tokenizer only; no torch needed), e.g. the project tokenizer venv:
    ~/codegen_data/venv_tok/Scripts/python.exe -I -B scripts/nl2mql_mql0009/build_mql0009_package.py \
        --tokenizer-dir ~/codegen_data/qwen3b_tok_488639f1 [--overwrite]

No model inference, no MongoDB / Atlas connection, no gold-SQL execution. SQLite files of the 16 held-out Spider-train
databases are READ (schema, row values) to derive the Atlas-form prompt context exactly as scripts/spider1/import_atlas.py
would store it; the derived per-collection counts and type-tagged row digests become the provisioning contract that
scoring must verify on Atlas (full parity) before any result is accepted.

Inputs (read only): approved corpus v2.1; spider-pool-v3 val_internal (item membership); spider-sql-v1 val_internal
(questions, gold SQL and execution hashes; gold stays local); Spider tables.json (train/val database entries only); the
MQL-0008 package (byte-identical source files, config and evaluation database names). Spider dev / test question or
SQL files are NOT opened.
"""
import argparse
import json
import random
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
M8_PKG = ROOT / "kaggle/nl2mql_mql0008_approved_corpus_v2_1"
sys.path.insert(0, str(HERE / "pkg_src"))
sys.path.insert(1, str(M8_PKG / "src"))
sys.path.insert(2, str(ROOT / "scripts/nl2mql"))
sys.path.insert(3, str(ROOT / "scripts/spider1"))
sys.path.insert(4, str(HERE))
import mql0009_core as C        # noqa: E402
import mql0008_core as M8       # noqa: E402
import patterns as PT           # noqa: E402
import dev1034_core as D        # noqa: E402  (unchanged MQL-0006 core: prompt isolation, gold-leak check, digests)
import mql0009_stats as STATS   # noqa: E402

PACKAGE = C.PACKAGE
PKG_OUT = ROOT / "kaggle" / PACKAGE
ZIP_OUT = ROOT / "kaggle" / f"{PACKAGE}_bundle.zip"
LOCAL = ROOT / "outputs/nl2mql_mql0009_retrieval_control"
CORPUS_SRC = ROOT / "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl"
V3_VAL = ROOT / "data/processed/spider_pool_v3/val_internal.jsonl"
V3_MANIFEST = ROOT / "data/processed/spider_pool_v3/manifest.json"
S1_VAL = ROOT / "data/processed/spider_sql_v1/val_internal.jsonl"
S1_MANIFEST = ROOT / "data/processed/spider_sql_v1/manifest.json"
V2_VAL = ROOT / "data/processed/spider_pool_v2/val_internal.jsonl"
TABLES = ROOT / "data/raw/spider1/spider_data/tables.json"
DB_DIR = ROOT / "data/raw/spider1/spider_data/database"
SPIDER1_CODE = ROOT / "scripts/spider1"
M8_CFG = M8_PKG / "configs/experiment.json"
DEV_ATLAS_MANIFEST = ROOT / "data/interim/spider1/atlas_collections.json"   # names of the 20 Spider-dev DBs on Atlas

SHARED_SRC = ("mql0008_core.py", "patterns.py", "retrieval.py", "rag_block.py")
RANDOM_SEEDS = [20261181, 20261182, 20261183, 20261184, 20261185]     # registered 2026-11-18 before any plan was built
FIXED_SEED = 20261186
NEAR_DUP = 0.8
PROPOSED_ATLAS_DB = "codegen_spider1_heldout"
MQL0008_RATES_S_PER_ITEM = {"zero_shot": 2.3 * 3600 / 892, "rag": 12198.8 / 892}   # MQL-0008 T4 wall clock


# ------------------------------------------------------------------ sources
def load_corpus():
    sha = C.sha256_file(CORPUS_SRC)
    C.need(sha == C.EXPECTED_CORPUS_SHA256, f"approved corpus sha256 {sha} != {C.EXPECTED_CORPUS_SHA256}")
    rows = C.read_jsonl(CORPUS_SRC)
    C.need(len(rows) == C.EXPECTED_CORPUS == len({r["id"] for r in rows}), "approved corpus is not 3,445 unique records")
    C.need(all(C.eligible_ok(r) for r in rows), "ineligible approved-corpus row")
    return rows


def build_eval_set():
    """Held-out evaluation set = spider-pool-v3 val_internal membership (audited: Tier A/B excluded in v2, semantic
    tiers excluded in v3), records taken from spider-sql-v1 val_internal (the source of the MQL-0006/0008 protocol:
    question text, gold SQL and execution hashes). Order = v3 file order. -> (records, funnel)."""
    v3 = C.read_jsonl(V3_VAL)
    s1 = {r["id"]: r for r in C.read_jsonl(S1_VAL)}
    v2_n = len(C.read_jsonl(V2_VAL))
    recs, ws = [], []
    for r in v3:
        C.need(r["id"] in s1, f"{r['id']}: not in spider-sql-v1 val_internal")
        s = s1[r["id"]]
        C.need(s["sql"] == r["sql"] and s["db_id"] == r["db_id"], f"{r['id']}: SQL/db differs between v3 and spider-sql-v1")
        C.need(s["tier"] == "exec_ok" and s["execution"]["status"] == "ok", f"{r['id']}: not exec_ok")
        C.need(s["execution"]["multiset_sha256"] == r["result_sha256"] or s["execution"]["ordered_sha256"] == r["result_sha256"],
               f"{r['id']}: result hash differs between v3 and spider-sql-v1")
        if s["question"] != r["question"]:
            ws.append({"id": r["id"], "v3_question": r["question"], "spider_sql_v1_question": s["question"]})
        recs.append({**s, "hardness": s["hardness"]})
    ids = [r["id"] for r in recs]
    C.need(len(ids) == len(set(ids)), "duplicate evaluation ids")
    funnel = {"spider_sql_v1_val_internal": len(s1), "spider_pool_v1_val_internal": 744, "spider_pool_v2_val_internal": v2_n,
              "spider_pool_v3_val_internal": len(v3),
              "notes": {"797->744": "spider-pool v1 build (docs/spider_pool_independent_audit.md)",
                        "744->734": "spider-pool v2: Tier A (4) + Tier B (6) exclusions",
                        "734->639": "spider-pool v3: 4 adjudicated semantic defects + 91 semantic-review (detector) hits removed"},
              "question_text_source": "spider-sql-v1 val_internal `question` (the MQL-0006/0008 protocol source); 4 items differ "
                                      "from spider-pool-v3 `question` in whitespace/punctuation only", "question_whitespace_diffs": ws}
    return recs, funnel


def protected_checks(recs, corpus):
    m8 = json.loads(M8_CFG.read_text(encoding="utf-8"))
    m8_eval_dbs = sorted({e["db_id"] for e in C.read_jsonl(M8_PKG / "data/eval/eval_inputs.jsonl")})
    dev_dbs = sorted({v["db_id"] for v in json.loads(DEV_ATLAS_MANIFEST.read_text(encoding="utf-8")).values()})
    eval_dbs = sorted({r["db_id"] for r in recs})
    s1man = json.loads(S1_MANIFEST.read_text(encoding="utf-8"))
    out = {"mql0008_eval_ids_sha256": m8["evaluation"]["ids_sha256"], "mql0008_eval_db_ids": m8_eval_dbs,
           "spider_dev_db_ids_on_atlas": dev_dbs, "eval_db_ids": eval_dbs,
           "eval_db_overlap_with_mql0008_eval": sorted(set(eval_dbs) & set(m8_eval_dbs)),
           "eval_db_overlap_with_spider_dev": sorted(set(eval_dbs) & set(dev_dbs)),
           "eval_ids_with_dev_prefix": [r["id"] for r in recs if not r["id"].startswith("spider_train_")],
           "eval_source_files": sorted({r["source"]["file"] for r in recs}),
           "corpus_db_overlap_with_spider_dev": sorted({r["db_id"] for r in corpus} & set(dev_dbs)),
           "spider_test_disjointness": "spider-sql-v1 manifest: train / val_internal / dev / test share no databases "
                                       "(tests/test_spider_sql_v1.py); Spider test files were not opened here",
           "spider_sql_v1_val_rule": "16 whole train_spider DBs picked by a seeded shuffle of db_ids (seed 20261027) "
                                     "(docs/spider_sql_v1_dataset.md)",
           "spider_sql_v1_manifest_sha256": C.sha256_file(S1_MANIFEST)}
    C.need(not out["eval_db_overlap_with_mql0008_eval"] and not out["eval_db_overlap_with_spider_dev"]
           and not out["eval_ids_with_dev_prefix"] and out["eval_source_files"] == ["train_spider.json"]
           and not out["corpus_db_overlap_with_spider_dev"], f"protected-set check failed: {out}")
    return out


# ------------------------------------------------------------------ Atlas-form context (offline; same mapping as import_atlas.py)
def derive_context(db_ids, db_dir=DB_DIR):
    """-> (atlas_collections, prompt_samples, mapping). Mirrors scripts/spider1/import_atlas.import_db without MongoDB:
    one collection per table `<db>__<table>`, columns in PRAGMA order, rows in rowid order, observed types via the same
    bson_type(); sample documents = first two rows (Atlas natural order = insertion order = rowid order)."""
    import importlib
    import spider1_kit as K
    K.DB_DIR = Path(db_dir)
    IA = importlib.import_module("import_atlas")    # byte-reuse of bson_type and _has_rowid (no connection on import)
    atlas, samples, colls = {}, {}, {}
    for db in sorted(db_ids):
        path = Path(db_dir) / db / f"{db}.sqlite"
        con = K.sqlite_ro(db)
        try:
            for table in K.tables(db):
                cols_decl = K.table_schema(db, table)
                cols = [c for c, _ in cols_decl]
                name = K.collection_name(db, table)
                q = ", ".join(f'"{c}"' for c in cols)
                rows = con.execute(f'SELECT {q} FROM "{table}" ORDER BY rowid').fetchall() if IA._has_rowid(con, table) \
                    else con.execute(f'SELECT {q} FROM "{table}"').fetchall()
                docs = [dict(zip(cols, r)) for r in rows]
                types = {}
                for d_ in docs:
                    for c in cols:
                        types.setdefault(c, set()).add(IA.bson_type(d_[c]))
                C.need(not any(isinstance(v, (bytes, bytearray)) for d_ in docs for v in d_.values()), f"{name}: BLOB values")
                atlas[name] = {"db_id": db, "table": table, "documents": len(docs),
                               "fields": {c: sorted(types.get(c, {"null"})) for c in cols}, "declared": dict(cols_decl)}
                samples[name] = [dict(d_) for d_ in docs[:2]]
                n, dig, replaced = D.sqlite_table_digest(path, table, cols)
                C.need(n == len(docs), f"{name}: digest row count")
                colls[name] = {"db_id": db, "table": table, "fields": cols, "declared_sql_types": dict(cols_decl),
                               "observed_bson_types": atlas[name]["fields"], "documents": len(docs), "sqlite_rows": n,
                               "row_multiset_sha256": dig, "utf8_replaced_strings": replaced}
        finally:
            con.close()
        C.need(D.utf8_replacements(path) == 0, f"{db}: invalid UTF-8 text (needs an Atlas-form reference rule)")
    mapping = {"atlas_db": PROPOSED_ATLAS_DB, "db_ids": sorted(db_ids), "collections": colls,
               "conventions": json.loads((ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2/data/atlas_mapping.json")
                                         .read_text(encoding="utf-8"))["conventions"],
               "status": "PROVISIONING CONTRACT - these collections do NOT exist on Atlas yet (2026-11-18)",
               "derivation": "offline from the Spider SQLite files with the import_atlas.py value mapping; never read from MongoDB",
               "sqlite_sha256": {db: C.sha256_file(Path(db_dir) / db / f"{db}.sqlite") for db in sorted(db_ids)}}
    return atlas, samples, mapping


def prompt_modules(context_dir):
    """The unmodified prompt_spider1.py + spider1_kit.py, pointed at the derived context (dev1034_core loader)."""
    return D.load_repo_modules(SPIDER1_CODE, context_dir, DB_DIR)


# ------------------------------------------------------------------ leakage audit
_KW = set("SELECT FROM WHERE GROUP BY ORDER HAVING LIMIT JOIN ON AS AND OR NOT IN LIKE BETWEEN IS NULL DISTINCT COUNT SUM AVG "
          "MIN MAX ASC DESC UNION INTERSECT EXCEPT EXISTS INNER LEFT OUTER CROSS NATURAL CASE WHEN THEN ELSE END ALL ANY".split())


def sql_fingerprint(sql):
    """Schema- and literal-masked SQL skeleton: keywords kept, identifiers -> ID, strings -> V, numbers -> N."""
    s = re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", " \x01 ", sql or "")
    s = re.sub(r"(?<![A-Za-z_])\d+(\.\d+)?", " \x02 ", s)
    out = []
    for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\x01|\x02|[^\sA-Za-z0-9_]", s):
        out.append("V" if t == "\x01" else "N" if t == "\x02" else t.upper() if t.upper() in _KW else "ID" if t[0].isalpha() or t[0] == "_" else t)
    return " ".join(out)


def leakage_audit(view_rows, recs, threshold=NEAR_DUP):
    """Pre-registered gates (decided before running):
      L1 database overlap ............ STOP on any hit
      L2 id / source-index overlap ... STOP on any hit
      L3 exact raw question .......... evaluation-specific retrieval exclusion (all demonstration arms)
      L4 normalised question ......... exclusion
      L5 token Jaccard >= 0.8 ........ exclusion (the MQL-0007/0008 rule, raw question vs corpus question / question_raw)
      L6 normalised SQL equality ..... REPORT ONLY (gold SQL never drives retrieval or prompts)
      L7 masked SQL skeleton equality  REPORT ONLY (structural match is what relevant retrieval is supposed to find)
    -> (local report incl. SQL info, exclusions, package report without any SQL)."""
    eval_dbs = {r["db_id"] for r in recs}
    l1 = sorted({r["db_id"] for r in view_rows} & eval_dbs)
    l2_ids = sorted({r["id"] for r in view_rows} & {r["id"] for r in recs})
    corp_idx = {(r["source"]["origin"]["file"], r["source"]["origin"]["index"]) for r in view_rows}
    l2_src = sorted(r["id"] for r in recs if (r["source"]["file"], r["source"]["index"]) in corp_idx)
    C.need(not l1 and not l2_ids and not l2_src, f"leakage STOP: db overlap {l1}, id overlap {l2_ids[:5]}, source overlap {l2_src[:5]}")
    raw = {}
    nq = {}
    for r in view_rows:
        for q in {r["question"], r.get("question_raw") or r["question"]}:
            raw.setdefault(q, set()).add(r["id"])
            nq.setdefault(M8.norm_question(q), set()).add(r["id"])
    csets = [(r["id"], M8.token_set(r["question"]) | M8.token_set(r.get("question_raw") or "")) for r in view_rows]
    l3, l4, l5, best = [], [], [], []
    for e in recs:
        for c in sorted(raw.get(e["question"], ())):
            l3.append({"eval_id": e["id"], "corpus_id": c})
        for c in sorted(nq.get(M8.norm_question(e["question"]), ())):
            l4.append({"eval_id": e["id"], "corpus_id": c})
        ts = M8.token_set(e["question"])
        top = 0.0
        for cid, cs in csets:
            j = M8.jaccard(ts, cs)
            top = max(top, j)
            if j >= threshold:
                l5.append({"eval_id": e["id"], "corpus_id": cid, "token_jaccard": round(j, 4)})
        best.append(top)
    excl = {}
    for gate, hits in (("L3_exact_question", l3), ("L4_normalised_question", l4), ("L5_token_jaccard_ge_0.8", l5)):
        for h in hits:
            excl.setdefault((h["eval_id"], h["corpus_id"]), set()).add(gate)
    exclusions = [{"eval_id": e, "corpus_id": c, "reason": sorted(g)} for (e, c), g in sorted(excl.items())]
    sql_n = {}
    sql_f = {}
    for r in view_rows:
        sql_n.setdefault(M8.norm_sql(r["sql"]), []).append(r["id"])
        sql_f.setdefault(sql_fingerprint(r["sql"]), []).append(r["id"])
    l6 = [{"eval_id": e["id"], "corpus_ids": sql_n[M8.norm_sql(e["sql"])]} for e in recs if M8.norm_sql(e["sql"]) in sql_n]
    l7 = Counter(min(len(sql_f.get(sql_fingerprint(e["sql"]), [])), 10) for e in recs)
    l7_items = sum(1 for e in recs if sql_fingerprint(e["sql"]) in sql_f)
    hist = Counter(round(b, 1) for b in best)
    local = {"gates": leakage_audit.__doc__, "n_eval": len(recs), "n_corpus_view": len(view_rows),
             "L1_db_overlap": l1, "L2_id_overlap": l2_ids, "L2_source_index_overlap": l2_src,
             "L3_exact_question": l3, "L4_normalised_question": l4, "L5_near_duplicates": l5,
             "max_token_jaccard_histogram": {str(k): v for k, v in sorted(hist.items())},
             "L6_normalised_sql_equal": {"items": len(l6), "pairs": l6, "action": "report only"},
             "L7_masked_sql_skeleton": {"items_with_any_corpus_match": l7_items, "match_count_histogram_capped10": dict(sorted(l7.items())),
                                        "action": "report only"},
             "exclusions": exclusions}
    pkg = {k: local[k] for k in ("n_eval", "n_corpus_view", "L1_db_overlap", "L2_id_overlap", "L2_source_index_overlap",
                                 "L3_exact_question", "L4_normalised_question", "L5_near_duplicates",
                                 "max_token_jaccard_histogram", "exclusions")}
    pkg["sql_based_checks"] = "L6/L7 were computed locally from gold SQL and are deliberately NOT shipped (report only)"
    return local, exclusions, pkg


# ------------------------------------------------------------------ fixed examples
def select_fixed(view_rows, recs, seed=FIXED_SEED, threshold=NEAR_DUP, k=C.K_DEMOS):
    """Registered procedure: shuffle the eligible corpus ids (sorted) with random.Random(seed); walk the permutation and
    accept a record iff (a) its database is not already used by an accepted record, (b) its normalised question and its
    MQL differ from the accepted ones, (c) its question passes the question-level leakage gates (no exact / normalised /
    Jaccard >= 0.8 match with ANY evaluation question). Gold SQL and evaluation labels are never used. Skips are logged;
    zero skips at (c) means the selection did not depend on the evaluation questions at all."""
    by = {r["id"]: r for r in view_rows}
    order = sorted(by)
    random.Random(seed).shuffle(order)
    eq_norm = {M8.norm_question(e["question"]) for e in recs}
    eq_raw = {e["question"] for e in recs}
    eq_sets = [M8.token_set(e["question"]) for e in recs]
    acc, log = [], []
    for i in order:
        r = by[i]
        if len(acc) == k:
            break
        qs = {r["question"], r.get("question_raw") or r["question"]}
        why = None
        if r["db_id"] in {by[a]["db_id"] for a in acc}:
            why = "database already used"
        elif M8.norm_question(r["question"]) in {M8.norm_question(by[a]["question"]) for a in acc} or \
                M8.compact(r["mql"]) in {M8.compact(by[a]["mql"]) for a in acc}:
            why = "duplicate question/MQL"
        elif qs & eq_raw or {M8.norm_question(q) for q in qs} & eq_norm or \
                any(M8.jaccard(M8.token_set(r["question"]) | M8.token_set(r.get("question_raw") or ""), s) >= threshold for s in eq_sets):
            why = "question-level leakage gate"
        log.append({"position": len(log), "corpus_id": i, "accepted": why is None, "skip_reason": why})
        if why is None:
            acc.append(i)
    C.need(len(acc) == k, "could not select the fixed examples")
    return acc, {"seed": seed, "procedure": select_fixed.__doc__, "draw_log": log,
                 "skipped_for_leakage": sum(x["skip_reason"] == "question-level leakage gate" for x in log),
                 "selected": [{"id": a, "db_id": by[a]["db_id"], "question": by[a]["question"],
                               "sql_pattern": sorted(PT.sql_patterns(by[a]["sql"]) - {"select"})} for a in acc]}


# ------------------------------------------------------------------ config
def make_config(recs, view, fixed, fixed_info, exclusions, prompts_sha, plans_sha, protected):
    m8 = json.loads(M8_CFG.read_text(encoding="utf-8"))
    ids = [r["id"] for r in recs]
    n = len(ids)
    zs_h = n * MQL0008_RATES_S_PER_ITEM["zero_shot"] / 3600
    rag_h = n * MQL0008_RATES_S_PER_ITEM["rag"] / 3600
    runs = {"A_zero_shot": {"family": "zero_shot", "k": 0, "label": "A. zero-shot (no demonstrations)"}}
    for i, s in enumerate(RANDOM_SEEDS, 1):
        runs[f"B_random_s{i}"] = {"family": "random", "k": 3, "seed": s,
                                  "label": f"B. random RAG seed {i}: retrieval.Retriever(method='random', seed={s})"}
    runs["C_fixed"] = {"family": "fixed", "k": 3, "label": "C. fixed-example RAG: the same 3 frozen corpus ids for every question"}
    runs["D_hybrid"] = {"family": "hybrid", "k": 3, "label": "D. MQL-0008 hybrid RAG (hybrid_masked, w=1.0), unchanged"}
    model = dict(m8["model"])
    model["dtype_note"] = ("bfloat16 exactly as MQL-0006/0007/0008. Those runs used a Tesla T4 (compute capability 7.5), which has "
                           "no native bf16: PyTorch emulated bf16 (torch.cuda.is_bf16_supported() is True with emulation). The "
                           "notebook records compute capability and bf16_native per run, accepts only the registered GPU, and "
                           "never substitutes fp16/fp32.")
    return {
        "experiment_id": C.EXPERIMENT,
        "title": "MQL-0009 - NL->MQL retrieval-control experiment (zero-shot / random / fixed / hybrid RAG)",
        "package": PACKAGE,
        "status": "PACKAGE BUILT - NOT EXECUTED (no prediction or score exists)",
        "question": STATS.ANALYSIS_PLAN["question"],
        "registered_at": "2026-11-18",
        "model": model,
        "hardware": {"allowed_gpus": ["Tesla T4"], "min_gpu_gib": 14.0, "bf16_emulation_allowed": True,
                     "historical_mql0008": {"gpu": "Tesla T4", "dtype": "bfloat16", "torch": "2.11.0+cu130", "transformers": "5.20.2",
                                            "accelerate": "1.15.0", "tokenizers": "0.23.3", "bf16_supported_reported": True,
                                            "resolved_revision_recorded": None,
                                            "source": "outputs/nl2mql_mql0008_approved_corpus_v2_1/scoring_runs/run_20261117T053551Z/"
                                                      "predictions/arm_b_hybrid_rag/run_config.json"},
                     "resolution": "keep bf16 (comparability with MQL-0008 arm B); T4 bf16 is emulated, recorded as such; all 8 "
                                   "runs must share GPU type, dtype, torch and transformers (resume refuses otherwise)"},
        "prompt": {"zero_shot_version": m8["prompt"]["zero_shot_version"], "zero_shot_template_sha256": m8["prompt"]["zero_shot_template_sha256"],
                   "zero_shot_prompts_sha256": prompts_sha, "rag_version": m8["prompt"]["rag_version"], "rag_rule": m8["prompt"]["rag_rule"],
                   "schema_context": "nl-mql-spider1-v1 Atlas-form context for the 16 held-out DBs, derived offline from SQLite with the "
                                     "import_atlas.py mapping (provisioning contract: scoring requires full Atlas parity)"},
        "evaluation": {"n_items": n, "n_databases": len({r["db_id"] for r in recs}), "db_ids": sorted({r["db_id"] for r in recs}),
                       "ids_sha256": C.sha256_text("\n".join(ids)),
                       "source": "spider-pool-v3 val_internal membership (639) x spider-sql-v1 val_internal records; database-level "
                                 "split of Spider train (16 whole DBs, seeded shuffle seed 20261027); gold SQL / rows NOT shipped",
                       "protected_db_ids": sorted(set(protected["spider_dev_db_ids_on_atlas"])),
                       "protected_note": "the 892-item MQL-0008 evaluation set (Spider dev) and the 65-item test partition are not used"},
        "corpus": {"name": "approved_corpus_v2_1", "source_path": "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl",
                   "package_path": C.CORPUS_FILE, "n_entries": C.EXPECTED_CORPUS, "source_sha256": C.EXPECTED_CORPUS_SHA256,
                   "eligible_view": view, "review_label": m8["corpus"]["review_label"]},
        "retrieval": m8["retrieval"],
        "context": m8["context"],
        "arm_runs": runs,
        "random_retrieval": {"seeds": RANDOM_SEEDS, "method": "retrieval.Retriever(method='random'): per question, order = "
                             "random.Random(int(sha256(f'{seed}:{qid}:{question}')[:16], 16)).shuffle(corpus order); target-db, "
                             "duplicate and exclusion filtering, then the shared budget loop takes the first 3 that fit (uniform "
                             "over the eligible view, distinct)", "registered_at": "2026-11-18, before any plan was built"},
        "fixed_retrieval": {"ids": fixed, "db_ids": [x["db_id"] for x in fixed_info["selected"]], "seed": FIXED_SEED,
                            "skipped_for_leakage": fixed_info["skipped_for_leakage"],
                            "procedure": "seeded draw from the eligible view; corpus attributes + question-level leakage gate only; "
                                         "see preflight/fixed_selection.json"},
        "leakage_gates": {"near_duplicate_token_jaccard": NEAR_DUP,
                          "policy": "L1 db / L2 id+source overlap: STOP; L3 exact, L4 normalised question, L5 Jaccard >= 0.8: "
                                    "evaluation-specific retrieval exclusion after ranking, identical for every demonstration arm; "
                                    "L6/L7 SQL-based: report only (gold never drives retrieval)"},
        "retrieval_exclusions": [{"eval_id": x["eval_id"], "corpus_id": x["corpus_id"]} for x in exclusions],
        "plans_sha256": plans_sha,
        "session_plan": {
            "estimate_basis": "MQL-0008 Tesla T4 wall clock: zero-shot 2.3 h / 892 items, hybrid 12,198.8 s / 892 items",
            "estimated_hours": {"zero_shot_run": round(zs_h, 2), "each_rag_run": round(rag_h, 2), "total": round(zs_h + 7 * rag_h, 2)},
            "1": {"runs": ["A_zero_shot", "D_hybrid", "B_random_s1"], "est_hours": round(zs_h + 2 * rag_h, 2)},
            "2": {"runs": ["C_fixed", "B_random_s2", "B_random_s3"], "est_hours": round(3 * rag_h, 2)},
            "3": {"runs": ["B_random_s4", "B_random_s5"], "est_hours": round(2 * rag_h, 2)},
            "rule": "sessions 2 and 3 resume from the previous version's output (RESUME_FROM='auto') in the pinned environment; "
                    "a 12-hour Kaggle session cannot hold all runs. Session 1 holds D and one random seed together so the primary "
                    "contrast has a same-session pair. If the pinned environment is lost (resume refused), regenerate ALL runs "
                    "in one new environment under a new output; never mix environments"},
        "analysis_plan_sha256": STATS.plan_sha(),
        "frozen_reference": {"experiment": "MQL-0008", "config_sha256": C.sha256_file(M8_CFG),
                             "package_manifest_sha256": C.sha256_file(M8_PKG / "package_manifest.json"),
                             "shared_source_sha256": {f: C.sha256_file(M8_PKG / "src" / f) for f in SHARED_SRC},
                             "must_equal": ["model", "retrieval", "context", "prompt.zero_shot_template_sha256", "prompt.rag_version"]},
        "stated_differences_from_MQL-0008": [
            "evaluation set: 639 held-out Spider-train questions on 16 databases (spider-sql-v1 val_internal / spider-pool-v3) "
            "instead of the 892 Spider-dev questions; schema context derived offline (those DBs are not on Atlas yet)",
            "arms: A, five random seeds, fixed, hybrid (8 runs) instead of A and hybrid",
            "retrieval exclusions recomputed for this evaluation set (question-level gates); the MQL-0008 exclusions refer to dev ids",
            "prediction records add prediction_id, arm_family, random_seed, precision, runtime, parse_status, normalized_mql",
            "the eligible corpus view equals the full approved corpus (no corpus database is an evaluation database)"],
    }


# ------------------------------------------------------------------ package
README_SRC = HERE / "package_README.md"
PKG_TEST_SRC = HERE / "package_tests" / "test_mql0009_package.py"
REQ_SRC = HERE / "requirements-kaggle.txt"
SCHEMA_SRC = HERE / "prediction_record.schema.json"


def build(tokenizer_dir, overwrite=False):
    from transformers import AutoTokenizer
    if PKG_OUT.exists() or ZIP_OUT.exists():
        C.need(overwrite, f"{PKG_OUT} or {ZIP_OUT} exists; pass --overwrite to rebuild (changes the package hashes)")
        C.need(PKG_OUT.name == PACKAGE and PKG_OUT.parent == ROOT / "kaggle", "refusing to delete an unexpected path")
        shutil.rmtree(PKG_OUT, ignore_errors=True)
        ZIP_OUT.unlink(missing_ok=True)
    pre = LOCAL / "preflight"
    corpus = load_corpus()
    recs, funnel = build_eval_set()
    protected = protected_checks(recs, corpus)
    eval_dbs = sorted({r["db_id"] for r in recs})
    view_rows, view = C.eligible_view(corpus, eval_dbs)
    local_lk, exclusions, pkg_lk = leakage_audit(view_rows, recs)
    fixed, fixed_info = select_fixed(view_rows, recs)

    # context + frozen zero-shot prompts (input isolation: PromptInput = id, db_id, question only)
    atlas, samples, mapping = derive_context(eval_dbs)
    tj = json.loads(TABLES.read_text(encoding="utf-8"))
    need_dbs = set(eval_dbs) | {r["db_id"] for r in view_rows}
    tables_subset = [t for t in tj if t["db_id"] in need_dbs]
    C.need({t["db_id"] for t in tables_subset} == need_dbs, "tables.json misses a database")
    ctx_stage = LOCAL / "work" / "prompt_context"
    ctx_stage.mkdir(parents=True, exist_ok=True)
    M8.write_json(ctx_stage / "atlas_collections.json", atlas)
    M8.write_json(ctx_stage / "prompt_samples.json", samples)
    M8.write_json(ctx_stage / "tables.json", tables_subset)
    K, P = prompt_modules(ctx_stage)
    C.need(P.template_hash() == "7401bf140dd1b8a9dac057edd9f729a8624823604fa2bd6ae9cfaf1c1890c1be", "prompt template changed")
    prompts = D.build_prompts(recs, P)
    D.check_no_gold_leak(prompts, recs)
    evals = [{"id": r["id"], "db_id": r["db_id"], "question": r["question"]} for r in recs]

    # plans for all 8 runs with the pinned tokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    count_tokens = lambda s: len(tok.encode(s, add_special_tokens=False))   # noqa: E731
    docs = M8.build_docs(view_rows, {t["db_id"]: t for t in tables_subset})
    docs_by_id = {d["doc_id"]: d for d in docs}
    infos = M8.eval_schema_infos(atlas, eval_dbs)

    PKG_OUT.mkdir(parents=True)
    w = lambda rel, obj: M8.write_json(PKG_OUT / rel, obj)        # noqa: E731
    (PKG_OUT / "data/eval").mkdir(parents=True)
    M8.write_jsonl(PKG_OUT / "data/eval/prompts.jsonl", prompts)
    prompts_sha = C.sha256_file(PKG_OUT / "data/eval/prompts.jsonl")
    pre_cfg = make_config(recs, view, fixed, fixed_info, exclusions, prompts_sha, {}, protected)
    runs = C.check_arm_design(pre_cfg)
    plans, plans_sha = {}, {}
    for run in runs:
        plan = C.build_plan(pre_cfg, run, evals, prompts, docs, infos, count_tokens)
        C.plan_checks(pre_cfg, run, plan, evals, docs_by_id)
        plans[run["name"]], plans_sha[run["name"]] = plan, C.plan_sha(plan)
    # hybrid run == MQL-0008 build_plan on the same inputs (byte-for-byte rows, modulo the two added keys)
    m8_cfg = {"arms": {"arm_b_hybrid_rag": {"retrieval": "hybrid_masked", "k": 3}}, "retrieval": pre_cfg["retrieval"],
              "context": pre_cfg["context"], "retrieval_exclusions": pre_cfg["retrieval_exclusions"]}
    m8_plan = M8.build_plan(m8_cfg, "arm_b_hybrid_rag", evals, prompts, docs, infos, count_tokens, arm_label="D_hybrid")
    C.need([{k: v for k, v in p.items() if k not in ("arm_family", "seed")} for p in plans["D_hybrid"]] == m8_plan,
           "D_hybrid plan differs from mql0008_core.build_plan")
    cfg = make_config(recs, view, fixed, fixed_info, exclusions, prompts_sha, plans_sha, protected)
    prompts_by_run = {n: C.prompts_for_run(plans[n], prompts, docs_by_id) for n in plans}
    C.check_prompt_equivalence(prompts_by_run, prompts)

    # ---- write package
    for f in SHARED_SRC:
        (PKG_OUT / "src").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(M8_PKG / "src" / f, PKG_OUT / "src" / f)
    shutil.copyfile(HERE / "pkg_src/mql0009_core.py", PKG_OUT / "src/mql0009_core.py")
    (PKG_OUT / "data/corpus").mkdir(parents=True)
    shutil.copyfile(CORPUS_SRC, PKG_OUT / C.CORPUS_FILE)
    M8.write_jsonl(PKG_OUT / "data/corpus/corpus_features.jsonl",
                   [{k: d[k] for k in ("doc_id", "db_id", "question", "masked_question", "patterns", "content")} for d in docs])
    w("data/corpus/eligible_view.json", {**view, "ids": [r["id"] for r in view_rows]})
    M8.write_jsonl(PKG_OUT / "data/eval/eval_inputs.jsonl", evals)
    w("data/context/atlas_collections.json", atlas)
    w("data/context/prompt_samples.json", samples)
    w("data/context/tables_subset.json", tables_subset)
    for n, plan in plans.items():
        M8.write_jsonl(PKG_OUT / f"data/plans/plan_{n}.jsonl", plan)
    w("data/leakage_report.json", pkg_lk)
    w("configs/experiment.json", cfg)
    shutil.copyfile(SCHEMA_SRC, PKG_OUT / "configs/prediction_record.schema.json")
    shutil.copyfile(REQ_SRC, PKG_OUT / "requirements-kaggle.txt")
    shutil.copyfile(README_SRC, PKG_OUT / "README.md")
    (PKG_OUT / "tests").mkdir()
    shutil.copyfile(PKG_TEST_SRC, PKG_OUT / "tests/test_mql0009_package.py")
    import make_mql0009_notebook as NB
    NB.build(PKG_OUT / f"{PACKAGE}.ipynb")
    (PKG_OUT / "provenance").mkdir()
    for f in ("build_mql0009_package.py", "make_mql0009_notebook.py", "mql0009_stats.py"):
        shutil.copyfile(HERE / f, PKG_OUT / "provenance" / f)

    # ---- local preflight records (never shipped: they hold gold-SQL-derived audit data)
    M8.write_json(pre / "eval_split_registration.json", {
        "experiment": C.EXPERIMENT, "registered_at": C.now(), "split": "spider-sql-v1 val_internal (database-level), spider-pool-v3 membership",
        "n_items": len(recs), "db_ids": eval_dbs, "items_per_db": dict(Counter(r["db_id"] for r in recs)),
        "hardness": dict(Counter(r["hardness"] for r in recs)), "ids": [r["id"] for r in recs],
        "ids_sha256": cfg["evaluation"]["ids_sha256"], "funnel": funnel, "protected_checks": protected,
        "inputs_sha256": {str(p.relative_to(ROOT)): C.sha256_file(p) for p in (V3_VAL, V3_MANIFEST, S1_VAL, S1_MANIFEST, V2_VAL, CORPUS_SRC, TABLES)}})
    M8.write_json(pre / "leakage_audit.json", local_lk)
    M8.write_json(pre / "fixed_selection.json", fixed_info)
    M8.write_json(LOCAL / "scoring_contract/atlas_mapping.json", mapping)
    M8.write_json(LOCAL / "scoring_contract/atlas_collections.json", atlas)
    STATS.write_plan()
    shutil.rmtree(LOCAL / "work", ignore_errors=True)

    # ---- manifest + checksums + zip
    finalize()
    return cfg, local_lk, fixed_info


def finalize():
    files = sorted(p.relative_to(PKG_OUT).as_posix() for p in PKG_OUT.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and ".pytest_cache" not in p.parts
                   and p.name not in ("package_manifest.json", "SHA256SUMS.txt"))
    cfg = json.loads((PKG_OUT / "configs/experiment.json").read_text(encoding="utf-8"))
    man = {"package": PACKAGE, "experiment": C.EXPERIMENT, "built_at": C.now(), "status": cfg["status"],
           "files": {n: {"sha256": C.sha256_file(PKG_OUT / n), "bytes": (PKG_OUT / n).stat().st_size} for n in files},
           "retrieval_corpus": {"path": C.CORPUS_FILE, "sha256": C.EXPECTED_CORPUS_SHA256, "records": C.EXPECTED_CORPUS},
           "config_sha256": C.sha256_file(PKG_OUT / "configs/experiment.json"), "analysis_plan_sha256": STATS.plan_sha(),
           "model": {"id": C.MODEL_ID, "revision": C.MODEL_REVISION}, "contains_gold_labels": False, "contains_credentials": False}
    M8.write_json(PKG_OUT / "package_manifest.json", man)
    names = files + ["package_manifest.json"]
    (PKG_OUT / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(PKG_OUT / n)}  {n}\n" for n in names), encoding="utf-8", newline="\n")
    tmp = ZIP_OUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names + ["SHA256SUMS.txt"]:
            zi = zipfile.ZipInfo(f"{PACKAGE}/{n}", date_time=(2026, 11, 18, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, (PKG_OUT / n).read_bytes())
    tmp.replace(ZIP_OUT)
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=False)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--finalize", action="store_true", help="re-hash after editing README/tests only")
    a = ap.parse_args(argv)
    if a.finalize:
        man = finalize()
        print("finalized", len(man["files"]), "files ->", ZIP_OUT)
        return
    C.need(a.tokenizer_dir, "--tokenizer-dir is required")
    cfg, lk, fx = build(Path(a.tokenizer_dir).expanduser(), a.overwrite)
    print(json.dumps({"package": str(PKG_OUT), "zip": str(ZIP_OUT), "zip_sha256": C.sha256_file(ZIP_OUT),
                      "eval_items": cfg["evaluation"]["n_items"], "exclusions": len(cfg["retrieval_exclusions"]),
                      "fixed_ids": cfg["fixed_retrieval"]["ids"], "fixed_skipped_for_leakage": fx["skipped_for_leakage"],
                      "L6_sql_equal_items": lk["L6_normalised_sql_equal"]["items"],
                      "L7_skeleton_items": lk["L7_masked_sql_skeleton"]["items_with_any_corpus_match"],
                      "session_plan": cfg["session_plan"]["estimated_hours"]}, indent=1))


if __name__ == "__main__":
    main()
