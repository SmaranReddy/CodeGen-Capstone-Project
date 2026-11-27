"""Build the MQL-0013 Kaggle package (generate only): R0 vs R3 retrieval, both through the MQL-0010 D0V recipe, on
Spider test_full (2,147 questions, 40 databases).

Run (venv_rag: transformers + torch + qdrant-client; no network, no MongoDB):
  ~/codegen_data/venv_rag/Scripts/python -B scripts/nl2mql_mql0013/build_mql0013_package.py [--overwrite]

Fail-closed steps, all local:
  1. benchmark: test_full ids/order; database AND id disjointness from every corpus / evaluation source used so far
     (approved corpus, MQL-0010 holdout + dev + build DBs, X0, spider-sql-v1 train / val_internal / dev_full, the
     expanded-split manifest by database only - the 65-item test file is never opened);
  2. registration (outputs/nl2mql_mql0013/registration/mql0013_registration.json) exists and its pinned source hashes
     match - the registration is written BEFORE this package and before any generation;
  3. Atlas-form context for the 40 test DBs (offline import_atlas mapping, test_database SQLite) + scoring contract;
  4. zero-shot base prompts with the unchanged nl-mql-spider1-v1 template (hash-checked), no gold SQL in any prompt;
  5. leakage audit L1-L7 against the 2,912-record MQL-0011 view; L3-L5 pairs excluded identically in both arms;
  6. plans: R0 = planner_dense lexical "bm25" (== MQL-0010 D0), R3 = lexical "rrf_fused" with the registered MQL-0011
     Qdrant collection (read-only, exact search), the pinned bge-small embedder and the CanonicalScorer;
  7. frozen per-arm prompts (system + user text) + hashes; shared / R0-only / R3-only partition.
Writes kaggle/nl2mql_mql0013_test_full/ (+ _bundle.zip), outputs/nl2mql_mql0013/{work,scoring_contract}/.
Never writes the corpus, the Qdrant storage, MQL-0008..0012 artifacts or any database.
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(1, str(HERE / "pkg_src"))
sys.path.insert(2, str(ROOT / "scripts/nl2mql_mql0011/src"))
sys.path.insert(3, str(ROOT / "scripts/nl2mql_mql0011"))
for p in ("scripts/nl2mql_mql0009", "scripts/nl2mql", "scripts/spider1"):
    sys.path.append(str(ROOT / p))
import mql0013_common as C      # noqa: E402
import mql0011_common as C11    # noqa: E402  (adds the MQL-0010 src and MQL-0009 package src to sys.path)

M10PKG_SRC = ROOT / "kaggle/nl2mql_mql0010_holdout/src"
PKG_SRC_FILES = ("mql0008_core.py", "mql0009_core.py", "patterns.py", "retrieval.py", "rag_block.py", "planner.py",
                 "prompt_v2.py", "recovery.py", "retriever_v2.py", "structure.py", "validator.py")
TEMPLATE_SHA = "7401bf140dd1b8a9dac057edd9f729a8624823604fa2bd6ae9cfaf1c1890c1be"
TOKENIZER_DIR = Path.home() / "codegen_data/qwen3b_tok_488639f1"
ZIP_DATE = (2026, 11, 22, 0, 0, 0)
need = C.need


# ------------------------------------------------------------------ 1. benchmark + disjointness
def load_benchmark():
    recs = C.read_jsonl(C.TEST_FULL)
    need(len(recs) == 2147 and len({r["id"] for r in recs}) == 2147, "test_full must hold 2,147 unique records")
    need(all(r["tier"] == "exec_ok" for r in recs), "test_full gold execution tier")
    return recs


def protected_sets():
    dbs, ids = {}, {}
    for name, p in C.PROTECTED_SPLITS.items():
        if p.suffix == ".jsonl":
            rows = C.read_jsonl(p)
            dbs[name] = {r["db_id"] for r in rows}
            ids[name] = {r["id"] for r in rows}
        elif name == "splits_expanded_manifest":
            m = json.loads(p.read_text(encoding="utf-8"))
            dbs[name] = set(m["database_assignments"])
            ids[name] = {e["id"] for e in m["examples"]}
        else:
            s = json.loads(p.read_text(encoding="utf-8"))
            dbs[name] = set(s["holdout"]["dbs"]) | set(s["dev"]["dbs"]) | set(s["build_dbs"] if isinstance(s["build_dbs"], list)
                                                                             else s["build_dbs"].get("dbs", []))
            ids[name] = set()
    corpus = C.read_jsonl(C.CORPUS)
    dbs["approved_corpus_v2_1"] = {r["db_id"] for r in corpus}
    ids["approved_corpus_v2_1"] = {r["id"] for r in corpus}
    return dbs, ids


def disjointness(recs):
    tdb, tid = {r["db_id"] for r in recs}, {r["id"] for r in recs}
    dbs, ids = protected_sets()
    rep = {name: {"db_overlap": sorted(tdb & d), "id_overlap": sorted(tid & ids[name])[:5]} for name, d in dbs.items()}
    bad = {k: v for k, v in rep.items() if v["db_overlap"] or v["id_overlap"]}
    need(not bad, f"benchmark is not disjoint: {bad}")
    return {"n_test_dbs": len(tdb), "checked": sorted(rep), "overlaps": 0}


# ------------------------------------------------------------------ 2. registration
def verify_registration():
    need(C.REGISTRATION.exists(), "registration missing: run register_mql0013.py first (before any package / generation)")
    reg = json.loads(C.REGISTRATION.read_text(encoding="utf-8"))
    for rel, h in reg["source_sha256"].items():
        need(C.sha256_file(ROOT / rel) == h, f"registered source changed after registration: {rel}")
    return reg


# ------------------------------------------------------------------ 3-7
def qdrant_snapshot():
    return {p.relative_to(C11.QDRANT_PATH).as_posix(): C.sha256_file(p) for p in sorted(C11.QDRANT_PATH.rglob("*"))
            if p.is_file() and p.name != ".lock"}


def build(overwrite=False):
    import build_mql0009_package as B9
    import dev1034_core as D
    import embedder as EM
    import mql0008_core as M8
    import planner as PL
    import planner_dense as PD
    import qdrant_store as QS
    import vector_store as VS
    from build_embedding_artifact import SPEC, load_artifact
    from transformers import AutoTokenizer

    recs = load_benchmark()
    disj = disjointness(recs)
    reg = verify_registration()
    need(reg["evaluation"]["ids_sha256"] == C.ids_sha256([r["id"] for r in recs]), "registered ids != test_full ids")
    if C.PKG_OUT.exists() or C.ZIP_OUT.exists():
        need(overwrite, f"{C.PKG_OUT} or {C.ZIP_OUT} exists; pass --overwrite")
        need(C.PKG_OUT.parent == ROOT / "kaggle" and C.PKG_OUT.name == C.PACKAGE, "unexpected package path")
        shutil.rmtree(C.PKG_OUT, ignore_errors=True)
        C.ZIP_OUT.unlink(missing_ok=True)
    tdb = sorted({r["db_id"] for r in recs})
    evals = [{"id": r["id"], "db_id": r["db_id"], "question": r["question"]} for r in recs]
    view, hdb = C11.load_view_rows()

    # 3. context + contract
    atlas, samples, mapping = B9.derive_context(tdb, db_dir=C.TEST_DB_DIR)
    mapping = {**mapping, "atlas_db": C.ATLAS_DB,
               "status": "PROVISIONING CONTRACT - MQL-0013 Spider test_full databases (insert-only target)"}
    C.write_json(C.CONTRACT, mapping)
    C.write_json(C.CONTRACT_DIR / "atlas_collections.json", atlas)
    test_tabs = {t["db_id"]: t for t in json.loads(C.TEST_TABLES.read_text(encoding="utf-8")) if t["db_id"] in set(tdb)}
    need(set(test_tabs) == set(tdb), "test_tables.json misses a test database")
    train_tabs = C11.load_tables()
    view_dbs = {r["db_id"] for r in view}
    tables_subset = [train_tabs[d] for d in sorted(view_dbs)] + [test_tabs[d] for d in tdb]
    stage = C.WORK / "prompt_context"
    C.write_json(stage / "atlas_collections.json", atlas)
    C.write_json(stage / "prompt_samples.json", samples)
    C.write_json(stage / "tables.json", [test_tabs[d] for d in tdb])

    # 4. zero-shot base prompts
    K, P = D.load_repo_modules(B9.SPIDER1_CODE, stage, C.TEST_DB_DIR)
    need(P.template_hash() == TEMPLATE_SHA, "zero-shot prompt template changed")
    prompts = D.build_prompts(evals, P)
    D.check_no_gold_leak(prompts, recs)

    # 5. leakage
    local, exclusions, pkg_leak = B9.leakage_audit(view, recs)
    C.write_json(C.WORK / "leakage_local.json", local)

    # 6. plans
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    count_tokens = lambda s: len(tok.encode(s, add_special_tokens=False))   # noqa: E731
    tables = {t["db_id"]: t for t in tables_subset}
    docs = M8.build_docs(view, tables)
    infos = M8.eval_schema_infos(atlas, tdb)
    spec, a_ids, _dbs, _shas, vec = load_artifact()
    q_before = qdrant_snapshot()
    store = QS.QdrantStore.open(spec, path=C11.QDRANT_PATH, exact=True)
    try:
        emb = EM.HFEmbedder(SPEC, threads=1)
        scorer = VS.CanonicalScorer(a_ids, vec, spec.dim)
        plans = {}
        for arm in C.PLAN_ARMS:
            flags = {"name": arm, "family": "hybrid", "lexical": C.LEXICAL[arm]}
            plans[arm] = PD.build_plan(flags, evals, prompts, docs, infos, atlas, tables, C.CTX, count_tokens, exclusions,
                                       forbidden_db_ids=sorted(hdb), store=store, embedder=emb, scorer=scorer)
    finally:
        store.close()
    need(qdrant_snapshot() == q_before, "Qdrant storage changed during plan building")

    # 7. frozen prompts per arm + partition
    by_doc = {d["doc_id"]: d for d in docs}
    fps = {p["id"]: p for p in prompts}
    arm_prompts = {}
    for arm in C.PLAN_ARMS:
        out = []
        for r in plans[arm]:
            system, user = PL.user_text_for(r, fps[r["id"]], by_doc)
            msgs = [{"role": "user", "content": user}]
            sha = PL.prompt_sha(system, msgs)
            need(sha == r["prompt_sha256"], f"{arm}/{r['id']}: prompt hash")
            out.append({"id": r["id"], "db_id": r["db_id"], "system": system, "messages": msgs, "prompt_sha256": sha})
        arm_prompts[arm] = out
    shared = [a["id"] for a, b in zip(arm_prompts["R0"], arm_prompts["R3"]) if a["prompt_sha256"] == b["prompt_sha256"]]
    sset = set(shared)
    diff = [e["id"] for e in evals if e["id"] not in sset]

    # package
    for d in ("src", "configs", "data/eval", "data/context", "data/plans", "data/prompts", "registration", "provenance"):
        (C.PKG_OUT / d).mkdir(parents=True, exist_ok=True)
    for n in PKG_SRC_FILES:
        shutil.copyfile(M10PKG_SRC / n, C.PKG_OUT / "src" / n)
    shutil.copyfile(HERE / "pkg_src/mql0013_core.py", C.PKG_OUT / "src/mql0013_core.py")
    shutil.copyfile(C.REGISTRATION, C.PKG_OUT / "registration" / C.REGISTRATION.name)
    C.write_jsonl(C.PKG_OUT / "data/eval/eval_inputs.jsonl", evals)
    C.write_jsonl(C.PKG_OUT / "data/eval/zero_shot_prompts.jsonl", prompts)
    C.write_json(C.PKG_OUT / "data/context/atlas_collections.json", atlas)
    for arm in C.PLAN_ARMS:
        C.write_jsonl(C.PKG_OUT / f"data/plans/plan_{arm}.jsonl", plans[arm])
        C.write_jsonl(C.PKG_OUT / f"data/prompts/prompts_{arm}.jsonl", arm_prompts[arm])
    for f in ("build_mql0013_package.py", "make_mql0013_notebook.py"):
        shutil.copyfile(HERE / f, C.PKG_OUT / "provenance" / f)
    C.write_json(C.WORK / "leakage_package.json", pkg_leak)
    canon = lambda o: json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))   # noqa: E731
    cfg = {
        "experiment_id": C.EXPERIMENT, "package": C.PACKAGE,
        "status": "PACKAGE BUILT - NOT EXECUTED (no prediction or score exists)",
        "registration_sha256": C.sha256_file(C.REGISTRATION),
        "model": {"id": "Qwen/Qwen2.5-Coder-3B-Instruct", "revision": "488639f1ff808d1d3d0ba301aef8c11461451ec5",
                  "label": "qwen2.5-coder-3b-instruct", "dtype": "bfloat16", "decoding": "greedy (do_sample=False)",
                  "max_new_tokens": 1024, "seed": 42, "batch_size_first_pass": 8, "batch_size_retry": 1,
                  "runtime": "transformers GPU, one card (CUDA_VISIBLE_DEVICES=0)"},
        "hardware": {"allowed_gpus": ["Tesla T4"], "min_gpu_gib": 14.0, "bf16_emulation_allowed": True},
        "context": C.CTX,
        "arms": {"R0": {"lexical": "bm25", "family": "hybrid", "k": 3, "diversify": "none"},
                 "R3": {"lexical": "rrf_fused", "family": "hybrid", "k": 3, "diversify": "none"},
                 "R0V": {"source": "R0", "feedback_mode": "schema"}, "R3V": {"source": "R3", "feedback_mode": "schema"}},
        "plans_sha256": {a: C.sha256_text(canon(plans[a])) for a in C.PLAN_ARMS},
        "prompts_sha256": {a: C.sha256_file(C.PKG_OUT / f"data/prompts/prompts_{a}.jsonl") for a in C.PLAN_ARMS},
        "partition": {"shared_ids": shared, "n_shared": len(shared), "n_diff": len(diff)},
        "prompt": {"zero_shot_version": "nl-mql-spider1-v1", "rag_version": "nl-mql-spider1-rag-v1",
                   "zero_shot_template_sha256": TEMPLATE_SHA},
        "evaluation": {"source": "data/processed/spider_sql_v1/test_full.jsonl", "n_items": len(evals),
                       "ids_sha256": C.ids_sha256([e["id"] for e in evals]), "db_ids": tdb},
        "retrieval": {"view_n": len(view), "view_ids_sha256": C11.VIEW_IDS_SHA256, "corpus_sha256": C11.CORPUS_SHA256,
                      "qdrant_index": C11.INDEX_NAME, "embedder_revision": C11.MODEL_REVISION,
                      "exclusions": exclusions, "n_exclusion_pairs": len(exclusions)},
        "scoring_contract": {"atlas_db": C.ATLAS_DB, "mapping_sha256": C.sha256_file(C.CONTRACT),
                             "collections": len(mapping["collections"]),
                             "documents": sum(m["documents"] for m in mapping["collections"].values())},
        "budget_drops": {a: sum(bool(r["dropped_budget"]) for r in plans[a]) for a in C.PLAN_ARMS},
        "disjointness": disj,
    }
    C.write_json(C.PKG_OUT / "configs/experiment.json", cfg)
    sys.path.insert(0, str(HERE))
    import make_mql0013_notebook as NB
    NB.build(C.PKG_OUT / f"{C.PACKAGE}.ipynb")
    man = finalize()
    C.write_json(C.WORK / "build_report.json", {
        "package_manifest_sha256": C.sha256_file(C.PKG_OUT / "package_manifest.json"), "zip_sha256": C.sha256_file(C.ZIP_OUT),
        "n_items": len(evals), "n_shared": len(shared), "n_diff": len(diff), "budget_drops": cfg["budget_drops"],
        "exclusion_pairs": len(exclusions), "scoring_contract": cfg["scoring_contract"], "disjointness": disj,
        "qdrant_unchanged": True, "tokenizer_dir": TOKENIZER_DIR.name})
    return cfg, man


def finalize():
    pkg = C.PKG_OUT
    files = sorted(p.relative_to(pkg).as_posix() for p in pkg.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
                   and p.name not in ("package_manifest.json", "SHA256SUMS.txt"))
    cfg = json.loads((pkg / "configs/experiment.json").read_text(encoding="utf-8"))
    man = {"package": C.PACKAGE, "experiment": C.EXPERIMENT, "status": cfg["status"],
           "files": {n: {"sha256": C.sha256_file(pkg / n), "bytes": (pkg / n).stat().st_size} for n in files},
           "config_sha256": C.sha256_file(pkg / "configs/experiment.json"), "registration_sha256": cfg["registration_sha256"],
           "model": {"id": cfg["model"]["id"], "revision": cfg["model"]["revision"]},
           "contains_gold_labels": False, "contains_credentials": False}
    C.write_json(pkg / "package_manifest.json", man)
    names = files + ["package_manifest.json"]
    (pkg / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(pkg / n)}  {n}\n" for n in names), encoding="utf-8",
                                        newline="\n")
    tmp = C.ZIP_OUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names + ["SHA256SUMS.txt"]:
            zi = zipfile.ZipInfo(f"{C.PACKAGE}/{n}", date_time=ZIP_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, (pkg / n).read_bytes())
    tmp.replace(C.ZIP_OUT)
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args(argv)
    if a.finalize:
        print("finalized", len(finalize()["files"]), "files")
        return
    cfg, man = build(a.overwrite)
    print(json.dumps({"zip": str(C.ZIP_OUT), "zip_sha256": C.sha256_file(C.ZIP_OUT), "files": len(man["files"]),
                      "n_items": cfg["evaluation"]["n_items"], "partition": {k: v for k, v in cfg["partition"].items()
                                                                            if k != "shared_ids"},
                      "budget_drops": cfg["budget_drops"], "exclusions": cfg["retrieval"]["n_exclusion_pairs"],
                      "scoring_contract": cfg["scoring_contract"]}, indent=1))


if __name__ == "__main__":
    main()
