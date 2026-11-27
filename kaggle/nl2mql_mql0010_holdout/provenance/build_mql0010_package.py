"""Build the MQL-0010 Kaggle package (generate-only) from the registered, tested implementation (registration A4).

Usage (needs a Python with `transformers` for the pinned tokenizer; no torch, no network, no database):
  ~/codegen_data/venv_tok/Scripts/python.exe -I -B scripts/nl2mql_mql0010/build_mql0010_package.py \
      --tokenizer-dir ~/codegen_data/qwen3b_tok_488639f1 [--overwrite]
  python -B scripts/nl2mql_mql0010/build_mql0010_package.py --finalize      # re-hash after editing README/notebook only

Fail-closed checks before anything is written:
  * registration_index.json and every document / source file it pins (A4) hash-match the repo;
  * approved corpus v2.1 hash; the retrieval view (2,912 records) via holdout_view.load_view; no holdout DB in docs;
  * holdout ids hash; frozen zero-shot prompts carry no gold SQL (dev1034_core.check_no_gold_leak);
  * plans A0 / D0 / E / EF built with the PINNED tokenizer and forbidden_db_ids = the 18 holdout DBs.
Writes: kaggle/nl2mql_mql0010_holdout/ (+ _bundle.zip) and outputs/nl2mql_mql0010/scoring_contract/ (Atlas contract).
Never writes the corpus, the registration files, MQL-0008/0009 artifacts or anything on a database.
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
M9PKG = ROOT / "kaggle/nl2mql_mql0009_retrieval_control"
sys.path.insert(0, str(HERE / "pkg_src"))
sys.path.insert(1, str(HERE / "src"))
sys.path.insert(2, str(M9PKG / "src"))
sys.path.insert(3, str(ROOT / "scripts/nl2mql_mql0009"))
sys.path.insert(4, str(ROOT / "scripts/nl2mql"))
sys.path.insert(5, str(ROOT / "scripts/spider1"))
import mql0010_core as C        # noqa: E402
import mql0008_core as M8       # noqa: E402
import holdout_view as HV       # noqa: E402
import planner as PL            # noqa: E402

PKG_OUT = ROOT / "kaggle" / C.PACKAGE
ZIP_OUT = ROOT / "kaggle" / f"{C.PACKAGE}_bundle.zip"
PRE = ROOT / "outputs/nl2mql_mql0010/preflight"
CONTRACT_DIR = ROOT / "outputs/nl2mql_mql0010/scoring_contract"
CORPUS = ROOT / "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl"
TABLES = ROOT / "data/raw/spider1/spider_data/tables.json"
HOLDOUT_ATLAS_DB = "codegen_spider1_holdout_mql0010"
SHARED_SRC = ("mql0008_core.py", "mql0009_core.py", "patterns.py", "retrieval.py", "rag_block.py")   # byte copies (MQL-0009 package)
TEMPLATE_SHA = "7401bf140dd1b8a9dac057edd9f729a8624823604fa2bd6ae9cfaf1c1890c1be"
ZIP_DATE = (2026, 11, 19, 0, 0, 0)


def verify_registration():
    idx_path = PRE / "registration_index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    for d in idx["documents"]:
        C.need(C.sha256_file(ROOT / d["path"]) == d["sha256"], f"registered document changed: {d['path']}")
    for name, h in idx["source_sha256"].items():
        C.need(C.sha256_file(HERE / "src" / name) == h, f"registered source changed: src/{name}")
    C.need(C.sha256_file(CORPUS) == idx["corpus_sha256"] == HV.CORPUS_SHA256, "approved corpus v2.1 changed")
    return idx_path, idx


def build(tokenizer_dir, overwrite=False):
    from transformers import AutoTokenizer
    import build_mql0009_package as B9          # derive_context / prompt_modules (offline, SQLite only)
    import dev1034_core as D
    idx_path, idx = verify_registration()
    docs_by_role = {d["role"]: ROOT / d["path"] for d in idx["documents"]}
    a4 = json.loads(docs_by_role["amendment_A4"].read_text(encoding="utf-8"))
    split = json.loads(docs_by_role["split_registration"].read_text(encoding="utf-8"))
    if PKG_OUT.exists() or ZIP_OUT.exists():
        C.need(overwrite, f"{PKG_OUT} or {ZIP_OUT} exists; pass --overwrite")
        C.need(PKG_OUT.name == C.PACKAGE and PKG_OUT.parent == ROOT / "kaggle", "refusing to delete an unexpected path")
        shutil.rmtree(PKG_OUT, ignore_errors=True)
        ZIP_OUT.unlink(missing_ok=True)
    hdb = sorted(HV.holdout_dbs(split))
    view = HV.load_view(CORPUS, split)                                              # R3
    items = M8.read_jsonl(PRE / "holdout_items.jsonl")
    C.need(C.sha256_text("\n".join(r["id"] for r in items)) == split["holdout"]["ids_sha256"], "holdout ids")
    evals = [{"id": r["id"], "db_id": r["db_id"], "question": r["question"]} for r in items]
    gold = {r["id"]: r for r in M8.read_jsonl(ROOT / "data/processed/spider_sql_v1/train.jsonl")}
    tj = json.loads(TABLES.read_text(encoding="utf-8"))
    need_dbs = set(hdb) | {r["db_id"] for r in view}
    tables_subset = [t for t in tj if t["db_id"] in need_dbs]
    C.need({t["db_id"] for t in tables_subset} == need_dbs, "tables.json misses a database")

    # Atlas-form context for the 18 holdout DBs (offline; import_atlas.py mapping) + scoring/provisioning contract
    atlas, samples, mapping = B9.derive_context(hdb)
    mapping = {**mapping, "atlas_db": HOLDOUT_ATLAS_DB,
               "status": "PROVISIONING CONTRACT - these collections do NOT exist on Atlas yet (MQL-0010, 2026-11-19)"}
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    M8.write_json(CONTRACT_DIR / "atlas_mapping.json", mapping)
    M8.write_json(CONTRACT_DIR / "atlas_collections.json", atlas)
    stage = ROOT / "outputs/nl2mql_mql0010/work/prompt_context"
    stage.mkdir(parents=True, exist_ok=True)
    M8.write_json(stage / "atlas_collections.json", atlas)
    M8.write_json(stage / "prompt_samples.json", samples)
    M8.write_json(stage / "tables.json", [t for t in tables_subset if t["db_id"] in set(hdb)])
    K, P = B9.prompt_modules(stage)
    C.need(P.template_hash() == TEMPLATE_SHA, "zero-shot prompt template changed")
    prompts = D.build_prompts(evals, P)
    D.check_no_gold_leak(prompts, [gold[r["id"]] for r in evals])

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    count_tokens = lambda s: len(tok.encode(s, add_special_tokens=False))   # noqa: E731
    tables = {t["db_id"]: t for t in tables_subset}
    docs = M8.build_docs(view, tables)
    HV.assert_docs_exclude(docs, set(hdb))
    infos = M8.eval_schema_infos(atlas, hdb)
    arms = {a["name"]: a for a in a4["arms"]}
    ctx = {"k": 3, "max_context_tokens": 1500, "n_ctx": 8192, "max_new_tokens": 1024, "margin_tokens": 64}
    plans = {}
    for arm in C.PLAN_ARMS:
        flags = {"name": arm, **{k: v for k, v in arms[arm]["flags"].items() if k != "validate_retry"}}
        plans[arm] = PL.build_plan(flags, evals, prompts, docs, infos, atlas, tables, ctx, count_tokens,
                                   split["leakage"]["L3_L5_exclusion_pairs"], forbidden_db_ids=hdb)
    dids = C.delta_ids(plans)

    PKG_OUT.mkdir(parents=True)
    for d in ("src", "registration", "configs", "data/corpus", "data/eval", "data/context", "data/plans", "provenance"):
        (PKG_OUT / d).mkdir(parents=True, exist_ok=True)
    for n in SHARED_SRC:
        shutil.copyfile(M9PKG / "src" / n, PKG_OUT / "src" / n)
    for n in idx["source_sha256"]:
        shutil.copyfile(HERE / "src" / n, PKG_OUT / "src" / n)
    shutil.copyfile(HERE / "pkg_src/mql0010_core.py", PKG_OUT / "src/mql0010_core.py")
    shutil.copyfile(idx_path, PKG_OUT / "registration/registration_index.json")
    for role, p in docs_by_role.items():
        shutil.copyfile(p, PKG_OUT / "registration" / p.name)
    shutil.copyfile(CORPUS, PKG_OUT / C.CORPUS_FILE)
    M8.write_jsonl(PKG_OUT / "data/eval/eval_inputs.jsonl", evals)
    M8.write_jsonl(PKG_OUT / "data/eval/prompts.jsonl", prompts)
    M8.write_json(PKG_OUT / "data/context/atlas_collections.json", atlas)
    M8.write_json(PKG_OUT / "data/context/prompt_samples.json", samples)
    M8.write_json(PKG_OUT / "data/context/tables_subset.json", tables_subset)
    for arm in C.PLAN_ARMS:
        M8.write_jsonl(PKG_OUT / f"data/plans/plan_{arm}.jsonl", plans[arm])
    for f in ("build_mql0010_package.py", "make_mql0010_notebook.py"):
        shutil.copyfile(HERE / f, PKG_OUT / "provenance" / f)
    shutil.copyfile(HERE / "package_README.md", PKG_OUT / "README.md")
    shutil.copyfile(HERE / "requirements-kaggle.txt", PKG_OUT / "requirements-kaggle.txt")
    cfg = {
        "experiment_id": C.EXPERIMENT, "package": C.PACKAGE,
        "status": "PACKAGE BUILT - NOT EXECUTED (no prediction or score exists)",
        "registration_index_sha256": C.sha256_file(idx_path),
        "model": {"id": C.MODEL_ID, "revision": C.MODEL_REVISION, "label": "qwen2.5-coder-3b-instruct", "dtype": "bfloat16",
                  "decoding": "greedy (do_sample=False)", "max_new_tokens": 1024, "seed": 42,
                  "batch_size_first_pass": 8, "batch_size_delta": 1, "batch_size_retry": 1,
                  "runtime": "transformers GPU, one card (CUDA_VISIBLE_DEVICES=0)"},
        "hardware": {"allowed_gpus": ["Tesla T4"], "min_gpu_gib": 14.0, "bf16_emulation_allowed": True},
        "context": ctx,
        "arms": a4["arms"],
        "passes": list(C.PASSES), "pass_dependencies": {k: list(v) for k, v in C.DEPENDS.items()},
        "session_plan": {"1": {"passes": list(C.FIRST_PASSES) + ["delta"], "estimated_t4_hours": 7.6},
                         "2": {"passes": list(C.RETRY_PASSES), "estimated_t4_hours": 2.4}},
        "plans_sha256": {a: C.sha256_text(C.canon(plans[a])) for a in C.PLAN_ARMS},
        "delta_ids": dids,
        "prompt": {"zero_shot_version": "nl-mql-spider1-v1", "rag_version": "nl-mql-spider1-rag-v1",
                   "zero_shot_template_sha256": TEMPLATE_SHA,
                   "zero_shot_prompts_sha256": C.sha256_file(PKG_OUT / "data/eval/prompts.jsonl")},
        "evaluation": {"n_items": len(evals), "ids_sha256": split["holdout"]["ids_sha256"], "db_ids": hdb},
        "corpus": {"path": C.CORPUS_FILE, "sha256": HV.CORPUS_SHA256, "view_n": len(view),
                   "view_ids_sha256": split["holdout_retrieval_view"]["ids_sha256"]},
        "retrieval_exclusions": split["leakage"]["L3_L5_exclusion_pairs"],
        "test_sequence": a4["effective_test_sequence"],
        "scoring_contract": {"atlas_db": HOLDOUT_ATLAS_DB, "mapping_sha256": C.sha256_file(CONTRACT_DIR / "atlas_mapping.json"),
                             "collections": len(mapping["collections"]),
                             "documents": sum(m["documents"] for m in mapping["collections"].values())},
        "budget_drops": {a: sum(bool(r["dropped_budget"]) for r in plans[a]) for a in C.PLAN_ARMS},
    }
    M8.write_json(PKG_OUT / "configs/experiment.json", cfg)
    sys.path.insert(0, str(HERE))
    import make_mql0010_notebook as NB
    NB.build(PKG_OUT / "nl2mql_mql0010_holdout.ipynb")
    man = finalize()
    return cfg, man


def finalize():
    files = sorted(p.relative_to(PKG_OUT).as_posix() for p in PKG_OUT.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and ".pytest_cache" not in p.parts
                   and p.name not in ("package_manifest.json", "SHA256SUMS.txt"))
    cfg = json.loads((PKG_OUT / "configs/experiment.json").read_text(encoding="utf-8"))
    man = {"package": C.PACKAGE, "experiment": C.EXPERIMENT, "status": cfg["status"],
           "files": {n: {"sha256": C.sha256_file(PKG_OUT / n), "bytes": (PKG_OUT / n).stat().st_size} for n in files},
           "retrieval_corpus": {"path": C.CORPUS_FILE, "sha256": HV.CORPUS_SHA256, "records": 3445, "view_records": cfg["corpus"]["view_n"]},
           "config_sha256": C.sha256_file(PKG_OUT / "configs/experiment.json"),
           "registration_index_sha256": cfg["registration_index_sha256"],
           "model": {"id": C.MODEL_ID, "revision": C.MODEL_REVISION}, "contains_gold_labels": False, "contains_credentials": False}
    M8.write_json(PKG_OUT / "package_manifest.json", man)
    names = files + ["package_manifest.json"]
    (PKG_OUT / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(PKG_OUT / n)}  {n}\n" for n in names), encoding="utf-8",
                                            newline="\n")
    tmp = ZIP_OUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names + ["SHA256SUMS.txt"]:
            zi = zipfile.ZipInfo(f"{C.PACKAGE}/{n}", date_time=ZIP_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, (PKG_OUT / n).read_bytes())
    tmp.replace(ZIP_OUT)
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args(argv)
    if a.finalize:
        man = finalize()
        print("finalized", len(man["files"]), "files ->", ZIP_OUT)
        return
    C.need(a.tokenizer_dir, "--tokenizer-dir is required")
    cfg, man = build(Path(a.tokenizer_dir).expanduser(), a.overwrite)
    print(json.dumps({"package": str(PKG_OUT), "zip": str(ZIP_OUT), "zip_sha256": C.sha256_file(ZIP_OUT),
                      "files": len(man["files"]), "eval_items": cfg["evaluation"]["n_items"], "delta_items": len(cfg["delta_ids"]),
                      "budget_drops": cfg["budget_drops"], "scoring_contract": cfg["scoring_contract"]}, indent=1))


if __name__ == "__main__":
    main()
