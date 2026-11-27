"""Build the MQL-0014 Kaggle package (generate only): R0 vs R3 on the MQL-0010 holdout (716 items), both + validator +
one schema retry. SECONDARY re-analysis of existing evaluation data.

Run (venv_rag): ~/codegen_data/venv_rag/Scripts/python -B scripts/nl2mql_mql0010_r3_recheck/build_package.py [--overwrite]

Inputs are the FROZEN MQL-0010 package (loaded through its own fail-closed loader: manifest, registration A4, 2,912-record
view, 716 ids, frozen zero-shot prompts, Atlas-form context) - nothing is re-derived from SQLite. Plans:
  R0 = planner_dense lexical "bm25" with the D0 flags; asserted equal to the frozen MQL-0010 plan_D0 row by row;
  R3 = lexical "rrf_fused" with the registered Qdrant collection (read-only), pinned embedder, CanonicalScorer.
Both with the registered L3-L5 exclusions; assertions: no example from a holdout DB or the question's own DB, no
excluded pair, Qdrant storage byte-identical. Writes kaggle/nl2mql_mql0014_r3_recheck/ (+ zip) and work/ files only.
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
sys.path.insert(1, str(ROOT / "kaggle/nl2mql_mql0010_holdout/src"))
sys.path.insert(2, str(ROOT / "scripts/nl2mql_mql0011/src"))
sys.path.insert(3, str(ROOT / "scripts/nl2mql_mql0011"))
import recheck_common as C      # noqa: E402
import mql0010_core as C10      # noqa: E402  (frozen MQL-0010 package core)
import mql0011_common as C11    # noqa: E402

PKG_SRC_FILES = ("mql0008_core.py", "mql0009_core.py", "patterns.py", "retrieval.py", "rag_block.py", "planner.py",
                 "prompt_v2.py", "recovery.py", "retriever_v2.py", "structure.py", "validator.py")
TOKENIZER_DIR = Path.home() / "codegen_data/qwen3b_tok_488639f1"
ZIP_DATE = (2026, 11, 23, 0, 0, 0)
need = C.need


def qdrant_snapshot():
    return {p.relative_to(C11.QDRANT_PATH).as_posix(): C.sha256_file(p) for p in sorted(C11.QDRANT_PATH.rglob("*"))
            if p.is_file() and p.name != ".lock"}


def verify_registration():
    need(C.REGISTRATION.exists(), "registration missing: run register.py first")
    reg = json.loads(C.REGISTRATION.read_text(encoding="utf-8"))
    for rel, h in reg["source_sha256"].items():
        need(C.sha256_file(ROOT / rel) == h, f"registered source changed after registration: {rel}")
    return reg


def build_plans(P, count_tokens):
    import embedder as EM
    import mql0008_core as M8
    import planner_dense as PD
    import qdrant_store as QS
    import vector_store as VS
    from build_embedding_artifact import SPEC, load_artifact
    cfg = P["cfg"]
    docs = M8.build_docs(P["view_rows"], P["tables"])
    hdb = set(P["holdout_dbs"])
    need(not {d["db_id"] for d in docs} & hdb, "retrieval docs contain a holdout DB")
    infos = M8.eval_schema_infos(P["atlas"], P["holdout_dbs"])
    excl = cfg["retrieval_exclusions"]
    d0 = {k: v for k, v in C10.arm_flags(cfg, "D0").items() if k != "validate_retry"}
    spec, a_ids, _d, _s, vec = load_artifact()
    before = qdrant_snapshot()
    store = QS.QdrantStore.open(spec, path=C11.QDRANT_PATH, exact=True)
    try:
        emb = EM.HFEmbedder(SPEC, threads=1)
        scorer = VS.CanonicalScorer(a_ids, vec, spec.dim)
        # forbidden_db_ids stays empty: targets ARE holdout DBs (the retriever refuses forbidden targets); the view
        # holds no holdout DB and leave-target-out is enforced by the retriever's universe + EligibilityFilter.
        plans = {arm: PD.build_plan({**d0, "name": arm, "lexical": C.LEXICAL[arm]}, P["evals"], P["prompts"], docs, infos,
                                    P["atlas"], P["tables"], cfg["context"], count_tokens, excl, forbidden_db_ids=(),
                                    store=store, embedder=emb, scorer=scorer) for arm in C.PLAN_ARMS}
    finally:
        store.close()
    need(qdrant_snapshot() == before, "Qdrant storage changed")
    frozen = C10.read_jsonl(C.M10_PKG / "data/plans/plan_D0.jsonl")
    strip = lambda r: {k: v for k, v in r.items() if k not in ("retrieval_mql0011", "arm")}   # noqa: E731
    need(len(frozen) == len(plans["R0"]) and all(strip(a) == strip(b) for a, b in zip(plans["R0"], frozen)),
         "R0 plan differs from the frozen MQL-0010 plan_D0")
    ex = {}
    for x in excl:
        ex.setdefault(x["eval_id"], set()).add(x["corpus_id"])
    for arm in C.PLAN_ARMS:
        for r in plans[arm]:
            need(not set(r["selected_db_ids"]) & hdb, f"{arm}/{r['id']}: example from a holdout DB")
            need(r["db_id"] not in r["selected_db_ids"], f"{arm}/{r['id']}: example from its own DB")
            need(not set(r["selected_ids"]) & ex.get(r["id"], set()), f"{arm}/{r['id']}: excluded pair selected")
    return docs, plans


def build(overwrite=False):
    from transformers import AutoTokenizer
    import planner as PL
    reg = verify_registration()
    P = C10.load_package(C.M10_PKG)
    need(P["manifest_sha256"] == reg["mql0010_package_manifest_sha256"], "MQL-0010 package differs from the registration")
    need(P["cfg"]["evaluation"]["ids_sha256"] == reg["evaluation"]["ids_sha256"], "holdout ids")
    if C.PKG_OUT.exists() or C.ZIP_OUT.exists():
        need(overwrite, f"{C.PKG_OUT} or {C.ZIP_OUT} exists; pass --overwrite")
        need(C.PKG_OUT.name == C.PACKAGE, "unexpected package path")
        shutil.rmtree(C.PKG_OUT, ignore_errors=True)
        C.ZIP_OUT.unlink(missing_ok=True)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    docs, plans = build_plans(P, lambda s: len(tok.encode(s, add_special_tokens=False)))
    by_doc = {d["doc_id"]: d for d in docs}
    fps = {p["id"]: p for p in P["prompts"]}
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
    ids = P["ids"]
    shared = [i for i, a, b in zip(ids, arm_prompts["R0"], arm_prompts["R3"]) if a["prompt_sha256"] == b["prompt_sha256"]]
    pkg = C.PKG_OUT
    for d in ("src", "configs", "data/eval", "data/context", "data/plans", "data/prompts", "registration", "provenance"):
        (pkg / d).mkdir(parents=True, exist_ok=True)
    for n in PKG_SRC_FILES:
        shutil.copyfile(C.M10_PKG / "src" / n, pkg / "src" / n)
    shutil.copyfile(HERE / "pkg_src/mql0014_core.py", pkg / "src/mql0014_core.py")
    shutil.copyfile(C.REGISTRATION, pkg / "registration" / C.REGISTRATION.name)
    C.write_jsonl(pkg / "data/eval/eval_inputs.jsonl", P["evals"])
    shutil.copyfile(C.M10_PKG / "data/context/atlas_collections.json", pkg / "data/context/atlas_collections.json")
    for arm in C.PLAN_ARMS:
        C.write_jsonl(pkg / f"data/plans/plan_{arm}.jsonl", plans[arm])
        C.write_jsonl(pkg / f"data/prompts/prompts_{arm}.jsonl", arm_prompts[arm])
    for f in ("build_package.py", "make_notebook.py"):
        shutil.copyfile(HERE / f, pkg / "provenance" / f)
    canon = lambda o: json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))   # noqa: E731
    m = P["cfg"]["model"]
    cfg = {
        "experiment_id": C.EXPERIMENT, "package": C.PACKAGE,
        "status": "PACKAGE BUILT - NOT EXECUTED (no prediction or score exists)",
        "nature": "SECONDARY re-analysis of the MQL-0010 holdout (716); not a fresh independent test",
        "registration_sha256": C.sha256_file(C.REGISTRATION),
        "model": {"id": m["id"], "revision": m["revision"], "label": m["label"], "dtype": "bfloat16",
                  "decoding": "greedy (do_sample=False)", "max_new_tokens": 1024, "seed": 42, "batch_size_first_pass": 8,
                  "batch_size_retry": 1, "runtime": "transformers GPU, one card (CUDA_VISIBLE_DEVICES=0)"},
        "hardware": P["cfg"]["hardware"],
        "context": P["cfg"]["context"],
        "arms": {"R0": {"lexical": "bm25", "flags": "MQL-0010 D0"}, "R3": {"lexical": "rrf_fused", "flags": "MQL-0010 D0"},
                 "R0V": {"source": "R0", "feedback_mode": "schema"}, "R3V": {"source": "R3", "feedback_mode": "schema"}},
        "plans_sha256": {a: C.sha256_text(canon(plans[a])) for a in C.PLAN_ARMS},
        "prompts_sha256": {a: C.sha256_file(pkg / f"data/prompts/prompts_{a}.jsonl") for a in C.PLAN_ARMS},
        "partition": {"shared_ids": shared, "n_shared": len(shared), "n_diff": len(ids) - len(shared)},
        "evaluation": {"source": "MQL-0010 holdout (frozen package nl2mql_mql0010_holdout)", "n_items": len(ids),
                       "ids_sha256": P["cfg"]["evaluation"]["ids_sha256"], "db_ids": P["holdout_dbs"]},
        "retrieval": {"view_n": len(P["view_rows"]), "view_ids_sha256": C11.VIEW_IDS_SHA256, "corpus_sha256": C11.CORPUS_SHA256,
                      "qdrant_index": C11.INDEX_NAME, "embedder_revision": C11.MODEL_REVISION,
                      "exclusions": P["cfg"]["retrieval_exclusions"], "R0_equals_frozen_mql0010_plan_D0": True},
        "mql0010_package_manifest_sha256": P["manifest_sha256"],
        "scoring_contract": {"atlas_db": C.ATLAS_DB, "mapping_sha256": C.sha256_file(C.M10_CONTRACT)},
        "budget_drops": {a: sum(bool(r["dropped_budget"]) for r in plans[a]) for a in C.PLAN_ARMS},
    }
    C.write_json(pkg / "configs/experiment.json", cfg)
    sys.path.insert(0, str(HERE))
    import make_notebook as NB
    NB.build(pkg / f"{C.PACKAGE}.ipynb")
    man = finalize()
    C.write_json(C.WORK / "build_report.json", {
        "package_manifest_sha256": C.sha256_file(pkg / "package_manifest.json"), "zip_sha256": C.sha256_file(C.ZIP_OUT),
        "n_items": len(ids), "n_shared": len(shared), "budget_drops": cfg["budget_drops"], "plans_sha256": cfg["plans_sha256"],
        "qdrant_unchanged": True, "R0_equals_frozen_mql0010_plan_D0": True})
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
    a = ap.parse_args(argv)
    cfg, man = build(a.overwrite)
    print(json.dumps({"zip": str(C.ZIP_OUT), "zip_sha256": C.sha256_file(C.ZIP_OUT), "files": len(man["files"]),
                      "n_items": cfg["evaluation"]["n_items"], "n_shared": cfg["partition"]["n_shared"],
                      "budget_drops": cfg["budget_drops"]}, indent=1))


if __name__ == "__main__":
    main()
