"""MQL-0010 R3 execution recheck: PREFLIGHT ONLY (no generation, no Atlas execution of predictions).

Secondary evaluation / re-analysis on the existing MQL-0010 holdout (716 questions, 18 databases). Not a fresh test.

Checks (venv_rag; reads only):
  1. frozen MQL-0010 package loads (manifest, registration A4, 2,912-record view, 716 ids, frozen zero-shot prompts);
  2. existing MQL-0010 predictions zip + run_final predictions / scoring hashes against RUN_MANIFEST;
  3. R0 plan (planner_dense lexical "bm25", D0 flags, registered L3-L5 exclusions) == frozen plan_D0, row by row;
  4. R3 plan (lexical "rrf_fused", registered Qdrant collection read-only, pinned embedder, CanonicalScorer);
     no selected example from any holdout database; Qdrant storage byte-identical before/after;
  5. R0 vs R3 prompt partition (shared / differing) -> generation estimate.
Writes outputs/nl2mql_mql0010_r3_execution_recheck/preflight/preflight_report.json (+ the two candidate plans).
Run: ~/codegen_data/venv_rag/Scripts/python -B scripts/nl2mql_mql0010_r3_recheck/preflight_recheck.py
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "kaggle/nl2mql_mql0010_holdout"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(1, str(ROOT / "scripts/nl2mql_mql0011/src"))
sys.path.insert(2, str(ROOT / "scripts/nl2mql_mql0011"))
import mql0010_core as C10      # noqa: E402  (frozen package core)
import mql0011_common as C11    # noqa: E402

OUT = ROOT / "outputs/nl2mql_mql0010_r3_execution_recheck"
PRE = OUT / "preflight"
RUN = ROOT / "outputs/nl2mql_mql0010/scoring_runs/run_final"
PRED_ZIP = ROOT / "mql0010_predictions.zip"
PRED_ZIP_SHA = "d18e7a85"
TOKENIZER_DIR = Path.home() / "codegen_data/qwen3b_tok_488639f1"


def qdrant_snapshot():
    return {p.relative_to(C11.QDRANT_PATH).as_posix(): C10.sha256_file(p) for p in sorted(C11.QDRANT_PATH.rglob("*"))
            if p.is_file() and p.name != ".lock"}


def main():
    import embedder as EM
    import mql0008_core as M8
    import planner_dense as PD
    import qdrant_store as QS
    import vector_store as VS
    from build_embedding_artifact import SPEC, load_artifact
    from transformers import AutoTokenizer
    rep = {"experiment": "MQL-0010 R3 execution recheck (secondary re-analysis of the MQL-0010 holdout)", "at": C10.now()}

    # 1. frozen package
    P = C10.load_package(PKG)
    cfg = P["cfg"]
    rep["package"] = {"manifest_sha256": P["manifest_sha256"], "registration_index_sha256": cfg["registration_index_sha256"],
                      "n_items": len(P["ids"]), "ids_sha256": cfg["evaluation"]["ids_sha256"], "n_dbs": len(P["holdout_dbs"]),
                      "view_n": len(P["view_rows"]), "exclusion_pairs": len(cfg["retrieval_exclusions"]),
                      "model": cfg["model"]["id"] + "@" + cfg["model"]["revision"], "context": cfg["context"]}

    # 2. existing predictions + scores
    man = json.loads((RUN / "scoring/RUN_MANIFEST.json").read_text(encoding="utf-8"))
    pz = C10.sha256_file(PRED_ZIP)
    pred_ok = {a: C10.sha256_file(RUN / "predictions" / a / "predictions.jsonl") == h for a, h in man["predictions_sha256"].items()}
    ana = json.loads((RUN / "scoring/analysis.json").read_text(encoding="utf-8"))
    d0v = C10.read_jsonl(RUN / "predictions/D0V/predictions.jsonl")
    rep["existing_mql0010"] = {"predictions_zip_sha256": pz, "zip_matches_memory_pin": pz.startswith(PRED_ZIP_SHA),
                               "run_final_prediction_hashes_match": pred_ok, "scored_at": man["scored_at"],
                               "atlas_db": man["atlas_db"], "accuracy": ana["accuracy"],
                               "d0v_runtime": sorted({json.dumps(r.get("runtime"), sort_keys=True) for r in d0v}),
                               "d0v_precision": sorted({json.dumps(r.get("precision"), sort_keys=True) for r in d0v})}

    # 3 + 4. plans
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    count_tokens = lambda s: len(tok.encode(s, add_special_tokens=False))   # noqa: E731
    docs = M8.build_docs(P["view_rows"], P["tables"])
    hdb = set(P["holdout_dbs"])
    assert not {d["db_id"] for d in docs} & hdb
    infos = M8.eval_schema_infos(P["atlas"], P["holdout_dbs"])
    excl = cfg["retrieval_exclusions"]
    d0_flags = {k: v for k, v in C10.arm_flags(cfg, "D0").items() if k != "validate_retry"}
    spec, a_ids, _dbs, _shas, vec = load_artifact()
    before = qdrant_snapshot()
    store = QS.QdrantStore.open(spec, path=C11.QDRANT_PATH, exact=True)
    try:
        emb = EM.HFEmbedder(SPEC, threads=1)
        scorer = VS.CanonicalScorer(a_ids, vec, spec.dim)
        # target DBs are holdout DBs; the view holds none of them, and leave-target-out is enforced by the retriever's
        # own universe + EligibilityFilter. forbidden_db_ids must stay empty (the retriever refuses forbidden targets).
        plans = {}
        for arm, lex in (("R0", "bm25"), ("R3", "rrf_fused")):
            plans[arm] = PD.build_plan({**d0_flags, "name": arm, "lexical": lex}, P["evals"], P["prompts"], docs, infos,
                                       P["atlas"], P["tables"], cfg["context"], count_tokens, excl, forbidden_db_ids=(),
                                       store=store, embedder=emb, scorer=scorer)
    finally:
        store.close()
    rep["qdrant_unchanged"] = qdrant_snapshot() == before
    frozen = C10.read_jsonl(PKG / "data/plans/plan_D0.jsonl")
    strip = lambda r: {k: v for k, v in r.items() if k not in ("retrieval_mql0011", "arm")}   # noqa: E731
    r0_diff = [a["id"] for a, b in zip(plans["R0"], frozen) if strip(a) != strip(b)]
    r0_prompt_diff = [a["id"] for a, b in zip(plans["R0"], frozen) if a["prompt_sha256"] != b["prompt_sha256"]]
    sel_db = {arm: sum(d in hdb for r in plans[arm] for d in r["selected_db_ids"]) for arm in plans}
    own_db = {arm: sum(r["db_id"] in r["selected_db_ids"] for r in plans[arm]) for arm in plans}
    ex = {}
    for x in excl:
        ex.setdefault(x["eval_id"], set()).add(x["corpus_id"])
    excl_hits = {arm: sum(bool(set(r["selected_ids"]) & ex.get(r["id"], set())) for r in plans[arm]) for arm in plans}
    shared = [a["id"] for a, b in zip(plans["R0"], plans["R3"]) if a["prompt_sha256"] == b["prompt_sha256"]]
    same_set = sum(set(a["selected_ids"]) == set(b["selected_ids"]) for a, b in zip(plans["R0"], plans["R3"]))
    rep["plans"] = {"R0_equals_frozen_D0_rows": not r0_diff, "R0_rows_differing": r0_diff[:10],
                    "R0_prompts_differing_from_frozen_D0": len(r0_prompt_diff),
                    "selected_from_holdout_dbs": sel_db, "selected_from_own_db": own_db, "selected_excluded_pairs": excl_hits,
                    "budget_drops": {a: sum(bool(r["dropped_budget"]) for r in plans[a]) for a in plans},
                    "n_shared_prompts": len(shared), "n_differing_prompts": len(P["ids"]) - len(shared),
                    "n_same_example_set": same_set,
                    "plan_sha256": {a: C10.sha256_text(C10.canon(plans[a])) for a in plans}}
    C10.write_jsonl(PRE / "plan_R0_candidate.jsonl", plans["R0"])
    C10.write_jsonl(PRE / "plan_R3_candidate.jsonl", plans["R3"])
    C10.write_json(PRE / "preflight_report.json", rep)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
