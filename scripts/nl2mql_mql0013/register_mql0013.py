"""Write the MQL-0013 registration (write-once; before the package is built and before any generation or scoring).

Run: python -B scripts/nl2mql_mql0013/register_mql0013.py
"""
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
import mql0013_common as C     # noqa: E402
import mql0013_analysis as AN  # noqa: E402

SOURCES = [
    "scripts/nl2mql_mql0013/src/mql0013_common.py", "scripts/nl2mql_mql0013/src/mql0013_analysis.py",
    "scripts/nl2mql_mql0013/pkg_src/mql0013_core.py", "scripts/nl2mql_mql0013/build_mql0013_package.py",
    "scripts/nl2mql_mql0013/make_mql0013_notebook.py", "scripts/nl2mql_mql0013/mql0013_scoring.py",
    "scripts/nl2mql_mql0013/provision_test_atlas.py",
    # reused, unchanged
    "scripts/nl2mql/dev1034_core.py", "scripts/nl2mql_mql0011/src/retriever_dense.py",
    "scripts/nl2mql_mql0011/src/planner_dense.py", "scripts/nl2mql_mql0011/src/vector_store.py",
    "scripts/nl2mql_mql0011/src/qdrant_store.py", "scripts/nl2mql_mql0011/src/embedder.py",
    "scripts/nl2mql_mql0010/src/planner.py", "scripts/nl2mql_mql0010/src/recovery.py",
    "scripts/nl2mql_mql0010/src/validator.py", "scripts/nl2mql_mql0010/src/prompt_v2.py",
    "scripts/nl2mql_mql0009/provision_heldout_atlas.py",
]


def main():
    if C.REGISTRATION.exists():
        raise SystemExit(f"{C.REGISTRATION} exists; the registration is write-once")
    recs = C.read_jsonl(C.TEST_FULL)
    ids = [r["id"] for r in recs]
    reg = {
        "experiment": C.EXPERIMENT,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "REGISTERED before package build, generation and scoring",
        "objective": "Measure strict Atlas execution accuracy of R3 (embedding-enhanced retrieval) against R0 (BM25 hybrid) "
                     "under an otherwise identical pipeline; user target >= 60% for R3V (descriptive, not assumed).",
        "user_authorisations_2026_11_22": ["benchmark = Spider test_full (consumes the ledger's 'evaluated once, at the end' "
                                           "test set for this NL->MQL comparison)",
                                           "provision ONE new Atlas database codegen_spider1_test_mql0013 (insert-only)"],
        "evaluation": {"source": "data/processed/spider_sql_v1/test_full.jsonl", "source_sha256": C.sha256_file(C.TEST_FULL),
                       "n_items": len(ids), "ids_sha256": C.ids_sha256(ids), "n_databases": len({r["db_id"] for r in recs}),
                       "filtering": "none (all 2,147; every gold query executes, tier exec_ok); no post-hoc exclusion",
                       "disjointness": "asserted at build: no database or id overlap with the approved corpus, MQL-0010 "
                                       "holdout/dev/build DBs, X0, spider-sql-v1 train/val_internal/dev_full, expanded "
                                       "split manifest (database level; the 65-item test file is never opened)"},
        "arms": list(AN.ARMS),
        "arm_definitions": {
            "R0": "MQL-0010 D0 = planner_dense lexical 'bm25' (BM25 over schema-masked question + regex-cue Jaccard), K=3, "
                  "diversify none, 1,500-token example budget; first answer",
            "R3": "identical except lexical 'rrf_fused' (MQL-0011 R3: RRF_60(BM25 rank) + RRF_60(dense rank), max-normalised, "
                  "+ Jaccard); bge-small-en-v1.5 @ 5c38ec7c, registered Qdrant collection (read-only, exact), CanonicalScorer",
            "R0V": "R0 + MQL-0010 component V (static schema validator; one schema-feedback retry; keep retry iff strictly fewer "
                   "issues)", "R3V": "R3 + the same component V"},
        "held_constant": ["Qwen/Qwen2.5-Coder-3B-Instruct @ 488639f1, bf16 on Tesla T4, greedy, max_new_tokens 1024",
                          "zero-shot base prompt nl-mql-spider1-v1 (template sha 7401bf14...) + RAG block nl-mql-spider1-rag-v1",
                          "retrieval view = the 2,912-record MQL-0011 view (the Qdrant collection's records) for BOTH arms",
                          "L3-L5 question-level leakage exclusions applied identically to both arms",
                          "validator, retry message, acceptance rule", "evaluator dev1034_core (unchanged), read-only Atlas user"],
        "generation_design": {"first_pass_batch": 8, "retry_batch": 1,
                              "shared_items": "items whose R0 and R3 prompts are byte-identical are generated once and the "
                                              "answer is used by both arms",
                              "diff_items": "R0-only and R3-only prompts are generated in separate passes with the same "
                                            "batch size and item order",
                              "retry": "each distinct retry prompt generated once at batch 1"},
        "scoring": {"metric": "strict execution accuracy on Atlas, cross-engine against SQLite gold (exec_match via "
                              "dev1034_core.evaluate_prediction); denominator 2,147 for every arm; anything not 'correct' "
                              "is wrong (incorrect, invalid, execution_failure, timeout, generation_failure, not_comparable)",
                    "dedup": "each distinct (item, raw_text) is executed once; arms with byte-identical text share the outcome",
                    "exec_timeout_ms": 30000, "indexes": "none created (as in every earlier evaluation database)"},
        "primary_hypothesis": {"comparison": "R3V vs R0V", "test": "exact two-sided McNemar", "alpha": AN.ALPHA,
                               "ci": f"paired bootstrap 95%, {AN.BOOT_REPS} resamples, seed {AN.BOOT_SEED}",
                               "multiplicity": "single primary hypothesis; everything else descriptive"},
        "descriptive": {"comparisons": [list(x) for x in AN.DESCRIPTIVE],
                        "target": f"R3V >= ceil(0.60 x 2147) = 1289 correct: reported yes/no, not a test",
                        "categories": ["Spider hardness (easy/medium/hard/extra)",
                                       "gold-SQL structure labels (structure.signature(patterns.sql_patterns(gold sql))), "
                                       "computed after scoring for reporting only", "reference_status"],
                        "taxonomy": list(AN.TAXONOMY)},
        "known_risks": ["aan_1 holds 243k rows (Citation 124,842); unindexed $lookup may time out (counted as timeout)",
                        "bf16 on T4 is emulated; batch composition can change greedy outputs (applies equally to both arms)",
                        "test_full is consumed by this experiment for NL->MQL"],
        "source_sha256": {p: C.sha256_file(ROOT / p) for p in SOURCES},
    }
    C.write_json(C.REGISTRATION, reg)
    print("registered:", C.REGISTRATION, C.sha256_file(C.REGISTRATION))


if __name__ == "__main__":
    main()
