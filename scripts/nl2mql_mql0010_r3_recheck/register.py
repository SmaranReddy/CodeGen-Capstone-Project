"""Write the MQL-0014 registration (write-once; before the package build, generation and scoring).
Run: python -B scripts/nl2mql_mql0010_r3_recheck/register.py
"""
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
import recheck_common as C     # noqa: E402
import recheck_analysis as AN  # noqa: E402

SOURCES = [
    "scripts/nl2mql_mql0010_r3_recheck/src/recheck_common.py", "scripts/nl2mql_mql0010_r3_recheck/src/recheck_analysis.py",
    "scripts/nl2mql_mql0010_r3_recheck/pkg_src/mql0014_core.py", "scripts/nl2mql_mql0010_r3_recheck/build_package.py",
    "scripts/nl2mql_mql0010_r3_recheck/make_notebook.py", "scripts/nl2mql_mql0010_r3_recheck/recheck_scoring.py",
    "scripts/nl2mql/dev1034_core.py", "scripts/nl2mql_mql0011/src/retriever_dense.py",
    "scripts/nl2mql_mql0011/src/planner_dense.py", "scripts/nl2mql_mql0011/src/vector_store.py",
    "scripts/nl2mql_mql0011/src/qdrant_store.py", "scripts/nl2mql_mql0011/src/embedder.py",
    "kaggle/nl2mql_mql0010_holdout/src/planner.py", "kaggle/nl2mql_mql0010_holdout/src/recovery.py",
    "kaggle/nl2mql_mql0010_holdout/src/validator.py", "kaggle/nl2mql_mql0010_holdout/src/prompt_v2.py",
]


def main():
    if C.REGISTRATION.exists():
        raise SystemExit(f"{C.REGISTRATION} exists; the registration is write-once")
    cfg10 = json.loads((C.M10_PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    pre = json.loads((C.OUT / "preflight/preflight_report.json").read_text(encoding="utf-8"))
    reg = {
        "experiment": C.EXPERIMENT,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "REGISTERED before package build, generation and scoring (user approved option B on 2026-11-23)",
        "nature": "SECONDARY evaluation / re-analysis of the existing MQL-0010 holdout. NOT a fresh independent test: "
                  "MQL-0010 outcomes on these 716 items are known (D0V 430/716) and the D0V recipe was selected on them. R3 "
                  "retrieval was developed on the MQL-0011 dev split / screened on X0 and never tuned on these items.",
        "conflicts_disclosed": [
            "MQL-0011 stage-1 gate failed (+2.65 < +3 pp) and registered 'no stage 2'; this is a separate, user-requested "
            "analysis and does not retroactively pass that gate",
            "MQL-0013 (Spider test_full) registration/package exist, unexecuted, superseded by this request; test_full unused",
        ],
        "evaluation": {"source": "kaggle/nl2mql_mql0010_holdout (frozen)", "n_items": cfg10["evaluation"]["n_items"],
                       "ids_sha256": cfg10["evaluation"]["ids_sha256"], "db_ids": cfg10["evaluation"]["db_ids"],
                       "filtering": "none; all 716 in every arm", "gold": "spider-sql-v1 train records; reference must "
                       "equal the frozen MQL-0010 reference sha " + C.M10_REFERENCE_SHA},
        "mql0010_package_manifest_sha256": pre["package"]["manifest_sha256"],
        "arms": list(AN.ARMS),
        "arm_definitions": {
            "R0": "planner_dense lexical 'bm25' with MQL-0010 D0 flags (== frozen MQL-0010 plan_D0, 716/716 rows)",
            "R3": "same, lexical 'rrf_fused' (MQL-0011 R3; registered Qdrant collection read-only, bge-small @ 5c38ec7c)",
            "R0V": "R0 + MQL-0010 component V (validator, one schema-feedback retry, keep iff strictly fewer issues)",
            "R3V": "R3 + the same component V"},
        "held_constant": ["Qwen/Qwen2.5-Coder-3B-Instruct @ 488639f1, bf16 Tesla T4, greedy, max_new_tokens 1024",
                          "frozen MQL-0010 zero-shot prompts + schema context; RAG block nl-mql-spider1-rag-v1; K=3, 1,500-token budget",
                          "2,912-record view; 13 registered L3-L5 exclusions in both arms", "validator / retry / acceptance rule",
                          "evaluator dev1034_core, read-only Atlas user, MQL-0010 contract on codegen_spider1_holdout_mql0010"],
        "generation_design": "regenerate BOTH arms in one Kaggle session: identical-prompt items once (shared), R0-only and "
                             "R3-only prompts in separate passes with the same batch size (8) and order, each distinct retry "
                             "prompt once at batch 1",
        "scoring": {"metric": "execution accuracy = correct / 716 (strict; incorrect, invalid, execution_failure, timeout, "
                              "generation_failure, not_comparable all count as wrong; infrastructure failures are retried "
                              "by the evaluator and reported separately if any remain)",
                    "dedup": "each distinct (item, raw_text) executed once; identical texts share the outcome",
                    "reproducibility_arm": "original MQL-0010 D0V predictions (sha " + C.M10_D0V_SHA[:12] + ") rescored "
                                           "in the same session; descriptive only"},
        "primary_hypothesis": {"comparison": "R3V vs R0V", "test": "exact two-sided McNemar", "alpha": AN.ALPHA,
                               "ci": f"paired bootstrap 95%, {AN.BOOT_REPS} resamples, seed {AN.BOOT_SEED}",
                               "reported": "both correct, both wrong, R3V-only, R0V-only, difference pp"},
        "descriptive": {"comparisons": [list(x) for x in AN.DESCRIPTIVE], "target_60pct": "R3V >= 430 correct (ceil 0.6*716)",
                        "breakdowns": ["Spider hardness", "join requirement", "nesting / set operation", "gold-SQL structure label",
                                       "reference status"], "taxonomy": list(AN.TAXONOMY),
                        "power_note": "with ~100 discordant pairs a true +2 pp gain has well under 50% power; a null is "
                                      "inconclusive"},
        "source_sha256": {p: C.sha256_file(ROOT / p) for p in SOURCES},
    }
    C.write_json(C.REGISTRATION, reg)
    print("registered:", C.REGISTRATION, C.sha256_file(C.REGISTRATION))


if __name__ == "__main__":
    main()
