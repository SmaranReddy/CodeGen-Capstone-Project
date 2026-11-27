"""Write the MQL-0010 registration: analysis plan (outputs/nl2mql_mql0010/preflight/analysis_plan.json) and the
experiment config (scripts/nl2mql_mql0010/configs/mql0010_experiment.json) that pins the plan, split and corpus
hashes. Refuses to overwrite an existing registration (registrations are immutable; amend with a new dated file).
No prediction exists when this runs.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
PRE = ROOT / "outputs/nl2mql_mql0010/preflight"
CFG = ROOT / "scripts/nl2mql_mql0010/configs/mql0010_experiment.json"
PLAN = PRE / "analysis_plan.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ANALYSIS_PLAN = {
    "experiment": "MQL-0010",
    "registered_at": "2026-11-19",
    "evaluation_set": "holdout (split_registration.json, amendment A1): 18 corpus DBs drawn by seed 20261119 before any inspection; items = spider-sql-v1 train exec_ok, non-degenerate gold, spider-pool-v3 members",
    "metric": "cross-engine execution accuracy on Atlas (registered MQL-0006..0009 evaluator, read-only user, full-parity preflight); correct / N, nothing excluded in the primary analysis",
    "arms": ["A0", "A0V", "D0", "D0V", "E", "EF", "FULL"],
    "primary": {
        "H1": {"comparison": "FULL vs D0", "test": "exact two-sided McNemar on per-question correctness", "alpha": 0.05,
               "effect": "paired difference in accuracy with question-level bootstrap 95% CI (10,000 resamples, seed 20261119)"}},
    "secondary_gatekept": {
        "rule": "tested only if H1 rejects; Holm over H2-H4 at family-wise alpha 0.05; each exact two-sided McNemar + bootstrap CI",
        "H2": "D0V vs D0  (controlled recovery alone; discordance can only come from kept retries)",
        "H3": "E vs D0    (signature-diversified selection alone; expected to be UNDERPOWERED - MDE ~4 pp)",
        "H4": "EF vs E    (type-incompatible join-key notes; prompts differ only on DBs with string/number FK pairs; UNDERPOWERED, descriptive in practice)"},
    "descriptive_no_claim": [
        "A0 vs D0 (replication of the MQL-0008/0009 RAG effect on new databases)",
        "A0V vs A0 (recovery without retrieval)",
        "recovery statistics per retry arm: retried, kept, retry with zero issues, retry that changed outcome (fix / break)",
        "outcome taxonomy per arm: correct / incorrect (wrong_row_count empty vs non-empty, wrong_column_count, wrong_values) / invalid / execution_error / timeout",
        "query family (gold-SQL labels, retrospective) and per-database accuracy",
        "retrieval traces: distinct signatures shown, rank of the kept examples, score plateaus"],
    "sensitivity": ["S1 reference_status ok only", "S2 without the L3-L5 exclusion items", "S3 without L6 (identical gold SQL in the view) items"],
    "power_note": "n = 716; with ~16% discordant pairs (MQL-0009 D vs C) the minimum detectable difference at alpha 0.05 / power 0.8 is ~4.2 pp; for D0V vs D0 discordance is bounded by the retried share (~20%) and the MDE is ~3 pp if about half the retries change the outcome",
    "decision_rules": {
        "adopt_V": "H1 and H2 reject",
        "adopt_E": "H1 and H3 reject; otherwise keep the MQL-0008 ranking order (E is reported as not shown)",
        "F": "reported descriptively; no adoption claim unless H4 rejects",
        "never": ["re-score or re-generate after seeing results", "change arms, thresholds or the holdout after registration",
                  "use the MQL-0009 639 or the Spider-dev 892 for any MQL-0010 decision", "pool random-seed outcomes as independent items"]},
    "integrity": ["all arms scored in ONE session against ONE full-parity Atlas preflight",
                  "prediction files hashed at export; derived arms record their source arm and reuse is verified by hash",
                  "Atlas tie-order nondeterminism: a single scoring pass; tie-dependent items reported"],
}

ARMS = [
    {"name": "A0", "kind": "generate", "label": "zero-shot (nl-mql-spider1-v1), unchanged", "flags": {"family": "zero_shot"}},
    {"name": "A0V", "kind": "retry", "source": "A0", "label": "A0 + controlled recovery (V)", "flags": {"family": "zero_shot", "validate_retry": True}},
    {"name": "D0", "kind": "generate", "label": "MQL-0008/0009 hybrid RAG, unchanged (baseline)", "flags": {"family": "hybrid"}},
    {"name": "D0V", "kind": "retry", "source": "D0", "label": "D0 + controlled recovery (V)", "flags": {"family": "hybrid", "validate_retry": True}},
    {"name": "E", "kind": "generate", "label": "D0 ranking + signature-diversified selection (E)", "flags": {"family": "hybrid", "diversify": "signature", "candidate_pool": 50}},
    {"name": "EF", "kind": "prompt_delta", "base": "E", "label": "E + type-incompatible join-key notes (F)",
     "flags": {"family": "hybrid", "diversify": "signature", "candidate_pool": 50, "fk_type_notes": "numeric_unified"}},
    {"name": "FULL", "kind": "retry", "source": "EF", "label": "E + F + V (improved architecture)",
     "flags": {"family": "hybrid", "diversify": "signature", "candidate_pool": 50, "fk_type_notes": "numeric_unified", "validate_retry": True}},
]


def main():
    if PLAN.exists() or CFG.exists():
        raise SystemExit("registration already exists; registrations are immutable (write a dated amendment instead)")
    split = PRE / "split_registration.json"
    reg = json.loads(split.read_text(encoding="utf-8"))
    PRE.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(ANALYSIS_PLAN, indent=1), encoding="utf-8", newline="\n")
    src = ROOT / "scripts/nl2mql_mql0010/src"
    cfg = {
        "experiment_id": "MQL-0010",
        "title": "MQL-0010 - controlled recovery, diversified selection and join-key notes on top of the MQL-0008 hybrid RAG",
        "status": "REGISTERED - NOT EXECUTED (no prediction, no score). Prerequisites in docs/mql0010_experiment_plan.md section 9",
        "registered_at": "2026-11-19",
        "model": {"id": "Qwen/Qwen2.5-Coder-3B-Instruct", "revision": "488639f1ff808d1d3d0ba301aef8c11461451ec5", "dtype": "bfloat16",
                  "decoding": "greedy (do_sample=False)", "max_new_tokens": 1024, "batch_size": 8, "seed": 42,
                  "gpu": "Tesla T4 only (bf16 emulated, as MQL-0006..0009)", "retry_max_new_tokens": 1024},
        "prompt": {"zero_shot_version": "nl-mql-spider1-v1", "rag_version": "nl-mql-spider1-rag-v1",
                   "rag_rule": "reference block inserted before the final 'Question:' line by the unchanged rag_block.insert_block",
                   "recovery": "one extra turn: [user: first prompt, assistant: first answer, user: validator feedback + RETRY_SUFFIX], same system prompt"},
        "corpus": {"path": "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl",
                   "sha256": "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf", "modified": False,
                   "holdout_view": reg["holdout_retrieval_view"]},
        "split": {"path": "outputs/nl2mql_mql0010/preflight/split_registration.json", "sha256": sha(split),
                  "holdout_n": reg["holdout"]["n_items"], "holdout_ids_sha256": reg["holdout"]["ids_sha256"],
                  "dev_n": reg["dev"]["n_items"], "dev_ids_sha256": reg["dev"]["ids_sha256"],
                  "retrieval_exclusions": reg["leakage"]["L3_L5_exclusion_pairs"]},
        "context": {"k": 3, "max_context_tokens": 1500, "n_ctx": 8192, "max_new_tokens": 1024, "margin_tokens": 64},
        "retrieval": {"base": "retrieval.Retriever(method='hybrid_masked', hybrid_weight=1.0), unchanged (BM25 k1 1.5, b 0.75)",
                      "diversify_candidate_pool": 50},
        "arms": ARMS,
        "not_adopted": {
            "structure_scoring_nb": "dev proxy hit@3 45.5 vs baseline 52.1 alone; +0.7 pt over baseline with diversity (outputs/nl2mql_mql0010/dev_evidence/retrieval_dev_proxy.json)",
            "lexical_tie_break": "no change on the dev proxy",
            "id_postprocessor": "offline counterfactual on MQL-0009: -10 to -17 correct per arm (outputs/nl2mql_mql0010/mql0009_post_analysis/offline_counterfactuals.json)",
            "unsupported_family_gate": "ceiling ~1 item on MQL-0009 (set-op family A 12.1% vs D 11.0%); NB gate precision 0.77 / recall 0.35 on train CV",
            "corpus_set_op_expansion": "proposal only (docs/mql0010_rag_architecture_design.md section 8)"},
        "analysis_plan_sha256": sha(PLAN),
        "source_sha256": {f.name: sha(f) for f in sorted(src.glob("*.py"))},
        "compute_estimate_t4_hours": {"A0": 2.0, "D0": 2.6, "E": 2.6, "EF_delta": 0.3, "retries_3_arms": 2.0, "total": 9.5,
                                      "basis": "MQL-0009 run configs: 9.9-12.8 s/item; ~20% of items retried (MQL-0009 validator flag rate)"},
    }
    CFG.parent.mkdir(parents=True, exist_ok=True)
    CFG.write_text(json.dumps(cfg, indent=1), encoding="utf-8", newline="\n")
    print("analysis plan", sha(PLAN))
    print("config", sha(CFG))


if __name__ == "__main__":
    main()
