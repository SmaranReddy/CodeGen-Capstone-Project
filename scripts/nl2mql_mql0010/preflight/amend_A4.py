"""Write amendment A4 (audit corrections R1-R5) and the registration index (R1). Write-once: refuses to overwrite.
No MQL-0010 prediction exists when this runs. The original config and analysis plan are NOT modified; the index
records which document governs which aspect, and pins every hash.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
PRE = ROOT / "outputs/nl2mql_mql0010/preflight"
A4 = PRE / "amendment_A4_2026-11-19.json"
INDEX = PRE / "registration_index.json"
SRC = ROOT / "scripts/nl2mql_mql0010/src"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return str(Path(p).relative_to(ROOT)).replace("\\", "/")


ARMS = [
    {"name": "A0", "kind": "generate", "flags": {"family": "zero_shot"}, "batch_size": 8},
    {"name": "A0V", "kind": "retry", "source": "A0", "feedback_mode": "schema", "flags": {"family": "zero_shot", "validate_retry": True}, "batch_size": 1},
    {"name": "D0", "kind": "generate", "flags": {"family": "hybrid"}, "batch_size": 8},
    {"name": "D0V", "kind": "retry", "source": "D0", "feedback_mode": "schema", "flags": {"family": "hybrid", "validate_retry": True}, "batch_size": 1},
    {"name": "D0G", "kind": "retry", "source": "D0", "feedback_mode": "generic", "flags": {"family": "hybrid", "validate_retry": True}, "batch_size": 1,
     "label": "R5 generic-retry control: same trigger (validator flag or unparseable), same items, same acceptance rule (strictly fewer issues), same model/decoding/retry budget (1); second turn = GENERIC_FEEDBACK, no schema-specific content"},
    {"name": "E", "kind": "generate", "flags": {"family": "hybrid", "diversify": "signature", "candidate_pool": 50}, "batch_size": 8},
    {"name": "EP", "kind": "matched_delta_control", "base": "E", "flags": {"family": "hybrid", "diversify": "signature", "candidate_pool": 50}, "batch_size": 1,
     "label": "R2 control: E's prompt regenerated for exactly the items whose EF prompt differs, in the same delta pass, batch size 1; all other items reuse E"},
    {"name": "EF", "kind": "matched_delta", "base": "E", "control": "EP",
     "flags": {"family": "hybrid", "diversify": "signature", "candidate_pool": 50, "fk_type_notes": "numeric_unified"}, "batch_size": 1},
    {"name": "FULL", "kind": "retry", "source": "EF", "feedback_mode": "schema",
     "flags": {"family": "hybrid", "diversify": "signature", "candidate_pool": 50, "fk_type_notes": "numeric_unified", "validate_retry": True}, "batch_size": 1},
]

AMENDMENT = {
    "experiment": "MQL-0010", "amendment": "A4", "date": "2026-11-19", "written_before_any_prediction": True,
    "origin": "docs/mql0010_pre_execution_audit.md corrections R1-R5, authorised by the user on 2026-11-19",
    "R1_registration_index": "outputs/nl2mql_mql0010/preflight/registration_index.json pins every registration document and source hash and states which document governs which aspect; tests/test_mql0010_registration.py enforces it",
    "R2_matched_type_note_comparison": {
        "change": "new control arm EP; EF becomes kind matched_delta; the type-note comparison is EF vs EP (was EF vs E)",
        "procedure": "for each item whose EF prompt hash differs from E's plan, generate E's prompt (EP) and EF's prompt in the same delta pass, one prompt per call (batch size 1), same model, decoding and max_new_tokens; all other items reuse E's record in both EP and EF",
        "implementation": "driver.matched_delta; verified by scoring.verify_exports"},
    "R3_holdout_view_enforcement": {
        "rule": "every retrieval index for holdout inference is built from holdout_view.load_view (corpus hash, registered view id hash and count, no holdout DB) and re-checked by holdout_view.assert_docs_exclude; planner.build_plan is called with forbidden_db_ids = the 18 holdout DBs and aborts on any such document",
        "implementation": "src/holdout_view.py; planner.build_plan(forbidden_db_ids); tests/test_mql0010_registration.py; preflight/dry_run.py"},
    "R4_power_correction_and_practical_criterion": {
        "correction": "the plan's 'MDE ~3 pp for D0V vs D0' is withdrawn. Audited estimate: with ~21% of items flagged and near-zero breaks, exact McNemar rejects at about +1 pp (power 0.79 at a 5% repair rate, mean effect +1.1 pp; power 1.00 at >=10%, +2.2 pp). Significance alone is therefore a low bar.",
        "criterion": "step 1 (D0V vs D0) is called PRACTICALLY MEANINGFUL only if it rejects AND the question-level bootstrap 95% CI lower bound of (D0V - D0) is >= +1.0 percentage point. The test itself is unchanged. Regressions caused by kept retries are reported explicitly."},
    "R5_generic_retry_control": {
        "arm": "D0G",
        "generic_feedback_text": "Your previous query may contain mistakes. Review it and return a corrected query.",
        "identical_to_D0V": ["first answers (D0's)", "trigger (unparseable or >=1 validator issue)", "items retried (verified)",
                             "acceptance rule (retry kept only if strictly fewer issues)", "retry budget (1)", "system prompt",
                             "model, revision, dtype, greedy decoding, max_new_tokens 1024", "batch size 1", "unparseable first answers receive the same FORMAT_FEEDBACK in both arms"],
        "differs": "only the content of the second user turn for parseable flagged answers (schema issue list vs generic sentence)",
        "estimand": "effect of schema-specific feedback content, given the validator trigger and acceptance rule"},
    "effective_test_sequence": {
        "alpha": 0.05, "procedure": "serial gatekeeping",
        "step1": "D0V vs D0 (exact two-sided McNemar) + R4 practical criterion",
        "step2": "only if step 1 rejects: Holm over {FULL vs D0, D0V vs D0G}",
        "step3": "only if BOTH step-2 hypotheses reject: Holm over {E vs D0, EF vs EP}",
        "descriptive_no_claim": ["D0 vs A0", "A0V vs A0", "D0G vs D0", "recovery statistics per retry arm", "outcome taxonomy", "per-DB and per-family accuracy"],
        "power_notes": {
            "step1": "see R4",
            "D0V_vs_D0G": "simulated at the Holm worst-case level 0.025: repair 20% vs 10%: power 0.53 (independent repairs) to 0.99 (generic repairs a subset); 10% vs 5%: 0.21-0.66; 20% vs 5%: 0.96-1.00",
            "E_vs_D0": "0.24-0.43 at a true +2 pp (discordance 8-16%); a non-rejection is inconclusive",
            "EF_vs_EP": "52 changed items; power <= 0.71 even at +8 net; descriptive in practice"},
        "supersedes": ["analysis_plan.json primary/secondary_gatekept ordering", "amendment A3 ordering (A3's step 1 is kept; its steps 2-3 are replaced)"]},
    "arms": ARMS,
    "inference": {"model": "Qwen/Qwen2.5-Coder-3B-Instruct", "revision": "488639f1ff808d1d3d0ba301aef8c11461451ec5", "dtype": "bfloat16 (T4, emulated)",
                  "decoding": "greedy", "max_new_tokens": 1024, "seed": 42,
                  "batch_size": {"first passes A0/D0/E": 8, "matched delta EP+EF": 1, "all retry calls (A0V, D0V, D0G, FULL)": 1},
                  "oom_policy_first_pass": "batch-1 fallback, recorded (as MQL-0009)"},
    "compute_estimate_t4_hours": {"first_passes": 7.2, "matched_delta_104_calls": 0.4, "retries_4_arms_about_620_calls": 2.4, "total": 10.0,
                                  "basis": "MQL-0009 9.9-12.8 s/item batched; batch-1 calls budgeted at 14 s; ~21% flag rate"},
}


def main():
    if A4.exists() or INDEX.exists():
        raise SystemExit("A4 / registration index already exist (write-once)")
    A4.write_text(json.dumps(AMENDMENT, indent=1), encoding="utf-8", newline="\n")
    docs = [
        ("config", ROOT / "scripts/nl2mql_mql0010/configs/mql0010_experiment.json",
         "model, prompts, corpus, split, context budget; its arms / compute / source_sha256 / analysis order are superseded by A4"),
        ("analysis_plan", PRE / "analysis_plan.json", "metric, sensitivity analyses, integrity rules; ordering superseded by A3 then A4"),
        ("split_registration", PRE / "split_registration.json", "holdout / dev databases and items, leakage audit (A1 applied)"),
        ("amendment_A2", PRE / "amendment_A2_2026-11-19.json", "dev proxy recomputed; no registration change"),
        ("amendment_A3", PRE / "amendment_A3_2026-11-19.json", "test order; superseded by A4 except step 1"),
        ("amendment_A4", A4, "GOVERNS arms, inference batching, test sequence, practical criterion, compute (R1-R5)"),
    ]
    index = {
        "experiment": "MQL-0010", "written": "2026-11-19", "status": "REGISTERED (A4) - NOT EXECUTED",
        "governing_order": "amendment_A4 > amendment_A3 > analysis_plan > config, for any aspect they both address",
        "documents": [{"role": r, "path": rel(p), "sha256": sha(p), "governs": g} for r, p, g in docs],
        "amendment_A1": "embedded in split_registration.json ('amendments')",
        "source_sha256": {f.name: sha(f) for f in sorted(SRC.glob("*.py"))},
        "corpus_sha256": "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf",
        "effective_arms": [a["name"] for a in ARMS],
    }
    INDEX.write_text(json.dumps(index, indent=1), encoding="utf-8", newline="\n")
    print("A4", sha(A4))
    print("index", sha(INDEX))


if __name__ == "__main__":
    main()
