"""Write the MQL-0015 registration (write-once, append-only; refuses to overwrite). CPU only, no network.

Inputs must already exist and verify: sampling artifacts (SHA256SUMS), registration_input_checks.json,
prompt_pins.jsonl, power_analysis.json. Output: outputs/nl2mql_final_1000/registration/mql0015_registration.json
and registration/SHA256SUMS.txt. Later changes go in dated amendment files, never in this file.
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMP = ROOT / "outputs/nl2mql_final_1000/sampling"
REG = ROOT / "outputs/nl2mql_final_1000/registration"
OUT = REG / "mql0015_registration.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def need(c, m):
    if not c:
        raise SystemExit(f"STOP: {m}")


def main():
    need(not OUT.exists(), f"{OUT} exists: the registration is write-once (add a dated amendment instead)")
    for line in (SAMP / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, n = line.split("  ", 1)
        need(sha((SAMP / n).resolve()) == h, f"sampling artifact changed: {n}")
    chk = json.loads((REG / "registration_input_checks.json").read_text(encoding="utf-8"))
    need(chk["ft_adapter"]["matches_verified"] and not chk["gold_scan"]["n_hits"]
         and not chk["schema_context"]["schema_prefix_differs"] and not chk["r3_plans"]["exclusion_violations"],
         "registration input checks not clean")
    need(sha(REG / "prompt_pins.jsonl") == chk["prompt_pins_sha256"], "prompt pins changed")
    power = json.loads((REG / "power_analysis.json").read_text(encoding="utf-8"))
    man = json.loads((SAMP / "sample_manifest.json").read_text(encoding="utf-8"))
    pk = chk["mql0013_package"]
    reg = {
        "experiment": "MQL-0015",
        "title": "Final independent NL->MQL evaluation: A0V vs R3V vs FT-R3V on 1,000 Spider test_full questions (Atlas execution)",
        "registered_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "REGISTERED before package build, generation and scoring. No prediction exists for any arm on these items.",
        "id_note": "MQL-0015 was unused (no file under docs/, outputs/, scripts/, kaggle/, tests/, configs/ mentioned it).",
        "relation_to_mql0013": (
            "MQL-0013 (R0V vs R3V on all 2,147 test_full items) stays SUPERSEDED and is not reactivated. Its frozen, "
            "never-executed package files are reused READ-ONLY as hash-pinned inputs (zero-shot prompts, R3 prompts and "
            "plans, Atlas-form context, scoring contract content). Nothing under outputs/nl2mql_mql0013/, "
            "kaggle/nl2mql_mql0013_test_full*, docs/mql0013_execution_runbook.md or scripts/nl2mql_mql0013/ is written."),
        "user_decisions_2026_11_23": [
            "evaluation = exactly 1,000 questions sampled from the untouched Spider test_full (2,147 / 40 DBs), seed 20261123",
            "three systems: base Qwen-3B zero-shot, base Qwen-3B + frozen hybrid R3, verified FT checkpoint + frozen hybrid R3",
            "R3 frozen (BM25 + bge-small-en-v1.5 dense + existing RRF + existing Jaccard structural rerank, top 3); no embedding-only retrieval",
            "the remaining 1,147 test_full questions stay unused",
            "preparation only: no Kaggle launch, generation, fine-tuning, Atlas provisioning or R3/corpus change before approval",
        ],
        "evaluation": {
            "source": man["source"],
            "sample": {"n": 1000, "seed": 20261123, "ids_file": "outputs/nl2mql_final_1000/sampling/sample_ids.txt",
                       "ids_sha256": man["sample_ids_sha256"], "manifest_sha256": sha(SAMP / "sample_manifest.json"),
                       "config_sha256": sha(SAMP / "sampling_config.json"), "unused_ids_sha256": man["unused_ids_sha256"],
                       "n_databases": 40, "method": "cell-level largest-remainder allocation over db_id x hardness; "
                       "within-cell structure-ordered systematic sample (sampling_config.json, PRE_DRAW_FREEZE.txt)"},
            "denominator": "1,000 for every arm; no post-hoc exclusion from the primary analysis",
            "gold_use": "gold SQL / gold results only for scoring and for metadata strata; never in prompts or the generation package",
        },
        "arms": {
            "A0V": {"model": "Qwen/Qwen2.5-Coder-3B-Instruct @ 488639f1ff808d1d3d0ba301aef8c11461451ec5 (original weights)",
                    "prompt": "frozen MQL-0013 zero-shot prompt per id (template nl-mql-spider1-v1, sha 7401bf14...), no demonstrations",
                    "prompt_pin": "prompt_pins.jsonl: zero_shot_prompt_sha256",
                    "component_V": "MQL-0010 static schema validator; one schema-feedback retry; keep retry iff strictly fewer issues"},
            "R3V": {"model": "Qwen/Qwen2.5-Coder-3B-Instruct @ 488639f1 (original weights)",
                    "prompt": "frozen MQL-0013 R3 prompt per id (nl-mql-spider1-rag-v1): same system prompt and schema block as A0V "
                              "plus 3 demonstrations from frozen R3 retrieval",
                    "retrieval": {"lexical": "rrf_fused = RRF_60(BM25 rank) + RRF_60(dense rank), max-normalised, + 1.0 x Jaccard "
                                             "structural score", "k": 3, "diversify": "none",
                                  "index": chk["r3_retrieval"]["index_name"], "embedder": "BAAI/bge-small-en-v1.5 @ 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                                  "view": "approved corpus v2.1 minus the 18 MQL-0010 holdout DBs: 2,912 records (ids sha cfde5e9c...)",
                                  "exclusions": "frozen MQL-0013 L3-L5 pairs (6 sampled items affected, 0 violations)",
                                  "plans": "frozen MQL-0013 plan_R3 rows of the 1,000 ids (not re-derived)"},
                    "prompt_pin": "prompt_pins.jsonl: r3_prompt_sha256",
                    "component_V": "identical to A0V"},
            "FT-R3V": {"model": "Qwen/Qwen2.5-Coder-3B-Instruct @ 488639f1 bf16 + LoRA q3b_r16_q10 merged with peft merge_and_unload "
                                "(as scored in MQL-0010-FT1)",
                       "adapter": {"path": chk["ft_adapter"]["path"], "sha256": chk["ft_adapter"]["file_sha256"],
                                   "adapter_config_sha256": chk["ft_adapter"]["adapter_config_sha256"],
                                   "handoff_zip_sha256": chk["ft_adapter"]["handoff_zip_sha256"],
                                   "r": 16, "lora_alpha": 32, "trained_on": "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit (QLoRA)"},
                       "tokenizer": "base tokenizer at 488639f1 (NOT the Unsloth tokenizer files in the handoff)",
                       "merged_weights_hash": "recorded at generation time (sha256 of the merged safetensors) and checked on resume",
                       "prompt": "byte-identical to R3V for every id (same prompt_pins r3_prompt_sha256)",
                       "component_V": "identical to A0V"},
        },
        "held_constant": [
            "same 1,000 ids in the same order; same system prompt and schema/sample-document context (verified item by item)",
            "same output contract: one JSON object {collection, pipeline}; same parser",
            "bfloat16 on Tesla T4 (emulated), greedy (do_sample=False), max_new_tokens 1024, seed 42",
            "first-pass batch 8 in the same item order for A0, R3 and FT-R3 (OOM -> batch-1 fallback, logged); retry batch 1",
            "same validator, retry prompt and keep rule; same evaluator, Atlas database and timeout for all arms",
            "no prompt truncation (generator has no truncation; 0 demo-budget drops; max R3 prompt 3,782 tokens est.)",
        ],
        "generation_design": {
            "package": "new MQL-0015 package (kaggle/nl2mql_mql0015_final_1000), built only after approval, from the pinned "
                       "frozen inputs filtered to the 1,000 ids; no gold, no hardness/structure labels",
            "passes": ["first:A0 (base)", "first:R3 (base)", "retry:A0V (base)", "retry:R3V (base)",
                       "first:FT-R3 (merged FT)", "retry:FT-R3V (merged FT)"],
            "sessions": "session 1 = base passes; session 2 = FT passes (merge at session start); environment pinned after session 1",
            "export_gate": "every id present for every arm; an item whose generation fails even at batch 1 is exported as "
                           "generation_failure (counted wrong), not silently dropped",
        },
        "scoring": {
            "metric": "strict execution accuracy on Atlas: correct / 1,000; anything not 'correct' is wrong",
            "evaluator": "scripts/nl2mql/dev1034_core.py evaluate_prediction (sha 81b86f0b...) unchanged, via a new MQL-0015 adapter",
            "reference": "gold SQL executed once on local SQLite test_database, frozen by hash before any arm is scored",
            "atlas": "one database holding the 40 test_full DBs (180 collections, 245,574 documents; contract content = "
                     "MQL-0013 atlas_mapping 32cbbc4d...), read-only scoring user, exec timeout 30,000 ms, no indexes; the "
                     "physical database name is a pending user decision (see preflight)",
            "session": "all three arms scored in ONE Atlas session after all exports are complete; identical (item, raw_text) "
                       "executed once and shared across arms",
            "taxonomy": ["correct", "incorrect", "invalid", "execution_failure", "timeout", "generation_failure", "infrastructure"],
            "infrastructure_rule": "connection/auth/server errors not caused by the query are re-executed up to 3 times in the "
                                   "same session; residual infrastructure outcomes count as wrong in the strict metric and are "
                                   "reported; if > 10 items (1%) remain infrastructure, the whole scoring run is repeated "
                                   "in a new session before analysis",
        },
        "analysis_plan": {
            "primary": {"comparison": "FT-R3V vs R3V", "question": "additional contribution of fine-tuning to hybrid RAG",
                        "test": "exact two-sided McNemar", "alpha": 0.05,
                        "ci": "paired bootstrap 95% (percentile), 10,000 resamples, seed 20261123"},
            "secondary": {"comparisons": [["R3V", "A0V"], ["FT-R3V", "A0V"]], "test": "exact two-sided McNemar",
                          "multiplicity": "Holm over the two secondaries at family alpha 0.05; Holm over all three comparisons "
                                          "reported as a robustness line", "ci": "same bootstrap"},
            "report_per_arm": ["correct / 1000 (%)", "incorrect", "invalid", "execution_failure", "timeout",
                               "generation_failure", "infrastructure", "retried / kept-retry counts"],
            "descriptive": [
                "first answers before retry: A0, R3, FT-R3 (and the retry effect per arm)",
                "by Spider hardness (easy/medium/hard/extra)",
                "by structure: join vs no join; nested; set operation; anti-join; primary structure class",
                "60% target reported as a fact only",
            ],
            "sensitivity_descriptive": [
                "cluster bootstrap by (db_id, gold SQL) for the primary CI (155 sampled paraphrase pairs)",
                "excluding degenerate-gold items (empty_result / all_null_result: 26)",
                "excluding the 5 items whose L5 near-duplicate partner is an FT training row",
                "excluding the 7 question_in_train items",
                "FT-R3V - R3V for R3 prompts > 2,048 tokens (144) vs <= 2,048 (856)",
                "population post-stratified accuracy (hardness x HAVING) vs raw accuracy (sample has HAVING 9.0% vs 7.5%)",
            ],
            "no_peeking": "no prediction is opened or scored before every arm is exported; no change to this plan after "
                          "registration except dated amendments written before generation",
            "power": {"note": "planning assumptions, NOT guarantees", **power},
        },
        "pins": {
            "mql0013_package_readonly_inputs": {k: pk[k] for k in pk if k.endswith("sha256")},
            "mql0013_plan_R3_canonical_sha256": "ab64d10238a5e3bf35ca9d15c110668f14bf76beef9cd94b3fbc4d7f81d5eb13",
            "r3_retrieval": chk["r3_retrieval"],
            "prompt_pins_sha256": chk["prompt_pins_sha256"],
            "registration_input_checks_sha256": sha(REG / "registration_input_checks.json"),
            "power_analysis_sha256": sha(REG / "power_analysis.json"),
            "atlas_contract": chk["atlas_contract"],
            "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted((ROOT / "scripts/nl2mql_final_1000").glob("*.py"))},
            "evaluator_sha256": {"scripts/nl2mql/dev1034_core.py": sha(ROOT / "scripts/nl2mql/dev1034_core.py")},
        },
        "known_risks": [
            "aan_1 holds 243k rows (Citation 124,842); unindexed $lookup can time out (counted as timeout, all arms)",
            "FT trained on prompts without demonstrations, foreign keys or sample documents and on sequences <= ~1.6k tokens; "
            "R3 prompts carry all three and 144/1000 exceed 2,048 tokens (distribution shift, measured by a sensitivity line)",
            "R3 demonstrations come from the 2,912-record view; the FT adapter was trained on 2,901 of these records "
            "(redundancy, not leakage: test DBs are disjoint)",
            "bf16 on T4 is emulated; batch composition/OOM fallbacks can change greedy outputs",
            "corpus lacks anti-join / intersect / union / nested families (MQL-0012); ~16% of the sample is in these families",
            "Atlas tie-order re-execution noise (~1-2% of items flipped in earlier runs); mitigated by single-session scoring",
            "test_full and the 40 DBs are consumed for NL->MQL by this experiment; the 1,147 unused items share these DBs",
        ],
    }
    REG.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    names = ["mql0015_registration.json", "registration_input_checks.json", "prompt_pins.jsonl", "power_analysis.json"]
    (REG / "SHA256SUMS.txt").write_text("".join(f"{sha(REG / n)}  {n}\n" for n in names), encoding="utf-8", newline="\n")
    print("registered", sha(OUT))


if __name__ == "__main__":
    sys.exit(main())
