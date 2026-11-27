"""Pre-register the AI review panel v1 run BEFORE any agent is launched (plan §E.7.3-§E.7.6).

    python -I -B scripts/ai_review_panel_v1/prereg.py   -> outputs/ai_review_panel_v1/prereg.json (refuses to overwrite)
"""
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANEL = ROOT / "outputs/ai_review_panel_v1"
MEMORY_INDEX = Path.home() / ".claude/projects/C--Users-Asus-OneDrive-Desktop-CodeGen/memory/MEMORY.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = PANEL / "prereg.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    man = json.loads((PANEL / "main/package_manifest.json").read_text(encoding="utf-8"))
    stab = sorted((PANEL / "stability/package").glob("batch_*.jsonl"))
    mem = MEMORY_INDEX.read_text(encoding="utf-8") if MEMORY_INDEX.exists() else ""
    leak_lines = [l for l in mem.splitlines() if any(w in l.lower() for w in ("cross-review", "/120", "audit of 120", "source label"))]
    doc = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "pre-registered before any agent launch",
        "label": "AI-reviewed (multi-agent panel); never human-reviewed",
        "protocol_change": "replaces the frozen human P2 gate with G-AIR for sql2mql-v2.1 / spider-pool-v3 (user decision "
                           "2026-11-12); pilot_freeze.json is not edited; the human P2 gate remains NOT passed",
        "prompts_sha256": {p.name: sha(p) for p in sorted((HERE / "prompts").glob("*.md"))},
        "tools_sha256": {p.name: sha(p) for p in sorted(HERE.glob("*.py"))},
        "roles": {"agent_1": "source SQL auditor (source only)", "agent_2": "translation auditor (translation only)",
                  "agent_3": "adversarial end-to-end (both)", "agent_4": "independent pass 1 (both) then adjudication"},
        "models": {"agent_1": "opus", "agent_2": "opus", "agent_3": "sonnet", "agent_4": "opus",
                   "diversity_note": "only Claude models are available to the orchestrator (Agent tool): no second model "
                                     "family. agent_3 (sonnet) differs in model size/version from the others in each "
                                     "dimension; shared blind spots remain likely (plan §E.7.2)."},
        "decision_rules": {
            "assessing_agents": {"source": ["agent_1", "agent_3", "agent_4"], "translation": ["agent_2", "agent_3", "agent_4"]},
            "unanimous_yes": "final YES without adjudication",
            "otherwise": "agent_4 adjudicates with fresh evidence; majority vote never used",
            "asymmetric_rule": "a record stays only if its final labels are YES (translation YES or NOT_APPLICABLE); "
                               "NO or UNSURE -> quarantine",
            "planted_caught": "final label NO in the planted dimension (strict); NO-or-UNSURE reported as secondary",
        },
        "thresholds": {"min_planted_per_dimension": 60, "min_caught_per_dimension": 55, "min_class_rate": 0.75,
                       "min_kappa": 0.6, "gair_n": 200, "gair_translation_upper": 0.05,
                       "gair_translation_error_count": "NO + UNSURE (conservative)",
                       "source_label_threshold": None, "source_label_note": "user decision 2026-11-13: report only"},
        "failure_handling": "if calibration (either dimension) or stability fails, the panel's labels are NOT used: every "
                            "S9/S10 candidate stays in C_semantic_review, G-AIR is reported as failed, and no prompt or "
                            "threshold is changed under this pre-registration (a retry needs a new prereg version and a "
                            "recorded deviation)",
        "items": man, "stability_batches_sha256": {p.name: sha(p) for p in stab},
        "planted_key_sha256": sha(PANEL / "_orchestrator_only/planted_key.jsonl"),
        "panel_key_sha256": {p.name: sha(p) for p in sorted((PANEL / "_orchestrator_only").glob("panel_key_*.jsonl"))},
        "isolation": {
            "memory_index_sha256": hashlib.sha256(mem.encode("utf-8")).hexdigest() if mem else None,
            "memory_index_not_scrubbed": "per the user's instruction (do not modify global memory), the auto-loaded "
                                         "MEMORY.md was NOT scrubbed; subagents auto-load it. Lines that mention earlier "
                                         "review results are listed below; they name no item, panel id or verdict for "
                                         "any item in this run, but they are a disclosed anchoring risk.",
            "memory_index_review_lines": leak_lines,
            "enforced_by": "instruction only (forbidden paths in standards.md); not by the sandbox. Each agent writes "
                           "context_disclosure.json.",
        },
        "deviations_from_plan": [
            "ro_exec.py is a derived copy of the run01 executor (same safety rules), because the original hard-codes "
            "run01 paths and logs inside frozen run01",
            f"planted share {man['planted_share']} instead of ~0.15, to bound the number of reviewed items; one combined "
            "phase (calibration + S9 + S10 + G-AIR) whose labels are used only after calibration and stability pass",
            "G-AIR sample drawn from the population that is in spider-pool-v3 regardless of the panel's S9/S10 outcomes, "
            "before spider-pool-v3 is built, so the panel runs once",
            "memory index not scrubbed (user instruction); disclosed above",
        ],
    }
    blob = json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True)
    doc["_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} body_sha256={doc['_sha256']} file_sha256={sha(out)}")


if __name__ == "__main__":
    main()
