"""MQL-0010 generation driver (model-agnostic; the Kaggle notebook injects the pinned Qwen model).

Three kinds of arm (configs/mql0010_experiment.json):
  * "generate"      first-pass generation for every item of the arm's plan              (A0, D0, E)
  * "prompt_delta"  generate ONLY items whose prompt hash differs from a base arm's plan; every other item reuses
                    the base arm's first-pass record verbatim (greedy decoding: same prompt -> same output)  (E+F)
  * "retry"         reuse a source arm's first-pass outputs and apply the single controlled recovery step
                    (recovery.recover) - a retry generation happens only for items the validator flags    (A0+V, D0+V, Full)
Reuse makes the derived comparisons attributable: an outcome difference between D0 and D0+V can only come from a
kept retry, and between E and E+F only from a changed prompt. It also removes batch-composition noise from them.

Amendment A4 (R2, R5):
  * "matched_delta" replaces "prompt_delta" for the type-note comparison: for every item whose prompt changes, BOTH the
    base prompt (control arm E', "EP") and the new prompt (EF) are generated in the same delta pass, one prompt per
    call (batch size 1), so the EF vs EP contrast differs only in the note - never in batch neighbours or padding.
    Unchanged items reuse E's record in both arms (identical prompt, greedy).
  * "retry" takes feedback_mode "schema" (V) or "generic" (R5 control D0G: same trigger, items, acceptance rule,
    model, decoding and retry budget; no schema-specific content). Every retry call is one prompt per call.

Prediction record (one per item and arm; the scored text is `raw_text`):
  id, db_id, arm, status, raw_text, first_raw_text, source ("generated" | "reused:<arm>"), prompt_sha256,
  recovery (trace or None), experiment.
`generate(system, messages) -> text` must be deterministic (greedy).
"""
import recovery as RC
import validator as V

EXPERIMENT = "MQL-0010"


def first_pass(arm_name, plan_rows, prompts, generate):
    """prompts: {id: (system, user_text)} built from the plan (hash-checked by the caller)."""
    out = []
    for r in plan_rows:
        system, user = prompts[r["id"]]
        text = generate(system, [{"role": "user", "content": user}])
        out.append({"id": r["id"], "db_id": r["db_id"], "arm": arm_name, "status": "ok", "raw_text": text,
                    "first_raw_text": text, "source": "generated", "prompt_sha256": r["prompt_sha256"],
                    "recovery": None, "experiment": EXPERIMENT})
    return out


def prompt_delta(arm_name, plan_rows, base_plan_rows, base_records, prompts, generate):
    base_plan = {r["id"]: r for r in base_plan_rows}
    base_rec = {r["id"]: r for r in base_records}
    out = []
    for r in plan_rows:
        if r["prompt_sha256"] == base_plan[r["id"]]["prompt_sha256"]:
            b = base_rec[r["id"]]
            out.append({**b, "arm": arm_name, "source": f"reused:{b['arm']}"})
        else:
            out.extend(first_pass(arm_name, [r], prompts, generate))
    return out


def matched_delta(arm_name, control_name, plan_rows, base_plan_rows, base_records, prompts_new, prompts_base, generate):
    """R2. -> (records of arm_name, records of control_name). Items whose prompt hash differs from the base plan are
    generated twice in this pass (control first, then new, per item, one prompt per call); all other items reuse the
    base record in both arms. The control arm's changed items carry source "generated:matched_delta"."""
    base_plan = {r["id"]: r for r in base_plan_rows}
    base_rec = {r["id"]: r for r in base_records}
    new, ctl = [], []
    for r in plan_rows:
        b = base_rec[r["id"]]
        if r["prompt_sha256"] == base_plan[r["id"]]["prompt_sha256"]:
            new.append({**b, "arm": arm_name, "source": f"reused:{b['arm']}"})
            ctl.append({**b, "arm": control_name, "source": f"reused:{b['arm']}"})
            continue
        bp = base_plan[r["id"]]
        c = first_pass(control_name, [bp], prompts_base, generate)[0]
        n = first_pass(arm_name, [r], prompts_new, generate)[0]
        ctl.append({**c, "source": "generated:matched_delta"})
        new.append({**n, "source": "generated:matched_delta"})
    return new, ctl


def retry_arm(arm_name, source_records, prompts, contexts, generate, enabled=True, feedback_mode="schema"):
    """contexts: {db_id: atlas_collections-style context}; one SchemaValidator per database."""
    vals = {}
    out = []
    for s in source_records:
        system, user = prompts[s["id"]]
        v = vals.setdefault(s["db_id"], V.SchemaValidator(s["db_id"], contexts))
        final, trace = RC.recover(system, user, s["first_raw_text"], v, generate, enabled=enabled, feedback_mode=feedback_mode)
        out.append({**s, "arm": arm_name, "raw_text": final, "source": f"reused:{s['arm']}", "recovery": trace})
    return out


def summarize_recovery(records):
    n = len(records)
    tr = [r["recovery"] for r in records if r.get("recovery")]
    return {"n": n, "retried": sum(t["retried"] for t in tr), "kept_retry": sum(t["kept"] == "retry" for t in tr),
            "retry_clean": sum(bool(t["retried"]) and t["retry_issues"] == [] for t in tr)}
