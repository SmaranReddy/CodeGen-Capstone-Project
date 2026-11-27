"""MQL-0010 component V (controlled recovery): validate the first answer; if the static validator reports issues,
ask the model ONCE more with the issues as feedback, and keep the retry only if it is strictly better.

Separate from retrieval: the same function is used for zero-shot and RAG arms. No database execution, no gold.

Decision rule (fixed before any MQL-0010 prediction):
  * first answer unparseable                     -> retry with a format message
  * first answer parses and has 0 issues         -> keep it, no retry
  * first answer has issues                      -> retry once
  * retry kept iff it parses and has FEWER issues than the first answer (ties keep the first answer).
`generate(system, messages) -> text` is injected (the Kaggle notebook passes the pinned Qwen model, greedy decoding).
The retry carries the SAME system prompt as the first pass (it holds the result contract).
"""
import mql0009_core as K9  # frozen MQL-0009 package core (parser identical to the evaluator)
import prompt_v2 as PV
import validator as V

FORMAT_FEEDBACK = ("Your previous answer was not a single valid JSON object with keys \"collection\" and \"pipeline\".")
# R5 generic-retry control (amendment A4): same trigger, same items, same acceptance rule and retry budget, but the
# second turn carries NO schema-specific content (no issue list, no field / collection names, no suggestions).
GENERIC_FEEDBACK = "Your previous query may contain mistakes. Review it and return a corrected query."
FEEDBACK_MODES = ("schema", "generic")


def parse_output(text):
    """The registered evaluator's parser (mql0009_core.parse_output == dev1034_core.parse_output, verbatim), plus
    the evaluator's normalisation (method defaulted, empty pipeline -> [{"$match": {}}]). -> dict or None."""
    n = K9.normalize_prediction(text, "ok")
    return n["normalized_mql"] if n["parse_status"] == "parsed" else None


def assess(text, validator):
    obj = parse_output(text)
    if obj is None:
        return None, None
    return obj, validator.validate(obj)


def recover(system, first_user_text, first_output, validator, generate, enabled=True, feedback_mode="schema"):
    """-> (final_text, trace). trace is JSON-serialisable and recorded per item.
    feedback_mode "schema": validator issues (component V); "generic": GENERIC_FEEDBACK (R5 control). The trigger
    (unparseable or >= 1 issue) and the acceptance rule (strictly fewer issues) are identical in both modes."""
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(feedback_mode)
    obj, issues = assess(first_output, validator)
    trace = {"enabled": enabled, "first_parsed": obj is not None,
             "first_issues": None if issues is None else [i.text() for i in issues],
             "retried": False, "retry_parsed": None, "retry_issues": None, "kept": "first", "feedback_mode": feedback_mode}
    if not enabled or (obj is not None and not issues):
        return first_output, trace
    if obj is None:
        feedback = FORMAT_FEEDBACK
    else:
        feedback = V.format_feedback(issues) if feedback_mode == "schema" else GENERIC_FEEDBACK
    msgs = PV.recovery_messages(first_user_text, first_output, feedback)
    second = generate(system, msgs)
    trace.update(retried=True, feedback=feedback)
    obj2, issues2 = assess(second, validator)
    trace["retry_parsed"] = obj2 is not None
    trace["retry_issues"] = None if issues2 is None else [i.text() for i in issues2]
    first_score = float("inf") if obj is None else len(issues)
    second_score = float("inf") if obj2 is None else len(issues2)
    if second_score < first_score:
        trace["kept"] = "retry"
        return second, trace
    return first_output, trace
