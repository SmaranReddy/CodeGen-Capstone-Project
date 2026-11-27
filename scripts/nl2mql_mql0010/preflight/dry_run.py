"""MQL-0010 MOCKED end-to-end dry run (no model, no Kaggle, no Atlas, no network).

What is real: the registered holdout ids, the enforced retrieval view (R3), the schema context derived offline from
the local Spider SQLite files (same derivation as MQL-0009), the frozen zero-shot prompts built by the unchanged
prompt_spider1 module, the plans for A0 / D0 / E / EF, the driver (all 9 arms incl. the matched delta pass and the
generic-retry control), export verification and the registered analysis sequence.
What is fake: `fake_generate` (deterministic text from a prompt hash) and `fake_evaluate` (deterministic outcome from a
hash; it never reads gold SQL or gold results). Token counting uses len/4 (the real package uses the pinned tokenizer).
The mocked "accuracies" are meaningless and are labelled MOCK in the report.

Run: python -B scripts/nl2mql_mql0010/preflight/dry_run.py      -> outputs/nl2mql_mql0010/dryrun/dry_run_report.json
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0010/src"))
sys.path.insert(1, str(ROOT / "kaggle/nl2mql_mql0009_retrieval_control/src"))
sys.path.insert(2, str(ROOT / "scripts/nl2mql_mql0009"))
sys.path.insert(3, str(ROOT / "scripts/nl2mql"))
sys.path.insert(4, str(ROOT / "scripts/spider1"))
import mql0008_core as M8          # noqa: E402
import planner as PL               # noqa: E402
import driver as DR                # noqa: E402
import scoring as SC               # noqa: E402
import holdout_view as HV          # noqa: E402

PRE = ROOT / "outputs/nl2mql_mql0010/preflight"
OUT = ROOT / "outputs/nl2mql_mql0010/dryrun"
CTX = {"k": 3, "max_context_tokens": 1500, "n_ctx": 8192, "max_new_tokens": 1024, "margin_tokens": 64}


def rl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def h(s):
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def fake_generate(system, messages):
    user = messages[0]["content"]
    colls = re.findall(r"^- (\w+__\w+)\s*$", user, flags=re.M)
    coll = colls[0] if colls else "unknown__collection"
    if len(messages) == 1:
        bad = h(system + user) % 100 < 25
        stage = {"$match": {"__missing_field__": 1}} if bad else {"$match": {}}
        return json.dumps({"collection": coll, "pipeline": [stage, {"$project": {"_id": 0}}]})
    fb = messages[2]["content"]
    p_fix = 70 if "does not exist" in fb else 35
    ok = h(user + fb) % 100 < p_fix
    stage = {"$match": {}} if ok else {"$match": {"__missing_field__": 2}}
    return json.dumps({"collection": coll, "pipeline": [stage, {"$limit": 5}, {"$project": {"_id": 0}}]})


def fake_evaluate(rec):
    t = rec["raw_text"]
    if "__missing_field__" in t:
        return "incorrect"
    return "correct" if h(rec["id"] + t) % 100 < 55 else "incorrect"


def main():
    import build_mql0009_package as B
    reg = HV.load_registration(PRE / "split_registration.json")
    hdb = sorted(HV.holdout_dbs(reg))
    view = HV.load_view(ROOT / "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl", reg)       # R3
    tj = json.loads((ROOT / "data/raw/spider1/spider_data/tables.json").read_text(encoding="utf-8"))
    tables = {t["db_id"]: t for t in tj if t["db_id"] in set(hdb) | {r["db_id"] for r in view}}
    docs = M8.build_docs(view, tables)
    HV.assert_docs_exclude(docs, set(hdb))                                                           # R3
    items = rl(PRE / "holdout_items.jsonl")
    assert hashlib.sha256("\n".join(r["id"] for r in items).encode()).hexdigest() == reg["holdout"]["ids_sha256"]
    gold_sql = {r["id"]: r for r in rl(ROOT / "data/processed/spider_sql_v1/train.jsonl")}          # leak check only
    atlas, samples, _ = B.derive_context(hdb)
    ctxdir = OUT / "context"
    ctxdir.mkdir(parents=True, exist_ok=True)
    M8.write_json(ctxdir / "atlas_collections.json", atlas)
    M8.write_json(ctxdir / "prompt_samples.json", samples)
    M8.write_json(ctxdir / "tables.json", [tables[d] for d in hdb])
    K, P = B.prompt_modules(ctxdir)
    assert P.template_hash() == "7401bf140dd1b8a9dac057edd9f729a8624823604fa2bd6ae9cfaf1c1890c1be"
    import dev1034_core as D
    prompts = D.build_prompts(items, P)
    D.check_no_gold_leak(prompts, [gold_sql[r["id"]] for r in items])
    fps = {p["id"]: p for p in prompts}
    infos = M8.eval_schema_infos(atlas, hdb)
    excl = reg["leakage"]["L3_L5_exclusion_pairs"]
    count = lambda s: len(s) // 4  # noqa: E731

    def plan(flags, name):
        return PL.build_plan({"name": name, **flags}, items, prompts, docs, infos, atlas, tables, CTX, count,
                             excl, forbidden_db_ids=hdb)
    plans = {"A0": plan({"family": "zero_shot"}, "A0"), "D0": plan({"family": "hybrid"}, "D0"),
             "E": plan({"family": "hybrid", "diversify": "signature", "candidate_pool": 50}, "E"),
             "EF": plan({"family": "hybrid", "diversify": "signature", "candidate_pool": 50, "fk_type_notes": "numeric_unified"}, "EF")}
    by_doc = {d["doc_id"]: d for d in docs}
    ptext = {a: {r["id"]: PL.user_text_for(r, fps[r["id"]], by_doc) for r in rows} for a, rows in plans.items()}
    calls = {"n": 0}

    def gen(system, messages):
        calls["n"] += 1
        return fake_generate(system, messages)

    recs = {}
    for a in ("A0", "D0", "E"):
        recs[a] = DR.first_pass(a, plans[a], ptext[a], gen)
    n_first = calls["n"]
    recs["EF"], recs["EP"] = DR.matched_delta("EF", "EP", plans["EF"], plans["E"], recs["E"], ptext["EF"], ptext["E"], gen)
    n_delta = calls["n"] - n_first
    recs["A0V"] = DR.retry_arm("A0V", recs["A0"], ptext["A0"], atlas, gen)
    recs["D0V"] = DR.retry_arm("D0V", recs["D0"], ptext["D0"], atlas, gen)
    recs["D0G"] = DR.retry_arm("D0G", recs["D0"], ptext["D0"], atlas, gen, feedback_mode="generic")
    recs["FULL"] = DR.retry_arm("FULL", recs["EF"], ptext["EF"], atlas, gen)
    n_retry = calls["n"] - n_first - n_delta
    ids = [r["id"] for r in items]
    ver = SC.verify_exports(recs, plans, ids)
    outcomes = SC.score_all(recs, fake_evaluate)
    ana = SC.analyze(outcomes, ids, recs, reps=2000)
    notes_dbs = sorted({r["db_id"] for r in plans["EF"] if r["schema_notes"]})
    rep = {"MOCK": "fake model + fake evaluator; numbers below are NOT results",
           "holdout_items": len(ids), "holdout_dbs": len(hdb), "view_records": len(view), "docs": len(docs),
           "view_ids_sha256_ok": True, "no_holdout_db_in_docs": True,
           "generation_calls": {"first_passes_A0_D0_E": n_first, "matched_delta_EP_plus_EF": n_delta, "retries_all_arms": n_retry},
           "delta_items": ver["delta_items"], "notes_dbs": notes_dbs, "retried": ver["retried"],
           "budget_drops": {a: sum(bool(r["dropped_budget"]) for r in rows) for a, rows in plans.items()},
           "verify_exports": ver, "analysis_MOCK": ana}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dry_run_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "analysis_MOCK"}, indent=1))
    print("MOCK step1:", ana["step1"]["diff_pp"], ana["step1"]["ci95_pp"], "step2:", type(ana["step2"]).__name__)


if __name__ == "__main__":
    main()
