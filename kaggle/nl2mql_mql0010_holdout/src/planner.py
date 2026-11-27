"""MQL-0010 per-question retrieval/prompt plan, one arm at a time, every component behind a flag.

Arm flags (configs/mql0010_experiment.json "arms"):
    family            "zero_shot" | "hybrid"
    k                 demonstrations (3)
    structure_scoring "baseline" | "nb"            (component S+R; evaluated, NOT adopted - see design doc)
    tie_break         "doc_id" | "lexical"
    diversify         "none" | "signature"         (component E)
    fk_type_notes     "off" | "numeric_unified"    (component F)
    validate_retry    false | true                 (component V; applied at generation time, recorded in the plan)
    candidate_pool    ranked candidates considered by the diversifier (default 50)

With family="hybrid", structure_scoring="baseline", tie_break="doc_id", diversify="none", fk_type_notes="off"
the plan row's selected ids and user prompt are identical to the registered MQL-0009 D_hybrid plan
(tests/test_mql0010_planner.py checks all 639 items).

Every row records: selected ids / scores / ranks / source DBs, the score trace of each selected document, the
top-10 ranking, exclusions applied, schema notes, dropped documents, token counts and prompt hashes.
"""
import hashlib
import json

import prompt_v2 as PV
import rag_block as RB
import retriever_v2 as R2

DEFAULT_ARM = {"family": "hybrid", "k": 3, "structure_scoring": "baseline", "tie_break": "doc_id",
               "diversify": "none", "fk_type_notes": "off", "validate_retry": False, "candidate_pool": 50}


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canon(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def prompt_sha(system, messages):
    return sha256_text(canon({"system": system, "messages": messages}))


def arm_settings(arm):
    unknown = set(arm) - set(DEFAULT_ARM) - {"name", "label"}
    if unknown:
        raise ValueError(f"unknown arm flags: {sorted(unknown)}")
    return {**DEFAULT_ARM, **arm}


def make_retriever(arm, docs, predictor=None):
    a = arm_settings(arm)
    if a["family"] == "zero_shot":
        return None
    return R2.StructureAwareRetriever(docs, predictor=predictor if a["structure_scoring"] == "nb" else None,
                                      structure_scoring=a["structure_scoring"], tie_break=a["tie_break"])


def build_plan(arm, eval_rows, frozen_prompts, docs, eval_infos, collections, tables_by_db, ctx_cfg, count_tokens,
               exclusions=(), predictor=None, forbidden_db_ids=()):
    """forbidden_db_ids (R3, amendment A4): databases whose documents must never be retrievable (the 18 holdout
    DBs for MQL-0010 holdout plans). Any such document in `docs` aborts the plan before an index is built."""
    a = arm_settings(arm)
    bad = sorted({d.get("db_id") for d in docs if d.get("db_id") in set(forbidden_db_ids)})
    if bad:
        raise ValueError(f"retrieval documents from forbidden databases: {bad}")
    retr = make_retriever(a, docs, predictor)
    by_prompt = {p["id"]: p for p in frozen_prompts}
    excl = {}
    for x in exclusions:
        excl.setdefault(x["eval_id"], set()).add(x["corpus_id"])
    rows = []
    for e in eval_rows:
        fp = by_prompt[e["id"]]
        system, base = fp["system"], fp["messages"][0]["content"]
        notes = (PV.fk_type_notes(e["db_id"], collections, tables_by_db[e["db_id"]], "numeric_unified")
                 if a["fk_type_notes"] == "numeric_unified" else [])
        ranked, rank_of, excluded, ordered = [], {}, [], []
        if retr is not None:
            full = retr.rank_examples(e["question"], eval_infos[e["db_id"]], qid=e["id"])
            rank_of = {d["doc_id"]: i + 1 for i, (d, _, _) in enumerate(full)}
            excluded = [{"corpus_id": d["doc_id"], "original_rank": i + 1}
                        for i, (d, _, _) in enumerate(full) if d["doc_id"] in excl.get(e["id"], ())]
            ranked = [t for t in full if t[0]["doc_id"] not in excl.get(e["id"], ())]
            if a["diversify"] == "signature":
                pool = ranked[:a["candidate_pool"]]
                first = R2.select_diverse(pool, a["k"])
                ids = {t[0]["doc_id"] for t in first}
                ordered = [t[0] for t in first] + [t[0] for t in ranked if t[0]["doc_id"] not in ids]
            elif a["diversify"] == "none":
                ordered = [t[0] for t in ranked]
            else:
                raise ValueError(a["diversify"])
        k = a["k"] if retr is not None else 0
        chosen, dropped, block, bt, total, notes_kept = PV.budgeted_block(
            e["db_id"], system, base, ordered, k, count_tokens, ctx_cfg["max_context_tokens"], ctx_cfg["n_ctx"],
            ctx_cfg["max_new_tokens"], ctx_cfg["margin_tokens"], notes)
        user = RB.insert_block(base, block)
        trace = {t[0]["doc_id"]: t[2] for t in ranked[:max(200, a["candidate_pool"])]}
        rows.append({
            "id": e["id"], "db_id": e["db_id"], "arm": a.get("name"),
            "selected_ids": [d["doc_id"] for d in chosen],
            "selected_scores": [trace.get(d["doc_id"], {}).get("score") for d in chosen],
            "selected_ranks": [rank_of[d["doc_id"]] for d in chosen],
            "selected_db_ids": [d["db_id"] for d in chosen],
            "selected_traces": [trace.get(d["doc_id"]) for d in chosen],
            "ranked_top10": [[t[0]["doc_id"], t[1]] for t in ranked[:10]],
            "excluded_by_rule": excluded, "schema_notes": notes_kept, "dropped_budget": dropped,
            "context_tokens": bt, "prompt_tokens_est": total, "empty_context": not block,
            "user_prompt_sha256": sha256_text(user),
            "prompt_sha256": prompt_sha(system, [{"role": "user", "content": user}]),
            "flags": {k2: a[k2] for k2 in DEFAULT_ARM},
        })
    return rows


def render_user_prompt(row_base_user, db_id, chosen_docs, notes):
    """Rebuild the user prompt from a plan row's selections (used by the generation notebook)."""
    return RB.insert_block(row_base_user, PV.build_block(db_id, chosen_docs, notes))


def user_text_for(row, frozen_prompt, docs_by_id):
    """Rebuild (system, user_text) for a plan row and verify it against the recorded hash (fail closed)."""
    system, base = frozen_prompt["system"], frozen_prompt["messages"][0]["content"]
    user = render_user_prompt(base, row["db_id"], [docs_by_id[i] for i in row["selected_ids"]], row["schema_notes"])
    if sha256_text(user) != row["user_prompt_sha256"]:
        raise ValueError(f"{row['id']}: rebuilt prompt differs from the plan")
    return system, user
