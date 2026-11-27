"""MQL-0011 planner: a copy of the registered `planner.build_plan` (scripts/nl2mql_mql0010/src/planner.py, NOT modified)
that takes the retrieval mode as one extra arm flag, `lexical`, and the VectorStore / Embedder as arguments.

Only differences from planner.build_plan:
  * flag `lexical` in {"bm25", "rrf_bm25", "dense_only", "rrf_fused"}; "bm25" builds the retriever with the ORIGINAL
    planner.make_retriever (identical object and ranking);
  * structure_scoring must be "baseline" (the NB predictor was not adopted in MQL-0010 and is not supported here);
  * each row carries one extra key, "retrieval_mql0011". Every other field is produced by the copied code and, for
    lexical="bm25", equals planner.build_plan's row (tests/test_mql0011_planner_equiv.py, 779 dev items).
"""
import planner as PL
import prompt_v2 as PV
import rag_block as RB
import retriever_dense as RD
import retriever_v2 as R2


def split_arm(arm):
    arm = dict(arm)
    lexical = arm.pop("lexical", "bm25")
    if lexical not in RD.LEXICAL_MODES:
        raise ValueError(f"unknown lexical mode {lexical}")
    a = PL.arm_settings(arm)
    if a["structure_scoring"] != "baseline" or a["tie_break"] != "doc_id":
        raise ValueError("MQL-0011 supports structure_scoring='baseline', tie_break='doc_id' only")
    return a, lexical


def make_retriever(a, lexical, docs, store=None, embedder=None, forbidden_db_ids=(), scorer=None):
    if a["family"] == "zero_shot":
        return None
    if lexical == "bm25":
        return PL.make_retriever(a, docs)
    return RD.DenseHybridRetriever(docs, lexical, store=store, embedder=embedder, forbidden_db_ids=forbidden_db_ids,
                                   scorer=scorer)


def build_plan(arm, eval_rows, frozen_prompts, docs, eval_infos, collections, tables_by_db, ctx_cfg, count_tokens,
               exclusions=(), forbidden_db_ids=(), store=None, embedder=None, scorer=None):
    a, lexical = split_arm(arm)
    bad = sorted({d.get("db_id") for d in docs if d.get("db_id") in set(forbidden_db_ids)})
    if bad:
        raise ValueError(f"retrieval documents from forbidden databases: {bad}")
    retr = make_retriever(a, lexical, docs, store, embedder, forbidden_db_ids, scorer)
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
            "user_prompt_sha256": PL.sha256_text(user),
            "prompt_sha256": PL.prompt_sha(system, [{"role": "user", "content": user}]),
            "flags": {k2: a[k2] for k2 in PL.DEFAULT_ARM},
            "retrieval_mql0011": {"lexical": lexical,
                                  "index_name": store.spec().index_name if (store is not None and
                                                                            lexical in RD.DENSE_MODES) else None},
        })
    return rows
