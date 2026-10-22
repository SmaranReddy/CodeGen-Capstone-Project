"""Offline retrieval evaluation (no model calls) -> outputs/rag_week3/retrieval_eval.json.

Relevance proxy: structural overlap between the query's gold-SQL pattern set and each retrieved
example's pattern set (patterns.sql_patterns; the generic `select`/`where` labels ignored).
Gold SQL is used ONLY here, to score retrieval; it never reaches a prompt or a retriever.

Selection protocol (decided before any RAG generation):
  * SELECT on train, leave-one-database-out (LODO): each train item queries the corpus with
    its own database's examples removed (the retriever always drops target-db documents), which
    mirrors the cross-database situation of validation/test queries.
  * CONFIRM on validation (72 items): reported, not used to choose.
Metrics at k: mean Jaccard of retrieved patterns, hit@k (some retrieved example has exactly the
query's pattern set), pattern recall@k (share of the query's patterns covered by the union of
the k examples). Guide selection precision/recall vs the gold patterns is reported for k_g=3.
Also reports precision/recall of the inference-time question cues against gold SQL labels.

Usage: python scripts/rag/retrieval_eval.py
"""
import json
import time

import common as C
from common import K
import patterns as PT
import retrieval as RT

OUT = C.ROOT / "outputs/rag_week3/retrieval_eval.json"
KS = (1, 3, 5)
HYBRID_WEIGHTS = (0.25, 0.5, 1.0)


def gold_patterns(row, info):
    return (PT.sql_patterns(row["gold_sql"]) | ({"text_number"} if info["text_number_fields"] else set())) - RT._GENERIC


def evaluate(retriever, rows, infos, k_guides=3):
    agg = {k: {"jacc": 0.0, "hit": 0, "recall": 0.0} for k in KS}
    g_prec = g_rec = 0.0
    t0 = time.perf_counter()
    for r in rows:
        info = infos[r["db_id"]]
        gold = gold_patterns(r, info)
        ranked = retriever.rank_examples(r["question"], info, r["id"])
        for k in KS:
            top = [d for d, _ in ranked[:k]]
            if not top:
                continue
            agg[k]["jacc"] += sum(RT.jaccard(gold, d["patterns"]) for d in top) / len(top)
            agg[k]["hit"] += any(set(d["patterns"]) - RT._GENERIC == gold for d in top)
            cov = set().union(*[set(d["patterns"]) for d in top]) & gold
            agg[k]["recall"] += len(cov) / len(gold) if gold else 1.0
        guides = [g for g, _ in retriever.rank_guides(r["question"], info, [d for d, _ in ranked[:3]])[:k_guides]]
        if guides:
            g_prec += sum(bool(set(g["patterns"]) & gold) for g in guides) / len(guides)
        gp = set().union(*[set(g["patterns"]) for g in guides]) if guides else set()
        g_rec += len(gp & gold) / len(gold) if gold else 1.0
    n = len(rows)
    return {"n": n,
            **{f"k{k}": {"mean_jaccard": round(v["jacc"] / n, 4), "hit_exact_patterns": round(v["hit"] / n, 4),
                         "pattern_recall": round(v["recall"] / n, 4)} for k, v in agg.items()},
            "guides_k3": {"precision": round(g_prec / n, 4), "pattern_recall": round(g_rec / n, 4)},
            "mean_latency_ms": round((time.perf_counter() - t0) / n * 1000, 3)}


def cue_quality(rows, infos):
    stats = {}
    for p in PT.PATTERNS:
        if p in ("select",):
            continue
        tp = fp = fn = 0
        for r in rows:
            info = infos[r["db_id"]]
            g = p in (PT.sql_patterns(r["gold_sql"]) | ({"text_number"} if info["text_number_fields"] else set()))
            c = p in PT.question_cues(r["question"], info)
            tp += g and c
            fp += c and not g
            fn += g and not c
        stats[p] = {"support": tp + fn, "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 3) if tp + fn else None}
    return stats


def main():
    rows = K.read_jsonl(C.DATASET)
    db_part, _ = C.load_split()
    corpus = C.load_corpus()
    collections = C.load_collections()
    infos = {db: PT.schema_info(db, collections) for db in db_part}
    train = [r for r in rows if db_part[r["db_id"]] == "train"]
    val = [r for r in rows if db_part[r["db_id"]] == "validation"]
    configs = [(m, None) for m in RT.EXAMPLE_METHODS if m != "hybrid_masked"]
    configs += [("hybrid_masked", w) for w in HYBRID_WEIGHTS]
    res = {"protocol": __doc__.split("Usage")[0].strip(), "corpus_sha256": C.sha256_file(C.CORPUS_DIR / "corpus.jsonl"),
           "train_lodo": {}, "validation": {}}
    for m, w in configs:
        name = m if w is None else f"{m}_w{w}"
        rt = RT.Retriever(corpus, m, hybrid_weight=w or 0.5)
        res["train_lodo"][name] = evaluate(rt, train, infos)
        res["validation"][name] = evaluate(rt, val, infos)
        print(f"{name:22s} train k3 jacc={res['train_lodo'][name]['k3']['mean_jaccard']:.3f} "
              f"hit={res['train_lodo'][name]['k3']['hit_exact_patterns']:.3f} | val k3 "
              f"jacc={res['validation'][name]['k3']['mean_jaccard']:.3f} hit={res['validation'][name]['k3']['hit_exact_patterns']:.3f}")
    best = max(res["train_lodo"], key=lambda n: (res["train_lodo"][n]["k3"]["mean_jaccard"], n))
    res["selected_on_train_lodo_by_k3_mean_jaccard"] = best
    res["cue_quality_train"] = cue_quality(train, infos)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("selected:", best)


if __name__ == "__main__":
    main()
