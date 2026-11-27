"""MQL-0009 post-experiment analysis (read-only, offline; no model, no database, no network).

Reads the registered, scored MQL-0009 artifacts and the frozen MQL-0009 package and writes derived tables to
outputs/nl2mql_mql0010/mql0009_post_analysis/. Nothing under outputs/nl2mql_mql0009_retrieval_control/ or
kaggle/nl2mql_mql0009_retrieval_control/ is written.

Evidence labels used in the output keys:
  * inference-time   only question text, target schema and corpus (what a deployed system can see)
  * DIAG             uses the evaluation items' gold SQL or outcomes; RETROSPECTIVE DESCRIPTION ONLY. Never used to
                     choose a threshold, weight or rule for MQL-0010.

Run:  python -B scripts/nl2mql_mql0010/analysis/analyze_mql0009.py
"""
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "kaggle/nl2mql_mql0009_retrieval_control"
RUN = ROOT / "outputs/nl2mql_mql0009_retrieval_control/scoring_runs/run_final"
OUT = ROOT / "outputs/nl2mql_mql0010/mql0009_post_analysis"
sys.path.insert(0, str(PKG / "src"))
import patterns as PT  # noqa: E402  (byte-identical to scripts/rag/patterns.py)
import retrieval as RT  # noqa: E402  (byte-identical to scripts/rag/retrieval.py)

ARMS = ["A_zero_shot", "B_random_s1", "B_random_s2", "B_random_s3", "B_random_s4", "B_random_s5", "C_fixed", "D_hybrid"]
SEEDS = ARMS[1:6]
UNSUPPORTED = {"anti_join", "nested", "intersect", "union"}   # zero records in approved corpus v2.1 (verified below)
IGNORE = {"select", "text_number"}
EXPECTED = {  # registered artifact hashes (ledger MQL-0009); the script refuses to analyse anything else
    "A_zero_shot": "1ed94e2cdc0842cae19fc139a0d93830f892472abdc2af6acc5e4d4b4d136477",
    "D_hybrid": "e7536b6b091694beaeb815aec7d591fac9031492eec7cb7ab62fbb4eccfd1037",
}


def rl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def sig(patterns):
    return frozenset(set(patterns) - IGNORE)


def stages(mql):
    """Top-level stage sequence; $lookup split into equality vs pipeline form."""
    if not isinstance(mql, dict) or not isinstance(mql.get("pipeline"), list):
        return None
    out = []
    for st in mql["pipeline"]:
        if not isinstance(st, dict) or len(st) != 1:
            out.append("?")
            continue
        k = next(iter(st))
        if k == "$lookup" and isinstance(st[k], dict) and "pipeline" in st[k]:
            k = "$lookup(pipeline)"
        out.append(k)
    return tuple(out)


def ops(obj):
    return PT.mql_operators(obj)


def holm(pvals):
    items = sorted(pvals.items(), key=lambda t: t[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = round(run, 5)
    return out


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    from math import comb
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_ci(vals, B=10000, seed=20261119):
    rng, n = random.Random(seed), len(vals)
    ms = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(ms[int(0.025 * B)], 4), round(ms[int(0.975 * B) - 1], 4)]


# ----------------------------------------------------------------------------------------------- loading
def load():
    for arm, h in EXPECTED.items():
        got = sha(RUN / "predictions" / arm / "predictions.jsonl")
        assert got == h, f"{arm}: predictions hash {got} != registered {h}"
    paired = {r["id"]: r for r in rl(RUN / "scoring/paired_per_question.jsonl")}
    evals = {a: {e["id"]: e for e in rl(RUN / f"scoring/arms/{a}/evaluation_results.jsonl")} for a in ARMS}
    rows = {a: {e["id"]: e for e in rl(RUN / f"scoring/arms/{a}/checkpoints/rows.partial.jsonl")} for a in ARMS}
    plans = {a: {p["id"]: p for p in rl(PKG / f"data/plans/plan_{a}.jsonl")} for a in ARMS}
    eval_in = {r["id"]: r for r in rl(PKG / "data/eval/eval_inputs.jsonl")}
    gold = {r["id"]: r for r in rl(ROOT / "data/processed/spider_sql_v1/val_internal.jsonl") if r["id"] in eval_in}
    assert len(gold) == len(eval_in) == len(paired) == 639
    corpus_rows = rl(PKG / "data/corpus/approved_corpus_v2_1.jsonl")
    corpus = {r["id"]: r for r in corpus_rows}
    ctx = json.loads((PKG / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
    tables = {t["db_id"]: t for t in json.loads((PKG / "data/context/tables_subset.json").read_text(encoding="utf-8"))}
    import mql0008_core as M8
    docs = M8.build_docs(corpus_rows, tables)   # exactly how the package derives retrieval documents
    frozen = rl(PKG / "data/corpus/corpus_features.jsonl")
    assert [{k: d[k] for k in ("doc_id", "db_id", "question", "masked_question", "patterns", "content")} for d in docs] == frozen
    cfg = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    return dict(paired=paired, evals=evals, rows=rows, plans=plans, eval_in=eval_in, gold=gold, docs=docs,
                corpus=corpus, ctx=ctx, tables=tables, cfg=cfg)


def ok(D, arm, i):
    return D["paired"][i][arm] == "correct"


# ----------------------------------------------------------------------------------------------- analyses
def a_stats(D):
    ids = sorted(D["paired"])
    acc = {a: sum(ok(D, a, i) for i in ids) for a in ARMS}
    d = [ok(D, "D_hybrid", i) - sum(ok(D, s, i) for s in SEEDS) / 5 for i in ids]
    out = {"accuracy": {a: {"correct": c, "n": 639, "pct": pct(c, 639)} for a, c in acc.items()},
           "H1_recomputed": {"diff": round(sum(d) / len(d), 4), "boot95": boot_ci(d)}}
    pv, pairs = {}, {}
    for s in SEEDS:
        b = sum(ok(D, "D_hybrid", i) and not ok(D, s, i) for i in ids)
        c = sum(ok(D, s, i) and not ok(D, "D_hybrid", i) for i in ids)
        pv[s] = mcnemar_exact(b, c)
        pairs[s] = {"D_only": b, "seed_only": c, "p": round(pv[s], 5)}
    out["D_vs_each_seed"] = pairs
    out["D_vs_each_seed_holm"] = holm(pv)
    nseeds = Counter(sum(ok(D, s, i) for s in SEEDS) for i in ids)
    out["questions_by_seeds_correct"] = dict(sorted(nseeds.items()))
    # decomposition of D - A into "any demonstrations" (Rbar - A) and "relevance" (D - Rbar)
    ra = [sum(ok(D, s, i) for s in SEEDS) / 5 - ok(D, "A_zero_shot", i) for i in ids]
    out["decomposition_pp"] = {"D_minus_A": pct(acc["D_hybrid"] - acc["A_zero_shot"], 639),
                               "Rbar_minus_A": round(100 * sum(ra) / 639, 2), "Rbar_minus_A_boot95": boot_ci(ra),
                               "D_minus_Rbar": round(100 * sum(d) / 639, 2)}
    return out


def gold_info(D):
    """DIAG: structural labels of the gold SQL per evaluation item."""
    g = {}
    for i, r in D["gold"].items():
        p = PT.sql_patterns(r["sql"])
        fam = ("set_op_or_subquery" if p & UNSUPPORTED else "join" if "join" in p else "single_table")
        g[i] = {"patterns": p, "sig": sig(p), "family": fam, "hardness": D["paired"][i]["hardness"]}
    return g


def a_failure_modes(D):
    out = {}
    for a in ARMS:
        c = Counter()
        for i, e in D["evals"][a].items():
            if e["outcome"] == "correct":
                continue
            det = e["detail"]
            if det == "wrong_row_count":
                rr = D["rows"][a].get(i)
                det = "wrong_row_count(empty)" if rr is not None and len(rr["rows"]) == 0 else "wrong_row_count(nonempty)"
            c[det] += 1
        out[a] = dict(c.most_common())
    return out


def a_transitions(D, G):
    """DIAG: A->D fixes / regressions by gold family and pattern; projection-only and `_id` fixes for every arm."""
    ids = sorted(D["paired"])
    fam = defaultdict(Counter)
    for i in ids:
        a, d = ok(D, "A_zero_shot", i), ok(D, "D_hybrid", i)
        k = "both_correct" if a and d else "fixed" if d else "broken" if a else "both_wrong"
        fam[G[i]["family"]][k] += 1
        fam["ALL"][k] += 1
    pat = {}
    for p in PT.PATTERNS:
        sub = [i for i in ids if p in G[i]["patterns"]]
        if not sub:
            continue
        f = sum(ok(D, "D_hybrid", i) and not ok(D, "A_zero_shot", i) for i in sub)
        b = sum(ok(D, "A_zero_shot", i) and not ok(D, "D_hybrid", i) for i in sub)
        pat[p] = {"n": len(sub), "fixed": f, "broken": b, "net": f - b}

    def proj_fix(arm):
        res = Counter()
        for i in ids:
            if not (ok(D, arm, i) and not ok(D, "A_zero_shot", i)):
                continue
            ea = D["evals"]["A_zero_shot"][i]
            res["fixes"] += 1
            if ea["detail"] != "wrong_column_count":
                continue
            res["A_wrong_column_count"] += 1
            sa, sb = stages(ea.get("mql")), stages(D["evals"][arm][i].get("mql"))
            if sa and sb and [s for s in sa if s != "$project"] == [s for s in sb if s != "$project"]:
                res["projection_only_change"] += 1
            cols = (D["rows"]["A_zero_shot"].get(i) or {}).get("columns") or []
            if "_id" in cols:
                res["A_result_has__id"] += 1
        return dict(res)

    proj = {a: proj_fix(a) for a in ARMS[1:]}
    # overlap of D's fixes with fixes of the random seeds / fixed arm
    fx = {a: {i for i in ids if ok(D, a, i) and not ok(D, "A_zero_shot", i)} for a in ARMS[1:]}
    anyrand = set().union(*(fx[s] for s in SEEDS))
    overlap = {"D_fixes": len(fx["D_hybrid"]), "also_fixed_by_any_random_seed": len(fx["D_hybrid"] & anyrand),
               "also_fixed_by_C": len(fx["D_hybrid"] & fx["C_fixed"]),
               "fixed_only_by_D": len(fx["D_hybrid"] - anyrand - fx["C_fixed"]),
               "also_fixed_by_each_seed": {s: len(fx["D_hybrid"] & fx[s]) for s in SEEDS},
               "seed_fixes": {s: len(fx[s]) for s in SEEDS}}
    # wrong_column_count across arms (inference-time observable result-shape failure)
    wcc = {a: sum(D["evals"][a][i]["detail"] == "wrong_column_count" for i in ids) for a in ARMS}
    idcol = {a: sum("_id" in ((D["rows"][a].get(i) or {}).get("columns") or []) for i in ids) for a in ARMS}
    return {"by_family_A_to_D": {k: dict(v) for k, v in fam.items()}, "by_pattern_A_to_D": pat,
            "fix_mechanism": proj, "fix_overlap": overlap, "wrong_column_count": wcc, "results_with__id_column": idcol}


def a_family_accuracy(D, G):
    ids = sorted(D["paired"])
    out = {}
    for key in ("family", "hardness"):
        groups = defaultdict(list)
        for i in ids:
            groups[G[i][key]].append(i)
        out[key] = {g: {"n": len(v), **{a: pct(sum(ok(D, a, i) for i in v), len(v)) for a in ARMS},
                        "Rbar": round(100 * sum(sum(ok(D, s, i) for s in SEEDS) / 5 for i in v) / len(v), 2)}
                    for g, v in sorted(groups.items())}
    return out


def a_ties(D):
    """inference-time: score ties in the hybrid ranking and the doc_id tie-break."""
    plan = D["plans"]["D_hybrid"]
    order = {d["doc_id"]: k for k, d in enumerate(sorted(D["docs"], key=lambda d: d["doc_id"]))}
    n = len(order)
    t34 = all10 = top1_eq_2 = top1_eq_1 = 0
    relpos_tied, relpos_all = [], []
    for p in plan.values():
        top = [s for _, s in p["ranked_top10"]]
        sel_s = p["selected_scores"]
        if len(top) >= 4 and top[2] == top[3]:
            t34 += 1
            relpos_tied += [order[x] / n for x in p["selected_ids"]]
        relpos_all += [order[x] / n for x in p["selected_ids"]]
        if len(set(top)) == 1:
            all10 += 1
        top1_eq_2 += top and top[0] == 2.0
        top1_eq_1 += top and top[0] == 1.0
        assert sel_s == sorted(sel_s, reverse=True)
    sel = Counter(x for p in plan.values() for x in p["selected_ids"])
    dbs = Counter(x for p in plan.values() for x in p["selected_db_ids"])
    return {"tie_at_rank3_4": t34, "all_top10_tied": all10, "top1_score_eq_2.0": top1_eq_2, "top1_score_eq_1.0": top1_eq_1,
            "mean_relative_corpus_position_selected_on_tied_items": round(statistics.mean(relpos_tied), 3) if relpos_tied else None,
            "mean_relative_corpus_position_selected_all": round(statistics.mean(relpos_all), 3),
            "distinct_docs_selected": len(sel), "max_reuse_of_one_doc": sel.most_common(1)[0],
            "top5_source_dbs": dbs.most_common(5), "n_items": len(plan)}


def a_redundancy(D):
    """inference-time: redundancy within the selected triple (pattern set / stage skeleton / database)."""
    docs = {d["doc_id"]: d for d in D["docs"]}
    out = {}
    for a in ("D_hybrid", "B_random_s1", "C_fixed"):
        same_sig = same_skel = same_db = 0
        for p in D["plans"][a].values():
            ss = [sig(docs[x]["patterns"]) for x in p["selected_ids"]]
            sk = [stages(docs[x]["mql"]) for x in p["selected_ids"]]
            same_sig += len(set(ss)) == 1
            same_skel += len(set(sk)) == 1
            same_db += len(set(p["selected_db_ids"])) < len(p["selected_db_ids"])
        out[a] = {"all3_same_pattern_set": same_sig, "all3_same_stage_skeleton": same_skel, "repeated_db": same_db}
    return out


def a_structural_relevance(D, G):
    """DIAG: does structural compatibility of the shown examples with the gold query predict correctness?

    (1) Observational, hybrid arm: D accuracy and D - Rbar split by whether any top-3 example has the gold pattern set.
    (2) Randomised, random arms: within a question, seeds differ only in which random examples were drawn, so the
        within-question contrast (matched seeds vs unmatched seeds) estimates the causal effect of showing a
        structurally matching example, free of question difficulty. Permutation test within question.
    The same within-question design is applied to lexical (BM25) similarity, the semantic channel of the retriever.
    """
    docs = {d["doc_id"]: d for d in D["docs"]}
    ids = sorted(D["paired"])
    retr = RT.Retriever(D["docs"], method="bm25_masked")
    infos = {db: PT.schema_info(db, D["ctx"]) for db in {D["eval_in"][i]["db_id"] for i in ids}}

    def matched(arm, i):
        return any(sig(docs[x]["patterns"]) == G[i]["sig"] for x in D["plans"][arm][i]["selected_ids"])

    def jac(arm, i):
        return max(RT.jaccard(G[i]["sig"], docs[x]["patterns"]) for x in D["plans"][arm][i]["selected_ids"])

    out = {}
    m = [i for i in ids if matched("D_hybrid", i)]
    u = [i for i in ids if not matched("D_hybrid", i)]

    def grp(v):
        return {"n": len(v), "A": pct(sum(ok(D, "A_zero_shot", i) for i in v), len(v)),
                "D": pct(sum(ok(D, "D_hybrid", i) for i in v), len(v)),
                "Rbar": round(100 * sum(sum(ok(D, s, i) for s in SEEDS) / 5 for i in v) / max(1, len(v)), 2),
                "D_fix": sum(ok(D, "D_hybrid", i) and not ok(D, "A_zero_shot", i) for i in v),
                "D_break": sum(ok(D, "A_zero_shot", i) and not ok(D, "D_hybrid", i) for i in v)}
    out["hybrid_exact_pattern_match_in_top3"] = {"matched": grp(m), "unmatched": grp(u)}
    out["exact_match_rate"] = {a: pct(sum(matched(a, i) for i in ids), 639) for a in ("D_hybrid", *SEEDS, "C_fixed")}
    # matched rate when the structure exists in the corpus at all
    corpus_sigs = Counter(sig(d["patterns"]) for d in D["docs"])
    avail = [i for i in ids if corpus_sigs[G[i]["sig"]] > 0]
    out["gold_signature_present_in_corpus"] = {"n": len(avail), "hybrid_matched": sum(matched("D_hybrid", i) for i in avail)}

    # (2) within-question randomised contrast
    lex = {}
    for i in ids:
        q, info = D["eval_in"][i]["question"], infos[D["eval_in"][i]["db_id"]]
        cand, idx = retr._index_for(info["db_id"])
        sc = idx.scores(PT.tokenize(retr.query_text(q, info)))
        top = max(sc) or 1.0
        pos = {d["doc_id"]: k for k, d in enumerate(cand)}
        lex[i] = {s: statistics.mean(sc[pos[x]] / top for x in D["plans"][s][i]["selected_ids"]) for s in SEEDS}

    def within(feature):
        diffs, n_q, rows = [], 0, []
        for i in ids:
            f = {s: feature(s, i) for s in SEEDS}
            hi = [s for s in SEEDS if f[s]]
            lo = [s for s in SEEDS if not f[s]]
            if not hi or not lo:
                continue
            n_q += 1
            ch = sum(ok(D, s, i) for s in hi) / len(hi)
            cl = sum(ok(D, s, i) for s in lo) / len(lo)
            diffs.append(ch - cl)
            rows.append((i, f))
        if not diffs:
            return {"n_questions": 0}
        obs = statistics.mean(diffs)
        rng, B, ge = random.Random(7), 5000, 0
        for _ in range(B):
            ds = []
            for i, f in rows:
                lab = [f[s] for s in SEEDS]
                rng.shuffle(lab)
                hi = [s for s, l in zip(SEEDS, lab) if l]
                lo = [s for s, l in zip(SEEDS, lab) if not l]
                ds.append(sum(ok(D, s, i) for s in hi) / len(hi) - sum(ok(D, s, i) for s in lo) / len(lo))
            ge += abs(statistics.mean(ds)) >= abs(obs) - 1e-12
        return {"n_questions": n_q, "mean_within_question_diff_pp": round(100 * obs, 2),
                "perm_p_two_sided": round((ge + 1) / (B + 1), 4)}

    out["randomised_within_question"] = {
        "exact_pattern_match": within(lambda s, i: matched(s, i)),
        "pattern_jaccard_ge_0.5": within(lambda s, i: jac(s, i) >= 0.5),
        "any_example_has_join_when_gold_single_table": within(
            lambda s, i: G[i]["family"] == "single_table" and any("join" in docs[x]["patterns"] for x in D["plans"][s][i]["selected_ids"])),
        "lexical_bm25_above_question_median": within(
            lambda s, i: lex[i][s] > statistics.median(lex[i].values())),
    }
    return out


def a_cue_accuracy(D):
    """inference-time property measured on Spider TRAIN questions (no evaluation item): question_cues vs gold SQL."""
    tr = [r for r in rl(ROOT / "data/processed/spider_sql_v1/train.jsonl")]
    sys.path.insert(0, str(PKG / "src"))
    import mql0008_core as M8
    tabs = {t["db_id"]: t for t in json.loads((ROOT / "data/raw/spider1/spider_data/tables.json").read_text(encoding="utf-8"))}
    infos = {}
    stats = defaultdict(Counter)
    exact = 0
    unsup = Counter()
    for r in tr:
        db = r["db_id"]
        if db not in infos:
            infos[db] = PT.schema_info(db, M8.schema_manifest_from_tables(tabs[db]))
        cues = PT.question_cues(r["question"], infos[db]) - IGNORE
        gp = PT.sql_patterns(r["sql"]) - IGNORE
        exact += (cues - {"where"}) == (gp - {"where"})
        for p in set(PT.PATTERNS) - IGNORE - {"select", "where"}:
            c, g = p in cues, p in gp
            stats[p]["tp" if c and g else "fp" if c else "fn" if g else "tn"] += 1
        cu, gu = bool(cues & UNSUPPORTED), bool(gp & UNSUPPORTED)
        unsup["tp" if cu and gu else "fp" if cu else "fn" if gu else "tn"] += 1
    per = {}
    for p, c in stats.items():
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
        per[p] = {"precision": round(prec, 3) if prec is not None else None, "recall": round(rec, 3) if rec is not None else None,
                  "support": c["tp"] + c["fn"], "fires": c["tp"] + c["fp"]}
    u = unsup
    return {"n_train_questions": len(tr), "exact_cue_set_agreement_pct": pct(exact, len(tr)), "per_pattern": per,
            "unsupported_family_cue": {"precision": round(u["tp"] / max(1, u["tp"] + u["fp"]), 3),
                                       "recall": round(u["tp"] / max(1, u["tp"] + u["fn"]), 3), **dict(u)}}


def a_idioms(D):
    """inference-time observable: copying of corpus idioms from shown examples into the output."""
    docs = {d["doc_id"]: d for d in D["docs"]}
    ids = sorted(D["paired"])
    checks = {"$lookup": lambda o, t: "$lookup" in o, "$unwind": lambda o, t: "$unwind" in o,
              "$facet": lambda o, t: "$facet" in o, "$expr": lambda o, t: "$expr" in o,
              "__jN alias": lambda o, t: re.search(r"__j\d", t) is not None,
              "type conversion": lambda o, t: bool(o & {"$toString", "$toInt", "$convert", "$toDouble", "$toLong"})}
    out = {}
    for a in ("A_zero_shot", "B_random_s1", "C_fixed", "D_hybrid"):
        res = {}
        for name, f in checks.items():
            in_ex, out_with, out_wo, n_with, n_wo, corr = 0, 0, 0, 0, 0, 0
            used = 0
            for i in ids:
                mq = D["evals"][a][i].get("mql")
                o, t = (ops(mq), json.dumps(mq)) if mq else (set(), "")
                ex = [docs[x] for x in D["plans"][a][i]["selected_ids"]]
                shown = any(f(ops(d["mql"]), json.dumps(d["mql"])) for d in ex)
                hit = f(o, t)
                used += hit
                corr += hit and ok(D, a, i)
                if shown:
                    n_with += 1
                    out_with += hit
                else:
                    n_wo += 1
                    out_wo += hit
            res[name] = {"outputs_using": used, "accuracy_when_used": pct(corr, used),
                         "use_rate_when_shown": pct(out_with, n_with), "use_rate_when_not_shown": pct(out_wo, n_wo),
                         "items_shown": n_with}
        out[a] = res
    return out


def fk_type_notes(db_id, collections, t):
    """Verbatim logic of scripts/rag/rag_prompt.fk_type_notes (schema-derived, inference-time)."""
    cols, tabs = t["column_names_original"], t["table_names_original"]
    notes = []
    for x, y in t["foreign_keys"]:
        a, b = f"{db_id}__{tabs[cols[x][0]]}", f"{db_id}__{tabs[cols[y][0]]}"
        ta = collections.get(a, {}).get("fields", {}).get(cols[x][1])
        tb = collections.get(b, {}).get("fields", {}).get(cols[y][1])
        if ta and tb and set(ta) != set(tb):
            notes.append((a, cols[x][1], tuple(ta), b, cols[y][1], tuple(tb)))
    return notes


def a_fk_types(D, G):
    """inference-time census + DIAG sizing: FK pairs whose observed BSON types differ, and items joining on them."""
    dbs = sorted({D["eval_in"][i]["db_id"] for i in D["eval_in"]})
    notes = {db: fk_type_notes(db, D["ctx"], D["tables"][db]) for db in dbs}
    notes = {k: v for k, v in notes.items() if v}
    ids = sorted(D["paired"])
    hit = []
    for i in ids:
        db = D["eval_in"][i]["db_id"]
        if db not in notes or "join" not in G[i]["patterns"]:
            continue
        s = D["gold"][i]["sql"].lower()
        for a, ca, _, b, cb, _ in notes[db]:
            ta, tb = a.split("__", 1)[1].lower(), b.split("__", 1)[1].lower()
            if re.search(rf"\b{re.escape(ta)}\b", s) and re.search(rf"\b{re.escape(tb)}\b", s) and ca.lower() in s and cb.lower() in s:
                hit.append(i)
                break
    acc = {a: sum(ok(D, a, i) for i in hit) for a in ARMS}
    conv = {a: sum(bool(ops(D["evals"][a][i].get("mql") or {}) & {"$toString", "$toInt", "$convert"}) for i in ids) for a in ARMS}
    return {"mismatched_fk_pairs": {k: [f"{a}.{ca} {list(ta)} -> {b}.{cb} {list(tb)}" for a, ca, ta, b, cb, tb in v] for k, v in notes.items()},
            "DIAG_items_joining_on_mismatched_pair": len(hit), "DIAG_correct_on_those": acc,
            "DIAG_items_by_db": dict(Counter(D["eval_in"][i]["db_id"] for i in hit)),
            "outputs_using_type_conversion": conv, "ids": hit}


def a_corpus(D, G):
    """inference-time corpus census (whole approved corpus v2.1) vs Spider train and the evaluation distribution."""
    rows = list(D["corpus"].values())
    sp = Counter(p for r in rows for p in PT.sql_patterns(r["sql"]))
    feats = Counter(f for r in rows for f in r.get("converter_features", []))
    opsc = Counter(o for r in rows for o in ops(r["mql"]))
    skel = Counter(stages(r["mql"]) for r in rows)
    sigs = Counter(sig(PT.sql_patterns(r["sql"])) for r in rows)
    tr = rl(ROOT / "data/processed/spider_sql_v1/train.jsonl")
    tr_sp = Counter(p for r in tr for p in PT.sql_patterns(r["sql"]))
    ev_sp = Counter(p for g in G.values() for p in g["patterns"])
    lookup_pipeline = sum("pipeline" in json.dumps(r["mql"]) and '"$lookup"' in json.dumps(r["mql"]) and '"let"' in json.dumps(r["mql"]) for r in rows)
    eval_sigs = Counter(g["sig"] for g in G.values())
    cover = sum(c for s, c in eval_sigs.items() if sigs[s] > 0)
    return {
        "n_records": len(rows), "n_dbs": len({r["db_id"] for r in rows}),
        "sql_patterns": {p: {"corpus": sp[p], "spider_train_7773": tr_sp[p], "eval_639_DIAG": ev_sp[p]} for p in PT.PATTERNS},
        "hardness": dict(Counter(r["hardness"] for r in rows)),
        "query_class": dict(Counter(r["query_class"] for r in rows).most_common()),
        "converter_features": dict(feats.most_common()),
        "mql_operators": dict(opsc.most_common()),
        "lookup_let_pipeline_records": lookup_pipeline,
        "distinct_stage_skeletons": len(skel), "top10_skeletons": [[" > ".join(k or ()), c] for k, c in skel.most_common(10)],
        "facet_count_wrapper_records": sum("$facet" in ops(r["mql"]) for r in rows),
        "distinct_pattern_signatures": len(sigs),
        "DIAG_eval_items_whose_gold_signature_occurs_in_corpus": cover,
        "duplicate_masked_questions": sum(c - 1 for c in Counter(d["masked_question"] for d in D["docs"]).values() if c > 1),
    }


def a_reproduce_hybrid(D):
    """inference-time: rebuild the hybrid ranking offline and check it reproduces the registered D plan selections
    (budget drops were zero, so the top-3 after exclusions is the selection)."""
    retr = RT.Retriever(D["docs"], method="hybrid_masked", hybrid_weight=D["cfg"]["retrieval"]["hybrid_weight"])
    excl = defaultdict(set)
    for x in D["cfg"].get("retrieval_exclusions", []):
        excl[x["eval_id"]].add(x["corpus_id"])
    infos = {}
    same = 0
    for i, p in D["plans"]["D_hybrid"].items():
        e = D["eval_in"][i]
        info = infos.setdefault(e["db_id"], PT.schema_info(e["db_id"], D["ctx"]))
        ranked = [d["doc_id"] for d, _ in retr.rank_examples(e["question"], info, qid=i) if d["doc_id"] not in excl[i]]
        same += ranked[:3] == p["selected_ids"]
    return {"reproduced_items": same, "n": len(D["plans"]["D_hybrid"]),
            "budget_drops": sum(bool(p["dropped_budget"]) for p in D["plans"]["D_hybrid"].values())}


def main():
    D = load()
    G = gold_info(D)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "inputs": {"run_dir": str(RUN.relative_to(ROOT)), "package": str(PKG.relative_to(ROOT)),
                   "analysis_json_sha256": sha(RUN / "scoring/analysis.json"),
                   "paired_sha256": sha(RUN / "scoring/paired_per_question.jsonl")},
        "stats": a_stats(D),
        "failure_modes": a_failure_modes(D),
        "family_accuracy_DIAG": a_family_accuracy(D, G),
        "transitions_DIAG": a_transitions(D, G),
        "ties": a_ties(D),
        "redundancy": a_redundancy(D),
        "structural_relevance_DIAG": a_structural_relevance(D, G),
        "cue_accuracy_train": a_cue_accuracy(D),
        "idioms": a_idioms(D),
        "fk_types": a_fk_types(D, G),
        "corpus": a_corpus(D, G),
        "reproduce_hybrid": a_reproduce_hybrid(D),
    }
    txt = json.dumps(res, indent=1, default=lambda o: sorted(o) if isinstance(o, (set, frozenset)) else str(o))
    (OUT / "mql0009_post_analysis.json").write_text(txt, encoding="utf-8", newline="\n")
    print(txt)


if __name__ == "__main__":
    main()
