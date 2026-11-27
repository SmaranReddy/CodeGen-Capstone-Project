"""MQL-0010 database-level split and leakage audit (offline; reads data only; writes outputs/nl2mql_mql0010/preflight/).

Registered rules (fixed in this file BEFORE the draw was run; the draw inspects no outcome, question or SQL):
  * Universe: the 123 databases of approved corpus v2.1 (sorted ids). Spider dev (MQL-0006/0008, 20 DBs) and the
    16 MQL-0009 evaluation DBs are not in this universe by construction (verified below).
  * HOLDOUT: 18 DBs drawn with random.Random(20261119).sample(sorted_universe, 18).
  * DEV:     15 DBs drawn with random.Random(20261120).sample(remaining_105_sorted, 15). The other 90 DBs are
             the "build" DBs (retrieval-only during development).
  * Items:   spider-sql-v1 train.jsonl rows of those DBs with tier == "exec_ok" and no "empty_result" /
             "all_null_result" flag (degenerate gold results). All query families are kept, including
             set operations and subqueries that the converter excluded from the corpus.
  * Retrieval view for the HOLDOUT arms: approved corpus v2.1 minus every record of a holdout DB (a view; the
    corpus file is not modified). DEV arms use leave-one-database-out over (corpus minus holdout DBs).
  * Structure-predictor training data: spider-sql-v1 train.jsonl minus holdout DBs, minus any row whose
    question is an exact / near duplicate (token Jaccard >= 0.8) of a holdout question.
  * Leakage levels (as MQL-0009): L1 DB overlap STOP; L2 id overlap STOP; L3/L4 exact or normalised question,
    L5 token Jaccard >= 0.8: question-level retrieval exclusion pairs, identical for every arm; L6 identical
    normalised gold SQL: report only (gold never drives retrieval).

AMENDMENT A1 (2026-11-19, before any MQL-0010 prediction and before any holdout outcome exists; the DB draw is
unchanged): items are additionally restricted to spider-pool-v3 membership (train.jsonl ids; quarantine and
exclusion ids are therefore out). This is the label-audit filter that built the MQL-0009 639
(val_internal intersected with spider-pool-v3), so the holdout gold labels meet the same audit standard.
"""
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/nl2mql_mql0010/preflight"
CORPUS = ROOT / "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl"
CORPUS_SHA256 = "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf"
TRAIN = ROOT / "data/processed/spider_sql_v1/train.jsonl"
MQL0009_EVAL = ROOT / "kaggle/nl2mql_mql0009_retrieval_control/data/eval/eval_inputs.jsonl"
DEV1034 = ROOT / "data/processed/spider_sql_v1/dev_full.jsonl"
POOL_V3_TRAIN = ROOT / "data/processed/spider_pool_v3/train.jsonl"
HOLDOUT_SEED, HOLDOUT_N = 20261119, 18
DEV_SEED, DEV_N = 20261120, 15
NEAR_DUP = 0.8
_W = re.compile(r"[a-z0-9]+")


def rl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm_q(q):
    return " ".join(_W.findall(q.lower()))


def toks(q):
    return set(_W.findall(q.lower()))


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def norm_sql(s):
    return " ".join(re.sub(r"\s+", " ", s.strip().rstrip(";")).lower().split())


def eligible_item(r):
    return r["tier"] == "exec_ok" and not ({"empty_result", "all_null_result"} & set(r["flags"]))


def draw():
    corpus = rl(CORPUS)
    universe = sorted({r["db_id"] for r in corpus})
    holdout = sorted(random.Random(HOLDOUT_SEED).sample(universe, HOLDOUT_N))
    rest = sorted(set(universe) - set(holdout))
    dev = sorted(random.Random(DEV_SEED).sample(rest, DEV_N))
    build = sorted(set(rest) - set(dev))
    return corpus, universe, holdout, dev, build


def near_pairs(queries, pool):
    """queries/pool: [(id, question)] -> exact/normalised and Jaccard>=0.8 pairs (query_id, pool_id, level, j)."""
    by_norm = defaultdict(list)
    for pid, q in pool:
        by_norm[norm_q(q)].append(pid)
    pool_t = [(pid, toks(q)) for pid, q in pool]
    out = []
    for qid, q in queries:
        n = norm_q(q)
        for pid in by_norm.get(n, []):
            out.append((qid, pid, "L3/L4", 1.0))
        tq = toks(q)
        for pid, tp in pool_t:
            j = jac(tq, tp)
            if j >= NEAR_DUP and pid not in by_norm.get(n, []):
                out.append((qid, pid, "L5", round(j, 4)))
    return out


def main():
    assert sha_file(CORPUS) == CORPUS_SHA256, "approved corpus v2.1 changed"
    corpus, universe, holdout, dev, build = draw()
    train = rl(TRAIN)
    m9 = {r["db_id"] for r in rl(MQL0009_EVAL)}
    d1034 = {r["db_id"] for r in rl(DEV1034)}
    assert not set(universe) & m9 and not set(universe) & d1034, "universe overlaps a protected evaluation set"

    pool = {r["id"] for r in rl(POOL_V3_TRAIN)}

    def items(dbs):
        return [r for r in train if r["db_id"] in dbs and eligible_item(r) and r["id"] in pool]

    hold_items, dev_items = items(set(holdout)), items(set(dev))
    view = [r for r in corpus if r["db_id"] not in set(holdout)]
    # L1 / L2
    assert not {r["db_id"] for r in view} & set(holdout)
    assert not {r["id"] for r in view} & {r["id"] for r in hold_items}
    # L3-L5 holdout vs retrieval view (question-level retrieval exclusions)
    pairs = near_pairs([(r["id"], r["question"]) for r in hold_items],
                       [(r["id"], r["question"]) for r in view] + [(r["id"], r.get("question_raw") or r["question"]) for r in view])
    pairs = sorted(set(pairs))
    # L6 report only: identical normalised gold SQL in the view
    view_sql = defaultdict(list)
    for r in view:
        view_sql[norm_sql(r["sql"])].append(r["id"])
    l6 = [(r["id"], view_sql[norm_sql(r["sql"])][:3]) for r in hold_items if norm_sql(r["sql"]) in view_sql]
    # structure-predictor training data: train minus holdout DBs minus near duplicates of holdout questions
    cand = [r for r in train if r["db_id"] not in set(holdout)]
    dup_train = {p for _, p, _, _ in near_pairs([(r["id"], r["question"]) for r in hold_items],
                                                [(r["id"], r["question"]) for r in cand])}
    sp_train = [r["id"] for r in cand if r["id"] not in dup_train]
    rep = {
        "experiment": "MQL-0010", "stage": "split", "rules": __doc__, "amendments": ["A1 2026-11-19: spider-pool-v3 membership filter on items"],
        "pool_v3_train_sha256": sha_file(POOL_V3_TRAIN),
        "items_before_A1": {"holdout": sum(1 for r in train if r["db_id"] in set(holdout) and eligible_item(r)),
                            "dev": sum(1 for r in train if r["db_id"] in set(dev) and eligible_item(r))},
        "corpus_sha256": CORPUS_SHA256,
        "universe_n_dbs": len(universe),
        "holdout": {"seed": HOLDOUT_SEED, "dbs": holdout, "n_items": len(hold_items),
                    "items_by_db": dict(Counter(r["db_id"] for r in hold_items)),
                    "ids_sha256": sha_text("\n".join(r["id"] for r in hold_items))},
        "dev": {"seed": DEV_SEED, "dbs": dev, "n_items": len(dev_items),
                "items_by_db": dict(Counter(r["db_id"] for r in dev_items)),
                "ids_sha256": sha_text("\n".join(r["id"] for r in dev_items))},
        "build_dbs": build,
        "holdout_retrieval_view": {"n_records": len(view), "n_dbs": len({r['db_id'] for r in view}),
                                   "ids_sha256": sha_text("\n".join(r["id"] for r in view))},
        "leakage": {"L1_db_overlap": [], "L2_id_overlap": [],
                    "L3_L5_exclusion_pairs": [{"eval_id": a, "corpus_id": b, "level": l, "jaccard": j} for a, b, l, j in pairs],
                    "L6_identical_gold_sql_report_only": l6,
                    "protected_overlap": {"mql0009_eval_dbs": sorted(set(universe) & m9), "spider_dev_dbs": sorted(set(universe) & d1034)}},
        "structure_predictor_training": {"n_rows": len(sp_train), "dropped_near_dups_of_holdout": len(dup_train),
                                         "ids_sha256": sha_text("\n".join(sp_train))},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split_registration.json").write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    for name, rows in (("holdout_items.jsonl", hold_items), ("dev_items.jsonl", dev_items)):
        with open(OUT / name, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "db_id": r["db_id"], "question": r["question"]}, ensure_ascii=False) + "\n")
    (OUT / "structure_predictor_train_ids.txt").write_text("\n".join(sp_train) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: rep[k] for k in ("holdout", "dev", "holdout_retrieval_view", "structure_predictor_training")}, indent=1))
    print("exclusion pairs:", len(pairs), "L6:", len(l6))


if __name__ == "__main__":
    main()
