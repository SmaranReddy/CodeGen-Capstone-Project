"""Development evidence for component R (offline retrieval proxy; no model, no database).

Dev questions: the 971 registered MQL-0010 dev items (15 corpus DBs). Retrieval pool: approved corpus v2.1 minus
the 18 holdout DBs (2,912 records), leave-the-question's-own-database-out (the retriever's built-in filter).
The structure predictor is refit WITHOUT the dev DBs for this proxy (so the dev questions are unseen by it).

Proxy metrics (gold SQL of DEV questions only; dev is a development split, not an evaluation set):
  hit@3   some top-3 example has exactly the question's gold structure signature
  hit@1   the top-1 example does
  mJ@3    mean over questions of the best structure-label Jaccard in the top 3
  ties34  questions whose rank-3 and rank-4 scores tie (selection decided by the tie-break)
reported overall and on the subset whose gold signature exists in the pool (the attainable subset).
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0010/src"))
sys.path.insert(1, str(ROOT / "kaggle/nl2mql_mql0009_retrieval_control/src"))
import patterns as PT           # noqa: E402
import retrieval as RT          # noqa: E402
import mql0008_core as M8       # noqa: E402
import structure as S           # noqa: E402
import retriever_v2 as R2       # noqa: E402

PRE = ROOT / "outputs/nl2mql_mql0010/preflight"
OUT = ROOT / "outputs/nl2mql_mql0010/dev_evidence"


def rl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    reg = json.loads((PRE / "split_registration.json").read_text(encoding="utf-8"))
    hold, dev = set(reg["holdout"]["dbs"]), set(reg["dev"]["dbs"])
    tabs = {t["db_id"]: t for t in json.loads((ROOT / "data/raw/spider1/spider_data/tables.json").read_text(encoding="utf-8"))}
    corpus = [r for r in rl(ROOT / "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl") if r["db_id"] not in hold]
    docs = M8.build_docs(corpus, tabs)
    train_ids = set((PRE / "structure_predictor_train_ids.txt").read_text(encoding="utf-8").split())
    train = {r["id"]: r for r in rl(ROOT / "data/processed/spider_sql_v1/train.jsonl")}
    infos = {}

    def info(db):
        if db not in infos:
            infos[db] = PT.schema_info(db, M8.schema_manifest_from_tables(tabs[db]))
        return infos[db]

    fit = [(S.features(train[i]["question"], info(train[i]["db_id"])), S.signature(PT.sql_patterns(train[i]["sql"])))
           for i in sorted(train_ids) if train[i]["db_id"] not in dev]
    pred = S.StructurePredictor().fit(fit)
    items = [train[r["id"]] for r in rl(PRE / "dev_items.jsonl")]
    pool_sigs = Counter(S.signature(d["patterns"]) for d in docs)

    configs = {"baseline (MQL-0008/0009)": dict(structure_scoring="baseline", tie_break="doc_id"),
               "baseline + lexical tie-break": dict(structure_scoring="baseline", tie_break="lexical")}
    for b in (0.25, 0.5, 1.0, 2.0):
        configs[f"nb beta={b}"] = dict(structure_scoring="nb", beta=b, tie_break="lexical")
    configs["baseline + signature diversity"] = dict(structure_scoring="baseline", tie_break="doc_id", diverse=True)
    for b in (0.5, 1.0, 2.0):
        configs[f"nb beta={b} + signature diversity"] = dict(structure_scoring="nb", beta=b, tie_break="lexical", diverse=True)
    res = {"n_dev_items": len(items), "pool_records": len(docs),
           "attainable_items": sum(pool_sigs[S.signature(PT.sql_patterns(r["sql"]))] > 0 for r in items)}
    for name, kw in configs.items():
        kw = dict(kw)
        diverse = kw.pop("diverse", False)
        R = R2.StructureAwareRetriever(docs, predictor=pred if kw["structure_scoring"] == "nb" else None, **kw)
        c = Counter()
        mj = []
        for r in items:
            g = S.signature(PT.sql_patterns(r["sql"]))
            ranked = R.rank_examples(r["question"], info(r["db_id"]), qid=r["id"])
            sel = R2.select_diverse(ranked, 3) if diverse else ranked[:3]
            top3 = [S.signature(d["patterns"]) for d, _, _ in sel]
            att = pool_sigs[g] > 0
            h3, h1 = g in top3, top3[:1] == [g]
            c["hit3"] += h3
            c["hit1"] += h1
            c["att"] += att
            c["hit3_att"] += h3 and att
            c["ties34"] += len(ranked) > 3 and ranked[2][1] == ranked[3][1]
            c["same_sig_triple"] += len(set(top3)) == 1
            mj.append(max(RT.jaccard(g, s) for s in top3))
        n = len(items)
        res[name] = {"hit@3": round(100 * c["hit3"] / n, 2), "hit@1": round(100 * c["hit1"] / n, 2),
                     "hit@3_attainable": round(100 * c["hit3_att"] / max(1, c["att"]), 2),
                     "mJ@3": round(sum(mj) / n, 4), "ties34": c["ties34"], "all3_same_signature": c["same_sig_triple"]}
        print(name, res[name])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retrieval_dev_proxy.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
