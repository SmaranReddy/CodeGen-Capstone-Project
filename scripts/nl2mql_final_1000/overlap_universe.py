"""MQL-0015 pre-draw universe audit (CPU only, read-only).

Checks all 2,147 Spider test_full questions for overlap with every training / retrieval / evaluation source BEFORE the
1,000-question draw, because the result can change the eligible universe.

Levels (same definitions as MQL-0010/0013): L1 database overlap, L2 id overlap, L3 exact question, L4 normalised
question, L5 token Jaccard >= 0.8 (report). Each L3/L4/L5 hit is classified by which arm can SEE the matched record:
  R3 retrieval view (2,912 records, MQL-0011 index)  -> R3V and FT-R3V prompts (retrieval excludes L3-L5 pairs per item)
  FT training rows (sft_train 2,395, trained)        -> FT-R3V weights
  FT dev rows (506, loss-only, not trained)          -> nobody's weights
  approved corpus outside the view (533)             -> no arm
Writes outputs/nl2mql_final_1000/sampling/universe_overlap_audit.json. Opens no 65-item test file
(the expanded split is checked by database via its manifest).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/nl2mql_final_1000/sampling/universe_overlap_audit.json"
TEST_FULL = ROOT / "data/processed/spider_sql_v1/test_full.jsonl"
CORPUS = ROOT / "kaggle/nl2mql_mql0010_holdout/data/corpus/approved_corpus_v2_1.jsonl"
SPLIT_REG = ROOT / "kaggle/nl2mql_mql0010_holdout/registration/split_registration.json"
HANDOFF = ROOT / "outputs/finetuning_reuse/handoff_extract/ft_handoff_q3b_r16_q10/data"
JSONL_SETS = {
    "mql0010_holdout_items": ROOT / "outputs/nl2mql_mql0010/preflight/holdout_items.jsonl",
    "mql0010_dev_items": ROOT / "outputs/nl2mql_mql0010/preflight/dev_items.jsonl",
    "mql0012_x0_items": ROOT / "outputs/nl2mql_mql0012/split/x0_items.jsonl",
    "spider_sql_v1_train": ROOT / "data/processed/spider_sql_v1/train.jsonl",
    "spider_sql_v1_val_internal": ROOT / "data/processed/spider_sql_v1/val_internal.jsonl",
    "spider_sql_v1_dev_full": ROOT / "data/processed/spider_sql_v1/dev_full.jsonl",
}
EXPANDED_MANIFEST = ROOT / "data/splits_expanded/split_manifest.json"
_W = re.compile(r"[a-z0-9]+")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p):
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(q):
    return " ".join(_W.findall(q.lower()))


def toks(q):
    return set(_W.findall(q.lower()))


def jacc(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    test = jl(TEST_FULL)
    corpus = {r["id"]: r for r in jl(CORPUS)}
    reg = json.loads(SPLIT_REG.read_text(encoding="utf-8"))
    hold_dbs = set(reg["holdout"]["dbs"])
    view = {i for i, r in corpus.items() if r["db_id"] not in hold_dbs}
    ft_train = {r["id"] for r in jl(HANDOFF / "sft_train.prompt_completion.jsonl")}
    ft_dev = {r["id"] for r in jl(HANDOFF / "sft_dev.prompt_completion.jsonl")}
    assert len(corpus) == 3445 and len(view) == 2912 and len(ft_train) == 2395 and len(ft_dev) == 506

    def visibility(cid):
        v = []
        if cid in view:
            v.append("R3_retrieval_view")
        if cid in ft_train:
            v.append("FT_trained")
        if cid in ft_dev:
            v.append("FT_dev_loss_only")
        return v or ["no_arm"]

    # ---- database / id level against every source
    test_dbs = {r["db_id"] for r in test}
    test_ids = {r["id"] for r in test}
    db_level, id_level = {}, {}
    sources = {"approved_corpus_v2_1": list(corpus.values()),
               "ft_sft_train": [dict(id=i, db_id=corpus[i]["db_id"]) for i in ft_train],
               "ft_sft_dev": [dict(id=i, db_id=corpus[i]["db_id"]) for i in ft_dev]}
    for name, p in JSONL_SETS.items():
        sources[name] = jl(p)
    for name, rows in sources.items():
        db_level[name] = sorted(test_dbs & {r["db_id"] for r in rows})
        id_level[name] = sorted(test_ids & {r["id"] for r in rows})
    m = json.loads(EXPANDED_MANIFEST.read_text(encoding="utf-8"))
    db_level["splits_expanded_manifest(database_assignments)"] = sorted(test_dbs & set(m["database_assignments"]))
    db_level["mql0010_split_registration(holdout+dev+build)"] = sorted(
        test_dbs & (hold_dbs | set(reg["dev"]["dbs"]) | (set(corpus[i]["db_id"] for i in corpus) - hold_dbs - set(reg["dev"]["dbs"]))))

    # ---- question level against corpus (question + question_raw) and evaluation question sets
    q_sources = {"approved_corpus_v2_1": [(r["id"], r["question"]) for r in corpus.values()]
                 + [(r["id"], r["question_raw"]) for r in corpus.values() if r.get("question_raw") and r["question_raw"] != r["question"]]}
    for name in ("mql0010_holdout_items", "mql0010_dev_items", "mql0012_x0_items",
                 "spider_sql_v1_train", "spider_sql_v1_val_internal", "spider_sql_v1_dev_full"):
        q_sources[name] = [(r["id"], r["question"]) for r in sources[name] if r.get("question")]
    q_level = {}
    for name, qs in q_sources.items():
        pre = [(i, q, norm(q), toks(q)) for i, q in qs]
        hits = []
        for r in test:
            q, nq, tq = r["question"], norm(r["question"]), toks(r["question"])
            for cid, cq, cn, ct in pre:
                lv = "L3_exact" if q == cq else "L4_normalised" if nq == cn else None
                j = jacc(tq, ct)
                if lv is None and j >= 0.8:
                    lv = "L5_jaccard_ge_0.8"
                if lv:
                    h = {"eval_id": r["id"], "eval_db": r["db_id"], "match_id": cid, "level": lv, "jaccard": round(j, 4)}
                    if name == "approved_corpus_v2_1":
                        h["visible_to"] = visibility(cid)
                    hits.append(h)
        uniq = {}
        for h in hits:  # one record per (eval, match)
            k = (h["eval_id"], h["match_id"])
            if k not in uniq or h["level"] < uniq[k]["level"]:
                uniq[k] = h
        hits = sorted(uniq.values(), key=lambda h: (h["eval_id"], h["match_id"]))
        q_level[name] = {
            "n_pairs": len(hits),
            "n_eval_items": len({h["eval_id"] for h in hits}),
            "by_level": {lv: sum(h["level"] == lv for h in hits) for lv in ("L3_exact", "L4_normalised", "L5_jaccard_ge_0.8")},
            "pairs": hits,
        }
    corp = q_level["approved_corpus_v2_1"]["pairs"]
    exact_visible = [h for h in corp if h["level"] in ("L3_exact", "L4_normalised") and h["visible_to"] != ["no_arm"]]
    flagged = sorted(r["id"] for r in test if "question_in_train" in r["flags"])
    out = {
        "what": "MQL-0015 pre-draw universe audit over all 2,147 test_full questions",
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                          [TEST_FULL, CORPUS, SPLIT_REG, HANDOFF / "sft_train.prompt_completion.jsonl",
                           HANDOFF / "sft_dev.prompt_completion.jsonl", EXPANDED_MANIFEST, *JSONL_SETS.values()]},
        "n_test": len(test), "n_test_dbs": len(test_dbs),
        "L1_database_overlap": db_level,
        "L2_id_overlap": id_level,
        "question_level": q_level,
        "corpus_exact_or_normalised_hits_visible_to_an_arm": exact_visible,
        "question_in_train_flag": {
            "n": len(flagged), "ids": flagged,
            "matched_in_spider_train_question_check": sorted({h["eval_id"] for h in q_level["spider_sql_v1_train"]["pairs"]
                                                              if h["level"] != "L5_jaccard_ge_0.8"} & set(flagged)),
            "matched_in_corpus_exact_or_norm": sorted({h["eval_id"] for h in corp if h["level"] != "L5_jaccard_ge_0.8"} & set(flagged)),
        },
        "blocking": bool(exact_visible) or any(db_level.values()) or any(id_level.values()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    summ = {k: v for k, v in out.items() if k not in ("question_level", "inputs_sha256", "question_in_train_flag")}
    summ["question_level"] = {k: {kk: vv for kk, vv in v.items() if kk != "pairs"} for k, v in q_level.items()}
    summ["question_in_train_flag"] = {k: (v if k != "ids" else len(v)) for k, v in out["question_in_train_flag"].items()}
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    sys.exit(main())
