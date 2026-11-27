"""Development evidence for component S: DB-grouped 5-fold cross-validation of the structure predictor against
the MQL-0008/0009 regex cues, on the registered structure-predictor training rows only (Spider train minus the
MQL-0010 holdout DBs; no MQL-0009 / Spider-dev / holdout item is touched). Writes
outputs/nl2mql_mql0010/dev_evidence/structure_predictor_cv.json.
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0010/src"))
sys.path.insert(1, str(ROOT / "kaggle/nl2mql_mql0009_retrieval_control/src"))
import patterns as PT          # noqa: E402
import mql0008_core as M8      # noqa: E402
import structure as S          # noqa: E402

OUT = ROOT / "outputs/nl2mql_mql0010/dev_evidence"


def rl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def load_training():
    ids = set((ROOT / "outputs/nl2mql_mql0010/preflight/structure_predictor_train_ids.txt").read_text(encoding="utf-8").split())
    rows = [r for r in rl(ROOT / "data/processed/spider_sql_v1/train.jsonl") if r["id"] in ids]
    tabs = {t["db_id"]: t for t in json.loads((ROOT / "data/raw/spider1/spider_data/tables.json").read_text(encoding="utf-8"))}
    infos = {}
    out = []
    for r in rows:
        db = r["db_id"]
        if db not in infos:
            infos[db] = PT.schema_info(db, M8.schema_manifest_from_tables(tabs[db]))
        out.append((r, infos[db]))
    return out


def fold_of(db, k=5):
    return int(hashlib.sha256(db.encode()).hexdigest(), 16) % k


def main():
    data = load_training()
    gold = [S.signature(PT.sql_patterns(r["sql"])) for r, _ in data]
    feats = [S.features(r["question"], info) for r, info in data]
    regex = [S.signature(PT.question_cues(r["question"], info)) for r, info in data]
    pred, unsup_p = [None] * len(data), [None] * len(data)
    for k in range(5):
        tr = [i for i, (r, _) in enumerate(data) if fold_of(r["db_id"]) != k]
        te = [i for i, (r, _) in enumerate(data) if fold_of(r["db_id"]) == k]
        m = S.StructurePredictor().fit((feats[i], gold[i]) for i in tr)
        for i in te:
            p = m.predict_proba(feats[i])
            pred[i] = S.predicted_signature(p)
            unsup_p[i] = S.unsupported_probability(p)

    def summary(sigs):
        per = {}
        for lab in S.STRUCT_LABELS:
            tp = sum(lab in s and lab in g for s, g in zip(sigs, gold))
            fp = sum(lab in s and lab not in g for s, g in zip(sigs, gold))
            fn = sum(lab not in s and lab in g for s, g in zip(sigs, gold))
            pr = tp / (tp + fp) if tp + fp else None
            rc = tp / (tp + fn) if tp + fn else None
            f1 = 2 * pr * rc / (pr + rc) if pr and rc else 0.0
            per[lab] = {"precision": round(pr, 3) if pr is not None else None,
                        "recall": round(rc, 3) if rc is not None else None, "f1": round(f1, 3), "support": tp + fn}
        exact = sum(s == g for s, g in zip(sigs, gold))
        exact_nw = sum((s - {"where"}) == (g - {"where"}) for s, g in zip(sigs, gold))
        return {"exact_signature": exact, "exact_signature_pct": round(100 * exact / len(gold), 2),
                "exact_ignoring_where_pct": round(100 * exact_nw / len(gold), 2),
                "macro_f1": round(sum(v["f1"] for v in per.values()) / len(per), 3), "per_label": per}

    def unsup_gate(th):
        tp = sum(p >= th and bool(g & S.UNSUPPORTED) for p, g in zip(unsup_p, gold))
        fp = sum(p >= th and not g & S.UNSUPPORTED for p, g in zip(unsup_p, gold))
        fn = sum(p < th and bool(g & S.UNSUPPORTED) for p, g in zip(unsup_p, gold))
        return {"threshold": th, "precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3), "fires": tp + fp}

    rgx_unsup = {"precision": None, "recall": None}
    tp = sum(bool(s & S.UNSUPPORTED) and bool(g & S.UNSUPPORTED) for s, g in zip(regex, gold))
    fp = sum(bool(s & S.UNSUPPORTED) and not g & S.UNSUPPORTED for s, g in zip(regex, gold))
    fn = sum(not s & S.UNSUPPORTED and bool(g & S.UNSUPPORTED) for s, g in zip(regex, gold))
    rgx_unsup = {"precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3), "fires": tp + fp}
    res = {"n": len(data), "n_dbs": len({r['db_id'] for r, _ in data}), "folds": "sha256(db_id) % 5 (DB-grouped)",
           "regex_cues": summary(regex), "structure_nb": summary(pred),
           "unsupported_gate": {"regex_cues": rgx_unsup, "nb": [unsup_gate(t) for t in (0.5, 0.7, 0.9)]},
           "gold_signature_distribution_top10": [[sorted(k), v] for k, v in Counter(gold).most_common(10)]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "structure_predictor_cv.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: (v if k not in ("regex_cues", "structure_nb") else {kk: vv for kk, vv in v.items() if kk != "per_label"})
                      for k, v in res.items()}, indent=1))
    for lab in S.STRUCT_LABELS:
        print(lab, res["regex_cues"]["per_label"][lab], res["structure_nb"]["per_label"][lab])


if __name__ == "__main__":
    main()
