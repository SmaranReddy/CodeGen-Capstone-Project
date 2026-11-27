"""MQL-0012 separate descriptive finding: corpus coverage of query structures (no corpus change, no retrieval).

Counts each structure label in (a) the full approved corpus v2.1 (3,445, SQL-derived patterns), (b) the registered
2,912-record retrieval view, and (c) the gold signatures of the X0 and MQL-0011 dev items, and lists which labels or
label combinations make items unattainable (gold signature absent from the view).
Writes outputs/nl2mql_mql0012/coverage/coverage.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mql0012_common as M   # noqa: E402

C = M.C11


def main():
    import patterns as PT
    import structure as S
    corpus = C.read_jsonl(C.CORPUS)
    view, _ = C.load_view_rows()
    train = {r["id"]: r for r in C.read_jsonl(C.SPIDER_TRAIN)}
    sig = lambda sql: S.signature(PT.sql_patterns(sql))   # noqa: E731
    corpus_labels = Counter(l for r in corpus for l in sig(r["sql"]))
    view_sigs = Counter(sig(r["sql"]) for r in view)
    view_labels = Counter(l for r in view for l in sig(r["sql"]))
    out = {"corpus_n": len(corpus), "view_n": len(view),
           "label_counts_corpus": dict(corpus_labels.most_common()), "label_counts_view": dict(view_labels.most_common()),
           "labels_absent_from_corpus": sorted(set(S.STRUCT_LABELS) - set(corpus_labels))}
    for name, path in (("x0", M.X0_ITEMS), ("mql0011_dev", C.DEV_ITEMS)):
        items = [train[r["id"]] for r in C.read_jsonl(path)]
        un = [r for r in items if view_sigs[sig(r["sql"])] == 0]
        miss = Counter()
        for r in un:
            absent = sorted(l for l in sig(r["sql"]) if view_labels[l] == 0)
            miss[" + ".join(absent) if absent else "(labels present, combination absent)"] += 1
        out[name] = {"n_items": len(items), "n_unattainable": len(un),
                     "unattainable_pct": round(100 * len(un) / len(items), 2),
                     "label_counts_gold": dict(Counter(l for r in items for l in sig(r["sql"])).most_common()),
                     "unattainable_by_absent_labels": miss.most_common(),
                     "unattainable_top_signatures": [[" ".join(sorted(s)), n] for s, n in
                                                     Counter(sig(r["sql"]) for r in un).most_common(10)]}
    C.write_json(M.OUT / "coverage/coverage.json", out)
    print(json.dumps({k: out[k] for k in ("labels_absent_from_corpus",)}, indent=1))
    for n in ("x0", "mql0011_dev"):
        print(n, out[n]["n_unattainable"], out[n]["unattainable_by_absent_labels"][:6])


if __name__ == "__main__":
    main()
