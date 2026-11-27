"""Draw the semantic-review sample: python -B scripts/audit_sql2mql_v1/semantic_sample.py

Seeded simple random sample (seed "semantic-20261025") of 120 of the 1,584 Kaggle-derived
labels, independent of data/finetune/ft_v1/semantic_audit_sample.json. Writes
outputs/audit_sql2mql_v1/semantic_sample.jsonl; the reviewer's judgements are recorded in
semantic_review.jsonl (one line per id).
"""
import json
import random

import common as K
import kaggle_census as KC

N = 120


def main():
    labels = sorted(KC.labels(), key=lambda r: r["id"])
    sample = random.Random("semantic-20261025").sample(labels, N)
    schema = json.loads((K.ROOT / "data/interim/finetune_v1/schema.json").read_text(encoding="utf-8"))["tables"]
    out = []
    for r in sample:
        ast, _, _ = K.S.parse_sql(r["reference"]["sql"])
        an = K.S.analyze(ast)
        out.append({"id": r["id"], "question": r["question"], "sql": r["reference"]["sql"],
                    "mql": K.C.compact(r["target_mql"]),
                    "schema": {t: schema[t] for t in an.columns}})
    K.write_jsonl(K.OUT / "semantic_sample.jsonl", out)
    for i, x in enumerate(out):
        print(f"[{i}] {x['id']}\n Q: {x['question']}\n S: {x['sql']}\n M: {x['mql']}\n T: {json.dumps(x['schema'])}")


if __name__ == "__main__":
    main()
