"""Re-run every failing (label, family) of kaggle_census and classify the row difference.

    python -B scripts/audit_sql2mql_v1/classify_kaggle_failures.py

Classes (decided from the actual rows, not from the SQL text):
  B1_empty_input_aggregate  gold has exactly one row, MQL returns none (ungrouped aggregate
                            over zero input rows: `$group: {_id: null}` emits no document);
  B2_sum_all_null           same rows except gold NULL where MQL returns 0 (SUM over only
                            NULL inputs; `$sum` ignores NULL and returns 0);
  B3_missing_field_error    MongoDB error on the missing-field representation;
  other                     anything else (inspected by hand).
Writes outputs/audit_sql2mql_v1/kaggle_failure_classes.jsonl.
"""
import collections
import hashlib
import json

import audit_fixtures as AF
import common as K
import kaggle_census as KC


def classify(gold, got, qualifying=None):
    if gold is None or got is None:
        return "other"
    if len(gold) == 1 and len(got) == 0:
        return "B1_empty_input_aggregate" if qualifying == 0 else "other"
    g = sorted(gold, key=repr)
    m = sorted(got, key=repr)
    if len(g) == len(m):
        def eqish(a, b):
            return a == b or (a is None and b == 0.0)
        # align rows by their non-aggregate values: compare as multisets with NULL->0 allowed
        rest = list(m)
        for row in g:
            hit = next((x for x in rest if len(x) == len(row) and all(eqish(a, b) for a, b in zip(row, x))), None)
            if hit is None:
                return "other"
            rest.remove(hit)
        return "B2_sum_all_null"
    return "other"


def main():
    schema = json.loads((K.ROOT / "data/interim/finetune_v1/schema.json").read_text(encoding="utf-8"))["tables"]
    by_id = {r["id"]: r for r in KC.labels()}
    out = []
    for rec in K.read_jsonl(KC.OUT):
        bad = [(k, v) for k, v in rec.get("families", {}).items() if v in ("fail", "mongo_error")]
        if not bad:
            continue
        r = by_id[rec["id"]]
        ast, _, _ = K.S.parse_sql(r["reference"]["sql"])
        an = K.S.analyze(ast)
        sch = {t: schema[t] for t in an.columns}
        clock = K.server_clock()
        seed = int(hashlib.sha256(r["id"].encode()).hexdigest()[:8], 16)
        fam = AF.families(ast, an, sch, clock, seed)
        for k, v in bad:
            name, _, rep = k.partition("@")
            missing = rep == "missing"
            if v == "mongo_error":
                cls = "B3_missing_field_error" if missing else "other_mongo_error"
                out.append({"id": rec["id"], "family": k, "class": cls, "sql": r["reference"]["sql"],
                            "detail": rec["details"].get(k, "")[:160]})
                continue
            tables = fam[name]
            gold = [tuple(K.norm(x) for x in row) for row in AF.gold(ast, tables, sch, clock)]
            docs, err = K.run_mongo(r["target_mql"], tables, missing=missing)
            got = None if docs is None else [tuple(K.norm(x) for x in row) for row in
                                             K.rows_of(docs, K.out_columns(r["target_mql"], sch, an.base))]
            con = K.S.sqlite_db(tables, {t: sch[t] for t in tables}, clock)
            try:
                qualifying = K.S.run_sqlite(con, K.S.count_matching_sql(ast))[1][0][0]
            finally:
                con.close()
            cls = classify(gold, got, qualifying)
            if missing and cls == "other":
                cls = "B4_missing_field_semantics"
            out.append({"id": rec["id"], "family": k, "class": cls, "sql": r["reference"]["sql"],
                        "gold_head": gold[:3], "got_head": (got or [])[:3]})
    K.write_jsonl(K.OUT / "kaggle_failure_classes.jsonl", out)
    per_label = collections.defaultdict(set)
    for x in out:
        per_label[(x["id"], "@missing" in x["family"])].add(x["class"])
    print("family-level:", collections.Counter((("missing" if "@" in x["family"] else "explicit"), x["class"]) for x in out))
    print("label-level explicit:", collections.Counter("+".join(sorted(v)) for (i, m), v in per_label.items() if not m))
    print("label-level missing:", collections.Counter("+".join(sorted(v)) for (i, m), v in per_label.items() if m))
    for x in out:
        if x["class"].startswith("other"):
            print("OTHER", x["id"], x["family"], x["sql"][:120], x.get("gold_head"), x.get("got_head"), x.get("detail", ""))


if __name__ == "__main__":
    main()
