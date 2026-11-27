"""Classify every failing Spider census query: python -B scripts/audit_sql2mql_v1/classify_spider_failures.py

Rules, applied in order (each uses the stored values of the original database):
  timeout             MongoDB maxTimeMS exceeded (error code 50): harness limit, not a verdict
  B1_empty_input      ungrouped aggregate, gold 1 row, MQL none, AND zero rows qualify on the SQL side
  literal_type        a column is compared with a literal whose type differs from every stored
                      non-NULL value of that column (e.g. TEXT '40' vs 50): SQLite coerces via
                      affinity, MongoDB compares types strictly
  text_numeric_agg    SUM/AVG/MIN/MAX over a column whose stored values are text
  B2_sum_all_null     gold NULL where MQL gives 0 (re-executed)
  divide_by_zero      MongoDB `$divide` error
  integer_division    SQL uses `/` (SQLite integer division vs the converter's real division)
  other               inspected by hand (listed)
Writes outputs/audit_sql2mql_v1/spider_failure_classes.jsonl.
"""
import collections

from sqlglot import exp

import common as K
import spider_census as SC


def classify(r):
    v, detail = r["explicit_null"]
    if v == "mongo_error" and detail.startswith("50:"):
        return "timeout"
    if v == "mongo_error" and "divide by zero" in detail:
        return "divide_by_zero"
    path, schema, rows = SC.load_db(r["db"])
    ast, _, _ = K.S.parse_sql(r["sql"])
    ast, _ = SC.canonicalise(ast, schema)
    an = K.S.analyze(ast)
    sel = [e.this if isinstance(e, exp.Alias) else e for e in ast.expressions]
    ungrouped_agg = any(isinstance(e, K.S.AGGS) for e in sel) and not ast.args.get("group")

    def col_types(c):
        t = an.aliases.get(c.table) if c.table else (an.base if not an.joins else None)
        if t is None:
            for tt in an.columns:
                if c.name in an.columns[tt]:
                    t = tt
        return {type(x.get(c.name)).__name__ for x in rows.get(t, []) if x.get(c.name) is not None}
    lit_mismatch = False
    for n in ast.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between, exp.In):
        for c in n.find_all(exp.Column):
            ty = col_types(c)
            for lit in n.find_all(exp.Literal):
                lt = "str" if lit.is_string else "num"
                if ty and ((lt == "str" and ty <= {"int", "float"}) or (lt == "num" and ty == {"str"})):
                    lit_mismatch = True
    text_agg = any(isinstance(a.this, exp.Column) and col_types(a.this) == {"str"}
                   for a in ast.find_all(exp.Sum, exp.Avg, exp.Min, exp.Max))
    gold = SC.real_db_rows(path, r["sql"])
    if ungrouped_agg and len(gold) == 1 and detail.startswith("rows 0 != 1"):
        # B1 only if NO row qualifies on the SQL side (otherwise the MQL's filter/join is at fault)
        try:
            qualifying = SC.real_db_rows(path, K.S.count_matching_sql(ast))[0][0]
        except Exception:  # noqa: BLE001 - e.g. a sqlsem UDF the plain database lacks
            qualifying = None
        if qualifying == 0:
            return "B1_empty_input"
    if lit_mismatch:
        return "literal_type"
    if text_agg:
        return "text_numeric_agg"
    tables = {t: rows[t] for t in an.columns}
    docs, err = K.run_mongo(r["mql"], tables)
    if docs is not None:
        got = K.rows_of(docs, K.out_columns(r["mql"], schema, an.base))
        g = sorted((tuple(K.norm(x) for x in row) for row in got), key=repr)
        e = sorted((tuple(K.norm(x) for x in row) for row in gold), key=repr)
        if len(g) == len(e) and all(all(a == b or (b is None and a == 0.0) for a, b in zip(x, y)) for x, y in zip(g, e)):
            return "B2_sum_all_null"
    if ast.find(exp.Div):
        return "integer_division"
    return "other"


def main():
    R = K.read_jsonl(SC.OUT)
    T = [r for r in R if r["status"] == "tested" and r["explicit_null"][0] in ("fail", "mongo_error")]
    out = []
    for r in sorted(T, key=lambda x: x["db"]):
        flagged = bool(r["diagnostics"]["mixed_type_columns"] or r["diagnostics"]["join_key_type_mismatch"])
        c = "join_key_type" if r["diagnostics"]["join_key_type_mismatch"] else None
        c = c or classify(r)
        if c in ("other",) and r["diagnostics"]["mixed_type_columns"]:
            c = "mixed_type_column"
        out.append({"id": r["id"], "db": r["db"], "stratum": "type_flagged" if flagged else "clean",
                    "class": c, "detail": r["explicit_null"][1][:120], "sql": r["sql"]})
    K.write_jsonl(K.OUT / "spider_failure_classes.jsonl", out)
    for s in ("clean", "type_flagged"):
        print(s, collections.Counter(x["class"] for x in out if x["stratum"] == s))
    empty_dbs = collections.Counter(x["db"] for x in out if x["class"] == "B1_empty_input")
    print("B1 by db (top):", empty_dbs.most_common(8))
    for x in out:
        if x["class"] == "other":
            print("OTHER", x["id"], x["detail"][:50], "|", x["sql"][:170])


if __name__ == "__main__":
    main()
