"""Recompute every figure quoted in docs/sql2mql_independent_audit.md from the audit outputs.

    python -B scripts/audit_sql2mql_v1/summarize.py   -> outputs/audit_sql2mql_v1/summary.json
"""
import collections
import json

import common as K


def adversarial():
    R = K.read_jsonl(K.OUT / "adversarial.jsonl")
    conv = [r for r in R if r["converter"] == "converted"]
    wrong = [r for r in conv if not r["explicit_null"]["pass"]]
    return {
        "cases": len(R), "converted": len(conv), "unsupported": len(R) - len(conv),
        "sqlite_agrees_with_hand_expectation": sum(bool(r["sqlite_agrees"]) for r in R),
        "pass_explicit_null": sum(r["explicit_null"]["pass"] for r in conv),
        "pass_missing_field": sum(r["missing_field"]["pass"] for r in conv),
        "wrong_explicit_null": [r["id"] for r in wrong],
        "wrong_and_existing_verifier_accepts": [r["id"] for r in wrong
                                                if r["existing_verifier"]["status"] == "fixture_equivalence_verified"],
        "wrong_and_existing_verifier_rejects": [r["id"] for r in wrong
                                                if r["existing_verifier"]["status"] != "fixture_equivalence_verified"],
        "fail_missing_only": [r["id"] for r in conv if r["explicit_null"]["pass"] and not r["missing_field"]["pass"]],
        "udf_checks_pass": sum(x["pass"] for x in json.load(open(K.OUT / "adversarial_udf.json", encoding="utf-8"))),
    }


def kaggle():
    R = K.read_jsonl(K.OUT / "kaggle_census.jsonl")
    n = len(R)
    ex = collections.Counter(r["verdict_explicit_null"] for r in R)
    mi = collections.Counter(r["verdict_missing_field"] for r in R)
    cls = K.read_jsonl(K.OUT / "kaggle_failure_classes.jsonl")
    per = collections.defaultdict(set)
    for x in cls:
        per[(x["id"], "@" in x["family"])].add(x["class"])
    b1 = {i for (i, m), v in per.items() if not m and "B1_empty_input_aggregate" in v}
    b2 = {i for (i, m), v in per.items() if not m and "B2_sum_all_null" in v}
    other = {i for (i, m), v in per.items() if not m and any(c.startswith("other") for c in v)}
    # failures that occur on non-degenerate data: rand*/dup families (not empty/zero/nullcols)
    nondegenerate = {x["id"] for x in cls if "@" not in x["family"] and x["family"].startswith(("rand", "dup"))}
    return {"labels": n, "explicit_null": dict(ex), "missing_field": dict(mi),
            "explicit_fail_rate": ex["fail"] / n, "explicit_fail_wilson95": K.wilson(ex["fail"], n),
            "missing_fail_rate": mi["fail"] / n, "missing_fail_wilson95": K.wilson(mi["fail"], n),
            "B1_labels": len(b1), "B2_labels": len(b2), "B1_and_B2": len(b1 & b2), "other_labels": len(other),
            "labels_failing_on_nondegenerate_random_data": len(nondegenerate),
            "labels_failing_on_nondegenerate_random_data_ids": sorted(nondegenerate),
            "missing_classes": dict(collections.Counter("+".join(sorted(v)) for (i, m), v in per.items() if m)),
            "reconvert_identical": sum(r.get("reconvert_identical") is True for r in R)}


def mutants():
    R = K.read_jsonl(K.OUT / "mql_mutants.jsonl")
    census = {x["id"]: x.get("families", {}) for x in K.read_jsonl(K.OUT / "kaggle_census.jsonl")}
    for r in R:  # an audit kill counts only on a family where the unmutated label itself passes
        ok = {f for f, v in census.get(r["id"], {}).items() if v == "pass" and "@" not in f}
        for m in r.get("mutants", []):
            if "killed_audit" in m:
                m["killed_audit"] = [f for f in m["killed_audit"] if f in ok]
    M = [m for r in R for m in r.get("mutants", [])]
    by = {}
    for op in sorted({m["op"] for m in M}):
        ms = [m for m in M if m["op"] == op]
        by[op] = {"n": len(ms), "killed_existing": sum(m["killed_existing"] for m in ms),
                  "survivor_killed_by_audit": sum((not m["killed_existing"]) and bool(m.get("killed_audit")) for m in ms),
                  "survivor_unkilled": sum((not m["killed_existing"]) and not m.get("killed_audit") for m in ms)}
    ke = sum(m["killed_existing"] for m in M)
    ka = sum((not m["killed_existing"]) and bool(m.get("killed_audit")) for m in M)
    labels_hit = sum(any((not m["killed_existing"]) and m.get("killed_audit") for m in r.get("mutants", [])) for r in R)
    return {"labels": len(R), "mutants": len(M), "killed_existing": ke,
            "survivors": len(M) - ke, "survivors_killed_by_audit": ka,
            "existing_kill_rate_of_known_nonequivalent": ke / (ke + ka) if ke + ka else None,
            "labels_with_accepted_nonequivalent_mutant": labels_hit,
            "labels_with_accepted_nonequivalent_mutant_wilson95": K.wilson(labels_hit, len(R)),
            "by_operator": by}


def spider():
    R = K.read_jsonl(K.OUT / "spider_census.jsonl")
    st = collections.Counter(r["status"] for r in R)
    T = [r for r in R if r["status"] == "tested"]
    flagged = lambda r: bool(r["diagnostics"]["mixed_type_columns"] or r["diagnostics"]["join_key_type_mismatch"])  # noqa: E731
    clean = [r for r in T if not flagged(r)]
    dirty = [r for r in T if flagged(r)]

    def ver(rs, rep):
        return dict(collections.Counter(r[rep][0] for r in rs))

    def rate(rs, rep):
        c = collections.Counter(r[rep][0] for r in rs)
        decided = c["pass"] + c["fail"] + c.get("mongo_error", 0)
        bad = c["fail"] + c.get("mongo_error", 0)
        return {"decided": decided, "wrong": bad, "wrong_rate": bad / decided if decided else None,
                "wilson95": K.wilson(bad, decided)}
    return {"unique_db_sql": len(R), "status": dict(st),
            "identifier_case_rewritten": sum(bool(r.get("identifier_case_rewritten")) for r in T),
            "tested": len(T), "clean": len(clean), "type_flagged": len(dirty),
            "explicit_all": ver(T, "explicit_null"), "explicit_clean": ver(clean, "explicit_null"),
            "explicit_flagged": ver(dirty, "explicit_null"),
            "missing_clean": ver(clean, "missing_field"),
            "rate_explicit_clean": rate(clean, "explicit_null"), "rate_explicit_all": rate(T, "explicit_null"),
            "rate_missing_clean": rate(clean, "missing_field"),
            "failure_classes": spider_classes(),
            "unsupported_reasons_top": collections.Counter(
                r["reason"].split(":")[0] + ":" + r["reason"].split(":")[1].split(";")[0] if ":" in r["reason"] else r["reason"]
                for r in R if r["status"] == "unsupported").most_common(12)}


def spider_classes():
    C = K.read_jsonl(K.OUT / "spider_failure_classes.jsonl")
    R = [r for r in K.read_jsonl(K.OUT / "spider_census.jsonl") if r["status"] == "tested"]
    out = {}
    for st in ("clean", "type_flagged"):
        cls = collections.Counter(x["class"] for x in C if x["stratum"] == st)
        rs = [r for r in R if bool(r["diagnostics"]["mixed_type_columns"] or r["diagnostics"]["join_key_type_mismatch"])
              == (st == "type_flagged")]
        passed = sum(r["explicit_null"][0] == "pass" for r in rs)
        wrong = sum(v for k, v in cls.items() if k != "timeout")
        decided = passed + wrong
        b1 = cls["B1_empty_input"]
        coerc = cls["literal_type"] + cls["text_numeric_agg"] + cls["join_key_type"] + cls["mixed_type_column"]
        logic = cls["other"]
        out[st] = {"classes": dict(cls), "pass": passed, "decided_excl_ties_timeouts": decided, "wrong": wrong,
                   "wrong_rate": wrong / decided, "wrong_wilson95": K.wilson(wrong, decided),
                   "B1": b1, "B1_rate": b1 / decided, "B1_wilson95": K.wilson(b1, decided),
                   "coercion": coerc, "coercion_rate": coerc / decided, "coercion_wilson95": K.wilson(coerc, decided),
                   "logic": logic, "logic_denominator": decided - b1 - coerc,
                   "logic_wilson95": K.wilson(logic, decided - b1 - coerc)}
    b1dbs = collections.Counter(x["db"] for x in C if x["class"] == "B1_empty_input")
    out["B1_by_db"] = dict(b1dbs)
    out["B1_from_fully_empty_dbs"] = sum(v for k, v in b1dbs.items()
                                         if k in ("academic", "yelp", "geo", "scholar", "imdb", "restaurants", "music_2"))
    return out


def datetime_equality():
    """Labels comparing a `datetime` column (inferred schema) for equality with a date-only value."""
    import re
    from sqlglot import exp
    import kaggle_census as KC
    schema = json.load(open(K.ROOT / "data/interim/finetune_v1/schema.json", encoding="utf-8"))["tables"]
    hits = []
    for r in KC.labels():
        ast, _, _ = K.S.parse_sql(r["reference"]["sql"])
        an = K.S.analyze(ast)
        for n in ast.find_all(exp.EQ):
            cols = [x for x in (n.this, n.expression) if isinstance(x, exp.Column)]
            other = [x for x in (n.this, n.expression) if not isinstance(x, exp.Column)]
            if len(cols) == 1 and other:
                c = cols[0]
                t = an.aliases.get(c.table) if c.table else an.base
                if schema.get(t, {}).get(c.name) != "datetime":
                    continue
                o = other[0]
                sh = K.S._date_shift(o)
                if (isinstance(o, exp.Literal) and o.is_string and re.fullmatch(r"\d{4}-\d{2}-\d{2}", o.this))                         or K.S.time_fn(o) == "today" or (sh and K.S.time_fn(sh[0]) == "today"):
                    hits.append({"id": r["id"], "sql": r["reference"]["sql"], "question": r["question"], "kind": "datetime = date"})
                    break
            elif len(cols) == 2:
                ts = [schema.get(an.aliases.get(c.table) if c.table else an.base, {}).get(c.name) for c in cols]
                if ts == ["datetime", "datetime"] and "same" in r["question"].lower():
                    hits.append({"id": r["id"], "sql": r["reference"]["sql"], "question": r["question"],
                                 "kind": "datetime = datetime for 'same date'"})
                    break
    K.write_jsonl(K.OUT / "datetime_date_equality.jsonl", hits)
    return {"labels": len(hits), "file": "outputs/audit_sql2mql_v1/datetime_date_equality.jsonl"}


def semantic():
    R = K.read_jsonl(K.OUT / "semantic_review.jsonl")
    n = len(R)
    lab = collections.Counter(r["label"] for r in R)
    inc = lab["incorrect"]
    return {"n": n, "mql_vs_sql": dict(collections.Counter(r["mql_vs_sql"] for r in R)),
            "sql_vs_question": dict(collections.Counter(r["sql_vs_question"] for r in R)),
            "label": dict(lab), "incorrect_wilson95": K.wilson(inc, n),
            "incorrect_or_arguable_wilson95": K.wilson(inc + lab["arguable"], n),
            "mql_unfaithful_on_nonempty_data_rule_of_three_upper": 3 / n}


def main():
    s = {}
    for name, fn in (("adversarial", adversarial), ("kaggle_census", kaggle), ("mql_mutants", mutants),
                     ("spider_census", spider), ("semantic_review", semantic),
                     ("datetime_date_equality", datetime_equality)):
        try:
            s[name] = fn()
        except FileNotFoundError as e:
            s[name] = {"missing": str(e)}
    (K.OUT / "summary.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    main()
