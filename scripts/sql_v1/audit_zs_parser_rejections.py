"""Diagnostic audit of official-parser rejections in SQL-0001 (dev only).

Read-only on the run artifacts. Writes only to <out>/ (a new folder). Uses the bundled official
process_sql/evaluation through the repo's OfficialEval wrapper; nothing in them is changed.
Rewrites below are diagnostics: they never replace the official metric.
"""
import collections, json, re, sys, traceback
from pathlib import Path

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "scripts/sql_v1"))
import spider_zs_sql as Z

bundle = repo / "kaggle/spider_sql_v1_dev_bundle_v1"
run = repo / "outputs/spider_sql_v1_zs_3b/v1/spider_sql_v1_zs_qwen25coder3b_v1"
variants_root = repo / "data/raw/spider1/testsuite"
out.mkdir(parents=True, exist_ok=True)
assert not any(out.iterdir()), f"{out} must be empty"

dev = Z.load_dev(bundle)
P = Z.read_jsonl(run / "predictions.jsonl")
R = Z.read_jsonl(run / "evaluation_results.jsonl")
assert [d["id"] for d in dev] == [p["id"] for p in P] == [r["id"] for r in R]
oe = Z.OfficialEval(bundle)
ps = oe.process_sql
assert oe.ev.get_sql is ps.get_sql

# ------------------------------------------------------------------ parsing with evidence
def parse_info(db_id, sql):
    """Official get_sql, plus where it failed (innermost process_sql frame) or how many tokens it left unread."""
    schema = oe.schema(db_id)
    try:
        toks = ps.tokenize(sql)
        twa = ps.get_tables_with_alias(schema.schema, toks)
        end, _ = ps.parse_sql(toks, 0, twa, schema)
        end = ps.skip_semicolon(toks, end)
        ok_official = True
        try:
            ps.get_sql(schema, sql)
        except Exception:  # noqa: BLE001
            ok_official = False
        return {"ok": ok_official, "unread_tokens": toks[end:]}
    except Exception as e:  # noqa: BLE001
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("process_sql.py")]
        tb = e.__traceback__
        loc = None
        while tb is not None:
            if tb.tb_frame.f_code.co_filename.endswith("process_sql.py"):
                loc = tb.tb_frame.f_locals
            tb = tb.tb_next
        tok, ctx = None, None
        if loc is not None:
            toks_l = loc.get("toks")
            i = loc.get("idx", loc.get("start_idx"))
            if isinstance(toks_l, list) and isinstance(i, int):
                tok = toks_l[i] if i < len(toks_l) else "<end>"
                ctx = toks_l[max(0, i - 4): i]
        inner = frames[-1] if frames else None
        return {"ok": False, "exc": type(e).__name__, "msg": str(e)[:160],
                "func": inner.name if inner else None, "line": inner.lineno if inner else None, "token": tok,
                "context": ctx}

# ------------------------------------------------------------------ string masking and features
LIT = re.compile(r"'(?:[^']|'')*'|\"[^\"]*\"")
def mask(sql):
    lits = []
    def sub(m):
        lits.append(m.group(0)); return f" __L{len(lits)-1}__ "
    return LIT.sub(sub, sql), lits
def unmask(s, lits):
    return re.sub(r"\s?__L(\d+)__\s?", lambda m: " " + lits[int(m.group(1))] + " ", s)

KW = set("""select from where group by order having limit join on as and or not in like between is null exists
union intersect except distinct asc desc inner left right outer cross natural full using case when then else end
count sum avg min max offset values with over partition""".split())
AGG = {"count", "sum", "avg", "min", "max"}

def tables_of(db_id):
    return set(oe.schema(db_id).schema)

def select_items(m):
    """[(start, end)] spans of each SELECT list (between SELECT and its FROM at the same depth) in masked SQL."""
    spans, depth, stack = [], 0, []
    toks = [(t.start(), t.group(0)) for t in re.finditer(r"\(|\)|\b[A-Za-z_]+\b", m)]
    open_sel = {}
    for pos, t in toks:
        if t == "(":
            depth += 1
        elif t == ")":
            open_sel.pop(depth, None); depth -= 1
        elif t.lower() == "select":
            open_sel[depth] = pos + len(t)
        elif t.lower() == "from" and depth in open_sel:
            spans.append((open_sel.pop(depth), pos))
    return spans

def split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(": depth += 1
        if ch == ")": depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts

ALIAS_AS = re.compile(r"^(?P<expr>.*\S)\s+AS\s+(?P<alias>[A-Za-z_]\w*|__L\d+__|`[^`]+`)\s*$", re.I | re.S)
ALIAS_BARE = re.compile(r"^(?P<expr>.*[\)\w])\s+(?P<alias>[A-Za-z_]\w*)\s*$", re.S)

def column_aliases(m):
    found = []
    for a, b in select_items(m):
        for item in split_top(m[a:b]):
            x = ALIAS_AS.match(item.strip())
            if not x:
                y = ALIAS_BARE.match(item.strip())
                if y and y.group("alias").lower() not in KW and y.group("expr").strip().lower() != "distinct":
                    x = y
            if x:
                found.append((x.group("expr").strip(), x.group("alias")))
    return found

def bare_table_alias_re(tabs):
    stop = "|".join(sorted(KW | {"where", "group", "order", "limit"}))
    return re.compile(r"\b(FROM|JOIN|,)\s+(" + "|".join(map(re.escape, sorted(tabs, key=len, reverse=True)))
                      + r")\s+(?!(?:" + stop + r")\b)([A-Za-z_]\w*)\b", re.I)

FUNC = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
def features(db_id, sql):
    m, lits = mask(sql)
    f = {}
    ca = column_aliases(m)
    if ca: f["column_alias"] = [a for _, a in ca]
    tabs = tables_of(db_id)
    if bare_table_alias_re(tabs).search(m): f["table_alias_without_as"] = True
    j = re.findall(r"\b((?:LEFT|RIGHT|INNER|CROSS|NATURAL|FULL)(?:\s+OUTER)?)\s+JOIN\b", m, re.I)
    if j: f["join_variant"] = sorted({x.upper() for x in j})
    if re.search(r"\bUSING\s*\(", m, re.I): f["join_using"] = True
    fn = sorted({x.lower() for x in FUNC.findall(m) if x.lower() not in KW | AGG})
    if fn: f["function"] = fn
    if re.search(r"\bCASE\b", m, re.I): f["case_expr"] = True
    if re.search(r"\bIS\s+(NOT\s+)?NULL\b", m, re.I): f["is_null"] = True
    if "<>" in m: f["angle_neq"] = True
    if re.search(r"\bIN\s*\(\s*(__L\d+__|-?\d+(\.\d+)?)\s*,", m, re.I): f["in_literal_list"] = True
    if re.search(r"\|\|", m): f["concat_op"] = True
    if re.search(r"^\s*WITH\b", m, re.I): f["cte"] = True
    if re.search(r"\bOVER\s*\(", m, re.I): f["window"] = True
    if re.search(r"\bFROM\s*\(\s*SELECT\b", m, re.I): f["subquery_in_from"] = True
    if re.search(r"\b\w+\.\*", m): f["qualified_star"] = True
    if re.search(r"\bOFFSET\b", m, re.I): f["offset"] = True
    if re.search(r"[*/+\-]\s*\d|\d\s*[*/+]|\b\d+\.\d+\b", re.sub(r"\bLIMIT\s+\d+", "", m, flags=re.I)):
        f["numeric_literal_arith"] = True
    if "`" in m: f["backtick_identifier"] = True
    if any(l.startswith('"') for l in lits): f["double_quoted_token"] = True
    if re.search(r"\b(TRUE|FALSE)\b", m, re.I): f["bool_literal"] = True
    return f

# ------------------------------------------------------------------ diagnostic rewrites (parse-only)
def rw_column_alias(db_id, sql):
    m, lits = mask(sql)
    spans = select_items(m)
    if not spans: return sql
    pieces, last, alias_map = [], 0, {}
    for a, b in spans:
        items = []
        for item in split_top(m[a:b]):
            s = item.strip()
            x = ALIAS_AS.match(s)
            if not x:
                y = ALIAS_BARE.match(s)
                if y and y.group("alias").lower() not in KW and y.group("expr").strip().lower() != "distinct":
                    x = y
            if x:
                alias_map[x.group("alias")] = x.group("expr").strip()
                items.append(" " + x.group("expr").strip() + " ")
            else:
                items.append(item)
        pieces.append(m[last:a]); pieces.append(",".join(items)); last = b
    pieces.append(m[last:])
    m2 = "".join(pieces)
    # references to a removed alias outside the select lists -> the aliased expression
    for alias, expr in alias_map.items():
        pat = re.compile(r"(?<![\w.])" + re.escape(alias) + r"(?![\w.(])", re.I)
        parts = re.split(r"(\bFROM\b)", m2, maxsplit=1, flags=re.I)
        if len(parts) == 3:
            # only after the first FROM; select items were already rewritten
            m2 = parts[0] + parts[1] + pat.sub(" " + expr + " ", parts[2])
    return unmask(m2, lits)

def rw_table_alias(db_id, sql):
    m, lits = mask(sql)
    return unmask(bare_table_alias_re(tables_of(db_id)).sub(r"\1 \2 AS \3", m), lits)

def rw_join_variant(db_id, sql):
    m, lits = mask(sql)
    return unmask(re.sub(r"\b(?:LEFT|RIGHT|INNER|CROSS|FULL)(?:\s+OUTER)?\s+JOIN\b", "JOIN", m, flags=re.I), lits)

def rw_quoting(db_id, sql):
    m, lits = mask(sql)
    m = re.sub(r"`([^`]+)`", r"\1", m).replace("<>", "!=")
    return unmask(m, lits)

REWRITES = {"column_alias": rw_column_alias, "table_alias_without_as": rw_table_alias,
            "join_variant": rw_join_variant, "quoting_or_neq": rw_quoting}
FORMAT_RW = {"column_alias", "table_alias_without_as", "quoting_or_neq"}

def same_rows(a, b, ordered):
    return (a == b) if ordered else (collections.Counter(a) == collections.Counter(b))

# ------------------------------------------------------------------ 1. reproduce parse status everywhere
recs = []
for d, p, r in zip(dev, P, R):
    pi = parse_info(d["db_id"], r["eval_sql"])
    gi = parse_info(d["db_id"], d["sql"])
    recs.append({"d": d, "p": p, "r": r, "pi": pi, "gi": gi})
repro = sum(x["pi"]["ok"] == x["r"]["official_parse_ok"] for x in recs)
rej = [x for x in recs if not x["r"]["official_parse_ok"]]
print(f"parse status reproduced for {repro}/1034; rejections {len(rej)}; gold parses {sum(x['gi']['ok'] for x in recs)}")
assert all(x["r"]["official_exec"] == 0 and x["r"]["official_exact"] == 0 for x in rej)

# ------------------------------------------------------------------ 2. classify rejections
def sqlite_class(err):
    e = (err or "").lower()
    if "syntax error" in e or "incomplete input" in e or "unrecognized token" in e: return "sqlite_syntax_error"
    if "no such function" in e: return "sqlite_no_such_function"
    if "no such column" in e or "no such table" in e or "ambiguous column" in e: return "sqlite_schema_reference_error"
    return "sqlite_other_error"

rows_cache = {}
def run_rows(db, sql):
    key = (db, sql)
    if key not in rows_cache:
        rows_cache[key] = oe._run(db, sql)
    return rows_cache[key]

audit = []
for x in rej:
    d, r, pi = x["d"], x["r"], x["pi"]
    sql, db_id = r["eval_sql"], d["db_id"]
    feats = features(db_id, sql)
    ws = lambda s: re.sub(r"\s+", "", s)
    rw = {}
    for name, fn in REWRITES.items():
        s2 = fn(db_id, sql)
        ch = ws(s2) != ws(sql)
        rw[name] = {"changed": ch, "parses": ch and parse_info(db_id, s2)["ok"], "sql": s2}
    order = ["quoting_or_neq", "table_alias_without_as", "column_alias", "join_variant"]
    def apply(names):
        s = sql
        for n in order:
            if n in names:
                s = REWRITES[n](db_id, s)
        return s
    changed = [n for n in order if rw[n]["changed"]]
    combo_all = apply(changed)
    combo_all_ok = bool(changed) and parse_info(db_id, combo_all)["ok"]
    combo = apply([n for n in changed if n != "join_variant"])
    combo_fmt_ok = ws(combo) != ws(sql) and parse_info(db_id, combo)["ok"]
    # rewrites needed: dropping one of them from the full set breaks parsing again (leave-one-out)
    needed = [n for n in changed if not parse_info(db_id, apply([c for c in changed if c != n]))["ok"]] if combo_all_ok else []
    residual = None if combo_all_ok else parse_info(db_id, combo_all)

    def residual_cause(info, f):
        ctx = info.get("context") or []
        tok = info.get("token")
        if info.get("func") == "parse_value" and tok == "," and "in" in ctx and "(" in ctx:
            return "in_literal_list"
        if info.get("msg", "").startswith("Alias"):
            return "alias_collides_with_table_name"
        if tok == "is" or tok == "null":
            return "is_null"
        if tok == "select" and len(ctx) >= 2 and ctx[-1] == "(" and ctx[-2] in ("or", "and", "where", "having"):
            return "scalar_subquery_as_left_operand"
        if info.get("exc") == "KeyError" and tok == "(" and "join" in ctx:
            return "subquery_in_join"
        for k in ("case_expr", "cte", "window", "concat_op", "subquery_in_from", "qualified_star", "join_using", "offset"):
            if k in f:
                return k
        return None

    if not r["sqlite_prepare_ok"]:
        sc = sqlite_class(r["sqlite_prepare_error"])
        cat = {"sqlite_syntax_error": "A_genuine_syntax_error",
               "sqlite_no_such_function": "D_sqlite_dialect_incompatibility",
               "sqlite_schema_reference_error": "E_invalid_schema_reference",
               "sqlite_other_error": "E_other_sqlite_error"}[sc]
        sub = sc
    else:
        causes = list(needed)
        if residual is not None:
            rc = residual_cause(residual, feats)
            causes += [n for n in changed if n not in causes]  # applied, but not shown sufficient
            causes.append(rc or f"residual:{residual.get('exc')}@{residual.get('func')}:{residual.get('line')} tok={residual.get('token')!r}")
        limitation = [c for c in causes if c not in FORMAT_RW and not c.startswith("residual:")]
        if residual is not None and residual_cause(residual, feats) is None:
            cat = "F_unclassified"
        elif limitation:
            cat = "B_unsupported_construct"
        else:
            cat = "C_alias_qualification_formatting"
        sub = "+".join(causes)
    audit.append({"id": d["id"], "db_id": db_id, "pred_sql": sql, "gold_sql": d["sql"], "hardness": r["hardness"],
                  "category": cat, "subcategory": sub, "parser_error": {k: pi.get(k) for k in ("exc", "msg", "func", "line", "token")},
                  "sqlite_prepare_ok": r["sqlite_prepare_ok"], "sqlite_error": r["sqlite_prepare_error"],
                  "features": feats, "rewrites_that_parse": [n for n, v in rw.items() if v["parses"]],
                  "rewrites_applied": changed, "rewrites_needed": needed,
                  "residual_parser_error": residual and {k: residual.get(k) for k in ("exc", "msg", "func", "line", "token", "context")},
                  "formatting_rewrites_combined_parse": combo_fmt_ok, "all_rewrites_combined_parse": combo_all_ok,
                  "normalized_sql": combo_all if combo_all_ok else (combo if combo_fmt_ok else None),
                  "unofficial_result_match": r["unofficial_result_match"], "exec_status": r["exec_status"]})

# ------------------------------------------------------------------ 3. parser-normalised diagnostic score (does NOT replace official)
for a in audit:
    a["normalized"] = None
    s2 = a["normalized_sql"]
    if not s2: continue
    d = next(x["d"] for x in rej if x["d"]["id"] == a["id"])
    db = oe.db_path(a["db_id"])
    st1, rows1, _ = run_rows(db, a["pred_sql"])
    st2, rows2, _ = run_rows(db, s2)
    preserved = st1 == st2 == "ok" and rows1 == rows2
    sc = oe.score_one(d, s2)
    a["normalized"] = {"rows_preserved_exactly": preserved, "official_parse_ok": sc["official_parse_ok"],
                       "official_exec": sc["official_exec"], "official_exact": sc["official_exact"]}

# ------------------------------------------------------------------ 4. test-suite variant check (dev DBs only)
def variant_check(db_id, pred, gold, ordered):
    vs = sorted((variants_root / db_id).glob("*.sqlite")) if (variants_root / db_id).is_dir() else []
    if not vs: return {"n_variants": 0, "status": "no_variants"}
    res = []
    for v in vs:
        gs, grows, gerr = run_rows(str(v), gold)
        pst, prows, perr = run_rows(str(v), pred)
        if gs != "ok": res.append("gold_failed")
        elif pst != "ok": res.append("pred_failed")
        else: res.append("match" if same_rows(prows, grows, ordered) else "mismatch")
    status = "all_match" if all(t == "match" for t in res) else ("gold_failed" if "gold_failed" in res else "distinguished")
    return {"n_variants": len(vs), "per_variant": res, "status": status}

def gold_shape(d):
    db = oe.db_path(d["db_id"])
    st, rows, _ = run_rows(db, d["sql"])
    return {"n_rows": len(rows), "n_cols": len(rows[0]) if rows else None,
            "empty": len(rows) == 0, "single_value": len(rows) == 1 and len(rows[0]) == 1}

groups = {
    "parse_rejected_result_match": [x for x in recs if not x["r"]["official_parse_ok"] and x["r"]["unofficial_result_match"]],
    "official_ex_correct": [x for x in recs if x["r"]["official_exec"] == 1],
    "parsed_ex0_result_match": [x for x in recs if x["r"]["official_parse_ok"] and not x["r"]["official_exec"] and x["r"]["unofficial_result_match"]],
    "ex1_result_mismatch": [x for x in recs if x["r"]["official_exec"] == 1 and not x["r"]["unofficial_result_match"]],
}
vc = {}
for gname, xs in groups.items():
    rows = []
    for x in xs:
        d = x["d"]
        g_sql = ps.get_sql(oe.schema(d["db_id"]), d["sql"])
        ordered = bool(g_sql["orderBy"])
        vr = variant_check(d["db_id"], x["r"]["eval_sql"], d["sql"], ordered)
        rows.append({"id": d["id"], "db_id": d["db_id"], "variant": vr, "gold_shape": gold_shape(d), "gold_ordered": ordered})
    vc[gname] = rows

# EX=1 but unofficial mismatch: explain with row comparisons on the official DB
ex1_mis = []
for x in groups["ex1_result_mismatch"]:
    d = x["d"]; db = oe.db_path(d["db_id"])
    _, g, _ = run_rows(db, d["sql"]); _, p, _ = run_rows(db, x["r"]["eval_sql"])
    reason = []
    if collections.Counter(p) == collections.Counter(g): reason.append("same_multiset_order_differs")
    elif collections.Counter(tuple(sorted(map(repr, t))) for t in p) == collections.Counter(tuple(sorted(map(repr, t))) for t in g):
        reason.append("same_values_columns_permuted")
    elif p and g and len(p[0]) != len(g[0]): reason.append(f"column_count_{len(p[0])}_vs_gold_{len(g[0])}")
    elif set(p) == set(g): reason.append("same_set_duplicates_differ")
    else: reason.append("rows_differ")
    ex1_mis.append({"id": d["id"], "reason": reason[0]})

# ------------------------------------------------------------------ 5. parser over-acceptance (tokens left unread)
unread_pred = [x["d"]["id"] for x in recs if x["pi"]["ok"] and x["pi"].get("unread_tokens")]
unread_gold = [x["d"]["id"] for x in recs if x["gi"]["ok"] and x["gi"].get("unread_tokens")]

# ------------------------------------------------------------------ 6. minimal probes: each construct alone, official parser
PROBES = [
    ("control (Spider style)", "concert_singer", "SELECT T1.name FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.singer_id = T2.singer_id"),
    ("column_alias", "concert_singer", "SELECT avg(age) AS avg_age FROM singer"),
    ("column_alias referenced in ORDER BY", "concert_singer", "SELECT country, count(*) AS n FROM singer GROUP BY country ORDER BY n DESC"),
    ("table_alias_without_as", "concert_singer", "SELECT s.name FROM singer s"),
    ("INNER JOIN", "concert_singer", "SELECT T1.name FROM singer AS T1 INNER JOIN singer_in_concert AS T2 ON T1.singer_id = T2.singer_id"),
    ("LEFT JOIN", "concert_singer", "SELECT T1.name FROM singer AS T1 LEFT JOIN singer_in_concert AS T2 ON T1.singer_id = T2.singer_id"),
    ("IN literal list", "concert_singer", "SELECT count(*) FROM concert WHERE year IN ('2014', '2015')"),
    ("IN single literal", "concert_singer", "SELECT count(*) FROM concert WHERE year IN ('2014')"),
    ("IS NULL", "concert_singer", "SELECT name FROM singer WHERE country IS NULL"),
    ("subquery in FROM with alias", "concert_singer", "SELECT T.c FROM (SELECT count(*) AS c FROM singer) AS T"),
    ("scalar subquery as left operand", "concert_singer", "SELECT name FROM stadium WHERE (SELECT count(*) FROM concert) > 3"),
    ("backtick identifier", "concert_singer", "SELECT `name` FROM singer"),
    ("<> operator", "concert_singer", "SELECT name FROM singer WHERE country <> 'France'"),
]
probes = []
for label, db_id, q in PROBES:
    info = parse_info(db_id, q)
    st, _, err = oe._run(oe.db_path(db_id), q)
    probes.append({"construct": label, "sql": q, "official_parse_ok": info["ok"], "sqlite_exec": st,
                   "parser_error": None if info["ok"] else f"{info.get('exc')} in {info.get('func')}:{info.get('line')} at token {info.get('token')!r}"})

# ------------------------------------------------------------------ write
Z.write_jsonl(out / "parser_rejections.jsonl", audit)
Z.write_jsonl(out / "variant_checks.jsonl", [{"group": g, **row} for g, rs in vc.items() for row in rs])
C = collections.Counter
summary = {
    "n_examples": len(recs), "parse_status_reproduced": repro, "n_rejected": len(rej),
    "gold_parse_ok": sum(x["gi"]["ok"] for x in recs),
    "category_counts": dict(C(a["category"] for a in audit)),
    "subcategory_counts": {c: dict(C(a["subcategory"] for a in audit if a["category"] == c).most_common()) for c in sorted({a["category"] for a in audit})},
    "category_by_result_match": {c: dict(C(a["unofficial_result_match"] for a in audit if a["category"] == c)) for c in sorted({a["category"] for a in audit})},
    "parser_error_locations": dict(C(f"{a['parser_error']['exc']}@{a['parser_error']['func']}:{a['parser_error']['line']}" for a in audit).most_common()),
    "feature_counts_sqlite_ok": dict(C(k for a in audit if a["sqlite_prepare_ok"] for k in a["features"]).most_common()),
    "functions_sqlite_ok": dict(C(fn for a in audit if a["sqlite_prepare_ok"] for fn in a["features"].get("function", [])).most_common()),
    "sqlite_errors": dict(C(re.sub(r"[:\s].*", "", (a["sqlite_error"] or "")) + " " + " ".join((a["sqlite_error"] or "").split()[:3]) for a in audit if not a["sqlite_prepare_ok"]).most_common(15)),
    "normalized": {
        "n_normalizable": sum(a["normalized"] is not None for a in audit),
        "rows_preserved": sum(bool(a["normalized"] and a["normalized"]["rows_preserved_exactly"]) for a in audit),
        "ex_gain_rows_preserved": sum(bool(a["normalized"] and a["normalized"]["rows_preserved_exactly"] and a["normalized"]["official_exec"]) for a in audit),
        "em_gain_rows_preserved": sum(bool(a["normalized"] and a["normalized"]["rows_preserved_exactly"] and a["normalized"]["official_exact"]) for a in audit),
        "ex_gain_formatting_only": sum(bool(a["normalized"] and a["normalized"]["rows_preserved_exactly"] and a["normalized"]["official_exec"] and a["category"].startswith("C_")) for a in audit),
        "em_gain_formatting_only": sum(bool(a["normalized"] and a["normalized"]["rows_preserved_exactly"] and a["normalized"]["official_exact"] and a["category"].startswith("C_")) for a in audit),
    },
    "variant_checks": {g: {"n": len(rs), "status": dict(C(r["variant"]["status"] for r in rs)),
                           "gold_empty": sum(r["gold_shape"]["empty"] for r in rs),
                           "gold_single_value": sum(r["gold_shape"]["single_value"] for r in rs)} for g, rs in vc.items()},
    "variant_db_coverage": sorted(p.name for p in variants_root.iterdir() if p.is_dir()),
    "ex1_result_mismatch_reasons": dict(C(e["reason"] for e in ex1_mis)),
    "parser_accepts_with_unread_tokens": {"pred": len(unread_pred), "pred_ids": unread_pred, "gold": len(unread_gold), "gold_ids": unread_gold},
    "probes": probes,
    "parse_ok_but_sqlite_fails":[x["d"]["id"] for x in recs if x["r"]["official_parse_ok"] and not x["r"]["sqlite_prepare_ok"]],
}
summary["variant_match_rejected_by_category"] = dict(C(
    (next(a["category"] for a in audit if a["id"] == r["id"]), r["variant"]["status"]) .__repr__() for r in vc["parse_rejected_result_match"]))
(out / "parser_audit_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k not in ("parser_accepts_with_unread_tokens",)}, indent=1)[:9000])
print("unread tokens:", summary["parser_accepts_with_unread_tokens"]["pred"], summary["parser_accepts_with_unread_tokens"]["gold"])
