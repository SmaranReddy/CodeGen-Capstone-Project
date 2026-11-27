"""Stage 1: extract, normalise, classify and deduplicate the Kaggle NL2SQL pairs.

    python scripts/finetune_v1/extract.py

Reads only data/raw/nl2sql_kaggle_v1/ (sha256-checked, never modified) and writes
data/interim/finetune_v1/{source_rows,pairs,schema,overlap,stage1_counts}.json(l).
Deterministic: no randomness, no network, no model calls; re-running reproduces the files.

Classes assigned per unique (question, SQL) pair (`stage1_class`):
  eligible            -> goes to SQL->MQL conversion + fixture verification (stage 2)
  unsupported_feature -> valid-looking SELECT outside the converter/verifier subset (kept as
                         an unverified candidate with the blocking construct recorded)
  ambiguous           -> question and SQL disagree (ungrounded literal, stated value missing
                         from the SQL, conflicting labels for one question, type conflict)
  rejected            -> not a SELECT, unparseable, placeholder SQL
  leakage_excluded    -> question (near-)duplicates a held-out benchmark question
Exact duplicates are collapsed into one pair (all source row ids kept); near-duplicates are
clustered (`near_dup_cluster`) and capped at build time.
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nk_common as C  # noqa: E402
import grounding as G  # noqa: E402
import sqlsem as S  # noqa: E402
from sqlglot import exp  # noqa: E402

MERGED = re.compile(r"^(?P<tail>.*?[^\s,])\s*,\s*\"?(?P<sql>(?:SELECT|WITH)\b.*)$", re.I | re.S)


# ---------------------------------------------------------------- raw rows
def read_raw():
    for name, sha in C.RAW_SHA256.items():
        got = C.sha256_file(C.RAW_DIR / name)
        if got != sha:
            raise SystemExit(f"raw file {name} sha256 {got} != pinned {sha}")
    with open(C.RAW_CSV, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != ["Prompt", "Query"]:
            raise SystemExit(f"unexpected header {header}")
        rows = []
        for i, rec in enumerate(reader, start=1):
            rows.append({"row": i, "raw": rec})
    return header, rows


def repair(raw):
    """-> (question, sql, repairs[]) for one CSV record."""
    repairs = []
    if len(raw) != 2:
        repairs.append(f"field_count_{len(raw)}")
        raw = [raw[0], ",".join(raw[1:])] if len(raw) > 2 else (raw + [""])[:2]
    q, s = raw
    if not re.match(r"\s*(SELECT|WITH)\b", s, re.I):
        m = MERGED.match(s)
        if m and not re.match(r"\s*(UPDATE|DELETE|INSERT|CREATE|DROP|ALTER)\b", s, re.I):
            q = f"{q.rstrip()}, {m.group('tail').strip()}"
            s = m.group("sql")
            repairs.append("unquoted_comma_in_prompt")
    if '\\"' in s:
        s = s.replace('\\"', '"')
        repairs.append("backslash_escaped_quotes")
    st = s.rstrip()
    if st.endswith('"') and st.count('"') % 2 == 1:
        s = st[:-1]
        repairs.append("stray_trailing_quote")
    q = C.norm_ws(q)
    s = C.norm_ws(s).rstrip(";").strip()
    return q, s, repairs


# ---------------------------------------------------------------- type inference
NAME_DT = re.compile(r"(_at$|_on$|timestamp|time$|^time|_time_)", re.I)
NAME_DATE = re.compile(r"(date|day$|_dt$|birthday|dob)", re.I)
NAME_INT = re.compile(r"(^id$|_id$|^id_|count$|quantity|qty|age$|_no$|number|^num_|year$|"
                      r"stock|level|order$|rank|position|views|votes)", re.I)
NAME_FLOAT = re.compile(r"(price|amount|balance|total|cost|salary|revenue|profit|score|rating|"
                        r"value|rate$|discount|fee|weight|height|reading|temperature|sales$)", re.I)


def infer_type(name, ev):
    """ev: Counter of evidence kinds for one column -> one of S.SQLITE_TYPES."""
    if ev.get("datetime") or ev.get("time"):
        if ev.get("datetime_literal_time") or NAME_DT.search(name) or ev.get("time"):
            return "datetime"
        return "date" if NAME_DATE.search(name) and not NAME_DT.search(name) else "datetime"
    if ev.get("date"):
        return "datetime" if NAME_DT.search(name) else "date"
    if ev.get("text") and ev["text"] >= max(ev.get("int", 0), ev.get("float", 0)):
        return "text"
    if ev.get("bool") and not ev.get("text") and not ev.get("float") and \
            (G.BOOLISH_COL.search(name) or ev.get("bool_literal")):
        return "bool"
    if ev.get("float"):
        return "float"
    if ev.get("int"):
        if G.BOOLISH_COL.search(name) and ev.get("int_01") == ev.get("int"):
            return "bool"
        return "float" if NAME_FLOAT.search(name) else "int"
    if ev.get("numeric"):
        return "float"
    if ev.get("textfn"):
        return "text"
    if G.BOOLISH_COL.search(name):
        return "bool"
    if NAME_DT.search(name):
        return "datetime"
    if NAME_DATE.search(name):
        return "date"
    if NAME_INT.search(name):
        return "int"
    if NAME_FLOAT.search(name):
        return "float"
    return "text"


def literal_evidence(lit):
    k, v = lit["kind"], lit["value"]
    if lit.get("fn") in ("year", "month", "day", "date"):
        return ["date"]
    if lit.get("fn") == "length":
        return ["textfn"]
    if lit.get("fn") in ("lower", "upper"):
        return ["text"]
    if k == "time":
        return ["time"]
    if k == "bool":
        return ["bool", "bool_literal"]
    if k == "int":
        return ["int"] + (["int_01", "bool"] if v in (0, 1) else [])
    if k == "float":
        return ["float"]
    if k == "str":
        if lit["op"] == "like":
            return ["text"]
        m = G.DATE_LIT.match(v)
        if m:
            return ["date"] + (["datetime", "datetime_literal_time"] if m.group(4) else [])
        return ["text"]
    return []


def compatible(lit, ctype):
    """Is this literal comparable with a column of canonical type ctype in BOTH engines?"""
    k, v = lit["kind"], lit["value"]
    if lit.get("fn") in ("year", "month", "day"):
        return ctype in ("date", "datetime") and k == "int"
    if lit.get("fn") == "date":
        return ctype in ("date", "datetime") and k == "str" and bool(G.DATE_LIT.match(v))
    if lit.get("fn") == "length":
        return ctype == "text" and k == "int"
    if lit.get("fn") in ("lower", "upper"):
        return ctype == "text" and k == "str"
    if k in ("time", "null"):
        return k == "null" or ctype in ("date", "datetime")
    if k == "bool":
        return ctype == "bool"
    if k == "int":
        return ctype in ("int", "float") or (ctype == "bool" and v in (0, 1))
    if k == "float":
        return ctype in ("int", "float")
    if k == "str":
        if lit["op"] == "like":
            return ctype == "text"
        if G.DATE_LIT.match(v):
            return ctype in ("date", "datetime")
        return ctype == "text"
    return False


# ---------------------------------------------------------------- overlap
STOP = set("the a an of in on for to and or with by from all what which who how many is are "
           "was were be list show find get give me select retrieve return each that their "
           "than more less have has do does".split())


def trigrams(s):
    s = " " + C.loose(s) + " "
    return {s[i:i + 3] for i in range(len(s) - 2)}


class NearIndex:
    def __init__(self, items):
        """items: [(key, text)]"""
        self.items = items
        self.tri = [trigrams(t) for _, t in items]
        self.inv = collections.defaultdict(set)
        for i, (_, t) in enumerate(items):
            for w in set(C.loose(t).split()) - STOP:
                self.inv[w].add(i)

    def best(self, text):
        words = set(C.loose(text).split()) - STOP
        cand = collections.Counter()
        for w in words:
            for i in self.inv.get(w, ()):
                cand[i] += 1
        tq = trigrams(text)
        best = (0.0, None)
        for i, _ in cand.most_common(200):
            j = len(tq & self.tri[i]) / max(1, len(tq | self.tri[i]))
            if j > best[0]:
                best = (j, self.items[i][0])
        return best


def reference_questions():
    """Benchmark (455, by split) and Spider 1.0 train/dev questions with provenance keys."""
    bench = C.read_jsonl(C.BENCHMARK)
    part = {}
    for p in ("train", "validation", "test"):
        for r in C.read_jsonl(C.SPLIT_DIR / f"{p}.jsonl"):
            part[r["id"]] = p
    heldout = [(f"bench:{r['id']}", r["question"]) for r in bench if part[r["id"]] != "train"]
    btrain = [(f"bench:{r['id']}", r["question"]) for r in bench if part[r["id"]] == "train"]
    spider = []
    for f in ("train_spider.json", "train_others.json", "dev.json"):
        p = C.SPIDER_DIR / f
        if p.exists():
            for i, r in enumerate(json.loads(p.read_text(encoding="utf-8"))):
                spider.append((f"spider:{f}:{i}:{r['db_id']}", r["question"]))
    tables = {}
    tj = C.SPIDER_DIR / "tables.json"
    if tj.exists():
        for t in json.loads(tj.read_text(encoding="utf-8")):
            for name in t["table_names_original"]:
                tables.setdefault(name.lower(), set()).add(t["db_id"])
    heldout_dbs = {r["db_id"] for r in bench if part[r["id"]] != "train"}
    return heldout, btrain, spider, tables, heldout_dbs


# ---------------------------------------------------------------- skeletons
def skeleton(ast):
    def tr(n):
        if isinstance(n, exp.Literal):
            return exp.Placeholder()
        if isinstance(n, exp.Boolean):
            return exp.Placeholder()
        return n
    return ast.copy().transform(tr).sql(dialect="mysql")


def masked_question(q, literals=()):
    """Question with its values masked: quoted spans, numbers/dates, and any (unquoted) text
    equal to one of the SQL's string literals ('facility TTT' -> 'facility <v>'), so template
    families that differ only in their values fall into one near-duplicate cluster."""
    s = q
    for lit in sorted({str(v) for v in literals if isinstance(v, str) and v.strip()}, key=len, reverse=True):
        for part in [x for x in re.split(r"[%_]+", lit) if len(x.strip()) >= 1]:
            s = re.sub(rf"(?<![A-Za-z0-9]){re.escape(part.strip())}(?![A-Za-z0-9])", " <v> ", s, flags=re.I)
    s = G.QUOTED.sub(" <v> ", s)
    s = re.sub(r"\d+(?:[-/.:]\d+)*", " <n> ", s)
    return C.loose(s)


def complexity(a):
    """nk-complexity-v1: a Spider-hardness-like bucket from the analysed components."""
    score = 0
    score += min(a.where_atoms, 4)
    score += 2 * len(a.joins)
    for f, w in (("group_by", 2), ("having", 2), ("order_by", 1), ("limit", 1), ("distinct", 1),
                 ("aggregate", 1), ("time_relative", 1), ("or", 1), ("not", 1), ("like", 1),
                 ("date_part", 1)):
        if f in a.features:
            score += w
    return "easy" if score <= 2 else "medium" if score <= 4 else "hard" if score <= 7 else "extra"


# ---------------------------------------------------------------- main
def main():
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    header, raw_rows = read_raw()
    counts = collections.OrderedDict()
    counts["raw_rows"] = len(raw_rows)

    srows = []
    for r in raw_rows:
        q, s, rep = repair(r["raw"])
        srows.append({"source_record_id": f"{C.SOURCE_ID}:row{r['row']:05d}", "row": r["row"],
                      "raw_prompt": r["raw"][0], "raw_query": r["raw"][1] if len(r["raw"]) > 1 else "",
                      "raw_sha256": C.sha_obj(r["raw"]), "question": q, "sql": s, "repairs": rep})
    C.write_jsonl(C.INTERIM / "source_rows.jsonl", srows)
    counts["rows_repaired"] = sum(bool(r["repairs"]) for r in srows)
    counts["repairs_by_type"] = dict(collections.Counter(x for r in srows for x in r["repairs"]))
    counts["rows_empty"] = sum(not r["question"] or not r["sql"] for r in srows)

    # exact duplicates: identical (question, SQL) after whitespace normalisation
    groups = collections.OrderedDict()
    for r in srows:
        if not r["question"] or not r["sql"]:
            continue
        groups.setdefault((r["question"], r["sql"]), []).append(r)
    counts["unique_pairs"] = len(groups)
    counts["exact_duplicate_rows_collapsed"] = sum(len(v) - 1 for v in groups.values())

    pairs = []
    for (q, s), rs in groups.items():
        pid = "nk_" + C.sha256_bytes(f"{q}\n{s}".encode("utf-8"))[:12]
        p = {"pair_id": pid, "question": q, "sql": s,
             "source_record_ids": [r["source_record_id"] for r in rs],
             "repairs": sorted({x for r in rs for x in r["repairs"]}),
             "n_source_rows": len(rs), "reasons": []}
        ph = G.placeholders(s)
        ast, dialect, err = S.parse_sql(s)
        p["dialect"], p["parse_error"] = dialect, err
        if ast is None:
            p["stage1_class"], p["reasons"] = "rejected", [f"parse_error:{err}"]
        elif not isinstance(ast, exp.Select):
            p["stage1_class"], p["reasons"] = "rejected", [f"not_select:{S.statement_kind(ast)}"]
        elif ph:
            p["stage1_class"], p["reasons"] = "rejected", [f"placeholder:{x}" for x in ph[:3]]
        p["_ast"] = ast
        pairs.append(p)
    by_class = lambda: dict(collections.Counter(p.get("stage1_class", "pending") for p in pairs))  # noqa: E731
    counts["after_parse_and_placeholder_filters"] = by_class()

    # analysis + schema evidence from every parsed SELECT
    evidence = collections.defaultdict(collections.Counter)
    tcols = collections.defaultdict(set)
    for p in pairs:
        if p.get("stage1_class"):
            continue
        a = S.analyze(p["_ast"])
        p["_an"] = a
        for t, cs in a.columns.items():
            tcols[t] |= cs
        for lit in a.literals:
            for e in literal_evidence(lit):
                evidence[(lit["table"], lit["column"])][e] += 1
        for e in p["_ast"].find_all(S.AGGS):
            if isinstance(e, (exp.Sum, exp.Avg)) and isinstance(e.this, exp.Column):
                t = a.aliases.get(e.this.table) if e.this.table else a.base
                if t:
                    evidence[(t, e.this.name)]["numeric"] += 1
        if a.unsupported:
            p["stage1_class"] = "unsupported_feature"
            p["reasons"] = [f"unsupported:{u}" for u in a.unsupported]
    schema = {t: {c: infer_type(c, evidence[(t, c)]) for c in sorted(cs)}
              for t, cs in sorted(tcols.items())}
    counts["after_supported_subset"] = by_class()

    # type conflicts against the canonical (dataset-wide) column types
    for p in pairs:
        if p.get("stage1_class"):
            continue
        bad = [f"type_conflict:{l['table']}.{l['column']}({schema[l['table']][l['column']]}) vs {l['kind']}:{l['value']!r}"
               for l in p["_an"].literals if not compatible(l, schema[l["table"]][l["column"]])]
        if bad:
            p["stage1_class"], p["reasons"] = "ambiguous", bad[:4]
    counts["after_type_check"] = by_class()

    # grounding
    for p in pairs:
        if not p.get("_an") or p.get("stage1_class") == "rejected":
            continue
        a, ast = p["_an"], p["_ast"]
        extra = []
        if a.limit is not None:
            extra.append((a.limit, {"role": "limit"}))
        if a.offset is not None:
            extra.append((a.offset, {"role": "offset"}))
        for node in ast.walk():
            sh = S._date_shift(node)
            if sh is not None:
                extra.append((abs(sh[1]), {"role": "interval", "unit": sh[2]}))
        for node in ast.find_all(exp.Mul):
            for side in (node.this, node.expression):
                if isinstance(side, exp.Literal) and not side.is_string and "." in side.this:
                    extra.append((float(side.this), {"role": "percent"}))
        lits = [l for l in a.literals]
        if G.SQL_IN_TEXT.search(p["question"]):
            p["grounding"] = {"ok": False, "failures": ["question_contains_sql"]}
            p["stage1_class"], p["reasons"] = "rejected", ["question_contains_sql"]
            continue
        fails = G.check(p["question"], p["sql"], lits, extra)
        if not a.unsupported:
            fails += G.structure_check(p["question"], ast, a, schema)
        p["grounding"] = {"ok": not fails, "failures": fails}
        if fails and not p.get("stage1_class"):
            p["stage1_class"], p["reasons"] = "ambiguous", fails[:6]
    counts["after_grounding"] = by_class()

    # conflicting labels: same (loosely normalised) question, different SQL
    byq = collections.defaultdict(list)
    for p in pairs:
        byq[C.loose(p["question"])].append(p)
    n_conf_q = 0
    for qk, ps in byq.items():
        sk = {S.to_sqlite(p["_ast"]) if p.get("_ast") is not None and isinstance(p["_ast"], exp.Select) else p["sql"]
              for p in ps}
        if len(sk) < 2:
            continue
        n_conf_q += 1
        survivors = [p for p in ps if p.get("stage1_class") in (None, "unsupported_feature")]
        distinct = {S.to_sqlite(p["_ast"]) for p in survivors}
        for p in ps:
            p["label_conflict"] = {"n_sql_variants": len(sk), "n_grounded_variants": len(distinct)}
        if len(distinct) > 1:
            for p in survivors:
                p["stage1_class"] = "ambiguous"
                p["reasons"] = [f"conflicting_labels:{len(distinct)}_grounded_sql_variants"]
    counts["questions_with_conflicting_sql"] = n_conf_q
    counts["after_conflict_resolution"] = by_class()

    # overlap with the benchmark and Spider (provenance; held-out -> excluded)
    heldout, btrain, spider, sp_tables, heldout_dbs = reference_questions()
    idx_h, idx_t, idx_s = NearIndex(heldout), NearIndex(btrain), NearIndex(spider)
    ov = {"thresholds": {"near_duplicate_trigram_jaccard": 0.8}, "pairs": {}}
    hist = collections.Counter()
    for p in pairs:
        h, t, s_ = idx_h.best(p["question"]), idx_t.best(p["question"]), idx_s.best(p["question"])
        p["overlap"] = {"heldout_max": round(h[0], 4), "heldout_key": h[1],
                        "bench_train_max": round(t[0], 4), "spider_max": round(s_[0], 4),
                        "spider_key": s_[1]}
        for name, v in (("heldout", h[0]), ("bench_train", t[0]), ("spider", s_[0])):
            hist[f"{name}>=0.8"] += v >= 0.8
            hist[f"{name}>=0.6"] += v >= 0.6
        if h[0] >= 0.8:
            p["stage1_class"] = "leakage_excluded"
            p["reasons"] = [f"near_duplicate_of_heldout:{h[1]}:{h[0]:.3f}"]
        tabs = set(p["_an"].columns) if p.get("_an") else set()
        hits = {t: sorted(sp_tables.get(t.lower(), ())) for t in tabs if t.lower() in sp_tables}
        if hits:
            p["overlap"]["spider_table_name_matches"] = hits
            p["overlap"]["heldout_db_table_name_matches"] = sorted(
                t for t, dbs in hits.items() if set(dbs) & heldout_dbs)
    kag_tables = set(schema)
    ov["question_similarity_counts"] = dict(hist)
    ov["table_name_overlap_with_spider"] = {t: sorted(sp_tables[t.lower()]) for t in sorted(kag_tables)
                                            if t.lower() in sp_tables}
    ov["table_name_overlap_with_heldout_dbs"] = sorted(
        t for t in kag_tables if t.lower() in sp_tables and set(sp_tables[t.lower()]) & heldout_dbs)
    ov["heldout_dbs"] = sorted(heldout_dbs)
    counts["after_leakage_check"] = by_class()

    # near-duplicate clusters among stage-2-bound pairs
    for p in pairs:
        if p.get("_ast") is not None and isinstance(p["_ast"], exp.Select):
            p["sql_skeleton"] = skeleton(p["_ast"])
            lits = [l["value"] for l in (p["_an"].literals if p.get("_an") else [])]
            p["masked_question"] = masked_question(p["question"], lits)
            p["near_dup_cluster"] = "cl_" + C.sha256_bytes(
                f"{p['masked_question']}\n{p['sql_skeleton']}".encode("utf-8"))[:10]
    for p in pairs:
        p.setdefault("stage1_class", "eligible")
    elig = [p for p in pairs if p["stage1_class"] == "eligible"]
    cl = collections.Counter(p["near_dup_cluster"] for p in elig)
    counts["eligible_near_dup_clusters"] = len(cl)
    counts["eligible_in_multi_member_clusters"] = sum(v for v in cl.values() if v > 1)
    counts["stage1_final"] = by_class()

    out = []
    for p in pairs:
        a = p.pop("_an", None)
        ast = p.pop("_ast", None)
        if a is not None:
            p["analysis"] = a.to_json()
            p["complexity"] = complexity(a)
            p["sqlite_sql"] = S.to_sqlite(ast)
        p["source_category"] = classify5(p)
        out.append(p)
    counts["five_way_classification"] = dict(collections.Counter(p["source_category"] for p in out))
    C.write_jsonl(C.INTERIM / "pairs.jsonl", out)
    (C.INTERIM / "schema.json").write_text(json.dumps(
        {"db_id": C.DB_ID, "method": "union of columns referenced by every parsed SELECT; type "
         "by majority literal/function evidence, then column-name heuristics",
         "tables": schema}, indent=1, sort_keys=True), encoding="utf-8")
    ov["pairs"] = {p["pair_id"]: p["overlap"] for p in out if "overlap" in p}
    (C.INTERIM / "overlap.json").write_text(json.dumps(ov, indent=1, sort_keys=True), encoding="utf-8")
    (C.INTERIM / "stage1_counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    print(json.dumps(counts, indent=1))


def classify5(p):
    """Map onto the five requested categories (see the expansion report, section 3)."""
    c = p["stage1_class"]
    if c == "rejected" or (c == "ambiguous" and any(r.startswith("type_conflict") for r in p["reasons"])):
        return "5_invalid_or_unsuitable" if c == "rejected" else "2_needs_schema_or_dialect_handling"
    if c == "ambiguous":
        return "5_invalid_or_unsuitable"
    if c == "leakage_excluded":
        return "4_duplicate_or_near_duplicate"
    if c == "unsupported_feature":
        return "3_needs_database_or_extended_converter"
    if p["repairs"] or p["dialect"] != "mysql":
        return "2_needs_schema_or_dialect_handling"
    return "1_directly_usable"


if __name__ == "__main__":
    main()
