"""Query-pattern vocabulary, question masking and tokenisation for the Week 3 RAG pipeline.

Three kinds of signal, kept strictly apart:
  * sql_patterns(sql)        -- labels from a gold SQL. Used ONLY for train-corpus metadata and
                                for OFFLINE retrieval evaluation; never for a query at inference.
  * question_cues(q, schema) -- heuristic labels from the question text plus the target schema
                                (both available at inference time).
  * mql_operators(obj)       -- every "$operator" key in an MQL object (corpus metadata).

Masking follows DAIL-SQL (Gao et al., 2023): schema mentions and literal values in a question
are replaced by placeholders so that similarity reflects the question's structure rather than
its domain. Every example in the corpus comes from a different database than every query, so
domain words cannot transfer anyway.
"""
import re

PATTERNS = ("select", "where", "count", "count_distinct", "distinct", "aggregate", "group_by",
            "having", "order_by", "limit", "join", "nested", "anti_join", "intersect", "union",
            "like", "text_number")

_STR_LIT = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")


def sql_patterns(sql):
    """Structural labels of a SQL query (string literals removed first)."""
    s = " " + re.sub(r"\s+", " ", _STR_LIT.sub("'v'", sql)).upper() + " "
    p = {"select"}
    rules = {
        "where": r"\bWHERE\b", "join": r"\bJOIN\b", "group_by": r"\bGROUP BY\b",
        "having": r"\bHAVING\b", "order_by": r"\bORDER BY\b", "limit": r"\bLIMIT\b",
        "count": r"\bCOUNT\s*\(", "count_distinct": r"\bCOUNT\s*\(\s*DISTINCT\b",
        "distinct": r"\bSELECT DISTINCT\b|\bCOUNT\s*\(\s*DISTINCT\b",
        "aggregate": r"\b(AVG|SUM|MIN|MAX)\s*\(", "intersect": r"\bINTERSECT\b",
        "union": r"\bUNION\b", "like": r"\bLIKE\b",
        "anti_join": r"\bNOT IN\b|\bEXCEPT\b|\bNOT EXISTS\b",
    }
    for name, rx in rules.items():
        if re.search(rx, s):
            p.add(name)
    # a sub-SELECT that is not just the right side of a set operation or an anti-join
    subs = len(re.findall(r"\(\s*SELECT\b", s))
    if subs > len(re.findall(r"\bNOT IN\s*\(\s*SELECT\b|\bNOT EXISTS\s*\(\s*SELECT\b", s)):
        p.add("nested")
    return p


_CUES = {
    "count": r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal number\b",
    "group_by": r"\beach\b|\bper\b|\bevery\b|\bfor all\b",
    "order_by": (r"\border\b|\bordered\b|\bsort|\bdescending\b|\bascending\b|\balphabetical"
                 r"|\bmost\b|\bleast\b|\bhighest\b|\blowest\b|\blargest\b|\bsmallest\b"
                 r"|\boldest\b|\byoungest\b|\bbiggest\b|\blongest\b|\bshortest\b|\bfewest\b"
                 r"|\bearliest\b|\blatest\b|\bmaximum\b|\bminimum\b|\btop\b"),
    "limit": (r"\bmost\b|\bleast\b|\bhighest\b|\blowest\b|\blargest\b|\bsmallest\b|\boldest\b"
              r"|\byoungest\b|\bbiggest\b|\blongest\b|\bshortest\b|\bfewest\b|\bearliest\b"
              r"|\blatest\b|\btop \d+\b|\bfirst \d+\b"),
    "aggregate": r"\baverage\b|\bmean\b|\bmaximum\b|\bminimum\b|\btotal\b|\bsum\b|\bmax\b|\bmin\b",
    "distinct": r"\bdistinct\b|\bdifferent\b|\bunique\b",
    "anti_join": r"\bnot\b|\bnever\b|\bwithout\b|\bno\b|n't\b|\bexcept\b|\bother than\b",
    "intersect": r"\bboth\b|\band also\b",
    "union": r"\beither\b|\bor\b",
    "having": (r"\bat least\b|\bat most\b|\bmore than (one|two|three|\d+)\b|\bfewer than\b"
               r"|\bless than (two|three|\d+)\b|\btwo or more\b"),
    "nested": (r"\b(above|below|greater|less|more|higher|lower|larger|smaller|older|younger)"
               r" than (the )?(average|any|all)\b"),
    "like": r"\bcontain|\bsubstring\b|\bstarts? with\b|\bends? with\b|\bthe (word|letter|string)\b",
}


def question_cues(question, schema_info=None):
    """Inference-time pattern guesses from the question (and the target schema's types)."""
    q = question.lower()
    cues = {"select"}
    for name, rx in _CUES.items():
        if re.search(rx, q):
            cues.add(name)
    if "having" in cues and "group_by" not in cues:
        cues.add("group_by")
    if schema_info:
        if len(mentioned_collections(question, schema_info)) >= 2:
            cues.add("join")
        if schema_info.get("text_number_fields"):
            cues.add("text_number")
    return cues


def mql_operators(obj):
    ops = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("$"):
                ops.add(k)
            ops |= mql_operators(v)
    elif isinstance(obj, list):
        for v in obj:
            ops |= mql_operators(v)
    return ops


# ---------------------------------------------------------------- tokenisation and masking
_WORD = re.compile(r"[a-z0-9]+")
NUMERIC_SQL = re.compile(r"INT|REAL|NUM|DEC|DOUBLE|FLOAT|BIGINT|SMALLINT|YEAR", re.I)


def _name_parts(name):
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return {w for w in _WORD.findall(name.lower()) if len(w) > 1}


def _variants(w):
    out = {w}
    if w.endswith("ies"):
        out.add(w[:-3] + "y")
    elif w.endswith("es"):
        out |= {w[:-2], w[:-1]}
    elif w.endswith("s"):
        out.add(w[:-1])
    else:
        out |= {w + "s", w + "es"}
    return out


def schema_info(db_id, collections_manifest):
    """Schema-only facts for one database (inference-time information).

    collections_manifest: data/interim/spider1/atlas_collections.json content.
    """
    tables, words, text_num = {}, set(), []
    for name, info in sorted(collections_manifest.items()):
        if info["db_id"] != db_id:
            continue
        t = info["table"]
        tables[name] = _name_parts(t)
        words |= _name_parts(t)
        for f, types in info["fields"].items():
            words |= _name_parts(f)
            decl = info["declared"].get(f) or ""
            if "string" in types and NUMERIC_SQL.search(decl):
                text_num.append(f"{name}.{f}")
    vocab = set()
    for w in words:
        vocab |= _variants(w)
    return {"db_id": db_id, "tables": tables, "schema_words": vocab,
            "text_number_fields": text_num}


def mentioned_collections(question, info):
    toks = set(_WORD.findall(question.lower()))
    hit = []
    for coll, parts in info["tables"].items():
        if parts and all(any(v in toks for v in _variants(p)) for p in parts):
            hit.append(coll)
    return hit


def tokenize(text):
    return _WORD.findall(text.lower())


def mask_question(question, info):
    """Replace schema words by SCH, numbers by NUM, quoted / capitalised literals by VAL."""
    q = _STR_LIT.sub(" VAL ", question)
    out = []
    raw = re.findall(r"[A-Za-z0-9']+", q)
    for i, w in enumerate(raw):
        lw = w.lower().strip("'")
        if w == "VAL":
            out.append("val")
        elif re.fullmatch(r"\d+(\.\d+)?", lw):
            out.append("num")
        elif info and lw in info["schema_words"]:
            out.append("sch")
        elif i > 0 and w[:1].isupper() and lw not in _KEEP_CAPS:
            out.append("val")
        else:
            out.append(lw)
    return " ".join(out)


_KEEP_CAPS = {"i", "what", "which", "who", "how", "list", "show", "find", "give", "return"}
