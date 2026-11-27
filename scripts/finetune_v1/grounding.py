"""Question <-> reference-SQL agreement checks (deterministic, documented).

A pair passes only if
  (1) every literal in the SQL (strings, numbers, dates, LIKE fragments, LIMIT/OFFSET and
      INTERVAL amounts) is stated in the question, allowing the explicit equivalences listed
      below (date formats, number words, 'last week' = 7 days / 1 week, 0/1/TRUE/FALSE for
      boolean-looking columns, LIMIT 1 for singular superlatives), and
  (2) every quoted value and every free-standing number in the question appears in the SQL.
(1) rejects invented literals ("a specific role" -> roles_id = 789) and placeholders;
(2) rejects label noise where the SQL drops or changes a stated value ('Pharmacist D' in the
question, 'Pharmacist B' in the SQL). Neither check can prove the SQL answers the question;
they remove the classes of disagreement that can be detected without a model.
"""
import re

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"])}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9
NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
             "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50, "hundred": 100,
             "a hundred": 100, "thousand": 1000, "single": 1, "a": 1, "an": 1}
# period words that state an interval amount without a number ('in the last week')
PERIOD_AMOUNTS = {"week": {1, 7}, "month": {1, 30, 31}, "year": {1, 12, 365}, "day": {1, 24},
                  "hour": {1, 60}, "minute": {1, 60}, "fortnight": {14, 2}, "yesterday": {1},
                  "today": {0, 1}, "tomorrow": {1}, "quarter": {3, 90}, "decade": {10}}
SUPERLATIVE = re.compile(r"\b(latest|earliest|newest|oldest|most|least|highest|lowest|maximum|"
                         r"minimum|max|min|top|first|last|single|one|best|worst|largest|"
                         r"smallest|biggest|longest|shortest|recent|cheapest|priciest)\b")
BOOLISH_COL = re.compile(r"(^is_|^has_|^can_|active|enabled|disabled|verified|deleted|open|"
                         r"closed|flag|approved|archived|visible|completed|paid|available)", re.I)
PLACEHOLDER = re.compile(r"<[^<>]*>|\[[A-Za-z_ ]+\]|\byour_\w+|\bspecific_\w+|:\w+_id\b|\?\s*(?:$|[,)])|"
                         r"\{[a-z_]+\}|\bxxx+\b", re.I)


def placeholders(sql):
    return [m.group(0) for m in PLACEHOLDER.finditer(sql)]


def _qdates(q):
    """All calendar dates / year-months / years stated in the question."""
    ql = q.lower()
    days, months, years = set(), set(), set()

    def add(y, m, d):
        try:
            y, m, d = int(y), int(m), int(d)
        except (TypeError, ValueError):
            return
        if 1 <= m <= 12 and 1 <= d <= 31:
            days.add(f"{y:04d}-{m:02d}-{d:02d}")
            months.add(f"{y:04d}-{m:02d}")
            years.add(f"{y:04d}")

    for y, m, d in re.findall(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", ql):
        add(y, m, d)
    for a, b, y in re.findall(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b", ql):
        add(y, a, b)
        add(y, b, a)
    mon = "|".join(sorted(MONTHS, key=len, reverse=True))
    for m, d, y in re.findall(rf"\b({mon})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", ql):
        add(y, MONTHS[m], d)
    for d, m, y in re.findall(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({mon})\.?,?\s+(\d{{4}})\b", ql):
        add(y, MONTHS[m], d)
    for m, y in re.findall(rf"\b({mon})\.?,?\s+(?:of\s+)?(\d{{4}})\b", ql):
        months.add(f"{int(y):04d}-{MONTHS[m]:02d}")
        years.add(y)
    for y, m in re.findall(r"\b(\d{4})[-/](\d{1,2})\b(?![-/]\d)", ql):
        if 1 <= int(m) <= 12:
            months.add(f"{int(y):04d}-{int(m):02d}")
    for y in re.findall(r"\b((?:19|20)\d{2})\b", ql):
        years.add(y)
    return days, months, years


def _qtimes(q):
    ql = q.lower()
    out = set()
    for h, mi, ap in re.findall(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", ql):
        h = int(h) % 12 + (12 if ap.startswith("p") else 0)
        out.add(f"{h:02d}:{int(mi or 0):02d}")
    for h, mi in re.findall(r"\b(\d{1,2}):(\d{2})\b", ql):
        out.add(f"{int(h):02d}:{mi}")
    if "midnight" in ql:
        out.add("00:00")
    if "noon" in ql:
        out.add("12:00")
    return out


def _qnumbers(q):
    ql = q.lower().replace(",000", "000")
    nums = set()
    for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])", ql):
        nums.add(float(n))
    for n in re.findall(r"\$?(\d+(?:\.\d+)?)k\b", ql):
        nums.add(float(n) * 1000)
    for w, v in NUM_WORDS.items():
        if re.search(rf"\b{w}\b", ql):
            nums.add(float(v))
    return nums


DATE_LIT = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?$")
TIME_LIT = re.compile(r"^(\d{2}):(\d{2})(?::(\d{2}))?$")


def date_grounded(lit, q):
    m = DATE_LIT.match(lit)
    if not m:
        return False
    days, months, years = _qdates(q)
    y, mo, d = m.group(1), m.group(2), m.group(3)
    hh, mm = m.group(4), m.group(5)
    if hh is not None and f"{hh}:{mm}" not in ("00:00", "23:59") and f"{hh}:{mm}" not in _qtimes(q):
        return False
    iso = f"{y}-{mo}-{d}"
    if iso in days:
        return True
    # month bounds of a stated month, year bounds of a stated year
    if f"{y}-{mo}" in months and (d == "01" or int(d) >= 28):
        return True
    if y in years and ((mo, d) in (("01", "01"), ("12", "31"))):
        return True
    return False


def string_grounded(lit, q, op):
    ql = q.lower()
    s = lit.lower()
    if op == "like":
        parts = [p for p in re.split(r"[%_]+", s) if p.strip()]
        return all(p.strip() in ql or p.strip().replace("_", " ") in ql for p in parts) if parts else True
    if DATE_LIT.match(lit):
        return date_grounded(lit, q)
    if TIME_LIT.match(lit):
        return lit[:5] in _qtimes(q)
    if s in ql or s.replace("_", " ") in ql or s.replace(" ", "_") in ql:
        return True
    if s in ("true", "false", "yes", "no", "y", "n", "1", "0"):
        return True
    return False


def number_grounded(v, q, ctx):
    nums = _qnumbers(q)
    if float(v) in nums:
        return True
    if v == 0:
        return True
    ql = q.lower()
    if ctx.get("bool_column") and v in (0, 1):
        return True
    if ctx.get("role") == "limit" and v == 1 and SUPERLATIVE.search(ql):
        return True
    if ctx.get("role") == "interval":
        unit = ctx.get("unit", "")
        for word, amounts in PERIOD_AMOUNTS.items():
            if re.search(rf"\b{word}s?\b|\b{word}ly\b", ql) and v in amounts:
                return True
        if unit == "day" and v in (7, 14, 30, 31, 90, 365) and re.search(r"\b(week|month|quarter|year)s?\b", ql):
            if (v // 7) in nums or (v // 30) in nums or v in nums:
                return True
    if ctx.get("role") == "date_part":
        days, months, years = _qdates(q)
        if str(v) in years or any(m.endswith(f"-{int(v):02d}") for m in months):
            return True
        if any(MONTHS.get(w) == v for w in re.findall(r"[a-z]+", ql)):
            return True
    if ctx.get("role") == "percent" and float(v) * 100 in nums:
        return True
    return False


QUOTED = re.compile(r"(?:(?<=\s)|(?<=^)|(?<=[(:=]))'([^']{1,80})'(?=[\s,.;:)?!]|$)|\"([^\"]{1,80})\"|`([^`]{1,80})`")


def question_values(q):
    """Quoted values and free-standing numbers stated by the question."""
    quoted = [a or b or c for a, b, c in QUOTED.findall(q)]
    stripped = QUOTED.sub(" ", q)
    # remove date expressions so their digits are not treated as free numbers
    stripped = re.sub(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?\b|"
                      r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", stripped)
    mon = "|".join(sorted(MONTHS, key=len, reverse=True))
    stripped = re.sub(rf"\b({mon})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b|"
                      rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?({mon})\b(?:,?\s+\d{{4}})?|"
                      rf"\b({mon})\.?,?\s+\d{{4}}\b", " ", stripped, flags=re.I)
    stripped = re.sub(r"\b\w*[a-z_]\d+\w*\b|\b\d+[a-z_]\w*\b", " ", stripped, flags=re.I)
    nums = [float(n) for n in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.]?\d)", stripped)]
    return quoted, nums


def check(question, sql_text, literals, extra_numbers):
    """-> list of failure reasons ([] = grounded).

    literals: Analysis.literals; extra_numbers: [(value, ctx)] for LIMIT/OFFSET/INTERVAL/etc.
    """
    fails = []
    for lit in literals:
        k, v = lit["kind"], lit["value"]
        if k in ("time", "null"):
            continue
        if k == "bool":
            continue
        if k == "str":
            if not string_grounded(v, question, lit["op"]):
                fails.append(f"sql_literal_not_in_question:{v!r}")
        elif k in ("int", "float"):
            role = "date_part" if lit.get("fn") in ("year", "month", "day") else "value"
            ctx = {"role": role, "bool_column": bool(BOOLISH_COL.search(lit["column"]))}
            if not number_grounded(v, question, ctx):
                fails.append(f"sql_literal_not_in_question:{v!r}")
    for v, ctx in extra_numbers:
        if not number_grounded(v, question, ctx):
            fails.append(f"sql_{ctx.get('role')}_not_in_question:{v!r}")
    quoted, nums = question_values(question)
    sl = sql_text.lower()
    sql_nums = set()
    for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])", sql_text):
        sql_nums.add(float(n))
    for qv in quoted:
        if qv.lower() not in sl and qv.lower().replace(" ", "_") not in sl:
            if not (DATE_LIT.match(qv) and qv[:10] in sl):
                fails.append(f"question_value_not_in_sql:{qv!r}")
    for n in nums:
        if n not in sql_nums and not (n == int(n) and f"{int(n)}" in sl):
            fails.append(f"question_number_not_in_sql:{n:g}")
    return fails


# ---------------------------------------------------------------- identifiers and structure
GENERIC_TOKENS = {"id", "ids", "at", "by", "is", "of", "the", "table", "tbl", "no", "num", "col",
                  "data", "info", "value", "has", "can", "on", "in", "to", "dt", "fk", "pk"}
ABBREV = {"qty": "quantity", "desc": "description", "dob": "birth", "org": "organization",
          "orgs": "organization", "addr": "address", "amt": "amount", "num": "number",
          "dept": "department", "emp": "employee", "cust": "customer", "prod": "product",
          "txn": "transaction", "msg": "message", "img": "image", "pharma": "pharmacy",
          "fullname": "name", "lname": "name", "fname": "name", "zip": "zip", "tel": "phone"}
TIME_CUES = re.compile(r"\b(day|days|week|weeks|month|months|year|years|hour|hours|minute|"
                       r"minutes|today|yesterday|tomorrow|recent|recently|latest|earliest|newest|"
                       r"oldest|ago|since|before|after|between|date|dates|time|timestamp|created|"
                       r"updated|when|current|last|past|next|upcoming|future|expired|from|until|"
                       r"\d{4})\b|\d{4}-\d{2}", re.I)


def stem(w):
    w = w.lower()
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ses", "s"), ("es", "e"), ("s", ""),
                     ("ing", ""), ("ed", "")):
        if len(w) > 4 and w.endswith(suf):
            return w[: len(w) - len(suf)] + rep
    return w


def _qstems(q):
    words = re.findall(r"[a-z]+", q.lower().replace("_", " "))
    return {stem(w) for w in words} | set(words)


def ident_mentioned(ident, q, ctype=None):
    ql = q.lower()
    compact_q = re.sub(r"[^a-z0-9]", "", ql)
    il = ident.lower()
    if il in ql or il.replace("_", " ") in ql or il.replace("_", "") in compact_q:
        return True
    toks = [t for t in re.split(r"[_\W\d]+", il) if t]
    toks = [ABBREV.get(t, t) for t in toks]
    content = [t for t in toks if t not in GENERIC_TOKENS and len(t) >= 2]
    if not content:
        return bool(re.search(r"\b(id|ids|identifier|identifiers|key)\b", ql)) or ident.lower() in ("id",)
    qs = _qstems(q)
    for t in content:
        st = stem(t)
        if t in qs or st in qs or (len(st) >= 4 and any(x.startswith(st) or st.startswith(x)
                                                        for x in qs if len(x) >= 4)):
            return True
        if len(t) >= 5 and t in compact_q:
            return True
    if ctype in ("date", "datetime") and TIME_CUES.search(q):
        return True
    return False


AGG_CUES = {
    "count": r"\b(count|counts|counted|number of|how many|total number|tally|quantity of)\b",
    "avg": r"\b(average|avg|mean)\b",
    "sum": r"\b(total|sum|overall|combined|aggregate)\b",
    "max": r"\b(max|maximum|highest|largest|latest|most recent|biggest|greatest|top|newest|last|"
           r"longest|peak|most)\b",
    "min": r"\b(min|minimum|lowest|smallest|earliest|oldest|first|cheapest|least|shortest)\b",
}
ORDER_CUE = re.compile(r"\b(order|ordered|ordering|sort|sorted|sorting|rank|ranked|ascending|"
                       r"descending|asc|desc|alphabetical|alphabetically|latest|earliest|top|"
                       r"bottom|highest|lowest|first|last|most|least|newest|oldest|recent|"
                       r"chronological|chronologically|largest|smallest|biggest|best|worst|"
                       r"increasing|decreasing)\b", re.I)
GROUP_CUE = re.compile(r"\b(each|per|by|group|grouped|grouping|every|breakdown|distribution|"
                       r"respective|categor\w*)\b", re.I)
DISTINCT_CUE = re.compile(r"\b(distinct|unique|different|deduplicated|without duplicates|"
                          r"each|all the)\b", re.I)
FILTER_CUE = re.compile(r"\b(after|before|between|greater|less|more than|fewer|above|below|"
                        r"where|whose|equal|equals|starting|starts|ending|ends|contain|contains|"
                        r"containing|within|since|older than|newer than|not null|is null|"
                        r"exceed|exceeds|at least|at most|in the last|past \d+)\b", re.I)
SQL_IN_TEXT = re.compile(r"\bSELECT\s+(\*|DISTINCT|[\w.]+\s*(,|FROM))[^\n]*\bFROM\b|"
                         r"\bWHERE\s+\w+\s*(=|>|<|LIKE|BETWEEN|IN)\s")
UNDERSPEC = re.compile(r"\b(specific|particular|certain|given|some specified|chosen)\b", re.I)


def structure_check(question, ast, a, schema):
    """Question/SQL agreement on referenced identifiers and query shape -> [failures]."""
    from sqlglot import exp as E
    fails = []
    q = question
    tabs = set(a.columns)
    for t in sorted(tabs):
        if not ident_mentioned(t, q):
            # a table may go unmentioned when the question names its columns instead
            cols = a.columns[t]
            if not cols or not all(ident_mentioned(c, q, schema.get(t, {}).get(c)) for c in cols):
                fails.append(f"table_not_in_question:{t}")
    for t, cols in sorted(a.columns.items()):
        for c in sorted(cols):
            if not ident_mentioned(c, q, schema.get(t, {}).get(c)):
                fails.append(f"column_not_in_question:{t}.{c}")
    aggs = {type(n).__name__.lower() for n in ast.find_all(E.Count, E.Sum, E.Avg, E.Min, E.Max)}
    for g in sorted(aggs):
        if not re.search(AGG_CUES[g], q, re.I):
            fails.append(f"aggregate_not_in_question:{g}")
    ql = q.lower()
    if not aggs and re.search(r"\b(how many|number of|count of|average|total number)\b", ql):
        fails.append("question_asks_aggregate_sql_has_none")
    if ast.args.get("order") is not None and not ORDER_CUE.search(q):
        fails.append("order_by_not_in_question")
    if ast.args.get("group") is not None and not GROUP_CUE.search(q):
        fails.append("group_by_not_in_question")
    if ast.args.get("distinct") is not None and not DISTINCT_CUE.search(q):
        fails.append("distinct_not_in_question")
    if ast.args.get("where") is None and ast.args.get("having") is None and FILTER_CUE.search(q):
        fails.append("question_implies_filter_sql_has_none")
    quoted, nums = question_values(q)
    if UNDERSPEC.search(q) and not quoted and not nums and not _qdates(q)[2]:
        fails.append("underspecified_question")
    return fails
