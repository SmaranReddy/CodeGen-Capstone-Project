"""Reference-SQL semantics for ft-v1: parse, analyse, run on SQLite fixtures, mutate.

* `parse_sql` tries the dialects in DIALECTS order with sqlglot and records the first that
  parses the text as exactly one statement.
* `analyze` walks the AST and reports tables/aliases, referenced columns, literals (with the
  column and operator they are compared with), query features and every construct outside
  the supported subset (`unsupported`). Only supported queries are executed and converted.
* `to_sqlite` renders the AST for SQLite with MySQL semantics where the engines differ:
  NOW()/CURDATE()/INTERVAL arithmetic/YEAR()/MONTH()/DAY()/DATE() become Python UDFs bound
  to a frozen UTC clock, TRUE/FALSE become 1/0, and `/` is real division (MySQL), not
  SQLite integer division. LIKE/ILIKE use SQLite LIKE (ASCII case-insensitive, like MySQL's
  default collation); `=` on text is case-sensitive (SQLite BINARY).
* `mutants` yields single-edit variants of the query (drop or negate one WHERE/HAVING atom,
  flip one ORDER BY direction, drop LIMIT/OFFSET/DISTINCT). A fixture set is adequate only
  if every mutant's result differs from the original on at least one fixture.
"""
import datetime as dt
import re
import sqlite3

import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel

DIALECTS = ("mysql", "postgres", "tsql", "sqlite")

TIME_NOW = {"NOW", "GETDATE", "SYSDATE", "LOCALTIMESTAMP"}
TIME_TODAY = {"CURDATE"}
UNIT_ALIASES = {
    "SECOND": "second", "SECONDS": "second", "MINUTE": "minute", "MINUTES": "minute",
    "HOUR": "hour", "HOURS": "hour", "DAY": "day", "DAYS": "day", "WEEK": "week",
    "WEEKS": "week", "MONTH": "month", "MONTHS": "month", "YEAR": "year", "YEARS": "year",
}
UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 7 * 86400,
                "month": 31 * 86400, "year": 366 * 86400}

CMP = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
BOOL_OPS = (exp.And, exp.Or, exp.Not, exp.Paren)
AGGS = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
DATE_PART = (exp.Year, exp.Month, exp.Day)
BOOLISH = re.compile(r"(^is_|^has_|^can_|active|enabled|disabled|verified|deleted|open|closed|"
                     r"flag|approved|archived|visible|completed|paid|available|fixed)", re.I)


# ---------------------------------------------------------------- parsing
def parse_sql(sql):
    """-> (ast | None, dialect | None, error | None)."""
    last = None
    for d in DIALECTS:
        try:
            trees = [t for t in sqlglot.parse(sql, read=d, error_level=ErrorLevel.RAISE) if t is not None]
        except Exception as e:  # noqa: BLE001 - any parser failure means "try the next dialect"
            last = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
            continue
        if len(trees) != 1:
            last = f"{len(trees)} statements"
            continue
        return trees[0], d, None
    return None, None, last


def statement_kind(ast):
    return type(ast).__name__.lower()


# ---------------------------------------------------------------- interval helpers
def interval_of(node):
    """Interval / DateSub / DateAdd operand -> (amount: int|float, unit) or None."""
    if isinstance(node, exp.Interval):
        lit, unit = node.this, node.args.get("unit")
        text = lit.this if isinstance(lit, exp.Literal) else None
        if text is None:
            return None
        text = str(text).strip()
        u = unit.name.upper() if unit is not None else None
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*([A-Za-z]+)?", text)
        if not m:
            return None
        amount, u2 = m.group(1), m.group(2)
        u = u or (u2.upper() if u2 else None)
        if u not in UNIT_ALIASES:
            return None
        amount = float(amount)
        return (int(amount) if amount.is_integer() else amount), UNIT_ALIASES[u]
    return None


def _date_shift(node):
    """Date-arithmetic node -> (base, signed amount, unit) or None."""
    if isinstance(node, (exp.DateSub, exp.DateAdd)):
        lit, unit = node.expression, node.args.get("unit")
        if not isinstance(lit, exp.Literal) or unit is None:
            return None
        u = UNIT_ALIASES.get(unit.name.upper())
        try:
            a = float(lit.this)
        except ValueError:
            return None
        if u is None:
            return None
        a = int(a) if a.is_integer() else a
        return node.this, (-a if isinstance(node, exp.DateSub) else a), u
    if isinstance(node, (exp.Sub, exp.Add)) and isinstance(node.expression, exp.Interval):
        iv = interval_of(node.expression)
        if iv is None:
            return None
        a, u = iv
        return node.this, (-a if isinstance(node, exp.Sub) else a), u
    return None


def time_fn(node):
    """'now' | 'today' | None for clock functions."""
    if isinstance(node, exp.CurrentTimestamp):
        return "now"
    if isinstance(node, exp.CurrentDate):
        return "today"
    if isinstance(node, exp.Anonymous) and not node.expressions:
        n = node.name.upper()
        if n in TIME_NOW:
            return "now"
        if n in TIME_TODAY:
            return "today"
    return None


def unwrap_date(node):
    """sqlglot wraps MySQL YEAR(x) arguments in TsOrDsToDate; strip that wrapper."""
    while isinstance(node, (exp.TsOrDsToDate,)):
        node = node.this
    return node


# ---------------------------------------------------------------- analysis
class Analysis:
    def __init__(self):
        self.base = None            # base table name
        self.aliases = {}           # alias/name -> table
        self.joins = []             # [{table, alias, kind, left:(t,c), right:(t,c)}]
        self.columns = {}           # table -> set(columns)
        self.literals = []          # [{table, column, op, kind, value, raw}]
        self.features = set()
        self.unsupported = []
        self.select_aliases = set()
        self.where_atoms = 0
        self.time_relative = False
        self.max_interval_s = 0
        self.order_columns = set()  # (table, col) used as sort keys
        self.group_columns = set()
        self.join_keys = set()
        self.limit = None
        self.offset = None

    def to_json(self):
        return {"base": self.base, "aliases": self.aliases, "joins": self.joins,
                "columns": {t: sorted(c) for t, c in sorted(self.columns.items())},
                "literals": self.literals, "features": sorted(self.features),
                "unsupported": self.unsupported, "where_atoms": self.where_atoms,
                "time_relative": self.time_relative, "max_interval_s": self.max_interval_s,
                "order_columns": sorted(map(list, self.order_columns)),
                "group_columns": sorted(map(list, self.group_columns)),
                "join_keys": sorted(map(list, self.join_keys)),
                "limit": self.limit, "offset": self.offset}


def _lit_value(node):
    if isinstance(node, exp.Boolean):
        return "bool", bool(node.this)
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal) and not node.this.is_string:
        k, v = _lit_value(node.this)
        return k, -v
    if isinstance(node, exp.Literal):
        if node.is_string:
            return "str", node.this
        t = node.this
        try:
            return ("int", int(t)) if re.fullmatch(r"-?\d+", t) else ("float", float(t))
        except ValueError:
            return "str", t
    return None, None


def is_literal(node):
    return isinstance(node, (exp.Literal, exp.Boolean)) or (
        isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal))


def analyze(ast):
    a = Analysis()
    if not isinstance(ast, exp.Select):
        a.unsupported.append(f"statement:{statement_kind(ast)}")
        return a
    for bad, name in ((exp.Subquery, "subquery"), (exp.Union, "union"), (exp.With, "cte"),
                      (exp.Window, "window"), (exp.Case, "case"), (exp.If, "if")):
        if ast.find(bad):
            a.unsupported.append(name)
    if a.unsupported:
        return a
    frm = ast.args.get("from")
    if frm is None or not isinstance(frm.this, exp.Table) or frm.this.args.get("db"):
        a.unsupported.append("from:not_a_plain_table")
        return a
    a.base = frm.this.name
    a.aliases[a.base] = a.base
    if frm.this.alias:
        a.aliases[frm.this.alias] = a.base
    a.columns[a.base] = set()
    for j in ast.args.get("joins") or []:
        t = j.this
        if not isinstance(t, exp.Table):
            a.unsupported.append("join:not_a_table")
            return a
        kind = (j.side or j.kind or "inner").lower()
        if kind not in ("inner", "left", "") or j.args.get("using"):
            a.unsupported.append(f"join:{kind or 'cross'}")
            return a
        a.aliases[t.name] = t.name
        if t.alias:
            a.aliases[t.alias] = t.name
        a.columns.setdefault(t.name, set())
        on = j.args.get("on")
        if not (isinstance(on, exp.EQ) and isinstance(on.this, exp.Column)
                and isinstance(on.expression, exp.Column)):
            a.unsupported.append("join:non_equi_on")
            return a
        a.joins.append({"table": t.name, "alias": t.alias or t.name, "kind": kind or "inner",
                        "on": [on.this.sql(), on.expression.sql()]})
        a.features.add("join")
    if len(set(a.aliases.values())) != len(a.aliases.values()) and len(a.joins) and \
            len({j["table"] for j in a.joins} | {a.base}) != len(a.joins) + 1:
        a.unsupported.append("join:self_join")
        return a

    # select-list aliases may be referenced by ORDER BY / HAVING
    for e in ast.expressions:
        if isinstance(e, exp.Alias):
            a.select_aliases.add(e.alias)

    def resolve(col):
        q = col.table
        if q:
            t = a.aliases.get(q)
            if t is None:
                a.unsupported.append(f"unknown_qualifier:{q}")
                return None
            return t
        if a.joins:
            a.unsupported.append("join:unqualified_column")
            return None
        return a.base

    for col in ast.find_all(exp.Column):
        if not col.table and col.name in a.select_aliases and not _in_where(col):
            continue
        if isinstance(col.this, exp.Star):
            a.unsupported.append("qualified_star")
            continue
        t = resolve(col)
        if t:
            a.columns[t].add(col.name)
    if a.unsupported:
        return a
    for j, jn in zip(ast.args.get("joins") or [], a.joins):
        on = j.args["on"]
        for c in (on.this, on.expression):
            a.join_keys.add((a.aliases[c.table] if c.table else a.base, c.name))

    _check_select(ast, a)
    where = ast.args.get("where")
    if where is not None:
        a.features.add("where")
        _check_pred(where.this, a, resolve, "where")
    grp = ast.args.get("group")
    if grp is not None:
        a.features.add("group_by")
        for g in grp.expressions:
            if isinstance(g, exp.Column):
                t = resolve(g)
                if t:
                    a.group_columns.add((t, g.name))
            elif not _ok_value_expr(unwrap_date(g) if isinstance(g, DATE_PART) else g, a):
                a.unsupported.append(f"group_by_expr:{type(g).__name__}")
    hav = ast.args.get("having")
    if hav is not None:
        a.features.add("having")
        _check_pred(hav.this, a, resolve, "having")
    order = ast.args.get("order")
    if order is not None:
        a.features.add("order_by")
        for o in order.expressions:
            e = o.this
            if isinstance(e, exp.Column) and not (not e.table and e.name in a.select_aliases):
                t = resolve(e)
                if t:
                    a.order_columns.add((t, e.name))
            elif isinstance(e, exp.Literal) and not e.is_string:
                a.features.add("order_by_ordinal")
            elif isinstance(e, AGGS) or isinstance(e, exp.Column):
                pass
            elif not _ok_value_expr(e, a):
                a.unsupported.append(f"order_by_expr:{type(e).__name__}")
    for key in ("limit", "offset"):
        node = ast.args.get(key)
        if node is not None:
            v = node.expression
            if not (isinstance(v, exp.Literal) and not v.is_string and re.fullmatch(r"\d+", v.this)):
                a.unsupported.append(f"{key}:non_literal")
            else:
                setattr(a, key, int(v.this))
                a.features.add(key)
    if ast.args.get("distinct"):
        a.features.add("distinct")
        if ast.args["distinct"].args.get("on"):
            a.unsupported.append("distinct_on")
    if a.time_relative:
        a.features.add("time_relative")
    a.unsupported = sorted(set(a.unsupported))
    return a


def _in_where(node):
    p = node.parent
    while p is not None:
        if isinstance(p, exp.Where):
            return True
        p = p.parent
    return False


def _check_select(ast, a):
    for e in ast.expressions:
        inner = e.this if isinstance(e, exp.Alias) else e
        if isinstance(inner, exp.Star):
            a.features.add("select_star")
            if a.joins:
                a.unsupported.append("select_star_with_join")
            continue
        if isinstance(inner, AGGS):
            a.features.add("aggregate")
            a.features.add(type(inner).__name__.lower())
            arg = inner.this
            if isinstance(arg, exp.Distinct):
                a.features.add("count_distinct")
                if len(arg.expressions) != 1 or not isinstance(arg.expressions[0], exp.Column):
                    a.unsupported.append("agg_distinct_expr")
                continue
            if isinstance(arg, exp.Star):
                if not isinstance(inner, exp.Count):
                    a.unsupported.append("agg_star")
                continue
            if not _ok_value_expr(arg, a):
                a.unsupported.append(f"agg_arg:{type(arg).__name__}")
            continue
        if not _ok_value_expr(inner, a):
            a.unsupported.append(f"select_expr:{type(inner).__name__}")


def _ok_value_expr(e, a):
    """Value expressions the converter supports (outside predicates)."""
    if isinstance(e, exp.Column) or is_literal(e):
        return True
    if isinstance(e, DATE_PART):
        a.features.add("date_part")
        return isinstance(unwrap_date(e.this), exp.Column)
    if isinstance(e, exp.TsOrDsToDate) or (isinstance(e, exp.Date) and len(e.expressions) <= 1):
        a.features.add("date_part")
        return isinstance(unwrap_date(e.this), exp.Column)
    if isinstance(e, (exp.Lower, exp.Upper, exp.Length)):
        a.features.add("string_fn")
        return isinstance(e.this, exp.Column)
    if isinstance(e, (exp.Add, exp.Sub, exp.Mul, exp.Div)) and not isinstance(e.expression, exp.Interval):
        a.features.add("arithmetic")
        return _ok_value_expr(e.this, a) and _ok_value_expr(e.expression, a)
    if isinstance(e, exp.Paren):
        return _ok_value_expr(e.this, a)
    if isinstance(e, AGGS):
        arg = e.this
        return isinstance(arg, exp.Star) or isinstance(arg, exp.Column) or (
            isinstance(arg, exp.Distinct) and len(arg.expressions) == 1)
    return False


def _time_expr(e, a):
    """Clock expression (NOW(), CURDATE(), +/- INTERVAL, DATE_SUB/ADD) -> True if supported."""
    if time_fn(e):
        a.time_relative = True
        return True
    sh = _date_shift(e)
    if sh is not None:
        base, amount, unit = sh
        if time_fn(base) or _date_shift(base) is not None:
            a.time_relative = True
            a.max_interval_s = max(a.max_interval_s, abs(amount) * UNIT_SECONDS[unit])
            a.features.add("interval")
            return _time_expr(base, a)
        return False
    if isinstance(e, exp.TsOrDsToDate) or (isinstance(e, exp.Date) and len(e.expressions) == 0):
        return _time_expr(e.this, a) if e.this is not None and not isinstance(e.this, exp.Column) else False
    return False


def _operand(e, a, resolve, ctx):
    """-> (kind, info): kind in column|literal|time|agg|value|bad."""
    if isinstance(e, exp.Paren):
        return _operand(e.this, a, resolve, ctx)
    if isinstance(e, exp.Column):
        if not e.table and e.name in a.select_aliases and ctx == "having":
            return "alias", e.name
        t = resolve(e)
        return ("column", (t, e.name)) if t else ("bad", None)
    if is_literal(e):
        return "literal", _lit_value(e)
    if isinstance(e, exp.Null):
        return "null", None
    if _time_expr(e, a):
        return "time", None
    if isinstance(e, DATE_PART) and isinstance(unwrap_date(e.this), exp.Column):
        a.features.add("date_part")
        c = unwrap_date(e.this)
        t = resolve(c)
        return ("fn", (t, c.name, type(e).__name__.lower())) if t else ("bad", None)
    if isinstance(e, DATE_PART) and _time_expr(unwrap_date(e.this), a):
        return "time", None
    if (isinstance(e, exp.TsOrDsToDate) or isinstance(e, exp.Date)) and isinstance(e.this, exp.Column):
        a.features.add("date_part")
        t = resolve(e.this)
        return ("fn", (t, e.this.name, "date")) if t else ("bad", None)
    if isinstance(e, (exp.Lower, exp.Upper, exp.Length)) and isinstance(e.this, exp.Column):
        a.features.add("string_fn")
        t = resolve(e.this)
        return ("fn", (t, e.this.name, type(e).__name__.lower())) if t else ("bad", None)
    if isinstance(e, AGGS) and ctx == "having":
        return "agg", None
    if isinstance(e, (exp.Add, exp.Sub, exp.Mul, exp.Div)) and _ok_value_expr(e, a):
        return "value", None
    return "bad", type(e).__name__


def _record_literal(a, col_info, op, lit, fn=None):
    kind, val = lit
    t, c = col_info[0], col_info[1]
    a.literals.append({"table": t, "column": c, "op": op, "kind": kind, "value": val,
                       "fn": fn})


def _check_pred(e, a, resolve, ctx):
    if isinstance(e, BOOL_OPS):
        if isinstance(e, exp.Or):
            a.features.add("or")
        if isinstance(e, exp.Not):
            a.features.add("not")
        for child in ([e.this] if isinstance(e, (exp.Not, exp.Paren)) else [e.this, e.expression]):
            _check_pred(child, a, resolve, ctx)
        return
    if ctx == "where":
        a.where_atoms += 1
    if isinstance(e, CMP):
        lk, li = _operand(e.this, a, resolve, ctx)
        rk, ri = _operand(e.expression, a, resolve, ctx)
        if "bad" in (lk, rk) or "null" in (lk, rk):
            a.unsupported.append(f"{ctx}_operand:{li if lk == 'bad' else ri if rk == 'bad' else 'null_compare'}")
            return
        op = type(e).__name__.lower()
        if lk in ("column", "fn") and rk == "literal":
            _record_literal(a, li, op, ri, li[2] if lk == "fn" else None)
        elif rk in ("column", "fn") and lk == "literal":
            _record_literal(a, ri, op, li, ri[2] if rk == "fn" else None)
        if lk == "literal" and rk == "literal":
            a.unsupported.append(f"{ctx}_literal_vs_literal")
        if "time" in (lk, rk):
            for k, i in ((lk, li), (rk, ri)):
                if k == "column":
                    a.literals.append({"table": i[0], "column": i[1], "op": op, "kind": "time",
                                       "value": None, "fn": None})
        if ctx == "having" and "agg" not in (lk, rk) and "alias" not in (lk, rk):
            a.unsupported.append("having_without_aggregate")
        return
    if isinstance(e, exp.Between):
        k, i = _operand(e.this, a, resolve, ctx)
        lo, hi = e.args["low"], e.args["high"]
        if k not in ("column", "fn", "agg", "alias"):
            a.unsupported.append(f"{ctx}_between_operand")
            return
        for bound, op in ((lo, "gte"), (hi, "lte")):
            bk, bi = _operand(bound, a, resolve, ctx)
            if bk == "literal" and k in ("column", "fn"):
                _record_literal(a, i, op, bi, i[2] if k == "fn" else None)
            elif bk == "time" and k == "column":
                a.literals.append({"table": i[0], "column": i[1], "op": op, "kind": "time",
                                   "value": None, "fn": None})
            elif bk not in ("literal", "time"):
                a.unsupported.append(f"{ctx}_between_bound")
        a.features.add("between")
        return
    if isinstance(e, exp.In):
        k, i = _operand(e.this, a, resolve, ctx)
        if k not in ("column", "fn") or e.args.get("query") or not e.expressions or \
                not all(is_literal(x) for x in e.expressions):
            a.unsupported.append(f"{ctx}_in_form")
            return
        for x in e.expressions:
            _record_literal(a, i, "in", _lit_value(x), i[2] if k == "fn" else None)
        a.features.add("in")
        return
    if isinstance(e, (exp.Like, exp.ILike)):
        k, i = _operand(e.this, a, resolve, ctx)
        pat = e.expression
        if k not in ("column", "fn") or not (isinstance(pat, exp.Literal) and pat.is_string) or \
                e.args.get("escape"):
            a.unsupported.append(f"{ctx}_like_form")
            return
        _record_literal(a, i, "like", ("str", pat.this), i[2] if k == "fn" else None)
        a.features.add("like")
        return
    if isinstance(e, exp.Is):
        k, i = _operand(e.this, a, resolve, ctx)
        if k != "column" or not isinstance(e.expression, exp.Null):
            a.unsupported.append(f"{ctx}_is_form")
            return
        a.literals.append({"table": i[0], "column": i[1], "op": "is_null", "kind": "null",
                           "value": None, "fn": None})
        a.features.add("null_check")
        return
    if isinstance(e, exp.Column):
        if not BOOLISH.search(e.name):
            a.unsupported.append(f"{ctx}_bare_non_boolean_column(truncated_sql?)")
            return
        t = resolve(e)
        if t:
            a.literals.append({"table": t, "column": e.name, "op": "truthy", "kind": "bool",
                               "value": True, "fn": None})
            a.features.add("bare_boolean_column")
        return
    a.unsupported.append(f"{ctx}_predicate:{type(e).__name__}")


# ---------------------------------------------------------------- SQLite engine
def _parse_dt(s):
    if s is None:
        return None, None
    s = str(s)
    for fmt, kind in (("%Y-%m-%d %H:%M:%S", "datetime"), ("%Y-%m-%d", "date"),
                      ("%Y-%m-%dT%H:%M:%S", "datetime"), ("%Y-%m-%d %H:%M", "datetime")):
        try:
            return dt.datetime.strptime(s[:19] if kind == "datetime" else s[:10], fmt), kind
        except ValueError:
            continue
    return None, None


def _add_months(d, n):
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    import calendar
    day = min(d.day, calendar.monthrange(y, m)[1])
    return d.replace(year=y, month=m, day=day)


def shift(d, amount, unit):
    if unit in ("month", "year"):
        return _add_months(d, int(amount) * (12 if unit == "year" else 1))
    return d + dt.timedelta(seconds=amount * UNIT_SECONDS[unit] if unit != "week"
                            else amount * 7 * 86400)


def fmt_dt(d, kind):
    return d.strftime("%Y-%m-%d" if kind == "date" else "%Y-%m-%d %H:%M:%S")


def make_udfs(clock):
    """clock: aware/naive UTC datetime frozen for the whole evaluation of one fixture."""
    def now():
        return fmt_dt(clock, "datetime")

    def today():
        return fmt_dt(clock, "date")

    def dtadd(s, amount, unit):
        d, kind = _parse_dt(s)
        if d is None:
            return None
        if kind == "date" and unit in ("hour", "minute", "second"):
            kind = "datetime"
        return fmt_dt(shift(d, amount, unit), kind)

    def part(which):
        def f(s):
            d, _ = _parse_dt(s)
            if d is None:
                return None
            return {"year": d.year, "month": d.month, "day": d.day}[which]
        return f

    def dateonly(s):
        d, _ = _parse_dt(s)
        return None if d is None else fmt_dt(d, "date")

    return {"_NOW": (0, now), "_CURDATE": (0, today), "_DTADD": (3, dtadd),
            "_YEAR": (1, part("year")), "_MONTH": (1, part("month")), "_DAY": (1, part("day")),
            "_DATE": (1, dateonly)}


def _to_sqlite_tree(ast):
    def tr(node):
        tf = time_fn(node)
        if tf == "now":
            return exp.Anonymous(this="_NOW", expressions=[])
        if tf == "today":
            return exp.Anonymous(this="_CURDATE", expressions=[])
        sh = _date_shift(node)
        if sh is not None:
            base, amount, unit = sh
            return exp.Anonymous(this="_DTADD", expressions=[
                tr(base.copy()), exp.Literal.number(amount), exp.Literal.string(unit)])
        if isinstance(node, DATE_PART):
            return exp.Anonymous(this=f"_{type(node).__name__.upper()}",
                                 expressions=[tr(unwrap_date(node.this).copy())])
        if isinstance(node, exp.TsOrDsToDate) or (isinstance(node, exp.Date) and len(node.expressions) == 0):
            return exp.Anonymous(this="_DATE", expressions=[tr(node.this.copy())])
        if isinstance(node, exp.Boolean):
            return exp.Literal.number(1 if node.this else 0)
        if isinstance(node, exp.ILike):
            return exp.Like(this=node.this, expression=node.expression)
        if isinstance(node, exp.Div):
            return exp.Div(this=exp.Cast(this=node.this, to=exp.DataType.build("REAL")),
                           expression=node.expression)
        return node
    return ast.copy().transform(tr)


def to_sqlite(ast):
    return _to_sqlite_tree(ast).sql(dialect="sqlite", identify=True)


SQLITE_TYPES = {"int": "INTEGER", "float": "REAL", "bool": "INTEGER", "text": "TEXT",
                "date": "TEXT", "datetime": "TEXT"}


def sqlite_db(tables, schema, clock):
    """tables: {table: [row dicts]}; schema: {table: {col: type}} -> sqlite3 connection."""
    con = sqlite3.connect(":memory:")
    for name, (n, fn) in make_udfs(clock).items():
        con.create_function(name, n, fn, deterministic=True)
    for t, rows in tables.items():
        cols = list(schema[t])
        con.execute(f'CREATE TABLE "{t}" ({", ".join(chr(34) + c + chr(34) + " " + SQLITE_TYPES[schema[t][c]] for c in cols)})')
        con.executemany(f'INSERT INTO "{t}" VALUES ({", ".join("?" * len(cols))})',
                        [tuple(r.get(c) for c in cols) for r in rows])
    return con


def run_sqlite(con, sql_text):
    cur = con.execute(sql_text)
    cols = [d[0] for d in cur.description] if cur.description else []
    return cols, [tuple(r) for r in cur.fetchall()]


# ---------------------------------------------------------------- mutants
def _atoms(node):
    if isinstance(node, (exp.And, exp.Or)):
        return _atoms(node.this) + _atoms(node.expression)
    if isinstance(node, (exp.Paren, exp.Not)):
        return _atoms(node.this)
    return [node]


def mutants(ast):
    """-> [(label, mutated ast)]. Textually duplicated atoms are skipped (equivalent mutants)."""
    out = []
    for clause in ("where", "having"):
        c = ast.args.get(clause)
        if c is None:
            continue
        atoms = _atoms(c.this)
        texts = [x.sql() for x in atoms]
        for i, atom in enumerate(atoms):
            if texts.count(texts[i]) > 1:
                continue
            path = _path_to(c.this, atom)
            if isinstance(atom, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
                swap = {exp.GT: exp.GTE, exp.GTE: exp.GT, exp.LT: exp.LTE, exp.LTE: exp.LT}[type(atom)]
                m = ast.copy()
                target = _follow(m.args[clause].this, path)
                repl = swap(this=target.this.copy(), expression=target.expression.copy())
                if target is m.args[clause].this:
                    m.args[clause].set("this", repl)
                else:
                    target.replace(repl)
                out.append((f"{clause}:opswap:{i}:{texts[i][:60]}", m))
            if isinstance(atom, exp.In) and len(atom.expressions) > 1:
                for k in range(len(atom.expressions)):
                    m = ast.copy()
                    target = _follow(m.args[clause].this, path)
                    target.set("expressions", [x.copy() for j, x in enumerate(target.expressions) if j != k])
                    out.append((f"{clause}:indrop{k}:{i}:{texts[i][:60]}", m))
            if isinstance(atom, (exp.Like, exp.ILike)) and isinstance(atom.expression, exp.Literal):
                pat = atom.expression.this
                for lab, newpat in (("unanchor_start", "%" + pat), ("unanchor_end", pat + "%")):
                    if newpat.startswith("%%") or newpat.endswith("%%"):
                        continue
                    m = ast.copy()
                    target = _follow(m.args[clause].this, path)
                    target.set("expression", exp.Literal.string(newpat))
                    out.append((f"{clause}:{lab}:{i}:{texts[i][:60]}", m))
            for kind in ("drop", "negate"):
                m = ast.copy()
                target = _follow(m.args[clause].this, path)
                repl = exp.Boolean(this=True) if kind == "drop" else exp.Not(this=exp.Paren(this=target.copy()))
                if target is m.args[clause].this:
                    m.args[clause].set("this", repl)
                else:
                    target.replace(repl)
                out.append((f"{clause}:{kind}:{i}:{texts[i][:60]}", m))
    order = ast.args.get("order")
    if order is not None:
        for i, o in enumerate(order.expressions):
            m = ast.copy()
            mo = m.args["order"].expressions[i]
            mo.set("desc", not bool(o.args.get("desc")))
            out.append((f"order:flip:{i}", m))
    for key in ("limit", "offset"):
        if ast.args.get(key) is not None:
            m = ast.copy()
            m.set(key, None)
            if key == "limit":
                m.set("offset", None)
            out.append((f"{key}:drop", m))
    if ast.args.get("distinct"):
        m = ast.copy()
        m.set("distinct", None)
        out.append(("distinct:drop", m))
    return out


def _path_to(root, target):
    """Child-index path from root to target (identity)."""
    if root is target:
        return []
    for key in ("this", "expression"):
        child = root.args.get(key)
        if isinstance(child, exp.Expression):
            p = _path_to(child, target)
            if p is not None:
                return [key] + p
    return None


def _follow(root, path):
    for key in path:
        root = root.args[key]
    return root


def count_matching_sql(ast):
    """SELECT COUNT(*) over FROM/JOIN/WHERE of the query (are any rows selected at all?)."""
    m = ast.copy()
    for key in ("group", "having", "order", "limit", "offset", "distinct"):
        m.set(key, None)
    m.set("expressions", [exp.Count(this=exp.Star())])
    return to_sqlite(m)


def sort_key_sql(ast):
    """The query with its ORDER BY expressions appended to the select list as __k{i}.

    Returns None when that could change the result (DISTINCT with a sort key outside the
    select list) or the query has no ORDER BY.
    """
    order = ast.args.get("order")
    if order is None:
        return None
    m = ast.copy()
    sel = [e.this if isinstance(e, exp.Alias) else e for e in m.expressions]
    alias_of = {e.alias: e.this for e in m.expressions if isinstance(e, exp.Alias)}
    keys = []
    for i, o in enumerate(order.expressions):
        e = o.this
        if isinstance(e, exp.Literal) and not e.is_string:
            e = sel[int(e.this) - 1]
        elif isinstance(e, exp.Column) and not e.table and e.name in alias_of:
            e = alias_of[e.name]
        if isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star)):
            return None
        keys.append(exp.alias_(e.copy(), f"__k{i}"))
    if m.args.get("distinct") and not all(any(k.this.sql() == s.sql() for s in sel) for k in keys):
        return None
    m.set("expressions", list(m.expressions) + keys)
    return to_sqlite(m), len(keys)
