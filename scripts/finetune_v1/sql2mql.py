"""Deterministic reference-SQL -> MQL candidate generator (converter `sql2mql-v1`).

Input: a SELECT inside the supported subset of sqlsem.analyze (single table or INNER/LEFT
equi-joins; WHERE/HAVING with comparisons, BETWEEN, IN, LIKE/ILIKE, IS [NOT] NULL, AND/OR/
NOT; GROUP BY with COUNT/SUM/AVG/MIN/MAX/COUNT(DISTINCT); ORDER BY; LIMIT/OFFSET;
DISTINCT; MySQL clock arithmetic and YEAR/MONTH/DAY/DATE/LOWER/UPPER/LENGTH).
Output: the benchmark's model-output contract, {"collection": ..., "pipeline": [...]}, plain
JSON (no shell helpers), read-only stages only.

The generator is a CANDIDATE source, not an oracle: every candidate is executed on MongoDB
and accepted only if its result equals the SQLite result of the reference SQL on every
fixture (verify.py). Semantics it implements deliberately:
  * SQL three-valued logic in WHERE: `<>`, NOT IN, NOT LIKE and negated comparisons never
    match NULL; `$expr` comparisons guard their column operands with `$ne: null`.
  * Inner-join keys that are NULL never match; LEFT joins keep unmatched rows.
  * LIKE is case-insensitive (`$options: "i"`), `%` -> `.*`, `_` -> `.`, anchored.
  * Clock functions use `$$NOW` (UTC); dates are compared as 'YYYY-MM-DD[ HH:MM:SS]' strings.
"""
import re

from sqlglot import exp

import nk_common as C
import sqlsem as S

CONVERTER_VERSION = "sql2mql-v1"


class Unsupported(Exception):
    pass


DT_FMT = {"date": "%Y-%m-%d", "datetime": "%Y-%m-%d %H:%M:%S"}
MQL_UNIT = {"second": "second", "minute": "minute", "hour": "hour", "day": "day",
            "week": "week", "month": "month", "year": "year"}


class Ctx:
    def __init__(self, a, schema):
        self.a = a
        self.schema = schema
        self.joined = {}  # table -> lookup "as" name

    def path(self, col):
        t = self.a.aliases[col.table] if col.table else self.a.base
        if t == self.a.base:
            return col.name, t
        return f"{self.joined[t]}.{col.name}", t

    def ctype(self, col):
        p, t = self.path(col)
        return self.schema[t][col.name]


def _lit(node):
    k, v = S._lit_value(node)
    if k == "bool":
        return 1 if v else 0
    if isinstance(v, str) and v.startswith("$"):
        return {"$literal": v}
    return v


def like_regex(pattern):
    out = []
    for ch in pattern:
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
    rx = "".join(out)
    # trivial simplifications that keep the anchored meaning
    if rx.startswith(".*"):
        rx = rx[2:]
    else:
        rx = "^" + rx
    if rx.endswith(".*") and not rx.endswith("\\.*"):
        rx = rx[:-2]
    else:
        rx = rx + "$"
    return rx


# ---------------------------------------------------------------- aggregation expressions
def clock_expr(node, ctx):
    """Clock/date-arithmetic SQL node -> (agg expression producing a string, kind)."""
    tf = S.time_fn(node)
    if tf == "now":
        return ("$$NOW", "datetime"), True
    if tf == "today":
        return ({"$dateTrunc": {"date": "$$NOW", "unit": "day"}}, "date"), True
    sh = S._date_shift(node)
    if sh is not None:
        base, amount, unit = sh
        (d, kind), is_clock = clock_expr(base, ctx)
        if not is_clock:
            raise Unsupported("date arithmetic on a column")
        if kind == "date" and unit in ("hour", "minute", "second"):
            kind = "datetime"
        op = "$dateSubtract" if amount < 0 else "$dateAdd"
        return ({op: {"startDate": d, "unit": MQL_UNIT[unit], "amount": abs(amount)}}, kind), True
    if isinstance(node, exp.TsOrDsToDate) or (isinstance(node, exp.Date) and not node.expressions):
        inner = node.this
        (d, kind), is_clock = clock_expr(inner, ctx)
        if is_clock:
            return (d, "date"), True
    return (None, None), False


def as_string(d, kind):
    return {"$dateToString": {"format": DT_FMT[kind], "date": d}}


def nullsafe(col_expr, expr):
    return {"$cond": [{"$eq": [col_expr, None]}, None, expr]}


def value_expr(node, ctx, agg_map=None):
    """SQL value expression -> MQL aggregation expression."""
    if isinstance(node, exp.Paren):
        return value_expr(node.this, ctx, agg_map)
    if agg_map is not None and node.sql() in agg_map:
        return "$" + agg_map[node.sql()]
    if isinstance(node, exp.Column):
        if agg_map is not None and not node.table and node.name in agg_map:
            return "$" + agg_map[node.name]
        return "$" + ctx.path(node)[0]
    if S.is_literal(node):
        return _lit(node)
    if isinstance(node, exp.Null):
        return None
    (d, kind), is_clock = clock_expr(node, ctx)
    if is_clock:
        return as_string(d, kind)
    if isinstance(node, S.DATE_PART):
        inner = S.unwrap_date(node.this)
        if isinstance(inner, exp.Column):
            c = value_expr(inner, ctx)
            pos = {"Year": (0, 4), "Month": (5, 2), "Day": (8, 2)}[type(node).__name__]
            return nullsafe(c, {"$toInt": {"$substrCP": [c, pos[0], pos[1]]}})
        (d, kind), is_clock = clock_expr(inner, ctx)
        if is_clock:
            return {f"${type(node).__name__.lower()}": d} if type(node).__name__ != "Day" else {"$dayOfMonth": d}
        raise Unsupported("date part of expression")
    if isinstance(node, exp.TsOrDsToDate) or (isinstance(node, exp.Date) and not node.expressions):
        c = value_expr(node.this, ctx)
        return nullsafe(c, {"$substrCP": [c, 0, 10]})
    if isinstance(node, exp.Lower):
        c = value_expr(node.this, ctx)
        return nullsafe(c, {"$toLower": c})
    if isinstance(node, exp.Upper):
        c = value_expr(node.this, ctx)
        return nullsafe(c, {"$toUpper": c})
    if isinstance(node, exp.Length):
        c = value_expr(node.this, ctx)
        return nullsafe(c, {"$strLenCP": c})
    for cls, op in ((exp.Add, "$add"), (exp.Sub, "$subtract"), (exp.Mul, "$multiply"),
                    (exp.Div, "$divide")):
        if isinstance(node, cls):
            return {op: [value_expr(node.this, ctx, agg_map), value_expr(node.expression, ctx, agg_map)]}
    raise Unsupported(f"value expression {type(node).__name__}")


def _is_value_col(node):
    return isinstance(node, exp.Column) or isinstance(node, (S.DATE_PART, exp.TsOrDsToDate, exp.Date,
                                                            exp.Lower, exp.Upper, exp.Length))


# ---------------------------------------------------------------- predicates
CMP_OP = {exp.EQ: "$eq", exp.NEQ: "$ne", exp.GT: "$gt", exp.GTE: "$gte", exp.LT: "$lt",
          exp.LTE: "$lte"}
FLIP = {"$gt": "$lt", "$gte": "$lte", "$lt": "$gt", "$lte": "$gte", "$eq": "$eq", "$ne": "$ne"}
NEGATE = {"$gt": "$lte", "$gte": "$lt", "$lt": "$gte", "$lte": "$gt", "$eq": "$ne", "$ne": "$eq"}


def predicate(node, ctx, negate=False, agg_map=None):
    """SQL boolean -> MQL $match document (query language; $expr where needed)."""
    if isinstance(node, exp.Paren):
        return predicate(node.this, ctx, negate, agg_map)
    if isinstance(node, exp.Not):
        return predicate(node.this, ctx, not negate, agg_map)
    if isinstance(node, (exp.And, exp.Or)):
        is_and = isinstance(node, exp.And) != negate
        parts = []
        for side in (node.this, node.expression):
            sub = predicate(side, ctx, negate, agg_map)
            key = "$and" if is_and else "$or"
            if list(sub) == [key]:
                parts.extend(sub[key])
            else:
                parts.append(sub)
        return {"$and" if is_and else "$or": parts}
    if isinstance(node, exp.Boolean):
        return {} if node.this != negate else {"$expr": False}
    if isinstance(node, tuple(CMP_OP)):
        op = CMP_OP[type(node)]
        left, right = node.this, node.expression
        if S.is_literal(left) and not S.is_literal(right):
            left, right, op = right, left, FLIP[op]
        if negate:
            op = NEGATE[op]
        if isinstance(left, exp.Column) and S.is_literal(right) and agg_map is None:
            f, v = ctx.path(left)[0], _lit(right)
            if op == "$eq":
                return {f: v}
            if op == "$ne":
                return {f: {"$nin": [v, None]}}
            return {f: {op: v}}
        return {"$expr": expr_compare(op, left, right, ctx, agg_map)}
    if isinstance(node, exp.Between):
        lo, hi = node.args["low"], node.args["high"]
        if negate:
            return {"$or": [predicate(exp.LT(this=node.this.copy(), expression=lo.copy()), ctx, False, agg_map),
                            predicate(exp.GT(this=node.this.copy(), expression=hi.copy()), ctx, False, agg_map)]}
        if isinstance(node.this, exp.Column) and S.is_literal(lo) and S.is_literal(hi) and agg_map is None:
            return {ctx.path(node.this)[0]: {"$gte": _lit(lo), "$lte": _lit(hi)}}
        return {"$and": [predicate(exp.GTE(this=node.this.copy(), expression=lo.copy()), ctx, False, agg_map),
                         predicate(exp.LTE(this=node.this.copy(), expression=hi.copy()), ctx, False, agg_map)]}
    if isinstance(node, exp.In):
        vals = [_lit(x) for x in node.expressions]
        if isinstance(node.this, exp.Column) and agg_map is None:
            f = ctx.path(node.this)[0]
            return {f: {"$nin": vals + [None]}} if negate else {f: {"$in": vals}}
        v = value_expr(node.this, ctx, agg_map)
        e = {"$in": [v, vals]}
        return {"$expr": {"$and": [{"$ne": [v, None]}, {"$not": [e]}]} if negate else e}
    if isinstance(node, (exp.Like, exp.ILike)):
        rx = like_regex(node.expression.this)
        if isinstance(node.this, exp.Column):
            f = ctx.path(node.this)[0]
            if negate:
                return {f: {"$not": {"$regex": rx, "$options": "i"}, "$ne": None}}
            return {f: {"$regex": rx, "$options": "i"}}
        v = value_expr(node.this, ctx, agg_map)
        m = {"$regexMatch": {"input": v, "regex": rx, "options": "i"}}
        return {"$expr": {"$and": [{"$ne": [v, None]}, {"$not": [m]} if negate else m]}}
    if isinstance(node, exp.Is):
        f = ctx.path(node.this)[0]
        return {f: {"$ne": None}} if negate else {f: None}
    if isinstance(node, exp.Column):
        f = ctx.path(node)[0]
        return {f: {"$in": [0, None]}} if negate else {f: {"$nin": [0, None]}}
    raise Unsupported(f"predicate {type(node).__name__}")


def expr_compare(op, left, right, ctx, agg_map):
    lv, rv = value_expr(left, ctx, agg_map), value_expr(right, ctx, agg_map)
    guards = [{"$ne": [x, None]} for x, n in ((lv, left), (rv, right)) if not S.is_literal(n)
              and not S.time_fn(n) and S._date_shift(n) is None]
    cmp_ = {op: [lv, rv]}
    return {"$and": guards + [cmp_]} if guards else cmp_


# ---------------------------------------------------------------- query
def out_name(e, i, used):
    if isinstance(e, exp.Alias):
        name = e.alias
    elif isinstance(e, exp.Column):
        name = e.name
    elif isinstance(e, exp.Count):
        name = "count"
    elif isinstance(e, S.AGGS):
        arg = e.this
        inner = arg.name if isinstance(arg, exp.Column) else "value"
        name = f"{type(e).__name__.lower()}_{inner}"
    else:
        name = f"col{i}"
    name = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or f"col{i}"
    if name[0].isdigit():
        name = "c_" + name
    base, k = name, 2
    while name in used or name == "_id":
        name = f"{base}_{k}"
        k += 1
    used.add(name)
    return name


def accumulator(agg, ctx):
    arg = agg.this
    if isinstance(agg, exp.Count):
        if isinstance(arg, exp.Star):
            return {"$sum": 1}, None
        if isinstance(arg, exp.Distinct):
            v = value_expr(arg.expressions[0], ctx)
            return {"$addToSet": v}, "distinct_count"
        v = value_expr(arg, ctx)
        return {"$sum": {"$cond": [{"$ne": [v, None]}, 1, 0]}}, None
    v = value_expr(arg, ctx)
    op = {"Sum": "$sum", "Avg": "$avg", "Min": "$min", "Max": "$max"}[type(agg).__name__]
    return {op: v}, None


def convert(ast, a, schema):
    """-> MQL dict, or raise Unsupported."""
    if a.unsupported:
        raise Unsupported(";".join(a.unsupported))
    ctx = Ctx(a, schema)
    pipe = []
    for j in a.joins:
        ctx.joined[j["table"]] = j["table"]
    for jn, j in zip(ast.args.get("joins") or [], a.joins):
        on = jn.args["on"]
        l, r = on.this, on.expression
        lt = a.aliases[l.table] if l.table else a.base
        if lt == j["table"]:
            l, r = r, l
        rt = a.aliases[r.table] if r.table else a.base
        if rt != j["table"]:
            raise Unsupported("join condition does not reference the joined table")
        lf, _ = ctx.path(l)
        as_ = ctx.joined[j["table"]]
        coll = C.coll(j["table"])
        if j["kind"] == "left":
            pipe.append({"$lookup": {"from": coll, "let": {"k": "$" + lf}, "pipeline": [
                {"$match": {"$expr": {"$and": [{"$ne": ["$$k", None]}, {"$eq": ["$" + r.name, "$$k"]}]}}}],
                "as": as_}})
            pipe.append({"$unwind": {"path": "$" + as_, "preserveNullAndEmptyArrays": True}})
        else:
            pipe.append({"$match": {lf: {"$ne": None}}})
            pipe.append({"$lookup": {"from": coll, "localField": lf, "foreignField": r.name, "as": as_}})
            pipe.append({"$unwind": "$" + as_})
    where = ast.args.get("where")
    if where is not None:
        m = predicate(where.this, ctx)
        if m:
            pipe.append({"$match": m})

    sel = list(ast.expressions)
    group = ast.args.get("group")
    has_agg = any(isinstance(e.this if isinstance(e, exp.Alias) else e, S.AGGS) for e in sel)
    having = ast.args.get("having")
    order = ast.args.get("order")
    used = set()
    names = [out_name(e, i, used) for i, e in enumerate(sel)]
    alias_expr = {e.alias: e.this for e in sel if isinstance(e, exp.Alias)}

    def sort_target(o, field_of):
        e = o.this
        if isinstance(e, exp.Literal) and not e.is_string:
            return names[int(e.this) - 1]
        if isinstance(e, exp.Column) and not e.table and e.name in alias_expr:
            e = alias_expr[e.name]
        return field_of(e)

    if group is not None or has_agg:
        gkeys = {}
        for i, g in enumerate(group.expressions if group is not None else []):
            gkeys[g.sql()] = (f"g{i}", value_expr(g, ctx))
        accs, post = {}, {}
        agg_map = {}

        def need_agg(node):
            k = node.sql()
            if k in agg_map:
                return agg_map[k]
            name = f"a{len(accs)}"
            acc, kind = accumulator(node, ctx)
            accs[name] = acc
            if kind == "distinct_count":
                post[name] = {"$size": {"$filter": {"input": "$" + name, "cond": {"$ne": ["$$this", None]}}}}
            agg_map[k] = name
            return name

        for n in [x for x in ast.find_all(*S.AGGS)]:
            need_agg(n)
        gid = {k: v for k, v in gkeys.values()} if gkeys else None
        stage = {"_id": gid}
        stage.update(accs)
        pipe.append({"$group": stage})
        if post:
            pipe.append({"$set": post})
        field = {k: f"_id.{v[0]}" for k, v in gkeys.items()}
        field.update(agg_map)

        def field_of(e):
            k = e.sql()
            if k in field:
                return field[k]
            if isinstance(e, exp.Column) and not e.table:
                for kk, v in field.items():
                    if kk.split(".")[-1] == e.name:
                        return v
            raise Unsupported(f"grouped expression not available: {k}")

        if having is not None:
            hmap = dict(field)
            for al, ex in alias_expr.items():
                if ex.sql() in field:
                    hmap[al] = field[ex.sql()]
            pipe.append({"$match": predicate(having.this, ctx, agg_map=hmap)})
        if order is not None:
            pipe.append({"$sort": {sort_target(o, field_of): -1 if o.args.get("desc") else 1
                                   for o in order.expressions}})
        _limits(pipe, a)
        proj = {"_id": 0}
        for name, e in zip(names, sel):
            inner = e.this if isinstance(e, exp.Alias) else e
            proj[name] = "$" + field_of(inner)
        pipe.append({"$project": proj})
    elif ast.args.get("distinct"):
        if any(isinstance(e, exp.Star) for e in sel):
            raise Unsupported("SELECT DISTINCT *")
        gid = {name: value_expr(e.this if isinstance(e, exp.Alias) else e, ctx)
               for name, e in zip(names, sel)}
        pipe.append({"$group": {"_id": gid}})
        by_sql = {(e.this if isinstance(e, exp.Alias) else e).sql(): f"_id.{n}" for n, e in zip(names, sel)}

        def field_of(e):
            if e.sql() in by_sql:
                return by_sql[e.sql()]
            if isinstance(e, exp.Column) and not e.table:
                for k, v in by_sql.items():
                    if k.split(".")[-1] == e.name:
                        return v
            raise Unsupported("DISTINCT with ORDER BY outside the select list")
        if order is not None:
            pipe.append({"$sort": {sort_target(o, field_of): -1 if o.args.get("desc") else 1
                                   for o in order.expressions}})
        _limits(pipe, a)
        pipe.append({"$project": {"_id": 0, **{n: f"$_id.{n}" for n in names}}})
    else:
        if order is not None:
            computed, spec = {}, {}
            for i, o in enumerate(order.expressions):
                def field_of(e, i=i):
                    if isinstance(e, exp.Column):
                        return ctx.path(e)[0]
                    computed[f"__s{i}"] = value_expr(e, ctx)
                    return f"__s{i}"
                spec[sort_target(o, field_of)] = -1 if o.args.get("desc") else 1
            if computed:
                pipe.append({"$set": computed})
            pipe.append({"$sort": spec})
        _limits(pipe, a)
        if len(sel) == 1 and isinstance(sel[0], exp.Star):
            pipe.append({"$project": {"_id": 0, **({k: 0 for k in computed} if order is not None and computed else {})}})
        else:
            proj = {"_id": 0}
            for name, e in zip(names, sel):
                inner = e.this if isinstance(e, exp.Alias) else e
                if isinstance(inner, exp.Star):
                    raise Unsupported("star mixed with columns")
                v = value_expr(inner, ctx)
                proj[name] = {"$literal": v} if not isinstance(v, (str, dict)) else v
            pipe.append({"$project": proj})
    return {"collection": C.coll(a.base), "pipeline": pipe}


def _limits(pipe, a):
    if a.offset:
        pipe.append({"$skip": a.offset})
    if a.limit is not None:
        pipe.append({"$limit": a.limit})
