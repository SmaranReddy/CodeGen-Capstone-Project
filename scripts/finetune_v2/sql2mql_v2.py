"""CONVERTER SIDE. Conservative SQLite-semantics SQL -> MQL translator (`sql2mql-v2`).

Input: one Spider (SQLite) SELECT plus the schema profile of its database. Output: the
benchmark's MQL contract {"collection": "<db>__<table>", "pipeline": [...]} (plain JSON,
read-only stages), or Unsupported(code). A translation is only a CANDIDATE: verifier.py decides.

Semantics contract (docs/sql2mql_v2_quality_assessment.md §2):
  * Identifiers are resolved case-insensitively to the stored spelling (F5). A double-quoted
    name that is not a column of any FROM table is a string literal (SQLite's fallback).
  * NULL: data may store SQL NULL as explicit null OR as an absent field (F15); every
    expression is safe under both ($ifNull before null tests, query-language null matching).
    Three-valued logic: a predicate that is NULL is false; NOT never turns NULL into true.
  * Types: a column's OBSERVED storage classes must be homogeneous (num or text) wherever its
    type matters (comparison, aggregate, arithmetic, join key, LIKE). Literals are converted
    with SQLite's column-affinity rules before comparison (F6); otherwise rejected. Join keys
    must have the same class (F7).
  * Aggregates: an ungrouped aggregate always yields one row ($facet + default row: COUNT -> 0,
    others -> NULL) (F1); SUM of no non-NULL input is NULL (F2); COUNT(col) counts non-NULL
    values; COUNT(DISTINCT col) ignores NULL.
  * DISTINCT with GROUP BY de-duplicates the grouped rows (F3; rejected with ORDER BY).
  * Division: NULL for a zero divisor; integer division (truncation) when both operands are
    integers (F8). Arithmetic on text is rejected (F4). ROUND and other functions are rejected (F14).
  * ORDER BY ordinals/aliases resolve to their select expressions (F9); bare-column predicates
    keep only non-zero / zero NON-NULL values (F10); LEFT JOIN guards are missing-aware (F11);
    joined documents live under reserved names `__j<k>` (F12); LIKE is ASCII-case-insensitive,
    `%`/`_` match newlines (`s` option), anchors are absolute (F13).
  * Bare (ungrouped) columns are allowed only when functionally determined by a grouped,
    non-NULL single-column primary key; LIMIT/OFFSET require ORDER BY.
Rejected by design: subqueries, set operations, CTEs, CASE/CAST/COALESCE and every other
function, non-LEFT outer joins, self-joins, joins without a single ON equality, NULL literals.

Must not import the verifier (verifier.py, comparator.py, fixtures.py, mongo_import.py).
"""
import re

import sqlglot
from sqlglot import exp

CONVERTER_VERSION = "sql2mql-v2"


class Unsupported(Exception):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


def U(code, detail=""):
    raise Unsupported(code, detail)


ALLOWED_NODES = (
    exp.Select, exp.From, exp.Join, exp.Table, exp.TableAlias, exp.Alias, exp.Column,
    exp.Identifier, exp.Star, exp.Literal, exp.Neg, exp.Paren, exp.And, exp.Or, exp.Not,
    exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between, exp.In, exp.Like, exp.Is,
    exp.Null, exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max, exp.Distinct, exp.Add, exp.Sub,
    exp.Mul, exp.Div, exp.Where, exp.Group, exp.Having, exp.Order, exp.Ordered, exp.Limit,
    exp.Offset)
AGGS = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
CMP = {exp.EQ: "$eq", exp.NEQ: "$ne", exp.GT: "$gt", exp.GTE: "$gte", exp.LT: "$lt", exp.LTE: "$lte"}
FLIP = {"$gt": "$lt", "$gte": "$lte", "$lt": "$gt", "$lte": "$gte", "$eq": "$eq", "$ne": "$ne"}
NEGATE = {"$gt": "$lte", "$gte": "$lt", "$lt": "$gte", "$lte": "$gt", "$eq": "$ne", "$ne": "$eq"}
NUMERIC_TEXT = re.compile(r"^\s*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\s*$")
REGEX_META = set("\\.^$|?*+()[]{}/")


# ---------------------------------------------------------------- typed values
class TV:
    """A translated value: aggregation expression, kind ('num'|'text'|'null'|'mixed'),
    whether every value is an integer, and the source column (plain column refs only)."""
    __slots__ = ("expr", "kind", "is_int", "col", "path")

    def __init__(self, expr, kind, is_int=False, col=None, path=None):
        self.expr, self.kind, self.is_int, self.col, self.path = expr, kind, is_int, col, path


def lit_expr(v):
    """A literal inside an aggregation expression (strings starting with '$' are paths)."""
    if isinstance(v, str) and v.startswith("$"):
        return {"$literal": v}
    return v


def not_null(e):
    return {"$ne": [{"$ifNull": [e, None]}, None]}


def compatible(a, b):
    return a == "null" or b == "null" or (a == b and a in ("num", "text"))


class Src:
    def __init__(self, table, alias, prefix, kind):
        self.table, self.alias, self.prefix, self.kind = table, alias, prefix, kind


# ---------------------------------------------------------------- converter
class Converter:
    def __init__(self, prof, sql):
        self.prof, self.sql = prof, sql.strip().rstrip(";")
        self.features = set()

    # ---- parsing and scope
    def parse(self):
        try:
            trees = sqlglot.parse(self.sql, read="sqlite")
        except sqlglot.errors.ParseError as e:
            U("parse", str(e)[:120])
        if len(trees) != 1 or not isinstance(trees[0], exp.Select):
            U("not_a_single_select")
        tree = trees[0]
        for node in tree.walk():
            if not isinstance(node, ALLOWED_NODES):
                U("construct", type(node).__name__)
            if isinstance(node, exp.Select) and node is not tree:
                U("construct", "subquery")
        for key in ("with", "qualify", "windows", "into", "laterals", "pivots", "sample"):
            if tree.args.get(key):
                U("construct", key)
        return tree

    def scope(self, tree):
        frm = tree.args.get("from")
        if frm is None or not isinstance(frm.this, exp.Table):
            U("construct", "FROM")
        self.srcs, self.by_alias = [], {}
        self.joins = []

        def add(tnode, prefix, kind):
            if tnode.args.get("db") or tnode.args.get("catalog"):
                U("construct", "qualified table")
            real = self.prof.table(tnode.name)
            if real is None:
                U("unknown_table", tnode.name)
            if any(s.table == real for s in self.srcs):
                U("self_join", real)
            alias = (tnode.alias or tnode.name).lower()
            if alias in self.by_alias:
                U("duplicate_alias", alias)
            for c in self.prof.tables[real]:
                if "." in c or c.startswith("$") or c.startswith("__"):
                    U("field_name", f"{real}.{c}")
            s = Src(real, alias, prefix, kind)
            self.srcs.append(s)
            self.by_alias[alias] = s
            return s

        self.base = add(frm.this, "", "base")
        for k, j in enumerate(tree.args.get("joins") or [], 1):
            side, kind = (j.args.get("side") or "").upper(), (j.args.get("kind") or "").upper()
            if j.args.get("using") or j.args.get("method"):
                U("join_kind", "USING/NATURAL")
            if side == "LEFT" and kind in ("", "OUTER"):
                jk = "left"
            elif not side and kind in ("", "INNER"):
                jk = "inner"
            else:
                U("join_kind", f"{side} {kind}".strip())
            if not isinstance(j.this, exp.Table):
                U("construct", "join source")
            s = add(j.this, f"__j{k}.", jk)
            on = j.args.get("on")
            if on is None:
                U("join_without_on")
            self.joins.append((s, on))
        self.features.add("join" if self.joins else "single_table")

    def resolve(self, col):
        """exp.Column -> ('col', Src, Column) | ('str', text)."""
        if isinstance(col.this, exp.Star):
            U("construct", "qualified star")
        name = col.name
        if col.table:
            s = self.by_alias.get(col.table.lower())
            if s is None:
                U("unknown_alias", col.table)
            c = self.prof.column(s.table, name)
            if c is None:
                U("unknown_column", f"{col.table}.{name}")
            return "col", s, c
        hits = [(s, self.prof.column(s.table, name)) for s in self.srcs]
        hits = [(s, c) for s, c in hits if c is not None]
        if len(hits) == 1:
            return "col", hits[0][0], hits[0][1]
        if len(hits) > 1:
            U("ambiguous_column", name)
        ident = col.this
        if isinstance(ident, exp.Identifier) and ident.quoted:
            self.features.add("double_quoted_string")
            return "str", name
        U("unknown_column", name)

    # ---- literals
    @staticmethod
    def literal(node):
        """-> (kind, value) for a literal node, else None."""
        neg = False
        if isinstance(node, exp.Neg):
            neg, node = True, node.this
        if isinstance(node, exp.Paren):
            return Converter.literal(node.this) if not neg else None
        if isinstance(node, exp.Literal):
            if node.is_string:
                return None if neg else ("text", node.this)
            t = node.this
            try:
                v = int(t)
            except ValueError:
                v = float(t)
            return "num", -v if neg else v
        return None

    def lit_of(self, node):
        lit = self.literal(node)
        if lit is not None:
            return lit
        if isinstance(node, exp.Column):
            r = self.resolve(node)
            if r[0] == "str":
                return "text", r[1]
        return None

    def convert_literal(self, col, lit):
        """SQLite comparison affinity: column-vs-literal (https://sqlite.org/datatype3.html §4.2)."""
        kind, v = lit
        aff = col.affinity
        if aff in ("INTEGER", "REAL", "NUMERIC") and kind == "text" and NUMERIC_TEXT.match(v):
            s = v.strip()
            try:
                v = int(s) if aff != "REAL" and re.fullmatch(r"[+-]?\d+", s) else float(s)
            except ValueError:
                U("literal_conversion", v)
            if aff == "NUMERIC" and isinstance(v, float) and v.is_integer() and abs(v) < 2 ** 53:
                v = int(v)
            kind = "num"
            self.features.add("implicit_cast")
        elif aff == "TEXT" and kind == "num":
            if not isinstance(v, int):
                U("literal_type_mismatch", "real literal vs TEXT column")
            v, kind = str(v), "text"
            self.features.add("implicit_cast")
        if not compatible(col.kind, kind):
            U("literal_type_mismatch", f"{col.table}.{col.name} ({col.kind}) vs {kind} literal")
        return kind, v

    # ---- row-level values (before any $group)
    def row_value(self, node):
        if isinstance(node, exp.Paren):
            return self.row_value(node.this)
        lit = self.literal(node)
        if lit is not None:
            return TV(lit_expr(lit[1]), lit[0], isinstance(lit[1], int))
        if isinstance(node, exp.Column):
            r = self.resolve(node)
            if r[0] == "str":
                return TV(lit_expr(r[1]), "text")
            _, s, c = r
            path = s.prefix + c.name
            return TV("$" + path, c.kind, c.all_int, col=c, path=path)
        if isinstance(node, exp.Neg):
            v = self.row_value(node.this)
            self.need_num(v, "negation")
            return TV({"$multiply": [-1, v.expr]}, "num", v.is_int)
        if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
            return self.arith(node, self.row_value)
        if isinstance(node, exp.Null):
            U("null_literal")
        if isinstance(node, AGGS):
            U("aggregate_in_row_context")
        U("value", type(node).__name__)

    def need_num(self, v, what):
        if v.kind not in ("num", "null"):
            U("arithmetic_type", f"{what} on {v.kind}")

    def arith(self, node, value):
        a, b = value(node.this), value(node.expression)
        self.need_num(a, type(node).__name__)
        self.need_num(b, type(node).__name__)
        self.features.add("arithmetic")
        if isinstance(node, exp.Div):
            self.features.add("division")
            is_int = a.is_int and b.is_int
            q = {"$divide": [a.expr, b.expr]}
            if is_int:
                q = {"$trunc": [q]}
            return TV({"$cond": [{"$eq": [b.expr, 0]}, None, q]}, "num", is_int)
        op = {exp.Add: "$add", exp.Sub: "$subtract", exp.Mul: "$multiply"}[type(node)]
        return TV({op: [a.expr, b.expr]}, "num", a.is_int and b.is_int)

    # ---- predicates
    def predicate(self, node, value, negate=False, query=True):
        """SQL boolean -> $match document. `value` translates operands; `query` allows
        query-language forms (row context only)."""
        if isinstance(node, exp.Paren):
            return self.predicate(node.this, value, negate, query)
        if isinstance(node, exp.Not):
            return self.predicate(node.this, value, not negate, query)
        if isinstance(node, (exp.And, exp.Or)):
            is_and = isinstance(node, exp.And) != negate
            key = "$and" if is_and else "$or"
            parts = []
            for side in (node.this, node.expression):
                sub = self.predicate(side, value, negate, query)
                parts.extend(sub[key] if list(sub) == [key] else [sub])
            return {key: parts}
        if isinstance(node, tuple(CMP)):
            return self.compare(CMP[type(node)], node.this, node.expression, value, negate, query)
        if isinstance(node, exp.Between):
            lo, hi = node.args["low"], node.args["high"]
            if negate:
                return {"$or": [self.compare("$lt", node.this, lo, value, False, query),
                                self.compare("$gt", node.this, hi, value, False, query)]}
            a = self.compare("$gte", node.this, lo, value, False, query)
            b = self.compare("$lte", node.this, hi, value, False, query)
            self.features.add("between")
            if query and len(a) == 1 and list(a) == list(b) and not next(iter(a)).startswith("$"):
                f = next(iter(a))
                if isinstance(a[f], dict) and isinstance(b[f], dict):
                    return {f: {**a[f], **b[f]}}
            return {"$and": [a, b]}
        if isinstance(node, exp.In):
            return self.in_list(node, value, negate, query)
        if isinstance(node, exp.Like):
            return self.like(node, value, negate, query)
        if isinstance(node, exp.Is):
            if not isinstance(node.expression, exp.Null):
                U("construct", "IS <non-NULL>")
            v = value(node.this)
            self.features.add("is_null")
            if query and v.path:
                return {v.path: {"$ne": None}} if negate else {v.path: None}
            t = {"$eq": [{"$ifNull": [v.expr, None]}, None]}
            return {"$expr": {"$not": [t]} if negate else t}
        if isinstance(node, exp.Column):
            v = value(node)
            if v.kind not in ("num", "null"):
                U("bare_predicate_type", v.kind)
            self.features.add("bare_boolean")
            if query and v.path:
                return {v.path: 0} if negate else {v.path: {"$nin": [0, None]}}
            cond = {"$eq": [v.expr, 0]} if negate else {"$ne": [v.expr, 0]}
            return {"$expr": {"$and": [not_null(v.expr), cond]}}
        U("predicate", type(node).__name__)

    def compare(self, op, left, right, value, negate, query):
        llit, rlit = self.lit_of(left), self.lit_of(right)
        if llit is not None and rlit is None:
            left, right, llit, rlit, op = right, left, rlit, llit, FLIP[op]
        if llit is not None:
            U("constant_predicate")
        if negate:
            op = NEGATE[op]
        lv = value(left)
        if rlit is not None:
            if lv.col is not None:
                kind, v = self.convert_literal(lv.col, rlit)
            else:
                kind, v = rlit
                if lv.kind == "mixed" or not compatible(lv.kind, kind):
                    U("literal_type_mismatch", f"expression ({lv.kind}) vs {kind} literal")
            if query and lv.path:
                if op == "$eq":
                    return {lv.path: v}
                if op == "$ne":
                    return {lv.path: {"$nin": [v, None]}}
                return {lv.path: {op: v}}
            return {"$expr": {"$and": [not_null(lv.expr), {op: [lv.expr, lit_expr(v)]}]}}
        rv = value(right)
        if lv.kind == "mixed" or rv.kind == "mixed" or not compatible(lv.kind, rv.kind):
            U("comparison_type_mismatch", f"{lv.kind} vs {rv.kind}")
        self.features.add("column_comparison")
        return {"$expr": {"$and": [not_null(lv.expr), not_null(rv.expr), {op: [lv.expr, rv.expr]}]}}

    def in_list(self, node, value, negate, query):
        if node.args.get("query") or node.args.get("unnest") or node.args.get("field"):
            U("construct", "IN subquery")
        lits = [self.lit_of(x) for x in node.expressions]
        if not lits or any(x is None for x in lits):
            U("in_list", "non-literal element")
        v = value(node.this)
        if v.col is not None:
            vals = [self.convert_literal(v.col, x)[1] for x in lits]
        else:
            if any(not compatible(v.kind, k) for k, _ in lits) or v.kind == "mixed":
                U("literal_type_mismatch", "IN list")
            vals = [x for _, x in lits]
        self.features.add("in_list")
        if query and v.path:
            return {v.path: {"$nin": vals + [None]}} if negate else {v.path: {"$in": vals}}
        t = {"$in": [v.expr, [lit_expr(x) for x in vals]]}
        return {"$expr": {"$and": [not_null(v.expr), {"$not": [t]} if negate else t]}}

    def like(self, node, value, negate, query):
        pat = self.lit_of(node.expression)
        if pat is None or pat[0] != "text":
            U("like", "pattern is not a string literal")
        pattern = pat[1]
        if any(ord(ch) > 127 and ch.isalpha() for ch in pattern):
            U("like", "non-ASCII letter in pattern (SQLite folds ASCII only)")
        v = value(node.this)
        if v.kind not in ("text", "null"):
            U("like", f"LIKE on {v.kind}")
        if v.col is not None and v.col.fold_trap:
            U("like", "column holds characters PCRE case-folds to ASCII")
        rx = like_regex(pattern)
        self.features.add("like")
        if query and v.path:
            if negate:
                return {v.path: {"$not": {"$regex": rx, "$options": "is"}, "$ne": None}}
            return {v.path: {"$regex": rx, "$options": "is"}}
        m = {"$regexMatch": {"input": v.expr, "regex": rx, "options": "is"}}
        return {"$expr": {"$and": [{"$eq": [{"$type": v.expr}, "string"]}, {"$not": [m]} if negate else m]}}

    # ---- main
    def convert(self):
        tree = self.parse()
        self.scope(tree)
        pipe = []
        for s, on in self.joins:
            pipe += self.join_stages(s, on)
        where = tree.args.get("where")
        if where is not None:
            self.features.add("where")
            pipe.append({"$match": self.predicate(where.this, self.row_value)})
        sel = list(tree.expressions)
        group = tree.args.get("group")
        having = tree.args.get("having")
        has_agg = any(n for e in sel for n in e.find_all(*AGGS)) or bool(having)
        order = tree.args.get("order")
        limit, offset = self.int_arg(tree.args.get("limit")), self.int_arg(tree.args.get("offset"))
        if (limit is not None or offset is not None) and order is None and not (has_agg and group is None):
            U("limit_without_order")
        if limit == 0:
            U("limit_zero")
        names = out_names(sel)
        if group is not None or has_agg:
            pipe += self.grouped(tree, sel, names, group, having, order, limit, offset)
        else:
            pipe += self.plain(tree, sel, names, order, limit, offset)
        return {"collection": f"{self.prof.db_id}__{self.base.table}", "pipeline": pipe}

    @staticmethod
    def int_arg(node):
        if node is None:
            return None
        e = node.expression if isinstance(node, (exp.Limit, exp.Offset)) else node
        if isinstance(e, exp.Literal) and not e.is_string and e.this.isdigit():
            return int(e.this)
        U("limit", "non-literal LIMIT/OFFSET")

    def join_stages(self, s, on):
        if not isinstance(on, exp.EQ) or not all(isinstance(x, exp.Column) for x in (on.this, on.expression)):
            U("join_condition", "not a single column equality")
        a, b = self.resolve(on.this), self.resolve(on.expression)
        if a[0] != "col" or b[0] != "col":
            U("join_condition", "string literal")
        if a[1] is s and b[1] is not s:
            a, b = b, a
        if b[1] is not s or a[1] is s or self.srcs.index(a[1]) > self.srcs.index(s):
            U("join_condition", "does not link the joined table to an earlier one")
        la, rb = a[2], b[2]
        if la.kind == "mixed" or rb.kind == "mixed" or not compatible(la.kind, rb.kind):
            U("join_key_types", f"{la.table}.{la.name} ({la.kind}) vs {rb.table}.{rb.name} ({rb.kind})")
        lf, as_ = a[1].prefix + la.name, s.prefix[:-1]
        if s.kind == "inner":
            return [{"$match": {lf: {"$ne": None}}},
                    {"$lookup": {"from": f"{self.prof.db_id}__{s.table}", "localField": lf,
                                 "foreignField": rb.name, "as": as_}},
                    {"$unwind": "$" + as_}]
        self.features.add("left_join")
        return [{"$lookup": {"from": f"{self.prof.db_id}__{s.table}", "let": {"k": "$" + lf},
                             "pipeline": [{"$match": {"$expr": {"$and": [not_null("$$k"),
                                                                         {"$eq": ["$" + rb.name, "$$k"]}]}}}],
                             "as": as_}},
                {"$unwind": {"path": "$" + as_, "preserveNullAndEmptyArrays": True}}]

    # ---- non-aggregated SELECT
    def plain(self, tree, sel, names, order, limit, offset):
        pipe = []
        star = len(sel) == 1 and isinstance(sel[0], exp.Star)
        if star:
            if self.joins:
                U("star_with_join")
            cols = list(self.prof.tables[self.base.table])
            items = [(c, TV("$" + c, None)) for c in cols]
            sel_nodes = [exp.column(c) for c in cols]
        else:
            if any(isinstance(e, exp.Star) for e in sel):
                U("construct", "star mixed with columns")
            sel_nodes = [e.this if isinstance(e, exp.Alias) else e for e in sel]
            items = list(zip(names, [self.row_value(e) for e in sel_nodes]))
        aliases = {e.alias.lower(): e.this for e in sel if isinstance(e, exp.Alias)}
        if tree.args.get("distinct"):
            self.features.add("distinct")
            pipe.append({"$group": {"_id": {n: {"$ifNull": [v.expr, None]} for n, v in items}}})
            if order is not None:
                spec = {}
                for o in order.expressions:
                    i = self.select_index(o.this, sel_nodes, aliases)
                    if i is None:
                        U("order_by", "DISTINCT ordered by a non-selected expression")
                    self.add_sort(spec, "_id." + items[i][0], o)
                pipe.append({"$sort": spec})
            pipe += limit_stages(limit, offset)
            pipe.append({"$project": {"_id": 0, **{n: "$_id." + n for n, _ in items}}})
            return pipe
        if order is not None:
            spec, temp = {}, {}
            for i, o in enumerate(order.expressions):
                node = self.order_target(o.this, sel_nodes, aliases)
                v = self.row_value(node)
                if v.path:
                    self.add_sort(spec, v.path, o)
                else:
                    temp[f"__s{i}"] = v.expr
                    self.add_sort(spec, f"__s{i}", o)
            if temp:
                pipe.append({"$set": temp})
            pipe.append({"$sort": spec})
        pipe += limit_stages(limit, offset)
        pipe.append({"$project": {"_id": 0, **{n: project_value(v.expr) for n, v in items}}})
        return pipe

    def select_index(self, node, sel_nodes, aliases):
        if isinstance(node, exp.Literal) and not node.is_string:
            i = int(node.this) - 1
            if not 0 <= i < len(sel_nodes):
                U("order_by", "ordinal out of range")
            return i
        if isinstance(node, exp.Column) and not node.table and node.name.lower() in aliases:
            node = aliases[node.name.lower()]
        for i, s in enumerate(sel_nodes):
            if s == node or (isinstance(s, exp.Column) and isinstance(node, exp.Column)
                             and self.same_column(s, node)):
                return i
        return None

    def same_column(self, a, b):
        ra, rb = self.resolve(a), self.resolve(b)
        return ra[0] == rb[0] == "col" and ra[1] is rb[1] and ra[2] is rb[2]

    def order_target(self, node, sel_nodes, aliases):
        if isinstance(node, exp.Literal) and not node.is_string:
            self.features.add("order_by_ordinal")
            return sel_nodes[self.select_index(node, sel_nodes, aliases)]
        if isinstance(node, exp.Column) and not node.table and node.name.lower() in aliases:
            hits = [s for s in self.srcs if self.prof.column(s.table, node.name)]
            if hits:
                U("order_by", "name is both an alias and a column")
            return aliases[node.name.lower()]
        return node

    def add_sort(self, spec, field, o):
        if field in spec:
            U("order_by", "duplicate sort key")
        spec[field] = -1 if o.args.get("desc") else 1
        self.features.add("order_by")

    # ---- aggregated SELECT
    def grouped(self, tree, sel, names, group, having, order, limit, offset):
        if any(isinstance(e, exp.Star) for e in sel):
            U("construct", "star in aggregate query")
        if having is not None and group is None:
            U("having_without_group_by")
        keys = []
        for g in (group.expressions if group is not None else []):
            if not isinstance(g, exp.Column):
                U("group_by", "non-column key")
            r = self.resolve(g)
            if r[0] != "col":
                U("group_by", "string literal key")
            keys.append(r)
        self.features.add("group_by" if keys else "ungrouped_aggregate")
        env = GroupEnv(self, keys)
        sel_nodes = [e.this if isinstance(e, exp.Alias) else e for e in sel]
        outs = [env.value(e) for e in sel_nodes]
        hav = None
        if having is not None:
            self.features.add("having")
            hav = self.predicate(having.this, env.value, query=False)
        aliases = {e.alias.lower(): e.this for e in sel if isinstance(e, exp.Alias)}
        sort_spec, sort_temp = {}, {}
        if order is not None:
            if not keys:
                U("order_by", "ORDER BY on a single aggregate row")
            if tree.args.get("distinct"):
                U("distinct_group_by_order")
            for i, o in enumerate(order.expressions):
                node = self.order_target(o.this, sel_nodes, aliases)
                v = env.value(node)
                if isinstance(v.expr, str) and v.expr.startswith("$") and not v.expr.startswith("$$"):
                    self.add_sort(sort_spec, v.expr[1:], o)
                else:
                    sort_temp[f"__s{i}"] = v.expr
                    self.add_sort(sort_spec, f"__s{i}", o)
        if offset is not None and not keys:
            U("limit", "OFFSET on a single aggregate row")
        pipe = []
        gid = {f"g{i}": {"$ifNull": [v, None]} for i, v in enumerate(env.key_exprs)} if keys else None
        acc = {"_id": gid, **env.accs}
        if keys:
            pipe.append({"$group": acc})
        else:   # F1: exactly one row, even over zero input rows
            pipe.append({"$facet": {"rows": [{"$group": acc}]}})
            pipe.append({"$project": {"row": {"$ifNull": [{"$arrayElemAt": ["$rows", 0]}, env.defaults()]}}})
            pipe.append({"$replaceRoot": {"newRoot": "$row"}})
        if env.post:
            pipe.append({"$set": env.post})
        if hav is not None:
            pipe.append({"$match": hav})
        if sort_temp:
            pipe.append({"$set": sort_temp})
        if sort_spec:
            pipe.append({"$sort": sort_spec})
        if keys:
            pipe += limit_stages(limit, offset)
        proj = {"_id": 0, **{n: project_value(v.expr) for n, v in zip(names, outs)}}
        if tree.args.get("distinct") and keys:   # F3
            self.features.add("distinct_group_by")
            pipe.append({"$group": {"_id": {n: {"$ifNull": [v.expr, None]} for n, v in zip(names, outs)}}})
            proj = {"_id": 0, **{n: "$_id." + n for n in names}}
        pipe.append({"$project": proj})
        return pipe


class GroupEnv:
    """Values AFTER $group: group keys -> $_id.g<i>, aggregates -> accumulator fields."""

    def __init__(self, conv, keys):
        self.c, self.keys = conv, keys
        self.key_exprs = ["$" + s.prefix + col.name for _, s, col in keys]
        self.accs, self.post, self.kinds, self.by_sql = {}, {}, {}, {}

    def defaults(self):
        d = {}
        for name, k in self.kinds.items():
            d[name] = 0 if k in ("count", "nn") else [] if k == "set" else None
        return d

    def _acc(self, key, spec, kind):
        name = f"a{len(self.accs)}"
        self.accs[name] = spec
        self.kinds[name] = kind
        return name

    def aggregate(self, node):
        key = node.sql(dialect="sqlite")
        if key in self.by_sql:
            return self.by_sql[key]
        arg = node.this
        self.c.features.add("count" if isinstance(node, exp.Count) else "aggregate")
        if isinstance(node, exp.Count) and isinstance(arg, exp.Star):
            tv = TV("$" + self._acc(key, {"$sum": 1}, "count"), "num", True)
        elif isinstance(node, exp.Count) and isinstance(arg, exp.Distinct):
            if len(arg.expressions) != 1:
                U("count_distinct", "several expressions")
            v = self.c.row_value(arg.expressions[0])
            if v.kind == "mixed":
                U("aggregate_type", "COUNT(DISTINCT) over mixed types")
            s = self._acc(key, {"$addToSet": v.expr}, "set")
            self.post[s] = {"$size": {"$filter": {"input": "$" + s, "cond": {"$ne": ["$$this", None]}}}}
            self.c.features.add("count_distinct")
            tv = TV("$" + s, "num", True)
        else:
            if isinstance(arg, exp.Distinct):
                U("aggregate", f"{type(node).__name__}(DISTINCT)")
            v = self.c.row_value(arg)
            nn = {"$sum": {"$cond": [{"$eq": [{"$ifNull": [v.expr, None]}, None]}, 0, 1]}}
            if isinstance(node, exp.Count):
                if v.kind == "mixed":
                    pass   # counting non-NULL values does not depend on their type
                tv = TV("$" + self._acc(key, nn, "count"), "num", True)
            elif isinstance(node, (exp.Sum, exp.Avg)):
                if v.kind not in ("num", "null"):
                    U("aggregate_type", f"{type(node).__name__} over {v.kind}")
                if isinstance(node, exp.Avg):
                    tv = TV("$" + self._acc(key, {"$avg": v.expr}, "value"), "num", False)
                else:   # F2: SUM of no non-NULL value is NULL
                    s = self._acc(key, {"$sum": v.expr}, "value")
                    n = self._acc(key + "#nn", nn, "nn")
                    self.post[s] = {"$cond": [{"$gt": ["$" + n, 0]}, "$" + s, None]}
                    tv = TV("$" + s, "num", v.is_int)
            else:
                if v.kind not in ("num", "text", "null"):
                    U("aggregate_type", f"{type(node).__name__} over {v.kind}")
                op = "$min" if isinstance(node, exp.Min) else "$max"
                tv = TV("$" + self._acc(key, {op: v.expr}, "value"), v.kind, v.is_int)
        self.by_sql[key] = tv
        return tv

    def value(self, node):
        if isinstance(node, exp.Paren):
            return self.value(node.this)
        if isinstance(node, AGGS):
            return self.aggregate(node)
        lit = self.c.literal(node)
        if lit is not None:
            return TV(lit_expr(lit[1]), lit[0], isinstance(lit[1], int))
        if isinstance(node, exp.Column):
            r = self.c.resolve(node)
            if r[0] == "str":
                return TV(lit_expr(r[1]), "text")
            _, s, col = r
            for i, (_, ks, kc) in enumerate(self.keys):
                if ks is s and kc is col:
                    return TV(f"$_id.g{i}", col.kind, col.all_int, col=col)
            return self.functional(s, col)
        if isinstance(node, exp.Neg):
            v = self.value(node.this)
            self.c.need_num(v, "negation")
            return TV({"$multiply": [-1, v.expr]}, "num", v.is_int)
        if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
            return self.c.arith(node, self.value)
        U("value", type(node).__name__)

    def functional(self, s, col):
        """A bare column is allowed only if a grouped key is the NON-NULL single-column
        primary key of the same table instance (so every row of the group shares it)."""
        pk = self.c.prof.pk[s.table]
        if len(pk) == 1 and any(ks is s and kc.name == pk[0] for _, ks, kc in self.keys):
            pkc = self.c.prof.tables[s.table][pk[0]]
            if not pkc.has_null:
                key = f"first:{s.alias}.{col.name}"
                if key not in self.by_sql:
                    name = self._acc(key, {"$first": "$" + s.prefix + col.name}, "value")
                    self.by_sql[key] = TV("$" + name, col.kind, col.all_int, col=col)
                self.c.features.add("functional_dependency")
                return self.by_sql[key]
        U("bare_column", f"{s.table}.{col.name} not determined by the GROUP BY")


# ---------------------------------------------------------------- helpers
def like_regex(pattern):
    """SQLite LIKE pattern -> PCRE (with options 'is'): '%' any run, '_' one character
    (newlines included), absolute anchors (\\z, not $, which also matches before a final \\n)."""
    parts = []
    for ch in pattern:
        if ch == "%":
            parts.append(".*")
        elif ch == "_":
            parts.append(".")
        else:
            parts.append("\\" + ch if ch in REGEX_META else ch)
    rx = "".join(parts)
    head = "" if rx.startswith(".*") else "^"
    if head == "" :
        rx = rx[2:]
    tail = "\\z"
    if rx.endswith(".*") and not rx.endswith("\\.*"):
        rx, tail = rx[:-2], ""
    return head + rx + tail


def project_value(e):
    if isinstance(e, str) and e.startswith("$"):
        return e
    return {"$literal": e} if not isinstance(e, dict) else e


def limit_stages(limit, offset):
    out = []
    if offset:
        out.append({"$skip": offset})
    if limit is not None:
        out.append({"$limit": limit})
    return out


def out_names(sel):
    used, names = set(), []
    for i, e in enumerate(sel):
        if isinstance(e, exp.Alias):
            name = e.alias
        elif isinstance(e, exp.Column):
            name = e.name
        elif isinstance(e, exp.Count):
            name = "count"
        elif isinstance(e, AGGS):
            arg = e.this
            name = f"{type(e).__name__.lower()}_{arg.name if isinstance(arg, exp.Column) else 'value'}"
        elif isinstance(e, exp.Star):
            name = "*"
        else:
            name = f"col{i}"
        name = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or f"col{i}"
        if name[0].isdigit():
            name = "c_" + name
        base, k = name, 2
        while name in used or name == "_id" or name.startswith("__"):
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        names.append(name)
    return names


def convert(sql, prof):
    """-> (mql, features) or raise Unsupported."""
    c = Converter(prof, sql)
    mql = c.convert()
    return mql, sorted(c.features)
