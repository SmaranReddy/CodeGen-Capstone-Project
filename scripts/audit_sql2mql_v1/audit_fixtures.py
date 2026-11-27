"""Independent fixture families for the audit (does not use fixtures.py).

Families for one query (all rows carry every schema column, NULL = explicit null unless the
`missing` representation is requested at execution time):
  rand0..rand3  random rows from small value domains seeded with the query's literals,
                case variants, +/- neighbours, clock-boundary timestamps, 20% NULLs, many
                duplicates and ties (small domains);
  empty         every referenced table empty;
  zero          base-table rows of which NONE satisfies WHERE (checked on SQLite);
  nullcols      rand0 with every column NOT referenced by WHERE/JOIN set to NULL in all rows
                (aggregates / projections over all-NULL inputs);
  dup           rand0 with every row duplicated (DISTINCT, COUNT, SUM multiplicity).

Judgement (`compare_family`) is the audit's own: positional columns (SQL select order vs the
final $project order), multiset for unordered queries, tie-aware sequence for ORDER BY,
inconclusive when an ORDER BY ... LIMIT boundary falls inside a tie, superset check for
LIMIT without ORDER BY. The gold side runs the reference SQL through sqlsem.to_sqlite (the
only way to execute MySQL clock functions on SQLite); its UDFs are checked independently in
adversarial_udf.json.
"""
import datetime as dt
import random
import string

from sqlglot import exp

import common as K

S = K.S
WORDS = ["alpha", "Bravo", "charlie", "DELTA", "echo", "fox trot"]


def _literals(ast):
    """{(qualifier|None, column): [values]} plus clock columns and date-part columns."""
    lits, clock, parts = {}, set(), {}
    for node in ast.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between, exp.In,
                             exp.Like, exp.ILike):
        cols = list(node.find_all(exp.Column))
        vals = []
        for lit in node.find_all(exp.Literal):
            k, v = S._lit_value(lit)
            if k is not None:
                vals.append(-v if isinstance(lit.parent, exp.Neg) and isinstance(v, (int, float)) else v)
        has_clock = any(S.time_fn(n) for n in node.walk()) or any(
            S._date_shift(n) is not None for n in node.walk())
        for c in cols:
            key = (c.table or None, c.name)
            lits.setdefault(key, []).extend(vals)
            if has_clock:
                clock.add(key)
            if isinstance(c.parent, S.DATE_PART) or isinstance(getattr(c.parent, "parent", None), S.DATE_PART):
                p = c.parent if isinstance(c.parent, S.DATE_PART) else c.parent.parent
                parts.setdefault(key, set()).add(type(p).__name__)
            if isinstance(node, (exp.Like, exp.ILike)):
                lits.setdefault(key + ("like",), []).extend(v for v in vals if isinstance(v, str))
    return lits, clock, parts


def _like_examples(p, rng):
    out = []
    for _ in range(3):
        s = "".join(rng.choice(["", "zz", " Q "]) if ch == "%" else rng.choice(string.ascii_letters) if ch == "_"
                    else ch for ch in p)
        out.append(s)
    core = p.replace("%", "").replace("_", "")
    out += [core.upper(), "x" + core, core + "x", core[:-1] if len(core) > 1 else "q"]
    return out


class Gen:
    def __init__(self, ast, an, schema, clock, seed):
        self.ast, self.an, self.schema, self.N = ast, an, schema, clock
        self.rng = random.Random(seed)
        self.lits, self.clock_cols, self.parts = _literals(ast)
        self.join_cols = {(t, c) for t, c in an.join_keys}
        self._pools = {}
        self.wide = False  # wide domains for ORDER BY keys (set per family)
        self.order_cols = {(t, c) for t, c in an.order_columns}

    def _key(self, t, c):
        # literal lookup by qualifier or by bare name (unqualified single-table queries)
        out = []
        for (q, name, *rest), vals in self.lits.items():
            if name != c or rest:
                continue
            if q is None or self.an.aliases.get(q) == t:
                out += vals
        likes = []
        for (q, name, *rest), vals in self.lits.items():
            if name == c and rest and (q is None or self.an.aliases.get(q) == t):
                likes += vals
        clock = any(name == c and (q is None or self.an.aliases.get(q) == t) for q, name in self.clock_cols)
        parts = set()
        for (q, name), ps in self.parts.items():
            if name == c and (q is None or self.an.aliases.get(q) == t):
                parts |= ps
        return out, likes, clock, parts

    def pool(self, t, c):
        rng, ty = self.rng, self.schema[t][c]
        vals, likes, clock, parts = self._key(t, c)
        if (t, c) in self.join_cols:
            return list(range(1, 5)) if ty in ("int", "float", "bool") else [f"K{i}" for i in range(1, 5)]
        p = []
        if ty in ("int", "float"):
            for v in vals:
                if isinstance(v, (int, float)):
                    p += [v, v - 1, v + 1] if ty == "int" else [v, v - 0.5, v + 0.5]
            p += [0, 1, 2, 3, 5] if ty == "int" else [0.0, 1.5, 2.25, 10.0]
            if ty == "int":
                p = [int(x) for x in p]
        elif ty == "bool":
            p = [0, 1]
        elif ty == "text":
            for v in vals:
                if isinstance(v, str):
                    p += [v, v, v.upper(), v.lower(), v.title(), v + "s", "x" + v]
                elif isinstance(v, (int, float)):
                    p += [str(v), str(v) + "0"]
            for lp in likes:
                p += _like_examples(lp, rng)
            p += WORDS[:3]
        elif ty in ("date", "datetime"):
            fmt = "%Y-%m-%d" if ty == "date" else "%Y-%m-%d %H:%M:%S"
            for v in vals:
                if isinstance(v, str):
                    d, _ = S._parse_dt(v)
                    if d is not None:
                        p += [d.strftime(fmt), (d + dt.timedelta(days=1)).strftime(fmt),
                              (d - dt.timedelta(days=1)).strftime(fmt)]
                elif isinstance(v, int) and parts:
                    for part in parts:
                        for vv in (v, v - 1, v + 1):
                            try:
                                y, m, dd = (vv, 6, 15) if part == "Year" else (2024, vv, 15) if part == "Month" \
                                    else (2024, 6, vv)
                                p.append(dt.datetime(y, m, dd, 10, 30).strftime(fmt))
                            except ValueError:
                                pass
            if clock:
                for off_days in (0, -1, -2, -6, -7, -8, -29, -30, -31, -89, -90, -91, -180, -364, -365, -366, -400, 1, 3):
                    for jitter in (-90, 90):
                        p.append((self.N + dt.timedelta(days=off_days, seconds=jitter)).strftime(fmt))
            p += ["2023-01-15" if ty == "date" else "2023-01-15 08:00:00",
                  "2024-06-30" if ty == "date" else "2024-06-30 23:59:59"]
        return p or [None]

    def wide_pool(self, t, c):
        rng, ty = self.rng, self.schema[t][c]
        if ty == "int":
            return rng.sample(range(100000), 200)
        if ty == "float":
            return [round(rng.uniform(0, 1e5), 3) for _ in range(200)]
        if ty in ("date", "datetime"):
            fmt = "%Y-%m-%d" if ty == "date" else "%Y-%m-%d %H:%M:%S"
            return [(self.N - dt.timedelta(seconds=rng.randint(0, 3 * 10 ** 8))).strftime(fmt) for _ in range(200)]
        if ty == "text":
            return ["".join(rng.choice(string.ascii_letters) for _ in range(8)) for _ in range(200)]
        return self.pool(t, c)

    def row(self, t, null_p=0.2):
        r = {}
        for c in self.schema[t]:
            if self.wide and (t, c) in self.order_cols:
                if ("wide", t, c) not in self._pools:
                    self._pools[("wide", t, c)] = self.wide_pool(t, c)
                r[c] = None if self.rng.random() < 0.05 else self.rng.choice(self._pools[("wide", t, c)])
                continue
            if (t, c) not in self._pools:
                self._pools[(t, c)] = self.pool(t, c)
            r[c] = None if self.rng.random() < null_p else self.rng.choice(self._pools[(t, c)])
        return r

    def tables(self, n):
        return {t: [self.row(t) for _ in range(n)] for t in self.an.columns}


def gold(ast, tables, schema, clock):
    return K.sqlite_rows(ast, tables, schema, clock)


def _where_false_rows(ast, an, g, schema, clock, n=30):
    where = ast.args.get("where")
    if where is None or an.joins:
        return None
    base = an.base
    cand = [g.row(base) for _ in range(n * 4)]
    sql = f'SELECT rowid FROM "{base}" WHERE ' + S._to_sqlite_tree(where.this).sql(dialect="sqlite", identify=True)
    con = S.sqlite_db({base: cand}, {base: schema[base]}, clock)
    try:
        ok = {r[0] for r in S.run_sqlite(con, sql)[1]}
    finally:
        con.close()
    return {base: [r for i, r in enumerate(cand, 1) if i not in ok][:n]}


def families(ast, an, schema, clock, seed):
    g = Gen(ast, an, schema, clock, seed)
    fam = {}
    for i, n in enumerate((8, 16, 30, 60)):
        g.wide = i >= 2  # rand2/rand3: distinct ORDER BY keys, so LIMIT boundaries are decidable
        fam[f"rand{i}"] = g.tables(n)
    g.wide = False
    fam["empty"] = {t: [] for t in an.columns}
    z = _where_false_rows(ast, an, g, schema, clock)
    if z is not None:
        fam["zero"] = z
    keep = {c.name for c in (ast.args["where"].find_all(exp.Column) if ast.args.get("where") else [])}
    keep |= {c for _, c in an.join_keys}
    fam["nullcols"] = {t: [{c: (v if c in keep else None) for c, v in r.items()} for r in rows]
                       for t, rows in fam["rand1"].items()}
    fam["dup"] = {t: rows + [dict(r) for r in rows] for t, rows in fam["rand1"].items()}
    return fam


# ---------------------------------------------------------------- judgement
def _blocks(ast, tables, schema, clock):
    sk = S.sort_key_sql(ast)
    if sk is None:
        return None
    con = S.sqlite_db(tables, {t: schema[t] for t in tables}, clock)
    try:
        sql, nk = sk
        keys = [tuple(r[-nk:]) for r in S.run_sqlite(con, sql)[1]]
    finally:
        con.close()
    out, b, prev = [], -1, object()
    for k in keys:
        if k != prev:
            b += 1
            prev = k
        out.append(b)
    return out


def _limit_tie(ast, an, tables, schema, clock):
    if an.limit is None or ast.args.get("order") is None:
        return False
    m = ast.copy()
    m.set("limit", None)
    m.set("offset", None)
    sk = S.sort_key_sql(m)
    if sk is None:
        return True
    con = S.sqlite_db(tables, {t: schema[t] for t in tables}, clock)
    try:
        keys = [tuple(r[-sk[1]:]) for r in S.run_sqlite(con, sk[0])[1]]
    finally:
        con.close()
    start, end = an.offset or 0, (an.offset or 0) + an.limit
    return (end < len(keys) and keys[end - 1] == keys[end]) or (0 < start < len(keys) and keys[start - 1] == keys[start])


def compare_family(ast, an, mql, tables, schema, clock, missing=False):
    """-> (verdict, detail): pass | fail | inconclusive | mongo_error | gold_error."""
    try:
        exp_rows = gold(ast, tables, schema, clock)
    except Exception as e:  # noqa: BLE001
        return "gold_error", f"{type(e).__name__}: {e}"
    docs, err = K.run_mongo(mql, tables, missing=missing)
    if docs is None:
        return "mongo_error", err
    cols = K.out_columns(mql, schema, an.base)
    got = K.rows_of(docs, cols)
    g = [tuple(K.norm(x) for x in r) for r in got]
    e = [tuple(K.norm(x) for x in r) for r in exp_rows]
    if e and g and len(e[0]) != len(g[0]):
        return "fail", f"column count {len(g[0])} != {len(e[0])}"
    ordered = ast.args.get("order") is not None
    if an.limit is not None and not ordered:
        m = ast.copy()
        m.set("limit", None)
        m.set("offset", None)
        sup = [tuple(K.norm(x) for x in r) for r in gold(m, tables, schema, clock)]
        ok = len(g) == len(e) and all(g.count(r) <= sup.count(r) for r in set(g))
        return ("pass" if ok else "fail"), "limit-without-order superset check"
    if ordered and _limit_tie(ast, an, tables, schema, clock):
        return "inconclusive", "tie at LIMIT/OFFSET boundary"
    if len(g) != len(e):
        return "fail", f"rows {len(g)} != {len(e)}"
    if not ordered:
        return ("pass" if sorted(g, key=repr) == sorted(e, key=repr) else "fail"), "multiset"
    blocks = _blocks(ast, tables, schema, clock)
    if blocks is None:
        return ("pass" if g == e else "fail"), "sequence"
    for b in set(blocks):
        ix = [i for i, x in enumerate(blocks) if x == b]
        if sorted((g[i] for i in ix), key=repr) != sorted((e[i] for i in ix), key=repr):
            return "fail", "tie-aware sequence"
    return "pass", "tie-aware sequence"
