"""Synthetic, discriminating fixture databases for one reference query (fixtures-v1).

The source dataset ships no database, so an example can only be checked for result
equivalence between its reference SQL (SQLite) and a candidate MQL (MongoDB) on data we
generate. Such a check is only meaningful if the data can tell wrong queries apart, so:

* every referenced table gets ALL its canonical columns (schema.json), with explicit NULLs;
* values are drawn from pools seeded with the query's own literals and their neighbours
  (v-1, v+1, adjacent dates, case variants, LIKE matches and near-misses), so each
  predicate has rows on both sides of it; clock-relative predicates get timestamps spread
  across their interval boundary around the frozen clock;
* join keys and GROUP BY columns use small domains so joins match and groups repeat;
* DISTINCT queries get duplicated rows; LIMIT queries get more candidate rows than the limit.

`adequate_set` keeps generating variants (seeded, escalating literal-hit probability and row
counts) until every SQL mutant (sqlsem.mutants) and a non-empty result are each witnessed
on at least one kept variant, with at least MIN_VARIANTS kept. Variants whose ORDER BY ...
LIMIT boundary falls inside a tie are discarded (ambiguous expected result).
"""
import datetime as dt
import random
import re
import string

import sqlsem as S
from sqlglot import exp

FIXTURE_VERSION = "fixtures-v1"
MIN_VARIANTS = 3
MAX_SEEDS = 40
NULL_P = 0.1
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
         "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
         "sierra", "tango", "Uniform", "Victor", "Whiskey", "Xray", "Yankee", "Zulu"]
BASE_CLOCK = dt.datetime(2026, 10, 24, 12, 0, 0)


def _rand_text(rng):
    return f"{rng.choice(WORDS)} {rng.choice(WORDS)}" if rng.random() < 0.5 else rng.choice(WORDS)


def _like_match(pattern, rng):
    out = []
    for ch in pattern:
        if ch == "%":
            out.append(rng.choice(["", rng.choice(WORDS), " " + rng.choice(WORDS) + " "]))
        elif ch == "_":
            out.append(rng.choice(string.ascii_lowercase))
        else:
            out.append(ch)
    s = "".join(out)
    return s.upper() if rng.random() < 0.15 else s


def _like_miss(pattern, rng):
    core = pattern.replace("%", "").replace("_", "")
    full = core
    if len(core) > 1:
        i = rng.randrange(len(core))
        core = core[:i] + core[i + 1:]
    # near misses: a dropped character, and the full core in the wrong position for anchored
    # patterns ('abc%' vs 'zz abc', '%abc' vs 'abc zz')
    return rng.choice([core + "qq", "qq" + core[::-1], _rand_text(rng), "qq " + full, full + " qq"])


def _fmt(d, ctype):
    return S.fmt_dt(d, "date" if ctype == "date" else "datetime")


class ColGen:
    def __init__(self, table, col, ctype, lits, roles, clock, max_interval_s, rng):
        self.t, self.c, self.ctype = table, col, ctype
        self.lits = lits
        self.roles = roles
        self.clock = clock
        self.span = max(max_interval_s, 3 * 86400)
        self.rng = rng
        self.hit_pool = self._hit_pool()

    def _hit_pool(self):
        rng, ct = self.rng, self.ctype
        pool = []
        for l in self.lits:
            k, v, op, fn = l["kind"], l["value"], l["op"], l.get("fn")
            if k == "time":
                continue
            if fn in ("year", "month", "day") and ct in ("date", "datetime"):
                for dv in (v, v - 1, v + 1):
                    for _ in range(3):
                        y = dv if fn == "year" else rng.choice([2022, 2023, 2024])
                        m = dv if fn == "month" else rng.randint(1, 12)
                        d = dv if fn == "day" else rng.randint(1, 28)
                        try:
                            pool.append(_fmt(dt.datetime(y, m, d, rng.randint(0, 23), rng.randint(0, 59)), ct))
                        except ValueError:
                            pass
                continue
            if fn == "length" and ct == "text":
                for n in (v - 1, v, v + 1):
                    if n > 0:
                        pool.append("".join(rng.choice(string.ascii_lowercase) for _ in range(int(n))))
                continue
            if k == "null":
                pool.append(None)
                continue
            if ct in ("int", "float", "bool"):
                if k == "bool" or ct == "bool":
                    pool += [0, 1]
                    continue
                if isinstance(v, (int, float)):
                    deltas = (0, -1, 1, 2, -2) if ct == "int" else (0, -0.5, 0.5, -1, 1)
                    pool += [type(v)(v + d) if ct == "int" else round(float(v) + d, 2) for d in deltas]
                    if ct == "int":
                        pool = [int(x) for x in pool]
                continue
            if ct == "text" and isinstance(v, str):
                if op == "like":
                    pool += [_like_match(v, rng) for _ in range(3)] + [_like_miss(v, rng) for _ in range(2)]
                else:
                    pool += [v, v, v.upper() if v.upper() != v else v.lower(), v + "x", "x" + v]
                continue
            if ct in ("date", "datetime") and isinstance(v, str):
                base, _ = S._parse_dt(v)
                if base is None:
                    continue
                for delta in (0, 0, -1, 1, -30, 30):
                    d = base + dt.timedelta(days=delta)
                    pool.append(_fmt(d, ct))
                if ct == "datetime":
                    pool += [_fmt(base + dt.timedelta(hours=rng.randint(1, 23), minutes=rng.randint(0, 59)), ct),
                             _fmt(base - dt.timedelta(seconds=1), ct), _fmt(base + dt.timedelta(seconds=1), ct)]
        if any(l["kind"] == "time" for l in self.lits) and self.ctype in ("date", "datetime"):
            for _ in range(8):
                off = self.rng.uniform(-2 * self.span, self.span)
                pool.append(_fmt(self.clock + dt.timedelta(seconds=off), self.ctype))
        return pool

    def random_value(self):
        rng, ct = self.rng, self.ctype
        if "join" in self.roles:
            return rng.randint(1, 6) if ct in ("int", "float", "bool") else f"K{rng.randint(1, 6)}"
        if "group" in self.roles:
            if ct in ("int", "float"):
                return rng.randint(1, 4)
            if ct == "bool":
                return rng.randint(0, 1)
            if ct in ("date", "datetime"):
                return _fmt(dt.datetime(rng.choice([2022, 2023]), rng.choice([1, 6]), rng.choice([1, 15])), ct)
            return rng.choice(WORDS[:4])
        if ct == "int":
            return rng.randint(0, 1000)
        if ct == "float":
            return round(rng.uniform(0, 1000), 2)
        if ct == "bool":
            return rng.randint(0, 1)
        if ct in ("date", "datetime"):
            if any(l["kind"] == "time" for l in self.lits):
                off = rng.uniform(-3 * self.span, 2 * self.span)
                return _fmt(self.clock + dt.timedelta(seconds=off), ct)
            d = dt.datetime(2021, 1, 1) + dt.timedelta(seconds=rng.randint(0, 6 * 365 * 86400))
            return _fmt(d, ct)
        return _rand_text(rng)

    def value(self, p_hit):
        if self.rng.random() < NULL_P:
            return None
        if self.hit_pool and self.rng.random() < p_hit:
            return self.rng.choice(self.hit_pool)
        return self.random_value()


# ---------------------------------------------------------------- witness rows
def _branches(node):
    """WHERE tree -> list of conjunctions [(atom, positive)], one per OR branch (capped)."""
    from sqlglot import exp as E
    if isinstance(node, E.Paren):
        return _branches(node.this)
    if isinstance(node, E.And):
        out = []
        for a in _branches(node.this):
            for b in _branches(node.expression):
                out.append(a + b)
                if len(out) > 16:
                    return out
        return out
    if isinstance(node, E.Or):
        return _branches(node.this) + _branches(node.expression)
    if isinstance(node, E.Not) and not isinstance(node.this, (E.And, E.Or)):
        return [[(x, not pos) for x, pos in br] for br in _branches(node.this)]
    return [[(node, True)]]


def _shift_value(v, ctype, direction, rng):
    """A value strictly above (direction=1) or below (-1) v for the column type."""
    if v is None:
        return None
    if ctype in ("int", "bool"):
        return int(v) + direction * rng.randint(1, 3)
    if ctype == "float":
        return round(float(v) + direction * rng.choice([0.5, 1, 7.25]), 2)
    if ctype in ("date", "datetime"):
        d, _ = S._parse_dt(v)
        if d is None:
            return None
        return _fmt(d + dt.timedelta(days=direction * rng.choice([1, 2, 9])), ctype)
    v = str(v)
    return (v + "a") if direction > 0 else (v[:-1] if v else None)


def _assign(atom, positive, an, schema, gens, clock_value, rng):
    """-> {column: value} making `atom` evaluate to `positive` (best effort), or None.
    Only a proposal: witness_rows confirms every candidate row with SQLite."""
    from sqlglot import exp as E

    def col_of(n):
        n = S.unwrap_date(n)
        if isinstance(n, E.Column):
            t = an["aliases"].get(n.table) if n.table else an["base"]
            if t == an["base"]:
                return n.name
        return None

    def lit_of(n):
        if S.is_literal(n):
            k, v = S._lit_value(n)
            return (1 if v else 0) if k == "bool" else v
        if S.time_fn(n) or S._date_shift(n) is not None:
            return clock_value(n)
        return None

    cols = schema[an["base"]]
    if isinstance(atom, (E.EQ, E.NEQ, E.GT, E.GTE, E.LT, E.LTE)):
        a, b = atom.this, atom.expression
        op = type(atom)
        if col_of(a) is None and col_of(b) is not None:
            a, b = b, a
            op = {E.GT: E.LT, E.GTE: E.LTE, E.LT: E.GT, E.LTE: E.GTE}.get(op, op)
        c = col_of(a)
        if c is None:
            return None
        if isinstance(a, S.DATE_PART):
            v = lit_of(b)
            if not isinstance(v, int):
                return None
            target = {E.EQ: v, E.NEQ: v + 1, E.GT: v + 1, E.GTE: v, E.LT: v - 1, E.LTE: v}[op]
            if not positive:
                target = {E.EQ: v + 1, E.NEQ: v, E.GT: v, E.GTE: v - 1, E.LT: v, E.LTE: v + 1}[op]
            y, m, d = 2023, 6, 15
            part = type(a).__name__
            if part == "Year":
                y = target
            elif part == "Month":
                m = target
            else:
                d = target
            try:
                return {c: _fmt(dt.datetime(y, m, d, rng.randint(0, 23), rng.randint(0, 59)), cols[c])}
            except ValueError:
                return None
        if not isinstance(a, E.Column):
            return None
        c2 = col_of(b)
        if c2 is not None:  # column vs column
            v = gens[c].random_value()
            if positive == (op in (E.EQ, E.GTE, E.LTE)) and cols[c] == cols[c2]:
                return {c: v, c2: v}
            return None
        v = lit_of(b)
        if v is None:
            return None
        ct = cols[c]
        exact_ok = not (S.time_fn(b) or S._date_shift(b) is not None)
        if op is E.EQ:
            return {c: v} if positive else {c: _shift_value(v, ct, rng.choice([1, -1]), rng)}
        if op is E.NEQ:
            return {c: _shift_value(v, ct, 1, rng)} if positive else {c: v}
        up = op in (E.GT, E.GTE)
        if positive:
            if op in (E.GTE, E.LTE) and exact_ok and rng.random() < 0.3:
                return {c: v}
            return {c: _shift_value(v, ct, 1 if up else -1, rng)}
        if op in (E.GT, E.LT) and exact_ok and rng.random() < 0.5:
            return {c: v}
        return {c: _shift_value(v, ct, -1 if up else 1, rng)}
    if isinstance(atom, E.Between):
        c = col_of(atom.this)
        if c is None or not isinstance(atom.this, E.Column):
            return None
        lo, hi = lit_of(atom.args["low"]), lit_of(atom.args["high"])
        if lo is None or hi is None:
            return None
        if positive:
            return {c: rng.choice([lo, hi])}
        if rng.random() < 0.5:
            return {c: _shift_value(lo, schema[an["base"]][c], -1, rng)}
        return {c: _shift_value(hi, schema[an["base"]][c], 1, rng)}
    if isinstance(atom, E.In):
        c = col_of(atom.this)
        if c is None or not isinstance(atom.this, E.Column):
            return None
        vals = [lit_of(x) for x in atom.expressions]
        if positive:
            return {c: rng.choice(vals)}
        return {c: _shift_value(rng.choice(vals), cols[c], 1, rng)}
    if isinstance(atom, (E.Like, E.ILike)):
        c = col_of(atom.this)
        if c is None or not isinstance(atom.this, E.Column):
            return None
        pat = atom.expression.this
        return {c: _like_match(pat, rng) if positive else _like_miss(pat, rng)}
    if isinstance(atom, E.Is):
        c = col_of(atom.this)
        if c is None:
            return None
        return {c: None} if positive else {c: gens[c].random_value()}
    if isinstance(atom, E.Column):
        c = col_of(atom)
        return {c: 1 if positive else 0} if c else None
    return None


def witness_rows(ast, an, schema, gens, rng, clock, n_witness=3):
    """Rows satisfying the WHERE clause (per OR branch) plus, for each atom of a branch, a copy
    with only that atom flipped. Every candidate is evaluated by SQLite, so only real
    witnesses and real near-misses (WHERE false) are returned."""
    where = ast.args.get("where")
    if where is None or an["joins"]:
        return []
    base = an["base"]
    cols = list(schema[base])
    cache = {}

    def clock_value(node):
        key = node.sql()
        if key not in cache:
            con = S.sqlite_db({}, {}, clock)
            try:
                cache[key] = S.run_sqlite(con, "SELECT " + S._to_sqlite_tree(node).sql(dialect="sqlite"))[1][0][0]
            finally:
                con.close()
        return cache[key]

    branches = _branches(where.this)[:8]
    cands = []
    for br in branches:
        for _ in range(n_witness * 3):
            row = {c: gens[c].random_value() for c in cols}
            order = list(br)
            rng.shuffle(order)
            for atom, pos in order:
                a = _assign(atom, pos, an, schema, gens, clock_value, rng)
                if a:
                    row.update(a)
            # several positive '%core%' LIKEs on one column: one value containing every core
            likes = {}
            for atom, pos in br:
                if pos and isinstance(atom, (exp.Like, exp.ILike)) and isinstance(atom.this, exp.Column):
                    likes.setdefault(atom.this.name, []).append(atom.expression.this)
            for c, pats in likes.items():
                if len(pats) > 1 and all(x.startswith("%") and x.endswith("%") for x in pats) and c in row:
                    row[c] = " ".join(x.strip("%").replace("%", " ") for x in pats)
            cands.append(row)
    wsql = f'SELECT rowid FROM "{base}" WHERE ' + S._to_sqlite_tree(where.this).sql(dialect="sqlite", identify=True)

    def passing(rows):
        con = S.sqlite_db({base: rows}, {base: schema[base]}, clock)
        try:
            return {r[0] for r in S.run_sqlite(con, wsql)[1]}
        finally:
            con.close()

    ok = passing(cands)
    witnesses = [r for i, r in enumerate(cands, 1) if i in ok][: n_witness * 2]
    if not witnesses:
        return []
    viol = []
    for w in witnesses[:n_witness]:
        for br in branches:
            for atom, pos in br:
                a = _assign(atom, not pos, an, schema, gens, clock_value, rng)
                if pos and isinstance(atom, (exp.Like, exp.ILike)) and isinstance(atom.this, exp.Column):
                    others = [x.expression.this for x, p2 in br if p2 and x is not atom
                              and isinstance(x, (exp.Like, exp.ILike)) and isinstance(x.this, exp.Column)
                              and x.this.name == atom.this.name]
                    if others and all(o.startswith("%") and o.endswith("%") for o in others):
                        a = {atom.this.name: " ".join(o.strip("%").replace("%", " ") for o in others)}
                if a:
                    r = dict(w)
                    r.update(a)
                    viol.append(r)
    if not viol:
        return witnesses
    bad = passing(viol)
    return witnesses + [r for i, r in enumerate(viol, 1) if i not in bad]


def generate(an, schema, seed, clock, p_hit=0.5, scale=1, ast=None):
    """-> {table: [row dict]} for one variant."""
    rng = random.Random(seed)
    tables = sorted(an["columns"])
    roles = {}
    for t, c in an["join_keys"]:
        roles.setdefault((t, c), set()).add("join")
    for t, c in an["group_columns"]:
        roles.setdefault((t, c), set()).add("group")
    lits = {}
    for l in an["literals"]:
        lits.setdefault((l["table"], l["column"]), []).append(l)
    limit = (an.get("limit") or 0) + (an.get("offset") or 0)
    n_base = 14 if an["joins"] else 24
    n = max(n_base, 2 * limit + 8) * scale
    n = min(n, 3000)
    out = {}
    all_gens = {t: {c: ColGen(t, c, schema[t][c], lits.get((t, c), []), roles.get((t, c), set()),
                              clock, an.get("max_interval_s") or 0, rng) for c in schema[t]}
                for t in tables}
    join_pool = [v for t in tables for c, g in all_gens[t].items() if "join" in g.roles for v in g.hit_pool]
    if join_pool:
        for t in tables:
            for g in all_gens[t].values():
                if "join" in g.roles:
                    g.hit_pool = list(join_pool)
    for t in tables:
        gens = all_gens[t]
        rows = [{c: g.value(p_hit) for c, g in gens.items()} for _ in range(n)]
        if ast is not None and t == an["base"] and seed % 2 == 0:
            for w in witness_rows(ast, an, schema, gens, rng, clock):
                rows.insert(rng.randrange(len(rows) + 1), w)
        if "distinct" in an["features"] and rows:
            rows += [dict(rng.choice(rows)) for _ in range(max(2, n // 5))]
        out[t] = rows
    return out


def compare(pred_cols, pred_rows, gold_cols, gold_rows, ordered, blocks):
    """Result comparison used everywhere in ft-v1 -> (match, reason).

    Fast path: when the predicted column names are exactly the expected column names, compare
    under that identity column mapping with the benchmark's normalisation (spider1_kit._norm)
    and the same ordered / tie-aware (blocks) / multiset rules as spider1_kit.exec_match. That
    is the one column permutation exec_match would accept first, so the fast path never
    accepts a result exec_match rejects; it avoids exec_match's permutation search, which is
    exponential when many columns share a value multiset (wide SELECT * over 0/1 columns).
    Otherwise falls back to spider1_kit.exec_match unchanged.
    """
    import collections
    import spider1_kit as K
    if pred_cols is not None and len(pred_cols) == len(gold_cols) and set(pred_cols) == set(gold_cols)             and len(set(gold_cols)) == len(gold_cols):
        idx = [pred_cols.index(c) for c in gold_cols]
        pred = [tuple(K._norm(r[i]) for i in idx) for r in pred_rows]
        gold = [tuple(K._norm(v) for v in r) for r in gold_rows]
        if len(pred) != len(gold):
            return False, f"row count {len(pred)} != gold {len(gold)}"
        if not ordered:
            return (collections.Counter(pred) == collections.Counter(gold)), "name-aligned multiset"
        if blocks is None:
            return pred == gold, "name-aligned ordered"
        for b in sorted(set(blocks)):
            ix = [i for i, x in enumerate(blocks) if x == b]
            if collections.Counter(pred[i] for i in ix) != collections.Counter(gold[i] for i in ix):
                return False, "name-aligned ordered (tie-aware) differs"
        return True, "name-aligned ordered (tie-aware)"
    return K.exec_match(pred_rows, gold_rows, blocks=blocks, ordered=ordered)


def _cmp(cols, rows_a, rows_b, ordered, blocks):
    return compare(cols, rows_a, cols, rows_b, ordered, blocks)[0]


def evaluate_variant(ast, an, schema, tables, clock, mutant_list):
    """-> dict with expected rows, blocks, validity and the set of mutants it kills."""
    con = S.sqlite_db(tables, {t: schema[t] for t in tables}, clock)
    try:
        cols, rows = S.run_sqlite(con, S.to_sqlite(ast))
        ordered = ast.args.get("order") is not None
        blocks = None
        valid, why = True, None
        if ordered:
            sk = S.sort_key_sql(ast)
            if sk is None:
                blocks = None
            else:
                sql_k, nk = sk
                _, krows = S.run_sqlite(con, sql_k)
                keys = [tuple(r[-nk:]) for r in krows]
                blocks, b, prev = [], -1, object()
                for k in keys:
                    if k != prev:
                        b += 1
                        prev = k
                    blocks.append(b)
                if an.get("limit") is not None:
                    m = ast.copy()
                    m.set("limit", None)
                    m.set("offset", None)
                    sk2 = S.sort_key_sql(m)
                    _, all_rows = S.run_sqlite(con, sk2[0])
                    all_keys = [tuple(r[-nk:]) for r in all_rows]
                    end = (an.get("offset") or 0) + an["limit"]
                    start = an.get("offset") or 0
                    if end < len(all_keys) and all_keys[end - 1] == all_keys[end]:
                        valid, why = False, "tie_at_limit_boundary"
                    if start > 0 and start < len(all_keys) and all_keys[start - 1] == all_keys[start]:
                        valid, why = False, "tie_at_offset_boundary"
        elif an.get("limit") is not None:
            # LIMIT without ORDER BY: natural order (insertion order) is used by both engines
            pass
        _, cnt = S.run_sqlite(con, S.count_matching_sql(ast))
        matching = cnt[0][0]
        kills = set()
        for label, m in mutant_list:
            try:
                _, mrows = S.run_sqlite(con, S.to_sqlite(m))
            except Exception:  # noqa: BLE001 - an unexecutable mutant cannot be witnessed
                continue
            if not _cmp(cols, mrows, rows, ordered, blocks if ordered else None):
                kills.add(label)
        return {"columns": cols, "rows": rows, "ordered": ordered, "blocks": blocks,
                "valid": valid, "invalid_reason": why, "matching_rows": matching,
                "nonempty": bool(rows) and matching > 0, "kills": kills}
    finally:
        con.close()


def implied_mutant(label, ast, an=None, schema=None):
    """Mutants that cannot change the result by construction (documented exemptions):
    * dropping `x IS NOT NULL` when another atom on x already rejects NULL;
    * swapping a strict/non-strict comparison (`>` vs `>=`) whose two sides can never be
      equal on well-formed data: a datetime column ('YYYY-MM-DD HH:MM:SS') against a
      date-only literal, or any column against a clock expression (second-resolution
      equality with NOW() is not a property a fixture can or should pin down)."""
    parts = label.split(":", 3)
    if len(parts) == 4 and parts[1] == "opswap":
        atom = S._atoms(ast.args.get(parts[0]).this)[int(parts[2])]
        sides = (atom.this, atom.expression)
        if any(S.time_fn(x) or S._date_shift(x) is not None for x in sides):
            return True
        from sqlglot import exp as E
        cols = [x for x in sides if isinstance(x, E.Column)]
        lits = [x for x in sides if isinstance(x, E.Literal) and x.is_string]
        if cols and lits and an is not None and schema is not None:
            c = cols[0]
            t = an["aliases"].get(c.table) if c.table else an["base"]
            ctype = schema.get(t, {}).get(c.name)
            if ctype == "datetime" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", lits[0].this):
                return True
        return False
    if len(parts) < 4 or parts[1] != "drop":
        return False
    text = parts[3]
    if " IS NULL" in text and text.startswith("NOT "):
        col = text[4:].split(" IS NULL")[0]
        clause = ast.args.get(parts[0])
        others = [x.sql() for x in S._atoms(clause.this) if x.sql() != text]
        return any(col in o for o in others)
    return False


def adequate_set(ast, an, schema, seed_base, clock):
    mlist = S.mutants(ast)
    need = {m[0] for m in mlist if not implied_mutant(m[0], ast, an, schema)} | {"nonempty"}
    covered, kept, tried, invalid = set(), [], 0, {}
    for i in range(MAX_SEEDS):
        tried += 1
        p_hit = (0.5, 0.65, 0.8, 0.9)[i % 4]
        scale = 1 + i // 8
        seed = seed_base * 1000 + i
        tables = generate(an, schema, seed, clock, p_hit, scale, ast=ast)
        res = evaluate_variant(ast, an, schema, tables, clock, mlist)
        if not res["valid"]:
            invalid[res["invalid_reason"]] = invalid.get(res["invalid_reason"], 0) + 1
            continue
        new = (res["kills"] | ({"nonempty"} if res["nonempty"] else set())) & need
        if new - covered or len(kept) < MIN_VARIANTS:
            kept.append({"seed": seed, "p_hit": p_hit, "scale": scale, "tables": tables, **res})
            covered |= new
        if covered >= need and len(kept) >= MIN_VARIANTS:
            break
    uncovered = sorted(need - covered) + (["min_variants"] if len(kept) < MIN_VARIANTS else [])
    return {"kept": kept, "need": sorted(need), "covered": sorted(covered),
            "uncovered": uncovered, "seeds_tried": tried, "invalid": invalid,
            "exempt_mutants": sorted({m[0] for m in mlist} - need - {"nonempty"})}
