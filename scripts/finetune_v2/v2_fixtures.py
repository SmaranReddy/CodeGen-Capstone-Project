"""VERIFIER SIDE. Paired adversarial fixture databases for differential verification.

A fixture is built FROM THE REAL DATABASE of the item: the referenced tables' original
`CREATE TABLE` text (so SQLite column affinity and constraints are identical), filled with
family-specific rows derived from the real rows and the query's literals. Rows are inserted
with INSERT OR IGNORE (constraint violations drop the row), and the rows SQLite actually
STORED are read back; exactly those rows are imported into MongoDB (mongo_import.load). The
two engines therefore always see the same data, whatever affinity conversion SQLite applied.

Families (V3 of docs/nl2mql_next_steps.md, plus sampling/shuffling for discrimination):
  empty          every referenced table empty
  zero_match     real rows minus every row containing a query literal (WHERE matches nothing)
  all_null       sample with every nullable, non-key referenced column set to NULL
  null_mix       sample with ~35% NULLs injected in nullable, non-key columns
  dup            sample plus duplicates of about half its rows (fresh integer key if any)
  case_variants  sample plus copies whose text values (and literal strings) change case
  newline        sample plus copies whose text values contain '\n'
  shuffle        sample with every non-key column shuffled independently (distinct values per
                 GROUP BY key; breaks column correlations so swapped columns are visible)
  zero_div       sample with ~35% of numeric non-key values set to 0
  text_numerics  sample plus copies whose TEXT-stored values are numeric-looking strings
  literal_hits   sample plus copies set to query literals and their neighbours (+-1, case)
  sample_a/b     two seeded random samples (tables <= SAMPLE_N rows are kept whole)
  -- added in verifier-2 (2026-10-30, after the iteration-1 mutant study) --
  boundary_minus / boundary_equal / boundary_plus
                 witness rows with every column compared to a numeric literal set IN PLACE
                 to literal-1 / literal / literal+1 (keys and joins untouched)
  boundary_case  witness rows with every column compared to a string literal set to a case
                 variant of the literal (LIKE patterns: a matching string, then case-swapped)
  orphans        copies of witness rows whose join-key columns hold values absent from the
                 joined table (same storage class), with fresh primary keys
Every sample-based family starts from the WITNESS rows (real rows that satisfy the query's
joins and WHERE, found by the verifier on the real database) plus a random sample, so large
tables cannot hide the rows the query is about. Copies receive fresh primary keys for single
integer, text and composite keys; a family whose added rows are all dropped by constraints is
reported `degenerate` and not counted as a run.

Column classes come from the real rows (verifier's own scan). A family never puts a value of
a new storage class into a column (text stays text, numbers stay numbers), except NULL.

Must not import the converter.
"""
import random
import sqlite3

from sqlglot import exp, parse_one

import v2_common as C

SAMPLE_N = 60
FAMILIES = ("empty", "zero_match", "all_null", "null_mix", "dup", "case_variants", "newline",
            "shuffle", "zero_div", "text_numerics", "literal_hits", "sample_a", "sample_b",
            "boundary_minus", "boundary_equal", "boundary_plus", "boundary_case", "orphans",
            "small_orphans", "null_atoms_even", "null_atoms_odd", "null_keys_even", "null_keys_odd",
            "all_null_keep")
ADDS_ROWS = ("dup", "case_variants", "newline", "text_numerics", "literal_hits", "orphans", "small_orphans")
AUDIT_V3 = {"empty tables": "empty", "zero-match WHERE": "zero_match",
            "all-NULL aggregate columns": "all_null", "duplicate rows": "dup",
            "case variants for LIKE": "case_variants",
            "distinct values per GROUP BY key": "shuffle", "zero divisors": "zero_div",
            "text-typed numerics": "text_numerics", "newline strings": "newline"}


class DbInfo:
    """Verifier-side schema facts of one SQLite file (PRAGMA + a scan of stored classes)."""

    def __init__(self, path, tables):
        con = C.sqlite_ro(path)
        try:
            self.ddl = {}
            for name, sql in con.execute("SELECT name, sql FROM sqlite_master WHERE type='table' "
                                         "AND name NOT LIKE 'sqlite_%'"):
                self.ddl[name] = sql
            self.cols, self.notnull, self.pk = {}, {}, {}
            for t in self.ddl:
                info = con.execute(f'PRAGMA table_info("{t}")').fetchall()
                self.cols[t] = [r[1] for r in info]
                self.notnull[t] = {r[1] for r in info if r[3]}
                self.pk[t] = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5]]
        finally:
            con.close()
        self.tables = tables      # {t: (cols, rows)} real rows (mongo_import.read_tables)
        self.cls = {}
        for t, (cols, rows) in tables.items():
            for i, c in enumerate(cols):
                kinds = set()
                for r in rows:
                    v = r[i]
                    if v is not None:
                        kinds.add("num" if isinstance(v, (int, float)) else
                                  "text" if isinstance(v, str) else "blob")
                self.cls[(t, c)] = kinds

    def table_of(self, name):
        hits = [t for t in self.ddl if t.lower() == name.lower()]
        return hits[0] if len(hits) == 1 else None


def referenced_tables(sql, info):
    out = set()
    for t in parse_one(sql, read="sqlite").find_all(exp.Table):
        real = info.table_of(t.name)
        if real:
            out.add(real)
    return out


def literals(sql, info):
    """Query literals: numbers, strings, and double-quoted names that are not columns of any
    referenced table (SQLite reads those as strings)."""
    tree = parse_one(sql, read="sqlite")
    allcols = {c.lower() for t in info.ddl for c in info.cols[t]}
    nums, strs = set(), set()
    for lit in tree.find_all(exp.Literal):
        if lit.is_string:
            s = lit.this
            strs.add(s)
            core = s.strip("%_")
            if core and core != s:
                strs.add(core)
        else:
            try:
                nums.add(int(lit.this))
            except ValueError:
                nums.add(float(lit.this))
    for col in tree.find_all(exp.Column):
        ident = col.this
        if isinstance(ident, exp.Identifier) and ident.quoted and ident.name.lower() not in allcols:
            strs.add(ident.name)
    return sorted(nums, key=float), sorted(strs)


def atoms(sql, info):
    """-> ([(table, column, node_type, [literal values])] from WHERE, [(table, column)] join keys)."""
    tree = parse_one(sql, read="sqlite")
    alias = {}
    for t in tree.find_all(exp.Table):
        real = info.table_of(t.name)
        if real:
            alias[(t.alias or t.name).lower()] = real
            alias.setdefault(t.name.lower(), real)
    tabs = set(alias.values())

    def colref(c):
        if not isinstance(c, exp.Column):
            return None
        if c.table:
            t = alias.get(c.table.lower())
        else:
            hits = [t for t in tabs if any(x.lower() == c.name.lower() for x in info.cols[t])]
            t = hits[0] if len(hits) == 1 else None
        if not t:
            return None
        hit = [x for x in info.cols[t] if x.lower() == c.name.lower()]
        return (t, hit[0]) if hit else None

    def litval(n):
        neg = isinstance(n, exp.Neg)
        if neg:
            n = n.this
        if isinstance(n, exp.Literal):
            if n.is_string:
                return None if neg else n.this
            try:
                v = int(n.this)
            except ValueError:
                v = float(n.this)
            return -v if neg else v
        if isinstance(n, exp.Column) and colref(n) is None and isinstance(n.this, exp.Identifier) \
                and n.this.quoted:
            return n.name
        return None

    out, where = [], tree.args.get("where")
    found = where.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between, exp.In,
                           exp.Like) if where is not None else []
    for n in found:
        if isinstance(n, exp.Between):
            c, vals = colref(n.this), [litval(n.args["low"]), litval(n.args["high"])]
        elif isinstance(n, exp.In):
            c, vals = colref(n.this), [litval(x) for x in n.expressions]
        else:
            c, v = colref(n.this), litval(n.expression)
            if c is None:
                c, v = colref(n.expression), litval(n.this)
            vals = [v]
        vals = [v for v in vals if v is not None]
        if c and vals:
            out.append((c[0], c[1], type(n).__name__, vals))
    having = tree.args.get("having")   # aggregate(column) <op> literal -> boundary on that column
    for n in (having.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE) if having is not None else []):
        for agg, lit in ((n.this, n.expression), (n.expression, n.this)):
            if isinstance(agg, (exp.Avg, exp.Sum, exp.Min, exp.Max)) and isinstance(agg.this, exp.Column):
                c, v = colref(agg.this), litval(lit)
                if c and v is not None:
                    out.append((c[0], c[1], type(n).__name__, [v]))
    joins = set()
    for j in tree.args.get("joins") or []:
        on = j.args.get("on")
        for eq in (on.find_all(exp.EQ) if on is not None else []):
            for side in (eq.this, eq.expression):
                r = colref(side)
                if r:
                    joins.add(r)
    return out, sorted(joins)


def _like_example(pattern):
    return pattern.replace("%", "").replace("_", "x")


def _false_value(op, vals, kind):
    """A value of the same storage class for which the atom is false (None if unknown)."""
    if kind == {"num"} and all(isinstance(v, (int, float)) for v in vals):
        if op in ("GT", "GTE"):
            return min(vals) - 1
        if op in ("LT", "LTE", "EQ", "In", "Between"):
            return max(vals) + 1
        if op == "NEQ":
            return vals[0]
    if kind == {"text"} and all(isinstance(v, str) for v in vals):
        if op in ("EQ", "In", "Like"):
            return "~no~"
        if op == "NEQ":
            return vals[0]
    return None


def _sample(rows, rng, witness=()):
    """Witness rows first, then a random sample of the other rows (up to SAMPLE_N in total)."""
    if len(rows) <= SAMPLE_N:
        return list(rows)
    left = list(rows)
    for w in witness:
        if w in left:
            left.remove(w)
    k = max(0, SAMPLE_N - len(witness))
    return list(witness) + [left[i] for i in sorted(rng.sample(range(len(left)), min(k, len(left))))]


def _case_variants(s):
    return [x for x in {s.upper(), s.lower(), s.swapcase(), s.title()} if x != s]


class Builder:
    def __init__(self, info, tables, sql, seed, witness=None):
        self.info, self.tables, self.seed = info, sorted(tables), seed
        self.nums, self.strs = literals(sql, info)
        self.witness = witness or {}
        self.atoms, self.join_keys = atoms(sql, info)
        self.base_len = {}

    def _rng(self, family):
        return random.Random(f"{self.seed}:{family}")

    def _keyish(self, t, c):
        return c in self.info.pk[t]

    def _fresh_keys(self, t, rows, base_rows=None):
        """Give copies a fresh primary key (single integer, text or composite): one key
        component changes, keeping its storage class (numbers: above every stored value;
        text: a '~<n>' suffix)."""
        pk = self.info.pk[t]
        cols, real = self.info.tables[t]
        if not pk or not rows:
            return rows
        comp = next((c for c in pk if self.info.cls[(t, c)] == {"num"}), pk[0])
        i = cols.index(comp)
        out = []
        if self.info.cls[(t, comp)] == {"num"}:
            nxt = int(max((r[i] for r in real if isinstance(r[i], (int, float))), default=0)) + 1
            for r in rows:
                r = list(r)
                r[i] = nxt
                nxt += 1
                out.append(tuple(r))
        else:
            for k, r in enumerate(rows):
                r = list(r)
                r[i] = f"{r[i] if r[i] is not None else ''}~{k}"
                out.append(tuple(r))
        return out

    def family(self, name):
        """-> {table: rows} (before SQLite storage)."""
        rng = self._rng(name)
        out = {}
        for t in self.tables:
            cols, real = self.info.tables[t]
            wit = self.witness.get(t, [])
            base = _sample(real, rng, wit)
            self.base_len[t] = len(base)
            idx = {c: i for i, c in enumerate(cols)}

            def nullable(c, t=t):
                return c not in self.info.notnull[t] and not self._keyish(t, c)

            def cls(c, t=t):
                return self.info.cls[(t, c)]

            if name == "empty":
                rows = []
            elif name in ("sample_a", "sample_b"):
                rows = base
            elif name == "zero_match":
                lits = {s.lower() for s in self.strs} | {float(n) for n in self.nums}

                def hit(v):
                    if isinstance(v, str):
                        return v.lower() in lits or any(s.lower() in v.lower() for s in self.strs if len(s) >= 2)
                    return isinstance(v, (int, float)) and float(v) in lits
                rows = [r for r in real if not any(hit(v) for v in r)][:max(SAMPLE_N * 3, 1)]
            elif name == "all_null":
                rows = [tuple(None if nullable(c) else r[idx[c]] for c in cols) for r in base]
            elif name == "null_mix":
                rows = [tuple(None if nullable(c) and rng.random() < 0.35 else r[idx[c]] for c in cols)
                        for r in base]
            elif name == "dup":
                extra = [r for r in base if rng.random() < 0.5]
                rows = base + self._fresh_keys(t, extra, real) + [r for r in extra if not self.info.pk[t]]
            elif name == "case_variants":
                copies = []
                for r in base:
                    if rng.random() < 0.6:
                        copies.append(tuple(rng.choice(_case_variants(v) or [v]) if isinstance(v, str)
                                            and not self._keyish(t, c) else v for c, v in zip(cols, r)))
                for s in self.strs:
                    for var in _case_variants(s):
                        r = list(rng.choice(base)) if base else None
                        if r is None:
                            break
                        tc = [c for c in cols if cls(c) == {"text"} and not self._keyish(t, c)]
                        if tc:
                            r[idx[rng.choice(tc)]] = var
                            copies.append(tuple(r))
                rows = base + self._fresh_keys(t, copies, real)
            elif name == "newline":
                copies = []
                for r in base:
                    if rng.random() < 0.6:
                        copies.append(tuple((v[:len(v) // 2] + "\n" + v[len(v) // 2:]) if isinstance(v, str)
                                            and v and not self._keyish(t, c) else v for c, v in zip(cols, r)))
                rows = base + self._fresh_keys(t, copies, real)
            elif name == "shuffle":
                colvals = {c: [r[idx[c]] for r in base] for c in cols}
                for c in cols:
                    if not self._keyish(t, c):
                        rng.shuffle(colvals[c])
                rows = [tuple(colvals[c][k] for c in cols) for k in range(len(base))]
            elif name == "zero_div":
                rows = [tuple(0 if cls(c) == {"num"} and not self._keyish(t, c) and rng.random() < 0.35
                              and v is not None else v for c, v in zip(cols, r)) for r in base]
            elif name == "text_numerics":
                pool = ["9", "10", "100", "-1", "1.5", "007", "2"] + [str(n) for n in self.nums]
                copies = []
                for r in base:
                    if rng.random() < 0.5:
                        copies.append(tuple(rng.choice(pool) if cls(c) == {"text"} and not self._keyish(t, c)
                                            else v for c, v in zip(cols, r)))
                rows = base + self._fresh_keys(t, copies, real)
            elif name == "literal_hits":
                copies = []
                numv = sorted({v for n in self.nums for v in (n - 1, n, n + 1)}, key=float)
                strv = sorted({v for s in self.strs for v in [s] + _case_variants(s)})
                for k in range(min(40, 4 * (len(numv) + len(strv)) + 4)):
                    if not base:
                        break
                    r = list(base[k % len(base)])
                    for c in cols:
                        if self._keyish(t, c):
                            continue
                        if cls(c) == {"num"} and numv and rng.random() < 0.6:
                            r[idx[c]] = rng.choice(numv)
                        elif cls(c) == {"text"} and strv and rng.random() < 0.6:
                            r[idx[c]] = rng.choice(strv)
                    copies.append(tuple(r))
                rows = base + self._fresh_keys(t, copies, real)
            elif name in ("boundary_minus", "boundary_equal", "boundary_plus", "boundary_case"):
                # Every sampled row (witnesses first) gets ONE target atom, cycling over the
                # table's atoms; per atom the rows alternate between "other atoms unchanged" and
                # "other atoms of the table made false", so a row's outcome can hinge on the target
                # alone (OR) or on all atoms (AND). Numeric literals compared with a TEXT column
                # use their decimal text. boundary_case alternates the exact string / a case variant.
                delta = {"boundary_minus": -1, "boundary_equal": 0, "boundary_plus": 1}.get(name)
                targets = [(c, op, vals) for tt, c, op, vals in self.atoms if tt == t and not self._keyish(t, c)]
                rows = []
                for k, r in enumerate(base):
                    r = list(r)
                    if targets:
                        n = len(targets)
                        ti, parity = k % n, (k // n) % 2
                        if parity == 1:
                            for j, (c, op, vals) in enumerate(targets):
                                if j != ti and c != targets[ti][0]:
                                    fv = _false_value(op, vals, cls(c))
                                    if fv is not None:
                                        r[idx[c]] = fv
                        c, op, vals = targets[ti]
                        v = vals[(k // (2 * n)) % len(vals)]
                        if isinstance(v, (int, float)) and delta is not None:
                            if cls(c) == {"num"}:
                                r[idx[c]] = v + delta
                            elif cls(c) == {"text"} and isinstance(v, int):
                                r[idx[c]] = str(v + delta)
                        elif cls(c) == {"text"} and isinstance(v, str):
                            ex = _like_example(v)
                            if name == "boundary_equal":
                                r[idx[c]] = ex
                            elif name == "boundary_case":
                                r[idx[c]] = ex if parity == 0 else rng.choice(_case_variants(ex) or [ex])
                    rows.append(tuple(r))
            elif name in ("null_keys_even", "null_keys_odd"):
                # NULL join keys on every other row of every table (predicate columns untouched)
                cols_n = {c for tt, c in self.join_keys if tt == t and nullable(c)}
                par = 0 if name.endswith("even") else 1
                rows = [tuple(None if (k % 2 == par and c in cols_n) else v for c, v in zip(cols, r))
                        for k, r in enumerate(base)]
            elif name in ("null_atoms_even", "null_atoms_odd"):
                # NULL in the predicate columns of every other witness row (join keys untouched)
                cols_n = {c for tt, c, _, _ in self.atoms if tt == t and nullable(c)}
                par = 0 if name.endswith("even") else 1
                wset = list(wit) if wit else base[:10]
                changed = [tuple(None if (k % 2 == par and c in cols_n) else v for c, v in zip(cols, r))
                           for k, r in enumerate(wset)]
                rest = list(base)
                for w in wset:
                    if w in rest:
                        rest.remove(w)
                rows = changed + rest
            elif name == "all_null_keep":
                # all-NULL aggregate inputs while the rows still qualify (filters, keys kept)
                keep = {c for tt, c, _, _ in self.atoms if tt == t} | {c for tt, c in self.join_keys if tt == t}
                rows = [tuple(None if nullable(c) and c not in keep else r[idx[c]] for c in cols) for r in base]
            elif name in ("orphans", "small_orphans"):
                keys = [c for tt, c in self.join_keys if tt == t]
                if name == "small_orphans":
                    base = (list(wit) if wit else base)[:3]
                    self.base_len[t] = len(base)
                src = list(wit) if wit else base[:10]
                if name == "small_orphans":
                    src = src[:3]
                copies = []
                ti = self.tables.index(t)   # per-table offsets: orphans of two tables never match
                # one copy per (row, join key): only that key is orphaned, the others still join
                top = max([0] + [v for tt, cc in self.join_keys for v in
                                 (x[self.info.tables[tt][0].index(cc)] for x in self.info.tables[tt][1])
                                 if isinstance(v, (int, float))])
                for k, (r, c) in enumerate((r, c) for r in src for c in keys):
                    r = list(r)
                    if cls(c) == {"num"}:
                        r[idx[c]] = int(top) + 1000 * (ti + 1) + k
                    elif cls(c) == {"text"}:
                        r[idx[c]] = f"{r[idx[c]]}~orphan{ti}.{k}"
                    copies.append(tuple(r))
                if not keys:
                    rows = base
                elif any(c in self.info.pk[t] for c in keys):
                    rows = base + copies
                else:
                    rows = base + self._fresh_keys(t, copies)
            else:
                raise ValueError(name)
            out[t] = rows
        return out

    def materialize(self, name):
        """-> (sqlite connection holding the fixture, {t: (cols, stored_rows)}, problems)."""
        rows = self.family(name)
        con = sqlite3.connect(":memory:")
        con.text_factory = lambda b: b.decode("utf-8", errors="surrogateescape")
        stored, problems = {}, []
        grew = [t for t in self.tables if len(rows[t]) > self.base_len.get(t, 0)]
        for t in self.tables:
            con.execute(self.info.ddl[t])
        for t in self.tables:
            cols = self.info.tables[t][0]
            q = ", ".join(f'"{c}"' for c in cols)
            ph = ", ".join("?" for _ in cols)
            con.executemany(f'INSERT OR IGNORE INTO "{t}" ({q}) VALUES ({ph})', rows[t])
        for t in self.tables:
            cols = self.info.tables[t][0]
            q = ", ".join(f'"{c}"' for c in cols)
            got = [tuple(r) for r in con.execute(f'SELECT {q} FROM "{t}" ORDER BY rowid')]
            for i, c in enumerate(cols):
                kinds = {"num" if isinstance(r[i], (int, float)) else "text" if isinstance(r[i], str)
                         else "blob" for r in got if r[i] is not None}
                if not kinds <= self.info.cls[(t, c)] | ({"num"} if not self.info.cls[(t, c)] else set()):
                    problems.append(f"{t}.{c}: storage class drift {sorted(kinds)}")
            stored[t] = (cols, got)
        if name in ADDS_ROWS and grew and all(len(stored[t][1]) <= self.base_len[t] for t in grew):
            problems.append("degenerate: every added row was dropped by a constraint")
        return con, stored, problems
