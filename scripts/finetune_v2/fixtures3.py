"""VERIFIER SIDE (verifier-3). Fixture families of verifier-2 (v2_fixtures.py, unchanged and
imported) PLUS families added in iteration 3 (2026-10-31). Every verifier-2 family is still run,
so verifier-3 is a strict superset.

Added families (`extra_families(builder)` lists the ones that apply to an item):
  <fam>_rev / <fam>_shuf   the verifier-2 family <fam> (sample_a, sample_b, dup, literal_hits,
                           boundary_equal, orphans) with every table's rows inserted in REVERSED /
                           seeded SHUFFLED order. Both engines receive the same order (SQLite
                           rowid order = MongoDB insertion order), so the SQL truth is unchanged
                           except for tie order, which the comparator already treats as free.
                           Kills MQL that only works because natural order equals sort order
                           (e.g. a dropped $sort before $limit 1, $first/$last of a group).
  dangle<i>                IN-PLACE orphan of ONE join-key column (the i-th of the query's
                           join keys): on every other witness row (then sampled row) that column
                           gets a value absent from every joined table, without copying the row
                           and without touching its primary key unless the key IS the primary key.
                           Unlike `orphans` (copies with fresh keys, which upstream tables cannot
                           reach), the changed rows keep every other link, so a middle table of a
                           multi-join chain gets rows whose downstream partner is missing.
  dangle_null<i>           the same rows with that join-key column set to NULL (nullable only).
  dangle_eq<i>, dangle_null_eq<i>
                           the same edits applied to the boundary_equal rows (WHERE literals set
                           in place), on every other row: reaches queries whose WHERE matches no
                           real row (no witnesses), where plain dangle rows are all filtered out.
  keyshift                 every integer join-key VALUE of the query's join graph shifted by a
                           per-table-family offset consistently on both sides of each join edge
                           (join structure preserved), so ids of different tables stop coinciding;
                           kills joins on the wrong column pair (e.g. swapped localField /
                           foreignField) that happen to agree when ids are 1..n everywhere.
  boundary_{minus,equal,plus}_alltrue
                           every predicate atom of a table set TRUE on every row except one
                           target atom (cycling) at literal-1 / literal / literal+1: boundaries of
                           one conjunct while the others hold (verifier-2's boundary families kept
                           the others unchanged or false).
  null_keys_both           NULL in every nullable join-key column on BOTH sides of every edge in
                           the same rows (k % 3 == 0), so NULL = NULL lookups can match in MongoDB
                           (SQL never joins NULLs).

Must not import the converter.
"""
import random

import v2_fixtures as F2
from v2_fixtures import DbInfo, referenced_tables, literals, atoms  # noqa: F401  (re-export)

REORDER_OF = ("sample_a", "sample_b", "dup", "literal_hits", "boundary_equal", "orphans")
BASE_FAMILIES = F2.FAMILIES
ALLTRUE = ("boundary_minus_alltrue", "boundary_equal_alltrue", "boundary_plus_alltrue")
FAMILIES = F2.FAMILIES + tuple(f"{f}_{o}" for f in REORDER_OF for o in ("rev", "shuf")) + \
    ("keyshift", "null_keys_both") + ALLTRUE
ADDS_ROWS = F2.ADDS_ROWS + tuple(f"{f}_{o}" for f in REORDER_OF if f in F2.ADDS_ROWS for o in ("rev", "shuf"))
SHIFT = 100000


def reorder_base(name):
    for o in ("_rev", "_shuf"):
        if name.endswith(o) and name[: -len(o)] in REORDER_OF:
            return name[: -len(o)]
    return None


def extra_families(builder):
    """Item-specific families (one per join-key column)."""
    out = []
    for i, _ in enumerate(builder.join_keys):
        out += [f"dangle{i}", f"dangle_null{i}", f"dangle_eq{i}", f"dangle_null_eq{i}"]
    return tuple(out)


class Builder(F2.Builder):
    def family(self, name):
        self._ref = None
        rows = self._family(name)
        self._n_rows = {t: len(r) for t, r in rows.items()}
        if self._ref is not None:        # in-place families: did any row change?
            self._changed = any(rows[t] != self._ref[t] for t in self.tables)
        return rows

    def _family(self, name):
        base = reorder_base(name)
        if base:
            return super().family(base)     # same rows; the ORDER is changed after storage
        if name.startswith("dangle"):
            return self._dangle(name)
        if name == "keyshift":
            return self._keyshift()
        if name == "null_keys_both":
            return self._null_keys_both()
        if name in ALLTRUE:
            return self._alltrue(name)
        return super().family(name)

    def materialize(self, name):
        """As verifier-2 (v2_fixtures.Builder.materialize), with this module's ADDS_ROWS for the
        degenerate-family check, and a family that changed no row reported as degenerate."""
        con, stored, problems = super().materialize(name)
        if reorder_base(name):
            # reorder the rows AS STORED (the MongoDB insertion order). Reordering before the
            # SQLite insert would be undone for INTEGER PRIMARY KEY tables (rowid = key).
            for t in self.tables:
                cols, rows = stored[t]
                rows = list(rows)
                if name.endswith("_rev"):
                    rows.reverse()
                else:
                    random.Random(f"{self.seed}:{name}:{t}").shuffle(rows)
                stored[t] = (cols, rows)
        grew = [t for t in self.tables if self._n_rows.get(t, 0) > self.base_len.get(t, 0)]
        if name in ADDS_ROWS and name not in F2.ADDS_ROWS and grew and                 all(len(stored[t][1]) <= self.base_len[t] for t in grew):
            problems.append("degenerate: every added row was dropped by a constraint")
        if self._ref is not None and not self._changed:
            problems.append("degenerate: no row changed")
        return con, stored, problems

    # ------------------------------------------------------------ helpers
    def _base(self, name):
        """The unchanged starting rows of an in-place family (also kept as self._ref)."""
        if name.startswith("dangle_eq") or name.startswith("dangle_null_eq"):
            out = super().family("boundary_equal")
        else:
            rng = self._rng(name)
            out = {}
            for t in self.tables:
                cols, real = self.info.tables[t]
                out[t] = F2._sample(real, rng, self.witness.get(t, []))
        for t in self.tables:
            self.base_len[t] = len(out[t])
        self._ref = {t: list(r) for t, r in out.items()}
        return out

    def _all_values(self, c_lower):
        vals = set()
        for tt, cc in self.join_keys:
            cols, real = self.info.tables[tt]
            i = cols.index(cc)
            vals |= {r[i] for r in real}
        return vals

    def _dangle(self, name):
        null = name.startswith("dangle_null")
        i = int(name[len(name.rstrip("0123456789")):])
        t, c = self.join_keys[i]
        eq = "_eq" in name
        base = self._base(name)
        cols = self.info.tables[t][0]
        j = cols.index(c)
        kind = self.info.cls[(t, c)]
        if null and (c in self.info.notnull[t] or c in self.info.pk[t]):
            return base   # cannot hold NULL: family degenerates to a sample (reported as such)
        used = self._all_values(c)
        top = max([0] + [v for v in used if isinstance(v, (int, float))])
        wit = self.witness.get(t, [])
        rows = []
        n_wit = 0
        for k, r in enumerate(base[t]):
            r = list(r)
            in_w = tuple(r) in wit
            n_wit += in_w
            # every other witness row; tables without witnesses (or rows already rewritten by
            # boundary_equal): every other row
            if wit and not eq:
                pick = in_w and n_wit % 2 == 1
            else:
                pick = k % 2 == 0
            if pick and r[j] is not None:
                if null:
                    r[j] = None
                elif kind == {"num"}:
                    r[j] = int(top) + 7000 + k
                elif kind == {"text"}:
                    r[j] = f"{r[j]}~dangle{k}"
            rows.append(tuple(r))
        base[t] = rows
        return base

    def _keyshift(self):
        """Shift integer values of each connected join-key group by its own offset."""
        base = self._base("keyshift")
        # union-find over join-key columns: the two sides of an ON a = b belong to one group
        groups = {}
        for jk in self.join_keys:
            groups[jk] = {jk}
        for a, b in self.join_edges:
            ga, gb = groups[a], groups[b]
            if ga is not gb:
                ga |= gb
                for x in gb:
                    groups[x] = ga
        seen, offset = [], {}
        for jk in self.join_keys:
            g = groups[jk]
            if not any(g is s for s in seen):
                seen.append(g)
            offset[jk] = SHIFT * (1 + [k for k, s in enumerate(seen) if s is g][0]) + 13 * len(seen)
        for t in self.tables:
            cols = self.info.tables[t][0]
            shift = {cols.index(c): offset[(tt, c)] for tt, c in self.join_keys
                     if tt == t and self.info.cls[(t, c)] == {"num"}}
            base[t] = [tuple(v + shift[q] if q in shift and isinstance(v, int) else v
                             for q, v in enumerate(r)) for r in base[t]]
        return base

    def _alltrue(self, name):
        """Conjunction boundaries: on every row, EVERY predicate atom of the table is set to a
        value that satisfies it, except one target atom (cycling) set to literal-1 / literal /
        literal+1 (strings: the literal / a LIKE match). A > 2000 vs > 2001 difference needs a
        row with the target at 2001 AND every other conjunct true, which verifier-2's
        boundary families (others unchanged / others false) did not guarantee."""
        delta = {"boundary_minus_alltrue": -1, "boundary_equal_alltrue": 0, "boundary_plus_alltrue": 1}[name]
        base = self._base(name)
        for t in self.tables:
            cols = self.info.tables[t][0]
            idx = {c: i for i, c in enumerate(cols)}
            targets = [(c, op, vals) for tt, c, op, vals in self.atoms if tt == t and not self._keyish(t, c)]
            if not targets:
                continue
            out = []
            for k, r in enumerate(base[t]):
                r = list(r)
                ti = k % len(targets)
                for j, (c, op, vals) in enumerate(targets):
                    kind = self.info.cls[(t, c)]
                    if j == ti:
                        v = vals[(k // len(targets)) % len(vals)]
                        nv = _boundary(v, delta, kind)
                    else:
                        nv = _true_value(op, vals, kind)
                    if nv is not None:
                        r[idx[c]] = nv
                out.append(tuple(r))
            base[t] = out
        return base

    def _null_keys_both(self):
        base = self._base("null_keys_both")
        for t in self.tables:
            cols = self.info.tables[t][0]
            ks = {cols.index(c) for tt, c in self.join_keys
                  if tt == t and c not in self.info.notnull[t] and c not in self.info.pk[t]}
            base[t] = [tuple(None if (q in ks and k % 3 == 0) else v for q, v in enumerate(r))
                       for k, r in enumerate(base[t])]
        return base

    @property
    def join_edges(self):
        if not hasattr(self, "_edges"):
            self._edges = join_edges(self._sql, self.info)
        return self._edges


def join_edges(sql, info):
    """[(colA, colB)] for every ON a = b of the top-level joins (resolved (table, column))."""
    from sqlglot import exp, parse_one
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

    out = []
    for j in tree.args.get("joins") or []:
        on = j.args.get("on")
        for eq in (on.find_all(exp.EQ) if on is not None else []):
            a, b = colref(eq.this), colref(eq.expression)
            if a and b:
                out.append((a, b))
    return out


def _boundary(v, delta, kind):
    if isinstance(v, (int, float)):
        if kind == {"num"}:
            return v + delta
        if kind == {"text"} and isinstance(v, int):
            return str(v + delta)
        return None
    if isinstance(v, str) and kind == {"text"}:
        return F2._like_example(v) if delta == 0 else None
    return None


def _true_value(op, vals, kind):
    """A value of the column's storage class that satisfies `<col> <op> vals` (None if unknown)."""
    v = vals[0]
    if kind == {"num"} and all(isinstance(x, (int, float)) for x in vals):
        return {"EQ": v, "In": v, "GT": v + 1, "GTE": v, "LT": v - 1, "LTE": v, "NEQ": v + 1,
                "Between": min(vals)}.get(op)
    if kind == {"text"} and all(isinstance(x, str) for x in vals):
        return {"EQ": v, "In": v, "Like": F2._like_example(v), "NEQ": v + "~"}.get(op)
    if kind == {"text"} and all(isinstance(x, int) for x in vals):
        return {"EQ": str(v), "In": str(v)}.get(op)
    return None


def make_builder(info, tables, sql, seed, witness):
    b = Builder(info, tables, sql, seed, witness)
    b._sql = sql
    return b
