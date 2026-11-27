"""VERIFIER SIDE. Fixture families for verifier 3.1 = every fixtures3 family (imported, unchanged) plus the
`fold_traps` families (plan §B.4.3), which close verifier-3's Unicode case-folding blind spot.

  fold_traps          base rows = fixtures3 `boundary_equal` rows (LIKE atoms made true), plus copies of every
                      base row in which each non-key TEXT value has k/K -> U+212A KELVIN SIGN and s/S -> U+017F
                      LATIN SMALL LETTER LONG S. SQLite LIKE never matches such a copy through a k/s pattern
                      character; a PCRE `i` regex does (Unicode folding), so an over-matching translation fails.
  fold_traps_dotted   the same with i -> U+0130, I -> U+0131, e/E <-> U+00E9/U+00C9 (accent case pair).

Copies get fresh primary keys (fixtures v2 rule). A family whose copies are all identical to the base rows
(no substitutable character) is reported degenerate and skipped. Applied only to labels whose SQL uses LIKE.
All non-ASCII characters are written as escapes.

Must not import the converter (sql2mql_v2*.py, schema_profile.py).
"""
import sqlglot
from sqlglot import exp

import fixtures3 as F3
from fixtures3 import FAMILIES as BASE_FAMILIES, DbInfo, referenced_tables, literals, atoms  # noqa: F401

KELVIN, LONG_S = "K", "ſ"
FOLD_MAP = {"k": KELVIN, "K": KELVIN, "s": LONG_S, "S": LONG_S}
DOTTED_MAP = {"i": "İ", "I": "ı", "é": "É", "É": "é", "e": "É", "E": "é"}
FOLD_FAMILIES = ("fold_traps", "fold_traps_dotted")
ADDS_ROWS = F3.ADDS_ROWS + FOLD_FAMILIES


def uses_like(sql):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return " like " in f" {sql.lower()} "
    return any(True for _ in tree.find_all(exp.Like))


def _sub(v, table):
    return "".join(table.get(ch, ch) for ch in v)


class Builder(F3.Builder):
    def _family(self, name):
        if name in FOLD_FAMILIES:
            return self._fold(name)
        return super()._family(name)

    def _fold(self, name):
        table = FOLD_MAP if name == "fold_traps" else DOTTED_MAP
        base = F3.Builder._family(self, "boundary_equal")
        out = {}
        self._fold_changed = False
        for t in self.tables:
            cols, real = self.info.tables[t]
            rows = list(base[t])
            copies = []
            for r in rows:
                c2 = tuple(_sub(v, table) if isinstance(v, str) and not self._keyish(t, c) else v
                           for c, v in zip(cols, r))
                if c2 != tuple(r):
                    copies.append(c2)
            if copies:
                self._fold_changed = True
            out[t] = rows + self._fresh_keys(t, copies, real)
        return out

    def materialize(self, name):
        con, stored, problems = super().materialize(name)
        if name in FOLD_FAMILIES and not getattr(self, "_fold_changed", False):
            problems.append("degenerate: no substitutable character in any base row")
        return con, stored, problems


def make_builder(info, tables, sql, seed, witness):
    b = Builder(info, tables, sql, seed, witness)
    b._sql = sql
    return b
