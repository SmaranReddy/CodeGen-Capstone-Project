"""CONVERTER SIDE. sql2mql-v2.1 = frozen sql2mql-v2 with ONE change: Unicode-safe LIKE (plan Part B).

sql2mql_v2.py and schema_profile.py are imported unchanged (their hashes stay pinned by
round3_code_freeze.json / pilot_v3_run01_runner_freeze.json). Only Converter.like is overridden.

Why: v2 emitted `$regex` with options "is". PCRE's `i` uses Unicode case folding, so it is wider than
SQLite LIKE (which folds the 26 ASCII letters only):
  * U+212A KELVIN SIGN folds to k and U+017F LATIN SMALL LETTER LONG S folds to s, so an ASCII pattern
    over-matches values holding those characters (R040, R082);
  * case-paired non-letters (circled letters U+24B6/U+24D0, Roman numerals U+2160/U+2170) pass v2's
    `isalpha()` pattern guard and are folded too.
v2.1 never asks the regex engine to fold case: each ASCII letter becomes a two-character class
`[xX]`, and the options become "s" (dot matches newline; `%`/`_` match newlines as in SQLite) in all
three emission paths ($regex, $not+$regex, $regexMatch). Every other character matches only itself,
which is SQLite's behaviour.

Kept unchanged (so the set of accepted inputs equals v2's): rejection of a non-literal pattern, of LIKE on
num/mixed columns, of non-ASCII letters in the pattern, and of columns flagged fold_trap (now redundant).

Must not import the verifier (verifier*.py, comparator.py, fixtures*.py, mongo_import.py).
"""
import sql2mql_v2 as V2
from sql2mql_v2 import U, Unsupported, REGEX_META  # noqa: F401  (re-export Unsupported)

CONVERTER_VERSION = "sql2mql-v2.1"
BASE_CONVERTER_VERSION = V2.CONVERTER_VERSION
LIKE_OPTIONS = "s"


def _ascii_letter(ch):
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def like_regex_ascii(pattern):
    """SQLite LIKE pattern -> PCRE for options "s" (no `i`).

    '%' -> '.*', '_' -> '.', ASCII letter c -> '[' + lower(c) + upper(c) + ']', REGEX_META characters
    backslash-escaped, anything else literal. Anchors '^' and '\\z' (absolute end; '$' would also match
    before a final newline), dropped next to a leading / trailing '%' exactly as in v2's like_regex.
    """
    parts = []
    for ch in pattern:
        if ch == "%":
            parts.append(".*")
        elif ch == "_":
            parts.append(".")
        elif _ascii_letter(ch):
            parts.append("[" + ch.lower() + ch.upper() + "]")
        else:
            parts.append("\\" + ch if ch in REGEX_META else ch)
    head = "^"
    if parts and parts[0] == ".*":
        head, parts = "", parts[1:]
    tail = "\\z"
    if parts and parts[-1] == ".*":
        tail, parts = "", parts[:-1]
    return head + "".join(parts) + tail


class Converter(V2.Converter):
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
        rx = like_regex_ascii(pattern)
        self.features.add("like")
        if query and v.path:
            if negate:
                return {v.path: {"$not": {"$regex": rx, "$options": LIKE_OPTIONS}, "$ne": None}}
            return {v.path: {"$regex": rx, "$options": LIKE_OPTIONS}}
        m = {"$regexMatch": {"input": v.expr, "regex": rx, "options": LIKE_OPTIONS}}
        return {"$expr": {"$and": [{"$eq": [{"$type": v.expr}, "string"]}, {"$not": [m]} if negate else m]}}


def convert(sql, prof):
    """-> (mql, features) or raise Unsupported. Same contract as sql2mql_v2.convert."""
    c = Converter(prof, sql)
    mql = c.convert()
    return mql, sorted(c.features)
