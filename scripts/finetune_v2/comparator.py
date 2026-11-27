"""VERIFIER SIDE. Strict, positional result comparison (independent of spider1_kit.exec_match).

Differences from exec_match, on purpose:
  * columns are compared POSITIONALLY (no column permutation): a swapped projection is wrong;
  * strings compare exactly; a string never equals a number ('10' != 10);
  * numbers compare by value with a tight relative tolerance (1e-9), not rounded to 4 places;
    int and float of equal value are equal (SQLite AVG -> 2.0, MongoDB may give 2);
  * any value that is not null / int / float / str / bytes (dates, objects, arrays, Decimal128)
    makes the comparison fail.
Ordering is checked against SQLite's ORDER BY tie blocks (see `match_ordered`).

Must not import the converter.
"""
import math

ALLOWED = (type(None), int, float, str, bytes)


class BadValue(Exception):
    pass


def _norm(v):
    if isinstance(v, bool):
        raise BadValue(f"boolean value {v!r}")
    try:
        from bson.int64 import Int64
        if isinstance(v, Int64):
            return int(v)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(v, bytearray):
        v = bytes(v)
    if not isinstance(v, ALLOWED):
        raise BadValue(f"unsupported value type {type(v).__name__}")
    if isinstance(v, float) and math.isnan(v):
        raise BadValue("NaN")
    return v


def value_eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    na, nb = isinstance(a, (int, float)), isinstance(b, (int, float))
    if na and nb:
        if a == b:
            return True
        if isinstance(a, float) or isinstance(b, float):
            return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
        return False
    if na or nb:
        return False
    return type(a) is type(b) and a == b


def row_eq(r, s):
    return len(r) == len(s) and all(value_eq(x, y) for x, y in zip(r, s))


def _sort_key(r):
    return tuple((0, 0) if v is None else (1, float(v)) if isinstance(v, (int, float))
                 else (2, v) if isinstance(v, str) else (3, v) for v in r)


GREEDY_MAX = 3000


def multiset_eq(got, exp):
    """Multiset equality under value_eq. Sort both sides and pair them; if that fails (values
    within tolerance may sort differently), fall back to exhaustive greedy matching."""
    if len(got) != len(exp):
        return False
    if any(len(r) != len(s) for r, s in zip(got, exp)):
        return False
    if all(row_eq(r, s) for r, s in zip(sorted(got, key=_sort_key), sorted(exp, key=_sort_key))):
        return True
    if len(got) > GREEDY_MAX:
        return False
    pool = list(exp)
    for r in got:
        for i, c in enumerate(pool):
            if row_eq(r, c):
                pool.pop(i)
                break
        else:
            return False
    return True


def docs_to_rows(docs, out_cols):
    """MongoDB documents -> positional rows over `out_cols` (absent field -> None)."""
    rows = []
    for d in docs:
        extra = set(d) - set(out_cols) - {"_id"}
        if extra:
            raise BadValue(f"unexpected output fields {sorted(extra)}")
        if "_id" in d:
            raise BadValue("output keeps _id")
        rows.append(tuple(_norm(d.get(c)) for c in out_cols))
    return rows


def norm_rows(rows):
    return [tuple(_norm(v) for v in r) for r in rows]


def tie_blocks(keyed, n_cols):
    """keyed: SQLite rows (answer cols + order-key cols), in ORDER BY order.
    -> list of (start, end) half-open ranges of equal order keys."""
    blocks, start = [], 0
    for i in range(1, len(keyed) + 1):
        if i == len(keyed) or not row_eq(keyed[i][n_cols:], keyed[start][n_cols:]):
            blocks.append((start, i))
            start = i
    return blocks


def window_plan(blocks, offset, limit, total):
    """-> (segments, cut). segments: [(block_start, block_end, n_taken_in_window)]."""
    lo = offset or 0
    hi = total if limit is None else min(total, lo + limit)
    segs, cut = [], False
    for s, e in blocks:
        a, b = max(s, lo), min(e, hi)
        if a < b:
            segs.append((s, e, b - a))
            if (a, b) != (s, e):
                cut = True
    return segs, cut


def match_ordered(got, keyed, n_cols, offset, limit):
    """Ordered match allowing any order WITHIN a tie block; a block cut by the window may be
    represented by any sub-multiset of the right size. -> (ok, reason, tie_cut)."""
    answers = [r[:n_cols] for r in keyed]
    segs, cut = window_plan(tie_blocks(keyed, n_cols), offset, limit, len(keyed))
    if len(got) != sum(n for _, _, n in segs):
        return False, f"row count {len(got)} != {sum(n for _, _, n in segs)}", cut
    pos = 0
    for s, e, n in segs:
        part, block = got[pos:pos + n], answers[s:e]
        pos += n
        if n == e - s:
            if not multiset_eq(part, block):
                return False, f"rows differ in order block {s}-{e}", cut
        else:
            pool = list(block)
            for r in part:
                for i, c in enumerate(pool):
                    if row_eq(r, c):
                        pool.pop(i)
                        break
                else:
                    return False, f"row not in cut tie block {s}-{e}", cut
    return True, "ordered match (tie-aware)", cut
