"""
Clean Spider 1.0 (via Smaran's spider-sql-v1 splits) into a reproducible pool of NL -> SQL examples
that are worth converting to MQL: each gold has a non-trivial, unique, verifiable answer on a
populated database, so a converted MQL query can later be accepted only if it reproduces it.

Inputs (both hash-checked, the build aborts on any mismatch)
  --raw     official spider_data/ from spider_data.zip (sha256 00636695...85b121b)
  --sqlv1   spider-sql-v1 processed folder (train / val_internal / dev_full + manifest.json)

Rules, applied in this order (first failing rule is the reject reason). Fixed before any MQL is written:
  R1 weak_verification   tier is not exec_ok: the official DB is empty (academic, geo, imdb, ...)
  R2 exec_error          gold SQL fails or times out on the official SQLite DB
  R3 result_mismatch     result differs from the hash recorded in spider-sql-v1 (audits that dataset;
                         the few hits are float sums that differ between SQLite versions)
  R4 weak_result         empty result or only NULLs: any wrong query can return that too
  R5 zero_only_result    a single 0 (e.g. COUNT = 0): same problem
  R6 too_many_rows       > 1000 rows
  R7 limit_tie           ORDER BY ... LIMIT cuts inside a tie: the gold answer is not unique
  R8 order_unverifiable  ORDER BY whose tie structure cannot be checked (set ops, DISTINCT, ...)
  R9 duplicate_question  same database + normalised question already kept (train, then val, then dev)

The test split is never read (it stays blind). Output is byte-for-byte reproducible: no timestamps,
fixed ordering and JSON settings; manifest.json records input, code and output hashes.

Usage
  python scripts/build_spider_pool.py --raw data/raw/spider1/spider_data \
      --sqlv1 ~/Downloads/spider_sql_v1/data/processed/spider_sql_v1 --out data/spider_pool
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict

SPLITS = [("train", "train.jsonl"), ("val_internal", "val_internal.jsonl"), ("dev", "dev_full.jsonl")]
RAW_FILES = ["train_spider.json", "train_others.json", "dev.json", "test.json", "tables.json",
             "test_tables.json"]
MAX_ROWS = 1000


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_line(row) -> str:
    return json.dumps([round(v, 6) if isinstance(v, float) else v for v in row])


def result_hashes(rows) -> tuple[str, str]:
    """Same scheme as spider-sql-v1: sha256 of the JSON rows (floats rounded to 6 places) joined by
    newlines; the multiset hash sorts the lines first."""
    lines = [row_line(r) for r in rows]
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
    return h("\n".join(lines)), h("\n".join(sorted(lines)))


def run_sql(sqlite_path: str, sql: str, timeout_s: float):
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: time.monotonic() > deadline, 10000)
    try:
        cur = con.execute(sql)
        return [d[0] for d in cur.description], [tuple(r) for r in cur.fetchall()]
    finally:
        con.close()


def sql_key(sql: str) -> str:
    s = re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()
    return re.sub(r"\s*([(),=<>])\s*", r"\1", s)


def question_key(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", q.lower())).strip()


def clean_question(q: str) -> str:
    """Undo Spider's tokenisation spacing ("employees ?" -> "employees?"); wording is unchanged."""
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"\s+([?.,!;:])", r"\1", q)
    return re.sub(r"\s+'s\b", "'s", q)


# ---------------------------------------------------------------- ORDER BY tie analysis
# Adapted from Smaran's scripts/spider1/spider1_kit.py (order_info / tie_analysis).
def _top_level_positions(sql: str, keyword: str) -> list[int]:
    out, depth, i, n, q = [], 0, 0, len(sql), None
    kw, low = keyword.lower(), sql.lower()
    while i < n:
        ch = sql[i]
        if q:
            if ch == q:
                q = None
        elif ch in ("'", '"'):
            q = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and low.startswith(kw, i):
            before = low[i - 1] if i else " "
            after = low[i + len(kw)] if i + len(kw) < n else " "
            if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                out.append(i)
        i += 1
    return out


def order_info(sql: str) -> dict:
    s = sql.strip().rstrip(";")
    info = {"order_by": False, "limit": None, "analyzable": False}
    set_op = any(_top_level_positions(s, k) for k in ("union", "intersect", "except"))
    ob = _top_level_positions(s, "order by")
    if not ob:
        lim = _top_level_positions(s, "limit")
        if lim:
            info["limit"] = int(re.findall(r"\d+", s[lim[-1]:])[0])
        return info
    info["order_by"] = True
    tail = s[ob[-1] + len("order by"):]
    lim = _top_level_positions(tail, "limit")
    if lim:
        info["limit"] = int(re.findall(r"\d+", tail[lim[-1]:])[0])
        tail = tail[:lim[-1]]
    info["order_expr"] = re.sub(r"\s+(asc|desc)\s*$", "", tail.strip(), flags=re.I)
    sel, frm = _top_level_positions(s, "select"), _top_level_positions(s, "from")
    if sel and frm and sel[0] == 0 and not set_op:
        info["select_list"] = s[len("select"):frm[0]].strip()
        info["from_rest"] = s[frm[0]:ob[-1]].strip()
        info["direction"] = "desc" if re.search(r"\bdesc\s*$", tail.strip(), re.I) else "asc"
        info["analyzable"] = True
    return info


def tie_analysis(sqlite_path: str, sql: str, gold_rows, timeout_s: float) -> dict:
    """Tie-block id per gold row, and whether a LIMIT cutoff falls inside a tie. Verified by
    re-deriving the gold rows from the same query with the ORDER BY key appended."""
    info = order_info(sql)
    res = {"order_by": info["order_by"], "limit": info["limit"], "tie_blocks": None,
           "tie_at_limit": False, "status": "no_order"}
    if not info["order_by"]:
        return res
    if not info["analyzable"]:
        res["status"] = "unanalyzable"
        return res
    sel = info["select_list"]
    if sel.lower().startswith("distinct "):
        res["status"] = "distinct_unverified"
        return res
    keyed = (f"SELECT {sel}, {info['order_expr']} {info['from_rest']} "
             f"ORDER BY {info['order_expr']} {info['direction']}")
    try:
        _, rows = run_sql(sqlite_path, keyed, timeout_s)
    except Exception as e:  # noqa: BLE001
        res["status"] = f"keyed_query_failed: {e}"
        return res
    k = info["limit"]
    head = rows[:k] if k is not None else rows
    if [r[:-1] for r in head] != list(gold_rows):
        res["status"] = "keyed_rows_differ"
        return res
    keys = [r[-1] for r in head]
    blocks, b = [], 0
    for i, key in enumerate(keys):
        if i and key != keys[i - 1]:
            b += 1
        blocks.append(b)
    res["tie_blocks"] = blocks
    res["tie_at_limit"] = bool(k is not None and len(rows) > k and rows[k][-1] == rows[k - 1][-1])
    res["status"] = "ok"
    return res


# ---------------------------------------------------------------- SQL features
def sql_features(sql: str) -> list[str]:
    s = re.sub(r"'[^']*'|\"[^\"]*\"", "''", sql.lower())
    n_join = len(re.findall(r"\bjoin\b", s))
    f = []
    if n_join:
        f.append("join" if n_join == 1 else "multi_join")
    if len(re.findall(r"\bselect\b", s)) > 1 and not re.search(r"\b(union|intersect|except)\b", s):
        f.append("nested")
    if re.search(r"\b(union|intersect|except)\b", s):
        f.append("set_op")
    for kw, name in (("group by", "group"), ("having", "having"), ("order by", "sort"),
                     ("limit", "limit"), ("distinct", "distinct"), ("like", "like")):
        if re.search(rf"\b{kw}\b", s):
            f.append(name)
    if re.search(r"\bcount\s*\(", s):
        f.append("count")
    if re.search(r"\b(sum|avg|min|max)\s*\(", s):
        f.append("aggregate_fn")
    if re.search(r"\b(or|not|in|between)\b", s.split(" from ", 1)[-1]):
        f.append("filter_complex")
    return f


def query_class(feats: list[str]) -> str:
    """One coarse class per example, same names as combine_datasets.py where they overlap."""
    for c in ("set_op", "nested", "multi_join", "join", "having", "group"):
        if c in feats:
            return c
    if "sort" in feats or "limit" in feats:
        return "sort_limit"
    if "count" in feats or "aggregate_fn" in feats:
        return "count"
    if "filter_complex" in feats:
        return "filter_complex"
    return "filter_simple"


# ---------------------------------------------------------------- build
def check_inputs(raw: str, sqlv1: str) -> dict:
    man = json.load(open(os.path.join(sqlv1, "manifest.json"), encoding="utf-8"))
    bad = []
    for name in RAW_FILES:
        got = sha256_file(os.path.join(raw, name))
        if got != man["inputs_sha256"][name]:
            bad.append(f"raw {name}: {got} != {man['inputs_sha256'][name]}")
    for _, name in SPLITS:
        got = sha256_file(os.path.join(sqlv1, name))
        if got != man["files_sha256"][name]:
            bad.append(f"spider-sql-v1 {name}: {got} != {man['files_sha256'][name]}")
    if bad:
        raise SystemExit("input hash mismatch:\n  " + "\n  ".join(bad))
    return man


def process(rec: dict, split: str, raw: str, timeout_s: float) -> tuple[dict | None, str]:
    db, sql = rec["db_id"], rec["sql"]
    sqlite_path = os.path.join(raw, "database", db, f"{db}.sqlite")
    if rec.get("tier") != "exec_ok":
        return None, "weak_verification"
    try:
        cols, rows = run_sql(sqlite_path, sql, timeout_s)
    except Exception:  # noqa: BLE001
        return None, "exec_error"
    ordered_h, multiset_h = result_hashes(rows)
    ex = rec["execution"]
    if ordered_h != ex["ordered_sha256"] or multiset_h != ex["multiset_sha256"]:
        return None, "result_mismatch"
    if not rows or all(v is None for r in rows for v in r):
        return None, "weak_result"
    if len(rows) == 1 and len(rows[0]) == 1 and rows[0][0] == 0:
        return None, "zero_only_result"
    if len(rows) > MAX_ROWS:
        return None, "too_many_rows"
    ties = tie_analysis(sqlite_path, sql, rows, timeout_s)
    if ties["order_by"] and ties["status"] != "ok":
        return None, "order_unverifiable"
    if ties["tie_at_limit"]:
        return None, "limit_tie"
    question = clean_question(rec["question"])
    feats = sql_features(sql)
    out = {
        "id": rec["id"], "split": split, "db_id": db, "question": question,
        "question_raw": rec["question"], "sql": sql, "hardness": rec.get("hardness"),
        "query_class": query_class(feats), "features": feats,
        "group": f"{db}:{hashlib.sha1(sql_key(sql).encode()).hexdigest()[:10]}",
        "ordered": ties["order_by"], "limit": ties["limit"], "tie_blocks": ties["tie_blocks"],
        "columns": cols, "n_rows": len(rows), "expected_output": [list(r) for r in rows],
        "result_sha256": ordered_h if ties["order_by"] else multiset_h,
        "source": rec["source"],
    }
    return out, "kept"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/spider1/spider_data")
    ap.add_argument("--sqlv1", default=os.path.expanduser(
        "~/Downloads/spider_sql_v1/data/processed/spider_sql_v1"))
    ap.add_argument("--out", default="data/spider_pool")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    check_inputs(args.raw, args.sqlv1)
    kept = defaultdict(list)
    rejected = []
    funnel = defaultdict(Counter)
    seen_q = set()
    used_dbs = set()
    for split, fname in SPLITS:
        with open(os.path.join(args.sqlv1, fname), encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        for rec in recs:
            funnel[split]["input"] += 1
            ex, reason = process(rec, split, args.raw, args.timeout)
            if ex is not None:
                key = (ex["db_id"], question_key(ex["question"]))
                if key in seen_q:
                    ex, reason = None, "duplicate_question"
                else:
                    seen_q.add(key)
            funnel[split][reason] += 1
            if ex is None:
                rejected.append({"id": rec["id"], "split": split, "db_id": rec["db_id"],
                                 "question": rec["question"], "sql": rec["sql"], "reject_reason": reason})
                continue
            kept[split].append(ex)
            used_dbs.add(ex["db_id"])
        print(f"{split}: {funnel[split]['input']} -> {funnel[split]['kept']}", flush=True)

    os.makedirs(args.out, exist_ok=True)

    def write_jsonl(name, rows):
        path = os.path.join(args.out, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        return sha256_file(path)

    files = {f"{s}.jsonl": write_jsonl(f"{s}.jsonl", kept[s]) for s, _ in SPLITS}
    files["rejected.jsonl"] = write_jsonl("rejected.jsonl", rejected)

    def split_stats(rows):
        return {"examples": len(rows), "databases": len({r["db_id"] for r in rows}),
                "distinct_queries": len({r["group"] for r in rows}),
                "query_class": dict(sorted(Counter(r["query_class"] for r in rows).items())),
                "hardness": dict(sorted(Counter(r["hardness"] for r in rows).items()))}

    stats = {s: split_stats(kept[s]) for s, _ in SPLITS}
    raw_man = json.load(open(os.path.join(args.sqlv1, "manifest.json"), encoding="utf-8"))
    manifest = {
        "version": "spider-pool-v1",
        "command": "python scripts/build_spider_pool.py --raw <spider_data> --sqlv1 <spider_sql_v1> "
                   f"--out <out> --timeout {args.timeout:g}",
        "code_sha256": sha256_file(__file__),
        "inputs": {
            "spider_data.zip": "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b",
            "raw_sha256": raw_man["inputs_sha256"],
            "spider_sql_v1_sha256": {name: raw_man["files_sha256"][name] for _, name in SPLITS},
            "sqlite_sha256": {db: sha256_file(os.path.join(args.raw, "database", db, f"{db}.sqlite"))
                              for db in sorted(used_dbs)},
        },
        "parameters": {"timeout_s": args.timeout, "max_rows": MAX_ROWS},
        # float sums differ between SQLite versions, so results (and R3) depend on it
        "sqlite_version": sqlite3.sqlite_version,
        "funnel": {s: dict(sorted(funnel[s].items())) for s, _ in SPLITS},
        "splits": stats,
        "files_sha256": files,
        "license": "CC BY-SA 4.0 (Spider)",
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"funnel": manifest["funnel"], "splits": stats}, indent=1))


if __name__ == "__main__":
    main()
