"""Build the expanded Spider 1.0 NL -> MQL benchmark from the full 493-item eligible pool.

No gold MQL is authored or required: the reference answer of every item is the result of
its official gold SQL on the official SQLite database, and a model's MQL is scored by
executing it on Atlas and comparing results (run_eval.score_output / exec_match).

Every pool item gets exactly one status (data/interim/spider1/expanded/status.jsonl):
  * sample items (300): the v1 curation stands -- the 279 v1-validated items are included
    (their gold rows must be byte-identical to data/processed/spider1_nl_mql.jsonl), and the
    21 v1 exclusions keep their written reasons (data/interim/spider1/adjudication.json)
  * the 193 other pool items: decision from expanded/audit_new_items.json (NL/SQL audit)
then two data gates for every candidate:
  * expected result: gold SQL re-executed now (SQLite, read-only) must reproduce the pool's
    frozen gold rows
  * parity (expanded/parity.json, scripts/spider1/parity_check.py): every table of the db_id
    must match its Atlas collection. A db_id whose only mismatch is strings that are not
    valid UTF-8 in SQLite (BSON can only hold the U+FFFD-replaced text) is admitted per
    item: the SQL is re-run on an in-memory copy where those strings are replaced exactly
    as stored on Atlas, and the item is kept only if its result is unchanged.

Outputs (the 279-item v1 release is never touched):
  data/processed/spider1_nl_mql_expanded.jsonl
  data/interim/spider1/expanded/status.jsonl
  data/interim/spider1/expanded/dataset_summary.json

Usage: python scripts/spider1/build_expanded.py
"""
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402

EXP = K.INTERIM / "expanded"
OUT = K.ROOT / "data/processed/spider1_nl_mql_expanded.jsonl"
V1 = K.ROOT / "data/processed/spider1_nl_mql.jsonl"


def rows_sha(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def utf8_only_mismatch(dbrep):
    """True if every non-matching table differs only by invalid-UTF-8 strings."""
    bad = [t for t in dbrep["tables"].values() if not t["match"]]
    return bool(bad) and not dbrep["atlas_collections_without_table"] and all(
        t.get("sqlite_rows") == t.get("atlas_docs") and not t.get("rows_only_in_sqlite")
        and not t.get("rows_only_in_atlas") and not t.get("extra_fields")
        and not t.get("missing_fields") and t.get("utf8_replaced_strings") for t in bad)


def atlas_form_copy(db_id):
    """In-memory copy of the dev DB with invalid-UTF-8 TEXT replaced as stored on Atlas."""
    src = sqlite3.connect("file:" + K.db_path(db_id).as_posix() + "?mode=ro", uri=True)
    mem = sqlite3.connect(":memory:")
    src.backup(mem)
    src.close()
    mem.text_factory = bytes
    changed = 0
    for table in K.tables(db_id):
        for col, _ in K.table_schema(db_id, table):
            for rowid, v in mem.execute(f'SELECT rowid, "{col}" FROM "{table}" WHERE typeof("{col}")=\'text\'').fetchall():
                try:
                    v.decode("utf-8")
                except UnicodeDecodeError:
                    mem.execute(f'UPDATE "{table}" SET "{col}"=? WHERE rowid=?',
                                (v.decode("utf-8", errors="replace"), rowid))
                    changed += 1
    mem.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return mem, changed


def main():
    pool = K.read_jsonl(K.INTERIM / "pool.jsonl")
    sample_ids = {r["id"] for r in K.read_jsonl(K.INTERIM / "sample.jsonl")}
    v1 = {r["id"]: r for r in K.read_jsonl(V1)}
    v1_status = {r["id"]: r for r in K.read_jsonl(K.INTERIM / "status.jsonl")}
    v1_adj = K.load_json(K.INTERIM / "adjudication.json")
    audit = K.load_json(EXP / "audit_new_items.json")["items"]
    parity = K.load_json(EXP / "parity.json")
    assert not set(audit) & sample_ids, "audit file must only cover non-sample pool items"
    assert set(audit) | sample_ids == {r["id"] for r in pool}, "every pool item needs a decision"

    mem_dbs = {}
    status, dataset = [], []
    for item in pool:
        iid, db = item["id"], item["db_id"]
        st = {"id": iid, "db_id": db, "hardness": item["hardness"], "in_v1_sample": iid in sample_ids}
        flags = []
        # 1. curation (NL/SQL audit)
        if iid in sample_ids:
            if iid not in v1:
                vs = v1_status[iid]
                st.update(status="excluded", stage="v1_adjudication",
                          kind=(vs.get("issue") or {}).get("kind"),
                          reason=(v1_adj.get(iid) or {}).get("reason") or vs.get("reason"))
                status.append(st)
                continue
            flags = list(v1[iid]["flags"])
            st["audit"] = "v1 (gold MQL validated on dev + test-suite variants, curator adjudication)"
        else:
            a = audit[iid]
            if a["decision"] != "include":
                st.update(status="excluded", stage="nl_sql_audit", kind=a["kind"], reason=a["reason"])
                status.append(st)
                continue
            if a["kind"]:
                flags = [a["kind"]]
                st["flag_reason"] = a["reason"]
            st["audit"] = "expanded NL/SQL audit"
        # 2. expected result: re-execute the gold SQL now
        cols, rows = K.run_sql(db, item["gold_sql"])
        rows = [list(r) for r in rows]
        if rows != item["gold_rows"]:
            st.update(status="excluded", stage="expected_result",
                      reason="gold SQL no longer reproduces the frozen pool result")
            status.append(st)
            continue
        if iid in v1:
            for k in ("gold_rows", "gold_columns", "order_by", "limit", "tie_blocks", "question", "gold_sql"):
                assert v1[iid][k] == item[k], (iid, k)
        # 3. SQLite <-> Atlas data parity
        dbrep = parity["databases"].get(db)
        if dbrep is None:
            st.update(status="excluded", stage="parity", reason="db_id not parity-checked")
            status.append(st)
            continue
        if dbrep["parity"]:
            st["parity"] = "all tables: type-tagged row multiset identical"
        elif utf8_only_mismatch(dbrep):
            if db not in mem_dbs:
                mem_dbs[db] = atlas_form_copy(db)
            mem, changed = mem_dbs[db]
            alt = [list(r) for r in mem.execute(item["gold_sql"]).fetchall()]
            if alt != rows:
                st.update(status="excluded", stage="parity",
                          reason=f"result depends on {changed} invalid-UTF-8 strings that Atlas stores replaced")
                status.append(st)
                continue
            st["parity"] = (f"identical except {changed} invalid-UTF-8 strings; result unchanged when "
                            "they are replaced as on Atlas")
        else:
            st.update(status="excluded", stage="parity", reason="SQLite and Atlas data differ")
            status.append(st)
            continue
        st["status"] = "included"
        st["flags"] = flags
        status.append(st)
        dataset.append({
            "id": iid, "source": "spider1_dev", "dev_index": item["dev_index"], "db_id": db,
            "question": item["question"], "hardness": item["hardness"], "gold_sql": item["gold_sql"],
            "gold_columns": cols, "gold_rows": rows, "n_rows": len(rows),
            "order_by": item["order_by"], "limit": item["limit"], "tie_blocks": item["tie_blocks"],
            "expected_rows_sha256": rows_sha(rows), "flags": flags,
            "in_v1_benchmark": iid in v1, "parity": st["parity"],
        })
    K.write_jsonl(OUT, dataset)
    K.write_jsonl(EXP / "status.jsonl", status)
    v1_ids = set(v1)
    assert v1_ids <= {d["id"] for d in dataset}, "a v1 item dropped out of the expanded set"
    excl = [s for s in status if s["status"] != "included"]
    summary = {
        "eligible_pool": len(pool),
        "included": len(dataset),
        "excluded": len(excl),
        "excluded_by_stage": dict(Counter(s["stage"] for s in excl)),
        "excluded_by_kind": dict(Counter(f"{s['stage']}:{s.get('kind')}" for s in excl)),
        "from_v1_279": sum(d["in_v1_benchmark"] for d in dataset),
        "new_items": sum(not d["in_v1_benchmark"] for d in dataset),
        "by_hardness": {h: sum(d["hardness"] == h for d in dataset) for h in ("easy", "medium", "hard", "extra")},
        "by_db": dict(sorted(Counter(d["db_id"] for d in dataset).items())),
        "flagged_kept": dict(Counter(f for d in dataset for f in d["flags"])),
        "parity_evidence": dict(Counter(d["parity"] for d in dataset)),
        "unique_questions": len({(d["db_id"], d["question"]) for d in dataset}),
        "dataset": str(OUT.relative_to(K.ROOT)).replace("\\", "/"),
        "dataset_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "pool_sha256": hashlib.sha256((K.INTERIM / "pool.jsonl").read_bytes()).hexdigest(),
    }
    (EXP / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
