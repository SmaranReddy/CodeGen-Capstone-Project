"""Independent human-review package for sql2mql-v2 labels (pilot gate P2 / V7).

    python -B scripts/finetune_v2/review_package.py dryrun
        DRY RUN on DEV items (the 200 iteration-1 diagnosis labels only): shows the reviewer the
        format. Not the pilot, not a gate measurement. -> outputs/sql2mql_v2/review_dryrun/
    python -B scripts/finetune_v2/review_package.py pilot --records outputs/sql2mql_v2/pilot/records.jsonl
        ONLY after Phase 2 has run (needs the pilot records). -> outputs/sql2mql_v2/review_pilot/

Selection rule (frozen here, pilot gate P2): accepted labels only; target min(120, accepted);
stratified by round3_stats.stratum (J2 / OL / J1 / G) with proportional allocation by largest
remainder and >= 1 per non-empty stratum; within a stratum a seeded random sample
(seed 20261101, ids sorted first). Order in the sheet is shuffled (seed) so strata are mixed.

For every item the package holds: question, original SQL, converted MQL, the referenced tables'
CREATE TABLE text and row counts, the SQLite result of the verbatim SQL and the MongoDB result
of the MQL (local 8.0.34, real database, first 15 rows + row counts), the converter decision
(features) and the verifier decision (status + check summary).
The reviewer columns of review_sheet.csv are left EMPTY: nothing here fills a judgement.
Local MongoDB only (v2_common.local_client).
"""
import argparse
import csv
import json
import random
import sqlite3
import sys
from pathlib import Path

from sqlglot import exp, parse_one

import v2_common as C
import round3_stats as S

SEED = 20261101
TARGET = 120
PREVIEW = 15
REVIEW_COLUMNS = [
    "reviewer_id",
    "nl_sql_agree (Y/N/U)",
    "sql_mql_equivalent (Y/N/U)",
    "mql_answers_question (Y/N/U)",
    "discrepancy_type",
    "semantic_discrepancy",
    "comments",
    "confidence (1-5)",
    "minutes_spent",
]
DISCREPANCY_TYPES = ["none", "wrong_filter", "wrong_join", "missing_or_extra_rows", "null_handling",
                     "aggregation", "grouping", "ordering_or_limit", "distinct", "projection_or_columns",
                     "type_or_case_semantics", "gold_sql_wrong_for_question", "other"]


def select(accepted, target=TARGET, seed=SEED):
    """Frozen P2 selection: -> list of ids."""
    n = min(target, len(accepted))
    by = {h: sorted(r["id"] for r in accepted if S.stratum(r["sql"]) == h) for h in S.STRATA}
    nonempty = [h for h in S.STRATA if by[h]]
    quota = {h: max(1, n * len(by[h]) / len(accepted)) for h in nonempty}
    alloc = {h: min(len(by[h]), int(quota[h])) for h in nonempty}
    rest = n - sum(alloc.values())
    for h in sorted(nonempty, key=lambda h: -(quota[h] - int(quota[h]))):
        if rest <= 0:
            break
        if alloc[h] < len(by[h]):
            alloc[h] += 1
            rest -= 1
    rng = random.Random(seed)
    picked = [i for h in nonempty for i in rng.sample(by[h], alloc[h])]
    rng.shuffle(picked)
    return picked


def schema_context(path, sql):
    tabs = {t.name.lower() for t in parse_one(sql, read="sqlite").find_all(exp.Table)}
    con = C.sqlite_ro(path)
    try:
        out = []
        for name, ddl in con.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"):
            if name.lower() in tabs:
                n = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
                out.append({"table": name, "rows": n, "ddl": ddl})
        return out
    finally:
        con.close()


def sqlite_result(path, sql):
    con = C.sqlite_ro(path)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = [list(r) for r in cur.fetchall()]
        return {"columns": cols, "row_count": len(rows), "rows": rows[:PREVIEW]}
    except sqlite3.Error as e:
        return {"error": str(e)}
    finally:
        con.close()


def mongo_result(client, mql):
    cols = [k for k in mql["pipeline"][-1]["$project"] if k != "_id"]
    docs = list(client[C.REAL_DB][mql["collection"]].aggregate(mql["pipeline"], maxTimeMS=C.TIMEOUT_MS))
    return {"columns": cols, "row_count": len(docs), "rows": [[d.get(c, "<absent>") for c in cols]
                                                               for d in docs[:PREVIEW]]}


def verifier_summary(v):
    ch = v.get("checks", {})
    fam = ch.get("fixtures", {})
    used = [k for k, x in fam.items() if not str(x).startswith("skipped")]
    return {"verifier": v.get("verifier"), "status": v.get("status"), "detail": v.get("detail"),
            "l1_explicit_null": ch.get("l1_explicit_null"), "l1_missing": ch.get("l1_missing"),
            "v6": ch.get("v6_determinism"), "real_rows": ch.get("real_rows"),
            "fixture_runs_passed": len(used), "fixture_runs_skipped": len(fam) - len(used),
            "extra_checks": {k: ch[k] for k in ch if k.startswith("v3_")}}


def fmt_rows(res):
    if "error" in res:
        return f"ERROR {res['error']}"
    head = " | ".join(map(str, res["columns"]))
    body = "\n".join(" | ".join(json.dumps(x, ensure_ascii=False, default=str) for x in r) for r in res["rows"])
    more = f"\n... ({res['row_count']} rows in total)" if res["row_count"] > len(res["rows"]) else ""
    return f"{head}\n{body}{more}" if res["rows"] else f"{head}\n(0 rows)"


def build(recs, out, banner, client, target=TARGET):
    acc = [r for r in recs if r.get("verification", {}).get("status") == "accepted"]
    ids = select(acc, target)
    byid = {r["id"]: r for r in acc}
    items = []
    for k, i in enumerate(ids, 1):
        r = byid[i]
        path = C.sqlite_path(r["db_id"])
        items.append({
            "review_id": f"R{k:03d}", "item_id": i, "db_id": r["db_id"], "stratum": S.stratum(r["sql"]),
            "hardness": r.get("hardness"), "question": r["question"], "sql": r["sql"], "mql": r["mql"],
            "schema": schema_context(path, r["sql"]),
            "sqlite_result": sqlite_result(path, r["sql"]),
            "mongo_result": mongo_result(client, r["mql"]),
            "converter": {"version": r.get("converter"), "decision": r.get("conversion"),
                          "features": r.get("features")},
            "verifier": verifier_summary(r["verification"]),
        })
    out.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(out / "review_items.jsonl", items)
    with (out / "review_sheet.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["package"] + ["review_id", "item_id", "db_id", "stratum", "hardness", "question", "sql",
                                  "mql", "sqlite_result_preview", "mongo_result_preview", "converter_decision",
                                  "verifier_decision"] + REVIEW_COLUMNS)
        for it in items:
            w.writerow([banner, it["review_id"], it["item_id"], it["db_id"], it["stratum"], it["hardness"],
                        it["question"], it["sql"], C.compact(it["mql"]), fmt_rows(it["sqlite_result"]),
                        fmt_rows(it["mongo_result"]),
                        f"{it['converter']['decision']} {it['converter']['features']}",
                        f"{it['verifier']['status']} ({it['verifier']['verifier']})"] + [""] * len(REVIEW_COLUMNS))
    lines = [f"# sql2mql-v2 human review packet — {banner}", "",
             "Record your judgements ONLY in review_sheet.csv (same review_id). Instructions: "
             "docs/sql2mql_v2_human_review.md.", ""]
    for it in items:
        lines += [f"## {it['review_id']} · {it['item_id']} · db `{it['db_id']}` · stratum {it['stratum']} · "
                  f"{it['hardness']}", "", f"**Question:** {it['question']}", "", "**Original SQL**", "",
                  "```sql", it["sql"], "```", "", "**Converted MQL**", "", "```json",
                  json.dumps(it["mql"], indent=1, ensure_ascii=False), "```", "", "**Schema (referenced tables)**", ""]
        for s in it["schema"]:
            lines += [f"- `{s['table']}` ({s['rows']} rows)", "", "```sql", s["ddl"], "```", ""]
        lines += ["**SQLite result (verbatim SQL)**", "", "```", fmt_rows(it["sqlite_result"]), "```", "",
                  "**MongoDB result (MQL, local 8.0.34, real data)**", "", "```", fmt_rows(it["mongo_result"]),
                  "```", "", f"**Converter:** {it['converter']['decision']}, features {it['converter']['features']}",
                  "", f"**Verifier:** {json.dumps(it['verifier'], ensure_ascii=False)}", "", "---", ""]
    (out / "review_packet.md").write_text("\n".join(lines), encoding="utf-8")
    meta = {"banner": banner, "items": len(items), "accepted_pool": len(acc),
            "selection": {"rule": "min(120, accepted), stratified J2/OL/J1/G proportional (largest remainder, "
                                  ">=1 per non-empty stratum), seeded", "seed": SEED},
            "by_stratum": {h: sum(it["stratum"] == h for it in items) for h in S.STRATA},
            "review_columns": REVIEW_COLUMNS, "discrepancy_types": DISCREPANCY_TYPES,
            "judgements_filled": 0, "sha256": {p.name: C.sha256_file(p) for p in
                                               (out / "review_items.jsonl", out / "review_sheet.csv",
                                                out / "review_packet.md")}}
    (out / "package_manifest.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("banner", "items", "accepted_pool", "by_stratum")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("dryrun", "pilot"))
    ap.add_argument("--records")
    a = ap.parse_args()
    client = C.local_client()
    C.check_server(client)
    if a.mode == "dryrun":
        diag = set(json.loads((C.OUT / "mutant_study" / "diagnosis_labels.json").read_text(encoding="utf-8"))["ids"])
        recs = [r for r in C.read_jsonl(C.OUT / "dev_iter2" / "records.jsonl") if r["id"] in diag]
        out = C.OUT / "review_dryrun"
        build(recs, out, "DRY RUN (24 DEV diagnosis items; NOT the pilot; not a gate measurement)", client, 24)
    else:
        if not a.records or not Path(a.records).exists():
            sys.exit("pilot mode needs the Phase-2 pilot records (the pilot has not been run)")
        out = C.OUT / "review_pilot"
        if (out / "review_sheet.csv").exists():
            sys.exit(f"{out} exists: refusing to overwrite a review package")
        fr = json.loads(C.FREEZE.read_text(encoding="utf-8"))
        recs = C.read_jsonl(Path(a.records))
        if {r["id"] for r in recs} != set(fr["pilot"]["ids"]):
            sys.exit("records are not exactly the 300 frozen pilot ids")
        build(recs, out, "PILOT P2 review (sql2mql-v2)", client)


if __name__ == "__main__":
    main()
