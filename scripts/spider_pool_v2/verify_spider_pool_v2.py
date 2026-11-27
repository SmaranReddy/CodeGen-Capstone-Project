"""Independent verification of data/processed/spider_pool_v2 (read-only).
Re-derives every expectation from the source ZIP and the audit evidence without importing the builder.
Protected sets are checked by ID only: dev.jsonl / rejected.jsonl / test_full.jsonl are parsed in memory and only
`id` (and `split`) is kept; no question, SQL or output field is read into any result.
Usage: python -I -B scripts/spider_pool_v2/verify_spider_pool_v2.py [--dir data/processed/spider_pool_v2] [--report out.json]"""
import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(ROOT / "data/processed/spider_pool_v2"))
ap.add_argument("--report", default=None)
a = ap.parse_args()
D = Path(a.dir)
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
checks = {}


def ok(name, cond, detail=None):
    checks[name] = {"pass": bool(cond), **({"detail": detail} if detail is not None else {})}


man = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
expected_files = {"train.jsonl", "val_internal.jsonl", "quarantine.jsonl", "exclusions.jsonl", "manifest.json"}
ok("exact_file_set", {p.name for p in D.iterdir()} == expected_files, sorted(p.name for p in D.iterdir()))

# ---- source
zb = (ROOT / "spider_pool.zip").read_bytes()
ok("source_zip_hash", sha(zb) == man["source"]["zip"]["sha256"] == "31848241861988ebb0619e1802c38fb5d90cfb62f818a65a53cf601340888a6b")
z = zipfile.ZipFile(ROOT / "spider_pool.zip")
zman = json.loads(z.read("spider_pool/manifest.json"))
src = {s: [json.loads(x) for x in z.read(f"spider_pool/{s}.jsonl").decode("utf-8").splitlines()] for s in ("train", "val_internal")}
ok("source_counts_match_zip_manifest", {s: len(v) for s, v in src.items()} == {s: zman["splits"][s]["examples"] for s in src}, {s: len(v) for s, v in src.items()})
S = {r["id"]: r for v in src.values() for r in v}

# ---- outputs
def load(name):
    b = (D / name).read_bytes()
    return b, [json.loads(x) for x in b.decode("utf-8").splitlines()]


out = {n: load(n) for n in ("train.jsonl", "val_internal.jsonl", "quarantine.jsonl", "exclusions.jsonl")}
ok("output_hashes_match_manifest", all(sha(out[n][0]) == man["files"][n]["sha256"] and len(out[n][1]) == man["files"][n]["records"] for n in out))
ok("lf_only_and_utf8", all(b"\r" not in out[n][0] and out[n][0].endswith(b"\n") for n in out))
tr, va, qu, ex = (out[n][1] for n in ("train.jsonl", "val_internal.jsonl", "quarantine.jsonl", "exclusions.jsonl"))

# ---- expected dispositions from evidence (independent re-derivation)
rec = [json.loads(x) for x in (ROOT / "outputs/spider_pool_audit_v1/records/recommended_exclusions.jsonl").read_text(encoding="utf-8").splitlines()]
ok("evidence_tier_list_hash", sha((ROOT / "outputs/spider_pool_audit_v1/records/recommended_exclusions.jsonl").read_bytes()) == man["exclusion_provenance"]["tier_list"]["sha256"])
want = {}
for x in rec:
    want[x["id"]] = ("excluded" if x["tier"][0] in "AB" else "quarantined", x["tier"], x["reason"])
recheck = json.loads((ROOT / "outputs/spider_pool_repro_audit_v1/stage9_empty_table_recheck.json").read_text(encoding="utf-8"))
untiered = [d["id"] for d in recheck["missed_detail"] if d["tier_now"] is None]
ok("added_quarantine_is_exactly_02208", untiered == ["spider_train_02208"], untiered)
for i in untiered:
    want[i] = ("quarantined", None, None)

ids = lambda rows: [r["id"] for r in rows]  # noqa: E731
all_out = ids(tr) + ids(va) + ids(qu)
ok("every_output_id_exists_in_source_train_val", set(all_out) <= set(S), sorted(set(all_out) - set(S))[:10])
ok("unique_ids_train", len(set(ids(tr))) == len(tr))
ok("unique_ids_val", len(set(ids(va))) == len(va))
ok("unique_ids_quarantine", len(set(ids(qu))) == len(qu))
ok("unique_ids_exclusions", len(set(ids(ex))) == len(ex))
ok("no_id_in_two_output_record_files", len(set(all_out)) == len(all_out))
ok("train_val_id_overlap_zero", not set(ids(tr)) & set(ids(va)))
ok("train_val_db_overlap_zero", not {r["db_id"] for r in tr} & {r["db_id"] for r in va}, sorted({r["db_id"] for r in tr} & {r["db_id"] for r in va}))
ok("train_val_group_overlap_zero", not {r["group"] for r in tr} & {r["group"] for r in va})
ok("split_fields_consistent", all(r["split"] == "train" for r in tr) and all(r["split"] == "val_internal" for r in va))

# disposition of every source record
disp = {}
for r in tr + va:
    disp[r["id"]] = "retained"
for r in qu:
    disp[r["id"]] = "quarantined"
exd = {e["id"]: e for e in ex}
for e in ex:
    if e["disposition"] == "excluded":
        disp[e["id"]] = "excluded"
ok("every_source_record_has_exactly_one_disposition", set(disp) == set(S) and len(all_out) + sum(e["disposition"] == "excluded" for e in ex) == len(S))
ok("retained_are_exactly_untiered", {i for i, d in disp.items() if d == "retained"} == set(S) - set(want))
ok("excluded_set_exact", {i for i, d in disp.items() if d == "excluded"} == {i for i, w in want.items() if w[0] == "excluded"})
ok("quarantine_set_exact", {i for i, d in disp.items() if d == "quarantined"} == {i for i, w in want.items() if w[0] == "quarantined"})
ok("exclusions_file_covers_all_removed", set(exd) == set(want) and all(exd[i]["disposition"] == want[i][0] for i in want))
ok("tier_and_reason_verbatim", all(i in exd and exd[i]["tier"] == want[i][1] and exd[i]["reason"] == want[i][2] for i in want if want[i][1] is not None))
ok("every_removed_has_nonempty_reason_and_evidence", all(e.get("reason") and e.get("evidence") and e.get("tier") for e in ex))
ok("exclusion_source_line_hash_matches_zip", all(
    e["source"]["line_sha256"] == sha(z.read(e["source"]["zip_member"]).decode("utf-8").splitlines()[e["source"]["line"] - 1].encode("utf-8")) for e in ex))

# content preservation: parsed equality with source, all fields
ok("retained_and_quarantined_content_identical_to_source", all(r == S.get(r["id"]) for r in tr + va + qu))
src_lines = {json.loads(x)["id"]: x for s in ("train", "val_internal") for x in z.read(f"spider_pool/{s}.jsonl").decode("utf-8").splitlines()}
ok("retained_and_quarantined_lines_byte_identical_to_source", all(l == src_lines.get(json.loads(l)["id"]) for n in ("train.jsonl", "val_internal.jsonl", "quarantine.jsonl") for l in out[n][0].decode("utf-8").splitlines()))
ok("source_order_preserved", ids(tr) == [r["id"] for r in src["train"] if disp[r["id"]] == "retained"] and ids(va) == [r["id"] for r in src["val_internal"] if disp[r["id"]] == "retained"])
ok("no_duplicate_records_introduced", len({json.dumps(r, sort_keys=True) for r in tr + va}) == len(tr) + len(va))

# group consistency: no excluded/quarantined query left with a retained paraphrase
grp_disp = {}
for i, d in disp.items():
    if i in S:
        grp_disp.setdefault(S[i]["group"], set()).add(d)
mixed = {g: sorted(v) for g, v in grp_disp.items() if len(v) > 1}
ok("no_query_group_split_across_dispositions", not mixed, mixed)

# ---- protected sets, ID only
dev_ids = set()
for name in ("spider_pool/dev.jsonl", "spider_pool/rejected.jsonl"):
    for x in z.read(name).decode("utf-8").splitlines():
        r = json.loads(x)
        if r["split"] == "dev":
            dev_ids.add(r["id"])
        del r
test_ids = set()
for x in (ROOT / "data/processed/spider_sql_v1/test_full.jsonl").read_text(encoding="utf-8").splitlines():
    test_ids.add(json.loads(x)["id"])
h65 = {json.loads(x)["mapped_id"] for x in (ROOT / "outputs/spider_pool_audit_v1/records/heldout65_overlap_ids.jsonl").read_text(encoding="utf-8").splitlines()}
everything = set(all_out) | set(exd)
ok("protected_sizes", (len(dev_ids), len(test_ids), len(h65)) == (1034, 2147, 65), [len(dev_ids), len(test_ids), len(h65)])
ok("no_spider_dev_ids", not everything & dev_ids)
ok("no_spider_test_ids", not everything & test_ids)
ok("no_heldout65_ids", not everything & h65)
ok("only_spider_train_prefix", all(i.startswith("spider_train_") for i in everything))
dev_dbs = {json.loads(x)["db_id"] for x in z.read("spider_pool/dev.jsonl").decode("utf-8").splitlines()}
ok("no_dev_database_in_output", not {r["db_id"] for r in tr + va + qu} & dev_dbs)

# ---- counts reconcile
cnt = {s: {"source": len(src[s]), "excluded": sum(1 for e in ex if e["split"] == s and e["disposition"] == "excluded"),
           "quarantined": sum(1 for r in qu if r["split"] == s), "retained": len(tr if s == "train" else va)} for s in src}
ok("counts_reconcile", all(c["source"] == c["excluded"] + c["quarantined"] + c["retained"] for c in cnt.values()) and cnt == man["counts"], cnt)
tiers = Counter((e["split"], e["tier"]) for e in ex)
ok("tier_counts_match_audit", {k: v for k, v in tiers.items()} == {("train", "A_confirmed_defect"): 53, ("val_internal", "A_confirmed_defect"): 4,
                                ("train", "B_engine_or_row_choice_dependent"): 31, ("val_internal", "B_engine_or_row_choice_dependent"): 6,
                                ("train", "C_review_before_use"): 42, ("train", "C_review_added_by_reproducibility_audit"): 1}, {f"{a}|{b}": v for (a, b), v in tiers.items()})
summary = {"all_pass": all(c["pass"] for c in checks.values()), "n_checks": len(checks), "counts": cnt,
           "dbs": {"train": len({r["db_id"] for r in tr}), "val_internal": len({r["db_id"] for r in va}), "quarantine": len({r["db_id"] for r in qu})},
           "groups": {"train": len({r["group"] for r in tr}), "val_internal": len({r["group"] for r in va})},
           "hardness": {"train": dict(sorted(Counter(r["hardness"] for r in tr).items())), "val_internal": dict(sorted(Counter(r["hardness"] for r in va).items()))},
           "checks": checks}
txt = json.dumps(summary, indent=1, ensure_ascii=False)
if a.report:
    Path(a.report).write_text(txt + "\n", encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=1))
print("FAILED:", [k for k, v in checks.items() if not v["pass"]])
sys.exit(0 if summary["all_pass"] else 1)
