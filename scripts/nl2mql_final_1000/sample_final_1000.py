"""MQL-0015 deterministic stratified sampling of 1,000 Spider test_full questions (CPU only).

The method is fixed in outputs/nl2mql_final_1000/sampling/sampling_config.json, which was written before the
first draw. This script implements it exactly:
  * cell-level largest-remainder allocation over db_id x hardness;
  * within each cell, implicit structure stratification plus a systematic sample;
  * randomness from sha256 only.

Gold SQL is read for metadata labels only (hardness is a source field; structure labels come from the frozen MQL-0013
patterns.sql_patterns). Nothing here reads a model prediction or an accuracy.

  python -I -B scripts/nl2mql_final_1000/sample_final_1000.py draw   [--out DIR]   # writes the sampling artifacts
  python -I -B scripts/nl2mql_final_1000/sample_final_1000.py ids    [--out DIR]   # in-memory draw, prints ids sha only
"""
import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/nl2mql_final_1000/sampling"
CONFIG = DEFAULT_OUT / "sampling_config.json"
PATTERNS_PY = ROOT / "kaggle/nl2mql_mql0013_test_full/src/patterns.py"
AUDIT = DEFAULT_OUT / "universe_overlap_audit.json"
HARDNESS = ("easy", "medium", "hard", "extra")
CLASSES = ("set_operation", "anti_join", "nested", "join", "aggregation", "simple")
REPORT_LABELS = ("join", "nested", "set_operation", "intersect", "union", "except", "anti_join", "anti_join_not_in_or_exists",
                 "group_by", "having", "aggregate", "count", "count_distinct", "distinct", "order_by", "limit", "where", "like")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def need(cond, msg):
    if not cond:
        raise SystemExit(f"STOP: {msg}")


def load_patterns():
    spec = importlib.util.spec_from_file_location("mql0013_patterns_frozen", PATTERNS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def labels(sql, P):
    s = set(P.sql_patterns(sql)) - {"select", "text_number"}
    up = " " + re.sub(r"\s+", " ", sql).upper() + " "
    if re.search(r"\bEXCEPT\b", up):
        s.add("except")
    if re.search(r"\bNOT IN\b|\bNOT EXISTS\b", up):
        s.add("anti_join_not_in_or_exists")
    if s & {"intersect", "union", "except"}:
        s.add("set_operation")
    return s


def primary_class(s):
    if "set_operation" in s:
        return "set_operation"
    if "anti_join_not_in_or_exists" in s:
        return "anti_join"
    if "nested" in s:
        return "nested"
    if "join" in s:
        return "join"
    if s & {"group_by", "having", "aggregate", "count"}:
        return "aggregation"
    return "simple"


def draw(cfg):
    seed = cfg["seed"]
    src = ROOT / cfg["source"]["path"]
    need(sha_file(src) == cfg["source"]["sha256"], "test_full sha256 differs from the registered source")
    recs = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    need(len(recs) == cfg["source"]["n_items"] == 2147 and len({r["id"] for r in recs}) == 2147, "2,147 unique ids")
    need(len({r["db_id"] for r in recs}) == 40, "40 databases")
    need({r["hardness"] for r in recs} <= set(HARDNESS), "hardness values")
    P = load_patterns()
    meta = {}
    for r in recs:
        s = labels(r["sql"], P)
        meta[r["id"]] = {"id": r["id"], "db_id": r["db_id"], "hardness": r["hardness"], "structure_labels": sorted(s),
                         "primary_class": primary_class(s), "flags": sorted(r["flags"]), "source_index": r["source"]["index"]}
    cells = defaultdict(list)
    for r in recs:
        cells[(r["db_id"], r["hardness"])].append(r["id"])
    N, n = len(recs), cfg["sample_size"]
    keys = sorted(cells)
    quota = {k: len(cells[k]) * n / N for k in keys}
    alloc = {k: math.floor(quota[k]) for k in keys}
    left = n - sum(alloc.values())
    order = sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), h(f"{seed}|alloc|{k[0]}|{k[1]}")))
    for k in order[:left]:
        need(quota[k] - alloc[k] > 0, "largest-remainder unit given to an integral cell")
        alloc[k] += 1
    need(sum(alloc.values()) == n, "allocation total")
    need(all(0 <= alloc[k] <= len(cells[k]) for k in keys), "allocation within cell sizes")
    per_db = Counter()
    for k in keys:
        per_db[k[0]] += alloc[k]
    need(len(per_db) == 40 and min(per_db.values()) >= 1, "database minimum guard")
    selected = []
    for k in keys:
        a, ids = alloc[k], cells[k]
        if a == 0:
            continue
        ordered = sorted(ids, key=lambda i: (CLASSES.index(meta[i]["primary_class"]), h(f"{seed}|item|{i}")))
        u = int(h(f"{seed}|start|{k[0]}|{k[1]}")[:16], 16) / 16 ** 16
        pos = [math.floor((u + i) * len(ids) / a) for i in range(a)]
        need(len(set(pos)) == a and max(pos) < len(ids), f"systematic positions {k}")
        selected += [ordered[p] for p in pos]
    need(len(selected) == n == len(set(selected)), "1,000 unique selections")
    src_order = {r["id"]: idx for idx, r in enumerate(recs)}
    selected.sort(key=src_order.get)
    unused = [r["id"] for r in recs if r["id"] not in set(selected)]
    need(len(unused) == N - n and not (set(unused) & set(selected)), "complement")
    return recs, meta, cells, quota, alloc, selected, unused


def dist(items, meta, key):
    c = Counter()
    for i in items:
        v = meta[i][key]
        for x in (v if isinstance(v, list) else [v]):
            c[x] += 1
    return c


def write_artifacts(out, cfg, recs, meta, cells, quota, alloc, selected, unused):
    out.mkdir(parents=True, exist_ok=True)
    ids_txt = "".join(i + "\n" for i in selected)
    (out / "sample_ids.txt").write_text(ids_txt, encoding="utf-8", newline="\n")
    (out / "unused_ids.txt").write_text("".join(i + "\n" for i in unused), encoding="utf-8", newline="\n")
    audit = json.loads(AUDIT.read_text(encoding="utf-8")) if AUDIT.exists() else None
    l5 = defaultdict(list)
    if audit:
        for p in audit["question_level"]["approved_corpus_v2_1"]["pairs"]:
            l5[p["eval_id"]].append({"corpus_id": p["match_id"], "jaccard": p["jaccard"], "visible_to": p["visible_to"]})
    all_ids = [r["id"] for r in recs]
    manifest = {
        "experiment": "MQL-0015",
        "source": cfg["source"],
        "seed": cfg["seed"],
        "sample_size": len(selected),
        "sample_ids_sha256": h(ids_txt),
        "unused_ids_sha256": h("".join(i + "\n" for i in unused)),
        "config_sha256": sha_file(CONFIG),
        "code_sha256": {"scripts/nl2mql_final_1000/sample_final_1000.py": sha_file(Path(__file__)),
                        "kaggle/nl2mql_mql0013_test_full/src/patterns.py": sha_file(PATTERNS_PY)},
        "gold_derived_fields_note": "hardness, structure_labels, primary_class and flags derive from gold SQL / gold execution; "
                                    "they are evaluation metadata and must NOT be copied into the generation package",
        "allocation": [{"db_id": k[0], "hardness": k[1], "n_source": len(cells[k]), "quota": round(quota[k], 6),
                        "allocated": alloc[k]} for k in sorted(cells)],
        "items": [dict(meta[i], l5_corpus_pairs=l5.get(i, [])) for i in selected],
    }
    (out / "sample_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "sampling_report.md").write_text(report(cfg, meta, cells, quota, alloc, selected, all_ids, l5, audit, manifest),
                                            encoding="utf-8")
    names = ["sampling_config.json", "sample_manifest.json", "sample_ids.txt", "unused_ids.txt", "sampling_report.md",
             "universe_overlap_audit.json"]
    lines = [f"{cfg['source']['sha256']}  ../../../{cfg['source']['path']}"]
    lines += [f"{sha_file(out / nm)}  {nm}" for nm in names if (out / nm).exists()]
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return manifest


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def report(cfg, meta, cells, quota, alloc, sel, all_ids, l5, audit, manifest):
    N, n = len(all_ids), len(sel)
    L = []
    w = L.append
    w("# MQL-0015 sampling report: 1,000 Spider test_full questions\n")
    w(f"Generated by `scripts/nl2mql_final_1000/sample_final_1000.py` from `sampling_config.json` (written before the draw). "
      f"Source `{cfg['source']['path']}` sha256 `{cfg['source']['sha256']}`, {N} questions / 40 databases. "
      f"Seed **{cfg['seed']}**. Sample ids sha256 (of `sample_ids.txt`) `{manifest['sample_ids_sha256']}`.\n")
    w("## Method\n")
    w("1. **Universe:** all 2,147 test_full items, with no filtering. The pre-draw overlap audit "
      "(`universe_overlap_audit.json`) found nothing that would change eligibility (see Independence below).")
    w("2. **Explicit strata:** database × Spider hardness, giving 152 non-empty cells.")
    w("3. **Allocation:** largest-remainder (Hamilton) proportional allocation at cell level.")
    w("   - Each cell's quota is q = n_cell × 1000 / 2147, and it first receives floor(q).")
    w("   - The remaining units go to the cells with the largest fractional remainders.")
    w("   - Ties are broken by `sha256('20261123|alloc|db|hardness')`, never by name.")
    w("   - Every database is guaranteed at least 1 item; the guard cannot bind, because the smallest database quota is 6.99.")
    w("4. **Within each cell:** items are ordered by structure priority class "
      "(set_operation > anti_join > nested > join > aggregation > simple), then by `sha256('20261123|item|id')`.")
    w("   - A systematic sample is taken along that order: positions floor((u + i)·n/a), with start "
      "u = `sha256('20261123|start|db|hardness')` mapped to [0,1).")
    w("   - This is implicit stratification: it keeps structure classes close to proportional inside every cell, "
      "without forcing any quota.")
    w("5. **Labels:** structure labels come from the gold SQL via the frozen MQL-0013 `patterns.sql_patterns`, plus "
      "`except` and `set_operation`. They are used for strata and reporting only.")
    w("6. **Randomness and inputs:** all randomness comes from sha256 of explicit strings. Nothing reads predictions, "
      "accuracies or model outputs. The draw never inspects questions for perceived ease.\n")
    w(f"**Unused:** the other {N - n} items are listed in `unused_ids.txt`. They are disjoint from the sample and stay unused.\n")

    def table(title, key, order=None, show_all=True):
        cf, cs = dist(all_ids, meta, key), dist(sel, meta, key)
        ks = order or sorted(cf, key=lambda k: (-cf[k], k))
        w(f"## {title}\n")
        w("| value | full n | full % | sample n | sample % | sample/full ratio |")
        w("|---|---:|---:|---:|---:|---:|")
        for k in ks:
            if not show_all and cf[k] == 0:
                continue
            w(f"| {k} | {cf[k]} | {pct(cf[k], N)} | {cs[k]} | {pct(cs[k], n)} | {cs[k] / cf[k]:.3f} |" if cf[k] else
              f"| {k} | 0 | - | {cs[k]} | - | - |")
        w("")
    table("Difficulty", "hardness", list(HARDNESS))
    table("Primary structure class (single label, used for within-cell ordering)", "primary_class", list(CLASSES))
    w("## Query-structure labels (multi-label; an item can carry several)\n")
    cf, cs = dist(all_ids, meta, "structure_labels"), dist(sel, meta, "structure_labels")
    w("| label | full n | full % | sample n | sample % | expected at 1000/2147 | sample − expected |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for k in REPORT_LABELS:
        e = cf[k] * n / N
        w(f"| {k} | {cf[k]} | {pct(cf[k], N)} | {cs[k]} | {pct(cs[k], n)} | {e:.1f} | {cs[k] - e:+.1f} |")
    w("\n`anti_join` follows the frozen labeller (NOT IN, NOT EXISTS or EXCEPT). `anti_join_not_in_or_exists` excludes "
      "EXCEPT, which is counted under `except` / `set_operation`.\n")
    w("## Join, nested, set-operation and anti-join representation\n")
    combos = {"join (any)": lambda s: "join" in s, "no join": lambda s: "join" not in s,
              "nested (sub-SELECT)": lambda s: "nested" in s, "set operation (INTERSECT/UNION/EXCEPT)": lambda s: "set_operation" in s,
              "anti-join (NOT IN / NOT EXISTS)": lambda s: "anti_join_not_in_or_exists" in s,
              "nested or set operation or anti-join": lambda s: bool(s & {"nested", "set_operation", "anti_join_not_in_or_exists"}),
              "flat (none of the three)": lambda s: not (s & {"nested", "set_operation", "anti_join_not_in_or_exists"})}
    w("| group | full n | full % | sample n | sample % | expected | sample − expected |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for nm, f in combos.items():
        a = sum(f(set(meta[i]["structure_labels"])) for i in all_ids)
        b = sum(f(set(meta[i]["structure_labels"])) for i in sel)
        w(f"| {nm} | {a} | {pct(a, N)} | {b} | {pct(b, n)} | {a * n / N:.1f} | {b - a * n / N:+.1f} |")
    w("")
    w("## Questions per database\n")
    w("| database | full n | sample n | quota (n·1000/2147) | easy | medium | hard | extra |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    dbf, dbs = dist(all_ids, meta, "db_id"), dist(sel, meta, "db_id")
    for d in sorted(dbf):
        hs = [alloc.get((d, x), 0) for x in HARDNESS]
        hf = [len(cells.get((d, x), [])) for x in HARDNESS]
        w(f"| {d} | {dbf[d]} | {dbs[d]} | {dbf[d] * n / N:.2f} | " + " | ".join(f"{a}/{b}" for a, b in zip(hs, hf)) + " |")
    w(f"\nDatabases represented: **{len(dbs)}/40**. Smallest sampled database: "
      f"{min(dbs.values())} items. Largest absolute deviation from the database quota: "
      f"{max(abs(dbs[d] - dbf[d] * n / N) for d in dbf):.2f} items. Cells with 0 allocated: "
      f"{sum(1 for k in alloc if alloc[k] == 0)} (source sizes: "
      f"{sorted(len(cells[k]) for k in alloc if alloc[k] == 0)}).\n")
    w("## Gold-result flags (descriptive; no filtering)\n")
    table_flags = dist(all_ids, meta, "flags"), dist(sel, meta, "flags")
    w("| flag | full n | sample n |")
    w("|---|---:|---:|")
    for k in sorted(table_flags[0]):
        w(f"| {k} | {table_flags[0][k]} | {table_flags[1][k]} |")
    deg_f = sum(bool({"empty_result", "all_null_result"} & set(meta[i]["flags"])) for i in all_ids)
    deg_s = sum(bool({"empty_result", "all_null_result"} & set(meta[i]["flags"])) for i in sel)
    w(f"\nDegenerate-gold items (`empty_result` or `all_null_result`): {deg_f} in full, {deg_s} in sample. They are kept, "
      "and a sensitivity analysis excluding them is pre-registered.\n")
    w("## Independence and leakage (summary; details in `universe_overlap_audit.json` and the preflight report)\n")
    if audit:
        w("- **Database level (L1) and id level (L2):** 0 overlaps with every source:")
        w("  - the approved corpus v2.1 (3,445 records);")
        w("  - FT `sft_train` and `sft_dev`;")
        w("  - MQL-0010 holdout, dev and build databases;")
        w("  - X0 (MQL-0012);")
        w("  - spider-sql-v1 train, val_internal and dev_full;")
        w("  - the expanded-split manifest. It is checked by database only, and the 65-item test file is never opened.")
        w("- **Question level:** 0 exact (L3) and 0 normalised (L4) matches against the approved corpus, "
          "MQL-0010 holdout and dev, X0, spider-sql-v1 train, val_internal and dev_full.")
        s5 = [i for i in sel if l5.get(i)]
        ft5 = [i for i in s5 if any("FT_trained" in p["visible_to"] for p in l5[i])]
        w(f"- **L5 near-duplicates** (token Jaccard ≥ 0.8) to corpus records: "
          f"{audit['question_level']['approved_corpus_v2_1']['n_eval_items']} items in full, **{len(s5)} in sample**.")
        w("  - Retrieval excludes these pairs per item, using the frozen MQL-0013 exclusion list.")
        w(f"  - {len(ft5)} sampled items have an L5 partner among the FT training rows. The FT weights saw a similar "
          "question about a different database, so a sensitivity analysis excluding them is pre-registered.")
        w(f"- **`question_in_train` source flag:** 27 items in full, {sum('question_in_train' in meta[i]['flags'] for i in sel)} in sample.")
        w("  - None matches any spider-sql-v1 train, corpus or FT question at L3 or L4. The flag comes from the raw Spider "
          "train files, whose matching records are outside every set any arm sees.")
    w("\n## Reproducibility\n")
    w("- **Same output on rerun:** re-running `draw` (or `ids`) with the same config gives byte-identical `sample_ids.txt`. "
      "This is tested in two fresh processes with different PYTHONHASHSEED values "
      "(`tests/test_mql0015_sampling.py`).")
    w("- **Source unchanged:** test_full sha256 is checked before and after.")
    w(f"- **Code hashes:** sampler `{manifest['code_sha256']['scripts/nl2mql_final_1000/sample_final_1000.py']}`; "
      f"patterns.py `{manifest['code_sha256']['kaggle/nl2mql_mql0013_test_full/src/patterns.py']}`; "
      f"config `{manifest['config_sha256']}`.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["draw", "ids"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    before = sha_file(ROOT / cfg["source"]["path"])
    recs, meta, cells, quota, alloc, sel, unused = draw(cfg)
    if a.mode == "draw":
        m = write_artifacts(Path(a.out), cfg, recs, meta, cells, quota, alloc, sel, unused)
        print(json.dumps({"n": len(sel), "dbs": len({meta[i]['db_id'] for i in sel}),
                          "sample_ids_sha256": m["sample_ids_sha256"]}))
    else:
        print(h("".join(i + "\n" for i in sel)))
    need(sha_file(ROOT / cfg["source"]["path"]) == before, "source file changed during the run")


if __name__ == "__main__":
    main()
