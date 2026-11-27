"""Stage 3: assemble the versioned fine-tuning dataset ft-v1 (deterministic).

    python scripts/finetune_v1/build_ft.py

Inputs: data/interim/finetune_v1/{pairs.jsonl, schema.json, verify/progress.jsonl},
data/rag/corpus_v1/corpus.jsonl (280 Atlas-verified Spider TRAIN-split examples), the
expanded benchmark split. Outputs in data/finetune/ft_v1/:

  train.jsonl                    SFT records (Spider-train 280 + capped new verified)
  dev_internal.jsonl             cluster-grouped 5% of NEW verified data (loss monitoring only;
                                 model selection uses the benchmark validation partition)
  verified_capped_near_dups.jsonl verified but beyond the per-cluster cap (not trained on)
  candidates_unverified.jsonl    eligible/unsupported pairs without a verified MQL + blocker
  rejected.jsonl                 rejected / ambiguous / leakage-excluded / failed, with reasons
  status_manifest.jsonl          one line per unique source pair: final status + reasons
  leakage_report.json            automated overlap checks against the 137 held-out items
  build_manifest.json            input/code hashes, counts per stage, commands

Training input = system + user prompt (question + schema context) -> assistant target MQL.
Reference SQL, expected results and fixture metadata live only in metadata fields.
"""
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nk_common as C  # noqa: E402
import fixtures as F  # noqa: E402
import verify as V  # noqa: E402

NEAR_DUP_CAP = 3
DEV_FRACTION = 0.05
N_DISTRACTORS = 2
SEED = 20261024
TYPE_VIEW = {"int": (["int"], "INTEGER"), "float": (["double"], "REAL"), "bool": (["int"], "BOOLEAN"),
             "text": (["string"], "TEXT"), "date": (["string"], "DATE"),
             "datetime": (["string"], "DATETIME")}


def load_prompt_module():
    import prompt_spider1 as P
    return P


def frozen_samples(schema):
    """2 sample documents per table, generic random values (no query literals), seeded by name."""
    out = {}
    for t, cols in sorted(schema.items()):
        rng = random.Random(f"{SEED}:{t}")
        gens = {c: F.ColGen(t, c, ty, [], set(), F.BASE_CLOCK, 0, rng) for c, ty in cols.items()}
        out[t] = [{c: g.random_value() for c, g in gens.items()} for _ in range(2)]
    return out


def name_tokens(t):
    import grounding as G
    return {G.stem(x) for x in re.split(r"[_\d]+", t.lower()) if len(x) >= 3}


def distractors(targets, schema, key):
    taken = set().union(*(name_tokens(t) for t in targets))
    pool = sorted(t for t in schema if t not in targets and not (name_tokens(t) & taken))
    rng = random.Random(f"{SEED}:{key}")
    return sorted(rng.sample(pool, min(N_DISTRACTORS, len(pool))))


def new_user_prompt(P, question, tables, schema, samples):
    lines, slines = [], []
    for t in sorted(tables):
        name = C.coll(t)
        lines.append(f"- {name}")
        for f, ty in schema[t].items():
            obs, decl = TYPE_VIEW[ty]
            lines.append(f"    {f}: {'/'.join(obs)} [{decl}]")
        for d in samples[t]:
            slines.append(f"- {name}: {json.dumps(d, ensure_ascii=False)}")
    return P.USER_TEMPLATE.format(db=C.DB_ID, schema="\n".join(lines), fks="(none)",
                                  samples="\n".join(slines), question=question)


def target_for_new(p, r, schema):
    """Order-preserving target for a verified Kaggle pair: re-run the deterministic converter
    and require it to be identical (as content) to the verified candidate."""
    import sql2mql as MM
    import sqlsem as SS
    if r.get("mql_text"):
        t = json.loads(r["mql_text"])
    else:
        ast = SS.parse_sql(p["sql"])[0]
        an = SS.analyze(ast)
        t = MM.convert(ast, an, {x: schema[x] for x in an.to_json()["columns"]})
    if C.canon(t) != C.canon(r["mql"]):
        raise SystemExit(f"{p['pair_id']}: regenerated MQL differs from the verified candidate")
    return t


def target_for_spider(d, raw_by_key):
    """Order-preserving target for a corpus example: the original model output text."""
    from run_nl_mql_eval import parse_output
    e = raw_by_key[(d["provenance"]["source_model"], d["provenance"]["dataset_id"])]
    parsed, err = parse_output(e["raw_text"])
    if parsed is None:
        raise SystemExit(f"{d['doc_id']}: cannot parse original prediction ({err})")
    t = {"collection": parsed["collection"], "pipeline": parsed["pipeline"]}
    if C.canon(t) != C.canon(d["mql"]):
        raise SystemExit(f"{d['doc_id']}: original prediction differs from the corpus MQL")
    return t


def sql_patterns(sql):
    import patterns as PT
    return sorted(PT.sql_patterns(sql))


def mql_ops(mql):
    import patterns as PT
    return sorted(PT.mql_operators(mql["pipeline"]))


def main():
    P = load_prompt_module()
    import common as RC  # scripts/rag/common.py
    import leakage as LK

    out = C.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    pairs = C.read_jsonl(C.INTERIM / "pairs.jsonl")
    schema_doc = json.loads((C.INTERIM / "schema.json").read_text(encoding="utf-8"))
    schema = schema_doc["tables"]
    schema_sha = C.sha_obj(schema_doc)
    chash = V.code_hash()
    latest = {}
    if V.PROGRESS.exists():
        for r in C.read_jsonl(V.PROGRESS):
            latest[r["pair_id"]] = r
    pmap = {p["pair_id"]: p for p in pairs}
    # only records produced by the current code + schema count
    current = {pid: r for pid, r in latest.items()
               if pid in pmap and r.get("input_sha256") == V.input_hash(pmap[pid], schema_sha, chash)}
    samples = frozen_samples(schema)

    status_rows, verified, unverified, rejected = [], [], [], []
    for p in pairs:
        cls = p["stage1_class"]
        base = {"pair_id": p["pair_id"], "question": p["question"], "reference_sql": p["sql"],
                "source_record_ids": p["source_record_ids"], "stage1_class": cls,
                "dialect": p.get("dialect"), "repairs": p["repairs"], "complexity": p.get("complexity"),
                "near_dup_cluster": p.get("near_dup_cluster")}
        if cls == "eligible":
            r = current.get(p["pair_id"])
            if r is None:
                st = {"status": "unverified", "reasons": ["not_yet_verified"]}
            else:
                st = {"status": r["status"], "reasons": r["reasons"]}
            if st["status"] == V.VERIFIED:
                verified.append((p, r))
            elif st["status"] in ("unverified", "error"):
                unverified.append({**base, **st, "candidate_mql": r.get("mql") if r else None})
            else:
                rejected.append({**base, **st, "candidate_mql": r.get("mql") if r else None})
        elif cls == "unsupported_feature":
            st = {"status": "unverified", "reasons": p["reasons"]}
            unverified.append({**base, **st, "candidate_mql": None})
        else:
            st = {"status": cls, "reasons": p["reasons"]}
            rejected.append({**base, **st})
        status_rows.append({**base, **st})

    # post-conversion duplicates: identical loose question + identical target MQL
    seen, dup_after = {}, []
    for p, r in sorted(verified, key=lambda x: C.sha256_bytes(f"{SEED}:{x[0]['pair_id']}".encode())):
        k = (C.loose(p["question"]), C.canon(r["mql"]))
        if k in seen:
            dup_after.append({"pair_id": p["pair_id"], "duplicate_of": seen[k], "question": p["question"],
                              "reference_sql": p["sql"], "status": "duplicate_after_conversion",
                              "reasons": [f"same normalised question and target as {seen[k]}"],
                              "source_record_ids": p["source_record_ids"]})
        else:
            seen[k] = p["pair_id"]
    dup_ids = {d["pair_id"] for d in dup_after}
    verified = [(p, r) for p, r in verified if p["pair_id"] not in dup_ids]
    rejected += dup_after

    # near-duplicate cap (deterministic) and cluster-grouped internal dev split
    by_cl = collections.defaultdict(list)
    for p, r in verified:
        by_cl[p["near_dup_cluster"]].append((p, r))
    kept, capped = [], []
    for cl, items in sorted(by_cl.items()):
        items.sort(key=lambda x: C.sha256_bytes(f"{SEED}:{x[0]['pair_id']}".encode()))
        kept += items[:NEAR_DUP_CAP]
        capped += items[NEAR_DUP_CAP:]
    dev_clusters = {cl for cl in by_cl
                    if int(C.sha256_bytes(f"dev:{SEED}:{cl}".encode())[:8], 16) / 0xFFFFFFFF < DEV_FRACTION}

    def new_record(p, r, split):
        tables = sorted(p["analysis"]["columns"])
        dis = distractors(tables, schema, p["pair_id"])
        user = new_user_prompt(P, p["question"], tables + dis, schema, samples)
        mql = target_for_new(p, r, schema)
        target = C.compact(mql)
        msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": user},
                {"role": "assistant", "content": target}]
        return {
            "id": f"ftv1_{p['pair_id']}", "split": split, "source_dataset": C.SOURCE_ID,
            "source_pair_id": p["pair_id"], "source_record_ids": p["source_record_ids"],
            "db_id": C.DB_ID, "question": p["question"],
            "context": {"collections": [C.coll(t) for t in tables], "distractor_collections": [C.coll(t) for t in dis],
                        "schema_source": "inferred canonical schema (schema.json)", "foreign_keys": "none listed",
                        "sample_documents": "frozen generic samples (no query literals)"},
            "messages": msgs, "target_mql": mql,
            "labels": {"sql_patterns": sql_patterns(p["sql"]), "mql_operators": mql_ops(mql),
                       "features": p["analysis"]["features"], "complexity": p["complexity"],
                       "complexity_scheme": "nk-complexity-v1"},
            "reference": {"sql": p["sql"], "dialect": p["dialect"], "sqlite_sql": p["sqlite_sql"],
                          "repairs": p["repairs"], "near_dup_cluster": p["near_dup_cluster"]},
            "validation": {"status": r["status"], "method": "fixture_equivalence: reference SQL on SQLite vs "
                           "candidate MQL on MongoDB 8.0 ($documents), spider1_kit.exec_match",
                           "candidate_source": r["converter"], "fixture_version": r["fixture_version"],
                           "fixtures": r["fixtures"], "execution": r["execution"],
                           "mql_mutant_drop_match_caught": r.get("mql_mutant_drop_match_caught"),
                           "validator": r["validator"], "verified_at": r["verified_at"],
                           "code_sha256": r["code_sha256"], "grounding": p.get("grounding")},
            "provenance": {"license": C.SOURCE_INFO["license"], "source_url": C.SOURCE_INFO["url"],
                           "raw_sha256": C.RAW_SHA256["NL2SQL_Query_Dataset.csv"],
                           "overlap": p.get("overlap")},
        }

    # existing verified Spider-train examples (RAG corpus v1), TRAIN partition only
    db_part, _ = RC.load_split()
    corpus = [d for d in C.read_jsonl(C.CORPUS) if d["kind"] == "example"]
    raw_by_key = {}
    for d in corpus:
        m = d["provenance"]["source_model"]
        if not any(k[0] == m for k in raw_by_key):
            for e in C.read_jsonl(C.ROOT / d["provenance"]["source_file"]):
                raw_by_key[(m, e["id"])] = e
    spider_recs = []
    for d in corpus:
        if db_part.get(d["db_id"]) != "train" or d.get("split") != "train":
            raise SystemExit(f"corpus example {d['doc_id']} is not train-partition")
        msgs = P.messages(d["db_id"], d["question"])
        smql = target_for_spider(d, raw_by_key)
        target = C.compact(smql)
        spider_recs.append({
            "id": f"ftv1_{d['provenance']['dataset_id']}", "split": "train", "source_dataset": "spider1_dev_expanded_benchmark_train",
            "source_pair_id": d["provenance"]["dataset_id"], "source_record_ids": [d["provenance"]["dataset_id"]],
            "db_id": d["db_id"], "question": d["question"],
            "context": {"collections": None, "schema_source": "benchmark prompt nl-mql-spider1-v1 (full database)"},
            "messages": [{"role": "system", "content": P.SYSTEM}] + msgs + [{"role": "assistant", "content": target}],
            "target_mql": smql,
            "labels": {"sql_patterns": d.get("patterns"), "mql_operators": d.get("operators"),
                       "complexity": d.get("hardness"), "complexity_scheme": "spider_hardness"},
            "reference": {"sql": None, "note": "gold SQL lives in data/processed/spider1_nl_mql_expanded.jsonl"},
            "validation": {"status": "execution_verified_source_db",
                           "method": d["provenance"]["verification"],
                           "candidate_source": d["provenance"]["source_model"],
                           "expected_rows_sha256": d["provenance"]["expected_rows_sha256"]},
            "provenance": {"license": "Spider 1.0: CC BY-SA 4.0", "corpus": "data/rag/corpus_v1",
                           "source_record_sha256": d["provenance"]["source_record_sha256"]},
        })

    train, dev = list(spider_recs), []
    for p, r in kept:
        if p["near_dup_cluster"] in dev_clusters:
            dev.append(new_record(p, r, "dev_internal"))
        else:
            train.append(new_record(p, r, "train"))
    capped_recs = [new_record(p, r, "capped") for p, r in capped]
    for rec in train + dev + capped_recs:
        rec["hashes"] = {"messages_sha256": C.sha_obj(rec["messages"]),
                         "target_sha256": C.sha_obj(rec["target_mql"])}
        rec["hashes"]["record_sha256"] = C.sha_obj({k: v for k, v in rec.items() if k != "hashes"})

    # ---------------------------------------------------------------- leakage checks
    bench = C.read_jsonl(C.BENCHMARK)
    collections_man = json.loads(RC.COLLECTIONS.read_text(encoding="utf-8"))
    db_part_ext = dict(db_part)
    db_part_ext[C.DB_ID] = "train"
    pseudo = [{"doc_id": r["id"], "kind": "example", "db_id": r["db_id"], "split": "train",
               "question": r["question"], "mql": r["target_mql"],
               "content": r["messages"][-1]["content"] + "\n" + r["question"],
               "provenance": {"dataset_id": r["source_pair_id"]}} for r in train + dev]
    lk_errors = LK.check(pseudo, bench, db_part_ext, collections_man)
    heldout = [b for b in bench if db_part[b["db_id"]] != "train"]
    hq = {C.loose(b["question"]) for b in heldout}
    hsql = {C.loose(b["gold_sql"]) for b in heldout}
    held_colls = {c for c, i in collections_man.items() if db_part[i["db_id"]] != "train"}
    extra = []
    for r in train + dev:
        blob = json.dumps(r["messages"], ensure_ascii=False)
        if C.loose(r["question"]) in hq:
            extra.append(f"{r['id']}: question equals a held-out question")
        for c in held_colls:
            if c in blob:
                extra.append(f"{r['id']}: prompt/target mentions held-out collection {c}")
                break
        if r["reference"].get("sql") and C.loose(r["reference"]["sql"]) in hsql:
            extra.append(f"{r['id']}: reference SQL equals a held-out gold SQL")
        if r["reference"].get("sql") and r["reference"]["sql"] in blob:
            extra.append(f"{r['id']}: reference SQL appears in the training messages")
    ov = json.loads((C.INTERIM / "overlap.json").read_text(encoding="utf-8"))
    new_ids = [r["source_pair_id"] for r in train + dev if r["source_dataset"] == C.SOURCE_ID]
    sims = [ov["pairs"][i] for i in new_ids]
    leakage = {
        "rag_leakage_check_errors": lk_errors, "extra_check_errors": extra,
        "heldout_items": len(heldout), "heldout_dbs": sorted({b["db_id"] for b in heldout}),
        "new_train_max_trigram_jaccard_vs_heldout": max((s["heldout_max"] for s in sims), default=0),
        "new_train_ge_0_6_vs_heldout": sum(s["heldout_max"] >= 0.6 for s in sims),
        "new_train_max_trigram_jaccard_vs_spider_all": max((s["spider_max"] for s in sims), default=0),
        "new_train_ge_0_8_vs_spider_all": sum(s["spider_max"] >= 0.8 for s in sims),
        "table_name_overlap_with_heldout_dbs": ov["table_name_overlap_with_heldout_dbs"],
        "table_name_overlap_with_spider": ov["table_name_overlap_with_spider"],
        "stage1_leakage_excluded": sum(p["stage1_class"] == "leakage_excluded" for p in pairs),
        "spider_examples_from_train_partition_only": all(db_part[r["db_id"]] == "train" for r in spider_recs),
        "benchmark_files_untouched_sha256": {
            "data/processed/spider1_nl_mql_expanded.jsonl": C.sha256_file(C.BENCHMARK),
            **{f"data/splits_expanded/{p}.jsonl": C.sha256_file(C.SPLIT_DIR / f"{p}.jsonl")
               for p in ("train", "validation", "test")},
            "data/splits_expanded/split_manifest.json": C.sha256_file(C.SPLIT_DIR / "split_manifest.json")},
    }
    if lk_errors or extra:
        (out / "leakage_report.json").write_text(json.dumps(leakage, indent=1), encoding="utf-8")
        raise SystemExit(f"LEAKAGE CHECK FAILED: {len(lk_errors)} + {len(extra)} errors (see leakage_report.json)")

    # ---------------------------------------------------------------- write
    C.write_jsonl(out / "train.jsonl", train)
    C.write_jsonl(out / "dev_internal.jsonl", dev)
    # TRL conversational prompt-completion view (loss on the completion only); input = prompt
    for name, recs in (("sft_train.prompt_completion.jsonl", train), ("sft_dev.prompt_completion.jsonl", dev)):
        C.write_jsonl(out / name, [{"id": r["id"], "prompt": r["messages"][:-1],
                                    "completion": r["messages"][-1:]} for r in recs])
    C.write_jsonl(out / "verified_capped_near_dups.jsonl", capped_recs)
    C.write_jsonl(out / "candidates_unverified.jsonl", unverified)
    C.write_jsonl(out / "rejected.jsonl", rejected)
    capped_ids = {p["pair_id"] for p, _ in capped}
    dev_ids = {r["source_pair_id"] for r in dev}
    for s in status_rows:
        if s["pair_id"] in dup_ids:
            s["status"], s["reasons"] = "duplicate_after_conversion", ["see rejected.jsonl"]
        if s["status"] == V.VERIFIED:
            s["dataset_role"] = ("capped_near_duplicate" if s["pair_id"] in capped_ids
                                 else "dev_internal" if s["pair_id"] in dev_ids else "train")
    C.write_jsonl(out / "status_manifest.jsonl", status_rows)
    (out / "leakage_report.json").write_text(json.dumps(leakage, indent=1), encoding="utf-8")

    s1 = json.loads((C.INTERIM / "stage1_counts.json").read_text(encoding="utf-8"))
    vstat = collections.Counter(current[p["pair_id"]]["status"] if p["pair_id"] in current else "not_run"
                                for p in pairs if p["stage1_class"] == "eligible")
    new_train = [r for r in train if r["source_dataset"] == C.SOURCE_ID]
    counts = collections.OrderedDict([
        ("stage1", s1),
        ("stage2_verification_of_eligible", dict(vstat)),
        ("verified_duplicates_after_conversion", len(dup_after)),
        ("verified_unique_pairs", len(verified)),
        ("verified_near_dup_clusters", len(by_cl)),
        ("verified_after_cap", len(kept)),
        ("verified_capped_out", len(capped)),
        ("new_train", len(new_train)),
        ("new_dev_internal", len(dev)),
        ("spider_train_existing", len(spider_recs)),
        ("train_total", len(train)),
        ("unverified_candidates", len(unverified)),
        ("rejected_or_excluded", len(rejected)),
        ("by_complexity_new_train", dict(collections.Counter(r["labels"]["complexity"] for r in new_train))),
        ("by_complexity_spider", dict(collections.Counter(r["labels"]["complexity"] for r in spider_recs))),
        ("by_feature_new_train", dict(collections.Counter(f for r in new_train for f in r["labels"]["features"]).most_common())),
        ("by_sql_pattern_new_train", dict(collections.Counter(f for r in new_train for f in r["labels"]["sql_patterns"]).most_common())),
        ("by_mql_operator_new_train", dict(collections.Counter(f for r in new_train for f in r["labels"]["mql_operators"]).most_common())),
        ("by_table_new_train", dict(collections.Counter(c for r in new_train for c in r["context"]["collections"]).most_common())),
        ("expansion_factor_vs_280_corpus", round(len(train) / 280, 3)),
        ("expansion_factor_vs_455_benchmark", round(len(train) / 455, 3)),
    ])
    files = {f: C.sha256_file(out / f) for f in ("train.jsonl", "dev_internal.jsonl", "sft_train.prompt_completion.jsonl",
                                                 "sft_dev.prompt_completion.jsonl", "verified_capped_near_dups.jsonl",
                                                 "candidates_unverified.jsonl", "rejected.jsonl",
                                                 "status_manifest.jsonl", "leakage_report.json")}
    manifest = {
        "version": C.BUILD_VERSION, "source": C.SOURCE_INFO,
        "commands": ["python scripts/finetune_v1/extract.py", "python scripts/finetune_v1/verify.py",
                     "python scripts/finetune_v1/build_ft.py"],
        "inputs": {"raw": C.RAW_SHA256, "pairs.jsonl": C.sha256_file(C.INTERIM / "pairs.jsonl"),
                   "schema.json": C.sha256_file(C.INTERIM / "schema.json"), "verify_code_sha256": chash,
                   "corpus_v1": C.sha256_file(C.CORPUS), "benchmark": C.sha256_file(C.BENCHMARK),
                   "split_manifest": C.sha256_file(C.SPLIT_DIR / "split_manifest.json"),
                   "prompt_template_sha256": P.template_hash(), "prompt_version": P.PROMPT_VERSION},
        "code_sha256": {f.name: C.sha256_file(f) for f in sorted(C.HERE.glob("*.py"))},
        "parameters": {"near_dup_cap": NEAR_DUP_CAP, "dev_fraction": DEV_FRACTION,
                       "n_distractors": N_DISTRACTORS, "seed": SEED},
        "counts": counts, "outputs_sha256": files,
        "evaluation_partitions": {"model_selection": "data/splits_expanded/validation.jsonl (72, unchanged)",
                                  "final_test": "data/splits_expanded/test.jsonl (65, unchanged, unused)"},
    }
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in counts.items() if k != "stage1"}, indent=1))


if __name__ == "__main__":
    main()
