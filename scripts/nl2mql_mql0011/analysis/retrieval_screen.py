"""MQL-0011 stage 1: offline retrieval screen on the registered MQL-0010 dev split (no model, no Atlas).

register : write outputs/nl2mql_mql0011/registration/screen_registration.json with the config sha256 (BEFORE running).
run      : verifies the config hash against the registration, reproduces the historical R0/R0E anchors exactly
           (aborts otherwise), runs the 7 registered arms over the 779 dev items through the embedded Qdrant store
           (exact; CanonicalScorer re-scoring; cached query vectors), computes the metrics and the pre-registered gate.
Writes outputs/nl2mql_mql0011/screen/{screen_results.json, per_item.jsonl, screen_summary.md}.
Gold SQL / corpus MQL of dev items are read for METRICS only (dev is a development split, not an evaluation set).
Run (venv_rag): python scripts/nl2mql_mql0011/analysis/retrieval_screen.py register | run
"""
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mql0011_common as C  # noqa: E402

REG = C.OUT / "registration/screen_registration.json"
SCREEN = C.OUT / "screen"


def source_hashes():
    """Every MQL-0011 source the screen depends on (metric definitions live in this file), plus the reused frozen
    modules, so the registration pins code as well as the config."""
    files = sorted((C.HERE / "src").glob("*.py")) + [C.HERE / "analysis/retrieval_screen.py",
                                                      C.HERE / "build_embedding_artifact.py"]
    import patterns as PT
    import retrieval as RT
    import retriever_v2 as R2
    import structure as S
    import mql0008_core as M8
    import holdout_view as HV
    files += [Path(m.__file__) for m in (PT, RT, R2, S, M8, HV)]
    return {str(Path(f).resolve().relative_to(C.ROOT)).replace("\\", "/"): C.sha256_file(f) for f in files}


def register():
    if REG.exists():
        raise SystemExit(f"{REG} exists; a registration is never overwritten")
    C.write_json(REG, {"experiment": "MQL-0011", "stage": 1, "config": str(C.CONFIG.relative_to(C.ROOT)).replace("\\", "/"),
                       "config_sha256": C.sha256_file(C.CONFIG), "registered_utc": datetime.now(timezone.utc).isoformat(),
                       "source_sha256": source_hashes(),
                       "artifact_manifest_sha256": C.sha256_file(C.ARTIFACT_DIR / "manifest.json"),
                       "planner_equivalence_sha256": C.sha256_file(C.OUT / "preflight/planner_equivalence.json"),
                       "backend_parity_sha256": C.sha256_file(C.OUT / "preflight/backend_parity.json")})
    print(REG.read_text(encoding="utf-8"))


def skeleton(mql):
    return tuple(next(iter(stage)) for stage in mql["pipeline"])


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def boot_ci(diffs, resamples, seed):
    rng = random.Random(seed)
    n = len(diffs)
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return [round(100 * stats[int(0.025 * resamples)], 2), round(100 * stats[int(0.975 * resamples) - 1], 2)]


def run():
    import embedder as EM
    import mql0008_core as M8
    import patterns as PT
    import qdrant_store as QS
    import retrieval as RT
    import retriever_dense as RD
    import retriever_v2 as R2
    import structure as S
    import vector_store as VS
    from build_embedding_artifact import SPEC, load_artifact

    reg = json.loads(REG.read_text(encoding="utf-8"))
    if C.sha256_file(C.CONFIG) != reg["config_sha256"]:
        raise SystemExit("config differs from its registration")
    if source_hashes() != reg["source_sha256"]:
        raise SystemExit("source code differs from its registration")
    for f, k in (("preflight/backend_parity.json", "backend_parity_sha256"),
                 ("preflight/planner_equivalence.json", "planner_equivalence_sha256")):
        rep = json.loads((C.OUT / f).read_text(encoding="utf-8"))
        if C.sha256_file(C.OUT / f) != reg[k] or not (rep.get("ok") or rep.get("equivalent")):
            raise SystemExit(f"precondition {f} missing, changed or failed")
    cfg = json.loads(C.CONFIG.read_text(encoding="utf-8"))
    spec, ids, dbs, shas, vec = load_artifact()
    rows, hdb = C.load_view_rows()
    tabs = C.load_tables()
    docs = M8.build_docs(rows, tabs)
    dev = C.read_jsonl(C.DEV_ITEMS)
    train = {r["id"]: r for r in C.read_jsonl(C.SPIDER_TRAIN)}
    corpus_by_id = {r["id"]: r for r in C.read_jsonl(C.CORPUS)}
    items = [train[r["id"]] for r in dev]
    infos = C.schema_infos(tabs, sorted({r["db_id"] for r in dev}))
    pool_sigs = Counter(S.signature(d["patterns"]) for d in docs)
    gold_sig = {r["id"]: S.signature(PT.sql_patterns(r["sql"])) for r in items}
    gold_skel = {r["id"]: skeleton(corpus_by_id[r["id"]]["mql"]) for r in items if r["id"] in corpus_by_id}
    doc_sig = {d["doc_id"]: S.signature(d["patterns"]) for d in docs}
    doc_skel = {d["doc_id"]: skeleton(d["mql"]) for d in docs}
    cache = EM.CachedEmbedder.from_dir(SPEC, C.ARTIFACT_DIR / "query_cache_dev")
    scorer = VS.CanonicalScorer(ids, vec, spec.dim)
    masked = {r["id"]: PT.mask_question(r["question"], infos[r["db_id"]]) for r in items}
    row_of = {d: i for i, d in enumerate(ids)}
    twin = {}
    for r in items:
        q = cache.encode([masked[r["id"]]])[0]
        full = scorer.V @ q
        twin[r["id"]] = any(d["db_id"] != r["db_id"] and doc_sig[d["doc_id"]] == gold_sig[r["id"]]
                            and (d["masked_question"] == masked[r["id"]] or full[row_of[d["doc_id"]]] >= 0.95)
                            for d in docs)
    results, sel = {"n_items": len(items), "n_attainable": sum(pool_sigs[gold_sig[r["id"]]] > 0 for r in items),
                    "n_skeleton_items": len(gold_skel), "n_paraphrase_twins": sum(twin.values()), "arms": {}}, {}
    per_item = {r["id"]: {"id": r["id"], "db_id": r["db_id"], "attainable": pool_sigs[gold_sig[r["id"]]] > 0,
                          "has_skeleton": r["id"] in gold_skel, "paraphrase_twin": twin[r["id"]]} for r in items}
    with QS.QdrantStore.open(spec, path=C.QDRANT_PATH, exact=True) as store:
        for arm in cfg["arms"]:
            name, mode = arm["name"], arm["lexical"]
            dense = mode in RD.DENSE_MODES
            R = RD.DenseHybridRetriever(docs, mode, store=store if dense else None, embedder=cache if dense else None,
                                        forbidden_db_ids=hdb, scorer=scorer if dense else None)
            c, mj, t = Counter(), [], 0.0
            for r in items:
                t0 = time.perf_counter()
                ranked = R.rank_examples(r["question"], infos[r["db_id"]], qid=r["id"])
                t += time.perf_counter() - t0
                top = R2.select_diverse(ranked, 3) if arm["diversify"] == "signature" else ranked[:3]
                tids = [d["doc_id"] for d, *_ in top]
                if any(d["db_id"] == r["db_id"] or d["db_id"] in hdb for d, *_ in top):
                    raise SystemExit(f"{name}: ineligible selection for {r['id']}")
                sigs = [doc_sig[i] for i in tids]
                g = gold_sig[r["id"]]
                pi = per_item[r["id"]]
                h3, h1 = g in sigs, sigs[:1] == [g]
                sk = (gold_skel[r["id"]] in [doc_skel[i] for i in tids]) if pi["has_skeleton"] else None
                pi[name] = {"top3": tids, "hit3": h3, "skel3": sk}
                c["hit3"] += h3
                c["hit1"] += h1
                c["hit3_att"] += h3 and pi["attainable"]
                c["skel3"] += bool(sk)
                c["ties34"] += len(ranked) > 3 and ranked[2][1] == ranked[3][1]
                c["same"] += len(set(sigs)) == 1
                mj.append(max(RT.jaccard(g, s) for s in sigs))
            n = len(items)
            results["arms"][name] = {
                "role": arm["role"], "hit@3": pct(c["hit3"], n), "hit@1": pct(c["hit1"], n),
                "hit@3_attainable": pct(c["hit3_att"], results["n_attainable"]), "mJ@3": round(sum(mj) / n, 4),
                "skeleton_hit@3": pct(c["skel3"], len(gold_skel)), "ties34": c["ties34"], "all3_same_signature": c["same"],
                "mean_ms_per_query": round(1000 * t / n, 2)}
            sel[name] = {r["id"]: frozenset(per_item[r["id"]][name]["top3"]) for r in items}
            print(name, results["arms"][name], flush=True)
    for fam, base in (("none", "R0"), ("signature", "R0E")):
        for arm in cfg["arms"]:
            if arm["diversify"] == fam:
                results["arms"][arm["name"]]["triple_change_vs_" + base] = pct(
                    sum(sel[arm["name"]][i] != sel[base][i] for i in sel[base]), len(items))
    # ---------------------------------------------------------------- anchors (abort on any difference)
    bad = {a: {k: (results["arms"][a][k] if k != "attainable_items" else results["n_attainable"], v)
               for k, v in exp.items() if (results["arms"][a][k] if k != "attainable_items" else results["n_attainable"]) != v}
           for a, exp in cfg["anchors"].items() if a in results["arms"]}
    results["anchor_mismatches"] = {a: v for a, v in bad.items() if v}
    if results["anchor_mismatches"]:
        C.write_json(SCREEN / "screen_results_ABORTED.json", results)
        raise SystemExit(f"anchor mismatch: {results['anchor_mismatches']}")
    # ---------------------------------------------------------------- gate
    g = cfg["gate"]["bootstrap"]

    def rate(arm, key, subset):
        v = [per_item[i][arm][key] for i in subset]
        return 100.0 * sum(map(bool, v)) / len(v)

    def gate_on(ids_):
        att = [i for i in ids_ if per_item[i]["attainable"]]
        sk = [i for i in ids_ if per_item[i]["has_skeleton"]]
        d30 = rate("R3", "hit3", att) - rate("R0", "hit3", att)
        d31 = [int(per_item[i]["R3"]["hit3"]) - int(per_item[i]["R1"]["hit3"]) for i in att]
        ci31 = boot_ci(d31, g["resamples"], g["seed"])
        d30_ci = boot_ci([int(per_item[i]["R3"]["hit3"]) - int(per_item[i]["R0"]["hit3"]) for i in att],
                         g["resamples"], g["seed"])
        dsk = rate("R3", "skel3", sk) - rate("R0", "skel3", sk)
        return {"n_attainable": len(att), "n_skeleton": len(sk),
                "R3_minus_R0_hit3_att_pp": round(d30, 2), "R3_minus_R0_ci95": d30_ci,
                "R3_minus_R1_hit3_att_pp": round(100.0 * sum(d31) / len(d31), 2), "R3_minus_R1_ci95": ci31,
                "R1_minus_R0_hit3_att_pp": round(rate("R1", "hit3", att) - rate("R0", "hit3", att), 2),
                "R2_minus_R0_hit3_att_pp": round(rate("R2", "hit3", att) - rate("R0", "hit3", att), 2),
                "R3_minus_R0_skeleton_pp": round(dsk, 2),
                "G1": d30 >= 3.0, "G2": ci31[0] > 0, "G3": dsk >= -1.0}
    all_ids = [r["id"] for r in items]
    full, nontwin = gate_on(all_ids), gate_on([i for i in all_ids if not twin[i]])
    results["gate"] = {"all_items": full, "non_twin_items": nontwin,
                       "G4": nontwin["G1"] and nontwin["G2"] and nontwin["G3"]}
    results["gate"]["PASS"] = full["G1"] and full["G2"] and full["G3"] and results["gate"]["G4"]
    results["gate"]["rescaling_flag"] = full["R1_minus_R0_hit3_att_pp"] >= 3.0
    results["config_sha256"] = reg["config_sha256"]
    results["completed_utc"] = datetime.now(timezone.utc).isoformat()
    C.write_json(SCREEN / "screen_results.json", results)
    C.write_jsonl(SCREEN / "per_item.jsonl", [per_item[i] for i in all_ids])
    write_summary(results)
    print(json.dumps(results["gate"], indent=1))


def write_summary(res):
    a = res["arms"]
    cols = ["hit@3", "hit@3_attainable", "hit@1", "mJ@3", "skeleton_hit@3", "all3_same_signature", "ties34",
            "triple_change_vs_R0", "triple_change_vs_R0E", "mean_ms_per_query"]
    lines = ["# MQL-0011 stage 1: offline retrieval screen (dev split)", "",
             "Offline structural retrieval proxy on the registered MQL-0010 dev split. **Not execution accuracy.**",
             f"Items {res['n_items']}; attainable {res['n_attainable']}; skeleton items {res['n_skeleton_items']}; "
             f"paraphrase twins {res['n_paraphrase_twins']}. Config sha256 `{res['config_sha256']}`.", "",
             "| arm | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in a.items():
        lines.append(f"| {name} | " + " | ".join(str(m.get(c, "")) for c in cols) + " |")
    gt = res["gate"]
    lines += ["", "## Gate (pre-registered)", "", "```", json.dumps(gt, indent=1), "```", "",
              f"**Verdict: {'PASS' if gt['PASS'] else 'FAIL'}.** Latencies are embedded-local and are not evidence of "
              "production-scale performance."]
    (SCREEN / "screen_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    {"register": register, "run": run}[sys.argv[1]]()
