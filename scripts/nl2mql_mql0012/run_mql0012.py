"""MQL-0012 runner (offline; no model, no Atlas). Run with ~/codegen_data/venv_rag/Scripts/python.

validate : on the MQL-0011 dev split (development reference, not confirmation), R3 and R0 built from the UNCHANGED
           MQL-0011 classes reproduce the registered MQL-0011 top-3 for all 779 items, and FlexRetriever(bm25+dense,
           regex) returns full rankings identical to R3. No X1/X2 metric is computed. -> preflight/validation.json
register : write-once registration (config, split, query cache, artifact, validation and source hashes).
run      : verifies the registration, runs R3, X1, X2 (+ R0 descriptive) on the 3,295 X0 items through the read-only
           Qdrant collection, then the pre-registered analysis. -> results/{per_item.jsonl, results.json, summary.md}
"""
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import mql0012_common as M   # noqa: E402

C = M.C11
PRE = M.OUT / "preflight"


def env():
    import embedder as EM
    import mql0008_core as M8
    import qdrant_store as QS
    import vector_store as VS
    from build_embedding_artifact import SPEC, load_artifact
    spec, ids, dbs, shas, vec = load_artifact()
    rows, hdb = C.load_view_rows()
    tabs = C.load_tables()
    docs = M8.build_docs(rows, tabs)
    store = QS.QdrantStore.open(spec, path=C.QDRANT_PATH, exact=True)
    return {"spec": spec, "ids": ids, "vec": vec, "hdb": hdb, "tabs": tabs, "docs": docs, "store": store,
            "scorer": VS.CanonicalScorer(ids, vec, spec.dim), "SPEC": SPEC, "EM": EM}


def arms(E, cache):
    import retriever_dense as RD
    import mql0012_retriever as FR
    common = dict(store=E["store"], embedder=cache, forbidden_db_ids=E["hdb"], scorer=E["scorer"])
    return {"R3": RD.DenseHybridRetriever(E["docs"], "rrf_fused", **common),
            "X1": FR.FlexRetriever(E["docs"], ("bm25", "dense"), "knn_vote", E["store"], cache, E["scorer"], E["hdb"]),
            "X2": FR.FlexRetriever(E["docs"], ("dense",), "regex", E["store"], cache, E["scorer"], E["hdb"]),
            "R0": RD.DenseHybridRetriever(E["docs"], "bm25", forbidden_db_ids=E["hdb"])}


# ------------------------------------------------------------------ validate
def validate():
    import mql0012_retriever as FR
    import retriever_dense as RD
    E = env()
    try:
        cache = E["EM"].CachedEmbedder.from_dir(E["SPEC"], C.ARTIFACT_DIR / "query_cache_dev")
        train = {r["id"]: r for r in C.read_jsonl(C.SPIDER_TRAIN)}
        dev = [train[r["id"]] for r in C.read_jsonl(C.DEV_ITEMS)]
        infos = C.schema_infos(E["tabs"], sorted({r["db_id"] for r in dev}))
        reg = {r["id"]: r for r in C.read_jsonl(M.MQL0011_PER_ITEM)}
        r3 = RD.DenseHybridRetriever(E["docs"], "rrf_fused", store=E["store"], embedder=cache,
                                     forbidden_db_ids=E["hdb"], scorer=E["scorer"])
        r0 = RD.DenseHybridRetriever(E["docs"], "bm25", forbidden_db_ids=E["hdb"])
        fx = FR.FlexRetriever(E["docs"], ("bm25", "dense"), "regex", E["store"], cache, E["scorer"], E["hdb"])
        c = Counter()
        for e in dev:
            info = infos[e["db_id"]]
            a = r3.rank_examples(e["question"], info, e["id"])
            b = fx.rank_examples(e["question"], info, e["id"])
            z = r0.rank_examples(e["question"], info, e["id"])
            c["r3_top3_equal_registered"] += [d["doc_id"] for d, *_ in a[:3]] == reg[e["id"]]["R3"]["top3"]
            c["r0_top3_equal_registered"] += [d["doc_id"] for d, *_ in z[:3]] == reg[e["id"]]["R0"]["top3"]
            c["flex_full_ranking_equal_r3"] += a == b
        rep = {"n_dev": len(dev), **c}
        rep["ok"] = all(v == len(dev) for k, v in c.items())
    finally:
        E["store"].close()
    C.write_json(PRE / "validation.json", rep)
    print(json.dumps(rep))
    if not rep["ok"]:
        sys.exit(1)


# ------------------------------------------------------------------ register
def source_hashes():
    files = sorted((M.HERE / "src").glob("*.py")) + [M.HERE / "run_mql0012.py", M.HERE / "build_x0_split.py",
                                                     M.HERE / "build_x0_query_cache.py"]
    mods = ["holdout_view", "mql0008_core", "patterns", "retrieval", "retriever_v2", "structure", "retriever_dense",
            "vector_store", "qdrant_store", "embedder", "mql0011_common", "build_embedding_artifact"]
    files += [Path(__import__(m).__file__) for m in mods]
    return {str(Path(f).resolve().relative_to(C.ROOT)).replace("\\", "/"): C.sha256_file(f) for f in files}


def register():
    if M.REGISTRATION.exists():
        raise SystemExit(f"{M.REGISTRATION} exists; never overwritten")
    v = json.loads((PRE / "validation.json").read_text(encoding="utf-8"))
    if not v["ok"]:
        raise SystemExit("validation did not pass")
    if (M.RESULTS / "results.json").exists():
        raise SystemExit("results already exist; cannot pre-register")
    split = json.loads(M.SPLIT.read_text(encoding="utf-8"))
    C.write_json(M.REGISTRATION, {
        "experiment": "MQL-0012", "registered_utc": datetime.now(timezone.utc).isoformat(),
        "config": "scripts/nl2mql_mql0012/configs/mql0012_config.json", "config_sha256": C.sha256_file(M.CONFIG),
        "split_sha256": C.sha256_file(M.SPLIT), "x0_items_sha256": C.sha256_file(M.X0_ITEMS),
        "x0_ids_sha256": split["ids_sha256"], "n_x0_items": split["n_items"],
        "retrieval_view_ids_sha256": split["retrieval_view"]["ids_sha256"],
        "query_cache_manifest_sha256": C.sha256_file(M.QUERY_CACHE_X0 / "manifest.json"),
        "embedding_artifact_manifest_sha256": C.sha256_file(C.ARTIFACT_DIR / "manifest.json"),
        "validation_sha256": C.sha256_file(PRE / "validation.json"),
        "source_sha256": source_hashes(),
        "statement": "registered before any X0 retrieval was run and before any X1/X2 metric existed on dev or X0"})
    print(M.REGISTRATION.read_text(encoding="utf-8")[:900])


# ------------------------------------------------------------------ run
def skeleton(mql):
    return tuple(next(iter(st)) for st in mql["pipeline"])


def subgroups(sig):
    g = []
    if "join" in sig:
        g.append("join")
    if sig & {"group_by", "having"}:
        g.append("group_by")
    if sig & {"order_by", "limit"}:
        g.append("order/limit")
    if sig & {"count", "count_distinct", "aggregate"} and "group_by" not in sig:
        g.append("ungrouped_aggregate")
    if not sig & {"join", "group_by", "having", "nested", "anti_join", "intersect", "union", "count", "count_distinct",
                  "aggregate", "order_by", "limit"}:
        g.append("simple_filter/projection")
    if sig & {"nested", "anti_join"}:
        g.append("nested/anti_join")
    if sig & {"intersect", "union"}:
        g.append("set_op")
    return g


class Boot:
    """Paired percentile bootstrap; one index matrix per item set, reused for every comparison on that set."""

    def __init__(self, n, B, seed):
        rng = np.random.default_rng(seed)
        self.idx = [rng.integers(0, n, size=(min(1000, B - s), n), dtype=np.int32) for s in range(0, B, 1000)]
        self.B = B

    def dist(self, diffs):
        d = np.asarray(diffs, dtype=np.float64)
        return np.sort(np.concatenate([d[ix].mean(axis=1) for ix in self.idx])) * 100

    def ci(self, dist):
        return [round(float(dist[int(0.025 * self.B)]), 2), round(float(dist[int(0.975 * self.B) - 1]), 2)]


def mcnemar(b, c):
    from math import comb
    n = b + c
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def compare(a, b, key, sub, boot, margin=None):
    x = [(bool(i[a][key]), bool(i[b][key])) for i in sub]
    ao = sum(p and not q for p, q in x)
    bo = sum(q and not p for p, q in x)
    dist = boot.dist([int(p) - int(q) for p, q in x])
    out = {"n": len(sub), "rate_a": round(100 * sum(p for p, _ in x) / len(sub), 2),
           "rate_b": round(100 * sum(q for _, q in x) / len(sub), 2), "a_only": ao, "b_only": bo,
           "both": sum(p and q for p, q in x), "neither": sum(not p and not q for p, q in x),
           "delta_pp": round(100 * (ao - bo) / len(sub), 2), "ci95": boot.ci(dist), "mcnemar_p": mcnemar(ao, bo)}
    if margin is not None:
        out["ni_p_one_sided"] = (1 + int((dist <= margin + 1e-12).sum())) / (boot.B + 1)
    return out


def run():
    import patterns as PT
    import retrieval as RT
    import structure as S
    reg = json.loads(M.REGISTRATION.read_text(encoding="utf-8"))
    checks = {"config": (C.sha256_file(M.CONFIG), reg["config_sha256"]),
              "split": (C.sha256_file(M.SPLIT), reg["split_sha256"]),
              "items": (C.sha256_file(M.X0_ITEMS), reg["x0_items_sha256"]),
              "cache": (C.sha256_file(M.QUERY_CACHE_X0 / "manifest.json"), reg["query_cache_manifest_sha256"]),
              "artifact": (C.sha256_file(C.ARTIFACT_DIR / "manifest.json"), reg["embedding_artifact_manifest_sha256"])}
    bad = [k for k, (a, b) in checks.items() if a != b]
    if bad or source_hashes() != reg["source_sha256"]:
        raise SystemExit(f"registration mismatch: {bad or 'source code'}")
    if (M.RESULTS / "results.json").exists():
        raise SystemExit("results exist; the confirmation run is executed once")
    cfg = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    E = env()
    try:
        cache = E["EM"].CachedEmbedder.from_dir(E["SPEC"], M.QUERY_CACHE_X0)
        A = arms(E, cache)
        train = {r["id"]: r for r in C.read_jsonl(C.SPIDER_TRAIN)}
        corpus = {r["id"]: r for r in C.read_jsonl(C.CORPUS)}
        x0 = [train[r["id"]] for r in C.read_jsonl(M.X0_ITEMS)]
        infos = C.schema_infos(E["tabs"], sorted({r["db_id"] for r in x0}))
        docs = E["docs"]
        dsig = {d["doc_id"]: S.signature(d["patterns"]) for d in docs}
        dskel = {d["doc_id"]: skeleton(d["mql"]) for d in docs}
        pool_sigs = Counter(dsig.values())
        row_of = {d: i for i, d in enumerate(E["ids"])}
        per, t_arm = [], Counter()
        for n_done, e in enumerate(x0):
            info = infos[e["db_id"]]
            g = S.signature(PT.sql_patterns(e["sql"]))
            masked = PT.mask_question(e["question"], info)
            q = cache.encode([masked])[0]
            full = E["scorer"].V @ q
            twin = any(d["db_id"] != e["db_id"] and dsig[d["doc_id"]] == g and
                       (d["masked_question"] == masked or full[row_of[d["doc_id"]]] >= 0.95) for d in docs)
            rec = {"id": e["id"], "db_id": e["db_id"], "attainable": pool_sigs[g] > 0, "gold_signature": sorted(g),
                   "has_skeleton": e["id"] in corpus, "subgroups": subgroups(g), "paraphrase_twin": twin,
                   "regex_cues": sorted(PT.question_cues(e["question"], info) - {"select"})}
            gs = skeleton(corpus[e["id"]]["mql"]) if rec["has_skeleton"] else None
            for name, R in A.items():
                t0 = time.perf_counter()
                ranked = R.rank_examples(e["question"], info, qid=e["id"])
                t_arm[name] += time.perf_counter() - t0
                top = ranked[:3]
                if any(d["db_id"] == e["db_id"] or d["db_id"] in E["hdb"] for d, *_ in top) or len(top) != 3:
                    raise SystemExit(f"{name}: ineligible or short selection for {e['id']}")
                ids = [d["doc_id"] for d, *_ in top]
                sigs = [dsig[i] for i in ids]
                rec[name] = {"top3": ids, "hit3": g in sigs, "hit1": sigs[0] == g,
                             "skel3": (gs in [dskel[i] for i in ids]) if gs else None,
                             "mJ": round(max(RT.jaccard(g, s) for s in sigs), 4),
                             "trace": [{k: t.get(k) for k in ("score", "structure", "bm25_rank", "dense_rank")}
                                       for _, _, t in top]}
                if name == "X1":
                    rec["X1"]["vote_cues"] = [c for c in R.last_cues if c != "select"]
                    rec["X1"]["votes"] = R.last_vote["votes"]
                    rec["X1"]["neighbours"] = R.last_vote["neighbours"]
            per.append(rec)
            if n_done % 500 == 0:
                print(f"{n_done}/{len(x0)}", flush=True)
    finally:
        E["store"].close()
    C.write_jsonl(M.RESULTS / "per_item.jsonl", per)
    analyze(per, cfg, {k: round(1000 * v / len(per), 1) for k, v in t_arm.items()})


def analyze(per, cfg, latency):
    st = cfg["statistics"]["bootstrap"]
    att = [i for i in per if i["attainable"]]
    shp = [i for i in per if i["has_skeleton"]]
    join = [i for i in att if "join" in i["subgroups"]]
    B_att, B_shp = Boot(len(att), st["resamples"], st["seed"]), Boot(len(shp), st["resamples"], st["seed"] + 1)
    B_join = Boot(len(join), st["resamples"], st["seed"] + 2)
    res = {"n_items": len(per), "n_attainable": len(att), "n_shape": len(shp), "n_join_attainable": len(join),
           "n_twins": sum(i["paraphrase_twin"] for i in per), "latency_ms_per_query_embedded_local": latency, "arms": {}}
    for a in ("R0", "R3", "X1", "X2"):
        res["arms"][a] = {"structure_match": round(100 * sum(i[a]["hit3"] for i in att) / len(att), 2),
                          "pipeline_shape": round(100 * sum(bool(i[a]["skel3"]) for i in shp) / len(shp), 2),
                          "hit1_att": round(100 * sum(i[a]["hit1"] for i in att) / len(att), 2),
                          "mJ3": round(sum(i[a]["mJ"] for i in per) / len(per), 4),
                          "join_structure_match": round(100 * sum(i[a]["hit3"] for i in join) / len(join), 2)}
    x1 = compare("X1", "R3", "hit3", att, B_att)
    x1_shape = compare("X1", "R3", "skel3", shp, B_shp)
    x1_join = compare("X1", "R3", "hit3", join, B_join)
    x2s = compare("X2", "R3", "hit3", att, B_att, margin=-1.0)
    x2p = compare("X2", "R3", "skel3", shp, B_shp, margin=-1.0)
    p_x1, p_x2 = x1["mcnemar_p"], max(x2s["ni_p_one_sided"], x2p["ni_p_one_sided"])
    order = sorted([("X1", p_x1), ("X2", p_x2)], key=lambda t: t[1])
    holm = {order[0][0]: order[0][1] <= 0.025}
    holm[order[1][0]] = holm[order[0][0]] and order[1][1] <= 0.05
    x1_pass = {"c1_delta_ge_2": x1["delta_pp"] >= 2.0, "c2_ci_lb_gt_0": x1["ci95"][0] > 0,
               "c3_join_decline_le_2": x1_join["delta_pp"] >= -2.0, "c4_holm": holm["X1"]}
    x2_pass = {"c1_structure_lb_gt_-1": x2s["ci95"][0] > -1.0, "c2_shape_lb_gt_-1": x2p["ci95"][0] > -1.0,
               "c3_holm": holm["X2"]}
    res["confirmatory"] = {
        "X1_vs_R3_structure": x1, "X1_vs_R3_join": x1_join, "X2_vs_R3_structure": x2s, "X2_vs_R3_shape": x2p,
        "p_X1": p_x1, "p_X2": p_x2, "holm_reject": holm,
        "X1": {**x1_pass, "PASS": all(x1_pass.values())}, "X2": {**x2_pass, "PASS": all(x2_pass.values())}}
    res["descriptive"] = {
        "X1_vs_R3_shape": x1_shape,
        "R3_vs_R0_structure": compare("R3", "R0", "hit3", att, B_att),
        "X1_vs_X2_structure": compare("X1", "X2", "hit3", att, B_att),
        "subgroups": {}, "twins": {}, "cue_quality": {}, "coverage": {}}
    groups = sorted({g for i in per for g in i["subgroups"]})
    for g in groups:
        a_ = [i for i in att if g in i["subgroups"]]
        res["descriptive"]["subgroups"][g] = {"n": sum(g in i["subgroups"] for i in per), "n_att": len(a_), **{
            a: (round(100 * sum(i[a]["hit3"] for i in a_) / len(a_), 1) if a_ else None) for a in ("R0", "R3", "X1", "X2")}}
    for flag in (True, False):
        a_ = [i for i in att if i["paraphrase_twin"] == flag]
        res["descriptive"]["twins"]["twin" if flag else "non_twin"] = {"n_att": len(a_), **{
            a: round(100 * sum(i[a]["hit3"] for i in a_) / len(a_), 2) for a in ("R0", "R3", "X1", "X2")}}
    labels = ("count", "group_by", "order_by", "limit", "join", "aggregate", "distinct", "anti_join", "nested",
              "intersect", "union", "having", "like", "count_distinct")
    for src, get in (("regex", lambda i: set(i["regex_cues"])), ("vote", lambda i: set(i["X1"]["vote_cues"]))):
        q = {}
        for l in labels:
            tp = sum(l in get(i) and l in i["gold_signature"] for i in per)
            fp = sum(l in get(i) and l not in i["gold_signature"] for i in per)
            fn = sum(l not in get(i) and l in i["gold_signature"] for i in per)
            q[l] = {"precision": round(tp / (tp + fp), 3) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 3) if tp + fn else None, "support": tp + fn}
        exact = sum((get(i) - {"text_number", "where"}) == (set(i["gold_signature"]) - {"where"}) for i in per)
        res["descriptive"]["cue_quality"][src] = {"exact_signature_agreement": round(exact / len(per), 3), "per_label": q}
    unatt = [i for i in per if not i["attainable"]]
    res["descriptive"]["coverage"] = {
        "n_unattainable": len(unatt),
        "by_subgroup": Counter(g for i in unatt for g in i["subgroups"]).most_common(),
        "signatures": Counter(" ".join(i["gold_signature"]) for i in unatt).most_common(10)}
    res["completed_utc"] = datetime.now(timezone.utc).isoformat()
    C.write_json(M.RESULTS / "results.json", res)
    lines = ["# MQL-0012 results (offline structural retrieval proxy on X0; NOT execution accuracy)", "",
             f"X0 items {res['n_items']}, attainable {res['n_attainable']}, shape items {res['n_shape']}, "
             f"join (attainable) {res['n_join_attainable']}, twins {res['n_twins']}.", "",
             "| arm | structure_match | pipeline_shape | join structure | hit@1 | mJ@3 | ms/query |", "|---|---|---|---|---|---|---|"]
    for a, m in res["arms"].items():
        lines.append(f"| {a} | {m['structure_match']} | {m['pipeline_shape']} | {m['join_structure_match']} | "
                     f"{m['hit1_att']} | {m['mJ3']} | {latency.get(a)} |")
    lines += ["", "## Confirmatory", "```", json.dumps(res["confirmatory"], indent=1), "```"]
    (M.RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"arms": res["arms"], "X1": res["confirmatory"]["X1"], "X2": res["confirmatory"]["X2"],
                      "p": [p_x1, p_x2]}, indent=1))


if __name__ == "__main__":
    {"validate": validate, "register": register, "run": run}[sys.argv[1]]()
