"""MQL-0011 baseline equivalence: original planner.build_plan vs planner_dense.build_plan(lexical="bm25").

Inputs: the 779 registered MQL-0010 DEV questions (the frozen MQL-0010 plan_D0/plan_E files hold the 716 HOLDOUT ids
only, so they are not used; the holdout is never read). Retrieval docs: the registered 2,912-record view.
Prompt base: a fixed synthetic zero-shot stand-in per question (schema lines from tables.json + the 'Question:' marker);
both planners receive identical inputs, and the pinned Qwen tokenizer counts tokens so the real 1,500-token budget
logic is exercised. Arms: D0 (diversify none) and E (signature diversity), K=3, as registered.

Writes outputs/nl2mql_mql0011/preflight/planner_equivalence.json.
Run: ~/codegen_data/venv_rag/Scripts/python scripts/nl2mql_mql0011/analysis/planner_equivalence.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mql0011_common as C  # noqa: E402

TOKENIZER_DIR = Path(os.environ.get("MQL0008_TOKENIZER_DIR", str(Path.home() / "codegen_data/qwen3b_tok_488639f1")))
CTX = {"k": 3, "max_context_tokens": 1500, "n_ctx": 8192, "max_new_tokens": 1024, "margin_tokens": 64}
ARMS = {"D0": {"family": "hybrid", "diversify": "none"}, "E": {"family": "hybrid", "diversify": "signature"}}


def synthetic_prompt(e, tab):
    cols = {}
    for t, c in tab["column_names_original"][1:]:
        cols.setdefault(t, []).append(c)
    schema = "\n".join(f"{tab['db_id']}__{tab['table_names_original'][t]}: {', '.join(cs)}" for t, cs in cols.items())
    return {"id": e["id"], "system": "You translate questions into MongoDB queries.",
            "messages": [{"role": "user", "content": f"Database {tab['db_id']}\n{schema}\n\nQuestion: {e['question']}"}]}


def compare(count_tokens):
    import mql0008_core as M8
    import planner as PL
    import planner_dense as PD
    rows, hdb = C.load_view_rows()
    tabs = C.load_tables()
    docs = M8.build_docs(rows, tabs)
    dev = C.read_jsonl(C.DEV_ITEMS)
    if len(dev) != C.DEV_N or C.ids_sha256([r["id"] for r in dev]) != C.DEV_IDS_SHA256:
        raise ValueError("dev items differ from the registration")
    if set(r["db_id"] for r in dev) & hdb:
        raise ValueError("dev item from a holdout DB")
    infos = C.schema_infos(tabs, sorted({r["db_id"] for r in dev}))
    prompts = [synthetic_prompt(e, tabs[e["db_id"]]) for e in dev]
    tables = {db: tabs[db] for db in infos}
    report = {"n_items": len(dev), "dev_ids_sha256": C.DEV_IDS_SHA256, "view_ids_sha256": C.VIEW_IDS_SHA256,
              "tokenizer_dir": TOKENIZER_DIR.name, "arms": {}}
    for name, flags in ARMS.items():
        orig =PL.build_plan({"name": name, **flags}, dev, prompts, docs, infos, {}, tables, CTX, count_tokens,
                             forbidden_db_ids=hdb)
        new = PD.build_plan({"name": name, "lexical": "bm25", **flags}, dev, prompts, docs, infos, {}, tables, CTX,
                            count_tokens, forbidden_db_ids=hdb)
        extra = {tuple(sorted(set(n) - set(o))) for o, n in zip(orig, new)}
        diff = [o["id"] for o, n in zip(orig, new) if {k: v for k, v in n.items() if k != "retrieval_mql0011"} != o]
        report["arms"][name] = {"n_orig": len(orig), "n_new": len(new), "rows_differing": len(diff),
                                "differing_ids": diff[:10], "extra_keys": [list(x) for x in extra],
                                "plan_sha256_orig": C.sha256_text(PL.canon(orig)),
                                "plan_sha256_new_minus_extra_key": C.sha256_text(PL.canon(
                                    [{k: v for k, v in n.items() if k != "retrieval_mql0011"} for n in new])),
                                "n_items_with_3_examples": sum(len(o["selected_ids"]) == 3 for o in orig)}
    report["equivalent"] = all(v["rows_differing"] == 0 and v["n_orig"] == v["n_new"] == C.DEV_N
                               and v["extra_keys"] == [["retrieval_mql0011"]] for v in report["arms"].values())
    return report


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    rep = compare(lambda s: len(tok.encode(s, add_special_tokens=False)))
    C.write_json(C.OUT / "preflight/planner_equivalence.json", rep)
    print({k: (v if k != "arms" else {a: {kk: vv for kk, vv in x.items() if kk != "differing_ids"}
                                       for a, x in v.items()}) for k, v in rep.items()})
    if not rep["equivalent"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
