"""MQL-0015 registration input checks (CPU only, read-only on every input).

Reads the frozen MQL-0013 package (as read-only reference inputs; MQL-0013 itself stays superseded and unexecuted),
the MQL-0015 sample, the verified FT handoff, the MQL-0011 embedding artifact and the Atlas contract. Writes:
  outputs/nl2mql_final_1000/registration/registration_input_checks.json
  outputs/nl2mql_final_1000/registration/prompt_pins.jsonl    (per sampled id: zero-shot + R3 prompt sha256, plan facts)

Checks:
  1. MQL-0013 package SHA256SUMS all OK; frozen file hashes equal the MQL-0013 config pins.
  2. The 1,000 ids are present in the zero-shot prompts, R3 prompts and R3 plans.
  3. Identical schema context: same system prompt; R3 user text = zero-shot text up to "\\n\\nQuestion:" + RAG block +
     the same question tail. Zero-shot template sha = FT training template (7401bf14...).
  4. R3 plans: k = 3, no budget drop, no empty context, lexical rrf_fused on the registered index; L3-L5 exclusion
     pairs honoured (no excluded corpus id selected).
  5. Context length: prompt tokens (registered Qwen tokenizer estimate) vs the FT training max_seq_length 2048 and the
     model context; generator code has no truncation.
  6. No gold labels in the generation inputs: for each sampled item its gold SQL text and gold result hashes are
     searched in every file of the MQL-0013 package directory and bundle zip.
  7. FT adapter: handoff SHA256SUMS + actual file hash = 7096760a...; adapter_config base and LoRA shape.
  8. R3 retrieval artifact hashes (embedding manifest, doc_index, vectors) and corpus/view hashes.
"""
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "kaggle/nl2mql_mql0013_test_full"
PKG_ZIP = ROOT / "kaggle/nl2mql_mql0013_test_full_bundle.zip"
SAMPLING = ROOT / "outputs/nl2mql_final_1000/sampling"
OUT = ROOT / "outputs/nl2mql_final_1000/registration"
HANDOFF = ROOT / "outputs/finetuning_reuse/handoff_extract/ft_handoff_q3b_r16_q10"
EMB = ROOT / "outputs/nl2mql_mql0011/embeddings/mql0011__v2_1_view2912__bge_small_en_v1_5__5c38ec7c__masked_q"
CONTRACT = ROOT / "outputs/nl2mql_mql0013/scoring_contract/atlas_mapping.json"
TEST_FULL = ROOT / "data/processed/spider_sql_v1/test_full.jsonl"
FT_TEMPLATE = "7401bf14"
ADAPTER_SHA = "7096760ac2ed7e4a6c2eefdd7c0c249d4b3dba723f069982a499b876d434e09d"


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def sha(p):
    return sha_b(Path(p).read_bytes())


def jl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def need(c, m):
    if not c:
        raise SystemExit(f"STOP: {m}")


def main():
    res = {}
    # 1. package integrity
    sums = [l.split("  ", 1) for l in (PKG / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = [n for h, n in sums if sha(PKG / n) != h]
    cfg = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    res["mql0013_package"] = {
        "sha256sums_files": len(sums), "sha256sums_bad": bad,
        "bundle_zip_sha256": sha(PKG_ZIP), "package_manifest_sha256": sha(PKG / "package_manifest.json"),
        "config_sha256": sha(PKG / "configs/experiment.json"),
        "zero_shot_prompts_sha256": sha(PKG / "data/eval/zero_shot_prompts.jsonl"),
        "prompts_R3_sha256": sha(PKG / "data/prompts/prompts_R3.jsonl"),
        "plan_R3_sha256": sha(PKG / "data/plans/plan_R3.jsonl"),
        "eval_inputs_sha256": sha(PKG / "data/eval/eval_inputs.jsonl"),
        "atlas_collections_sha256": sha(PKG / "data/context/atlas_collections.json"),
        # prompts pin = file bytes; plans pin = sha256 of canonical JSON of the plan list (build_mql0013_package.py:223)
        "config_pins_match": sha(PKG / "data/prompts/prompts_R3.jsonl") == cfg["prompts_sha256"]["R3"]
                             and sha_b(json.dumps(jl(PKG / "data/plans/plan_R3.jsonl"), sort_keys=True, ensure_ascii=False,
                                                  separators=(",", ":")).encode("utf-8")) == cfg["plans_sha256"]["R3"],
        "mql0013_status_in_config": cfg["status"],
    }
    need(not bad and res["mql0013_package"]["config_pins_match"], "MQL-0013 package integrity")
    # 2. ids
    ids = (SAMPLING / "sample_ids.txt").read_text(encoding="utf-8").split()
    need(len(ids) == 1000 == len(set(ids)), "sample ids")
    zs = {r["id"]: r for r in jl(PKG / "data/eval/zero_shot_prompts.jsonl")}
    r3 = {r["id"]: r for r in jl(PKG / "data/prompts/prompts_R3.jsonl")}
    pl = {r["id"]: r for r in jl(PKG / "data/plans/plan_R3.jsonl")}
    ev = {r["id"]: r for r in jl(PKG / "data/eval/eval_inputs.jsonl")}
    need(all(i in zs and i in r3 and i in pl and i in ev for i in ids), "sampled ids missing from MQL-0013 package")
    # 3. identical schema context
    sys_bad, prefix_bad, tail_bad, tmpl = [], [], [], set()
    for i in ids:
        z, r = zs[i], r3[i]
        tmpl.add(z["template_sha256"])
        if z["system"] != r["system"]:
            sys_bad.append(i)
        zu, ru = z["messages"][0]["content"], r["messages"][0]["content"]
        cut = zu.index("\n\nQuestion:")
        if not ru.startswith(zu[:cut]):
            prefix_bad.append(i)
        if not ru.endswith(zu[cut:]):
            tail_bad.append(i)
    res["schema_context"] = {"system_prompt_differs": sys_bad, "schema_prefix_differs": prefix_bad,
                             "question_tail_differs": tail_bad, "zero_shot_template_sha256": sorted(tmpl),
                             "equals_ft_training_template": all(t.startswith(FT_TEMPLATE) for t in tmpl),
                             "distinct_system_prompts": len({zs[i]["system"] for i in ids})}
    need(not (sys_bad or prefix_bad or tail_bad), "schema context not identical across arms")
    # 4. plans + exclusions
    excl = {}
    for e in cfg["retrieval"]["exclusions"]:
        excl.setdefault(e["eval_id"], set()).add(e["corpus_id"])
    viol = [i for i in ids if excl.get(i, set()) & set(pl[i]["selected_ids"])]
    res["r3_plans"] = {
        "k_not_3": [i for i in ids if len(pl[i]["selected_ids"]) != 3],
        "budget_drops": [i for i in ids if pl[i]["dropped_budget"]],
        "empty_context": [i for i in ids if pl[i]["empty_context"]],
        "lexical": sorted({pl[i]["retrieval_mql0011"]["lexical"] for i in ids}),
        "index_name": sorted({pl[i]["retrieval_mql0011"]["index_name"] for i in ids}),
        "prompt_sha_matches_plan": all(pl[i]["prompt_sha256"] == r3[i]["prompt_sha256"] for i in ids),
        "sampled_items_with_exclusion_pairs": sorted(i for i in ids if i in excl),
        "exclusion_violations": viol,
        "selected_db_ids_overlap_test_dbs": sorted({d for i in ids for d in pl[i]["selected_db_ids"]} & {ev[i]["db_id"] for i in ids}),
    }
    need(not viol and not res["r3_plans"]["k_not_3"] and not res["r3_plans"]["budget_drops"], "R3 plan checks")
    # 5. lengths
    toks = sorted(pl[i]["prompt_tokens_est"] for i in ids)
    res["context_length"] = {
        "r3_prompt_tokens_est": {"median": toks[len(toks) // 2], "p95": toks[int(0.95 * (len(toks) - 1))],
                                 "p99": toks[int(0.99 * (len(toks) - 1))], "max": toks[-1]},
        "r3_prompts_over_2048": sum(t > 2048 for t in toks),
        "r3_prompt_plus_1024_new_over_4096": sum(t + 1024 > 4096 for t in toks),
        "ft_training_max_seq_length": 2048,
        "model_max_position_embeddings": 32768,
        "mql0013_context_cfg": cfg["context"],
        "generator_truncation": "none: tok(texts, return_tensors='pt', padding=True) with no truncation/max_length "
                                "(kaggle/nl2mql_mql0013_test_full/src/mql0008_core.py run_batch); demo block assembled "
                                "whole or skipped (never truncated) with 0 budget drops",
        "ids_over_2048": [i for i in ids if pl[i]["prompt_tokens_est"] > 2048],
    }
    # 6. gold scan
    gold = {r["id"]: r for r in jl(TEST_FULL)}
    needles = {}
    for i in ids:
        g = gold[i]
        needles[g["sql"]] = i
        needles[g["execution"]["ordered_sha256"]] = i
        needles[g["execution"]["multiset_sha256"]] = i
    texts = {}
    for p in PKG.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            texts[str(p.relative_to(ROOT))] = p.read_bytes().decode("utf-8", "replace")
    with zipfile.ZipFile(PKG_ZIP) as z:
        for n in z.namelist():
            if not n.endswith("/"):
                texts[f"zip:{n}"] = z.read(n).decode("utf-8", "replace")
    hits = []
    for name, t in texts.items():
        for nd, i in needles.items():
            if len(nd) >= 12 and nd in t:
                hits.append({"file": name, "item": i, "needle": nd[:80]})
    res["gold_scan"] = {"files_scanned": len(texts), "needles": len(needles), "hits": hits[:50], "n_hits": len(hits),
                        "note": "gold SQL strings + gold result hashes of the 1,000 sampled items; needles shorter "
                                "than 12 characters skipped"}
    # 7. FT adapter
    hs = dict(reversed(l.split("  ", 1)) for l in (HANDOFF / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if l.strip())
    ad_path = HANDOFF / "checkpoint/q3b_r16_q10/adapter_model.safetensors"
    acfg = json.loads((HANDOFF / "checkpoint/q3b_r16_q10/adapter_config.json").read_text(encoding="utf-8"))
    res["ft_adapter"] = {
        "path": str(ad_path.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": hs.get("checkpoint/q3b_r16_q10/adapter_model.safetensors"),
        "file_sha256": sha(ad_path), "bytes": ad_path.stat().st_size,
        "adapter_config_sha256": sha(HANDOFF / "checkpoint/q3b_r16_q10/adapter_config.json"),
        "handoff_zip_sha256": sha(ROOT / "ft_handoff_q3b_r16_q10.zip"),
        "base_model_name_or_path": acfg["base_model_name_or_path"], "r": acfg["r"], "lora_alpha": acfg["lora_alpha"],
        "target_modules": sorted(acfg["target_modules"]), "peft_type": acfg["peft_type"],
        "matches_verified": None,
    }
    res["ft_adapter"]["matches_verified"] = res["ft_adapter"]["manifest_sha256"] == res["ft_adapter"]["file_sha256"] == ADAPTER_SHA
    need(res["ft_adapter"]["matches_verified"], "adapter hash")
    # 8. retrieval artifact
    em = json.loads((EMB / "manifest.json").read_text(encoding="utf-8"))
    res["r3_retrieval"] = {
        "index_name": em["index_spec"]["index_name"], "embedder": em["embedder"]["model_id"],
        "embedder_revision": em["embedder"]["model_revision"], "manifest_sha256": sha(EMB / "manifest.json"),
        "doc_index_sha256": sha(EMB / "doc_index.jsonl"), "doc_vectors_npy_sha256": sha(EMB / "doc_vectors.npy"),
        "manifest_files_match": em["files_sha256"] == {"doc_index.jsonl": sha(EMB / "doc_index.jsonl"),
                                                       "doc_vectors.npy": sha(EMB / "doc_vectors.npy")},
        "vectors_sha256_spec": em["index_spec"]["vectors_sha256"], "view_ids_sha256": em["index_spec"]["view_ids_sha256"],
        "corpus_sha256": em["index_spec"]["corpus_sha256"], "n_records": em["index_spec"]["n_records"],
        "package_view_ids_sha256": cfg["retrieval"]["view_ids_sha256"],
        "consistent_with_package": em["index_spec"]["view_ids_sha256"] == cfg["retrieval"]["view_ids_sha256"]
                                   and em["index_spec"]["index_name"] == cfg["retrieval"]["qdrant_index"],
    }
    # Atlas contract
    ct = json.loads(CONTRACT.read_text(encoding="utf-8"))
    sdb = {gold[i]["db_id"] for i in ids}
    res["atlas_contract"] = {"path": str(CONTRACT.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(CONTRACT),
                             "atlas_db": ct["atlas_db"], "collections": len(ct["collections"]),
                             "covers_all_sampled_dbs": sdb <= set(ct["db_ids"]), "sampled_dbs": len(sdb)}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "prompt_pins.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in ids:
            f.write(json.dumps({"id": i, "db_id": ev[i]["db_id"], "zero_shot_prompt_sha256": zs[i]["prompt_sha256"],
                                "r3_prompt_sha256": r3[i]["prompt_sha256"], "r3_selected_ids": pl[i]["selected_ids"],
                                "r3_prompt_tokens_est": pl[i]["prompt_tokens_est"]}) + "\n")
    res["prompt_pins_sha256"] = sha(OUT / "prompt_pins.jsonl")
    (OUT / "registration_input_checks.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    short = json.loads(json.dumps(res))
    short["context_length"].pop("ids_over_2048")
    print(json.dumps(short, indent=1)[:7000])


if __name__ == "__main__":
    sys.exit(main())
