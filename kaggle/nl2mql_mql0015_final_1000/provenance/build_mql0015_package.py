"""Build the MQL-0015 Kaggle package (CPU only, no network, no database). Writes:
  kaggle/nl2mql_mql0015_final_1000/ (+ _bundle.zip)
  outputs/nl2mql_final_1000/scoring_contract/atlas_mapping.json  (MQL-0015 copy of the MQL-0013 contract)
  outputs/nl2mql_final_1000/preparation/build_report.json

Inputs (read-only, hash-checked): the registration + amendment A1, the 1,000-item sample, the per-id prompt pins, the
frozen MQL-0013 package (prompt / plan LINES copied byte-for-byte for the 1,000 ids; MQL-0013 is not reactivated), the
verified FT adapter (shipped inside the package), the MQL-0013 Atlas contract (copied with only the MQL-0015 fields
changed). Never writes the corpus, R3 / Qdrant, MQL-0010/0013/0014 artifacts, the sample or any database.

  python -I -B scripts/nl2mql_mql0015/build_mql0015_package.py [--overwrite] [--pkg-out DIR --zip-out FILE --contract-out FILE]
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import mql0015_common as CM   # noqa: E402

ZIP_DATE = (2026, 11, 23, 0, 0, 0)
SRC_COPIES = ("mql0008_core.py", "mql0009_core.py", "patterns.py", "planner.py", "prompt_v2.py", "rag_block.py",
              "recovery.py", "retrieval.py", "retriever_v2.py", "structure.py", "validator.py")
CONTRACT_CHANGED_FIELDS = {"atlas_db": CM.ATLAS_DB,
                           "status": "PROVISIONING CONTRACT - MQL-0015 Spider test_full databases (insert-only target "
                                     "codegen_spider1_test_mql0015); content copied from the MQL-0013 contract 32cbbc4d..."}
need = CM.need


def lines_by_id(path, ids):
    """Byte-exact JSONL lines of the given ids, in the order of `ids`."""
    raw = Path(path).read_bytes().split(b"\n")
    by = {}
    for ln in raw:
        if ln.strip():
            by[json.loads(ln)["id"]] = ln
    need(all(i in by for i in ids), f"{path}: ids missing")
    return b"".join(by[i] + b"\n" for i in ids)


def verify_inputs():
    ids = CM.sample_ids()
    checks = {
        "registration": (CM.REGISTRATION, CM.REGISTRATION_SHA256),
        "sample_manifest": (CM.SAMPLE_MANIFEST, CM.SAMPLE_MANIFEST_SHA256),
        "prompt_pins": (CM.PROMPT_PINS, CM.PROMPT_PINS_SHA256),
        "test_full": (CM.TEST_FULL, CM.TEST_FULL_SHA256),
        "mql0013_bundle_zip": (CM.ROOT / "kaggle/nl2mql_mql0013_test_full_bundle.zip", CM.MQL13_BUNDLE_SHA256),
        "mql0013_package_manifest": (CM.MQL13_PKG / "package_manifest.json", CM.MQL13_MANIFEST_SHA256),
        "mql0013_contract": (CM.MQL13_CONTRACT, CM.MQL13_CONTRACT_SHA256),
        "adapter": (CM.ADAPTER_DIR / "adapter_model.safetensors", CM.ADAPTER_SHA256),
        "adapter_config": (CM.ADAPTER_DIR / "adapter_config.json", CM.ADAPTER_CONFIG_SHA256),
        "corpus": (CM.CORPUS, CM.CORPUS_SHA256),
    }
    got = {}
    for k, (p, h) in checks.items():
        got[k] = CM.sha256_file(p)
        need(got[k] == h, f"{k}: {p} sha256 {got[k]} != pinned {h}")
    need(CM.AMENDMENT_A1.exists(), "amendment A1 missing (write it first)")
    for line in (CM.MQL13_PKG / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, n = line.split("  ", 1)
        need(CM.sha256_file(CM.MQL13_PKG / n) == h, f"MQL-0013 package file changed: {n}")
    hs = dict(reversed(l.split("  ", 1)) for l in (CM.HANDOFF / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if l.strip())
    need(hs["checkpoint/q3b_r16_q10/adapter_model.safetensors"] == CM.ADAPTER_SHA256, "handoff manifest adapter hash")
    em = json.loads(CM.EMB_MANIFEST.read_text(encoding="utf-8"))
    need(em["index_spec"]["index_name"] == CM.R3_INDEX and em["index_spec"]["view_ids_sha256"] == CM.VIEW_IDS_SHA256
         and em["index_spec"]["corpus_sha256"] == CM.CORPUS_SHA256 and em["embedder"]["model_revision"] == CM.EMBEDDER_REVISION,
         "R3 embedding manifest differs from the pins")
    m13 = json.loads((CM.MQL13_PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    need(m13["retrieval"]["qdrant_index"] == CM.R3_INDEX and m13["retrieval"]["view_ids_sha256"] == CM.VIEW_IDS_SHA256, "MQL-0013 R3 pins")
    reg = json.loads(CM.REGISTRATION.read_text(encoding="utf-8"))
    need(reg["evaluation"]["sample"]["ids_sha256"] == CM.SAMPLE_IDS_SHA256, "registration sample pin")
    return ids, got, m13, em


def make_contract(dst):
    src = json.loads(CM.MQL13_CONTRACT.read_text(encoding="utf-8"))
    new = {**src, **CONTRACT_CHANGED_FIELDS}
    diff = sorted(k for k in set(src) | set(new) if src.get(k) != new.get(k))
    need(diff == sorted(CONTRACT_CHANGED_FIELDS), f"contract copy changed unexpected fields: {diff}")
    text = json.dumps(new, indent=1, ensure_ascii=False) + "\n"
    dst = Path(dst)
    if dst.exists():
        need(dst.read_text(encoding="utf-8") == text, f"{dst} exists with different content")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8", newline="\n")
    return new, CM.sha256_file(dst), diff


def gold_scan(pkg, ids):
    gold = {r["id"]: r for r in CM.read_jsonl(CM.TEST_FULL)}
    needles = {}
    for i in ids:
        g = gold[i]
        for nd in (g["sql"], g["execution"]["ordered_sha256"], g["execution"]["multiset_sha256"]):
            if len(nd) >= 12:
                needles[nd] = i
    hits, files = [], 0
    for p in sorted(Path(pkg).rglob("*")):
        if not p.is_file() or p.suffix == ".safetensors":
            continue
        files += 1
        t = p.read_bytes().decode("utf-8", "replace")
        hits += [{"file": p.relative_to(pkg).as_posix(), "item": i} for nd, i in needles.items() if nd in t]
    return {"files_scanned": files, "needles": len(needles), "hits": hits}


def build(pkg_out=CM.PKG_OUT, zip_out=CM.ZIP_OUT, contract_out=CM.CONTRACT, overwrite=False, report_out=None):
    pkg_out, zip_out = Path(pkg_out), Path(zip_out)
    ids, got, m13, em = verify_inputs()
    if pkg_out.exists():
        need(overwrite, f"{pkg_out} exists (use --overwrite)")
        shutil.rmtree(pkg_out)
    contract, contract_sha, contract_diff = make_contract(contract_out)
    need(set(contract["db_ids"]) >= {r["db_id"] for r in CM.read_jsonl(CM.TEST_FULL) if r["id"] in set(ids)}, "contract DBs")
    (pkg_out / "src").mkdir(parents=True)
    for n in SRC_COPIES:
        shutil.copyfile(CM.MQL13_PKG / "src" / n, pkg_out / "src" / n)
    shutil.copyfile(HERE / "pkg_src/mql0015_core.py", pkg_out / "src/mql0015_core.py")
    data = {"data/eval/eval_inputs.jsonl": CM.MQL13_PKG / "data/eval/eval_inputs.jsonl",
            "data/prompts/prompts_A0.jsonl": CM.MQL13_PKG / "data/eval/zero_shot_prompts.jsonl",
            "data/prompts/prompts_R3.jsonl": CM.MQL13_PKG / "data/prompts/prompts_R3.jsonl",
            "data/plans/plan_R3.jsonl": CM.MQL13_PKG / "data/plans/plan_R3.jsonl"}
    for rel, src in data.items():
        (pkg_out / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg_out / rel).write_bytes(lines_by_id(src, ids))
    (pkg_out / "data/context").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CM.MQL13_PKG / "data/context/atlas_collections.json", pkg_out / "data/context/atlas_collections.json")
    shutil.copyfile(CM.PROMPT_PINS, pkg_out / "data/prompt_pins.jsonl")
    (pkg_out / "model/ft_adapter").mkdir(parents=True)
    for n in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copyfile(CM.ADAPTER_DIR / n, pkg_out / "model/ft_adapter" / n)
    (pkg_out / "registration").mkdir()
    shutil.copyfile(CM.REGISTRATION, pkg_out / "registration/mql0015_registration.json")
    shutil.copyfile(CM.AMENDMENT_A1, pkg_out / "registration/amendment_A1_2026-11-23.json")
    (pkg_out / "provenance").mkdir()
    for n in ("build_mql0015_package.py", "make_mql0015_notebook.py"):
        shutil.copyfile(HERE / n, pkg_out / "provenance" / n)
    plan = CM.read_jsonl(pkg_out / "data/plans/plan_R3.jsonl")
    toks = [r["prompt_tokens_est"] for r in plan]
    cfg = {
        "experiment_id": CM.EXPERIMENT,
        "package": CM.PACKAGE,
        "status": "PACKAGE BUILT - NOT EXECUTED (no prediction or score exists)",
        "registration_sha256": CM.sha256_file(CM.REGISTRATION),
        "amendment_A1_sha256": CM.sha256_file(CM.AMENDMENT_A1),
        "model": {"id": CM.MODEL_ID, "revision": CM.MODEL_REVISION, "label": "qwen2.5-coder-3b-instruct", "dtype": "bfloat16",
                  "decoding": "greedy (do_sample=False)", "max_new_tokens": 1024, "seed": 42, "batch_size_first_pass": 8,
                  "batch_size_retry": 1, "runtime": "transformers GPU, one card (CUDA_VISIBLE_DEVICES=0)"},
        "hardware": m13["hardware"],
        "context": m13["context"],
        "ft": {"adapter_path_in_package": "model/ft_adapter", "adapter_sha256": CM.ADAPTER_SHA256,
               "adapter_config_sha256": CM.ADAPTER_CONFIG_SHA256, "lora": {"r": 16, "alpha": 32},
               "trained_on": "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit (QLoRA)",
               "merge": "peft merge_and_unload into the bf16 base 488639f1 (as MQL-0010-FT1)",
               "tokenizer": "base tokenizer at 488639f1 (not the Unsloth tokenizer files)",
               "only_checkpoint": "the notebook loads only this in-package adapter and refuses other attached adapters"},
        "retrieval": {"family": "hybrid R3 (frozen)", "lexical": "rrf_fused", "k": 3, "index": CM.R3_INDEX,
                      "embedder": "BAAI/bge-small-en-v1.5", "embedder_revision": CM.EMBEDDER_REVISION,
                      "corpus_sha256": CM.CORPUS_SHA256, "view_ids_sha256": CM.VIEW_IDS_SHA256, "view_n": 2912,
                      "embedding_manifest_sha256": CM.sha256_file(CM.EMB_MANIFEST),
                      "embedding_files_sha256": em["files_sha256"], "plans": "frozen MQL-0013 plan_R3 rows (not re-derived)"},
        "arms": {"A0V": {"model": "base", "prompts": "A0", "component_V": "schema"},
                 "R3V": {"model": "base", "prompts": "R3", "component_V": "schema"},
                 "FT-R3V": {"model": "ft", "prompts": "R3", "component_V": "schema"}},
        "evaluation": {"source": "data/processed/spider_sql_v1/test_full.jsonl", "n_items": 1000,
                       "ids_sha256": CM.SAMPLE_IDS_SHA256, "n_databases": len({r["db_id"] for r in plan})},
        "prompt_lengths": {"r3_tokens_median": sorted(toks)[len(toks) // 2], "r3_tokens_max": max(toks),
                           "r3_over_2048": sum(t > 2048 for t in toks), "truncation": "none"},
        "frozen_inputs_sha256": {"mql0013_bundle_zip": got["mql0013_bundle_zip"],
                                 "mql0013_package_manifest": got["mql0013_package_manifest"],
                                 "prompt_pins": got["prompt_pins"], "sample_manifest": got["sample_manifest"]},
        "scoring_contract": {"path": "outputs/nl2mql_final_1000/scoring_contract/atlas_mapping.json", "sha256": contract_sha,
                             "atlas_db": CM.ATLAS_DB, "collections": len(contract["collections"]),
                             "documents": sum(m["documents"] for m in contract["collections"].values()),
                             "copied_from": {"path": "outputs/nl2mql_mql0013/scoring_contract/atlas_mapping.json",
                                             "sha256": CM.MQL13_CONTRACT_SHA256, "changed_fields": contract_diff}},
        "prompt_context_sha256": {n: CM.sha256_file(CM.MQL13_PROMPT_CONTEXT / n) for n in ("prompt_samples.json", "tables.json")},
        "analysis_sha256": {"scripts/nl2mql_mql0015/src/mql0015_analysis.py": CM.sha256_file(HERE / "src/mql0015_analysis.py")},
        "scoring_code_sha256": {"scripts/nl2mql_mql0015/mql0015_scoring.py": CM.sha256_file(HERE / "mql0015_scoring.py"),
                                "scripts/nl2mql_mql0015/src/mql0015_common.py": CM.sha256_file(HERE / "src/mql0015_common.py"),
                                "scripts/nl2mql/dev1034_core.py": CM.sha256_file(CM.ROOT / "scripts/nl2mql/dev1034_core.py")},
        "session_plan": {"1": ["first:A0", "first:R3", "retry:base"], "2": ["first:FT-R3", "retry:FT"]},
    }
    CM.write_json(pkg_out / "configs/experiment.json", cfg)
    sys.path.insert(0, str(HERE))
    import make_mql0015_notebook as NB
    NB.build(pkg_out / f"{CM.PACKAGE}.ipynb")
    files = sorted(p.relative_to(pkg_out).as_posix() for p in pkg_out.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    man = {"package": CM.PACKAGE, "experiment": CM.EXPERIMENT, "status": cfg["status"],
           "files": {n: {"sha256": CM.sha256_file(pkg_out / n), "bytes": (pkg_out / n).stat().st_size} for n in files},
           "config_sha256": CM.sha256_file(pkg_out / "configs/experiment.json"), "registration_sha256": cfg["registration_sha256"],
           "model": {"id": CM.MODEL_ID, "revision": CM.MODEL_REVISION}, "adapter_sha256": CM.ADAPTER_SHA256,
           "contains_gold_labels": False, "contains_credentials": False}
    CM.write_json(pkg_out / "package_manifest.json", man)
    names = files + ["package_manifest.json"]
    (pkg_out / "SHA256SUMS.txt").write_text("".join(f"{CM.sha256_file(pkg_out / n)}  {n}\n" for n in names), encoding="utf-8",
                                            newline="\n")
    scan = gold_scan(pkg_out, ids)
    need(not scan["hits"], f"gold found in package: {scan['hits'][:3]}")
    sys.path.insert(0, str(pkg_out / "src"))
    import mql0015_core as C
    P = C.load_package(pkg_out)                        # the runtime's own fail-closed checks
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names + ["SHA256SUMS.txt"]:
            zi = zipfile.ZipInfo(f"{CM.PACKAGE}/{n}", date_time=ZIP_DATE)
            zi.compress_type = zipfile.ZIP_STORED if n.endswith(".safetensors") else zipfile.ZIP_DEFLATED
            z.writestr(zi, (pkg_out / n).read_bytes())
    tmp.replace(zip_out)
    rep = {"package_dir": str(pkg_out), "zip": str(zip_out), "zip_sha256": CM.sha256_file(zip_out),
           "package_manifest_sha256": CM.sha256_file(pkg_out / "package_manifest.json"),
           "config_sha256": man["config_sha256"], "files": len(names), "n_items": len(P["ids"]),
           "gold_scan": {k: v for k, v in scan.items() if k != "hits"} | {"hits": len(scan["hits"])},
           "inputs_sha256": got, "contract": cfg["scoring_contract"], "prompt_lengths": cfg["prompt_lengths"]}
    if report_out:
        CM.write_json(report_out, rep)
    return cfg, man, rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pkg-out", default=str(CM.PKG_OUT))
    ap.add_argument("--zip-out", default=str(CM.ZIP_OUT))
    ap.add_argument("--contract-out", default=str(CM.CONTRACT))
    ap.add_argument("--report-out", default=str(CM.PREP / "build_report.json"))
    a = ap.parse_args(argv)
    _, _, rep = build(a.pkg_out, a.zip_out, a.contract_out, a.overwrite, a.report_out)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
