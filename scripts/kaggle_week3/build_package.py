"""Build the private Kaggle input package for the clean Week 3 control experiment (no model calls).

Three arms on the SAME 318 train items, generated on one Kaggle GPU with one model load:
  zs_rerun (k = 0), random_k3 (seeded random examples), hybrid_k3 (hybrid_masked retriever).

Every prompt is built HERE, on Windows, with the canonical pipeline
(scripts/rag/run_rag_eval.retrieve_and_prompt + the local llama-cpp tokenizer for budgeting) and
frozen into the package, so Kaggle runs generation only: no retrieval code, no corpus, no gold
SQL / gold rows, no split manifest (no test ids), no Atlas client, no .env.

Checks before anything is written (the build aborts on any failure):
  * hybrid_k3 selections == outputs/rag_week3/hybrid_k3__train (318/318)
  * random_k3 selections == the 143 saved outputs/rag_week3/random_k3__train records
  * random selection rebuilt twice is identical (seeded, order-independent)
  * no selected example comes from the target database or is the item itself
  * zs_rerun user text == the zero-shot prompt (prompt_spider1.messages), empty context
  * every non-zs prompt == zero-shot prompt + one inserted reference block
  * the zip holds only allowlisted files and no credential-like strings

Usage: python scripts/kaggle_week3/build_package.py [--version v1]
  -> kaggle/week3_control_<version>/ (package dir) and kaggle/week3_control_<version>.zip
"""
import argparse
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rag"))

import common as C  # noqa: E402
from common import K  # noqa: E402
import prompt_spider1 as P  # noqa: E402
import rag_prompt as RP  # noqa: E402
import retrieval as RT  # noqa: E402
import run_rag_eval as RR  # noqa: E402

ARMS = ("zs_rerun", "random_k3", "hybrid_k3")
SECRET_PATTERNS = [r"mongodb(\+srv)?://", r"MONGODB_URI\s*=", r"gsk_[A-Za-z0-9]{20,}", r"AIza[0-9A-Za-z_\-]{30,}",
                   r"sk-ant-[A-Za-z0-9\-]{20,}", r"sk-[A-Za-z0-9]{32,}", r"hf_[A-Za-z0-9]{30,}",
                   r"(API_KEY|PASSWORD|SECRET)\s*=\s*\S+"]
CODE_FILES = ("retrieval.py", "patterns.py", "rag_prompt.py", "run_rag_eval.py")


def prompt_sha(system, messages):
    return C.sha256_bytes(C.canon({"system": system, "messages": messages}).encode("utf-8"))


def train_rows(base, cfg):
    rows = K.read_jsonl(C.ROOT / base["dataset"])
    db_part, _ = C.load_split(C.ROOT / cfg["split_dir"])
    return [r for r in rows if db_part[r["db_id"]] == "train"]


def build_arm(arm, rows, cfg, base, corpus, collections, counter):
    exp = cfg["experiments"][arm]
    retriever = RT.Retriever(corpus, exp["retrieval"] if exp["retrieval"] != "none" else "bm25_masked",
                             hybrid_weight=exp.get("hybrid_weight", 0.5), seed=exp.get("seed", 20261020))
    out = []
    for ex in rows:
        info = RR.PT.schema_info(ex["db_id"], collections)
        msgs, ret = RR.retrieve_and_prompt(ex, exp, retriever, info, counter, cfg, base, collections)
        ret.pop("retrieval_latency_s")
        ret.pop("context_build_latency_s")
        out.append({"id": ex["id"], "db_id": ex["db_id"], "hardness": ex["hardness"], "question": ex["question"],
                    "arm": arm, "messages": msgs, "prompt_sha256": prompt_sha(P.SYSTEM, msgs), "retrieval": ret})
    return out


def check_arm(arm, recs, corpus_by_id):
    """Fairness invariants that hold for every arm."""
    for r in recs:
        zs_user = P.messages(r["db_id"], r["question"])[0]["content"]
        user = r["messages"][0]["content"]
        sel = r["retrieval"]["selected_examples"]
        if arm == "zs_rerun":
            assert user == zs_user and not sel and r["retrieval"]["empty_context"], r["id"]
            assert r["retrieval"]["context"] == "" and "Solved examples" not in user, r["id"]
        else:
            assert len(sel) == 3 and len(set(sel)) == 3, (arm, r["id"], sel)
            block = r["retrieval"]["context"]
            assert user == RP.insert_block(zs_user, block), r["id"]
            for d in sel:
                assert corpus_by_id[d]["db_id"] != r["db_id"], (arm, r["id"], d)
                assert d != "ex_" + r["id"], (arm, r["id"], d)
                assert corpus_by_id[d]["kind"] == "example" and corpus_by_id[d]["split"] == "train", d


def compare_saved(arm, recs, saved_dir, expect_n=None):
    path = C.ROOT / "outputs/rag_week3" / saved_dir / "executions.jsonl"
    saved = {r["id"]: r["retrieval"]["selected_examples"] for r in K.read_jsonl(path)}
    now = {r["id"]: r["retrieval"]["selected_examples"] for r in recs}
    same = sum(now[i] == s for i, s in saved.items())
    assert same == len(saved) and (expect_n is None or len(saved) == expect_n), (arm, same, len(saved))
    return {"saved_run": f"outputs/rag_week3/{saved_dir}", "saved_records": len(saved), "identical_selections": same}


def secret_scan(paths):
    hits = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            if re.search(pat, text):
                hits.append(f"{p.name}: {pat}")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    a = ap.parse_args()
    name = f"week3_control_{a.version}"
    pkg = C.ROOT / "kaggle" / name
    if pkg.exists():
        sys.exit(f"{pkg} exists; packages are immutable, use a new --version")

    cfg, base, base_sha = RR.load_configs()
    rows = train_rows(base, cfg)
    assert len(rows) == 318, len(rows)
    corpus_dir = C.ROOT / cfg["corpus_dir"]
    corpus = C.load_corpus(corpus_dir)
    corpus_by_id = {d["doc_id"]: d for d in corpus}
    corpus_sha = C.sha256_file(corpus_dir / "corpus.jsonl")
    collections = C.load_collections()
    counter = RR.make_counter(cfg)
    code_sha = C.sha256_bytes(b"".join((C.HERE / f).read_bytes() for f in CODE_FILES))
    mc = base["models"][cfg["provider"]]

    arms, checks = {}, {}
    for arm in ARMS:
        recs = build_arm(arm, rows, cfg, base, corpus, collections, counter)
        check_arm(arm, recs, corpus_by_id)
        arms[arm] = recs
        print(f"{arm}: {len(recs)} prompts built and checked", flush=True)
    checks["hybrid_k3"] = compare_saved("hybrid_k3", arms["hybrid_k3"], "hybrid_k3__train", 318)
    checks["random_k3"] = compare_saved("random_k3", arms["random_k3"], "random_k3__train")
    again = build_arm("random_k3", list(reversed(rows)), cfg, base, corpus, collections, counter)
    rebuilt = {r["id"]: r["prompt_sha256"] for r in again}
    assert all(rebuilt[r["id"]] == r["prompt_sha256"] for r in arms["random_k3"])
    checks["random_k3"]["rebuilt_in_reverse_order_identical"] = 318
    checks["fairness"] = ("every arm: same 318 ids in dataset order, same system prompt and zero-shot base; "
                          "zs_rerun has no examples; random/hybrid insert exactly 3 examples, none from "
                          "the target database or the item itself")
    print(json.dumps(checks, indent=1))

    pkg.mkdir(parents=True)
    (pkg / "prompts").mkdir()
    for arm in ARMS:
        K.write_jsonl(pkg / "prompts" / f"{arm}.jsonl", arms[arm])
    shutil.copyfile(HERE / "kaggle_generate.py", pkg / "kaggle_generate.py")
    shutil.copyfile(C.ROOT / "scripts/evaluation/providers.py", pkg / "providers.py")
    shutil.copyfile(HERE / "week3_control.ipynb", pkg / "week3_control.ipynb")
    shutil.copyfile(HERE / "KAGGLE_SETUP.md", pkg / "KAGGLE_SETUP.md")

    ids = [r["id"] for r in rows]
    manifest = {
        "package": name, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "clean Week 3 control: zs_rerun vs random_k3 vs hybrid_k3 on the 318 train items",
        "partition": "train", "n_items": len(ids), "item_ids": ids,
        "item_ids_sha256": C.sha256_bytes("\n".join(ids).encode("utf-8")),
        "arms": {arm: {"experiment": arm, "config": cfg["experiments"][arm],
                       "experiment_sha256": RR.experiment_hash(arm, cfg["experiments"][arm], cfg, base, corpus_sha),
                       "prompt_file": f"prompts/{arm}.jsonl"} for arm in ARMS},
        "arm_order": list(ARMS),
        "random_seed": cfg["experiments"]["random_k3"]["seed"],
        "system_prompt": P.SYSTEM, "system_prompt_sha256": C.sha256_bytes(P.SYSTEM.encode("utf-8")),
        "zero_shot_prompt_version": P.PROMPT_VERSION, "zero_shot_prompt_sha256": P.template_hash(),
        "rag_prompt_version": RP.RAG_PROMPT_VERSION, "rag_prompt_sha256": RP.template_hash(),
        "generation": base["generation"], "context": cfg["context"],
        "model": {"provider": cfg["provider"], "id": mc["id"], "file": Path(mc["gguf_path"]).name,
                  "gguf_sha256": mc["gguf_sha256"], "n_ctx": mc["n_ctx"],
                  "hf_repo": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"},
        "inference": {"backend": "llama-cpp-python", "llama_cpp_python_version": "0.3.36",
                      "n_gpu_layers": -1, "kv_reset_per_item": True,
                      "note": "prompts were token-budgeted with llama-cpp-python 0.3.36 (vocab-only load "
                              "of the same GGUF); Kaggle must use the same version"},
        "provenance": {"dataset": base["dataset"], "dataset_sha256": C.sha256_file(C.ROOT / base["dataset"]),
                       "split_dir": cfg["split_dir"],
                       "split_manifest_sha256": C.sha256_file(C.ROOT / cfg["split_dir"] / "split_manifest.json"),
                       "corpus_dir": cfg["corpus_dir"], "corpus_sha256": corpus_sha,
                       "rag_config_sha256": C.sha256_file(RR.CONFIG),
                       "base_config": cfg["base_config"], "base_config_sha256": base_sha,
                       "code_sha256": code_sha, "code_files": [f"scripts/rag/{f}" for f in CODE_FILES],
                       "builder_sha256": C.sha256_file(Path(__file__))},
        "checks": checks,
        "analysis_plan": {
            "frozen_at": "package build, before any GPU output exists",
            "primary": "hybrid_k3 vs random_k3 (do retrieved examples beat random examples?)",
            "confirmatory_family": ["hybrid_k3_vs_random_k3", "hybrid_k3_vs_zs_rerun", "random_k3_vs_zs_rerun"],
            "tests": "exact McNemar per contrast, Holm over the family, alpha 0.05; paired bootstrap 95% CI "
                     "(10k resamples, seed 20261020); Wilson 95% CI per arm",
            "exploratory": ["hardness and gold-SQL pattern slices", "helped/hurt items", "latency",
                            "top-1 example similarity", "agreement with earlier CPU outputs"],
            "caveats": ["train is a development partition: the hybrid weight was chosen on train-LODO",
                        "hybrid_k3 vs zero-shot on train is already known from the CPU runs, and 143 CPU "
                        "random_k3 outputs were seen, so the train comparison is not blind",
                        "GPU latency is not comparable with the earlier CPU latency"]},
        "excluded_on_purpose": ["gold_sql", "gold_rows", "validation/test items", "split manifest",
                                "retrieval corpus", "Atlas client and scorer", ".env / credentials"],
    }
    files = sorted(p for p in pkg.rglob("*") if p.is_file())
    manifest["files"] = {p.relative_to(pkg).as_posix(): C.sha256_file(p) for p in files}
    (pkg / "package_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    files = sorted(p for p in pkg.rglob("*") if p.is_file())
    (pkg / "SHA256SUMS.txt").write_text("".join(f"{C.sha256_file(p)}  {p.relative_to(pkg).as_posix()}\n"
                                                for p in files), encoding="utf-8")
    files = sorted(p for p in pkg.rglob("*") if p.is_file())

    allow = {"package_manifest.json", "SHA256SUMS.txt", "kaggle_generate.py", "providers.py",
             "week3_control.ipynb", "KAGGLE_SETUP.md"} | {f"prompts/{arm}.jsonl" for arm in ARMS}
    names = {p.relative_to(pkg).as_posix() for p in files}
    assert names == allow, names ^ allow
    hits = secret_scan(files)
    if hits:
        shutil.rmtree(pkg)
        sys.exit(f"credential-like strings found, package removed: {hits}")
    zpath = pkg.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(pkg).as_posix())
    print(f"package {pkg}\nzip {zpath} sha256 {C.sha256_file(zpath)}")
    for p in files:
        print(f"  {p.relative_to(pkg).as_posix():32s} {p.stat().st_size:>10,d}")


if __name__ == "__main__":
    main()
