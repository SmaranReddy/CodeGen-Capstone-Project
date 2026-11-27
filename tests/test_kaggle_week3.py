"""Clean Week 3 control (Kaggle generation / Windows scoring): package integrity and fairness,
checkpoint repair, untrusted-zip handling and output verification. No model calls, no network."""
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/kaggle_week3"))
sys.path.insert(0, str(ROOT / "scripts/rag"))
import build_package as BP  # noqa: E402
import kaggle_generate as KG  # noqa: E402
import score_kaggle_outputs as SK  # noqa: E402
import common as C  # noqa: E402
import prompt_spider1 as P  # noqa: E402

PKG = ROOT / "kaggle/week3_control_v1"
needs_pkg = pytest.mark.skipif(not PKG.exists(), reason="package not built")


def _man():
    return json.loads((PKG / "package_manifest.json").read_text(encoding="utf-8"))


def _prompts(arm):
    return C.K.read_jsonl(PKG / f"prompts/{arm}.jsonl")


@needs_pkg
def test_package_files_match_manifest_and_sums():
    man = _man()
    for name, h in man["files"].items():
        assert C.sha256_file(PKG / name) == h, name
    for line in (PKG / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, name = line.split("  ", 1)
        assert C.sha256_file(PKG / name) == h, name


@needs_pkg
def test_zip_is_allowlisted_and_has_no_credentials():
    with zipfile.ZipFile(PKG.with_suffix(".zip")) as z:
        names = set(z.namelist())
        assert not any(n.endswith(".env") or "/." in n or n.startswith(".") for n in names)
        assert names == {"package_manifest.json", "SHA256SUMS.txt", "kaggle_generate.py", "providers.py",
                         "week3_control.ipynb", "KAGGLE_SETUP.md", "prompts/zs_rerun.jsonl",
                         "prompts/random_k3.jsonl", "prompts/hybrid_k3.jsonl"}
        for n in names:
            text = z.read(n).decode("utf-8", "ignore")
            for pat in BP.SECRET_PATTERNS:
                assert not re.search(pat, text), (n, pat)
            if n.startswith("prompts/"):
                assert "gold_sql" not in text and "gold_rows" not in text, n


@needs_pkg
def test_same_items_and_settings_for_every_arm():
    man = _man()
    assert man["arm_order"] == ["zs_rerun", "random_k3", "hybrid_k3"] and man["n_items"] == 318
    assert man["random_seed"] == 20261020 and man["generation"]["temperature"] == 0
    for arm in man["arm_order"]:
        ps = _prompts(arm)
        assert [p["id"] for p in ps] == man["item_ids"]
        for p in ps:
            assert KG.canon_sha({"system": man["system_prompt"], "messages": p["messages"]}) == p["prompt_sha256"]
    assert man["system_prompt"] == P.SYSTEM


@needs_pkg
def test_zero_shot_arm_has_no_examples_and_others_exclude_target_db():
    corpus = {d["doc_id"]: d for d in C.load_corpus()}
    for p in _prompts("zs_rerun"):
        assert p["retrieval"]["selected_examples"] == [] and "Solved examples" not in p["messages"][0]["content"]
    for arm in ("random_k3", "hybrid_k3"):
        for p in _prompts(arm):
            sel = p["retrieval"]["selected_examples"]
            assert len(sel) == 3 and all(corpus[d]["db_id"] != p["db_id"] for d in sel)


@needs_pkg
def test_package_has_no_validation_or_test_items():
    db_part, _ = C.load_split()
    for arm in ("zs_rerun", "random_k3", "hybrid_k3"):
        assert {db_part[p["db_id"]] for p in _prompts(arm)} == {"train"}


def test_checkpoint_drops_only_a_truncated_last_line(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c", "raw', encoding="utf-8")
    assert [r["id"] for r in KG.load_checkpoint(p, print)] == ["a", "b"]
    assert p.read_text(encoding="utf-8") == '{"id": "a"}\n{"id": "b"}\n'
    p.write_text('{"id": "a"}\n{"id": \n{"id": "c"}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        KG.load_checkpoint(p, print)


def test_append_record_writes_one_line(tmp_path):
    p = tmp_path / "g.jsonl"
    KG.append_record(p, {"id": "a", "raw_text": "x\ny"})
    KG.append_record(p, {"id": "b"})
    assert [r["id"] for r in KG.load_checkpoint(p, print)] == ["a", "b"]


def test_generator_has_no_database_or_scoring_dependency():
    src = (ROOT / "scripts/kaggle_week3/kaggle_generate.py").read_text(encoding="utf-8")
    for bad in ("pymongo", "run_eval", "mongo_client", "dotenv", "MONGODB", "score_output"):
        assert bad not in src, bad


@pytest.mark.parametrize("name", ["../evil.txt", "/abs.txt", "C:/x.txt", "a\\..\\b.txt"])
def test_safe_extract_rejects_unsafe_paths(tmp_path, name):
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr(name, "x")
    with pytest.raises(SystemExit):
        SK.safe_extract(z, tmp_path / "dest")


def test_safe_extract_never_overwrites(tmp_path):
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("a.txt", "x")
    (tmp_path / "dest").mkdir()
    with pytest.raises(SystemExit):
        SK.safe_extract(z, tmp_path / "dest")


def test_verify_sums_detects_tampering_and_extra_files(tmp_path):
    (tmp_path / "a.json").write_text("1", encoding="utf-8")
    h = hashlib.sha256(b"1").hexdigest()
    (tmp_path / "SHA256SUMS.txt").write_text(f"{h}  a.json\n", encoding="utf-8")
    assert SK.verify_sums(tmp_path) == {"files_listed": 1, "mismatched": [], "unlisted_files": []}
    (tmp_path / "a.json").write_text("2", encoding="utf-8")
    (tmp_path / "b.json").write_text("3", encoding="utf-8")
    r = SK.verify_sums(tmp_path)
    assert r["mismatched"] == ["a.json"] and r["unlisted_files"] == ["b.json"]
