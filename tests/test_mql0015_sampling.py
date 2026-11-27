"""MQL-0015 sampling tests (CPU only; read-only on the source; writes only to pytest tmp dirs)."""
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLER = ROOT / "scripts/nl2mql_final_1000/sample_final_1000.py"
SAMPLING = ROOT / "outputs/nl2mql_final_1000/sampling"
SOURCE = ROOT / "data/processed/spider_sql_v1/test_full.jsonl"
SOURCE_SHA = "c6601ecdec5a9461aafb9015d334233ff92d00d69d659314a08784e34d6552af"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _run(out, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    r = subprocess.run([sys.executable, "-I", "-B", str(SAMPLER), "draw", "--out", str(out)], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    return (out / "sample_ids.txt").read_bytes()


def test_same_seed_same_ids_across_fresh_processes(tmp_path):
    before = _sha(SOURCE)
    a = _run(tmp_path / "a", 1)
    b = _run(tmp_path / "b", 987654)
    assert a == b
    assert a == (SAMPLING / "sample_ids.txt").read_bytes(), "registered sample_ids.txt differs from a fresh draw"
    assert _sha(SOURCE) == before == SOURCE_SHA


def test_registered_sample_properties():
    ids = (SAMPLING / "sample_ids.txt").read_text(encoding="utf-8").split()
    unused = (SAMPLING / "unused_ids.txt").read_text(encoding="utf-8").split()
    src = [json.loads(x) for x in SOURCE.read_text(encoding="utf-8").splitlines() if x.strip()]
    db = {r["id"]: r["db_id"] for r in src}
    assert len(ids) == 1000 == len(set(ids))
    assert len(unused) == 1147 == len(set(unused))
    assert not set(ids) & set(unused)
    assert set(ids) | set(unused) == set(db)
    assert len({db[i] for i in ids}) == 40
    m = json.loads((SAMPLING / "sample_manifest.json").read_text(encoding="utf-8"))
    assert [it["id"] for it in m["items"]] == ids
    assert m["sample_ids_sha256"] == hashlib.sha256((SAMPLING / "sample_ids.txt").read_bytes()).hexdigest()
    assert sum(c["allocated"] for c in m["allocation"]) == 1000


def test_allocation_is_largest_remainder():
    m = json.loads((SAMPLING / "sample_manifest.json").read_text(encoding="utf-8"))
    alloc = m["allocation"]
    floors = [int(c["quota"] // 1) for c in alloc]
    extra = [c["allocated"] - f for c, f in zip(alloc, floors)]
    assert set(extra) <= {0, 1}
    rem_up = [c["quota"] - f for c, f, e in zip(alloc, floors, extra) if e == 1]
    rem_dn = [c["quota"] - f for c, f, e in zip(alloc, floors, extra) if e == 0]
    assert min(rem_up) >= max(rem_dn) - 1e-9
    got = Counter(it["db_id"] + "|" + it["hardness"] for it in m["items"])
    assert all(got[c["db_id"] + "|" + c["hardness"]] == c["allocated"] for c in alloc)


def test_sha256sums_match():
    for line in (SAMPLING / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, name = line.split("  ", 1)
        assert _sha((SAMPLING / name).resolve()) == h, name
