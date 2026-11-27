"""Plan §E.3 item 6: every frozen artifact still has its recorded hash, and the v2.1 freeze holds."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/pilot_v3_followup"))
import hash_baseline as HB  # noqa: E402


def test_recorded_manifest_hashes_match():
    rows = HB.check_recorded()
    bad = [r for r in rows if r["status"] not in ("ok", "protected_not_opened")]
    assert not bad, bad[:5]
    assert sum(r["status"] == "ok" for r in rows) >= 217


@pytest.mark.skipif(not HB.BASE.exists(), reason="no S0 baseline")
def test_frozen_dirs_unchanged_since_baseline():
    base = json.loads(HB.BASE.read_text(encoding="utf-8"))
    snap = HB.snapshot()
    changed = [k for k, v in base["snapshot"].items() if snap.get(k) != v]
    added = [k for k in snap if k not in base["snapshot"] and not k.startswith("data/interim/finetune_v2/")]
    assert not changed and not added, (changed[:5], added[:5])


def test_v2_1_code_freeze_holds():
    r = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts/finetune_v2/freeze_v2_1_code.py"), "check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_frozen_v2_modules_not_edited_by_v2_1():
    fz = json.loads((ROOT / "data/interim/finetune_v2/round3_code_freeze.json").read_text(encoding="utf-8"))
    for k in ("sql2mql_v2.py", "schema_profile.py", "verifier3.py", "fixtures3.py"):
        assert HB.sha(ROOT / "scripts/finetune_v2" / k) == fz["code_sha256"][k]
