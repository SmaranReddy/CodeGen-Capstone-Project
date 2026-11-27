"""Source-label detectors D1-D6 on the 120 run01 review items (plan §E.3 item 4, §D7).

The 120 items are DEVELOPMENT data for the detectors: this asserts must-detect recall and reports (does not
assert) flags among the 110 audit-accepted items.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/source_label_audit_v1"))
import detectors as D  # noqa: E402
import run_audit as A  # noqa: E402


@pytest.fixture(scope="module")
def hits():
    return {r["review_id"]: D.run_all(r) for r in A.review_records()}


def test_prereg_matches_code():
    pr = json.loads(A.PREREG.read_text(encoding="utf-8"))
    assert pr["detectors_sha256"] == hashlib.sha256((ROOT / "scripts/source_label_audit_v1/detectors.py").read_bytes()).hexdigest()
    assert pr["params_sha256"] == A.sha_obj([D.PARAMS, D.LEXICON])


@pytest.mark.parametrize("rid", sorted(A.MUST))
def test_must_detect(hits, rid):
    subs = {h["subclass"] for h in hits[rid]}
    for det, allowed in A.MUST[rid].items():
        assert subs & set(allowed), (rid, det, subs)


def test_D2_does_not_flag_R084(hits):
    assert not any(h["detector"] == "D2" for h in hits["R084"])


def test_report_flags_among_110(hits, capsys):
    from collections import Counter
    c = Counter(h["subclass"] for rid, hs in hits.items() if rid not in A.FINDINGS for h in hs if h["action"] == "quarantine")
    print("flags among 110 audit-accepted items (false positives under the audit's labels):", dict(c))
    assert sum(c.values()) < 110


def test_detectors_never_write(tmp_path):
    src = (ROOT / "scripts/source_label_audit_v1/detectors.py").read_text(encoding="utf-8")
    assert "mode=ro" in src and "INSERT" not in src and "UPDATE " not in src and ".commit(" not in src
