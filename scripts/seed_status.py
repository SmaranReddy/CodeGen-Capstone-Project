"""Seed data/interim/conversion_status.jsonl for all 135 examples.

Sets a status + concrete reason for every example from the existing query-pattern
classification and manifest, before any new conversion is attempted. The driver
(scripts/run_conversions.py) later updates the examples it actually converts.
Does not overwrite a status that already exists unless --force is given.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conversion_kit import (EXISTING_VALIDATED, ROOT, load_status, manifest, pattern_record,
                            patterns, save_status)

FORCE = "--force" in sys.argv


def base(man, pat):
    return {
        "instance_id": man["instance_id"],
        "db": man["db"],
        "question": man["question"],
        "category": pat["category"],
        "category_name": pat["category_name"],
        "basis": pat["basis"],
        "flags": pat["flags"],
        "n_gold_csv": pat["n_gold_csv"],
        "artifact_path": None,
        "source_semantics": "gold_sql" if pat["basis"] == "gold_sql" else None,
        "validation": None,
        "newly_converted_this_run": False,
    }


def default_status(man, pat):
    iid = man["instance_id"]
    flags = set(pat["flags"])
    if iid in EXISTING_VALIDATED:
        return "existing_validated", "Validated prototype from the earlier round (artifact in data/interim/pilot)."
    if pat["category"] == 4:
        return "blocked", "not_faithful: gold CSV is not reproducible from the shipped gold SQL at value level."
    if "gold_unreliable" in flags:
        return "blocked", "gold_unreliable: shipped gold result is inconsistent."
    if "needs_mentor" in flags:
        return "deferred", "needs_mentor: question is underspecified or has differently-shaped golds requiring mentor input."
    if "view_dependent" in flags:
        return "deferred", "view_dependent: question depends on a SQL view not present in DDL; materialization rule not yet approved."
    if "ext_knowledge" in flags:
        return "deferred", "ext_knowledge: question depends on an external-knowledge document; excluded per scope."
    if pat["category"] == 3:
        return "deferred", "difficult_ambiguous: procedural/underspecified; deferred to a later wave."
    return "not_attempted", "Eligible (category 1-2); not yet attempted in this run."


def main():
    man = manifest()
    pats = patterns()
    status = load_status()
    for iid in sorted(man):
        if iid in status and not FORCE:
            continue
        st, reason = default_status(man[iid], pats[iid])
        d = base(man[iid], pats[iid])
        d["status"] = st
        d["reason"] = reason
        status[iid] = d
    save_status(status)
    from collections import Counter
    c = Counter(v["status"] for v in status.values())
    print("wrote", len(status), "status rows ->", ROOT / "data/interim/conversion_status.jsonl")
    print("status counts:", dict(c))


if __name__ == "__main__":
    main()
