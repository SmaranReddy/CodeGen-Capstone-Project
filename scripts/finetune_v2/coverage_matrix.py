"""G1 coverage matrix: every remediation item (F1-F15) and every audited defect -> how v2
handles it (implemented | rejected) and its regression evidence, checked against the code base:
  * every named unit test exists in tests/test_finetune_v2.py;
  * every named adversarial case exists in outputs/sql2mql_v2/adversarial/suite.jsonl and its
    recorded outcome is correct (converted, correct under both NULL representations, accepted
    or conservatively rejected by the verifier) or a correct rejection for `rejected` items.
Writes outputs/sql2mql_v2/coverage_matrix.json; exit 1 if any row lacks passing evidence.
"""
import json
import re
import sys

import v2_common as C

MATRIX = [
    # id, handling, unit tests, adversarial cases
    ("F1", "implemented", ["test_F1_ungrouped_aggregate_has_default_row", "test_live_family_catches_b1_that_l1_misses"], ["V_F1a", "A1", "A2"]),
    ("F2", "implemented", ["test_F2_sum_guarded_by_non_null_count"], ["V_F2a", "A6", "A6b", "V_HAV1", "V_CD1"]),
    ("F3", "implemented", ["test_F3_distinct_group_by_dedupes"], ["V_F3a"]),
    ("F4", "rejected", ["test_F4_date_arithmetic_rejected"], ["V_F4a"]),
    ("F5", "implemented", ["test_F5_identifiers_canonicalised", "test_double_quoted_string_literal"], ["V_F5a", "V_F5b", "V_DQ1"]),
    ("F6", "implemented+rejected", ["test_F6_affinity_literals"], ["V_F6a", "V_F6b", "V_F6c", "V_F6d", "V_F6e", "V_F6f", "W17", "V_NUM1"]),
    ("F7", "rejected", ["test_F7_mixed_type_join_rejected"], ["V_F7a", "J8"]),
    ("F8", "implemented", ["test_F8_integer_division_and_zero_guard"], ["V_F8a", "V_F8b", "V_F8c", "N1", "N1b", "N3", "V_AVG1"]),
    ("F9", "implemented", ["test_F9_order_by_ordinal_and_alias"], ["V_F9a", "V_F9b", "O4", "O4c", "O5"]),
    ("F10", "implemented", ["test_F10_not_bare_boolean_excludes_null"], ["V_F10a", "W14"]),
    ("F11", "implemented", ["test_F11_left_join_guards_are_missing_aware"], ["V_F11a", "J4", "J4b"]),
    ("F12", "implemented", ["test_F12_lookup_alias_is_reserved"], ["V_F12a", "J6"]),
    ("F13", "implemented", ["test_F13_like_regex"], ["V_F13a", "V_F13b", "V_F13c", "V_F13d", "W10d", "W10e"]),
    ("F14", "rejected", ["test_F14_functions_rejected"], ["V_F14a"]),
    ("F15", "implemented", ["test_F15_null_tests_are_missing_safe", "test_live_import_parity"], ["W19", "W19b", "W27", "A5", "A5b", "A13", "J3", "J5"]),
    ("A1b (empty table)", "implemented", ["test_F1_ungrouped_aggregate_has_default_row"], ["A1b"]),
    ("A3", "implemented", [], ["A3"]),
    ("D4", "implemented", [], ["D4"]),
    ("C4@missing (clock)", "rejected", [], ["C4"]),
    ("DISTINCT+GROUP BY", "implemented", ["test_F3_distinct_group_by_dedupes"], ["V_F3a"]),
    ("date subtraction", "rejected", ["test_F4_date_arithmetic_rejected"], ["V_F4a"]),
    ("identifier case", "implemented", ["test_F5_identifiers_canonicalised"], ["V_F5a", "V_F5b"]),
    ("implicit cast", "implemented+rejected", ["test_F6_affinity_literals"], ["V_F6a", "V_F6b", "V_F6d"]),
    ("int-vs-text join", "rejected", ["test_F7_mixed_type_join_rejected"], ["V_F7a", "J8"]),
    ("bare column (SQLite arbitrary row)", "implemented+rejected", ["test_bare_columns_and_limits"], ["V_BC1", "V_BC2"]),
    ("LIMIT without ORDER BY", "rejected", ["test_bare_columns_and_limits"], ["V_LIM1"]),
]


def main():
    tests = set(re.findall(r"^def (test_\w+)", (C.ROOT / "tests/test_finetune_v2.py").read_text(encoding="utf-8"), re.M))
    suite = {r["id"]: r for r in C.read_jsonl(C.OUT / "adversarial/suite.jsonl")}
    rows, problems = [], []
    for fid, handling, utests, cases in MATRIX:
        ev = []
        for t in utests:
            if t not in tests:
                problems.append(f"{fid}: unit test {t} missing")
        for cid in cases:
            r = suite.get(cid)
            if r is None:
                problems.append(f"{fid}: adversarial case {cid} missing")
                continue
            if r["converter"] == "rejected":
                outcome = f"converter rejected ({r['code']})"
                good = True
            elif r["converter"] == "converted":
                correct = r["explicit_null"]["correct"] and r["missing"]["correct"]
                if r["expect_reject"]:
                    good, outcome = False, "converted although rejection expected"
                else:
                    good = bool(correct)
                    outcome = f"converted, correct={correct}, verifier={r['verifier']}"
            else:
                good, outcome = False, f"crash {r.get('code')}"
            if not good:
                problems.append(f"{fid}: case {cid}: {outcome}")
            ev.append({"case": cid, "outcome": outcome})
        if not utests and not cases:
            problems.append(f"{fid}: no regression evidence")
        rows.append({"item": fid, "handling": handling, "unit_tests": utests, "adversarial": ev})
    out = {"rows": rows, "problems": problems, "complete": not problems}
    (C.OUT / "coverage_matrix.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(rows), "complete": not problems, "problems": problems}, indent=1))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
