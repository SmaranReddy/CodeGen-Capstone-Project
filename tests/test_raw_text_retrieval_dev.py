import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0011/analysis"))

from raw_text_retrieval_dev import slices, summarize  # noqa: E402


def pct(numerator, denominator):
    return round(100 * numerator / denominator, 2) if denominator else None


class RawTextRetrievalDevTest(unittest.TestCase):
    def test_report_flags_can_overlap(self):
        sql = "SELECT a WHERE a NOT\n  IN (SELECT b) UNION SELECT c"
        self.assertEqual(slices(sql, {"union"}), ["overall", "anti_join", "set_operation"])
        self.assertEqual(slices("SELECT a EXCEPT SELECT b", {"anti_join"}), ["overall", "set_operation"])

    def test_summary_counts_eligible_examples_and_paired_changes(self):
        miss = {"hit1": False, "hit3": False, "skeleton_hit3": False, "max_jaccard": 0.5}
        hit = {"hit1": True, "hit3": True, "skeleton_hit3": True, "max_jaccard": 1.0}
        rows = [
            {"eligible_matching_examples": 2, "has_gold_skeleton": True, "masked": miss, "raw": hit},
            {"eligible_matching_examples": 0, "has_gold_skeleton": False, "masked": miss, "raw": miss},
        ]
        result = summarize(rows, pct)
        self.assertEqual(result["questions"], 2)
        self.assertEqual(result["questions_with_eligible_signature_match"], 1)
        self.assertEqual(result["eligible_matching_examples_total_across_questions"], 2)
        self.assertEqual(result["raw"]["hit@3"], 50.0)
        self.assertEqual(result["raw"]["hit@3_attainable"], 100.0)
        self.assertEqual(result["raw"]["skeleton_hit@3"], 100.0)
        self.assertEqual(result["raw"]["mJ@3"], 0.75)
        self.assertEqual((result["hit@3_gains"], result["hit@3_losses"], result["hit@3_delta_pp"]), (1, 0, 50.0))

    def test_empty_slice_has_no_rates(self):
        result = summarize([], pct)
        self.assertEqual(result["questions"], 0)
        self.assertIsNone(result["raw"]["hit@3"])
        self.assertIsNone(result["masked"]["hit@3_attainable"])
        self.assertIsNone(result["hit@3_delta_pp"])
