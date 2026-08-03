#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from run_external_vqa_qwen import select_answer_scope, summarize


class ExternalVqaQwenScopeTests(unittest.TestCase):
    def test_pathvqa_yes_no_only_selects_exact_frozen_count(self) -> None:
        rows = [
            *({"answer_type": "yes_no", "index": index} for index in range(3362)),
            *({"answer_type": "free_form", "index": 3362 + index} for index in range(3357)),
        ]
        selected = select_answer_scope("pathvqa", "yes_no_only", rows)
        self.assertEqual(len(selected), 3362)
        self.assertTrue(all(row["answer_type"] == "yes_no" for row in selected))

    def test_non_pathvqa_rejects_yes_no_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "only to PathVQA"):
            select_answer_scope("omnimedvqa", "yes_no_only", [])

    def test_pathvqa_wrong_yes_no_count_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 3362"):
            select_answer_scope("pathvqa", "yes_no_only", [{"answer_type": "yes_no"}])

    def test_yes_no_summary_excludes_free_form_metrics_without_division(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            data = root / "data.json"
            data.write_text("[]\n", encoding="utf-8")
            predictions = root / "predictions.jsonl"
            predictions.write_text("{}\n", encoding="utf-8")
            args = argparse.Namespace(
                task="pathvqa",
                split_role="external_test",
                backend="qwen2_5_vl",
                model=model,
                data=data,
                max_new_tokens=64,
                pathvqa_answer_scope="yes_no_only",
            )
            row = {
                "completion": "yes",
                "generated_token_count": 1,
                "reached_generation_cap": False,
                "answer_type": "yes_no",
                "exact_match": True,
                "official_token_overlap_score": 1.0,
                "official_token_f1_score": 1.0,
                "contract_aligned_exact_match": True,
            }
            result = summarize([row], args, predictions)
            self.assertEqual(result["answer_scope"], "yes_no_only")
            self.assertTrue(result["free_form_inference_excluded"])
            self.assertNotIn("paper_free_form_macro_token_f1", result)


if __name__ == "__main__":
    unittest.main()
