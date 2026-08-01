#!/usr/bin/env python3
"""Unit tests for the offline matched Stage3 penalty case analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import analyze_stage3_penalty_cases as analysis


class PenaltyCaseAnalysisTest(unittest.TestCase):
    def test_choice_from_completion(self) -> None:
        self.assertEqual(
            analysis.choice_from_completion("<think>x</think><answer>C) value</answer>"), "C"
        )
        self.assertIsNone(analysis.choice_from_completion("no answer tags"))

    def make_run(self, root: Path, label: str, correctness: list[int]) -> Path:
        run = root / label
        audit = run / "reward_audit"
        cache = run / "judge" / "cache" / "aa"
        validation = run / "validation"
        audit.mkdir(parents=True)
        cache.mkdir(parents=True)
        validation.mkdir(parents=True)

        by_rank: dict[int, list[dict[str, object]]] = {0: [], 1: []}
        for rank in (0, 1):
            for reward_type, reward in (("accuracy", float(rank)), ("format", 1.0), ("process", 0.75)):
                by_rank[rank].append(
                    {
                        "reward_type": reward_type,
                        "reward": reward,
                        "rank": rank,
                        "call_index": rank,
                        "item_index": 0,
                        "record_index": rank,
                        "problem": f"problem-{rank}",
                        "solution": "<answer>A) gold</answer>",
                        "image_sha256": f"image-{rank}",
                        "completion": "<think>reason</think><answer>A) value</answer>",
                    }
                )
        for rank, rows in by_rank.items():
            (audit / f"rank_{rank:02d}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

        judge_value = {
            "cache_key": f"cache-{label}",
            "served_model": "judge",
            "scores": {
                "penalty": 0.4,
                "integrity": 0.8,
                "knowledge": 0.9,
                "process": 0.85,
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "image_feature_analysis_present": True,
                                    "option_elimination_present": False,
                                    "medical_knowledge_support_present": True,
                                    "histological_definition_error": False,
                                    "logical_contradiction": False,
                                    "outdated_or_incorrect_pathology_criterion": False,
                                }
                            )
                        }
                    }
                ]
            },
        }
        (cache / "record.json").write_text(json.dumps(judge_value), encoding="utf-8")

        rows = []
        for index, correct in enumerate(correctness):
            choice = "A" if correct else "B"
            rows.append(
                {
                    "index": index,
                    "image": f"image-{index}",
                    "problem": f"question-{index}",
                    "solution": "<answer>A) gold</answer>",
                    "predicted_choice": choice,
                    "accuracy_reward": float(correct),
                    "format_reward": 1.0,
                    "generated_token_count": 10 + index,
                    "completion": f"<think>{label}</think><answer>{choice}) value</answer>",
                }
            )
        (validation / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (validation / "metrics.json").write_text(
            json.dumps({"status": "completed", "count": len(rows)}), encoding="utf-8"
        )
        return run

    def test_matched_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arms = [
                ("0.3", self.make_run(root, "arm03", [0, 1, 0, 1])),
                ("0.4", self.make_run(root, "arm04", [0, 1, 0, 1])),
                ("0.5", self.make_run(root, "arm05", [0, 1, 1, 0])),
            ]
            result = analysis.analyze(arms, "validation")
            self.assertEqual(result["network_calls"], 0)
            self.assertEqual(result["training_alignment"]["aligned_rollout_count"], 2)
            comparison = result["validation_comparison"]
            self.assertEqual(comparison["pattern_counts"], {"000": 1, "001": 1, "110": 1, "111": 1})
            self.assertEqual(comparison["disagreement_case_count"], 2)
            self.assertEqual(comparison["pairwise"]["0.4_to_0.5"]["improved"], 1)
            self.assertEqual(comparison["pairwise"]["0.4_to_0.5"]["regressed"], 1)


if __name__ == "__main__":
    unittest.main()
