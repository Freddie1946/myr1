#!/usr/bin/env python3
from __future__ import annotations

import unittest

from prepare_pathvqa_capped_unresolved_rerun import select_records


class PreparePathvqaCorrectiveSubsetTests(unittest.TestCase):
    def test_only_capped_and_unresolved_rows_are_selected(self) -> None:
        records = [
            {"answer_type": "yes_no", "question": str(index), "answer": "yes", "image": "x"}
            for index in range(3362)
        ]
        from external_vqa_contract import record_sha256

        predictions = []
        scores = []
        for index, record in enumerate(records):
            source_hash = record_sha256(record)
            predictions.append(
                {
                    "index": index,
                    "source_record_sha256": source_hash,
                    "reached_generation_cap": index in {4, 7, 9},
                }
            )
            scores.append(
                {
                    "yes_no_index": index,
                    "source_record_sha256": source_hash,
                    "completion_sha256": str(index),
                    "v3_extracted_answer": "yes" if index == 7 else None,
                }
            )
        selected, audit = select_records(records, predictions, scores)
        self.assertEqual([row["question"] for row in selected], ["4", "9"])
        self.assertEqual([row["yes_no_index"] for row in audit], [4, 9])


if __name__ == "__main__":
    unittest.main()
