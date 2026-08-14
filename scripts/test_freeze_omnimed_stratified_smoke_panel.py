#!/usr/bin/env python3

from __future__ import annotations

import unittest

from freeze_omnimed_stratified_smoke_panel import EXPECTED_SOURCES, freeze


class FreezeOmniMedSmokePanelTests(unittest.TestCase):
    def test_deterministic_and_balanced(self) -> None:
        records = []
        for source in EXPECTED_SOURCES:
            for index in range(6):
                records.append({"dataset": source, "question_id": f"{source}-{index}"})
        forward = freeze(records, 3)
        reverse = freeze(list(reversed(records)), 3)
        self.assertEqual(forward, reverse)
        self.assertEqual(len(forward), 12)
        self.assertEqual(
            {source: sum(row["dataset"] == source for row in forward) for source in EXPECTED_SOURCES},
            {source: 3 for source in EXPECTED_SOURCES},
        )

    def test_rejects_missing_source(self) -> None:
        records = [
            {"dataset": source, "question_id": source}
            for source in EXPECTED_SOURCES[:-1]
        ]
        with self.assertRaisesRegex(ValueError, "unexpected OmniMedVQA sources"):
            freeze(records, 1)


if __name__ == "__main__":
    unittest.main()
