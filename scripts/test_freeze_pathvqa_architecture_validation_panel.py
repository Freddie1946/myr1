#!/usr/bin/env python3

import unittest

from freeze_pathvqa_architecture_validation_panel import select_panel


class ArchitecturePanelTests(unittest.TestCase):
    def test_balanced_unique_and_excluded(self):
        rows = []
        for index in range(12):
            rows.append({
                "source_index": index,
                "image": f"image-{index}",
                "image_sha256": f"sha-{index}",
                "question": f"q-{index}",
                "answer": "yes" if index % 2 == 0 else "no",
            })
        selected = select_panel(rows, {0, 1}, 3)
        self.assertEqual(len(selected), 6)
        self.assertEqual(sum(row["answer"] == "yes" for row in selected), 3)
        self.assertEqual(sum(row["answer"] == "no" for row in selected), 3)
        self.assertFalse({row["source_index"] for row in selected} & {0, 1})
        self.assertEqual(len({row["image_sha256"] for row in selected}), 6)


if __name__ == "__main__":
    unittest.main()
