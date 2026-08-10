#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from prepare_pathvqa_validation_calibration import build_records


class PreparePathvqaValidationCalibrationTests(unittest.TestCase):
    def test_build_records_deduplicates_images_and_types_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "validation.parquet"
            table = pa.table(
                {
                    "image": [
                        {"bytes": b"same-image", "path": "one.jpg"},
                        {"bytes": b"same-image", "path": "two.jpg"},
                    ],
                    "question": ["Is it present?", "What is shown?"],
                    "answer": ["Yes", "adenocarcinoma"],
                }
            )
            pq.write_table(table, parquet)
            records = build_records([parquet], root / "images")
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["answer_type"], "yes_no")
            self.assertEqual(records[1]["answer_type"], "free_form")
            self.assertEqual(records[0]["image_sha256"], records[1]["image_sha256"])
            self.assertEqual(len(list((root / "images").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
