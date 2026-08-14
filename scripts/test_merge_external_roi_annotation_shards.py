import json
import tempfile
import unittest
from pathlib import Path

from merge_external_roi_annotation_shards import choose_rows


class MergeExternalAnnotationTest(unittest.TestCase):
    def test_valid_repair_replaces_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"; b = Path(tmp) / "b.jsonl"
            a.write_text(json.dumps({"panel_index": 3, "status": "failed"}) + "\n")
            b.write_text(json.dumps({"panel_index": 3, "status": "validated", "annotation": {"x": 1}}) + "\n")
            rows, hashes = choose_rows([a, b])
            self.assertEqual(rows[3]["status"], "validated")
            self.assertEqual(set(hashes), {str(a.resolve()), str(b.resolve())})

    def test_conflicting_valid_rows_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"; b = Path(tmp) / "b.jsonl"
            a.write_text(json.dumps({"panel_index": 3, "status": "validated", "annotation": {"x": 1}}) + "\n")
            b.write_text(json.dumps({"panel_index": 3, "status": "validated", "annotation": {"x": 2}}) + "\n")
            with self.assertRaises(ValueError):
                choose_rows([a, b])


if __name__ == "__main__":
    unittest.main()
