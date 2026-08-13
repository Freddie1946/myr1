import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "audit_final_nonhuman", Path(__file__).with_name("audit_final_nonhuman_data_completion.py")
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FinalCompletionAuditTests(unittest.TestCase):
    def test_completed_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps({"status": "completed", "count": 999}))
            self.assertEqual(MODULE.completed_metrics(path, 999), (True, "completed"))

    def test_wrong_count_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps({"status": "completed", "count": 998}))
            ok, detail = MODULE.completed_metrics(path, 999)
            self.assertFalse(ok)
            self.assertIn("expected=999", detail)


if __name__ == "__main__":
    unittest.main()
