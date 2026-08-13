import importlib.util
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "build_final_results_tables", Path(__file__).with_name("build_final_results_tables.py")
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FinalResultsTableTests(unittest.TestCase):
    def test_metric_uses_first_available_named_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text('{"accuracy": 0.0, "contract_aligned_exact_accuracy": 0.625}\n')
            value, _ = MODULE.metric(path, ("contract_aligned_exact_accuracy", "accuracy"))
            self.assertEqual(value, 0.625)

    def test_missing_metric_is_na(self):
        value, data = MODULE.metric(Path("/definitely/missing"), ("accuracy",))
        self.assertIsNone(value)
        self.assertIsNone(data)
        self.assertEqual(MODULE.pct(value), "NA")

    def test_percent_render(self):
        self.assertEqual(MODULE.pct(0.634635), "63.46")


if __name__ == "__main__":
    unittest.main()
