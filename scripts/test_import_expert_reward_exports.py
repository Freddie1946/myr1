import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from import_expert_reward_exports import EVENTS, import_archive


class ImportExpertRewardExportsTest(unittest.TestCase):
    def make_archive(self, root: Path, count: int = 60) -> Path:
        reviewer = "pathologist_A"
        prefix = f"pathvlm_reward_review_{reviewer}"
        files = {}
        events = {event: (index % 2 == 0) for index, event in enumerate(EVENTS)}
        for number in range(1, count + 1):
            case_id = f"HRA-{number:03d}"
            files[f"{case_id}.json"] = (json.dumps({
                "reviewer_id": reviewer, "case_id": case_id, "events": events,
            }) + "\n").encode()
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["reviewer_id", "case_id", *EVENTS])
        files["reward_six_event_ratings.csv"] = csv_buffer.getvalue().encode()
        records = [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()]
        manifest = {
            "reviewer_id": reviewer, "completed_cases": count, "expected_cases": 60,
            "complete": count == 60, "files": records,
        }
        path = root / "ratings.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in files.items():
                archive.writestr(f"{prefix}/{name}", data)
            archive.writestr(f"{prefix}/EXPORT_MANIFEST.json", json.dumps(manifest))
        return path

    def test_complete_archive_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = import_archive(self.make_archive(root), root / "out", False)
            self.assertEqual(result["completed_cases"], 60)
            self.assertTrue((root / "out/pathologist_A/HRA-060.json").is_file())

    def test_partial_requires_explicit_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.make_archive(root, 3)
            with self.assertRaises(ValueError):
                import_archive(archive, root / "out", False)
            result = import_archive(archive, root / "out", True)
            self.assertEqual(result["completed_cases"], 3)


if __name__ == "__main__":
    unittest.main()
