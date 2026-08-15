#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("build_stage2_stage3_human_review_packet.py")
SPEC = importlib.util.spec_from_file_location("review_packet", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class ReviewPacketTest(unittest.TestCase):
    def test_blinding_and_strata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "x.png"
            image.write_bytes(b"png")
            def row(index, is_correct, completion):
                return {"index": index, "source_record_sha256": f"h{index}", "image": str(image),
                        "problem": "Q", "solution": "R", "completion": completion,
                        "predicted_choice": "A" if is_correct else "B", "target_choice": "A",
                        "accuracy_reward": float(is_correct)}
            s2 = {0: row(0, False, "old0"), 1: row(1, True, "old1")}
            s3 = {0: row(0, True, "new0"), 1: row(1, False, "new1")}
            blind = MODULE.build_blind_panel(stage2=s2, stage3=s3, indices=[0, 1],
                                               output=root / "blind", seed=7, labels=("old", "new"))
            self.assertEqual(blind["case_count"], 2)
            review_text = (root / "blind/review_packet/case_001/review.html").read_text()
            self.assertNotIn("Stage2", review_text)
            improved = MODULE.build_improvements(stage2=s2, stage3=s3, output=root / "improved",
                                                  labels=("old", "new"))
            self.assertEqual(improved["wrong_to_right"], 1)
            self.assertEqual(improved["right_to_wrong"], 1)
            payload = json.loads(next((root / "improved").glob("improved_*/case.json")).read_text())
            self.assertEqual(payload["source_index"], 0)


if __name__ == "__main__":
    unittest.main()
