import tempfile
import unittest
from pathlib import Path

from PIL import Image

from build_expert_review_browse_packets import annotated_image, scaled_box


class ExpertReviewPacketTest(unittest.TestCase):
    def test_scaled_box_uses_normalized_canvas(self):
        self.assertEqual(scaled_box([100, 200, 900, 800], 500, 250), (50, 50, 450, 200))

    def test_annotated_image_writes_file_and_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            target = root / "annotated.jpg"
            Image.new("RGB", (200, 100), "white").save(source)
            regions = annotated_image(source, target, [[100, 100, 500, 500], [600, 200, 900, 800]])
            self.assertTrue(target.is_file())
            self.assertEqual([row["region_id"] for row in regions], ["R1", "R2"])
            self.assertEqual(regions[0]["box_pixels"], [20, 10, 100, 50])

    def test_rejects_degenerate_box(self):
        with self.assertRaises(ValueError):
            scaled_box([100, 100, 100, 200], 500, 500)


if __name__ == "__main__":
    unittest.main()
