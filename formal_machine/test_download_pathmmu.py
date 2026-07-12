#!/usr/bin/env python3

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

import download_pathmmu


class DownloadPathMMUTest(unittest.TestCase):
    def test_extracts_only_frozen_images_from_nested_archive(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            split_root = root / "split"
            split_dir = split_root / "splits"
            split_dir.mkdir(parents=True)
            rows = [
                {"image": "/old/path/a.png", "problem": "p", "solution": "s"},
                {"image": "/old/path/b.jpg", "problem": "p", "solution": "s"},
            ]
            (split_dir / "sft_train_with_cot.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )

            sources = root / "sources"
            sources.mkdir()
            Image.new("RGB", (3, 2), "red").save(sources / "a.png")
            Image.new("RGB", (2, 3), "blue").save(sources / "b.jpg")
            Image.new("RGB", (1, 1), "green").save(sources / "unused.png")
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.write(sources / "a.png", "images/a.png")
                bundle.write(sources / "b.jpg", "other/nested/b.jpg")
                bundle.write(sources / "unused.png", "images/unused.png")

            required = download_pathmmu.required_image_names(split_root)
            output = root / "output"
            existing, extracted, missing = download_pathmmu.extract_required(
                archive, output, required, overwrite=False
            )
            self.assertEqual(existing, 0)
            self.assertEqual(extracted, 2)
            self.assertEqual(missing, [])
            self.assertEqual({p.name for p in output.iterdir()}, {"a.png", "b.jpg"})

            existing, extracted, missing = download_pathmmu.extract_required(
                archive, output, required, overwrite=False
            )
            self.assertEqual(existing, 2)
            self.assertEqual(extracted, 0)
            self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
