#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_visual_fidelity_run import verify_run


class VerifyVisualFidelityRunTests(unittest.TestCase):
    def make_run(self, root: Path) -> tuple[Path, Path]:
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        run = root / "run"
        figures = run / "figures"
        figures.mkdir(parents=True)
        rows = [
            {
                "panel_index": index,
                "source_record_sha256": f"{index:064x}",
                "patch_importance_log_target_probability_drop": [0.0] * 36,
            }
            for index in range(24)
        ]
        cases = run / "case_results.jsonl"
        cases.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        for index in range(24):
            (figures / f"case_{index:02d}_heatmap.png").write_bytes(b"png")
        (figures / "aggregate_deletion_insertion_curves.png").write_bytes(b"png")
        metrics = {
            "status": "completed",
            "method": "patch_occlusion_target_option_probability_fidelity_v1",
            "attention_claim": False,
            "selection_uses_model_outputs": False,
            "model_label": "arm",
            "model_path": str(model.resolve()),
            "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            "panel_sha256": "a" * 64,
            "case_count": 24,
            "grid_rows": 6,
            "grid_columns": 6,
            "random_permutations": 5,
            "case_results": str(cases.resolve()),
            "case_results_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
        }
        metrics_path = run / "metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        return model, metrics_path

    def test_accepts_complete_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model, metrics = self.make_run(Path(temporary))
            result = verify_run(
                metrics,
                expected_label="arm",
                expected_model=model,
                expected_panel_sha256="a" * 64,
            )
            self.assertEqual(result["case_count"], 24)
            self.assertEqual(result["figure_count"], 25)

    def test_rejects_modified_case_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model, metrics = self.make_run(Path(temporary))
            (metrics.parent / "case_results.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_run(
                    metrics,
                    expected_label="arm",
                    expected_model=model,
                    expected_panel_sha256="a" * 64,
                )


if __name__ == "__main__":
    unittest.main()
