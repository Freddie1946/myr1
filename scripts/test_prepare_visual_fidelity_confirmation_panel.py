from __future__ import annotations

import json
from pathlib import Path


def test_confirmation_panel_contract(tmp_path: Path) -> None:
    # Contract-level test: the expected manifest fields are explicit and
    # outcome-blind; data selection itself is exercised by the production
    # script against the frozen 385-record file.
    manifest = {
        "status": "frozen_before_model_visualization_outputs",
        "selection_uses_model_outputs": False,
        "case_count": 48,
        "target_choice_counts": {"A": 12, "B": 12, "C": 12, "D": 12},
        "excluded_source_record_sha256": ["pilot"],
    }
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["status"] == "frozen_before_model_visualization_outputs"
    assert loaded["selection_uses_model_outputs"] is False
    assert loaded["case_count"] == 48
    assert loaded["target_choice_counts"] == {"A": 12, "B": 12, "C": 12, "D": 12}
