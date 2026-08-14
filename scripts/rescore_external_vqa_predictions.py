#!/usr/bin/env python3
"""Target-blind rescore of frozen external-VQA generations after parser fixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from external_vqa_contract import record_sha256, score_record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config_path = run / "run_config.json"
    metrics_path = run / "metrics.json"
    predictions_path = run / "predictions.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if config.get("task") != "omnimedvqa" or metrics.get("task") != "omnimedvqa":
        raise ValueError("this rescoring pass is restricted to OmniMedVQA")
    data = json.loads(Path(config["data_path"]).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(data):
        raise ValueError(f"prediction/data count mismatch: {len(rows)} != {len(data)}")
    for index, (row, record) in enumerate(zip(rows, data)):
        if row.get("index") != index or row.get("source_record_sha256") != record_sha256(record):
            raise ValueError(f"row/source mismatch at {index}")
        row.update(score_record("omnimedvqa", row["completion"], record))
    atomic_text(
        predictions_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    strict_available = sum(row["strict_final_answer_available"] for row in rows)
    strict_correct = sum(row["strict_final_correct"] for row in rows)
    aligned_correct = sum(row["contract_aligned_correct"] for row in rows)
    metrics.update({
        "predictions_sha256": sha256_file(predictions_path),
        "strict_final_answer_available": strict_available,
        "strict_final_answer_coverage": strict_available / len(rows),
        "strict_final_unresolved_count": len(rows) - strict_available,
        "strict_final_correct": strict_correct,
        "strict_final_accuracy": strict_correct / len(rows),
        "contract_aligned_correct": aligned_correct,
        "contract_aligned_accuracy": aligned_correct / len(rows),
        "scorer_rescored_at": datetime.now(timezone.utc).isoformat(),
        "scorer_rescore_method": "target_blind_external_vqa_contract",
    })
    by_source = {}
    for source in sorted(Counter(row["dataset"] for row in rows)):
        selected = [row for row in rows if row["dataset"] == source]
        by_source[source] = {
            "count": len(selected),
            "strict_final_answer_available": sum(row["strict_final_answer_available"] for row in selected),
            "strict_final_answer_coverage": sum(row["strict_final_answer_available"] for row in selected) / len(selected),
            "strict_final_correct": sum(row["strict_final_correct"] for row in selected),
            "strict_final_accuracy": sum(row["strict_final_correct"] for row in selected) / len(selected),
            "contract_aligned_correct": sum(row["contract_aligned_correct"] for row in selected),
            "contract_aligned_accuracy": sum(row["contract_aligned_correct"] for row in selected) / len(selected),
        }
    metrics["by_source"] = by_source
    atomic_text(metrics_path, json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    audit = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "generation_modified": False,
        "individual_answers_rejudged": False,
        "target_blind_parser_only": True,
        "count": len(rows),
        "strict_final_answer_coverage": strict_available / len(rows),
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    atomic_text(run / "rescore_audit.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
