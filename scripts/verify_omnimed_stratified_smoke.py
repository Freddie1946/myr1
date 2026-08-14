#!/usr/bin/env python3
"""Gate one model's stratified OmniMedVQA smoke before its full evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SOURCES = {
    "Chest CT Scan": 16,
    "Diabetic Retinopathy": 16,
    "ISIC2020": 16,
    "Retinal OCT-C8": 16,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"prediction line {line_number} is not an object")
            rows.append(value)
    return rows


def output_faults(text: str) -> list[str]:
    faults: list[str] = []
    if "\ufffd" in text:
        faults.append("unicode_replacement_character")
    controls = sorted(
        {
            f"U+{ord(char):04X}"
            for char in text
            if unicodedata.category(char) in {"Cc", "Cf"} and char not in "\n\r\t"
        }
    )
    if controls:
        faults.append("unicode_control_or_format:" + ",".join(controls))
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens = re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE)
    if re.search(r"(\b\w+\b)(?:\s+\1){7,}", normalized, re.I):
        faults.append("single_token_loop")
    # Flag only adjacent phrase loops. Ordinary medical explanations often
    # reuse bigrams such as ``the mass`` several times non-consecutively.
    for width in range(2, 13):
        repeats_required = max(4, (20 + width - 1) // width)
        for start in range(0, len(tokens) - width * repeats_required + 1):
            unit = tokens[start : start + width]
            if all(
                tokens[start + repeat * width : start + (repeat + 1) * width] == unit
                for repeat in range(1, repeats_required)
            ):
                faults.append(f"adjacent_repeated_{width}gram")
                break
        if faults and faults[-1].startswith("adjacent_repeated_"):
            break
    return faults


def verify(
    metrics_path: Path,
    predictions_path: Path,
    *,
    expected_data_sha256: str,
    expected_model_config_sha256: str,
    minimum_strict_final_coverage: float = 0.95,
    maximum_cap_hit_count: int = 1,
    maximum_empty_count: int = 0,
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = load_jsonl(predictions_path)
    expected = {
        "status": "completed",
        "task": "omnimedvqa",
        "split_role": "adapter_smoke",
        "count": 64,
        "generation_contract": "omnimed_domain_think_answer_v4_1024",
        "max_new_tokens": 1024,
        "data_sha256": expected_data_sha256,
        "model_config_sha256": expected_model_config_sha256,
        "predictions_sha256": sha256_file(predictions_path),
    }
    mismatches = {
        key: {"expected": value, "actual": metrics.get(key)}
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise ValueError(f"metrics contract mismatch: {mismatches}")
    if len(rows) != 64:
        raise ValueError(f"expected 64 predictions, got {len(rows)}")
    if [row.get("index") for row in rows] != list(range(64)):
        raise ValueError("prediction indices are not contiguous 0..63")
    sources = Counter(str(row.get("dataset")) for row in rows)
    if sources != Counter(EXPECTED_SOURCES):
        raise ValueError(f"source coverage mismatch: {dict(sources)}")

    empty = [row["index"] for row in rows if not str(row.get("completion") or "").strip()]
    cap_hits = [row["index"] for row in rows if bool(row.get("reached_generation_cap"))]
    unparseable = [
        row["index"] for row in rows if not bool(row.get("strict_final_answer_available"))
    ]
    faults = {
        int(row["index"]): output_faults(str(row.get("completion") or ""))
        for row in rows
    }
    faults = {index: values for index, values in faults.items() if values}
    coverage = (64 - len(unparseable)) / 64
    failures: dict[str, Any] = {}
    if len(empty) > maximum_empty_count:
        failures["empty_completions"] = {
            "maximum": maximum_empty_count,
            "actual": len(empty),
            "indices": empty,
        }
    if len(cap_hits) > maximum_cap_hit_count:
        failures["generation_cap_hits"] = {
            "maximum": maximum_cap_hit_count,
            "actual": len(cap_hits),
            "indices": cap_hits,
        }
    if coverage < minimum_strict_final_coverage:
        failures["strict_final_coverage"] = {
            "minimum": minimum_strict_final_coverage,
            "actual": coverage,
            "unparseable_indices": unparseable,
        }
    if faults:
        failures["garbled_or_repetitive_outputs"] = faults
    if failures:
        raise ValueError(f"OmniMedVQA smoke failed: {failures}")

    return {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_result": False,
        "task": "omnimedvqa",
        "split_role": "adapter_smoke",
        "count": 64,
        "source_counts": dict(sorted(sources.items())),
        "strict_final_answer_coverage": coverage,
        "strict_final_unparseable_count": len(unparseable),
        "generation_cap_hit_count": len(cap_hits),
        "empty_completion_count": len(empty),
        "garbled_or_repetitive_output_count": len(faults),
        "accuracy_used_as_gate": False,
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": expected["predictions_sha256"],
        "data_sha256": expected_data_sha256,
        "model_config_sha256": expected_model_config_sha256,
        "thresholds": {
            "minimum_strict_final_coverage": minimum_strict_final_coverage,
            "maximum_cap_hit_count": maximum_cap_hit_count,
            "maximum_empty_completion_count": maximum_empty_count,
            "maximum_garbled_or_repetitive_output_count": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-model-config-sha256", required=True)
    parser.add_argument("--minimum-strict-final-coverage", type=float, default=0.95)
    parser.add_argument("--maximum-cap-hit-count", type=int, default=1)
    parser.add_argument("--maximum-empty-count", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(
            args.metrics,
            args.predictions,
            expected_data_sha256=args.expected_data_sha256,
            expected_model_config_sha256=args.expected_model_config_sha256,
            minimum_strict_final_coverage=args.minimum_strict_final_coverage,
            maximum_cap_hit_count=args.maximum_cap_hit_count,
            maximum_empty_count=args.maximum_empty_count,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"OmniMedVQA stratified smoke verification failed: {error}") from error
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, args.output)
    print(text, end="")


if __name__ == "__main__":
    main()
