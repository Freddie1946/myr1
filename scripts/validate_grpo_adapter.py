#!/usr/bin/env python3
"""Validate a PathMMU GRPO JSON adapter on the runtime host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    records = json.loads(args.json_path.read_text(encoding="utf-8"))
    if len(records) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} records, found {len(records)}")
    missing_images = []
    for index, record in enumerate(records):
        for field in ("image", "problem", "solution"):
            if not str(record.get(field, "")).strip():
                raise ValueError(f"record {index} has empty/missing {field}")
        if not Path(record["image"]).is_file():
            missing_images.append(record["image"])
    if missing_images:
        raise FileNotFoundError(f"missing {len(missing_images)} images: {missing_images[:3]}")
    print(f"GRPO adapter validation: PASS ({len(records)} records, all images present)")


if __name__ == "__main__":
    main()
