#!/usr/bin/env python3
"""Create an audited, reindexed PathVQA yes/no-only prediction file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from external_vqa_contract import normalize_short_answer, sha256_file


def select(rows: list[dict]) -> list[dict]:
    selected = [
        row for row in rows
        if row.get("answer_type") == "yes_no"
        or normalize_short_answer(str(row.get("answer") or "")) in {"yes", "no"}
    ]
    if len(selected) != 3362:
        raise ValueError(f"expected 3362 PathVQA yes/no rows, got {len(selected)}")
    output = []
    for yes_no_index, row in enumerate(selected):
        item = dict(row)
        item["source_index"] = row.get("index")
        item["index"] = yes_no_index
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    selected = select(rows)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "predictions.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "selection": "PathVQA yes/no only",
        "count": len(selected),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "count": len(selected), "output": str(output.resolve())}))


if __name__ == "__main__":
    main()
