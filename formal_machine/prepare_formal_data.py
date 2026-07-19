#!/usr/bin/env python3
"""Rewrite image paths and build formal LLaMA-Factory/GRPO adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


BASE_SPLIT_SPECS = {
    "sft_0500": "subsets/sft/sft_0500_with_cot.json",
    "sft_1000": "subsets/sft/sft_1000_with_cot.json",
    "sft_2000": "subsets/sft/sft_2000_with_cot.json",
    "sft_3000": "subsets/sft/sft_3000_with_cot.json",
    "rl_0250": "subsets/rl/rl_0250_with_cot.json",
    "rl_0500": "subsets/rl/rl_0500_with_cot.json",
    "rl_1000": "subsets/rl/rl_1000_with_cot.json",
    "validation_0385": "splits/validation_385_with_cot.json",
}
BASE_EXPECTED_COUNTS = {
    "sft_0500": 500,
    "sft_1000": 1000,
    "sft_2000": 2000,
    "sft_3000": 3000,
    "rl_0250": 250,
    "rl_0500": 500,
    "rl_1000": 1000,
    "validation_0385": 385,
}


def data_contract(split_root: Path) -> tuple[str, dict[str, str], dict[str, int], str]:
    manifest = json.loads((split_root / "manifest.json").read_text(encoding="utf-8"))
    version = manifest.get("version")
    specs = dict(BASE_SPLIT_SPECS)
    counts = dict(BASE_EXPECTED_COUNTS)
    if version == "pathmmu_image_disjoint_v2":
        test_key = "test_0999"
        specs[test_key] = "splits/test_0999_with_cot.json"
        counts[test_key] = 999
    elif version == "pathmmu_image_disjoint_v1":
        test_key = "test_1000"
        specs[test_key] = "splits/test_1000_with_cot.json"
        counts[test_key] = 1000
    else:
        raise ValueError(f"unsupported data version: {version}")
    return version, specs, counts, test_key


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"expected JSON list: {path}")
    return records


def image_target(original: str, image_root: Path) -> Path:
    return (image_root / Path(original).name).resolve()


def rewrite_records(records: list[dict], image_root: Path) -> list[dict]:
    rewritten = []
    missing = []
    for index, record in enumerate(records):
        for key in ("image", "problem", "solution"):
            if not str(record.get(key, "")).strip():
                raise ValueError(f"record {index} has empty/missing {key}")
        target = image_target(record["image"], image_root)
        if not target.is_file():
            missing.append(str(target))
        rewritten.append({**record, "image": str(target)})
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images; examples: {missing[:5]}")
    return rewritten


def lf_record(record: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"<image>{record['problem']}"},
            {"role": "assistant", "content": record["solution"]},
        ],
        "images": [record["image"]],
    }


def dataset_info_entry(filename: str) -> dict:
    return {
        "file_name": filename,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }


def image_set(records: list[dict]) -> set[str]:
    return {Path(record["image"]).name for record in records}


def emit_text(path: Path, content: str, reuse: bool, expected_files: set[Path]) -> None:
    resolved = path.resolve()
    expected_files.add(resolved)
    if reuse:
        if not path.is_file():
            raise FileNotFoundError(f"existing adapter output is missing: {path}")
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"existing adapter output differs from deterministic rebuild: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-if-valid", action="store_true")
    args = parser.parse_args()

    data_version, split_specs, expected_counts, test_key = data_contract(args.split_root)

    output_nonempty = args.output_root.exists() and any(args.output_root.iterdir())
    reuse_existing = output_nonempty and args.reuse_if_valid and not args.overwrite
    if output_nonempty and not args.overwrite and not reuse_existing:
        raise FileExistsError(
            f"output is not empty: {args.output_root}; use --reuse-if-valid or --overwrite intentionally"
        )
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    lf_dir = args.output_root / "llamafactory"
    grpo_dir = args.output_root / "grpo"
    records_dir = args.output_root / "rewritten_records"
    for directory in (lf_dir, grpo_dir, records_dir):
        if not reuse_existing:
            directory.mkdir(parents=True, exist_ok=True)

    rewritten: dict[str, list[dict]] = {}
    sources = {}
    expected_files: set[Path] = set()
    for name, relative in split_specs.items():
        source = (args.split_root / relative).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        records = rewrite_records(load_records(source), args.image_root)
        expected = expected_counts[name]
        if len(records) != expected:
            raise ValueError(f"{name}: expected {expected}, found {len(records)}")
        rewritten[name] = records
        sources[name] = {"path": str(source), "sha256": sha256(source), "count": len(records)}
        emit_text(
            records_dir / f"{name}.json",
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            reuse_existing,
            expected_files,
        )

    # Verify nestedness.
    for small, large in (("sft_0500", "sft_1000"), ("sft_1000", "sft_2000"), ("sft_2000", "sft_3000"),
                         ("rl_0250", "rl_0500"), ("rl_0500", "rl_1000")):
        small_keys = [(r["image"], r["problem"], r["solution"]) for r in rewritten[small]]
        large_keys = {(r["image"], r["problem"], r["solution"]) for r in rewritten[large]}
        if not set(small_keys).issubset(large_keys):
            raise ValueError(f"nestedness failed: {small} is not a subset of {large}")

    # Verify image-disjoint top-level partitions.
    partitions = {
        "sft": image_set(rewritten["sft_3000"]),
        "rl": image_set(rewritten["rl_1000"]),
        "validation": image_set(rewritten["validation_0385"]),
        "test": image_set(rewritten[test_key]),
    }
    overlaps = {}
    names = list(partitions)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            count = len(partitions[left].intersection(partitions[right]))
            overlaps[f"{left}__{right}"] = count
            if count:
                raise ValueError(f"image leakage: {left} and {right} overlap by {count}")

    dataset_info = {}
    for size in (500, 1000, 2000, 3000):
        key = f"sft_{size:04d}"
        dataset_name = f"pathvlm_sft_n{size:04d}"
        filename = f"{dataset_name}.json"
        emit_text(
            lf_dir / filename,
            json.dumps([lf_record(r) for r in rewritten[key]], ensure_ascii=False, indent=2) + "\n",
            reuse_existing,
            expected_files,
        )
        dataset_info[dataset_name] = dataset_info_entry(filename)

    for key, dataset_name in (("validation_0385", "pathvlm_validation_n0385"),
                              (test_key, f"pathvlm_test_n{expected_counts[test_key]:04d}")):
        filename = f"{dataset_name}.json"
        emit_text(
            lf_dir / filename,
            json.dumps([lf_record(r) for r in rewritten[key]], ensure_ascii=False, indent=2) + "\n",
            reuse_existing,
            expected_files,
        )
        dataset_info[dataset_name] = dataset_info_entry(filename)

    smoke_name = "pathvlm_sft_smoke_n0008"
    smoke_file = f"{smoke_name}.json"
    emit_text(
        lf_dir / smoke_file,
        json.dumps([lf_record(r) for r in rewritten["sft_0500"][:8]], ensure_ascii=False, indent=2) + "\n",
        reuse_existing,
        expected_files,
    )
    dataset_info[smoke_name] = dataset_info_entry(smoke_file)
    emit_text(
        lf_dir / "dataset_info.json",
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        reuse_existing,
        expected_files,
    )

    for size in (250, 500, 1000):
        key = f"rl_{size:04d}"
        data_file = grpo_dir / f"pathvlm_rl_n{size:04d}.json"
        yaml_file = grpo_dir / f"pathvlm_rl_n{size:04d}.yaml"
        emit_text(
            data_file,
            json.dumps(rewritten[key], ensure_ascii=False, indent=2) + "\n",
            reuse_existing,
            expected_files,
        )
        emit_text(
            yaml_file,
            f"datasets:\n  - json_path: {data_file.resolve()}\n    sampling_strategy: all\n",
            reuse_existing,
            expected_files,
        )
    smoke_grpo = grpo_dir / "pathvlm_rl_smoke_n0008.json"
    emit_text(
        smoke_grpo,
        json.dumps(rewritten["rl_0250"][:8], ensure_ascii=False, indent=2) + "\n",
        reuse_existing,
        expected_files,
    )
    emit_text(
        grpo_dir / "pathvlm_rl_smoke_n0008.yaml",
        f"datasets:\n  - json_path: {smoke_grpo.resolve()}\n    sampling_strategy: all\n",
        reuse_existing,
        expected_files,
    )

    manifest = {
        "data_version": data_version,
        "picked_json_used": False,
        "image_root": str(args.image_root.resolve()),
        "sources": sources,
        "image_counts": {name: len(images) for name, images in partitions.items()},
        "pairwise_image_overlaps": overlaps,
        "llamafactory_dataset_info": str((lf_dir / "dataset_info.json").resolve()),
        "test_policy": "evaluation_only_never_training_or_selection",
    }
    manifest_path = args.output_root / "formal_data_manifest.json"
    emit_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        reuse_existing,
        expected_files,
    )
    if reuse_existing:
        actual_files = {path.resolve() for path in args.output_root.rglob("*") if path.is_file()}
        missing = sorted(str(path) for path in expected_files - actual_files)
        unexpected = sorted(str(path) for path in actual_files - expected_files)
        if missing or unexpected:
            raise ValueError(
                "existing adapter output file set differs from deterministic rebuild; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        print(f"validated and reused {len(expected_files)} deterministic adapter files")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
