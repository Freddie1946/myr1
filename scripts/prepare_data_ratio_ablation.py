#!/usr/bin/env python3
"""Build deterministic, image-disjoint adapters for the sparse data-ratio ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"<answer>\s*([A-D])(?:\)|\b)", re.IGNORECASE)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_record_key(record: dict[str, Any]) -> str:
    payload = "\0".join(
        (Path(record["image"]).name, str(record["problem"]), str(record["solution"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def answer_letter(record: dict[str, Any]) -> str:
    match = ANSWER_RE.search(str(record["solution"]))
    if not match:
        raise ValueError(f"cannot parse answer from record {canonical_record_key(record)}")
    return match.group(1).upper()


def load_source(repo: Path, spec: dict[str, Any], image_root: Path) -> list[dict[str, Any]]:
    path = (repo / spec["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if actual_hash != spec["sha256"]:
        raise ValueError(f"source hash mismatch for {path}: {actual_hash}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != int(spec["count"]):
        raise ValueError(f"source count mismatch for {path}")
    rewritten = []
    for index, row in enumerate(rows):
        if not all(str(row.get(key, "")).strip() for key in ("image", "problem", "solution")):
            raise ValueError(f"source row {index} is incomplete: {path}")
        image = (image_root / Path(row["image"]).name).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        rewritten.append({"image": str(image), "problem": row["problem"], "solution": row["solution"]})
    if len({canonical_record_key(row) for row in rewritten}) != len(rewritten):
        raise ValueError(f"source contains duplicate records: {path}")
    return rewritten


def group_by_image(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[Path(row["image"]).name].append(row)
    return dict(groups)


def exact_group_subset(
    groups: dict[str, list[dict[str, Any]]], candidates: list[str], target: int, salt: str
) -> set[str]:
    ordered = sorted(
        candidates,
        key=lambda image: hashlib.sha256(f"{salt}:{image}".encode("utf-8")).hexdigest(),
    )
    choices: dict[int, tuple[str, ...]] = {0: ()}
    for image in ordered:
        weight = len(groups[image])
        for total in sorted(tuple(choices), reverse=True):
            new_total = total + weight
            if new_total <= target and new_total not in choices:
                choices[new_total] = choices[total] + (image,)
        if target in choices:
            return set(choices[target])
    raise ValueError(f"cannot make an image-complete subset of exactly {target} QA")


def partition_source(rows: list[dict[str, Any]], source: str, seed: int) -> list[list[dict[str, Any]]]:
    groups = group_by_image(rows)
    remaining = list(groups)
    image_bins: list[set[str]] = []
    for bin_index in range(3):
        remaining_rows = [row for image in remaining for row in groups[image]]
        remaining_answers = Counter(answer_letter(row) for row in remaining_rows)
        bins_left = 4 - bin_index
        target_answers = {
            letter: remaining_answers[letter] / bins_left for letter in "ABCD"
        }
        candidates = []
        for trial in range(512):
            selected_trial = exact_group_subset(
                groups,
                remaining,
                125,
                f"{seed}:{source}:{bin_index + 1}:trial{trial}",
            )
            counts = Counter(
                answer_letter(row)
                for image in selected_trial
                for row in groups[image]
            )
            score = sum((counts[letter] - target_answers[letter]) ** 2 for letter in "ABCD")
            tie = hashlib.sha256("\0".join(sorted(selected_trial)).encode("utf-8")).hexdigest()
            candidates.append((score, tie, selected_trial))
        selected = min(candidates, key=lambda item: (item[0], item[1]))[2]
        image_bins.append(selected)
        remaining = [image for image in remaining if image not in selected]
    if sum(len(groups[image]) for image in remaining) != 125:
        raise ValueError(f"last {source} bin is not exactly 125 QA")
    image_bins.append(set(remaining))
    result = []
    for index, images in enumerate(image_bins, start=1):
        selected_rows = [row for row in rows if Path(row["image"]).name in images]
        selected_rows.sort(key=lambda row: canonical_record_key(row))
        if len(selected_rows) != 125:
            raise AssertionError(f"{source} bin {index} count mismatch")
        result.append(selected_rows)
    if set().union(*image_bins) != set(groups) or sum(map(len, image_bins)) != len(groups):
        raise AssertionError(f"{source} image partition is not exhaustive and disjoint")
    return result


def lf_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"<image>{record['problem']}"},
            {"role": "assistant", "content": record["solution"]},
        ],
        "images": [record["image"]],
    }


def dataset_info_entry(filename: str) -> dict[str, Any]:
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


def json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "qa_count": len(rows),
        "image_count": len({Path(row["image"]).name for row in rows}),
        "answer_counts": dict(sorted(Counter(answer_letter(row) for row in rows).items())),
        "records_sha256": sha256_bytes(json_text(rows).encode("utf-8")),
    }


def add_json(files: dict[str, str], relative: str, payload: Any) -> None:
    files[relative] = json_text(payload)


def build_files(repo: Path, image_root: Path, protocol: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    specs = protocol["source_files"]
    source_rows = {
        name: load_source(repo, specs[name], image_root)
        for name in ("sft_0500", "rl_0500", "sft_3000", "rl_1000")
    }
    if {Path(row["image"]).name for row in source_rows["sft_3000"]}.intersection(
        Path(row["image"]).name for row in source_rows["rl_1000"]
    ):
        raise ValueError("frozen SFT3000 and RL1000 image pools overlap")

    seed = int(protocol["seed"])
    sft_bins = partition_source(source_rows["sft_0500"], "sft", seed)
    rl_bins = partition_source(source_rows["rl_0500"], "rl", seed)
    bins = [sorted(sft_bins[i] + rl_bins[i], key=canonical_record_key) for i in range(4)]
    if any(len(rows) != 250 for rows in bins):
        raise AssertionError("combined mini-pool bins must each contain 250 QA")
    common_pool = [row for rows in bins for row in rows]
    if len({canonical_record_key(row) for row in common_pool}) != 1000:
        raise ValueError("common mini-pool is not 1000 unique records")

    definitions = {
        "sft0250_rl0750": ([0], [1, 2, 3]),
        "sft0500_rl0500": ([0, 1], [2, 3]),
        "sft0750_rl0250": ([0, 1, 2], [3]),
    }
    arms: dict[str, dict[str, Any]] = {}
    files: dict[str, str] = {}
    dataset_info: dict[str, Any] = {}
    add_json(files, "records/common_pool_1000.json", common_pool)
    provenance_rows = []
    for bin_index, rows in enumerate(bins, start=1):
        add_json(files, f"records/bin_{bin_index:02d}_n0250.json", rows)
        sft_keys = {canonical_record_key(row) for row in sft_bins[bin_index - 1]}
        for row in rows:
            provenance_rows.append({
                "record_sha256": canonical_record_key(row),
                "bin": bin_index,
                "original_source": "sft_0500" if canonical_record_key(row) in sft_keys else "rl_0500",
                "image_basename": Path(row["image"]).name,
                "answer": answer_letter(row),
            })
    add_json(files, "records/provenance_index.json", provenance_rows)

    for arm, (sft_indices, rl_indices) in definitions.items():
        sft_rows = [row for index in sft_indices for row in bins[index]]
        rl_rows = [row for index in rl_indices for row in bins[index]]
        sft_images = {Path(row["image"]).name for row in sft_rows}
        rl_images = {Path(row["image"]).name for row in rl_rows}
        if sft_images.intersection(rl_images):
            raise ValueError(f"{arm} has SFT/RL image overlap")
        if {canonical_record_key(row) for row in sft_rows}.intersection(
            canonical_record_key(row) for row in rl_rows
        ):
            raise ValueError(f"{arm} has SFT/RL record overlap")
        sft_name = f"pathvlm_ratio_{arm}_sft"
        sft_filename = f"{sft_name}.json"
        add_json(files, f"llamafactory/{sft_filename}", [lf_record(row) for row in sft_rows])
        dataset_info[sft_name] = dataset_info_entry(sft_filename)
        rl_relative = f"grpo/pathvlm_ratio_{arm}_rl.json"
        add_json(files, rl_relative, rl_rows)
        files[f"grpo/pathvlm_ratio_{arm}_rl.yaml"] = (
            "datasets:\n"
            f"  - json_path: __OUTPUT_ROOT__/{rl_relative}\n"
            "    sampling_strategy: all\n"
        )
        arms[arm] = {
            "sft": {**stats(sft_rows), "dataset_name": sft_name},
            "rl": {**stats(rl_rows), "dataset_yaml": f"grpo/pathvlm_ratio_{arm}_rl.yaml"},
            "sft_bins": [index + 1 for index in sft_indices],
            "rl_bins": [index + 1 for index in rl_indices],
            "image_overlap": 0,
            "record_overlap": 0,
        }
        smoke_sft = sft_rows[:8]
        smoke_rl = rl_rows[:8]
        smoke_name = f"pathvlm_ratio_smoke_{arm}_sft"
        smoke_filename = f"{smoke_name}.json"
        add_json(files, f"llamafactory/{smoke_filename}", [lf_record(row) for row in smoke_sft])
        dataset_info[smoke_name] = dataset_info_entry(smoke_filename)
        smoke_rl_relative = f"grpo/pathvlm_ratio_smoke_{arm}_rl.json"
        add_json(files, smoke_rl_relative, smoke_rl)
        files[f"grpo/pathvlm_ratio_smoke_{arm}_rl.yaml"] = (
            "datasets:\n"
            f"  - json_path: __OUTPUT_ROOT__/{smoke_rl_relative}\n"
            "    sampling_strategy: all\n"
        )

    all_4000 = sorted(source_rows["sft_3000"] + source_rows["rl_1000"], key=canonical_record_key)
    if len(all_4000) != 4000 or len({canonical_record_key(row) for row in all_4000}) != 4000:
        raise ValueError("full 4000-QA union is not unique and complete")
    add_json(files, "grpo/pathvlm_ratio_base_rule_rl4000.json", all_4000)
    files["grpo/pathvlm_ratio_base_rule_rl4000.yaml"] = (
        "datasets:\n"
        "  - json_path: __OUTPUT_ROOT__/grpo/pathvlm_ratio_base_rule_rl4000.json\n"
        "    sampling_strategy: all\n"
    )
    add_json(files, "grpo/pathvlm_ratio_smoke_base_rule_rl4000.json", all_4000[:8])
    files["grpo/pathvlm_ratio_smoke_base_rule_rl4000.yaml"] = (
        "datasets:\n"
        "  - json_path: __OUTPUT_ROOT__/grpo/pathvlm_ratio_smoke_base_rule_rl4000.json\n"
        "    sampling_strategy: all\n"
    )
    continuation = sorted(source_rows["rl_1000"], key=canonical_record_key)
    add_json(files, "grpo/pathvlm_ratio_stage2_continue_rule_rl1000.json", continuation)
    files["grpo/pathvlm_ratio_stage2_continue_rule_rl1000.yaml"] = (
        "datasets:\n"
        "  - json_path: __OUTPUT_ROOT__/grpo/pathvlm_ratio_stage2_continue_rule_rl1000.json\n"
        "    sampling_strategy: all\n"
    )
    add_json(files, "grpo/pathvlm_ratio_smoke_stage2_continue_rule_rl1000.json", continuation[:8])
    files["grpo/pathvlm_ratio_smoke_stage2_continue_rule_rl1000.yaml"] = (
        "datasets:\n"
        "  - json_path: __OUTPUT_ROOT__/grpo/pathvlm_ratio_smoke_stage2_continue_rule_rl1000.json\n"
        "    sampling_strategy: all\n"
    )
    add_json(files, "llamafactory/dataset_info.json", dataset_info)

    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "created_at": protocol["created_at"],
        "data_version": protocol["data_version"],
        "seed": seed,
        "common_pool": stats(common_pool),
        "bins": {str(i + 1): stats(rows) for i, rows in enumerate(bins)},
        "arms": arms,
        "base_rule_rl4000": stats(all_4000),
        "stage2_continue_rule_rl1000": stats(continuation),
        "source_files": {
            name: {**specs[name], "resolved_path": str((repo / specs[name]["path"]).resolve())}
            for name in specs
        },
        "gates": {
            "source_hashes_match": True,
            "source_counts_match": True,
            "common_pool_exactly_1000": True,
            "four_bins_exactly_250": True,
            "each_bin_has_125_from_each_source": True,
            "all_arm_sft_rl_image_overlaps_zero": True,
            "all_arm_sft_rl_record_overlaps_zero": True,
            "full_union_exactly_4000": True,
            "all_images_exist": True,
            "test_accessed": False,
        },
        "test_accessed": False,
    }
    return files, manifest


def materialize(output_root: Path, files: dict[str, str], manifest: dict[str, Any], reuse: bool) -> None:
    output_root = output_root.resolve()
    rendered = {
        relative: content.replace("__OUTPUT_ROOT__", str(output_root))
        for relative, content in files.items()
    }
    file_entries = {}
    for relative, content in sorted(rendered.items()):
        path = output_root / relative
        payload = content.encode("utf-8")
        if path.exists():
            if not reuse:
                raise FileExistsError(path)
            if path.read_bytes() != payload:
                raise ValueError(f"existing deterministic artifact differs: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
            temporary.write_bytes(payload)
            os.replace(temporary, path)
        file_entries[relative] = {"size_bytes": len(payload), "sha256": sha256_bytes(payload)}
    manifest = {**manifest, "output_root": str(output_root), "files": file_entries}
    manifest_text = json_text(manifest)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and reuse:
        if manifest_path.read_text(encoding="utf-8") != manifest_text:
            raise ValueError(f"existing manifest differs: {manifest_path}")
    elif manifest_path.exists():
        raise FileExistsError(manifest_path)
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest_text, encoding="utf-8")
    print(json.dumps({
        "status": "prepared",
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "file_count": len(file_entries),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-if-valid", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    if protocol_path.parent != (repo / "protocol").resolve():
        raise ValueError("protocol must be a checked-in file under repo/protocol")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("test_accessed") is not False:
        raise ValueError("protocol does not explicitly keep test inaccessible")
    files, manifest = build_files(repo, args.image_root.resolve(), protocol)
    materialize(args.output_root, files, manifest, args.reuse_if_valid)


if __name__ == "__main__":
    main()
