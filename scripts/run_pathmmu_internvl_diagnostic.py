#!/usr/bin/env python3
"""Run resumable InternVL PathMMU development diagnostics with the official chat interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from pathmmu_rewards import accuracy_reward, choice_letter, format_reward


QUESTION_TEMPLATE = (
    "<image>\n{question}\n"
    "Give concise image-grounded reasoning inside <think>...</think>. Then output exactly one "
    "option letter (A, B, C, or D) inside <answer>...</answer>."
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB")),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    width, height = image.size
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda value: value[0] * value[1],
    )
    target_ratio = find_closest_aspect_ratio(
        width / height, target_ratios, width, height, image_size
    )
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    resized = image.resize((target_width, target_height))
    images = []
    for index in range(target_ratio[0] * target_ratio[1]):
        left = (index % target_ratio[0]) * image_size
        upper = (index // target_ratio[0]) * image_size
        images.append(
            resized.crop((left, upper, left + image_size, upper + image_size))
        )
    if use_thumbnail and len(images) != 1:
        images.append(image.resize((image_size, image_size)))
    return images


def load_image(path: Path, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    transform = build_transform(input_size)
    with Image.open(path) as source:
        images = dynamic_preprocess(
            source.convert("RGB"),
            image_size=input_size,
            max_num=max_num,
            use_thumbnail=True,
        )
    return torch.stack([transform(image) for image in images])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_existing(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for index, row in enumerate(rows):
        if row.get("index") != index:
            raise ValueError(f"non-contiguous existing prediction at row {index}")
        if row.get("source_record_sha256") != record_sha256(records[index]):
            raise ValueError(f"source record changed at row {index}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split-role",
        required=True,
        choices=("validation_smoke", "test999_development"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens != 1024:
        raise ValueError("the approved diagnostic contract requires max_new_tokens=1024")
    records = json.loads(args.data.read_text(encoding="utf-8"))
    expected = 999 if args.split_role == "test999_development" else 385
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, got {len(records)}")
    if args.limit is not None:
        if args.split_role == "test999_development":
            raise ValueError("test999 development run must cover all 999 records")
        records = records[: args.limit]
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    config_path = args.output_dir / "run_config.json"
    if metrics_path.exists():
        raise FileExistsError(metrics_path)

    config = {
        "schema_version": 1,
        "status": "running",
        "backend": "internvl_official_chat",
        "split_role": args.split_role,
        "test_accessed": args.split_role == "test999_development",
        "diagnostic_only": args.split_role == "test999_development",
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "selected_count": len(records),
        "prompt_template": QUESTION_TEMPLATE,
        "dtype": "bfloat16",
        "quantization": "none",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "image_size": 448,
        "maximum_tiles": 12,
    }
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if {**old, "status": "running"} != config:
            raise ValueError("resume configuration differs")
    else:
        write_json(config_path, config)
    rows = load_existing(predictions_path, records)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False, local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
            use_flash_attn=False,
        )
        .eval()
        .cuda()
    )
    generation_config = {"max_new_tokens": 1024, "do_sample": False}

    for index in range(len(rows), len(records)):
        record = records[index]
        pixel_values = load_image(Path(record["image"])).to(torch.bfloat16).cuda()
        question = QUESTION_TEMPLATE.format(question=record["problem"])
        with torch.inference_mode():
            completion = model.chat(
                tokenizer, pixel_values, question, generation_config
            )
        token_count = len(tokenizer.encode(completion, add_special_tokens=False))
        wrapped = [[{"role": "assistant", "content": completion}]]
        row = {
            "index": index,
            "source_record_sha256": record_sha256(record),
            "image": record["image"],
            "problem": record["problem"],
            "solution": record["solution"],
            "completion": completion,
            "generated_token_count": token_count,
            "ended_with_eos": None,
            "reached_generation_cap": token_count >= 1024,
            "predicted_choice": choice_letter(completion),
            "target_choice": choice_letter(record["solution"]),
            "accuracy_reward": accuracy_reward(wrapped, [record["solution"]])[0],
            "format_reward": format_reward(wrapped)[0],
            "test999_development_diagnostic": args.split_role
            == "test999_development",
        }
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        rows.append(row)
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "total": len(records),
                        "correct": sum(int(item["accuracy_reward"]) for item in rows),
                    }
                ),
                flush=True,
            )

    lengths = [int(row["generated_token_count"]) for row in rows]
    metrics = {
        **config,
        "status": "completed",
        "eligible_for_untouched_final_stage3_claim": False
        if args.split_role == "test999_development"
        else None,
        "count": len(rows),
        "correct": sum(int(row["accuracy_reward"]) for row in rows),
        "accuracy": sum(float(row["accuracy_reward"]) for row in rows) / len(rows),
        "format_correct": sum(int(row["format_reward"]) for row in rows),
        "choice_extracted": sum(row["predicted_choice"] is not None for row in rows),
        "empty_completion_count": sum(not str(row["completion"]) for row in rows),
        "mean_generated_tokens": statistics.fmean(lengths),
        "median_generated_tokens": statistics.median(lengths),
        "maximum_generated_tokens": max(lengths),
        "generation_cap_hit_count": sum(
            bool(row["reached_generation_cap"]) for row in rows
        ),
        "predictions_sha256": sha256_file(predictions_path),
    }
    write_json(metrics_path, metrics)
    config["status"] = "completed"
    write_json(config_path, config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
