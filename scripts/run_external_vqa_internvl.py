#!/usr/bin/env python3
"""Resumable external VQA through InternVL's official chat interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from external_vqa_contract import (
    PATHVQA_PROMPT,
    OMNIMEDVQA_PROMPT,
    prompt_for_record,
    score_record,
    sha256_file,
)
from external_vqa_native_utils import (
    append_row,
    load_existing,
    result_row,
    summarize,
    write_json,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB")),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_ratio(
    aspect: float,
    ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best, difference, area = (1, 1), float("inf"), width * height
    for ratio in ratios:
        current = abs(aspect - ratio[0] / ratio[1])
        if current < difference or (
            current == difference
            and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best, difference = ratio, current
    return best


def load_image(path: Path, image_size: int = 448, max_tiles: int = 12) -> torch.Tensor:
    transform = build_transform(image_size)
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    ratios = sorted(
        {
            (i, j)
            for n in range(1, max_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_tiles
        },
        key=lambda value: value[0] * value[1],
    )
    ratio = find_ratio(width / height, ratios, width, height, image_size)
    resized = image.resize((image_size * ratio[0], image_size * ratio[1]))
    tiles = []
    for index in range(ratio[0] * ratio[1]):
        left = (index % ratio[0]) * image_size
        upper = (index // ratio[0]) * image_size
        tiles.append(resized.crop((left, upper, left + image_size, upper + image_size)))
    if len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return torch.stack([transform(tile) for tile in tiles])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("pathvqa", "omnimedvqa"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split-role", required=True, choices=("adapter_smoke", "external_test")
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens != 64:
        raise ValueError("frozen contract requires max_new_tokens=64")
    records = json.loads(args.data.read_text(encoding="utf-8"))
    expected = 6719 if args.task == "pathvqa" else 8518
    if len(records) != expected:
        raise ValueError(f"expected {expected}, got {len(records)}")
    if args.limit is not None:
        if args.split_role != "adapter_smoke" or not 1 <= args.limit <= len(records):
            raise ValueError("invalid smoke limit")
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
        "task": args.task,
        "split_role": args.split_role,
        "backend": "internvl_official_chat",
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "selected_count": len(records),
        "prompt_template": PATHVQA_PROMPT if args.task == "pathvqa" else OMNIMEDVQA_PROMPT,
        "dtype": "bfloat16",
        "quantization": "none",
        "do_sample": False,
        "max_new_tokens": 64,
        "image_size": 448,
        "maximum_tiles": 12,
    }
    if config_path.exists():
        if {**json.loads(config_path.read_text()), "status": "running"} != config:
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
    generation_config = {"max_new_tokens": 64, "do_sample": False}
    for index in range(len(rows), len(records)):
        record = records[index]
        pixels = load_image(Path(record["image"])).to(torch.bfloat16).cuda()
        question = "<image>\n" + prompt_for_record(args.task, record)
        with torch.inference_mode():
            completion = model.chat(tokenizer, pixels, question, generation_config)
        completion = completion.strip()
        token_count = len(tokenizer.encode(completion, add_special_tokens=False))
        row = result_row(
            index=index,
            task=args.task,
            record=record,
            completion=completion,
            token_count=token_count,
            ended_with_eos=None,
            score=score_record(args.task, completion, record),
            max_new_tokens=64,
        )
        append_row(predictions_path, row)
        rows.append(row)
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            key = "exact_match" if args.task == "pathvqa" else "official_most_similar_correct"
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "total": len(records),
                        "correct": sum(item[key] for item in rows),
                    }
                ),
                flush=True,
            )
    metrics = summarize(
        rows, task=args.task, common=config, predictions_path=predictions_path
    )
    write_json(metrics_path, metrics)
    write_json(config_path, {**config, "status": "completed"})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
