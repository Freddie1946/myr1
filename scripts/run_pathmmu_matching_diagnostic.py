#!/usr/bin/env python3
"""Run PLIP or CONCH image-option matching on PathMMU."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


OPTION_RE = re.compile(r"^([A-D])\)\s*(.+?)\s*$", re.MULTILINE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def target_choice(solution: str) -> str:
    match = re.search(r"<answer>\s*([A-D])(?:\)|\s|<)", solution, re.IGNORECASE)
    if not match:
        raise ValueError("solution has no answer letter")
    return match.group(1).upper()


def options(problem: str) -> tuple[list[str], list[str]]:
    matches = OPTION_RE.findall(problem)
    letters = [letter for letter, _ in matches]
    texts = [text for _, text in matches]
    if letters != ["A", "B", "C", "D"]:
        raise ValueError(f"unexpected options: {letters}")
    return letters, texts


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("plip", "conch"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split-role",
        required=True,
        choices=("validation_smoke", "test999_development"),
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_backend(backend: str, model_path: Path):
    if backend == "plip":
        from transformers import CLIPModel, CLIPProcessor

        model = (
            CLIPModel.from_pretrained(model_path, local_files_only=True)
            .to(device="cuda", dtype=torch.bfloat16)
            .eval()
        )
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)

        def score(image: Image.Image, texts: list[str]) -> torch.Tensor:
            inputs = processor(
                text=texts, images=image, return_tensors="pt", padding=True
            )
            input_ids = inputs["input_ids"].cuda()
            attention_mask = inputs["attention_mask"].cuda()
            pixel_values = inputs["pixel_values"].to(
                device="cuda", dtype=torch.bfloat16
            )
            with torch.inference_mode():
                image_features = model.get_image_features(pixel_values=pixel_values)
                text_features = model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            image_features = F.normalize(image_features.float(), dim=-1)
            text_features = F.normalize(text_features.float(), dim=-1)
            return (image_features @ text_features.T)[0].cpu()

        return model, score

    from conch.open_clip_custom import (
        create_model_from_pretrained,
        get_tokenizer,
        tokenize,
    )

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", checkpoint_path=str(model_path / "pytorch_model.bin")
    )
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    tokenizer = get_tokenizer()

    def score(image: Image.Image, texts: list[str]) -> torch.Tensor:
        image_tensor = preprocess(image).unsqueeze(0).to(
            device="cuda", dtype=torch.bfloat16
        )
        text_tokens = tokenize(tokenizer, texts).cuda()
        with torch.inference_mode():
            image_features = model.encode_image(
                image_tensor, proj_contrast=True, normalize=True
            )
            text_features = model.encode_text(text_tokens, normalize=True)
        return (image_features.float() @ text_features.float().T)[0].cpu()

    return model, score


def main() -> None:
    args = parse_args()
    records = json.loads(args.data.read_text(encoding="utf-8"))
    expected = 999 if args.split_role == "test999_development" else 385
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, got {len(records)}")
    if args.limit is not None:
        if args.split_role == "test999_development":
            raise ValueError("test999 development run must cover all 999 records")
        records = records[: args.limit]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    predictions_path = args.output_dir / "predictions.jsonl"
    config = {
        "schema_version": 1,
        "status": "running",
        "backend": args.backend,
        "task": "image_to_raw_option_text_matching",
        "option_text_template": "{raw_option_text}",
        "split_role": args.split_role,
        "test_accessed": args.split_role == "test999_development",
        "diagnostic_only": args.split_role == "test999_development",
        "eligible_for_untouched_final_stage3_claim": (
            False if args.split_role == "test999_development" else None
        ),
        "model_path": str(args.model.resolve()),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "selected_count": len(records),
        "dtype": "bfloat16",
        "quantization": "none",
    }
    write_json(args.output_dir / "run_config.json", config)
    _, score = load_backend(args.backend, args.model)

    correct = 0
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            letters, texts = options(record["problem"])
            with Image.open(record["image"]) as source:
                similarities = score(source.convert("RGB"), texts)
            predicted = letters[int(similarities.argmax().item())]
            target = target_choice(record["solution"])
            is_correct = predicted == target
            correct += int(is_correct)
            row = {
                "index": index,
                "source_record_sha256": record_sha256(record),
                "image": record["image"],
                "problem": record["problem"],
                "solution": record["solution"],
                "option_texts": dict(zip(letters, texts)),
                "similarities": {
                    letter: float(value)
                    for letter, value in zip(letters, similarities.tolist())
                },
                "predicted_choice": predicted,
                "target_choice": target,
                "accuracy_reward": float(is_correct),
                "test999_development_diagnostic": args.split_role
                == "test999_development",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if (index + 1) % 25 == 0 or index + 1 == len(records):
                print(
                    json.dumps(
                        {
                            "completed": index + 1,
                            "total": len(records),
                            "correct": correct,
                        }
                    ),
                    flush=True,
                )

    metrics = {
        **config,
        "status": "completed",
        "count": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "choice_extracted": len(records),
        "predictions_sha256": sha256_file(predictions_path),
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "run_config.json", {**config, "status": "completed"})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
