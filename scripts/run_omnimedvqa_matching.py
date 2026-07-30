#!/usr/bin/env python3
"""Run PLIP or CONCH raw-option matching on the fixed OmniMedVQA four-source set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from external_vqa_contract import (
    omnimed_options,
    omnimed_target_choice,
    record_sha256,
    sha256_file,
)
from external_vqa_native_utils import write_json


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
            inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
            with torch.inference_mode():
                image_features = model.get_image_features(
                    pixel_values=inputs["pixel_values"].to(
                        device="cuda", dtype=torch.bfloat16
                    )
                )
                text_features = model.get_text_features(
                    input_ids=inputs["input_ids"].cuda(),
                    attention_mask=inputs["attention_mask"].cuda(),
                )
            return (
                F.normalize(image_features.float(), dim=-1)
                @ F.normalize(text_features.float(), dim=-1).T
            )[0].cpu()

        return score

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

    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("plip", "conch"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split-role", required=True, choices=("adapter_smoke", "external_test")
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    records = json.loads(args.data.read_text(encoding="utf-8"))
    if len(records) != 8518:
        raise ValueError(f"expected 8518 records, got {len(records)}")
    if args.limit is not None:
        if args.split_role != "adapter_smoke" or not 1 <= args.limit <= len(records):
            raise ValueError("invalid smoke limit")
        records = records[: args.limit]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    config = {
        "schema_version": 1,
        "status": "running",
        "task": "omnimedvqa",
        "split_role": args.split_role,
        "backend": args.backend,
        "matching_contract": "image_to_raw_option_text_cosine_similarity",
        "model_path": str(args.model.resolve()),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "selected_count": len(records),
        "dtype": "bfloat16",
        "quantization": "none",
    }
    write_json(args.output_dir / "run_config.json", config)
    score = load_backend(args.backend, args.model)
    correct = 0
    by_source: dict[str, dict[str, int]] = {}
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            letters, texts = omnimed_options(record)
            with Image.open(record["image"]) as source:
                similarities = score(source.convert("RGB"), texts)
            predicted = letters[int(similarities.argmax().item())]
            target = omnimed_target_choice(record)
            is_correct = predicted == target
            correct += int(is_correct)
            source_stats = by_source.setdefault(record["dataset"], {"count": 0, "correct": 0})
            source_stats["count"] += 1
            source_stats["correct"] += int(is_correct)
            row = {
                "index": index,
                "source_record_sha256": record_sha256(record),
                "image": record["image"],
                "dataset": record["dataset"],
                "question_id": record["question_id"],
                "question": record["question"],
                "question_type": record["question_type"],
                "gt_answer": record["gt_answer"],
                "option_texts": dict(zip(letters, texts)),
                "similarities": dict(zip(letters, map(float, similarities.tolist()))),
                "predicted_choice": predicted,
                "target_choice": target,
                "correct": is_correct,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if (index + 1) % 100 == 0 or index + 1 == len(records):
                print(
                    json.dumps(
                        {"completed": index + 1, "total": len(records), "correct": correct}
                    ),
                    flush=True,
                )
    for stats in by_source.values():
        stats["accuracy"] = stats["correct"] / stats["count"]
    metrics = {
        **config,
        "status": "completed",
        "primary_metric": "raw_option_text_cosine_matching_accuracy",
        "count": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "by_source": by_source,
        "predictions_sha256": sha256_file(predictions_path),
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "run_config.json", {**config, "status": "completed"})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
