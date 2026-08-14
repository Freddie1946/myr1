#!/usr/bin/env python3
"""Run resumable PathMMU diagnostics through the native LLaVA-Med interface."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from external_vqa_contract import normalize_short_answer
from pathmmu_rewards import accuracy_reward, choice_letter, format_reward


REASONING_QUESTION_TEMPLATE = (
    "{question}\n"
    "Give concise image-grounded reasoning inside <think>...</think>. Then output exactly one "
    "option letter (A, B, C, or D) inside <answer>...</answer>."
)
LETTER_ONLY_QUESTION_TEMPLATE = (
    "{question}\n"
    "Return exactly one uppercase option letter: A, B, C, or D. Do not output any other text."
)
OPTION_TEXT_QUESTION_TEMPLATE = (
    "{question}\n"
    "Answer using only the complete text of the single best option. Do not explain."
)


def option_texts(problem: str) -> dict[str, str]:
    matches = re.findall(
        r"(?m)^\s*([A-D])\)\s*(.+?)(?=\n\s*[A-D]\)\s*|\Z)", str(problem).strip()
    )
    result = {letter: text.strip() for letter, text in matches}
    if len(result) < 2:
        raise ValueError("could not parse PathMMU option texts")
    return result


def option_text_choice(completion: str, problem: str) -> tuple[str | None, str, dict[str, float]]:
    options = option_texts(problem)
    explicit = choice_letter(completion)
    if explicit in options:
        return explicit, "explicit_choice_letter", {}
    candidate = normalize_short_answer(completion)
    exact = [
        letter for letter, text in options.items()
        if normalize_short_answer(text) == candidate
    ]
    if len(exact) == 1:
        return exact[0], "exact_option_text", {}
    similarities = {
        letter: difflib.SequenceMatcher(None, text, str(completion)).ratio()
        for letter, text in options.items()
    }
    if not similarities or max(similarities.values()) <= 0:
        return None, "unresolved", similarities
    return max(similarities, key=similarities.get), "option_text_sequence_matcher", similarities


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    parser.add_argument(
        "--prompt-contract",
        choices=("reasoning_v1", "letter_only_v2", "option_text_v3"),
        default="reasoning_v1",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_cap = {
        "reasoning_v1": 1024,
        "letter_only_v2": 16,
        "option_text_v3": 128,
    }[args.prompt_contract]
    if args.max_new_tokens != expected_cap:
        raise ValueError(
            f"{args.prompt_contract} requires max_new_tokens={expected_cap}"
        )
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

    question_template = {
        "reasoning_v1": REASONING_QUESTION_TEMPLATE,
        "letter_only_v2": LETTER_ONLY_QUESTION_TEMPLATE,
        "option_text_v3": OPTION_TEXT_QUESTION_TEMPLATE,
    }[args.prompt_contract]
    config = {
        "schema_version": 1,
        "status": "running",
        "backend": "llava_med_native",
        "split_role": args.split_role,
        "test_accessed": args.split_role == "test999_development",
        "diagnostic_only": args.split_role == "test999_development",
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "selected_count": len(records),
        "prompt_contract": args.prompt_contract,
        "prompt_template": "<image>\n" + question_template,
        "conversation_mode": "mistral_instruct",
        "dtype": "bfloat16",
        "quantization": "none",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "stopping": "eos_or_generation_cap",
    }
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if {**old, "status": "running"} != config:
            raise ValueError("resume configuration differs")
    else:
        write_json(config_path, config)
    rows = load_existing(predictions_path, records)

    disable_torch_init()
    model_name = get_model_name_from_path(str(args.model))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(args.model), None, model_name
    )
    model.to(device="cuda", dtype=torch.bfloat16).eval()
    model.get_vision_tower().to(device="cuda", dtype=torch.bfloat16)
    model.model.mm_projector.to(device="cuda", dtype=torch.bfloat16)

    for index in range(len(rows), len(records)):
        record = records[index]
        question = question_template.format(question=record["problem"])
        if model.config.mm_use_im_start_end:
            question = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + question
            )
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question
        conversation = conv_templates["mistral_instruct"].copy()
        conversation.append_message(conversation.roles[0], question)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
            image_tensor = process_images(
                [image], image_processor, model.config
            )[0].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        sequence = output_ids[0]
        if sequence.shape[0] >= input_ids.shape[1] and torch.equal(
            sequence[: input_ids.shape[1]], input_ids[0]
        ):
            sequence = sequence[input_ids.shape[1] :]
        generated_ids = sequence.detach().cpu().tolist()
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        token_count = len(generated_ids)
        wrapped = [[{"role": "assistant", "content": completion}]]
        target_choice = choice_letter(record["solution"])
        if args.prompt_contract == "option_text_v3":
            predicted_choice, prediction_source, option_similarities = option_text_choice(
                completion, record["problem"]
            )
            correct = predicted_choice == target_choice
        else:
            predicted_choice = choice_letter(completion)
            prediction_source = "choice_letter_parser"
            option_similarities = None
            correct = bool(accuracy_reward(wrapped, [record["solution"]])[0])
        row = {
            "index": index,
            "source_record_sha256": record_sha256(record),
            "image": record["image"],
            "problem": record["problem"],
            "solution": record["solution"],
            "completion": completion,
            "generated_token_count": token_count,
            "ended_with_eos": bool(
                generated_ids and generated_ids[-1] == tokenizer.eos_token_id
            ),
            "reached_generation_cap": token_count >= args.max_new_tokens,
            "predicted_choice": predicted_choice,
            "prediction_source": prediction_source,
            "option_similarities": option_similarities,
            "target_choice": target_choice,
            "accuracy_reward": float(correct),
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
                        "correct": sum(item["accuracy_reward"] for item in rows),
                    }
                ),
                flush=True,
            )

    token_counts = [row["generated_token_count"] for row in rows]
    correct = int(sum(row["accuracy_reward"] for row in rows))
    metrics = {
        **config,
        "status": "completed",
        "eligible_for_untouched_final_stage3_claim": (
            False if args.split_role == "test999_development" else None
        ),
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "primary_metric": (
            "contract_aligned_option_text_sequence_matcher_accuracy"
            if args.prompt_contract == "option_text_v3"
            else "explicit_choice_accuracy"
        ),
        "prediction_source_counts": {
            source: sum(row.get("prediction_source") == source for row in rows)
            for source in sorted({row.get("prediction_source") for row in rows})
        },
        "format_correct": int(sum(row["format_reward"] for row in rows)),
        "choice_extracted": sum(row["predicted_choice"] is not None for row in rows),
        "empty_completion_count": sum(not row["completion"] for row in rows),
        "mean_generated_tokens": statistics.mean(token_counts),
        "median_generated_tokens": statistics.median(token_counts),
        "maximum_generated_tokens": max(token_counts),
        "generation_cap_hit_count": sum(row["reached_generation_cap"] for row in rows),
        "predictions_sha256": sha256_file(predictions_path),
    }
    write_json(metrics_path, metrics)
    write_json(config_path, {**config, "status": "completed"})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
