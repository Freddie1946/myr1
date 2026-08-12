#!/usr/bin/env python3
"""Run frozen PathVQA validation prompt/format calibration for Qwen2.5-VL checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageStat
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from external_vqa_contract import (
    PATHVQA_PROMPT,
    normalize_short_answer,
    pathvqa_score,
    record_sha256,
    sha256_file,
)
from grpo_pathmmu_audit import STRICT_PROMPT_SUFFIX


PROMPT_CONTRACTS = {
    "short_v1_192": {
        "max_new_tokens": 192,
        "template": PATHVQA_PROMPT,
        "answer_encoding": "yes_no",
    },
    "xml_v1_192": {
        "max_new_tokens": 192,
        "template": (
            "{question}\n"
            "This is a binary medical visual question. The only valid final answers are Yes and No. "
            "You may reason before the final answer, but you must end with exactly "
            "<answer>Yes</answer> or <answer>No</answer>. Do not put an option letter in the answer tag."
        ),
        "answer_encoding": "yes_no",
    },
    "xml_v1_1024": {
        "max_new_tokens": 1024,
        "template": (
            "{question}\n"
            "This is a binary medical visual question. The only valid final answers are Yes and No. "
            "You may reason before the final answer, but you must end with exactly "
            "<answer>Yes</answer> or <answer>No</answer>. Do not put an option letter in the answer tag."
        ),
        "answer_encoding": "yes_no",
    },
    "xml_v1_4096": {
        "max_new_tokens": 4096,
        "template": (
            "{question}\n"
            "This is a binary medical visual question. The only valid final answers are Yes and No. "
            "You may reason before the final answer, but you must end with exactly "
            "<answer>Yes</answer> or <answer>No</answer>. Do not put an option letter in the answer tag."
        ),
        "answer_encoding": "yes_no",
    },
    "domain_think_answer_v2_2048": {
        "max_new_tokens": 2048,
        "template": (
            "{question}\n"
            "Answer this binary pathology image question using the same structured format as the "
            "target-domain task. Give concise image-grounded pathological reasoning inside "
            "<think>...</think>. Then output exactly one final binary answer: "
            "<answer>Yes</answer> or <answer>No</answer>. Do not put an option letter or any other "
            "text inside the answer tag."
        ),
        "answer_encoding": "yes_no",
    },
    "choice_v1_32": {
        "max_new_tokens": 32,
        "template": (
            "{question}\nChoose exactly one answer:\nA) Yes\nB) No\n"
            "Return only A or B without explanation."
        ),
        "answer_encoding": "ab_choice",
    },
    "choice_v1_192": {
        "max_new_tokens": 192,
        "template": (
            "{question}\nChoose exactly one answer:\nA) Yes\nB) No\n"
            "Return the selected letter and answer in the final answer."
        ),
        "answer_encoding": "ab_choice",
        "choice_map": {"A": "yes", "B": "no"},
    },
    "pathmmu_ab_v1_2048": {
        "max_new_tokens": 2048,
        "template": "{question}\nOptions:\nA) Yes\nB) No" + STRICT_PROMPT_SUFFIX,
        "answer_encoding": "ab_choice",
        "choice_map": {"A": "yes", "B": "no"},
    },
}


def strict_pathmmu_choice_format(completion: str, contract: str) -> bool | None:
    """Check the exact PathMMU-style final answer while scoring semantics separately."""
    if contract != "pathmmu_ab_v1_2048":
        return None
    tagged = re.findall(
        r"<answer\b[^>]*>\s*(.*?)(?:</answer\s*>|$)", completion, re.I | re.S
    )
    if not tagged:
        return False
    value = re.sub(r"\s+", " ", tagged[-1].strip())
    return bool(re.fullmatch(r"(?:A\) Yes|B\) No)", value, re.I))


def eos_ids(value: int | list[int] | tuple[int, ...] | None) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def extract_calibration_answer(completion: str, contract: str) -> tuple[str | None, str]:
    if contract not in PROMPT_CONTRACTS:
        raise ValueError(f"unknown prompt contract: {contract}")
    if PROMPT_CONTRACTS[contract]["answer_encoding"] == "yes_no":
        score = pathvqa_score(completion, "yes")
        value = normalize_short_answer(score["contract_aligned_answer"])
        return (
            value if value in {"yes", "no"} else None,
            score["contract_aligned_answer_source"],
        )
    tagged = re.findall(
        r"<answer\b[^>]*>\s*(.*?)(?:</answer\s*>|$)", completion, re.I | re.S
    )
    candidate = tagged[-1].strip() if tagged else completion.strip()
    explicit = re.match(r"^\s*\(?\s*([AB])(?=[\s).,:;\-]|$)", candidate, re.I)
    if explicit:
        letter = explicit.group(1).upper()
        choice_map = PROMPT_CONTRACTS[contract].get(
            "choice_map", {"A": "yes", "B": "no"}
        )
        return choice_map[letter], "leading_ab"
    direct = re.match(r"^\s*(yes|no)\b", candidate, re.I)
    if direct:
        return direct.group(1).lower(), "leading_yes_no"
    return None, "unresolved"


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cyclic_image_map(records: list[dict[str, Any]]) -> dict[str, str]:
    paths = sorted({str(row["image"]) for row in records})
    if len(paths) < 2:
        raise ValueError("cyclic mismatch requires at least two unique images")
    return {path: paths[(index + 1) % len(paths)] for index, path in enumerate(paths)}


def intervention_image(
    source_path: str, mode: str, mismatch: dict[str, str]
) -> tuple[Image.Image, str]:
    input_path = mismatch[source_path] if mode == "cyclic_mismatch" else source_path
    with Image.open(input_path) as source:
        image = source.convert("RGB")
    if mode == "global_mean_blank":
        mean = tuple(round(value) for value in ImageStat.Stat(image).mean)
        image = Image.new("RGB", image.size, mean)
    return image, input_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument(
        "--panel-role",
        choices=("frozen_calibration_128", "architecture_validation_512"),
        default="frozen_calibration_128",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--image-mode",
        choices=("original", "cyclic_mismatch", "global_mean_blank"),
        default="original",
    )
    parser.add_argument(
        "--prompt-contracts",
        default=",".join(PROMPT_CONTRACTS),
        help="Comma-separated frozen prompt contracts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    contracts = [item.strip() for item in args.prompt_contracts.split(",") if item.strip()]
    if not contracts or len(contracts) != len(set(contracts)):
        raise ValueError("prompt contracts must be non-empty and unique")
    unknown = set(contracts) - set(PROMPT_CONTRACTS)
    if unknown:
        raise ValueError(f"unknown prompt contracts: {sorted(unknown)}")
    panel_payload = json.loads(args.panel.read_text(encoding="utf-8"))
    if args.panel_role == "frozen_calibration_128":
        if not isinstance(panel_payload, dict) or panel_payload.get("status") != "frozen_before_calibration_inference":
            raise ValueError("panel is not frozen before calibration inference")
        records = panel_payload.get("records")
        if not isinstance(records, list) or len(records) != 128:
            raise ValueError("expected the frozen 128-case panel")
    else:
        records = panel_payload.get("records") if isinstance(panel_payload, dict) else panel_payload
        if not isinstance(records, list) or len(records) != 512:
            raise ValueError("expected the frozen 512-case architecture validation panel")
    if {normalize_short_answer(row["answer"]) for row in records} != {"yes", "no"}:
        raise ValueError("calibration panel is not binary")
    mismatch = cyclic_image_map(records) if args.image_mode == "cyclic_mismatch" else {}
    args.output_root.mkdir(parents=True)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = 65536
        processor.image_processor.min_pixels = 3136
    compatible_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if isinstance(getattr(compatible_config, "text_config", None), dict):
        delattr(compatible_config, "text_config")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        config=compatible_config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    if args.adapter is not None:
        if not (args.adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(args.adapter / "adapter_config.json")
        if not (args.adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(args.adapter / "adapter_model.safetensors")
        model = PeftModel.from_pretrained(
            model, args.adapter, local_files_only=True, is_trainable=False
        )
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    stop_ids = eos_ids(model.generation_config.eos_token_id)
    index: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "formal_result": False,
        "split_role": (
            "pathvqa_validation_prompt_parser_calibration"
            if args.panel_role == "frozen_calibration_128"
            else "pathvqa_architecture_validation_format_aligned_diagnostic"
        ),
        "panel_role": args.panel_role,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "adapter_path": str(args.adapter.resolve()) if args.adapter is not None else None,
        "adapter_config_sha256": (
            sha256_file(args.adapter / "adapter_config.json") if args.adapter is not None else None
        ),
        "adapter_model_sha256": (
            sha256_file(args.adapter / "adapter_model.safetensors") if args.adapter is not None else None
        ),
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256_file(args.panel),
        "accuracy_used_for_prompt_selection": False,
        "image_mode": args.image_mode,
        "contracts": {},
    }
    for contract in contracts:
        spec = PROMPT_CONTRACTS[contract]
        contract_root = args.output_root / contract
        contract_root.mkdir()
        predictions_path = contract_root / "predictions.jsonl"
        rows: list[dict[str, Any]] = []
        for batch_start in range(0, len(records), args.batch_size):
            batch = records[batch_start : batch_start + args.batch_size]
            images = []
            input_image_paths = []
            prompts = []
            for record in batch:
                source_path = str(record["image"])
                image, input_image_path = intervention_image(
                    source_path, args.image_mode, mismatch
                )
                images.append(image)
                input_image_paths.append(input_image_path)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": spec["template"].format(question=record["question"])},
                    ],
                }]
                prompts.append(processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ))
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=int(spec["max_new_tokens"]),
                    do_sample=False,
                    use_cache=True,
                )
            completion_ids = generated[:, inputs["input_ids"].shape[1] :]
            for offset, record in enumerate(batch):
                token_ids = [int(value) for value in completion_ids[offset].tolist()]
                token_count = len(token_ids)
                for token_index, token_id in enumerate(token_ids):
                    if token_id in stop_ids:
                        token_count = token_index + 1
                        break
                token_ids = token_ids[:token_count]
                completion = processor.decode(token_ids, skip_special_tokens=True).strip()
                answer, source = extract_calibration_answer(completion, contract)
                target = normalize_short_answer(record["answer"])
                row = {
                    "index": batch_start + offset,
                    "panel_index": record.get("panel_index", batch_start + offset),
                    "source_index": record["source_index"],
                    "source_record_sha256": record_sha256(record),
                    "image": record["image"],
                    "input_image": input_image_paths[offset],
                    "image_mode": args.image_mode,
                    "question": record["question"],
                    "target": target,
                    "completion": completion,
                    "generated_token_count": token_count,
                    "ended_with_eos": bool(token_ids and token_ids[-1] in stop_ids),
                    "reached_generation_cap": token_count >= int(spec["max_new_tokens"]),
                    "parsed_answer": answer,
                    "parse_source": source,
                    "parseable": answer is not None,
                    "correct": answer == target,
                    "strict_pathmmu_choice_format": strict_pathmmu_choice_format(
                        completion, contract
                    ),
                }
                with predictions_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                rows.append(row)
        lengths = [row["generated_token_count"] for row in rows]
        metrics = {
            "schema_version": 1,
            "status": "completed",
            "formal_result": False,
            "contract": contract,
            "prompt_template": spec["template"],
            "image_mode": args.image_mode,
            "max_new_tokens": spec["max_new_tokens"],
            "count": len(rows),
            "parseable_count": sum(row["parseable"] for row in rows),
            "parseable_rate": sum(row["parseable"] for row in rows) / len(rows),
            "generation_cap_hit_count": sum(row["reached_generation_cap"] for row in rows),
            "generation_cap_hit_rate": sum(row["reached_generation_cap"] for row in rows) / len(rows),
            "correct": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "accuracy_diagnostic_not_for_selection": sum(row["correct"] for row in rows) / len(rows),
            "conditional_accuracy_when_parseable": (
                sum(row["correct"] for row in rows) / sum(row["parseable"] for row in rows)
                if any(row["parseable"] for row in rows) else None
            ),
            "mean_generated_tokens": statistics.fmean(lengths),
            "median_generated_tokens": statistics.median(lengths),
            "maximum_generated_tokens": max(lengths),
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": sha256_file(predictions_path),
        }
        if contract == "pathmmu_ab_v1_2048":
            metrics["strict_pathmmu_choice_format_count"] = sum(
                bool(row["strict_pathmmu_choice_format"]) for row in rows
            )
            metrics["strict_pathmmu_choice_format_rate"] = (
                metrics["strict_pathmmu_choice_format_count"] / len(rows)
            )
        write_json(contract_root / "metrics.json", metrics)
        index["contracts"][contract] = metrics
        print(json.dumps({"contract": contract, "parseable_rate": metrics["parseable_rate"]}), flush=True)
    write_json(args.output_root / "index.json", index)


if __name__ == "__main__":
    main()
