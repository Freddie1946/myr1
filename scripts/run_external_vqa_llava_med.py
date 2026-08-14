#!/usr/bin/env python3
"""Resumable deterministic PathVQA/OmniMedVQA inference for LLaVA-Med.

The legacy short-answer contract is the primary external-baseline contract.  The
fixed A=Yes/B=No contract is a supplementary interface diagnostic and is never
silently pooled with legacy-contract scores.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from external_vqa_contract import (
    normalize_short_answer,
    prompt_for_record,
    record_sha256,
    score_record,
    sha256_file,
)
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


PATHVQA_AB_PROMPT = (
    "{question}\n"
    "Options:\nA) Yes\nB) No\n"
    "First give concise image-grounded reasoning inside <think>...</think>. Then output "
    "exactly <answer>A) Yes</answer> or <answer>B) No</answer>."
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def pathvqa_ab_score(completion: str, answer: str) -> dict[str, Any]:
    tagged = re.findall(
        r"<answer\b[^>]*>\s*(.*?)(?:</answer\s*>|$)", completion, re.I | re.S
    )
    candidate = tagged[-1].strip() if tagged else ""
    match = re.match(r"^\s*\(?([AB])\)?(?:\s*[).,:;\-]\s*|\s+)(Yes|No)\s*$", candidate, re.I)
    choice = match.group(1).upper() if match else None
    consistent = bool(
        match
        and ((choice == "A" and match.group(2).lower() == "yes")
             or (choice == "B" and match.group(2).lower() == "no"))
    )
    semantic = {"A": "yes", "B": "no"}.get(choice) if consistent else None
    target = normalize_short_answer(answer)
    strict = bool(
        consistent and re.search(r"<think\b[^>]*>.*?</think\s*>", completion, re.I | re.S)
    )
    return {
        "pathvqa_ab_choice": choice,
        "pathvqa_ab_semantic_answer": semantic,
        "pathvqa_ab_parseable": semantic is not None,
        "strict_pathmmu_choice_format": strict,
        "contract_aligned_answer": semantic or "",
        "contract_aligned_answer_source": "answer_tag_fixed_ab_mapping" if semantic else "unresolved",
        "contract_aligned_exact_match": semantic == target,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("pathvqa", "omnimedvqa"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-role", required=True, choices=("adapter_smoke", "external_test"))
    parser.add_argument(
        "--generation-contract",
        choices=("legacy_v1_64", "pathvqa_pathmmu_ab_v6_2048"),
        default="legacy_v1_64",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--pathvqa-answer-scope", choices=("all", "yes_no_only"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def selected_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = json.loads(args.data.read_text(encoding="utf-8"))
    expected = 6719 if args.task == "pathvqa" else 8518
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, got {len(records)}")
    if args.task != "pathvqa" and args.pathvqa_answer_scope != "all":
        raise ValueError("PathVQA answer scope cannot be used for OmniMedVQA")
    if args.task == "pathvqa" and args.pathvqa_answer_scope == "yes_no_only":
        records = [row for row in records if row.get("answer_type") == "yes_no"]
        if len(records) != 3362:
            raise ValueError(f"expected 3362 PathVQA yes/no records, got {len(records)}")
    if args.limit is not None:
        if args.split_role != "adapter_smoke":
            raise ValueError("only adapter smoke permits --limit")
        if not 1 <= args.limit <= len(records):
            raise ValueError("invalid limit")
        records = records[: args.limit]
    return records


def score(task: str, contract: str, completion: str, record: dict[str, Any]) -> dict[str, Any]:
    result = score_record(task, completion, record)
    if contract == "pathvqa_pathmmu_ab_v6_2048":
        result.update({
            "semantic_yesno_answer": result["contract_aligned_answer"],
            "semantic_yesno_answer_source": result["contract_aligned_answer_source"],
            "semantic_yesno_correct": result["contract_aligned_exact_match"],
        })
        result.update(pathvqa_ab_score(completion, record["answer"]))
    return result


def load_existing(
    path: Path, records: list[dict[str, Any]], task: str, contract: str
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for index, row in enumerate(rows):
        if row.get("index") != index:
            raise ValueError(f"non-contiguous prediction at row {index}")
        if row.get("source_record_sha256") != record_sha256(records[index]):
            raise ValueError(f"source record changed at row {index}")
        row.update(score(task, contract, row["completion"], records[index]))
    return rows


def summarize(
    rows: list[dict[str, Any]], args: argparse.Namespace, predictions_path: Path
) -> dict[str, Any]:
    lengths = [int(row["generated_token_count"]) for row in rows]
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "backend": "llava_med_native",
        "task": args.task,
        "split_role": args.split_role,
        "generation_contract": args.generation_contract,
        "count": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "empty_completion_count": sum(not row["completion"].strip() for row in rows),
        "mean_generated_tokens": statistics.fmean(lengths),
        "median_generated_tokens": statistics.median(lengths),
        "maximum_generated_tokens": max(lengths),
        "generation_cap_hit_count": sum(row["reached_generation_cap"] for row in rows),
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "predictions_sha256": sha256_file(predictions_path),
        "dtype": "bfloat16",
        "quantization": "none",
        "do_sample": False,
    }
    if args.task == "pathvqa":
        correct = sum(row["contract_aligned_exact_match"] for row in rows)
        metrics.update({
            "primary_metric": (
                "pathvqa_yes_no_fixed_ab_contract_aligned_accuracy"
                if args.generation_contract == "pathvqa_pathmmu_ab_v6_2048"
                else "pathvqa_yes_no_contract_aligned_accuracy"
            ),
            "answer_scope": args.pathvqa_answer_scope,
            "contract_aligned_exact_correct": correct,
            "contract_aligned_exact_accuracy": correct / len(rows),
        })
        if args.generation_contract == "pathvqa_pathmmu_ab_v6_2048":
            metrics.update({
                "fixed_option_mapping": {"A": "Yes", "B": "No"},
                "option_order_reversal_used": False,
                "ab_parseable_count": sum(row["pathvqa_ab_parseable"] for row in rows),
                "ab_parseable_rate": sum(row["pathvqa_ab_parseable"] for row in rows) / len(rows),
                "strict_pathmmu_choice_format_count": sum(
                    row["strict_pathmmu_choice_format"] for row in rows
                ),
                "strict_pathmmu_choice_format_rate": sum(
                    row["strict_pathmmu_choice_format"] for row in rows
                ) / len(rows),
                "semantic_yesno_correct": sum(row["semantic_yesno_correct"] for row in rows),
                "semantic_yesno_accuracy": sum(row["semantic_yesno_correct"] for row in rows) / len(rows),
            })
    else:
        aligned = sum(row["contract_aligned_correct"] for row in rows)
        official = sum(row["official_most_similar_correct"] for row in rows)
        by_source = {}
        for source in sorted({row["dataset"] for row in rows}):
            selected = [row for row in rows if row["dataset"] == source]
            by_source[source] = {
                "count": len(selected),
                "contract_aligned_correct": sum(row["contract_aligned_correct"] for row in selected),
                "contract_aligned_accuracy": sum(row["contract_aligned_correct"] for row in selected) / len(selected),
                "official_correct": sum(row["official_most_similar_correct"] for row in selected),
                "official_accuracy": sum(row["official_most_similar_correct"] for row in selected) / len(selected),
            }
        metrics.update({
            "primary_metric": "contract_aligned_sequence_matcher_option_accuracy",
            "contract_aligned_correct": aligned,
            "contract_aligned_accuracy": aligned / len(rows),
            "official_correct": official,
            "official_accuracy": official / len(rows),
            "by_source": by_source,
        })
    return metrics


def main() -> None:
    args = parse_args()
    if args.generation_contract == "legacy_v1_64":
        if args.max_new_tokens != 64:
            raise ValueError("legacy_v1_64 requires max_new_tokens=64")
    else:
        if args.task != "pathvqa" or args.pathvqa_answer_scope != "yes_no_only":
            raise ValueError("A/B contract requires PathVQA yes_no_only")
        if args.max_new_tokens != 2048:
            raise ValueError("A/B contract requires max_new_tokens=2048")
    records = selected_records(args)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    config_path = args.output_dir / "run_config.json"
    if metrics_path.exists():
        raise FileExistsError(metrics_path)
    prompt_template = PATHVQA_AB_PROMPT if args.generation_contract != "legacy_v1_64" else "frozen legacy external VQA prompt"
    config = {
        "schema_version": 1,
        "status": "running",
        "backend": "llava_med_native",
        "task": args.task,
        "split_role": args.split_role,
        "generation_contract": args.generation_contract,
        "selected_count": len(records),
        "answer_scope": args.pathvqa_answer_scope if args.task == "pathvqa" else "all",
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "prompt_template": prompt_template,
        "conversation_mode": "mistral_instruct",
        "dtype": "bfloat16",
        "quantization": "none",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "resume": args.resume,
    }
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if {**old, "resume": args.resume, "status": "running"} != config:
            raise ValueError("resume configuration differs")
    else:
        write_json(config_path, config)
    rows = load_existing(
        predictions_path, records, args.task, args.generation_contract
    )

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
        record_prompt = (
            PATHVQA_AB_PROMPT.format(question=record["question"])
            if args.generation_contract == "pathvqa_pathmmu_ab_v6_2048"
            else prompt_for_record(args.task, record)
        )
        if model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + record_prompt
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + record_prompt
        conversation = conv_templates["mistral_instruct"].copy()
        conversation.append_message(conversation.roles[0], question)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).to(
                device="cuda", dtype=torch.bfloat16
            )
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
        row = {
            "index": index,
            "source_record_sha256": record_sha256(record),
            "image": record["image"],
            "question": record["question"],
            "completion": completion,
            "generated_token_count": len(generated_ids),
            "ended_with_eos": bool(generated_ids and generated_ids[-1] == tokenizer.eos_token_id),
            "reached_generation_cap": len(generated_ids) >= args.max_new_tokens,
            **score(args.task, args.generation_contract, completion, record),
        }
        if args.task == "pathvqa":
            row["answer"] = record["answer"]
        else:
            row.update({
                "dataset": record["dataset"],
                "question_id": record["question_id"],
                "question_type": record["question_type"],
                "gt_answer": record["gt_answer"],
            })
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        rows.append(row)
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(json.dumps({"completed": index + 1, "total": len(records)}), flush=True)

    metrics = summarize(rows, args, predictions_path)
    write_json(metrics_path, metrics)
    write_json(config_path, {**config, "status": "completed"})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
