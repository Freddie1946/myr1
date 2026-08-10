#!/usr/bin/env python3
"""Resumable deterministic PathVQA/OmniMedVQA inference for Transformers-native VLMs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from external_vqa_contract import (
    PATHVQA_PROMPT,
    OMNIMEDVQA_PROMPT,
    prompt_for_record,
    record_sha256,
    score_record,
    sha256_file,
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def eos_ids(value: int | list[int] | tuple[int, ...] | None) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def load_existing(
    path: Path, records: list[dict[str, Any]], task: str
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for index, row in enumerate(rows):
        if row.get("index") != index:
            raise ValueError(f"non-contiguous prediction at row {index}")
        if index >= len(records):
            raise ValueError("existing predictions exceed selected records")
        if row.get("source_record_sha256") != record_sha256(records[index]):
            raise ValueError(f"source record changed at row {index}")
        # Permit a run started under scorer v1 to resume under the additive v2
        # contract without regenerating or silently keeping stale in-memory scores.
        row.update(score_record(task, row["completion"], records[index]))
    return rows


def select_answer_scope(
    task: str, answer_scope: str, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if task != "pathvqa":
        if answer_scope != "all":
            raise ValueError("answer scope applies only to PathVQA")
        return records
    if answer_scope == "all":
        return records
    if answer_scope == "yes_no_only":
        selected = [row for row in records if row.get("answer_type") == "yes_no"]
        if len(selected) != 3362:
            raise ValueError(f"expected 3362 PathVQA yes/no records, got {len(selected)}")
        return selected
    raise ValueError(f"unsupported PathVQA answer scope: {answer_scope}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("pathvqa", "omnimedvqa"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--backend",
        required=True,
        choices=("gemma3", "mllama", "qwen2_vl", "qwen2_5_vl", "qwen3_5"),
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split-role", required=True, choices=("adapter_smoke", "external_test")
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--generation-contract",
        choices=("legacy_v1_64", "omnimed_corrective_v3_192"),
        default="legacy_v1_64",
        help="Pinned generation/scoring contract; corrective v3 is OmniMedVQA-only.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pathvqa-answer-scope",
        choices=("all", "yes_no_only"),
        default="all",
    )
    return parser.parse_args()


def summarize(
    rows: list[dict[str, Any]], args: argparse.Namespace, predictions_path: Path
) -> dict[str, Any]:
    lengths = [int(row["generated_token_count"]) for row in rows]
    common = {
        "schema_version": 2,
        "status": "completed",
        "task": args.task,
        "split_role": args.split_role,
        "count": len(rows),
        "empty_completion_count": sum(not row["completion"].strip() for row in rows),
        "mean_generated_tokens": statistics.fmean(lengths),
        "median_generated_tokens": statistics.median(lengths),
        "maximum_generated_tokens": max(lengths),
        "generation_cap_hit_count": sum(row["reached_generation_cap"] for row in rows),
        "max_new_tokens": args.max_new_tokens,
        "generation_contract": args.generation_contract,
        "do_sample": False,
        "dtype": "bfloat16",
        "quantization": "none",
        "backend": args.backend,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "predictions_sha256": sha256_file(predictions_path),
    }
    if args.task == "pathvqa":
        yes_no = [row for row in rows if row["answer_type"] == "yes_no"]
        free = [row for row in rows if row["answer_type"] == "free_form"]
        correct = sum(row["exact_match"] for row in rows)
        pathvqa_metrics = {
            "primary_metric": "pathvqa_paper_metric_family_by_answer_type",
            "answer_scope": args.pathvqa_answer_scope,
            "repository_token_overlap_mean_diagnostic": statistics.fmean(
                row["official_token_overlap_score"] for row in rows
            ),
            "repository_token_f1_mean": statistics.fmean(
                row["official_token_f1_score"] for row in rows
            ),
            "contract_aligned_exact_correct": sum(
                row["contract_aligned_exact_match"] for row in rows
            ),
            "contract_aligned_exact_accuracy": sum(
                row["contract_aligned_exact_match"] for row in rows
            ) / len(rows),
            "legacy_strict_exact_metric": "normalized_whole_completion_exact_match",
            "correct": correct,
            "accuracy": correct / len(rows),
            "yes_no_count": len(yes_no),
            "yes_no_correct": sum(row["exact_match"] for row in yes_no),
            "yes_no_accuracy": sum(row["exact_match"] for row in yes_no) / len(yes_no),
            "paper_yes_no_contract_aligned_accuracy": sum(
                row["contract_aligned_exact_match"] for row in yes_no
            ) / len(yes_no),
        }
        if args.pathvqa_answer_scope == "all":
            pathvqa_metrics.update({
                "free_form_count": len(free),
                "free_form_correct": sum(row["exact_match"] for row in free),
                "free_form_accuracy": sum(row["exact_match"] for row in free) / len(free),
                "paper_free_form_strict_exact_accuracy": sum(
                    row["strict_exact_match"] for row in free
                ) / len(free),
                "paper_free_form_macro_token_f1": statistics.fmean(
                    row["repository_token_f1_score"] for row in free
                ),
                "paper_free_form_sentence_bleu_1": statistics.fmean(
                    row["sentence_bleu_1"] for row in free
                ),
                "paper_free_form_sentence_bleu_2": statistics.fmean(
                    row["sentence_bleu_2"] for row in free
                ),
                "paper_free_form_sentence_bleu_3": statistics.fmean(
                    row["sentence_bleu_3"] for row in free
                ),
            })
        else:
            pathvqa_metrics.update({
                "primary_metric": "pathvqa_yes_no_contract_aligned_accuracy",
                "free_form_count": 0,
                "free_form_inference_excluded": True,
            })
        common.update(pathvqa_metrics)
    else:
        official_correct = sum(row["official_most_similar_correct"] for row in rows)
        aligned_correct = sum(row["contract_aligned_correct"] for row in rows)
        strict_correct = sum(row["strict_text_correct"] for row in rows)
        strict_final_available = sum(
            row["strict_final_answer_available"] for row in rows
        )
        strict_final_correct = sum(row["strict_final_correct"] for row in rows)
        by_source = {}
        for source in sorted({row["dataset"] for row in rows}):
            selected = [row for row in rows if row["dataset"] == source]
            by_source[source] = {
                "count": len(selected),
                "official_correct": sum(
                    row["official_most_similar_correct"] for row in selected
                ),
                "official_accuracy": sum(
                    row["official_most_similar_correct"] for row in selected
                )
                / len(selected),
                "strict_text_correct": sum(row["strict_text_correct"] for row in selected),
                "strict_text_accuracy": sum(
                    row["strict_text_correct"] for row in selected
                )
                / len(selected),
                "contract_aligned_correct": sum(
                    row["contract_aligned_correct"] for row in selected
                ),
                "contract_aligned_accuracy": sum(
                    row["contract_aligned_correct"] for row in selected
                ) / len(selected),
                "strict_final_answer_available": sum(
                    row["strict_final_answer_available"] for row in selected
                ),
                "strict_final_answer_coverage": sum(
                    row["strict_final_answer_available"] for row in selected
                ) / len(selected),
                "strict_final_correct": sum(
                    row["strict_final_correct"] for row in selected
                ),
                "strict_final_accuracy": sum(
                    row["strict_final_correct"] for row in selected
                ) / len(selected),
            }
        common.update(
            {
                "primary_metric": (
                    "strict_final_option_accuracy"
                    if args.generation_contract == "omnimed_corrective_v3_192"
                    else "contract_aligned_sequence_matcher_option_accuracy"
                ),
                "strict_final_correct": strict_final_correct,
                "strict_final_accuracy": strict_final_correct / len(rows),
                "strict_final_answer_available": strict_final_available,
                "strict_final_answer_coverage": strict_final_available / len(rows),
                "strict_final_unresolved_count": len(rows) - strict_final_available,
                "contract_aligned_correct": aligned_correct,
                "contract_aligned_accuracy": aligned_correct / len(rows),
                "official_raw_completion_metric": "official_sequence_matcher_option_accuracy",
                "official_correct": official_correct,
                "official_accuracy": official_correct / len(rows),
                "strict_text_correct": strict_correct,
                "strict_text_accuracy": strict_correct / len(rows),
                "by_source": by_source,
            }
        )
    return common


def main() -> None:
    args = parse_args()
    if args.generation_contract == "legacy_v1_64":
        if args.max_new_tokens != 64:
            raise ValueError("legacy external VQA contract requires max_new_tokens=64")
    elif args.generation_contract == "omnimed_corrective_v3_192":
        if args.task != "omnimedvqa":
            raise ValueError("corrective v3 generation contract is OmniMedVQA-only")
        if args.max_new_tokens != 192:
            raise ValueError("corrective v3 generation contract requires max_new_tokens=192")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    records = json.loads(args.data.read_text(encoding="utf-8"))
    expected = 6719 if args.task == "pathvqa" else 8518
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, got {len(records)}")
    records = select_answer_scope(args.task, args.pathvqa_answer_scope, records)
    if args.limit is not None:
        if args.split_role != "adapter_smoke":
            raise ValueError("only adapter smoke permits --limit")
        if not 1 <= args.limit <= len(records):
            raise ValueError("invalid --limit")
        records = records[: args.limit]
    if args.split_role == "external_test" and args.limit is not None:
        raise ValueError("external test must cover the full split")

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    config_path = args.output_dir / "run_config.json"
    if metrics_path.exists():
        raise FileExistsError(metrics_path)
    prompt_template = PATHVQA_PROMPT if args.task == "pathvqa" else OMNIMEDVQA_PROMPT
    config = {
        "schema_version": 2,
        "status": "running",
        "task": args.task,
        "split_role": args.split_role,
        "selected_count": len(records),
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "backend": args.backend,
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "prompt_template": prompt_template,
        "dtype": "bfloat16",
        "quantization": "none",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "generation_contract": args.generation_contract,
        "batch_size": args.batch_size,
        "answer_scope": args.pathvqa_answer_scope if args.task == "pathvqa" else "all",
        "resume": args.resume,
    }
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if {**old, "resume": args.resume, "status": "running"} != config:
            raise ValueError("resume configuration differs")
    else:
        write_json(config_path, config)
    rows = load_existing(predictions_path, records, args.task)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = 65536
        processor.image_processor.min_pixels = 3136
    if args.backend == "gemma3":
        from transformers import Gemma3ForConditionalGeneration

        model_class = Gemma3ForConditionalGeneration
    elif args.backend == "mllama":
        from transformers import MllamaForConditionalGeneration

        model_class = MllamaForConditionalGeneration
    elif args.backend == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration

        model_class = Qwen3_5ForConditionalGeneration
    else:
        model_class = {
            "qwen2_vl": Qwen2VLForConditionalGeneration,
            "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
        }[args.backend]
    load_kwargs = {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if args.backend == "qwen2_5_vl":
        compatible_config = AutoConfig.from_pretrained(
            args.model, local_files_only=True
        )
        if isinstance(getattr(compatible_config, "text_config", None), dict):
            delattr(compatible_config, "text_config")
        load_kwargs["config"] = compatible_config
    if args.backend == "mllama":
        # The 90B baseline cannot fit on one accelerator.  Accelerate's deterministic automatic
        # placement uses the visible GPUs without quantization or CPU/disk offload on 8xA100-80G.
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {
            index: "76GiB" for index in range(torch.cuda.device_count())
        }
        model = model_class.from_pretrained(args.model, **load_kwargs)
    else:
        model = model_class.from_pretrained(args.model, **load_kwargs).to("cuda")
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    stop_ids = eos_ids(model.generation_config.eos_token_id)

    for batch_start in range(len(rows), len(records), args.batch_size):
        batch_records = records[batch_start : batch_start + args.batch_size]
        images = []
        prompts = []
        for record in batch_records:
            with Image.open(record["image"]) as source:
                images.append(source.convert("RGB"))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_for_record(args.task, record)},
                    ],
                }
            ]
            template_kwargs = {}
            if args.backend == "qwen3_5":
                template_kwargs["enable_thinking"] = False
            prompts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            )
        processor_images = [[image] for image in images] if args.backend == "gemma3" else images
        inputs = processor(
            text=prompts, images=processor_images, return_tensors="pt", padding=True
        )
        input_device = next(model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        completion_ids = generated[:, inputs["input_ids"].shape[1] :]
        for offset, record in enumerate(batch_records):
            token_ids = [int(value) for value in completion_ids[offset].tolist()]
            token_count = len(token_ids)
            for token_index, token_id in enumerate(token_ids):
                if token_id in stop_ids:
                    token_count = token_index + 1
                    break
            token_ids = token_ids[:token_count]
            completion = processor.decode(token_ids, skip_special_tokens=True).strip()
            index = batch_start + offset
            score = score_record(args.task, completion, record)
            row = {
                "index": index,
                "source_record_sha256": record_sha256(record),
                "image": record["image"],
                "question": record["question"],
                "completion": completion,
                "generated_token_count": token_count,
                "ended_with_eos": bool(token_ids and token_ids[-1] in stop_ids),
                "reached_generation_cap": token_count >= args.max_new_tokens,
                **score,
            }
            if args.task == "pathvqa":
                row["answer"] = record["answer"]
            else:
                row.update(
                    {
                        "dataset": record["dataset"],
                        "question_id": record["question_id"],
                        "question_type": record["question_type"],
                        "gt_answer": record["gt_answer"],
                        "option_texts": {
                            letter: record[f"option_{letter}"]
                            for letter in "ABCD"
                            if record.get(f"option_{letter}") is not None
                        },
                    }
                )
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            rows.append(row)
        completed = batch_start + len(batch_records)
        if completed % 100 < args.batch_size or completed == len(records):
            if args.task == "pathvqa":
                correct = sum(row["contract_aligned_exact_match"] for row in rows)
            elif args.generation_contract == "omnimed_corrective_v3_192":
                correct = sum(row["strict_final_correct"] for row in rows)
            else:
                correct = sum(row["contract_aligned_correct"] for row in rows)
            print(
                json.dumps(
                    {"completed": completed, "total": len(records), "correct": correct}
                ),
                flush=True,
            )

    metrics = summarize(rows, args, predictions_path)
    write_json(metrics_path, metrics)
    write_json(config_path, {**config, "status": "completed"})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
