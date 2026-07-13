#!/usr/bin/env python3
"""Load a gathered Qwen2.5-VL SFT checkpoint on CPU and record identity/counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    AutoProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "checkpoint": str(checkpoint),
        "loaded": True,
        "model_type": config.model_type,
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "total_parameters": total,
        "torch_dtype_counts": {},
    }
    for parameter in model.parameters():
        key = str(parameter.dtype)
        result["torch_dtype_counts"][key] = result["torch_dtype_counts"].get(key, 0) + parameter.numel()
    result["passed"] = (
        result["model_type"] == "qwen2_5_vl"
        and result["hidden_size"] == 3584
        and total > 8_000_000_000
    )
    rendered = json.dumps(result, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
