#!/usr/bin/env python3
"""Merge a selected SFT LoRA/Projector adapter into an immutable full Qwen2.5-VL model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.base / "config.json", args.adapter / "adapter_config.json", args.adapter / "adapter_model.safetensors"):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = AutoConfig.from_pretrained(args.base, local_files_only=True)
    if isinstance(getattr(config, "text_config", None), dict):
        delattr(config, "text_config")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, config=config, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True, is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    args.output_dir.mkdir(parents=True)
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(args.base, local_files_only=True).save_pretrained(args.output_dir)
    shards = sorted(args.output_dir.glob("*.safetensors"))
    manifest = {
        "schema_version": 1,
        "status": "merged",
        "base": str(args.base.resolve()),
        "base_config_sha256": sha256(args.base / "config.json"),
        "adapter": str(args.adapter.resolve()),
        "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"),
        "adapter_model_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "output_config_sha256": sha256(args.output_dir / "config.json"),
        "weight_shards": {path.name: sha256(path) for path in shards},
        "safe_merge": True,
    }
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
