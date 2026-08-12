#!/usr/bin/env python3
"""Fail closed on the trainability and saved-weight contract of a PEFT smoke run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from safetensors import safe_open


LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
ADAPTER_PREFIX = "base_model.model."


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_lora_module(key: str) -> tuple[str, str] | None:
    if not key.startswith(ADAPTER_PREFIX):
        return None
    for suffix in LORA_SUFFIXES:
        if key.endswith(suffix):
            return key[len(ADAPTER_PREFIX) : -len(suffix)], suffix.split(".")[1]
    return None


def base_tensor(model_dir: Path, name: str):
    index = load_json(model_dir / "model.safetensors.index.json")
    shard = index["weight_map"].get(name)
    if shard is None:
        raise KeyError(f"base tensor missing from index: {name}")
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def verify(manifest_path: Path, output_dir: Path) -> dict:
    manifest = load_json(manifest_path)
    arm = manifest["arm"]
    if arm == "c0":
        raise ValueError("C0 is a full-model run and requires the separate full-checkpoint gate")

    adapter_config_path = output_dir / "adapter_config.json"
    adapter_model_path = output_dir / "adapter_model.safetensors"
    train_results_path = output_dir / "train_results.json"
    for required in (adapter_config_path, adapter_model_path, train_results_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    adapter_config = load_json(adapter_config_path)
    expected_targets = set(manifest["lora_target_modules"])
    expect_merger = manifest["projector_policy"] == "full_train_visual.merger"
    saved_targets = set(adapter_config["target_modules"])
    if saved_targets != expected_targets:
        missing = sorted(expected_targets - saved_targets)
        unexpected = sorted(saved_targets - expected_targets)
        raise ValueError(f"LoRA target mismatch: missing={missing}, unexpected={unexpected}")
    expected_modules_to_save = {"visual.merger"} if expect_merger else set()
    if set(adapter_config.get("modules_to_save") or []) != expected_modules_to_save:
        raise ValueError(
            f"modules_to_save mismatch: expected={sorted(expected_modules_to_save)}"
        )

    seen: dict[str, set[str]] = {}
    merger_tensors: dict[str, object] = {}
    all_finite = True
    lora_b_nonzero = 0
    lora_b_elements = 0
    with safe_open(adapter_model_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            parsed = canonical_lora_module(key)
            if parsed is not None:
                module, kind = parsed
                seen.setdefault(module, set()).add(kind)
                tensor = handle.get_tensor(key)
                all_finite = all_finite and bool(tensor.isfinite().all())
                if kind == "lora_B":
                    lora_b_nonzero += int((tensor != 0).sum())
                    lora_b_elements += tensor.numel()
            elif key.startswith(ADAPTER_PREFIX + "visual.merger."):
                tensor = handle.get_tensor(key)
                all_finite = all_finite and bool(tensor.isfinite().all())
                merger_tensors[key[len(ADAPTER_PREFIX) :]] = tensor

    if set(seen) != expected_targets:
        missing = sorted(expected_targets - set(seen))
        unexpected = sorted(set(seen) - expected_targets)
        raise ValueError(f"saved LoRA module mismatch: missing={missing}, unexpected={unexpected}")
    incomplete = sorted(module for module, kinds in seen.items() if kinds != {"lora_A", "lora_B"})
    if incomplete:
        raise ValueError(f"incomplete LoRA tensor pairs: {incomplete}")
    if not all_finite:
        raise ValueError("adapter contains non-finite tensors")
    if lora_b_nonzero == 0:
        raise ValueError("all LoRA-B weights remained zero")
    if expect_merger and not merger_tensors:
        raise ValueError("saved adapter contains no visual.merger tensors")
    if not expect_merger and merger_tensors:
        raise ValueError("frozen visual.merger unexpectedly appears in the adapter")

    model_dir = Path(adapter_config["base_model_name_or_path"])
    merger_changed = 0
    merger_max_abs_delta = 0.0
    for name, trained in merger_tensors.items():
        original = base_tensor(model_dir, name)
        if original.shape != trained.shape:
            raise ValueError(f"merger tensor shape mismatch for {name}")
        delta = (trained.float() - original.float()).abs()
        value = float(delta.max()) if delta.numel() else 0.0
        if not math.isfinite(value):
            raise ValueError(f"non-finite merger delta for {name}")
        merger_max_abs_delta = max(merger_max_abs_delta, value)
        merger_changed += int(bool((delta != 0).any()))
    if expect_merger and merger_changed == 0:
        raise ValueError("visual.merger was saved but did not change from the base model")

    expected_visual_blocks = {
        int(module.split(".")[2])
        for module in expected_targets
        if module.startswith("visual.blocks.")
    }
    saved_visual_blocks = {
        int(module.split(".")[2])
        for module in seen
        if module.startswith("visual.blocks.")
    }
    if saved_visual_blocks != expected_visual_blocks:
        raise ValueError(
            f"visual block mismatch: expected={sorted(expected_visual_blocks)}, "
            f"saved={sorted(saved_visual_blocks)}"
        )

    train_results = load_json(train_results_path)
    runtime = float(train_results["train_runtime"])
    if runtime <= 0:
        raise ValueError("non-positive train runtime")
    report = {
        "schema_version": 1,
        "formal_result": False,
        "status": "passed",
        "arm": arm,
        "lora_module_count": len(seen),
        "vision_blocks": sorted(saved_visual_blocks),
        "merger_tensor_count": len(merger_tensors),
        "merger_changed_tensor_count": merger_changed,
        "merger_max_abs_delta": merger_max_abs_delta,
        "lora_b_nonzero_elements": lora_b_nonzero,
        "lora_b_total_elements": lora_b_elements,
        "train_runtime_seconds": runtime,
        "train_samples_per_second": float(train_results["train_samples_per_second"]),
        "adapter_model_bytes": adapter_model_path.stat().st_size,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = verify(args.manifest, args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
