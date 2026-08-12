#!/usr/bin/env python3
"""Compare effective LoRA update directions and paired decisions for RL arms."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def adapter_scale(adapter: Path) -> float:
    config = load_json(adapter / "adapter_config.json")
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    return alpha / math.sqrt(rank) if config.get("use_rslora") else alpha / rank


def lora_prefixes(adapter: Path) -> list[str]:
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        return sorted(
            key.removesuffix(".lora_A.weight")
            for key in handle.keys()
            if key.endswith(".lora_A.weight")
        )


def module_metadata(prefix: str) -> tuple[str, str]:
    match = LAYER_PATTERN.search(prefix)
    layer = f"layer_{int(match.group(1)):02d}" if match else "non_layer"
    module = prefix.rsplit(".", 1)[-1]
    return layer, module


def compare_adapters(left: Path, right: Path) -> dict:
    left_prefixes, right_prefixes = lora_prefixes(left), lora_prefixes(right)
    if left_prefixes != right_prefixes:
        raise ValueError("LoRA module sets do not match")
    left_scale, right_scale = adapter_scale(left), adapter_scale(right)
    groups: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    per_module = []
    factor_a_dot = factor_a_left_sq = factor_a_right_sq = 0.0
    with safe_open(str(left / "adapter_model.safetensors"), framework="pt", device="cpu") as lhs, safe_open(
        str(right / "adapter_model.safetensors"), framework="pt", device="cpu"
    ) as rhs:
        for prefix in left_prefixes:
            a1 = lhs.get_tensor(prefix + ".lora_A.weight").float()
            b1 = lhs.get_tensor(prefix + ".lora_B.weight").float()
            a2 = rhs.get_tensor(prefix + ".lora_A.weight").float()
            b2 = rhs.get_tensor(prefix + ".lora_B.weight").float()
            factor_a_dot += float((a1 * a2).sum().double())
            factor_a_left_sq += float((a1 * a1).sum().double())
            factor_a_right_sq += float((a2 * a2).sum().double())
            # <B1 A1, B2 A2> = sum((B1^T B2) * (A1 A2^T)); no dense delta is formed.
            inner = left_scale * right_scale * float(((b1.T @ b2) * (a1 @ a2.T)).sum().double())
            norm1_sq = left_scale * left_scale * float(((b1.T @ b1) * (a1 @ a1.T)).sum().double())
            norm2_sq = right_scale * right_scale * float(((b2.T @ b2) * (a2 @ a2.T)).sum().double())
            layer, module = module_metadata(prefix)
            cosine = inner / math.sqrt(norm1_sq * norm2_sq) if norm1_sq > 0 and norm2_sq > 0 else None
            per_module.append(
                {
                    "prefix": prefix,
                    "layer": layer,
                    "module": module,
                    "cosine": cosine,
                    "left_norm": math.sqrt(max(norm1_sq, 0.0)),
                    "right_norm": math.sqrt(max(norm2_sq, 0.0)),
                }
            )
            for key in (("overall", "all"), ("layer", layer), ("module", module)):
                values = groups[key]
                values[0] += inner
                values[1] += norm1_sq
                values[2] += norm2_sq

    def group_rows(kind: str) -> list[dict]:
        rows = []
        for (group_kind, name), (inner, norm1_sq, norm2_sq) in sorted(groups.items()):
            if group_kind != kind:
                continue
            rows.append(
                {
                    "name": name,
                    "cosine": inner / math.sqrt(norm1_sq * norm2_sq),
                    "left_norm": math.sqrt(max(norm1_sq, 0.0)),
                    "right_norm": math.sqrt(max(norm2_sq, 0.0)),
                }
            )
        return rows

    factor_a_cosine = factor_a_dot / math.sqrt(factor_a_left_sq * factor_a_right_sq)
    return {
        "left": str(left.resolve()),
        "right": str(right.resolve()),
        "module_count": len(left_prefixes),
        "factor_A_cosine": factor_a_cosine,
        "shared_low_rank_basis_supported": factor_a_cosine >= 0.99,
        "causal_recipe_direction_interpretation_valid": factor_a_cosine >= 0.99,
        "direction_interpretation_warning": (
            None if factor_a_cosine >= 0.99 else
            "Final LoRA A factors are not aligned. Effective-update cosine is mathematically exact "
            "for the realized adapters but cannot isolate optimizer-recipe direction from independently "
            "initialized low-rank subspaces."
        ),
        "overall": group_rows("overall")[0],
        "by_layer": group_rows("layer"),
        "by_module": group_rows("module"),
        "per_layer_module": per_module,
    }


def paired_flips(parent_path: Path, candidate_path: Path) -> dict:
    parent, candidate = load_jsonl(parent_path), load_jsonl(candidate_path)
    identities = [row["source_record_sha256"] for row in parent]
    if identities != [row["source_record_sha256"] for row in candidate]:
        raise ValueError("paired prediction identity mismatch")
    wrong_to_right = right_to_wrong = stable_right = stable_wrong = 0
    for before, after in zip(parent, candidate):
        before_correct = bool(before["accuracy_reward"])
        after_correct = bool(after["accuracy_reward"])
        wrong_to_right += int(not before_correct and after_correct)
        right_to_wrong += int(before_correct and not after_correct)
        stable_right += int(before_correct and after_correct)
        stable_wrong += int(not before_correct and not after_correct)
    changed = wrong_to_right + right_to_wrong
    smaller = min(wrong_to_right, right_to_wrong)
    p_value = 1.0 if changed == 0 else min(
        1.0, 2 * sum(math.comb(changed, index) for index in range(smaller + 1)) / 2**changed
    )
    return {
        "count": len(parent),
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "net_correct": wrong_to_right - right_to_wrong,
        "stable_right": stable_right,
        "stable_wrong": stable_wrong,
        "decision_churn_rate": changed / len(parent),
        "mcnemar_exact_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent-eval-root", required=True, type=Path)
    parser.add_argument("--phase1-eval-root", required=True, type=Path)
    parser.add_argument("--phase2-eval-root", required=True, type=Path)
    parser.add_argument("--r1-adapter", required=True, type=Path)
    parser.add_argument("--lr3-adapter", required=True, type=Path)
    parser.add_argument("--g2-adapter", required=True, type=Path)
    args = parser.parse_args()
    adapters = {"g20_lr1": args.r1_adapter, "g20_lr3": args.lr3_adapter, "g2_lr1": args.g2_adapter}
    directions = []
    names = list(adapters)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            row = compare_adapters(adapters[left], adapters[right])
            row.update(left_arm=left, right_arm=right)
            directions.append(row)
    eval_paths = {
        "g20_lr1": {
            "train_probe": args.phase1_eval_root / "train_probe/r1_step025/predictions.jsonl",
            "pathmmu_val": args.phase1_eval_root / "pathmmu_val/r1_step025/predictions.jsonl",
        },
        "g20_lr3": {
            "train_probe": args.phase2_eval_root / "train_probe/g20_lr3_step025/predictions.jsonl",
            "pathmmu_val": args.phase2_eval_root / "pathmmu_val/g20_lr3_step025/predictions.jsonl",
        },
        "g2_lr1": {
            "train_probe": args.phase2_eval_root / "train_probe/g2_lr1_step250/predictions.jsonl",
            "pathmmu_val": args.phase2_eval_root / "pathmmu_val/g2_lr1_step250/predictions.jsonl",
        },
    }
    parent_paths = {
        "train_probe": args.parent_eval_root / "train_probe/parent/predictions.jsonl",
        "pathmmu_val": args.parent_eval_root / "pathmmu_val/parent/predictions.jsonl",
    }
    flips = {
        arm: {split: paired_flips(parent_paths[split], path) for split, path in split_paths.items()}
        for arm, split_paths in eval_paths.items()
    }
    payload = {
        "schema_version": 1,
        "status": "completed",
        "claim_boundary": (
            "Cosines compare exact realized effective LoRA BA updates. A causal optimizer-recipe "
            "direction claim is forbidden when the final A factors do not support a shared low-rank "
            "basis. Paired flips use deterministic greedy predictions against the same SFT parent."
        ),
        "test_accessed": False,
        "update_direction_pairs": directions,
        "paired_flips_vs_sft_parent": flips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    concise = {
        "overall_cosines": {
            f"{row['left_arm']}__{row['right_arm']}": row["overall"]["cosine"] for row in directions
        },
        "paired_flips_vs_sft_parent": flips,
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
