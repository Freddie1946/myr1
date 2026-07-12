#!/usr/bin/env python3
"""Compare representative language and visual tensors between two sharded checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def index_map(checkpoint: Path) -> dict[str, str]:
    payload = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return payload["weight_map"]


def choose_tensor(names: set[str], kind: str) -> str:
    if kind == "language":
        preferred = [
            name
            for name in names
            if "layers.0.self_attn.q_proj.weight" in name and "visual" not in name
        ]
        fallback = [name for name in names if "visual" not in name and name.endswith("weight")]
    else:
        preferred = [name for name in names if "visual.blocks.0" in name and name.endswith("weight")]
        fallback = [name for name in names if "visual" in name and name.endswith("weight")]
    choices = sorted(preferred or fallback)
    if not choices:
        raise ValueError(f"no {kind} tensor candidate found")
    return choices[0]


def load_tensor(checkpoint: Path, weights: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(checkpoint / weights[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def compare(parent: torch.Tensor, child: torch.Tensor) -> dict:
    if parent.shape != child.shape:
        raise ValueError(f"shape mismatch: {parent.shape} != {child.shape}")
    delta = child.float() - parent.float()
    return {
        "shape": list(parent.shape),
        "parent_dtype": str(parent.dtype),
        "child_dtype": str(child.dtype),
        "numel": parent.numel(),
        "changed_elements": int(torch.count_nonzero(delta).item()),
        "max_abs_delta": float(delta.abs().max().item()),
        "mean_abs_delta": float(delta.abs().mean().item()),
        "l2_delta": float(torch.linalg.vector_norm(delta).item()),
        "exactly_equal": bool(torch.equal(parent, child)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--child", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    parent_map = index_map(args.parent)
    child_map = index_map(args.child)
    common = set(parent_map).intersection(child_map)
    result = {
        "parent": str(args.parent),
        "child": str(args.child),
        "common_tensor_count": len(common),
        "tensors": {},
    }
    for kind in ("language", "visual"):
        name = choose_tensor(common, kind)
        result["tensors"][kind] = {
            "name": name,
            **compare(
                load_tensor(args.parent, parent_map, name),
                load_tensor(args.child, child_map, name),
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
