#!/usr/bin/env python3
"""Summarize matched-exposure R1/G20-LR3/G2-LR1 screening evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from summarize_rule_rl_reward_isolation import load_jsonl, reward_prefix


def load(path: Path):
    return json.loads(path.read_text())


def direct_flips(reference: Path, candidate: Path) -> dict:
    left, right = load_jsonl(reference), load_jsonl(candidate)
    if [x["source_record_sha256"] for x in left] != [x["source_record_sha256"] for x in right]:
        raise RuntimeError("paired prediction identity mismatch")
    reference_only = candidate_only = 0
    for before, after in zip(left, right):
        a, b = bool(before["accuracy_reward"]), bool(after["accuracy_reward"])
        reference_only += int(a and not b)
        candidate_only += int(b and not a)
    n = reference_only + candidate_only
    smaller = min(reference_only, candidate_only)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(smaller + 1)) / 2**n)
    return {
        "reference_only_correct": reference_only,
        "candidate_only_correct": candidate_only,
        "net_candidate_correct": candidate_only - reference_only,
        "mcnemar_exact_p": p,
    }


def effective_lora_delta_norm(adapter: Path) -> float:
    config = load(adapter / "adapter_config.json")
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    scale = alpha / math.sqrt(rank) if config.get("use_rslora") else alpha / rank
    state = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    prefixes = sorted(key.removesuffix(".lora_A.weight") for key in state if key.endswith(".lora_A.weight"))
    total = 0.0
    for prefix in prefixes:
        a = state[prefix + ".lora_A.weight"].double()
        b = state[prefix + ".lora_B.weight"].double()
        aa = a @ a.T
        bb = b.T @ b
        total += scale * scale * float((aa * bb.T).sum())
    return math.sqrt(max(total, 0.0))


def row(name: str, run_dir: Path, checkpoint: Path, eval_root: Path, eval_name: str, step: int) -> dict:
    contract = load(run_dir / "launch_contract.json")
    audit = load(run_dir / "output/pathvlm_train_state_audit.json")
    logs = [item for item in audit["log_history"] if int(item.get("step", 0)) <= step and "grad_norm" in item]
    train = load(eval_root / f"train_probe/{eval_name}/metrics.json")
    val = load(eval_root / f"pathmmu_val/{eval_name}/metrics.json")
    return {
        "arm": name,
        "step": step,
        "prompt_exposure": step * int(contract["unique_prompts_per_optimizer_step"]),
        "optimizer_updates": step,
        "learning_rate": contract["learning_rate"],
        "prompts_per_update": contract["unique_prompts_per_optimizer_step"],
        "sampled": reward_prefix(run_dir, step, "accuracy_only"),
        "mean_trainer_kl": sum(float(item["kl"]) for item in logs) / len(logs),
        "final_trainer_kl": float(logs[-1]["kl"]),
        "mean_grad_norm": sum(float(item["grad_norm"]) for item in logs) / len(logs),
        "effective_lora_delta_frobenius": effective_lora_delta_norm(checkpoint),
        "train_probe": {"correct": train["correct"], "count": train["count"], "accuracy": train["accuracy"], "format_rate": train["format_correct"] / train["count"]},
        "pathmmu_val": {"correct": val["correct"], "count": val["count"], "accuracy": val["accuracy"], "format_rate": val["format_correct"] / val["count"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--phase1-train-root", required=True, type=Path)
    parser.add_argument("--phase1-eval-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline_run = args.phase1_train_root / "phase1/r1_short"
    baseline_ckpt = baseline_run / "output/checkpoint-25"
    # Reuse immutable Phase-1 metrics by referencing them in a temporary row-shaped view.
    baseline = row("r1_g20_lr1", baseline_run, baseline_ckpt, args.phase1_eval_root, "r1_step025", 25)
    lr3_run = args.train_root / "g20_lr3"
    g2_run = args.train_root / "g2_lr1"
    lr3 = row("r1_g20_lr3", lr3_run, lr3_run / "output/checkpoint-25", args.eval_root, "g20_lr3_step025", 25)
    g2 = row("r1_g2_lr1", g2_run, g2_run / "output/checkpoint-250", args.eval_root, "g2_lr1_step250", 250)
    baseline_sampled_format = float(baseline["sampled"]["sampled_format_rate"])
    for candidate, eval_name in ((lr3, "g20_lr3_step025"), (g2, "g2_lr1_step250")):
        candidate["relative_sampled_format_drop"] = baseline_sampled_format - float(candidate["sampled"]["sampled_format_rate"])
        candidate["format_gate_triggered"] = (
            candidate["relative_sampled_format_drop"] > 0.05
            or candidate["train_probe"]["format_rate"] < 0.98
            or candidate["pathmmu_val"]["format_rate"] < 0.98
        )
        for split in ("train_probe", "pathmmu_val"):
            candidate[split]["paired_vs_r1_g20_lr1"] = direct_flips(
                args.phase1_eval_root / f"{split}/r1_step025/predictions.jsonl",
                args.eval_root / f"{split}/{eval_name}/predictions.jsonl",
            )
    payload = {
        "schema_version": 1,
        "status": "lean_phase2_screening_complete_manual_analysis_required",
        "claim_boundary": "G2 versus G20 is a small-batch/high-update-frequency recipe comparison, not a pure optimizer-granularity isolation.",
        "matched_prompt_exposure": 500,
        "test_accessed": False,
        "rows": [baseline, lr3, g2],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
