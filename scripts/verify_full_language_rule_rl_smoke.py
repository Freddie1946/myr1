#!/usr/bin/env python3
"""Fail-closed verification for the full-language, frozen-vision GRPO smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing required audit file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-step", type=int, default=2)
    parser.add_argument("--load-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = run_dir / "output"
    checkpoint = output_dir / f"checkpoint-{args.expected_step}"
    launch = load_json(run_dir / "launch_contract.json")
    audit = load_json(output_dir / "pathvlm_train_state_audit.json")
    reward = load_json(run_dir / "reward_alignment_verification.json")
    load_metrics = load_json(args.load_metrics.resolve())

    trainability = audit.get("trainability", {})
    parameters = trainability.get("parameters", {})
    language = parameters.get("language", {})
    visual = parameters.get("visual", {})
    projector = parameters.get("multimodal_projector", {})
    history = [row for row in audit.get("log_history", []) if "grad_norm" in row]
    checkpoint_weight_files = sorted(
        path.name for path in checkpoint.glob("*.safetensors")
    )
    if (checkpoint / "model.safetensors.index.json").is_file():
        checkpoint_weight_files.append("model.safetensors.index.json")

    gates = {
        "launch_declares_full_language": launch.get("trainable_parameters") == "full language model"
        and launch.get("use_peft") is False,
        "launch_freezes_visual_and_projector": launch.get("vision_encoder") == "frozen"
        and launch.get("multimodal_projector") == "frozen",
        "completed_expected_step": audit.get("global_step") == args.expected_step,
        "trainability_report_passed": trainability.get("passed") is True,
        "runtime_language_mode_full": trainability.get("language_mode") == "full",
        "all_language_parameters_trainable": int(language.get("total", 0)) > 0
        and int(language.get("trainable", -1)) == int(language.get("total", 0)),
        "visual_fully_frozen": int(visual.get("total", 0)) > 0
        and int(visual.get("trainable", -1)) == 0,
        "projector_fully_frozen": int(projector.get("total", 0)) > 0
        and int(projector.get("trainable", -1)) == 0,
        "finite_nonzero_gradients": len(history) == args.expected_step
        and all(math.isfinite(float(row["grad_norm"])) and float(row["grad_norm"]) > 0 for row in history),
        "finite_rewards_and_kl": len(history) == args.expected_step
        and all(
            math.isfinite(float(row[key]))
            for row in history
            for key in ("reward", "reward_std", "kl")
        ),
        "reward_alignment_passed": reward.get("passed") is True,
        "checkpoint_metadata_present": all(
            (checkpoint / name).is_file()
            for name in ("config.json", "trainer_state.json", "tokenizer_config.json")
        ),
        "checkpoint_weights_present": bool(checkpoint_weight_files),
        "checkpoint_load_completed": load_metrics.get("status") == "completed"
        and int(load_metrics.get("count", 0)) == 8,
        "checkpoint_load_not_test": load_metrics.get("test_accessed") is False,
        "checkpoint_load_no_truncation": int(load_metrics.get("generation_cap_hit_count", -1)) == 0,
    }
    result = {
        "schema_version": 1,
        "passed": all(gates.values()),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "expected_step": args.expected_step,
        "gates": gates,
        "parameter_counts": {
            "language": language,
            "visual": visual,
            "multimodal_projector": projector,
        },
        "gradient_norms": [float(row["grad_norm"]) for row in history],
        "checkpoint_weight_files": checkpoint_weight_files,
        "load_metrics": load_metrics,
    }
    output = args.output or run_dir / "smoke_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
