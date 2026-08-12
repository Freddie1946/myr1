#!/usr/bin/env python3
"""Reproduce the frozen visual-evidence suite for n=4 step 500 versus step 1000.

The sequence waits for the clean n=4/n=8 training queue to release all accelerators, then runs two
checkpoint-matched diagnostics on the same model-blind 96-case PathMMU validation panel:

1. appearance-matched paired-image counterfactuals;
2. option-conditioned RISE with independent hard deletion/retention validation.

Raw attention is intentionally not promoted: its previous spatial confirmation gate failed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from run_option_conditioned_rise_confirmation import aggregate as aggregate_rise
from run_visual_understanding_counterfactual import aggregate as aggregate_counterfactual
from run_option_conditioned_visual_evidence import sha256_file


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
TRAIN_QUEUE_STATE = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
    "full_language_rule_rl_clean_n4_n8_step1000_20260812/sequence_state.json"
)
PANEL = REPO / "protocol/visual_understanding_confirmation_panel_v2_96case_20260808.json"
PANEL_SHA256 = "c1222d9ff2d3a1b038a744b2f1c1addd18e777de3fb4f31c09107acef7e52931"
PAIRS = REPO / "protocol/visual_understanding_counterfactual_pairs_v1_96case_20260808.json"
PAIRS_SHA256 = "69bbe60cc057266448cb71bcc0bd6ff52281b9bd298995a7d0cb314be4e7a4d6"
ROOT = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
    "full_language_rule_rl_n4_step500_to1000_explainability_20260812"
)
STATE = ROOT / "sequence_state.json"
MODELS = {
    "n4_step500": Path(
        "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
        "full_language_rule_rl_capacity_20260811/model_snapshots/checkpoint-500"
    ),
    "n4_step1000_continued": Path(
        "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
        "full_language_rule_rl_n4_epoch2_resume_20260812/model_snapshots/checkpoint-1000"
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def gpu_processes() -> list[str]:
    output = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def wait_for_training_and_gpus(state: dict[str, Any]) -> None:
    while True:
        status = None
        if TRAIN_QUEUE_STATE.is_file():
            status = json.loads(TRAIN_QUEUE_STATE.read_text(encoding="utf-8")).get("status")
        if status == "failed":
            raise RuntimeError("clean n=4/n=8 training queue failed")
        active = gpu_processes()
        if status == "completed" and not active:
            return
        state.update(
            stage="waiting_for_clean_training_queue",
            prerequisite_status=status,
            active_gpu_process_count=len(active),
            updated_at=now(),
        )
        write_json(STATE, state)
        time.sleep(60)


def run_sharded_phase(
    phase: str,
    command_builder: Callable[[str, Path, int, Path], list[str]],
    state: dict[str, Any],
) -> dict[str, list[Path]]:
    processes: list[tuple[subprocess.Popen[Any], Any, str, int, Path]] = []
    outputs: dict[str, list[Path]] = {label: [] for label in MODELS}
    gpu = 0
    for label, model in MODELS.items():
        for shard in range(4):
            output = ROOT / phase / label / f"shard{shard}"
            if output.exists():
                raise FileExistsError(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            log_path = ROOT / "logs" / f"{phase}_{label}_shard{shard}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": str(REPO / "scripts"),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            process = subprocess.Popen(
                command_builder(label, model, shard, output), cwd=REPO, env=env,
                stdout=handle, stderr=subprocess.STDOUT,
            )
            processes.append((process, handle, label, shard, output))
            outputs[label].append(output)
            gpu += 1
    state.update(stage=phase, updated_at=now())
    write_json(STATE, state)
    failures = []
    for process, handle, label, shard, output in processes:
        code = process.wait()
        handle.close()
        if code != 0 or not (output / "metrics.json").is_file():
            failures.append({"label": label, "shard": shard, "returncode": code})
    if failures:
        raise RuntimeError(f"{phase} shard failures: {failures}")
    return outputs


def counterfactual_command(label: str, model: Path, shard: int, output: Path) -> list[str]:
    command = [
        str(PYTHON), str(REPO / "scripts/run_visual_understanding_counterfactual.py"),
        "--model", str(model), "--model-label", label,
        "--panel", str(PANEL), "--expected-panel-sha256", PANEL_SHA256,
        "--counterfactual-manifest", str(PAIRS),
        "--expected-counterfactual-sha256", PAIRS_SHA256,
        "--output-dir", str(output),
    ]
    for index in range(shard, 96, 4):
        command.extend(("--panel-index", str(index)))
    return command


def rise_command(label: str, model: Path, shard: int, output: Path) -> list[str]:
    command = [
        str(PYTHON), str(REPO / "scripts/run_option_conditioned_rise_confirmation.py"),
        "--model", str(model), "--model-label", label,
        "--panel", str(PANEL), "--expected-panel-sha256", PANEL_SHA256,
        "--output-dir", str(output), "--rise-count", "256", "--rise-cells", "7",
        "--random-controls", "5", "--attribution-long-edge", "256", "--batch-size", "1",
    ]
    for index in range(shard, 96, 4):
        command.extend(("--panel-index", str(index)))
    return command


def merge_shards(
    phase: str, label: str, shards: list[Path],
    aggregate: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> tuple[Path, list[dict[str, Any]]]:
    rows = []
    for shard in shards:
        rows.extend(
            json.loads(line) for line in (shard / "case_results.jsonl").open(encoding="utf-8")
            if line.strip()
        )
    rows.sort(key=lambda row: row["panel_index"])
    if [row["panel_index"] for row in rows] != list(range(96)):
        raise RuntimeError(f"{phase}/{label}: incomplete or duplicate panel")
    merged = ROOT / phase / label / "merged"
    merged.mkdir()
    cases = merged / "case_results.jsonl"
    cases.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metrics = {
        "schema_version": 1,
        "created_at": now(),
        "status": "completed_confirmation",
        "formal_result": False,
        "longitudinal_reuse_of_previously_frozen_panel": True,
        "model_label": label,
        "model_path": str(MODELS[label].resolve()),
        "model_config_sha256": sha256_file(MODELS[label] / "config.json"),
        "panel_path": str(PANEL.resolve()),
        "panel_sha256": PANEL_SHA256,
        "panel_indices": list(range(96)),
        "case_count": 96,
        "case_results": str(cases.resolve()),
        "case_results_sha256": sha256_file(cases),
        **aggregate(rows),
    }
    write_json(merged / "metrics.json", metrics)
    return merged / "metrics.json", rows


def bootstrap_ci(values: list[float], name: str) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def paired_summary(
    counterfactual: dict[str, list[dict[str, Any]]],
    rise: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    def counterfactual_gain(row: dict[str, Any]) -> float:
        conditions = row["conditions"]
        return float(
            conditions["original"]["target_margin"]
            - conditions["nearest_same_target"]["target_margin"]
        )

    def rise_advantage(row: dict[str, Any], mode: str) -> float:
        random_auc = float(np.mean([
            control["target_margin_auc"]
            for control in row["curves"][mode]["random_shifted_shape_matched"]
        ]))
        observed = float(row["curves"][mode]["rise_high"]["target_margin_auc"])
        return random_auc - observed if mode == "deletion" else observed - random_auc

    before_cf = counterfactual["n4_step500"]
    after_cf = counterfactual["n4_step1000_continued"]
    cf_delta = [counterfactual_gain(b) - counterfactual_gain(a) for a, b in zip(before_cf, after_cf)]
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_post_hoc_longitudinal_comparison",
        "comparison": "same n4 full-RL trajectory, step1000 continued minus step500",
        "claim_boundary": (
            "The model-blind panel was frozen before the original visualization runs, but this is "
            "a post-hoc longitudinal reuse and not a new independent confirmation panel."
        ),
        "paired_image_same_target_margin_gain_delta": {
            "mean": float(np.mean(cf_delta)),
            "bootstrap_95ci": bootstrap_ci(cf_delta, "n4-500-1000-counterfactual"),
            "positive_case_rate": float(np.mean(np.asarray(cf_delta) > 0)),
        },
        "rise": {},
    }
    for mode in ("deletion", "retention"):
        before = [rise_advantage(row, mode) for row in rise["n4_step500"]]
        after = [rise_advantage(row, mode) for row in rise["n4_step1000_continued"]]
        delta = [b - a for a, b in zip(before, after)]
        output["rise"][mode] = {
            "step500_mean_fidelity_advantage": float(np.mean(before)),
            "step1000_mean_fidelity_advantage": float(np.mean(after)),
            "step1000_minus_step500_mean": float(np.mean(delta)),
            "step1000_minus_step500_bootstrap_95ci": bootstrap_ci(
                delta, f"n4-500-1000-rise-{mode}"
            ),
            "positive_case_rate": float(np.mean(np.asarray(delta) > 0)),
        }
    return output


def main() -> None:
    if STATE.exists():
        raise FileExistsError(STATE)
    for path in (PANEL, PAIRS, *(model / "model.safetensors.index.json" for model in MODELS.values())):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(PANEL) != PANEL_SHA256 or sha256_file(PAIRS) != PAIRS_SHA256:
        raise RuntimeError("frozen explainability inputs changed")
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "stage": "queued",
        "started_at": now(),
        "models": {key: str(value.resolve()) for key, value in MODELS.items()},
        "methods": ["paired_image_counterfactual", "option_conditioned_rise"],
        "raw_attention_promoted": False,
        "training_queue_prerequisite": str(TRAIN_QUEUE_STATE),
    }
    write_json(STATE, state)
    try:
        wait_for_training_and_gpus(state)
        cf_shards = run_sharded_phase("paired_image_counterfactual", counterfactual_command, state)
        cf_rows: dict[str, list[dict[str, Any]]] = {}
        for label, shards in cf_shards.items():
            _, cf_rows[label] = merge_shards(
                "paired_image_counterfactual", label, shards, aggregate_counterfactual
            )
        rise_shards = run_sharded_phase("option_conditioned_rise", rise_command, state)
        rise_rows: dict[str, list[dict[str, Any]]] = {}
        for label, shards in rise_shards.items():
            _, rise_rows[label] = merge_shards(
                "option_conditioned_rise", label, shards, aggregate_rise
            )
        write_json(ROOT / "paired_step500_to1000_summary.json", paired_summary(cf_rows, rise_rows))
        state.update(status="completed", stage="complete", finished_at=now())
        write_json(STATE, state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), finished_at=now())
        write_json(STATE, state)
        raise


if __name__ == "__main__":
    main()
