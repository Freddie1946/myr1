#!/usr/bin/env python3
"""Core step-1000 evaluation concurrently with the clean n=4 training arm."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
MODEL = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
    "full_language_rule_rl_n4_epoch2_resume_20260812/model_snapshots/checkpoint-1000"
)
OUTPUT = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
    "full_language_rule_rl_n4_epoch2_resume_20260812/step1000_core_concurrent"
)
TRAIN_PROBE = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
    "rule_rl_mechanism_train_probe256_20260811/records_n0256.json"
)
PATHMMU_VAL = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/"
    "pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json"
)
PATHMMU_TEST = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/"
    "pathmmu_image_disjoint_v2/rewritten_records/test_0999.json"
)
PATHVQA_VAL = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
    "pathvqa_architecture_validation_v1_20260811/"
    "pathvqa_validation_balanced_image_unique_512.json"
)
MMMU = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
    "mmmu_nonmedical_dev_retention_v1_20260811/panel.json"
)
STATE = OUTPUT / "evaluation_state.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_command(name: str, gpu: int, command: list[str], output: Path, state: dict,
                lock: threading.Lock, command_output_root: Path | None = None) -> None:
    # Most runners receive ``output`` directly.  Multi-contract calibration receives
    # an output *root* and creates a nested contract directory itself; pre-creating
    # that root would make its safety check fail.
    (command_output_root or output).parent.mkdir(parents=True, exist_ok=True)
    log = OUTPUT / "logs" / f"{name}.log"
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(REPO / "scripts"),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    with lock:
        state["jobs"][name] = {"gpu": gpu, "status": "running", "started_at": now()}
        write_json(STATE, state)
    with log.open("w") as handle:
        code = subprocess.run(command, cwd=REPO, env=env, stdout=handle,
                              stderr=subprocess.STDOUT).returncode
    success = code == 0 and (output / "metrics.json").is_file()
    with lock:
        state["jobs"][name].update(
            status="completed" if success else "failed", returncode=code,
            finished_at=now(),
        )
        write_json(STATE, state)
    if not success:
        raise RuntimeError(f"evaluation job failed: {name}")


def lane_pathmmu(state: dict, lock: threading.Lock, failures: list[str]) -> None:
    qwen = str(REPO / "scripts/run_pathmmu_qwen_diagnostic.py")
    jobs = (
        ("train_probe", TRAIN_PROBE, "rl_train_probe", OUTPUT / "train_probe"),
        ("pathmmu_val", PATHMMU_VAL, "validation_smoke", OUTPUT / "pathmmu_val"),
        ("pathmmu_test999", PATHMMU_TEST, "test999_development", OUTPUT / "pathmmu_test999"),
    )
    try:
        for name, data, role, output in jobs:
            run_command(name, 0, [
                str(PYTHON), qwen, "--model", str(MODEL), "--backend", "qwen2_5_vl",
                "--data", str(data), "--output-dir", str(output), "--batch-size", "8",
                "--split-role", role, "--max-new-tokens", "1024",
            ], output, state, lock)
    except BaseException as error:
        failures.append(f"pathmmu_lane: {error!r}")


def lane_pathvqa(state: dict, lock: threading.Lock, failures: list[str]) -> None:
    try:
        forced = OUTPUT / "pathvqa_val/forced_binary_normal"
        run_command("pathvqa_forced_binary_normal", 1, [
            str(PYTHON), str(REPO / "scripts/run_pathvqa_forced_binary_logits.py"),
            "--model", str(MODEL), "--data", str(PATHVQA_VAL),
            "--output-dir", str(forced), "--batch-size", "8",
            "--split-role", "validation_diagnostic", "--image-mode", "original",
        ], forced, state, lock)
        generated_root = OUTPUT / "pathvqa_val/pathmmu_ab_v1_2048"
        run_command("pathvqa_ab_generated", 1, [
            str(PYTHON), str(REPO / "scripts/run_pathvqa_yesno_prompt_calibration.py"),
            "--model", str(MODEL), "--panel", str(PATHVQA_VAL),
            "--panel-role", "architecture_validation_512",
            "--output-root", str(generated_root), "--batch-size", "4",
            "--prompt-contracts", "pathmmu_ab_v1_2048",
        ], generated_root / "pathmmu_ab_v1_2048", state, lock,
            command_output_root=generated_root)
    except BaseException as error:
        failures.append(f"pathvqa_lane: {error!r}")


def lane_mmmu(state: dict, lock: threading.Lock, failures: list[str]) -> None:
    output = OUTPUT / "mmmu"
    try:
        run_command("mmmu", 2, [
            str(PYTHON), str(REPO / "scripts/run_mmmu_retention_diagnostic.py"),
            "--model", str(MODEL), "--panel", str(MMMU),
            "--output-dir", str(output), "--batch-size", "2",
            "--max-new-tokens", "4096",
        ], output, state, lock)
    except BaseException as error:
        failures.append(f"mmmu_lane: {error!r}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in (MODEL / "model.safetensors.index.json", TRAIN_PROBE, PATHMMU_VAL,
                 PATHMMU_TEST, PATHVQA_VAL, MMMU):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "logs").mkdir()
    state = {
        "schema_version": 1, "status": "running", "started_at": now(),
        "model": str(MODEL.resolve()), "concurrent_with_training": True,
        "test999_role": "development diagnostic; not a training continuation gate",
        "jobs": {},
    }
    write_json(STATE, state)
    lock = threading.Lock()
    failures: list[str] = []
    threads = [
        threading.Thread(target=lane_pathmmu, args=(state, lock, failures)),
        threading.Thread(target=lane_pathvqa, args=(state, lock, failures)),
        threading.Thread(target=lane_mmmu, args=(state, lock, failures)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        state.update(status="failed", failures=failures, finished_at=now())
        write_json(STATE, state)
        raise SystemExit("; ".join(failures))
    metrics = {}
    for name, path in {
        "train_probe": OUTPUT / "train_probe/metrics.json",
        "pathmmu_val": OUTPUT / "pathmmu_val/metrics.json",
        "pathmmu_test999": OUTPUT / "pathmmu_test999/metrics.json",
        "pathvqa_forced_binary_normal": OUTPUT / "pathvqa_val/forced_binary_normal/metrics.json",
        "pathvqa_ab_generated": OUTPUT / "pathvqa_val/pathmmu_ab_v1_2048/pathmmu_ab_v1_2048/metrics.json",
        "mmmu": OUTPUT / "mmmu/metrics.json",
    }.items():
        metrics[name] = json.loads(path.read_text())
    write_json(OUTPUT / "complete_summary.json", {
        "schema_version": 1, "status": "completed", "metrics": metrics,
    })
    state.update(status="completed", finished_at=now())
    write_json(STATE, state)


if __name__ == "__main__":
    main()
