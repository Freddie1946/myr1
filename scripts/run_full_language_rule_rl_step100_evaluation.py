#!/usr/bin/env python3
"""Eight-A100 evaluation gate for the full-language GRPO step-100 checkpoint."""

from __future__ import annotations

import json
import os
import subprocess
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
MODEL = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
    "full_language_rule_rl_capacity_20260811/formal_g2_lr1_step100/output/checkpoint-100"
)
OUTPUT = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
    "full_language_rule_rl_capacity_20260811/step100_gate"
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
PATHVQA_TEST = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
    "external_vqa_contract_v1_20260729/pathvqa_test_6719.json"
)
MMMU = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
    "mmmu_nonmedical_dev_retention_v1_20260811/panel.json"
)


@dataclass(frozen=True)
class Job:
    name: str
    gpu: int
    command: tuple[str, ...]
    output: Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flips(reference_path: Path, candidate_path: Path) -> dict:
    reference = [json.loads(line) for line in reference_path.open(encoding="utf-8")]
    candidate = [json.loads(line) for line in candidate_path.open(encoding="utf-8")]
    if len(reference) != len(candidate):
        raise RuntimeError("paired prediction counts differ")
    wrong_to_right = right_to_wrong = both_right = both_wrong = 0
    for left, right in zip(reference, candidate):
        if left["source_record_sha256"] != right["source_record_sha256"]:
            raise RuntimeError("paired source records differ")
        a = bool(left["accuracy_reward"])
        b = bool(right["accuracy_reward"])
        if not a and b:
            wrong_to_right += 1
        elif a and not b:
            right_to_wrong += 1
        elif a and b:
            both_right += 1
        else:
            both_wrong += 1
    return {
        "count": len(reference),
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "net_correct_change": wrong_to_right - right_to_wrong,
        "both_right": both_right,
        "both_wrong": both_wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    model = args.model.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    required = (model / "model.safetensors.index.json", TRAIN_PROBE, PATHMMU_VAL,
                PATHMMU_TEST, PATHVQA_VAL, PATHVQA_TEST, MMMU)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    active = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    if active:
        raise RuntimeError(f"GPU compute processes already active: {active}")

    outputs = {
        "train_probe": output / "train_probe",
        "pathmmu_val": output / "pathmmu_val",
        "pathmmu_test999": output / "pathmmu_test999",
        "pathvqa_original": output / "pathvqa_val/original",
        "pathvqa_shuffle": output / "pathvqa_val/cyclic_mismatch",
        "pathvqa_blank": output / "pathvqa_val/global_mean_blank",
        "pathvqa_ab_generated": output / "pathvqa_val/pathmmu_ab_generated",
        "mmmu": output / "mmmu",
        "pathvqa_test_generated": output / "pathvqa_test_yesno3362_pathmmu_ab_generated",
    }
    qwen = str(REPO / "scripts/run_pathmmu_qwen_diagnostic.py")
    pathvqa = str(REPO / "scripts/run_pathvqa_forced_binary_logits.py")
    jobs = [
        Job("train_probe", 0, (str(PYTHON), qwen, "--model", str(model), "--backend", "qwen2_5_vl",
            "--data", str(TRAIN_PROBE), "--output-dir", str(outputs["train_probe"]),
            "--batch-size", "32", "--split-role", "rl_train_probe", "--max-new-tokens", "1024"), outputs["train_probe"]),
        Job("pathmmu_val", 1, (str(PYTHON), qwen, "--model", str(model), "--backend", "qwen2_5_vl",
            "--data", str(PATHMMU_VAL), "--output-dir", str(outputs["pathmmu_val"]),
            "--batch-size", "32", "--split-role", "validation_smoke", "--max-new-tokens", "1024"), outputs["pathmmu_val"]),
        Job("pathmmu_test999", 2, (str(PYTHON), qwen, "--model", str(model), "--backend", "qwen2_5_vl",
            "--data", str(PATHMMU_TEST), "--output-dir", str(outputs["pathmmu_test999"]),
            "--batch-size", "32", "--split-role", "test999_development", "--max-new-tokens", "1024"), outputs["pathmmu_test999"]),
    ]
    for name, gpu, mode in (
        ("pathvqa_original", 3, "original"),
        ("pathvqa_shuffle", 4, "cyclic_mismatch"),
        ("pathvqa_blank", 5, "global_mean_blank"),
    ):
        jobs.append(Job(name, gpu, (str(PYTHON), pathvqa, "--model", str(model),
            "--data", str(PATHVQA_VAL), "--output-dir", str(outputs[name]),
            "--batch-size", "32", "--split-role", "validation_diagnostic",
            "--image-mode", mode), outputs[name]))
    jobs.extend([
        Job("mmmu", 6, (str(PYTHON), str(REPO / "scripts/run_mmmu_retention_diagnostic.py"),
            "--model", str(model), "--panel", str(MMMU), "--output-dir", str(outputs["mmmu"]),
            "--batch-size", "4", "--max-new-tokens", "4096"), outputs["mmmu"]),
        Job("pathvqa_test_generated", 7, (str(PYTHON), str(REPO / "scripts/run_external_vqa_qwen.py"),
            "--task", "pathvqa", "--model", str(model), "--backend", "qwen2_5_vl",
            "--data", str(PATHVQA_TEST), "--output-dir", str(outputs["pathvqa_test_generated"]),
            "--split-role", "external_test", "--pathvqa-answer-scope", "yes_no_only",
            "--generation-contract", "pathvqa_pathmmu_ab_v6_2048",
            "--max-new-tokens", "2048", "--batch-size", "16"), outputs["pathvqa_test_generated"]),
    ])

    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    state = {
        "schema_version": 1, "status": "running", "started_at": now(),
        "test_accessed": True, "test_used_for_selection": False,
        "jobs": [], "automatic_retry": False,
    }
    running = []
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = str(REPO / "scripts")
    for job in jobs:
        job.output.parent.mkdir(parents=True, exist_ok=True)
        handle = (logs / f"{job.name}.log").open("w", encoding="utf-8")
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        process = subprocess.Popen(job.command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        running.append((process, job, handle))
        state["jobs"].append({"name": job.name, "gpu": job.gpu, "status": "running", "started_at": now()})
    state_path = output / "evaluation_state.json"
    write_json(state_path, state)
    failed = False
    while running:
        for process, job, handle in list(running):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            success = code == 0 and (job.output / "metrics.json").is_file()
            row = next(item for item in state["jobs"] if item["name"] == job.name)
            row.update(status="completed" if success else "failed", returncode=code, finished_at=now())
            failed = failed or not success
            running.remove((process, job, handle))
            write_json(state_path, state)
        if running:
            time.sleep(2)
    if failed:
        state.update(status="failed", finished_at=now())
        write_json(state_path, state)
        raise SystemExit("one or more evaluation jobs failed")

    # The 512-case aligned free-generation diagnostic is deliberately run after the
    # eight parallel lanes.  Its runner creates one directory per prompt contract,
    # so keeping it outside the generic Job abstraction also avoids pre-creating its
    # output root.  Forced binary remains available below only as a visual-dependence
    # diagnostic; this A/B generation result is the primary PathVQA validation score.
    ab_root = outputs["pathvqa_ab_generated"]
    ab_command = (
        str(PYTHON), str(REPO / "scripts/run_pathvqa_yesno_prompt_calibration.py"),
        "--model", str(model), "--panel", str(PATHVQA_VAL),
        "--panel-role", "architecture_validation_512",
        "--output-root", str(ab_root), "--batch-size", "16",
        "--prompt-contracts", "pathmmu_ab_v1_2048",
    )
    ab_log = logs / "pathvqa_ab_generated.log"
    ab_env = env_base.copy()
    ab_env["CUDA_VISIBLE_DEVICES"] = "0"
    state["jobs"].append({
        "name": "pathvqa_ab_generated", "gpu": 0, "status": "running",
        "started_at": now(),
    })
    write_json(state_path, state)
    with ab_log.open("w", encoding="utf-8") as handle:
        ab_code = subprocess.run(
            ab_command, cwd=REPO, env=ab_env, stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode
    ab_metrics_path = ab_root / "pathmmu_ab_v1_2048/metrics.json"
    ab_success = ab_code == 0 and ab_metrics_path.is_file()
    ab_row = next(item for item in state["jobs"] if item["name"] == "pathvqa_ab_generated")
    ab_row.update(
        status="completed" if ab_success else "failed", returncode=ab_code,
        finished_at=now(),
    )
    write_json(state_path, state)
    if not ab_success:
        state.update(status="failed", finished_at=now())
        write_json(state_path, state)
        raise SystemExit("PathVQA A/B validation evaluation failed")

    train = load(outputs["train_probe"] / "metrics.json")
    val = load(outputs["pathmmu_val"] / "metrics.json")
    test = load(outputs["pathmmu_test999"] / "metrics.json")
    normal = load(outputs["pathvqa_original"] / "metrics.json")
    shuffle = load(outputs["pathvqa_shuffle"] / "metrics.json")
    blank = load(outputs["pathvqa_blank"] / "metrics.json")
    pv_ab = load(ab_metrics_path)
    mmmu = load(outputs["mmmu"] / "metrics.json")
    pv_test = load(outputs["pathvqa_test_generated"] / "metrics.json")
    reference_root = Path(
        "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
        "formal_selected_sft3000_20260811"
    )
    phase1 = Path(
        "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
        "rule_rl_mechanism_funnel_20260811/formal_v1/phase1_core"
    )
    parent_val_predictions = reference_root / "pathmmu/step080/predictions.jsonl"
    if not parent_val_predictions.is_file():
        parent_val_predictions = phase1 / "pathmmu_val/parent/predictions.jsonl"
    summary = {
        "schema_version": 1,
        "status": "full_language_rule_rl_evaluation_complete",
        "test_used_for_selection": False,
        "claim_boundaries": {
            "pathmmu_test999": "development diagnostic; already-touched test; never used for continuation",
            "pathvqa_primary": "free generation with fixed A=Yes/B=No PathMMU-style options and think/answer tags",
            "pathvqa_val_controls": "forced-binary policy/visual-dependence diagnostic, not the primary semantic VQA score",
            "pathvqa_test": "free generation under the frozen 2048-token fixed-A/B PathMMU-style contract",
        },
        "candidate": {
            "train_probe": {key: train[key] for key in ("correct", "count", "accuracy", "format_correct", "generation_cap_hit_count")},
            "pathmmu_val": {key: val[key] for key in ("correct", "count", "accuracy", "format_correct", "generation_cap_hit_count")},
            "pathmmu_test999": {key: test[key] for key in ("correct", "count", "accuracy", "format_correct", "generation_cap_hit_count")},
            "pathvqa_val": {
                "primary_ab_generated": {
                    "correct": pv_ab["correct"], "count": pv_ab["count"],
                    "accuracy": pv_ab["accuracy"],
                    "parseable_rate": pv_ab["parseable_rate"],
                    "strict_pathmmu_choice_format_rate": pv_ab["strict_pathmmu_choice_format_rate"],
                    "generation_cap_hit_count": pv_ab["generation_cap_hit_count"],
                    "fixed_option_mapping": {"A": "Yes", "B": "No"},
                    "option_order_reversal_used": False,
                },
                "secondary_forced_binary_visual_dependence": {
                    "normal": normal["accuracy"], "shuffle": shuffle["accuracy"],
                    "blank": blank["accuracy"],
                    "normal_minus_shuffle": normal["accuracy"] - shuffle["accuracy"],
                    "normal_minus_blank": normal["accuracy"] - blank["accuracy"],
                },
            },
            "mmmu": {key: mmmu[key] for key in ("correct", "count", "accuracy", "generation_cap_hit_count")},
            "pathvqa_test_yesno_ab_generated": {
                "correct": pv_test["contract_aligned_exact_correct"],
                "count": pv_test["count"],
                "accuracy": pv_test["contract_aligned_exact_accuracy"],
                "parseable_rate": pv_test["ab_parseable_rate"],
                "strict_pathmmu_choice_format_rate": pv_test["strict_pathmmu_choice_format_rate"],
                "fixed_option_mapping": pv_test["fixed_option_mapping"],
                "option_order_reversal_used": pv_test["option_order_reversal_used"],
                "nonempty_completion_coverage": (
                    pv_test["count"] - pv_test["empty_completion_count"]
                ) / pv_test["count"],
                "generation_cap_hit_count": pv_test["generation_cap_hit_count"],
                "maximum_generated_tokens": pv_test["maximum_generated_tokens"],
            },
        },
        "paired_pathmmu_val_flips_vs_sft_parent": flips(
            parent_val_predictions, outputs["pathmmu_val"] / "predictions.jsonl"
        ),
        "paired_pathmmu_test_flips_vs_sft_parent": flips(
            reference_root / "pathmmu_test999/step080/predictions.jsonl",
            outputs["pathmmu_test999"] / "predictions.jsonl",
        ),
    }
    write_json(output / "complete_summary.json", summary)
    state.update(status="completed", finished_at=now())
    write_json(state_path, state)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
