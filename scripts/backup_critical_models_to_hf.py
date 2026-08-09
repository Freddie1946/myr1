#!/usr/bin/env python3
"""Back up the three explicitly selected critical models with fixed visibility.

The allowlist is intentionally narrow:

* Grok Stage3: private;
* Stage2 continued rule-RL: public with manual gated access;
* base + 4,000 rule-RL: public with manual gated access.

Small data-ratio checkpoints are deliberately absent.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from backup_data_ratio_models_to_hf import (
    atomic_json,
    build_manifest,
    now_iso,
    selected_files,
    verify_remote,
)
from backup_future_data_ratio_models_to_hf import formal_task_complete, load_json


CRITICAL_SPECS = (
    {
        "task_id": "grok43_stage3",
        "repo_id": "Freddie1946/PathVLM-R1-Process-GRPO-Grok43-n1000-seed42-epoch3",
        "private": True,
        "gated": False,
        "expected_step": 1500,
        "source_kind": "grok",
        "summary": "Process-GRPO Stage3 model trained with Grok 4.3 as the judge.",
    },
    {
        "task_id": "stage2_continue_rule_rl1000",
        "repo_id": "Freddie1946/PathVLM-R1-Stage2-Continued-RuleRL1000-seed42",
        "private": False,
        "gated": "manual",
        "expected_step": 1500,
        "source_kind": "formal",
        "summary": "Stage2 outcome-RL checkpoint continued with rule-only RL.",
    },
    {
        "task_id": "base_rule_rl4000",
        "repo_id": "Freddie1946/PathVLM-R1-Base-RuleRL4000-seed42",
        "private": False,
        "gated": "manual",
        "expected_step": 6000,
        "source_kind": "formal",
        "summary": "Qwen2.5-VL base trained with the full 4,000-example rule-only RL control.",
    },
)


def resolved_tasks(formal_root: Path, grok_output: Path) -> list[dict[str, Any]]:
    tasks = [dict(spec) for spec in CRITICAL_SPECS]
    for task in tasks:
        if task["source_kind"] == "grok":
            task["source"] = grok_output
        else:
            task["source"] = formal_root / "tasks" / task["task_id"] / "output"
    return tasks


def grok_complete(output: Path, expected_step: int) -> bool:
    state_path = output / f"checkpoint-{expected_step}" / "trainer_state.json"
    result_path = output / "train_results.json"
    if not state_path.is_file() or not result_path.is_file():
        return False
    state = load_json(state_path)
    return int(state.get("global_step", -1)) == expected_step


def model_card(task: dict[str, Any]) -> str:
    gated_line = "gated: true\n" if task["gated"] else ""
    access = (
        "The repository is public for discovery, but weight downloads require manual "
        "approval by the repository owner."
        if task["gated"]
        else "This repository is private."
    )
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-7B-Instruct
library_name: transformers
pipeline_tag: image-text-to-text
{gated_line}tags:
- multimodal
- pathology
- grpo
- research
---

# {task['repo_id'].split('/', 1)[1]}

{task['summary']}

This is a research checkpoint derived from Qwen2.5-VL-7B-Instruct. It is archived to support
the controlled experiments and reviewer-response analysis for PathVLM-R1. It is not intended for
clinical deployment or diagnosis.

## Access

{access}

## Training record

- Expected terminal optimizer step: `{task['expected_step']}`
- Seed: `42`
- Backup type: model-only; optimizer and RNG state are not included
- Exact file hashes are provided in `snapshot_manifest.json`

The complete experimental protocol, evaluation scope and limitations are maintained in the
associated PathVLM-R1 revision repository.
"""


def prepare_repo(api: HfApi, task: dict[str, Any]) -> dict[str, Any]:
    api.create_repo(
        task["repo_id"], repo_type="model", private=bool(task["private"]), exist_ok=True
    )
    api.update_repo_settings(
        repo_id=task["repo_id"],
        repo_type="model",
        private=bool(task["private"]),
        gated=task["gated"],
    )
    info = api.model_info(task["repo_id"])
    actual_gated = getattr(info, "gated", False)
    if bool(info.private) != bool(task["private"]):
        raise RuntimeError(f"visibility mismatch for {task['repo_id']}")
    if task["gated"] == "manual" and actual_gated != "manual":
        raise RuntimeError(
            f"manual gate not active for {task['repo_id']}: {actual_gated!r}"
        )
    if not task["gated"] and actual_gated:
        raise RuntimeError(f"unexpected gate for {task['repo_id']}: {actual_gated!r}")
    return {"private": bool(info.private), "gated": actual_gated}


def upload_task(
    api: HfApi, hf_bin: str, task: dict[str, Any], state_root: Path
) -> dict[str, Any]:
    files = selected_files(task["source"])
    manifest_path = state_root / "manifests" / f"{task['task_id']}.json"
    manifest = build_manifest(task, files)
    manifest.update(
        {
            "private": bool(task["private"]),
            "gated": task["gated"],
            "access_policy": "manual_approval" if task["gated"] else "owner_only",
        }
    )
    atomic_json(manifest_path, manifest)
    settings = prepare_repo(api, task)

    card_path = state_root / "cards" / f"{task['task_id']}.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(model_card(task), encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=task["repo_id"],
        repo_type="model",
        commit_message="Add model card and access policy",
    )

    command = [
        hf_bin,
        "upload-large-folder",
        task["repo_id"],
        str(task["source"]),
        "--repo-type",
        "model",
        "--num-workers",
        "2",
        "--no-bars",
    ]
    if task["private"]:
        command.append("--private")
    for path in files:
        command.extend(["--include", path.name])
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"HF upload exited {completed.returncode} for {task['repo_id']}")
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="snapshot_manifest.json",
        repo_id=task["repo_id"],
        repo_type="model",
        commit_message=f"Add verified model-only manifest for {task['task_id']}",
    )
    verification = verify_remote(api, task["repo_id"], manifest)
    final = api.model_info(task["repo_id"])
    if bool(final.private) != bool(task["private"]):
        raise RuntimeError(f"visibility changed during upload for {task['repo_id']}")
    if task["gated"] == "manual" and getattr(final, "gated", False) != "manual":
        raise RuntimeError(f"manual gate changed during upload for {task['repo_id']}")
    verification.update(settings)
    return verification


def wait_until_complete(
    task: dict[str, Any], formal_state: Path, poll_seconds: int
) -> None:
    while True:
        if task["source_kind"] == "grok":
            done = grok_complete(task["source"], task["expected_step"])
        else:
            done = formal_task_complete(
                formal_state, task["task_id"], task["expected_step"]
            )
        if done:
            selected_files(task["source"])
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--grok-output", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--hf-bin", default="/usr/local/bin/hf")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    formal_root = args.formal_root.resolve()
    formal_state = formal_root / "state.json"
    state_root = args.state_root.resolve()
    state_path = state_root / "state.json"
    tasks = resolved_tasks(formal_root, args.grok_output.resolve())
    state = (
        load_json(state_path)
        if state_path.is_file()
        else {"schema_version": 1, "created_at": now_iso(), "tasks": {}}
    )
    state.update(
        {
            "status": "running",
            "pid": os.getpid(),
            "updated_at": now_iso(),
            "policy": "three_explicit_critical_models_only",
        }
    )
    atomic_json(state_path, state)
    api = HfApi()
    for task in tasks:
        if state["tasks"].get(task["task_id"], {}).get("status") == "completed":
            continue
        state["current_task"] = task["task_id"]
        state["tasks"][task["task_id"]] = {
            "status": "waiting_for_completion",
            "repo_id": task["repo_id"],
            "private": task["private"],
            "gated": task["gated"],
            "expected_step": task["expected_step"],
        }
        atomic_json(state_path, state)
        try:
            wait_until_complete(task, formal_state, args.poll_seconds)
            state["tasks"][task["task_id"]].update(
                {"status": "uploading", "started_at": now_iso()}
            )
            atomic_json(state_path, state)
            verification = upload_task(api, args.hf_bin, task, state_root)
        except Exception as exc:
            state["status"] = "failed"
            state["tasks"][task["task_id"]].update(
                {
                    "status": "failed",
                    "failed_at": now_iso(),
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            state["updated_at"] = now_iso()
            atomic_json(state_path, state)
            raise
        state["tasks"][task["task_id"]].update(
            {"status": "completed", "completed_at": now_iso(), "verification": verification}
        )
        state["updated_at"] = now_iso()
        atomic_json(state_path, state)
    state.update({"status": "completed", "completed_at": now_iso(), "current_task": None})
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()
