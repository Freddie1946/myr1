#!/usr/bin/env python3
"""Wait for the repaired OmniMedVQA wave and upload a final immutable archive."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


WORK = Path("/home/dataset-assist-0/czy/wjy")
REPO = WORK / "myr1"
RUN_ROOT = WORK / "pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815"
REMOTE_REPO = "Freddie1946/PathVLM-R1-Migration-Archive-20260815"
MODELS = (
    "qwen2_5_vl_3b", "lingshu_7b", "medvlm_r1", "medgemma_4b_it",
    "scalereasoner_r1", "llama3_2_vision_11b", "huatuogpt_vision_7b",
    "internvl3_8b", "deepseek_vl2", "llava_med_7b", "llama3_2_vision_90b",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def statuses() -> dict[str, str]:
    values = {}
    for model in MODELS:
        path = RUN_ROOT / model / "state.json"
        try:
            values[model] = json.loads(path.read_text()).get("status", "unknown")
        except Exception:
            values[model] = "missing"
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=60)
    args = parser.parse_args()
    state_root = args.state_root.resolve()
    state_path = state_root / "state.json"
    state = {"schema_version": 1, "started_at": now(), "pid": os.getpid()}
    while True:
        current = statuses()
        state.update({"status": "waiting_for_omnimed", "models": current, "updated_at": now()})
        write(state_path, state)
        if all(value == "completed" for value in current.values()):
            break
        if any(value == "failed" for value in current.values()):
            state.update({"status": "blocked_by_failed_model", "updated_at": now()})
            write(state_path, state)
            raise RuntimeError(f"Omni model failed: {current}")
        time.sleep(args.poll_seconds)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = WORK / "backup_archives" / f"evaluation_results_omnimed_final_{tag}"
    subprocess.run(
        [
            "python3", str(REPO / "scripts/build_evaluation_results_archive.py"),
            "--workspace", str(WORK), "--output-dir", str(output),
        ],
        check=True,
    )
    prefix = f"evaluation_snapshots/omnimed_final_{tag}"
    api = HfApi()
    attempt = 0
    while True:
        attempt += 1
        state.update({"status": "uploading", "attempt": attempt, "prefix": prefix, "updated_at": now()})
        write(state_path, state)
        try:
            api.create_repo(REMOTE_REPO, repo_type="dataset", private=False, exist_ok=True)
            api.update_repo_settings(
                REMOTE_REPO, repo_type="dataset", private=False, gated="manual"
            )
            api.upload_folder(
                folder_path=str(output), path_in_repo=prefix, repo_id=REMOTE_REPO,
                repo_type="dataset", commit_message=f"Add final OmniMedVQA result archive {tag}",
            )
            info = api.dataset_info(REMOTE_REPO, files_metadata=True)
            remote = {entry.rfilename: entry for entry in info.siblings or []}
            for local in output.iterdir():
                if not local.is_file():
                    continue
                found = remote.get(f"{prefix}/{local.name}")
                if found is None or (found.size is not None and int(found.size) != local.stat().st_size):
                    raise RuntimeError(f"remote verification failed for {local.name}")
            if info.private or getattr(info, "gated", False) != "manual":
                raise RuntimeError("result repository visibility/gate mismatch")
            break
        except Exception as error:
            state.update({"status": "retry_wait", "last_error": repr(error), "updated_at": now()})
            write(state_path, state)
            time.sleep(args.retry_seconds)
    verification = json.loads((output / "ARCHIVE_VERIFICATION.json").read_text())
    state.update(
        {"status": "completed", "completed_at": now(), "revision": info.sha, **verification}
    )
    write(state_path, state)


if __name__ == "__main__":
    main()
