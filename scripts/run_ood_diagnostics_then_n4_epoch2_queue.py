#!/usr/bin/env python3
"""Wait for the two approved diagnostics, then launch the eight-GPU n4 continuation."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
STATE = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
    "full_language_rule_rl_n4_epoch2_resume_20260812_queue_state.json"
)
PREREQUISITES = (
    Path(
        "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
        "omnimedvqa_parent_n4_n8_20260812/sequence_state.json"
    ),
    Path(
        "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
        "pathvqa_choice_parent_n4_n8_20260812/sequence_state.json"
    ),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(status: str, stage: str, **extra) -> None:
    payload = {"schema_version": 1, "status": status, "stage": stage,
               "updated_at": now(), "prerequisites": [str(x) for x in PREREQUISITES], **extra}
    temporary = STATE.with_suffix(STATE.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATE)


def gpu_processes() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    if STATE.exists():
        raise FileExistsError(STATE)
    write_state("running", "waiting_for_diagnostics")
    try:
        last_report = 0.0
        while True:
            values = []
            for path in PREREQUISITES:
                values.append(json.loads(path.read_text()) if path.exists() else {"status": "missing"})
            if any(value.get("status") == "failed" for value in values):
                raise RuntimeError(f"diagnostic prerequisite failed: {values}")
            if all(value.get("status") == "completed" for value in values):
                break
            if time.monotonic() - last_report >= 300:
                write_state("running", "waiting_for_diagnostics",
                            prerequisite_statuses=[value.get("status") for value in values])
                last_report = time.monotonic()
            time.sleep(10)
        write_state("running", "waiting_for_gpu_release")
        for _ in range(60):
            if not gpu_processes():
                break
            time.sleep(5)
        else:
            raise RuntimeError("diagnostics completed but GPU processes did not exit")
        write_state("running", "n4_epoch2_resume")
        subprocess.run([
            str(PYTHON), str(REPO / "scripts/run_full_language_rule_rl_n4_epoch2_resume.py")
        ], cwd=REPO, check=True)
        write_state("completed", "complete")
    except BaseException:
        write_state("failed", "failed")
        raise


if __name__ == "__main__":
    main()
