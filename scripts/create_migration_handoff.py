#!/usr/bin/env python3
"""Create a compact, secret-free migration handoff and backup inventory.

The handoff records paths, sizes, hashes for small metadata/result files, and
the current queue/process state.  It deliberately does not copy credentials,
API keys, raw images, optimizer shards, or conversation-internal data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path, limit: int = 64 * 1024 * 1024) -> str | None:
    if not path.is_file() or path.stat().st_size > limit:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path: Path, root: Path) -> dict:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "relative_to_workspace": os.path.relpath(path, root),
        "exists": True,
        "bytes": size,
        "sha256_if_le_64MiB": sha256(path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=Path("/home/dataset-assist-0/czy/wjy"))
    ap.add_argument("--repo", type=Path, default=Path("/home/dataset-assist-0/czy/wjy/myr1"))
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    root = args.workspace.resolve()
    repo = args.repo.resolve()
    out = (args.out_dir or (repo / "docs" / "migration_backup_20260812")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    install = root / "pathvlm_r1_v1_a100"
    eval_root = root / "pathvlm_revision_eval_a100"
    n8_root = install / "runs/full_language_rule_rl_clean_n4_n8_step1000_20260812"
    files = [
        repo / "protocol/predefined_evidence_candidate_panel_v1_80case_20260812.json",
        repo / "docs/20260812_n4_step500_to1000_explainability_plan.md",
        n8_root / "sequence_state.json",
        n8_root / "n4_fresh_step1000/launch_contract.json",
        n8_root / "n4_fresh_step1000/completion_verification.json",
        n8_root / "n4_fresh_step1000/reward_alignment_verification.json",
        n8_root / "n4_fresh_step1000/output/train_results.json",
        n8_root / "n4_fresh_step1000_pathmmu_eval/complete_summary.json",
        eval_root / "runs/option_conditioned_rise_96case_gpt4o_stage3_epoch2_step1000_20260809/metrics.json",
        eval_root / "runs/visual_understanding_counterfactual_96case_gpt4o_stage3_epoch2_step1000_20260808/metrics.json",
        eval_root / "runs/visual_adaptation_architecture_validation_20260811/pathmmu/l/metrics.json",
        eval_root / "runs/visual_adaptation_architecture_validation_20260811/pathvqa/l/original/metrics.json",
        eval_root / "runs/visual_adaptation_architecture_validation_20260811/mmmu_nonmedical_dev116/l_max4096_rescored_v2/metrics.json",
    ]
    inventory = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "machine migration handoff; no secrets",
        "workspace": str(root),
        "git_repo": str(repo),
        "git_branch": subprocess.run(["git", "-C", str(repo), "branch", "--show-current"], text=True, capture_output=True).stdout.strip(),
        "git_head": subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip(),
        "n8_training": {
            "queue_state": str(n8_root / "sequence_state.json"),
            "train_log": str(n8_root / "n8_fresh_step1000/train.log"),
            "expected": "num_generations=8, max_steps=1000, LR=1e-6, full language RL, vision/projector frozen",
            "resume_policy": "retain output/checkpoint-400 and later snapshots; do not treat partial n8 as final",
        },
        "critical_model_backup_allowlist": [
            {"label": "SFT parent", "path": str(install / "runs/formal_selected_rule_rl1000_20260811/parents/sft_step080_merged"), "policy": "private HF model repo"},
            {"label": "n4 full-RL step1000", "path": str(n8_root / "n4_fresh_step1000/model_snapshots/checkpoint-1000"), "policy": "private HF model repo"},
            {"label": "n8 full-RL step1000", "path": str(n8_root / "n8_fresh_step1000/model_snapshots/checkpoint-1000"), "policy": "private HF model repo after completion"},
            {"label": "selected Stage3 checkpoints", "path": str(install / "runs/stage3_process_grpo"), "policy": "existing HF backups; do not duplicate until selected"},
        ],
        "excluded_from_long_term_model_backup": ["smoke/throughput checkpoints", "failed attempts", "small data-ratio ablations", "optimizer/ZeRO shards unless resumability is explicitly required", "API keys and credential files"],
        "records": [record(p, root) for p in files],
        "conversation_history": "Codex cannot export the platform's raw transcript; this handoff is a synthesized record of decisions, commands, paths, and resume policy.",
    }
    (out / "backup_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = f"""# Migration backup handoff — 2026-08-12

This is a secret-free, synthesized handoff for moving the PathVLM-R1 work to another machine. The raw Codex transcript is not exportable from the filesystem; this document records the active plan and durable pointers.

## Current execution

- n=8 full-language rule-RL is still running from the merged L-r16 SFT step-80 parent (`num_generations=8`, `max_steps=1000`, LR `1e-6`, 8×A100, vision/projector frozen).
- n=4 step-1000 is complete and verified; its PathMMU validation/test999 are in the inventory JSON.
- After n=8 completes: run n=8 PathMMU validation and diagnostic test999, then the split audit, then select the Stage3 starting checkpoint.
- Stage3 uses Aigcbest GPT-4o with balance checks and resumable checkpoints every 100 steps; no fixed dollar cap was requested.
- Explainability uses an 80-case externally annotated candidate panel, clean-correct primary cases, controlled evidence/neighbor/random deletion, representation tracing, activation patching, and 7×7 option-conditioned RISE as auxiliary evidence.

## Backup policy

Long-term model backups are limited to the SFT parent and final n=4/n=8 checkpoints (private HF repositories), plus already selected Stage3 models. Smoke, failed, and small ablation checkpoints are excluded. Evaluation summaries, manifests, and this handoff are small enough for GitHub/HF archival. No credential, token, or API-key file is copied.

See [`backup_inventory.json`](./backup_inventory.json) for exact absolute paths, sizes, and hashes where practical.

## Resume checklist

1. Clone the repository branch recorded in `backup_inventory.json`.
2. Recreate the workspace environment and HF authentication on the new machine; never copy secrets from this handoff.
3. Verify the listed paths/checkpoint manifests and the n=8 queue state before launching anything.
4. If n=8 is incomplete, resume only from the latest verified checkpoint; do not use `checkpoint-400` as a final result.
5. Keep PathVQA/OmniMedVQA test sealed for selection; PathMMU test999 is diagnostic only.

Generated at {inventory['created_at']}.
"""
    (out / "README.md").write_text(md, encoding="utf-8")
    print(out / "backup_inventory.json")
    print(out / "README.md")


if __name__ == "__main__":
    main()
