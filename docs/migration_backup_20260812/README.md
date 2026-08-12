# Migration backup handoff — 2026-08-12

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

The compact evaluation/protocol archive is stored privately at
`Freddie1946/PathVLM-R1-Evaluation-Archive-20260812` (archive SHA-256:
`5f2262f74d414b4d6cabf92683a39a039190cf8910188e0f77a009b6e53d7d55`). It
contains metrics, summaries, manifests, launch contracts, and selected docs,
but no raw images or credentials.

## Resume checklist

1. Clone the repository branch recorded in `backup_inventory.json`.
2. Recreate the workspace environment and HF authentication on the new machine; never copy secrets from this handoff.
3. Verify the listed paths/checkpoint manifests and the n=8 queue state before launching anything.
4. If n=8 is incomplete, resume only from the latest verified checkpoint; do not use `checkpoint-400` as a final result.
5. Keep PathVQA/OmniMedVQA test sealed for selection; PathMMU test999 is diagnostic only.

Generated at 2026-08-12T11:06:43.490852+00:00.
