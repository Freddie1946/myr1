# PathVLM-R1 revision repository instructions

These instructions apply to every Codex/agent working in this repository, especially on the formal training machine.

## Mandatory reading order

Before changing files or launching work, read completely:

1. `CODEX_START_HERE.md`
2. `docs/LATEST.md`
3. The timestamped documents referenced by `docs/LATEST.md`
4. `manuscript/README.md` and the materials in its mandatory reading order
5. `protocol/training_plan.md`
6. `protocol/code_hash_manifest_20260712_144811.json`

Do not rely on chat memory from another machine. The Git repository is the authoritative handoff record.

## Scientific invariants

- Formal base model: `Qwen/Qwen2.5-VL-7B-Instruct` revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Data version: `pathmmu_image_disjoint_v1`.
- `picked.json` is deprecated and forbidden.
- SFT/RL/validation/test must remain image-disjoint.
- Test is evaluation-only. Never use it for prompt, seed, checkpoint, threshold, reward, or hyperparameter selection.
- Validation is the model/configuration selection split.
- Full-parameter language-model fine-tuning is required; the vision tower and multimodal projector remain frozen.
- Main formal runs target training seeds 42, 43, and 44. Do not select a favorable seed.
- Save exact sample counts, data/model/code hashes, seeds, raw predictions, and online reward events.
- Stage 3 Process Reward is reconstructed/completed work, not proven historical recovery. Do not invent its scientific definition.

## Current status that must not be misinterpreted

- The 3B SFT one-step smoke run is valid and had nonzero gradient.
- Outcome GRPO Attempt02 completed plumbing but had zero reward variance and no effective update; its model shards were pruned.
- Outcome GRPO Attempt03 had nonzero gradient and changed language parameters while visual parameters stayed identical, but its old accuracy parser falsely credited an answer that enumerated all choices. Attempt03 is scientifically invalid; its model shards were pruned.
- Parser v2 fixes explicit-choice parsing and option-enumeration false positives; regression tests pass.
- Outcome GRPO Attempt04 is prepared but has not run because the debug GPUs became occupied.
- Do not use any 3B debug checkpoint as a parent of formal 7B experiments.

## Formal-machine bootstrap

The expected human entry point is:

```bash
cp formal_machine/formal_machine.env.example formal_machine.env
# Edit formal_machine.env.
bash setup_formal_machine.sh formal_machine.env
```

The bootstrap must finish with a passing `preflight_report.json`. If it does not, diagnose and document the exact failure; do not start training.

## Execution gates

Before formal SFT:

- Verify base revision and model identity.
- Verify adapter counts and every image path.
- Verify all pairwise split image overlaps are zero.
- Run one optimizer step on formal hardware.
- Confirm language parameters are trainable and vision/projector parameters are frozen.
- Save, reload, and resume a checkpoint.

Before formal Outcome GRPO:

- Parent must be a newly trained formal SFT checkpoint.
- Run parser regression tests.
- Audit online completion/reward JSONL.
- Require reward parser consistency online versus offline.
- Diagnose persistent zero reward variance.
- Verify a language tensor changes and a frozen visual tensor does not.

## Documentation discipline

- Every material plan, attempt, failure, correction, and completion gets a new `YYYYMMDD_HHMMSS_*.md` file under `docs/`.
- Update `docs/LATEST.md` after every material state change.
- Update the associated run manifest; never overwrite history to make a failed run look successful.
- Mark engineering smoke results `formal_result: false`.

## Git and storage

- Commit source, configs, manifests, small JSON data, and documentation.
- Never commit model weights, images, venvs, caches, generated outputs, credentials, or API keys.
- Preserve unrelated user changes and dirty working trees.
- Do not force-push or rewrite shared history without explicit permission.
