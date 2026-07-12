# Codex start here: formal-machine handoff

Timestamp of this handoff: 2026-07-12 19:18:13 Asia/Shanghai

This repository replaces cross-machine chat memory. A new Codex session on the formal machine must treat the repository, timestamped documents, and run manifests as its source of truth.

## First response expected from the new Codex

After reading `AGENTS.md` and this file, report:

1. Current Git commit and worktree status.
2. Formal machine GPU/driver, CPU RAM, swap, disk, CUDA toolkit, and existing conda state.
3. Whether the machine is online or must use an offline wheelhouse/model snapshot.
4. Resolved paths for frozen splits, images, LLaMA-Factory, vendored Open-R1, and the exact 7B model.
5. Any mismatch with the fixed protocol before making changes.

Then configure the machine; do not launch full experiments immediately.

## Human quick start

```bash
git clone <YOUR_REPOSITORY_URL> pathvlm-r1-revision
cd pathvlm-r1-revision
cp formal_machine/formal_machine.env.example formal_machine.env
nano formal_machine.env  # or use a graphical editor
bash setup_formal_machine.sh formal_machine.env
```

At minimum, edit:

- `INSTALL_ROOT`
- `SPLIT_ROOT`
- `IMAGE_ROOT`
- `CONDA_EXE`
- `ONLINE`
- `BASE_MODEL_SOURCE` when offline

Before online PathMMU setup, the human must request access at
`https://huggingface.co/datasets/jamessyx/PathMMU`, accept its terms, and authenticate on the
formal machine with `hf auth login` or `HF_TOKEN`. This is a gated dataset and the bootstrap
cannot bypass its manual approval step.

If the script succeeds, it creates:

- `$INSTALL_ROOT/envs/sft`
- `$INSTALL_ROOT/envs/grpo`
- `$INSTALL_ROOT/models/...`
- `$INSTALL_ROOT/data/pathmmu_image_disjoint_v1`
- `$INSTALL_ROOT/reports/preflight_report.json`
- `$INSTALL_ROOT/FORMAL_PATHS.env`

## Online and offline modes

Online mode downloads wheels, clones the exact LLaMA-Factory revision, downloads the exact Qwen
7B revision, downloads official PathMMU `images.zip`, and extracts only the 3,809 unique images
referenced by the frozen split. It writes `reports/pathmmu_download_report.json` and aborts if any
required image is unavailable or corrupt.

Offline mode requires these to be copied through the remote-desktop shared drive:

- Git repository checkout.
- `wheelhouse/`.
- Exact 7B snapshot.
- Frozen split JSON directory.
- Required PathMMU images.
- LLaMA-Factory source tree at the audited revision.

Set `ONLINE=0`, `WHEELHOUSE`, `BASE_MODEL_SOURCE`, and `LLAMAFACTORY_SRC` accordingly.

## Current experimental truth

### Data

- SFT: 3000 QA / 2121 images.
- RL: 1000 QA / 708 images.
- Validation: 385 QA / 272 images.
- Test: 1000 QA / 708 images.
- Pairwise image overlap: zero.
- Nested SFT: 500/1000/2000/3000.
- Nested RL: 250/500/1000.

### SFT debugging

The 3B proxy completed a genuine full language-model optimizer step:

- 3,085,938,688 trainable language parameters.
- Vision tower and projector frozen.
- Loss 1.6429958.
- Gradient norm 25.1862.
- Exit status 0 and loadable checkpoint.

This validates SFT plumbing, not final performance.

### Outcome GRPO debugging

Attempt02: online plumbing worked but both completions had zero reward; no effective update.

Attempt03: generated four completions and had nonzero gradient. Tensor comparison proved the language layer changed and the visual layer was identical. Case audit then found an accuracy-parser false positive, so the policy update is invalid.

Parser v2 fixes the defect and passes regression tests. Attempt04 is the next debug action when two GPUs are free. On the formal machine, reproduce a one-step gate with parser v2 before any long GRPO run.

### Process Reward

Not finalized. Read `docs/20260712_144517_stage3_process_reward_decisions_pending.md`. Ask the user to agree on process units, supervision source, label space, aggregation, online implementation, and claims before training Stage 3.

## Formal experiment order

1. Bootstrap and pass preflight.
2. One-step 7B full-language SFT smoke on formal hardware; save/reload/resume.
3. SFT scale study with seed 42: 500/1000/2000/3000.
4. Select the training protocol on validation only.
5. Main SFT at 3000 with seeds 42/43/44.
6. One-step Outcome GRPO parser/gradient/parameter-delta gate.
7. RL scale study with seed 42: 250/500/1000.
8. Main Outcome GRPO with seeds 42/43/44.
9. Implement and smoke-test Stage 3 only after its definition is agreed.
10. Generalization evaluations.
11. Lock prompts/checkpoints/scoring, then evaluate test.

## Important files

- `AGENTS.md`: mandatory behavioral/scientific rules.
- `docs/LATEST.md`: chronological recovery pointer.
- `formal_machine/formal_machine.env.example`: user configuration.
- `setup_formal_machine.sh`: single setup entry point.
- `formal_machine/prepare_formal_data.py`: image-path rewrite and adapter generation.
- `formal_machine/download_pathmmu.py`: gated download and frozen-image extraction.
- `formal_machine/preflight_formal_machine.py`: hardware/environment/model/data gates.
- `configs/`: debug and formal reference configurations.
- `scripts/pathmmu_rewards.py`: shared online/offline reward implementation.
- `scripts/test_pathmmu_rewards.py`: mandatory reward regression tests.
- `protocol/`: model/data/code manifests and training plan.

## What not to do

- Do not use `picked.json`.
- Do not tune on test.
- Do not reuse obsolete historical 7B checkpoints.
- Do not silently change split, prompt, reward parser, seed policy, or model revision.
- Do not interpret an exit-0 GRPO run as learning without checking reward variance, gradient norm, and parameter deltas.
- Do not claim Attempt03 is valid.
