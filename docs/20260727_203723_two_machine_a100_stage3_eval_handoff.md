# Two-machine A100 Stage 3/evaluation handoff

Recorded: `2026-07-27T20:37:23+08:00`

This is a preparation and responsibility handoff with `formal_result: false`. No data/model was
downloaded, no environment was modified, no GPU was used and no training/inference/test was run.

## Decision

The existing formal machine will run only the next approved 4000-SFT training control. The new
machine at `/home/dataset-assist-0/czy/wjy`, with eight A100 GPUs, will own Stage 3 and all
validation/test/baseline/OOD work. It also inherits every remaining repository experiment; machine
location is not a criterion for cancelling scientific obligations.

Validation counts as evaluation under this split. The existing machine may train and save
predeclared 4000-SFT candidates but may not select among them. A100 receives all candidates with an
exact transfer manifest and performs validation selection.

The exact 4000-SFT control is not frozen: fresh-base SFT on all 4,000 examples and continued SFT from
the selected n=3000 checkpoint are different scientific comparisons. A separate protocol decision
is required before the existing machine launches either one.

## Implemented handoff material

- `A100_STAGE3_EVAL_CODEX_START.md`: complete human/Codex entry point, personal Codex setup,
  two-machine roles, workspace layout, transfer protocol and inherited experiment order.
- `formal_machine/a100_stage3_eval_workspace.env.example`: non-secret A100 workspace paths and
  isolated caches.
- `scripts/launch_personal_codex_a100.sh`: launches only the personal Codex binary/state, from the
  correct repository, with workspace-write plus on-request approvals.
- `protocol/a100_codex_bootstrap_prompt_20260727.txt`: first prompt that reads the entire repository
  and then automatically performs preparation, but forbids training/inference/test/API spending.
- `protocol/a100_stage3_eval_handoff_manifest_20260727_203723.json`: machine responsibilities,
  paths, inherited work and execution gates.
- `protocol/code_hash_manifest_20260727_203723.json`: new active bootstrap/code snapshot. It replaces
  the old hardcoded bootstrap pointer without changing the historical 2026-07-19 manifest. Mutable
  living handoff/reviewer documents are intentionally excluded from executable-code verification,
  preventing legitimate future status updates from breaking a fresh-machine bootstrap.

The existing `FORMAL_MACHINE_CODEX_GUIDE.md`, `CODEX_START_HERE.md`, `AGENTS.md` and
`protocol/training_plan.md` are updated to route a memoryless A100 Codex into this handoff.

## Personal Codex isolation

The A100 setup uses:

- `CODEX_HOME=/home/dataset-assist-0/czy/wjy/.codex-wjy`;
- `CODEX_INSTALL_DIR=/home/dataset-assist-0/czy/wjy/codex-bin`;
- file credential storage under the private `CODEX_HOME`;
- `sandbox_mode=workspace-write`, `approval_policy=on-request`;
- explicit `--add-dir /home/dataset-assist-0/czy/wjy`;
- no global Codex configuration, state, auth or skills.

This follows the current official Codex behavior: `CODEX_HOME` changes the root of config, auth,
logs, sessions, skills and package metadata; the standalone installer honors `CODEX_INSTALL_DIR`;
headless CLI login supports `codex login --device-auth`.

## Synchronization gate

The local handoff branch was already ahead of its GitHub remote before this work. The A100 machine
must not clone until the branch containing this handoff and all earlier commits is pushed. A new
execution branch `codex/a100-stage3-eval` avoids simultaneous two-machine writes to one branch.
No force push or history rewrite is allowed.

Weights, images, environments, caches, raw generations and checkpoints remain outside Git. Small
manifests/docs/code return through Git. The 4000-SFT checkpoint transfer requires exact file hashes
and source-run provenance.

## Remaining gates

1. User confirms and freezes the scientific definition of the 4000-SFT control.
2. Push the complete handoff branch so A100 can clone it.
3. Install/login the personal Codex on A100.
4. Audit A100 driver, GPU occupancy, disk/quota, Conda and network.
5. Authenticate Hugging Face and recheck every gated repository.
6. Complete preparation and passing formal/external preflights.
7. Separately freeze and authorize every training/evaluation stage.
