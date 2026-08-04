# Kimi segment02 checkpoint-load failure and correction

Recorded at: `2026-08-04 23:13:44 CST`

## Failure

After all eight GPUs became idle, the approved Kimi Stage3 recovery supervisor recorded segment02
at launch index 2 with `checkpoint-200` as its resume source. All eight workers loaded the four
model shards, then stopped before any optimizer step or Judge request while DeepSpeed loaded the
ZeRO optimizer state.

PyTorch 2.6 defaults `torch.load` to `weights_only=True`. The locally generated DeepSpeed state
contains `deepspeed.runtime.zero.config.ZeroStageEnum`, which the restricted loader rejects. The
failure was `_pickle.UnpicklingError: Weights only load failed`. This is the same known local-
checkpoint compatibility behavior already handled in the GPT-4o launcher and earlier Kimi pilot,
but the full Kimi launcher had omitted the scoped environment setting.

The request ledger remained exactly at 2,223 physical requests and `$4.015588330000004`; it had
zero reservations. No Judge call, reward, optimizer update or checkpoint mutation occurred.

## Secondary accounting gate

The supervisor correctly classified the load error as terminal, then its settlement step failed
closed because the approved request cap was frozen in code as 12,976 while the existing on-disk
ledger still recorded its original 12,361 contract. Consequently, segment02 has a start event but
no automatically generated failure event. This history will be completed explicitly after the
ledger migration, without rewriting the existing audit.

## Correction

- Export `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` only after a resume checkpoint is structurally
  validated as an immediate child of the run output directory.
- Explicitly clear the setting for fresh runs.
- Add a locked `raise-request-cap` recovery command. It permits only an increase, requires the
  exact old USD/reserve/request contract, refuses unresolved reservations, preserves every ledger
  history row and appends a timestamped contract amendment before atomically replacing the JSON.
- Retain the approved old/new caps 12,361 and 12,976 and the unchanged USD ceiling.

The relevant recovery/OpenRouter/AIGCBest tests pass 58/58, shell syntax and Python compilation
pass, and all GPUs are idle. This record does not claim that segment03 has started.

## Code hashes

- `scripts/stage3_checkpoint_recovery.py`:
  `d7f7501eb8631ac52ed45d152ce95ffef8105154892eaab8f7ad45ef933e344d`
- `scripts/launch_stage3_kimi26_formal.sh`:
  `c4fb15228b0e4ac9e8561fde8bd0c58a15504dabd2aed9ba89576407066e7dda`
- `scripts/test_stage3_checkpoint_recovery.py`:
  `267f20d574483e6ef53fb39bc4f277700cebd241f7e586126481db936a8731b1`
- `scripts/test_stage3_kimi26_recovery_contract.py`:
  `f13d86f47976916fa89cc83c50436fefe43a5fe17166a18312ecc1e3c8f9adde`
