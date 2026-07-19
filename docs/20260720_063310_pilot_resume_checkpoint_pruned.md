# Pilot resume checkpoint pruned after complete verification

Timestamp: `2026-07-20T06:33:10+08:00`

## Authorization and exact scope

The user explicitly approved deletion after careful confirmation. Only this directory was removed:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage2_outcome_grpo/priority_n1000_seed0042/stage2_priority_n1000_seed0042_20260720_035156/pilot/output/checkpoint-25`

It was the engineering pilot's full ZeRO resume state. It had already been successfully consumed by
a fresh process that resumed from global step 25 and completed global step 50. It is not the formal
SFT parent and cannot be used as the formal n=1000 parent.

## Pre-deletion gates and inventory

All checks passed before deletion:

- target resolved to the exact expected absolute path inside the exact pilot run;
- target was a directory, not a symlink;
- `trainer_state.json`, segment-A audit and segment-B resume audit proved step 25 -> step 50;
- the pilot manifest remained completed with every gate true;
- the pipeline had stopped only at the later storage projection gate;
- no formal n=1000 directory existed;
- the independent final model, raw reward events, logs, train-state audits, tensor comparison and
  validation predictions/metrics all existed outside the target.

The target contained 44 files totaling 109,344,802,837 bytes. Every file was read and hashed before
deletion. The retained inventory is:

`pilot/checkpoint25_pre_pruning_sha256.txt`

Its SHA-256 is:

`9483d4ce1ece4731e75a2c2e1a7375b27e5107f2914bc71900473c5ed1d6bebb`

The associated machine-local JSON record is `pilot/checkpoint25_pruning_record.json`.

## Post-deletion verification

- Exact checkpoint target absent: pass.
- SHA-256 inventory retained and unchanged: pass.
- Pilot final model index plus four safetensor shards retained: pass.
- Pilot manifest and all gates retained: pass.
- Eight rank logs for each of the two reward segments retained: pass.
- Exactly 385 validation predictions retained with their original SHA-256: pass.
- Validation metrics, tensor audit and logs retained: pass.
- Formal n=1000 directory still absent: pass.
- Test accessed: false.

Free space increased from 839,667,310,592 to 949,008,969,728 bytes. The measured increase was
109,341,659,136 bytes. The frozen formal-start projection requires 897,060,103,530 bytes, so the
current margin is 51,948,866,198 bytes. The original pipeline manifest retains its truthful failed-
at-storage-gate status and now contains an appended storage-pruning event; history was not rewritten.

## Next boundary

Formal training has not started. A formal-only continuation must reuse the passing pilot evidence,
restart from the original selected SFT n=3000 epoch-3 parent, and rerun exact Git/code/data/parent,
GPU, port and storage gates. It must not rerun the pilot, use its final model as parent, access test,
or start another experiment. Launch requires the user's next confirmation.
