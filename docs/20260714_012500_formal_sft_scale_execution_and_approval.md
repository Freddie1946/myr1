# Formal SFT scale execution and pre-approved launchers

Timestamp: `2026-07-14 01:25:00 Asia/Shanghai`

## Authorized scope

The user pre-authorized the following fixed shell entry points so execution may continue while the
user is offline:

- `bash scripts/launch_formal_sft_run.sh`
- `bash scripts/launch_formal_validation.sh`
- `bash scripts/launch_formal_outcome_grpo_smoke.sh`

The authorization is intentionally narrower than arbitrary Python or shell execution. The SFT
backend accepts only the generated seed-42 scale configs for 500, 1000, 2000, or 3000 records. The
validation backend accepts only the frozen 385-QA validation split and a completed formal SFT parent.
The Outcome-GRPO launcher remains locked until a newly trained validation-qualified SFT checkpoint
exists and the Stage-2 backend has been audited. It does not authorize a long GRPO run. Stage 3 is
deferred.

## Frozen hardware topology

The formal allocation document fixes physical GPUs `0,1,2,3,4,5,6,7` with eight workers. The SFT
backend now rejects any other topology; it will not silently use seven GPUs and thereby change the
effective global batch. If GPU 0 or any other selected GPU still has a compute process, the launcher
refuses before creating a run directory. Codex will not stop or disturb that process.

## SFT backend gates

Before launch, the backend requires:

- the exact generated config path and an allowlisted seed/count;
- exact base model, revision, DeepSpeed policy, full language-model tuning, and visual/projector
  freeze settings;
- passing formal preflight gates, exact adapter count, at least one image per adapter, and every
  referenced image path present;
- eight idle allocated GPUs, free distributed port 29700, and at least 500 GiB free storage;
- offline execution with proxy variables removed and W&B disabled.

Each run receives a unique directory, resolved config, exact command, environment snapshots,
resource samples, and a manifest. Checkpoints are saved every 100 steps with the last two retained
plus the final gathered model. Completion requires a finite loss history, finite nonzero gradients,
independent final-checkpoint reload, a changed language tensor, an exactly unchanged visual tensor,
and no test access.

## Execution order and exact commands

After a fresh all-eight-GPU idle check, run the four seed-42 candidates sequentially, never
concurrently:

```bash
source /home/wjy/pathvlm_r1_v1_formal/FORMAL_PATHS.env
bash scripts/launch_formal_sft_run.sh --config /home/wjy/pathvlm_r1_v1_formal/generated_configs/sft/sft_n0500_seed0042.yaml
bash scripts/launch_formal_sft_run.sh --config /home/wjy/pathvlm_r1_v1_formal/generated_configs/sft/sft_n1000_seed0042.yaml
bash scripts/launch_formal_sft_run.sh --config /home/wjy/pathvlm_r1_v1_formal/generated_configs/sft/sft_n2000_seed0042.yaml
bash scripts/launch_formal_sft_run.sh --config /home/wjy/pathvlm_r1_v1_formal/generated_configs/sft/sft_n3000_seed0042.yaml
```

After each successful training manifest, run deterministic validation on a newly idle single GPU:

```bash
source /home/wjy/pathvlm_r1_v1_formal/FORMAL_PATHS.env
bash scripts/launch_formal_validation.sh \
  --checkpoint <exact-final-checkpoint-from-parent-manifest> \
  --parent-manifest <exact-completed-parent-run-manifest> \
  --gpu <freshly-idle-physical-gpu>
```

Any failed gate stops the sequence. Persistent foreign GPU occupancy also pauses the sequence rather
than changing topology. Test remains sealed.

## Validation-only selection rule

The four scale candidates deliberately share the current predeclared candidate schedule (10 epochs,
learning rate 2e-5, per-device batch 1, seed 42) and differ only in training-set size. Their frozen
validation predictions and metrics provide the scale evidence. These runs do not by themselves prove
that the inherited schedule is optimal. Before the seed-43/44 main runs, the training protocol must
be frozen using validation-only evidence and any additional candidate comparison must be explicitly
documented. No test result may influence that choice.

After the scale sequence, the validation-qualified n=3000 seed-42 checkpoint may be used only for the
audited Outcome-GRPO engineering smoke. Stage-2 scale/long runs and Stage 3 require later gates.
