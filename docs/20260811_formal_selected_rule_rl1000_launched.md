# Formal selected rule-RL1000 launch

Date: 2026-08-11

## Outcome

Formal rule-only GRPO has started from the frozen L-SFT3000 step-80 parent. The launch contract
was verified and the first real optimizer step completed with finite gradients and nonzero reward
variance. This record establishes successful launch, not formal training completion.

## Parent and frozen policy

- Parent: L-SFT3000 checkpoint step 80, selected before test access by the validation,
  retention and RL-readiness funnel.
- Fixed data: 1,000 PathMMU RL prompts, exactly three equivalent prompt exposures.
- Vision encoder: frozen.
- Multimodal projector: frozen.
- Language model: LoRA only, r16/alpha32/dropout0.05.
- Rewards: audited answer accuracy plus strict think/answer format; no Judge LLM.
- Seed and data seed: 42.

## Throughput and truncation smoke selection

The original 192-token PDB10 smoke was fast but had 13/320 missing closing answer tags, including
four outputs at the token cap. It was rejected before formal launch. Three safer candidates were
then compared:

| PDB/GPU | Max completion | Throughput | Missing close tags | Sampled worst GPU | Decision |
|---:|---:|---:|---:|---:|---|
| 10 | 512 | 2.712 samples/s | 2/320 | 80.7/81.9 GiB | reject: insufficient memory margin |
| 5 | 512 | 1.554 samples/s | 2/160 | 39.2/81.9 GiB | safe but too slow |
| 10 | 384 | 3.226 samples/s | 3/320 | 71.4/81.9 GiB | selected |

The 384-token configuration retains about 10.5 GiB sampled headroom and cuts the observed
format-failure rate from 4.06% at 192 tokens to 0.94%. Two pathological outputs remained extremely
long even at 512 tokens, so further increasing the cap would mainly increase tail latency and OOM
risk. These rare outputs receive zero format reward under the existing rule contract rather than
being silently treated as valid.

## Formal configuration and expected duration

| Field | Value |
|---|---:|
| GPUs | 8 x A100 80 GiB |
| Per-device batch | 10 |
| Completion batch/step | 80 |
| Unique prompts/step | 20 |
| Generations/prompt | 4 |
| Optimizer steps | 150 |
| Max completion | 384 tokens |
| Checkpoint schedule | step 100 plus final output |
| Estimated wall time | 65-80 minutes |

The estimate is based on the selected four-step smoke. Long-tail generations make individual
steps variable; the first formal step took 44.88 seconds and is not used alone as the ETA.

## First-step verification

The launch audit reported exactly 42,860,544 trainable language-LoRA parameters, zero trainable
Vision parameters and zero trainable Projector parameters. Step 1 then reported:

| Metric | Value |
|---|---:|
| Gradient norm | 0.123364 |
| Accuracy reward | 0.2375 |
| Format reward | 0.9375 |
| Total reward | 1.1750 |
| Reward standard deviation | 0.3080 |
| Mean completion length | 108.65 |
| KL | 0.0 |

All values are finite, and reward variance is nonzero. Formal training continues in the run
directory below. Per the goal boundary, completion monitoring and post-RL validation/OOD/test
evaluation belong to a later task.

## Recovery paths

- Run: `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_20260811/formal_z2_gc0_mbs10_len384`
- Live log: `train.log` under that run directory
- Frozen launch contract: `launch_contract.json` under that run directory
- Protocol manifest: `protocol/formal_selected_rule_rl1000_launch_20260811.json`
