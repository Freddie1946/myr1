# Data-ratio rule-RL ablation prepared and sequential smoke completed

Timestamp: 2026-08-09 03:32:36 Asia/Shanghai

## Decision and scientific role

The user replaced the previously broad SFT/RL matrix with a sparse allocation design. The three
new middle arms use one fixed 1,000-QA union and change only which image-complete bins receive SFT
versus rule-based Outcome-GRPO:

| Arm | SFT QA | rule-RL QA | Status before formal launch |
| --- | ---: | ---: | --- |
| SFT-heavy | 750 | 250 | Prepared |
| balanced | 500 | 500 | Prepared |
| RL-heavy | 250 | 750 | Prepared |

The fixed union is the exact union of the already frozen SFT-500 and RL-500 subsets. It is divided
into four deterministic 250-QA bins. Each bin contains 125 QA from each original source, keeps all
QAs for one image together and has near-balanced answers:

| Bin | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| 1 | 64 | 62 | 62 | 62 |
| 2 | 63 | 62 | 62 | 63 |
| 3 | 63 | 62 | 63 | 62 |
| 4 | 62 | 63 | 64 | 61 |

The SFT subsets are nested (bin 1; bins 1--2; bins 1--3), while the corresponding RL subsets are
the exact complement. Every arm has zero SFT/RL record overlap and zero SFT/RL image overlap.

Two 4,000-QA extremes remain in the response design: the existing continued-SFT 4,000-exposure
control is not rerun, while base-to-rule-RL on the complete SFT3000+RL1000 union is new and receives
a 50-step gate before the 6,000-step stress test. The latter is explicitly not compute-matched to
the former and must not be presented as a clean symmetric causal contrast.

Separately, the exact selected Stage2 checkpoint will receive another 1,500 steps of rule-only RL
with a fresh optimizer. It is matched to the extra optimization schedule of Stage3 and is the
control needed to distinguish LLM-judge process reward from merely training longer. It is not one
of the data-allocation points.

## Frozen artifacts

- Protocol: `protocol/data_ratio_rule_rl_ablation_v1_20260809.json`
- Deterministic data builder: `scripts/prepare_data_ratio_ablation.py`
- Stateful runner: `scripts/run_data_ratio_ablation_sequence.py`
- Detached launcher: `scripts/launch_data_ratio_ablation_sequence.sh`
- Regression tests: `scripts/test_data_ratio_ablation_sequence.py`
- Prepared data root:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/data_ratio_rule_rl_ablation_v1`
- Prepared manifest SHA-256:
  `2761b64e10eec83ab67fd3e74973558996a6c1084bb5cc09630f2eeed3d6ae79`

All RL commands use only `accuracy` and `format`; Stage3 process/Judge variables are explicitly
removed from their environments. Test is not loaded by data preparation, training or smoke.

## Sequential smoke

Four CUDA-free regression tests passed. Both smoke and formal preflight then passed with eight idle
80-GB A100 GPUs, the pinned base model, the fixed Stage2 parent, the passing formal-machine report,
and more than the 700-GiB retained disk reserve.

The real eight-GPU smoke ran six tasks in one recoverable queue:

| Task | Result |
| --- | --- |
| 750-arm SFT step | loss 1.7511; grad norm 29.7897 |
| 750→250 rule-RL step | accuracy 0.625; format 1.000; reward std 0.25; grad norm 4.4655 |
| 250-arm SFT step | loss 1.7511; grad norm 29.7925 |
| 250→750 rule-RL step | accuracy 0.375; format 0.875; reward std 0.50; grad norm 4.0653 |
| base→rule-RL step | accuracy 0.000; format 0.125; reward std 0.25; grad norm 4.3708 |
| Stage2→continued-rule-RL step | accuracy 0.500; format 1.000; reward std 0; grad norm 0 |

All six model outputs are structurally loadable. Every RL task retained eight accuracy and eight
format audit events and no process reward. The base result demonstrates why the 0+4000 arm needs a
longer 50-step variance/format gate. The single zero-variance Stage2 batch is retained rather than
misreported as learning; the formal 1,500-step control records the full trajectory.

The first execution found a post-training-only defect: the reward summarizer omitted the
`Counter` import. Both training stages had already completed. The runner was hardened to recognize
a complete model and train-state audit, reuse the prior reward events, redo only the failed audit,
and continue the remaining queue. The recovery succeeded without repeating the completed training.

Final smoke state:

`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/data_ratio_rule_rl_ablation_v1/smoke_sequence/state.json`

SHA-256: `da088c3b4d6af7fb24bb8791dff43ff0298544980aba95462db94be67356fb4a`

The smoke tree occupies about 186 GiB; approximately 3.8 TiB remains free.

## Formal queue behavior

The frozen order is:

1. 750 SFT, then 250 rule-RL;
2. 500 SFT, then 500 rule-RL;
3. 250 SFT, then 750 rule-RL;
4. selected Stage2 parent, then 1,500 fresh-optimizer rule-RL steps;
5. base-to-rule-RL 50-step gate;
6. base-to-rule-RL 6,000-step full stress test only if the gate passes.

SFT uses ten epochs; rule-RL uses the precomputed three-epoch-equivalent step counts. Formal
checkpoints save every 100 steps and retain the latest two resume checkpoints. A completed task is
hash/structure checked and skipped; a partial task resumes from its latest full checkpoint. A
failure stops the queue and is recorded; there is no unbounded automatic retry.

Formal execution is launched detached. After its process is alive, state/log paths exist and the
first task has entered real training, this Codex goal ends; completing all training and evaluating
the resulting checkpoints are later goals.
