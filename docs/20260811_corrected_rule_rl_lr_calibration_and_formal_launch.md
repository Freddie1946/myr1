# Corrected rule-RL learning-rate calibration and formal launch

Date: 2026-08-11 Asia/Shanghai

After correcting the multimodal custom-reward alignment failure, two engineering arms compared
`1e-6` and `3e-6` from the same selected L-SFT3000 step-80 parent. Both used 8 x A100, PDB10,
four generations, a 384-token cap, and stopped after the first ten steps of the same planned
150-step scheduler. Both post-run alignment gates passed with 800 accuracy and 800 format events
and zero errors.

| LR | Mean accuracy reward | Mean KL, steps 2--10 | Max grad norm | PathMMU val385 | PathVQA val512 |
|---:|---:|---:|---:|---:|---:|
| `1e-6` | 47.88% | 0.000330 | 0.1868 | **218/385 = 56.62%** | **329/512 = 64.26%** |
| `3e-6` | 48.88% | 0.000326 | 0.1581 | 215/385 = 55.84% | 328/512 = 64.06% |

The selection used no test data. Because `1e-6` had the higher point estimate on both validation
benchmarks while both arms passed alignment and numerical gates, `1e-6` is frozen for the corrected
formal 150-step run. The formal run retains the same parent, data, seed, batch, rewards, component
freezing and completion cap as the invalidated attempt; only the reward-alignment implementation
and its fail-closed gate differ.

Machine-readable record:
`protocol/corrected_rule_rl_lr_selection_formal_launch_20260811.json`.
