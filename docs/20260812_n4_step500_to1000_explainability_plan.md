# n=4 step500 → step1000 visual-evidence comparison

This is a post-hoc longitudinal diagnostic comparing the continued full-language n=4 rule-RL
checkpoint at step 500 with its continued step-1000 checkpoint. Both checkpoints use the same SFT
parent, vision/projector freezing, reward, seed, and rollout configuration; only additional RL
exposure differs.

The frozen 96-case PathMMU validation panel and appearance-matched counterfactual manifest are reused
with their recorded SHA-256 values. This makes the comparison paired and reproducible, but it is not
a new independent confirmation panel.

## Reproduced methods

1. **Paired-image counterfactual:** score the original image and model-blind nearest same-answer-
   position mismatch. Report paired accuracy and ground-truth option-margin loss.
2. **Option-conditioned RISE:** generate one saliency map per A/B/C/D option from 256 random soft
   masks. Validate the target-option high-saliency region with hard mean-fill deletion and retention,
   against the low-saliency region and five circular-shifted area/shape-matched controls.

Raw attention and Grad×Attention remain an exploratory comparator only: their earlier spatial
confirmation gate was negative. Layer-20/24 activation patching can be added after this comparison
if the two primary methods show a meaningful checkpoint difference.

The queued runner is `scripts/run_n4_step500_to1000_explainability_sequence.py`. It waits for the
clean n=4/n=8 training queue to finish and all GPUs to become idle, then evaluates both checkpoints
in eight shards and writes paired bootstrap intervals.
