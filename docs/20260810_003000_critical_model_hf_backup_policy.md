# Critical-model Hugging Face backup policy

Date: 2026-08-10 00:30 Asia/Shanghai

The results-only rule for small data-ratio checkpoints remains in force. The user explicitly
selected exactly three exceptions:

1. Grok 4.3 Stage3, step 1500: private HF model repository;
2. Stage2 continued with rule-only RL, step 1500: public repository with manual gated access;
3. base plus 4,000-example rule-only RL, step 6000: public repository with manual gated access.

For each gated model, the worker creates the repository, activates and verifies `manual` gating,
uploads a model card, and only then uploads the weights. The repository name, card and metadata are
public; downloads require individual owner approval. The worker verifies the terminal training
step, file completeness, visibility, gate mode, sizes and hashes. Optimizer/RNG state and all small
ratio checkpoints are excluded.

The Grok output is already complete and is backed up first. The two rule-RL controls are uploaded
only after their formal task state and train-state audit prove exact completion. This policy keeps
the expensive Grok arm private while moving reproducibility controls to public-storage accounting.
