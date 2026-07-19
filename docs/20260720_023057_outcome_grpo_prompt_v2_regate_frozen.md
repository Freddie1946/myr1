# Outcome-GRPO prompt-contract v2 re-gate frozen

Timestamp: 2026-07-20 02:30:57 Asia/Shanghai

## User decision

The user explicitly approved the recommended correction after the first one-step gate exposed eight
zero format rewards. The correction removes only the trailing sentence `Output the final answer in
JSON format.` from the PathMMU GRPO question template. Accuracy parser v2 and the strict full-match
`<think>...</think><answer>...</answer>` format reward are unchanged.

No SFT run is repeated, no test data is accessed, and no long Outcome-GRPO run is authorized.

## Implementation and fail-closed behavior

The repository does not rewrite the vendored historical trainer. A PathMMU-specific dataset subclass
calls the vendored adapter, verifies that its generated text still ends in the exact known legacy JSON
sentence, and then replaces the text with contract `pathmmu_think_answer_only_v2`. If the upstream
prompt changes unexpectedly, the run fails instead of applying an approximate replacement.

The previous gate and its 16-GiB output remain preserved. This re-gate restarts from the same exact
validation-selected formal SFT n=3000 epoch-3/step-1125 parent.

All previous gates remain required. The new post-run gate additionally requires at least one positive
format reward among the eight raw completions. Positive total-reward group variance, positive trainer
reward standard deviation, a finite nonzero gradient, language-tensor change, exact visual-tensor
equality, save/load success, online/offline parser consistency, test isolation and `picked.json`
absence remain mandatory.

## Frozen evidence

- Protocol: `protocol/outcome_grpo_prompt_v2_gate_manifest_20260720_023057.json`.
- Code manifest: `protocol/outcome_grpo_prompt_v2_gate_code_manifest_20260720_023057.json`.
- Prompt helper, legacy-suffix failure guard, reward logger and synthetic post-run auditor tests pass.
- Engineering result only: every outcome remains `formal_result: false`.
- Failure policy: preserve artifacts and stop; no automatic retry or long RL launch.

## Reviewer mapping

This correction strengthens the auditable Outcome-GRPO prerequisite for R2-4, R3-3 and R3-5. It is
not itself a formal RL-scale result and does not resolve Stage 3 or expert-evaluation requests.
