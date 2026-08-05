# Grok-parallel GPT-4o admission and Gemini 3.5 route audit

Recorded at: `2026-08-05 22:07:16 CST`

## Authorization and boundary

The user authorized other useful evaluations and experiments to run while the formal Grok 4.3
Stage3 arm remains active. This record prepares only the previously frozen GPT-4o selected-model
closed-question evaluations and its independent 24-case visual-fidelity arm. PathVQA free-form
work remains postponed. Previously stopped hosted Grok external-dataset baselines and Doubao are
not resumed by this authorization.

Secondary failure must stop the secondary process, not the Grok supervisor. No code in this change
signals, terminates or restarts the Stage3 owner.

## Shared-GPU admission

Default behavior is unchanged: without an explicit environment flag, both launchers require an
idle GPU with at most 10 MiB used. The opt-in mode requires:

- `PATHVLM_ALLOW_SHARED_GPU_WITH_STAGE3=true`;
- an explicit owner under `pathvlm_r1_v1_a100/runs/stage3_process_grpo/`;
- at least eight live owner workers containing `grpo_pathmmu.py`;
- evidence that the owner has completed an optimizer step;
- at most 30,000 MiB initially used and at least 45,000 MiB initially free on every secondary GPU.

The three-dataset launcher runs its existing 16-case behavior smokes first and rechecks the owner
before full PathMMU test999, PathVQA yes/no-3362 and OmniMedVQA-8518 inference. The evaluation run
contract and visual completion record both preserve the admission mode, owner and thresholds.

At preparation time Grok had completed step 15. All eight GPUs used approximately 20.7 GiB of
81.9 GiB, leaving about 61.2 GiB each. The prepared assignment is GPUs 0/1/2 for the three selected
GPT-4o tasks and GPU 3 for the visual-fidelity arm.

## Gemini 3.5 no-thinking audit

One zero-retry, non-training PathMMU-validation smoke requested
`gemini-3.5-flash-nothinking` through AIGCBest. The response was HTTP-successful and stopped
normally, but reported served identity `gemini-3.5-flash`. The strict candidate gate therefore
failed before any stability run. Estimated request cost was `$0.005394`.

This confirms that AIGCBest exposes no-thinking as a routing/configuration alias rather than an
independently returned model identity. It may later be reported transparently as Gemini 3.5 Flash
with thinking disabled only after an explicit alias contract; it is not eligible under the current
exact requested/served identity gate.

Report SHA-256:
`a22340ab73a61ea4595a2f3bf84de5278a92194f81c3a8e13cbcfa3ba60042ab`.

## Verification

- Shell syntax passed for both launchers.
- Four shared-admission static tests passed.
- Thirteen external-VQA scope and smoke/full integrity tests passed.
- Eleven visual preparation, execution, verification and aggregation tests passed in the frozen
  SFT environment.
- The system Python lacked matplotlib; this was an environment-selection error, not a code or
  formal-environment failure. No package was installed or changed.

No selected-model evaluation or visual inference is claimed complete by this preparation record.
