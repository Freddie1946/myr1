# Meta Llama Vision weight-access gate

Created: 2026-07-28 01:53:59 +08:00

This is a preparation-only correction record. No model was loaded, no inference or evaluation was
run, and no training or paid API request was started.

## Corrected access status

Authenticated Hugging Face metadata lookup succeeds for both frozen repositories below, but an
actual weight download fails with `Access denied. This repository requires approval.`:

- `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `meta-llama/Llama-3.2-90B-Vision-Instruct`

The failed download loops were terminated after reproducing the repository gate. A successful
metadata request must therefore not be treated as evidence of weight access.

The preparation environments and pinned source revisions remain valid. The only new blocker is
that the Hugging Face account used on this machine must be approved for both gated repositories
before their exact weights can be staged.

## Required user action

While authenticated as the same Hugging Face account used by this workspace, accept/request access
on both repository pages:

- <https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct>
- <https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct>

After approval, rerun the fixed-revision downloads and verify exact weight-file totals before any
smoke or formal evaluation. Do not substitute another Llama revision or an ungated derivative.
