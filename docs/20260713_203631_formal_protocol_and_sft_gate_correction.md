# Formal protocol, split-hash, environment, and SFT gate correction

Timestamp: `2026-07-13 20:36:31 Asia/Shanghai`

## Scope

This is an engineering/protocol correction performed on the formal machine before downloading data
or models and before launching training. No test examples were used for selection, no model weights
were downloaded, and no training was started.

## Protocol corrections

- Formal Stage 1 is fixed to full-parameter language-model fine-tuning. Vision and multimodal
  projector parameters remain frozen. LoRA/QLoRA is not an allowed formal-paper substitute.
- Stage 3 remains reconstructed/completed work pending explicit agreement on its scientific
  definition. The older outline that presented online GPT process reward as already fixed is
  superseded.

## Frozen split hash correction

The 22 JSON hashes in the frozen data report were created from CRLF-serialized files. Git stores the
same content with LF line endings, so raw checkout hashes differ. No split content changed.
`formal_machine/verify_frozen_splits.py` now canonicalizes line endings to CRLF for historical hash
verification and independently checks:

- all 22 recorded hashes;
- exact 3000/1000/385/1000 QA and 2121/708/272/708 image counts;
- the exact 5,385-QA union;
- all six pairwise image overlaps equal zero;
- nested, image-complete SFT 500/1000/2000/3000 and RL 250/500/1000 subsets;
- absence of `picked.json`.

The read-only formal-machine check passed every gate.

## Environment correction

The formal host has NVIDIA driver 580.142. The official PyTorch wheel index was queried without
installing packages and contains Torch 2.6.0+cu124 and torchvision 0.21.0+cu124. The formal template
now selects cu124 instead of the old cu118 default. Separate SFT and GRPO environments remain
required. The inherited `wjy` environment is not a formal environment.

## Formal SFT smoke gate implementation

Added an audited launcher and helpers that will, after bootstrap and explicit GPU allocation:

1. validate the 8-sample SFT adapter, fixed 7B parent, seed 42, full fine-tuning, and freeze flags;
2. snapshot Git, LLaMA-Factory, pip/Conda, CUDA, NVIDIA, DeepSpeed, model, data, and code-manifest state;
3. save exact run-specific configs, `command.txt`, and `run_manifest.yaml`;
4. train one optimizer step and require a gathered `checkpoint-1`;
5. independently load model and processor from `checkpoint-1` on CPU;
6. resume from `checkpoint-1` to global step 2 and require `checkpoint-2`;
7. independently load `checkpoint-2`;
8. require a representative language tensor to change and a frozen visual tensor to remain exactly
   equal for both the initial and resumed updates;
9. parse and record LLaMA-Factory trainable parameter counts and visual/projector freeze messages.
10. record loss/gradient history and sampled GPU-memory, utilization, RAM, and swap peaks for both steps.

The launcher is intentionally not executed in this correction. It remains an engineering smoke with
`formal_result: false`.

## Preflight strengthening

Preflight now checks the exact `config.json`, model index, and tokenizer-config hashes from the fixed
base-model manifest; frozen-split verification; exact adapter source counts; CUDA availability; and
the pinned SFT/GRPO package versions. Bootstrap must pass this strengthened report before the SFT
smoke launcher is authorized.

## Remaining external gates

- The user must complete Hugging Face authentication for the gated PathMMU dataset.
- The user must confirm which physical GPUs are allocated on the shared server.
- Only then may machine-local configuration and bootstrap/download proceed.
