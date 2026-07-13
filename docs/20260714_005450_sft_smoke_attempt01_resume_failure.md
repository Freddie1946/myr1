# Formal 7B SFT smoke Attempt01 resume failure

Timestamp: `2026-07-14 00:54:50 Asia/Shanghai`

The first formal-machine 7B SFT save/reload/resume smoke used physical GPUs 1--4 and the exact frozen
Qwen2.5-VL-7B base. Its initial optimizer step completed with loss 1.8846 and gradient norm 45.9465.
LLaMA-Factory reported 7,615,616,512 trainable language parameters out of 8,292,166,656 total, and
explicitly logged the vision tower and multimodal projector as frozen. `checkpoint-1` was gathered,
saved with optimizer state, and independently reloaded as a Qwen2.5-VL 7B checkpoint.

The resume process then failed before the second optimizer step. PyTorch 2.6 changed the default for
`torch.load` to `weights_only=True`; DeepSpeed 0.15.4 calls `torch.load` without an explicit value when
reading its own optimizer checkpoint, whose serialized `ZeroStageEnum` is not on the default safe
allowlist. All ranks therefore stopped with `_pickle.UnpicklingError`. This is an environment
compatibility failure, not an OOM, data failure, or model-identity failure. The run manifest correctly
records `status: failed` and `formal_result: false`; the failed run directory is preserved.

The installed PyTorch source documents `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` as the compatibility
override when the call site did not explicitly set `weights_only`. Because the checkpoint is generated
locally in the immediately preceding audited step, its trust boundary is explicit. The smoke runner
now removes any conflicting force-safe variable and sets this override only for the resume subprocess.
The manifest records the compatibility choice. A fresh smoke attempt is required; no formal long run
is authorized by this partial result.
