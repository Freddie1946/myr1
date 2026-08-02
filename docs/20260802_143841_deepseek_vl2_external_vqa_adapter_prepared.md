# DeepSeek-VL2 external-VQA adapter prepared

Recorded at: `2026-08-02 14:38:41 CST`

DeepSeek-VL2 is a mandatory original-manuscript local baseline.  Its official native PathMMU
adapter previously produced nonempty responses but failed the fixed extractable-answer response
contract, so PathMMU test999 was correctly not run.  That failure does not by itself determine
whether its short-answer PathVQA and candidate-text OmniMedVQA interfaces are valid.

`scripts/run_external_vqa_deepseek_vl2.py` now provides a separate, unrun external-evaluation
adapter using the official `DeepseekVLV2Processor`, `prepare_inputs_embeds` and language-model
generation interface.  It inherits the same frozen prompts, source-record SHA-256, append-only
resumption, deterministic BF16/no-quantization decoding and scoring utilities used by the other
local baselines.  Full evaluation cannot use `--limit`; adapter smoke is the only mode that can.

This preparation made no model load, GPU inference or test-data access.  After GPT-4o Stage3 and
its three-epoch validation finish, DeepSeek-VL2 must pass separate PathVQA and OmniMedVQA adapter
smokes before either full run.  A PathMMU response-contract failure remains recorded and is not
silently repaired with a model-specific post-test parser.

Verification:

- isolated DeepSeek-VL2 environment imports the new runner and parses `--help`;
- Python compilation passed;
- 14 relevant static/contract tests passed;
- `git diff --check` passed.

