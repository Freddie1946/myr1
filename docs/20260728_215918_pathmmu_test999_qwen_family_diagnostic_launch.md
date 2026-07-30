# PathMMU test999 Qwen-family diagnostic launch

Created: 2026-07-28 21:59:18 +08:00

This launch follows the explicit reclassification in
`20260728_215133_pathmmu_test999_reclassified_as_stage3_development_diagnostic.md`. The outputs are
pre-Stage-3 development diagnostics and may be used for bad-case analysis; they are not an
untouched final Stage 3 test.

Four fixed-revision local models passed an eight-record, batch-size-eight validation smoke with
nonempty outputs, 100% choice extraction, 100% strict format and zero generation-cap hits:

- Qwen2.5-VL-7B-Instruct;
- Qwen2.5-VL-3B-Instruct;
- Lingshu-7B;
- MedVLM-R1.

They are authorized to run all 999 corrected PathMMU v2 diagnostic records in parallel on physical
GPUs 0--3. The frozen execution contract is BF16, no quantization, greedy decoding, batch size 8,
`max_new_tokens=1024`, the approved common prompt and parser v2. Each raw output is appended
atomically with its source-record hash and can be resumed without regenerating completed records.

Frozen hashes and paths are in
`protocol/pathmmu_test999_qwen_family_diagnostic_launch_manifest_20260728_215918.json`.

The selected Stage 2 Outcome-GRPO epoch-2/step-1000 checkpoint is not on this A100 machine or in the
authenticated Hugging Face account. Its diagnostic run remains blocked on exact transfer; no
available model is a valid substitute.
