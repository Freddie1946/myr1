# PathMMU Qwen runner import correction

Created: `2026-07-29T22:20:00+08:00`

The first two SFT4000/Stage2 validation-smoke processes failed during Python module import, before
argument parsing, output-directory creation, data access, model loading or inference. The pinned
Qwen environment uses Transformers 4.49.0 and supports `Qwen2_5_VLForConditionalGeneration`, but
does not export `Gemma3ForConditionalGeneration`. The shared runner imported both eagerly.

The runner now imports the Gemma 3 class only when the `gemma3` backend is selected. No Qwen prompt,
decoding, parser, data or checkpoint behavior changed. Python compilation and the PathMMU reward
tests pass. The correction manifest is
`protocol/pathmmu_test999_sft4000_stage2_import_correction_20260729_222000.json`.
