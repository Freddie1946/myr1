# External VQA variable-option serialization correction

Created: `2026-07-29T22:57:33+08:00`

The first MedVLM-R1 OmniMedVQA process stopped after preserving indices 0–540. The next source
record has only options A and B. Although prompt construction and official scoring already handled
the actual number of options correctly, the provenance row attempted to serialize A, B, C and D
unconditionally and raised `KeyError: option_C` after that record's inference.

The prepared four-source set contains 1,278 two-option and 7,240 four-option records. The earlier
contract document's description of all 8,518 records as four-choice was therefore inaccurate.

The correction changes only provenance serialization to retain the option keys that exist in each
source record. It does not change model input, prompt text, candidate ordering, answer mapping,
generation, metric or any saved prediction. A new two-option regression test was added; all six
tests and compilation pass. MedVLM-R1 resumed at index 541 after validating every existing source
record hash.

Corrected hashes and the failed-attempt facts are in
`protocol/external_vqa_variable_option_serialization_correction_20260729_225733.json`.
