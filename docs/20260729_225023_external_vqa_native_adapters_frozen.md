# External VQA native adapters frozen

Created: `2026-07-29T22:50:23+08:00`

While the eight Transformers-native full runs were in progress, separate native adapters were
added for InternVL3, HuatuoGPT-Vision and OmniMedVQA PLIP/CONCH matching. They inherit the already
frozen datasets, prompts, scoring functions and generation limits without modification.

InternVL uses its official `model.chat` interface and the same dynamic 448-pixel tiling used by the
existing audited PathMMU runner. Huatuo uses the repository-native `HuatuoChatbot` preprocessing
and generation path. PLIP and CONCH score each OmniMedVQA image against that question's four raw
option texts by cosine similarity; these matching results will be reported separately from
generative question-answering scores.

All four new files compile and the shared utilities import successfully. GPU adapter smokes remain
pending until GPUs become available. Frozen hashes are recorded in
`protocol/external_vqa_native_adapter_addendum_20260729_225023.json`.
