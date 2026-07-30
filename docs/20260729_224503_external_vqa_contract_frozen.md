# PathVQA and OmniMedVQA common contract frozen

Created: `2026-07-29T22:45:03+08:00`

The external evaluation contract was frozen only after the SFT4000 control and PathMMU triad
completed. Stage3 remains unstarted.

PathVQA contains 6,719 test QAs over 858 unique exact image contents: 3,362 yes/no and 3,357
free-form records. Generative models receive the image and question with one fixed instruction to
return only a short answer. Primary reporting is conservative normalized exact match for the
yes/no, free-form and overall strata. The normalization does not remove articles, stem words,
introduce semantic aliases, match against the global answer pool, or use an LLM judge.

OmniMedVQA contains 8,518 four-choice QAs from the four user-requested sources: Chest CT Scan 871,
Diabetic Retinopathy 2,051, ISIC2020 1,580 and Retinal OCT-C8 4,016. The prompt and primary scoring
follow the released official question-answering semantics: show the candidate answer texts,
request only the selected answer text, then map the raw completion to the option with the highest
Python `SequenceMatcher` ratio. Strict normalized option-text exact match is retained as a
secondary diagnostic. Both micro and per-source scores will be reported.

All generative runs are zero-shot, greedy BF16 without quantization, with a fixed 64-token cap.
Raw predictions, mapped answers, source-record hashes and configuration hashes are mandatory.
Sixteen-record adapter smokes may validate only backend plumbing; because they touch test records,
they cannot change the frozen prompt, metric, parser or threshold.

PLIP and CONCH are compatible only with OmniMedVQA's per-question candidate options, not PathVQA's
open answer space. UNI has no text interface. DeepSeek-VL2 and LLaVA-Med require separate native
adapter gates; unavailable Meta weights and hosted or paid APIs remain outside this run.

The immutable contract is
`protocol/external_vqa_common_contract_manifest_20260729_224503.json`. Five deterministic scoring
regression tests and Python compilation passed before inference.
