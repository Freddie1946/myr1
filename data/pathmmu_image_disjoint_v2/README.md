# PathMMU exact-content-disjoint split v2

Frozen: 2026-07-19

This version is a deterministic correction of `pathmmu_image_disjoint_v1`. SFT, RL, validation,
and every nested SFT/RL subset are unchanged. One opaque test QA was removed before any formal RL
or test evaluation because its image file was byte-for-byte identical to an RL image under a
different basename. The test split therefore contains 999 QA and 707 image basenames.

The split unit is now audited at two levels: image basename and SHA-256 of exact image-file bytes.
All six pairwise SFT/RL/validation/test overlaps are zero at both levels. `picked.json` remains
forbidden. Test remains evaluation-only.

The correction decision used only image basename and file digest. The removed test question,
answer, model output, and model performance were not inspected or used. See `manifest.json`,
`validation_report.json`, and `image_content_sha256.json` for exact provenance.
