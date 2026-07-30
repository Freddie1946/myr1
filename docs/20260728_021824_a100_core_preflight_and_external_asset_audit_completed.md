# A100 core preflight and external-asset audit completed

Created: 2026-07-28 02:18:24 +08:00

This is a preparation-only completion record. No training, model generation, validation/test
inference, paid API call, or cleanup was run. `formal_result` is false.

## Formal-machine preflight

The fixed Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5` is now local as five safetensors shards totaling
16,584,414,560 bytes. The frozen config, model-index and tokenizer-config hashes match
`protocol/base_model_manifest.json`.

`setup_formal_machine.sh` completed successfully and wrote:

- `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/reports/preflight_report.json`
- SHA-256:
  `04c772a2db6a9f10282d1f557f828f54959b9a72c3d1add78beca63f1235e80e`
- `passed=true`; all 22 named gates are true.

The final SFT environment is Torch 2.6.0+cu124, Transformers 4.49.0, Tokenizers 0.21.0,
Accelerate 1.2.1, Datasets 3.2.0 and TRL 0.9.6. The final GRPO environment is Torch
2.6.0+cu124, Transformers 4.49.0, Tokenizers 0.21.0, Accelerate 1.4.0, Datasets 3.3.2 and
TRL 0.15.2. Both `pip check` calls and both source import gates pass.

The setup log emitted a temporary resolver warning while the shared packages were installed into
the SFT environment. The subsequent editable LLaMA-Factory installation correctly restored its
required Accelerate/Datasets versions. This was an intermediate installation state, not a final
dependency failure.

The preflight reverified all 3,808 physical PathMMU images, frozen split hashes, parent-file
invariance, nested subsets, zero basename/content overlap, test blinding and `picked.json`
absence. Eight A100 GPUs were visible and idle; no model was loaded onto them.

## External evaluation assets

The fixed OmniMedVQA archive is 10,698,178,715 bytes with SHA-256
`12245e0f99afbc7d6f70e4ca3c2e5a7979a01816cd8159ae34539f4aa76adee0`.
Only the four approved sources were extracted.

The preparation-only audit report is:

- `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/external_eval_asset_audit_20260728.json`
- SHA-256:
  `9681e51dbbe6de1b438c0ecd0392857ea2dc9ebe3cf39ad538a86a5b83122aaa`

All gates pass:

| Source | QA count | Referenced images | Unique exact contents | Missing | PathMMU exact overlap |
|---|---:|---:|---:|---:|---:|
| Chest CT Scan | 871 | 382 | 349 | 0 | 0 |
| ISIC2020 | 1,580 | 1,499 | 1,499 | 0 | 0 |
| Retinal OCT-C8 | 4,016 | 3,224 | 3,222 | 0 | 0 |
| Diabetic Retinopathy | 2,051 | 1,966 | 1,966 | 0 | 0 |

PathVQA contains 6,719 QA rows and 858 unique image contents. It has zero exact-content overlap
with formal PathMMU and zero exact-content overlap with the four OmniMedVQA sources.

The report records `formal_split_labels_read=false` and `model_inference_performed=false`.

## Remaining preparation gates

- All isolated evaluation environments are already built and pass their CPU import/dependency
  gates.
- Qwen2.5-VL-3B-Instruct and MedVLM-R1 weights now pass exact-size completion gates. CONCH, UNI
  and PLIP weights were completed earlier.
- MedGemma, Lingshu, DeepSeek-VL2, InternVL3, HuatuoGPT-Vision and LLaVA-Med weight staging remains
  in resumable fixed-revision download queues.
- The two Meta Llama 3.2 Vision weight repositories remain approval-gated as recorded in
  `20260728_015359_llama_vision_weight_access_gate.md`.
- The exact n3000 epoch-3/step-1125 SFT parent is still absent from this A100 machine.
- No baseline smoke or evaluation may start until the common evaluation contract is approved by
  the user. Stage 3 remains explicitly unstarted.
