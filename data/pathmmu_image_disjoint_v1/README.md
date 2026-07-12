# PathMMU image-disjoint split v1

Frozen: 2026-07-11

## Primary split

| Split | QA | Unique images | PubMed QA | EduContent QA |
|---|---:|---:|---:|---:|
| SFT train | 3000 | 2121 | 1839 | 1161 |
| RL train | 1000 | 708 | 613 | 387 |
| Validation | 385 | 272 | 236 | 149 |
| Test | 1000 | 708 | 613 | 387 |

All questions associated with one image remain in the same split. Pairwise image overlap between the four primary splits is zero.

The split was selected using source balance, answer-option balance, and questions-per-image balance only. No model outputs or model performance were used.

## Usage policy

- Use `validation` for checkpoint selection, training steps, prompts, parsing rules, reward weights, and other development decisions.
- Keep `test` sealed until the full training and evaluation protocol is fixed.
- Training seeds may change model optimization but must not regenerate any split file.
- `/home/wjy/processed_data/picked.json` is deprecated and is not part of this dataset version.

## Nested data-scaling subsets

SFT subsets are nested and image-complete:

- 500 QA
- 1000 QA
- 2000 QA
- 3000 QA

RL subsets are nested and image-complete:

- 250 QA
- 500 QA
- 1000 QA

## Reproducibility

- Split base seed: `20260711`
- Subset seed: `20260711`
- Full statistics and SHA256 hashes: `manifest.json`
- Integrity checks: `validation_report.json`
