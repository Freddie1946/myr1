# Option-conditioned RISE 96-case confirmation

Primary tests compare the ground-truth-option RISE-high region with five area/shape-matched random shifts under hard mean-fill validation. One-sided sign-flip p-values are Holm-adjusted over two models x deletion/retention (four tests). RISE generation used blurred replacement; validation therefore uses a different perturbation.

| Model | Mode | High vs random [95% CI] | Holm p | High vs low [95% CI] | Holm p | Positive cases |
|---|---|---:|---:|---:|---:|---:|
| gpt4o_stage3_epoch2_step1000 | deletion | +0.2407 [+0.1940, +0.2903] | 4e-05 | +0.4593 [+0.3772, +0.5465] | 4e-05 | 0.938 |
| gpt4o_stage3_epoch2_step1000 | retention | +0.2097 [+0.1642, +0.2601] | 4e-05 | +0.4445 [+0.3676, +0.5245] | 4e-05 | 0.906 |
| grok43_stage3_step1500 | deletion | +0.2205 [+0.1690, +0.2802] | 4e-05 | +0.4489 [+0.3675, +0.5372] | 4e-05 | 0.917 |
| grok43_stage3_step1500 | retention | +0.2241 [+0.1754, +0.2784] | 4e-05 | +0.4189 [+0.3371, +0.5053] | 4e-05 | 0.885 |

## Bidirectional diagnostic

- gpt4o_stage3_epoch2_step1000: both directions positive in 87.5% of cases; mean per-case minimum +0.1305 [+0.0972, +0.1647].
- grok43_stage3_step1500: both directions positive in 82.3% of cases; mean per-case minimum +0.1291 [+0.0924, +0.1679].

## Reproducibility gates

- gpt4o_stage3_epoch2_step1000: baseline/masked/intervention maximum A/B/C/D probability differences = 0.0/0.0/0.0.
- grok43_stage3_step1500: baseline/masked/intervention maximum A/B/C/D probability differences = 0.0/0.0/0.0.
