# PathMMU test999 image-cluster bootstrap results

Questions: 999; unique image clusters: 707; bootstrap replicates: 10000.

| Model | Accuracy | Image-cluster bootstrap 95% CI |
| --- | ---: | ---: |
| L-r16-SFT3000-step80 | 56.66% | [53.53%, 59.76%] |
| full-rule-RL-n8-step1000 | 63.46% | [60.49%, 66.47%] |
| GPT4o-Stage3-step1500 | 63.86% | [60.90%, 66.77%] |

Paired differences use `GPT4o-Stage3-step1500` as the reference.

| Comparator | Difference | 95% CI | Two-sided bootstrap p |
| --- | ---: | ---: | ---: |
| L-r16-SFT3000-step80 | +7.21% | [+4.24%, +10.23%] | 0.0000 |
| full-rule-RL-n8-step1000 | +0.40% | [-1.82%, +2.61%] | 0.7580 |
