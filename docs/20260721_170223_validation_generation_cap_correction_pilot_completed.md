# Validation generation-cap correction pilot completed

Timestamp: `2026-07-21T17:02:23+08:00`

## Disposition

The user-authorized validation-only Base/SFT/Outcome-GRPO pilot completed successfully with
deterministic decoding and `max_new_tokens=768`. All three jobs exited zero, saved 385 raw
predictions, passed source-record and offline parser-v2 consistency checks, and returned their GPUs
to the 18-MiB idle baseline. Test was not accessed. This pilot diagnoses and corrects an evaluation
protocol problem; it remains `formal_result: false` until every checkpoint participating in formal
selection is rerun under the corrected protocol.

Run root:

`/home/wjy/pathvlm_r1_v1_formal/runs/validation_generation_cap_pilot/validation_generation_cap_pilot_20260721_161141`

## Results

| Model | Correct / 385 | Accuracy | Strict format | Choice extracted | Mean tokens | Maximum tokens | Cap hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 195 | 50.6494% | 345 | 383 | 207.12 | 417 | 0 |
| SFT n=3000 epoch 3 | 233 | 60.5195% | 384 | 385 | 75.67 | 160 | 0 |
| Outcome-GRPO n=1000 epoch 2 | 241 | 62.5974% | 385 | 385 | 73.82 | 156 | 0 |

The Base 768-token completions are byte-for-byte identical for all 385 source identities to the
earlier 512-token diagnostic. The SFT and Outcome-GRPO 768-token completions are respectively
byte-for-byte identical for all 385 records to their historical 192-token outputs. Therefore the
corrected cap changes the invalid historical Base comparison but does not change the selected SFT
or Outcome-GRPO validation scores.

## Paired comparisons

- Base versus SFT: Base-only correct 56, SFT-only correct 94; exact two-sided McNemar
  `p=0.0024055`.
- Base versus Outcome-GRPO: Base-only correct 50, Outcome-GRPO-only correct 96;
  `p=0.0001750`.
- SFT versus Outcome-GRPO: SFT-only correct 21, Outcome-GRPO-only correct 29;
  `p=0.3222363`.

SFT and Outcome-GRPO remain significantly better than the fairly evaluated Base on this validation
split. The eight-question Outcome-GRPO advantage over SFT is not statistically significant on 385
validation records and must not be overstated.

## Scientific correction

The historical Base result `100/385 = 25.9740%` arose because 231 records reached the 192-token
generation cap before many could emit a final answer. It must remain preserved as historical run
evidence but must not be used for a comparable Base-to-trained-model improvement claim. Under the
corrected pilot the corresponding gains are 9.8701 percentage points for SFT and 11.9481 points for
Outcome-GRPO, rather than 34.5455 and 36.6234 points.

The next gate is a separate documented corrected-protocol validation run over every checkpoint
that participates in SFT scale/epoch or Outcome-GRPO epoch selection. Test remains sealed. Stage 3
has not started; external process supervision may help some negative-transfer cases, but this is a
hypothesis requiring a frozen scientific definition and validation evidence.

## Reviewer mapping

This correction directly strengthens R1-7 baseline prompt/configuration disclosure, R2-3 fair
same-base comparison, and the paired statistical analysis requested for the staged claims. It does
not itself complete the multi-seed, generalized-baseline, Stage 3, or final test requirements.

