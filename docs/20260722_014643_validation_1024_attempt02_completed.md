# Corrected 1024-token validation Attempt02 completed

Completed: `2026-07-22T01:46:43.396123+08:00`

Run: `/home/wjy/pathvlm_r1_v1_formal/runs/validation_1024_all_candidates/validation_1024_20260722_003447`

Run-manifest SHA-256: `bcc4b463a455cc500f69152ca7dad42a5682c7cfa598f5c1724c7cf057734a6c`

Repository commit: `67e3ae62ebb1552b7ecade132c5e71f65b4a09d0`

Formal Stage-2 completion record:
`protocol/stage2_formal_n1000_seed42_completion_manifest_20260722_014643.json`

Attempt02 completed all 26 frozen candidates and saved exactly 10,010 raw predictions from the
385-QA validation split. Every job passed checkpoint integrity, source/index, deterministic decoding,
online/offline parser consistency, raw-output, and test-sealing gates. The aggregate run is
`formal_result: true`. Test was not accessed.

The two degenerate completions previously exposed by Attempt01 were hard-truncated at 1,024 tokens,
preserved verbatim, scored by unchanged parser v2, and reported without stopping the queue:
`n2000_epoch01` and `n2000_epoch05` each had one cap hit. The other 24 candidates had zero cap hits.

## Complete validation curve

| Candidate | Correct / 385 | Accuracy | Strict format | Max tokens | Cap hits |
|---|---:|---:|---:|---:|---:|
| Base | 195 | 50.65% | 89.61% | 418 | 0 |
| SFT n=500 final/e10 | 192 | 49.87% | 99.48% | 128 | 0 |
| SFT n=1000 final/e10 | 198 | 51.43% | 95.84% | 138 | 0 |
| SFT n=2000 e1 | 204 | 52.99% | 99.74% | 1024 | 1 |
| SFT n=2000 e2 | 212 | 55.06% | 100.00% | 131 | 0 |
| SFT n=2000 e3 | 202 | 52.47% | 99.48% | 221 | 0 |
| SFT n=2000 e4 | 188 | 48.83% | 99.74% | 143 | 0 |
| SFT n=2000 e5 | 223 | 57.92% | 99.74% | 1024 | 1 |
| SFT n=2000 e6 | 205 | 53.25% | 100.00% | 217 | 0 |
| SFT n=2000 e7 | 197 | 51.17% | 100.00% | 178 | 0 |
| SFT n=2000 e8 | 204 | 52.99% | 100.00% | 150 | 0 |
| SFT n=2000 e9 | 197 | 51.17% | 100.00% | 150 | 0 |
| SFT n=2000 e10 | 200 | 51.95% | 100.00% | 145 | 0 |
| SFT n=3000 e1 | 227 | 58.96% | 100.00% | 119 | 0 |
| SFT n=3000 e2 | 220 | 57.14% | 98.96% | 116 | 0 |
| SFT n=3000 e3 | 233 | 60.52% | 99.74% | 161 | 0 |
| SFT n=3000 e4 | 220 | 57.14% | 83.12% | 190 | 0 |
| SFT n=3000 e5 | 225 | 58.44% | 100.00% | 166 | 0 |
| SFT n=3000 e6 | 207 | 53.77% | 100.00% | 191 | 0 |
| SFT n=3000 e7 | 216 | 56.10% | 100.00% | 180 | 0 |
| SFT n=3000 e8 | 211 | 54.81% | 100.00% | 180 | 0 |
| SFT n=3000 e9 | 213 | 55.32% | 99.74% | 176 | 0 |
| SFT n=3000 e10 | 214 | 55.58% | 100.00% | 185 | 0 |
| Outcome GRPO n=1000 e1 | 237 | 61.56% | 100.00% | 149 | 0 |
| Outcome GRPO n=1000 e2 | 241 | 62.60% | 100.00% | 157 | 0 |
| Outcome GRPO n=1000 e3 | 237 | 61.56% | 100.00% | 141 | 0 |

The frozen selection rule (maximum validation accuracy, then strict format, then earliest epoch)
selects SFT n=2000 epoch 5, SFT n=3000 epoch 3, and Outcome-GRPO n=1000 epoch 2. No test result is
available or implied. No new training, Stage 3, prompt selection, parser change, or test inference was
performed in this validation run.
