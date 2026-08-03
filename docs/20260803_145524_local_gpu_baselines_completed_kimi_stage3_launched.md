# Local GPU baselines completed and fresh Kimi Stage3 launched

Recorded at: `2026-08-03 14:55:24 CST`

## Outcome

The approved local-GPU sequence completed without changing prompts, parsers, generation limits or
individual answers. All completed full runs passed count, ordered-index, source-record, data,
model-config and predictions-file SHA-256 verification. The ModelScope-mirror provenance for both
Llama rows remains mandatory disclosure; this execution does not turn it into official Hugging
Face provenance.

| Model / task | Primary reported result | Generation cap hits |
| --- | ---: | ---: |
| Llama 3.2 11B / PathMMU test999 | 222/999 (22.22%) | 633/999 |
| Llama 3.2 11B / OmniMedVQA | 4591/8518 (53.90%) contract-aligned | 1481/8518 |
| DeepSeek-VL2 / PathVQA | yes/no 59.52%; free-form macro token F1 3.77% | 25/6719 |
| DeepSeek-VL2 / OmniMedVQA | 4392/8518 (51.56%) contract-aligned | 27/8518 |
| Llama 3.2 90B / PathMMU test999 | 571/999 (57.16%) | 0/999 |
| Llama 3.2 90B / PathVQA | yes/no 62.49%; free-form macro token F1 4.14% | 11/6719 |
| Llama 3.2 90B / OmniMedVQA | 5800/8518 (68.09%) contract-aligned | 5/8518 |

PathVQA is not collapsed into one misleading score: its paper-aligned yes/no accuracy and
free-form macro token F1 remain separate. Llama 11B PathVQA was not run full because its frozen
smoke failed the cap-hit gate. DeepSeek-VL2 PathMMU remains excluded by its earlier terminal
response-contract failure. Llama 11B PathMMU's high full-run cap-hit rate is retained as an
explicit result-quality caveat; it was not used to retroactively edit outputs or change the
predeclared smoke decision.

All three Llama 90B 16-case smokes passed before test/full access. PathMMU and OmniMedVQA were
16/16 nonempty and parseable with zero cap hits. PathVQA was 16/16 nonempty, 14/16 parseable,
covered both answer types and had zero cap hits. Accuracy was not a smoke gate.

## Fresh Kimi Stage3 launch

After the final 90B integrity check released all GPUs, a new synthetic paid contract smoke passed
for `moonshotai/kimi-k2.6` served by `Inceptron`:

- one physical attempt, cost `$0.00072115`;
- ZDR required and data collection denied;
- reasoning disabled, 1,024 Judge-token limit;
- fresh run budget unchanged at `$26.71782984`;
- zero unresolved reservations when the smoke marker was written;
- marker SHA-256
  `ce7a95573d1049192b6c26b45b93088fa1a928aa85234d23dcc7923b5a3c3d2e`.

The bounded supervisor recorded `segment00` start at
`2026-08-03T06:54:18.683037+00:00`. The launcher passed the frozen source, model, trainability,
eight-GPU and disk preflights, created eight worker processes, and entered the 1,500-step training
loop with save interval 100 and at most three recognized transient recoveries. This satisfies the
user's requested stopping boundary: Kimi is launched, and no continued Codex monitoring is
required by this sequence.

The stable sequence-completion record has SHA-256
`a2787628d8e81fba3dedd3f53b8450e94909a717a71332dd73fd7e483ac647f6`; the stable sequence event
log has SHA-256 `2b63067a63b8466c227dabdfad62b0b6456d4512b69e657ad511ff17a512100c`.

