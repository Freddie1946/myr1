# GPT-4o visual completed and partial full evaluation started

Recorded at: `2026-08-05 22:20:15 CST`

## Visual-fidelity completion

The selected GPT-4o Stage3 checkpoint completed the frozen 24-case PathMMU-validation
perturbation-fidelity arm while sharing GPU 3 with Grok Stage3. Verification passed for all ordered
cases, 36 patch scores per case, raw result hashes and the exact 25-figure inventory.

Primary results:

- panel answer accuracy: 12/24 (50.00%);
- mean deletion AUC advantage over random: `0.0214979178`;
- mean insertion AUC advantage over random: `0.0184948114`;
- mean top-25% comprehensiveness: `0.0253713042`;
- mean random-25% comprehensiveness: `-0.0153533619`.

The output is a single verified arm, not the complete five-arm comparison. It supports only the
frozen perturbation-fidelity wording and is not attention, causal localization or lesion IoU.

Artifacts:

- completed record SHA-256:
  `33b0a663ba1e0439a75d2a070d9466256307ce7d57b84a9d64e96117bd5d21ea`;
- metrics SHA-256:
  `883a296523a0c4db691eba39baedb232a9967269264a85d7c09dec408fe28e35`;
- verification SHA-256:
  `2ccdbf7fa45271a97219a0adc3dee328cd8026b77b8cef5f82ca7ecbe4dcf1a6`.

## Partial selected-model evaluation

A fresh run explicitly selected `pathmmu,pathvqa`, used GPUs 0/1 and reran both 16-case behavior
smokes. Both passed under the unchanged aggregate policy, after which the launcher recorded
`full_phase_started` and began:

- PathMMU test999 as development diagnostic;
- PathVQA yes/no-only 3,362-case external evaluation.

At this record boundary the raw prediction files contained 25 and 176 rows respectively. These
counts are progress only and are not converted into partial scores. The run contract SHA-256 is
`d71c710e175e0e9c3cc3d81e8ebd8d54a0bd3c47a0cc103c30b95279af10d1bd`.

OmniMedVQA is not part of this new run and remains missing after the independently retained
15/16 generation-cap smoke failure. PathVQA free-form remains excluded.

## Grok owner status

Grok Stage3 retained all eight workers, reached step 29, and had 241 completed Judge requests with
`$1.11930625` committed at this boundary. No OOM, Judge failure, fallback-limit event or supervisor
recovery occurred during the visual completion and evaluation launch.

The completed visual tree and eventual completed selected-model evaluation require a new immutable
private-HF results backup. No partial prediction tree is presented as a completed backup.
