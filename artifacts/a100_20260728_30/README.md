# A100 experiment logs and metric artifacts

This directory contains the compact, non-model artifacts copied from the A100 execution workspace
for Git/GitHub preservation.

Included:

- SFT4000 training, resource, configuration, reload and parameter-delta logs;
- Kimi 2.6 50-step Stage3 pilot launch/recovery logs and validation metrics;
- completed PathMMU, PathVQA and OmniMedVQA metric JSON files.

Intentionally excluded:

- checkpoints, optimizer states and model weights;
- raw benchmark records, images and full prediction files;
- OpenRouter request/response caches, Judge evidence and API accounting records containing request
  content;
- credentials and the local `scripts/keepintouch.txt` file.

Timestamped protocol JSON files and result summaries under `protocol/` and `docs/` remain the
authoritative interpretation layer. These copied logs are supporting execution evidence and retain
their original filenames. The ongoing full Kimi Stage3 arm is recorded by its launch protocol now;
its final log and artifacts must be added only after that arm terminates.
