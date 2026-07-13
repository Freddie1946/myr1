# PathMMU Xet hang and standard-HTTP retry

Timestamp: `2026-07-13 21:32:48 Asia/Shanghai`

## Attempt result

The corrected bootstrap successfully prepared both isolated environments before reaching PathMMU:

- SFT: Python 3.10, Torch 2.6.0+cu124, Transformers 4.49.0, Tokenizers 0.21.0, TRL 0.9.6,
  LLaMA-Factory 0.9.2;
- GRPO: Python 3.10, Torch 2.6.0+cu124, Transformers 4.49.0, Tokenizers 0.21.0, TRL 0.15.2,
  vendored Open-R1 editable install.

The PathMMU request authenticated and resolved the dataset revision, but the Xet download transport
then remained in `xet_get` with a zero-byte incomplete archive. After several minutes, the process had
a `CLOSE-WAIT` HTTPS socket, showing that the peer had closed the connection. No image bytes had been
written. Codex interrupted only its own bootstrap process; no other user's process was touched.

## Correction

The bootstrap now defaults and exports `HF_HUB_DISABLE_XET=1`, which forces the ordinary Hugging Face
Hub HTTP download path. The setting is explicit in the example and gitignored machine-local configs.
This changes transport only; dataset ID, revision, archive hash reporting, exact 3,809-image extraction,
and all frozen-split checks remain unchanged.

The environments will be reused on retry. No training has run, and GPU 0 remained occupied at the
latest hardware snapshot.
