# PathMMU download and processing on the formal machine

Timestamp: `2026-07-12 21:33:40 Asia/Shanghai`

## Source and access gate

The authoritative source is `https://huggingface.co/datasets/jamessyx/PathMMU`. It is gated under
CC BY-ND 4.0. The account owner must complete the linked application, use the same email address,
accept the Hugging Face conditions, and wait for approval. No script can bypass this step.

After approval, authenticate on the formal machine with either:

```bash
hf auth login
```

or a session-only token:

```bash
export HF_TOKEN='<read token>'
```

Never commit the token or place it in `formal_machine.env`.

On a completely bare machine the `hf` command may not exist yet. In that case, export `HF_TOKEN`
for the first setup run. Alternatively, let setup create `$INSTALL_ROOT/envs/grpo`, then run
`$INSTALL_ROOT/envs/grpo/bin/hf auth login` and rerun setup.

## Online end-to-end setup

```bash
git clone -b codex/formal-machine-handoff https://github.com/Freddie1946/myr1.git
cd myr1
cp formal_machine/formal_machine.env.example formal_machine.env
```

Edit `formal_machine.env`. Keep these values for online acquisition:

```bash
ONLINE=1
PATHMMU_AUTO_DOWNLOAD=1
PATHMMU_DATASET_ID="jamessyx/PathMMU"
PATHMMU_DATASET_REVISION="main"
PATHMMU_ARCHIVE=""
```

Set valid `INSTALL_ROOT`, `BUNDLE_ROOT`, `SPLIT_ROOT`, `IMAGE_ROOT`, `CONDA_EXE`, and GPU fields,
then run:

```bash
bash setup_formal_machine.sh formal_machine.env
```

The setup creates the GRPO environment before data acquisition, uses `huggingface_hub` to fetch
the official `images.zip` (about 1.93 GB), and reads every frozen `*_with_cot.json`. Only the 3,809
unique basenames required by SFT/RL/validation/test are extracted. ZIP directory names are ignored,
duplicate basenames are rejected, and Pillow verifies every resulting image.

The report is written to:

```text
$INSTALL_ROOT/reports/pathmmu_download_report.json
```

It records the resolved Hugging Face revision, archive path/size/SHA-256, required count, extracted
count, and missing names. Setup stops if even one frozen-split image is missing. This matters because
the official PathMMU documentation states that some source subsets require additional acquisition;
we must verify the frozen 3,809-image subset empirically after authenticated download.

After image extraction, existing `prepare_formal_data.py` rewrites paths, verifies exact QA/image
counts, checks nested scale subsets and all pairwise image-disjoint constraints, then creates the
LLaMA-Factory and GRPO adapters. The formal preflight runs last.

## Reusing an already downloaded archive

If `images.zip` was transferred through the remote-desktop shared drive:

```bash
PATHMMU_AUTO_DOWNLOAD=1
PATHMMU_ARCHIVE="/absolute/path/to/images.zip"
```

The same whitelist extraction and validation are used without a second dataset download. Fully
offline setup additionally requires the wheelhouse, exact 7B snapshot, and audited LLaMA-Factory
source described in `CODEX_START_HERE.md`.

## Failure handling

- `401/403` or gated-repository error: confirm approval and run `hf auth login` again.
- Network interruption: rerun setup; Hugging Face cache and already validated images are reused.
- Missing images in the report: do not train. Follow the official PathMMU additional acquisition
  instructions for the named source subsets, then rerun with `PATHMMU_AUTO_DOWNLOAD=0` so the data
  preparation stage validates the completed `IMAGE_ROOT`.
- Corrupt existing image: remove only the named corrupt file and rerun extraction.

The script never downloads obsolete checkpoints and never changes the frozen JSON split.
