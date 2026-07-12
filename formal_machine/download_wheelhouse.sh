#!/usr/bin/env bash
set -euo pipefail

DEST="${1:?usage: download_wheelhouse.sh /path/to/wheelhouse [/path/to/LLaMA-Factory]}"
LLAMAFACTORY_SRC="${2:-}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
mkdir -p "$DEST"

python3 -m pip download -d "$DEST" --index-url "$PYTORCH_INDEX_URL" \
  torch==2.6.0 torchvision==0.21.0
python3 -m pip download -d "$DEST" \
  transformers==4.49.0 tokenizers==0.21.0 accelerate==1.4.0 datasets==3.3.2 \
  deepspeed==0.15.4 peft pillow safetensors sentencepiece qwen-vl-utils \
  math-verify pyyaml packaging psutil scipy rich tyro bitsandbytes liger-kernel==0.5.2
python3 -m pip download --no-deps -d "$DEST" trl==0.9.6
python3 -m pip download --no-deps -d "$DEST" trl==0.15.2

if [[ -n "$LLAMAFACTORY_SRC" ]]; then
  [[ -f "$LLAMAFACTORY_SRC/pyproject.toml" || -f "$LLAMAFACTORY_SRC/setup.py" ]] || {
    echo "Invalid LLaMA-Factory source: $LLAMAFACTORY_SRC" >&2; exit 1;
  }
  python3 -m pip download -d "$DEST" "$LLAMAFACTORY_SRC"
fi

echo "Wheelhouse created at $DEST"
