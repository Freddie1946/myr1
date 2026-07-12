#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:?usage: build_transfer_bundle.sh /path/to/output.tar.gz [split_root] [image_root] [llamafactory_src]}"
SPLIT_ROOT="${2:-$REPO_ROOT/data/pathmmu_image_disjoint_v1}"
IMAGE_ROOT="${3:-}"
LLAMAFACTORY_SRC="${4:-}"

[[ -d "$SPLIT_ROOT" ]] || { echo "Missing split root: $SPLIT_ROOT" >&2; exit 1; }
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/pathvlm-formal-handoff/data/pathmmu_image_disjoint_v1"

rsync -a --exclude='.git' --exclude='tmp' --exclude='*.safetensors' --exclude='output/' \
  "$REPO_ROOT/" "$STAGE/pathvlm-formal-handoff/"
rsync -a "$SPLIT_ROOT/" "$STAGE/pathvlm-formal-handoff/data/pathmmu_image_disjoint_v1/"

if [[ -n "$IMAGE_ROOT" ]]; then
  mkdir -p "$STAGE/pathvlm-formal-handoff/images"
  python3 - "$SPLIT_ROOT" "$IMAGE_ROOT" "$STAGE/pathvlm-formal-handoff/images" <<'PY'
import json, shutil, sys
from pathlib import Path
split_root, image_root, output = map(Path, sys.argv[1:])
names = set()
for path in split_root.rglob('*_with_cot.json'):
    for row in json.loads(path.read_text(encoding='utf-8')):
        names.add(Path(row['image']).name)
missing = []
for name in sorted(names):
    source = image_root / name
    if not source.is_file(): missing.append(str(source)); continue
    shutil.copy2(source, output / name)
if missing: raise SystemExit(f'missing {len(missing)} images: {missing[:5]}')
print(f'copied {len(names)} unique images')
PY
fi

if [[ -n "$LLAMAFACTORY_SRC" ]]; then
  [[ -f "$LLAMAFACTORY_SRC/pyproject.toml" || -f "$LLAMAFACTORY_SRC/setup.py" ]] || {
    echo "Invalid LLaMA-Factory source: $LLAMAFACTORY_SRC" >&2; exit 1;
  }
  mkdir -p "$STAGE/pathvlm-formal-handoff/vendor/LLaMA-Factory"
  rsync -a --exclude='.git' --exclude='saves/' --exclude='output/' --exclude='__pycache__/' \
    "$LLAMAFACTORY_SRC/" "$STAGE/pathvlm-formal-handoff/vendor/LLaMA-Factory/"
fi

tar -C "$STAGE" -czf "$OUTPUT" .
sha256sum "$OUTPUT" > "$OUTPUT.sha256"
echo "Created $OUTPUT"
echo "Created $OUTPUT.sha256"
