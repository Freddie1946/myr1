#!/usr/bin/env bash
set -euo pipefail

REPO_ID="Freddie1946/PathVLM-R1-Migration-Archive-20260815"
REVISION="5d2485312266ec0670a14494b1fe999d10fe94e1"
PREFIX="increments/20260815_final_experiment_records_v1"
ARCHIVE="pathvlm_final_experiment_records_20260815T123626Z.tar.gz"
EXPECTED_SHA256="0c00a19b827ec8d3ec0c539ca6741507355397569d4e851e5eb60c66ae6cb625"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 TARGET_ROOT [--extract]" >&2
  exit 2
fi
TARGET_ROOT="$(realpath -m "$1")"
EXTRACT=false
if [[ ${2:-} == "--extract" ]]; then
  EXTRACT=true
elif [[ -n ${2:-} ]]; then
  echo "Unknown option: $2" >&2
  exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI ('hf'). Install huggingface_hub first." >&2
  exit 1
fi
mkdir -p "$TARGET_ROOT/download"
hf download "$REPO_ID" \
  --repo-type dataset \
  --revision "$REVISION" \
  --include "$PREFIX/*" \
  --local-dir "$TARGET_ROOT/download"

SNAPSHOT_DIR="$TARGET_ROOT/download/$PREFIX"
printf '%s  %s\n' "$EXPECTED_SHA256" "$SNAPSHOT_DIR/$ARCHIVE" | sha256sum --check --strict
python3 "$(dirname "$0")/verify_final_experiment_records_snapshot.py" "$SNAPSHOT_DIR"

if [[ "$EXTRACT" == true ]]; then
  mkdir -p "$TARGET_ROOT/extracted"
  if find "$TARGET_ROOT/extracted" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to extract into non-empty directory: $TARGET_ROOT/extracted" >&2
    exit 1
  fi
  tar -xzf "$SNAPSHOT_DIR/$ARCHIVE" -C "$TARGET_ROOT/extracted"
fi

echo "Verified snapshot: $SNAPSHOT_DIR"
if [[ "$EXTRACT" == true ]]; then
  echo "Extracted records: $TARGET_ROOT/extracted/records"
fi
