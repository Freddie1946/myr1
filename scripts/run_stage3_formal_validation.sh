#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE="/home/dataset-assist-0/czy/wjy"
REPO="$WORKSPACE/myr1"
INSTALL="$WORKSPACE/pathvlm_r1_v1_a100"
PYTHON="$INSTALL/envs/sft/bin/python"
VALIDATE="$REPO/scripts/run_stage3_validation385_parallel.sh"
VERIFY_SNAPSHOT="$REPO/scripts/verify_checkpoint_snapshot.py"
SELECT="$REPO/scripts/select_stage3_validation_checkpoint.py"

: "${PATHVLM_STAGE3_COMPLETED_RUN_DIR:?Set the completed formal Stage3 run directory}"
RUN_DIR="$(readlink -m "$PATHVLM_STAGE3_COMPLETED_RUN_DIR")"
EXPECTED_PARENT="$INSTALL/runs/stage3_process_grpo"
case "$RUN_DIR" in
  "$EXPECTED_PARENT"/*) ;;
  *) echo "Stage3 run must be under $EXPECTED_PARENT" >&2; exit 2 ;;
esac

VALIDATION_ROOT="${PATHVLM_STAGE3_VALIDATION_ROOT:-$RUN_DIR/validation_0385_epochs}"
[[ -x "$PYTHON" && -f "$VALIDATE" && -f "$VERIFY_SNAPSHOT" && -f "$SELECT" ]] || exit 2
[[ ! -e "$VALIDATION_ROOT/selection.json" ]] || {
  echo "Refusing to overwrite a completed Stage3 selection: $VALIDATION_ROOT/selection.json" >&2
  exit 2
}

for step in 500 1000 1500; do
  epoch="$((step / 500))"
  model="$RUN_DIR/epoch_model_snapshots/checkpoint-$step"
  output="$VALIDATION_ROOT/checkpoint-$step"
  "$PYTHON" "$VERIFY_SNAPSHOT" "$model" \
    --expected-step "$step" --expected-epoch "$epoch" \
    --output "$VALIDATION_ROOT/checkpoint-${step}_snapshot_verification.json"
  [[ ! -e "$output" ]] || {
    echo "Refusing to reuse or overwrite validation output: $output" >&2
    exit 2
  }
  PATHVLM_STAGE3_VALIDATION_MODEL="$model" \
  PATHVLM_STAGE3_VALIDATION_RUN_ROOT="$output" \
    bash "$VALIDATE"
done

"$PYTHON" "$SELECT" \
  --run-dir "$RUN_DIR" \
  --validation-root "$VALIDATION_ROOT" \
  --output "$VALIDATION_ROOT/selection.json"

