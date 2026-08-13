#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/dataset-assist-0/czy/wjy/myr1
python_bin=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python
eval_root=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100
run_root="$eval_root/runs/stage3_gpt4o_n8_checkpoint1500_final_eval_20260813"
annotation_root="$eval_root/runs/predefined_evidence_annotation_aigcbest_v2_20260813"
annotation_summary="$annotation_root/case_annotations.summary.json"
panel="$repo_root/protocol/predefined_evidence_selected_panel_v2_20260813.json"
model=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_n8_parent_full_20260813/epoch_model_snapshots/checkpoint-1500
stage3_pathvqa_gate="$eval_root/runs/stage3_gpt4o_n8_checkpoint_ood_comparison_20260813/pathvqa_test.done"

annotation_complete() {
  "$python_bin" - "$annotation_summary" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("annotation_count") == 160 else 1)
PY
}

until annotation_complete; do
  sleep 30
done

"$python_bin" "$repo_root/scripts/freeze_annotated_visual_evidence_panel_v2.py" \
  --panel "$repo_root/protocol/predefined_evidence_candidate_panel_v2_160case_20260813.json" \
  --annotations "$annotation_root/case_annotations.jsonl" \
  --output "$panel"

export PYTHONPATH="$repo_root/scripts"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=6 "$python_bin" "$repo_root/scripts/run_reference_evidence_behavior.py" \
  --model "$model" \
  --panel "$panel" \
  --output-dir "$run_root/reference_evidence_behavior_v2" \
  > "$eval_root/reference_evidence_behavior_v2.log" 2>&1 &
behavior_pid=$!

until [[ -f "$stage3_pathvqa_gate" ]]; do
  sleep 30
done

CUDA_VISIBLE_DEVICES=7 "$python_bin" "$repo_root/scripts/run_reference_evidence_dual_stream_patching.py" \
  --model "$model" \
  --panel "$panel" \
  --output-dir "$run_root/reference_evidence_dual_stream_v2" \
  > "$eval_root/reference_evidence_dual_stream_v2.log" 2>&1 &
dual_pid=$!

wait "$behavior_pid"
wait "$dual_pid"

"$python_bin" "$repo_root/scripts/summarize_reference_evidence_causal_results.py" \
  --behavior-dir "$run_root/reference_evidence_behavior_v2" \
  --dual-stream-dir "$run_root/reference_evidence_dual_stream_v2" \
  --output-root "$run_root/reference_evidence_summary_v2"

printf 'complete\n' > "$run_root/reference_evidence_v2.done"
