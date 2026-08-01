#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE="/home/dataset-assist-0/czy/wjy"
REPO="$WORKSPACE/myr1"
INSTALL="$WORKSPACE/pathvlm_r1_v1_a100"
PYTHON="$INSTALL/envs/sft/bin/python"
MODEL="${PATHVLM_STAGE3_VALIDATION_MODEL:-$INSTALL/runs/stage3_process_grpo/kimi26_50step_seed42_20260730/output_attempt03}"
DATA="$INSTALL/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json"
CHAT_TEMPLATE="$MODEL/chat_template.json"
RUN_ROOT="${PATHVLM_STAGE3_VALIDATION_RUN_ROOT:-$INSTALL/runs/stage3_process_grpo/kimi26_50step_seed42_20260730/validation_0385_postpilot_parallel_20260730_101722}"
RUNNER="$REPO/scripts/infer_and_score_pathmmu_with_tokens.py"
EXPECTED_DATA_SHA256="f52c9a412da579db8fb8bf52f6fa320e7c4ae9cb42b6cf1bdf62f77ab158dcf0"

[[ -x "$PYTHON" && -f "$RUNNER" ]] || {
  echo "validation Python or runner is missing" >&2
  exit 2
}
[[ -f "$MODEL/model.safetensors.index.json" && -f "$CHAT_TEMPLATE" ]] || {
  echo "Stage3 final model is incomplete" >&2
  exit 2
}
[[ "$(sha256sum "$DATA" | awk '{print $1}')" == "$EXPECTED_DATA_SHA256" ]] || {
  echo "frozen validation data hash mismatch" >&2
  exit 2
}
[[ ! -e "$RUN_ROOT" ]] || {
  echo "refusing to overwrite validation run: $RUN_ROOT" >&2
  exit 2
}

mapfile -t GPU_MEMORY < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
[[ "${#GPU_MEMORY[@]}" -eq 8 ]] || {
  echo "exactly eight GPUs are required" >&2
  exit 2
}
for index in "${!GPU_MEMORY[@]}"; do
  used="${GPU_MEMORY[$index]//[[:space:]]/}"
  (( used <= 10 )) || {
    echo "GPU $index is not idle: ${used} MiB" >&2
    exit 2
  }
done

mkdir -p "$RUN_ROOT/inputs" "$RUN_ROOT/shards"
"$PYTHON" - "$DATA" "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
records = json.loads(data_path.read_text(encoding="utf-8"))
if len(records) != 385:
    raise SystemExit(f"expected 385 validation records, found {len(records)}")
shards = []
for shard_id in range(8):
    start = len(records) * shard_id // 8
    end = len(records) * (shard_id + 1) // 8
    path = run_root / "inputs" / f"shard_{shard_id:02d}.json"
    path.write_text(
        json.dumps(records[start:end], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shards.append(
        {
            "shard_id": shard_id,
            "gpu": shard_id,
            "start": start,
            "end": end,
            "count": end - start,
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
(run_root / "shards.json").write_text(
    json.dumps(shards, indent=2) + "\n", encoding="utf-8"
)
PY

export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export WANDB_MODE="disabled"
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="$REPO/scripts"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  shard="$(printf '%02d' "$gpu")"
  output="$RUN_ROOT/shards/shard_$shard"
  log="$RUN_ROOT/shard_$shard.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" "$RUNNER" \
      --model "$MODEL" \
      --data "$RUN_ROOT/inputs/shard_$shard.json" \
      --output-dir "$output" \
      --max-new-tokens 1024 \
      --chat-template-file "$CHAT_TEMPLATE"
  ) >"$log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "validation shard $index failed; see $RUN_ROOT/shard_$(printf '%02d' "$index").log" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 1

"$PYTHON" - "$DATA" "$MODEL" "$CHAT_TEMPLATE" "$RUN_ROOT" <<'PY'
import hashlib
import json
import statistics
import sys
from pathlib import Path

from pathmmu_rewards import accuracy_reward, choice_letter, format_reward

data_path = Path(sys.argv[1]).resolve()
model = Path(sys.argv[2]).resolve()
chat_template = Path(sys.argv[3]).resolve()
run_root = Path(sys.argv[4]).resolve()
source = json.loads(data_path.read_text(encoding="utf-8"))
shards = json.loads((run_root / "shards.json").read_text(encoding="utf-8"))
merged = []
for shard in shards:
    shard_id = int(shard["shard_id"])
    start = int(shard["start"])
    end = int(shard["end"])
    predictions = run_root / "shards" / f"shard_{shard_id:02d}" / "predictions.jsonl"
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    if len(rows) != end - start:
        raise SystemExit(f"shard {shard_id} prediction count mismatch")
    for local_index, row in enumerate(rows):
        global_index = start + local_index
        expected = source[global_index]
        if row.get("index") != local_index:
            raise SystemExit(f"shard {shard_id} local index mismatch")
        for key in ("image", "problem", "solution"):
            if row.get(key) != expected.get(key):
                raise SystemExit(f"shard {shard_id} source mismatch at {global_index}: {key}")
        wrapped = [[{"role": "assistant", "content": row["completion"]}]]
        accuracy = accuracy_reward(wrapped, [row["solution"]])[0]
        formatting = format_reward(wrapped)[0]
        if accuracy != row["accuracy_reward"] or formatting != row["format_reward"]:
            raise SystemExit(f"offline reward mismatch at global index {global_index}")
        if choice_letter(row["completion"]) != row["predicted_choice"]:
            raise SystemExit(f"offline choice mismatch at global index {global_index}")
        row["index"] = global_index
        row["shard_id"] = shard_id
        row["shard_local_index"] = local_index
        merged.append(row)

if len(merged) != 385 or [row["index"] for row in merged] != list(range(385)):
    raise SystemExit("merged validation index/count gate failed")
predictions_path = run_root / "predictions.jsonl"
with predictions_path.open("w", encoding="utf-8") as handle:
    for row in merged:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

lengths = [int(row["generated_token_count"]) for row in merged]
metrics = {
    "schema_version": 1,
    "status": "completed",
    "split_role": "validation_0385",
    "test_accessed": False,
    "count": len(merged),
    "correct": sum(int(row["accuracy_reward"]) for row in merged),
    "accuracy": statistics.fmean(float(row["accuracy_reward"]) for row in merged),
    "format_correct": sum(int(row["format_reward"]) for row in merged),
    "format_accuracy": statistics.fmean(float(row["format_reward"]) for row in merged),
    "choice_extracted": sum(row["predicted_choice"] is not None for row in merged),
    "empty_completion_count": sum(not row["completion"] for row in merged),
    "mean_generated_tokens": statistics.fmean(lengths),
    "median_generated_tokens": statistics.median(lengths),
    "maximum_generated_tokens": max(lengths),
    "eos_terminated_count": sum(bool(row["ended_with_eos"]) for row in merged),
    "generation_cap_hit_count": sum(bool(row["reached_generation_cap"]) for row in merged),
    "max_new_tokens": 1024,
    "do_sample": False,
    "dtype": "bfloat16",
    "quantization": "none",
    "parallel_scheduling_only": True,
    "per_gpu_batch_size": 1,
    "model_path": str(model),
    "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
    "data_path": str(data_path),
    "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    "chat_template_file": str(chat_template),
    "chat_template_sha256": hashlib.sha256(chat_template.read_bytes()).hexdigest(),
    "predictions_file": str(predictions_path),
    "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    "shards": shards,
}
(run_root / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
PY
