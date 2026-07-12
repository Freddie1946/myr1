#!/usr/bin/env python3
"""Render machine-resolved SFT and Outcome-GRPO configs without launching training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def sft_yaml(*, model: Path, lf_data: Path, ds: Path, output: Path, dataset: str,
             max_samples: int, seed: int, smoke: bool) -> str:
    schedule = (
        "max_steps: 1\nnum_train_epochs: 1\nlr_scheduler_type: constant\nwarmup_ratio: 0.0"
        if smoke
        else "num_train_epochs: 10\nlr_scheduler_type: cosine\nwarmup_ratio: 0.03"
    )
    save = 'save_strategy: "no"\nsave_only_model: true' if smoke else "save_strategy: steps\nsave_steps: 100\nsave_only_model: false"
    return f"""### Generated formal-machine config. Review before launch.
### model
model_name_or_path: {model}
trust_remote_code: true
image_max_pixels: 65536
video_max_pixels: 8192
flash_attn: sdpa
use_cache: false

### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
deepspeed: {ds}

### dataset
dataset_dir: {lf_data}
dataset: {dataset}
template: qwen2_vl
cutoff_len: 512
max_samples: {max_samples}
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {output}
logging_steps: 1
{save}
plot_loss: true
overwrite_output_dir: false
report_to: none

### training
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 2.0e-5
{schedule}
bf16: true
gradient_checkpointing: true
seed: {seed}
data_seed: {seed}
ddp_timeout: 180000000
"""


def grpo_script(*, python: Path, repo: Path, open_r1: Path, model_placeholder: str,
                data_yaml: Path, output: Path, run_dir: Path, seed: int, nproc: int,
                port: int, cuda_visible: str, per_device_batch: int, max_steps: int | None) -> str:
    max_steps_line = f"cmd+=(--max_steps {max_steps})" if max_steps is not None else ""
    return f"""#!/usr/bin/env bash
set -euo pipefail
: "${{MODEL_PATH:={model_placeholder}}}"
if [[ "$MODEL_PATH" == "__SET_TO_NEW_FORMAL_SFT_CHECKPOINT__" ]]; then
  echo "Set MODEL_PATH to a newly trained formal SFT checkpoint" >&2; exit 2
fi
mkdir -p "{run_dir}"
export CUDA_VISIBLE_DEVICES="{cuda_visible}"
export NCCL_P2P_DISABLE="${{NCCL_P2P_DISABLE:-1}}"
export NCCL_IB_DISABLE="${{NCCL_IB_DISABLE:-1}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export WANDB_MODE="${{WANDB_MODE:-disabled}}"
export DEBUG_MODE=true
export LOG_PATH="{run_dir}/online_rewards.jsonl"
cd "{open_r1}"
cmd=(
  "{python}" -m torch.distributed.run --nproc_per_node={nproc} --master_port={port}
  "{repo}/scripts/grpo_pathmmu.py"
  --deepspeed "{repo}/configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json"
  --output_dir "{output}"
  --model_name_or_path "$MODEL_PATH"
  --dataset_name "{data_yaml}" --image_root /
  --reward_funcs accuracy format --freeze_vision_modules true
  --max_pixels 65536 --min_pixels 3136
  --num_generations 4 --max_completion_length 192
  --per_device_train_batch_size {per_device_batch} --gradient_accumulation_steps 1
  --learning_rate 1.0e-6 --logging_steps 1 --bf16 true --torch_dtype bfloat16
  --gradient_checkpointing true --attn_implementation sdpa
  --beta 0.04 --num_iterations 1 --save_steps 100 --save_only_model false
  --report_to none --seed {seed} --data_seed {seed}
)
{max_steps_line}
"${{cmd[@]}}" 2>&1 | tee "{run_dir}/train.log"
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--install-root", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--sft-python", required=True, type=Path)
    p.add_argument("--grpo-python", required=True, type=Path)
    p.add_argument("--llamafactory-src", required=True, type=Path)
    p.add_argument("--open-r1-src", required=True, type=Path)
    p.add_argument("--cuda-visible-devices", required=True)
    p.add_argument("--nproc-per-node", required=True, type=int)
    p.add_argument("--master-port-base", required=True, type=int)
    p.add_argument("--grpo-per-device-batch", required=True, type=int)
    args = p.parse_args()

    if (args.nproc_per_node * args.grpo_per_device_batch) % 4:
        raise ValueError("NPROC_PER_NODE * GRPO_PER_DEVICE_BATCH must be divisible by num_generations=4")
    if len(args.cuda_visible_devices.split(",")) != args.nproc_per_node:
        raise ValueError("CUDA_VISIBLE_DEVICES count must equal NPROC_PER_NODE")

    generated = args.install_root / "generated_configs"
    sft_dir, grpo_dir = generated / "sft", generated / "outcome_grpo"
    sft_dir.mkdir(parents=True, exist_ok=True)
    grpo_dir.mkdir(parents=True, exist_ok=True)
    runs = args.install_root / "runs"
    ds = args.repo_root / "configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json"
    lf_data = args.data_root / "llamafactory"
    grpo_data = args.data_root / "grpo"

    specs = [("smoke_n0008", "pathvlm_sft_smoke_n0008", 8, 42, True)]
    specs += [(f"n{size:04d}_seed0042", f"pathvlm_sft_n{size:04d}", size, 42, False)
              for size in (500, 1000, 2000, 3000)]
    specs += [(f"n3000_seed{seed:04d}", "pathvlm_sft_n3000", 3000, seed, False)
              for seed in (43, 44)]
    for name, dataset, count, seed, smoke in specs:
        path = sft_dir / f"sft_{name}.yaml"
        path.write_text(sft_yaml(model=args.model_dir, lf_data=lf_data, ds=ds,
                                 output=runs / "stage1_sft" / name / "output",
                                 dataset=dataset, max_samples=count, seed=seed, smoke=smoke), encoding="utf-8")

    grpo_specs = [("smoke_n0008_seed0042", "pathvlm_rl_smoke_n0008", 42, 1)]
    grpo_specs += [(f"n{size:04d}_seed0042", f"pathvlm_rl_n{size:04d}", 42, None)
                   for size in (250, 500, 1000)]
    grpo_specs += [(f"n1000_seed{seed:04d}", "pathvlm_rl_n1000", seed, None) for seed in (43, 44)]
    for offset, (name, dataset, seed, max_steps) in enumerate(grpo_specs):
        run_dir = runs / "stage2_outcome_grpo" / name
        script = grpo_dir / f"grpo_{name}.sh"
        script.write_text(grpo_script(
            python=args.grpo_python, repo=args.repo_root, open_r1=args.open_r1_src,
            model_placeholder="__SET_TO_NEW_FORMAL_SFT_CHECKPOINT__",
            data_yaml=grpo_data / f"{dataset}.yaml", output=run_dir / "output",
            run_dir=run_dir, seed=seed, nproc=args.nproc_per_node,
            port=args.master_port_base + offset, cuda_visible=args.cuda_visible_devices,
            per_device_batch=args.grpo_per_device_batch, max_steps=max_steps), encoding="utf-8")
        script.chmod(0o755)

    manifest = {
        "generated_root": str(generated),
        "sft_configs": sorted(str(p) for p in sft_dir.glob("*.yaml")),
        "grpo_scripts": sorted(str(p) for p in grpo_dir.glob("*.sh")),
        "warning": "Review configs and complete one-step gates before any full run.",
    }
    (generated / "generated_config_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
