#!/usr/bin/env python3
"""Preflight a prepared formal-machine installation and emit a JSON report."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(command: list[str]) -> dict:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def env_versions(python: Path) -> dict:
    code = (
        "import json,torch,transformers,tokenizers,accelerate,datasets,deepspeed; "
        "d={'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,"
        "'tokenizers':tokenizers.__version__,'accelerate':accelerate.__version__,"
        "'datasets':datasets.__version__,'deepspeed':deepspeed.__version__}; "
        "import trl; d['trl']=trl.__version__; print(json.dumps(d))"
    )
    result = run([str(python), "-c", code])
    if result["returncode"]:
        raise RuntimeError(result)
    return json.loads(result["stdout"].strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-source-manifest", required=True, type=Path)
    parser.add_argument("--expected-model-id", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sft_python = args.install_root / "envs" / "sft" / "bin" / "python"
    grpo_python = args.install_root / "envs" / "grpo" / "bin" / "python"
    for path in (sft_python, grpo_python, args.model_dir / "config.json",
                 args.model_source_manifest, args.data_root / "formal_data_manifest.json"):
        if not path.exists():
            raise FileNotFoundError(path)

    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    model_source = json.loads(args.model_source_manifest.read_text(encoding="utf-8"))
    manifest = json.loads((args.data_root / "formal_data_manifest.json").read_text(encoding="utf-8"))
    model_shards = sorted(args.model_dir.glob("*.safetensors"))
    report = {
        "nvidia_smi": run(["nvidia-smi"]),
        "disk": run(["df", "-h", str(args.install_root)]),
        "sft_environment": env_versions(sft_python),
        "grpo_environment": env_versions(grpo_python),
        "model": {
            "path": str(args.model_dir.resolve()),
            "model_type": config.get("model_type"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "weight_shard_count": len(model_shards),
            "weight_bytes": sum(path.stat().st_size for path in model_shards),
            "source_manifest": model_source,
        },
        "data": manifest,
        "import_checks": {
            "llamafactory": run([str(sft_python), "-c", "import llamafactory; print('llamafactory import PASS')"]),
            "open_r1": run([str(grpo_python), "-c", "from open_r1.trainer import Qwen2VLGRPOTrainer; print('open_r1 import PASS')"]),
            "reward_tests": run([str(grpo_python), str(Path(__file__).resolve().parents[1] / "scripts/test_pathmmu_rewards.py")]),
        },
        "gates": {
            "model_is_qwen2_5_vl": config.get("model_type") == "qwen2_5_vl",
            "model_is_7b_shape": config.get("hidden_size") == 3584,
            "model_has_weights": bool(model_shards) and sum(path.stat().st_size for path in model_shards) > 10_000_000_000,
            "model_id_matches": model_source.get("model_id") == args.expected_model_id,
            "model_revision_matches": model_source.get("revision") == args.expected_revision,
            "no_image_overlap": all(v == 0 for v in manifest["pairwise_image_overlaps"].values()),
            "picked_json_unused": manifest.get("picked_json_used") is False,
        },
    }
    report["gates"]["llamafactory_imports"] = report["import_checks"]["llamafactory"]["returncode"] == 0
    report["gates"]["open_r1_imports"] = report["import_checks"]["open_r1"]["returncode"] == 0
    report["gates"]["reward_tests_pass"] = report["import_checks"]["reward_tests"]["returncode"] == 0
    report["passed"] = all(report["gates"].values()) and report["nvidia_smi"]["returncode"] == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
