#!/usr/bin/env python3
"""CUDA-independent audit helpers for the formal PathMMU GRPO wrapper."""

from __future__ import annotations

import json
import os
import hashlib
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_AUDIT_CALLS: defaultdict[str, int] = defaultdict(int)

STRICT_PROMPT_CONTRACT = "pathmmu_think_answer_only_v2"
STRICT_PROMPT_SUFFIX = (
    " First output the thinking process in <think> </think> tags and then output the final answer "
    "in <answer> </answer> tags."
)
LEGACY_JSON_SUFFIX = " Output the final answer in JSON format."


def strict_prompt_text(question: str) -> str:
    """Return the exact prompt contract scored by the strict format reward."""
    return str(question) + STRICT_PROMPT_SUFFIX


def replace_legacy_json_prompt(original: str, question: str) -> str:
    """Fail closed unless the vendored prompt has exactly the audited legacy suffix."""
    if not str(original).endswith(LEGACY_JSON_SUFFIX):
        raise RuntimeError("vendored prompt no longer has the expected legacy JSON suffix")
    corrected = strict_prompt_text(question)
    if "JSON format" in corrected or not corrected.endswith("<answer> </answer> tags."):
        raise RuntimeError("strict PathMMU prompt contract construction failed")
    return corrected


def completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)


def aligned_solutions(
    solution: Any, completion_count: int, *, require_exact: bool = False
) -> list[Any]:
    if solution is None:
        if require_exact:
            raise RuntimeError("formal reward solutions are missing")
        return [""] * completion_count
    values = list(solution)
    if require_exact and len(values) != completion_count:
        raise RuntimeError(
            "formal reward solution length mismatch: "
            f"{len(values)} != {completion_count}"
        )
    aligned = values[:completion_count]
    if len(aligned) != completion_count:
        raise RuntimeError(
            "reward solution length mismatch: "
            f"{len(aligned)} != {completion_count}"
        )
    return aligned


def append_audit_events(
    reward_type: str,
    completions: list[Any],
    solutions: list[Any],
    rewards: list[float],
    metadata: list[dict[str, Any]] | None = None,
) -> None:
    """Write one JSONL per rank so distributed writes cannot interleave."""
    root = os.getenv("PATHVLM_REWARD_LOG_DIR")
    if not root:
        return
    if metadata is None:
        metadata = [{} for _ in completions]
    if not (len(completions) == len(solutions) == len(rewards) == len(metadata)):
        raise RuntimeError(
            "reward audit length mismatch: "
            f"{len(completions)}, {len(solutions)}, {len(rewards)}, {len(metadata)}"
        )
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    call_index = _AUDIT_CALLS[reward_type]
    _AUDIT_CALLS[reward_type] += 1
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank_{rank:02d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for item_index, (completion, solution, reward, source) in enumerate(
            zip(completions, solutions, rewards, metadata)
        ):
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rank": rank,
                "local_rank": local_rank,
                "pid": os.getpid(),
                "reward_type": reward_type,
                "call_index": call_index,
                "item_index": item_index,
                "reward": float(reward),
                "completion": completion_text(completion),
                "solution": str(solution),
                "training_segment": os.getenv("PATHVLM_TRAINING_SEGMENT", "unspecified"),
                **source,
            }
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()


MODEL_SNAPSHOT_PATTERNS = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
    "vocab.json",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_model_snapshot_file(name: str) -> bool:
    return name in MODEL_SNAPSHOT_PATTERNS or (
        name.startswith("model-") and name.endswith(".safetensors")
    )


def create_model_only_snapshot(
    checkpoint: Path, destination: Path, *, global_step: int, epoch: float
) -> dict[str, Any]:
    """Atomically retain a loadable model without optimizer/resume state."""
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise RuntimeError(f"checkpoint is missing: {checkpoint}")
    if destination.exists():
        raise RuntimeError(f"snapshot destination already exists: {destination}")
    required = {"config.json", "preprocessor_config.json", "tokenizer_config.json"}
    selected = sorted(
        path for path in checkpoint.iterdir()
        if path.is_file() and _is_model_snapshot_file(path.name)
    )
    names = {path.name for path in selected}
    if not required.issubset(names):
        raise RuntimeError(f"checkpoint model metadata is incomplete: {sorted(required - names)}")
    if not ({"model.safetensors"} <= names or "model.safetensors.index.json" in names):
        raise RuntimeError("checkpoint has no gathered model weights")
    if any(path.stat().st_size <= 0 for path in selected):
        raise RuntimeError("checkpoint contains an empty model/processor file")

    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        entries = []
        for source in selected:
            target = temporary / source.name
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            entries.append({
                "name": source.name,
                "size_bytes": target.stat().st_size,
                "sha256": file_sha256(target),
            })
        manifest = {
            "schema_version": 1,
            "global_step": int(global_step),
            "epoch": float(epoch),
            "model_only": True,
            "resumable": False,
            "source_checkpoint": str(checkpoint),
            "files": entries,
        }
        manifest_path = temporary / "snapshot_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parameter_count(parameter) -> int:
    return int(getattr(parameter, "ds_numel", parameter.numel()))


def trainability_report(model, language_mode: str = "full") -> dict[str, Any]:
    if language_mode not in {"full", "lora"}:
        raise ValueError(f"invalid language mode: {language_mode}")
    buckets = {
        "all": {"total": 0, "trainable": 0},
        "language": {"total": 0, "trainable": 0},
        "visual": {"total": 0, "trainable": 0},
        "multimodal_projector": {"total": 0, "trainable": 0},
    }
    trainable_language_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = _parameter_count(parameter)
        trainable = count if parameter.requires_grad else 0
        buckets["all"]["total"] += count
        buckets["all"]["trainable"] += trainable
        bucket = "visual" if "visual" in name else "language"
        buckets[bucket]["total"] += count
        buckets[bucket]["trainable"] += trainable
        if bucket == "language" and parameter.requires_grad:
            trainable_language_names.append(name)
        if "visual.merger" in name or ".merger." in name:
            buckets["multimodal_projector"]["total"] += count
            buckets["multimodal_projector"]["trainable"] += trainable
    report: dict[str, Any] = {
        "parameters": buckets,
        "language_mode": language_mode,
        "trainable_language_parameter_names": trainable_language_names,
    }
    language_gate = (
        buckets["language"]["trainable"] == buckets["language"]["total"]
        if language_mode == "full"
        else (
            buckets["language"]["trainable"] > 0
            and all("lora_" in name for name in trainable_language_names)
        )
    )
    report["gates"] = {
        "language_nonempty": buckets["language"]["total"] > 0,
        "language_policy_satisfied": language_gate,
        "visual_nonempty": buckets["visual"]["total"] > 0,
        "visual_fully_frozen": buckets["visual"]["trainable"] == 0,
        "projector_nonempty": buckets["multimodal_projector"]["total"] > 0,
        "projector_fully_frozen": buckets["multimodal_projector"]["trainable"] == 0,
    }
    report["passed"] = all(report["gates"].values())
    return report
