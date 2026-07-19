#!/usr/bin/env python3
"""CUDA-independent audit helpers for the formal PathMMU GRPO wrapper."""

from __future__ import annotations

import json
import os
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


def aligned_solutions(solution: Any, completion_count: int) -> list[Any]:
    if solution is None:
        return [""] * completion_count
    return list(solution)[:completion_count]


def append_audit_events(
    reward_type: str,
    completions: list[Any],
    solutions: list[Any],
    rewards: list[float],
) -> None:
    """Write one JSONL per rank so distributed writes cannot interleave."""
    root = os.getenv("PATHVLM_REWARD_LOG_DIR")
    if not root:
        return
    if not (len(completions) == len(solutions) == len(rewards)):
        raise RuntimeError(
            f"reward audit length mismatch: {len(completions)}, {len(solutions)}, {len(rewards)}"
        )
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    call_index = _AUDIT_CALLS[reward_type]
    _AUDIT_CALLS[reward_type] += 1
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank_{rank:02d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for item_index, (completion, solution, reward) in enumerate(
            zip(completions, solutions, rewards)
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
            }
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()


def _parameter_count(parameter) -> int:
    return int(getattr(parameter, "ds_numel", parameter.numel()))


def trainability_report(model) -> dict[str, Any]:
    buckets = {
        "all": {"total": 0, "trainable": 0},
        "language": {"total": 0, "trainable": 0},
        "visual": {"total": 0, "trainable": 0},
        "multimodal_projector": {"total": 0, "trainable": 0},
    }
    for name, parameter in model.named_parameters():
        count = _parameter_count(parameter)
        trainable = count if parameter.requires_grad else 0
        buckets["all"]["total"] += count
        buckets["all"]["trainable"] += trainable
        bucket = "visual" if "visual" in name else "language"
        buckets[bucket]["total"] += count
        buckets[bucket]["trainable"] += trainable
        if "visual.merger" in name or ".merger." in name:
            buckets["multimodal_projector"]["total"] += count
            buckets["multimodal_projector"]["trainable"] += trainable
    report: dict[str, Any] = {"parameters": buckets}
    report["gates"] = {
        "language_nonempty": buckets["language"]["total"] > 0,
        "language_fully_trainable": (
            buckets["language"]["trainable"] == buckets["language"]["total"]
        ),
        "visual_nonempty": buckets["visual"]["total"] > 0,
        "visual_fully_frozen": buckets["visual"]["trainable"] == 0,
        "projector_nonempty": buckets["multimodal_projector"]["total"] > 0,
        "projector_fully_frozen": buckets["multimodal_projector"]["trainable"] == 0,
    }
    report["passed"] = all(report["gates"].values())
    return report
