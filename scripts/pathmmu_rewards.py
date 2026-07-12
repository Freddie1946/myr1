"""PathMMU online/offline reward functions shared by debug GRPO and evaluation."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
FORMAT_RE = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.IGNORECASE | re.DOTALL)


def answer_text(text: str) -> str:
    matches = ANSWER_RE.findall(str(text))
    return matches[-1].strip() if matches else str(text).strip()


def choice_letter(text: str) -> str | None:
    value = answer_text(text).upper().strip()
    # Prefer an explicit selected-value field. Do not treat keys such as
    # {"Option A": ..., "Option B": ...} as a prediction of A.
    patterns = (
        r'"(?:FINAL[_ ]?ANSWER|ANSWER|OPTION|CHOICE)"\s*:\s*"?\s*([A-D])(?:\b|\s*[\)])',
        r"\b(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:|=)\s*([A-D])\b",
        r"^\s*([A-D])(?:\s*[\)\].:]|\s*$)",
    )
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    # Accept a single unambiguous choice-like occurrence, but reject answers
    # that enumerate several options without selecting one.
    candidates = re.findall(r"\b([A-D])\s*[\)\].:]", value)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)


def _append_debug(event: dict[str, Any]) -> None:
    if os.getenv("DEBUG_MODE", "").lower() != "true":
        return
    log_path = os.getenv("LOG_PATH")
    if not log_path:
        return
    event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with Path(log_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def accuracy_reward(completions, solution, **kwargs):
    rewards = []
    for completion, target in zip(completions, solution):
        prediction = _completion_text(completion)
        predicted_choice = choice_letter(prediction)
        target_choice = choice_letter(str(target))
        if predicted_choice is not None and target_choice is not None:
            reward = float(predicted_choice == target_choice)
        else:
            reward = float(answer_text(prediction).casefold() == answer_text(str(target)).casefold())
        rewards.append(reward)
        _append_debug(
            {
                "reward_type": "accuracy",
                "reward": reward,
                "predicted_choice": predicted_choice,
                "target_choice": target_choice,
                "completion": prediction,
                "solution": str(target),
            }
        )
    return rewards


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        prediction = _completion_text(completion)
        reward = float(FORMAT_RE.fullmatch(prediction) is not None)
        rewards.append(reward)
        _append_debug({"reward_type": "format", "reward": reward, "completion": prediction})
    return rewards
