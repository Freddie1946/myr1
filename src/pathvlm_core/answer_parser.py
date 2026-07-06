"""Answer parsing utilities for PathMMU-style <answer>...</answer> outputs."""

from __future__ import annotations

import re
from typing import Optional

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
SPACE_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Normalize answer text while preserving option wording."""
    return SPACE_RE.sub(" ", (text or "").strip())


def extract_answer_block(text: str) -> Optional[str]:
    """Return the last <answer> block content, or None if absent."""
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return None
    return normalize_answer(matches[-1])


def exact_answer_match(model_output: str, ground_truth: str) -> bool:
    """Compare extracted <answer> contents exactly after whitespace normalization."""
    pred = extract_answer_block(model_output)
    gold = extract_answer_block(ground_truth) or normalize_answer(ground_truth)
    return pred is not None and pred == gold


def has_think_answer_format(text: str) -> bool:
    """Check only structural <think>...</think><answer>...</answer> format."""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.IGNORECASE | re.DOTALL)
    return bool(pattern.fullmatch((text or "").strip()))
