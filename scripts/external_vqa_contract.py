#!/usr/bin/env python3
"""Frozen prompt and deterministic scoring helpers for external VQA evaluation."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import string
import unicodedata
from pathlib import Path
from typing import Any


PATHVQA_PROMPT = (
    "{question}\n"
    "Answer the question using only a short answer. Do not include an explanation or any other "
    "text."
)
OMNIMEDVQA_PROMPT = (
    "{question}Here are {option_count} candidate answers:{options} Only return what you think is "
    "the correct answer from the candidate answers, do not return any other irrelevant text!"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_short_answer(value: str) -> str:
    """Conservative normalization for exact-answer scoring.

    It does not remove articles, stem words, expand abbreviations, or apply semantic aliases.
    """

    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.strip(string.whitespace + "\"'`")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(string.punctuation + " ")
    return text


def pathvqa_answer_type(answer: str) -> str:
    return "yes_no" if normalize_short_answer(answer) in {"yes", "no"} else "free_form"


def pathvqa_score(completion: str, answer: str) -> dict[str, Any]:
    predicted = normalize_short_answer(completion)
    target = normalize_short_answer(answer)
    return {
        "normalized_completion": predicted,
        "normalized_target": target,
        "answer_type": pathvqa_answer_type(answer),
        "exact_match": predicted == target,
    }


def omnimed_options(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    letters = [letter for letter in "ABCD" if record.get(f"option_{letter}") is not None]
    texts = [str(record[f"option_{letter}"]) for letter in letters]
    if len(letters) < 2:
        raise ValueError("OmniMedVQA record has fewer than two options")
    return letters, texts


def omnimed_target_choice(record: dict[str, Any]) -> str:
    letters, texts = omnimed_options(record)
    hits = [letter for letter, text in zip(letters, texts) if text == record["gt_answer"]]
    if len(hits) != 1:
        raise ValueError(f"expected one exact target option, got {hits}")
    return hits[0]


def omnimed_prompt(record: dict[str, Any]) -> str:
    _, texts = omnimed_options(record)
    return OMNIMEDVQA_PROMPT.format(
        question=record["question"],
        option_count=len(texts),
        options=str(texts),
    )


def official_most_similar_option(completion: str, texts: list[str]) -> tuple[int | None, list[float]]:
    """Mirror the official OmniMedVQA QA evaluator's SequenceMatcher rule."""

    similarities = [
        difflib.SequenceMatcher(None, text, str(completion)).ratio() for text in texts
    ]
    if not similarities or max(similarities) <= 0:
        return None, similarities
    # Python's max/index behavior preserves the first option on an exact tie, matching the source.
    return similarities.index(max(similarities)), similarities


def omnimed_score(completion: str, record: dict[str, Any]) -> dict[str, Any]:
    letters, texts = omnimed_options(record)
    target = omnimed_target_choice(record)
    best_index, similarities = official_most_similar_option(completion, texts)
    predicted = letters[best_index] if best_index is not None else None
    normalized_completion = normalize_short_answer(completion)
    strict_matches = [
        letter
        for letter, text in zip(letters, texts)
        if normalize_short_answer(text) == normalized_completion
    ]
    strict_predicted = strict_matches[0] if len(strict_matches) == 1 else None
    return {
        "target_choice": target,
        "official_predicted_choice": predicted,
        "official_option_similarities": dict(zip(letters, similarities)),
        "official_most_similar_correct": predicted == target,
        "strict_text_predicted_choice": strict_predicted,
        "strict_text_correct": strict_predicted == target,
        "normalized_completion": normalized_completion,
    }


def prompt_for_record(task: str, record: dict[str, Any]) -> str:
    if task == "pathvqa":
        return PATHVQA_PROMPT.format(question=record["question"])
    if task == "omnimedvqa":
        return omnimed_prompt(record)
    raise ValueError(f"unsupported task: {task}")


def score_record(task: str, completion: str, record: dict[str, Any]) -> dict[str, Any]:
    if task == "pathvqa":
        return pathvqa_score(completion, record["answer"])
    if task == "omnimedvqa":
        return omnimed_score(completion, record)
    raise ValueError(f"unsupported task: {task}")
