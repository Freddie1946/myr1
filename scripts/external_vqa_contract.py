#!/usr/bin/env python3
"""Frozen prompt and deterministic scoring helpers for external VQA evaluation."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import string
import unicodedata
from collections import defaultdict
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
OMNIMEDVQA_DOMAIN_PROMPT = (
    "{question}\nOptions:\n{options}\n"
    "Give concise image-grounded medical reasoning inside <think>...</think>. Then output exactly "
    "one option letter inside <answer>...</answer>."
)
OMNIMEDVQA_NATIVE_CHOICE_PROMPT = (
    "{question}\nOptions:\n{options}\n"
    "Return only the single letter of the best answer: A, B, C, or D. "
    "Do not provide reasoning or any other text."
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


def _pathvqa_official_words(value: str) -> dict[str, int]:
    """Mirror PathVQA's published ``split_sentence(..., 1)`` tokenization."""

    words: dict[str, int] = defaultdict(int)
    for word in re.sub("[^a-zA-Z ]", "", str(value)).lower().strip().split():
        words[word] += 1
    return dict(words)


def pathvqa_official_token_overlap(completion: str, answer: str) -> float:
    """Mirror the repository's unfortunately named ``calculate_exactmatch`` code."""

    candidate = _pathvqa_official_words(completion)
    reference = _pathvqa_official_words(answer)
    total = sum(candidate.values())
    if not total:
        return 0.0
    # The official source counts each distinct reference word once when present.
    return sum(word in candidate for word in reference) / total


def pathvqa_official_token_f1(completion: str, answer: str) -> float:
    """Mirror the PathVQA repository's ``calculate_f1score`` implementation."""

    candidate = _pathvqa_official_words(completion)
    reference = _pathvqa_official_words(answer)
    if not candidate or not reference:
        return 0.0
    true_positive = false_positive = false_negative = 0
    for word in set(candidate) | set(reference):
        if word in candidate and word in reference:
            true_positive += candidate[word]
        elif word in candidate:
            false_positive += candidate[word]
        else:
            false_negative += reference[word]
    if not true_positive:
        return 0.0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2 * precision * recall / (precision + recall)


def pathvqa_sentence_bleu(completion: str, answer: str, maximum_order: int) -> float:
    """Deterministic sentence BLEU-N with uniform weights and no smoothing."""

    if maximum_order not in (1, 2, 3):
        raise ValueError("PathVQA BLEU order must be 1, 2, or 3")
    # Preserve token order; the counter helper above intentionally does not.
    candidate_tokens = re.sub("[^a-zA-Z ]", "", str(completion)).lower().split()
    reference_tokens = re.sub("[^a-zA-Z ]", "", str(answer)).lower().split()
    if not candidate_tokens or not reference_tokens:
        return 0.0
    precisions = []
    for order in range(1, maximum_order + 1):
        candidate_ngrams: dict[tuple[str, ...], int] = defaultdict(int)
        reference_ngrams: dict[tuple[str, ...], int] = defaultdict(int)
        for index in range(len(candidate_tokens) - order + 1):
            candidate_ngrams[tuple(candidate_tokens[index : index + order])] += 1
        for index in range(len(reference_tokens) - order + 1):
            reference_ngrams[tuple(reference_tokens[index : index + order])] += 1
        denominator = sum(candidate_ngrams.values())
        if not denominator:
            return 0.0
        clipped = sum(
            min(count, reference_ngrams.get(ngram, 0))
            for ngram, count in candidate_ngrams.items()
        )
        if not clipped:
            return 0.0
        precisions.append(clipped / denominator)
    brevity_penalty = (
        1.0
        if len(candidate_tokens) > len(reference_tokens)
        else math.exp(1.0 - len(reference_tokens) / len(candidate_tokens))
    )
    return brevity_penalty * math.exp(
        sum(math.log(value) for value in precisions) / maximum_order
    )


def extract_pathvqa_answer(completion: str, answer_type: str) -> tuple[str, str]:
    """Target-blind extraction for models that insist on a reasoning wrapper."""

    tagged = re.findall(
        r"<answer\b[^>]*>\s*(.*?)(?:</answer\s*>|$)", str(completion), re.I | re.S
    )
    if tagged:
        answer = tagged[-1].strip().rstrip("</ ")
        if answer_type == "yes_no":
            # Some PathMMU-tuned checkpoints preserve their learned multiple-choice
            # wrapper on PathVQA and emit e.g. ``<answer>B) No</answer>``.  Decode
            # only an unambiguous leading yes/no token; never search an explanation
            # for a target word.
            tagged_yes_no = re.match(
                r"^\s*(?:\(?[A-D]\)?\s*[).,:;\-]?\s*)?(yes|no)\b",
                answer,
                re.I,
            )
            if tagged_yes_no:
                return tagged_yes_no.group(1).lower(), "answer_tag_leading_yes_no"
        return answer, "answer_tag"
    if answer_type == "yes_no":
        marked = re.findall(
            r"(?:final\s+answer|answer)\s*(?:is|:)\s*\b(yes|no)\b",
            str(completion),
            re.I,
        )
        if marked:
            return marked[-1].lower(), "answer_marker"
        leading = re.match(r"^\s*(yes|no)\b", str(completion), re.I)
        if leading:
            return leading.group(1).lower(), "leading_yes_no"
    return str(completion).strip(), "raw_completion"


def pathvqa_score(completion: str, answer: str) -> dict[str, Any]:
    predicted = normalize_short_answer(completion)
    target = normalize_short_answer(answer)
    answer_type = pathvqa_answer_type(answer)
    extracted, extraction_source = extract_pathvqa_answer(completion, answer_type)
    return {
        "normalized_completion": predicted,
        "normalized_target": target,
        "answer_type": answer_type,
        # Retained for compatibility with frozen v1 prediction files.
        "exact_match": predicted == target,
        "strict_exact_match": predicted == target,
        "repository_token_overlap_score": pathvqa_official_token_overlap(completion, answer),
        "repository_token_f1_score": pathvqa_official_token_f1(completion, answer),
        "sentence_bleu_1": pathvqa_sentence_bleu(completion, answer, 1),
        "sentence_bleu_2": pathvqa_sentence_bleu(completion, answer, 2),
        "sentence_bleu_3": pathvqa_sentence_bleu(completion, answer, 3),
        # Compatibility aliases for the short-lived v2 draft scorer.
        "official_token_overlap_score": pathvqa_official_token_overlap(completion, answer),
        "official_token_f1_score": pathvqa_official_token_f1(completion, answer),
        "contract_aligned_answer": extracted,
        "contract_aligned_answer_source": extraction_source,
        "contract_aligned_exact_match": normalize_short_answer(extracted) == target,
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


def omnimed_domain_prompt(record: dict[str, Any]) -> str:
    """Build the corrected labeled-option OmniMedVQA generation prompt."""

    letters, texts = omnimed_options(record)
    options = "\n".join(
        f"{letter}) {text}" for letter, text in zip(letters, texts)
    )
    return OMNIMEDVQA_DOMAIN_PROMPT.format(
        question=record["question"], options=options
    )


def omnimed_native_choice_prompt(record: dict[str, Any]) -> str:
    """Build the labeled, no-example native choice adapter prompt."""

    letters, texts = omnimed_options(record)
    options = "\n".join(
        f"{letter}) {text}" for letter, text in zip(letters, texts)
    )
    return OMNIMEDVQA_NATIVE_CHOICE_PROMPT.format(
        question=record["question"], options=options
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


def extract_omnimed_answer(completion: str) -> tuple[str, str]:
    """Extract a target-blind final answer before applying the official matcher."""

    tagged = re.findall(
        r"<answer\b[^>]*>\s*(.*?)(?:</answer\s*>|$)", str(completion), re.I | re.S
    )
    if tagged:
        return tagged[-1].strip().rstrip("</ "), "answer_tag"
    markdown_marked = re.findall(
        r"^\s*\*{1,2}\s*(?:final\s+answer|answer)\s*:\s*\*{1,2}"
        r"\s*(.+?)(?:\n|$)",
        str(completion),
        re.I | re.M,
    )
    if markdown_marked:
        return markdown_marked[-1].strip(), "markdown_answer_marker"
    marked = re.findall(
        # Accept common Markdown wrappers such as ``*Answer*: D) CT`` while
        # keeping extraction target-blind and anchored to a dedicated line.
        # Do not interpret prose such as ``the answer is ...`` inside a
        # reasoning paragraph as a final-answer marker.
        r"^\s*(?:\*{0,2}\s*)?(?:final\s+answer|answer)(?:\s*\*{0,2})"
        r"\s*(?:is|:)\s*(.+?)(?:\n|$)",
        str(completion),
        re.I | re.M,
    )
    if marked:
        return marked[-1].strip(), "answer_marker"
    malformed_closing = re.search(
        r"</answer\s*>\s*([A-D])(?:\s*[).,:;\-].*)?\s*$",
        str(completion),
        re.I | re.S,
    )
    if malformed_closing:
        return malformed_closing.group(1).upper(), "malformed_closing_answer_tag"
    explicit_diagnosis = re.findall(
        r"(?:most\s+likely\s+(?:diagnosis|answer)|correct\s+answer|final\s+answer|answer)"
        r"[ \t]*(?:is[ \t]*:|is|:)[ \t]*(?:\r?\n[ \t]*)+"
        r"\(?([A-D])(?=[\s).,:;\-]|$)",
        str(completion),
        re.I,
    )
    if explicit_diagnosis:
        return explicit_diagnosis[-1].upper(), "explicit_diagnosis_block"
    option_correct = re.findall(
        r"^\s*Option\s+([A-D])\s*[:).,;\-].*\b(?:is\s+)?(?:the\s+)?correct\s+answer\b",
        str(completion),
        re.I | re.M,
    )
    if option_correct:
        return option_correct[-1].upper(), "explicit_option_correct_statement"
    explicit_choice_statement = re.findall(
        r"^\s*(?:The\s+)?(?:correct\s+answer|correct\s+option|selected\s+option\s+letter|"
        r"best\s+answer(?:\s+for\s+this\s+abnormality)?)\s*(?:is|:)\s*"
        r"\(?([A-D])(?=[\s).,:;\-]|$)",
        str(completion),
        re.I | re.M,
    )
    if explicit_choice_statement:
        return explicit_choice_statement[-1].upper(), "explicit_choice_statement"
    # Some instruction-tuned VLMs follow the semantic contract but omit the
    # XML wrapper, placing a single explicit choice on the final non-empty
    # line.  Extract only that final line; never search free-form reasoning for
    # an option letter.
    lines = [line.strip() for line in str(completion).splitlines() if line.strip()]
    if lines:
        bare_marker = re.match(
            r"^(?:final\s+answer|answer)\s*(?:is|:)?\s*([A-D])"
            r"(?=[\s).,:;\-]|$)",
            lines[-1],
            re.I,
        )
        if bare_marker:
            return bare_marker.group(1).upper(), "trailing_answer_marker_choice"
    if lines and re.match(r"^\(?\s*[A-D](?=[\s).,:;\-]|$)", lines[-1], re.I):
        return lines[-1], "trailing_explicit_choice_line"
    return str(completion).strip(), "raw_completion"


def strict_omnimed_final_choice(
    completion: str, letters: list[str], texts: list[str]
) -> tuple[str | None, str]:
    """Parse an explicit final option without fuzzy similarity or target access.

    A choice is available only when the extracted answer begins with an option
    letter or its first non-empty line exactly equals one candidate answer after
    conservative normalization.  This deliberately leaves truncated reasoning
    without a final answer unresolved rather than guessing from lexical overlap.
    """

    answer, answer_source = extract_omnimed_answer(completion)
    explicit_letter = re.match(
        r"^\s*\(?\s*([A-D])(?=[\s).,:;\-]|$)", answer, re.I
    )
    if explicit_letter and explicit_letter.group(1).upper() in letters:
        return (
            explicit_letter.group(1).upper(),
            f"{answer_source}:explicit_leading_choice_letter",
        )
    first_line = next((line.strip() for line in answer.splitlines() if line.strip()), "")
    exact_matches = [
        letter
        for letter, text in zip(letters, texts)
        if normalize_short_answer(first_line) == normalize_short_answer(text)
    ]
    if len(exact_matches) == 1:
        return exact_matches[0], f"{answer_source}:exact_first_line_option_text"
    return None, f"{answer_source}:unresolved"


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
    strict_final_predicted, strict_final_source = strict_omnimed_final_choice(
        completion, letters, texts
    )
    aligned_answer, aligned_source = extract_omnimed_answer(completion)
    explicit_letter = re.match(
        r"^\s*\(?\s*([A-D])(?=[\s).,:;\-]|$)", aligned_answer, re.I
    )
    if explicit_letter and explicit_letter.group(1).upper() in letters:
        aligned_predicted = explicit_letter.group(1).upper()
        aligned_similarities = None
        aligned_prediction_source = "explicit_leading_choice_letter"
    else:
        first_line = next(
            (line.strip() for line in aligned_answer.splitlines() if line.strip()), ""
        )
        leading_matches = [
            letter
            for letter, text in zip(letters, texts)
            if normalize_short_answer(first_line) == normalize_short_answer(text)
        ]
        if len(leading_matches) == 1:
            aligned_predicted = leading_matches[0]
            aligned_similarities = None
            aligned_prediction_source = "exact_leading_option_text"
        else:
            aligned_index, aligned_similarities_list = official_most_similar_option(
                aligned_answer, texts
            )
            aligned_predicted = letters[aligned_index] if aligned_index is not None else None
            aligned_similarities = dict(zip(letters, aligned_similarities_list))
            aligned_prediction_source = "official_option_similarity"
    return {
        "target_choice": target,
        "official_predicted_choice": predicted,
        "official_option_similarities": dict(zip(letters, similarities)),
        "official_most_similar_correct": predicted == target,
        "strict_text_predicted_choice": strict_predicted,
        "strict_text_correct": strict_predicted == target,
        "strict_final_predicted_choice": strict_final_predicted,
        "strict_final_prediction_source": strict_final_source,
        "strict_final_answer_available": strict_final_predicted is not None,
        "strict_final_correct": strict_final_predicted == target,
        "contract_aligned_answer": aligned_answer,
        "contract_aligned_answer_source": aligned_source,
        "contract_aligned_predicted_choice": aligned_predicted,
        "contract_aligned_prediction_source": aligned_prediction_source,
        "contract_aligned_option_similarities": aligned_similarities,
        "contract_aligned_correct": aligned_predicted == target,
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
