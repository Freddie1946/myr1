"""GPT-4o process reward interface adapted from evaluate_with_gpt4o_v2/ewg.

The training reward is intentionally narrow and paper-compatible:
process_reward = (integrity_score + knowledge_score) / 2.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from io import BytesIO
from typing import Any, Optional

from openai import OpenAI


def image_to_base64(image: Any) -> Optional[str]:
    """Accept a PIL image or a filesystem path and return base64 PNG/JPEG data."""
    if image is None:
        return None
    if isinstance(image, str):
        with open(image, "rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")
    if hasattr(image, "save"):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    return None


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def build_process_prompt(question: str, model_output: str, ground_truth: str, is_correct: bool) -> str:
    correctness_info = "The model's final answer is CORRECT." if is_correct else "The model's final answer is INCORRECT."
    return f"""
You are a medical image pathology expert. Evaluate the quality of this diagnostic thinking chain:

[Question]
{question}

[Model's Complete Output]
{model_output}

[Standard Answer]
{ground_truth}

[Answer Correctness]
{correctness_info}

Scoring criteria (each with a full score of 1 point):
1. Thinking chain integrity (deduct 0.4 points for each missing step, until 0):
   - Image feature analysis
   - Step-by-step option elimination (at least 2 options)
   - Medical knowledge citation
2. Medical knowledge rationality (deduct 0.4 points for each error, until 0):
   - Violation of basic histological definitions
   - Logical contradiction
   - Use of outdated or incorrect pathological criteria

Output format: JSON only, no markdown, no explanation:
{{"integrity_score": 0.0, "knowledge_score": 0.0}}
""".strip()


def parse_scores(raw: str) -> dict[str, float]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        if fence:
            parsed = json.loads(fence.group(1).strip())
        else:
            obj = re.search(r"\{[^}]*\"integrity_score\"[^}]*\"knowledge_score\"[^}]*\}", raw, re.DOTALL)
            if not obj:
                raise
            parsed = json.loads(obj.group(0))
    return {
        "integrity_score": max(0.0, min(1.0, float(parsed.get("integrity_score", 0.0)))),
        "knowledge_score": max(0.0, min(1.0, float(parsed.get("knowledge_score", 0.0)))),
    }


def call_gpt4o_process_reward(
    question: str,
    model_output: str,
    ground_truth: str,
    is_correct: bool,
    image: Any = None,
    max_retries: int = 3,
) -> dict[str, float]:
    client = get_client()
    content: list[dict[str, Any]] = [{"type": "text", "text": build_process_prompt(question, model_output, ground_truth, is_correct)}]
    encoded_image = image_to_base64(image)
    if encoded_image:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}})

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=os.environ.get("PATHVLM_GPT_REWARD_MODEL", "gpt-4o"),
                messages=[{"role": "user", "content": content}],
                temperature=float(os.environ.get("PATHVLM_GPT_REWARD_TEMPERATURE", "0.1")),
                max_tokens=int(os.environ.get("PATHVLM_GPT_REWARD_MAX_TOKENS", "200")),
            )
            return parse_scores(response.choices[0].message.content.strip())
        except Exception as exc:  # noqa: BLE001 - reward fallback should not crash training by default
            last_error = exc
            time.sleep(2 ** attempt)
    if os.environ.get("PATHVLM_STRICT_GPT_REWARD", "false").lower() == "true" and last_error:
        raise last_error
    return {"integrity_score": 0.0, "knowledge_score": 0.0}
