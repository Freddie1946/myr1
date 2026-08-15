#!/usr/bin/env python3
"""Generate resumable English/Chinese medical translations for expert packets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORK = Path("/home/dataset-assist-0/czy/wjy")
API_URL = "https://api2.aigcbest.top/v1/chat/completions"
MODEL = "claude-sonnet-4-6"
PROMPT_VERSION = "expert_packet_medical_en_zh_translation_v1"
SYSTEM_PROMPT = """You are a meticulous bilingual medical editor specializing in pathology.
Translate every English string value in the supplied JSON object into clear Simplified Chinese.
Do not summarize, omit, correct, or reinterpret the source. Preserve option labels A)/B)/C)/D),
numbers, units, gene and marker names, and the literal XML tags <think>, </think>, <answer>,
</answer>. Keep established English abbreviations in parentheses when useful. This translation is
a reading aid; fidelity to the English source is more important than stylistic elegance.

Do NOT return JSON, because medical quotations can break JSON escaping. For every supplied key,
return exactly one plain-text block in the supplied key order:
<<<TRANSLATION:key>>>
translated text
<<<END:key>>>
Do not use Markdown fences and do not write text outside these blocks."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def judge_map(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    result = {}
    for row in load_jsonl(path):
        if int(row["run"]) == 0 and row["candidate_label"] in {"Stage2-outcome-GRPO", "Stage3-GPT4o-selected"}:
            result[(int(row["index"]), row["candidate_label"])] = row["scores"]
    return result


def source_records() -> dict[str, list[dict[str, Any]]]:
    blinded = WORK / "backup_archives/human_review_distribution_20260815_v2/reviewer_blinded"
    reward_cases = load_jsonl(WORK / "pathvlm_revision_eval_a100/human_review/reward_agreement_60case_20260814/blinded_packet/cases.jsonl")
    reward_refs = {row["case_id"]: row for row in load_jsonl(WORK / "pathvlm_revision_eval_a100/human_review/external_reference_reward_review_20260815/references.jsonl")}
    reward = [{
        "case_id": row["case_id"],
        "texts": {
            "question_and_options": row["question_and_options"],
            "dataset_reference_answer": row["dataset_reference_answer"],
            "candidate_completion": row["candidate_completion"],
            "external_overall_reason": reward_refs[row["case_id"]]["reference"]["overall_reason"],
        },
    } for row in reward_cases]

    effective = json.loads((WORK / "myr1/protocol/effective_interpretability_panel_20case_20260814.json").read_text(encoding="utf-8"))
    effective_by_panel = {int(row["panel_index"]): row for row in effective["cases"]}
    mapping = json.loads((WORK / "pathvlm_revision_eval_a100/human_review/expert_review_browse_packets_20260815/INTERNAL_DO_NOT_SEND_TO_REVIEWERS/source_mapping.json").read_text(encoding="utf-8"))["roi_cases"]
    gemini = {int(row["panel_index"]): row["annotation"] for row in load_jsonl(WORK / "pathvlm_revision_eval_a100/runs/exhaustive_reference_annotation_gemini31pro_formal_20260814/all160_validated.jsonl")}
    opus = {int(row["panel_index"]): row["annotation"] for row in load_jsonl(WORK / "pathvlm_revision_eval_a100/runs/dual_model_roi_formal_20260814/opus_annotations/all120_validated.jsonl")}
    roi = []
    for item in mapping:
        case_id = item["case_id"]
        panel = int(item["source_panel_index"])
        case = effective_by_panel[panel]
        g, o = gemini[panel], opus[panel]
        texts = {
            "question_and_options": case["problem"],
            "dataset_reference_answer": case["solution"],
            "gemini_rationale": str(g.get("rationale", "")),
            "opus_rationale": str(o.get("rationale", "")),
        }
        for index, region in enumerate(case["external_annotation"]["regions"], 1):
            texts[f"consensus_region_{index:02d}_feature"] = str(region.get("feature", ""))
        for prefix, annotation in (("gemini", g), ("opus", o)):
            for index, region in enumerate(annotation.get("regions", []), 1):
                texts[f"{prefix}_region_{index:02d}_feature"] = str(region.get("feature", ""))
        roi.append({"case_id": case_id, "texts": texts})

    answer_key_path = WORK / "pathvlm_revision_eval_a100/runs/stage2_stage3_human_review_packet_20260815/blind_pairwise_multijudge_panel100/adjudication_key_keep_separate/answer_key.csv"
    with answer_key_path.open(encoding="utf-8-sig") as handle:
        answer_keys = {row["case_id"]: row for row in csv.DictReader(handle)}
    claude = judge_map(WORK / "pathvlm_revision_eval_a100/runs/multijudge_reasoning_eval_20260810/formal_claude/judgments.jsonl")
    gemini_judge = judge_map(WORK / "pathvlm_revision_eval_a100/runs/multijudge_reasoning_eval_20260810/formal_gemini/judgments.jsonl")
    generation = []
    for case_number in range(1, 101):
        case_id = f"case_{case_number:03d}"
        case = json.loads((blinded / "stage2_stage3_blind_pairwise_100" / case_id / "case.json").read_text(encoding="utf-8"))
        key = answer_keys[case_id]
        source_index = int(key["source_index"])
        texts = {
            "question_and_options": case["problem"],
            "dataset_reference_answer": case["reference_reasoning"],
            "response_a": case["response_a"],
            "response_b": case["response_b"],
        }
        for anonymous in ("a", "b"):
            model = key[f"response_{anonymous}_model"]
            texts[f"claude_response_{anonymous}_reason"] = claude[(source_index, model)]["overall_reason"]
            texts[f"gemini_response_{anonymous}_reason"] = gemini_judge[(source_index, model)]["overall_reason"]
        generation.append({"case_id": case_id, "texts": texts})
    return {"reward": reward, "roi": roi, "generation": generation}


def parse_translation(content: str, source: dict[str, str]) -> dict[str, str]:
    translated = {}
    for key in source:
        pattern = re.compile(
            rf"<<<TRANSLATION:{re.escape(key)}>>>\s*\n?(.*?)\n?\s*<<<END:{re.escape(key)}>>>",
            re.DOTALL,
        )
        matches = pattern.findall(content)
        if len(matches) != 1:
            raise ValueError(f"expected one translation block for {key}, got {len(matches)}")
        # Chinese models often typography-normalize an immutable option marker
        # such as ``A)`` to ``A）``. Restore the source contract before the
        # structural check; this changes punctuation only and never remaps a
        # label or alters its translated content.
        translated[key] = re.sub(r"\b([ABCD])）", r"\1)", matches[0].strip())
    for key, text in translated.items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty translation: {key}")
        if not re.search(r"[\u4e00-\u9fff]", text):
            raise ValueError(f"translation lacks Chinese characters: {key}")
        original = source[key]
        for tag in ("<think>", "</think>", "<answer>", "</answer>"):
            if original.count(tag) != text.count(tag):
                raise ValueError(f"tag mismatch for {key}: {tag}")
        for option in ("A)", "B)", "C)", "D)"):
            source_has_label = bool(re.search(rf"(?m)^(?:\s*){re.escape(option)}|<answer>{re.escape(option)}", original))
            target_has_label = bool(re.search(rf"(?m)^(?:\s*){re.escape(option)}|<answer>{re.escape(option)}", text))
            if source_has_label and not target_has_label:
                raise ValueError(f"option-label mismatch for {key}: {option}")
    return translated


def request_translation(api_key: str, source: dict[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": canonical(source)},
        ],
        "temperature": 0,
        "max_tokens": 4096,
        "stream": False,
    }
    request = urllib.request.Request(API_URL, data=canonical(payload).encode(), method="POST", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode("utf-8", errors="replace"))
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or body.get("model") != MODEL:
        raise ValueError("invalid response or served-model mismatch")
    content = choices[0].get("message", {}).get("content", "")
    try:
        translated = parse_translation(content, source)
    except Exception as error:
        raise ValueError(f"invalid translation payload: {error}; excerpt={content[:800]!r}") from error
    usage = body.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    return translated, {
        "response_id": body.get("id"),
        "served_model": body.get("model"),
        "finish_reason": choices[0].get("finish_reason"),
        "usage": usage,
        "estimated_cost_usd": (prompt_tokens * 3.0 + completion_tokens * 15.0) / 1_000_000,
        "latency_seconds": time.monotonic() - started,
    }


def translate_with_retry(api_key: str, task: str, record: dict[str, Any], retries: int) -> dict[str, Any]:
    errors = []
    for attempt in range(1, retries + 1):
        try:
            translations, metadata = request_translation(api_key, record["texts"])
            return {
                "schema_version": 1,
                "task": task,
                "case_id": record["case_id"],
                "prompt_version": PROMPT_VERSION,
                "requested_model": MODEL,
                "source_texts_sha256": sha256_text(canonical(record["texts"])),
                "translations": translations,
                "metadata": metadata,
                "attempt": attempt,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            if attempt < retries:
                time.sleep(min(30, 2 ** (attempt - 1)) + random.random())
    raise RuntimeError(f"{task}/{record['case_id']} failed: {' | '.join(errors)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=("reward", "roi", "generation"), default=["reward", "roi", "generation"])
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    api_key = os.environ.get("AIGCBEST_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    sources = source_records()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for task in args.tasks:
        records = sources[task]
        if args.limit_per_task is not None:
            records = records[: args.limit_per_task]
        output_path = args.output_dir / f"{task}.jsonl"
        existing = {row["case_id"]: row for row in load_jsonl(output_path)} if output_path.exists() else {}
        pending = [row for row in records if row["case_id"] not in existing]
        completed = dict(existing)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(translate_with_retry, api_key, task, row, args.retries): row["case_id"] for row in pending}
            for future in as_completed(futures):
                result = future.result()
                completed[result["case_id"]] = result
                ordered = [completed[key] for key in sorted(completed)]
                temporary = output_path.with_suffix(".jsonl.tmp")
                temporary.write_text("".join(canonical(row) + "\n" for row in ordered), encoding="utf-8")
                os.replace(temporary, output_path)
                print(canonical({"task": task, "completed": len(completed), "target": len(records), "case_id": result["case_id"]}), flush=True)
        selected = [completed[row["case_id"]] for row in records]
        manifests[task] = {
            "case_count": len(selected),
            "file": str(output_path.resolve()),
            "file_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "estimated_cost_usd": sum(float(row["metadata"]["estimated_cost_usd"]) for row in selected),
        }
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "scientific_boundary": "Chinese translations are reading aids; English source remains authoritative",
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "tasks": manifests,
        "total_estimated_cost_usd": sum(row["estimated_cost_usd"] for row in manifests.values()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
