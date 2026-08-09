#!/usr/bin/env python3
"""Blind, resumable multi-judge evaluation of PathMMU reasoning quality."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_URL = "https://api2.aigcbest.top/v1/chat/completions"
PROMPT_VERSION = "pathmmu_blind_six_metric_multijudge_v1"
TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504, 520, 529}
JUDGES = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "gemini-3.1-pro": {"input": 2.0, "output": 12.0},
    "gpt-4o-2024-08-06": {"input": 2.5, "output": 10.0},
}
METRICS = (
    "r_acc", "k_acc", "rigor", "professionalism", "clarity", "conciseness"
)
SYSTEM_PROMPT = """You are an independent evaluator of pathology visual question-answering.
You receive an image, a multiple-choice question, a reference answer/reasoning, and an anonymous
candidate response. Do not infer the candidate model identity. Evaluate the candidate itself, not
its writing similarity to the reference. The reference is context and may not be the only valid
reasoning path. Use the image when judging visual claims.

Return exactly one JSON object with these seven fields:
{"r_acc": number, "k_acc": number, "rigor": number, "professionalism": number,
 "clarity": number, "conciseness": number, "overall_reason": string}

Every numeric score must lie in [0,1]. Use the full continuous range and no more than two decimal
places. Definitions:
- r_acc: reasoning logically and evidentially supports the candidate's final answer; concise,
  sufficient reasoning is preferred. Do not award this merely because the final option is correct.
- k_acc: medical/pathology facts, terminology, principles and diagnostic criteria are accurate.
- rigor: reasoning flows without unsupported leaps or contradictions, and conclusions follow.
- professionalism: terminology and language are appropriate for a medical expert audience.
- clarity: reasoning is organized and understandable with adequate explanation.
- conciseness: response is direct and avoids unnecessary repetition or verbosity.

Score wrong but well-written answers fairly by dimension: their style scores may be high while
r_acc/k_acc are low. Score a bare answer low on reasoning dimensions even if its option is correct.
The overall_reason must be at most 240 characters and must not mention a model identity."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty prediction file: {path}")
    seen: set[int] = set()
    for row in rows:
        index = int(row["index"])
        if index in seen:
            raise ValueError(f"duplicate index {index} in {path}")
        seen.add(index)
    return rows


def load_manifest(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("manifest.models must be a nonempty list")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for item in models:
        label = str(item["label"])
        prediction_path = Path(item["predictions"]).resolve()
        if label in loaded or not prediction_path.is_file():
            raise ValueError(f"invalid model entry: {item}")
        loaded[label] = load_jsonl(prediction_path)
        item["predictions"] = str(prediction_path)
        item["predictions_sha256"] = sha256_file(prediction_path)
    return loaded, manifest


def freeze_panel(
    loaded: dict[str, list[dict[str, Any]]], *, count: int, repeat_count: int, seed: int
) -> dict[str, Any]:
    maps = {label: {int(row["index"]): row for row in rows} for label, rows in loaded.items()}
    common = set.intersection(*(set(rows) for rows in maps.values()))
    if len(common) < count:
        raise ValueError(f"only {len(common)} common cases; requested {count}")
    anchor_label = next(iter(maps))
    anchor = maps[anchor_label]
    eligible: list[tuple[str, int]] = []
    for index in common:
        source_hash = str(anchor[index].get("source_record_sha256", ""))
        if not source_hash:
            raise ValueError(f"case {index} lacks source_record_sha256")
        for label, rows in maps.items():
            if str(rows[index].get("source_record_sha256")) != source_hash:
                raise ValueError(f"source mismatch at index {index}: {anchor_label} vs {label}")
        rank = hashlib.sha256(f"panel:{seed}:{source_hash}".encode()).hexdigest()
        eligible.append((rank, index))
    chosen = [index for _, index in sorted(eligible)[:count]]
    repeat_ranked = sorted(
        (hashlib.sha256(f"repeat:{seed}:{anchor[i]['source_record_sha256']}".encode()).hexdigest(), i)
        for i in chosen
    )
    repeated = [index for _, index in repeat_ranked[:repeat_count]]
    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "selection": "sha256-ranked common indices; independent of model outputs and correctness",
        "seed": seed,
        "case_count": count,
        "repeat_case_count": repeat_count,
        "indices": chosen,
        "repeat_indices": repeated,
        "source_record_sha256": {str(i): anchor[i]["source_record_sha256"] for i in chosen},
    }


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def prompt_text(row: dict[str, Any]) -> str:
    return (
        "QUESTION AND OPTIONS:\n" + str(row["problem"]) +
        "\n\nREFERENCE ANSWER/REASONING (context, not a template):\n" + str(row["solution"]) +
        "\n\nANONYMOUS CANDIDATE RESPONSE:\n" + str(row["completion"])
    )


def parse_scores(text: str) -> dict[str, Any]:
    value = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    if not isinstance(value, dict) or set(value) != {*METRICS, "overall_reason"}:
        raise ValueError("invalid judge field set")
    for key in METRICS:
        score = value[key]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
            raise ValueError(f"invalid {key} score")
        value[key] = float(score)
    reason = value["overall_reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 480:
        raise ValueError("invalid overall_reason")
    return value


def response_cost(usage: Any, judge: str) -> float:
    if not isinstance(usage, dict):
        raise ValueError("response lacks usage")
    prompt = int(usage["prompt_tokens"])
    completion = int(usage["completion_tokens"])
    price = JUDGES[judge]
    return (prompt * price["input"] + completion * price["output"]) / 1_000_000


def request_once(api_key: str, judge: str, row: dict[str, Any], run: int) -> tuple[dict[str, Any], dict[str, Any]]:
    image = Path(row["image"] if "image" in row else row["image_path"]).resolve()
    payload = {
        "model": judge,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text(row)},
                {"type": "image_url", "image_url": {"url": image_data_url(image)}},
            ]},
        ],
        "temperature": 0,
        "seed": 42 + run,
        "max_tokens": 2048 if judge == "gemini-3.1-pro" else 320,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    if judge == "gemini-3.1-pro":
        # Gemini 3.1 Pro is a reasoning model. Its default thinking can consume a
        # 320-token completion cap before the JSON answer begins. Low effort plus
        # a larger ceiling preserves the requested semantic judge while avoiding
        # empty/truncated visible content.
        payload["reasoning_effort"] = "low"
    request = urllib.request.Request(
        API_URL, data=canonical_json(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw[:500]}") from exc
    body = json.loads(raw)
    choices = body.get("choices")
    if status != 200 or not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("successful response lacks exactly one choice")
    served = body.get("model")
    if served != judge:
        raise ValueError(f"served-model mismatch: requested {judge}, got {served}")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("response content is not text")
    try:
        scores = parse_scores(content)
    except (json.JSONDecodeError, ValueError) as exc:
        excerpt = content[:1000].replace("\x00", "<NUL>")
        raise ValueError(
            f"invalid judge JSON: {exc}; finish_reason={choices[0].get('finish_reason')!r}; "
            f"usage={body.get('usage')!r}; content={excerpt!r}"
        ) from exc
    metadata = {
        "response_id": body.get("id"),
        "served_model": served,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": body.get("usage"),
        "latency_seconds": time.monotonic() - started,
        "cost_usd": response_cost(body.get("usage"), judge),
    }
    return scores, metadata


def load_budget(path: Path, limit: float) -> dict[str, Any]:
    if path.exists():
        budget = json.loads(path.read_text(encoding="utf-8"))
        if float(budget["limit_usd"]) != limit:
            raise ValueError("existing budget limit differs")
        return budget
    return {"schema_version": 1, "limit_usd": limit, "spent_usd": 0.0, "attempts": []}


def account_budget(path: Path, budget: dict[str, Any], event: dict[str, Any]) -> None:
    budget["attempts"].append(event)
    budget["spent_usd"] = sum(float(x.get("accounted_cost_usd", 0)) for x in budget["attempts"])
    budget["remaining_usd"] = float(budget["limit_usd"]) - float(budget["spent_usd"])
    budget["updated_at"] = now_iso()
    atomic_json(path, budget)


def result_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return str(row["judge"]), str(row["candidate_label"]), int(row["index"]), int(row["run"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--budget-limit-usd", type=float, default=40.0)
    parser.add_argument("--case-count", type=int, default=100)
    parser.add_argument("--repeat-case-count", type=int, default=30)
    parser.add_argument("--panel-seed", type=int, default=20260810)
    parser.add_argument("--judges", nargs="+", choices=tuple(JUDGES), default=tuple(JUDGES))
    parser.add_argument("--reuse-smoke-results", type=Path)
    args = parser.parse_args()
    api_key = os.getenv("AIGCBEST_API_KEY", "")
    if not api_key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    if not 0 <= args.repeat_case_count <= args.case_count:
        raise ValueError("invalid repeat-case count")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    loaded, manifest = load_manifest(args.manifest)
    panel_path = output / "frozen_panel.json"
    if panel_path.exists():
        panel = json.loads(panel_path.read_text(encoding="utf-8"))
    else:
        panel = freeze_panel(
            loaded, count=args.case_count, repeat_count=args.repeat_case_count, seed=args.panel_seed
        )
        atomic_json(panel_path, panel)
    maps = {label: {int(row["index"]): row for row in rows} for label, rows in loaded.items()}
    results_path = output / ("smoke_results.jsonl" if args.mode == "smoke" else "judgments.jsonl")
    existing: set[tuple[str, str, int, int]] = set()
    if results_path.exists():
        existing = {result_key(json.loads(line)) for line in results_path.read_text().splitlines() if line.strip()}
    reused_count = 0
    if args.mode == "full" and args.reuse_smoke_results:
        for line in args.reuse_smoke_results.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prior = json.loads(line)
            key = result_key(prior)
            judge, label, index, run = key
            if (
                judge not in args.judges or label not in loaded or index not in panel["indices"]
                or run != 0 or key in existing
            ):
                continue
            expected_hash = maps[label][index]["source_record_sha256"]
            if prior.get("source_record_sha256") != expected_hash:
                raise ValueError(f"reused smoke source mismatch: {key}")
            append_jsonl(results_path, {**prior, "origin": "identity_smoke_reused"})
            existing.add(key)
            reused_count += 1
    budget_path = output / "budget_ledger.json"
    budget = load_budget(budget_path, args.budget_limit_usd)
    labels = list(loaded)
    indices = panel["indices"][:4] if args.mode == "smoke" else panel["indices"]
    tasks: list[tuple[str, str, int, int]] = []
    for judge in args.judges:
        for label in labels:
            for index in indices:
                tasks.append((judge, label, int(index), 0))
                if args.mode == "full" and int(index) in set(panel["repeat_indices"]):
                    tasks.extend((judge, label, int(index), run) for run in (1, 2))
    random.Random(args.panel_seed + (0 if args.mode == "smoke" else 1)).shuffle(tasks)

    audit_path = output / "request_audit.jsonl"
    for ordinal, (judge, label, index, run) in enumerate(tasks, 1):
        key = (judge, label, index, run)
        if key in existing:
            continue
        remaining = args.budget_limit_usd - float(budget.get("spent_usd", 0))
        if remaining < 0.05:
            raise RuntimeError(f"budget reserve gate reached; remaining=${remaining:.4f}")
        row = maps[label][index]
        last_error: Exception | None = None
        for attempt in range(1, 4):
            if attempt > 1:
                time.sleep((2, 8)[attempt - 2])
            event_base = {
                "recorded_at": now_iso(), "mode": args.mode, "judge": judge,
                "candidate_label": label, "index": index, "run": run, "attempt": attempt,
            }
            try:
                scores, metadata = request_once(api_key, judge, row, run)
            except Exception as exc:
                last_error = exc
                event = {**event_base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                append_jsonl(audit_path, event)
                # A response that failed validation may still be billed. Reserve a conservative amount.
                accounted = 0.02 if isinstance(exc, ValueError) else 0.005
                account_budget(budget_path, budget, {**event, "accounted_cost_usd": accounted})
                if attempt < 3:
                    continue
                break
            event = {**event_base, "status": "passed", **metadata}
            account_budget(
                budget_path, budget, {**event, "accounted_cost_usd": metadata["cost_usd"]}
            )
            result = {
                "schema_version": 1, "recorded_at": now_iso(), "prompt_version": PROMPT_VERSION,
                "judge": judge, "candidate_label": label, "index": index, "run": run,
                "source_record_sha256": row["source_record_sha256"],
                "target_choice": row.get("target_choice"),
                "predicted_choice": row.get("predicted_choice"),
                "candidate_correct": row.get("accuracy_reward") == 1.0,
                "scores": scores, **metadata,
            }
            append_jsonl(results_path, result)
            existing.add(key)
            print(
                f"[{ordinal}/{len(tasks)}] {judge} {label} index={index} run={run} "
                f"cost=${metadata['cost_usd']:.5f} total=${budget['spent_usd']:.4f}", flush=True
            )
            break
        else:
            raise AssertionError("unreachable")
        if key not in existing:
            raise RuntimeError(f"request failed after retries: {key}: {last_error}")

    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    summary = {
        "schema_version": 1, "updated_at": now_iso(), "mode": args.mode,
        "manifest": manifest, "manifest_sha256": sha256_file(args.manifest.resolve()),
        "panel": str(panel_path), "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "judges": args.judges, "model_count": len(labels), "result_count": len(rows),
        "reused_smoke_result_count": reused_count,
        "served_models": sorted({row["served_model"] for row in rows}),
        "spent_usd": budget["spent_usd"], "budget_limit_usd": args.budget_limit_usd,
    }
    atomic_json(output / f"{args.mode}_summary.json", summary)


if __name__ == "__main__":
    main()
