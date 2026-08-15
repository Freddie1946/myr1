#!/usr/bin/env python3
"""Fail-closed audit of final non-human results and human-review inputs.

Human ratings are intentionally not required.  The audit verifies that every
declared evaluation has its full expected sample count, that repeated-inference
summaries contain five runs, and that the frozen human-review packets are
internally complete before distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORK = Path("/home/dataset-assist-0/czy/wjy")
EVAL = WORK / "pathvlm_revision_eval_a100/runs"
REPO = WORK / "myr1"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_count(value: dict[str, Any]) -> int | None:
    for key in ("count", "source_count", "completed", "expected_count"):
        if type(value.get(key)) is int:
            return int(value[key])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks: list[dict[str, Any]] = []

    def record(label: str, path: Path, ok: bool, detail: str, category: str) -> None:
        checks.append({
            "label": label,
            "category": category,
            "path": str(path),
            "ok": bool(ok),
            "detail": detail,
            "sha256": sha256(path) if path.is_file() else None,
        })

    def file_check(label: str, path: Path, category: str = "documentation") -> None:
        record(label, path, path.is_file(), "present" if path.is_file() else "missing", category)

    def metrics_check(label: str, path: Path, count: int, category: str) -> None:
        if not path.is_file():
            record(label, path, False, "missing", category)
            return
        value = load(path)
        actual = metric_count(value)
        status = str(value.get("status", ""))
        ok = actual == count and (not status or status.startswith("completed"))
        record(label, path, ok, f"count={actual}; status={status or 'implicit-complete'}", category)

    # Current paper-line core evaluations are registry-driven.
    registry = load(REPO / "protocol/final_unified_results_tables_20260814.json")
    for row in registry.get("core_current_contract", []):
        name = str(row["model"])
        paths = row.get("paths", {})
        for endpoint, expected in (
            ("pathmmu_val", 385), ("pathmmu_test", 999),
            ("pathvqa_fixed_ab", 3362), ("omni", 8518), ("mmmu", 116),
        ):
            if endpoint in paths:
                metrics_check(f"core:{name}:{endpoint}", Path(paths[endpoint]), expected, "core_results")

    # The two paper gaps closed on 2026-08-15.
    missing = EVAL / "missing_paper_evaluations_20260815"
    for step in (500, 1000, 1500):
        metrics_check(
            f"Stage3-{step}:PathVQA-free-YesNo",
            missing / f"stage3_gpt4o_checkpoint{step}_pathvqa_yesno_v5/full3362/metrics.json",
            3362,
            "newly_completed_results",
        )
    metrics_check(
        "Stage2-continued-ruleRL1000:OmniMedVQA",
        missing / "stage2_continued_rule_rl1000_omnimedvqa_v4/full8518/metrics.json",
        8518,
        "newly_completed_results",
    )

    # Uniform Omni baseline reruns.
    omni = EVAL / "omnimedvqa_per_model_gated_v2_20260815"
    baseline_names = (
        "qwen2_5_vl_3b", "lingshu_7b", "medvlm_r1", "medgemma_4b_it",
        "scalereasoner_r1", "llama3_2_vision_11b", "huatuogpt_vision_7b",
        "internvl3_8b", "deepseek_vl2", "llama3_2_vision_90b", "llava_med_7b",
    )
    for name in baseline_names:
        metrics_check(f"baseline:{name}:OmniMedVQA", omni / name / "full8518/metrics.json", 8518, "baselines")

    # Sparse 1,000-sample allocation screen.
    ratio = EVAL / "data_ratio_uniform_eval_20260814"
    for arm in ("sft0250_rl0750", "sft0500_rl0500", "sft0750_rl0250"):
        for endpoint, expected in (
            ("pathmmu_val385", 385), ("pathmmu_test999", 999),
            ("pathvqa_test3362_ab", 3362), ("omnimedvqa_8518", 8518),
            ("mmmu_nonmedical116", 116),
        ):
            metrics_check(f"ratio:{arm}:{endpoint}", ratio / arm / endpoint / "metrics.json", expected, "data_ratio")

    # Five-seed stochastic inference repeats for selected main stages.
    repeat_roots = (
        EVAL / "repeated_inference_pathmmu_test999_20260814",
        EVAL / "repeated_inference_pathmmu_test999_20260815",
    )
    expected_repeats = {
        "l_r16_sft80", "full_rule_rl_n8_step1000", "stage3_gpt4o_step1500",
        "base_qwen2_5_vl_7b", "rule_rl4000_checkpoint2500_provisional",
        "stage2_continued_rule_rl1000", "stage2_outcome_grpo",
        "stage3_gpt4o_step500", "stage3_gpt4o_step1000",
    }
    found: set[str] = set()
    for root in repeat_roots:
        for path in root.glob("*/summary.json"):
            if path.parent.is_symlink() or path.parent.name not in expected_repeats:
                continue
            value = load(path)
            ok = value.get("run_count") == 5
            record(f"repeat:{path.parent.name}", path, ok, f"run_count={value.get('run_count')}", "repeated_inference")
            found.add(path.parent.name)
    for name in sorted(expected_repeats - found):
        record(f"repeat:{name}", repeat_roots[0] / name / "summary.json", False, "missing", "repeated_inference")

    file_check("multi-Judge completion", REPO / "protocol/final_evaluation_multijudge_completion_20260810.json", "reasoning_quality")
    interpretability_path = EVAL / "stage3_gpt4o_n8_checkpoint1500_final_eval_20260813/reference_evidence_dual_stream_v2/metrics.json"
    if interpretability_path.is_file():
        value = load(interpretability_path)
        ok = value.get("status") == "completed" and value.get("primary_reasoning_clean_correct_and_interface_agreement_count") == 30
        record(
            "interpretability dual-stream", interpretability_path, ok,
            f"status={value.get('status')}; primary_cases={value.get('primary_reasoning_clean_correct_and_interface_agreement_count')}",
            "interpretability",
        )
    else:
        record("interpretability dual-stream", interpretability_path, False, "missing", "interpretability")
    file_check("result lineage", REPO / "docs/result_catalog_20260815/result_lineage.json")
    file_check("paper tables", REPO / "docs/result_catalog_20260815/paper_tables.md")

    # Human inputs are required; human ratings themselves are deliberately pending.
    browse = WORK / "pathvlm_revision_eval_a100/human_review/expert_review_browse_packets_20260815"
    browse_manifest = browse / "manifest.json"
    if browse_manifest.is_file():
        value = load(browse_manifest)
        listed = value.get("file_sha256", {})
        mismatches = []
        for rel, expected in listed.items():
            path = browse / rel
            if not path.is_file() or sha256(path) != expected:
                mismatches.append(rel)
        ok = (
            value.get("reward_case_count") == 60
            and value.get("roi_case_count") == 20
            and not mismatches
        )
        record(
            "human:reward60+ROI20 browse packet", browse_manifest, ok,
            f"reward={value.get('reward_case_count')}; roi={value.get('roi_case_count')}; hash_mismatches={len(mismatches)}",
            "human_inputs",
        )
    else:
        record("human:reward60+ROI20 browse packet", browse_manifest, False, "missing", "human_inputs")

    pair = EVAL / "stage2_stage3_human_review_packet_20260815/manifest.json"
    if pair.is_file():
        value = load(pair)
        ok = value.get("blind_pairwise", {}).get("case_count") == 100
        record("human:Stage2-vs-Stage3 blind100", pair, ok, f"cases={value.get('blind_pairwise', {}).get('case_count')}", "human_inputs")
    else:
        record("human:Stage2-vs-Stage3 blind100", pair, False, "missing", "human_inputs")

    failed = [item for item in checks if not item["ok"]]
    by_category: dict[str, dict[str, int]] = {}
    for item in checks:
        summary = by_category.setdefault(item["category"], {"total": 0, "failed": 0})
        summary["total"] += 1
        summary["failed"] += int(not item["ok"])
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "paper-facing non-human results plus frozen human-review inputs",
        "status": "complete_except_human_ratings" if not failed else "incomplete",
        "human_ratings_status": "pending_by_design",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "category_summary": by_category,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks), "failed": [x["label"] for x in failed]}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
