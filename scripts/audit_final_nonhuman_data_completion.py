#!/usr/bin/env python3
"""Fail-closed audit for the final non-human result/backup package."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


WORK = Path("/home/dataset-assist-0/czy/wjy")
EVAL = WORK / "pathvlm_revision_eval_a100/runs"
REPO = WORK / "myr1"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def completed_metrics(path: Path, expected_count: int) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    data = json.loads(path.read_text(encoding="utf-8"))
    count = data.get("count", data.get("source_count"))
    if count != expected_count:
        return False, f"count={count}, expected={expected_count}"
    status = str(data.get("status", ""))
    if status and not status.startswith("completed"):
        return False, f"status={status}"
    return True, "completed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-final-backup", action="store_true")
    args = parser.parse_args()

    checks: list[dict[str, object]] = []

    def add(label: str, path: Path, expected_count: int | None = None) -> None:
        if expected_count is None:
            ok, detail = path.is_file(), "present" if path.is_file() else "missing"
        else:
            ok, detail = completed_metrics(path, expected_count)
        checks.append({
            "label": label, "path": str(path), "required": True,
            "ok": ok, "detail": detail,
            "sha256": sha256(path) if ok and path.is_file() else None,
        })

    ratio_root = EVAL / "data_ratio_uniform_eval_20260814"
    for label in ("sft0250_rl0750", "sft0500_rl0500", "sft0750_rl0250"):
        root = ratio_root / label
        add(f"{label}:PathMMU-val", root / "pathmmu_val385/metrics.json", 385)
        add(f"{label}:PathMMU-test", root / "pathmmu_test999/metrics.json", 999)
        add(f"{label}:PathVQA-A/B", root / "pathvqa_test3362_ab/metrics.json", 3362)
        add(f"{label}:MMMU", root / "mmmu_nonmedical116/metrics.json", 116)
        add(f"{label}:OmniMedVQA", root / "omnimedvqa_8518/metrics.json", 8518)

    repeat_root = EVAL / "repeated_inference_pathmmu_test999_20260814"
    for label in ("l_r16_sft80", "full_rule_rl_n8_step1000", "stage3_gpt4o_step1500"):
        add(f"repeat:{label}", repeat_root / label / "summary.json")

    add(
        "LoRA-SFT4000:OmniMedVQA",
        EVAL / "lora_sft4000_control_20260814/formal_8gpu_gbs96/omnimedvqa_8518/metrics.json",
        8518,
    )
    add(
        "historical-Stage2:OmniMedVQA",
        EVAL / "core_historical_corrected_eval_20260813/stage2/omnimedvqa_8518/metrics.json",
        8518,
    )
    add("unified Markdown", REPO / "docs/20260814_final_unified_results_tables.md")
    add("unified JSON", REPO / "protocol/final_unified_results_tables_20260814.json")
    add(
        "minimal core paired statistics",
        REPO / "protocol/final_core_pathmmu_statistics_20260814/cluster_bootstrap_results.json",
    )
    add("legacy external baselines and multijudge", REPO / "docs/20260810_final_closed_benchmark_and_multijudge_results.md")
    add(
        "explainability dual-stream",
        EVAL / "stage3_gpt4o_n8_checkpoint1500_final_eval_20260813/reference_evidence_dual_stream_v2/metrics.json",
    )
    if args.require_final_backup:
        backup_path = REPO / "protocol/final_hf_results_backup_20260814.json"
        add("final HF backup verification", backup_path)
        if backup_path.is_file():
            backup = json.loads(backup_path.read_text(encoding="utf-8"))
            archive = Path(backup["full_archive"]["local_path"])
            expected = backup["full_archive"]["sha256"]
            actual = sha256(archive) if archive.is_file() else None
            checks.append({
                "label": "final full local archive hash",
                "path": str(archive),
                "required": True,
                "ok": actual == expected,
                "detail": "hash_verified" if actual == expected else f"actual={actual}, expected={expected}",
                "sha256": actual,
            })

    missing = [item for item in checks if not item["ok"]]
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "all required non-human report data; human expert scoring explicitly excluded",
        "status": "complete" if not missing else "incomplete",
        "human_expert_scoring": "excluded_pending_user",
        "required_check_count": len(checks),
        "failed_check_count": len(missing),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failed": [item["label"] for item in missing]}, ensure_ascii=False))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
