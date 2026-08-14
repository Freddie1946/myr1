#!/usr/bin/env python3
"""Freeze a blinded, stratified human audit packet from formal Stage3 rollouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVENTS = (
    "image_feature_analysis_present",
    "option_elimination_present",
    "medical_knowledge_support_present",
    "histological_definition_error",
    "logical_contradiction",
    "outdated_or_incorrect_pathology_criterion",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_key(seed: int, row: dict[str, Any]) -> str:
    identity = "|".join(str(row[x]) for x in ("rank", "call_index", "item_index", "record_index"))
    return hashlib.sha256(f"{seed}|{identity}|{row['completion_sha256']}".encode()).hexdigest()


def process_band(score: float) -> str:
    if score >= 0.999:
        return "high_1p0"
    if score >= 0.799:
        return "mid_0p8"
    return "low_le_0p6"


def load_rows(run_root: Path) -> list[dict[str, Any]]:
    reward_paths = sorted((run_root / "reward_audit" / "segment00").glob("rank_*.jsonl"))
    judge_paths = sorted((run_root / "judge" / "audit" / "segment00").glob("rank_*.jsonl"))
    if len(reward_paths) != 8 or len(judge_paths) != 8:
        raise ValueError("expected exactly eight aligned reward and judge shards")
    rows = []
    for rank, (reward_path, judge_path) in enumerate(zip(reward_paths, judge_paths)):
        groups: dict[tuple[int, int], dict[str, Any]] = defaultdict(dict)
        for raw in reward_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(raw)
            if int(row["rank"]) != rank:
                raise ValueError("rank mismatch in reward audit")
            groups[(int(row["call_index"]), int(row["item_index"]))][row["reward_type"]] = row
        if len(groups) != 1500 or any(set(x) != {"accuracy", "format", "process"} for x in groups.values()):
            raise ValueError("reward triplets are incomplete")
        judges = [json.loads(x) for x in judge_path.read_text(encoding="utf-8").splitlines()]
        if len(judges) != 1500:
            raise ValueError("judge shard is incomplete")
        for ordinal, key in enumerate(sorted(groups)):
            call_index, item_index = key
            triplet = groups[key]
            source = triplet["process"]
            judge = judges[ordinal]
            if int(judge["record_index"]) != int(source["record_index"]):
                raise ValueError("judge/reward record mismatch")
            if abs(float(judge["process_reward"]) - float(source["reward"])) > 1e-6:
                raise ValueError("judge/reward score mismatch")
            if judge.get("served_model") != "gpt-4o-2024-08-06" or judge.get("status") != "completed":
                raise ValueError("unexpected judge identity or status")
            image = Path(source["image_path"])
            if not image.is_file() or sha256_file(image) != source["image_sha256"]:
                raise ValueError("source image integrity failure")
            events = judge["events"]
            if any(name not in events for name in EVENTS):
                raise ValueError("judge event schema incomplete")
            rows.append({
                "rank": rank, "call_index": call_index, "item_index": item_index,
                "record_index": int(source["record_index"]), "stage_third": call_index // 500,
                "accuracy_reward": float(triplet["accuracy"]["reward"]),
                "format_reward": float(triplet["format"]["reward"]),
                "process_reward": float(source["reward"]), "process_band": process_band(float(source["reward"])),
                "events": {name: bool(events[name]) for name in EVENTS},
                "event_evidence": events.get("evidence", {}),
                "problem": source["problem"], "solution": source["solution"],
                "completion": source["completion"],
                "completion_sha256": hashlib.sha256(source["completion"].encode()).hexdigest(),
                "image_path": str(image.resolve()), "image_sha256": source["image_sha256"],
                "judge_cache_key": judge["cache_key"], "judge_response_id": judge.get("response_id"),
            })
    if len(rows) != 12000:
        raise ValueError("formal run must contain 12,000 aligned trajectories")
    return rows


def select(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    used_records: set[int] = set()
    selected: list[dict[str, Any]] = []

    def take(pool: list[dict[str, Any]], n: int, group: str, stratum: str) -> None:
        eligible = [x for x in pool if x["record_index"] not in used_records]
        eligible.sort(key=lambda x: stable_key(seed, x))
        if len(eligible) < n:
            raise ValueError(f"insufficient unique records for {group}/{stratum}: {len(eligible)} < {n}")
        for row in eligible[:n]:
            copy = dict(row); copy["selection_group"] = group; copy["selection_stratum"] = stratum
            selected.append(copy); used_records.add(copy["record_index"])

    # Probability-style representative component: 18 cells x 2 trajectories.
    for stage in range(3):
        for accuracy in (0.0, 1.0):
            for band in ("low_le_0p6", "mid_0p8", "high_1p0"):
                pool = [x for x in rows if x["stage_third"] == stage and x["accuracy_reward"] == accuracy
                        and x["process_band"] == band]
                take(pool, 2, "representative", f"stage{stage}_acc{int(accuracy)}_{band}")

    # Deliberately separate challenge component; never mix it into prevalence estimates.
    take([x for x in rows if x["process_reward"] == 1.0 and x["accuracy_reward"] == 0.0],
         6, "challenge", "high_process_wrong_answer")
    take([x for x in rows if x["process_reward"] <= 0.6 and x["accuracy_reward"] == 1.0],
         6, "challenge", "low_process_correct_answer")
    take([x for x in rows if x["events"]["histological_definition_error"]],
         4, "challenge", "histological_error_positive")
    take([x for x in rows if x["events"]["logical_contradiction"]],
         4, "challenge", "logical_contradiction_positive")
    take([x for x in rows if x["events"]["outdated_or_incorrect_pathology_criterion"]],
         4, "challenge", "incorrect_criterion_positive")
    if len(selected) != 60 or len(used_records) != 60:
        raise AssertionError("selection contract did not yield 60 unique prompts")
    return selected


def write_packet(rows: list[dict[str, Any]], output_dir: Path, seed: int, run_root: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    blind_dir = output_dir / "blinded_packet"
    image_dir = blind_dir / "images"
    image_dir.mkdir(parents=True)
    order = sorted(rows, key=lambda x: stable_key(seed + 1, x))
    blinded = []
    internal = []
    for number, row in enumerate(order, 1):
        case_id = f"HRA-{number:03d}"
        source_image = Path(row["image_path"])
        target_image = image_dir / f"{case_id}{source_image.suffix.lower()}"
        shutil.copy2(source_image, target_image)
        if sha256_file(target_image) != row["image_sha256"]:
            raise ValueError("copied packet image hash mismatch")
        blind = {
            "case_id": case_id, "image_file": str(Path("images") / target_image.name),
            "image_sha256": row["image_sha256"], "question_and_options": row["problem"],
            "dataset_reference_answer": row["solution"], "candidate_completion": row["completion"],
            "ratings": {name: None for name in EVENTS} | {
                "pathology_reasoning_correctness_1_to_5": None,
                "image_grounding_1_to_5": None,
                "logic_consistency_1_to_5": None,
                "hallucination_present": None,
                "reference_answer_ambiguous_or_noisy": None,
                "reviewer_confidence_1_to_5": None,
                "reviewer_notes": "",
            },
        }
        blinded.append(blind)
        internal.append({"case_id": case_id, **row})
    blind_path = blind_dir / "cases.jsonl"
    blind_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in blinded), encoding="utf-8")
    key_path = output_dir / "internal_selection_key.jsonl"
    key_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in internal), encoding="utf-8")
    with (blind_dir / "ratings_template.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["case_id", *EVENTS, "pathology_reasoning_correctness_1_to_5", "image_grounding_1_to_5",
                  "logic_consistency_1_to_5", "hallucination_present", "reference_answer_ambiguous_or_noisy",
                  "reviewer_confidence_1_to_5", "reviewer_notes"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for x in blinded: writer.writerow({"case_id": x["case_id"]})
    instructions = """# Blinded human reward-agreement packet

Score every case independently from the image, question/options, dataset reference answer and candidate completion.
Do not infer missing image evidence from the answer. For the six event fields use `true`, `false`, or `uncertain`.
The first three fields mean a valid component is present; the last three mean the named error is present.
Use integers 1--5 for the three quality dimensions and reviewer confidence. Do not consult model identity,
training stage, automated reward, or another reviewer. Reviewer A scores all 60 cases; reviewer B scores at
least 30 overlapping cases. The representative and challenge subsets are disclosed only after ratings freeze.
This packet evaluates agreement with an automated process auditor; it is not a clinical-validity study.
"""
    (blind_dir / "README.md").write_text(instructions, encoding="utf-8")
    forbidden = ("gpt", "process_reward", "accuracy_reward", "selection_group", "stage_third", "rank", "call_index")
    lower = blind_path.read_text(encoding="utf-8").lower()
    if any(token in lower for token in forbidden):
        raise ValueError("blinding audit failed")
    counts = Counter(x["selection_group"] for x in rows)
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "frozen",
        "source_run": str(run_root.resolve()), "source_trajectory_count": 12000,
        "unique_source_prompt_count": 1000, "selected_case_count": 60,
        "unique_selected_prompt_count": len({x["record_index"] for x in rows}),
        "selection_counts": dict(counts), "representative_case_count": counts["representative"],
        "challenge_case_count": counts["challenge"], "minimum_required_case_count": 50,
        "reviewer_a_assignment": 60, "reviewer_b_minimum_overlap": 30,
        "selection_seed": seed, "blinded_to_judge_and_training_identity": True,
        "blinded_cases_sha256": sha256_file(blind_path), "internal_key_sha256": sha256_file(key_path),
        "ratings_template_sha256": sha256_file(blind_dir / "ratings_template.csv"),
        "image_count": len(list(image_dir.iterdir())),
        "interpretation_boundary": "representative and challenge subsets must be reported separately",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260814)
    args = p.parse_args()
    rows = load_rows(args.run_root)
    write_packet(select(rows, args.seed), args.output_dir, args.seed, args.run_root)


if __name__ == "__main__":
    main()
