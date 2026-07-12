#!/usr/bin/env python3
"""Re-score saved online completions with the current reward implementation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from pathmmu_rewards import accuracy_reward, choice_letter, format_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    events = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    accuracy_events = [event for event in events if event.get("reward_type") == "accuracy"]
    rows = []
    for index, event in enumerate(accuracy_events):
        completion = event["completion"]
        solution = event["solution"]
        wrapped = [[{"role": "assistant", "content": completion}]]
        accuracy = accuracy_reward(wrapped, [solution])[0]
        format_score = format_reward(wrapped)[0]
        rows.append(
            {
                "index": index,
                "old_accuracy_reward": event.get("reward"),
                "corrected_accuracy_reward": accuracy,
                "corrected_format_reward": format_score,
                "corrected_total_reward": accuracy + format_score,
                "predicted_choice": choice_letter(completion),
                "target_choice": choice_letter(solution),
            }
        )
    totals = [row["corrected_total_reward"] for row in rows]
    result = {
        "completion_count": len(rows),
        "rows": rows,
        "corrected_mean_accuracy": sum(row["corrected_accuracy_reward"] for row in rows) / len(rows),
        "corrected_mean_format": sum(row["corrected_format_reward"] for row in rows) / len(rows),
        "corrected_mean_total_reward": sum(totals) / len(totals),
        "corrected_sample_reward_std": statistics.stdev(totals) if len(totals) > 1 else 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
