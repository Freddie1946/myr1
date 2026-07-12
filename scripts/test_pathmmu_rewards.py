#!/usr/bin/env python3

from pathmmu_rewards import accuracy_reward, choice_letter, format_reward


def wrapped(text: str):
    return [[{"role": "assistant", "content": text}]]


assert choice_letter("<answer>D) diagnosis</answer>") == "D"
assert choice_letter("The answer is B.") == "B"
assert choice_letter('<answer>{"option": "B") Melanocytes"}</answer>') == "B"
assert choice_letter('<answer>{"answer": "C) Diagnosis"}</answer>') == "C"
assert choice_letter(
    '<answer>{"Option A": "Keratinocytes", "Option B": "Melanocytes", '
    '"Option C": "Sebaceous cells", "Option D": "Fibroblasts"}</answer>'
) is None
assert accuracy_reward(wrapped("<answer>D</answer>"), ["<answer>D) target</answer>"]) == [1.0]
assert accuracy_reward(wrapped("<answer>A</answer>"), ["<answer>D) target</answer>"]) == [0.0]
assert accuracy_reward(
    wrapped('<answer>{"Option A": "Keratinocytes", "Option B": "Melanocytes"}</answer>'),
    ["<answer>A) Keratinocytes</answer>"],
) == [0.0]
assert format_reward(wrapped("<think>reason</think><answer>D</answer>")) == [1.0]
assert format_reward(wrapped("D")) == [0.0]
print("pathmmu reward tests: PASS")
