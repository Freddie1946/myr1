#!/usr/bin/env python3

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("freeze_pathvqa_cross_format_semantic_panel.py")
SPEC = importlib.util.spec_from_file_location("cross_format_panel", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(index, question, image, base_ok, sft_ok, target="yes"):
    key = f"{index:064x}"
    base_answer = target if base_ok else ("no" if target == "yes" else "yes")
    sft_answer = target if sft_ok else ("no" if target == "yes" else "yes")
    common = {
        "source_index": index,
        "source_record_sha256": key,
        "image": image,
        "question": question,
        "target": target,
    }
    base = {
        **common,
        "forced_binary_correct": base_ok,
        "forced_binary_answer": base_answer,
        "target_margin": 0.5 if base_ok else -0.5,
    }
    sft = {
        **common,
        "forced_binary_correct": sft_ok,
        "forced_binary_answer": sft_answer,
        "target_margin": 0.5 if sft_ok else -0.5,
    }
    return key, base, sft


def test_category_contract():
    assert MODULE.category("Is liver present?") == "short_presence"
    assert MODULE.category("Does this image show Hashimoto thyroiditis?") == "image_diagnosis"
    assert MODULE.category("Does the iron stain show deposits?") == "stain_marker"
    assert MODULE.category("Are pleomorphic nuclei visible?") == "morphology"
    assert MODULE.category("Is the diagnosis supported?") == "other"


def test_outcome_contract():
    _, base, sft = row(1, "Is liver present?", "a.jpg", True, False)
    assert MODULE.outcome_group(base, sft) == "regression"
    _, base, sft = row(2, "Is liver present?", "b.jpg", False, True)
    assert MODULE.outcome_group(base, sft) == "improvement"
    _, base, sft = row(3, "Is liver present?", "c.jpg", True, True)
    assert MODULE.outcome_group(base, sft) == "stable_correct"
    _, base, sft = row(4, "Is liver present?", "d.jpg", False, False)
    assert MODULE.outcome_group(base, sft) is None
