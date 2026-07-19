#!/usr/bin/env python3
"""CUDA-independent frozen-evidence tests for formal-only Stage 2 continuation."""

from __future__ import annotations

from formal_machine import run_stage2_formal_only as formal


protocol = formal.verify_continuation_protocol()
assert protocol["verified"]["pilot_rerun_authorized"] is False
evidence = formal.verify_passing_pilot_and_pruning()
assert evidence["pilot_gate_passed"] is True
assert evidence["pilot_validation_correct"] == 232
assert evidence["checkpoint_pruned"] is True
assert evidence["formal_directory_absent"] is True
storage = formal.verify_storage()
assert storage["passed"] is True
assert storage["margin_bytes"] > 0
print("Stage-2 formal-only continuation tests: PASS")
