from summarize_reference_evidence_causal_results import dual_summary


def _record(with_controls: bool) -> dict:
    stream = {"raw_margin_recovery": 0.5, "recovery_fraction": 0.75}
    if with_controls:
        stream.update({
            "correct_minus_permuted_margin_recovery": 0.4,
            "correct_minus_opposite_margin_recovery": 0.6,
        })
    return {
        "patched_by_layer": {
            "0": {
                "visual_token_patch": dict(stream),
                "query_position_patch": dict(stream),
            }
        }
    }


def test_dual_summary_accepts_legacy_records_without_direction_controls() -> None:
    summary = dual_summary([_record(with_controls=False)])
    visual = summary["by_layer"]["0"]["visual_token_patch"]
    assert visual["raw_margin_recovery"]["mean"] == 0.5
    assert "correct_minus_permuted_margin_recovery" not in visual


def test_dual_summary_includes_direction_controls_when_present() -> None:
    summary = dual_summary([_record(with_controls=True)])
    visual = summary["by_layer"]["0"]["visual_token_patch"]
    assert visual["correct_minus_permuted_margin_recovery"]["mean"] == 0.4
    assert visual["correct_minus_opposite_margin_recovery"]["mean"] == 0.6
