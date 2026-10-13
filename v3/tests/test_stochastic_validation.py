from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stochastic_validation import compare_trial, frozen_cases, frozen_target_cases  # noqa: E402


def test_frozen_target_selection_uses_only_stored_mapping() -> None:
    frozen = {
        "baseline_trace_index": {"deepagents.filesystem.read_file": ["a"]},
        "impacted_eval_mapping": {"mutation": ["a"]},
        "corpus": [
            {
                "eval_id": "a",
                "prompt": "read",
                "files": [["/a", "A"]],
                "expected_substring": "A",
            },
            {
                "eval_id": "b",
                "prompt": "other",
                "files": [["/b", "B"]],
                "expected_substring": "B",
            },
        ],
    }
    frozen["baseline_trace_index"]["deepagents.filesystem.ls"] = ["b"]
    cases = frozen_cases(frozen)

    assert [
        case.eval_id for case in frozen_target_cases(frozen, cases, "mutation")
    ] == ["a"]


def test_compare_trial_preserves_false_negative_and_false_positive() -> None:
    baseline = {
        "results": [
            {
                "eval_id": "a",
                "passed": True,
                "expected_tool": "read_file",
                "selected_tools": ["read_file"],
            },
            {
                "eval_id": "b",
                "passed": True,
                "expected_tool": "ls",
                "selected_tools": ["ls"],
            },
        ]
    }
    full = {
        "seed": 1,
        "results": [
            {"eval_id": "a", "passed": False, "selected_tools": ["read_file"]},
            {"eval_id": "b", "passed": True, "selected_tools": ["ls"]},
        ],
    }
    targeted = {
        "results": [
            {"eval_id": "a", "passed": True},
            {"eval_id": "b", "passed": False},
        ]
    }

    result = compare_trial(
        baseline,
        full,
        targeted,
        frozen_target_ids={"a", "b"},
        changed_dependency="deepagents.filesystem.read_file",
    )

    assert result["false_negative_ids"] == ["a"]
    assert result["false_positive_ids"] == ["b"]
    assert result["root_cause_correct_ids"] == ["a"]
