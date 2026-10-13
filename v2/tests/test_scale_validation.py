from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scale_validation import (  # noqa: E402
    MUTATIONS,
    _arguments,
    affected_cases,
    build_manifest,
    dependency_index,
    make_corpus,
    manifest_changes,
)


def test_wrong_route_is_recordable_instead_of_crashing() -> None:
    arguments = _arguments(
        "grep",
        "Read file '/datasets/read_0/note.txt' and return its contents.",
        {"required": ["pattern"]},
    )

    assert arguments == {
        "pattern": "/datasets/read_0/note.txt",
        "path": "/datasets/read_0/note.txt",
    }


def test_corpus_has_twenty_four_unannotated_cases() -> None:
    cases = make_corpus()

    assert len(cases) == 24
    assert all(not hasattr(case, "dependency_id") for case in cases)


def test_manifest_detects_description_change() -> None:
    changes = manifest_changes(build_manifest(), build_manifest(MUTATIONS[0]))

    assert [change["dependency_id"] for change in changes] == [
        "deepagents.filesystem.read_file"
    ]
    assert changes[0]["path"] == "tools.read_file.description"


def test_manifest_detects_schema_change() -> None:
    changes = manifest_changes(build_manifest(), build_manifest(MUTATIONS[1]))

    assert [change["dependency_id"] for change in changes] == [
        "deepagents.filesystem.glob"
    ]
    assert changes[0]["path"] == "tools.glob.input_schema"


def test_dependency_mapping_comes_from_observed_calls() -> None:
    baseline = {
        "results": [
            {"eval_id": "a", "tool_calls": ["read_file"]},
            {"eval_id": "b", "tool_calls": ["glob"]},
            {"eval_id": "c", "tool_calls": ["read_file"]},
        ]
    }

    assert dependency_index(baseline) == {
        "deepagents.filesystem.glob": ["b"],
        "deepagents.filesystem.read_file": ["a", "c"],
    }


def test_affected_selection_uses_change_and_trace_index() -> None:
    cases = make_corpus(cases_per_tool=1)
    changes = manifest_changes(build_manifest(), build_manifest(MUTATIONS[2]))
    selected = affected_cases(
        cases,
        changes,
        {"deepagents.filesystem.grep": ["grep_literal_00"]},
    )

    assert [case.eval_id for case in selected] == ["grep_literal_00"]
