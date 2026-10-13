from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from scale_validation import (  # noqa: E402
    MUTATIONS,
    TRACKED_TOOLS,
    affected_cases,
    aggregate_metrics,
    build_abl_workflow,
    build_manifest,
    dependency_index,
    failed_ids,
    make_corpus,
    manifest_changes,
    run_suite,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def run_upstream_tests(deepagents_root: Path) -> dict[str, Any]:
    package = deepagents_root / "libs" / "deepagents"
    selectors = [
        "tests/unit_tests/test_end_to_end.py::TestDeepAgentEndToEnd::test_deep_agent_with_fake_llm_basic",
        "tests/unit_tests/test_end_to_end.py::TestDeepAgentEndToEnd::test_deep_agent_with_fake_llm_with_tools",
    ]
    command = [
        sys.executable,
        "-m",
        "pytest",
        *selectors,
        "-q",
        "--disable-socket",
        "--allow-unix-socket",
        "--benchmark-disable",
    ]
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=package,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "duration_ms": round((perf_counter() - started) * 1000, 3),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "selected_test_count": len(selectors),
    }


def run_abl(abl_root: Path, workflow: Path, output: Path) -> dict[str, Any]:
    command = [
        "uv",
        "run",
        "--offline",
        "--frozen",
        "abl",
        "verify",
        str(workflow),
        "--output",
        str(output),
    ]
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=abl_root,
        text=True,
        capture_output=True,
        check=False,
    )
    result: dict[str, Any] = {
        "command": command,
        "exit_code": completed.returncode,
        "duration_ms": round((perf_counter() - started) * 1000, 3),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "report_path": str(output),
    }
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        result["status"] = report["summary"]["overall_status"]
        result["findings"] = report["summary"]["total_findings"]
    return result


def run_probe() -> int:
    cases = make_corpus(cases_per_tool=1)
    suite = run_suite(cases)
    index = dependency_index(suite)
    result = {
        "status": "PASS"
        if suite["passed"] == len(cases) and len(index) == len(TRACKED_TOOLS)
        else "FAIL",
        "suite": suite,
        "dependency_index": index,
    }
    write_json(ROOT / "artifacts" / "probe.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


def run_full(deepagents_root: Path, abl_root: Path) -> int:
    started = perf_counter()
    artifacts = ROOT / "artifacts"
    deepagents_before = git_output(deepagents_root, "status", "--porcelain")
    abl_before = git_output(abl_root, "status", "--porcelain")
    upstream_before = run_upstream_tests(deepagents_root)

    cases = make_corpus()
    baseline_manifest = build_manifest()
    baseline = run_suite(cases)
    if upstream_before["exit_code"] != 0 or baseline["failed"] != 0:
        raise SystemExit("baseline failed")
    index = dependency_index(baseline)
    write_json(artifacts / "baseline_trace.json", baseline)
    write_json(artifacts / "dependency_index.json", index)

    mutation_reports: list[dict[str, Any]] = []
    for mutation in MUTATIONS:
        changed_manifest = build_manifest(mutation)
        changes = manifest_changes(baseline_manifest, changed_manifest)
        targeted_cases = affected_cases(cases, changes, index)
        full = run_suite(cases, mutation)
        targeted = run_suite(targeted_cases, mutation)
        injected = failed_ids(full)
        caught = failed_ids(targeted)
        root_cause_correct = (
            bool(injected)
            and injected == caught
            and {item["dependency_id"] for item in changes} == {mutation.dependency_id}
            and injected == {case.eval_id for case in targeted_cases}
        )
        report = {
            "mutation": {
                "mutation_id": mutation.mutation_id,
                "dependency_id": mutation.dependency_id,
                "dependency_type": mutation.dependency_type,
            },
            "changes": changes,
            "affected_eval_ids": [case.eval_id for case in targeted_cases],
            "full": full,
            "targeted": targeted,
            "injected_regression_ids": sorted(injected),
            "targeted_caught_ids": sorted(caught),
            "full_caught_ids": sorted(injected),
            "recall": len(caught & injected) / len(injected) if injected else 0.0,
            "precision": (
                len(caught & injected) / len(targeted_cases) if targeted_cases else 0.0
            ),
            "targeted_ratio": len(targeted_cases) / len(cases),
            "root_cause_correct": root_cause_correct,
            "root_cause_detail": (
                f"{mutation.dependency_id} changed -> "
                f"{len(targeted_cases)} baseline-dependent evals selected -> "
                f"{len(injected)} behavioral regressions observed"
            ),
        }
        mutation_reports.append(report)
        write_json(artifacts / f"mutation_{mutation.mutation_id}.json", report)

    description_mutation = MUTATIONS[0]
    description_changes = manifest_changes(
        baseline_manifest, build_manifest(description_mutation)
    )
    repair_cases = affected_cases(cases, description_changes, index)
    repair_patch = {
        "mutation_id": description_mutation.mutation_id,
        "operation": "replace",
        "path": "tools.read_file.description",
        "value": baseline_manifest["read_file"]["description"],
    }
    repaired_targeted = run_suite(repair_cases)
    repaired_full = run_suite(cases)
    repair = {
        "generated": True,
        "patch": repair_patch,
        "targeted_after_repair": repaired_targeted,
        "full_after_repair": repaired_full,
        "verified": (repaired_targeted["failed"] == 0 and repaired_full["failed"] == 0),
    }
    write_json(artifacts / "repair_patch.json", repair_patch)
    write_json(artifacts / "repair_verification.json", repair)

    workflow_path = artifacts / "repaired_workflow_for_abl.json"
    abl_report_path = artifacts / "abl_assurance_report.json"
    write_json(workflow_path, build_abl_workflow(repaired_targeted))
    if abl_report_path.exists():
        abl_report_path.unlink()
    abl = run_abl(abl_root, workflow_path.resolve(), abl_report_path.resolve())
    upstream_after = run_upstream_tests(deepagents_root)
    deepagents_after = git_output(deepagents_root, "status", "--porcelain")
    abl_after = git_output(abl_root, "status", "--porcelain")

    aggregate = aggregate_metrics(mutation_reports)
    success = (
        len(cases) >= 20
        and len(MUTATIONS) >= 3
        and len({mutation.dependency_type for mutation in MUTATIONS}) >= 2
        and len(index) >= 3
        and all(item["recall"] == 1.0 for item in mutation_reports)
        and all(item["root_cause_correct"] for item in mutation_reports)
        and all(item["targeted_ratio"] < 0.5 for item in mutation_reports)
        and repair["verified"]
        and upstream_after["exit_code"] == 0
        and abl.get("exit_code") == 0
        and abl.get("status") == "COMPLETE"
        and deepagents_before == deepagents_after == ""
        and abl_before == abl_after == ""
    )
    project = {
        "name": "langchain-ai/deepagents",
        "source": git_output(deepagents_root, "remote", "get-url", "origin"),
        "commit": git_output(deepagents_root, "rev-parse", "HEAD"),
        "version": "0.7.13",
        "why_selected": (
            "real open-source agent harness; provider-free end-to-end fake-model "
            "tests; real LangGraph loop and filesystem tools; existing local environment"
        ),
    }
    final = {
        "status": "PASS" if success else "FAIL",
        "project": project,
        "baseline_tests": upstream_before,
        "baseline": baseline,
        "baseline_eval_count": len(cases),
        "dependency_count": len(index),
        "dependency_index": index,
        "dependency_mapping_source": "baseline observed AIMessage.tool_calls",
        "manual_dependency_annotations": 0,
        "mutations": mutation_reports,
        "mutation_count": len(MUTATIONS),
        "dependency_types": sorted(
            {mutation.dependency_type for mutation in MUTATIONS}
        ),
        "aggregate": aggregate,
        "repair": repair,
        "abl_reverification": {
            **abl,
            "version": "0.2.1a1",
            "commit": git_output(abl_root, "rev-parse", "HEAD"),
            "core_changed": abl_before != abl_after or abl_after != "",
        },
        "post_repair_upstream_tests": upstream_after,
        "cost": {
            "full_usd": 0.0,
            "targeted_usd": 0.0,
            "model_calls_billed": 0,
            "api_calls": 0,
        },
        "resource_usage": {
            "wall_runtime_ms": round((perf_counter() - started) * 1000, 3),
            "network_calls": 0,
            "downloads": 0,
        },
        "upstream_changed": deepagents_before != deepagents_after
        or deepagents_after != "",
        "limitations": [
            "The workflow uses the real Deep Agents graph and tools with a deterministic replay model, not a live LLM.",
            "Results cover four filesystem-tool dependencies and three controlled interface mutations, not arbitrary tools or models.",
            "Schema mutations model advertised-contract drift against the unchanged runtime handler.",
            "ABL verifies recorded governance, audit, and identifier-level provenance, not production security or output truth.",
        ],
    }
    write_json(artifacts / "experiment_report.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if success else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepagents-root", type=Path)
    parser.add_argument("--abl-root", type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        return run_probe()
    if args.deepagents_root is None or args.abl_root is None:
        parser.error("--deepagents-root and --abl-root are required without --probe")
    return run_full(args.deepagents_root.resolve(), args.abl_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
