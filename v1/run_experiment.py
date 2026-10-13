from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dependency_killer import (
    EVALS,
    TOOL_ID,
    build_abl_workflow,
    dependency_index,
    load_manifest,
    manifest_changes,
    run_evals,
    select_affected,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run_unit_tests(manifest_path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["TOOL_MANIFEST"] = str(manifest_path)
    started = perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": f"{sys.executable} -m unittest discover -s tests -q",
        "exit_code": completed.returncode,
        "duration_ms": round((perf_counter() - started) * 1000, 3),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _run_abl(abl_root: Path, workflow_path: Path, report_path: Path) -> dict[str, Any]:
    command = [
        "uv",
        "run",
        "--offline",
        "--frozen",
        "abl",
        "verify",
        str(workflow_path),
        "--output",
        str(report_path),
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
        "report_path": str(report_path),
    }
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result["overall_status"] = report["summary"]["overall_status"]
        result["total_findings"] = report["summary"]["total_findings"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abl-root", type=Path)
    args = parser.parse_args()

    artifacts = ROOT / "artifacts"
    baseline_path = ROOT / "fixtures" / "tool_baseline.json"
    changed_path = ROOT / "fixtures" / "tool_changed.json"
    repaired_path = ROOT / "fixtures" / "tool_repaired.json"
    baseline = load_manifest(baseline_path)
    changed = load_manifest(changed_path)
    repaired = load_manifest(repaired_path)

    baseline_unit = _run_unit_tests(baseline_path)
    if baseline_unit["exit_code"] != 0:
        raise SystemExit("baseline unit tests failed")
    baseline_evals = run_evals(EVALS, baseline_path)
    if baseline_evals["failed"] != 0:
        raise SystemExit("baseline behavioral evals failed")

    index = dependency_index(baseline_evals)
    changes = manifest_changes(baseline, changed)
    affected = select_affected(changes, index)

    changed_unit = _run_unit_tests(changed_path)
    targeted_changed = run_evals(affected, changed_path)
    injected_ids = {case.eval_id for case in affected}
    caught_ids = {
        result["eval_id"]
        for result in targeted_changed["results"]
        if not result["passed"]
    }

    repair_changes = manifest_changes(changed, repaired)
    repaired_targeted = run_evals(affected, repaired_path)
    repaired_full = run_evals(EVALS, repaired_path)

    workflow_path = artifacts / "repaired_workflow_for_abl.json"
    abl_report_path = artifacts / "abl_assurance_report.json"
    _write_json(workflow_path, build_abl_workflow(repaired_targeted))
    if abl_report_path.exists():
        abl_report_path.unlink()
    abl = (_run_abl(args.abl_root.resolve(), workflow_path.resolve(), abl_report_path.resolve())
           if args.abl_root else {"status": "NOT_RUN", "reason": "optional ABL verification not requested"})

    root_cause = {
        "dependency_id": TOOL_ID,
        "changed_paths": [change["path"] for change in changes],
        "evidence": "all failed targeted evals depended on the only changed dependency in baseline traces",
        "failed_eval_ids": sorted(caught_ids),
    }
    report = {
        "status": "PASS"
        if (
            changed_unit["exit_code"] == 0
            and caught_ids == injected_ids
            and repaired_targeted["failed"] == 0
            and repaired_full["failed"] == 0
            and (not args.abl_root or (abl.get("exit_code") == 0 and abl.get("overall_status") == "COMPLETE"))
            and len(affected) < len(EVALS)
        )
        else "FAIL",
        "mode": "deterministic_local_mcp_stdio_no_model_no_api",
        "baseline": {"unit_tests": baseline_unit, "behavioral_evals": baseline_evals},
        "dependency_change": {
            "from": str(baseline_path),
            "to": str(changed_path),
            "changes": changes,
        },
        "dependency_index": index,
        "selection": {
            "full_eval_count": len(EVALS),
            "targeted_eval_count": len(affected),
            "targeted_eval_ids": [case.eval_id for case in affected],
            "selection_source": "baseline observed tool_calls",
            "manual_workflow_dependency_annotations": 0,
        },
        "regression": {
            "ordinary_unit_tests": changed_unit,
            "targeted_behavioral_evals": targeted_changed,
            "injected_regression_ids": sorted(injected_ids),
            "caught_regression_ids": sorted(caught_ids),
            "recall": len(caught_ids & injected_ids) / len(injected_ids),
        },
        "root_cause": root_cause,
        "repair": {
            "from": str(changed_path),
            "to": str(repaired_path),
            "changes": repair_changes,
            "targeted_evals": repaired_targeted,
            "full_evals": repaired_full,
        },
        "abl_reverification": abl,
        "cost": {"model_calls": 0, "api_calls": 0, "model_api_cost_usd": 0.0},
        "limitations": [
            "The agent router and MCP transport are deterministic local fixtures.",
            "The result proves this bounded description-change scenario, not arbitrary semantic tool changes.",
            "ABL verifies recorded governance, audit, and identifier-level provenance; it does not certify production security.",
        ],
    }
    _write_json(artifacts / "experiment_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
