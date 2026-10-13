from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

TOOL_ID = "mcp.support.lookup_ticket"
TICKETS = {
    "TCK-100": {"status": "open", "priority": "high"},
    "TCK-200": {"status": "closed", "priority": "low"},
}


@dataclass(frozen=True)
class EvalCase:
    eval_id: str
    kind: str
    prompt: str
    expected: str


@dataclass(frozen=True)
class EvalResult:
    eval_id: str
    passed: bool
    expected: str
    actual: str
    tool_calls: tuple[str, ...]
    duration_ms: float


EVALS = (
    EvalCase(
        "incident_triage",
        "agent",
        "Triage support ticket TCK-100 and report urgency.",
        "ESCALATE:TCK-100:high",
    ),
    EvalCase(
        "ticket_status",
        "agent",
        "Give the status of support ticket TCK-200.",
        "STATUS:TCK-200:closed",
    ),
    EvalCase("arithmetic", "local", "Add two and two.", "4"),
    EvalCase(
        "password_faq",
        "local",
        "How do I reset my password?",
        "Use the password-reset link.",
    ),
)


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("tool manifest must be an object")
    required = {"name", "description", "inputSchema"}
    if set(value) != required:
        raise ValueError(f"tool manifest fields must be exactly {sorted(required)}")
    if value["name"] != TOOL_ID:
        raise ValueError(f"unexpected tool name: {value['name']!r}")
    schema = value["inputSchema"]
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("inputSchema must describe an object")
    if schema.get("required") != ["ticket_id"]:
        raise ValueError("ticket_id must be the sole required input")
    return value


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def manifest_changes(
    before: dict[str, Any], after: dict[str, Any]
) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            changes.append(
                {
                    "dependency_id": TOOL_ID,
                    "path": f"tools.{before.get('name', after.get('name'))}.{key}",
                    "before_sha256": canonical_hash(before.get(key)),
                    "after_sha256": canonical_hash(after.get(key)),
                }
            )
    return changes


def _tokens(text: str) -> set[str]:
    stop = {"a", "an", "and", "by", "for", "i", "of", "the", "to", "use"}
    return {token for token in re.findall(r"[a-z]+", text.lower()) if token not in stop}


def _select_tool(prompt: str, manifest: dict[str, Any]) -> bool:
    return len(_tokens(prompt) & _tokens(str(manifest["description"]))) >= 2


def dispatch_mcp_tool(
    manifest: dict[str, Any], arguments: dict[str, str]
) -> dict[str, str]:
    required = manifest["inputSchema"]["required"]
    missing = [name for name in required if not arguments.get(name)]
    if missing:
        raise ValueError(f"missing required MCP arguments: {missing}")
    ticket_id = arguments["ticket_id"]
    if ticket_id not in TICKETS:
        raise KeyError(ticket_id)
    return {"ticket_id": ticket_id, **TICKETS[ticket_id]}


async def _run_eval(
    case: EvalCase, session: ClientSession, advertised_tool: Any
) -> EvalResult:
    started = perf_counter()
    calls: tuple[str, ...] = ()
    if case.eval_id == "arithmetic":
        actual = "4"
    elif case.eval_id == "password_faq":
        actual = "Use the password-reset link."
    elif not _select_tool(
        case.prompt,
        {
            "name": advertised_tool.name,
            "description": advertised_tool.description or "",
            "inputSchema": advertised_tool.input_schema,
        },
    ):
        actual = "NO_TOOL_SELECTED"
    else:
        match = re.search(r"TCK-\d+", case.prompt)
        if match is None:
            actual = "MISSING_TICKET_ID"
        else:
            response = await session.call_tool(
                advertised_tool.name, {"ticket_id": match.group(0)}
            )
            if response.is_error or not isinstance(response.structured_content, dict):
                actual = "MCP_TOOL_ERROR"
                return EvalResult(
                    case.eval_id,
                    False,
                    case.expected,
                    actual,
                    calls,
                    (perf_counter() - started) * 1000,
                )
            record = response.structured_content
            calls = (advertised_tool.name,)
            if case.eval_id == "incident_triage":
                actual = f"ESCALATE:{record['ticket_id']}:{record['priority']}"
            else:
                actual = f"STATUS:{record['ticket_id']}:{record['status']}"
    elapsed_ms = (perf_counter() - started) * 1000
    return EvalResult(
        case.eval_id, actual == case.expected, case.expected, actual, calls, elapsed_ms
    )


async def _run_evals_async(
    cases: tuple[EvalCase, ...], manifest_path: Path
) -> dict[str, Any]:
    started = perf_counter()
    server_path = Path(__file__).with_name("mcp_tool_server.py")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path), str(manifest_path.resolve())],
        cwd=server_path.parent,
    )
    with open(os.devnull, "w", encoding="utf-8") as errlog:  # noqa: ASYNC230
        async with stdio_client(parameters, errlog=errlog) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                listed = await session.list_tools()
                if len(listed.tools) != 1:
                    raise RuntimeError(
                        f"expected one MCP tool, received {len(listed.tools)}"
                    )
                advertised_tool = listed.tools[0]
                results = tuple(
                    [await _run_eval(case, session, advertised_tool) for case in cases]
                )
    return {
        "count": len(results),
        "passed": sum(result.passed for result in results),
        "failed": sum(not result.passed for result in results),
        "duration_ms": round((perf_counter() - started) * 1000, 3),
        "mcp_transport": "stdio",
        "mcp_protocol_version": initialized.protocol_version,
        "advertised_tool": {
            "name": advertised_tool.name,
            "description": advertised_tool.description,
            "inputSchema": advertised_tool.input_schema,
        },
        "results": [asdict(result) for result in results],
    }


def run_evals(cases: tuple[EvalCase, ...], manifest_path: Path) -> dict[str, Any]:
    return anyio.run(_run_evals_async, cases, manifest_path)


def dependency_index(baseline_results: dict[str, Any]) -> dict[str, list[str]]:
    index: dict[str, set[str]] = {}
    for result in baseline_results["results"]:
        for dependency_id in result["tool_calls"]:
            index.setdefault(dependency_id, set()).add(result["eval_id"])
    return {key: sorted(value) for key, value in sorted(index.items())}


def select_affected(
    changes: list[dict[str, str]], index: dict[str, list[str]]
) -> tuple[EvalCase, ...]:
    ids = {
        eval_id
        for change in changes
        for eval_id in index.get(change["dependency_id"], [])
    }
    return tuple(case for case in EVALS if case.eval_id in ids)


def build_abl_workflow(repaired_results: dict[str, Any]) -> dict[str, Any]:
    incident = next(
        result
        for result in repaired_results["results"]
        if result["eval_id"] == "incident_triage"
    )
    if not incident["passed"] or tuple(incident["tool_calls"]) != (TOOL_ID,):
        raise ValueError(
            "repaired incident workflow is not eligible for ABL verification"
        )
    return {
        "workflow_id": "dependency_killer_repaired_incident_triage",
        "title": "Repaired deterministic MCP ticket-triage workflow",
        "research_goal": "Verify the recorded repaired workflow and its local fixture lineage.",
        "events": [
            {
                "event_id": "lookup_ticket",
                "event_type": "mcp_tool_call",
                "action": "look up support ticket TCK-100",
                "tool": TOOL_ID,
                "executed": True,
                "evidence_refs": ["ticket_record"],
                "observation": "bounded local fixture record returned",
                "sequence": 1,
            },
            {
                "event_id": "produce_triage",
                "event_type": "agent_response",
                "action": "produce deterministic triage result",
                "tool": "deterministic_agent_router",
                "executed": True,
                "evidence_refs": ["ticket_record"],
                "observation": incident["actual"],
                "sequence": 2,
                "parent_event_id": "lookup_ticket",
            },
        ],
        "sources": [
            {
                "source_id": "local_ticket_fixture",
                "source_type": "local_fixture",
                "locator": "fixture:TCK-100",
                "title": "Bounded synthetic ticket store",
            }
        ],
        "evidence": [
            {
                "evidence_id": "ticket_record",
                "source_refs": ["local_ticket_fixture"],
                "produced_by_event": "lookup_ticket",
            }
        ],
        "claims": [
            {
                "claim_id": "triage_result",
                "text": incident["actual"],
                "evidence_refs": ["ticket_record"],
            }
        ],
        "artifacts": [
            {
                "artifact_id": "triage_output",
                "artifact_type": "agent_response",
                "artifact_role": "FINAL_OUTPUT",
                "produced_by_event": "produce_triage",
                "evidence_refs": ["ticket_record"],
            }
        ],
        "assurance_context": {
            "events": [
                {
                    "event_id": "lookup_ticket",
                    "governance_decision": "ALLOW",
                    "policy_ref": "local-demo-policy-v1",
                    "approval_requirement": "NOT_REQUIRED",
                },
                {
                    "event_id": "produce_triage",
                    "governance_decision": "ALLOW",
                    "policy_ref": "local-demo-policy-v1",
                    "approval_requirement": "NOT_REQUIRED",
                },
            ]
        },
    }
