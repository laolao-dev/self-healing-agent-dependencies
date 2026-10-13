from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import median
from time import perf_counter
from typing import Any

from deepagents.backends.state import StateBackend
from deepagents.backends.utils import create_file_data
from deepagents.graph import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


TRACKED_TOOLS = ("ls", "read_file", "glob", "grep")
DEPENDENCY_PREFIX = "deepagents.filesystem."


@dataclass(frozen=True)
class EvalCase:
    eval_id: str
    prompt: str
    files: tuple[tuple[str, str], ...]
    expected_substring: str


@dataclass(frozen=True)
class Mutation:
    mutation_id: str
    dependency_id: str
    dependency_type: str
    description_override: str | None = None
    schema_rename: tuple[str, str] | None = None


@dataclass(frozen=True)
class EvalResult:
    eval_id: str
    passed: bool
    expected_substring: str
    tool_calls: tuple[str, ...]
    tool_arguments: tuple[dict[str, object], ...]
    tool_statuses: tuple[str, ...]
    tool_output: str
    duration_ms: float


MUTATIONS = (
    Mutation(
        mutation_id="read_file_description_semantic_narrowing",
        dependency_id=f"{DEPENDENCY_PREFIX}read_file",
        dependency_type="tool_description",
        description_override=(
            "Loads a binary media asset from an opaque asset locator for preview "
            "generation."
        ),
    ),
    Mutation(
        mutation_id="glob_schema_pattern_to_query",
        dependency_id=f"{DEPENDENCY_PREFIX}glob",
        dependency_type="tool_schema",
        schema_rename=("pattern", "query"),
    ),
    Mutation(
        mutation_id="grep_schema_pattern_to_search_text",
        dependency_id=f"{DEPENDENCY_PREFIX}grep",
        dependency_type="tool_schema",
        schema_rename=("pattern", "search_text"),
    ),
)


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_corpus(cases_per_tool: int = 6) -> tuple[EvalCase, ...]:
    cases: list[EvalCase] = []
    for index in range(cases_per_tool):
        cases.extend(
            [
                EvalCase(
                    eval_id=f"list_directory_{index:02d}",
                    prompt=(
                        f"List directory '/datasets/list_{index}' and report its files."
                    ),
                    files=((f"/datasets/list_{index}/item_{index}.txt", "listed"),),
                    expected_substring=f"item_{index}.txt",
                ),
                EvalCase(
                    eval_id=f"read_file_{index:02d}",
                    prompt=(
                        f"Read file '/datasets/read_{index}/note.txt' and return "
                        "its contents."
                    ),
                    files=(
                        (
                            f"/datasets/read_{index}/note.txt",
                            f"READ_MARKER_{index}",
                        ),
                    ),
                    expected_substring=f"READ_MARKER_{index}",
                ),
                EvalCase(
                    eval_id=f"glob_markdown_{index:02d}",
                    prompt=(
                        f"Find files matching glob pattern '*.md' under "
                        f"'/datasets/glob_{index}'."
                    ),
                    files=(
                        (f"/datasets/glob_{index}/match_{index}.md", "match"),
                        (f"/datasets/glob_{index}/skip_{index}.txt", "skip"),
                    ),
                    expected_substring=f"match_{index}.md",
                ),
                EvalCase(
                    eval_id=f"grep_literal_{index:02d}",
                    prompt=(
                        f"Search literal text 'needle_{index}' across files under "
                        f"'/datasets/grep_{index}'."
                    ),
                    files=(
                        (
                            f"/datasets/grep_{index}/data_{index}.txt",
                            f"prefix needle_{index} suffix",
                        ),
                    ),
                    expected_substring=f"data_{index}.txt",
                ),
            ]
        )
    return tuple(cases)


def _tool_spec(tool: BaseTool | dict[str, Any] | type | Callable) -> dict[str, Any]:
    converted = convert_to_openai_tool(tool)
    function = converted.get("function", converted)
    return {
        "name": str(function["name"]),
        "description": str(function.get("description", "")),
        "input_schema": copy.deepcopy(function.get("parameters", {})),
    }


def _rename_schema_field(
    schema: dict[str, Any], old_name: str, new_name: str
) -> dict[str, Any]:
    changed = copy.deepcopy(schema)
    properties = changed.get("properties")
    if not isinstance(properties, dict) or old_name not in properties:
        msg = f"schema does not expose field {old_name!r}"
        raise ValueError(msg)
    properties[new_name] = properties.pop(old_name)
    required = changed.get("required", [])
    if isinstance(required, list):
        changed["required"] = [
            new_name if item == old_name else item for item in required
        ]
    return changed


def build_manifest(mutation: Mutation | None = None) -> dict[str, dict[str, Any]]:
    descriptions: dict[str, str] = {}
    if mutation is not None and mutation.description_override is not None:
        descriptions[mutation.dependency_id.removeprefix(DEPENDENCY_PREFIX)] = (
            mutation.description_override
        )
    middleware = FilesystemMiddleware(
        backend=StateBackend(),
        tools=list(TRACKED_TOOLS),
        custom_tool_descriptions=descriptions,
    )
    manifest = {tool.name: _tool_spec(tool) for tool in middleware.tools}
    if mutation is not None and mutation.schema_rename is not None:
        tool_name = mutation.dependency_id.removeprefix(DEPENDENCY_PREFIX)
        old_name, new_name = mutation.schema_rename
        manifest[tool_name]["input_schema"] = _rename_schema_field(
            manifest[tool_name]["input_schema"], old_name, new_name
        )
    return manifest


def manifest_changes(
    baseline: dict[str, dict[str, Any]], changed: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    for tool_name in sorted(set(baseline) | set(changed)):
        before = baseline.get(tool_name, {})
        after = changed.get(tool_name, {})
        for field in ("description", "input_schema"):
            if before.get(field) != after.get(field):
                changes.append(
                    {
                        "dependency_id": f"{DEPENDENCY_PREFIX}{tool_name}",
                        "path": f"tools.{tool_name}.{field}",
                        "before_sha256": canonical_hash(before.get(field)),
                        "after_sha256": canonical_hash(after.get(field)),
                    }
                )
    return changes


def _tokens(text: str) -> set[str]:
    stop = {
        "a",
        "all",
        "and",
        "for",
        "from",
        "in",
        "its",
        "of",
        "the",
        "this",
        "to",
        "under",
    }
    values: set[str] = set()
    for token in re.findall(r"[a-z]+", text.lower()):
        if token in stop:
            continue
        values.add(token[:-1] if token.endswith("s") and len(token) > 3 else token)
    return values


def _quoted(prompt: str) -> list[str]:
    return re.findall(r"'([^']+)'", prompt)


def _required_key(schema: dict[str, Any], fallback: str) -> str:
    required = schema.get("required")
    if isinstance(required, list) and required and isinstance(required[0], str):
        return required[0]
    return fallback


def _arguments(tool_name: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    quoted = _quoted(prompt)
    if tool_name == "ls":
        return {_required_key(schema, "path"): quoted[0]}
    if tool_name == "read_file":
        return {_required_key(schema, "file_path"): quoted[0]}
    if tool_name in {"glob", "grep"}:
        # A dependency regression may route a one-argument prompt to a
        # two-argument tool. Keep the replay evaluator total so that the real
        # tool node records the bad behavior instead of crashing the suite.
        search_root = quoted[1] if len(quoted) > 1 else quoted[0]
        return {
            _required_key(schema, "pattern"): quoted[0],
            "path": search_root,
        }
    return {}


class RoutingReplayModel(BaseChatModel):
    """Deterministic tool-selection replay model over real bound tool contracts."""

    schema_renames: dict[str, tuple[str, str]] = Field(default_factory=dict)
    advertised_tools: dict[str, dict[str, Any]] = Field(
        default_factory=dict, exclude=True
    )

    @property
    def _llm_type(self) -> str:
        return "deepagents-dependency-replay"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        del tool_choice, kwargs
        manifest = {
            spec["name"]: spec
            for spec in (_tool_spec(tool) for tool in tools)
            if spec["name"] in TRACKED_TOOLS
        }
        for tool_name, (old_name, new_name) in self.schema_renames.items():
            manifest[tool_name]["input_schema"] = _rename_schema_field(
                manifest[tool_name]["input_schema"], old_name, new_name
            )
        self.advertised_tools = manifest
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        if messages and isinstance(messages[-1], ToolMessage):
            message = AIMessage(content="WORKFLOW_COMPLETE")
            return ChatResult(generations=[ChatGeneration(message=message)])
        prompt = next(
            str(message.content)
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )
        prompt_tokens = _tokens(prompt)
        scored = sorted(
            (
                len(prompt_tokens & _tokens(str(spec["description"]))),
                name,
                spec,
            )
            for name, spec in self.advertised_tools.items()
        )
        score, name, spec = scored[-1]
        if score < 2:
            message = AIMessage(content="NO_TOOL_SELECTED")
        else:
            arguments = _arguments(name, prompt, spec["input_schema"])
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": arguments,
                        "id": "dependency-replay-call",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, default=str)


def _run_case(agent: Any, case: EvalCase) -> EvalResult:
    started = perf_counter()
    state_files = {path: create_file_data(content) for path, content in case.files}
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=case.prompt)],
            "files": state_files,
        }
    )
    messages = result["messages"]
    calls = tuple(
        call
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    )
    tool_messages = tuple(
        message for message in messages if isinstance(message, ToolMessage)
    )
    output = "\n".join(_content_text(message.content) for message in tool_messages)
    statuses = tuple(
        str(getattr(message, "status", "success")) for message in tool_messages
    )
    passed = (
        len(calls) == 1
        and len(tool_messages) == 1
        and "error" not in statuses
        and case.expected_substring in output
    )
    return EvalResult(
        eval_id=case.eval_id,
        passed=passed,
        expected_substring=case.expected_substring,
        tool_calls=tuple(str(call["name"]) for call in calls),
        tool_arguments=tuple(dict(call["args"]) for call in calls),
        tool_statuses=statuses,
        tool_output=output,
        duration_ms=round((perf_counter() - started) * 1000, 3),
    )


def run_suite(
    cases: tuple[EvalCase, ...], mutation: Mutation | None = None
) -> dict[str, Any]:
    descriptions: dict[str, str] = {}
    schema_renames: dict[str, tuple[str, str]] = {}
    if mutation is not None:
        tool_name = mutation.dependency_id.removeprefix(DEPENDENCY_PREFIX)
        if mutation.description_override is not None:
            descriptions[tool_name] = mutation.description_override
        if mutation.schema_rename is not None:
            schema_renames[tool_name] = mutation.schema_rename
    backend = StateBackend()
    filesystem = FilesystemMiddleware(
        backend=backend,
        tools=list(TRACKED_TOOLS),
        custom_tool_descriptions=descriptions,
    )
    model = RoutingReplayModel(schema_renames=schema_renames)
    agent = create_deep_agent(
        model=model,
        backend=backend,
        middleware=[filesystem],
        system_prompt=(
            "Use the available filesystem tools to satisfy exactly one bounded local "
            "request."
        ),
    )
    started = perf_counter()
    results = tuple(_run_case(agent, case) for case in cases)
    return {
        "count": len(results),
        "passed": sum(result.passed for result in results),
        "failed": sum(not result.passed for result in results),
        "duration_ms": round((perf_counter() - started) * 1000, 3),
        "model_calls_cost_usd": 0.0,
        "mode": "deepagents_real_graph_deterministic_replay_model",
        "advertised_tools": model.advertised_tools,
        "results": [asdict(result) for result in results],
    }


def dependency_index(baseline_results: dict[str, Any]) -> dict[str, list[str]]:
    index: dict[str, set[str]] = {}
    for result in baseline_results["results"]:
        for tool_name in result["tool_calls"]:
            dependency_id = f"{DEPENDENCY_PREFIX}{tool_name}"
            index.setdefault(dependency_id, set()).add(result["eval_id"])
    return {key: sorted(value) for key, value in sorted(index.items())}


def affected_cases(
    cases: tuple[EvalCase, ...],
    changes: list[dict[str, str]],
    index: dict[str, list[str]],
) -> tuple[EvalCase, ...]:
    affected_ids = {
        eval_id
        for change in changes
        for eval_id in index.get(change["dependency_id"], [])
    }
    return tuple(case for case in cases if case.eval_id in affected_ids)


def failed_ids(suite: dict[str, Any]) -> set[str]:
    return {result["eval_id"] for result in suite["results"] if not result["passed"]}


def aggregate_metrics(mutations: list[dict[str, Any]]) -> dict[str, float]:
    full_times = [item["full"]["duration_ms"] for item in mutations]
    targeted_times = [item["targeted"]["duration_ms"] for item in mutations]
    injected = sum(len(item["injected_regression_ids"]) for item in mutations)
    caught = sum(len(item["targeted_caught_ids"]) for item in mutations)
    targeted_count = sum(item["targeted"]["count"] for item in mutations)
    median_full = median(full_times)
    median_targeted = median(targeted_times)
    return {
        "median_full_runtime_ms": round(median_full, 3),
        "median_targeted_runtime_ms": round(median_targeted, 3),
        "runtime_reduction": round(1 - (median_targeted / median_full), 6),
        "regression_recall": caught / injected if injected else 0.0,
        "precision": caught / targeted_count if targeted_count else 0.0,
    }


def build_abl_workflow(repaired_suite: dict[str, Any]) -> dict[str, Any]:
    result = next(
        item for item in repaired_suite["results"] if item["eval_id"] == "read_file_00"
    )
    if not result["passed"] or tuple(result["tool_calls"]) != ("read_file",):
        msg = "repaired read_file workflow is not eligible for ABL verification"
        raise ValueError(msg)
    return {
        "workflow_id": "deepagents_dependency_v2_repaired_read_file",
        "title": "Repaired Deep Agents filesystem workflow",
        "research_goal": (
            "Verify the recorded repaired tool workflow and local fixture lineage."
        ),
        "events": [
            {
                "event_id": "read_fixture",
                "event_type": "function_call",
                "action": "read bounded local fixture",
                "tool": "deepagents.filesystem.read_file",
                "executed": True,
                "evidence_refs": ["read_result"],
                "observation": "READ_MARKER_0 returned",
                "sequence": 1,
            },
            {
                "event_id": "evaluate_result",
                "event_type": "behavioral_eval",
                "action": "compare tool result with expected marker",
                "tool": "dependency_v2_eval",
                "executed": True,
                "evidence_refs": ["read_result"],
                "observation": "repaired eval passed",
                "sequence": 2,
                "parent_event_id": "read_fixture",
            },
        ],
        "sources": [
            {
                "source_id": "deepagents_local_fixture",
                "source_type": "local_fixture",
                "locator": "local:/datasets/read_0/note.txt",
                "title": "Bounded Deep Agents StateBackend fixture",
            }
        ],
        "evidence": [
            {
                "evidence_id": "read_result",
                "source_refs": ["deepagents_local_fixture"],
                "produced_by_event": "read_fixture",
            }
        ],
        "claims": [
            {
                "claim_id": "repair_behavior_restored",
                "text": "The repaired workflow returned READ_MARKER_0.",
                "evidence_refs": ["read_result"],
            }
        ],
        "artifacts": [
            {
                "artifact_id": "repaired_eval_output",
                "artifact_type": "behavioral_eval_result",
                "artifact_role": "FINAL_OUTPUT",
                "produced_by_event": "evaluate_result",
                "evidence_refs": ["read_result"],
            }
        ],
        "assurance_context": {
            "events": [
                {
                    "event_id": "read_fixture",
                    "governance_decision": "ALLOW",
                    "policy_ref": "local-offline-replay-policy-v2",
                    "approval_requirement": "NOT_REQUIRED",
                },
                {
                    "event_id": "evaluate_result",
                    "governance_decision": "ALLOW",
                    "policy_ref": "local-offline-replay-policy-v2",
                    "approval_requirement": "NOT_REQUIRED",
                },
            ]
        },
    }
