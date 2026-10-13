from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Self

from deepagents.backends.state import StateBackend
from deepagents.backends.utils import create_file_data
from deepagents.graph import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import Field, PrivateAttr


TRACKED_TOOLS = ("ls", "read_file", "glob", "grep")
DEPENDENCY_PREFIX = "deepagents.filesystem."
SYSTEM_PROMPT = """You are running a filesystem-tool evaluation.
For each user request, call exactly one appropriate filesystem tool from ls,
read_file, glob, or grep. Copy paths and search expressions exactly from the
request. Do not merely explain what you would do. After a tool result arrives,
return exactly WORKFLOW_COMPLETE and do not call another tool.
"""


@dataclass(frozen=True)
class FrozenCase:
    eval_id: str
    prompt: str
    files: tuple[tuple[str, str], ...]
    expected_substring: str
    expected_tool: str


@dataclass(frozen=True)
class CaseResult:
    eval_id: str
    passed: bool
    expected_tool: str
    expected_substring: str
    selected_tools: tuple[str, ...]
    tool_arguments: tuple[dict[str, object], ...]
    tool_statuses: tuple[str, ...]
    tool_outputs: tuple[str, ...]
    final_content: str
    failure_type: str | None
    exception: str | None
    duration_ms: float


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def load_frozen_inputs(path: Path) -> dict[str, Any]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    expected = frozen["semantic_hashes"]
    values = {
        "dependency_manifest": frozen["dependency_manifest"],
        "baseline_trace_index": frozen["baseline_trace_index"],
        "impacted_eval_mapping": frozen["impacted_eval_mapping"],
        "corpus": frozen["corpus"],
        "mutations": frozen["mutations"],
        "root_cause_logic": frozen["root_cause_logic"],
    }
    observed = {key: canonical_hash(value) for key, value in values.items()}
    if observed != expected:
        raise RuntimeError("frozen V2 semantic hashes do not verify")
    return frozen


def frozen_cases(frozen: dict[str, Any]) -> tuple[FrozenCase, ...]:
    expected_tools = {
        eval_id: dependency.removeprefix(DEPENDENCY_PREFIX)
        for dependency, eval_ids in frozen["baseline_trace_index"].items()
        for eval_id in eval_ids
    }
    return tuple(
        FrozenCase(
            eval_id=item["eval_id"],
            prompt=item["prompt"],
            files=tuple(tuple(pair) for pair in item["files"]),
            expected_substring=item["expected_substring"],
            expected_tool=expected_tools[item["eval_id"]],
        )
        for item in frozen["corpus"]
    )


def mutation_by_id(frozen: dict[str, Any], mutation_id: str) -> dict[str, Any]:
    return next(
        item for item in frozen["mutations"] if item["mutation_id"] == mutation_id
    )


def _rename_schema_field(
    schema: dict[str, Any], old_name: str, new_name: str
) -> dict[str, Any]:
    changed = copy.deepcopy(schema)
    properties = changed.get("properties")
    if not isinstance(properties, dict) or old_name not in properties:
        raise ValueError(f"schema does not expose field {old_name!r}")
    properties[new_name] = properties.pop(old_name)
    required = changed.get("required", [])
    if isinstance(required, list):
        changed["required"] = [
            new_name if item == old_name else item for item in required
        ]
    return changed


def advertised_manifest(
    frozen: dict[str, Any], mutation: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    manifest = copy.deepcopy(frozen["dependency_manifest"])
    if mutation is None:
        return manifest
    tool_name = mutation["dependency_id"].removeprefix(DEPENDENCY_PREFIX)
    if mutation["description_override"] is not None:
        manifest[tool_name]["description"] = mutation["description_override"]
    if mutation["schema_rename"] is not None:
        old_name, new_name = mutation["schema_rename"]
        manifest[tool_name]["input_schema"] = _rename_schema_field(
            manifest[tool_name]["input_schema"], old_name, new_name
        )
    return manifest


def _openai_tool(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec["description"],
            "parameters": spec["input_schema"],
        },
    }


class OllamaToolModel(BaseChatModel):
    """Local stochastic model with V2-frozen tool advertisements."""

    model_name: str = "qwen3:8b"
    temperature: float = 0.6
    seed: int | None = None
    base_url: str = "http://127.0.0.1:11434/v1"
    manifest: dict[str, dict[str, Any]] = Field(default_factory=dict, exclude=True)
    _bound: Runnable[LanguageModelInput, AIMessage] | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "ollama-openai-compatible-frozen-v2-tools"

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Self:
        del tools, tool_choice, kwargs
        tool_specs = [_openai_tool(self.manifest[name]) for name in TRACKED_TOOLS]
        client = ChatOpenAI(
            model=self.model_name,
            api_key="ollama-local",
            base_url=self.base_url,
            temperature=self.temperature,
            seed=self.seed,
            timeout=120,
            max_retries=0,
            extra_body={"options": {"num_ctx": 8192}, "think": False},
        )
        self._bound = client.bind_tools(tool_specs, parallel_tool_calls=False)
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
        if self._bound is None:
            raise RuntimeError("model was invoked before tools were bound")
        message = self._bound.invoke(messages)
        if not isinstance(message, AIMessage):
            raise TypeError("local model did not return AIMessage")
        return ChatResult(generations=[ChatGeneration(message=message)])


def build_agent(
    frozen: dict[str, Any], mutation: dict[str, Any] | None, seed: int | None
) -> Any:
    manifest = advertised_manifest(frozen, mutation)
    descriptions = {name: spec["description"] for name, spec in manifest.items()}
    middleware = FilesystemMiddleware(
        backend=StateBackend(),
        tools=list(TRACKED_TOOLS),
        custom_tool_descriptions=descriptions,
    )
    model = OllamaToolModel(seed=seed, manifest=manifest)
    return create_deep_agent(
        model=model,
        middleware=[middleware],
        system_prompt=SYSTEM_PROMPT,
    )


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def run_case(agent: Any, case: FrozenCase) -> CaseResult:
    started = perf_counter()
    try:
        result = agent.invoke(
            {
                "messages": [{"role": "user", "content": case.prompt}],
                "files": {
                    path: create_file_data(content) for path, content in case.files
                },
            },
            config={"recursion_limit": 8},
        )
        messages = result["messages"]
        tool_calls = tuple(
            call
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        )
        tool_messages = tuple(
            message for message in messages if isinstance(message, ToolMessage)
        )
        selected = tuple(str(call["name"]) for call in tool_calls)
        arguments = tuple(dict(call["args"]) for call in tool_calls)
        statuses = tuple(str(message.status or "success") for message in tool_messages)
        outputs = tuple(_text(message.content) for message in tool_messages)
        final = next(
            (
                _text(message.content)
                for message in reversed(messages)
                if isinstance(message, AIMessage) and not message.tool_calls
            ),
            "",
        )
        output_text = "\n".join(outputs)
        passed = (
            case.expected_tool in selected
            and case.expected_substring in output_text
            and "error" not in statuses
        )
        if passed:
            failure_type = None
        elif not selected:
            failure_type = "no_tool_call"
        elif case.expected_tool not in selected:
            failure_type = "wrong_tool"
        elif "error" in statuses:
            failure_type = "tool_error"
        else:
            failure_type = "output_mismatch"
        exception = None
    except Exception as error:  # noqa: BLE001 - every stochastic failure is evidence
        selected = ()
        arguments = ()
        statuses = ()
        outputs = ()
        final = ""
        passed = False
        failure_type = "agent_exception"
        exception = f"{type(error).__name__}: {error}"
    return CaseResult(
        eval_id=case.eval_id,
        passed=passed,
        expected_tool=case.expected_tool,
        expected_substring=case.expected_substring,
        selected_tools=selected,
        tool_arguments=arguments,
        tool_statuses=statuses,
        tool_outputs=outputs,
        final_content=final,
        failure_type=failure_type,
        exception=exception,
        duration_ms=round((perf_counter() - started) * 1000, 3),
    )


def run_suite(
    frozen: dict[str, Any],
    cases: tuple[FrozenCase, ...],
    *,
    mutation: dict[str, Any] | None,
    seed: int | None,
) -> dict[str, Any]:
    agent = build_agent(frozen, mutation, seed)
    started = perf_counter()
    results = tuple(run_case(agent, case) for case in cases)
    return {
        "seed": seed,
        "mutation_id": None if mutation is None else mutation["mutation_id"],
        "count": len(results),
        "passed": sum(item.passed for item in results),
        "failed": sum(not item.passed for item in results),
        "duration_ms": round((perf_counter() - started) * 1000, 3),
        "results": [asdict(item) for item in results],
    }


def frozen_target_cases(
    frozen: dict[str, Any],
    cases: tuple[FrozenCase, ...],
    mutation_id: str,
) -> tuple[FrozenCase, ...]:
    selected_ids = set(frozen["impacted_eval_mapping"][mutation_id])
    return tuple(case for case in cases if case.eval_id in selected_ids)


def compare_trial(
    baseline: dict[str, Any],
    full: dict[str, Any],
    targeted: dict[str, Any],
    *,
    frozen_target_ids: set[str],
    changed_dependency: str,
) -> dict[str, Any]:
    baseline_by_id = {item["eval_id"]: item for item in baseline["results"]}
    full_by_id = {item["eval_id"]: item for item in full["results"]}
    targeted_by_id = {item["eval_id"]: item for item in targeted["results"]}
    regression_ids = {
        eval_id
        for eval_id, item in full_by_id.items()
        if baseline_by_id[eval_id]["passed"] and not item["passed"]
    }
    detected_ids = {
        eval_id
        for eval_id in regression_ids
        if eval_id in targeted_by_id and not targeted_by_id[eval_id]["passed"]
    }
    false_negative_ids = regression_ids - detected_ids
    targeted_failure_ids = {
        eval_id for eval_id, item in targeted_by_id.items() if not item["passed"]
    }
    false_positive_ids = targeted_failure_ids - regression_ids
    true_negative_ids = {
        eval_id
        for eval_id, item in targeted_by_id.items()
        if item["passed"] and eval_id not in regression_ids
    }
    changed_tool = changed_dependency.removeprefix(DEPENDENCY_PREFIX)
    root_cause_correct_ids = {
        eval_id
        for eval_id in regression_ids
        if eval_id in frozen_target_ids
        and changed_tool in baseline_by_id[eval_id]["selected_tools"]
    }
    unseen_paths = []
    for eval_id, item in baseline_by_id.items():
        selected = set(item["selected_tools"])
        expected = {item["expected_tool"]}
        if selected and selected != expected:
            unseen_paths.append(
                {
                    "eval_id": eval_id,
                    "phase": "baseline",
                    "expected": sorted(expected),
                    "observed": sorted(selected),
                }
            )
    for eval_id in sorted(regression_ids - frozen_target_ids):
        unseen_paths.append(
            {
                "eval_id": eval_id,
                "phase": "mutated_full",
                "expected_frozen_target": False,
                "observed_tools": full_by_id[eval_id]["selected_tools"],
            }
        )
    return {
        "seed": full["seed"],
        "eligible_baseline_passes": sum(
            item["passed"] for item in baseline_by_id.values()
        ),
        "full_regression_ids": sorted(regression_ids),
        "targeted_detected_ids": sorted(detected_ids),
        "false_negative_ids": sorted(false_negative_ids),
        "false_positive_ids": sorted(false_positive_ids),
        "true_negative_ids": sorted(true_negative_ids),
        "root_cause_correct_ids": sorted(root_cause_correct_ids),
        "root_cause_incorrect_ids": sorted(regression_ids - root_cause_correct_ids),
        "unseen_stochastic_dependency_paths": unseen_paths,
        "behavioral_regression_rate": (
            len(regression_ids) / len(baseline_by_id) if baseline_by_id else 0.0
        ),
    }
