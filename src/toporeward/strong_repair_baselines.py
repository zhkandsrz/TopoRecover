from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Sequence

from .actions import Extrude, RegisterProfile
from .feature_plan import feature_plan_to_topology_contract, normalize_feature_plan
from .lm.parsing import parse_action_line
from .natural_repair_recall import evaluate_repaired_history, parse_patch
from .topology_history_repair import canonical_action_lines
from .verifier import TopoVerifier


PATCH_SCHEMA = '{"edits":[{"start":int,"end":int,"replacement":["Action",...]}]}'
HISTORY_SCHEMA = '{"actions":["Action",...]}'
RECAD_FEEDBACK_SCHEMA = (
    '{"diagnoses":[{"block_start":int,"block_end":int,'
    '"error_type":"type","description":"text"}]}'
)


@dataclass(frozen=True)
class RuntimeTrace:
    execution_status: str
    first_rejected_step: int | None
    failure_type: str | None
    repair_hint: str | None
    valid_prefix_length: int
    profile_def_use: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["profile_def_use"] = [dict(item) for item in self.profile_def_use]
        return row


def runtime_trace(observed_lines: Sequence[str]) -> RuntimeTrace:
    """Extract execution evidence without consulting a topology contract."""

    observed = canonical_action_lines(observed_lines)
    verifier = TopoVerifier()
    state = verifier.initial_state()
    first_rejected_step: int | None = None
    failure_type: str | None = None
    repair_hint: str | None = None
    valid_prefix_length = 0
    definitions: dict[str, int] = {}
    uses: dict[str, list[int]] = {}

    for index, line in enumerate(observed):
        action = parse_action_line(line)
        if action is None:
            first_rejected_step = index
            failure_type = "unparseable_action"
            repair_hint = "Replace the unparseable action with a supported typed action."
            break
        if isinstance(action, RegisterProfile):
            definitions[str(action.profile_id)] = index
        elif isinstance(action, Extrude):
            uses.setdefault(str(action.profile_id), []).append(index)
        result = verifier.step(state, action)
        if not result.valid or result.next_state is None:
            first_rejected_step = index
            failure_type = (
                str(getattr(result.failure_type, "value", result.failure_type))
                if result.failure_type is not None
                else "unknown"
            )
            repair_hint = result.repair_hint
            break
        state = result.next_state
        valid_prefix_length = index + 1

    def_use = tuple(
        {
            "profile_id": profile_id,
            "definition_step": definitions.get(profile_id),
            "use_steps": sorted(uses.get(profile_id, [])),
        }
        for profile_id in sorted(set(definitions) | set(uses))
    )
    ended = bool(getattr(state, "ended", False))
    if first_rejected_step is not None:
        status = "rejected"
    elif ended:
        status = "valid_ended"
    else:
        status = "valid_incomplete"
    return RuntimeTrace(
        execution_status=status,
        first_rejected_step=first_rejected_step,
        failure_type=failure_type,
        repair_hint=repair_hint,
        valid_prefix_length=valid_prefix_length,
        profile_def_use=def_use,
    )


def _history_text(lines: Sequence[str]) -> str:
    return "\n".join(f"{index}: {line}" for index, line in enumerate(lines))


def _dsl_text() -> str:
    return "\n".join(
        [
            "StartSketch; StartFace; StartLoop(kind=outer|inner);",
            "AddLine(start=(x,y), end=(x,y));",
            "AddArc(start=(x,y), mid=(x,y), end=(x,y));",
            "AddCircle(center=(x,y), radius=r); EndLoop; EndFace;",
            "RegisterProfile(profile_id=id); EndSketch;",
            "Extrude(profile_id=id, depth=d, op=add|cut|intersect); End.",
        ]
    )


def tracecad_style_prompt(
    *,
    observed_lines: Sequence[str],
    feature_plan: Mapping[str, Any],
    skills: Sequence[Mapping[str, Any]] = (),
) -> tuple[str, RuntimeTrace]:
    """Build a TraceCAD-style persistent-state localized-repair prompt.

    The model receives upstream requirements, modeling steps, and execution/def-use
    evidence. It must choose its own edit locus and never receives TopoRecover's
    committed-discrepancy certificate.
    """

    observed = canonical_action_lines(observed_lines)
    plan = normalize_feature_plan(feature_plan)
    trace = runtime_trace(observed)
    memory = list(skills)
    prompt = "\n".join(
        [
            "[TASK]",
            "Repair a failed parametric CAD history using persistent requirements, modeling steps, and runtime evidence.",
            "Choose the faulty operation or dependency region yourself. Preserve unrelated operations.",
            "Return only one JSON patch with schema:",
            PATCH_SCHEMA,
            "Use at most two non-overlapping edits and at most 240 replacement actions per edit.",
            "The complete repaired history must execute, end with End, and realize the feature plan.",
            "Supported typed CAD actions:",
            _dsl_text(),
            "[PERSISTENT_REQUIREMENTS]",
            json.dumps(plan, sort_keys=True, separators=(",", ":")),
            "[RUNTIME_AND_DEPENDENCY_EVIDENCE]",
            json.dumps(trace.to_dict(), sort_keys=True, separators=(",", ":")),
            "[REPAIR_SKILL_MEMORY]",
            json.dumps(memory, sort_keys=True, separators=(",", ":")),
            "[MODELING_STEPS]",
            _history_text(observed),
            "[PATCH]",
        ]
    )
    return prompt, trace


def _observed_summary(observed_lines: Sequence[str]) -> dict[str, Any]:
    observed = canonical_action_lines(observed_lines)
    trace = runtime_trace(observed)
    verifier = TopoVerifier()
    state = verifier.initial_state()
    for line in observed[: trace.valid_prefix_length]:
        action = parse_action_line(line)
        assert action is not None
        result = verifier.step(state, action)
        assert result.valid and result.next_state is not None
        state = result.next_state
    profiles = []
    for profile_id in sorted(state.profiles):
        profile = state.profiles[profile_id]
        profiles.append(
            {
                "profile_id": str(profile_id),
                "loop_roles": sorted(loop.kind for loop in profile.loops),
            }
        )
    return {
        "execution_status": trace.execution_status,
        "profile_count": len(profiles),
        "profiles": profiles,
        "extrusion_operations": [str(row["op"]) for row in state.extrusions],
    }


def requirement_validation_feedback(
    *, history_lines: Sequence[str], feature_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Return public requirement checks without exposing a repair location."""

    history = canonical_action_lines(history_lines)
    plan = normalize_feature_plan(feature_plan)
    contract = feature_plan_to_topology_contract(plan)
    evaluation = evaluate_repaired_history(history, contract)
    partial = dict(evaluation["partial_contract"])
    checks = [
        {
            "requirement": "history_executes_and_ends",
            "passed": bool(evaluation["valid_ended"]),
        },
        {
            "requirement": "registered_profile_count",
            "passed": partial.get("profile_count") == contract.get("profile_count"),
        },
        {
            "requirement": "profile_outer_inner_loop_roles",
            "passed": partial.get("profiles") == contract.get("profiles"),
        },
        {
            "requirement": "extrusion_feature_count",
            "passed": partial.get("extrusion_count") == contract.get("extrusion_count"),
        },
        {
            "requirement": "extrusion_operation_counts",
            "passed": partial.get("operation_counts") == contract.get("operation_counts"),
        },
        {
            "requirement": "profile_to_extrusion_reference_graph",
            "passed": partial.get("extrusion_graph") == contract.get("extrusion_graph"),
        },
    ]
    return {
        "checks": checks,
        "passed": sum(bool(item["passed"]) for item in checks),
        "total": len(checks),
        "execution_failure_type": evaluation.get("failure_type"),
        "valid_prefix_length": int(evaluation["valid_prefix_length"]),
        "contract_distance": int(evaluation["contract_distance"]),
    }


def requirement_candidate_key(
    *,
    candidate_lines: Sequence[str],
    feature_plan: Mapping[str, Any],
    original_lines: Sequence[str],
) -> tuple[Any, ...]:
    feedback = requirement_validation_feedback(
        history_lines=candidate_lines, feature_plan=feature_plan
    )
    original = canonical_action_lines(original_lines)
    candidate = canonical_action_lines(candidate_lines)
    lcs = SequenceMatcher(a=original, b=candidate, autojunk=False).ratio()
    execution_passed = next(
        bool(row["passed"])
        for row in feedback["checks"]
        if row["requirement"] == "history_executes_and_ends"
    )
    return (
        int(feedback["passed"]),
        int(execution_passed),
        -int(feedback["contract_distance"]),
        float(lcs),
    )


def cadcodeverify_style_prompt(
    *,
    observed_lines: Sequence[str],
    feature_plan: Mapping[str, Any],
    round_index: int = 1,
    max_rounds: int = 2,
) -> tuple[str, RuntimeTrace]:
    """Build a method-adapted binary-feedback refinement prompt.

    CADCodeVerify answers visual binary questions. The typed adaptation uses
    executor/state answers to the same requirement-level questions, without a
    repair location or target history.
    """

    observed = canonical_action_lines(observed_lines)
    plan = normalize_feature_plan(feature_plan)
    trace = runtime_trace(observed)
    feedback = requirement_validation_feedback(
        history_lines=observed, feature_plan=plan
    )
    prompt = "\n".join(
        [
            "[TASK]",
            f"Requirement-feedback refinement round {round_index} of {max_rounds}.",
            "Revise the complete typed CAD history using binary requirement checks.",
            "Return only one JSON object with schema:",
            HISTORY_SCHEMA,
            "Preserve correct actions when possible. The revised history must execute, end with End, and satisfy the feature plan.",
            "The checker does not identify the faulty step. Infer the necessary edit scope yourself.",
            "Supported typed CAD actions:",
            _dsl_text(),
            "[FEATURE_PLAN]",
            json.dumps(plan, sort_keys=True, separators=(",", ":")),
            "[BINARY_VALIDATION_FEEDBACK]",
            json.dumps(feedback, sort_keys=True, separators=(",", ":")),
            "[OBSERVED_HISTORY]",
            _history_text(observed),
            "[REVISED_HISTORY]",
        ]
    )
    return prompt, trace


def caddesigner_style_prompt(
    *,
    observed_lines: Sequence[str],
    feature_plan: Mapping[str, Any],
    design_brief: str,
    round_index: int = 1,
    max_rounds: int = 2,
) -> tuple[str, RuntimeTrace]:
    """Build a CADDesigner-style requirement expansion and full-history revision prompt."""

    observed = canonical_action_lines(observed_lines)
    plan = normalize_feature_plan(feature_plan)
    trace = runtime_trace(observed)
    feedback = requirement_validation_feedback(
        history_lines=observed, feature_plan=plan
    )
    prompt = "\n".join(
        [
            "[TASK]",
            f"Agentic CAD revision round {round_index} of {max_rounds}.",
            "Act as a CAD design agent. Reconcile the requirement, feature plan, failed execution trace, and full modeling history.",
            "Return a complete revised typed CAD history as exactly one JSON object:",
            HISTORY_SCHEMA,
            "The revised history must execute, end with End, and implement the feature plan. Reuse correct modeling steps when possible.",
            "Supported typed CAD actions:",
            _dsl_text(),
            "[DESIGN_REQUIREMENT]",
            design_brief.strip(),
            "[EXPANDED_FEATURE_PLAN]",
            json.dumps(plan, sort_keys=True, separators=(",", ":")),
            "[EXECUTION_TRACE]",
            json.dumps(trace.to_dict(), sort_keys=True, separators=(",", ":")),
            "[EXECUTION_AND_REQUIREMENT_FEEDBACK]",
            json.dumps(feedback, sort_keys=True, separators=(",", ":")),
            "[FAILED_MODELING_HISTORY]",
            _history_text(observed),
            "[REVISED_HISTORY]",
        ]
    )
    return prompt, trace


def recad_style_feedback_prompt(
    *,
    observed_lines: Sequence[str],
    feature_plan: Mapping[str, Any],
    design_brief: str,
    round_index: int = 1,
    max_rounds: int = 2,
) -> tuple[str, RuntimeTrace]:
    """Adapt ReCAD's feedback generator to typed histories and plan evidence."""

    observed = canonical_action_lines(observed_lines)
    plan = normalize_feature_plan(feature_plan)
    trace = runtime_trace(observed)
    state_summary = _observed_summary(observed)
    prompt = "\n".join(
        [
            "[TASK]",
            f"CAD review feedback round {round_index} of {max_rounds}.",
            "Compare the requested design and feature plan with the current typed CAD program and its dynamic-state summary.",
            "Identify the erroneous action block(s), classify each error, and explain the required correction.",
            "Infer the faulty block yourself; no repair location is supplied.",
            "Return only one JSON object with schema:",
            RECAD_FEEDBACK_SCHEMA,
            "Use at most four diagnoses. block_start is inclusive and block_end is exclusive.",
            "Allowed error types: primitive, rotation, position, size, constant, logic, missing_block, redundant_block, wrong_reference, wrong_operation.",
            "[DESIGN_REQUIREMENT]",
            design_brief.strip(),
            "[FEATURE_PLAN]",
            json.dumps(plan, sort_keys=True, separators=(",", ":")),
            "[DYNAMIC_STATE_SUMMARY]",
            json.dumps(state_summary, sort_keys=True, separators=(",", ":")),
            "[RUNTIME_TRACE]",
            json.dumps(trace.to_dict(), sort_keys=True, separators=(",", ":")),
            "[INDEXED_CAD_PROGRAM]",
            _history_text(observed),
            "[REVIEW_FEEDBACK]",
        ]
    )
    return prompt, trace


def parse_recad_feedback(
    text: str, *, observed_length: int, max_diagnoses: int = 4
) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping) or not isinstance(value.get("diagnoses"), list):
            continue
        diagnoses: list[dict[str, Any]] = []
        for row in value["diagnoses"]:
            if not isinstance(row, Mapping):
                return None
            try:
                start = int(row["block_start"])
                end = int(row["block_end"])
            except (KeyError, TypeError, ValueError):
                return None
            if not 0 <= start <= end <= observed_length:
                return None
            diagnoses.append(
                {
                    "block_start": start,
                    "block_end": end,
                    "error_type": str(row.get("error_type") or "unknown")[:80],
                    "description": str(row.get("description") or "")[:1000],
                }
            )
        if not 1 <= len(diagnoses) <= max_diagnoses:
            continue
        return {"diagnoses": diagnoses}
    return None


def recad_style_editor_prompt(
    *,
    observed_lines: Sequence[str],
    feature_plan: Mapping[str, Any],
    design_brief: str,
    generated_feedback: Mapping[str, Any],
    round_index: int = 1,
    max_rounds: int = 2,
) -> str:
    """Adapt ReCAD's code editor while preserving model-generated localization."""

    observed = canonical_action_lines(observed_lines)
    plan = normalize_feature_plan(feature_plan)
    trace = runtime_trace(observed)
    return "\n".join(
        [
            "[TASK]",
            f"CAD program correction round {round_index} of {max_rounds}.",
            "Use the generated review feedback to correct the complete typed CAD program.",
            "Return only one JSON object with schema:",
            HISTORY_SCHEMA,
            "Preserve action blocks not implicated by the review whenever possible.",
            "The final program must execute, end with End, and implement the feature plan.",
            "Supported typed CAD actions:",
            _dsl_text(),
            "[DESIGN_REQUIREMENT]",
            design_brief.strip(),
            "[FEATURE_PLAN]",
            json.dumps(plan, sort_keys=True, separators=(",", ":")),
            "[GENERATED_REVIEW_FEEDBACK]",
            json.dumps(dict(generated_feedback), sort_keys=True, separators=(",", ":")),
            "[RUNTIME_TRACE]",
            json.dumps(trace.to_dict(), sort_keys=True, separators=(",", ":")),
            "[INDEXED_CAD_PROGRAM]",
            _history_text(observed),
            "[CORRECTED_PROGRAM]",
        ]
    )


def parse_full_history(text: str, *, max_actions: int = 320) -> list[str] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping) or not isinstance(value.get("actions"), list):
            continue
        try:
            actions = canonical_action_lines(str(line) for line in value["actions"])
        except ValueError:
            continue
        if not 1 <= len(actions) <= max_actions:
            continue
        if any(parse_action_line(line) is None for line in actions):
            continue
        return actions
    return None


def parse_unrestricted_bounded_patch(
    text: str,
    *,
    observed_length: int,
    max_edits: int = 2,
    max_replacement_actions: int = 240,
) -> dict[str, Any] | None:
    patch = parse_patch(text)
    if patch is None:
        return None
    edits = sorted(
        list(patch.get("edits") or []),
        key=lambda item: (int(item["start"]), int(item["end"])),
    )
    if not 1 <= len(edits) <= max_edits:
        return None
    cursor = 0
    for edit in edits:
        start = int(edit["start"])
        end = int(edit["end"])
        replacement = canonical_action_lines(edit.get("replacement") or [])
        if not cursor <= start <= end <= observed_length:
            return None
        if len(replacement) > max_replacement_actions:
            return None
        if any(parse_action_line(line) is None for line in replacement):
            return None
        edit["replacement"] = replacement
        cursor = end
    return {"edits": edits}


def prompt_input_audit(prompt: str) -> dict[str, bool]:
    forbidden = {
        "first_contract_divergence_step": "first_contract_divergence_step",
        "j_topo": "j_topo",
        "semantic_repair_start": "semantic_repair_start",
        "topology_delta": "topology_delta",
        "mismatch_map": "mismatch_map",
        "oracle_patch": "oracle_patch",
        "target_actions": "target_actions",
        "target_geometry": "target_geometry",
    }
    return {key: token.lower() in prompt.lower() for key, token in forbidden.items()}


def operation_counts(feature_plan: Mapping[str, Any]) -> dict[str, int]:
    plan = normalize_feature_plan(feature_plan)
    return dict(Counter(str(feature["operation"]) for feature in plan["features"]))
