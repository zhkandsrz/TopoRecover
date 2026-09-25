from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from typing import Any, Iterable, Mapping, Sequence

from .actions import Action
from .bounded_patch_repair import replay_lines, strict_target_reached
from .failure_certificate import build_failure_certificate
from .lm.parsing import parse_action_line
from .rlvr import structure_stats
from .structure_family import structure_family_id
from .topology_history_repair import canonical_action_lines
from .topology_history_repair import topology_intent_contract
from .topoplan_v2 import infer_phase
from .verifier import TopoVerifier


PATCH_SCHEMA = (
    '{"edits":[{"start":int,"end":int,"replacement":["Action",...]}]}'
)


def apply_patch_operations(
    observed_lines: Sequence[str], operations: Sequence[Mapping[str, Any]],
    *, preserve_source_text: bool = False,
) -> list[str]:
    observed = list(observed_lines) if preserve_source_text else canonical_action_lines(observed_lines)
    normalized = sorted(
        (
            {
                "start": int(operation["start"]),
                "end": int(operation["end"]),
                "replacement": canonical_action_lines(operation.get("replacement") or []),
            }
            for operation in operations
        ),
        key=lambda item: (item["start"], item["end"]),
    )
    result: list[str] = []
    cursor = 0
    for operation in normalized:
        start = int(operation["start"])
        end = int(operation["end"])
        if not cursor <= start <= end <= len(observed):
            raise ValueError("invalid or overlapping patch operation")
        result.extend(observed[cursor:start])
        result.extend(operation["replacement"])
        cursor = end
    result.extend(observed[cursor:])
    return result


def stable_id(*parts: object) -> str:
    payload = "\n".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parsed_actions(lines: Sequence[str]) -> list[Action]:
    actions = [parse_action_line(str(line)) for line in lines]
    if any(action is None for action in actions):
        raise ValueError("history contains an unparseable CAD action")
    return [action for action in actions if action is not None]


def target_counts(lines: Sequence[str]) -> dict[str, int]:
    stats = structure_stats(parsed_actions(lines))
    return {
        "profile_count": int(stats.get("profile_count", 0) or 0),
        "hole_count": int(stats.get("hole_count", 0) or 0),
        "circle_hole_count": int(stats.get("circle_hole_count", 0) or 0),
        "extrude_count": int(stats.get("extrude_count", 0) or 0),
    }


def replay_prefix(lines: Sequence[str]) -> tuple[TopoVerifier, Any, list[Action]]:
    verifier = TopoVerifier()
    state = verifier.initial_state()
    actions: list[Action] = []
    for index, action in enumerate(parsed_actions(lines)):
        result = verifier.step(state, action)
        if not result.valid or result.next_state is None:
            raise ValueError(f"stored valid prefix fails at action {index}")
        state = result.next_state
        actions.append(action)
    return verifier, state, actions


def failure_certificate_for_case(row: Mapping[str, Any]) -> dict[str, Any]:
    prefix_lines = [str(line) for line in row.get("valid_prefix_actions") or []]
    attempted_line = str(row.get("attempted_action") or "")
    target_lines = [str(line) for line in row.get("target_actions") or []]
    verifier, state, prefix_actions = replay_prefix(prefix_lines)
    attempted = parse_action_line(attempted_line)
    if attempted is None:
        raise ValueError("attempted action is unparseable")
    counts = target_counts(target_lines)
    phase = infer_phase(prefix_actions, counts, state=state)
    certificate = build_failure_certificate(
        prefix_actions=prefix_actions,
        state=state,
        candidate_action=attempted,
        target_stats=counts,
        micro_phase=phase,
        failing_step=int(row["failure_step"]),
        verifier=verifier,
    )
    if certificate is None:
        result = verifier.step(state, attempted)
        raise ValueError(
            "TopoVerifier did not reproduce a certificate: "
            f"valid={result.valid} failure={result.failure_type}"
        )
    return certificate.to_dict()


def _history_text(lines: Sequence[str]) -> str:
    return "\n".join(f"{index}: {line}" for index, line in enumerate(lines))


def repair_prompt(
    *,
    observed_lines: Sequence[str],
    failure_step: int,
    topology_contract: Mapping[str, Any],
    feedback: Mapping[str, Any] | str,
) -> str:
    feedback_text = (
        json.dumps(feedback, sort_keys=True, separators=(",", ":"))
        if isinstance(feedback, Mapping)
        else str(feedback)
    )
    return "\n".join(
        [
            "[TASK]",
            (
                "Repair this typed CAD action history. Preserve unaffected actions, "
                "make the whole history executable and ended, and satisfy the topology goal."
            ),
            "Return only one JSON object with schema:",
            PATCH_SCHEMA,
            "Indices are zero-based half-open spans in OBSERVED_HISTORY.",
            (
                "Use at most two edits and at most 160 replacement actions per edit. "
                "You may regenerate the suffix from the first failure."
            ),
            (
                "By default preserve every action before FIRST_FAILURE_STEP and replace "
                "the suffix [FIRST_FAILURE_STEP, len(OBSERVED_HISTORY)). Roll back earlier "
                "only when the reported state cannot be repaired from that step."
            ),
            "Use only these CAD DSL forms:",
            "StartSketch",
            "StartFace",
            "StartLoop(kind=outer|inner)",
            "AddLine(start=(x,y), end=(x,y))",
            "AddArc(start=(x,y), mid=(x,y), end=(x,y))",
            "AddCircle(center=(x,y), radius=r)",
            "EndLoop",
            "EndFace",
            "RegisterProfile(profile_id=profile_N)",
            "EndSketch",
            "Extrude(profile_id=profile_N, depth=d, op=add|cut|intersect)",
            "End",
            "Do not invent any other action type. The repaired history must end with End.",
            "[DESIRED_TOPOLOGY_CONTRACT]",
            json.dumps(topology_contract, sort_keys=True, separators=(",", ":")),
            "[FIRST_FAILURE_STEP]",
            str(int(failure_step)),
            "[HISTORY_LENGTH]",
            str(len(observed_lines)),
            "[OBSERVED_HISTORY]",
            _history_text(observed_lines),
            "[FAILURE_FEEDBACK]",
            feedback_text,
            "[PATCH]",
        ]
    )


def build_public_private_case(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = canonical_action_lines(row.get("observed_actions") or [])
    target = canonical_action_lines(row.get("target_actions") or [])
    valid_prefix = canonical_action_lines(row.get("valid_prefix_actions") or [])
    failure_step = int(row["failure_step"])
    if failure_step != len(valid_prefix):
        raise ValueError("failure_step does not equal valid-prefix length")
    if failure_step >= len(observed):
        raise ValueError("failure_step is outside observed history")
    certificate = failure_certificate_for_case(row)
    counts = target_counts(target)
    desired_contract = topology_intent_contract(target, level="topology")
    if desired_contract is None:
        raise ValueError("target history does not define an executable topology contract")
    family = structure_family_id(parsed_actions(target))
    case_id = stable_id(
        "toporecover-natural-recall-v1",
        row.get("program_id"),
        family,
        failure_step,
        row.get("attempted_action"),
    )
    generic_feedback = {
        "source": "generic_execution",
        "message": "The CAD executor rejected the action at FIRST_FAILURE_STEP.",
    }
    public = {
        "schema_version": "toporecover_natural_recall_v1",
        "case_id": case_id,
        "program_id": str(row.get("program_id") or ""),
        "structure_family_id": family,
        "source_file": row.get("source_file"),
        "failure_step": failure_step,
        "failure_type": str(row.get("failure_type") or "unknown"),
        "goal_counts": counts,
        "desired_topology_contract": desired_contract,
        "observed_actions": observed,
        "generic_execution_prompt": repair_prompt(
            observed_lines=observed,
            failure_step=failure_step,
            topology_contract=desired_contract,
            feedback=generic_feedback,
        ),
        "topoverifier_prompt": repair_prompt(
            observed_lines=observed,
            failure_step=failure_step,
            topology_contract=desired_contract,
            feedback=certificate,
        ),
        "topoverifier_certificate": certificate,
        "leakage_audit": {
            "target_action_history_visible": False,
            "oracle_next_action_visible": False,
            "oracle_patch_visible": False,
            "candidate_specific_validity_visible": False,
            "goal_counts_visible": True,
            "topology_contract_visible": True,
        },
    }
    private = {
        "case_id": case_id,
        "program_id": str(row.get("program_id") or ""),
        "structure_family_id": family,
        "target_actions": target,
        "target_counts": counts,
        "desired_topology_contract": desired_contract,
        "oracle_retry_action": row.get("oracle_retry_action"),
        "oracle_single_retry_rejoins_target_prefix": bool(
            row.get("oracle_single_retry_rejoins_target_prefix")
        ),
    }
    return public, private


def parse_patch(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or not isinstance(value.get("edits"), list):
            continue
        try:
            edits = [
                {
                    "start": int(edit["start"]),
                    "end": int(edit["end"]),
                    "replacement": [str(line) for line in edit["replacement"]],
                }
                for edit in value["edits"]
            ]
        except (KeyError, TypeError, ValueError):
            continue
        return {"edits": edits}
    return None


def _bounded_patch(patch: Mapping[str, Any], observed_length: int) -> bool:
    edits = list(patch.get("edits") or [])
    if not 1 <= len(edits) <= 2:
        return False
    cursor = 0
    for edit in sorted(edits, key=lambda item: (int(item["start"]), int(item["end"]))):
        start = int(edit["start"])
        end = int(edit["end"])
        replacement = list(edit.get("replacement") or [])
        if not cursor <= start <= end <= observed_length:
            return False
        if not 0 <= len(replacement) <= 160:
            return False
        if any(parse_action_line(str(line)) is None for line in replacement):
            return False
        cursor = end
    return True


def normalize_append_off_by_one(
    patch: Mapping[str, Any], observed_length: int
) -> dict[str, Any]:
    """Normalize the common ``[len, len+1)`` spelling of tail insertion.

    This adapter is purely syntactic: it does not inspect verifier state, the
    desired contract, or replacement actions. All other spans remain strict.
    """

    edits: list[dict[str, Any]] = []
    for raw in patch.get("edits") or []:
        start = int(raw["start"])
        end = int(raw["end"])
        if start == observed_length and end == observed_length + 1:
            end = observed_length
        edits.append(
            {
                "start": start,
                "end": end,
                "replacement": [str(line) for line in raw.get("replacement") or []],
            }
        )
    return {"edits": edits}


def evaluate_patch_candidate(
    public: Mapping[str, Any],
    private: Mapping[str, Any],
    text: str,
    *,
    horizon: int = 6,
    normalize_tail_append: bool = False,
) -> dict[str, Any]:
    observed = [str(line) for line in public.get("observed_actions") or []]
    patch = parse_patch(text)
    if patch is None:
        return {
            "generated": text,
            "patch": None,
            "parseable": False,
            "bounded": False,
            "immediate_valid": False,
            "k_step_survivable": False,
            "valid_ended": False,
            "intent_satisfied": False,
            "exact_target": False,
        }
    raw_patch = patch
    if normalize_tail_append:
        patch = normalize_append_off_by_one(patch, len(observed))
    bounded = _bounded_patch(patch, len(observed))
    if not bounded:
        return {
            "generated": text,
            "patch": patch,
            "raw_patch": raw_patch,
            "tail_append_normalized": patch != raw_patch,
            "parseable": True,
            "bounded": False,
            "immediate_valid": False,
            "k_step_survivable": False,
            "valid_ended": False,
            "intent_satisfied": False,
            "exact_target": False,
        }
    try:
        repaired = apply_patch_operations(observed, list(patch["edits"]))
    except (TypeError, ValueError):
        repaired = []
    outcome = replay_lines(repaired) if repaired else None
    first_edit = min(int(edit["start"]) for edit in patch["edits"])
    replacement_end = first_edit + sum(
        len(list(edit.get("replacement") or [])) for edit in patch["edits"]
    )
    local_limit = min(len(repaired), replacement_end + int(horizon))
    immediate = replay_lines(repaired[:replacement_end]) if repaired else None
    local = replay_lines(repaired[:local_limit]) if repaired else None
    target = [str(line) for line in private.get("target_actions") or []]
    hidden_target_counts = private.get("target_counts")
    if hidden_target_counts is None:
        hidden_target_counts = target_counts(target)
    repaired_contract = (
        topology_intent_contract(repaired, level="topology")
        if outcome and outcome.valid and outcome.ended
        else None
    )
    topology_goal_satisfied = bool(
        outcome and strict_target_reached(outcome, dict(hidden_target_counts))
    )
    intent_satisfied = bool(
        repaired_contract is not None
        and repaired_contract == dict(private["desired_topology_contract"])
    )
    return {
        "generated": text,
        "patch": patch,
        "raw_patch": raw_patch,
        "tail_append_normalized": patch != raw_patch,
        "parseable": True,
        "bounded": True,
        "immediate_valid": bool(immediate and immediate.valid),
        "k_step_survivable": bool(local and local.valid),
        "valid_ended": bool(outcome and outcome.valid and outcome.ended),
        "topology_goal_satisfied": topology_goal_satisfied,
        "intent_satisfied": intent_satisfied,
        "exact_target": repaired == target,
        "repaired_actions": repaired,
        "replay_failure_step": outcome.failure_step if outcome else None,
        "replay_failure_type": outcome.failure_type if outcome else "patch_application_failure",
    }


def _replay_details(lines: Sequence[str]) -> dict[str, Any]:
    verifier = TopoVerifier()
    state = verifier.initial_state()
    actions: list[Action] = []
    for index, line in enumerate(lines):
        action = parse_action_line(str(line))
        if action is None:
            return {
                "valid": False,
                "ended": False,
                "failure_step": index,
                "failure_type": "unparseable_action",
                "failed_action": None,
                "state": state,
                "actions": actions,
            }
        result = verifier.step(state, action)
        if not result.valid or result.next_state is None:
            return {
                "valid": False,
                "ended": False,
                "failure_step": index,
                "failure_type": str(result.failure_type or "unknown"),
                "failed_action": action,
                "state": state,
                "actions": actions,
            }
        state = result.next_state
        actions.append(action)
    return {
        "valid": True,
        "ended": bool(state.ended),
        "failure_step": None,
        "failure_type": None,
        "failed_action": None,
        "state": state,
        "actions": actions,
    }


def _partial_contract(details: Mapping[str, Any]) -> dict[str, Any]:
    state = details["state"]
    profile_rows: list[dict[str, Any]] = []
    profile_index: dict[str, int] = {}
    inner_count = 0
    for index, profile_id in enumerate(sorted(state.profiles)):
        loop_roles = sorted(loop.kind for loop in state.profiles[profile_id].loops)
        profile_index[str(profile_id)] = index
        inner_count += sum(role == "inner" for role in loop_roles)
        profile_rows.append({"profile_index": index, "loop_roles": loop_roles})
    operations = Counter(str(row["op"]) for row in state.extrusions)
    extrusions = [
        {
            "profile_index": profile_index.get(str(row["profile_id"]), -1),
            "operation": str(row["op"]),
        }
        for row in state.extrusions
    ]
    return {
        "profile_count": len(state.profiles),
        "inner_loop_count": inner_count,
        "extrusion_count": len(state.extrusions),
        "operation_counts": dict(sorted(operations.items())),
        "profiles": profile_rows,
        "extrusion_graph": extrusions,
    }


def _iterative_state_summary(details: Mapping[str, Any]) -> dict[str, Any]:
    state = details["state"]
    loop = state.current_loop
    return {
        "hierarchy_stack": list(state.stack),
        "open_loop_kind": loop.kind if loop is not None else None,
        "loop_start": loop.start if loop is not None else None,
        "loop_tail": loop.tail if loop is not None else None,
        "loop_segments": len(loop.segments) if loop is not None else 0,
        "completed_face_loops": [loop.kind for loop in state.current_face_loops],
        "pending_face": state.pending_face is not None,
        "registered_profiles": sorted(state.profiles),
        "extrusions": [
            {
                "profile_id": str(row["profile_id"]),
                "operation": str(row["op"]),
                "depth": float(row["depth"]),
            }
            for row in state.extrusions
        ],
    }


def _completion_prerequisite(details: Mapping[str, Any]) -> str:
    state = details["state"]
    top = state.stack[-1] if state.stack else None
    if top == "loop":
        return (
            "close the current geometry, then EndLoop; finish the face, register "
            "its profile, close the sketch, complete required extrusions, then End"
        )
    if top == "face":
        return (
            "EndFace, register the completed profile, close the sketch, complete "
            "required extrusions, then End"
        )
    if state.pending_face is not None:
        return "register the pending face before closing its sketch"
    if top == "sketch":
        return "EndSketch before any remaining Extrude actions, then emit End"
    return "complete outstanding extrusions, then emit End"


def topology_contract_distance(
    observed: Mapping[str, Any], desired: Mapping[str, Any]
) -> int:
    distance = sum(
        abs(int(observed.get(key, 0) or 0) - int(desired.get(key, 0) or 0))
        for key in ("profile_count", "inner_loop_count", "extrusion_count")
    )
    observed_operations = Counter(observed.get("operation_counts") or {})
    desired_operations = Counter(desired.get("operation_counts") or {})
    distance += sum((observed_operations - desired_operations).values())
    distance += sum((desired_operations - observed_operations).values())
    observed_profiles = Counter(
        tuple(sorted(str(role) for role in (row.get("loop_roles") or [])))
        for row in observed.get("profiles") or []
    )
    desired_profiles = Counter(
        tuple(sorted(str(role) for role in (row.get("loop_roles") or [])))
        for row in desired.get("profiles") or []
    )
    distance += sum((observed_profiles - desired_profiles).values())
    distance += sum((desired_profiles - observed_profiles).values())
    observed_extrusions = Counter(
        (int(row.get("profile_index", -1)), str(row.get("operation") or ""))
        for row in observed.get("extrusion_graph") or []
    )
    desired_extrusions = Counter(
        (int(row.get("profile_index", -1)), str(row.get("operation") or ""))
        for row in desired.get("extrusion_graph") or []
    )
    distance += sum((observed_extrusions - desired_extrusions).values())
    distance += sum((desired_extrusions - observed_extrusions).values())
    return int(distance)


def topology_contract_equivalent(
    observed: Mapping[str, Any], desired: Mapping[str, Any]
) -> bool:
    """Compare topology contracts modulo non-semantic loop-role ordering."""

    return topology_contract_distance(observed, desired) == 0


def evaluate_repaired_history(
    lines: Sequence[str], desired_contract: Mapping[str, Any]
) -> dict[str, Any]:
    canonical = canonical_action_lines(lines)
    details = _replay_details(canonical)
    partial_contract = _partial_contract(details)
    contract = (
        topology_intent_contract(canonical, level="topology")
        if details["valid"] and details["ended"]
        else None
    )
    intent_satisfied = (
        topology_contract_equivalent(contract, desired_contract)
        if contract is not None
        else False
    )
    valid_prefix_length = (
        len(canonical)
        if details["valid"]
        else int(details.get("failure_step") or 0)
    )
    return {
        "valid": bool(details["valid"]),
        "ended": bool(details["ended"]),
        "valid_ended": bool(details["valid"] and details["ended"]),
        "intent_satisfied": bool(intent_satisfied),
        "valid_and_intent_satisfied": bool(
            details["valid"] and details["ended"] and intent_satisfied
        ),
        "failure_step": details["failure_step"],
        "failure_type": details["failure_type"],
        "valid_prefix_length": valid_prefix_length,
        "partial_contract": partial_contract,
        "contract_distance": topology_contract_distance(
            partial_contract, desired_contract
        ),
    }


def transactional_selection_key(
    evaluation: Mapping[str, Any], *, selector: str, edit_cost: int = 0
) -> tuple[Any, ...]:
    if selector not in {"execution", "topoverifier"}:
        raise ValueError(f"unknown transactional selector: {selector}")
    common = (
        int(bool(evaluation.get("valid_ended"))),
        int(bool(evaluation.get("valid"))),
    )
    if selector == "execution":
        return (
            *common,
            int(evaluation.get("valid_prefix_length", 0) or 0),
            -int(edit_cost),
        )
    return (
        int(bool(evaluation.get("valid_and_intent_satisfied"))),
        int(bool(evaluation.get("valid"))),
        -int(evaluation.get("contract_distance", 10**9) or 0),
        int(bool(evaluation.get("valid_ended"))),
        int(evaluation.get("valid_prefix_length", 0) or 0),
        -int(edit_cost),
    )


def iterative_feedback(
    *,
    history_lines: Sequence[str],
    desired_contract: Mapping[str, Any],
    target_counts: Mapping[str, Any],
    structured: bool,
) -> tuple[int, Mapping[str, Any] | str]:
    canonical = canonical_action_lines(history_lines)
    details = _replay_details(canonical)
    if not details["valid"]:
        step = int(details.get("failure_step") or 0)
        if not structured:
            return step, "The CAD executor rejected the action at CURRENT_REPAIR_LOCUS_STEP."
        failed_action = details.get("failed_action")
        if failed_action is not None:
            phase = infer_phase(
                details["actions"], dict(target_counts), state=details["state"]
            )
            certificate = build_failure_certificate(
                prefix_actions=details["actions"],
                state=details["state"],
                candidate_action=failed_action,
                target_stats=dict(target_counts),
                micro_phase=phase,
                failing_step=step,
                verifier=TopoVerifier(),
            )
            if certificate is not None:
                return step, certificate.to_dict()
        return step, {
            "failing_step": step,
            "violation": details.get("failure_type") or "unknown",
            "observed_topology": _partial_contract(details),
            "expected_prerequisite": "restore a verifier-valid CAD prefix",
        }

    current_contract = _partial_contract(details)
    if not details["ended"]:
        step = len(canonical)
        if not structured:
            return step, "The CAD history is executable so far but does not terminate."
        return step, {
            "failing_step": step,
            "violation": "incomplete_history",
            "observed_state": _iterative_state_summary(details),
            "observed_topology": current_contract,
            "desired_topology": dict(desired_contract),
            "contract_distance": topology_contract_distance(
                current_contract, desired_contract
            ),
            "repair_locus": {
                "start_step": step,
                "end_step": step,
                "recommended_operation": "insert_before",
            },
            "expected_prerequisite": _completion_prerequisite(details),
        }

    exact_contract = topology_intent_contract(canonical, level="topology")
    step = max(0, len(canonical) - 1)
    if not structured:
        return step, "The CAD history executes and terminates but misses the requested topology goal."
    return step, {
        "failing_step": step,
        "violation": "topology_contract_mismatch",
        "observed_topology": exact_contract,
        "observed_state": _iterative_state_summary(details),
        "desired_topology": dict(desired_contract),
        "contract_distance": topology_contract_distance(
            exact_contract or {}, desired_contract
        ),
        "repair_locus": {
            "start_step": step,
            "end_step": step,
            "recommended_operation": "replace",
        },
    }


def summarize_candidate_sets(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float:
        return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0

    return {
        "cases": len(rows),
        "proposal_parse_recall": rate("any_parseable"),
        "bounded_patch_recall": rate("any_bounded"),
        "immediate_valid_candidate_recall": rate("any_immediate_valid"),
        "k_step_survivable_candidate_recall": rate("any_k_step_survivable"),
        "valid_ended_candidate_recall": rate("any_valid_ended"),
        "topology_goal_candidate_recall": rate("any_topology_goal_satisfied"),
        "intent_satisfying_candidate_recall": rate("any_intent_satisfied"),
        "valid_and_intent_satisfying_candidate_recall": rate(
            "any_valid_and_intent_satisfied"
        ),
        "exact_target_candidate_recall": rate("any_exact_target"),
        "mean_unique_candidates": (
            sum(int(row.get("candidate_count", 0)) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
    }


def candidate_set_record(
    *,
    public: Mapping[str, Any],
    private: Mapping[str, Any],
    condition: str,
    generated: Iterable[str],
    horizon: int = 6,
) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for text in generated:
        result = evaluate_patch_candidate(public, private, str(text), horizon=horizon)
        key = json.dumps(result.get("patch"), sort_keys=True, separators=(",", ":"))
        unique.setdefault(key, result)
    candidates = list(unique.values())
    return {
        "case_id": public["case_id"],
        "program_id": public["program_id"],
        "structure_family_id": public["structure_family_id"],
        "failure_type": public["failure_type"],
        "condition": condition,
        "candidate_count": len(candidates),
        "any_parseable": any(row["parseable"] for row in candidates),
        "any_bounded": any(row["bounded"] for row in candidates),
        "any_immediate_valid": any(row["immediate_valid"] for row in candidates),
        "any_k_step_survivable": any(row["k_step_survivable"] for row in candidates),
        "any_valid_ended": any(row["valid_ended"] for row in candidates),
        "any_topology_goal_satisfied": any(
            row.get("topology_goal_satisfied", False) for row in candidates
        ),
        "any_intent_satisfied": any(row["intent_satisfied"] for row in candidates),
        "any_valid_and_intent_satisfied": any(
            row["valid_ended"] and row["intent_satisfied"] for row in candidates
        ),
        "any_exact_target": any(row["exact_target"] for row in candidates),
        "candidates": candidates,
    }
