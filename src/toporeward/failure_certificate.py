from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from .actions import (
    Action,
    AddArc,
    AddCircle,
    AddLine,
    End,
    EndFace,
    EndLoop,
    EndSketch,
    Extrude,
    RegisterProfile,
    StartFace,
    StartLoop,
    StartSketch,
)
from .structure_stats import structure_stats
from .structured_consequence import evaluate_structured_consequence
from .verifier import TopoVerifier, VerificationState


RepairOperation = Literal["replace", "insert_before", "delete"]


@dataclass(frozen=True)
class FailureCertificate:
    """Structured certificate for the earliest inconsistent history action.

    The certificate reports a verifier-observed violation and a bounded repair
    locus. It does not expose an oracle repair action or claim causal minimality.
    """

    failing_step: int
    micro_phase: str
    violation: str
    immediate_valid: bool
    affected_objects: dict[str, Any]
    expected_prerequisite: str
    observed_state: dict[str, Any]
    repair_locus: dict[str, Any]
    verifier_evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _counts(target_stats: dict[str, Any]) -> dict[str, int]:
    return {
        "profile_count": int(target_stats.get("profile_count", 0) or 0),
        "hole_count": int(target_stats.get("hole_count", 0) or 0),
        "extrude_count": int(target_stats.get("extrude_count", 0) or 0),
    }


def _remaining(prefix_actions: Sequence[Action], target_stats: dict[str, Any]) -> dict[str, int]:
    current = structure_stats(list(prefix_actions))
    target = _counts(target_stats)
    return {
        key: max(0, target[key] - int(current.get(key, 0) or 0))
        for key in target
    }


def _action_objects(action: Action) -> dict[str, Any]:
    if isinstance(action, Extrude):
        return {
            "action_type": "Extrude",
            "profile_id": action.profile_id,
            "depth": action.depth,
            "operation": action.op,
        }
    if isinstance(action, RegisterProfile):
        return {"action_type": "RegisterProfile", "profile_id": action.profile_id}
    if isinstance(action, StartLoop):
        return {"action_type": "StartLoop", "loop_kind": action.kind}
    if isinstance(action, AddLine):
        return {"action_type": "AddLine", "start": action.start, "end": action.end}
    if isinstance(action, AddArc):
        return {
            "action_type": "AddArc",
            "start": action.start,
            "mid": action.mid,
            "end": action.end,
        }
    if isinstance(action, AddCircle):
        return {
            "action_type": "AddCircle",
            "center": action.center,
            "radius": action.radius,
        }
    return {"action_type": action.__class__.__name__}


def _state_summary(
    state: VerificationState,
    prefix_actions: Sequence[Action],
    target_stats: dict[str, Any],
) -> dict[str, Any]:
    loop = state.current_loop
    return {
        "hierarchy_stack": list(state.stack),
        "open_loop_kind": loop.kind if loop is not None else None,
        "loop_start": loop.start if loop is not None else None,
        "loop_tail": loop.tail if loop is not None else None,
        "loop_points": len(loop.points) if loop is not None else 0,
        "loop_segments": len(loop.segments) if loop is not None else 0,
        "pending_face": state.pending_face is not None,
        "registered_profiles": sorted(state.profiles),
        "extrusions_done": len(state.extrusions),
        "remaining": _remaining(prefix_actions, target_stats),
    }


def _prerequisite(violation: str, state: VerificationState) -> str:
    loop = state.current_loop
    if violation == "hierarchy_error":
        top = state.stack[-1] if state.stack else "ROOT"
        return f"emit an action admitted by hierarchy state {top}"
    if violation == "endpoint_discontinuity":
        return f"curve start must equal current loop tail {loop.tail if loop else None}"
    if violation == "unclosed_loop":
        return f"loop tail must return to loop start {loop.start if loop else None}"
    if violation == "invalid_profile_reference":
        profiles = sorted(state.profiles)
        return f"reference one registered profile from {profiles}"
    if violation == "invalid_hole_containment":
        return "inner geometry must remain strictly inside the completed outer loop"
    if violation == "degenerate_curve":
        return "use non-zero geometry with distinct defining points"
    if violation == "self_intersection":
        return "new geometry must not intersect the existing loop except at closure"
    if violation == "wrong_reference":
        return "use the profile required by the next outstanding extrusion"
    if violation == "wrong_extrude_parameter":
        return "match the outstanding extrusion depth and boolean operation"
    if violation == "premature":
        return "complete the remaining topology goals before closing this phase"
    if violation == "overbuild":
        return "stop extending a topology goal that is already satisfied"
    if violation == "wrong_phase":
        return "select an action compatible with the current construction micro-phase"
    if violation == "dead_end_within_k":
        return "choose an action with a verifier-valid bounded continuation"
    return "satisfy the dynamic topology invariant reported by the verifier"


def _recommended_operation(violation: str, action: Action) -> RepairOperation:
    if violation == "premature" and isinstance(action, (End, EndFace, EndSketch)):
        return "insert_before"
    if violation == "overbuild":
        return "delete"
    return "replace"


def _expected_extrude_violation(
    action: Action,
    expected_extrude: dict[str, Any] | None,
) -> str | None:
    if not isinstance(action, Extrude) or not expected_extrude:
        return None
    expected_profile = str(expected_extrude.get("profile_id") or "")
    if expected_profile and action.profile_id != expected_profile:
        return "wrong_reference"
    try:
        expected_depth = float(expected_extrude.get("depth"))
    except (TypeError, ValueError):
        expected_depth = action.depth
    expected_op = str(expected_extrude.get("op") or action.op)
    if abs(action.depth - expected_depth) > 1e-6 or action.op != expected_op:
        return "wrong_extrude_parameter"
    return None


def _premature_terminal_violation(
    action: Action,
    prefix_actions: Sequence[Action],
    target_stats: dict[str, Any],
    micro_phase: str,
) -> bool:
    """Recognize terminal actions whose missing prerequisite is outstanding work.

    The immediate verifier may report a hierarchy error (for example, ``End``
    before any extrusion). The process certificate retains that raw failure as
    evidence but classifies the repair-relevant cause as premature termination.
    """

    remaining = _remaining(prefix_actions, target_stats)
    phase = str(micro_phase or "").upper()
    if isinstance(action, End):
        return phase in {"EXTRUDE", "EXTRUDE_CONTINUE"} and remaining["extrude_count"] > 0
    if isinstance(action, EndSketch):
        return phase != "END_SKETCH" and remaining["profile_count"] > 0
    if isinstance(action, EndFace):
        return phase == "CLOSE_FACE_INNER" and remaining["hole_count"] > 0
    return False


def build_failure_certificate(
    *,
    prefix_actions: Sequence[Action],
    state: VerificationState,
    candidate_action: Action,
    target_stats: dict[str, Any],
    micro_phase: str,
    expected_extrude: dict[str, Any] | None = None,
    failing_step: int | None = None,
    verifier: TopoVerifier | None = None,
) -> FailureCertificate | None:
    """Return a certificate when one action violates immediate or process state."""

    verifier = verifier or TopoVerifier()
    consequence = evaluate_structured_consequence(
        prefix_actions=prefix_actions,
        state=state,
        candidate_action=candidate_action,
        target_stats=_counts(target_stats),
        continuation_provider=lambda _actions, _state: (),
        process_phase=micro_phase,
        horizon=1,
        beam_width=1,
        verifier=verifier,
    )
    violation: str | None = None
    if _premature_terminal_violation(
        candidate_action,
        prefix_actions,
        target_stats,
        micro_phase,
    ):
        violation = "premature"
    elif not consequence.immediate_valid:
        violation = consequence.failure_type
    else:
        violation = _expected_extrude_violation(candidate_action, expected_extrude)
        # The generic structured-consequence label uses global hole/profile
        # counts. That is appropriate for single-profile frontiers but can call
        # a safe face close "premature" when holes belong to a future face.
        # Premature terminal actions are therefore classified by micro-phase
        # above; the global label remains available in verifier_evidence.
        if violation is None and consequence.terminal_class == "overbuild":
            violation = "overbuild"
        if violation is None and not consequence.phase_consistent:
            violation = "wrong_phase"
        if violation is None and consequence.dead_end_within_k:
            violation = "dead_end_within_k"
    if violation is None:
        return None

    step = len(prefix_actions) if failing_step is None else int(failing_step)
    operation = _recommended_operation(violation, candidate_action)
    return FailureCertificate(
        failing_step=step,
        micro_phase=str(micro_phase or "UNKNOWN"),
        violation=violation,
        immediate_valid=consequence.immediate_valid,
        affected_objects=_action_objects(candidate_action),
        expected_prerequisite=_prerequisite(violation, state),
        observed_state=_state_summary(state, prefix_actions, target_stats),
        repair_locus={
            "start_step": step,
            "end_step": step,
            "recommended_operation": operation,
            "bounded_window": [max(0, step - 1), step],
        },
        verifier_evidence={
            "failure_type": consequence.failure_type,
            "terminal_class": consequence.terminal_class,
            "phase_consistent": consequence.phase_consistent,
            "reference_consistent": consequence.reference_consistent,
            "geometry_consistent": consequence.geometry_consistent,
            "target_satisfied": consequence.target_satisfied,
        },
    )
