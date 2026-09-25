from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from statistics import median
from typing import Any, Mapping, Sequence

from .actions import (
    Action,
    AddArc,
    AddCircle,
    AddLine,
    End,
    EndLoop,
    Extrude,
    RegisterProfile,
    StartLoop,
    action_to_text,
)
from .failure_certificate import build_failure_certificate
from .lm.parsing import parse_action_line
from .natural_repair_recall import (
    apply_patch_operations,
    evaluate_repaired_history,
    topology_contract_distance,
)
from .topology_history_repair import canonical_action_lines
from .topoplan_v2 import infer_phase
from .verifier import TopoVerifier, VerificationState


@dataclass(frozen=True)
class RepairObligation:
    """Action-free topology obligations induced by the first failed transition."""

    failure_step: int
    violation: str
    micro_phase: str
    affected_objects: dict[str, Any]
    missing_prerequisite: str
    topology_delta: dict[str, Any]
    open_transaction: dict[str, Any]
    semantic_repair_start: int
    repair_scope: str
    candidate_input_audit: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TypedRepairTransaction:
    """A bounded typed edit compiled from verifier state and an explicit goal."""

    transaction_id: str
    operator: str
    repair_start: int
    repair_end: int
    replacement: tuple[str, ...]
    rollback_length: int
    provenance: dict[str, Any]

    def patch(self) -> dict[str, Any]:
        return {
            "edits": [
                {
                    "start": self.repair_start,
                    "end": self.repair_end,
                    "replacement": list(self.replacement),
                }
            ]
        }

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["replacement"] = list(self.replacement)
        row["patch"] = self.patch()
        return row


@dataclass(frozen=True)
class EvaluatedRepairTransaction:
    transaction: TypedRepairTransaction
    repaired_actions: tuple[str, ...]
    immediate_valid: bool
    k_step_survivable: bool
    valid: bool
    ended: bool
    intent_satisfied: bool
    contract_distance: int
    valid_prefix_length: int
    preservation_ratio: float
    edit_action_cost: int

    @property
    def valid_and_intent_satisfied(self) -> bool:
        return self.valid and self.ended and self.intent_satisfied

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["transaction"] = self.transaction.to_dict()
        row["repaired_actions"] = list(self.repaired_actions)
        row["valid_and_intent_satisfied"] = self.valid_and_intent_satisfied
        return row


@dataclass(frozen=True)
class ContractReconciliation:
    """Minimal edits that align already committed profiles with the contract."""

    edits: tuple[dict[str, Any], ...]
    reconciled_actions: tuple[str, ...]
    shifted_failure_step: int
    profile_mismatches_fixed: int
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["edits"] = [dict(edit) for edit in self.edits]
        row["reconciled_actions"] = list(self.reconciled_actions)
        return row


@dataclass(frozen=True)
class TopologyRepairPlan:
    obligation: RepairObligation
    reconciliation: ContractReconciliation | None
    completion: EvaluatedRepairTransaction
    final_actions: tuple[str, ...]
    action_lcs_preservation: float
    first_changed_action: int

    @property
    def valid_and_intent_satisfied(self) -> bool:
        return self.completion.valid_and_intent_satisfied

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation": self.obligation.to_dict(),
            "reconciliation": (
                self.reconciliation.to_dict()
                if self.reconciliation is not None
                else None
            ),
            "completion": self.completion.to_dict(),
            "final_actions": list(self.final_actions),
            "action_lcs_preservation": self.action_lcs_preservation,
            "first_changed_action": self.first_changed_action,
            "valid_and_intent_satisfied": self.valid_and_intent_satisfied,
        }


@dataclass(frozen=True)
class _LoopSpan:
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class _ProfileSpan:
    profile_id: str
    face_start: int
    end_face: int
    register_end: int
    loops: tuple[_LoopSpan, ...]


def _replay_prefix(
    lines: Sequence[str], verifier: TopoVerifier
) -> tuple[VerificationState, list[Action]] | None:
    state = verifier.initial_state()
    actions: list[Action] = []
    for line in lines:
        action = parse_action_line(str(line))
        if action is None:
            return None
        result = verifier.step(state, action)
        if not result.valid or result.next_state is None:
            return None
        state = result.next_state
        actions.append(action)
    return state, actions


def _profile_rows(state: VerificationState) -> list[dict[str, Any]]:
    return [
        {
            "profile_id": str(profile_id),
            "profile_index": index,
            "loop_roles": sorted(loop.kind for loop in state.profiles[profile_id].loops),
        }
        for index, profile_id in enumerate(sorted(state.profiles))
    ]


def _topology_delta(
    state: VerificationState, contract: Mapping[str, Any]
) -> dict[str, Any]:
    existing_profiles = _profile_rows(state)
    desired_profiles = list(contract.get("profiles") or [])
    desired_graph = list(contract.get("extrusion_graph") or [])
    observed_operations = Counter(str(row["op"]) for row in state.extrusions)
    desired_operations = Counter(contract.get("operation_counts") or {})
    return {
        "profiles_remaining": max(0, len(desired_profiles) - len(existing_profiles)),
        "inner_loops_remaining": max(
            0,
            int(contract.get("inner_loop_count", 0) or 0)
            - sum(
                role == "inner"
                for row in existing_profiles
                for role in row["loop_roles"]
            ),
        ),
        "extrusions_remaining": max(0, len(desired_graph) - len(state.extrusions)),
        "operation_deficit": dict(
            sorted((desired_operations - observed_operations).items())
        ),
    }


def _open_transaction(lines: Sequence[str]) -> dict[str, Any]:
    sketch_start: int | None = None
    face_start: int | None = None
    loop_start: int | None = None
    for index, line in enumerate(lines):
        action = parse_action_line(str(line))
        if action is None:
            continue
        name = action.__class__.__name__
        if name == "StartSketch":
            sketch_start = index
        elif name == "StartFace":
            face_start = index
        elif isinstance(action, StartLoop):
            loop_start = index
        elif name == "EndLoop":
            loop_start = None
        elif name == "EndFace":
            face_start = None
        elif name == "EndSketch":
            sketch_start = None
    return {
        "sketch_start": sketch_start,
        "face_start": face_start,
        "loop_start": loop_start,
    }


def derive_repair_obligation(
    observed_lines: Sequence[str],
    *,
    failure_step: int,
    contract: Mapping[str, Any],
    verifier: TopoVerifier | None = None,
) -> RepairObligation:
    verifier = verifier or TopoVerifier()
    observed = canonical_action_lines(observed_lines)
    if not 0 <= failure_step < len(observed):
        raise ValueError("failure_step is outside observed history")
    replayed = _replay_prefix(observed[:failure_step], verifier)
    if replayed is None:
        raise ValueError("history before failure_step is not verifier-valid")
    state, actions = replayed
    attempted = parse_action_line(observed[failure_step])
    if attempted is None:
        raise ValueError("failed action is unparseable")
    target_stats = {
        "profile_count": int(contract.get("profile_count", 0) or 0),
        "hole_count": int(contract.get("inner_loop_count", 0) or 0),
        "extrude_count": int(contract.get("extrusion_count", 0) or 0),
    }
    phase = infer_phase(actions, target_stats, state=state)
    certificate = build_failure_certificate(
        prefix_actions=actions,
        state=state,
        candidate_action=attempted,
        target_stats=target_stats,
        micro_phase=phase,
        failing_step=failure_step,
        verifier=verifier,
    )
    violation = (
        certificate.violation if certificate is not None else "incomplete_history"
    )
    semantic_repair_start = _latest_contract_compatible_boundary(
        observed, failure_step, contract, verifier
    )
    prefix_contract_compatible = _contract_prefix_compatible(state, contract)
    return RepairObligation(
        failure_step=failure_step,
        violation=violation,
        micro_phase=str(phase),
        affected_objects=(
            dict(certificate.affected_objects)
            if certificate is not None
            else {"action_type": attempted.__class__.__name__}
        ),
        missing_prerequisite=(
            certificate.expected_prerequisite
            if certificate is not None
            else "complete the outstanding topology contract"
        ),
        topology_delta=_topology_delta(state, contract),
        open_transaction=_open_transaction(observed[:failure_step]),
        semantic_repair_start=semantic_repair_start,
        repair_scope=(
            "execution_local"
            if prefix_contract_compatible
            else "contract_divergence"
        ),
        candidate_input_audit={
            "uses_observed_history": True,
            "uses_dynamic_verifier_state": True,
            "uses_explicit_topology_contract": True,
            "uses_target_action_history": False,
            "uses_oracle_patch": False,
            "uses_candidate_validity_as_input": False,
        },
    )


def _append_valid(
    state: VerificationState,
    lines: list[str],
    action: Action,
    verifier: TopoVerifier,
) -> VerificationState | None:
    result = verifier.step(state, action)
    if not result.valid or result.next_state is None:
        return None
    lines.append(action_to_text(action))
    return result.next_state


def _default_depth(lines: Sequence[str]) -> float:
    depths = []
    for line in lines:
        action = parse_action_line(str(line))
        if isinstance(action, Extrude) and action.depth > 0:
            depths.append(float(action.depth))
    return float(median(depths)) if depths else 1.0


def _canonical_outer_actions(profile_index: int) -> list[Action]:
    x0 = float(profile_index * 3)
    x1 = x0 + 2.0
    y0, y1 = 0.0, 2.0
    return [
        AddLine((x0, y0), (x1, y0)),
        AddLine((x1, y0), (x1, y1)),
        AddLine((x1, y1), (x0, y1)),
        AddLine((x0, y1), (x0, y0)),
    ]


def _inner_circle_action(
    state: VerificationState, inner_index: int, inner_total: int
) -> AddCircle | None:
    outer = next(
        (loop for loop in state.current_face_loops if loop.kind == "outer"), None
    )
    if outer is None or not outer.points:
        return None
    xs = [point[0] for point in outer.points]
    ys = [point[1] for point in outer.points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return None
    center = (
        min(xs) + (inner_index + 1) * width / (inner_total + 1),
        min(ys) + 0.5 * height,
    )
    radius = 0.12 * min(width, height) / max(1, inner_total)
    return AddCircle(center=center, radius=radius)


def _inner_circle_candidates(state: VerificationState) -> list[AddCircle]:
    outer = next(
        (loop for loop in state.current_face_loops if loop.kind == "outer"), None
    )
    if outer is None or not outer.points:
        return []
    xs = [point[0] for point in outer.points]
    ys = [point[1] for point in outer.points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return []
    radius = 0.045 * min(width, height)
    fractions = (0.2, 0.35, 0.5, 0.65, 0.8)
    return [
        AddCircle(
            center=(min(xs) + x * width, min(ys) + y * height),
            radius=radius,
        )
        for y in fractions
        for x in fractions
    ]


def _append_inner_loop(
    state: VerificationState,
    lines: list[str],
    verifier: TopoVerifier,
) -> VerificationState | None:
    next_state = _append_valid(state, lines, StartLoop("inner"), verifier)
    if next_state is None:
        return None
    state = next_state
    chosen: VerificationState | None = None
    chosen_action: AddCircle | None = None
    for candidate in _inner_circle_candidates(state):
        result = verifier.step(state, candidate)
        if result.valid and result.next_state is not None:
            chosen = result.next_state
            chosen_action = candidate
            break
    if chosen is None or chosen_action is None:
        return None
    lines.append(action_to_text(chosen_action))
    return _append_valid(chosen, lines, EndLoop(), verifier)


def _profile_transaction_spans(
    lines: Sequence[str], failure_step: int
) -> list[_ProfileSpan]:
    spans: list[_ProfileSpan] = []
    face_start: int | None = None
    end_face: int | None = None
    loop_start: int | None = None
    loop_kind: str | None = None
    loops: list[_LoopSpan] = []
    for index, line in enumerate(lines[:failure_step]):
        action = parse_action_line(str(line))
        if action is None:
            continue
        name = action.__class__.__name__
        if name == "StartFace":
            face_start = index
            end_face = None
            loops = []
        elif isinstance(action, StartLoop):
            loop_start = index
            loop_kind = action.kind
        elif isinstance(action, EndLoop) and loop_start is not None and loop_kind:
            loops.append(_LoopSpan(loop_kind, loop_start, index + 1))
            loop_start = None
            loop_kind = None
        elif name == "EndFace":
            end_face = index
        elif isinstance(action, RegisterProfile):
            if face_start is not None and end_face is not None:
                spans.append(
                    _ProfileSpan(
                        profile_id=action.profile_id,
                        face_start=face_start,
                        end_face=end_face,
                        register_end=index + 1,
                        loops=tuple(loops),
                    )
                )
            face_start = None
            end_face = None
            loops = []
    return spans


def reconcile_committed_topology(
    observed_lines: Sequence[str],
    *,
    failure_step: int,
    contract: Mapping[str, Any],
    verifier: TopoVerifier | None = None,
) -> ContractReconciliation | None:
    """Patch committed profile loop roles without reading target geometry."""

    verifier = verifier or TopoVerifier()
    observed = canonical_action_lines(observed_lines)
    spans = sorted(
        _profile_transaction_spans(observed, failure_step),
        key=lambda span: span.profile_id,
    )
    desired_profiles = list(contract.get("profiles") or [])
    if len(spans) > len(desired_profiles):
        return None
    edits: list[dict[str, Any]] = []
    mismatches = 0
    for profile_index, span in enumerate(spans):
        desired_roles = sorted(
            str(role)
            for role in desired_profiles[profile_index].get("loop_roles") or []
        )
        observed_inner = [loop for loop in span.loops if loop.kind == "inner"]
        observed_outer = [loop for loop in span.loops if loop.kind == "outer"]
        if len(observed_outer) != desired_roles.count("outer"):
            return None
        desired_inner = desired_roles.count("inner")
        if len(observed_inner) == desired_inner:
            continue
        mismatches += 1
        if len(observed_inner) > desired_inner:
            for loop in observed_inner[desired_inner:]:
                edits.append(
                    {"start": loop.start, "end": loop.end, "replacement": []}
                )
            continue

        replayed = _replay_prefix(observed[: span.end_face], verifier)
        if replayed is None:
            return None
        state, _ = replayed
        inserted: list[str] = []
        for _ in range(len(observed_inner), desired_inner):
            next_state = _append_inner_loop(state, inserted, verifier)
            if next_state is None:
                return None
            state = next_state
        edits.append(
            {
                "start": span.end_face,
                "end": span.end_face,
                "replacement": inserted,
            }
        )

    if not edits:
        return ContractReconciliation(
            edits=(),
            reconciled_actions=tuple(observed),
            shifted_failure_step=failure_step,
            profile_mismatches_fixed=0,
            provenance={
                "compiler": "topoverifier_profile_role_reconciliation_v1",
                "uses_target_action_history": False,
                "uses_oracle_geometry": False,
            },
        )
    reconciled = apply_patch_operations(observed, edits)
    shift = sum(
        len(edit["replacement"]) - (int(edit["end"]) - int(edit["start"]))
        for edit in edits
        if int(edit["start"]) < failure_step
    )
    shifted_failure = failure_step + shift
    replayed = _replay_prefix(reconciled[:shifted_failure], verifier)
    if replayed is None:
        return None
    state, _ = replayed
    if not _contract_prefix_compatible(state, contract):
        return None
    return ContractReconciliation(
        edits=tuple(edits),
        reconciled_actions=tuple(reconciled),
        shifted_failure_step=shifted_failure,
        profile_mismatches_fixed=mismatches,
        provenance={
            "compiler": "topoverifier_profile_role_reconciliation_v1",
            "uses_target_action_history": False,
            "uses_oracle_geometry": False,
            "preserves_outer_loops": True,
            "preserves_extrusion_parameters": True,
        },
    )


def _complete_open_loop(
    state: VerificationState,
    lines: list[str],
    *,
    profile_index: int,
    desired_inner_total: int,
    verifier: TopoVerifier,
) -> VerificationState | None:
    loop = state.current_loop
    if loop is None:
        return state
    if loop.circle_closed:
        return _append_valid(state, lines, EndLoop(), verifier)

    if not loop.segments:
        if loop.kind == "outer":
            actions = _canonical_outer_actions(profile_index)
        else:
            circle = _inner_circle_action(
                state, len([x for x in state.current_face_loops if x.kind == "inner"]), desired_inner_total
            )
            actions = [circle] if circle is not None else []
        if not actions:
            return None
        for action in actions:
            next_state = _append_valid(state, lines, action, verifier)
            if next_state is None:
                return None
            state = next_state
    elif len(loop.segments) == 1 and loop.start is not None and loop.tail is not None:
        x0, y0 = loop.start
        x1, y1 = loop.tail
        dx, dy = x1 - x0, y1 - y0
        scale = max((dx * dx + dy * dy) ** 0.5, 1.0) * 0.25
        nx, ny = -dy, dx
        norm = max((nx * nx + ny * ny) ** 0.5, 1e-9)
        offset = (scale * nx / norm, scale * ny / norm)
        actions = [
            AddLine(loop.tail, (x1 + offset[0], y1 + offset[1])),
            AddLine(
                (x1 + offset[0], y1 + offset[1]),
                (x0 + offset[0], y0 + offset[1]),
            ),
            AddLine((x0 + offset[0], y0 + offset[1]), loop.start),
        ]
        for action in actions:
            next_state = _append_valid(state, lines, action, verifier)
            if next_state is None:
                return None
            state = next_state
    elif loop.start is not None and loop.tail is not None and loop.tail != loop.start:
        next_state = _append_valid(
            state, lines, AddLine(loop.tail, loop.start), verifier
        )
        if next_state is None:
            return None
        state = next_state
    return _append_valid(state, lines, EndLoop(), verifier)


def _contract_prefix_compatible(
    state: VerificationState, contract: Mapping[str, Any]
) -> bool:
    desired_profiles = list(contract.get("profiles") or [])
    existing_profiles = _profile_rows(state)
    if len(existing_profiles) > len(desired_profiles):
        return False
    for existing, desired in zip(existing_profiles, desired_profiles):
        if existing["loop_roles"] != sorted(str(x) for x in desired.get("loop_roles") or []):
            return False
    desired_graph = list(contract.get("extrusion_graph") or [])
    if len(state.extrusions) > len(desired_graph):
        return False
    profile_index = {
        str(row["profile_id"]): int(row["profile_index"])
        for row in existing_profiles
    }
    for existing, desired in zip(state.extrusions, desired_graph):
        if profile_index.get(str(existing["profile_id"]), -1) != int(
            desired.get("profile_index", -1)
        ):
            return False
        if str(existing["op"]) != str(desired.get("operation") or ""):
            return False
    return True


def _latest_contract_compatible_boundary(
    observed: Sequence[str],
    failure_step: int,
    contract: Mapping[str, Any],
    verifier: TopoVerifier,
) -> int:
    for start in range(failure_step, -1, -1):
        replayed = _replay_prefix(observed[:start], verifier)
        if replayed is None:
            continue
        state, _ = replayed
        if (
            state.stack in ([], ["sketch"])
            and state.pending_face is None
            and _contract_prefix_compatible(state, contract)
        ):
            return start
    return 0


def compile_goal_completion(
    prefix_lines: Sequence[str],
    contract: Mapping[str, Any],
    *,
    support_lines: Sequence[str] = (),
    max_actions: int = 160,
    verifier: TopoVerifier | None = None,
) -> list[str] | None:
    """Compile a topology-complete suffix from an arbitrary valid prefix state."""

    verifier = verifier or TopoVerifier()
    replayed = _replay_prefix(prefix_lines, verifier)
    if replayed is None:
        return None
    state, _ = replayed
    if state.ended or not _contract_prefix_compatible(state, contract):
        return [] if state.ended else None
    desired_profiles = list(contract.get("profiles") or [])
    lines: list[str] = []

    if state.current_loop is not None:
        profile_index = len(state.profiles)
        if profile_index >= len(desired_profiles):
            return None
        desired_roles = [
            str(role) for role in desired_profiles[profile_index].get("loop_roles") or []
        ]
        state = _complete_open_loop(
            state,
            lines,
            profile_index=profile_index,
            desired_inner_total=sum(role == "inner" for role in desired_roles),
            verifier=verifier,
        )
        if state is None:
            return None

    if state.stack and state.stack[-1] == "face":
        profile_index = len(state.profiles)
        if profile_index >= len(desired_profiles):
            return None
        desired_roles = sorted(
            str(role) for role in desired_profiles[profile_index].get("loop_roles") or []
        )
        current_roles = sorted(loop.kind for loop in state.current_face_loops)
        if any(current_roles.count(role) > desired_roles.count(role) for role in set(current_roles)):
            return None
        if "outer" not in current_roles:
            next_state = _append_valid(state, lines, StartLoop("outer"), verifier)
            if next_state is None:
                return None
            state = next_state
            for action in _canonical_outer_actions(profile_index):
                next_state = _append_valid(state, lines, action, verifier)
                if next_state is None:
                    return None
                state = next_state
            next_state = _append_valid(state, lines, EndLoop(), verifier)
            if next_state is None:
                return None
            state = next_state
            current_roles.append("outer")
        desired_inner = desired_roles.count("inner")
        current_inner = current_roles.count("inner")
        for inner_index in range(current_inner, desired_inner):
            next_state = _append_valid(state, lines, StartLoop("inner"), verifier)
            if next_state is None:
                return None
            state = next_state
            circle = _inner_circle_action(state, inner_index, desired_inner)
            if circle is None:
                return None
            next_state = _append_valid(state, lines, circle, verifier)
            if next_state is None:
                return None
            state = next_state
            next_state = _append_valid(state, lines, EndLoop(), verifier)
            if next_state is None:
                return None
            state = next_state
        end_face = parse_action_line("EndFace")
        assert end_face is not None
        next_state = _append_valid(state, lines, end_face, verifier)
        if next_state is None:
            return None
        state = next_state

    if state.pending_face is not None:
        profile_index = len(state.profiles)
        # Contract profile indices are defined by sorted profile IDs. New IDs
        # sort after every preserved native ID, so adding a repair profile does
        # not silently remap earlier extrusion references (notably profile_10).
        profile_id = f"zz_repair_profile_{profile_index:04d}"
        next_state = _append_valid(
            state, lines, RegisterProfile(profile_id), verifier
        )
        if next_state is None:
            return None
        state = next_state

    existing_profiles = _profile_rows(state)
    if len(existing_profiles) < len(desired_profiles):
        # CAD histories may interleave sketch/profile transactions with
        # extrusion transactions (Text2CAD emits one such block per part).
        # Existing extrusions are already checked against the contract prefix,
        # so adding later profiles is safe and avoids an unnecessary restart.
        if not state.stack:
            start_sketch = parse_action_line("StartSketch")
            assert start_sketch is not None
            next_state = _append_valid(state, lines, start_sketch, verifier)
            if next_state is None:
                return None
            state = next_state
        if state.stack != ["sketch"]:
            return None
        for profile_index in range(len(existing_profiles), len(desired_profiles)):
            start_face = parse_action_line("StartFace")
            assert start_face is not None
            next_state = _append_valid(state, lines, start_face, verifier)
            if next_state is None:
                return None
            state = next_state
            desired_roles = sorted(
                str(role)
                for role in desired_profiles[profile_index].get("loop_roles") or []
            )
            next_state = _append_valid(state, lines, StartLoop("outer"), verifier)
            if next_state is None:
                return None
            state = next_state
            for action in _canonical_outer_actions(profile_index):
                next_state = _append_valid(state, lines, action, verifier)
                if next_state is None:
                    return None
                state = next_state
            next_state = _append_valid(state, lines, EndLoop(), verifier)
            if next_state is None:
                return None
            state = next_state
            desired_inner = desired_roles.count("inner")
            for inner_index in range(desired_inner):
                next_state = _append_valid(state, lines, StartLoop("inner"), verifier)
                if next_state is None:
                    return None
                state = next_state
                circle = _inner_circle_action(state, inner_index, desired_inner)
                if circle is None:
                    return None
                next_state = _append_valid(state, lines, circle, verifier)
                if next_state is None:
                    return None
                state = next_state
                next_state = _append_valid(state, lines, EndLoop(), verifier)
                if next_state is None:
                    return None
                state = next_state
            end_face = parse_action_line("EndFace")
            assert end_face is not None
            next_state = _append_valid(state, lines, end_face, verifier)
            if next_state is None:
                return None
            state = next_state
            profile_id = f"zz_repair_profile_{profile_index:04d}"
            next_state = _append_valid(
                state, lines, RegisterProfile(profile_id), verifier
            )
            if next_state is None:
                return None
            state = next_state

    if state.stack == ["sketch"]:
        end_sketch = parse_action_line("EndSketch")
        assert end_sketch is not None
        next_state = _append_valid(state, lines, end_sketch, verifier)
        if next_state is None:
            return None
        state = next_state
    if state.stack or state.pending_face is not None:
        return None

    profile_rows = _profile_rows(state)
    profile_ids = [str(row["profile_id"]) for row in profile_rows]
    desired_graph = list(contract.get("extrusion_graph") or [])
    depth = _default_depth([*prefix_lines, *support_lines])
    for row in desired_graph[len(state.extrusions) :]:
        profile_index = int(row.get("profile_index", -1))
        operation = str(row.get("operation") or "")
        if not 0 <= profile_index < len(profile_ids):
            return None
        if operation not in {"add", "cut", "intersect"}:
            return None
        if row.get("depth") is not None:
            action_depth = float(row["depth"])
        else:
            # A topology-only contract omits geometric depth. Using identical
            # add/cut depths on the same profile can erase the entire solid in
            # OCC, so subtractive defaults are conservatively bounded below
            # the additive construction depth. This is an execution-safe
            # canonical value, not a claim of parameter-intent recovery.
            action_depth = depth if operation == "add" else 0.5 * depth
        next_state = _append_valid(
            state,
            lines,
            Extrude(profile_ids[profile_index], action_depth, operation),
            verifier,
        )
        if next_state is None:
            return None
        state = next_state
    if len(state.extrusions) != len(desired_graph):
        return None
    next_state = _append_valid(state, lines, End(), verifier)
    if next_state is None:
        return None
    if len(lines) > max_actions:
        return None
    return lines


def _candidate_starts(
    observed: Sequence[str],
    failure_step: int,
    max_rollback: int,
    contract: Mapping[str, Any],
) -> list[tuple[int, str]]:
    open_tx = _open_transaction(observed[:failure_step])
    starts: list[tuple[int, str]] = [(failure_step, "complete_at_failure")]
    for key, operator in (
        ("loop_start", "rollback_loop_transaction"),
        ("face_start", "rollback_face_transaction"),
        ("sketch_start", "rollback_sketch_transaction"),
    ):
        value = open_tx.get(key)
        if value is not None and failure_step - int(value) <= max_rollback:
            starts.append((int(value), operator))

    lower = max(0, failure_step - max_rollback)
    verifier = TopoVerifier()
    for start in range(failure_step, lower - 1, -1):
        replayed = _replay_prefix(observed[:start], verifier)
        if replayed is None:
            continue
        state, _ = replayed
        if state.stack in ([], ["sketch"]) and state.pending_face is None:
            starts.append((start, "boundary_regenerate"))
            break

    # Immediate legality can remain intact after an earlier topology-goal
    # divergence (for example, registering a hole-free profile when the goal
    # requires a hole). Find the latest clean prefix that is still compatible
    # with the explicit contract, even when it lies outside the local window.
    starts.append(
        (
            _latest_contract_compatible_boundary(
                observed, failure_step, contract, verifier
            ),
            "contract_divergence_repair",
        )
    )
    unique: list[tuple[int, str]] = []
    seen: set[int] = set()
    for start, operator in starts:
        if start in seen:
            continue
        seen.add(start)
        unique.append((start, operator))
    return unique


def compile_typed_transactions(
    observed_lines: Sequence[str],
    *,
    failure_step: int,
    contract: Mapping[str, Any],
    max_rollback: int = 32,
    max_replacement_actions: int = 160,
    verifier: TopoVerifier | None = None,
) -> tuple[RepairObligation, list[TypedRepairTransaction]]:
    verifier = verifier or TopoVerifier()
    observed = canonical_action_lines(observed_lines)
    obligation = derive_repair_obligation(
        observed,
        failure_step=failure_step,
        contract=contract,
        verifier=verifier,
    )
    transactions: list[TypedRepairTransaction] = []
    for index, (start, operator) in enumerate(
        _candidate_starts(observed, failure_step, max_rollback, contract)
    ):
        suffix = compile_goal_completion(
            observed[:start],
            contract,
            support_lines=observed,
            max_actions=max_replacement_actions,
            verifier=verifier,
        )
        if suffix is None:
            continue
        transactions.append(
            TypedRepairTransaction(
                transaction_id=f"tx-{index:02d}-{operator}",
                operator=operator,
                repair_start=start,
                repair_end=len(observed),
                replacement=tuple(suffix),
                rollback_length=max(0, failure_step - start),
                provenance={
                    "compiler": "topoverifier_typed_transaction_v1",
                    "violation": obligation.violation,
                    "micro_phase": obligation.micro_phase,
                    "uses_target_action_history": False,
                    "uses_oracle_patch": False,
                    "contract_level": "topology",
                    "restart_fallback": start == 0,
                    "near_full_regeneration": start <= 1,
                },
            )
        )
    return obligation, transactions


def evaluate_transaction(
    observed_lines: Sequence[str],
    transaction: TypedRepairTransaction,
    contract: Mapping[str, Any],
    *,
    failure_step: int,
    horizon: int = 6,
) -> EvaluatedRepairTransaction:
    observed = canonical_action_lines(observed_lines)
    repaired = apply_patch_operations(observed, transaction.patch()["edits"])
    evaluation = evaluate_repaired_history(repaired, contract)
    local_end = min(
        len(repaired), transaction.repair_start + max(1, int(horizon))
    )
    local = evaluate_repaired_history(repaired[:local_end], contract)
    immediate_end = min(len(repaired), transaction.repair_start + 1)
    immediate = evaluate_repaired_history(repaired[:immediate_end], contract)
    edit_cost = (
        transaction.repair_end
        - transaction.repair_start
        + len(transaction.replacement)
    )
    return EvaluatedRepairTransaction(
        transaction=transaction,
        repaired_actions=tuple(repaired),
        immediate_valid=bool(immediate["valid"]),
        k_step_survivable=bool(local["valid"]),
        valid=bool(evaluation["valid"]),
        ended=bool(evaluation["ended"]),
        intent_satisfied=bool(evaluation["intent_satisfied"]),
        contract_distance=int(evaluation["contract_distance"]),
        valid_prefix_length=int(evaluation["valid_prefix_length"]),
        preservation_ratio=(
            transaction.repair_start / len(observed) if observed else 1.0
        ),
        edit_action_cost=edit_cost,
    )


def select_transaction(
    evaluations: Sequence[EvaluatedRepairTransaction],
) -> EvaluatedRepairTransaction | None:
    if not evaluations:
        return None

    def key(item: EvaluatedRepairTransaction) -> tuple[Any, ...]:
        return (
            int(item.valid_and_intent_satisfied),
            int(item.k_step_survivable),
            int(item.valid),
            -item.contract_distance,
            item.transaction.repair_start,
            -item.edit_action_cost,
            -len(item.transaction.replacement),
        )

    return max(evaluations, key=key)


def _action_lcs_preservation(
    original: Sequence[str], repaired: Sequence[str]
) -> float:
    if not original:
        return 1.0
    matcher = SequenceMatcher(a=list(original), b=list(repaired), autojunk=False)
    preserved = sum(block.size for block in matcher.get_matching_blocks())
    return preserved / len(original)


def _common_prefix_length(first: Sequence[str], second: Sequence[str]) -> int:
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    return length


def repair_topology_history(
    observed_lines: Sequence[str],
    *,
    failure_step: int,
    contract: Mapping[str, Any],
    horizon: int = 6,
    max_rollback: int = 32,
    max_replacement_actions: int = 160,
    verifier: TopoVerifier | None = None,
) -> TopologyRepairPlan | None:
    """Repair committed topology deltas, then complete the execution-local fault.

    A transaction-local reconciliation is preferred when it preserves more of
    the observed history. When already committed surplus topology cannot be
    edited in place, the compiler falls back to the latest clean boundary that
    is still a prefix of the explicit contract.
    """

    verifier = verifier or TopoVerifier()
    original = canonical_action_lines(observed_lines)
    original_obligation = derive_repair_obligation(
        original,
        failure_step=failure_step,
        contract=contract,
        verifier=verifier,
    )
    reconciliation = reconcile_committed_topology(
        original,
        failure_step=failure_step,
        contract=contract,
        verifier=verifier,
    )
    working = (
        list(reconciliation.reconciled_actions)
        if reconciliation is not None
        else original
    )
    working_failure = (
        reconciliation.shifted_failure_step
        if reconciliation is not None
        else failure_step
    )
    _working_obligation, transactions = compile_typed_transactions(
        working,
        failure_step=working_failure,
        contract=contract,
        max_rollback=max_rollback,
        max_replacement_actions=max_replacement_actions,
        verifier=verifier,
    )
    evaluations = [
        evaluate_transaction(
            working,
            transaction,
            contract,
            failure_step=working_failure,
            horizon=horizon,
        )
        for transaction in transactions
    ]
    local = select_transaction(
        [
            evaluation
            for evaluation in evaluations
            if evaluation.transaction.operator != "contract_divergence_repair"
        ]
    )
    selected = (
        local
        if local is not None and local.valid_and_intent_satisfied
        else select_transaction(evaluations)
    )
    staged_plan = None
    if selected is not None:
        final_actions = tuple(selected.repaired_actions)
        staged_plan = TopologyRepairPlan(
            obligation=original_obligation,
            reconciliation=reconciliation,
            completion=selected,
            final_actions=final_actions,
            action_lcs_preservation=_action_lcs_preservation(original, final_actions),
            first_changed_action=_common_prefix_length(original, final_actions),
        )

    # A clean-boundary recompile is the safe final transaction for committed
    # surplus profiles or incompatible extrusion graphs. It reads the same
    # anonymous contract as local reconciliation and never reads target actions.
    from .contract_repair_search import search_contract_repair

    boundary = search_contract_repair(
        original,
        failure_step=failure_step,
        contract=contract,
        max_rollback=len(original),
        max_replacement_actions=max_replacement_actions,
    )
    boundary_plan = None
    if boundary is not None:
        boundary_transaction = TypedRepairTransaction(
            transaction_id=f"contract_boundary_recompile:{boundary.rollback_start}",
            operator="contract_boundary_recompile",
            repair_start=boundary.rollback_start,
            repair_end=len(original),
            replacement=tuple(boundary.replacement),
            rollback_length=len(original) - boundary.rollback_start,
            provenance={
                "compiler": "topoverifier_contract_boundary_recompile_v1",
                "contract_level": "topology",
                "uses_target_action_history": False,
                "uses_oracle_geometry": False,
                "uses_oracle_patch": False,
                "verifier_calls": boundary.verifier_calls,
            },
        )
        boundary_evaluation = evaluate_transaction(
            original,
            boundary_transaction,
            contract,
            failure_step=failure_step,
            horizon=horizon,
        )
        if boundary_evaluation.valid_and_intent_satisfied:
            boundary_actions = tuple(boundary_evaluation.repaired_actions)
            boundary_plan = TopologyRepairPlan(
                obligation=original_obligation,
                reconciliation=None,
                completion=boundary_evaluation,
                final_actions=boundary_actions,
                action_lcs_preservation=_action_lcs_preservation(
                    original, boundary_actions
                ),
                first_changed_action=_common_prefix_length(
                    original, boundary_actions
                ),
            )

    candidates = [
        plan for plan in (staged_plan, boundary_plan) if plan is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda plan: (
            int(plan.valid_and_intent_satisfied),
            plan.action_lcs_preservation,
            plan.first_changed_action,
            -plan.completion.edit_action_cost,
            int(plan.completion.transaction.operator != "contract_boundary_recompile"),
        ),
    )
