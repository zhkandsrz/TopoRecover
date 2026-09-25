from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Literal, Sequence

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
from .rlvr import structure_stats
from .verifier import TopoVerifier, VerificationState


PROCESS_KEYS = ("profile_count", "hole_count", "extrude_count")
GEOMETRY_FAILURE_TYPES = frozenset(
    {
        "degenerate_curve",
        "endpoint_discontinuity",
        "invalid_hole_containment",
        "kernel_execution_failure",
        "self_intersection",
        "unclosed_loop",
    }
)
REFERENCE_FAILURE_TYPES = frozenset({"invalid_profile_reference"})
TerminalClass = Literal["none", "premature", "exact", "overbuild"]
ContinuationProvider = Callable[[Sequence[Action], VerificationState], Iterable[Action]]


@dataclass(frozen=True)
class StructuredConsequence:
    """Verifier-derived future consequence for one candidate action.

    The decision unit is always one action. ``lookahead_steps`` records how far
    verifier replay could continue after that action, up to ``horizon``.
    """

    immediate_valid: bool
    continuation_feasible: bool
    target_satisfied: bool
    exact_terminal: bool
    dead_end_within_k: bool
    terminal_class: TerminalClass
    progress_profiles: int
    progress_holes: int
    progress_extrudes: int
    overbuild_profiles: int
    overbuild_holes: int
    overbuild_extrudes: int
    reference_consistent: bool
    geometry_consistent: bool
    phase_consistent: bool
    lookahead_steps: int
    horizon: int
    failure_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Node:
    actions: tuple[Action, ...]
    state: VerificationState
    depth: int


def _target_counts(target_stats: dict[str, Any]) -> dict[str, int]:
    return {key: int(target_stats.get(key, 0) or 0) for key in PROCESS_KEYS}


def _goal_progress(
    before: dict[str, int],
    after: dict[str, int],
    target: dict[str, int],
) -> tuple[int, int, int]:
    values = []
    for key in PROCESS_KEYS:
        values.append(max(0, min(after[key], target[key]) - min(before[key], target[key])))
    return tuple(values)  # type: ignore[return-value]


def _overbuild(after: dict[str, int], target: dict[str, int]) -> tuple[int, int, int]:
    return tuple(max(0, after[key] - target[key]) for key in PROCESS_KEYS)  # type: ignore[return-value]


def _target_satisfied(stats: dict[str, int], target: dict[str, int]) -> bool:
    return all(stats[key] >= target[key] for key in PROCESS_KEYS)


def _terminal_class(
    *,
    ended: bool,
    satisfied: bool,
    overbuild: tuple[int, int, int],
) -> TerminalClass:
    if any(overbuild):
        return "overbuild"
    if ended and satisfied:
        return "exact"
    if ended and not satisfied:
        return "premature"
    return "none"


def _phase_terminal_override(
    *,
    candidate_action: Action,
    before: dict[str, int],
    target: dict[str, int],
    default: TerminalClass,
) -> TerminalClass:
    """Calibrate local process termination using goal counts.

    These rules describe generic hierarchy boundaries, not an oracle next
    action: closing a face while required holes remain is premature, while
    opening additional construction after its target count is met is overbuild.
    """

    if isinstance(candidate_action, EndFace) and before["hole_count"] < target["hole_count"]:
        return "premature"
    if isinstance(candidate_action, EndSketch) and before["profile_count"] < target["profile_count"]:
        return "premature"
    if isinstance(candidate_action, StartLoop) and candidate_action.kind == "inner":
        if before["hole_count"] >= target["hole_count"]:
            return "overbuild"
    if isinstance(candidate_action, StartFace) and before["profile_count"] >= target["profile_count"]:
        return "overbuild"
    return default


def phase_consistent_action(
    *,
    state: VerificationState,
    candidate_action: Action,
    before: dict[str, int],
    target: dict[str, int],
    process_phase: str | None = None,
) -> bool:
    """Return whether an action respects outstanding construction prerequisites.

    This is not an oracle-next-action test. It accepts whole action families at
    a verifier-derived process phase, while rejecting immediately legal actions
    that skip an unfinished hierarchy boundary. For example, ``Extrude`` is
    verifier-valid while a sketch remains open, but is phase-inconsistent when
    more profiles still need to be constructed.
    """

    phase = str(process_phase or "").upper()
    top = state.stack[-1] if state.stack else None
    loop = state.current_loop
    curve_or_close = isinstance(candidate_action, (AddLine, AddArc, AddCircle, EndLoop))

    if phase == "START_SKETCH":
        return isinstance(candidate_action, StartSketch)
    if phase == "START_FACE":
        return isinstance(candidate_action, StartFace)
    if phase == "REGISTER_PROFILE":
        return isinstance(candidate_action, RegisterProfile)
    if phase == "END_SKETCH":
        return isinstance(candidate_action, EndSketch)
    if phase in {"EXTRUDE", "EXTRUDE_CONTINUE"}:
        return isinstance(candidate_action, Extrude)
    if phase in {"EXTRUDE_END", "END"}:
        return isinstance(candidate_action, End)
    if phase == "CLOSE_FACE_INNER":
        return (
            isinstance(candidate_action, StartLoop)
            and candidate_action.kind == "inner"
        )
    if phase in {"CLOSE_FACE", "CLOSE_FACE_SAFE"}:
        return isinstance(candidate_action, EndFace)
    if phase == "BUILD_OUTER_START":
        return (
            isinstance(candidate_action, StartLoop)
            and candidate_action.kind == "outer"
        )
    if phase in {"BUILD_OUTER", "BUILD_OUTER_DRAW", "BUILD_OUTER_CLOSE"}:
        if top == "face" and loop is None:
            return (
                isinstance(candidate_action, StartLoop)
                and candidate_action.kind == "outer"
            )
        return bool(loop is not None and loop.kind == "outer" and curve_or_close)
    if phase == "BUILD_INNER":
        if top == "face" and loop is None:
            return (
                isinstance(candidate_action, StartLoop)
                and candidate_action.kind == "inner"
            )
        return bool(loop is not None and loop.kind == "inner" and curve_or_close)

    # Fallback for callers that do not provide a refined process phase.
    if top is None:
        if not state.profiles and not state.extrusions:
            return isinstance(candidate_action, StartSketch)
        if before["extrude_count"] < target["extrude_count"]:
            return isinstance(candidate_action, Extrude)
        return isinstance(candidate_action, End)
    if top == "sketch":
        if state.pending_face is not None:
            return isinstance(candidate_action, RegisterProfile)
        if before["profile_count"] < target["profile_count"]:
            return isinstance(candidate_action, StartFace)
        return isinstance(candidate_action, EndSketch)
    if top == "face" and loop is None:
        if not state.current_face_loops:
            return (
                isinstance(candidate_action, StartLoop)
                and candidate_action.kind == "outer"
            )
        if before["hole_count"] < target["hole_count"]:
            return (
                isinstance(candidate_action, StartLoop)
                and candidate_action.kind == "inner"
            )
        return isinstance(candidate_action, EndFace)
    if top == "loop" and loop is not None:
        return curve_or_close
    return False


def consequence_preference_key(value: StructuredConsequence) -> tuple[int, ...]:
    """Pre-registered lexicographic order without a hand-tuned scalar value."""

    terminal_quality = {
        "premature": 0,
        "overbuild": 0,
        "none": 1,
        "exact": 2,
    }[value.terminal_class]
    return (
        int(value.immediate_valid),
        int(value.reference_consistent),
        int(value.geometry_consistent),
        int(value.phase_consistent),
        terminal_quality,
        int(value.target_satisfied),
        int(value.continuation_feasible or value.exact_terminal),
        value.progress_profiles + value.progress_holes + value.progress_extrudes,
        -(
            value.overbuild_profiles
            + value.overbuild_holes
            + value.overbuild_extrudes
        ),
        value.lookahead_steps,
    )


def preferred_consequence_indices(values: Sequence[StructuredConsequence]) -> list[int]:
    if not values:
        return []
    keys = [consequence_preference_key(value) for value in values]
    best = max(keys)
    return [index for index, key in enumerate(keys) if key == best]


def _node_key(
    node: _Node,
    *,
    before: dict[str, int],
    target: dict[str, int],
) -> tuple[int, ...]:
    after = structure_stats(list(node.actions))
    progress = _goal_progress(before, after, target)
    excess = _overbuild(after, target)
    satisfied = _target_satisfied(after, target)
    terminal = _terminal_class(
        ended=bool(node.state.ended),
        satisfied=satisfied,
        overbuild=excess,
    )
    terminal_quality = {"premature": 0, "overbuild": 0, "none": 1, "exact": 2}[terminal]
    return (
        terminal_quality,
        int(satisfied),
        sum(progress),
        -sum(excess),
        node.depth,
    )


def evaluate_structured_consequence(
    *,
    prefix_actions: Sequence[Action],
    state: VerificationState,
    candidate_action: Action,
    target_stats: dict[str, Any],
    continuation_provider: ContinuationProvider,
    process_phase: str | None = None,
    horizon: int = 6,
    beam_width: int = 8,
    verifier: TopoVerifier | None = None,
) -> StructuredConsequence:
    """Replay one action and search a bounded set of verifier-valid continuations."""

    if horizon < 1:
        raise ValueError("horizon must be at least one")
    if beam_width < 1:
        raise ValueError("beam_width must be at least one")
    verifier = verifier or TopoVerifier()
    before = structure_stats(list(prefix_actions))
    target = _target_counts(target_stats)
    phase_consistent = phase_consistent_action(
        state=state,
        candidate_action=candidate_action,
        before=before,
        target=target,
        process_phase=process_phase,
    )
    first = verifier.step(state, candidate_action)
    if not first.valid or first.next_state is None:
        failure = str(first.failure_type or "invalid")
        return StructuredConsequence(
            immediate_valid=False,
            continuation_feasible=False,
            target_satisfied=False,
            exact_terminal=False,
            dead_end_within_k=True,
            terminal_class="none",
            progress_profiles=0,
            progress_holes=0,
            progress_extrudes=0,
            overbuild_profiles=0,
            overbuild_holes=0,
            overbuild_extrudes=0,
            reference_consistent=failure not in REFERENCE_FAILURE_TYPES,
            geometry_consistent=failure not in GEOMETRY_FAILURE_TYPES,
            phase_consistent=phase_consistent,
            lookahead_steps=0,
            horizon=horizon,
            failure_type=failure,
        )

    root = _Node(
        actions=tuple([*prefix_actions, candidate_action]),
        state=first.next_state,
        depth=1,
    )
    frontier = [root]
    visited = [root]
    had_valid_continuation = bool(root.state.ended) or horizon == 1

    for _ in range(1, horizon):
        children: list[_Node] = []
        for node in frontier:
            if node.state.ended:
                continue
            seen_lines: set[str] = set()
            for action in continuation_provider(node.actions, node.state):
                marker = repr(action)
                if marker in seen_lines:
                    continue
                seen_lines.add(marker)
                result = verifier.step(node.state, action)
                if not result.valid or result.next_state is None:
                    continue
                children.append(
                    _Node(
                        actions=tuple([*node.actions, action]),
                        state=result.next_state,
                        depth=node.depth + 1,
                    )
                )
        if not children:
            break
        had_valid_continuation = True
        children.sort(
            key=lambda node: _node_key(node, before=before, target=target),
            reverse=True,
        )
        frontier = children[:beam_width]
        visited.extend(frontier)

    best = max(visited, key=lambda node: _node_key(node, before=before, target=target))
    after = structure_stats(list(best.actions))
    progress = _goal_progress(before, after, target)
    excess = _overbuild(after, target)
    satisfied = _target_satisfied(after, target)
    terminal = _terminal_class(
        ended=bool(best.state.ended),
        satisfied=satisfied,
        overbuild=excess,
    )
    terminal = _phase_terminal_override(
        candidate_action=candidate_action,
        before=before,
        target=target,
        default=terminal,
    )
    reached_horizon = any(node.depth >= horizon for node in visited)
    exact_terminal = any(
        node.state.ended
        and _target_satisfied(structure_stats(list(node.actions)), target)
        and not any(_overbuild(structure_stats(list(node.actions)), target))
        for node in visited
    )
    return StructuredConsequence(
        immediate_valid=True,
        continuation_feasible=had_valid_continuation,
        target_satisfied=satisfied,
        exact_terminal=exact_terminal,
        dead_end_within_k=not reached_horizon and not exact_terminal,
        terminal_class=terminal,
        progress_profiles=progress[0],
        progress_holes=progress[1],
        progress_extrudes=progress[2],
        overbuild_profiles=excess[0],
        overbuild_holes=excess[1],
        overbuild_extrudes=excess[2],
        reference_consistent=True,
        geometry_consistent=True,
        phase_consistent=phase_consistent,
        lookahead_steps=max(node.depth for node in visited),
        horizon=horizon,
        failure_type="none",
    )


def is_terminal_action(action: Action) -> bool:
    return isinstance(action, End)


def is_extrude_action(action: Action) -> bool:
    return isinstance(action, Extrude)
