from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .actions import (
    Action,
    AddCircle,
    End,
    EndFace,
    EndSketch,
    EndLoop,
    Extrude,
    RegisterProfile,
    StartFace,
    StartLoop,
    StartSketch,
    action_to_text,
)
from .lm.parsing import parse_action_line
from .structure_stats import structure_stats
from .topoplan import coerce_target_stats, profile_plan_from_actions, progress_score
from .verifier import TopoVerifier, VerificationState
from .verifier.geometry import (
    close_points,
    circle_as_polygon,
    on_segment,
    point_in_polygon,
    polygon_segments,
    polygons_cross,
)


PHASES = [
    "START_SKETCH",
    "START_FACE",
    "BUILD_OUTER",
    "BUILD_INNER",
    "CLOSE_FACE",
    "REGISTER_PROFILE",
    "END_SKETCH",
    "EXTRUDE",
    "END",
    "DONE",
]

PROGRESS_ROLES = {"desired", "progress"}
NEGATIVE_ROLES = {"stagnant", "premature", "overbuild", "wrong_phase", "wrong_reference", "invalid"}

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class TopoPlanV2State:
    phase: str
    target_stats: dict[str, int]
    progress_stats: dict[str, int]
    remaining_profiles: int
    remaining_holes: int
    remaining_circle_holes: int
    remaining_extrusions: int
    should_continue: bool
    should_stop: bool
    should_close_phase: bool
    valid_progress_actions: tuple[str, ...]
    open_loop: str | None
    open_face: bool
    pending_face: bool
    registered_profiles: tuple[str, ...]
    loop_subphase: str
    current_loop_points: int
    current_loop_segments: int
    current_loop_closed: bool
    current_loop_start: str | None
    current_loop_tail: str | None
    hole_subphase: str
    must_action_types: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    next_oracle_action: str | None = None


def replay_prefix(prefix_actions: list[Action] | tuple[Action, ...]) -> VerificationState | None:
    verifier = TopoVerifier()
    state = verifier.initial_state()
    for action in prefix_actions:
        result = verifier.step(state, action)
        if not result.valid:
            return None
        assert result.next_state is not None
        state = result.next_state
    return state


def _point_text(point: Point | None) -> str | None:
    if point is None:
        return None
    return f"({point[0]:.4g},{point[1]:.4g})"


def _stack_top(state: VerificationState | None) -> str | None:
    if state is None or not state.stack:
        return None
    return state.stack[-1]


def _remaining(target: dict[str, int], progress: dict[str, int], key: str) -> int:
    return max(0, int(target.get(key, 0)) - int(progress.get(key, 0)))


def _target_satisfied(target: dict[str, int], progress: dict[str, int]) -> bool:
    return (
        progress["profile_count"] >= target["profile_count"]
        and progress["hole_count"] >= target["hole_count"]
        and progress["circle_hole_count"] >= target["circle_hole_count"]
        and progress["extrude_count"] >= target["extrude_count"]
    )


def infer_phase(
    prefix_actions: list[Action] | tuple[Action, ...],
    target_stats: dict[str, Any] | None,
    *,
    state: VerificationState | None = None,
    target_actions: list[Action] | tuple[Action, ...] | None = None,
) -> str:
    target = coerce_target_stats(target_stats)
    progress = structure_stats(prefix_actions)
    state = replay_prefix(list(prefix_actions)) if state is None else state
    next_action = _next_oracle_action(prefix_actions, target_actions)
    if state is None:
        return "START_SKETCH"
    if state.ended:
        return "DONE"

    top = _stack_top(state)
    if top is None:
        if isinstance(next_action, Extrude):
            return "EXTRUDE"
        if isinstance(next_action, End):
            return "END"
        if isinstance(next_action, StartSketch):
            return "START_SKETCH"
        if not prefix_actions:
            return "START_SKETCH"
        if progress["extrude_count"] < target["extrude_count"]:
            return "EXTRUDE"
        return "END"

    if top == "loop":
        if state.current_loop is not None and state.current_loop.kind == "inner":
            return "BUILD_INNER"
        return "BUILD_OUTER"

    if top == "face":
        if not state.current_face_loops:
            return "BUILD_OUTER"
        if isinstance(next_action, StartLoop) and next_action.kind == "inner":
            return "BUILD_INNER"
        if isinstance(next_action, EndFace):
            return "CLOSE_FACE"
        remaining_holes = _remaining(target, progress, "hole_count")
        layout = layout_feasibility_state(target_stats, prefix_actions, state=state)
        if remaining_holes > 0 and layout["safe_inner_progress"]:
            return "BUILD_INNER"
        return "CLOSE_FACE"

    if top == "sketch":
        if state.pending_face is not None:
            return "REGISTER_PROFILE"
        if isinstance(next_action, StartFace):
            return "START_FACE"
        if isinstance(next_action, EndSketch):
            return "END_SKETCH"
        if progress["profile_count"] < target["profile_count"]:
            return "START_FACE"
        return "END_SKETCH"

    return "START_SKETCH"


def expected_action_types_for_phase(phase: str) -> tuple[str, ...]:
    if phase == "START_SKETCH":
        return ("StartSketch",)
    if phase == "START_FACE":
        return ("StartFace",)
    if phase == "BUILD_OUTER":
        return ("StartLoop", "AddLine", "AddArc", "AddCircle", "EndLoop")
    if phase == "BUILD_INNER":
        return ("StartLoop", "AddLine", "AddArc", "AddCircle", "EndLoop")
    if phase == "CLOSE_FACE":
        return ("EndFace",)
    if phase == "REGISTER_PROFILE":
        return ("RegisterProfile",)
    if phase == "END_SKETCH":
        return ("EndSketch",)
    if phase == "EXTRUDE":
        return ("Extrude",)
    if phase == "END":
        return ("End",)
    return ()


def _action_type(action: Action) -> str:
    return action.type


def _loop_is_closed(state: VerificationState | None) -> bool:
    if state is None or state.current_loop is None:
        return False
    loop = state.current_loop
    if loop.circle_closed:
        return True
    return loop.start is not None and loop.tail is not None and close_points(loop.start, loop.tail)


def _action_signature(action: Action) -> str:
    if isinstance(action, StartLoop):
        return f"StartLoop.{action.kind}"
    return action.type


def _next_oracle_action(
    prefix_actions: list[Action] | tuple[Action, ...],
    target_actions: list[Action] | tuple[Action, ...] | None,
) -> Action | None:
    if target_actions is None or len(prefix_actions) >= len(target_actions):
        return None
    return target_actions[len(prefix_actions)]


def _loop_subphase(
    state: VerificationState | None,
    phase: str,
    next_action: Action | None,
) -> str:
    if state is None:
        return "NONE"
    top = _stack_top(state)
    if top == "face" and state.current_loop is None:
        if phase == "BUILD_INNER":
            return "START_INNER"
        if phase == "BUILD_OUTER":
            return "START_OUTER"
        if phase == "CLOSE_FACE":
            return "CLOSE_FACE"
        return "FACE_IDLE"
    if top != "loop" or state.current_loop is None:
        return "NONE"

    loop_kind = state.current_loop.kind
    if next_action is not None and isinstance(next_action, EndLoop):
        return "CLOSE_INNER" if loop_kind == "inner" else "CLOSE_OUTER"
    if loop_kind == "inner":
        if _loop_is_closed(state):
            return "CLOSE_INNER"
        return "DRAW_INNER"
    if _loop_is_closed(state):
        return "CLOSE_OUTER"
    return "DRAW_OUTER"


def _forbidden_actions(
    state: VerificationState | None,
    phase: str,
    remaining_profiles: int,
    remaining_holes: int,
    remaining_extrusions: int,
) -> tuple[str, ...]:
    if state is None:
        return tuple()
    top = _stack_top(state)
    forbidden: list[str] = []

    if top == "loop":
        forbidden.extend(
            [
                "StartLoop(kind=inner)",
                "StartLoop(kind=outer)",
                "StartFace",
                "EndFace",
                "RegisterProfile",
                "EndSketch",
                "Extrude",
                "End",
            ]
        )
    elif top == "face":
        if phase == "BUILD_INNER" and remaining_holes > 0:
            forbidden.extend(
                [
                    "StartLoop(kind=outer)",
                    "EndFace",
                    "RegisterProfile",
                    "EndSketch",
                    "Extrude",
                    "End",
                    "StartFace",
                ]
            )
        elif phase == "CLOSE_FACE":
            forbidden.extend(
                [
                    "StartLoop(kind=inner)",
                    "StartLoop(kind=outer)",
                    "StartFace",
                    "RegisterProfile",
                    "EndSketch",
                    "Extrude",
                    "End",
                ]
            )
    elif top == "sketch":
        if state.pending_face is not None:
            forbidden.extend(["StartFace", "StartLoop", "EndSketch", "Extrude", "End"])
        elif remaining_profiles > 0:
            forbidden.extend(["EndSketch", "Extrude", "End"])
        elif remaining_extrusions > 0:
            forbidden.extend(["End"])
    elif top is None:
        if remaining_extrusions > 0:
            forbidden.extend(["End"])

    return tuple(dict.fromkeys(forbidden))


def _canonical_lines(actions: Iterable[Action]) -> tuple[str, ...]:
    seen = set()
    lines = []
    for action in actions:
        line = action_to_text(action)
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return tuple(lines)


def _profiles_by_id(target_stats: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    profiles = (target_stats or {}).get("profiles")
    if not isinstance(profiles, list):
        return {}
    return {
        str(profile.get("profile_id")): profile
        for profile in profiles
        if isinstance(profile, dict) and profile.get("profile_id") is not None
    }


def _profiles_list(target_stats: dict[str, Any] | None) -> list[dict[str, Any]]:
    profiles = (target_stats or {}).get("profiles")
    if not isinstance(profiles, list):
        return []
    return [profile for profile in profiles if isinstance(profile, dict)]


def _extrudes_list(target_stats: dict[str, Any] | None) -> list[dict[str, Any]]:
    extrudes = (target_stats or {}).get("extrudes")
    if not isinstance(extrudes, list):
        return []
    return [extrude for extrude in extrudes if isinstance(extrude, dict)]


def _bbox(points: list[Point]) -> BBox | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0 = min(xs)
    y0 = min(ys)
    return (round(x0, 6), round(y0, 6), round(max(xs) - x0, 6), round(max(ys) - y0, 6))


def _bbox_from_meta(data: dict[str, Any] | None) -> BBox | None:
    if not isinstance(data, dict):
        return None
    try:
        return (
            round(float(data["x"]), 6),
            round(float(data["y"]), 6),
            round(float(data["w"]), 6),
            round(float(data["h"]), 6),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _bbox_close(a: BBox | None, b: BBox | None, *, tol: float = 1e-4) -> bool | None:
    if a is None or b is None:
        return None
    return max(abs(x - y) for x, y in zip(a, b)) <= tol


def _bbox_text(value: BBox | None) -> str:
    if value is None:
        return "UNKNOWN"
    return f"({value[0]:.6g},{value[1]:.6g},{value[2]:.6g},{value[3]:.6g})"


def _bool_text(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return str(value).lower()


def _num_text(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return "UNKNOWN"


def _outer_loop_from_face(face: Any) -> Any | None:
    for loop in getattr(face, "loops", []) or []:
        if getattr(loop, "kind", None) == "outer":
            return loop
    return None


def _current_outer_points(state: VerificationState | None) -> list[Point]:
    if state is None:
        return []
    if state.current_loop is not None and state.current_loop.kind == "outer":
        return list(state.current_loop.points)
    if state.current_face_loops:
        for loop in state.current_face_loops:
            if loop.kind == "outer":
                return list(loop.points)
    if state.pending_face is not None:
        outer = _outer_loop_from_face(state.pending_face)
        return list(getattr(outer, "points", []) or []) if outer is not None else []
    return []


def _strict_inside(point: Point, polygon: list[Point]) -> bool:
    if not point_in_polygon(point, polygon):
        return False
    return not any(on_segment(start, point, end) for start, end in polygon_segments(polygon))


def _rect_points(x: float, y: float, w: float, h: float) -> list[Point]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _hole_polygon(hole: dict[str, Any]) -> list[Point] | None:
    try:
        x = float(hole["x"])
        y = float(hole["y"])
        w = float(hole["w"])
        h = float(hole["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if hole.get("kind") == "circle":
        radius = round(min(w, h) * 0.45, 4)
        center = (round(x + w * 0.5, 4), round(y + h * 0.5, 4))
        return circle_as_polygon(center, radius)
    return _rect_points(x, y, w, h)


def _hole_summary_text(holes: Any, *, limit: int = 4) -> str:
    if not isinstance(holes, list) or not holes:
        return "NONE"
    parts: list[str] = []
    for hole in holes[:limit]:
        if not isinstance(hole, dict):
            continue
        kind = str(hole.get("kind") or hole.get("source_curve_kind") or "hole")
        parts.append(f"{kind}:{_bbox_text(_bbox_from_meta(hole))}")
    if len(holes) > limit:
        parts.append(f"+{len(holes) - limit}")
    return ";".join(parts) if parts else "NONE"


def _target_holes_fit_outer(profile_meta: dict[str, Any] | None, outer_points: list[Point]) -> bool | None:
    if not isinstance(profile_meta, dict) or not outer_points:
        return None
    holes = profile_meta.get("holes")
    if not isinstance(holes, list):
        return None
    for hole in holes:
        if not isinstance(hole, dict):
            continue
        polygon = _hole_polygon(hole)
        if polygon is None:
            continue
        if polygons_cross(outer_points, polygon):
            return False
        if not all(_strict_inside(point, outer_points) for point in polygon):
            return False
    return True


def layout_feasibility_state(
    target_stats: dict[str, Any] | None,
    prefix_actions: list[Action] | tuple[Action, ...],
    *,
    state: VerificationState | None = None,
) -> dict[str, Any]:
    """Return layout-conditioned process state for safe hole progress.

    This deliberately uses the original target_stats object, not the coerced
    count-only target, because profile/hole feasibility depends on per-profile
    layout metadata.
    """

    target = coerce_target_stats(target_stats)
    progress = structure_stats(prefix_actions)
    profiles = _profiles_by_id(target_stats)
    state = replay_prefix(list(prefix_actions)) if state is None else state
    profile_index = min(
        int(progress["profile_count"]),
        max(0, int(target["profile_count"]) - 1),
    )
    next_profile_id = f"profile_{profile_index}" if progress["profile_count"] < target["profile_count"] else "NONE"
    next_profile = profiles.get(next_profile_id)
    profile0 = profiles.get("profile_0")
    expected_bbox = _bbox_from_meta((next_profile or {}).get("outer"))
    profile0_bbox = _bbox_from_meta((profile0 or {}).get("outer"))
    outer_points = _current_outer_points(state)
    current_bbox = _bbox(outer_points)
    layout_match = _bbox_close(current_bbox, expected_bbox)
    repeated_profile0 = (
        _bbox_close(current_bbox, profile0_bbox) if next_profile_id not in {"NONE", "profile_0"} else False
    )
    holes = (next_profile or {}).get("holes")
    target_holes = len(holes) if isinstance(holes, list) else 0
    hole_fit = _target_holes_fit_outer(next_profile, outer_points)
    top = _stack_top(state)
    safe_inner_progress = (
        top == "face"
        and state is not None
        and state.current_loop is None
        and bool(state.current_face_loops)
        and target["hole_count"] > progress["hole_count"]
        and target_holes > 0
        and hole_fit is True
    )
    unsafe_inner_progress = (
        top == "face"
        and state is not None
        and state.current_loop is None
        and bool(state.current_face_loops)
        and target["hole_count"] > progress["hole_count"]
        and not safe_inner_progress
    )
    return {
        "next_profile_id": next_profile_id,
        "expected_outer_bbox": expected_bbox,
        "current_outer_bbox": current_bbox,
        "layout_match": layout_match,
        "repeated_profile0_layout": repeated_profile0,
        "target_holes_for_profile": target_holes,
        "target_holes_fit_current_outer": hole_fit,
        "safe_inner_progress": safe_inner_progress,
        "unsafe_inner_progress": unsafe_inner_progress,
    }


def _next_profile_plan_lines(
    target_stats: dict[str, Any] | None,
    plan: TopoPlanV2State,
) -> list[str]:
    profiles = _profiles_list(target_stats)
    if not profiles:
        return []
    progress_index = int(plan.progress_stats["profile_count"])
    if progress_index >= len(profiles):
        return [
            "[NEXT_PROFILE_PLAN]",
            "next_profile_id=NONE expected_outer_bbox=NONE expected_holes=0 expected_hole_bboxes=NONE",
        ]
    profile = profiles[progress_index]
    profile_id = str(profile.get("profile_id", f"profile_{progress_index}"))
    holes = profile.get("holes")
    hole_count = len(holes) if isinstance(holes, list) else 0
    return [
        "[NEXT_PROFILE_PLAN]",
        (
            f"next_profile_id={profile_id} "
            f"expected_outer_bbox={_bbox_text(_bbox_from_meta(profile.get('outer')))} "
            f"expected_holes={hole_count} "
            f"expected_hole_bboxes={_hole_summary_text(holes)}"
        ),
    ]


def _extrude_plan_lines(
    target_stats: dict[str, Any] | None,
    plan: TopoPlanV2State,
    *,
    limit: int = 6,
) -> list[str]:
    extrudes = _extrudes_list(target_stats)
    if not extrudes:
        return []
    index = int(plan.progress_stats["extrude_count"])
    if index >= len(extrudes):
        return [
            "[EXTRUDE_PLAN]",
            "next_extrude=NONE remaining_extrude_profile_ids=NONE",
        ]
    next_extrude = extrudes[index]
    remaining_ids = [
        str(item.get("profile_id", "UNKNOWN"))
        for item in extrudes[index : index + limit]
        if isinstance(item, dict)
    ]
    if len(extrudes) > index + limit:
        remaining_ids.append(f"+{len(extrudes) - index - limit}")
    return [
        "[EXTRUDE_PLAN]",
        (
            "next_extrude="
            f"profile_id={next_extrude.get('profile_id', 'UNKNOWN')},"
            f"depth={_num_text(next_extrude.get('depth'))},"
            f"op={next_extrude.get('op', 'UNKNOWN')} "
            f"remaining_extrude_profile_ids={','.join(remaining_ids) if remaining_ids else 'NONE'}"
        ),
    ]


def _plan_context_lines(
    target_stats: dict[str, Any] | None,
    plan: TopoPlanV2State,
) -> list[str]:
    lines: list[str] = []
    lines.extend(_next_profile_plan_lines(target_stats, plan))
    lines.extend(_extrude_plan_lines(target_stats, plan))
    return lines


def _layout_state_lines(
    target_stats: dict[str, Any] | None,
    prefix_actions: list[Action] | tuple[Action, ...],
    plan: TopoPlanV2State,
) -> list[str]:
    if not _profiles_by_id(target_stats):
        return []
    layout = layout_feasibility_state(target_stats, prefix_actions)
    return [
        "[LAYOUT_STATE]",
        (
            f"next_profile_id={layout['next_profile_id']} "
            f"expected_outer_bbox={_bbox_text(layout['expected_outer_bbox'])} "
            f"current_outer_bbox={_bbox_text(layout['current_outer_bbox'])} "
            f"layout_match={_bool_text(layout['layout_match'])} "
            f"repeated_profile0_layout={_bool_text(layout['repeated_profile0_layout'])} "
            f"target_holes_for_profile={layout['target_holes_for_profile']} "
            f"target_holes_fit_current_outer={_bool_text(layout['target_holes_fit_current_outer'])} "
            f"safe_inner_progress={_bool_text(layout['safe_inner_progress'])}"
        ),
    ]


def suggested_progress_actions(
    prefix_actions: list[Action] | tuple[Action, ...],
    target_stats: dict[str, Any] | None,
    *,
    state: VerificationState | None = None,
    target_actions: list[Action] | tuple[Action, ...] | None = None,
) -> tuple[str, ...]:
    target = coerce_target_stats(target_stats)
    progress = structure_stats(prefix_actions)
    state = replay_prefix(list(prefix_actions)) if state is None else state
    phase = infer_phase(prefix_actions, target, state=state, target_actions=target_actions)
    suggestions: list[Action] = []

    if target_actions is not None and len(prefix_actions) < len(target_actions):
        suggestions.append(target_actions[len(prefix_actions)])

    if phase == "START_SKETCH":
        suggestions.append(StartSketch())
    elif phase == "START_FACE":
        suggestions.append(StartFace())
    elif phase == "BUILD_OUTER":
        if state is not None and _stack_top(state) == "face" and state.current_loop is None:
            suggestions.append(StartLoop("outer"))
        elif _loop_is_closed(state):
            suggestions.append(EndLoop())
    elif phase == "BUILD_INNER":
        if state is not None and _stack_top(state) == "face" and state.current_loop is None:
            suggestions.append(StartLoop("inner"))
        elif _loop_is_closed(state):
            suggestions.append(EndLoop())
    elif phase == "CLOSE_FACE":
        suggestions.append(EndFace())
    elif phase == "REGISTER_PROFILE":
        suggestions.append(RegisterProfile(f"profile_{progress['profile_count']}"))
    elif phase == "END_SKETCH":
        suggestions.append(EndSketch())
    elif phase == "EXTRUDE":
        profile_id = f"profile_{min(progress['extrude_count'], max(progress['profile_count'] - 1, 0))}"
        suggestions.append(Extrude(profile_id=profile_id, depth=0.5, op="add"))
    elif phase == "END":
        suggestions.append(End())

    if (
        progress["hole_count"] < target["hole_count"]
        and phase == "BUILD_INNER"
        and state is not None
        and _stack_top(state) == "face"
        and state.current_loop is None
    ):
        suggestions.append(StartLoop("inner"))
    if progress["profile_count"] < target["profile_count"] and phase in {"START_FACE", "END_SKETCH"}:
        suggestions.append(StartFace())
    return _canonical_lines(suggestions)


def topoplan_v2_state(
    target_stats: dict[str, Any] | None,
    prefix_actions: list[Action] | tuple[Action, ...],
    *,
    target_actions: list[Action] | tuple[Action, ...] | None = None,
    include_oracle: bool = True,
) -> TopoPlanV2State:
    target = coerce_target_stats(target_stats)
    progress = structure_stats(prefix_actions)
    state = replay_prefix(list(prefix_actions))
    oracle_actions = target_actions if include_oracle else None
    phase = infer_phase(prefix_actions, target, state=state, target_actions=oracle_actions)
    remaining_profiles = _remaining(target, progress, "profile_count")
    remaining_holes = _remaining(target, progress, "hole_count")
    remaining_circle_holes = _remaining(target, progress, "circle_hole_count")
    remaining_extrusions = _remaining(target, progress, "extrude_count")
    should_stop = phase == "DONE" or (phase == "END" and _target_satisfied(target, progress))
    should_close_phase = phase in {"CLOSE_FACE", "REGISTER_PROFILE", "END_SKETCH", "EXTRUDE", "END"}
    should_continue = not should_stop and phase != "DONE"
    next_oracle_action = _next_oracle_action(prefix_actions, oracle_actions)
    next_oracle = action_to_text(next_oracle_action) if next_oracle_action is not None else None
    loop_subphase = _loop_subphase(state, phase, next_oracle_action)
    hole_subphase = loop_subphase if loop_subphase in {"START_INNER", "DRAW_INNER", "CLOSE_INNER"} else "NONE"
    must_action_types = (
        (_action_signature(next_oracle_action),)
        if next_oracle_action is not None
        else expected_action_types_for_phase(phase)
    )
    current_loop = state.current_loop if state is not None else None
    return TopoPlanV2State(
        phase=phase,
        target_stats=target,
        progress_stats=progress,
        remaining_profiles=remaining_profiles,
        remaining_holes=remaining_holes,
        remaining_circle_holes=remaining_circle_holes,
        remaining_extrusions=remaining_extrusions,
        should_continue=should_continue,
        should_stop=should_stop,
        should_close_phase=should_close_phase,
        valid_progress_actions=suggested_progress_actions(
            prefix_actions,
            target,
            state=state,
            target_actions=oracle_actions,
        ),
        open_loop=state.current_loop.kind if state is not None and state.current_loop is not None else None,
        open_face=state is not None and _stack_top(state) == "face",
        pending_face=state is not None and state.pending_face is not None,
        registered_profiles=tuple(sorted(state.profiles)) if state is not None else tuple(),
        loop_subphase=loop_subphase,
        current_loop_points=len(current_loop.points) if current_loop is not None else 0,
        current_loop_segments=len(current_loop.segments) if current_loop is not None else 0,
        current_loop_closed=_loop_is_closed(state),
        current_loop_start=_point_text(current_loop.start) if current_loop is not None else None,
        current_loop_tail=_point_text(current_loop.tail) if current_loop is not None else None,
        hole_subphase=hole_subphase,
        must_action_types=must_action_types,
        forbidden_actions=_forbidden_actions(
            state,
            phase,
            remaining_profiles,
            remaining_holes,
            remaining_extrusions,
        ),
        next_oracle_action=next_oracle,
    )


def topoplan_v2_block(
    target_stats: dict[str, Any] | None,
    prefix_actions: list[Action] | tuple[Action, ...],
    *,
    target_actions: list[Action] | tuple[Action, ...] | None = None,
    include_oracle: bool = True,
) -> str:
    plan = topoplan_v2_state(
        target_stats,
        prefix_actions,
        target_actions=target_actions,
        include_oracle=include_oracle,
    )
    profile_plans = profile_plan_from_actions(target_actions or [])
    valid_progress = ", ".join(plan.valid_progress_actions) if plan.valid_progress_actions else "NONE"
    profiles = ", ".join(plan.registered_profiles) if plan.registered_profiles else "[]"
    must_actions = ", ".join(plan.must_action_types) if plan.must_action_types else "NONE"
    forbidden = ", ".join(plan.forbidden_actions) if plan.forbidden_actions else "NONE"
    lines = [
        "<topoplan_v2>",
        "[TARGET]",
        (
            f"profiles={plan.target_stats['profile_count']} "
            f"holes={plan.target_stats['hole_count']} "
            f"circle_holes={plan.target_stats['circle_hole_count']} "
            f"extrusions={plan.target_stats['extrude_count']} "
            f"actions={plan.target_stats['action_count']}"
        ),
        "[CURRENT]",
        (
            f"profiles_done={plan.progress_stats['profile_count']} "
            f"holes_done={plan.progress_stats['hole_count']} "
            f"circle_holes_done={plan.progress_stats['circle_hole_count']} "
            f"extrusions_done={plan.progress_stats['extrude_count']} "
            f"actions_done={plan.progress_stats['action_count']} "
            f"open_loop={plan.open_loop or 'NONE'} "
            f"open_face={int(plan.open_face)} "
            f"pending_face={int(plan.pending_face)} "
            f"registered_profiles={profiles}"
        ),
        "[LOOP_STATE]",
        (
            f"loop_subphase={plan.loop_subphase} "
            f"hole_subphase={plan.hole_subphase} "
            f"loop_points={plan.current_loop_points} "
            f"loop_segments={plan.current_loop_segments} "
            f"loop_closed={int(plan.current_loop_closed)} "
            f"loop_start={plan.current_loop_start or 'NONE'} "
            f"loop_tail={plan.current_loop_tail or 'NONE'}"
        ),
        "[REMAINING_TASK]",
        (
            f"profiles_needed={plan.remaining_profiles} "
            f"holes_needed={plan.remaining_holes} "
            f"circle_holes_needed={plan.remaining_circle_holes} "
            f"extrusions_needed={plan.remaining_extrusions}"
        ),
        "[PHASE]",
        plan.phase,
        "[CONTROL]",
        (
            f"should_continue={str(plan.should_continue).lower()} "
            f"should_stop={str(plan.should_stop).lower()} "
            f"should_close_phase={str(plan.should_close_phase).lower()}"
        ),
        "[MUST_ACTION_TYPES]",
        must_actions,
        "[FORBIDDEN_ACTIONS]",
        forbidden,
        "[VALID_PROGRESS_ACTIONS]",
        valid_progress,
    ]
    if profile_plans:
        lines.append("[PROFILE_PLAN]")
        for profile in profile_plans:
            lines.append(
                f"{profile.profile_id}: outer_loops=1 inner_loops={profile.inner_loops} "
                f"circle_holes={profile.circle_holes}"
            )
    lines.extend(_plan_context_lines(target_stats, plan))
    lines.extend(_layout_state_lines(target_stats, prefix_actions, plan))
    if plan.next_oracle_action is not None:
        lines.extend(["[ORACLE_NEXT_ACTION]", plan.next_oracle_action])
    lines.append("</topoplan_v2>")
    return "\n".join(lines) + "\n"


def topoplan_v2_decision_tail(
    target_stats: dict[str, Any] | None,
    prefix_actions: list[Action] | tuple[Action, ...],
    *,
    target_actions: list[Action] | tuple[Action, ...] | None = None,
    include_oracle: bool = True,
) -> str:
    """Compact process state placed immediately before the next-action slot.

    Long CAD prefixes can push the main TopoPlan block out of the model context
    window when prompts are truncated from the left. This tail keeps the
    decision-critical state near the logits being scored.
    """

    plan = topoplan_v2_state(
        target_stats,
        prefix_actions,
        target_actions=target_actions,
        include_oracle=include_oracle,
    )
    valid_progress = ", ".join(plan.valid_progress_actions) if plan.valid_progress_actions else "NONE"
    must_actions = ", ".join(plan.must_action_types) if plan.must_action_types else "NONE"
    forbidden = ", ".join(plan.forbidden_actions) if plan.forbidden_actions else "NONE"
    profiles = ", ".join(plan.registered_profiles) if plan.registered_profiles else "[]"
    lines = [
        "<decision_state>",
        "[CURRENT]",
        (
            f"profiles_done={plan.progress_stats['profile_count']} "
            f"holes_done={plan.progress_stats['hole_count']} "
            f"circle_holes_done={plan.progress_stats['circle_hole_count']} "
            f"extrusions_done={plan.progress_stats['extrude_count']} "
            f"actions_done={plan.progress_stats['action_count']} "
            f"open_loop={plan.open_loop or 'NONE'} "
            f"open_face={int(plan.open_face)} "
            f"pending_face={int(plan.pending_face)} "
            f"registered_profiles={profiles}"
        ),
        "[LOOP_STATE]",
        (
            f"loop_subphase={plan.loop_subphase} "
            f"hole_subphase={plan.hole_subphase} "
            f"loop_points={plan.current_loop_points} "
            f"loop_segments={plan.current_loop_segments} "
            f"loop_closed={int(plan.current_loop_closed)} "
            f"loop_start={plan.current_loop_start or 'NONE'} "
            f"loop_tail={plan.current_loop_tail or 'NONE'}"
        ),
        "[REMAINING_TASK]",
        (
            f"profiles_needed={plan.remaining_profiles} "
            f"holes_needed={plan.remaining_holes} "
            f"circle_holes_needed={plan.remaining_circle_holes} "
            f"extrusions_needed={plan.remaining_extrusions}"
        ),
        "[PHASE]",
        plan.phase,
        "[CONTROL]",
        (
            f"should_continue={str(plan.should_continue).lower()} "
            f"should_stop={str(plan.should_stop).lower()} "
            f"should_close_phase={str(plan.should_close_phase).lower()}"
        ),
        "[MUST_ACTION_TYPES]",
        must_actions,
        "[FORBIDDEN_ACTIONS]",
        forbidden,
        "[VALID_PROGRESS_ACTIONS]",
        valid_progress,
    ]
    lines.extend(_plan_context_lines(target_stats, plan))
    lines.extend(_layout_state_lines(target_stats, prefix_actions, plan))
    if plan.next_oracle_action is not None:
        lines.extend(["[ORACLE_NEXT_ACTION]", plan.next_oracle_action])
    lines.append("</decision_state>")
    return "\n".join(lines) + "\n"


def role_for_action(
    prefix_actions: list[Action] | tuple[Action, ...],
    candidate_action: Action,
    target_stats: dict[str, Any] | None,
    *,
    target_actions: list[Action] | tuple[Action, ...] | None = None,
    state: VerificationState | None = None,
) -> dict[str, Any]:
    target = coerce_target_stats(target_stats)
    prefix = list(prefix_actions)
    state = replay_prefix(prefix) if state is None else state
    verifier = TopoVerifier()
    if state is None:
        return {
            "validity": "invalid",
            "role": "invalid",
            "phase": "START_SKETCH",
            "progress_delta": 0.0,
            "failure_type": "invalid_prefix",
            "repair_hint": "Repair the prefix before choosing the next CAD action.",
        }
    result = verifier.step(state, candidate_action)
    phase = infer_phase(prefix, target, state=state, target_actions=target_actions)
    if not result.valid:
        failure_type = result.failure_type
        role = "wrong_reference" if failure_type == "invalid_profile_reference" else "invalid"
        return {
            "validity": "invalid",
            "role": role,
            "phase": phase,
            "progress_delta": 0.0,
            "failure_type": failure_type,
            "repair_hint": result.repair_hint,
        }

    before = structure_stats(prefix)
    after = structure_stats(prefix + [candidate_action])
    line = action_to_text(candidate_action)
    oracle_line = None
    if target_actions is not None and len(prefix) < len(target_actions):
        oracle_line = action_to_text(target_actions[len(prefix)])
    progress_delta = progress_score(prefix, candidate_action, target)

    expected_types = expected_action_types_for_phase(phase)
    if oracle_line is not None and line == oracle_line:
        role = "desired"
    elif isinstance(candidate_action, (EndFace, EndSketch, End, Extrude)) and (
        after["profile_count"] < target["profile_count"]
        or after["hole_count"] < target["hole_count"]
        or after["extrude_count"] < target["extrude_count"]
    ):
        role = "premature"
    elif any(after[key] > target[key] for key in ["profile_count", "hole_count", "circle_hole_count", "extrude_count"]):
        role = "overbuild"
    elif expected_types and _action_type(candidate_action) not in expected_types:
        role = "wrong_phase"
    elif progress_delta > 0.0:
        role = "progress"
    else:
        role = "stagnant"

    return {
        "validity": "valid",
        "role": role,
        "phase": phase,
        "progress_delta": float(progress_delta),
        "failure_type": None,
        "repair_hint": None,
        "before_stats": before,
        "after_stats": after,
    }


def parse_action_lines(lines: Iterable[str]) -> list[Action]:
    actions = []
    for line in lines:
        action = parse_action_line(line)
        if action is not None:
            actions.append(action)
    return actions
