from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ..actions import (
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
from .failures import FailureType, REPAIR_HINTS
from .geometry import (
    EPS,
    Point,
    Segment,
    arc_as_polyline,
    circle_as_polygon,
    close_points,
    collinear,
    distance,
    on_segment,
    point_in_polygon,
    polygon_segments,
    polygons_cross,
    signed_area,
    would_self_intersect,
)


@dataclass
class LoopState:
    kind: str
    start: Point | None = None
    tail: Point | None = None
    points: list[Point] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    circle_closed: bool = False


@dataclass
class CompletedLoop:
    kind: str
    points: list[Point]
    area: float


@dataclass
class CompletedFace:
    loops: list[CompletedLoop]


@dataclass
class VerificationState:
    stack: list[str] = field(default_factory=list)
    current_loop: LoopState | None = None
    current_face_loops: list[CompletedLoop] = field(default_factory=list)
    pending_face: CompletedFace | None = None
    profiles: dict[str, CompletedFace] = field(default_factory=dict)
    sketch_start_profile_count: int | None = None
    extrusions: list[dict[str, Any]] = field(default_factory=list)
    ended: bool = False

    def clone(self) -> "VerificationState":
        return deepcopy(self)


@dataclass(frozen=True)
class StepResult:
    valid: bool
    next_state: VerificationState | None
    failure_type: FailureType | None = None
    repair_hint: str | None = None


class TopoVerifier:
    def initial_state(self) -> VerificationState:
        return VerificationState()

    def verify_program(self, actions: list[Action] | tuple[Action, ...]) -> StepResult:
        state = self.initial_state()
        last_result: StepResult = StepResult(valid=True, next_state=state)
        for action in actions:
            last_result = self.step(state, action)
            if not last_result.valid:
                return last_result
            assert last_result.next_state is not None
            state = last_result.next_state
        return last_result

    def step(self, state: VerificationState, action: Action) -> StepResult:
        if state.ended:
            return self._fail("hierarchy_error")

        next_state = state.clone()
        try:
            if isinstance(action, StartSketch):
                return self._start_sketch(next_state)
            if isinstance(action, StartFace):
                return self._start_face(next_state)
            if isinstance(action, StartLoop):
                return self._start_loop(next_state, action)
            if isinstance(action, AddLine):
                return self._add_line(next_state, action.start, action.end)
            if isinstance(action, AddArc):
                return self._add_arc(next_state, action)
            if isinstance(action, AddCircle):
                return self._add_circle(next_state, action)
            if isinstance(action, EndLoop):
                return self._end_loop(next_state)
            if isinstance(action, EndFace):
                return self._end_face(next_state)
            if isinstance(action, RegisterProfile):
                return self._register_profile(next_state, action)
            if isinstance(action, EndSketch):
                return self._end_sketch(next_state)
            if isinstance(action, Extrude):
                return self._extrude(next_state, action)
            if isinstance(action, End):
                return self._end(next_state)
        except Exception:
            return self._fail("kernel_execution_failure")
        return self._fail("hierarchy_error")

    def _ok(self, state: VerificationState) -> StepResult:
        return StepResult(valid=True, next_state=state)

    def _fail(self, failure_type: FailureType) -> StepResult:
        return StepResult(
            valid=False,
            next_state=None,
            failure_type=failure_type,
            repair_hint=REPAIR_HINTS[failure_type],
        )

    def _stack_top(self, state: VerificationState) -> str | None:
        return state.stack[-1] if state.stack else None

    def _strictly_inside_polygon(self, point: Point, polygon: list[Point]) -> bool:
        if not point_in_polygon(point, polygon):
            return False
        return not any(on_segment(start, point, end) for start, end in polygon_segments(polygon))

    def _start_sketch(self, state: VerificationState) -> StepResult:
        if state.stack or state.pending_face is not None or state.ended:
            return self._fail("hierarchy_error")
        state.stack.append("sketch")
        state.sketch_start_profile_count = len(state.profiles)
        return self._ok(state)

    def _start_face(self, state: VerificationState) -> StepResult:
        if self._stack_top(state) != "sketch" or state.pending_face is not None:
            return self._fail("hierarchy_error")
        state.stack.append("face")
        state.current_face_loops = []
        return self._ok(state)

    def _start_loop(self, state: VerificationState, action: StartLoop) -> StepResult:
        if self._stack_top(state) != "face" or state.current_loop is not None:
            return self._fail("hierarchy_error")
        if not state.current_face_loops and action.kind != "outer":
            return self._fail("hierarchy_error")
        if state.current_face_loops and action.kind != "inner":
            return self._fail("hierarchy_error")
        state.stack.append("loop")
        state.current_loop = LoopState(kind=action.kind)
        return self._ok(state)

    def _add_line(self, state: VerificationState, start: Point, end: Point) -> StepResult:
        loop = state.current_loop
        if self._stack_top(state) != "loop" or loop is None or loop.circle_closed:
            return self._fail("hierarchy_error")
        if distance(start, end) <= EPS:
            return self._fail("degenerate_curve")
        if loop.tail is not None and not close_points(start, loop.tail):
            return self._fail("endpoint_discontinuity")

        if loop.kind == "inner":
            outer_loops = [completed for completed in state.current_face_loops if completed.kind == "outer"]
            if len(outer_loops) != 1:
                return self._fail("hierarchy_error")
            outer = outer_loops[0]
            if not self._strictly_inside_polygon(start, outer.points) or not self._strictly_inside_polygon(end, outer.points):
                return self._fail("invalid_hole_containment")

        closing = loop.start is not None and close_points(end, loop.start)
        new_segment = (start, end)
        if would_self_intersect(loop.segments, new_segment, closing):
            return self._fail("self_intersection")

        if loop.start is None:
            loop.start = start
            loop.points.append(start)
        loop.tail = end
        loop.points.append(end)
        loop.segments.append(new_segment)
        return self._ok(state)

    def _add_arc(self, state: VerificationState, action: AddArc) -> StepResult:
        if distance(action.start, action.mid) <= EPS or distance(action.mid, action.end) <= EPS:
            return self._fail("degenerate_curve")
        if distance(action.start, action.end) <= EPS or collinear(action.start, action.mid, action.end):
            return self._fail("degenerate_curve")
        points = arc_as_polyline(action.start, action.mid, action.end)
        for start, end in zip(points, points[1:]):
            result = self._add_line(state, start, end)
            if not result.valid:
                return result
        return self._ok(state)

    def _add_circle(self, state: VerificationState, action: AddCircle) -> StepResult:
        loop = state.current_loop
        if self._stack_top(state) != "loop" or loop is None:
            return self._fail("hierarchy_error")
        if loop.points or loop.segments:
            return self._fail("hierarchy_error")
        if action.radius <= EPS:
            return self._fail("degenerate_curve")
        circle_points = circle_as_polygon(action.center, action.radius)
        if loop.kind == "inner":
            outer_loops = [completed for completed in state.current_face_loops if completed.kind == "outer"]
            if len(outer_loops) != 1:
                return self._fail("hierarchy_error")
            outer = outer_loops[0]
            if not all(self._strictly_inside_polygon(point, outer.points) for point in circle_points):
                return self._fail("invalid_hole_containment")
            for completed in state.current_face_loops:
                if completed.kind != "inner":
                    continue
                if polygons_cross(completed.points, circle_points):
                    return self._fail("invalid_hole_containment")
                if any(point_in_polygon(point, completed.points) for point in circle_points):
                    return self._fail("invalid_hole_containment")
                if any(point_in_polygon(point, circle_points) for point in completed.points):
                    return self._fail("invalid_hole_containment")
        loop.points = circle_points
        loop.circle_closed = True
        loop.start = loop.points[0]
        loop.tail = loop.points[0]
        return self._ok(state)

    def _end_loop(self, state: VerificationState) -> StepResult:
        loop = state.current_loop
        if self._stack_top(state) != "loop" or loop is None:
            return self._fail("hierarchy_error")
        if not loop.circle_closed:
            if loop.start is None or loop.tail is None or not close_points(loop.tail, loop.start):
                return self._fail("unclosed_loop")
            if len(loop.points) < 4:
                return self._fail("degenerate_curve")
            polygon = loop.points[:-1]
        else:
            polygon = loop.points
        area = signed_area(polygon)
        if abs(area) <= EPS:
            return self._fail("degenerate_curve")

        if loop.kind == "inner":
            outer_loops = [completed for completed in state.current_face_loops if completed.kind == "outer"]
            if len(outer_loops) != 1:
                return self._fail("hierarchy_error")
            outer = outer_loops[0]
            if polygons_cross(outer.points, polygon):
                return self._fail("invalid_hole_containment")
            if not all(point_in_polygon(point, outer.points) for point in polygon):
                return self._fail("invalid_hole_containment")
            for completed in state.current_face_loops:
                if completed.kind != "inner":
                    continue
                if polygons_cross(completed.points, polygon):
                    return self._fail("invalid_hole_containment")
                if any(point_in_polygon(point, completed.points) for point in polygon):
                    return self._fail("invalid_hole_containment")
                if any(point_in_polygon(point, polygon) for point in completed.points):
                    return self._fail("invalid_hole_containment")

        state.stack.pop()
        state.current_loop = None
        state.current_face_loops.append(CompletedLoop(kind=loop.kind, points=list(polygon), area=area))
        return self._ok(state)

    def _end_face(self, state: VerificationState) -> StepResult:
        if self._stack_top(state) != "face" or state.current_loop is not None:
            return self._fail("hierarchy_error")
        if not state.current_face_loops:
            return self._fail("hierarchy_error")

        outer_loops = [loop for loop in state.current_face_loops if loop.kind == "outer"]
        inner_loops = [loop for loop in state.current_face_loops if loop.kind == "inner"]
        if len(outer_loops) != 1:
            return self._fail("hierarchy_error")
        outer = outer_loops[0]

        for inner in inner_loops:
            if polygons_cross(outer.points, inner.points):
                return self._fail("invalid_hole_containment")
            if not all(point_in_polygon(point, outer.points) for point in inner.points):
                return self._fail("invalid_hole_containment")

        for i, first in enumerate(inner_loops):
            for second in inner_loops[i + 1 :]:
                if polygons_cross(first.points, second.points):
                    return self._fail("invalid_hole_containment")
                if any(point_in_polygon(point, first.points) for point in second.points):
                    return self._fail("invalid_hole_containment")
                if any(point_in_polygon(point, second.points) for point in first.points):
                    return self._fail("invalid_hole_containment")

        state.stack.pop()
        state.pending_face = CompletedFace(loops=list(state.current_face_loops))
        state.current_face_loops = []
        return self._ok(state)

    def _register_profile(self, state: VerificationState, action: RegisterProfile) -> StepResult:
        if self._stack_top(state) != "sketch" or state.pending_face is None:
            return self._fail("hierarchy_error")
        if action.profile_id in state.profiles:
            return self._fail("invalid_profile_reference")
        state.profiles[action.profile_id] = state.pending_face
        state.pending_face = None
        return self._ok(state)

    def _end_sketch(self, state: VerificationState) -> StepResult:
        if self._stack_top(state) != "sketch" or state.pending_face is not None:
            return self._fail("hierarchy_error")
        if len(state.profiles) <= (state.sketch_start_profile_count or 0):
            return self._fail("hierarchy_error")
        state.stack.pop()
        state.sketch_start_profile_count = None
        return self._ok(state)

    def _extrude(self, state: VerificationState, action: Extrude) -> StepResult:
        if state.stack not in ([], ["sketch"]) or state.pending_face is not None:
            return self._fail("hierarchy_error")
        if action.profile_id not in state.profiles:
            return self._fail("invalid_profile_reference")
        if action.depth <= EPS or action.op not in {"add", "cut", "intersect"}:
            return self._fail("degenerate_curve")
        state.extrusions.append({"profile_id": action.profile_id, "depth": action.depth, "op": action.op})
        return self._ok(state)

    def _end(self, state: VerificationState) -> StepResult:
        if state.stack or not state.extrusions:
            return self._fail("hierarchy_error")
        state.ended = True
        return self._ok(state)
