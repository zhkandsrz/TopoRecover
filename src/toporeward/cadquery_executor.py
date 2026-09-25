from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .actions import (
    Action,
    AddArc,
    AddCircle,
    AddLine,
    EndFace,
    EndLoop,
    Extrude,
    RegisterProfile,
    StartFace,
    StartLoop,
)


class CadQueryExecutionError(RuntimeError):
    pass


def _cadquery() -> Any:
    try:
        import cadquery as cq  # type: ignore
    except ImportError as exc:
        raise CadQueryExecutionError("CadQuery is not installed") from exc
    return cq


@dataclass(frozen=True)
class CadQueryExecutionResult:
    success: bool
    failure_action_index: int | None
    failure_type: str
    failure_message: str
    registered_profiles: int
    extrusions: int
    volume: float
    solids: int
    faces: int
    shape_valid: bool
    brep_evidence: dict[str, Any] | None = None
    surface_evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.brep_evidence is None:
            result.pop("brep_evidence")
        if self.surface_evidence is None:
            result.pop("surface_evidence")
        return result


@dataclass
class _Loop:
    kind: str
    primitives: list[Action]


@dataclass
class _FaceDefinition:
    outer: list[_Loop]
    inner: list[_Loop]


def _vector(cq: Any, point: tuple[float, float]) -> Any:
    return cq.Vector(float(point[0]), float(point[1]), 0.0)


def _wire_from_loop(cq: Any, loop: _Loop) -> Any:
    if not loop.primitives:
        raise CadQueryExecutionError("empty_loop")
    if len(loop.primitives) == 1 and isinstance(loop.primitives[0], AddCircle):
        circle = loop.primitives[0]
        if circle.radius <= 0:
            raise CadQueryExecutionError("nonpositive_circle_radius")
        edge = cq.Edge.makeCircle(
            float(circle.radius),
            _vector(cq, circle.center),
            cq.Vector(0.0, 0.0, 1.0),
        )
        return cq.Wire.assembleEdges([edge])

    edges: list[Any] = []
    for primitive in loop.primitives:
        if isinstance(primitive, AddLine):
            edges.append(cq.Edge.makeLine(_vector(cq, primitive.start), _vector(cq, primitive.end)))
        elif isinstance(primitive, AddArc):
            edges.append(
                cq.Edge.makeThreePointArc(
                    _vector(cq, primitive.start),
                    _vector(cq, primitive.mid),
                    _vector(cq, primitive.end),
                )
            )
        elif isinstance(primitive, AddCircle):
            raise CadQueryExecutionError("circle_mixed_with_other_primitives")
        else:
            raise CadQueryExecutionError(f"unsupported_curve:{type(primitive).__name__}")
    wire = cq.Wire.assembleEdges(edges)
    if not wire.IsClosed():
        raise CadQueryExecutionError("wire_not_closed")
    return wire


def _face_from_definition(cq: Any, definition: _FaceDefinition) -> Any:
    if len(definition.outer) != 1:
        raise CadQueryExecutionError(f"expected_one_outer_loop:found_{len(definition.outer)}")
    outer = _wire_from_loop(cq, definition.outer[0])
    inner = [_wire_from_loop(cq, loop) for loop in definition.inner]
    face = cq.Face.makeFromWires(outer, inner)
    if not face.isValid():
        raise CadQueryExecutionError("invalid_face")
    return face


def _shape_counts(shape: Any | None) -> tuple[float, int, int, bool]:
    if shape is None or (hasattr(shape, "isNull") and shape.isNull()):
        return 0.0, 0, 0, False
    volume = float(shape.Volume())
    solids = len(shape.Solids())
    faces = len(shape.Faces())
    valid = bool(shape.isValid()) and volume > 1e-9 and solids > 0
    return volume, solids, faces, valid


def _round_number(value: float) -> float:
    return float(f"{float(value):.8g}")


def _point_tuple(value: Any) -> list[float]:
    return [
        _round_number(float(value.x)),
        _round_number(float(value.y)),
        _round_number(float(value.z)),
    ]


def _shape_brep_evidence(shape: Any | None) -> dict[str, Any] | None:
    if shape is None or (hasattr(shape, "isNull") and shape.isNull()):
        return None
    try:
        solids = list(shape.Solids())
        faces = list(shape.Faces())
        edges = list(shape.Edges())
        vertices = list(shape.Vertices())
        bbox = shape.BoundingBox()
        face_types: dict[str, int] = {}
        for face in faces:
            kind = str(face.geomType()).lower()
            face_types[kind] = face_types.get(kind, 0) + 1
        edge_types: dict[str, int] = {}
        for edge in edges:
            kind = str(edge.geomType()).lower()
            edge_types[kind] = edge_types.get(kind, 0) + 1
        samples = {
            tuple(_point_tuple(vertex.Center())) for vertex in vertices
        } | {
            tuple(_point_tuple(face.Center())) for face in faces
        }
        ordered_samples = [list(point) for point in sorted(samples)[:128]]
        return {
            "solid_count": len(solids),
            "shell_count": len(shape.Shells()),
            "face_count": len(faces),
            "edge_count": len(edges),
            "vertex_count": len(vertices),
            "volume": _round_number(float(shape.Volume())),
            "surface_area": _round_number(
                sum(float(face.Area()) for face in faces)
            ),
            "bbox": [
                _round_number(bbox.xmin),
                _round_number(bbox.ymin),
                _round_number(bbox.zmin),
                _round_number(bbox.xmax),
                _round_number(bbox.ymax),
                _round_number(bbox.zmax),
            ],
            "face_type_histogram": dict(sorted(face_types.items())),
            "edge_type_histogram": dict(sorted(edge_types.items())),
            "euler_characteristic": len(vertices) - len(edges) + len(faces),
            "point_samples": ordered_samples,
        }
    except Exception:
        return None


def _halton(index: int, base: int) -> float:
    value = 0.0
    fraction = 1.0 / base
    while index:
        value += fraction * (index % base)
        index //= base
        fraction /= base
    return value


def _shape_surface_evidence(
    shape: Any | None, *, sample_count: int = 1024
) -> dict[str, Any] | None:
    """Return deterministic area-weighted points from an OCC tessellation."""
    if shape is None or (hasattr(shape, "isNull") and shape.isNull()):
        return None
    try:
        bbox = shape.BoundingBox()
        diagonal = math.sqrt(
            (float(bbox.xmax) - float(bbox.xmin)) ** 2
            + (float(bbox.ymax) - float(bbox.ymin)) ** 2
            + (float(bbox.zmax) - float(bbox.zmin)) ** 2
        )
        linear_tolerance = max(diagonal * 1e-3, 1e-5)
        vertices, triangles = shape.tessellate(linear_tolerance, 0.1)
        points = [tuple(float(value) for value in vertex.toTuple()) for vertex in vertices]
        weighted_triangles: list[tuple[float, tuple[int, int, int]]] = []
        cumulative_area = 0.0
        for triangle in triangles:
            first, second, third = (points[int(index)] for index in triangle)
            ab = tuple(second[i] - first[i] for i in range(3))
            ac = tuple(third[i] - first[i] for i in range(3))
            cross = (
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            )
            area = 0.5 * math.sqrt(sum(value * value for value in cross))
            if area <= 1e-12:
                continue
            cumulative_area += area
            weighted_triangles.append(
                (cumulative_area, tuple(int(index) for index in triangle))
            )
        if not weighted_triangles or cumulative_area <= 1e-12:
            return None

        cumulative = [row[0] for row in weighted_triangles]
        samples: list[list[float]] = []
        for sample_index in range(sample_count):
            area_position = (sample_index + 0.5) * cumulative_area / sample_count
            triangle_index = min(
                bisect.bisect_left(cumulative, area_position),
                len(weighted_triangles) - 1,
            )
            _, triangle = weighted_triangles[triangle_index]
            first, second, third = (points[index] for index in triangle)
            root_u = math.sqrt(_halton(sample_index + 1, 2))
            v = _halton(sample_index + 1, 3)
            weights = (1.0 - root_u, root_u * (1.0 - v), root_u * v)
            samples.append(
                [
                    _round_number(
                        weights[0] * first[axis]
                        + weights[1] * second[axis]
                        + weights[2] * third[axis]
                    )
                    for axis in range(3)
                ]
            )
        return {
            "sampling": "area_weighted_halton_over_occ_tessellation",
            "sample_count": len(samples),
            "linear_tolerance": _round_number(linear_tolerance),
            "angular_tolerance": 0.1,
            "bbox": [
                _round_number(bbox.xmin),
                _round_number(bbox.ymin),
                _round_number(bbox.zmin),
                _round_number(bbox.xmax),
                _round_number(bbox.ymax),
                _round_number(bbox.zmax),
            ],
            "mesh_vertices": len(vertices),
            "mesh_triangles": len(weighted_triangles),
            "point_samples": samples,
        }
    except Exception:
        return None


def execute_actions(
    actions: Sequence[Action],
    *,
    include_brep_evidence: bool = False,
    include_surface_evidence: bool = False,
    surface_sample_count: int = 1024,
) -> CadQueryExecutionResult:
    """Execute the TopoReward DSL with an OCC geometry kernel.

    This path intentionally does not call ``TopoVerifier``. It is an
    independent check that a completed action sequence constructs a valid B-rep.
    """

    cq = _cadquery()
    current_loop: _Loop | None = None
    current_face = _FaceDefinition(outer=[], inner=[])
    pending_face: _FaceDefinition | None = None
    profiles: dict[str, Any] = {}
    model: Any | None = None
    extrusions = 0

    try:
        for action_index, action in enumerate(actions):
            try:
                if isinstance(action, StartFace):
                    if current_loop is not None:
                        raise CadQueryExecutionError("face_started_with_open_loop")
                    current_face = _FaceDefinition(outer=[], inner=[])
                    pending_face = None
                elif isinstance(action, StartLoop):
                    if current_loop is not None:
                        raise CadQueryExecutionError("nested_loop")
                    current_loop = _Loop(kind=action.kind, primitives=[])
                elif isinstance(action, (AddLine, AddArc, AddCircle)):
                    if current_loop is None:
                        raise CadQueryExecutionError("curve_outside_loop")
                    current_loop.primitives.append(action)
                elif isinstance(action, EndLoop):
                    if current_loop is None:
                        raise CadQueryExecutionError("end_loop_without_loop")
                    if current_loop.kind == "outer":
                        current_face.outer.append(current_loop)
                    else:
                        current_face.inner.append(current_loop)
                    current_loop = None
                elif isinstance(action, EndFace):
                    if current_loop is not None:
                        raise CadQueryExecutionError("face_ended_with_open_loop")
                    pending_face = current_face
                elif isinstance(action, RegisterProfile):
                    if pending_face is None:
                        raise CadQueryExecutionError("register_without_face")
                    if action.profile_id in profiles:
                        raise CadQueryExecutionError("duplicate_profile_id")
                    profiles[action.profile_id] = _face_from_definition(cq, pending_face)
                    pending_face = None
                elif isinstance(action, Extrude):
                    face = profiles.get(action.profile_id)
                    if face is None:
                        raise CadQueryExecutionError("unknown_profile_reference")
                    if abs(float(action.depth)) <= 1e-9:
                        raise CadQueryExecutionError("zero_extrude_depth")
                    solid = cq.Solid.extrudeLinear(
                        face.outerWire(),
                        list(face.innerWires()),
                        cq.Vector(0.0, 0.0, float(action.depth)),
                    )
                    if solid.isNull() or not solid.isValid() or float(solid.Volume()) <= 1e-9:
                        raise CadQueryExecutionError("invalid_extruded_solid")
                    if action.op == "add":
                        model = solid if model is None else model.fuse(solid)
                    elif action.op == "cut":
                        if model is None:
                            raise CadQueryExecutionError("cut_without_base_solid")
                        model = model.cut(solid)
                    elif action.op == "intersect":
                        if model is None:
                            raise CadQueryExecutionError("intersect_without_base_solid")
                        model = model.intersect(solid)
                    else:
                        raise CadQueryExecutionError(f"unknown_extrude_operation:{action.op}")
                    if model.isNull():
                        raise CadQueryExecutionError("empty_boolean_result")
                    if not model.isValid():
                        raise CadQueryExecutionError("invalid_boolean_result")
                    extrusions += 1
            except Exception as exc:
                if isinstance(exc, CadQueryExecutionError):
                    failure = exc
                elif "Null TopoDS_Shape" in str(exc):
                    failure = CadQueryExecutionError("empty_boolean_result")
                else:
                    failure = CadQueryExecutionError(
                        f"kernel_exception:{type(exc).__name__}:{exc}"
                    )
                volume, solids, faces, valid = _shape_counts(model)
                return CadQueryExecutionResult(
                    success=False,
                    failure_action_index=action_index,
                    failure_type=str(failure).split(":", 1)[0],
                    failure_message=str(failure),
                    registered_profiles=len(profiles),
                    extrusions=extrusions,
                    volume=volume,
                    solids=solids,
                    faces=faces,
                    shape_valid=valid,
                    brep_evidence=(
                        _shape_brep_evidence(model)
                        if include_brep_evidence
                        else None
                    ),
                )
    except Exception as exc:
        return CadQueryExecutionResult(
            success=False,
            failure_action_index=None,
            failure_type="executor_exception",
            failure_message=f"{type(exc).__name__}:{exc}",
            registered_profiles=len(profiles),
            extrusions=extrusions,
            volume=0.0,
            solids=0,
            faces=0,
            shape_valid=False,
        )

    volume, solids, faces, valid = _shape_counts(model)
    success = valid and extrusions > 0
    return CadQueryExecutionResult(
        success=success,
        failure_action_index=None if success else len(actions),
        failure_type="none" if success else "no_valid_final_solid",
        failure_message="" if success else "program did not produce a valid positive-volume solid",
        registered_profiles=len(profiles),
        extrusions=extrusions,
        volume=volume,
        solids=solids,
        faces=faces,
        shape_valid=valid,
        brep_evidence=(
            _shape_brep_evidence(model) if include_brep_evidence else None
        ),
        surface_evidence=(
            _shape_surface_evidence(model, sample_count=surface_sample_count)
            if include_surface_evidence
            else None
        ),
    )
