from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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


_OPERATION_MAP = {
    "NewBodyFeatureOperation": "add",
    "JoinFeatureOperation": "add",
    "CutFeatureOperation": "cut",
    "IntersectFeatureOperation": "intersect",
    "add": "add",
    "cut": "cut",
    "intersect": "intersect",
}


@dataclass(frozen=True)
class Text2CADBridgeResult:
    actions: tuple[Action, ...]
    action_provenance: tuple[dict[str, Any], ...]
    topology_faithful: bool
    operation_faithful: bool
    geometry_faithful: bool
    issues: tuple[str, ...]
    sketch_extrusions: int
    faces: int
    loops: int
    curves: int


def _numbered_items(value: Any, prefix: str) -> list[tuple[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a mapping of {prefix} entries")
    numbered: list[tuple[int, str, Any]] = []
    for key, item in value.items():
        match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", str(key))
        if match is None:
            continue
        numbered.append((int(match.group(1)), str(key), item))
    if not numbered:
        raise ValueError(f"No {prefix}_N entries found")
    return [(key, item) for _index, key, item in sorted(numbered)]


def _numbered_values(value: Any, prefix: str) -> list[Any]:
    return [item for _key, item in _numbered_items(value, prefix)]


def _point(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a coordinate sequence")
    if len(value) < 2:
        raise ValueError(f"{name} must contain at least two coordinates")
    return (float(value[0]), float(value[1]))


def _curve_action(key: str, payload: Any) -> Action:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Curve {key!r} must be a mapping")
    kind = key.split("_", 1)[0].lower()
    if kind == "line":
        return AddLine(
            start=_point(payload.get("Start Point"), "Start Point"),
            end=_point(payload.get("End Point"), "End Point"),
        )
    if kind == "arc":
        return AddArc(
            start=_point(payload.get("Start Point"), "Start Point"),
            mid=_point(payload.get("Mid Point"), "Mid Point"),
            end=_point(payload.get("End Point"), "End Point"),
        )
    if kind == "circle":
        radius = float(payload.get("Radius"))
        if radius <= 0:
            raise ValueError("Circle radius must be positive")
        return AddCircle(
            center=_point(payload.get("Center"), "Center"),
            radius=radius,
        )
    raise ValueError(f"Unsupported Text2CAD curve type: {kind!r}")


def _curves(loop: Any) -> list[Action]:
    if not isinstance(loop, Mapping):
        raise ValueError("Loop must be a mapping")
    actions: list[Action] = []
    for key, payload in loop.items():
        match = re.fullmatch(r"(?:line|arc|circle)_(\d+)", str(key).lower())
        if match is None:
            raise ValueError(f"Malformed Text2CAD curve key: {key!r}")
        actions.append(_curve_action(str(key), payload))
    if not actions:
        raise ValueError("Loop contains no curves")
    # CADSequence._json() preserves insertion order, including mixed curve
    # types whose per-type numeric suffixes are not a global ordering.
    return actions


def _near_zero(values: Any, tolerance: float) -> bool:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return False
    return all(abs(float(value)) <= tolerance for value in values)


def text2cad_minimal_json_to_actions(
    payload: Mapping[str, Any],
    *,
    zero_tolerance: float = 1e-6,
) -> Text2CADBridgeResult:
    """Convert Text2CAD's denumericalized minimal JSON into typed actions.

    Text2CAD's official ``CADSequence._json()`` representation preserves the
    sketch curves, coordinate system, two extrusion extents, and boolean/body
    operation. The TopoReward DSL represents local 2D sketch topology and one
    extrusion depth, so every omitted native quantity is surfaced as an issue.
    """

    parts = _numbered_values(payload.get("parts"), "part")
    actions: list[Action] = []
    provenance: list[dict[str, Any]] = []
    issues: set[str] = set()
    face_count = 0
    loop_count = 0
    curve_count = 0

    def append_action(
        action: Action,
        *,
        part_key: str | None = None,
        face_key: str | None = None,
        loop_key: str | None = None,
        curve_key: str | None = None,
        role: str,
    ) -> None:
        actions.append(action)
        provenance.append(
            {
                "part_key": part_key,
                "face_key": face_key,
                "loop_key": loop_key,
                "curve_key": curve_key,
                "role": role,
            }
        )

    for part_index, (part_key, part) in enumerate(
        _numbered_items(payload.get("parts"), "part")
    ):
        if not isinstance(part, Mapping):
            raise ValueError("Part must be a mapping")
        coordinate_system = part.get("coordinate_system")
        if not isinstance(coordinate_system, Mapping):
            raise ValueError("Part is missing coordinate_system")
        if not _near_zero(coordinate_system.get("Euler Angles"), zero_tolerance):
            issues.add("nonidentity_sketch_plane_not_represented")
        if not _near_zero(
            coordinate_system.get("Translation Vector"), zero_tolerance
        ):
            issues.add("nonzero_sketch_origin_not_represented")

        extrusion = part.get("extrusion")
        if not isinstance(extrusion, Mapping):
            raise ValueError("Part is missing extrusion")
        extent_one = float(extrusion.get("extrude_depth_towards_normal", 0.0))
        extent_two = float(extrusion.get("extrude_depth_opposite_normal", 0.0))
        nonzero_extents = [
            abs(value)
            for value in (extent_one, extent_two)
            if abs(value) > zero_tolerance
        ]
        if len(nonzero_extents) > 1:
            issues.add("two_sided_extrusion_not_represented")
        depth = max(nonzero_extents, default=0.0)
        if depth <= zero_tolerance:
            issues.add("zero_extrusion_depth")

        native_operation = str(extrusion.get("operation"))
        if native_operation not in _OPERATION_MAP:
            raise ValueError(f"Unsupported Text2CAD operation: {native_operation!r}")
        operation = _OPERATION_MAP[native_operation]
        if native_operation in {"NewBodyFeatureOperation", "JoinFeatureOperation"}:
            issues.add("body_identity_not_represented")

        append_action(StartSketch(), part_key=part_key, role="start_sketch")
        profile_ids: list[str] = []
        for face_index, (face_key, face) in enumerate(
            _numbered_items(part.get("sketch"), "face")
        ):
            append_action(
                StartFace(), part_key=part_key, face_key=face_key, role="start_face"
            )
            for loop_index, (loop_key, loop) in enumerate(
                _numbered_items(face, "loop")
            ):
                loop_kind = "outer" if loop_index == 0 else "inner"
                append_action(
                    StartLoop(loop_kind),
                    part_key=part_key,
                    face_key=face_key,
                    loop_key=loop_key,
                    role="start_loop",
                )
                curve_actions = _curves(loop)
                curve_items = list(loop.items())
                for (curve_key, _payload), curve_action in zip(
                    curve_items, curve_actions, strict=True
                ):
                    append_action(
                        curve_action,
                        part_key=part_key,
                        face_key=face_key,
                        loop_key=loop_key,
                        curve_key=str(curve_key),
                        role="curve",
                    )
                append_action(
                    EndLoop(),
                    part_key=part_key,
                    face_key=face_key,
                    loop_key=loop_key,
                    role="end_loop",
                )
                loop_count += 1
                curve_count += len(curve_actions)
            append_action(
                EndFace(), part_key=part_key, face_key=face_key, role="end_face"
            )
            profile_id = f"profile_{part_index}_{face_index}"
            append_action(
                RegisterProfile(profile_id),
                part_key=part_key,
                face_key=face_key,
                role="register_profile",
            )
            profile_ids.append(profile_id)
            face_count += 1
        append_action(EndSketch(), part_key=part_key, role="end_sketch")
        for face_index, profile_id in enumerate(profile_ids):
            append_action(
                Extrude(profile_id, depth, operation),
                part_key=part_key,
                face_key=f"face_{face_index + 1}",
                role="extrude",
            )

    append_action(End(), role="end")
    geometry_issues = {
        "nonidentity_sketch_plane_not_represented",
        "nonzero_sketch_origin_not_represented",
        "two_sided_extrusion_not_represented",
    }
    return Text2CADBridgeResult(
        actions=tuple(actions),
        action_provenance=tuple(provenance),
        topology_faithful="zero_extrusion_depth" not in issues,
        operation_faithful="body_identity_not_represented" not in issues,
        geometry_faithful=not bool(issues & geometry_issues),
        issues=tuple(sorted(issues)),
        sketch_extrusions=len(parts),
        faces=face_count,
        loops=loop_count,
        curves=curve_count,
    )
