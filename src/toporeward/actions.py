from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

Point: TypeAlias = tuple[float, float]


@dataclass(frozen=True)
class StartSketch:
    type: Literal["StartSketch"] = "StartSketch"


@dataclass(frozen=True)
class StartFace:
    type: Literal["StartFace"] = "StartFace"


@dataclass(frozen=True)
class StartLoop:
    kind: Literal["outer", "inner"] = "outer"
    type: Literal["StartLoop"] = "StartLoop"


@dataclass(frozen=True)
class AddLine:
    start: Point
    end: Point
    type: Literal["AddLine"] = "AddLine"


@dataclass(frozen=True)
class AddArc:
    start: Point
    mid: Point
    end: Point
    type: Literal["AddArc"] = "AddArc"


@dataclass(frozen=True)
class AddCircle:
    center: Point
    radius: float
    type: Literal["AddCircle"] = "AddCircle"


@dataclass(frozen=True)
class EndLoop:
    type: Literal["EndLoop"] = "EndLoop"


@dataclass(frozen=True)
class EndFace:
    type: Literal["EndFace"] = "EndFace"


@dataclass(frozen=True)
class RegisterProfile:
    profile_id: str
    type: Literal["RegisterProfile"] = "RegisterProfile"


@dataclass(frozen=True)
class EndSketch:
    type: Literal["EndSketch"] = "EndSketch"


@dataclass(frozen=True)
class Extrude:
    profile_id: str
    depth: float
    op: Literal["add", "cut", "intersect"] = "add"
    type: Literal["Extrude"] = "Extrude"


@dataclass(frozen=True)
class End:
    type: Literal["End"] = "End"


Action: TypeAlias = (
    StartSketch
    | StartFace
    | StartLoop
    | AddLine
    | AddArc
    | AddCircle
    | EndLoop
    | EndFace
    | RegisterProfile
    | EndSketch
    | Extrude
    | End
)


_ACTION_TYPES = {
    "StartSketch": StartSketch,
    "StartFace": StartFace,
    "StartLoop": StartLoop,
    "AddLine": AddLine,
    "AddArc": AddArc,
    "AddCircle": AddCircle,
    "EndLoop": EndLoop,
    "EndFace": EndFace,
    "RegisterProfile": RegisterProfile,
    "EndSketch": EndSketch,
    "Extrude": Extrude,
    "End": End,
}


def _point(value: Any) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Expected a 2D point, got {value!r}")
    return (float(value[0]), float(value[1]))


def action_to_dict(action: Action) -> dict[str, Any]:
    data = asdict(action)
    for key in ("start", "mid", "end", "center"):
        if key in data:
            data[key] = list(data[key])
    return data


def action_from_dict(data: dict[str, Any]) -> Action:
    action_type = data.get("type")
    if action_type not in _ACTION_TYPES:
        raise ValueError(f"Unknown action type: {action_type!r}")

    if action_type in {"StartSketch", "StartFace", "EndLoop", "EndFace", "EndSketch", "End"}:
        return _ACTION_TYPES[action_type]()
    if action_type == "StartLoop":
        return StartLoop(kind=data.get("kind", "outer"))
    if action_type == "AddLine":
        return AddLine(start=_point(data["start"]), end=_point(data["end"]))
    if action_type == "AddArc":
        return AddArc(start=_point(data["start"]), mid=_point(data["mid"]), end=_point(data["end"]))
    if action_type == "AddCircle":
        return AddCircle(center=_point(data["center"]), radius=float(data["radius"]))
    if action_type == "RegisterProfile":
        return RegisterProfile(profile_id=str(data["profile_id"]))
    if action_type == "Extrude":
        return Extrude(
            profile_id=str(data["profile_id"]),
            depth=float(data["depth"]),
            op=data.get("op", "add"),
        )
    raise AssertionError(f"Unhandled action type: {action_type}")


def actions_from_dicts(items: list[dict[str, Any]]) -> list[Action]:
    return [action_from_dict(item) for item in items]


def actions_to_dicts(actions: list[Action] | tuple[Action, ...]) -> list[dict[str, Any]]:
    return [action_to_dict(action) for action in actions]


def point_to_text(point: Point) -> str:
    return f"({point[0]:.6g},{point[1]:.6g})"


def action_to_text(action: Action) -> str:
    if isinstance(action, StartLoop):
        return f"StartLoop(kind={action.kind})"
    if isinstance(action, AddLine):
        return f"AddLine(start={point_to_text(action.start)}, end={point_to_text(action.end)})"
    if isinstance(action, AddArc):
        return (
            "AddArc("
            f"start={point_to_text(action.start)}, "
            f"mid={point_to_text(action.mid)}, "
            f"end={point_to_text(action.end)})"
        )
    if isinstance(action, AddCircle):
        return f"AddCircle(center={point_to_text(action.center)}, radius={action.radius:.6g})"
    if isinstance(action, RegisterProfile):
        return f"RegisterProfile(profile_id={action.profile_id})"
    if isinstance(action, Extrude):
        return f"Extrude(profile_id={action.profile_id}, depth={action.depth:.6g}, op={action.op})"
    return action.type


def actions_to_text(actions: list[Action] | tuple[Action, ...]) -> str:
    return "\n".join(action_to_text(action) for action in actions)

