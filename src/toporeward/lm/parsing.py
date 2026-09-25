from __future__ import annotations

import re
from dataclasses import asdict
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
from ..verifier import TopoVerifier

NUMBER = r"-?\d+(?:\.\d+)?(?:e-?\d+)?"
POINT = rf"\(({NUMBER}),\s*({NUMBER})\)"
ADD_LINE_RE = re.compile(rf"^AddLine\(start={POINT},\s*end={POINT}\)$")
ADD_ARC_RE = re.compile(rf"^AddArc\(start={POINT},\s*mid={POINT},\s*end={POINT}\)$")
ADD_CIRCLE_RE = re.compile(rf"^AddCircle\(center={POINT},\s*radius=({NUMBER})\)$")
START_LOOP_RE = re.compile(r"^StartLoop\(kind=(outer|inner)\)$")
END_LOOP_WITH_KIND_RE = re.compile(r"^EndLoop\(kind=(outer|inner)\)$")
REGISTER_RE = re.compile(r"^RegisterProfile\(profile_id=([A-Za-z0-9_\-]+)\)$")
EXTRUDE_RE = re.compile(rf"^Extrude\(profile_id=([A-Za-z0-9_\-]+),\s*depth=({NUMBER}),\s*op=(add|cut|intersect)\)$")


def parse_action_line(line: str) -> Action | None:
    line = line.strip()
    if line == "StartSketch":
        return StartSketch()
    if line == "StartFace":
        return StartFace()
    if line == "EndLoop":
        return EndLoop()
    if line == "EndFace":
        return EndFace()
    if line == "EndSketch":
        return EndSketch()
    if line == "End":
        return End()

    start_loop = START_LOOP_RE.match(line)
    if start_loop:
        return StartLoop(start_loop.group(1))
    if END_LOOP_WITH_KIND_RE.match(line):
        return EndLoop()

    add_line = ADD_LINE_RE.match(line)
    if add_line:
        values = [float(item) for item in add_line.groups()]
        return AddLine((values[0], values[1]), (values[2], values[3]))

    add_arc = ADD_ARC_RE.match(line)
    if add_arc:
        values = [float(item) for item in add_arc.groups()]
        return AddArc((values[0], values[1]), (values[2], values[3]), (values[4], values[5]))

    add_circle = ADD_CIRCLE_RE.match(line)
    if add_circle:
        values = [float(item) for item in add_circle.groups()]
        return AddCircle((values[0], values[1]), values[2])

    register = REGISTER_RE.match(line)
    if register:
        return RegisterProfile(register.group(1))

    extrude = EXTRUDE_RE.match(line)
    if extrude:
        return Extrude(extrude.group(1), float(extrude.group(2)), extrude.group(3))

    return None


def parse_action_text(text: str) -> list[Action]:
    actions: list[Action] = []
    for line in text.splitlines():
        action = parse_action_line(line)
        if action is not None:
            actions.append(action)
    return actions


def replay_action_text(text: str) -> dict[str, Any]:
    actions = parse_action_text(text)
    verifier = TopoVerifier()
    state = verifier.initial_state()
    for index, action in enumerate(actions):
        result = verifier.step(state, action)
        if not result.valid:
            return {
                "num_parsed_actions": len(actions),
                "valid_prefix_length": index,
                "first_invalid_action": asdict(action),
                "failure_type": result.failure_type,
                "repair_hint": result.repair_hint,
                "ended": False,
            }
        assert result.next_state is not None
        state = result.next_state
    return {
        "num_parsed_actions": len(actions),
        "valid_prefix_length": len(actions),
        "first_invalid_action": None,
        "failure_type": None,
        "repair_hint": None,
        "ended": state.ended,
    }
