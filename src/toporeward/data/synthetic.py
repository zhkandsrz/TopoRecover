from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ..actions import (
    Action,
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
    action_to_dict,
    actions_to_dicts,
)
from ..program import Program
from ..verifier import TopoVerifier


def rectangle_loop(x: float, y: float, w: float, h: float) -> list[Action]:
    p0 = (x, y)
    p1 = (x + w, y)
    p2 = (x + w, y + h)
    p3 = (x, y + h)
    return [
        AddLine(p0, p1),
        AddLine(p1, p2),
        AddLine(p2, p3),
        AddLine(p3, p0),
    ]


def make_rectangle_program(program_id: str, with_hole: bool = False) -> Program:
    actions: list[Action] = [StartSketch(), StartFace(), StartLoop("outer")]
    actions.extend(rectangle_loop(0.0, 0.0, 1.0, 1.0))
    actions.append(EndLoop())
    if with_hole:
        actions.append(StartLoop("inner"))
        actions.extend(rectangle_loop(0.25, 0.25, 0.2, 0.2))
        actions.append(EndLoop())
    actions.extend(
        [
            EndFace(),
            RegisterProfile("profile_0"),
            EndSketch(),
            Extrude("profile_0", depth=0.5, op="add"),
            End(),
        ]
    )
    return Program(tuple(actions), program_id=program_id, source="synthetic")


def make_valid_programs(count: int, seed: int = 0) -> list[Program]:
    rng = random.Random(seed)
    programs: list[Program] = []
    for idx in range(count):
        programs.append(make_rectangle_program(f"synthetic_{idx:06d}", with_hole=rng.random() < 0.35))
    return programs


def _shift_point(point: tuple[float, float], dx: float = 0.173) -> tuple[float, float]:
    return (point[0] + dx, point[1] + dx)


def build_preference_pairs(programs: list[Program]) -> list[dict[str, Any]]:
    verifier = TopoVerifier()
    pairs: list[dict[str, Any]] = []

    for program in programs:
        prefix: list[Action] = []
        state = verifier.initial_state()
        for action in program.actions:
            chosen_result = verifier.step(state, action)
            if not chosen_result.valid:
                break

            rejected: Action | None = None
            expected_failure: str | None = None
            if isinstance(action, AddLine) and len(prefix) >= 4:
                rejected = AddLine(start=_shift_point(action.start), end=action.end)
                expected_failure = "endpoint_discontinuity"
            elif isinstance(action, Extrude):
                rejected = Extrude("missing_profile", depth=action.depth, op=action.op)
                expected_failure = "invalid_profile_reference"

            if rejected is not None and expected_failure is not None:
                rejected_result = verifier.step(state, rejected)
                if not rejected_result.valid:
                    pairs.append(
                        {
                            "program_id": program.program_id,
                            "prefix": actions_to_dicts(prefix),
                            "chosen": {
                                "action": action_to_dict(action),
                                "valid": True,
                            },
                            "rejected": {
                                "action": action_to_dict(rejected),
                                "valid": False,
                                "failure_type": rejected_result.failure_type,
                                "repair_hint": rejected_result.repair_hint,
                            },
                        }
                    )

            prefix.append(action)
            assert chosen_result.next_state is not None
            state = chosen_result.next_state

    pairs.extend(_manual_failure_pairs())
    return pairs


def build_negative_examples(programs: list[Program]) -> list[dict[str, Any]]:
    pairs = build_preference_pairs(programs)
    negatives = [
        {
            "program_id": pair["program_id"],
            "prefix": pair["prefix"],
            "action": pair["rejected"]["action"],
            "valid": False,
            "failure_type": pair["rejected"]["failure_type"],
            "repair_hint": pair["rejected"]["repair_hint"],
        }
        for pair in pairs
    ]
    negatives.extend(_manual_negative_examples())
    return negatives


def _manual_failure_pairs() -> list[dict[str, Any]]:
    items: list[tuple[str, list[Action], Action, Action]] = []

    unclosed_prefix: list[Action] = [
        StartSketch(),
        StartFace(),
        StartLoop("outer"),
        AddLine((0.0, 0.0), (1.0, 0.0)),
        AddLine((1.0, 0.0), (1.0, 1.0)),
        AddLine((1.0, 1.0), (0.0, 1.0)),
    ]
    items.append(("manual_unclosed_loop", unclosed_prefix, AddLine((0.0, 1.0), (0.0, 0.0)), EndLoop()))

    intersect_prefix: list[Action] = [
        StartSketch(),
        StartFace(),
        StartLoop("outer"),
        AddLine((0.0, 0.0), (1.0, 1.0)),
        AddLine((1.0, 1.0), (0.0, 1.0)),
    ]
    items.append(("manual_self_intersection", intersect_prefix, AddLine((0.0, 1.0), (0.0, 0.0)), AddLine((0.0, 1.0), (1.0, 0.0))))

    degenerate_prefix: list[Action] = [StartSketch(), StartFace(), StartLoop("outer")]
    items.append(("manual_degenerate_curve", degenerate_prefix, AddLine((0.0, 0.0), (1.0, 0.0)), AddLine((0.0, 0.0), (0.0, 0.0))))

    verifier = TopoVerifier()
    pairs: list[dict[str, Any]] = []
    for name, prefix, chosen, rejected in items:
        state = verifier.initial_state()
        ok = True
        for action in prefix:
            result = verifier.step(state, action)
            if not result.valid:
                ok = False
                break
            assert result.next_state is not None
            state = result.next_state
        if not ok:
            continue
        chosen_result = verifier.step(state, chosen)
        rejected_result = verifier.step(state, rejected)
        if chosen_result.valid and not rejected_result.valid:
            pairs.append(
                {
                    "program_id": name,
                    "prefix": actions_to_dicts(prefix),
                    "chosen": {"action": action_to_dict(chosen), "valid": True},
                    "rejected": {
                        "action": action_to_dict(rejected),
                        "valid": False,
                        "failure_type": rejected_result.failure_type,
                        "repair_hint": rejected_result.repair_hint,
                    },
                }
            )

    return pairs


def _manual_negative_examples() -> list[dict[str, Any]]:
    invalid_hole_prefix: list[Action] = [
        StartSketch(),
        StartFace(),
        StartLoop("outer"),
        *rectangle_loop(0.0, 0.0, 1.0, 1.0),
        EndLoop(),
        StartLoop("inner"),
        *rectangle_loop(1.5, 1.5, 0.2, 0.2),
        EndLoop(),
    ]
    verifier = TopoVerifier()
    state = verifier.initial_state()
    for action in invalid_hole_prefix:
        result = verifier.step(state, action)
        if not result.valid:
            return []
        assert result.next_state is not None
        state = result.next_state
    result = verifier.step(state, EndFace())
    if result.valid:
        return []
    return [
        {
            "program_id": "manual_invalid_hole_containment",
            "prefix": actions_to_dicts(invalid_hole_prefix),
            "action": action_to_dict(EndFace()),
            "valid": False,
            "failure_type": result.failure_type,
            "repair_hint": result.repair_hint,
        }
    ]


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_programs(path: Path, programs: list[Program]) -> None:
    write_jsonl(path, [program.to_dict() for program in programs])
