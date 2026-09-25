from __future__ import annotations

from typing import Literal, TypeAlias

FailureType: TypeAlias = Literal[
    "hierarchy_error",
    "endpoint_discontinuity",
    "unclosed_loop",
    "degenerate_curve",
    "self_intersection",
    "invalid_hole_containment",
    "invalid_profile_reference",
    "kernel_execution_failure",
]


REPAIR_HINTS: dict[str, str] = {
    "hierarchy_error": "Respect the sketch/face/loop/extrude hierarchy before emitting this action.",
    "endpoint_discontinuity": "Next curve must start from the current loop tail.",
    "unclosed_loop": "Close the loop by returning to the loop start point before EndLoop.",
    "degenerate_curve": "Use distinct points and non-zero geometric parameters.",
    "self_intersection": "Choose a curve that does not cross existing curves in the current loop.",
    "invalid_hole_containment": "Inner loops must be inside the outer loop and cannot cross it.",
    "invalid_profile_reference": "Register a closed profile before extrusion and reference an existing profile id.",
    "kernel_execution_failure": "The final CAD kernel rejected this geometry; simplify or repair the solid operation.",
}

