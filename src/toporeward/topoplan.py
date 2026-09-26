from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import (
    Action,
    AddCircle,
    EndFace,
    EndSketch,
    EndLoop,
    Extrude,
    RegisterProfile,
    StartFace,
    StartLoop,
    action_to_text,
)
from .structure_stats import structure_stats


@dataclass(frozen=True)
class ProfilePlan:
    profile_id: str
    inner_loops: int
    circle_holes: int


def coerce_target_stats(target_stats: dict[str, Any] | None) -> dict[str, int]:
    target_stats = target_stats or {}
    return {
        "action_count": int(target_stats.get("action_count", 0) or 0),
        "profile_count": int(target_stats.get("profile_count", 0) or 0),
        "hole_count": int(target_stats.get("hole_count", 0) or 0),
        "circle_hole_count": int(target_stats.get("circle_hole_count", 0) or 0),
        "extrude_count": int(target_stats.get("extrude_count", 0) or 0),
    }


def profile_plan_from_actions(actions: list[Action] | tuple[Action, ...]) -> list[ProfilePlan]:
    plans: list[ProfilePlan] = []
    active_face = False
    inner_loops = 0
    circle_holes = 0
    pending_inner_circle = False
    for action in actions:
        if isinstance(action, StartFace):
            active_face = True
            inner_loops = 0
            circle_holes = 0
            pending_inner_circle = False
            continue
        if active_face and isinstance(action, StartLoop) and action.kind == "inner":
            inner_loops += 1
            pending_inner_circle = True
            continue
        if active_face and pending_inner_circle and action_to_text(action).startswith("AddCircle("):
            circle_holes += 1
            pending_inner_circle = False
            continue
        if isinstance(action, EndFace):
            active_face = False
            pending_inner_circle = False
            continue
        if isinstance(action, RegisterProfile):
            plans.append(ProfilePlan(action.profile_id, inner_loops, circle_holes))
            inner_loops = 0
            circle_holes = 0
            pending_inner_circle = False
    return plans


def next_milestone(prefix_actions: list[Action] | tuple[Action, ...], target_stats: dict[str, int]) -> str:
    progress = structure_stats(prefix_actions)
    if progress["profile_count"] < target_stats["profile_count"]:
        if progress["hole_count"] < target_stats["hole_count"]:
            return "complete_remaining_loops_before_registering_profile"
        return "register_remaining_profile"
    if progress["extrude_count"] < target_stats["extrude_count"]:
        return "extrude_registered_profiles"
    if not prefix_actions or not isinstance(prefix_actions[-1], EndSketch):
        return "close_sketch_then_finish"
    return "finish_program"


def current_loop_kind(actions: list[Action] | tuple[Action, ...]) -> str | None:
    stack: list[str] = []
    for action in actions:
        if isinstance(action, StartLoop):
            stack.append(action.kind)
        elif isinstance(action, EndLoop) and stack:
            stack.pop()
        elif isinstance(action, EndFace):
            stack.clear()
    return stack[-1] if stack else None


def topoplan_block(
    target_stats: dict[str, Any] | None,
    prefix_actions: list[Action] | tuple[Action, ...] | None = None,
    target_actions: list[Action] | tuple[Action, ...] | None = None,
) -> str:
    target = coerce_target_stats(target_stats)
    prefix = list(prefix_actions or [])
    progress = structure_stats(prefix)
    profile_plans = profile_plan_from_actions(target_actions or [])
    remaining_profiles = max(0, target["profile_count"] - progress["profile_count"])
    remaining_holes = max(0, target["hole_count"] - progress["hole_count"])
    remaining_extrudes = max(0, target["extrude_count"] - progress["extrude_count"])

    lines = [
        "<topoplan>",
        (
            "target: "
            f"profiles={target['profile_count']}, "
            f"holes={target['hole_count']}, "
            f"circle_holes={target['circle_hole_count']}, "
            f"extrusions={target['extrude_count']}, "
            f"actions={target['action_count']}"
        ),
        (
            "progress: "
            f"profiles={progress['profile_count']}, "
            f"holes={progress['hole_count']}, "
            f"circle_holes={progress['circle_hole_count']}, "
            f"extrusions={progress['extrude_count']}, "
            f"actions={progress['action_count']}"
        ),
        (
            "remaining: "
            f"profiles={remaining_profiles}, "
            f"holes={remaining_holes}, "
            f"extrusions={remaining_extrudes}"
        ),
    ]
    if profile_plans:
        for index, plan in enumerate(profile_plans):
            lines.append(
                f"profile_{index}: id={plan.profile_id}, outer_loops=1, "
                f"inner_loops={plan.inner_loops}, circle_holes={plan.circle_holes}"
            )
    else:
        lines.append("profile_plan: infer from target counts and avoid premature terminal actions")
    lines.append(f"next_milestone: {next_milestone(prefix, target)}")
    lines.append("terminal_rule: use EndFace/EndSketch/End only after required profiles, holes, and extrusions are satisfied")
    lines.append("</topoplan>")
    return "\n".join(lines) + "\n"


def progress_score(
    prefix_actions: list[Action] | tuple[Action, ...],
    candidate_action: Action,
    target_stats: dict[str, Any] | None,
) -> float:
    target = coerce_target_stats(target_stats)
    before = structure_stats(prefix_actions)
    after = structure_stats(list(prefix_actions) + [candidate_action])
    score = 0.0
    for key in ["profile_count", "hole_count", "circle_hole_count", "extrude_count"]:
        target_value = target[key]
        if target_value <= 0:
            if after[key] > before[key]:
                score -= 0.5
            continue
        before_value = min(before[key], target_value)
        after_value = min(after[key], target_value)
        if (
            key == "circle_hole_count"
            and after_value > before_value
            and not (isinstance(candidate_action, AddCircle) and current_loop_kind(prefix_actions) == "inner")
        ):
            after_value = before_value
        score += (after_value - before_value) / target_value
        if after[key] > target_value:
            score -= 0.5 * (after[key] - target_value)

    if isinstance(candidate_action, (EndFace, EndSketch)) and (
        after["profile_count"] < target["profile_count"] or after["hole_count"] < target["hole_count"]
    ):
        score -= 1.0
    if action_to_text(candidate_action) == "End" and (
        after["profile_count"] < target["profile_count"]
        or after["hole_count"] < target["hole_count"]
        or after["extrude_count"] < target["extrude_count"]
    ):
        score -= 1.5
    if isinstance(candidate_action, Extrude):
        if before["profile_count"] < target["profile_count"] or before["hole_count"] < target["hole_count"]:
            score -= 1.0
        elif after["extrude_count"] <= target["extrude_count"]:
            score += 0.25
    if isinstance(candidate_action, StartFace) and before["profile_count"] < target["profile_count"]:
        score += 0.1
    return score
