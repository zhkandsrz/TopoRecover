"""Command counts shared by diagnosis and bounded repair."""

from .actions import Action, AddCircle, Extrude, RegisterProfile, StartLoop


def structure_stats(actions: list[Action] | tuple[Action, ...]) -> dict[str, int]:
    return {
        "action_count": len(actions),
        "profile_count": sum(isinstance(action, RegisterProfile) for action in actions),
        "hole_count": sum(isinstance(action, StartLoop) and action.kind == "inner" for action in actions),
        "circle_hole_count": sum(isinstance(action, AddCircle) for action in actions),
        "extrude_count": sum(isinstance(action, Extrude) for action in actions),
    }
