from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from toporeward.actions import Action, action_to_dict


def _action_dict(action: Action | Mapping[str, Any]) -> dict[str, Any]:
    return dict(action) if isinstance(action, Mapping) else action_to_dict(action)


def structure_family_payload(
    actions: Sequence[Action | Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a geometry-invariant CAD construction-family description."""

    profile_ids: dict[str, str] = {}
    unknown_profile_ids: dict[str, str] = {}
    skeleton: list[str] = []
    loop_stack: list[dict[str, Any]] = []
    loop_graph: list[dict[str, Any]] = []
    registration_graph: list[dict[str, Any]] = []
    extrusion_graph: list[dict[str, Any]] = []
    sketch_index = -1
    face_index = -1

    def canonical_profile(raw: Any, *, registered: bool) -> str:
        value = str(raw)
        if value in profile_ids:
            return profile_ids[value]
        if registered:
            canonical = f"profile_{len(profile_ids)}"
            profile_ids[value] = canonical
            return canonical
        if value not in unknown_profile_ids:
            unknown_profile_ids[value] = f"unregistered_{len(unknown_profile_ids)}"
        return unknown_profile_ids[value]

    for action_index, raw_action in enumerate(actions):
        action = _action_dict(raw_action)
        action_type = str(action.get("type") or "UNKNOWN")
        if action_type == "StartSketch":
            sketch_index += 1
            skeleton.append("StartSketch")
        elif action_type == "StartFace":
            face_index += 1
            skeleton.append("StartFace")
        elif action_type == "StartLoop":
            kind = str(action.get("kind") or "outer")
            loop_index = len(loop_graph) + len(loop_stack)
            parent = loop_stack[-1]["loop_index"] if loop_stack else None
            loop_stack.append(
                {
                    "loop_index": loop_index,
                    "kind": kind,
                    "parent": parent,
                    "sketch": sketch_index,
                    "face": face_index,
                    "primitives": [],
                }
            )
            skeleton.append(f"StartLoop:{kind}")
        elif action_type in {"AddLine", "AddArc", "AddCircle"}:
            skeleton.append(action_type)
            if loop_stack:
                loop_stack[-1]["primitives"].append(action_type)
        elif action_type == "EndLoop":
            skeleton.append("EndLoop")
            if loop_stack:
                loop_graph.append(loop_stack.pop())
            else:
                loop_graph.append(
                    {
                        "loop_index": f"unmatched_end_{action_index}",
                        "kind": "unknown",
                        "parent": None,
                        "sketch": sketch_index,
                        "face": face_index,
                        "primitives": [],
                    }
                )
        elif action_type == "RegisterProfile":
            profile = canonical_profile(action.get("profile_id"), registered=True)
            skeleton.append(f"RegisterProfile:{profile}")
            registration_graph.append(
                {"profile": profile, "sketch": sketch_index, "face": face_index}
            )
        elif action_type == "Extrude":
            profile = canonical_profile(action.get("profile_id"), registered=False)
            operation = str(action.get("op") or "add")
            skeleton.append(f"Extrude:{profile}:{operation}")
            extrusion_graph.append(
                {
                    "profile": profile,
                    "operation": operation,
                    "index": len(extrusion_graph),
                }
            )
        else:
            skeleton.append(action_type)

    while loop_stack:
        unfinished = dict(loop_stack.pop())
        unfinished["unclosed"] = True
        loop_graph.append(unfinished)

    return {
        "normalized_action_skeleton": skeleton,
        "loop_topology": loop_graph,
        "profile_registration_graph": registration_graph,
        "extrusion_reference_graph": extrusion_graph,
        "extrusion_operation_sequence": [item["operation"] for item in extrusion_graph],
        "count_tuple": {
            "profiles": len(registration_graph),
            "holes": sum(item["kind"] == "inner" for item in loop_graph),
            "extrusions": len(extrusion_graph),
        },
    }


def structure_family_id(actions: Sequence[Action | Mapping[str, Any]]) -> str:
    payload = structure_family_payload(actions)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()
