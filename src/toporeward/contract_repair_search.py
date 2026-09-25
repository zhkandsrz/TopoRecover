from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .lm.parsing import parse_action_line
from .natural_repair_recall import evaluate_repaired_history
from .verifier import TopoVerifier


@dataclass(frozen=True)
class ContractRepair:
    rollback_start: int
    replacement: list[str]
    repaired_actions: list[str]
    verifier_calls: int

    def patch(self, observed_length: int) -> dict[str, Any]:
        return {
            "edits": [
                {
                    "start": self.rollback_start,
                    "end": observed_length,
                    "replacement": self.replacement,
                }
            ]
        }


def _replay_prefix(lines: Sequence[str]):
    verifier = TopoVerifier()
    state = verifier.initial_state()
    for line in lines:
        action = parse_action_line(str(line))
        if action is None:
            return None
        result = verifier.step(state, action)
        if not result.valid or result.next_state is None:
            return None
        state = result.next_state
    return state


def _profile_rows(state) -> list[dict[str, Any]]:
    rows = []
    for index, profile_id in enumerate(sorted(state.profiles)):
        rows.append(
            {
                "profile_id": str(profile_id),
                "profile_index": index,
                "loop_roles": sorted(loop.kind for loop in state.profiles[profile_id].loops),
            }
        )
    return rows


def _clean_contract_prefix(state, contract: Mapping[str, Any]) -> bool:
    if state.ended or state.pending_face is not None:
        return False
    if state.stack not in ([], ["sketch"]):
        return False

    desired_profiles = list(contract.get("profiles") or [])
    existing_profiles = _profile_rows(state)
    if len(existing_profiles) > len(desired_profiles):
        return False
    for existing, desired in zip(existing_profiles, desired_profiles):
        if existing["loop_roles"] != sorted(str(role) for role in desired["loop_roles"]):
            return False

    desired_graph = list(contract.get("extrusion_graph") or [])
    if len(state.extrusions) > len(desired_graph):
        return False
    profile_index = {
        str(row["profile_id"]): int(row["profile_index"])
        for row in existing_profiles
    }
    for existing, desired in zip(state.extrusions, desired_graph):
        if profile_index.get(str(existing["profile_id"]), -1) != int(
            desired["profile_index"]
        ):
            return False
        if str(existing["op"]) != str(desired["operation"]):
            return False
    return True


def _canonical_face(
    profile_index: int,
    loop_roles: Sequence[str],
    *,
    profile_id: str | None = None,
) -> list[str]:
    x0 = float(profile_index * 3)
    x1 = x0 + 2.0
    y0, y1 = 0.0, 2.0
    lines = [
        "StartFace",
        "StartLoop(kind=outer)",
        f"AddLine(start=({x0:g},{y0:g}), end=({x1:g},{y0:g}))",
        f"AddLine(start=({x1:g},{y0:g}), end=({x1:g},{y1:g}))",
        f"AddLine(start=({x1:g},{y1:g}), end=({x0:g},{y1:g}))",
        f"AddLine(start=({x0:g},{y1:g}), end=({x0:g},{y0:g}))",
        "EndLoop",
    ]
    inner_count = sum(str(role) == "inner" for role in loop_roles)
    for inner_index in range(inner_count):
        center_x = x0 + (inner_index + 1) * 2.0 / (inner_count + 1)
        radius = min(0.15, 0.5 / (inner_count + 1))
        lines.extend(
            [
                "StartLoop(kind=inner)",
                f"AddCircle(center=({center_x:g},1), radius={radius:g})",
                "EndLoop",
            ]
        )
    profile_id = profile_id or f"profile_{profile_index}"
    lines.extend(["EndFace", f"RegisterProfile(profile_id={profile_id})"])
    return lines


def compile_contract_suffix(state, contract: Mapping[str, Any]) -> list[str] | None:
    """Compile a canonical topology suffix from a clean verifier boundary.

    Geometry values for missing faces and extrusion depth are canonical defaults;
    the compiler promises topology-contract satisfaction, not geometric recovery.
    """

    if not _clean_contract_prefix(state, contract):
        return None
    desired_profiles = list(contract.get("profiles") or [])
    existing_profiles = _profile_rows(state)
    profile_ids = [str(row["profile_id"]) for row in existing_profiles]
    lines: list[str] = []

    if len(existing_profiles) < len(desired_profiles) and not state.stack:
        lines.append("StartSketch")
    for index in range(len(existing_profiles), len(desired_profiles)):
        desired = desired_profiles[index]
        if int(desired["profile_index"]) != index:
            return None
        profile_id = f"profile_{index}"
        if profile_id in state.profiles or profile_id in profile_ids:
            profile_id = f"repair_profile_{index}"
        profile_ids.append(profile_id)
        lines.extend(
            _canonical_face(
                index,
                list(desired["loop_roles"]),
                profile_id=profile_id,
            )
        )

    if state.stack == ["sketch"] or len(existing_profiles) < len(desired_profiles):
        lines.append("EndSketch")

    desired_graph = list(contract.get("extrusion_graph") or [])
    for row in desired_graph[len(state.extrusions) :]:
        profile_index = int(row["profile_index"])
        if not 0 <= profile_index < len(desired_profiles):
            return None
        operation = str(row["operation"])
        if operation not in {"add", "cut", "intersect"}:
            return None
        lines.append(
            f"Extrude(profile_id={profile_ids[profile_index]}, depth=1, op={operation})"
        )
    lines.append("End")
    return lines


def search_contract_repair(
    observed_lines: Sequence[str],
    failure_step: int,
    contract: Mapping[str, Any],
    *,
    max_rollback: int = 32,
    max_replacement_actions: int = 160,
) -> ContractRepair | None:
    """Find the latest clean boundary whose canonical suffix satisfies contract."""

    observed = [str(line) for line in observed_lines]
    upper = min(int(failure_step), len(observed))
    lower = max(0, upper - int(max_rollback))
    verifier_calls = 0
    for start in range(upper, lower - 1, -1):
        state = _replay_prefix(observed[:start])
        verifier_calls += 1
        if state is None:
            continue
        suffix = compile_contract_suffix(state, contract)
        if suffix is None or len(suffix) > max_replacement_actions:
            continue
        repaired = [*observed[:start], *suffix]
        outcome = evaluate_repaired_history(repaired, contract)
        verifier_calls += 1
        if outcome["valid"] and outcome["ended"] and outcome["intent_satisfied"]:
            return ContractRepair(
                rollback_start=start,
                replacement=suffix,
                repaired_actions=repaired,
                verifier_calls=verifier_calls,
            )
    return None
