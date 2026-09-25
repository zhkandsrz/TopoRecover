from __future__ import annotations

import itertools
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence

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
    action_to_text,
    actions_from_dicts,
)
from .lm.parsing import parse_action_line
from .verifier import TopoVerifier, VerificationState


@dataclass(frozen=True)
class FinalReplay:
    valid: bool
    ended: bool
    failure_step: int | None
    failure_type: str | None
    state: VerificationState | None


@dataclass(frozen=True)
class EditHunk:
    hunk_id: int
    tag: str
    observed_start: int
    observed_end: int
    target_start: int
    target_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalRepair:
    selected_hunk_ids: tuple[int, ...]
    all_hunks: tuple[EditHunk, ...]
    intervention_certified: bool
    dependency_slice: tuple[int, ...]
    repaired_lines: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_hunk_ids": list(self.selected_hunk_ids),
            "all_hunks": [hunk.to_dict() for hunk in self.all_hunks],
            "intervention_certified": self.intervention_certified,
            "dependency_slice": list(self.dependency_slice),
            "repaired_lines": list(self.repaired_lines),
        }


def canonical_action_lines(lines: Sequence[str]) -> list[str]:
    canonical: list[str] = []
    for index, line in enumerate(lines):
        action = parse_action_line(str(line).strip())
        if action is None:
            raise ValueError(f"Unparseable action at index {index}: {line}")
        canonical.append(action_to_text(action))
    return canonical


def program_action_lines(row: dict[str, Any]) -> list[str]:
    return [action_to_text(action) for action in actions_from_dicts(row["actions"])]


def replay_final(
    lines: Sequence[str], verifier: TopoVerifier | None = None
) -> FinalReplay:
    verifier = verifier or TopoVerifier()
    state = verifier.initial_state()
    for index, line in enumerate(lines):
        action = parse_action_line(str(line).strip())
        if action is None:
            return FinalReplay(False, False, index, "unparseable_action", None)
        result = verifier.step(state, action)
        if not result.valid or result.next_state is None:
            return FinalReplay(
                False,
                False,
                index,
                str(result.failure_type or "unknown"),
                None,
            )
        state = result.next_state
    return FinalReplay(True, state.ended, None, None, state)


def semantic_state_signature(state: VerificationState) -> str:
    """Serialize the complete verifier state as the intended-history signature."""

    return json.dumps(asdict(state), sort_keys=True, separators=(",", ":"))


def semantic_history_signature(
    lines: Sequence[str], verifier: TopoVerifier | None = None
) -> str | None:
    outcome = replay_final(lines, verifier)
    if not outcome.valid or not outcome.ended or outcome.state is None:
        return None
    return semantic_state_signature(outcome.state)


def topology_intent_contract(
    lines: Sequence[str],
    *,
    level: str = "topology",
    verifier: TopoVerifier | None = None,
) -> dict[str, Any] | None:
    """Return an intent contract that never exposes an oracle action suffix.

    ``counts`` captures only aggregate construction targets. ``topology`` adds
    profile loop roles and the extrusion-reference graph. ``parameters`` also
    includes extrusion depths. These levels make the repair observability
    assumption explicit instead of silently passing the target history.
    """

    if level not in {"legality", "counts", "topology", "parameters", "semantic"}:
        raise ValueError(f"Unknown intent-contract level: {level}")
    outcome = replay_final(lines, verifier)
    if not outcome.valid or not outcome.ended or outcome.state is None:
        return None
    state = outcome.state
    if level == "legality":
        return {"valid": True, "ended": True}
    if level == "semantic":
        return asdict(state)

    profile_rows: list[dict[str, Any]] = []
    total_inner = 0
    profile_index: dict[str, int] = {}
    for index, profile_id in enumerate(sorted(state.profiles)):
        face = state.profiles[profile_id]
        loop_roles = sorted(loop.kind for loop in face.loops)
        total_inner += sum(role == "inner" for role in loop_roles)
        profile_index[profile_id] = index
        profile_rows.append(
            {
                "profile_index": index,
                "loop_roles": loop_roles,
            }
        )
    operations = Counter(str(row["op"]) for row in state.extrusions)
    counts = {
        "profile_count": len(state.profiles),
        "inner_loop_count": total_inner,
        "extrusion_count": len(state.extrusions),
        "operation_counts": dict(sorted(operations.items())),
    }
    if level == "counts":
        return counts

    extrusions: list[dict[str, Any]] = []
    for row in state.extrusions:
        item = {
            "profile_index": profile_index.get(str(row["profile_id"]), -1),
            "operation": str(row["op"]),
        }
        if level == "parameters":
            item["depth"] = round(float(row["depth"]), 6)
        extrusions.append(item)
    return {
        **counts,
        "profiles": profile_rows,
        "extrusion_graph": extrusions,
    }


def intent_contract_distinguishes(
    observed_lines: Sequence[str],
    target_lines: Sequence[str],
    *,
    level: str,
    verifier: TopoVerifier | None = None,
) -> bool:
    observed = topology_intent_contract(observed_lines, level=level, verifier=verifier)
    target = topology_intent_contract(target_lines, level=level, verifier=verifier)
    return observed is not None and target is not None and observed != target


def intent_recovered(
    candidate_lines: Sequence[str],
    target_lines: Sequence[str],
    verifier: TopoVerifier | None = None,
) -> bool:
    verifier = verifier or TopoVerifier()
    candidate = semantic_history_signature(candidate_lines, verifier)
    target = semantic_history_signature(target_lines, verifier)
    return candidate is not None and target is not None and candidate == target


def diff_hunks(
    observed_lines: Sequence[str], target_lines: Sequence[str]
) -> list[EditHunk]:
    observed = canonical_action_lines(observed_lines)
    target = canonical_action_lines(target_lines)
    matcher = SequenceMatcher(a=observed, b=target, autojunk=False)
    hunks: list[EditHunk] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        hunks.append(
            EditHunk(
                hunk_id=len(hunks),
                tag=tag,
                observed_start=i1,
                observed_end=i2,
                target_start=j1,
                target_end=j2,
            )
        )
    return hunks


def apply_target_hunks(
    observed_lines: Sequence[str],
    target_lines: Sequence[str],
    hunks: Sequence[EditHunk],
    selected_hunk_ids: Iterable[int],
) -> list[str]:
    observed = canonical_action_lines(observed_lines)
    target = canonical_action_lines(target_lines)
    selected = set(selected_hunk_ids)
    repaired: list[str] = []
    cursor = 0
    for hunk in hunks:
        repaired.extend(observed[cursor : hunk.observed_start])
        if hunk.hunk_id in selected:
            repaired.extend(target[hunk.target_start : hunk.target_end])
        else:
            repaired.extend(observed[hunk.observed_start : hunk.observed_end])
        cursor = hunk.observed_end
    repaired.extend(observed[cursor:])
    return repaired


def action_dependency_graph(lines: Sequence[str]) -> dict[int, set[int]]:
    """Build the explicit hierarchy/reference dependencies of a CAD history."""

    actions = [parse_action_line(line) for line in canonical_action_lines(lines)]
    graph: dict[int, set[int]] = {index: set() for index in range(len(actions))}
    sketch_start: int | None = None
    face_start: int | None = None
    loop_start: int | None = None
    loop_members: list[int] = []
    face_loop_ends: list[int] = []
    profile_sources: dict[str, int] = {}
    last_end_face: int | None = None
    last_end_sketch: int | None = None
    extrudes: list[int] = []

    for index, action in enumerate(actions):
        assert action is not None
        if isinstance(action, StartSketch):
            sketch_start = index
        elif isinstance(action, StartFace):
            if sketch_start is not None:
                graph[index].add(sketch_start)
            face_start = index
            face_loop_ends = []
        elif isinstance(action, StartLoop):
            if face_start is not None:
                graph[index].add(face_start)
            loop_start = index
            loop_members = []
        elif isinstance(action, (AddLine, AddArc, AddCircle)):
            if loop_start is not None:
                graph[index].add(loop_start)
            if loop_members:
                graph[index].add(loop_members[-1])
            loop_members.append(index)
        elif isinstance(action, EndLoop):
            if loop_start is not None:
                graph[index].add(loop_start)
            graph[index].update(loop_members)
            face_loop_ends.append(index)
            loop_start = None
            loop_members = []
        elif isinstance(action, EndFace):
            if face_start is not None:
                graph[index].add(face_start)
            graph[index].update(face_loop_ends)
            last_end_face = index
            face_start = None
            face_loop_ends = []
        elif isinstance(action, RegisterProfile):
            if last_end_face is not None:
                graph[index].add(last_end_face)
            profile_sources[action.profile_id] = index
        elif isinstance(action, EndSketch):
            if sketch_start is not None:
                graph[index].add(sketch_start)
            graph[index].update(profile_sources.values())
            last_end_sketch = index
            sketch_start = None
        elif isinstance(action, Extrude):
            if last_end_sketch is not None:
                graph[index].add(last_end_sketch)
            source = profile_sources.get(action.profile_id)
            if source is not None:
                graph[index].add(source)
            extrudes.append(index)
        elif isinstance(action, End):
            if last_end_sketch is not None:
                graph[index].add(last_end_sketch)
            graph[index].update(extrudes)
    return graph


def dependency_closure(graph: dict[int, set[int]], seeds: Iterable[int]) -> set[int]:
    closure = {index for index in seeds if index in graph}
    frontier = list(closure)
    while frontier:
        index = frontier.pop()
        for parent in graph.get(index, set()):
            if parent not in closure:
                closure.add(parent)
                frontier.append(parent)
    return closure


def _hunk_seed_indices(hunks: Sequence[EditHunk], selected: Iterable[int]) -> set[int]:
    selected_ids = set(selected)
    seeds: set[int] = set()
    for hunk in hunks:
        if hunk.hunk_id not in selected_ids:
            continue
        if hunk.observed_start < hunk.observed_end:
            seeds.update(range(hunk.observed_start, hunk.observed_end))
        else:
            seeds.add(max(0, hunk.observed_start - 1))
    return seeds


def minimum_causal_repair(
    observed_lines: Sequence[str],
    target_lines: Sequence[str],
    *,
    max_hunks: int = 12,
    verifier: TopoVerifier | None = None,
) -> CausalRepair | None:
    """Find a minimum set of target-diff interventions that restores intent.

    The returned slice is called causal only because every selected hunk is
    necessary under a concrete intervention test. It is not inferred from text
    similarity alone.
    """

    verifier = verifier or TopoVerifier()
    hunks = diff_hunks(observed_lines, target_lines)
    if not hunks or len(hunks) > max_hunks:
        return None
    selected: tuple[int, ...] | None = None
    repaired: list[str] | None = None
    for size in range(1, len(hunks) + 1):
        for candidate in itertools.combinations(range(len(hunks)), size):
            lines = apply_target_hunks(observed_lines, target_lines, hunks, candidate)
            if intent_recovered(lines, target_lines, verifier):
                selected = candidate
                repaired = lines
                break
        if selected is not None:
            break
    if selected is None or repaired is None:
        return None

    necessary = all(
        not intent_recovered(
            apply_target_hunks(
                observed_lines,
                target_lines,
                hunks,
                [item for item in selected if item != removed],
            ),
            target_lines,
            verifier,
        )
        for removed in selected
    )
    graph = action_dependency_graph(observed_lines)
    closure = dependency_closure(graph, _hunk_seed_indices(hunks, selected))
    return CausalRepair(
        selected_hunk_ids=selected,
        all_hunks=tuple(hunks),
        intervention_certified=necessary,
        dependency_slice=tuple(sorted(closure)),
        repaired_lines=tuple(repaired),
    )


def oracle_last_action_retry_recovers(
    observed_lines: Sequence[str],
    target_lines: Sequence[str],
    verifier: TopoVerifier | None = None,
) -> bool:
    if not observed_lines or not target_lines:
        return False
    candidate = [*canonical_action_lines(observed_lines[:-1]), canonical_action_lines(target_lines)[-1]]
    return intent_recovered(candidate, target_lines, verifier)


def _inner_loop_spans(actions: Sequence[Action]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, action in enumerate(actions):
        if isinstance(action, StartLoop) and action.kind == "inner":
            start = index
        elif isinstance(action, EndLoop) and start is not None:
            spans.append((start, index + 1))
            start = None
    return spans


def build_controlled_composite_fault(
    target_lines: Sequence[str],
    *,
    fault_count: int = 2,
    variant_index: int = 0,
    verifier: TopoVerifier | None = None,
) -> dict[str, Any] | None:
    """Create a verifier-valid, ended history with multiple semantic faults."""

    verifier = verifier or TopoVerifier()
    canonical = canonical_action_lines(target_lines)
    actions = [parse_action_line(line) for line in canonical]
    assert all(action is not None for action in actions)
    typed_actions = [action for action in actions if action is not None]
    mutations: list[dict[str, Any]] = []

    for start, end in _inner_loop_spans(typed_actions):
        mutations.append(
            {
                "kind": "delete_inner_loop",
                "start": start,
                "end": end,
                "replacement": [],
            }
        )
    profile_ids = [
        action.profile_id for action in typed_actions if isinstance(action, RegisterProfile)
    ]
    extrusion_indices = [
        index for index, action in enumerate(typed_actions) if isinstance(action, Extrude)
    ]
    for index, action in enumerate(typed_actions):
        if not isinstance(action, Extrude):
            continue
        depth = round(float(action.depth) + 0.1, 6)
        replacement = action_to_text(Extrude(action.profile_id, depth, action.op))
        mutations.append(
            {
                "kind": "wrong_extrude_depth",
                "start": index,
                "end": index + 1,
                "replacement": [replacement],
            }
        )
        alternatives = sorted(profile_id for profile_id in profile_ids if profile_id != action.profile_id)
        if alternatives:
            replacement = action_to_text(
                Extrude(alternatives[0], float(action.depth), action.op)
            )
            mutations.append(
                {
                    "kind": "wrong_extrude_reference",
                    "start": index,
                    "end": index + 1,
                    "replacement": [replacement],
                }
            )
        if index != extrusion_indices[0]:
            operation = "add" if action.op != "add" else "cut"
            replacement = action_to_text(
                Extrude(action.profile_id, float(action.depth), operation)
            )
            mutations.append(
                {
                    "kind": "wrong_extrude_operation",
                    "start": index,
                    "end": index + 1,
                    "replacement": [replacement],
                }
            )
        if len(extrusion_indices) > 1:
            mutations.append(
                {
                    "kind": "delete_extrusion",
                    "start": index,
                    "end": index + 1,
                    "replacement": [],
                }
            )

    combinations: list[tuple[dict[str, Any], ...]] = []
    for candidate in itertools.combinations(mutations, fault_count):
        if len({str(item["kind"]) for item in candidate}) < fault_count:
            continue
        spans = [
            set(range(int(item["start"]), int(item["end"]))) for item in candidate
        ]
        if any(first & second for first, second in itertools.combinations(spans, 2)):
            continue
        combinations.append(candidate)
    combinations.sort(
        key=lambda candidate: hashlib.sha256(
            (
                "\n".join(canonical)
                + "|"
                + "|".join(
                    f"{item['kind']}:{item['start']}:{item['end']}" for item in candidate
                )
            ).encode("utf-8")
        ).hexdigest()
    )
    valid_variant = 0
    for selected_tuple in combinations:
        selected = list(selected_tuple)
        corrupted = list(canonical)
        for mutation in sorted(
            selected, key=lambda item: int(item["start"]), reverse=True
        ):
            corrupted[int(mutation["start"]) : int(mutation["end"])] = list(
                mutation["replacement"]
            )
        outcome = replay_final(corrupted, verifier)
        if (
            not outcome.valid
            or not outcome.ended
            or intent_recovered(corrupted, canonical, verifier)
        ):
            continue
        causal = minimum_causal_repair(corrupted, canonical, verifier=verifier)
        if causal is None or len(causal.selected_hunk_ids) < fault_count:
            continue
        if valid_variant < int(variant_index):
            valid_variant += 1
            continue
        return {
            "target_actions": canonical,
            "corrupted_actions": corrupted,
            "injected_faults": selected,
            "causal_repair": causal.to_dict(),
            "oracle_last_action_retry_recovers": oracle_last_action_retry_recovers(
                corrupted, canonical, verifier
            ),
        }
    return None
