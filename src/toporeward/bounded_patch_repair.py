from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from .actions import (
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
    action_to_text,
)
from .lm.parsing import parse_action_line
from .rlvr import structure_stats
from .verifier import TopoVerifier, VerificationState


@dataclass(frozen=True)
class ProgramReplay:
    valid: bool
    ended: bool
    failure_step: int | None
    failure_type: str | None
    stats: dict[str, int]


@dataclass(frozen=True)
class CertifiedPatch:
    source: str
    repair_start: int
    rollback_length: int
    patch_lines: tuple[str, ...]
    target_resume_index: int
    proposal_evaluations: int
    replay_valid: bool
    replay_ended: bool
    strict_target_recovered: bool
    replay_failure_step: int | None
    replay_failure_type: str | None
    replay_stats: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["patch_lines"] = list(self.patch_lines)
        row["patch_length"] = len(self.patch_lines)
        return row


@dataclass
class _PartialPatch:
    repair_start: int
    rollback_length: int
    state: VerificationState
    actions: list[Action]
    patch_lines: tuple[str, ...]


def state_signature(state: VerificationState) -> str:
    """Return a deterministic signature for verifier-state resynchronization."""

    return json.dumps(asdict(state), sort_keys=True, separators=(",", ":"))


def replay_lines(
    lines: Sequence[str], verifier: TopoVerifier | None = None
) -> ProgramReplay:
    verifier = verifier or TopoVerifier()
    state = verifier.initial_state()
    actions: list[Action] = []
    for index, line in enumerate(lines):
        action = parse_action_line(line)
        if action is None:
            return ProgramReplay(
                False,
                False,
                index,
                "unparseable_action",
                structure_stats(actions),
            )
        result = verifier.step(state, action)
        if not result.valid or result.next_state is None:
            return ProgramReplay(
                False,
                False,
                index,
                str(result.failure_type or "unknown"),
                structure_stats(actions),
            )
        state = result.next_state
        actions.append(action)
    return ProgramReplay(True, state.ended, None, None, structure_stats(actions))


def strict_target_reached(outcome: ProgramReplay, target_stats: dict[str, Any]) -> bool:
    if not outcome.valid or not outcome.ended:
        return False
    for key in ("profile_count", "hole_count", "extrude_count"):
        if int(outcome.stats.get(key, 0) or 0) != int(target_stats.get(key, 0) or 0):
            return False
    return True


def replay_prefix(
    lines: Sequence[str], verifier: TopoVerifier
) -> tuple[VerificationState, list[Action]] | None:
    state = verifier.initial_state()
    actions: list[Action] = []
    for line in lines:
        action = parse_action_line(line)
        if action is None:
            return None
        result = verifier.step(state, action)
        if not result.valid or result.next_state is None:
            return None
        state = result.next_state
        actions.append(action)
    return state, actions


def target_state_index(
    target_lines: Sequence[str], verifier: TopoVerifier
) -> tuple[dict[str, list[int]], list[Action]]:
    state = verifier.initial_state()
    signatures: dict[str, list[int]] = {state_signature(state): [0]}
    actions: list[Action] = []
    for index, line in enumerate(target_lines, start=1):
        action = parse_action_line(line)
        if action is None:
            raise ValueError(f"Unparseable target action at {index - 1}: {line}")
        result = verifier.step(state, action)
        if not result.valid or result.next_state is None:
            raise ValueError(
                f"Target action fails verifier replay at {index - 1}: {line}"
            )
        state = result.next_state
        actions.append(action)
        signatures.setdefault(state_signature(state), []).append(index)
    return signatures, actions


def _unique_lines(lines: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = str(line).strip()
        if not normalized or normalized in seen or parse_action_line(normalized) is None:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def event_candidate_lines(event: dict[str, Any], limit: int = 16) -> list[str]:
    return _unique_lines(
        str((item or {}).get("line") or "") for item in event.get("top", [])
    )[:limit]


def generic_repair_lines(state: VerificationState, support_lines: Sequence[str]) -> list[str]:
    """Construct state-only repair actions without reading target actions."""

    lines = [
        action_to_text(StartSketch()),
        action_to_text(StartFace()),
        action_to_text(StartLoop("outer")),
        action_to_text(StartLoop("inner")),
        action_to_text(EndLoop()),
        action_to_text(EndFace()),
        action_to_text(EndSketch()),
        action_to_text(End()),
    ]
    loop = state.current_loop
    if loop is not None and loop.start is not None and loop.tail is not None:
        lines.append(action_to_text(AddLine(start=loop.tail, end=loop.start)))
    lines.append(action_to_text(RegisterProfile(f"profile_{len(state.profiles)}")))

    observed_extrudes = [
        parse_action_line(line)
        for line in support_lines
        if line.startswith("Extrude(")
    ]
    observed_depth_ops = {
        (float(action.depth), str(action.op))
        for action in observed_extrudes
        if isinstance(action, Extrude)
    }
    for profile_id in sorted(state.profiles):
        for depth, operation in sorted(observed_depth_ops):
            lines.append(action_to_text(Extrude(profile_id, depth, operation)))
    return _unique_lines(lines)


def _partial_priority(partial: _PartialPatch, target_stats: dict[str, Any]) -> tuple[Any, ...]:
    stats = structure_stats(partial.actions)
    deficit = sum(
        max(0, int(target_stats.get(key, 0) or 0) - int(stats.get(key, 0) or 0))
        for key in ("profile_count", "hole_count", "extrude_count")
    )
    return (
        deficit,
        partial.rollback_length,
        len(partial.patch_lines),
        state_signature(partial.state),
    )


def _certify(
    *,
    source: str,
    history_lines: Sequence[str],
    target_lines: Sequence[str],
    target_stats: dict[str, Any],
    repair_start: int,
    failure_index: int,
    patch_lines: Sequence[str],
    target_resume_index: int,
    proposal_evaluations: int,
    verifier: TopoVerifier,
) -> CertifiedPatch:
    repaired = [
        *history_lines[:repair_start],
        *patch_lines,
        *target_lines[target_resume_index:],
    ]
    outcome = replay_lines(repaired, verifier)
    return CertifiedPatch(
        source=source,
        repair_start=repair_start,
        rollback_length=max(0, failure_index - repair_start + 1),
        patch_lines=tuple(patch_lines),
        target_resume_index=target_resume_index,
        proposal_evaluations=proposal_evaluations,
        replay_valid=outcome.valid,
        replay_ended=outcome.ended,
        strict_target_recovered=strict_target_reached(outcome, target_stats),
        replay_failure_step=outcome.failure_step,
        replay_failure_type=outcome.failure_type,
        replay_stats=outcome.stats,
    )


def find_support_patch(
    *,
    history_lines: Sequence[str],
    failure_index: int,
    trace: Sequence[dict[str, Any]],
    target_lines: Sequence[str],
    target_stats: dict[str, Any],
    max_patch_length: int = 6,
    max_rollback: int = 6,
    beam_size: int = 16,
    max_proposal_evaluations: int = 128,
    verifier: TopoVerifier | None = None,
) -> CertifiedPatch | None:
    """Find a bounded patch using only candidates recorded in the failed trace.

    Target states are used only to test resynchronization and to independently
    certify the untouched target suffix. They never contribute proposal text.
    """

    verifier = verifier or TopoVerifier()
    target_signatures, _ = target_state_index(target_lines, verifier)
    selected_events = [
        event
        for event in trace
        if event.get("status") == "selected" and (event.get("selected") or {}).get("line")
    ]
    support_catalog = _unique_lines(
        line
        for event in selected_events
        for line in event_candidate_lines(event)
    )
    starts = range(failure_index, max(-1, failure_index - max_rollback - 1), -1)
    frontier: list[_PartialPatch] = []
    for repair_start in starts:
        replayed = replay_prefix(history_lines[:repair_start], verifier)
        if replayed is None:
            continue
        state, actions = replayed
        frontier.append(
            _PartialPatch(
                repair_start=repair_start,
                rollback_length=failure_index - repair_start + 1,
                state=state,
                actions=actions,
                patch_lines=(),
            )
        )

    proposal_evaluations = 0
    for _depth in range(1, max_patch_length + 1):
        next_frontier: list[_PartialPatch] = []
        successes: list[CertifiedPatch] = []
        for partial in sorted(frontier, key=lambda item: _partial_priority(item, target_stats)):
            event_index = min(
                failure_index,
                partial.repair_start + len(partial.patch_lines),
            )
            local_lines = (
                event_candidate_lines(selected_events[event_index])
                if 0 <= event_index < len(selected_events)
                else []
            )
            failure_lines = (
                event_candidate_lines(selected_events[failure_index])
                if 0 <= failure_index < len(selected_events)
                else []
            )
            candidates = _unique_lines(
                [
                    *local_lines,
                    *failure_lines,
                    *generic_repair_lines(partial.state, support_catalog),
                    *support_catalog,
                ]
            )
            for line in candidates:
                if proposal_evaluations >= max_proposal_evaluations:
                    break
                proposal_evaluations += 1
                action = parse_action_line(line)
                if action is None:
                    continue
                result = verifier.step(partial.state, action)
                if not result.valid or result.next_state is None:
                    continue
                updated = _PartialPatch(
                    repair_start=partial.repair_start,
                    rollback_length=partial.rollback_length,
                    state=result.next_state,
                    actions=[*partial.actions, action],
                    patch_lines=(*partial.patch_lines, line),
                )
                resume_indices = target_signatures.get(state_signature(updated.state), [])
                for resume_index in sorted(resume_indices, reverse=True):
                    certified = _certify(
                        source="support_only",
                        history_lines=history_lines,
                        target_lines=target_lines,
                        target_stats=target_stats,
                        repair_start=updated.repair_start,
                        failure_index=failure_index,
                        patch_lines=updated.patch_lines,
                        target_resume_index=resume_index,
                        proposal_evaluations=proposal_evaluations,
                        verifier=verifier,
                    )
                    if certified.strict_target_recovered:
                        successes.append(certified)
                        break
                next_frontier.append(updated)
            if proposal_evaluations >= max_proposal_evaluations:
                break
        if successes:
            return min(
                successes,
                key=lambda patch: (
                    len(patch.patch_lines),
                    patch.rollback_length,
                    -patch.target_resume_index,
                ),
            )
        deduplicated: dict[str, _PartialPatch] = {}
        for partial in sorted(next_frontier, key=lambda item: _partial_priority(item, target_stats)):
            signature = state_signature(partial.state)
            deduplicated.setdefault(signature, partial)
        frontier = list(deduplicated.values())[:beam_size]
        if not frontier or proposal_evaluations >= max_proposal_evaluations:
            break
    return None


def find_oracle_template_patch(
    *,
    history_lines: Sequence[str],
    failure_index: int,
    target_lines: Sequence[str],
    target_stats: dict[str, Any],
    max_patch_length: int = 6,
    max_rollback: int = 6,
    max_proposal_evaluations: int = 128,
    allowed_repair_starts: Iterable[int] | None = None,
    source: str = "oracle_template",
    verifier: TopoVerifier | None = None,
) -> CertifiedPatch | None:
    """Search bounded contiguous target templates as an explicit oracle bound.

    ``allowed_repair_starts`` restricts the editable locus while leaving the
    patch vocabulary, proposal budget, and certification rule unchanged. This
    supports localization attribution; target templates remain an oracle
    capability bound and are not an autonomous repair method.
    """

    verifier = verifier or TopoVerifier()
    target_signatures, target_actions = target_state_index(target_lines, verifier)
    attempts: list[tuple[int, int, int]] = []
    if allowed_repair_starts is None:
        repair_starts = list(
            range(
                failure_index,
                max(-1, failure_index - max_rollback - 1),
                -1,
            )
        )
    else:
        repair_starts = sorted(
            {
                int(index)
                for index in allowed_repair_starts
                if 0 <= int(index) <= failure_index
            },
            reverse=True,
        )
    for repair_start in repair_starts:
        low = max(0, repair_start - max_rollback)
        high = min(len(target_actions), repair_start + max_rollback + 1)
        for target_start in range(low, high):
            for patch_length in range(1, max_patch_length + 1):
                if target_start + patch_length > len(target_actions):
                    break
                attempts.append((repair_start, target_start, patch_length))
    attempts.sort(
        key=lambda item: (
            item[2],
            failure_index - item[0],
            abs(item[1] - item[0]),
            item[1],
        )
    )

    proposal_evaluations = 0
    successes: list[CertifiedPatch] = []
    prefix_cache = {
        repair_start: replay_prefix(history_lines[:repair_start], verifier)
        for repair_start in repair_starts
    }
    for repair_start, target_start, patch_length in attempts:
        replayed = prefix_cache.get(repair_start)
        if replayed is None:
            continue
        state, _ = replayed
        patch = target_actions[target_start : target_start + patch_length]
        valid = True
        for action in patch:
            if proposal_evaluations >= max_proposal_evaluations:
                valid = False
                break
            proposal_evaluations += 1
            result = verifier.step(state, action)
            if not result.valid or result.next_state is None:
                valid = False
                break
            state = result.next_state
        if not valid:
            if proposal_evaluations >= max_proposal_evaluations:
                break
            continue
        resume_index = target_start + patch_length
        if resume_index not in target_signatures.get(state_signature(state), []):
            continue
        patch_lines = target_lines[target_start:resume_index]
        certified = _certify(
            source=source,
            history_lines=history_lines,
            target_lines=target_lines,
            target_stats=target_stats,
            repair_start=repair_start,
            failure_index=failure_index,
            patch_lines=patch_lines,
            target_resume_index=resume_index,
            proposal_evaluations=proposal_evaluations,
            verifier=verifier,
        )
        if certified.strict_target_recovered:
            successes.append(certified)
            break
    return min(
        successes,
        key=lambda patch: (
            len(patch.patch_lines),
            patch.rollback_length,
            -patch.target_resume_index,
        ),
    ) if successes else None


def find_minimum_oracle_locus_patch(
    *,
    history_lines: Sequence[str],
    failure_index: int,
    target_lines: Sequence[str],
    target_stats: dict[str, Any],
    max_patch_length: int = 6,
    max_transition_evaluations: int = 8192,
    allowed_repair_starts: Iterable[int] | None = None,
    verifier: TopoVerifier | None = None,
) -> CertifiedPatch | None:
    """Find a minimum target-template resynchronization as an oracle bound.

    Unlike the budget-matched patcher, this routine caches each incremental
    target-template transition. Its budget counts unique verifier transitions,
    so it is reported only as an upper-bound localization oracle.
    """

    verifier = verifier or TopoVerifier()
    target_signatures, target_actions = target_state_index(target_lines, verifier)
    if allowed_repair_starts is None:
        repair_starts = list(range(failure_index, -1, -1))
    else:
        repair_starts = sorted(
            {
                int(index)
                for index in allowed_repair_starts
                if 0 <= int(index) <= failure_index
            },
            reverse=True,
        )
    prefix_cache = {
        repair_start: replay_prefix(history_lines[:repair_start], verifier)
        for repair_start in repair_starts
    }
    attempts: list[tuple[int, int, int]] = []
    for repair_start in repair_starts:
        low = max(0, repair_start - 6)
        high = min(len(target_actions), repair_start + 7)
        for target_start in range(low, high):
            for patch_length in range(1, max_patch_length + 1):
                if target_start + patch_length > len(target_actions):
                    break
                attempts.append((repair_start, target_start, patch_length))
    attempts.sort(
        key=lambda item: (
            item[2],
            failure_index - item[0],
            abs(item[1] - item[0]),
            item[1],
        )
    )

    state_cache: dict[tuple[int, int, int], VerificationState | None] = {}
    transition_evaluations = 0
    for repair_start, target_start, patch_length in attempts:
        if transition_evaluations >= max_transition_evaluations:
            break
        if patch_length == 1:
            replayed = prefix_cache.get(repair_start)
            state = replayed[0] if replayed is not None else None
        else:
            state = state_cache.get((repair_start, target_start, patch_length - 1))
        if state is None:
            state_cache[(repair_start, target_start, patch_length)] = None
            continue
        action_index = target_start + patch_length - 1
        transition_evaluations += 1
        result = verifier.step(state, target_actions[action_index])
        if not result.valid or result.next_state is None:
            state_cache[(repair_start, target_start, patch_length)] = None
            continue
        state = result.next_state
        state_cache[(repair_start, target_start, patch_length)] = state
        resume_index = target_start + patch_length
        if resume_index not in target_signatures.get(state_signature(state), []):
            continue
        patch_lines = target_lines[target_start:resume_index]
        certified = _certify(
            source="oracle_minimum_locus_upper_bound",
            history_lines=history_lines,
            target_lines=target_lines,
            target_stats=target_stats,
            repair_start=repair_start,
            failure_index=failure_index,
            patch_lines=patch_lines,
            target_resume_index=resume_index,
            proposal_evaluations=transition_evaluations,
            verifier=verifier,
        )
        if certified.strict_target_recovered:
            return certified
    return None
