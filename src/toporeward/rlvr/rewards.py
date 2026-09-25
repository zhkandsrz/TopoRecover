from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

from ..actions import (
    Action,
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
    action_to_dict,
    action_to_text,
    actions_from_dicts,
    actions_to_dicts,
)
from ..lm.parsing import parse_action_line
from ..prm.topoprm_registry import TopoPRMModel, load_topoprm_model, topoprm_predict_class, topoprm_valid_probability
from ..verifier import TopoVerifier


@dataclass(frozen=True)
class RewardWeights:
    format: float = 0.5
    process_topo: float = 1.0
    final_executable: float = 2.0
    target_complexity: float = 1.5
    target_progress: float = 0.8
    geometry_quality: float = 0.5
    overbuild_penalty: float = 0.8
    collapse_penalty: float = 1.5
    premature_terminal_penalty: float = 1.25
    invalid_transition_penalty: float = 1.0


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    format_reward: float
    process_topo_reward: float
    final_executable_reward: float
    target_complexity_reward: float
    target_progress_reward: float
    geometry_quality_reward: float
    overbuild_penalty: float
    collapse_penalty: float
    premature_terminal_penalty: float
    invalid_transition_penalty: float
    parsed_actions: int
    valid_actions: int
    ended: bool
    first_failure_type: str | None
    generated_stats: dict[str, int]
    target_stats: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionRewardWeights:
    format: float = 0.8
    valid_transition: float = 1.5
    target_action: float = 2.0
    target_action_type: float = 0.4
    target_progress: float = 1.4
    overbuild_penalty: float = 0.8
    premature_terminal_penalty: float = 2.5
    invalid_transition_penalty: float = 2.0
    extra_text_penalty: float = 0.6


@dataclass(frozen=True)
class ActionRewardBreakdown:
    total: float
    format_reward: float
    valid_transition_reward: float
    target_action_reward: float
    target_action_type_reward: float
    target_progress_reward: float
    overbuild_penalty: float
    premature_terminal_penalty: float
    invalid_transition_penalty: float
    extra_text_penalty: float
    parsed_actions: int
    valid_actions: int
    first_failure_type: str | None
    selected_action: str | None
    target_action: str | None
    generated_stats: dict[str, int]
    target_stats: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessActionRewardWeights:
    format: float = 0.8
    valid_transition: float = 1.2
    target_action: float = 1.8
    target_action_type: float = 0.35
    target_progress: float = 1.0
    profile_reach: float = 0.75
    hole_reach: float = 0.9
    safe_inner_progress: float = 0.8
    extrude_frontier: float = 1.4
    extrude_phase_reach: float = 0.8
    unsafe_inner_progress_penalty: float = 2.6
    wrong_extrude_penalty: float = 1.4
    extra_startface_penalty: float = 1.2
    premature_phase_shift_penalty: float = 1.1
    overbuild_penalty: float = 0.8
    premature_terminal_penalty: float = 2.2
    invalid_transition_penalty: float = 2.0
    extra_text_penalty: float = 0.7


@dataclass(frozen=True)
class ProcessActionRewardBreakdown:
    total: float
    format_reward: float
    valid_transition_reward: float
    target_action_reward: float
    target_action_type_reward: float
    target_progress_reward: float
    profile_reach_reward: float
    hole_reach_reward: float
    safe_inner_progress_reward: float
    extrude_frontier_reward: float
    extrude_phase_reach_reward: float
    unsafe_inner_progress_penalty: float
    wrong_extrude_penalty: float
    extra_startface_penalty: float
    premature_phase_shift_penalty: float
    overbuild_penalty: float
    premature_terminal_penalty: float
    invalid_transition_penalty: float
    extra_text_penalty: float
    parsed_actions: int
    valid_actions: int
    first_failure_type: str | None
    selected_action: str | None
    target_action: str | None
    generated_stats: dict[str, int]
    target_stats: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TopoPRMActionRewardWeights:
    format: float = 0.5
    verifier_valid: float = 0.8
    topoprm_valid: float = 2.5
    target_action: float = 1.4
    target_action_type: float = 0.35
    target_progress: float = 1.0
    overbuild_penalty: float = 0.8
    premature_terminal_penalty: float = 2.0
    topoprm_invalid_penalty: float = 2.5
    extra_text_penalty: float = 0.6


@dataclass(frozen=True)
class TopoPRMActionRewardBreakdown:
    total: float
    format_reward: float
    verifier_valid_reward: float
    topoprm_valid_probability: float
    target_action_reward: float
    target_action_type_reward: float
    target_progress_reward: float
    overbuild_penalty: float
    premature_terminal_penalty: float
    topoprm_invalid_penalty: float
    extra_text_penalty: float
    parsed_actions: int
    valid_actions: int
    verifier_failure_type: str | None
    topoprm_predicted_class: str | None
    selected_action: str | None
    target_action: str | None
    generated_stats: dict[str, int]
    target_stats: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TOPOPRM_CACHE: tuple[str, TopoPRMModel] | None = None


def _default_topoprm_path() -> Path:
    return Path(__file__).resolve().parents[3] / "checkpoints" / "topoprm_v1_linear.json"


def _load_topoprm_from_env() -> TopoPRMModel:
    global _TOPOPRM_CACHE
    path = Path(os.environ.get("TOPOREWARD_TOPOPRM_PATH", _default_topoprm_path()))
    cache_key = str(path.resolve())
    if _TOPOPRM_CACHE is None or _TOPOPRM_CACHE[0] != cache_key:
        if not path.exists():
            raise FileNotFoundError(
                f"TopoPRM checkpoint not found: {path}. "
                "Pass --topoprm to train_grpo_smoke.py or set TOPOREWARD_TOPOPRM_PATH."
            )
        _TOPOPRM_CACHE = (cache_key, load_topoprm_model(path))
    return _TOPOPRM_CACHE[1]


def structure_stats(actions: list[Action] | tuple[Action, ...]) -> dict[str, int]:
    return {
        "action_count": len(actions),
        "profile_count": sum(isinstance(action, RegisterProfile) for action in actions),
        "hole_count": sum(isinstance(action, StartLoop) and action.kind == "inner" for action in actions),
        "circle_hole_count": sum(isinstance(action, AddCircle) for action in actions),
        "extrude_count": sum(isinstance(action, Extrude) for action in actions),
    }


def _normalize_target_stats(target_stats: dict[str, Any] | None) -> dict[str, int]:
    target_stats = target_stats or {}
    return {
        "action_count": int(target_stats.get("action_count", 0) or 0),
        "profile_count": int(target_stats.get("profile_count", 0) or 0),
        "hole_count": int(target_stats.get("hole_count", 0) or 0),
        "circle_hole_count": int(target_stats.get("circle_hole_count", 0) or 0),
        "extrude_count": int(target_stats.get("extrude_count", 0) or 0),
    }


def _parse_completion(completion: str) -> tuple[list[Action], int]:
    actions: list[Action] = []
    invalid_lines = 0
    for raw_line in completion.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        action = parse_action_line(line)
        if action is None:
            invalid_lines += 1
            continue
        actions.append(action)
    return actions, invalid_lines


def _replay_prefix(prefix_actions: list[Action]) -> Any | None:
    verifier = TopoVerifier()
    state = verifier.initial_state()
    for action in prefix_actions:
        result = verifier.step(state, action)
        if not result.valid:
            return None
        assert result.next_state is not None
        state = result.next_state
    return state


def _replay_completion(prefix_actions: list[Action], completion_actions: list[Action]) -> tuple[list[Action], bool, str | None]:
    verifier = TopoVerifier()
    state = _replay_prefix(prefix_actions)
    if state is None:
        return [], False, "invalid_prefix"

    valid_actions: list[Action] = []
    for action in completion_actions:
        result = verifier.step(state, action)
        if not result.valid:
            return valid_actions, False, result.failure_type
        assert result.next_state is not None
        state = result.next_state
        valid_actions.append(action)
    return valid_actions, state.ended, None


def _format_reward(parsed_actions: int, invalid_lines: int) -> float:
    total_lines = parsed_actions + invalid_lines
    if total_lines == 0:
        return 0.0
    return parsed_actions / total_lines


def _process_reward(valid_actions: int, parsed_actions: int, ended: bool) -> float:
    if parsed_actions == 0:
        return 0.0
    reward = valid_actions / parsed_actions
    if ended and valid_actions == parsed_actions:
        reward = min(1.0, reward + 0.1)
    return reward


def _has_structural_target(target_stats: dict[str, int]) -> bool:
    return any(target_stats.get(key, 0) > 0 for key in ["profile_count", "hole_count", "circle_hole_count", "extrude_count"])


def _target_complexity_reward(generated_stats: dict[str, int], target_stats: dict[str, int]) -> float:
    keys = ["profile_count", "hole_count", "circle_hole_count", "extrude_count"]
    scores = []
    for key in keys:
        target = target_stats.get(key, 0)
        generated = generated_stats.get(key, 0)
        if target <= 0:
            scores.append(1.0 if generated == 0 else 0.6)
        else:
            scores.append(min(generated / target, 1.0))
    return sum(scores) / len(scores)


def _target_progress_reward(
    prefix_stats: dict[str, int],
    generated_stats: dict[str, int],
    target_stats: dict[str, int],
) -> float:
    keys = ["profile_count", "hole_count", "circle_hole_count", "extrude_count"]
    reward = 0.0
    for key in keys:
        target = target_stats.get(key, 0)
        if target <= 0:
            continue
        before = min(prefix_stats.get(key, 0) / target, 1.0)
        after = min(generated_stats.get(key, 0) / target, 1.0)
        reward += max(0.0, after - before)
    return reward


def _geometry_quality_reward(generated_stats: dict[str, int], target_stats: dict[str, int]) -> float:
    target_actions = max(target_stats.get("action_count", 0), 1)
    generated_actions = generated_stats.get("action_count", 0)
    length_score = max(0.0, 1.0 - min(abs(generated_actions - target_actions) / target_actions, 1.0))

    target_extrudes = max(target_stats.get("extrude_count", 0), 1)
    generated_extrudes = generated_stats.get("extrude_count", 0)
    repeated_extrude_overflow = max(0, generated_extrudes - target_extrudes)
    repeated_extrude_penalty = min(1.0, repeated_extrude_overflow / max(target_extrudes + 3, 1))
    return max(0.0, length_score - repeated_extrude_penalty)


def _overbuild_penalty(generated_stats: dict[str, int], target_stats: dict[str, int]) -> float:
    keys = ["profile_count", "hole_count", "circle_hole_count", "extrude_count"]
    penalty = 0.0
    for key in keys:
        target = target_stats.get(key, 0)
        generated = generated_stats.get(key, 0)
        if target <= 0:
            penalty += 1.0 if generated > 0 else 0.0
        else:
            penalty += max(0.0, (generated - target) / target)
    return min(1.0, penalty)


def _final_executable_reward(ended: bool, target_stats: dict[str, int], complexity_value: float, overbuild_value: float) -> float:
    if not ended:
        return 0.0
    if not _has_structural_target(target_stats):
        return 1.0
    if complexity_value >= 1.0 and overbuild_value <= 0.0:
        return 1.0
    return 0.25


def _collapse_penalty(generated_stats: dict[str, int], target_stats: dict[str, int], ended: bool) -> float:
    if not ended:
        return 0.0
    complexity = _target_complexity_reward(generated_stats, target_stats)
    target_actions = max(target_stats.get("action_count", 0), 1)
    generated_actions = generated_stats.get("action_count", 0)
    too_short = max(0.0, (0.75 * target_actions - generated_actions) / target_actions)
    under_complexity = max(0.0, 1.0 - complexity)
    return min(1.0, under_complexity + too_short)


def _premature_terminal_penalty(
    completion_actions: list[Action],
    generated_stats: dict[str, int],
    target_stats: dict[str, int],
) -> float:
    if not completion_actions:
        return 0.0
    last_action = completion_actions[-1]
    complexity_gap = max(0.0, 1.0 - _target_complexity_reward(generated_stats, target_stats))
    if complexity_gap <= 0.0:
        return 0.0

    if isinstance(last_action, End):
        target_actions = max(target_stats.get("action_count", 0), 1)
        generated_actions = generated_stats.get("action_count", 0)
        too_short = max(0.0, (0.75 * target_actions - generated_actions) / target_actions)
        return min(1.0, complexity_gap + too_short)
    if isinstance(last_action, EndSketch):
        unmet_sketch_complexity = (
            generated_stats.get("profile_count", 0) < target_stats.get("profile_count", 0)
            or generated_stats.get("hole_count", 0) < target_stats.get("hole_count", 0)
            or generated_stats.get("circle_hole_count", 0) < target_stats.get("circle_hole_count", 0)
        )
        return min(1.0, complexity_gap) if unmet_sketch_complexity else 0.0
    if isinstance(last_action, EndFace):
        unmet_face_complexity = (
            generated_stats.get("hole_count", 0) < target_stats.get("hole_count", 0)
            or generated_stats.get("circle_hole_count", 0) < target_stats.get("circle_hole_count", 0)
        )
        return min(1.0, complexity_gap) if unmet_face_complexity else 0.0
    return 0.0


def _coerce_prefix_actions(prefix_actions: list[Action] | list[dict[str, Any]] | None) -> list[Action]:
    if not prefix_actions:
        return []
    first = prefix_actions[0]
    if isinstance(first, dict):
        return actions_from_dicts(prefix_actions)  # type: ignore[arg-type]
    return list(prefix_actions)  # type: ignore[arg-type]


def _coerce_prefix_dicts(prefix_actions: list[Action] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not prefix_actions:
        return []
    first = prefix_actions[0]
    if isinstance(first, dict):
        return list(prefix_actions)  # type: ignore[return-value]
    return actions_to_dicts(list(prefix_actions))  # type: ignore[arg-type]


def score_completion(
    completion: str,
    prefix_actions: list[Action] | list[dict[str, Any]] | None = None,
    target_stats: dict[str, Any] | None = None,
    weights: RewardWeights = RewardWeights(),
) -> RewardBreakdown:
    prefix = _coerce_prefix_actions(prefix_actions)
    target = _normalize_target_stats(target_stats)
    parsed_actions, invalid_lines = _parse_completion(completion)
    valid_completion_actions, ended, first_failure_type = _replay_completion(prefix, parsed_actions)
    full_valid_actions = prefix + valid_completion_actions
    prefix_stats = structure_stats(prefix)
    generated = structure_stats(full_valid_actions)

    format_value = _format_reward(len(parsed_actions), invalid_lines)
    process_value = _process_reward(len(valid_completion_actions), len(parsed_actions), ended)
    complexity_value = _target_complexity_reward(generated, target)
    progress_value = _target_progress_reward(prefix_stats, generated, target)
    geometry_value = _geometry_quality_reward(generated, target)
    overbuild_value = _overbuild_penalty(generated, target)
    final_value = _final_executable_reward(ended, target, complexity_value, overbuild_value)
    collapse_value = _collapse_penalty(generated, target, ended)
    premature_terminal_value = _premature_terminal_penalty(valid_completion_actions, generated, target)
    invalid_transition_value = 1.0 if first_failure_type is not None and len(parsed_actions) > len(valid_completion_actions) else 0.0

    total = (
        weights.format * format_value
        + weights.process_topo * process_value
        + weights.final_executable * final_value
        + weights.target_complexity * complexity_value
        + weights.target_progress * progress_value
        + weights.geometry_quality * geometry_value
        - weights.overbuild_penalty * overbuild_value
        - weights.collapse_penalty * collapse_value
        - weights.premature_terminal_penalty * premature_terminal_value
        - weights.invalid_transition_penalty * invalid_transition_value
    )
    if invalid_lines:
        total -= min(1.0, invalid_lines * 0.2)

    return RewardBreakdown(
        total=float(total),
        format_reward=float(format_value),
        process_topo_reward=float(process_value),
        final_executable_reward=float(final_value),
        target_complexity_reward=float(complexity_value),
        target_progress_reward=float(progress_value),
        geometry_quality_reward=float(geometry_value),
        overbuild_penalty=float(overbuild_value),
        collapse_penalty=float(collapse_value),
        premature_terminal_penalty=float(premature_terminal_value),
        invalid_transition_penalty=float(invalid_transition_value),
        parsed_actions=len(parsed_actions),
        valid_actions=len(valid_completion_actions),
        ended=ended,
        first_failure_type=first_failure_type,
        generated_stats=generated,
        target_stats=target,
    )


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _first_nonempty_line(text: str | None) -> str | None:
    if text is None:
        return None
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if line:
            return line
    return None


def _same_action_type(left: Action | None, right: Action | None) -> bool:
    if left is None or right is None:
        return False
    return type(left) is type(right)


def _same_extrude(left: Extrude, right: Extrude) -> bool:
    return (
        left.profile_id == right.profile_id
        and abs(float(left.depth) - float(right.depth)) <= 1e-6
        and left.op == right.op
    )


def _is_outer_loop(action: Action | None) -> bool:
    return isinstance(action, StartLoop) and action.kind == "outer"


def _is_inner_loop(action: Action | None) -> bool:
    return isinstance(action, StartLoop) and action.kind == "inner"


def _is_profile_reach_action(action: Action | None, *, inside_inner_loop: bool = False) -> bool:
    if action is None:
        return False
    if inside_inner_loop:
        return False
    if _is_outer_loop(action):
        return True
    return isinstance(action, (StartFace, AddLine, EndLoop, EndFace, RegisterProfile))


def _is_hole_reach_action(action: Action | None, *, inside_inner_loop: bool = False) -> bool:
    if action is None:
        return False
    if _is_inner_loop(action):
        return True
    if isinstance(action, AddCircle):
        return True
    return inside_inner_loop and isinstance(action, (AddLine, EndLoop))


def _matching_reach_action(selected: Action | None, target: Action | None) -> float:
    if selected is None or target is None:
        return 0.0
    if action_to_text(selected) == action_to_text(target):
        return 1.0
    if isinstance(selected, StartLoop) and isinstance(target, StartLoop) and selected.kind == target.kind:
        return 0.6
    if type(selected) is type(target):
        return 0.45
    return 0.0


def _layout_feasibility_for_prefix(
    prefix_actions: list[Action],
    target_stats: dict[str, Any] | None,
    state: Any,
) -> dict[str, Any]:
    try:
        from ..topoplan_v2 import layout_feasibility_state
    except ImportError:
        return {"safe_inner_progress": False, "unsafe_inner_progress": False}
    return layout_feasibility_state(target_stats, prefix_actions, state=state)


def _wrong_extrude_penalty(selected: Action | None, target: Action | None) -> float:
    if not isinstance(selected, Extrude) or not isinstance(target, Extrude):
        return 0.0
    if _same_extrude(selected, target):
        return 0.0
    penalty = 0.0
    if selected.profile_id != target.profile_id:
        penalty += 0.5
    if abs(float(selected.depth) - float(target.depth)) > 1e-6:
        penalty += 0.3
    if selected.op != target.op:
        penalty += 0.2
    return min(1.0, penalty)


def score_action_completion(
    completion: str,
    prefix_actions: list[Action] | list[dict[str, Any]] | None = None,
    target_stats: dict[str, Any] | None = None,
    target_completion: str | None = None,
    weights: ActionRewardWeights = ActionRewardWeights(),
) -> ActionRewardBreakdown:
    """Score a GRPO completion as exactly one next CAD action.

    This reward is intentionally sharper than the full-completion reward. It is
    for process-level GRPO where each generated sample should choose the next
    topology-building action from the current verifier state.
    """

    prefix = _coerce_prefix_actions(prefix_actions)
    target = _normalize_target_stats(target_stats)
    parsed_actions, invalid_lines = _parse_completion(completion)
    selected_action = parsed_actions[0] if parsed_actions else None
    selected_line = action_to_text(selected_action) if selected_action is not None else None

    target_line = _first_nonempty_line(target_completion)
    target_action = parse_action_line(target_line) if target_line is not None else None
    canonical_target_line = action_to_text(target_action) if target_action is not None else target_line

    verifier = TopoVerifier()
    state = _replay_prefix(prefix)
    valid_transition = False
    first_failure_type: str | None = None
    valid_action_list: list[Action] = []
    if state is None:
        first_failure_type = "invalid_prefix"
    elif selected_action is None:
        first_failure_type = "parse_error" if invalid_lines else "empty_completion"
    else:
        result = verifier.step(state, selected_action)
        if result.valid:
            valid_transition = True
            valid_action_list.append(selected_action)
        else:
            first_failure_type = result.failure_type

    prefix_stats = structure_stats(prefix)
    generated = structure_stats(prefix + valid_action_list)

    extra_action_count = max(0, len(parsed_actions) - 1)
    extra_text_value = min(1.0, 0.5 * extra_action_count + 0.5 * invalid_lines)
    format_value = 1.0 if selected_action is not None and extra_action_count == 0 and invalid_lines == 0 else 0.0
    valid_value = 1.0 if valid_transition else 0.0
    target_action_value = (
        1.0
        if valid_transition
        and selected_line is not None
        and canonical_target_line is not None
        and selected_line == canonical_target_line
        else 0.0
    )
    target_action_type_value = 1.0 if valid_transition and _same_action_type(selected_action, target_action) else 0.0
    progress_value = _target_progress_reward(prefix_stats, generated, target) if valid_transition else 0.0
    overbuild_value = _overbuild_penalty(generated, target) if valid_transition else 0.0
    premature_terminal_value = (
        _premature_terminal_penalty(valid_action_list, generated, target) if valid_transition else 0.0
    )
    invalid_transition_value = 0.0 if valid_transition else 1.0

    total = (
        weights.format * format_value
        + weights.valid_transition * valid_value
        + weights.target_action * target_action_value
        + weights.target_action_type * target_action_type_value
        + weights.target_progress * progress_value
        - weights.overbuild_penalty * overbuild_value
        - weights.premature_terminal_penalty * premature_terminal_value
        - weights.invalid_transition_penalty * invalid_transition_value
        - weights.extra_text_penalty * extra_text_value
    )

    return ActionRewardBreakdown(
        total=float(total),
        format_reward=float(format_value),
        valid_transition_reward=float(valid_value),
        target_action_reward=float(target_action_value),
        target_action_type_reward=float(target_action_type_value),
        target_progress_reward=float(progress_value),
        overbuild_penalty=float(overbuild_value),
        premature_terminal_penalty=float(premature_terminal_value),
        invalid_transition_penalty=float(invalid_transition_value),
        extra_text_penalty=float(extra_text_value),
        parsed_actions=len(parsed_actions),
        valid_actions=len(valid_action_list),
        first_failure_type=first_failure_type,
        selected_action=selected_line,
        target_action=canonical_target_line,
        generated_stats=generated,
        target_stats=target,
    )


def score_process_action_completion(
    completion: str,
    prefix_actions: list[Action] | list[dict[str, Any]] | None = None,
    target_stats: dict[str, Any] | None = None,
    target_completion: str | None = None,
    weights: ProcessActionRewardWeights = ProcessActionRewardWeights(),
) -> ProcessActionRewardBreakdown:
    """Score one next-action sample with phase-aware process shaping.

    This variant keeps the exact next-action signal but adds the bounded
    process rewards exposed by the rollout diagnosis: staying in extrude
    scheduling once the sketch is closed, avoiding extra StartFace when
    profiles/holes are already sufficient, and penalizing wrong Extrude
    references/parameters.
    """

    prefix = _coerce_prefix_actions(prefix_actions)
    target = _normalize_target_stats(target_stats)
    parsed_actions, invalid_lines = _parse_completion(completion)
    selected_action = parsed_actions[0] if parsed_actions else None
    selected_line = action_to_text(selected_action) if selected_action is not None else None

    target_line = _first_nonempty_line(target_completion)
    target_action = parse_action_line(target_line) if target_line is not None else None
    canonical_target_line = action_to_text(target_action) if target_action is not None else target_line

    state = _replay_prefix(prefix)
    valid_transition = False
    first_failure_type: str | None = None
    valid_action_list: list[Action] = []
    if state is None:
        first_failure_type = "invalid_prefix"
    elif selected_action is None:
        first_failure_type = "parse_error" if invalid_lines else "empty_completion"
    else:
        result = TopoVerifier().step(state, selected_action)
        if result.valid:
            valid_transition = True
            valid_action_list.append(selected_action)
        else:
            first_failure_type = result.failure_type

    prefix_stats = structure_stats(prefix)
    generated = structure_stats(prefix + valid_action_list)
    last_prefix_action = prefix[-1] if prefix else None

    extra_action_count = max(0, len(parsed_actions) - 1)
    extra_text_value = min(1.0, 0.5 * extra_action_count + 0.5 * invalid_lines)
    format_value = 1.0 if selected_action is not None and extra_action_count == 0 and invalid_lines == 0 else 0.0
    valid_value = 1.0 if valid_transition else 0.0
    target_action_value = (
        1.0
        if valid_transition
        and selected_line is not None
        and canonical_target_line is not None
        and selected_line == canonical_target_line
        else 0.0
    )
    target_action_type_value = 1.0 if valid_transition and _same_action_type(selected_action, target_action) else 0.0
    progress_value = _target_progress_reward(prefix_stats, generated, target) if valid_transition else 0.0

    remaining_profiles = max(0, target.get("profile_count", 0) - prefix_stats.get("profile_count", 0))
    remaining_holes = max(0, target.get("hole_count", 0) - prefix_stats.get("hole_count", 0))
    remaining_extrudes = max(0, target.get("extrude_count", 0) - prefix_stats.get("extrude_count", 0))
    inside_inner_loop = bool(state is not None and state.current_loop is not None and state.current_loop.kind == "inner")
    profile_reach_value = (
        _matching_reach_action(selected_action, target_action)
        if valid_transition
        and remaining_profiles > 0
        and _is_profile_reach_action(target_action, inside_inner_loop=inside_inner_loop)
        else 0.0
    )
    layout_state = _layout_feasibility_for_prefix(prefix, target_stats, state)
    selected_starts_inner = _is_inner_loop(selected_action)
    unsafe_inner_progress_value = (
        1.0
        if valid_transition
        and selected_starts_inner
        and bool(layout_state.get("unsafe_inner_progress"))
        else 0.0
    )
    safe_inner_progress_value = (
        1.0
        if valid_transition
        and selected_starts_inner
        and bool(layout_state.get("safe_inner_progress"))
        else 0.0
    )
    if unsafe_inner_progress_value:
        target_action_value = 0.0
        target_action_type_value = 0.0
        progress_value = 0.0
    hole_reach_allowed = not selected_starts_inner or bool(layout_state.get("safe_inner_progress"))
    hole_reach_value = (
        _matching_reach_action(selected_action, target_action)
        if valid_transition
        and remaining_holes > 0
        and hole_reach_allowed
        and _is_hole_reach_action(target_action, inside_inner_loop=inside_inner_loop)
        else 0.0
    )

    at_extrude_frontier = isinstance(last_prefix_action, (EndSketch, Extrude))
    target_is_extrude = isinstance(target_action, Extrude)
    selected_is_exact_extrude = (
        isinstance(selected_action, Extrude)
        and isinstance(target_action, Extrude)
        and _same_extrude(selected_action, target_action)
    )
    extrude_frontier_value = 1.0 if at_extrude_frontier and target_is_extrude and selected_is_exact_extrude else 0.0

    profiles_done = prefix_stats.get("profile_count", 0) >= target.get("profile_count", 0)
    holes_done = prefix_stats.get("hole_count", 0) >= target.get("hole_count", 0)
    extrudes_remaining = prefix_stats.get("extrude_count", 0) < target.get("extrude_count", 0)
    should_stay_extrude_or_close = profiles_done and holes_done and extrudes_remaining

    extrude_phase_reach_value = (
        1.0
        if valid_transition
        and should_stay_extrude_or_close
        and isinstance(selected_action, (EndSketch, Extrude))
        else 0.0
    )
    extra_startface_value = (
        1.0
        if valid_transition
        and should_stay_extrude_or_close
        and isinstance(selected_action, StartFace)
        else 0.0
    )
    wrong_extrude_value = _wrong_extrude_penalty(selected_action, target_action) if valid_transition else 0.0
    premature_phase_shift_value = (
        1.0
        if valid_transition
        and (remaining_profiles > 0 or remaining_holes > 0)
        and remaining_extrudes > 0
        and isinstance(selected_action, (EndSketch, Extrude, End))
        else 0.0
    )
    overbuild_value = _overbuild_penalty(generated, target) if valid_transition else 0.0
    premature_terminal_value = (
        _premature_terminal_penalty(valid_action_list, generated, target) if valid_transition else 0.0
    )
    invalid_transition_value = 0.0 if valid_transition else 1.0

    total = (
        weights.format * format_value
        + weights.valid_transition * valid_value
        + weights.target_action * target_action_value
        + weights.target_action_type * target_action_type_value
        + weights.target_progress * progress_value
        + weights.profile_reach * profile_reach_value
        + weights.hole_reach * hole_reach_value
        + weights.safe_inner_progress * safe_inner_progress_value
        + weights.extrude_frontier * extrude_frontier_value
        + weights.extrude_phase_reach * extrude_phase_reach_value
        - weights.unsafe_inner_progress_penalty * unsafe_inner_progress_value
        - weights.wrong_extrude_penalty * wrong_extrude_value
        - weights.extra_startface_penalty * extra_startface_value
        - weights.premature_phase_shift_penalty * premature_phase_shift_value
        - weights.overbuild_penalty * overbuild_value
        - weights.premature_terminal_penalty * premature_terminal_value
        - weights.invalid_transition_penalty * invalid_transition_value
        - weights.extra_text_penalty * extra_text_value
    )

    return ProcessActionRewardBreakdown(
        total=float(total),
        format_reward=float(format_value),
        valid_transition_reward=float(valid_value),
        target_action_reward=float(target_action_value),
        target_action_type_reward=float(target_action_type_value),
        target_progress_reward=float(progress_value),
        profile_reach_reward=float(profile_reach_value),
        hole_reach_reward=float(hole_reach_value),
        safe_inner_progress_reward=float(safe_inner_progress_value),
        extrude_frontier_reward=float(extrude_frontier_value),
        extrude_phase_reach_reward=float(extrude_phase_reach_value),
        unsafe_inner_progress_penalty=float(unsafe_inner_progress_value),
        wrong_extrude_penalty=float(wrong_extrude_value),
        extra_startface_penalty=float(extra_startface_value),
        premature_phase_shift_penalty=float(premature_phase_shift_value),
        overbuild_penalty=float(overbuild_value),
        premature_terminal_penalty=float(premature_terminal_value),
        invalid_transition_penalty=float(invalid_transition_value),
        extra_text_penalty=float(extra_text_value),
        parsed_actions=len(parsed_actions),
        valid_actions=len(valid_action_list),
        first_failure_type=first_failure_type,
        selected_action=selected_line,
        target_action=canonical_target_line,
        generated_stats=generated,
        target_stats=target,
    )


def score_topoprm_action_completion(
    completion: str,
    prefix_actions: list[Action] | list[dict[str, Any]] | None = None,
    target_stats: dict[str, Any] | None = None,
    target_completion: str | None = None,
    topoprm: TopoPRMModel | None = None,
    weights: TopoPRMActionRewardWeights = TopoPRMActionRewardWeights(),
) -> TopoPRMActionRewardBreakdown:
    """Score one next-action sample with a learned TopoPRM process reward.

    The verifier still supplies the exact transition label for diagnostics, but
    the dense optimization signal is the learned TopoPRM valid probability. This
    is the local reward-v6 path for reducing policy mass on invalid actions.
    """

    prefix = _coerce_prefix_actions(prefix_actions)
    prefix_dicts = _coerce_prefix_dicts(prefix_actions)
    target = _normalize_target_stats(target_stats)
    parsed_actions, invalid_lines = _parse_completion(completion)
    selected_action = parsed_actions[0] if parsed_actions else None
    selected_line = action_to_text(selected_action) if selected_action is not None else None

    target_line = _first_nonempty_line(target_completion)
    target_action = parse_action_line(target_line) if target_line is not None else None
    canonical_target_line = action_to_text(target_action) if target_action is not None else target_line

    verifier = TopoVerifier()
    state = _replay_prefix(prefix)
    valid_transition = False
    verifier_failure_type: str | None = None
    valid_action_list: list[Action] = []
    if state is None:
        verifier_failure_type = "invalid_prefix"
    elif selected_action is None:
        verifier_failure_type = "parse_error" if invalid_lines else "empty_completion"
    else:
        result = verifier.step(state, selected_action)
        if result.valid:
            valid_transition = True
            valid_action_list.append(selected_action)
        else:
            verifier_failure_type = result.failure_type

    topoprm_model = topoprm or _load_topoprm_from_env()
    topoprm_valid_prob = 0.0
    topoprm_predicted_class: str | None = None
    if selected_action is not None:
        action_dict = action_to_dict(selected_action)
        topoprm_valid_prob = topoprm_valid_probability(topoprm_model, prefix_dicts, action_dict, target)
        topoprm_predicted_class = topoprm_predict_class(topoprm_model, prefix_dicts, action_dict, target)

    prefix_stats = structure_stats(prefix)
    generated = structure_stats(prefix + valid_action_list)

    extra_action_count = max(0, len(parsed_actions) - 1)
    extra_text_value = min(1.0, 0.5 * extra_action_count + 0.5 * invalid_lines)
    format_value = 1.0 if selected_action is not None and extra_action_count == 0 and invalid_lines == 0 else 0.0
    verifier_valid_value = 1.0 if valid_transition else 0.0
    target_action_value = (
        1.0
        if selected_line is not None
        and canonical_target_line is not None
        and selected_line == canonical_target_line
        else 0.0
    )
    target_action_type_value = 1.0 if _same_action_type(selected_action, target_action) else 0.0
    progress_value = _target_progress_reward(prefix_stats, generated, target) if valid_transition else 0.0
    overbuild_value = _overbuild_penalty(generated, target) if valid_transition else 0.0
    premature_terminal_value = (
        _premature_terminal_penalty(valid_action_list, generated, target) if valid_transition else 0.0
    )
    topoprm_invalid_value = 1.0 - topoprm_valid_prob

    total = (
        weights.format * format_value
        + weights.verifier_valid * verifier_valid_value
        + weights.topoprm_valid * topoprm_valid_prob
        + weights.target_action * target_action_value
        + weights.target_action_type * target_action_type_value
        + weights.target_progress * progress_value
        - weights.overbuild_penalty * overbuild_value
        - weights.premature_terminal_penalty * premature_terminal_value
        - weights.topoprm_invalid_penalty * topoprm_invalid_value
        - weights.extra_text_penalty * extra_text_value
    )

    return TopoPRMActionRewardBreakdown(
        total=float(total),
        format_reward=float(format_value),
        verifier_valid_reward=float(verifier_valid_value),
        topoprm_valid_probability=float(topoprm_valid_prob),
        target_action_reward=float(target_action_value),
        target_action_type_reward=float(target_action_type_value),
        target_progress_reward=float(progress_value),
        overbuild_penalty=float(overbuild_value),
        premature_terminal_penalty=float(premature_terminal_value),
        topoprm_invalid_penalty=float(topoprm_invalid_value),
        extra_text_penalty=float(extra_text_value),
        parsed_actions=len(parsed_actions),
        valid_actions=len(valid_action_list),
        verifier_failure_type=verifier_failure_type,
        topoprm_predicted_class=topoprm_predicted_class,
        selected_action=selected_line,
        target_action=canonical_target_line,
        generated_stats=generated,
        target_stats=target,
    )


def grpo_reward_function(
    prompts: list[Any],
    completions: list[Any],
    prefix_actions: list[Any] | None = None,
    target_stats: list[dict[str, Any]] | None = None,
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        prefix = prefix_actions[idx] if prefix_actions is not None else None
        target = target_stats[idx] if target_stats is not None else None
        breakdown = score_completion(_completion_to_text(completion), prefix_actions=prefix, target_stats=target)
        rewards.append(breakdown.total)
    return rewards


def grpo_action_reward_function(
    prompts: list[Any],
    completions: list[Any],
    prefix_actions: list[Any] | None = None,
    target_stats: list[dict[str, Any]] | None = None,
    target_completion: list[str] | None = None,
    completion: list[str] | None = None,
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    targets = target_completion if target_completion is not None else completion
    for idx, generated_completion in enumerate(completions):
        prefix = prefix_actions[idx] if prefix_actions is not None else None
        target = target_stats[idx] if target_stats is not None else None
        target_text = targets[idx] if targets is not None else None
        breakdown = score_action_completion(
            _completion_to_text(generated_completion),
            prefix_actions=prefix,
            target_stats=target,
            target_completion=target_text,
        )
        rewards.append(breakdown.total)
    return rewards


def grpo_topoprm_action_reward_function(
    prompts: list[Any],
    completions: list[Any],
    prefix_actions: list[Any] | None = None,
    target_stats: list[dict[str, Any]] | None = None,
    target_completion: list[str] | None = None,
    completion: list[str] | None = None,
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    targets = target_completion if target_completion is not None else completion
    topoprm = _load_topoprm_from_env()
    for idx, generated_completion in enumerate(completions):
        prefix = prefix_actions[idx] if prefix_actions is not None else None
        target = target_stats[idx] if target_stats is not None else None
        target_text = targets[idx] if targets is not None else None
        breakdown = score_topoprm_action_completion(
            _completion_to_text(generated_completion),
            prefix_actions=prefix,
            target_stats=target,
            target_completion=target_text,
            topoprm=topoprm,
        )
        rewards.append(breakdown.total)
    return rewards


def grpo_process_action_reward_function(
    prompts: list[Any],
    completions: list[Any],
    prefix_actions: list[Any] | None = None,
    target_stats: list[dict[str, Any]] | None = None,
    target_completion: list[str] | None = None,
    completion: list[str] | None = None,
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    targets = target_completion if target_completion is not None else completion
    for idx, generated_completion in enumerate(completions):
        prefix = prefix_actions[idx] if prefix_actions is not None else None
        target = target_stats[idx] if target_stats is not None else None
        target_text = targets[idx] if targets is not None else None
        breakdown = score_process_action_completion(
            _completion_to_text(generated_completion),
            prefix_actions=prefix,
            target_stats=target,
            target_completion=target_text,
        )
        rewards.append(breakdown.total)
    return rewards
