from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..actions import (
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
    action_from_dict,
    actions_from_dicts,
)
from ..verifier import TopoVerifier
from ..verifier.failures import REPAIR_HINTS
from ..verifier.geometry import EPS, close_points, distance, on_segment, point_in_polygon, polygon_segments


FeatureVector = dict[str, float]
TOPOPRM_CLASSES = ["valid", *REPAIR_HINTS.keys()]


def _add(features: FeatureVector, name: str, value: float = 1.0) -> None:
    features[name] = features.get(name, 0.0) + value


def extract_features(prefix_dicts: list[dict[str, Any]], action_dict: dict[str, Any]) -> FeatureVector:
    prefix = actions_from_dicts(prefix_dicts)
    action = action_from_dict(action_dict)
    features: FeatureVector = {"bias": 1.0}
    _add(features, f"action={action.type}")
    _add(features, f"prefix_len_bucket={min(len(prefix) // 4, 8)}")
    if prefix:
        _add(features, f"last_action={prefix[-1].type}")

    registered_profiles = {item.profile_id for item in prefix if isinstance(item, RegisterProfile)}
    previous_lines = [item for item in prefix if isinstance(item, AddLine)]
    last_line = previous_lines[-1] if previous_lines else None

    if isinstance(action, AddLine):
        if distance(action.start, action.end) <= 1e-7:
            _add(features, "line_zero_length")
        if last_line is not None:
            if close_points(action.start, last_line.end):
                _add(features, "line_start_matches_previous_tail")
            else:
                _add(features, "line_start_mismatch_previous_tail")
    elif isinstance(action, EndLoop):
        loop_start = None
        loop_tail = None
        for item in reversed(prefix):
            if isinstance(item, AddLine) and loop_tail is None:
                loop_tail = item.end
            if item.type == "StartLoop":
                break
            if isinstance(item, AddLine):
                loop_start = item.start
        if loop_start is not None and loop_tail is not None:
            if close_points(loop_start, loop_tail):
                _add(features, "endloop_closed")
            else:
                _add(features, "endloop_unclosed")
    elif isinstance(action, Extrude):
        if action.profile_id in registered_profiles:
            _add(features, "extrude_profile_exists")
        else:
            _add(features, "extrude_profile_missing")

    return features


def _replay_prefix_state(prefix_dicts: list[dict[str, Any]]) -> Any | None:
    verifier = TopoVerifier()
    state = verifier.initial_state()
    for action in actions_from_dicts(prefix_dicts):
        result = verifier.step(state, action)
        if not result.valid:
            return None
        assert result.next_state is not None
        state = result.next_state
    return state


def _stack_top(state: Any | None) -> str:
    if state is None or not state.stack:
        return "empty"
    return str(state.stack[-1])


def _strictly_inside_outer(state: Any | None, point: tuple[float, float]) -> bool:
    if state is None:
        return False
    outer_loops = [loop for loop in state.current_face_loops if loop.kind == "outer"]
    if len(outer_loops) != 1:
        return False
    outer = outer_loops[0]
    if not point_in_polygon(point, outer.points):
        return False
    return not any(on_segment(start, point, end) for start, end in polygon_segments(outer.points))


def extract_topoprm_features(prefix_dicts: list[dict[str, Any]], action_dict: dict[str, Any]) -> FeatureVector:
    features = extract_features(prefix_dicts, action_dict)
    action = action_from_dict(action_dict)
    state = _replay_prefix_state(prefix_dicts)
    top = _stack_top(state)
    _add(features, f"stack_top={top}")
    if state is None:
        _add(features, "prefix_replay_invalid")
        return features

    _add(features, f"stack_depth={min(len(state.stack), 4)}")
    _add(features, f"profile_count_bucket={min(len(state.profiles), 4)}")
    _add(features, f"extrude_count_bucket={min(len(state.extrusions), 4)}")
    if state.pending_face is not None:
        _add(features, "has_pending_face")
    if state.ended:
        _add(features, "state_ended")
    if state.current_loop is not None:
        loop = state.current_loop
        _add(features, f"current_loop_kind={loop.kind}")
        _add(features, f"current_loop_points_bucket={min(len(loop.points), 8)}")
        if loop.start is not None:
            _add(features, "current_loop_has_start")
        if loop.tail is not None:
            _add(features, "current_loop_has_tail")
        if loop.circle_closed:
            _add(features, "current_loop_circle_closed")
    _add(features, f"face_loop_count_bucket={min(len(state.current_face_loops), 4)}")
    if any(loop.kind == "outer" for loop in state.current_face_loops):
        _add(features, "face_has_outer")
    if any(loop.kind == "inner" for loop in state.current_face_loops):
        _add(features, "face_has_inner")

    if isinstance(action, StartSketch):
        if not state.stack and not state.profiles and not state.extrusions and state.pending_face is None:
            _add(features, "start_sketch_context_ok")
    elif isinstance(action, StartFace):
        if top == "sketch" and state.pending_face is None:
            _add(features, "start_face_context_ok")
    elif isinstance(action, StartLoop):
        if top == "face" and state.current_loop is None:
            _add(features, "start_loop_context_ok")
        if not state.current_face_loops and action.kind == "outer":
            _add(features, "start_loop_expected_outer")
        if state.current_face_loops and action.kind == "inner":
            _add(features, "start_loop_expected_inner")
    elif isinstance(action, AddLine):
        loop = state.current_loop
        if top == "loop" and loop is not None and not loop.circle_closed:
            _add(features, "add_line_context_ok")
        if distance(action.start, action.end) <= EPS:
            _add(features, "geom_degenerate_line")
        if loop is not None and loop.tail is not None:
            if close_points(action.start, loop.tail):
                _add(features, "geom_start_matches_loop_tail")
            else:
                _add(features, "geom_start_misses_loop_tail")
        if loop is not None and loop.kind == "inner":
            if _strictly_inside_outer(state, action.start) and _strictly_inside_outer(state, action.end):
                _add(features, "geom_inner_line_inside_outer")
            else:
                _add(features, "geom_inner_line_not_inside_outer")
    elif isinstance(action, AddCircle):
        loop = state.current_loop
        if top == "loop" and loop is not None and not loop.points and not loop.segments:
            _add(features, "add_circle_context_ok")
        if action.radius <= EPS:
            _add(features, "geom_degenerate_circle")
        if loop is not None and loop.kind == "inner":
            if _strictly_inside_outer(state, action.center):
                _add(features, "geom_inner_circle_center_inside_outer")
            else:
                _add(features, "geom_inner_circle_center_not_inside_outer")
    elif isinstance(action, EndLoop):
        loop = state.current_loop
        if top == "loop" and loop is not None:
            _add(features, "end_loop_context_ok")
            if loop.circle_closed:
                _add(features, "end_loop_circle_closed")
            elif loop.start is not None and loop.tail is not None and close_points(loop.tail, loop.start):
                _add(features, "end_loop_tail_closes_start")
            else:
                _add(features, "end_loop_tail_open")
    elif isinstance(action, EndFace):
        if top == "face" and state.current_loop is None and state.current_face_loops:
            _add(features, "end_face_context_ok")
        if len([loop for loop in state.current_face_loops if loop.kind == "outer"]) == 1:
            _add(features, "end_face_has_one_outer")
    elif isinstance(action, RegisterProfile):
        if top == "sketch" and state.pending_face is not None:
            _add(features, "register_profile_context_ok")
        if action.profile_id in state.profiles:
            _add(features, "register_profile_duplicate")
    elif isinstance(action, EndSketch):
        if top == "sketch" and state.pending_face is None:
            _add(features, "end_sketch_context_ok")
        if len(state.profiles) > (state.sketch_start_profile_count or 0):
            _add(features, "end_sketch_has_new_profile")
    elif isinstance(action, Extrude):
        if state.stack in ([], ["sketch"]) and state.pending_face is None:
            _add(features, "extrude_context_ok")
        if action.profile_id in state.profiles:
            _add(features, "extrude_registered_profile")
        if action.depth > EPS:
            _add(features, "extrude_positive_depth")
        if action.op in {"add", "cut", "intersect"}:
            _add(features, "extrude_known_op")
    elif isinstance(action, End):
        if not state.stack:
            _add(features, "end_context_empty_stack")
        if state.extrusions:
            _add(features, "end_has_extrusions")

    return features


def load_pair_examples(path: Path) -> list[tuple[FeatureVector, int]]:
    examples: list[tuple[FeatureVector, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append((extract_features(row["prefix"], row["chosen"]["action"]), 1))
            examples.append((extract_features(row["prefix"], row["rejected"]["action"]), 0))
    return examples


def load_cad_prm_examples(path: Path) -> list[tuple[FeatureVector, str]]:
    examples: list[tuple[FeatureVector, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = "valid" if row["valid"] else str(row["failure_type"])
            examples.append((extract_topoprm_features(row["prefix_actions"], row["action"]), label))
    return examples


@dataclass
class LinearPRM:
    weights: dict[str, float]

    def score(self, features: FeatureVector) -> float:
        return sum(self.weights.get(name, 0.0) * value for name, value in features.items())

    def probability(self, features: FeatureVector) -> float:
        score = max(min(self.score(features), 40.0), -40.0)
        return 1.0 / (1.0 + math.exp(-score))

    def predict(self, features: FeatureVector) -> int:
        return 1 if self.probability(features) >= 0.5 else 0

    def to_dict(self) -> dict[str, Any]:
        return {"weights": self.weights}


def train_linear_prm(
    examples: list[tuple[FeatureVector, int]],
    *,
    epochs: int = 20,
    lr: float = 0.2,
    l2: float = 1e-5,
    seed: int = 0,
) -> LinearPRM:
    rng = random.Random(seed)
    weights: dict[str, float] = {}
    for _ in range(epochs):
        shuffled = list(examples)
        rng.shuffle(shuffled)
        for features, label in shuffled:
            score = sum(weights.get(name, 0.0) * value for name, value in features.items())
            score = max(min(score, 40.0), -40.0)
            prob = 1.0 / (1.0 + math.exp(-score))
            error = label - prob
            for name, value in features.items():
                weights[name] = weights.get(name, 0.0) + lr * (error * value - l2 * weights.get(name, 0.0))
    return LinearPRM(weights)


def evaluate(model: LinearPRM, examples: list[tuple[FeatureVector, int]]) -> dict[str, Any]:
    if not examples:
        return {"num_examples": 0, "accuracy": None}
    correct = 0
    total_loss = 0.0
    for features, label in examples:
        prob = min(max(model.probability(features), 1e-8), 1.0 - 1e-8)
        pred = 1 if prob >= 0.5 else 0
        correct += int(pred == label)
        total_loss += -(label * math.log(prob) + (1 - label) * math.log(1 - prob))
    return {
        "num_examples": len(examples),
        "accuracy": correct / len(examples),
        "log_loss": total_loss / len(examples),
    }


def train_val_split(
    examples: list[tuple[FeatureVector, int]],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[tuple[FeatureVector, int]], list[tuple[FeatureVector, int]]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_fraction))
    return shuffled[val_size:], shuffled[:val_size]


@dataclass
class LinearTopoPRM:
    classes: list[str]
    weights: dict[str, dict[str, float]]

    def logits(self, features: FeatureVector) -> dict[str, float]:
        return {
            label: sum(self.weights.get(label, {}).get(name, 0.0) * value for name, value in features.items())
            for label in self.classes
        }

    def probabilities(self, features: FeatureVector) -> dict[str, float]:
        logits = self.logits(features)
        max_logit = max(logits.values()) if logits else 0.0
        exp_values = {label: math.exp(max(min(value - max_logit, 40.0), -40.0)) for label, value in logits.items()}
        total = sum(exp_values.values()) or 1.0
        return {label: value / total for label, value in exp_values.items()}

    def valid_probability(self, features: FeatureVector) -> float:
        return self.probabilities(features).get("valid", 0.0)

    def predict_class(self, features: FeatureVector) -> str:
        probs = self.probabilities(features)
        return max(probs.items(), key=lambda item: item[1])[0]

    def to_dict(self) -> dict[str, Any]:
        return {"classes": self.classes, "weights": self.weights}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearTopoPRM":
        return cls(classes=list(data["classes"]), weights={label: dict(weights) for label, weights in data["weights"].items()})


def train_linear_topoprm(
    examples: list[tuple[FeatureVector, str]],
    *,
    classes: list[str] | None = None,
    epochs: int = 25,
    lr: float = 0.1,
    l2: float = 1e-5,
    seed: int = 0,
) -> LinearTopoPRM:
    rng = random.Random(seed)
    labels = sorted({label for _, label in examples})
    ordered_classes = classes or [label for label in TOPOPRM_CLASSES if label in labels]
    for label in labels:
        if label not in ordered_classes:
            ordered_classes.append(label)
    weights: dict[str, dict[str, float]] = {label: {} for label in ordered_classes}
    for _ in range(epochs):
        shuffled = list(examples)
        rng.shuffle(shuffled)
        for features, gold in shuffled:
            model = LinearTopoPRM(ordered_classes, weights)
            probs = model.probabilities(features)
            for label in ordered_classes:
                error = (1.0 if label == gold else 0.0) - probs.get(label, 0.0)
                label_weights = weights[label]
                for name, value in features.items():
                    current = label_weights.get(name, 0.0)
                    label_weights[name] = current + lr * (error * value - l2 * current)
    return LinearTopoPRM(ordered_classes, weights)


def evaluate_topoprm(model: LinearTopoPRM, examples: list[tuple[FeatureVector, str]]) -> dict[str, Any]:
    if not examples:
        return {"num_examples": 0, "accuracy": None}
    correct = 0
    valid_total = valid_correct = 0
    invalid_total = invalid_correct = 0
    failure_total = failure_correct = 0
    total_loss = 0.0
    predicted_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for features, gold in examples:
        probs = model.probabilities(features)
        pred = max(probs.items(), key=lambda item: item[1])[0]
        prob = min(max(probs.get(gold, 1e-8), 1e-8), 1.0)
        total_loss += -math.log(prob)
        correct += int(pred == gold)
        predicted_counts[pred] = predicted_counts.get(pred, 0) + 1
        gold_counts[gold] = gold_counts.get(gold, 0) + 1
        gold_valid = gold == "valid"
        pred_valid = pred == "valid"
        if gold_valid:
            valid_total += 1
            valid_correct += int(pred_valid)
        else:
            invalid_total += 1
            invalid_correct += int(not pred_valid)
            failure_total += 1
            failure_correct += int(pred == gold)
    return {
        "num_examples": len(examples),
        "accuracy": correct / len(examples),
        "log_loss": total_loss / len(examples),
        "valid_recall": valid_correct / valid_total if valid_total else None,
        "invalid_recall": invalid_correct / invalid_total if invalid_total else None,
        "failure_type_accuracy_on_invalid": failure_correct / failure_total if failure_total else None,
        "gold_counts": gold_counts,
        "predicted_counts": predicted_counts,
    }


def save_topoprm(model: LinearTopoPRM, path: Path, metadata: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_type": "linear_topoprm_v1", "metadata": metadata or {}, **model.to_dict()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_topoprm(path: Path) -> LinearTopoPRM:
    return LinearTopoPRM.from_dict(json.loads(path.read_text(encoding="utf-8")))
