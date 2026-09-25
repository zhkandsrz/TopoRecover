from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .simple_classifier import extract_topoprm_features, load_topoprm


TopoPRMModel = Any


def load_topoprm_model(path: Path, device: str = "cpu") -> TopoPRMModel:
    if path.is_dir():
        config_path = path / "config.json"
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            if str(payload.get("model_type", "")).startswith("neural_topoprm"):
                from .neural_topoprm import load_neural_topoprm

                return load_neural_topoprm(path, device=device)
        raise ValueError(f"Unsupported TopoPRM directory: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    model_type = str(payload.get("model_type", "linear_topoprm_v1"))
    if model_type.startswith("linear_topoprm"):
        return load_topoprm(path)
    if model_type.startswith("neural_topoprm"):
        from .neural_topoprm import load_neural_topoprm

        return load_neural_topoprm(path.parent, device=device)
    raise ValueError(f"Unsupported TopoPRM model_type={model_type!r} at {path}")


def _has_neural_api(model: TopoPRMModel) -> bool:
    return hasattr(model, "valid_probability_for") and hasattr(model, "predict_class_for")


def topoprm_probabilities(
    model: TopoPRMModel,
    prefix_dicts: list[dict[str, Any]],
    action_dict: dict[str, Any],
    target_stats: dict[str, Any] | None = None,
) -> dict[str, float]:
    if _has_neural_api(model):
        return model.probabilities_for(prefix_dicts, action_dict, target_stats)
    features = extract_topoprm_features(prefix_dicts, action_dict)
    return model.probabilities(features)


def topoprm_valid_probability(
    model: TopoPRMModel,
    prefix_dicts: list[dict[str, Any]],
    action_dict: dict[str, Any],
    target_stats: dict[str, Any] | None = None,
) -> float:
    if _has_neural_api(model):
        return model.valid_probability_for(prefix_dicts, action_dict, target_stats)
    features = extract_topoprm_features(prefix_dicts, action_dict)
    return float(model.valid_probability(features))


def topoprm_predict_class(
    model: TopoPRMModel,
    prefix_dicts: list[dict[str, Any]],
    action_dict: dict[str, Any],
    target_stats: dict[str, Any] | None = None,
) -> str:
    if _has_neural_api(model):
        return model.predict_class_for(prefix_dicts, action_dict, target_stats)
    features = extract_topoprm_features(prefix_dicts, action_dict)
    return str(model.predict_class(features))
