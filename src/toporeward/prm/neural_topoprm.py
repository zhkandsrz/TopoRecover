from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..actions import action_from_dict, action_to_text, actions_from_dicts, actions_to_text
from ..verifier import TopoVerifier
from .simple_classifier import FeatureVector, TOPOPRM_CLASSES, extract_topoprm_features


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class NeuralTopoPRMConfig:
    classes: list[str]
    char_vocab: dict[str, int]
    feature_names: list[str]
    max_length: int = 768
    embedding_dim: int = 48
    hidden_dim: int = 96
    dropout: float = 0.1


class NeuralTopoPRMNet(nn.Module):
    def __init__(self, config: NeuralTopoPRMConfig):
        super().__init__()
        self.config = config
        vocab_size = max(config.char_vocab.values()) + 1
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(config.embedding_dim, config.hidden_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(config.embedding_dim, config.hidden_dim, kernel_size=5, padding=2)
        self.feature_proj = nn.Linear(len(config.feature_names), config.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, len(config.classes)),
        )

    def forward(self, input_ids: torch.Tensor, feature_values: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(0).unsqueeze(1)
        embedded = self.embedding(input_ids).transpose(1, 2)
        conv3 = torch.relu(self.conv3(embedded)).masked_fill(~mask, -1e4).amax(dim=-1)
        conv5 = torch.relu(self.conv5(embedded)).masked_fill(~mask, -1e4).amax(dim=-1)
        dense = torch.relu(self.feature_proj(feature_values))
        return self.classifier(torch.cat([conv3, conv5, dense], dim=-1))


def _format_point(point: Any) -> str:
    if point is None:
        return "None"
    return f"({float(point[0]):.6g},{float(point[1]):.6g})"


def _state_summary(prefix_dicts: list[dict[str, Any]]) -> str:
    verifier = TopoVerifier()
    state = verifier.initial_state()
    for action in actions_from_dicts(prefix_dicts):
        result = verifier.step(state, action)
        if not result.valid:
            return f"STATE invalid_prefix failure={result.failure_type}"
        assert result.next_state is not None
        state = result.next_state

    parts = [
        f"stack={'/'.join(state.stack) if state.stack else 'empty'}",
        f"ended={int(state.ended)}",
        f"profiles={','.join(sorted(state.profiles)) if state.profiles else 'none'}",
        f"profile_count={len(state.profiles)}",
        f"extrude_count={len(state.extrusions)}",
        f"face_loop_count={len(state.current_face_loops)}",
        f"pending_face={int(state.pending_face is not None)}",
    ]
    if state.current_loop is not None:
        loop = state.current_loop
        parts.extend(
            [
                f"current_loop_kind={loop.kind}",
                f"current_loop_points={len(loop.points)}",
                f"current_loop_start={_format_point(loop.start)}",
                f"current_loop_tail={_format_point(loop.tail)}",
                f"current_loop_circle_closed={int(loop.circle_closed)}",
            ]
        )
    else:
        parts.append("current_loop=None")
    for index, loop in enumerate(state.current_face_loops):
        parts.append(f"face_loop_{index}={loop.kind}:points={len(loop.points)}:area={loop.area:.6g}")
    return "STATE " + " ".join(parts)


def serialize_topoprm_input(
    prefix_dicts: list[dict[str, Any]],
    action_dict: dict[str, Any],
    target_stats: dict[str, Any] | None = None,
) -> str:
    target_stats = target_stats or {}
    prefix_actions = actions_from_dicts(prefix_dicts) if prefix_dicts else []
    action = action_from_dict(action_dict)
    target = (
        "TARGET "
        f"actions={int(target_stats.get('action_count', 0) or 0)} "
        f"profiles={int(target_stats.get('profile_count', 0) or 0)} "
        f"holes={int(target_stats.get('hole_count', 0) or 0)} "
        f"circle_holes={int(target_stats.get('circle_hole_count', 0) or 0)} "
        f"extrusions={int(target_stats.get('extrude_count', 0) or 0)}"
    )
    prefix_text = actions_to_text(prefix_actions) if prefix_actions else "EMPTY_PREFIX"
    return "\n".join(
        [
            target,
            _state_summary(prefix_dicts),
            "PREFIX",
            prefix_text,
            "ACTION",
            action_to_text(action),
        ]
    )


def row_to_neural_topoprm_example(row: dict[str, Any]) -> tuple[str, FeatureVector, str]:
    label = "valid" if row["valid"] else str(row["failure_type"])
    text = serialize_topoprm_input(row["prefix_actions"], row["action"], row.get("target_stats"))
    features = extract_topoprm_features(row["prefix_actions"], row["action"])
    return text, features, label


def read_cad_prm_rows(path: Path, max_rows: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def build_char_vocab(texts: list[str]) -> dict[str, int]:
    chars = sorted({char for text in texts for char in text})
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for char in chars:
        if char not in vocab:
            vocab[char] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    ids = [vocab.get(char, vocab[UNK_TOKEN]) for char in text[:max_length]]
    if len(ids) < max_length:
        ids.extend([0] * (max_length - len(ids)))
    return ids


def encode_features(features: FeatureVector, feature_names: list[str]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in feature_names]


class NeuralTopoPRMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: list[tuple[str, FeatureVector, str]],
        config: NeuralTopoPRMConfig,
        class_to_id: dict[str, int],
    ):
        self.examples = examples
        self.config = config
        self.class_to_id = class_to_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text, features, label = self.examples[index]
        return {
            "input_ids": torch.tensor(encode_text(text, self.config.char_vocab, self.config.max_length), dtype=torch.long),
            "feature_values": torch.tensor(encode_features(features, self.config.feature_names), dtype=torch.float32),
            "label": torch.tensor(self.class_to_id[label], dtype=torch.long),
        }


class NeuralTopoPRMScorer:
    def __init__(self, config: NeuralTopoPRMConfig, model: NeuralTopoPRMNet, device: torch.device | None = None):
        self.config = config
        self.model = model
        self.device = device or torch.device("cpu")
        self.class_to_id = {label: idx for idx, label in enumerate(config.classes)}
        self.model.to(self.device)
        self.model.eval()

    def probabilities_for(
        self,
        prefix_dicts: list[dict[str, Any]],
        action_dict: dict[str, Any],
        target_stats: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        text = serialize_topoprm_input(prefix_dicts, action_dict, target_stats)
        features = extract_topoprm_features(prefix_dicts, action_dict)
        with torch.no_grad():
            input_ids = torch.tensor(
                [encode_text(text, self.config.char_vocab, self.config.max_length)],
                dtype=torch.long,
                device=self.device,
            )
            feature_values = torch.tensor(
                [encode_features(features, self.config.feature_names)],
                dtype=torch.float32,
                device=self.device,
            )
            probs = torch.softmax(self.model(input_ids, feature_values), dim=-1)[0].detach().cpu().tolist()
        return {label: float(probs[idx]) for idx, label in enumerate(self.config.classes)}

    def valid_probability_for(
        self,
        prefix_dicts: list[dict[str, Any]],
        action_dict: dict[str, Any],
        target_stats: dict[str, Any] | None = None,
    ) -> float:
        return self.probabilities_for(prefix_dicts, action_dict, target_stats).get("valid", 0.0)

    def predict_class_for(
        self,
        prefix_dicts: list[dict[str, Any]],
        action_dict: dict[str, Any],
        target_stats: dict[str, Any] | None = None,
    ) -> str:
        probs = self.probabilities_for(prefix_dicts, action_dict, target_stats)
        return max(probs.items(), key=lambda item: item[1])[0]


def save_neural_topoprm(
    scorer: NeuralTopoPRMScorer,
    out_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "model_type": "neural_topoprm_text_cnn_v1",
        "metadata": metadata or {},
        "config": asdict(scorer.config),
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(scorer.model.state_dict(), out_dir / "model.pt")


def load_neural_topoprm(path: Path, device: str | torch.device = "cpu") -> NeuralTopoPRMScorer:
    config_path = path / "config.json" if path.is_dir() else path
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = NeuralTopoPRMConfig(**payload["config"])
    model = NeuralTopoPRMNet(config)
    model_path = config_path.parent / "model.pt"
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    return NeuralTopoPRMScorer(config=config, model=model, device=torch.device(device))


def class_weights(labels: list[str], classes: list[str]) -> torch.Tensor:
    counts = {label: 0 for label in classes}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    weights = []
    for label in classes:
        count = max(counts.get(label, 0), 1)
        weights.append(math.sqrt(total / (len(classes) * count)))
    return torch.tensor(weights, dtype=torch.float32)

