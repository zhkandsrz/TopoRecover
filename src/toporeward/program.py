from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import Action, actions_from_dicts, actions_to_dicts, actions_to_text


@dataclass(frozen=True)
class Program:
    actions: tuple[Action, ...]
    program_id: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "source": self.source,
            "actions": actions_to_dicts(self.actions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Program":
        return cls(
            actions=tuple(actions_from_dicts(data["actions"])),
            program_id=data.get("program_id"),
            source=data.get("source"),
        )

    def to_text(self) -> str:
        return actions_to_text(self.actions)

