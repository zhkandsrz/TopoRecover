from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


ALLOWED_OPERATIONS = ("add", "cut", "intersect")
ALLOWED_ROLES = ("outer", "inner")
OPERATION_ALIASES = {
    "add": "add",
    "union": "add",
    "cut": "cut",
    "subtract": "cut",
    "intersect": "intersect",
    "intersection": "intersect",
}


@dataclass(frozen=True)
class ContractSelection:
    contract: dict[str, Any] | None
    valid_samples: int
    agreement: int
    total_samples: int


def _json_objects(text: str) -> Iterable[Mapping[str, Any]]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            yield value


def normalize_topology_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value.get("topology_contract"), Mapping):
        value = value["topology_contract"]

    raw_profiles = list(value.get("profiles") or [])
    raw_graph = list(value.get("extrusion_graph") or [])
    profile_count = len(raw_profiles)
    extrusion_count = len(raw_graph)
    if not 1 <= profile_count <= 64:
        raise ValueError("profile_count must be in [1, 64]")
    if not 1 <= extrusion_count <= 64:
        raise ValueError("extrusion_count must be in [1, 64]")

    profiles: list[dict[str, Any]] = []
    inner_count = 0
    for expected_index, raw in enumerate(raw_profiles):
        if not isinstance(raw, Mapping):
            raise ValueError("profile entries must be objects")
        profile_index = int(raw.get("profile_index", expected_index))
        if profile_index != expected_index:
            raise ValueError("profile indices must be contiguous and ordered")
        roles = [str(role).strip().lower() for role in raw.get("loop_roles") or []]
        if not roles or any(role not in ALLOWED_ROLES for role in roles):
            raise ValueError("loop_roles must contain only outer/inner")
        if roles.count("outer") != 1:
            raise ValueError("each profile must contain exactly one outer loop")
        canonical_roles = ["outer", *(["inner"] * roles.count("inner"))]
        inner_count += roles.count("inner")
        profiles.append(
            {"profile_index": profile_index, "loop_roles": canonical_roles}
        )

    graph: list[dict[str, Any]] = []
    for raw in raw_graph:
        if not isinstance(raw, Mapping):
            raise ValueError("extrusion entries must be objects")
        profile_index = int(raw.get("profile_index", -1))
        operation = OPERATION_ALIASES.get(
            str(raw.get("operation") or "").strip().lower(), ""
        )
        if not 0 <= profile_index < profile_count:
            raise ValueError("extrusion profile_index is out of range")
        if operation not in ALLOWED_OPERATIONS:
            raise ValueError("unsupported extrusion operation")
        graph.append({"profile_index": profile_index, "operation": operation})

    operation_counts = dict(Counter(row["operation"] for row in graph))
    return {
        "profile_count": profile_count,
        "profiles": profiles,
        "inner_loop_count": inner_count,
        "extrusion_count": extrusion_count,
        "extrusion_graph": graph,
        "operation_counts": operation_counts,
    }


def parse_topology_contract(text: str) -> dict[str, Any] | None:
    for value in _json_objects(text):
        try:
            return normalize_topology_contract(value)
        except (TypeError, ValueError):
            continue
    return None


def contract_signature(contract: Mapping[str, Any]) -> str:
    normalized = normalize_topology_contract(contract)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def select_contract_consensus(texts: Sequence[str]) -> ContractSelection:
    parsed = [parse_topology_contract(text) for text in texts]
    valid = [contract for contract in parsed if contract is not None]
    if not valid:
        return ContractSelection(None, 0, 0, len(texts))
    signatures = [contract_signature(contract) for contract in valid]
    counts = Counter(signatures)
    best_count = max(counts.values())
    best_signature = next(
        signature for signature in signatures if counts[signature] == best_count
    )
    selected = next(
        contract
        for contract, signature in zip(valid, signatures)
        if signature == best_signature
    )
    return ContractSelection(selected, len(valid), best_count, len(texts))


def topology_contract_metrics(
    predicted: Mapping[str, Any] | None,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    truth = normalize_topology_contract(target)
    if predicted is None:
        return {
            "contract_exact": False,
            "profile_count_exact": False,
            "inner_loop_count_exact": False,
            "extrusion_count_exact": False,
            "profile_roles_exact": False,
            "extrusion_graph_exact": False,
        }
    guess = normalize_topology_contract(predicted)
    return {
        "contract_exact": guess == truth,
        "profile_count_exact": guess["profile_count"] == truth["profile_count"],
        "inner_loop_count_exact": (
            guess["inner_loop_count"] == truth["inner_loop_count"]
        ),
        "extrusion_count_exact": (
            guess["extrusion_count"] == truth["extrusion_count"]
        ),
        "profile_roles_exact": guess["profiles"] == truth["profiles"],
        "extrusion_graph_exact": (
            guess["extrusion_graph"] == truth["extrusion_graph"]
        ),
    }


def contract_induction_prompt(
    *,
    design_brief: str,
    observed_actions: Sequence[str] | None = None,
) -> str:
    history = ""
    if observed_actions is not None:
        history = (
            "\n\nFAILED GENERATED HISTORY (evidence, not the requested design):\n"
            + "\n".join(f"{index}: {line}" for index, line in enumerate(observed_actions))
        )
    return f"""Infer the requested topology of a parametric CAD construction.

Use the design brief as the source of intent. A failed generated history may be
provided only as noisy evidence. Do not copy an accidental extra or missing
feature from that history when it conflicts with the brief.

Return exactly one JSON object with this schema:
{{
  "profile_count": integer,
  "profiles": [
    {{"profile_index": 0, "loop_roles": ["outer", "inner"]}}
  ],
  "inner_loop_count": integer,
  "extrusion_count": integer,
  "extrusion_graph": [
    {{"profile_index": 0, "operation": "add"}}
  ]
}}

Rules:
- Every profile has exactly one outer loop and zero or more inner loops.
- profile_count must equal the number of profiles.
- inner_loop_count must equal the total number of inner entries in profiles.
- extrusion_count must equal the number of entries in extrusion_graph.
- Each extrusion references a profile by zero-based index.
- operation is add, cut, or intersect.
- Do not output dimensions, coordinates, explanations, or markdown.

DESIGN BRIEF:
{design_brief.strip()}{history}
"""
