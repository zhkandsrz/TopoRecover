from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .prompt_contract import normalize_topology_contract


ALLOWED_OPERATIONS = ("add", "cut", "intersect")
ALLOWED_LOOP_ROLES = ("outer", "inner")
ALLOWED_FEATURE_TYPES = ("extrude",)


@dataclass(frozen=True)
class ParsedFeaturePlan:
    plan: dict[str, Any] | None
    error: str | None


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


def normalize_feature_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a generator-authored plan without reading a target CAD.

    The schema deliberately contains only information that may exist before
    low-level history generation. Geometry parameters may be retained for the
    generator, but TopoRecover's topology contract projection ignores them.
    """

    if isinstance(value.get("feature_plan"), Mapping):
        value = value["feature_plan"]

    raw_profiles = list(value.get("profiles") or [])
    raw_features = list(value.get("features") or [])
    if not 1 <= len(raw_profiles) <= 64:
        raise ValueError("feature plan must contain 1 to 64 profiles")
    if not 1 <= len(raw_features) <= 64:
        raise ValueError("feature plan must contain 1 to 64 features")

    profiles: list[dict[str, Any]] = []
    profile_ids: set[str] = set()
    for index, raw in enumerate(raw_profiles):
        if not isinstance(raw, Mapping):
            raise ValueError("profile entries must be objects")
        profile_id = str(raw.get("profile_id") or f"profile_{index}").strip()
        if not profile_id or profile_id in profile_ids:
            raise ValueError("profile_id values must be non-empty and unique")
        roles = [str(role).strip().lower() for role in raw.get("loop_roles") or []]
        if not roles or any(role not in ALLOWED_LOOP_ROLES for role in roles):
            raise ValueError("loop_roles must contain only outer/inner")
        if roles.count("outer") != 1:
            raise ValueError("each profile must contain exactly one outer loop")
        profile_ids.add(profile_id)
        profile = {
            "profile_id": profile_id,
            "loop_roles": ["outer", *(["inner"] * roles.count("inner"))],
        }
        if raw.get("sketch_plane") is not None:
            profile["sketch_plane"] = str(raw["sketch_plane"]).strip()
        profiles.append(profile)

    features: list[dict[str, Any]] = []
    feature_ids: set[str] = set()
    for index, raw in enumerate(raw_features):
        if not isinstance(raw, Mapping):
            raise ValueError("feature entries must be objects")
        feature_id = str(raw.get("feature_id") or f"feature_{index}").strip()
        feature_type = str(raw.get("type") or "").strip().lower()
        profile_id = str(raw.get("profile_id") or "").strip()
        operation = str(raw.get("operation") or "").strip().lower()
        if not feature_id or feature_id in feature_ids:
            raise ValueError("feature_id values must be non-empty and unique")
        if feature_type not in ALLOWED_FEATURE_TYPES:
            raise ValueError(f"unsupported feature type: {feature_type or '<empty>'}")
        if profile_id not in profile_ids:
            raise ValueError("feature profile_id must reference a declared profile")
        if operation not in ALLOWED_OPERATIONS:
            raise ValueError("feature operation must be add, cut, or intersect")
        feature_ids.add(feature_id)
        feature = {
            "feature_id": feature_id,
            "type": feature_type,
            "profile_id": profile_id,
            "operation": operation,
        }
        if raw.get("depth") is not None:
            feature["depth"] = float(raw["depth"])
        features.append(feature)

    return {
        "schema_version": "toporecover_feature_plan_v1",
        "profiles": profiles,
        "features": features,
    }


def parse_feature_plan(text: str) -> ParsedFeaturePlan:
    errors: list[str] = []
    for value in _json_objects(text):
        try:
            return ParsedFeaturePlan(normalize_feature_plan(value), None)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    error = errors[-1] if errors else "no JSON feature plan found"
    return ParsedFeaturePlan(None, error)


def feature_plan_signature(plan: Mapping[str, Any]) -> str:
    normalized = normalize_feature_plan(plan)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_plan_to_topology_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Project a frozen upstream plan into TopoRecover's anonymous contract."""

    normalized = normalize_feature_plan(plan)
    profile_index = {
        profile["profile_id"]: index
        for index, profile in enumerate(normalized["profiles"])
    }
    contract = {
        "profiles": [
            {
                "profile_index": index,
                "loop_roles": list(profile["loop_roles"]),
            }
            for index, profile in enumerate(normalized["profiles"])
        ],
        "extrusion_graph": [
            {
                "profile_index": profile_index[feature["profile_id"]],
                "operation": feature["operation"],
            }
            for feature in normalized["features"]
            if feature["type"] == "extrude"
        ],
    }
    return normalize_topology_contract(contract)


def feature_plan_prompt(*, design_brief: str) -> str:
    return f"""Create a high-level feature plan for the requested parametric CAD model.

Return exactly one JSON object with this schema:
{{
  "feature_plan": {{
    "profiles": [
      {{"profile_id": "p0", "loop_roles": ["outer", "inner"], "sketch_plane": "XY"}}
    ],
    "features": [
      {{"feature_id": "f0", "type": "extrude", "profile_id": "p0", "operation": "add", "depth": 10.0}}
    ]
  }}
}}

Rules:
- The plan is written before any CAD history is generated.
- Every profile has exactly one outer loop and zero or more inner loops.
- Every feature references a declared profile.
- Supported feature type is extrude; operation is add, cut, or intersect.
- Include dimensions only when stated or unambiguously implied by the brief.
- Do not output CAD actions, code, a target geometry, or an explanation.

DESIGN BRIEF:
{design_brief.strip()}
"""


def plan_conditioned_history_prompt(
    *,
    design_brief: str,
    feature_plan: Mapping[str, Any],
    action_grammar: Sequence[str] | None = None,
) -> str:
    normalized = normalize_feature_plan(feature_plan)
    grammar = ""
    if action_grammar:
        grammar = "\nALLOWED ACTION FORMS:\n" + "\n".join(
            f"- {line}" for line in action_grammar
        )
    return f"""Generate one executable typed CAD construction history.

The FEATURE PLAN below was produced and frozen before history generation. Follow
its profile roles, feature order, profile references, and Boolean operations.
Return only one action per line. Do not revise or reinterpret the feature plan.

DESIGN BRIEF:
{design_brief.strip()}

FROZEN FEATURE PLAN:
{json.dumps(normalized, sort_keys=True, separators=(",", ":"))}{grammar}
"""


def text2cad_plan_conditioned_brief(
    *, design_brief: str, feature_plan: Mapping[str, Any]
) -> str:
    """Render a frozen plan as compact natural language for Text2CAD."""

    normalized = normalize_feature_plan(feature_plan)
    profile_sentences = []
    for profile in normalized["profiles"]:
        inner = profile["loop_roles"].count("inner")
        profile_sentences.append(
            f"Profile {profile['profile_id']} has one outer loop and {inner} inner "
            f"loop{'s' if inner != 1 else ''}."
        )
    feature_sentences = [
        f"Feature {feature['feature_id']} extrudes profile {feature['profile_id']} "
        f"with Boolean operation {feature['operation']}."
        for feature in normalized["features"]
    ]
    plan_text = " ".join([*profile_sentences, *feature_sentences])
    return (
        f"{design_brief.strip()} Follow this frozen construction plan exactly: "
        f"{plan_text} Preserve the stated profile roles, references, feature order, "
        "and Boolean operations."
    )
