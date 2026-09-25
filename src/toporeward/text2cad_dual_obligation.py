from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .actions import Action
from .natural_repair_recall import topology_contract_equivalent
from .topology_history_repair import topology_intent_contract
from .verifier import TopoVerifier, VerificationState


@dataclass(frozen=True)
class DualObligationAudit:
    """Execution and goal-topology failures for one generated CAD history."""

    execution_valid_ended: bool
    first_invalid_step: int | None
    first_invalid_failure_type: str | None
    first_contract_divergence_step: int | None
    goal_contract_match: bool
    goal_failure: bool
    earlier_contract_divergence: bool
    failure_scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _profile_rows(state: VerificationState) -> list[dict[str, Any]]:
    return [
        {
            "profile_index": index,
            "loop_roles": sorted(
                loop.kind for loop in state.profiles[profile_id].loops
            ),
        }
        for index, profile_id in enumerate(sorted(state.profiles))
    ]


def contract_prefix_compatible(
    state: VerificationState, contract: Mapping[str, Any]
) -> bool:
    """Return whether committed state can still be a prefix of ``contract``."""

    desired_profiles = list(contract.get("profiles") or [])
    existing_profiles = _profile_rows(state)
    if len(existing_profiles) > len(desired_profiles):
        return False
    for existing, desired in zip(existing_profiles, desired_profiles):
        desired_roles = sorted(str(x) for x in desired.get("loop_roles") or [])
        if existing["loop_roles"] != desired_roles:
            return False

    desired_graph = list(contract.get("extrusion_graph") or [])
    if len(state.extrusions) > len(desired_graph):
        return False
    profile_index = {
        str(profile_id): index
        for index, profile_id in enumerate(sorted(state.profiles))
    }
    for existing, desired in zip(state.extrusions, desired_graph):
        if profile_index.get(str(existing["profile_id"]), -1) != int(
            desired.get("profile_index", -1)
        ):
            return False
        if str(existing["op"]) != str(desired.get("operation") or ""):
            return False
    return True


def audit_dual_obligation(
    actions: Sequence[Action],
    contract: Mapping[str, Any],
    *,
    verifier: TopoVerifier | None = None,
) -> DualObligationAudit:
    """Audit execution failure and earlier committed-topology divergence.

    The target action history is neither required nor inspected. ``contract``
    contains only anonymous profile roles and the extrusion-reference graph.
    """

    verifier = verifier or TopoVerifier()
    state = verifier.initial_state()
    first_invalid_step: int | None = None
    first_invalid_failure_type: str | None = None
    first_contract_divergence_step: int | None = None

    for step, action in enumerate(actions):
        result = verifier.step(state, action)
        if not result.valid or result.next_state is None:
            first_invalid_step = step
            first_invalid_failure_type = result.failure_type
            break
        state = result.next_state
        if (
            first_contract_divergence_step is None
            and not contract_prefix_compatible(state, contract)
        ):
            first_contract_divergence_step = step

    execution_valid_ended = first_invalid_step is None and bool(state.ended)
    observed_contract = None
    if execution_valid_ended:
        from .actions import action_to_text

        observed_contract = topology_intent_contract(
            [action_to_text(action) for action in actions],
            level="topology",
            verifier=verifier,
        )
    goal_contract_match = (
        topology_contract_equivalent(observed_contract, contract)
        if observed_contract is not None
        else False
    )
    if (
        execution_valid_ended
        and not goal_contract_match
        and first_contract_divergence_step is None
    ):
        # A missing suffix remains prefix-compatible until the terminal action.
        first_contract_divergence_step = max(0, len(actions) - 1)

    earlier_contract_divergence = (
        first_contract_divergence_step is not None
        and (
            first_invalid_step is None
            or first_contract_divergence_step < first_invalid_step
        )
    )
    goal_failure = not (execution_valid_ended and goal_contract_match)
    if not goal_failure:
        failure_scope = "success"
    elif earlier_contract_divergence and first_invalid_step is not None:
        failure_scope = "dual_obligation"
    elif earlier_contract_divergence:
        failure_scope = "contract_only"
    else:
        failure_scope = "execution_local"

    return DualObligationAudit(
        execution_valid_ended=execution_valid_ended,
        first_invalid_step=first_invalid_step,
        first_invalid_failure_type=first_invalid_failure_type,
        first_contract_divergence_step=first_contract_divergence_step,
        goal_contract_match=goal_contract_match,
        goal_failure=goal_failure,
        earlier_contract_divergence=earlier_contract_divergence,
        failure_scope=failure_scope,
    )


def summarize_dual_obligations(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures = [row for row in rows if row.get("goal_failure")]
    scopes = Counter(str(row.get("failure_scope")) for row in failures)
    first_invalid = [row for row in failures if row.get("first_invalid_step") is not None]
    earlier = [row for row in failures if row.get("earlier_contract_divergence")]
    return {
        "rows": len(rows),
        "natural_goal_failures": len(failures),
        "execution_failures": len(first_invalid),
        "earlier_contract_divergence_failures": len(earlier),
        "earlier_contract_divergence_given_failure": (
            len(earlier) / len(failures) if failures else 0.0
        ),
        "failure_scopes": dict(sorted(scopes.items())),
    }
