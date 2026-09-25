"""Evidence-grounded Stage B; no target geometry or new Stage A decisions."""
from dataclasses import asdict, dataclass
from copy import deepcopy
from math import isfinite

from .actions import Extrude, action_to_text
from .geometry_grounded_repair import ground_repair_geometry
from .feature_plan import normalize_feature_plan, feature_plan_to_topology_contract
from .llm_stage_a import complete_after_patch, profile_diagnostics
from .lm.parsing import parse_action_line
from .natural_repair_recall import evaluate_repaired_history
from .stage_a_multi_transaction import committed_spans


@dataclass(frozen=True)
class PreservedCompletion:
    actions: tuple[str, ...]
    status: str
    evidence: tuple[dict, ...] = ()

    def to_dict(self):
        return asdict(self)


def bind_plan_dimensions(contract, plan):
    """Carry public upstream depths alongside, not inside, the old topology projection."""
    bound = deepcopy(contract)
    if not plan:
        return bound, []
    normalized = normalize_feature_plan(plan)
    projected = feature_plan_to_topology_contract(normalized)
    roles = lambda c: [sorted(p['loop_roles']) for p in c.get('profiles', [])]
    graph = lambda c: [(g['profile_index'], g['operation']) for g in c.get('extrusion_graph', [])]
    if roles(contract) != roles(projected) or graph(contract) != graph(projected):
        raise ValueError('public_plan_contract_disagreement')
    evidence = []
    for index, feature in enumerate(normalized['features']):
        if 'depth' not in feature:
            continue
        depth = feature['depth']
        if not isfinite(depth) or depth <= 0:
            raise ValueError('invalid_public_plan_depth')
        existing = bound['extrusion_graph'][index].get('depth')
        if existing is not None and float(existing) != depth:
            raise ValueError('public_plan_contract_depth_disagreement')
        bound['extrusion_graph'][index]['depth'] = depth
        evidence.append({'operation_index': index, 'feature_id': feature['feature_id'],
                         'depth': depth, 'source': 'public_upstream_feature_plan'})
    return bound, evidence


def complete_preserving_geometry(lines, contract, *, protected_end):
    """Use accepted Stage A history, not pre-patch history, as geometry evidence."""
    original = list(lines)
    if not 0 <= protected_end <= len(original):
        raise ValueError('invalid_protected_end')
    if profile_diagnostics(original, contract):
        return PreservedCompletion((), 'committed_topology_unresolved')
    graph = contract.get('extrusion_graph', [])
    spans = sorted(committed_spans(original), key=lambda p: p.profile_id)
    if len({p.profile_id for p in spans}) != len(spans):
        return PreservedCompletion((), 'ambiguous_profile_identity')
    observed = [(i, parse_action_line(s)) for i, s in enumerate(original)
                if isinstance(parse_action_line(s), Extrude)]
    working, evidence = list(original), []
    # Equal counts identify operation occurrences even if ref/op is the fault.
    # Unequal counts are left to the existing compiler and correspondence audit.
    if len(observed) == len(graph):
        for ordinal, ((index, action), spec) in enumerate(zip(observed, graph)):
            target = spec.get('profile_index')
            operation = spec.get('operation')
            if type(target) is not int or not 0 <= target < len(spans):
                return PreservedCompletion((), 'unresolved_profile_reference')
            if operation not in ('add', 'cut', 'intersect'):
                return PreservedCompletion((), 'invalid_required_operation')
            explicit = spec.get('depth')
            depth = float(action.depth if explicit is None else explicit)
            if not isfinite(depth) or depth <= 0:
                return PreservedCompletion((), 'invalid_supported_depth')
            fixed = Extrude(spans[target].profile_id, depth, operation)
            if fixed != action:
                if index < protected_end:
                    return PreservedCompletion((), 'would_change_protected_operation')
                working[index] = action_to_text(fixed)
                evidence.append({'action_index': index, 'operation_index': ordinal,
                                 'depth_source': 'observed_occurrence' if explicit is None else 'explicit_contract',
                                 'depth': depth, 'reference_and_operation_source': 'contract'})
    proposed, reason = complete_after_patch(working, contract, protected_end=protected_end,
                                            preserve_reference_parameters=True)
    if proposed is None:
        return PreservedCompletion((), reason, tuple(evidence))
    grounded = ground_repair_geometry(original, proposed, contract)
    evidence.extend(grounded.evidence)
    if grounded.status != 'candidate':
        return PreservedCompletion((), grounded.reason, tuple(evidence))
    result = list(grounded.actions)
    # Serialization may normalize decimals. Check semantics, then restore exact
    # protected text and all other semantically unchanged action strings.
    if len(result) < protected_end or any(parse_action_line(a) != parse_action_line(b)
            for a, b in zip(original[:protected_end], result[:protected_end])):
        return PreservedCompletion((), 'stage_b_changed_protected_history', tuple(evidence))
    for i, line in enumerate(result):
        if i < len(original) and parse_action_line(line) == parse_action_line(original[i]):
            result[i] = original[i]
    if not evaluate_repaired_history(result, contract)['valid_and_intent_satisfied']:
        return PreservedCompletion((), 'grounded_topology_replay_failed', tuple(evidence))
    return PreservedCompletion(tuple(result), 'evidence_preserving_completion', tuple(evidence))
