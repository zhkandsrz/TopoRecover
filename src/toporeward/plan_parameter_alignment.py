"""Bind public plan parameters by references, not declaration-list position."""
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from math import isclose, isfinite

from .actions import Extrude, RegisterProfile, action_to_text
from .feature_plan import normalize_feature_plan
from .lm.parsing import parse_action_line


_METRES = {'mm': .001, 'cm': .01, 'm': 1., 'inch': .0254}


def bind_aligned_plan_dimensions(contract, plan, observed_lines):
    """Only explicit identities/ordered feature references supply alignment.

Unreferenced profile roles must still match as a multiset. Units are converted
only if both source and history units are declared. Legacy unitless inputs keep
their existing shared-coordinate convention. No scale is inferred from solids.
"""
    bound = deepcopy(contract)
    if not plan:
        return bound, []
    normalized = normalize_feature_plan(plan)
    profiles, graph = bound.get('profiles', []), bound.get('extrusion_graph', [])
    if len(profiles) != len(normalized['profiles']) or len(graph) != len(normalized['features']):
        raise ValueError('public_plan_contract_count_disagreement')
    plan_profiles = {p['profile_id']: p for p in normalized['profiles']}
    parsed = [parse_action_line(s) for s in observed_lines]
    if any(a is None for a in parsed):
        raise ValueError('unparseable_observed_history')
    ids = [a.profile_id for a in parsed if isinstance(a, RegisterProfile)]
    if len(set(ids)) != len(ids):
        raise ValueError('ambiguous_profile_identity')
    mapping, inverse, origins = {}, {}, {}

    def assign(identity, index, source):
        if type(index) is not int or not 0 <= index < len(profiles):
            raise ValueError('invalid_contract_profile_index')
        if identity in mapping and mapping[identity] != index:
            raise ValueError('public_plan_reference_order_conflict')
        if index in inverse and inverse[index] != identity:
            raise ValueError('public_plan_ambiguous_profile_correspondence')
        if sorted(plan_profiles[identity]['loop_roles']) != sorted(profiles[index]['loop_roles']):
            raise ValueError('public_plan_contract_role_disagreement')
        mapping[identity], inverse[index] = index, identity
        origins.setdefault(identity, source)

    # Contract indices follow sorted native IDs when all registrations exist.
    # Matching public names are stronger evidence than anonymous isomorphism.
    if len(ids) == len(profiles):
        for index, identity in enumerate(sorted(ids)):
            if identity in plan_profiles:
                assign(identity, index, 'observed_profile_identity')
    for feature, spec in zip(normalized['features'], graph):
        if feature['operation'] != spec.get('operation'):
            raise ValueError('public_plan_contract_operation_disagreement')
        assign(feature['profile_id'], spec.get('profile_index'), 'ordered_feature_reference')
    left = Counter(tuple(sorted(p['loop_roles'])) for pid, p in plan_profiles.items() if pid not in mapping)
    right = Counter(tuple(sorted(p['loop_roles'])) for i, p in enumerate(profiles) if i not in inverse)
    if left != right:
        raise ValueError('public_plan_contract_role_disagreement')

    raw = plan.get('feature_plan', plan)
    plan_unit, history_unit = raw.get('length_unit'), contract.get('length_unit')
    scale = 1.
    has_depth = any('depth' in f for f in normalized['features'])
    if has_depth and (plan_unit is not None or history_unit is not None):
        if plan_unit not in _METRES or history_unit not in _METRES:
            raise ValueError('missing_or_unsupported_length_units')
        scale = _METRES[plan_unit] / _METRES[history_unit]
    evidence = []
    observed_extrudes = [a for a in parsed if isinstance(a, Extrude)]
    for index, (feature, spec) in enumerate(zip(normalized['features'], graph)):
        if 'depth' not in feature:
            continue
        # Decimal input values and unit factors avoid introducing an edit solely
        # through binary multiplication (for example 0.7 cm -> 0.007 m).
        depth = float(Decimal(str(feature['depth'])) * Decimal(str(scale)))
        if not isfinite(depth) or depth <= 0:
            raise ValueError('invalid_public_plan_depth')
        existing = spec.get('depth')
        serialization_source = 'converted_public_plan'
        if existing is not None:
            existing = float(existing)
            if not isfinite(existing) or existing <= 0 or not isclose(existing, depth, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('public_plan_contract_depth_disagreement')
            depth = existing
            serialization_source = 'explicit_contract'
        else:
            candidates = []
            if len(observed_extrudes) == len(graph):
                candidates = [observed_extrudes[index]]
            elif len(ids) == len(profiles):
                identity = sorted(ids)[spec['profile_index']]
                pair_count = sum(s['profile_index'] == spec['profile_index'] and
                                 s['operation'] == spec['operation'] for s in graph)
                if pair_count == 1:
                    candidates = [a for a in observed_extrudes if a.profile_id == identity and
                                  a.op == spec['operation']]
            if len(candidates) == 1 and isclose(candidates[0].depth, depth, rel_tol=1e-12, abs_tol=1e-15):
                depth = candidates[0].depth
                serialization_source = 'numerically_equivalent_observed_occurrence'
        if serialization_source == 'converted_public_plan':
            native = parse_action_line(action_to_text(Extrude(
                feature['profile_id'], depth, feature['operation']))).depth
            # Match the native writer only for numerical noise, not a material
            # reduction in requested precision. Explicit/observed values win.
            if isclose(native, depth, rel_tol=1e-12, abs_tol=1e-15):
                depth = native
                serialization_source = 'numerically_equivalent_native_literal'
        bound['extrusion_graph'][index]['depth'] = depth
        evidence.append({'operation_index': index, 'feature_id': feature['feature_id'],
                         'plan_profile_id': feature['profile_id'],
                         'contract_profile_index': mapping[feature['profile_id']],
                         'correspondence_source': origins[feature['profile_id']],
                         'depth': depth, 'plan_depth': feature['depth'],
                         'plan_unit': plan_unit, 'history_unit': history_unit, 'scale': scale,
                         'depth_serialization_source': serialization_source,
                         'source': 'public_upstream_feature_plan'})
    return bound, evidence
