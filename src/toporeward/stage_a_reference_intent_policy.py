"""Resolve anonymous contract indices using explicit public feature identities."""
from copy import deepcopy

from .actions import RegisterProfile
from .feature_plan import normalize_feature_plan
from .lm.parsing import parse_action_line
from .natural_repair_recall import topology_contract_equivalent
from .stage_a_intent_policy import prepare_intent_policy, submit_intent_policy
from .stage_a_transaction_intent_policy import _finish as finish_transaction


def align_contract_indices(contract, plan, lines):
    normalized = normalize_feature_plan(plan)
    profiles = contract.get('profiles', [])
    graph = contract.get('extrusion_graph', [])
    ids = [a.profile_id for line in lines if isinstance(a := parse_action_line(line), RegisterProfile)]
    plan_profiles = {p['profile_id']: p for p in normalized['profiles']}
    if (len(ids) != len(profiles) or len(ids) != len(set(ids)) or set(ids) != set(plan_profiles)
            or len(graph) != len(normalized['features'])):
        return deepcopy(contract), []
    indices = {name: i for i, name in enumerate(sorted(ids))}
    mapping, inverse = {}, {}
    for feature, spec in zip(normalized['features'], graph):
        old = spec['profile_index']
        new = indices[feature['profile_id']]
        if feature['operation'] != spec['operation'] or sorted(profiles[old]['loop_roles']) != sorted(
                plan_profiles[feature['profile_id']]['loop_roles']):
            raise ValueError('public_plan_contract_correspondence_conflict')
        if old in mapping and mapping[old] != new or new in inverse and inverse[new] != old:
            raise ValueError('public_plan_ambiguous_profile_correspondence')
        mapping[old], inverse[new] = new, old
    # Do not invent correspondences for unreferenced or ambiguous profiles.
    if len(mapping) != len(profiles):
        return deepcopy(contract), []
    result = deepcopy(contract)
    for old, new in mapping.items():
        result['profiles'][new] = {**deepcopy(profiles[old]), 'profile_index': new}
    for feature in result['extrusion_graph']:
        feature['profile_index'] = mapping[feature['profile_index']]
    if not topology_contract_equivalent(contract, result):
        raise ValueError('contract_semantics_changed')
    evidence = [{'contract_profile_index': old, 'native_profile_index': new,
                 'native_profile_id': sorted(ids)[new], 'source': 'explicit_public_feature_identity'}
                for old, new in sorted(mapping.items())]
    return result, evidence


def _finish(state):
    if state['phase'] != 'complete':
        return state
    result = state['result']
    if result.get('patched_actions') and result.get('stage_a_success'):
        accepted = result['patched_actions']
    elif result.get('route') in ('stage_b_only', 'llm_retention_then_stage_b'):
        accepted = state['working_source']['observed_actions']
    else:
        return finish_transaction(state)
    working = state['working_source']
    try:
        contract, evidence = align_contract_indices(working['topology_contract'], working['feature_plan'], accepted)
    except ValueError:
        # The existing strict binder records the original conflict/abstention.
        return finish_transaction(state)
    changed = deepcopy(state)
    changed['working_source']['topology_contract'] = contract
    final = finish_transaction(changed)
    final['result']['public_reference_index_audit'] = evidence
    final['result']['contract_reindexing_only'] = True
    return final


def prepare_reference_intent(row):
    return _finish(prepare_intent_policy(row))


def submit_reference_intent(row, state, raw):
    return _finish(submit_intent_policy(row, state, raw))
