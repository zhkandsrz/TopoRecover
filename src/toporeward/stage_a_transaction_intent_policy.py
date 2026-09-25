"""Versioned development policy: LLM Stage A followed by profile-preserving B."""
from copy import deepcopy
from difflib import SequenceMatcher

from .stage_a_intent_policy import prepare_intent_policy, submit_intent_policy
from .stage_a_multi_transaction import committed_spans
from .plan_parameter_alignment import bind_aligned_plan_dimensions
from .stage_b_transaction_preserving import complete_transaction_preserving


def _finish(state):
    if state['phase'] != 'complete':
        return state
    state = deepcopy(state)
    result = state['result']
    state['legacy_completion_diagnostic'] = deepcopy(result)
    working = state['working_source']
    if result.get('patched_actions') and result.get('stage_a_success'):
        accepted, protected = result['patched_actions'], result['protected_end']
    elif result.get('route') in ('stage_b_only', 'llm_retention_then_stage_b'):
        accepted = working['observed_actions']
        protected = max((s.register_end for s in committed_spans(accepted)), default=0)
    else:
        return state
    try:
        contract, evidence = bind_aligned_plan_dimensions(working['topology_contract'], working['feature_plan'], accepted)
    except ValueError as error:
        result.update(final_actions=[], topology_success=False, failure=str(error), action_lcs=0.)
        return state
    completion = complete_transaction_preserving(accepted, contract, protected_end=protected)
    final = list(completion.actions)
    result.update(final_actions=final, topology_success=bool(final), completion_status=completion.status,
                  stage_b_geometry_audit=completion.to_dict(), public_plan_dimension_evidence=evidence,
                  geometry_reference_visible=False, symbolic_stage_a_fallback=False,
                  stage_b_protection='accepted_profile_transactions')
    if final:
        result.pop('failure', None)
    else:
        result['failure'] = completion.status
    original = state['source']['observed_actions']
    result['action_lcs'] = sum(b.size for b in SequenceMatcher(a=original, b=final, autojunk=False).get_matching_blocks()) / max(1, len(original))
    return state


def prepare_transaction_intent(row):
    return _finish(prepare_intent_policy(row))


def submit_transaction_intent(row, state, raw):
    return _finish(submit_intent_policy(row, state, raw))
