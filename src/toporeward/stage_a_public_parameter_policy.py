"""Versioned development wrapper: public dimension evidence, then LLM repair."""
from copy import deepcopy

from .public_plan_parameters import bind_public_plan_parameters
from .stage_a_packet_policy import PUBLIC_SOURCE_FIELDS
from .stage_a_structure_policy import prepare_structure_intent, submit_structure_intent


def _wrap(source, bound, evidence, child):
    state = dict(source=source, bound_source=bound, parameter_evidence=evidence,
                 child=child, phase=child['phase'], requests=child['requests'],
                 _actual_calls=child.get('_actual_calls',0),
                 _actual_tokens=child.get('_actual_tokens',0))
    if child['phase'] == 'complete':
        state['result'] = {**deepcopy(child['result']), 'case_id':source['case_id'],
                           'public_parameter_evidence':deepcopy(evidence),
                           'private_reference_parameter_selection':False}
    return state


def prepare_public_parameter_intent(row):
    source = {k:deepcopy(row[k]) for k in PUBLIC_SOURCE_FIELDS}
    try:
        bound, evidence = bind_public_plan_parameters(source)
    except ValueError as error:
        return dict(source=source,phase='complete',requests=[],result=dict(
            case_id=source['case_id'],final_actions=[],stage_a_success=False,
            topology_success=False,llm_calls=0,calls=0,output_tokens=0,
            action_lcs=0.,failure=str(error),symbolic_stage_a_fallback=False,
            route='public_parameter_evidence_rejected'))
    if evidence:
        # Different public prompt content must not share a cached request ID.
        bound['case_id'] = source['case_id']+'::public_parameters'
    return _wrap(source,bound,evidence,prepare_structure_intent(bound))


def submit_public_parameter_intent(row,state,raw):
    if state['phase'] == 'complete':
        raise ValueError('public_parameter_session_complete')
    if any(row[k] != state['source'][k] for k in PUBLIC_SOURCE_FIELDS):
        raise ValueError('stale_public_parameter_source')
    child = submit_structure_intent(state['bound_source'],state['child'],raw)
    return _wrap(state['source'],state['bound_source'],state['parameter_evidence'],child)
