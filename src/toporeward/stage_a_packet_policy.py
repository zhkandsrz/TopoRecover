"""Stateful local-LLM Stage A interface for complete-cohort evaluation.

The saved session owns the source snapshot and call budget. Only its requests
are sent to the LM. Unsupported cases remain explicit failures, never fallbacks.
"""
from copy import deepcopy

from .llm_stage_a import profile_diagnostics, complete_after_patch
from .stage_a_multi_transaction import committed_spans
from .stage_a_packet_feedback import evaluate_packet_set, quantitative_feedback
from .stage_a_profile_packets import prepare_profile_packets, packet_grammar


PUBLIC_SOURCE_FIELDS = ('case_id', 'observed_actions', 'topology_contract', 'design_brief', 'feature_plan')


def prepare_session(row):
    source = {key: deepcopy(row[key]) for key in PUBLIC_SOURCE_FIELDS}
    session = {'source': source, 'requests': [], 'maximum_calls_per_profile': 2,
               'symbolic_stage_a_fallback': False}
    if not profile_diagnostics(source['observed_actions'], source['topology_contract']):
        session['route'] = 'stage_b_only'
        return session
    try:
        packets = prepare_profile_packets(source)
        for q in packets:
            packet_grammar(q)
    except ValueError as error:
        return {**session, 'route': 'unsupported', 'reason': str(error)}
    return {**session, 'route': 'llm_stage_a', 'requests': packets}


def _check_source(row, session):
    if any(row[key] != session['source'][key] for key in PUBLIC_SOURCE_FIELDS):
        raise ValueError('stale_source_contract_or_brief')


def _responses(requests, raw):
    ids = [r['source_uid'] for r in raw]
    if len(ids) != len(set(ids)) or set(ids) != {q['source_uid'] for q in requests}:
        raise ValueError('incomplete_or_duplicate_response_coverage')
    by_id = {r['source_uid']: r for r in raw}
    return [by_id[q['source_uid']]['response'] for q in requests]


def accept_first(row, session, raw):
    _check_source(row, session)
    if 'first_result' in session:
        raise ValueError('first_round_already_consumed')
    qs = session['requests']
    replies = _responses(qs, raw)
    updated = deepcopy(session)
    if session['route'] == 'llm_stage_a':
        result = evaluate_packet_set(row, qs, replies)
        updated['feedback_requests'] = quantitative_feedback(row, qs, replies)
    else:
        result = {'case_id': row['case_id'], 'stage_a_success': False,
                  'topology_success': False, 'final_actions': [], 'fallback_used': False}
        updated['feedback_requests'] = []
        if session['route'] == 'unsupported':
            result['failure'] = session['reason']
        else:
            committed_end = max((s.register_end for s in committed_spans(row['observed_actions'])), default=0)
            final, reason = complete_after_patch(row['observed_actions'], row['topology_contract'],
                protected_end=committed_end, preserve_reference_parameters=True)
            result.update(final_actions=final or [], topology_success=final is not None,
                          completion_status=reason, protected_end=committed_end)
            if final is None:
                result['failure'] = reason
    result.update(route=session['route'], llm_calls=len(qs), calls=len(qs),
                  output_tokens=sum(r.get('output_tokens', 0) for r in raw))
    updated['first_result'] = result
    return updated


def finalize_session(row, session, raw):
    _check_source(row, session)
    if session.get('finalized'):
        raise ValueError('session_already_finalized')
    requests = session['feedback_requests']
    replies = _responses(requests, raw)
    result = deepcopy(session['first_result'])
    if requests:
        if result['stage_a_success'] or any(q['attempt'] != 1 for q in requests):
            raise ValueError('invalid_feedback_state')
        result = evaluate_packet_set(row, requests, replies)
        calls = session['first_result']['llm_calls'] + len(requests)
        result.update(route=session['route'], llm_calls=calls, calls=calls,
            output_tokens=session['first_result']['output_tokens']+sum(r.get('output_tokens', 0) for r in raw))
    if result['llm_calls'] > 2 * len(session['requests']):
        raise ValueError('call_budget_exceeded')
    return {**deepcopy(session), 'finalized': True, 'final_result': result}
