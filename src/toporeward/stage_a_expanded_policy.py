"""Sequential LLM profile retention, local loop repair, and protected Stage B.

Unlike a fallback, retention is selected from the input obligation before any
inner-loop generation. Every model decision and dependency deletion is recorded.
"""
from copy import deepcopy
from difflib import SequenceMatcher

from .stage_a_multi_transaction import committed_spans
from .stage_a_packet_policy import (
    PUBLIC_SOURCE_FIELDS, prepare_session, accept_first, finalize_session)
from .stage_a_profile_retention import prepare_retention, apply_retention


def _check(row, state):
    if any(row[key] != state['source'][key] for key in PUBLIC_SOURCE_FIELDS):
        raise ValueError('stale_expanded_source')


def _finish(state, result):
    result = deepcopy(result)
    if state.get('retention_result'):
        result.update(retention_result=state['retention_result'],
                      llm_calls=result['llm_calls']+1,
                      output_tokens=result.get('output_tokens', 0)+state['retention_tokens'])
        result['calls'] = result['llm_calls']
        result['route'] = 'llm_retention_and_local' if state['inner_session']['requests'] else 'llm_retention_then_stage_b'
        # An accepted LLM retention decision is Stage A, not Stage B-only credit.
        if result['topology_success']:
            result['stage_a_success'] = True
    original = state['source']['observed_actions']
    final = result.get('final_actions') or []
    result['action_lcs'] = sum(b.size for b in SequenceMatcher(
        a=original, b=final, autojunk=False).get_matching_blocks()) / max(1, len(original))
    return {**state, 'phase': 'complete', 'requests': [], 'result': result}


def _open_inner(state):
    session = prepare_session(state['working_source'])
    state = {**state, 'inner_session': session}
    if session['route'] != 'llm_stage_a':
        first = accept_first(state['working_source'], session, [])
        final = finalize_session(state['working_source'], first, [])
        return _finish(state, final['final_result'])
    return {**state, 'phase': 'inner_first', 'requests': session['requests']}


def prepare_expanded(row):
    source = {key: deepcopy(row[key]) for key in PUBLIC_SOURCE_FIELDS}
    state = {'source': source, 'working_source': deepcopy(source),
             'retention_result': None, 'retention_tokens': 0, 'symbolic_stage_a_fallback': False}
    desired = source['topology_contract'].get('profiles')
    if isinstance(desired, list) and len(committed_spans(source['observed_actions'])) > len(desired):
        try:
            request = prepare_retention(source)
        except ValueError as error:
            return {**state, 'phase': 'complete', 'requests': [], 'result': {
                'case_id': source['case_id'], 'route': 'unsupported', 'failure': str(error),
                'topology_success': False, 'stage_a_success': False, 'llm_calls': 0,
                'calls': 0, 'final_actions': [], 'action_lcs': 0., 'fallback_used': False}}
        return {**state, 'phase': 'retention', 'requests': [request]}
    return _open_inner(state)


def submit_expanded(row, state, raw):
    _check(row, state)
    if state['phase'] == 'complete':
        raise ValueError('expanded_session_already_complete')
    state = deepcopy(state)
    if state['phase'] == 'retention':
        q = state['requests'][0]
        if len(raw) != 1 or raw[0]['source_uid'] != q['source_uid']:
            raise ValueError('retention_response_coverage')
        try:
            result = apply_retention(state['source'], q, raw[0]['response'])
        except (ValueError, TypeError, KeyError) as error:
            return {**state, 'phase': 'complete', 'requests': [], 'result': {
                'case_id': row['case_id'], 'route': 'llm_retention', 'failure': str(error),
                'stage_a_success': False, 'topology_success': False, 'llm_calls': 1, 'calls': 1,
                'output_tokens': raw[0].get('output_tokens', 0),
                'final_actions': [], 'action_lcs': 0., 'fallback_used': False}}
        state['retention_result'] = result
        state['retention_tokens'] = raw[0].get('output_tokens', 0)
        state['working_source']['observed_actions'] = result['observed_actions']
        return _open_inner(state)
    if state['phase'] == 'inner_first':
        first = accept_first(state['working_source'], state['inner_session'], raw)
        state['inner_session'] = first
        if first['feedback_requests']:
            return {**state, 'phase': 'inner_feedback', 'requests': first['feedback_requests']}
        final = finalize_session(state['working_source'], first, [])
        return _finish(state, final['final_result'])
    if state['phase'] == 'inner_feedback':
        final = finalize_session(state['working_source'], state['inner_session'], raw)
        return _finish(state, final['final_result'])
    raise ValueError('unknown_expanded_session_phase')
