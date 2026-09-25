"""Verifier-localized LLM terminal-marker edits before profile repair/Stage B."""
from copy import deepcopy
from difflib import SequenceMatcher
import json

from .actions import End, EndSketch
from .cad_input_normalization import normalize_history
from .lm.parsing import parse_action_line
from .natural_repair_recall import evaluate_repaired_history
from .stage_a_packet_policy import PUBLIC_SOURCE_FIELDS
from .stage_a_reference_intent_policy import prepare_reference_intent, submit_reference_intent
from .verifier import TopoVerifier

MAX_TERMINAL_EDITS = 8


def terminal_certificate(row):
    lines = row['observed_actions']
    verifier = TopoVerifier()
    state = verifier.initial_state()
    for index, line in enumerate(lines):
        action = parse_action_line(line)
        if action is None:
            return None
        result = verifier.step(state, action)
        if not result.valid or result.next_state is None:
            if (isinstance(action, End) and state.stack == ['sketch']
                    and state.pending_face is None and state.extrusions
                    and any(not isinstance(parse_action_line(s), (End, EndSketch))
                            for s in lines[index+1:])):
                closed_prefix = lines[:index] + ['EndSketch', 'End']
                if evaluate_repaired_history(closed_prefix, row['topology_contract'])['valid_and_intent_satisfied']:
                    return None
                candidates = [i for i, s in enumerate(lines) if parse_action_line(s) == End()
                              and any(parse_action_line(t) != End() for t in lines[i+1:])]
                return {'committed_terminal_step': index, 'next_action_step': index+1,
                        'registered_profiles': sorted(state.profiles),
                        'completed_extrusions': len(state.extrusions),
                        'contract_satisfied_at_stop': False, 'candidate_terminal_steps': candidates,
                        'terminal_rejected': True, 'active_stack': list(state.stack)}
            return None
        state = result.next_state
        if isinstance(action, End):
            if not any(parse_action_line(s) != End() for s in lines[index+1:]):
                return None
            if evaluate_repaired_history(lines[:index+1], row['topology_contract'])['valid_and_intent_satisfied']:
                return None
            candidates = [i for i, s in enumerate(lines) if parse_action_line(s) == End()
                          and any(parse_action_line(t) != End() for t in lines[i+1:])]
            return {'committed_terminal_step': index, 'next_action_step': index+1,
                    'registered_profiles': sorted(state.profiles), 'completed_extrusions': len(state.extrusions),
                    'contract_satisfied_at_stop': False, 'candidate_terminal_steps': candidates}
    return None


def _request(state, attempt, feedback=None):
    row, certificate = state['normalized_source'], state['certificate']
    public = {'history': [{'step': i, 'action': s} for i, s in enumerate(row['observed_actions'])],
              'topology_contract': row['topology_contract'], 'feature_plan': row['feature_plan'],
              'design_brief': row['design_brief'], 'replay_state': certificate,
              'max_terminal_edits': MAX_TERMINAL_EDITS}
    if feedback is not None:
        public['previous_attempt'] = feedback
    note = ('The indicated End was rejected because the sketch is still open. '
            'It is not an accepted terminal; later profile geometry remains in the observed suffix. '
            if certificate.get('terminal_rejected') else '')
    return {'case_id': row['case_id'], 'source_uid': row['case_id']+f'::terminal_boundary::{attempt}',
            'terminal_boundary': True, 'attempt': attempt,
            'candidate_terminal_steps': certificate['candidate_terminal_steps'],
            'prompt': note+('Repair premature termination of this CAD history. Replay reached an End before the '
                       'requested topology was complete, but more construction actions follow. Decide which '
                       'listed End steps must be removed to execute the intended remaining features. '
                       'Do not modify, add or remove geometry, profile registrations, extrusion parameters '
                       'or the final End. Return ONLY {"remove_terminal_steps":[indices]} using distinct '
                       'listed indices, at most 8. Return an empty list to abstain. An index is a step '
                       'in the ORIGINAL history below, including on retry.\nINPUT:\n')+json.dumps(public, sort_keys=True)}


def _failure(state, reason):
    state = deepcopy(state)
    state.update(phase='complete', requests=[], result={'case_id': state['source']['case_id'],
                 'route': 'llm_terminal_boundary', 'failure': reason, 'final_actions': [],
                 'stage_a_success': False, 'topology_success': False, 'action_lcs': 0.,
                 'llm_calls': state['boundary_calls'], 'output_tokens': state['boundary_tokens'],
                 'fallback_used': False, 'symbolic_stage_a_fallback': False})
    state['result']['input_normalization_audit'] = state.get('input_normalization_audit', [])
    return state


def _child(state, child):
    state = deepcopy(state)
    state.update(child=child, phase='child', requests=child['requests'])
    state['_actual_calls'] = state['boundary_calls'] + child.get('_actual_calls', 0)
    state['_actual_tokens'] = state['boundary_tokens'] + child.get(
        '_actual_tokens', child.get('result', {}).get('output_tokens', 0))
    if child['phase'] == 'complete':
        result = deepcopy(child['result'])
        result['llm_calls'] = state['boundary_calls'] + result.get('llm_calls', 0)
        result['calls'] = result['llm_calls']
        result['output_tokens'] = state['boundary_tokens'] + result.get('output_tokens', 0)
        result['terminal_boundary_calls'] = state['boundary_calls']
        result['terminal_boundary_patch'] = state.get('terminal_boundary_patch')
        result['input_normalization_audit'] = state['input_normalization_audit']
        result['symbolic_stage_a_fallback'] = False
        if state.get('terminal_boundary_patch'):
            result['route'] = 'llm_terminal_boundary_then_'+result.get('route', 'profile_repair')
            result['stage_a_boundary_accepted'] = True
        original, final = state['source']['observed_actions'], result.get('final_actions', [])
        result['action_lcs'] = sum(b.size for b in SequenceMatcher(a=original, b=final, autojunk=False).get_matching_blocks())/max(1, len(original))
        state.update(phase='complete', requests=[], result=result)
    return state


def prepare_terminal_intent(row):
    source = {k: deepcopy(row[k]) for k in PUBLIC_SOURCE_FIELDS}
    state = {'source': source, 'boundary_calls': 0, 'boundary_tokens': 0,
             '_actual_calls': 0, '_actual_tokens': 0}
    try:
        lines, audit = normalize_history(source['observed_actions'])
    except ValueError as error:
        return _failure(state, str(error))
    normalized = {**deepcopy(source), 'observed_actions': lines}
    state.update(normalized_source=normalized, input_normalization_audit=audit)
    certificate = terminal_certificate(normalized)
    if certificate is not None:
        state.update(certificate=certificate, phase='terminal_boundary', attempt=0)
        state['requests'] = [_request(state, 0)]
        return state
    return _child(state, prepare_reference_intent(normalized))


def submit_terminal_intent(row, state, raw):
    if any(row[k] != state['source'][k] for k in PUBLIC_SOURCE_FIELDS):
        raise ValueError('stale_terminal_source')
    if state['phase'] == 'child':
        return _child(state, submit_reference_intent(state['normalized_source'], state['child'], raw))
    if state['phase'] != 'terminal_boundary':
        raise ValueError('terminal_session_complete')
    if len(raw) != 1 or raw[0]['source_uid'] != state['requests'][0]['source_uid']:
        raise ValueError('terminal_response_coverage')
    state = deepcopy(state)
    state['boundary_calls'] += 1
    state['_actual_calls'] = state['boundary_calls']
    state['boundary_tokens'] += raw[0].get('output_tokens', 0)
    state['_actual_tokens'] = state['boundary_tokens']
    try:
        value = json.loads(raw[0]['response'])
        if not isinstance(value, dict) or set(value) != {'remove_terminal_steps'}:
            raise ValueError('terminal_patch_schema')
        selected = value['remove_terminal_steps']
        if (not isinstance(selected, list) or any(type(i) is not int for i in selected)
                or len(selected) != len(set(selected)) or len(selected) > MAX_TERMINAL_EDITS
                or not set(selected).issubset(state['certificate']['candidate_terminal_steps'])):
            raise ValueError('terminal_patch_outside_scope')
        if not selected:
            return _failure(state, 'terminal_patch_abstention')
    except (ValueError, TypeError) as error:
        return _failure(state, str(error))
    original = state['normalized_source']
    patched = {**deepcopy(original), 'observed_actions': [s for i, s in enumerate(original['observed_actions']) if i not in selected]}
    remaining = terminal_certificate(patched)
    if remaining is not None:
        if state['attempt'] == 0:
            state.update(attempt=1, requests=[_request(state, 1, {'selected_steps': selected,
                         'replay_failure': 'premature_terminal_remains'})])
            return state
        return _failure(state, 'premature_terminal_remains')
    state['terminal_boundary_patch'] = {'removed_original_steps': sorted(selected),
        'geometry_actions_changed': False, 'selection_source': 'model_response',
        'model_response': raw[0]['response'], 'max_edits': MAX_TERMINAL_EDITS}
    state['normalized_source'] = patched
    return _child(state, prepare_reference_intent(patched))
