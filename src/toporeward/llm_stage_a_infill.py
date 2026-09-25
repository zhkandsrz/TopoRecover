"""Single missing-loop infill with a verifier-selected, fixed insertion boundary."""
from collections import Counter
import json

from .actions import AddArc, AddCircle, AddLine, EndLoop, StartLoop
from .llm_stage_a import evaluate_response, profile_diagnostics
from .lm.parsing import parse_action_line
from .strong_repair_baselines import _dsl_text, runtime_trace
from .topology_transaction_repair import _profile_transaction_spans

MODES = ('slot_only', 'slot_obligations')


def locate_slot(row):
    history = row['observed_actions']
    diagnostics = profile_diagnostics(history, row['topology_contract'])
    if len(diagnostics) != 1:
        raise ValueError('requires_one_mismatched_profile')
    d = diagnostics[0]
    actual, required = Counter(d['observed_roles']), Counter(d['required_roles'])
    if required != actual + Counter(inner=1):
        raise ValueError('requires_exactly_one_missing_inner_loop')
    trace = runtime_trace(history)
    stop = len(history) if trace.first_rejected_step is None else trace.first_rejected_step
    spans = _profile_transaction_spans(history, stop)
    span = next(s for s in spans if s.profile_id == d['profile_id'])
    return {'offset': span.end_face, 'profile_id': span.profile_id,
            'observed_roles': d['observed_roles'], 'required_roles': d['required_roles']}


def prepare_infill_request(row, mode='slot_obligations'):
    if mode not in MODES:
        raise ValueError('unknown_infill_mode')
    slot = locate_slot(row)
    history = row['observed_actions']
    trace = runtime_trace(history)
    visible = {
        'design_brief': row['design_brief'], 'feature_plan': row['feature_plan'],
        'readonly_before_slot': history[:slot['offset']],
        'readonly_after_slot': history[slot['offset']:],
        'active_profile': slot['profile_id'],
        'runtime_failure_type': trace.failure_type,
    }
    if mode == 'slot_obligations':
        visible['profile_obligation'] = {k: slot[k] for k in ('observed_roles', 'required_roles')}
    prompt = (
        'Restore the missing inner loop in the fixed slot between readonly_before_slot and '
        'readonly_after_slot. The slot is already localized. Do not choose edit offsets or '
        'rewrite any existing action. Follow the supplied geometric requirements and frozen '
        'feature plan. Return only JSON {"replacement":["one complete CAD action per string"]}. '
        'Output the complete missing loop: StartLoop(kind=inner), curve actions, EndLoop. '
        'Use numeric literals, not arithmetic expressions. Do not split an action and its '
        'arguments into separate strings. At most32 actions. Return {"replacement":[]} to '
        'abstain. No prose or code fences. The host only inserts your actions; it does not '
        'add delimiters, calculate coordinates or replace failed proposals. A downstream '
        'Stage B may repair execution references after the loop is valid.\nSUPPORTED DSL:\n'
        + _dsl_text() + '\nINPUT:\n' + json.dumps(visible, sort_keys=True)
    )
    return {
        'source_uid': f"{row['case_id']}::{mode}", 'case_id': row['case_id'], 'mode': mode,
        'prompt': prompt, 'slot': slot,
        'input_audit': {'target_actions_visible': False, 'teacher_patch_visible': False,
                        'target_geometry_visible': False, 'supplied_design_dimensions_visible': True,
                        'host_selects_slot': True, 'slot_source': 'observed_history_and_contract',
                        'host_generates_geometry': False, 'structured_obligations_visible': mode == 'slot_obligations'},
    }


def parse_infill(response):
    try:
        payload = json.loads(response)
    except (TypeError, ValueError) as error:
        raise ValueError('infill_json_invalid') from error
    if not isinstance(payload, dict) or set(payload) != {'replacement'}:
        raise ValueError('infill_schema_invalid')
    lines = payload['replacement']
    if not isinstance(lines, list) or not lines:
        raise ValueError('infill_abstention_or_not_list')
    if len(lines) > 32 or not all(isinstance(s, str) for s in lines):
        raise ValueError('infill_action_budget_or_type')
    actions = [parse_action_line(s) for s in lines]
    if any(a is None for a in actions):
        raise ValueError('infill_unparseable_action')
    if (len(actions) < 3 or not isinstance(actions[0], StartLoop) or actions[0].kind != 'inner'
            or not isinstance(actions[-1], EndLoop)
            or not all(isinstance(a, (AddCircle, AddLine, AddArc)) for a in actions[1:-1])):
        raise ValueError('infill_not_one_complete_inner_loop')
    return lines


def evaluate_infill(row, request, response):
    failed = {'case_id': row['case_id'], 'mode': request['mode'], 'patch_accepted': False,
              'stage_a_success': False, 'topology_success': False, 'fallback_used': False,
              'final_actions': [], 'action_lcs': 0.0}
    try:
        slot = locate_slot(row)
        if slot != request['slot']:
            raise ValueError('infill_slot_mismatch')
        lines = parse_infill(response)
    except (ValueError, TypeError, KeyError) as error:
        return {**failed, 'failure': str(error)}
    index = slot['offset']
    patch = {'edits': [{'start': index, 'end': index, 'replacement': lines}]}
    bounded = {**request, 'editable_regions': [{'start': index, 'end': index}]}
    return evaluate_response(row, bounded, json.dumps(patch), preserve_reference_parameters=True)
