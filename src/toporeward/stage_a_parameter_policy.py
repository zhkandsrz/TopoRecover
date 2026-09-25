"""Learned geometry proposal between verifier localization and existing Stage B."""
import json

from .llm_stage_a_geometry import compile_geometry_response, prepare_geometry_request, scope_for_case
from .llm_stage_a_infill import evaluate_infill, locate_slot

MODES = ('localized_parameters', 'full_history_parameters')


def prepare_parameter_request(row, mode='localized_parameters'):
    if mode not in MODES:
        raise ValueError('unknown_parameter_policy_mode')
    request = prepare_geometry_request(row, 'loop_parameters')
    if mode == 'full_history_parameters':
        instructions, payload = request['prompt'].split('\nINPUT:\n', 1)
        visible = json.loads(payload)
        visible['readonly_full_history'] = row['observed_actions']
        request['prompt'] = instructions+'\nINPUT:\n'+json.dumps(visible, sort_keys=True)
    request.update(mode=mode, source_uid=f"{row['case_id']}::{mode}", slot=locate_slot(row))
    request['input_audit'].update(model_chooses_shape_and_dimensions=True,
        fixed_geometry_constructor=True, host_geometry_search=False,
        target_actions_visible=False, teacher_patch_visible=False)
    return request


def compile_parameter_patch(row, request, response):
    if request['mode'] not in MODES or request['scope'] != scope_for_case(row) or request['slot'] != locate_slot(row):
        raise ValueError('parameter_policy_scope_mismatch')
    data = json.loads(response)
    if not isinstance(data, dict) or set(data) != {'profile_id', 'loop'}:
        raise ValueError('parameter_policy_schema')
    loop = data['loop']
    if not isinstance(loop, dict):
        raise ValueError('loop_schema')
    fields = {'circle': {'shape', 'center', 'radius'},
              'rectangle': {'shape', 'center', 'width', 'height'}}
    shape = loop.get('shape')
    if not isinstance(shape, str) or shape not in fields or set(loop) != fields[shape]:
        raise ValueError('loop_fields_or_shape')
    return compile_geometry_response(row, {**request, 'mode': 'loop_parameters'}, response)


def evaluate_parameter_response(row, request, response):
    try:
        patch = compile_parameter_patch(row, request, response)
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        return {'case_id': row['case_id'], 'mode': request['mode'], 'patch_accepted': False,
                'stage_a_success': False, 'topology_success': False, 'fallback_used': False,
                'final_actions': [], 'action_lcs': 0.0, 'failure': str(error)}
    edit = patch['edits'][0]
    if edit['start'] != request['slot']['offset'] or edit['end'] != edit['start']:
        raise ValueError('constructor_changed_localization')
    result = evaluate_infill(row, request, json.dumps({'replacement': edit['replacement']}))
    result['compiled_lm_patch'] = patch
    result['fixed_geometry_constructor'] = True
    result['geometry_source'] = 'llm_response'
    return result
