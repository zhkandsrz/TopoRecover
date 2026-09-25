"""Observed-curve grounding for LLM patches, without feasible-candidate search."""
from copy import deepcopy
import json
import math

from .actions import AddArc, AddCircle, AddLine, EndLoop, StartLoop
from .lm.parsing import parse_action_line
from .llm_stage_a_geometry import number, positive
from .stage_a_parameter_policy import prepare_parameter_request, evaluate_parameter_response
from .stage_a_dual_frame_policy import normalized_loops
from .strong_repair_baselines import runtime_trace
from .verifier.geometry import EPS, arc_as_polyline, circle_as_polygon


def observed_profile_frame(row, scope):
    lines = row['observed_actions'][scope['source_start']:scope['source_end']]
    outer, current, outer_count = [], None, 0
    for line in lines:
        action = parse_action_line(line)
        if isinstance(action, StartLoop):
            current = action.kind
            outer_count += current == 'outer'
        elif isinstance(action, EndLoop):
            current = None
        elif current == 'outer':
            if isinstance(action, AddLine):
                outer.extend([action.start, action.end])
            elif isinstance(action, AddCircle):
                outer.extend(circle_as_polygon(action.center, action.radius))
            elif isinstance(action, AddArc):
                outer.extend(arc_as_polyline(action.start, action.mid, action.end))
    if outer_count != 1 or not outer:
        raise ValueError('requires_one_observed_outer_loop')
    if not all(math.isfinite(x) for p in outer for x in p):
        raise ValueError('nonfinite_observed_boundary')
    lo = [min(p[d] for p in outer) for d in (0, 1)]
    hi = [max(p[d] for p in outer) for d in (0, 1)]
    w, h = positive(hi[0] - lo[0]), positive(hi[1] - lo[1])
    return {'origin': lo, 'width': w, 'height': h, 'shorter_side': min(w, h),
            'bounds': [*lo, *hi], 'provenance': 'observed_outer_curves_only',
            'arc_bounds': 'same_polyline_approximation_as_TopoVerifier',
            'box_is_not_feasible_region': True}


def prepare_grounded_request(row):
    q = prepare_parameter_request(row)
    frame = observed_profile_frame(row, q['scope'])
    scope = q['scope']
    lines = row['observed_actions'][scope['source_start']:scope['source_end']]
    # Public numerical tolerances are candidate-independent, not a chosen radius.
    circle_coefficient = 12 * math.sin(math.pi / 12)
    min_radius = max(EPS, math.sqrt(EPS / circle_coefficient))
    tolerances = {'minimum_loop_area_world_strict': EPS,
                  'minimum_circle_radius_world_strict': min_radius,
                  'minimum_circle_radius_fraction_strict': min_radius / frame['shorter_side'],
                  'minimum_rectangle_fraction_area_strict': EPS / (frame['width'] * frame['height'])}
    public = {'design_brief': row['design_brief'], 'feature_plan': row['feature_plan'],
              'profile_id': scope['profile_id'], 'observed_roles': scope['observed_roles'],
              'required_roles': scope['required_roles'], 'localized_profile': lines,
              'observed_outer_frame': frame, 'observed_loops_in_frame': normalized_loops(lines, frame),
              'public_verifier_tolerances': tolerances}
    q['prompt'] = (
        'Insert ONE missing inner loop in the localized CAD profile. Preserve existing curves and features. '
        'Choose circle or rectangle and its parameters according to the brief. Return JSON only. '
        'The observed loops describe actual existing geometry, NOT the desired patch. '
        'The outer bounding box is a coordinate system, NOT a guarantee of containment. '
        'Use the outer curve shape: a point in the bounding box can be outside a circle, triangle or concave outline. '
        'Avoid all existing inner loops. Keep positive clearance and exceed the public numerical tolerances. '
        'For explicit world dimensions, use coordinate_frame="world" and copy the specified quantities faithfully '
        '(diameter must be divided by two). Otherwise prefer coordinate_frame="profile_fraction". '
        'In profile_fraction, center=[u,v] is dimensionless: [0.5,0.5] is the box CENTER, regardless of world scale. '
        'Circle radius is divided by the shorter box side; rectangle width/height by box width/height. '
        'Do NOT put world coordinates into fractional fields or normalize twice. '
        'For a circular outer loop, its displayed center_fraction is the outer circle center; '
        'it is not the lower-left box corner. For polygons, follow the displayed connected start/end vertices. '
        'Unspecified dimensions are a feasibility choice, not recoverable original intent. '
        'The host only converts coordinates and constructs your exact proposal; it never moves, shrinks, '
        'searches or substitutes geometry. One rejected proposal may receive one actual verifier diagnosis. '
        'Return {"profile_id":"<given>","coordinate_frame":"world" or "profile_fraction",'
        '"loop":{"shape":"circle","center":[x,y],"radius":r}} or the same fields with '
        '"loop":{"shape":"rectangle","center":[x,y],"width":w,"height":h}.\nINPUT:\n'
    ) + json.dumps(public, sort_keys=True)
    q.update(attempt=0)
    q['input_audit'].update(frame_from_all_observed_curves=True, candidate_feasibility_search=False,
                            target_geometry_visible=False, candidate_specific_geometry_features=False)
    return q


def decode_grounded(row, q, response):
    data = json.loads(response)
    if not isinstance(data, dict) or set(data) != {'profile_id', 'coordinate_frame', 'loop'}:
        raise ValueError('grounded_patch_schema')
    if data['profile_id'] != q['scope']['profile_id']:
        raise ValueError('wrong_profile_id')
    loop = data['loop']
    fields = {'circle': {'shape', 'center', 'radius'},
              'rectangle': {'shape', 'center', 'width', 'height'}}
    if (not isinstance(loop, dict) or not isinstance(loop.get('shape'), str) or
            loop['shape'] not in fields or set(loop) != fields[loop['shape']]):
        raise ValueError('grounded_loop_schema')
    if not isinstance(loop['center'], list) or len(loop['center']) != 2:
        raise ValueError('invalid_center')
    center = list(map(number, loop['center']))
    lengths = {k: positive(loop[k]) for k in fields[loop['shape']] - {'shape', 'center'}}
    if data['coordinate_frame'] == 'world':
        return {'profile_id': data['profile_id'], 'loop': {'shape': loop['shape'], 'center': center, **lengths}}
    if data['coordinate_frame'] != 'profile_fraction':
        raise ValueError('unknown_coordinate_frame')
    if not all(0 < v < 1 for v in center):
        raise ValueError('local_center_outside')
    frame = observed_profile_frame(row, q['scope'])
    result = {'shape': loop['shape'], 'center': [frame['origin'][i] + center[i] * frame[k]
                                                 for i, k in enumerate(('width', 'height'))]}
    if loop['shape'] == 'circle':
        result['radius'] = lengths['radius'] * frame['shorter_side']
    else:
        result.update(width=lengths['width'] * frame['width'], height=lengths['height'] * frame['height'])
    return {'profile_id': data['profile_id'], 'loop': result}


def evaluate_grounded_response(row, q, response):
    try:
        world = decode_grounded(row, q, response)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return {'case_id': row['case_id'], 'mode': q['mode'], 'stage_a_success': False,
                'topology_success': False, 'patch_accepted': False, 'final_actions': [],
                'action_lcs': 0.0, 'failure': str(error), 'fallback_used': False}
    result = evaluate_parameter_response(row, q, json.dumps(world))
    result.update(model_world_parameters=world, model_parameter_response=response,
                  geometry_clipped_or_replaced=False, attempt=q['attempt'])
    return result


def grounded_feedback(row, q, response):
    if q['attempt'] != 0:
        raise ValueError('feedback_budget_exhausted')
    result = evaluate_grounded_response(row, q, response)
    if result['stage_a_success']:
        return None
    diagnosis = {'rejected_response': response, 'failure': result['failure']}
    if result.get('patched_actions'):
        trace = runtime_trace(result['patched_actions']).to_dict()
        diagnosis.update(failure_type=trace['failure_type'], repair_hint=trace['repair_hint'])
    instructions, payload = q['prompt'].split('\nINPUT:\n', 1)
    visible = json.loads(payload)
    visible['last_rejection'] = diagnosis
    revised = deepcopy(q)
    revised.update(attempt=1, source_uid=q['source_uid'] + '::feedback',
                   prompt=instructions + '\nINPUT:\n' + json.dumps(visible, sort_keys=True))
    return revised
