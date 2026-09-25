"""Quantitative rejection feedback for the model's own profile patch.

Only observed geometry and the actual proposal are inspected. This module never
searches for a replacement, uses a target solid, or changes verifier tolerances.
"""
from copy import deepcopy
import json
import math

from .lm.parsing import parse_action_line
from .stage_a_multi_transaction import prepare_multi_request, evaluate_multi_response
from .stage_a_profile_grounding import decode_grounded
from .stage_a_profile_packets import prepare_profile_packets, merge_packet_responses
from .verifier.core import TopoVerifier
from .verifier.geometry import EPS, circle_as_polygon, polygon_segments, polygons_cross, point_in_polygon, signed_area


def _distance_to_boundary(point, polygon):
    distances = []
    for a, b in polygon_segments(polygon):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        t = 0. if length2 == 0 else max(0., min(1., (
            (point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2))
        distances.append(math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy))
    return min(distances)


def _observed_face(row, profile_id):
    verifier = TopoVerifier()
    state = verifier.initial_state()
    for line in row['observed_actions']:
        step = verifier.step(state, parse_action_line(line))
        if not step.valid:
            break
        state = step.next_state
        if profile_id in state.profiles:
            return state.profiles[profile_id]
    raise ValueError('profile_not_committed')


def proposal_diagnostics(row, packet, response):
    """Measure this proposal; values are diagnostics, not suggested patch parameters."""
    data = json.loads(response)
    if not isinstance(data, dict) or not isinstance(data.get('profiles'), list) or len(data['profiles']) != 1:
        raise ValueError('packet_abstention_or_multiple_profiles')
    proposal = data['profiles'][0]
    if proposal['profile_id'] != packet['profile_id']:
        raise ValueError('wrong_packet_profile')
    face = _observed_face(row, packet['profile_id'])
    outer = next(loop.points for loop in face.loops if loop.kind == 'outer')
    keep = proposal['keep_loop_ids']
    obstacles = [(f'inner_{i}', loop.points) for i, loop in enumerate(
        loop for loop in face.loops if loop.kind == 'inner') if f'inner_{i}' in keep]
    scope = packet['scopes'][0]
    measurements = []
    for i, added in enumerate(proposal['add_loops']):
        world = decode_grounded(row, {'scope': scope}, json.dumps({
            'profile_id': packet['profile_id'], **added}))['loop']
        cx, cy = world['center']
        if world['shape'] == 'circle':
            polygon = circle_as_polygon((cx, cy), world['radius'])
        else:
            w, h = world['width'], world['height']
            polygon = [(cx-w/2, cy-h/2), (cx+w/2, cy-h/2),
                       (cx+w/2, cy+h/2), (cx-w/2, cy+h/2)]
        verifier = TopoVerifier()
        outside = [p for p in polygon if not verifier._strictly_inside_polygon(p, outer)]
        collisions = [name for name, points in obstacles if polygons_cross(points, polygon)
                      or any(point_in_polygon(p, points) for p in polygon)
                      or any(point_in_polygon(p, polygon) for p in points)]
        center_inside = verifier._strictly_inside_polygon((cx, cy), outer)
        measurement = {
            'proposal_index': i, 'decoded_world_geometry': world,
            'center_strictly_inside_outer': center_inside,
            'center_to_outer_boundary_distance_world': _distance_to_boundary((cx, cy), outer),
            'polygon_area_world': abs(signed_area(polygon)),
            'area_above_verifier_tolerance': abs(signed_area(polygon)) > EPS,
            'outside_or_touching_vertex_count': len(outside),
            'outside_or_touching_examples_world': outside[:4],
            'boundary_crossing_or_touching': polygons_cross(outer, polygon),
            'intersecting_retained_or_preceding_new_loops': collisions,
        }
        if world['shape'] == 'circle':
            measurement['circle_radius_world'] = world['radius']
        measurements.append(measurement)
        obstacles.append((f'new_{i}', polygon))
    return {'profile_id': packet['profile_id'], 'outer_vertices_world': outer,
            'minimum_polygon_area_world_strict': EPS,
            'boundary_representation': 'same_24_vertex_circle_and_arc_polyline_as_verifier',
            'proposal_measurements': measurements, 'host_geometry_search': False}


def evaluate_packet_set(row, packets, responses):
    """Atomic Stage A followed by the existing prefix-preserving Stage B."""
    try:
        merged = merge_packet_responses(row, packets, responses)
    except (ValueError, KeyError, TypeError) as error:
        return {'case_id': row['case_id'], 'stage_a_success': False,
                'topology_success': False, 'final_actions': [], 'action_lcs': 0.,
                'fallback_used': False, 'failure': str(error)}
    result = evaluate_multi_response(row, prepare_multi_request(row), merged)
    result['merged_model_response'] = merged
    return result


def quantitative_feedback(row, packets, responses):
    if any(q['attempt'] != 0 for q in packets):
        raise ValueError('feedback_budget_exhausted')
    result = evaluate_packet_set(row, packets, responses)
    # No retry on geometry-to-target mismatch or on an accepted Stage A patch.
    if result['stage_a_success']:
        return []
    retries = []
    for packet, response in zip(packets, responses):
        diagnostic = {'failure': result['failure'], 'failure_type': result.get('failure_type'),
                      'this_profile_response': response}
        try:
            diagnostic['geometry_diagnosis'] = proposal_diagnostics(row, packet, response)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            diagnostic['proposal_parse_error'] = str(error)
        instructions, payload = packet['prompt'].split('\nINPUT:\n', 1)
        public = json.loads(payload)
        public['last_rejection'] = diagnostic
        extra = (
            'This is the single permitted revision after a rejected Stage A patch. '
            'The diagnosis measures YOUR previous proposal, not a suggested replacement. '
            'If its center is outside, reducing only the radius cannot fix that center. '
            'Use the actual outer boundary, not just the box, to reason about placement. '
            'For a circle, its radius must be less than the distance from its inside center '
            'to every boundary edge, with positive clearance. Avoid retained/new loops. '
            'Respect explicit dimensions in the brief; choose feasible dimensions only when unspecified. '
            'Preserve an already-correct profile when rejection belongs to another profile. '
        )
        retry = deepcopy(packet)
        retry.update(attempt=1, source_uid=packet['source_uid']+'::feedback',
                     prompt=instructions+extra+'\nINPUT:\n'+json.dumps(public, sort_keys=True))
        retries.append(retry)
    return retries
