"""Observed boundary equations for LM-authored patches; no feasible-point search."""
from copy import deepcopy
import json
import math

from .actions import AddCircle
from .lm.parsing import parse_action_line
from .stage_a_packet_feedback import _observed_face
from .stage_a_profile_grounding import observed_profile_frame
from .stage_a_expanded_policy import prepare_expanded, submit_expanded
from .verifier.geometry import EPS, polygon_segments, signed_area


def boundary_context(row, packet):
    scope = packet['scopes'][0]
    frame = observed_profile_frame(row, scope)
    face = _observed_face(row, packet['profile_id'])
    outer = next(loop.points for loop in face.loops if loop.kind == 'outer')
    w, h = frame['width'], frame['height']
    vertices = [((x-frame['origin'][0])/w, (y-frame['origin'][1])/h) for x, y in outer]
    orientation = 1 if signed_area(vertices) > 0 else -1
    planes = []
    for a, b in polygon_segments(vertices):
        dx, dy = b[0]-a[0], b[1]-a[1]
        # For a CCW boundary, its interior lies left of every directed edge.
        A, B = -orientation*dy, orientation*dx
        C = orientation*(dy*a[0]-dx*a[1])
        norm = math.hypot(A, B)
        if norm:
            planes.append([A/norm, B/norm, C/norm])
    convex = all(A*x+B*y+C >= -1e-10 for A, B, C in planes for x, y in vertices)
    result = {
        'source': 'observed_outer_only_not_target_or_candidate_search',
        'frame': frame,
        'normalized_outer_vertices': vertices,
        'convex': convex,
        'minimum_circle_radius_fraction_strict': math.sqrt(EPS/(12*math.sin(math.pi/12)))/frame['shorter_side'],
        'minimum_rectangle_fraction_area_strict': EPS/(w*h),
        'host_selects_patch_geometry': False,
    }
    if convex:
        result['halfplanes_Au_Bv_C_positive'] = planes
        result['circle_condition_each_edge'] = 'A*u+B*v+C > radius_fraction*shorter_side*sqrt((A/width)^2+(B/height)^2)'
        result['rectangle_condition_each_edge'] = 'A*u+B*v+C > abs(A)*width_fraction/2+abs(B)*height_fraction/2'
    # Keep an exact circle description alongside the verifier's polygon approximation.
    lines = row['observed_actions'][scope['source_start']:scope['source_end']]
    in_outer = False
    for line in lines:
        action = parse_action_line(line)
        if action.type == 'StartLoop':
            in_outer = action.kind == 'outer'
        elif action.type == 'EndLoop':
            in_outer = False
        elif in_outer and isinstance(action, AddCircle):
            result['observed_circle'] = {
                'center_fraction': [(action.center[0]-frame['origin'][0])/w, (action.center[1]-frame['origin'][1])/h],
                'radius_over_shorter_side': action.radius/frame['shorter_side'],
                'polygon_replay_still_required': True,
            }
    return result


def augment_requests(state):
    if state['phase'] not in ('inner_first', 'inner_feedback'):
        return state
    state = deepcopy(state)
    requests = []
    for packet in state['requests']:
        instructions, payload = packet['prompt'].split('\nINPUT:\n', 1)
        public = json.loads(payload)
        if 'observed_boundary_constraints' not in public:
            public['observed_boundary_constraints'] = boundary_context(state['working_source'], packet)
            instructions += (
                'Use observed_boundary_constraints to check your own new loops. '
                'These equations describe existing geometry, not a proposed patch. '
                'For unspecified positions, choose your own interior center and dimensions '
                'satisfying ALL edges, avoiding retained/new holes and exceeding the minimum area. '
                'Convex halfplanes do not apply to a concave outline. '
                'Fractional radius must exceed minimum_circle_radius_fraction_strict; '
                'it is NOT a world-unit radius. Explicit geometry must remain as specified. '
            )
        q = deepcopy(packet)
        q['prompt'] = instructions+'\nINPUT:\n'+json.dumps(public, sort_keys=True)
        requests.append(q)
    state['requests'] = requests
    key = 'requests' if state['phase'] == 'inner_first' else 'feedback_requests'
    state['inner_session'][key] = deepcopy(requests)
    return state


def prepare_boundary(row):
    return augment_requests(prepare_expanded(row))


def submit_boundary(row, state, raw):
    return augment_requests(submit_expanded(row, state, raw))
