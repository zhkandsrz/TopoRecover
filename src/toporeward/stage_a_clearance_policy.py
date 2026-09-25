"""LLM-selected local coordinates with an explicit CAD clearance decoder.

This is geometric parameterization, not candidate-feature-free neural geometry.
No search, resampling, clipping, or symbolic replacement of rejected patches.
"""
from copy import deepcopy
import json
import math

from shapely.geometry import Polygon, Point
from shapely.ops import triangulate

from .stage_a_boundary_context import prepare_boundary, submit_boundary
from .stage_a_packet_feedback import _observed_face, _distance_to_boundary
from .stage_a_multi_transaction import committed_spans
from .verifier.geometry import EPS, signed_area, circle_as_polygon


def regions(row, packet):
    face = _observed_face(row, packet['profile_id'])
    outer = next(loop.points for loop in face.loops if loop.kind == 'outer')
    polygon = Polygon(outer)
    if not polygon.is_valid or polygon.area <= EPS:
        raise ValueError('invalid_observed_outer')
    items = []
    for triangle in triangulate(polygon):
        if not polygon.covers(triangle):
            continue
        vertices = sorted([list(p) for p in list(triangle.exterior.coords)[:3]])
        items.append({'vertices': vertices, 'area': triangle.area})
    # Public deterministic geometry representation; no proposed patch is scored.
    items.sort(key=lambda r: (-r['area'], r['vertices']))
    return [{'triangle_id': i, **r} for i, r in enumerate(items)]


def decode_clearance(row, packet, response):
    data = json.loads(response)
    if data == {'profiles': []}:
        return response, []
    if not isinstance(data, dict) or len(data.get('profiles', [])) != 1:
        raise ValueError('clearance_profile_schema')
    p = data['profiles'][0]
    if p.get('profile_id') != packet['profile_id']:
        raise ValueError('wrong_packet_profile')
    if not any(a.get('coordinate_frame') == 'interior_clearance' for a in p['add_loops']):
        return response, []
    face = _observed_face(row, packet['profile_id'])
    outer = next(loop.points for loop in face.loops if loop.kind == 'outer')
    inner = [loop.points for loop in face.loops if loop.kind == 'inner']
    ids = [f'inner_{i}' for i in range(len(inner))]
    keep = p['keep_loop_ids']
    if len(keep) != len(set(keep)) or not set(keep).issubset(ids):
        raise ValueError('invalid_keep_ids')
    obstacles = [inner[ids.index(k)] for k in keep]
    triangles = regions(row, packet)
    audit = []
    for i, added in enumerate(p['add_loops']):
        if added['coordinate_frame'] == 'interior_clearance':
            spec = added['loop']
            if set(spec) != {'shape', 'triangle_id', 'weights', 'radius_fraction'} or spec['shape'] != 'circle':
                raise ValueError('clearance_circle_schema')
            index, weights, fraction = spec['triangle_id'], spec['weights'], spec['radius_fraction']
            if type(index) is not int or not 0 <= index < len(triangles):
                raise ValueError('invalid_triangle_id')
            if (not isinstance(weights, list) or len(weights) != 3
                    or any(type(w) is not int or not 1 <= w <= 9 for w in weights)
                    or type(fraction) not in (int, float) or not .1 <= fraction <= .9):
                raise ValueError('invalid_clearance_parameters')
            vertices = triangles[index]['vertices']
            center = [sum(w*v[j] for w, v in zip(weights, vertices))/sum(weights) for j in range(2)]
            point = Point(center)
            if not Polygon(outer).contains(point) or any(Polygon(p).covers(point) for p in obstacles):
                raise ValueError('clearance_center_not_free')
            clearance = min(_distance_to_boundary(center, poly) for poly in [outer, *obstacles])
            lower = 1.1 * math.sqrt(EPS / (12*math.sin(math.pi/12)))
            upper = .9 * clearance
            if upper <= lower:
                raise ValueError('insufficient_clearance_above_area_tolerance')
            radius = lower + fraction*(upper-lower)
            world = {'shape': 'circle', 'center': center, 'radius': radius}
            p['add_loops'][i] = {'coordinate_frame': 'world', 'loop': world}
            audit.append({'proposal_index': i, 'model_parameters': deepcopy(spec),
                          'clearance_world': clearance, 'minimum_radius_world': lower,
                          'decoded_world': world, 'symbolic_geometry_search': False})
        else:
            from .stage_a_profile_grounding import decode_grounded
            world = decode_grounded(row, {'scope': packet['scopes'][0]}, json.dumps({
                'profile_id': packet['profile_id'], **added}))['loop']
        if world['shape'] == 'circle':
            points = circle_as_polygon(world['center'], world['radius'])
        else:
            x, y = world['center']; w, h = world['width'], world['height']
            points = [(x-w/2, y-h/2), (x+w/2, y-h/2), (x+w/2, y+h/2), (x-w/2, y+h/2)]
        obstacles.append(points)
    return json.dumps(data), audit


def augment(state):
    if state['phase'] not in ('inner_first', 'inner_feedback'):
        return state
    state = deepcopy(state)
    for q in state['requests']:
        if q['scopes'][0]['add_count'] == 0:
            old = json.loads(q['prompt'].split('\nINPUT:\n', 1)[1])
            profile = old.get('current_profile')
            if profile is None:
                profile = {'keep_count': old['keep_count'], 'inner_loops': old['existing_inner_loops']}
            order = [p.profile_id for p in committed_spans(state['working_source']['observed_actions'])]
            public = {
                'design_brief': old['design_brief'], 'current_profile_id': q['profile_id'],
                'retained_profiles_in_history_order': order,
                'current_profile_position_one_based': order.index(q['profile_id'])+1,
                'keep_count': profile['keep_count'],
                'existing_inner_loops': [{'loop_id': loop['loop_id'], 'world_geometry': loop['world_geometry']}
                                         for loop in profile['inner_loops']]}
            if 'last_rejection' in old:
                public['last_rejection'] = old['last_rejection']
            q['prompt'] = (
                'Select which existing inner loops to KEEP in the current CAD profile. '
                'This request adds no geometry. Read the design brief for this profile, '
                'using its position in the retained profile order. Match the requested '
                'existing opening to its exact world center, shape and dimensions below. '
                'Return exactly keep_count distinct loop IDs; unselected loops are deleted. '
                'Keep means preserve, not remove. Do not select the profile center unless '
                'the brief requests that opening. Other profiles are handled separately. '
                'Return JSON with profiles containing one object with profile_id, '
                'keep_loop_ids, and add_loops=[]; or profiles=[] to abstain.\nINPUT:\n'
            ) + json.dumps(public, sort_keys=True)
            q['retention_only_context'] = True
            continue
        instructions, payload = q['prompt'].split('\nINPUT:\n', 1)
        public = json.loads(payload)
        triangles = regions(state['working_source'], q)
        public['local_geometry_coordinates'] = {
            'triangles_world': triangles,
            'decoder': 'center=sum(weights[i]*vertices[i])/sum(weights); radius=r_min+fraction*(0.9*clearance-r_min)',
            'clearance': 'distance from model-chosen center to outer, kept holes and preceding new holes',
            'r_min': '1.1*sqrt(EPS/(12*sin(pi/12))); EPS=1e-7',
            'no_search_or_replacement': True}
        instructions += (
            'Additional CAD-native coordinate option: ONLY for a circle with unspecified '
            'center AND radius, prefer interior_clearance instead of inventing world numbers. '
            'Choose one observed triangle_id, three positive integer weights (1..9) for its '
            'listed vertices, and radius_fraction between0.1 and0.9. Larger-area triangles '
            'usually offer more clearance. The host evaluates exactly your selected '
            'coordinates; an obstructed or too-small region is rejected, not replaced. '
            'This option generates {"coordinate_frame":"interior_clearance",'
            '"loop":{"shape":"circle","triangle_id":0,"weights":[1,1,1],"radius_fraction":0.5}}. '
            'For ANY explicitly specified center or dimension, keep the original world '
            'or profile_fraction option and reproduce the requested parameters exactly. '
            'Do not change specified geometry to make execution easier. ')
        q['prompt'] = instructions+'\nINPUT:\n'+json.dumps(public, sort_keys=True)
        q['clearance_coordinates'] = True
    key = 'requests' if state['phase'] == 'inner_first' else 'feedback_requests'
    state['inner_session'][key] = deepcopy(state['requests'])
    return state


def prepare_clearance(row):
    return augment(prepare_boundary(row))


def submit_clearance(row, state, raw):
    if state['phase'] == 'retention':
        return augment(submit_boundary(row, state, raw))
    converted, trace = [], []
    by_id = {q['source_uid']: q for q in state['requests']}
    for r in raw:
        q = by_id[r['source_uid']]
        try:
            response, audit = decode_clearance(state['working_source'], q, r['response'])
            error = None
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            response, audit, error = '{"profiles":[]}', [], str(exc)
        converted.append({**r, 'response': response})
        trace.append({'source_uid': r['source_uid'], 'raw_response': r['response'],
                      'decoded_response': response, 'geometry_decoder': audit, 'decode_error': error})
    updated = submit_boundary(row, state, converted)
    updated['coordinate_trace'] = state.get('coordinate_trace', []) + trace
    if updated['phase'] == 'complete':
        updated['result']['coordinate_trace'] = updated['coordinate_trace']
        updated['result']['geometry_parameterization'] = 'llm_selected_barycentric_center_clearance_radius'
    return augment(updated)
