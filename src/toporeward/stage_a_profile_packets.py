"""Profile-local LLM requests with explicit retention decisions.

All packets must be merged and pass the existing multi-profile verifier before
Stage B. Complementing model-selected keep IDs is serialization, not a host choice.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import re

from .lm.parsing import parse_action_line
from .stage_a_multi_transaction import prepare_multi_request
from .stage_a_multi_grammar import request_grammar as multi_grammar


def prepare_profile_packets(row):
    parent = prepare_multi_request(row)
    public = json.loads(parent['prompt'].split('\nINPUT:\n', 1)[1])
    packets = []
    for profile in public['profiles']:
        profile = deepcopy(profile)
        profile['keep_count'] = len(profile['inner_loops']) - profile['remove_count']
        for loop in profile['inner_loops']:
            loop['world_geometry'] = [
                {'action_type': type(action).__name__, **asdict(action)}
                for action in (parse_action_line(s) for s in loop['actions'][1:-1])]
        visible = {'design_brief': row['design_brief'], 'feature_plan': row['feature_plan'],
                   'current_profile': profile,
                   'minimum_loop_area_world_strict': public['minimum_loop_area_world_strict']}
        text = (
            'Repair ONLY the current profile. Other profiles are handled separately. '
            'Select existing inner loops to KEEP, not loops to delete. Match their world '
            'centers, shapes and dimensions to the requested openings for this profile '
            'in the design brief. Do not keep a loop merely because it is first in the list. '
            'Return exactly keep_count distinct keep_loop_ids from this profile; all '
            'other observed inner loops will be removed. Preserve kept geometry exactly. '
            'Add exactly add_count missing loops. Use only geometry belonging to the '
            'current profile, not dimensions or centers from a different profile. '
            'The host only converts your selected IDs/parameters into edits; it does not '
            'correct, move, shrink, search or substitute geometry. '
            'For explicit coordinates/dimensions, use coordinate_frame="world". '
            'For unspecified geometry prefer "profile_fraction": center=[u,v] relative '
            'to the lower-left outer box, circle radius divided by its shorter side, '
            'rectangle width/height divided by box width/height. '
            'The bounding box is NOT the feasible region. Check the actual connected '
            'outer edges or outer circle: the new loop must lie strictly inside them, '
            'avoid existing and new holes, and have area above the public tolerance. '
            'In particular, never assume the box center is strictly inside a triangle. '
            'Unspecified geometry is a feasibility choice, not known intended geometry. '
            'Return JSON only: {"profiles":[{"profile_id":"<current profile>",'
            '"keep_loop_ids":["inner_0"],"add_loops":[{"coordinate_frame":"world",'
            '"loop":{"shape":"circle","center":[x,y],"radius":r}}]}]}. '
            'A rectangle uses shape="rectangle", center, width and height instead. '
            'Use empty lists for zero counts. {"profiles":[]} abstains.\nINPUT:\n'
        ) + json.dumps(visible, sort_keys=True)
        packets.append({'case_id': row['case_id'], 'profile_id': profile['profile_id'],
                        'source_uid': parent['source_uid'] + '::packet::' + profile['profile_id'],
                        'attempt': 0, 'prompt': text, 'scopes': [next(
                            deepcopy(s) for s in parent['scopes'] if s['profile_id'] == profile['profile_id'])]})
    return packets


def packet_grammar(packet):
    converted = deepcopy(packet)
    scope = converted['scopes'][0]
    scope['remove_count'] = len(scope['inner_loops']) - scope['remove_count']
    return multi_grammar(converted).replace(re.escape(json.dumps('remove_loop_ids')),
                                           re.escape(json.dumps('keep_loop_ids')))


def merge_packet_responses(row, packets, responses):
    expected = prepare_profile_packets(row)
    if [p['profile_id'] for p in packets] != [p['profile_id'] for p in expected]:
        raise ValueError('packet_profile_coverage')
    if len(responses) != len(packets):
        raise ValueError('packet_response_coverage')
    profiles = []
    for packet, original, response in zip(packets, expected, responses):
        if packet['scopes'] != original['scopes']:
            raise ValueError('stale_packet_scope')
        data = json.loads(response)
        if not isinstance(data, dict) or set(data) != {'profiles'}:
            raise ValueError('packet_schema')
        if not isinstance(data['profiles'], list) or len(data['profiles']) != 1:
            raise ValueError('packet_abstention_or_multiple_profiles')
        profile = data['profiles'][0]
        if not isinstance(profile, dict) or set(profile) != {'profile_id', 'keep_loop_ids', 'add_loops'}:
            raise ValueError('packet_profile_schema')
        if profile['profile_id'] != packet['profile_id']:
            raise ValueError('wrong_packet_profile')
        scope = packet['scopes'][0]
        ids = [loop['loop_id'] for loop in scope['inner_loops']]
        keep = profile['keep_loop_ids']
        if (not isinstance(keep, list) or not all(isinstance(v, str) for v in keep)
                or len(keep) != len(ids) - scope['remove_count']
                or len(set(keep)) != len(keep) or not set(keep).issubset(ids)):
            raise ValueError('invalid_keep_ids')
        profiles.append({'profile_id': profile['profile_id'],
                         'remove_loop_ids': [v for v in ids if v not in keep],
                         'add_loops': profile['add_loops']})
    return json.dumps({'profiles': profiles})
