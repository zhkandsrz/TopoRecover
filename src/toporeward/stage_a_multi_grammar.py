"""Serialization grammar for multi-profile patches, not a geometry-validity mask."""
import json
import re


def request_grammar(request):
    # Pretty-printed nested JSON commonly uses 4+ spaces. Bounding whitespace
    # per node can block the nonempty branch while leaving an empty-list branch
    # available. Bound generation tokens, not legal JSON indentation.
    ws = r'[ \n\t\r]*'
    scalar = r'(0|[1-9][0-9]{0,8})(\.[0-9]{1,10})?'
    fraction = r'(0(\.[0-9]{1,10})?|1(\.0{1,10})?)'

    def literal(value):
        return re.escape(json.dumps(value, separators=(',', ':')))

    def field(key, value):
        return literal(key) + ws + ':' + ws + value

    def obj(fields):
        return r'\{' + ws + (ws + ',' + ws).join(fields) + ws + r'\}'

    def array(items):
        return r'\[' + ws + (ws + ',' + ws).join(items) + ws + r'\]'

    choices = []
    for frame in ('world', 'profile_fraction'):
        number = fraction if frame == 'profile_fraction' else scalar
        coord = number if frame == 'profile_fraction' else '-?' + number
        for shape in ('circle', 'rectangle'):
            fields = [field('shape', literal(shape)), field('center', array([coord, coord]))]
            dimensions = ('radius',) if shape == 'circle' else ('width', 'height')
            fields.extend(field(k, number) for k in dimensions)
            choices.append(obj([field('coordinate_frame', literal(frame)), field('loop', obj(fields))]))
    new_loop = '(?:' + '|'.join(choices) + ')'
    profiles = []
    for scope in request['scopes']:
        # The LM chooses observed IDs. Duplicate deletion IDs are rejected by the
        # patch validator, not repaired or silently deduplicated by the host.
        removal = '(?:' + '|'.join(literal(loop['loop_id']) for loop in scope['inner_loops']) + ')'
        profiles.append(obj([
            field('profile_id', literal(scope['profile_id'])),
            field('remove_loop_ids', array([removal] * scope['remove_count'])),
            field('add_loops', array([new_loop] * scope['add_count'])),
        ]))
    complete = obj([field('profiles', array(profiles))])
    abstain = obj([field('profiles', array([]))])
    return ws + '(?:' + complete + '|' + abstain + ')' + ws
