"""Original packet grammar plus a CAD-native interior-circle parameterization."""
import json
import re


def request_grammar(request):
    if 'keep_count' in request:
        from retention_grammar_source import request_grammar as retention_grammar
        return retention_grammar(request)
    ws = r'[ \n\t\r]*'
    def lit(x):
        return re.escape(json.dumps(x, separators=(',', ':')))
    def field(k, v):
        return lit(k)+ws+':'+ws+v
    def obj(fs):
        return r'\{'+ws+(ws+','+ws).join(fs)+ws+r'\}'
    def arr(xs):
        return r'\['+ws+(ws+','+ws).join(xs)+ws+r'\]'
    num = r'(0|[1-9][0-9]{0,8})(\.[0-9]{1,10})?'
    frac = r'(0(\.[0-9]{1,10})?|1(\.0{1,10})?)'
    choices = []
    for frame in ('world', 'profile_fraction'):
        n = frac if frame == 'profile_fraction' else num
        c = n if frame == 'profile_fraction' else '-?'+n
        for shape in ('circle', 'rectangle'):
            fields = [field('shape', lit(shape)), field('center', arr([c, c]))]
            fields += [field(k, n) for k in (('radius',) if shape == 'circle' else ('width', 'height'))]
            choices.append(obj([field('coordinate_frame', lit(frame)), field('loop', obj(fields))]))
    choices.append(obj([field('coordinate_frame', lit('interior_clearance')), field('loop', obj([
        field('shape', lit('circle')), field('triangle_id', r'(0|[1-9][0-9]{0,2})'),
        field('weights', arr([r'[1-9]']*3)), field('radius_fraction', r'0\.[1-9]')]))]))
    scope = request['scopes'][0]
    count = len(scope['inner_loops'])-scope['remove_count']
    keep = '(?:'+'|'.join(lit(p['loop_id']) for p in scope['inner_loops'])+')'
    profile = obj([field('profile_id', lit(scope['profile_id'])),
        field('keep_loop_ids', arr([keep]*count)),
        field('add_loops', arr(['(?:'+'|'.join(choices)+')']*scope['add_count']))])
    return ws+'(?:'+obj([field('profiles', arr([profile]))])+'|'+obj([field('profiles', arr([]))])+')'+ws
