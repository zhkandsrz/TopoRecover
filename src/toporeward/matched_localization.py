"""Region-only interventions for a shared bounded-patch diagnostic.

This is an attribution harness, not a replacement for the deployed packet policy.
Every arm sees identical diagnostics; only editable_regions changes.
"""
from copy import deepcopy
import json

from .actions import Extrude, StartSketch, EndSketch, StartFace, EndFace, StartLoop, EndLoop
from .llm_stage_a import prepare_request, profile_diagnostics
from .lm.parsing import parse_action_line
from .strong_repair_baselines import runtime_trace
from .topology_transaction_repair import _profile_transaction_spans


SELECTORS = ('rejected_command', 'runtime_dependency', 'active_block',
             'topology', 'full_history')


def _region(start, end):
    return {'start': start, 'end': end}


def _active_block(lines, trigger):
    opening = {StartSketch: EndSketch, StartFace: EndFace, StartLoop: EndLoop}
    stack = []
    for i, line in enumerate(lines[:trigger]):
        action = parse_action_line(line)
        if type(action) in opening:
            stack.append((i, opening[type(action)]))
        elif stack and type(action) is stack[-1][1]:
            stack.pop()
    if not stack:
        return _region(trigger, min(trigger + 1, len(lines)))
    start, closing = stack[0]
    end = next((i + 1 for i in range(trigger, len(lines))
                if isinstance(parse_action_line(lines[i]), closing)), len(lines))
    return _region(start, end)


def select_regions(row, selector):
    if selector not in SELECTORS:
        raise ValueError('unknown_region_selector')
    lines = row['observed_actions']
    if not lines:
        return [_region(0, 0)]
    trace = runtime_trace(lines)
    trigger = trace.first_rejected_step
    trigger = len(lines) - 1 if trigger is None else trigger
    local = _active_block(lines, trigger)
    if selector == 'full_history':
        return [_region(0, len(lines))]
    if selector == 'rejected_command':
        return [_region(trigger, trigger + 1)]
    if selector == 'active_block':
        return [local]
    if selector == 'runtime_dependency':
        action = parse_action_line(lines[trigger])
        if isinstance(action, Extrude):
            spans = _profile_transaction_spans(lines, trigger)
            definitions = [s for s in spans if s.profile_id == action.profile_id]
            if definitions:
                return [_region(definitions[-1].face_start, trigger + 1)]
        return [local]
    diagnoses = profile_diagnostics(lines, row['topology_contract'])
    return ([_region(d['source_start'], d['source_end']) for d in diagnoses]
            or [local])


def request(row, selector):
    common = prepare_request(row, 'obligations')
    header, payload = common['prompt'].split('\nINPUT:\n', 1)
    public = json.loads(payload)
    regions = select_regions(row, selector)
    public['editable_regions'] = regions
    result = deepcopy(common)
    result.update(editable_regions=regions,
                  source_uid=row['case_id'] + '::region::' + selector,
                  prompt=header + '\nINPUT:\n' + json.dumps(public, sort_keys=True))
    return result


def assert_only_regions_change(requests):
    common = None
    for req in requests:
        header, payload = req['prompt'].split('\nINPUT:\n', 1)
        public = json.loads(payload)
        assert public.pop('editable_regions') == req['editable_regions']
        value = (header, public)
        if common is None:
            common = value
        assert value == common, 'non-region prompt content differs'


def syntax_grammar():
    # Syntax is common to all arms. Semantic scope/geometry checks occur later.
    ws = r'\s*'
    integer = r'(0|[1-9][0-9]*)'
    number = r'[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?'
    point = rf'\({number},\s*{number}\)'
    ident = r'[A-Za-z0-9_\-]+'
    commands = [r'StartSketch', r'StartFace', r'StartLoop\(kind=(outer|inner)\)',
                rf'AddLine\(start={point},\s*end={point}\)',
                rf'AddArc\(start={point},\s*mid={point},\s*end={point}\)',
                rf'AddCircle\(center={point},\s*radius={number}\)',
                r'EndLoop', r'EndFace', rf'RegisterProfile\(profile_id={ident}\)',
                r'EndSketch', rf'Extrude\(profile_id={ident},\s*depth={number},\s*op=(add|cut|intersect)\)', r'End']
    action = '"(' + '|'.join(commands) + ')"'
    actions = r'\[' + ws + '(' + action + '(' + ws + ',' + ws + action + r')*)?' + ws + r'\]'
    edit = (r'\{' + ws + '"start"' + ws + ':' + ws + integer + ws + ',' + ws
            + '"end"' + ws + ':' + ws + integer + ws + ',' + ws
            + '"replacement"' + ws + ':' + ws + actions + ws + r'\}')
    return (r'\{' + ws + '"edits"' + ws + ':' + ws + r'\[' + ws
            + '(' + edit + '(' + ws + ',' + ws + edit + ')*)?' + ws + r'\]' + ws + r'\}')
