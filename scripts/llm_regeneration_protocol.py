"""Public-only full-program generation prompts and validation; no model training."""
from copy import deepcopy
import json
import re

from toporeward.lm.parsing import parse_action_line, NUMBER
from toporeward.strong_repair_baselines import _dsl_text, runtime_trace, requirement_validation_feedback
from toporeward.stage_b_preserving_geometry import bind_plan_dimensions
from toporeward.public_profile_geometry import check_public_geometry
from toporeward.stage_a_compiler_fallback import validate_public_output

METHODS = ('plan_generation', 'feedback_regeneration')
FIELDS = ('case_id', 'design_brief', 'feature_plan', 'observed_actions', 'topology_contract')


def full_program_grammar():
    point = rf'\({NUMBER},\s*{NUMBER}\)'
    ident = r'[A-Za-z0-9_\-]+'
    actions = [r'StartSketch', r'StartFace', r'StartLoop\(kind=(outer|inner)\)',
               rf'AddLine\(start={point},\s*end={point}\)',
               rf'AddArc\(start={point},\s*mid={point},\s*end={point}\)',
               rf'AddCircle\(center={point},\s*radius={NUMBER}\)',
               r'EndLoop', r'EndFace', rf'RegisterProfile\(profile_id={ident}\)',
               r'EndSketch', rf'Extrude\(profile_id={ident},\s*depth={NUMBER},\s*op=(add|cut|intersect)\)', r'End']
    action = '"(' + '|'.join(actions) + ')"'
    # Syntax only: invalid references, ordering, dimensions and nontermination
    # remain possible and are checked after generation, never masked here.
    return r'\{\s*"actions"\s*:\s*\[\s*' + action + r'(\s*,\s*' + action + r')*\s*\]\s*\}'


def parse_program(raw):
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [], 'invalid_json'
    if not isinstance(value, dict) or set(value) != {'actions'}:
        return [], 'invalid_program_schema'
    lines = value['actions']
    if not isinstance(lines, list) or not 0 < len(lines) <= 240:
        return [], 'invalid_action_count'
    if any(not isinstance(line, str) or parse_action_line(line) is None for line in lines):
        return [], 'unparseable_action'
    return lines, None


def check_program(source, lines):
    public = {k: deepcopy(source[k]) for k in FIELDS}
    try:
        public['topology_contract'], _ = bind_plan_dimensions(public['topology_contract'], public['feature_plan'])
    except (ValueError, KeyError, TypeError) as error:
        return dict(accepted=False, reason=str(error))
    geometry = check_public_geometry(public, lines)
    checked = validate_public_output(lines, public['topology_contract'])
    return dict(accepted=bool(geometry['accepted'] and checked['accepted']),
                reason=checked['reason'] if not checked['accepted'] else geometry['reason'],
                public_geometry=geometry, execution=checked)


def feedback(source, lines):
    # Only public requirement checks and the current execution error, never the
    # certificate locating an earlier committed-topology mismatch.
    result = check_program(source, lines)
    trace = runtime_trace(lines)
    return dict(runtime=dict(execution_status=trace.execution_status,
        first_rejected_step=trace.first_rejected_step, failure_type=trace.failure_type,
        repair_hint=trace.repair_hint), checks=result,
        requirements=requirement_validation_feedback(history_lines=lines, feature_plan=source['feature_plan']))


def prompt(source, method, *, previous=None, public_feedback=None):
    if method not in METHODS:
        raise ValueError('unknown_regeneration_method')
    parts = ['Generate a complete executable parametric CAD command sequence that satisfies the design requirements.',
        'Return only JSON with the schema {"actions":["Action", ...]}. Do not output a patch or explanation.',
        'Use at most 240 commands. The complete program must terminate with End.',
        'Supported command syntax:', _dsl_text(),
        'A sketch contains faces. Each face begins with an outer loop followed by its inner loops.',
        'Close each loop and face, register the completed profile, then end the sketch before extrusion.',
        'Use unique profile identifiers. Extrusions must reference registered profiles and honor the plan operation and depth.',
        'For a polygon, successive edges must share endpoints and the final edge must close the loop.',
        'Use supplied geometry and dimensions; never silently change an explicit requirement.',
        '[DESIGN BRIEF]', str(source['design_brief']),
        '[FROZEN FEATURE PLAN]', json.dumps(source['feature_plan'], sort_keys=True)]
    if method == 'feedback_regeneration':
        current = source['observed_actions'] if previous is None else previous
        report = feedback(source, current) if public_feedback is None else public_feedback
        parts += ['Re-generate the entire program to fix the errors. Reuse compatible original geometry and dimensions when possible.',
                  '[ORIGINAL FAILED PROGRAM]', json.dumps(source['observed_actions']),
                  '[CURRENT PROGRAM]', json.dumps(current),
                  '[PUBLIC EXECUTION AND REQUIREMENT FEEDBACK]', json.dumps(report, sort_keys=True)]
    return '\n'.join(parts)
