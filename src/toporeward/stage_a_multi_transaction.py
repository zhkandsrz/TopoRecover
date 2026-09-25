"""Development-only LM inner-loop edits across verifier-localized profiles.

The host supplies scope and serialization, not removal choices or new geometry.
The frozen single-loop policies and published experiments are unchanged.
"""
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
import json

from .llm_stage_a import complete_after_patch, profile_diagnostics
from .natural_repair_recall import apply_patch_operations
from .stage_a_dual_frame_policy import normalized_loops
from .stage_a_profile_grounding import decode_grounded, observed_profile_frame
from .strong_repair_baselines import runtime_trace
from .topology_transaction_repair import _profile_transaction_spans
from .verifier.geometry import EPS


MAX_PROFILES = 8
MAX_LOOP_EDITS = 16
MAX_NEW_ACTIONS = 128
MODE = "multi_profile_inner_loop_patch_v12"


def committed_spans(lines):
    trace = runtime_trace(lines)
    stop = trace.first_rejected_step
    return _profile_transaction_spans(lines, len(lines) if stop is None else stop)


def transaction_scopes(row):
    lines = row['observed_actions']
    diagnostics = profile_diagnostics(lines, row['topology_contract'])
    if not diagnostics:
        raise ValueError('no_committed_stage_a_obligation')
    if len(diagnostics) > MAX_PROFILES:
        raise ValueError('profile_budget_exceeded')
    spans = {s.profile_id: s for s in committed_spans(lines)}
    scopes = []
    for diagnostic in diagnostics:
        actual = Counter(diagnostic['observed_roles'])
        required = Counter(diagnostic['required_roles'])
        if actual['outer'] != 1 or required['outer'] != 1:
            raise ValueError('unsupported_outer_or_whole_profile_obligation')
        if (set(actual) | set(required)) - {'outer', 'inner'}:
            raise ValueError('unsupported_loop_role')
        span = spans[diagnostic['profile_id']]
        inner = [loop for loop in span.loops if loop.kind == 'inner']
        scopes.append({
            **diagnostic, 'insertion_index': span.end_face,
            'add_count': max(0, required['inner'] - actual['inner']),
            'remove_count': max(0, actual['inner'] - required['inner']),
            'inner_loops': [
                {'loop_id': f'inner_{i}', 'start': loop.start, 'end': loop.end,
                 'actions': list(lines[loop.start:loop.end])}
                for i, loop in enumerate(inner)],
        })
    if sum(s['add_count'] + s['remove_count'] for s in scopes) > MAX_LOOP_EDITS:
        raise ValueError('loop_edit_budget_exceeded')
    return scopes


def prepare_multi_request(row):
    scopes = transaction_scopes(row)
    profiles = []
    for scope in scopes:
        lines = row['observed_actions'][scope['source_start']:scope['source_end']]
        frame = observed_profile_frame(row, scope)
        profiles.append({**scope, 'localized_profile': lines,
                         'observed_outer_frame': frame,
                         'observed_loops_in_frame': normalized_loops(lines, frame)})
    public = {'design_brief': row['design_brief'], 'feature_plan': row['feature_plan'],
              'profiles': profiles, 'minimum_loop_area_world_strict': EPS}
    prompt = (
        'Repair every listed committed profile. Return one JSON object with a profiles list. '
        'For each profile, choose exactly remove_count observed inner-loop IDs to remove '
        'and propose exactly add_count new inner loops. Preserve all other loops and actions. '
        'Use the brief and frozen plan to choose which observed inner loops to retain. '
        'Do not change outer loops, registration IDs, extrusion parameters or the plan. '
        'Deletion IDs are local to their profile. Reuse the given profile order. '
        'New loops must have positive area, lie strictly inside the actual outer outline '
        'and not intersect retained or other new inner loops. The bounding box is only a '
        'coordinate frame, not a feasible region. Respect specified shape and dimensions. '
        'Unspecified geometry is a feasibility choice, not inferred ground truth. '
        'Use world coordinates for explicit dimensions, or profile_fraction: center=[u,v] '
        'relative to the lower-left outer box; radius divided by its shorter side; '
        'width/height divided by box width/height. Never normalize twice. '
        'Host code only applies these edits and converts coordinates; it will not choose '
        'a deletion, search, move, shrink, substitute or fix your geometry. All profiles '
        'must pass replay and contract checks before downstream Stage B runs. '
        'Schema: {"profiles":[{"profile_id":"<given>","remove_loop_ids":["inner_0"],'
        '"add_loops":[{"coordinate_frame":"world","loop":{"shape":"circle",'
        '"center":[x,y],"radius":r}}]}]}. Use empty lists where counts are zero. '
        'A rectangle instead has shape="rectangle", center=[x,y], width=w, height=h. '
        'Output JSON only, or {"profiles":[]} to abstain.\nINPUT:\n'
    ) + json.dumps(public, sort_keys=True)
    return {'case_id': row['case_id'], 'source_uid': f"{row['case_id']}::{MODE}",
            'mode': MODE, 'attempt': 0, 'scopes': scopes, 'prompt': prompt,
            'source_actions': list(row['observed_actions']),
            'source_contract': deepcopy(row['topology_contract']),
            'input_audit': {'private_targets_visible': False, 'oracle_patch_visible': False,
                            'host_selects_deletion': False, 'host_selects_geometry': False,
                            'symbolic_stage_a_fallback': False}}


def _exact_fields(value, fields, reason):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(reason)


def _new_loop(row, scope, proposal):
    _exact_fields(proposal, ('coordinate_frame', 'loop'), 'new_loop_schema')
    world = decode_grounded(row, {'scope': scope}, json.dumps({
        'profile_id': scope['profile_id'], **proposal}))['loop']
    cx, cy = world['center']
    if world['shape'] == 'circle':
        curves = [f"AddCircle(center=({cx},{cy}), radius={world['radius']})"]
    else:
        w, h = world['width'], world['height']
        points = [(cx-w/2, cy-h/2), (cx+w/2, cy-h/2),
                  (cx+w/2, cy+h/2), (cx-w/2, cy+h/2)]
        curves = [f'AddLine(start={a}, end={b})'
                  for a, b in zip(points, points[1:] + points[:1])]
    return ['StartLoop(kind=inner)', *curves, 'EndLoop'], world


def compile_multi_patch(row, request, response):
    if (row['observed_actions'] != request['source_actions']
            or row['topology_contract'] != request['source_contract']):
        raise ValueError('stale_source_or_contract')
    scopes = transaction_scopes(row)
    if scopes != request['scopes']:
        raise ValueError('stale_localization')
    data = json.loads(response)
    _exact_fields(data, ('profiles',), 'multi_patch_schema')
    proposals = data['profiles']
    if not isinstance(proposals, list) or len(proposals) != len(scopes):
        raise ValueError('all_localized_profiles_required')
    by_id = {s['profile_id']: s for s in scopes}
    seen, edits, choices = set(), [], []
    for proposal in proposals:
        _exact_fields(proposal, ('profile_id', 'remove_loop_ids', 'add_loops'), 'profile_patch_schema')
        pid = proposal['profile_id']
        if not isinstance(pid, str) or pid not in by_id or pid in seen:
            raise ValueError('unknown_or_duplicate_profile')
        seen.add(pid)
        scope = by_id[pid]
        remove, add = proposal['remove_loop_ids'], proposal['add_loops']
        if (not isinstance(remove, list) or not all(isinstance(x, str) for x in remove)
                or len(remove) != scope['remove_count'] or len(set(remove)) != len(remove)):
            raise ValueError('wrong_or_duplicate_removal_count')
        if not isinstance(add, list) or len(add) != scope['add_count']:
            raise ValueError('wrong_addition_count')
        inner = {loop['loop_id']: loop for loop in scope['inner_loops']}
        for loop_id in remove:
            if loop_id not in inner:
                raise ValueError('unknown_or_protected_loop')
            loop = inner[loop_id]
            edits.append({'start': loop['start'], 'end': loop['end'], 'replacement': []})
        replacement, parameters = [], []
        for loop in add:
            actions, world = _new_loop(row, scope, loop)
            replacement.extend(actions)
            parameters.append(world)
        if replacement:
            edits.append({'start': scope['insertion_index'], 'end': scope['insertion_index'],
                          'replacement': replacement})
        choices.append({'profile_id': pid, 'removed_loop_ids': remove, 'new_world_loops': parameters})
    if sum(len(e['replacement']) for e in edits) > MAX_NEW_ACTIONS:
        raise ValueError('new_action_budget_exceeded')
    edits.sort(key=lambda e: (e['start'], e['end']))
    # All insertions for one profile are combined; every index refers to original source.
    patched = apply_patch_operations(row['observed_actions'], edits, preserve_source_text=True)
    last_commit = max(s['source_end'] for s in scopes)
    protected_end = last_commit + sum(len(e['replacement']) - (e['end']-e['start'])
                                     for e in edits if e['end'] <= last_commit)
    return patched, edits, protected_end, choices


def evaluate_multi_response(row, request, response):
    result = {'case_id': row['case_id'], 'mode': MODE, 'attempt': request['attempt'],
              'patch_accepted': False, 'stage_a_success': False, 'topology_success': False,
              'fallback_used': False, 'geometry_clipped_or_replaced': False,
              'final_actions': [], 'action_lcs': 0.0}
    try:
        patched, edits, protected_end, choices = compile_multi_patch(row, request, response)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return {**result, 'failure': str(error)}
    result.update(patch_accepted=True, patched_actions=patched, edits=edits,
                  protected_end=protected_end, model_choices=choices)
    trace = runtime_trace(patched)
    if trace.first_rejected_step is not None and trace.first_rejected_step < protected_end:
        return {**result, 'failure': 'stage_a_replay_rejected',
                'failure_type': trace.failure_type, 'failure_step': trace.first_rejected_step}
    before = {s.profile_id for s in committed_spans(row['observed_actions'])}
    after = {s.profile_id for s in committed_spans(patched)}
    if not before.issubset(after) or profile_diagnostics(patched, row['topology_contract']):
        return {**result, 'failure': 'stage_a_obligations_unresolved'}
    result['stage_a_success'] = True
    final, reason = complete_after_patch(patched, row['topology_contract'],
                                        protected_end=protected_end,
                                        preserve_reference_parameters=True)
    result['completion_status'] = reason
    if final is None:
        return {**result, 'failure': reason}
    # Stage B may fix the suffix, but cannot revise any accepted profile transaction.
    if final[:protected_end] != patched[:protected_end]:
        return {**result, 'failure': 'stage_b_changed_protected_history'}
    result.update(topology_success=True, final_actions=final)
    original = row['observed_actions']
    result['action_lcs'] = sum(b.size for b in SequenceMatcher(
        a=original, b=final, autojunk=False).get_matching_blocks()) / max(1, len(original))
    return result


def multi_feedback(row, request, response):
    if request['attempt'] != 0:
        raise ValueError('feedback_budget_exhausted')
    result = evaluate_multi_response(row, request, response)
    if result['stage_a_success']:
        return None
    diagnosis = {'rejected_response': response, 'failure': result['failure']}
    if result.get('patched_actions'):
        trace = runtime_trace(result['patched_actions'])
        diagnosis.update(first_rejected_step=trace.first_rejected_step,
                         failure_type=trace.failure_type, repair_hint=trace.repair_hint)
    instructions, payload = request['prompt'].split('\nINPUT:\n', 1)
    public = json.loads(payload)
    public['last_rejection'] = diagnosis
    revised = deepcopy(request)
    revised.update(attempt=1, source_uid=request['source_uid'] + '::feedback',
                   prompt=instructions + '\nINPUT:\n' + json.dumps(public, sort_keys=True))
    return revised
