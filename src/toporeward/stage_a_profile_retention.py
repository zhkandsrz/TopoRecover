"""LLM-selected whole-profile retention with explicit deletion dependency closure.

This is a development extension, not an activated fallback for the inner-loop
policy. The model chooses retained profiles. The host only applies the choice
and removes direct uses of deleted profiles and empty sketch envelopes.
"""
from copy import deepcopy
import json
import re

from .actions import EndSketch, Extrude, StartSketch
from .lm.parsing import parse_action_line
from .natural_repair_recall import apply_patch_operations
from .stage_a_multi_transaction import committed_spans


def prepare_retention(row):
    spans = sorted(committed_spans(row['observed_actions']), key=lambda s: s.profile_id)
    desired = row['topology_contract'].get('profiles')
    if not isinstance(desired, list) or not desired:
        raise ValueError('explicit_nonempty_profile_contract_required')
    if len(spans) <= len(desired):
        raise ValueError('no_surplus_committed_profiles')
    if len(spans) > 16:
        raise ValueError('retention_profile_budget_exceeded')
    profiles = [{'profile_id': s.profile_id, 'source_start': s.face_start,
                 'source_end': s.register_end,
                 'actions': row['observed_actions'][s.face_start:s.register_end]} for s in spans]
    dependencies = [{'step': i, 'action': line, 'profile_id': action.profile_id}
                    for i, line in enumerate(row['observed_actions'])
                    if isinstance(action := parse_action_line(line), Extrude)]
    public = {'design_brief': row['design_brief'], 'feature_plan': row['feature_plan'],
              'topology_contract': row['topology_contract'], 'committed_profiles': profiles,
              'extrusion_dependencies': dependencies, 'keep_count': len(desired)}
    prompt = (
        'The CAD history committed more profiles than its frozen plan requires. '
        'Choose exactly keep_count distinct observed profile IDs to KEEP. Match the '
        'brief and plan, using their actual geometry; do not simply keep the first IDs. '
        'You may not invent or modify geometry in this selection step. '
        'The unselected profile transactions will be deleted. Their dependent Extrude '
        'actions and any now-empty sketch envelopes will also be deleted as an explicit '
        'execution dependency closure. Other actions will stay unchanged. '
        'Later inner-loop repair and Stage B must still satisfy the complete contract. '
        'Return JSON only: {"keep_profile_ids":["<observed id>"]}. '
        'An empty list abstains and will not be replaced by a host selection.\nINPUT:\n'
    ) + json.dumps(public, sort_keys=True)
    return {'case_id': row['case_id'], 'source_uid': row['case_id']+'::profile_retention',
            'attempt': 0, 'prompt': prompt, 'profiles': profiles, 'keep_count': len(desired),
            'source_actions': deepcopy(row['observed_actions']),
            'source_contract': deepcopy(row['topology_contract'])}


def retention_grammar(request):
    ws = r'[ \n\t\r]*'
    ids = '(?:'+'|'.join(re.escape(json.dumps(p['profile_id'])) for p in request['profiles'])+')'
    nonempty = (ws+','+ws).join([ids]*request['keep_count'])
    return ws+r'\{'+ws+'"keep_profile_ids"'+ws+':'+ws+r'\['+ws+'(?:'+nonempty+'|)'+ws+r'\]'+ws+r'\}'+ws


def apply_retention(row, request, response):
    if (row['observed_actions'] != request['source_actions'] or
            row['topology_contract'] != request['source_contract']):
        raise ValueError('stale_retention_source')
    data = json.loads(response)
    if not isinstance(data, dict) or set(data) != {'keep_profile_ids'}:
        raise ValueError('retention_schema')
    keep = data['keep_profile_ids']
    known = {p['profile_id'] for p in request['profiles']}
    if (not isinstance(keep, list) or not all(isinstance(p, str) for p in keep)
            or len(keep) != request['keep_count'] or len(set(keep)) != len(keep)
            or not set(keep).issubset(known)):
        raise ValueError('invalid_retention_selection')
    drop = known - set(keep)
    lines = row['observed_actions']
    removed, provenance = set(), []
    for profile in request['profiles']:
        if profile['profile_id'] in drop:
            removed.update(range(profile['source_start'], profile['source_end']))
            provenance.append({'kind': 'model_selected_profile_deletion',
                'profile_id': profile['profile_id'], 'start': profile['source_start'], 'end': profile['source_end']})
    sketch_start = None
    for i, line in enumerate(lines):
        action = parse_action_line(line)
        if isinstance(action, Extrude) and action.profile_id in drop:
            removed.add(i)
            provenance.append({'kind': 'deleted_profile_direct_use', 'start': i, 'end': i+1,
                               'profile_id': action.profile_id})
        if isinstance(action, StartSketch):
            sketch_start = i
        elif isinstance(action, EndSketch) and sketch_start is not None:
            if all(j in removed for j in range(sketch_start+1, i)):
                removed.update((sketch_start, i))
                provenance.append({'kind': 'empty_sketch_envelope', 'start': sketch_start, 'end': i+1})
            sketch_start = None
    edits = []
    for index in sorted(removed):
        if edits and edits[-1]['end'] == index:
            edits[-1]['end'] += 1
        else:
            edits.append({'start': index, 'end': index+1, 'replacement': []})
    patched = apply_patch_operations(lines, edits, preserve_source_text=True)
    actual = {s.profile_id for s in committed_spans(patched)}
    if actual != set(keep):
        raise ValueError('retained_profiles_not_replayable')
    for profile in request['profiles']:
        if profile['profile_id'] in keep:
            block = lines[profile['source_start']:profile['source_end']]
            if not any(patched[i:i+len(block)] == block for i in range(len(patched)-len(block)+1)):
                raise ValueError('retained_profile_changed')
    return {'observed_actions': patched, 'kept_profile_ids': keep,
            'dropped_profile_ids': sorted(drop), 'edits': edits,
            'dependency_closure': provenance, 'host_chose_profiles': False,
            'geometry_changed': False, 'symbolic_stage_a_fallback': False}
