"""LLM-selected structural edits in a verifier-localized uncommitted sketch."""
from copy import deepcopy
from difflib import SequenceMatcher
import json

from .actions import AddCircle, AddLine, End, EndSketch, Extrude, RegisterProfile, StartFace, StartSketch
from .lm.parsing import parse_action_line
from .stage_a_packet_policy import PUBLIC_SOURCE_FIELDS
from .stage_a_terminal_policy import prepare_terminal_intent, submit_terminal_intent
from .stage_a_reference_intent_policy import align_contract_indices
from .stage_a_multi_transaction import committed_spans
from .plan_parameter_alignment import bind_aligned_plan_dimensions
from .stage_b_transaction_preserving import complete_transaction_preserving
from .verifier import TopoVerifier

CONTROL = ('StartSketch', 'StartFace', 'StartLoop(kind=outer)', 'StartLoop(kind=inner)',
           'EndLoop', 'EndFace', 'EndSketch', 'End')
MAX_EDITS = 32
MAX_WINDOW = 128


class StructureReplayError(ValueError):
    def __init__(self, feedback):
        super().__init__('structure_patch_no_replay_progress')
        self.feedback = feedback


def state_summary(state):
    return {'active_stack':list(state.stack),
            'completed_loop_roles':[loop.kind for loop in state.current_face_loops],
            'active_loop_kind':state.current_loop.kind if state.current_loop is not None else None,
            'pending_face':state.pending_face is not None,
            'pending_face_roles':[loop.kind for loop in state.pending_face.loops] if state.pending_face else [],
            'registered_profiles':sorted(state.profiles),
            'completed_extrusions':len(state.extrusions)}


def first_failure(lines):
    verifier = TopoVerifier()
    state = verifier.initial_state()
    for index, line in enumerate(lines):
        action = parse_action_line(line)
        if action is None:
            return index, state, 'unparseable_action'
        result = verifier.step(state, action)
        if not result.valid:
            return index, state, str(result.failure_type)
        state = result.next_state
    return None, state, None


def structure_certificate(row):
    lines = row['observed_actions']
    index, state, failure = first_failure(lines)
    if index is None or isinstance(parse_action_line(lines[index]), (End, EndSketch)):
        return None
    parsed = [parse_action_line(s) for s in lines]
    if any(a is None for a in parsed):
        return None
    faces = [i for i,a in enumerate(parsed[:index+1]) if isinstance(a, StartFace)]
    if not faces:
        return None
    start = faces[-1]
    if start and isinstance(parsed[start-1], StartSketch):
        start -= 1
    end = next((i+1 for i in range(index+1,len(lines)) if isinstance(parsed[i], EndSketch)),len(lines))
    if end-start > MAX_WINDOW:
        return None
    plan_ids = {p['profile_id'] for p in row['feature_plan'].get('profiles', [])}
    duplicate_ids = sorted({a.profile_id for a in parsed[start:end] if isinstance(a, RegisterProfile)
                            and a.profile_id in state.profiles and a.profile_id in plan_ids})
    matching = sorted(pid for pid, face in state.profiles.items()
                      if state.pending_face is not None and face == state.pending_face)
    # A redundant unregistered face can be discarded only when the stored shape
    # already exists and all publicly required profile identities are present.
    single_face = sum(isinstance(a,StartFace) for a in parsed[start:end]) == 1
    redundant = single_face and (bool(duplicate_ids) or (bool(matching) and set(state.profiles) == plan_ids))
    deletable = [i for i in range(start,end) if lines[i] in CONTROL or
                 redundant and isinstance(parsed[i], (AddLine, AddCircle, RegisterProfile))]
    protected = [i for i in range(start,end) if i not in deletable]
    return {'rejected_step': index, 'failure_type': failure, 'scope_start': start, 'scope_end': end,
            'rejected_action':lines[index], **state_summary(state),
            'duplicate_registered_ids': duplicate_ids, 'pending_face_matches': matching,
            'redundant_transaction_evidence': redundant, 'deletable_steps': deletable,
            'protected_steps': protected}


def structure_request(row, certificate, attempt, feedback=None):
    public = {'required_profiles':row['feature_plan']['profiles']}
    public['commands'] = [{'step':i,'action':row['observed_actions'][i]}
                          for i in range(certificate['scope_start'],certificate['scope_end'])]
    public['verifier_state'] = certificate
    public['allowed_insertions'] = list(CONTROL)
    if feedback is not None:
        public['previous_attempt'] = feedback
    prompt = (
        'You are repairing CAD command delimiters, not generating a new shape. '
        'Return the smallest edit that fixes the rejected transition and any later malformed delimiters.\n'
        'STATE MACHINE:\n'
        '- StartSketch enters sketch; StartFace enters face.\n'
        '- StartLoop(outer) requires face with zero completed loops. '
        'StartLoop(inner) requires face with a completed outer loop. Both enter loop.\n'
        '- AddLine/AddCircle require an active loop. EndLoop closes it and returns to face.\n'
        '- EndFace closes a face and sets pending_face=true, returning to sketch.\n'
        '- RegisterProfile requires sketch + pending_face=true and an unused profile ID. '
        'It consumes the pending face. A subsequent outer loop needs a NEW StartFace.\n'
        '- Extrude uses an already registered profile; it requires no open face/loop and no pending face. '
        'It needs no new sketch or face. EndSketch closes sketch; End requires an empty stack.\n'
        'REPAIR:\n'
        'Compare rejected_action with the actual verifier_state. Preserve correctly placed delimiters. '
        'In particular, moving an EndFace does NOT require deleting an EndLoop that closes geometry. '
        'If registration occurs while a face is still open, move its EndFace before RegisterProfile. '
        'Do not put a second EndFace after the profile has already been consumed.\n'
        'When redundant_transaction_evidence is true, the profile already exists: discard the redundant '
        'construction, not its protected Extrude. Rebuilding an empty face would create another failure. '
        'Otherwise all geometry must remain. Never alter protected commands or numeric parameters. '
        'Later execution completion handles explicit feature-plan extrusion corrections.\n'
        'Output edits use ORIGINAL step numbers below, never renumbered window positions. '
        'Only remove listed deletable_steps and only insert allowed_insertions within the scope. '
        'before_step may equal scope_end. Same-position insertions execute in list order before removal. '
        'At most 32 edits total. If previous_attempt is present, it was rejected and NOT applied; '
        'use its replay failure to revise the patch against the SAME original commands.\n'
        'Return ONLY a JSON object with one key edits, a list of edit operations. '
        'Each operation starts with op, followed by its arguments in the order below:\n'
        '- move: step, before_step. Move the EXISTING control command at step to before '
        'the ORIGINAL command at before_step. The command is copied exactly and its old occurrence removed. '
        'Use move for misplaced delimiters; do not separately insert and delete the same delimiter.\n'
        '- insert: before_step, action. Insert a missing control command before an ORIGINAL command.\n'
        '- delete: start_step, end_step. Remove the ORIGINAL half-open range '
        '[start_step, end_step), excluding end_step. Every removed step must be listed as deletable. '
        'Never include a protected Extrude in a deletion range.\n'
        'All edits refer to the same ORIGINAL commands, regardless of earlier edits in the list. '
        'Moves count as two edits and deletions count once per removed command toward the 32-edit budget. '
        'Choose the positions from the supplied commands and certificate, not a generic template. '
        'An empty edits list means abstention.\nINPUT:\n'
    )+json.dumps(public,sort_keys=True)
    return {'case_id':row['case_id'], 'source_uid':row['case_id']+f'::structure::{attempt}',
            'structure_repair':True, 'prompt':prompt, 'attempt':attempt,
            'structure_edit_operations':True,
            'movable_steps':[i for i in certificate['deletable_steps'] if row['observed_actions'][i] in CONTROL],
            'insertion_positions':list(range(certificate['scope_start'],certificate['scope_end']+1)),
            'deletable_steps':list(certificate['deletable_steps']),
            'input_audit':{'private_targets_visible':False,'geometry_parameters_locked':True,
                           'geometry_deletion_requires_duplicate_certificate':True,
                           'host_selects_edits':False,'symbolic_stage_a_fallback':False}}


def expand_edit_operations(row, certificate, edits):
    if not isinstance(edits,list) or not 0 < len(edits) <= MAX_EDITS:
        raise ValueError('structure_patch_empty_or_budget')
    insertions, remove = [], []
    for edit in edits:
        if not isinstance(edit,dict):
            raise ValueError('structure_edit_schema')
        op = edit.get('op')
        if op == 'move' and set(edit) == {'op','step','before_step'}:
            step = edit['step']
            if (type(step) is not int or step not in certificate['deletable_steps']
                    or row['observed_actions'][step] not in CONTROL):
                raise ValueError('structure_move_outside_scope_or_protected')
            if edit['before_step'] in (step,step+1):
                raise ValueError('structure_move_noop')
            insertions.append({'before_step':edit['before_step'],'action':row['observed_actions'][step]})
            remove.append(step)
        elif op == 'insert' and set(edit) == {'op','before_step','action'}:
            insertions.append({'before_step':edit['before_step'],'action':edit['action']})
        elif op == 'delete' and set(edit) == {'op','start_step','end_step'}:
            start, end = edit['start_step'],edit['end_step']
            if (type(start) is not int or type(end) is not int
                    or not certificate['scope_start'] <= start < end <= certificate['scope_end']):
                raise ValueError('structure_delete_range_outside_scope')
            remove.extend(range(start,end))
        else:
            raise ValueError('structure_edit_schema')
    return insertions,remove


def compile_structure_patch(row, certificate, response):
    if structure_certificate(row) != certificate:
        raise ValueError('stale_structure_certificate')
    value = json.loads(response)
    model_edits = None
    if isinstance(value,dict) and set(value) == {'edits'}:
        model_edits = value['edits']
        insertions,removals = expand_edit_operations(row,certificate,model_edits)
        value = {'insertions':insertions,'remove_steps':removals}
    if not isinstance(value,dict) or set(value) not in ({'insertions','remove_steps'},
                                                       {'diagnosis','insertions','remove_steps'}):
        raise ValueError('structure_patch_schema')
    if 'diagnosis' in value and (not isinstance(value['diagnosis'],str)
                                or not value['diagnosis'].strip() or len(value['diagnosis']) > 400):
        raise ValueError('structure_diagnosis_schema')
    ins, remove = value['insertions'], value['remove_steps']
    if not isinstance(ins,list) or not isinstance(remove,list) or not 0 < len(ins)+len(remove) <= MAX_EDITS:
        raise ValueError('structure_patch_empty_or_budget')
    if any(type(i) is not int for i in remove) or len(set(remove)) != len(remove):
        raise ValueError('invalid_structure_removal')
    if not set(remove).issubset(certificate['deletable_steps']):
        raise ValueError('structure_removal_outside_scope_or_protected')
    by_position = {}
    for item in ins:
        if (not isinstance(item,dict) or set(item) != {'before_step','action'}
                or type(item['before_step']) is not int
                or not certificate['scope_start'] <= item['before_step'] <= certificate['scope_end']
                or item['action'] not in CONTROL):
            raise ValueError('structure_insertion_outside_scope_or_not_control')
        by_position.setdefault(item['before_step'],[]).append(item['action'])
    lines, origins = [], []
    original = row['observed_actions']
    for index in range(len(original)+1):
        additions = by_position.get(index,[])
        lines.extend(additions); origins.extend([None]*len(additions))
        if index < len(original) and index not in remove:
            lines.append(original[index]); origins.append(index)
    rejected, replay_state, failure = first_failure(lines)
    if rejected is not None:
        remaining_origins = [x for x in origins[rejected:] if x is not None]
        frontier = next((i for i in range(certificate['rejected_step'],len(original))
                         if i not in certificate['deletable_steps']),None)
        # Removing a rejected delimiter is not progress if the next preserved
        # action still cannot execute. Require a retained frontier to pass.
        if (not remaining_origins or remaining_origins[0] <= certificate['rejected_step']
                or frontier is None or frontier not in origins[:rejected]):
            raise StructureReplayError({'candidate_step':rejected,'rejected_action':lines[rejected],
                                        'failure_type':failure,'state':state_summary(replay_state),
                                        'required_retained_frontier':frontier,
                                        'next_original_step':remaining_origins[0] if remaining_origins else None})
    return lines, {'insertions':deepcopy(ins),'removed_original_steps':remove,
                   'source_start':certificate['scope_start'],'source_end':certificate['scope_end'],
                   'redundant_transaction_evidence':certificate['redundant_transaction_evidence'],
                   'model_diagnosis':value.get('diagnosis'),
                   'model_edits':deepcopy(model_edits),
                   'selection_source':'model_response','model_response':response}


def _finish(state, child):
    state = deepcopy(state)
    state['child'] = child
    state['requests'] = child['requests']
    state['_actual_calls'] = state['prior_calls']+child.get('_actual_calls',0)
    state['_actual_tokens'] = state['prior_tokens']+child.get(
        '_actual_tokens',child.get('result',{}).get('output_tokens',0))
    if child['phase'] != 'complete':
        if state['rounds'] >= 6:
            state.update(phase='complete',requests=[],result={
                'case_id':state['source']['case_id'],'final_actions':[],
                'stage_a_success':False,'topology_success':False,
                'failure':'structure_composite_round_budget_exhausted','route':'llm_structure',
                'llm_calls':state['_actual_calls'],'output_tokens':state['_actual_tokens'],
                'structure_model_calls':state['structure_calls'],'structure_patches':state['patches'],
                'action_lcs':0.,'symbolic_stage_a_fallback':False})
            return state
        state['phase'] = 'child'
        return state
    result = deepcopy(child['result'])
    working = deepcopy(child.get('normalized_source',state['working_source']))
    eligible = (result.get('stage_a_success') and result.get('patched_actions')
                or result.get('route','').endswith(('stage_b_only','llm_retention_then_stage_b')))
    if not result.get('topology_success') and eligible:
        accepted = (result['patched_actions']
                    if result.get('stage_a_success') and result.get('patched_actions')
                    else working['observed_actions'])
        n_features = sum(isinstance(parse_action_line(s),Extrude) for s in accepted)
        if 0 < n_features < len(working['feature_plan'].get('features',[])):
            try:
                aligned, index_evidence = align_contract_indices(working['topology_contract'],working['feature_plan'],accepted)
                aligned, dimensions = bind_aligned_plan_dimensions(aligned,working['feature_plan'],accepted)
                protected = max((s.register_end for s in committed_spans(accepted)),default=0)
                completion = complete_transaction_preserving(accepted,aligned,protected_end=protected,
                                                             align_missing_features=True)
                result['feature_alignment_audit'] = completion.to_dict()
                if completion.actions:
                    result.update(final_actions=list(completion.actions),topology_success=True,
                                  completion_status=completion.status,
                                  public_reference_index_audit=index_evidence,
                                  public_plan_dimension_evidence=dimensions,
                                  geometry_reference_visible=False)
                    result.pop('failure',None)
            except ValueError as error:
                result['feature_alignment_failure'] = str(error)
    if (not result.get('topology_success') and not state.get('structure_disabled')
            and state['structure_calls'] < 4 and state['rounds'] < 6):
        working = deepcopy(child.get('normalized_source',state['working_source']))
        if result.get('stage_a_success') and result.get('patched_actions'):
            working['observed_actions'] = result['patched_actions']
        certificate = structure_certificate(working)
        if certificate is not None:
            state.update(working_source=working,certificate=certificate,phase='structure')
            state['requests'] = [structure_request(working,certificate,state['structure_calls'])]
            return state
    result['llm_calls'] = state['prior_calls']+result.get('llm_calls',0)
    result['calls'] = result['llm_calls']
    result['output_tokens'] = state['prior_tokens']+result.get('output_tokens',0)
    result['structure_model_calls'] = state['structure_calls']
    result['structure_patches'] = state['patches']
    result['structure_attempts'] = state['attempts']
    if state['patches']:
        result['route'] = 'llm_structure_then_'+result.get('route','completion')
    result['symbolic_stage_a_fallback'] = False
    original, final = state['source']['observed_actions'],result.get('final_actions',[])
    result['action_lcs'] = sum(b.size for b in SequenceMatcher(a=original,b=final,autojunk=False).get_matching_blocks())/max(1,len(original))
    state.update(phase='complete',requests=[],result=result)
    return state


def prepare_structure_intent(row):
    source = {k:deepcopy(row[k]) for k in PUBLIC_SOURCE_FIELDS}
    child = prepare_terminal_intent(source)
    working = deepcopy(child.get('normalized_source',source))
    return _finish({'source':source,'working_source':working,'prior_calls':0,'prior_tokens':0,
                    'structure_calls':0,'rounds':0,'patches':[],'attempts':[],'window_retry':0},child)


def submit_structure_intent(row,state,raw):
    if any(row[k] != state['source'][k] for k in PUBLIC_SOURCE_FIELDS):
        raise ValueError('stale_structure_source')
    state = deepcopy(state)
    state['rounds'] += 1
    if state['phase'] == 'child':
        child = submit_terminal_intent(state['child']['source'],state['child'],raw)
        return _finish(state,child)
    if state['phase'] != 'structure' or len(raw) != 1 or raw[0]['source_uid'] != state['requests'][0]['source_uid']:
        raise ValueError('structure_response_coverage')
    previous = state['child']['result']
    state['prior_calls'] += previous.get('llm_calls',0)+1
    state['prior_tokens'] += previous.get('output_tokens',0)+raw[0].get('output_tokens',0)
    state['structure_calls'] += 1
    try:
        lines, patch = compile_structure_patch(state['working_source'],state['certificate'],raw[0]['response'])
    except (ValueError,TypeError) as error:
        # An invalid model edit is an explicit failed candidate, not a symbolic rescue.
        failed = {'phase':'complete','requests':[],'result':{'case_id':row['case_id'],
                  'topology_success':False,'stage_a_success':False,'final_actions':[],
                  'failure':str(error),'route':'llm_structure','llm_calls':0,'output_tokens':0}}
        feedback = {'response':raw[0]['response'],'failure':str(error),
                    'replay':getattr(error,'feedback',None)}
        state['attempts'].append({'source_uid':raw[0]['source_uid'],'accepted':False,**feedback})
        if (isinstance(error,StructureReplayError) and state['window_retry'] < 1
                and state['structure_calls'] < 4 and state['rounds'] < 6):
            state.update(child=failed,window_retry=1,_actual_calls=state['prior_calls'],
                         _actual_tokens=state['prior_tokens'])
            state['requests'] = [structure_request(state['working_source'],state['certificate'],
                                                    state['structure_calls'],feedback)]
            return state
        state['structure_disabled'] = True
        return _finish(state,failed)
    state['attempts'].append({'source_uid':raw[0]['source_uid'],'accepted':True,'response':raw[0]['response']})
    state['window_retry'] = 0
    state['patches'].append(patch)
    state['working_source']['observed_actions'] = lines
    return _finish(state,prepare_terminal_intent(state['working_source']))
